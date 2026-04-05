"""
Training Pipeline Orchestrator.

Runs the full end-to-end process:
Raw Text -> Tokenizer -> PyTorch Model -> ONNX Model

Features:
- Configurable model architecture (RoBERTa, BERT)
- Resume from checkpoint support
- Learning rate warmup and weight decay
- Training metrics logging
- Beautiful Rich terminal output

Example:
    >>> from myspellchecker.training import TrainingPipeline, TrainingConfig
    >>> config = TrainingConfig(
    ...     input_file="corpus.txt",
    ...     output_dir="./model",
    ...     architecture="roberta",  # or "bert"
    ...     epochs=5,
    ... )
    >>> pipeline = TrainingPipeline()
    >>> model_path = pipeline.run(config)
"""

import os
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from myspellchecker.training.constants import (
    DEFAULT_EMBEDDING_LR,
    DEFAULT_EMBEDDING_WARMUP_STEPS,
    DEFAULT_MIN_FREQUENCY,
    DEFAULT_VOCAB_SIZE,
    DEFAULT_WARMUP_RATIO,
    DEFAULT_WEIGHT_DECAY,
)
from myspellchecker.training.exporter import ONNXExporter
from myspellchecker.training.reporter import TrainingReporter
from myspellchecker.training.trainer import ModelArchitecture, ModelTrainer
from myspellchecker.utils.logging_utils import get_logger


@dataclass
class TrainingConfig:
    """
    Configuration for the training pipeline.

    Attributes:
        input_file: Path to training corpus (one sentence per line).
        output_dir: Directory to save trained model and artifacts.
        vocab_size: Vocabulary size for BPE tokenizer.
        min_frequency: Minimum frequency for token inclusion.
        epochs: Number of training epochs.
        batch_size: Batch size per device.
        learning_rate: Peak learning rate.
        hidden_size: Size of hidden layers.
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads.
        max_length: Maximum sequence length.
        keep_checkpoints: Keep intermediate checkpoints.
        architecture: Model architecture ("roberta" or "bert").
        resume_from_checkpoint: Path to checkpoint to resume from.
        warmup_ratio: Ratio of steps for learning rate warmup.
        weight_decay: Weight decay for optimizer.
        save_metrics: Save training metrics to JSON file.
        checkpoint_dir: Persistent directory for checkpoints (e.g. /opt/ml/checkpoints).
            When set, tokenizer and model checkpoints are saved here and survive
            job restarts. On resume, completed steps are automatically skipped.

    Example:
        >>> config = TrainingConfig(
        ...     input_file="corpus.txt",
        ...     output_dir="./output",
        ...     architecture="roberta",
        ...     epochs=10,
        ...     warmup_ratio=0.1,
        ... )
    """

    input_file: str
    output_dir: str
    vocab_size: int = DEFAULT_VOCAB_SIZE
    min_frequency: int = DEFAULT_MIN_FREQUENCY
    epochs: int = 5
    batch_size: int = 16
    learning_rate: float = 5e-5
    hidden_size: int = 256
    num_layers: int = 4
    num_heads: int = 4
    max_length: int = 128
    keep_checkpoints: bool = False
    architecture: str = "roberta"
    resume_from_checkpoint: str | None = None
    warmup_ratio: float = DEFAULT_WARMUP_RATIO
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    save_metrics: bool = True
    streaming: bool = False
    checkpoint_dir: str | None = None
    max_steps: int | None = None
    word_boundary_aware: bool = False
    whole_word_masking: bool = False
    pos_file: str | None = None
    denoising_ratio: float = 0.0
    fp16: bool = False
    gradient_accumulation_steps: int = 1
    confusable_masking: bool = False
    confusable_mask_ratio: float = 0.3
    confusable_words_file: str | None = None
    lr_scheduler_type: str = "linear"
    corruption_ratio: float = 0.0
    mlm_probability: float = 0.15
    embedding_surgery: bool = False
    embedding_warmup_steps: int = DEFAULT_EMBEDDING_WARMUP_STEPS
    embedding_lr: float = DEFAULT_EMBEDDING_LR

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")
        if self.vocab_size < 100:
            raise ValueError("vocab_size must be >= 100")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if self.num_heads < 1:
            raise ValueError("num_heads must be >= 1")
        if self.max_length < 16:
            raise ValueError("max_length must be >= 16")
        if self.hidden_size < 1:
            raise ValueError("hidden_size must be >= 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if not 0 <= self.warmup_ratio < 1:
            raise ValueError("warmup_ratio must be >= 0 and < 1")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be >= 0")
        if self.confusable_masking and not self.whole_word_masking:
            raise ValueError(
                "confusable_masking requires whole_word_masking=True "
                "(confusable masking operates on whole words)"
            )
        if self.confusable_masking and not 0.0 < self.confusable_mask_ratio <= 1.0:
            raise ValueError("confusable_mask_ratio must be in (0.0, 1.0]")

        # Validate architecture
        try:
            ModelArchitecture.from_string(self.architecture)
        except ValueError as e:
            raise ValueError(f"Invalid architecture: {self.architecture}. {e}") from e

        # Transformer architecture constraint: hidden_size must be divisible by num_heads
        # This is required for multi-head attention to split embeddings evenly across heads
        if self.hidden_size % self.num_heads != 0:
            valid_heads = [h for h in [1, 2, 4, 8, 16] if self.hidden_size % h == 0]
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_heads ({self.num_heads}). "
                f"Valid num_heads values for hidden_size={self.hidden_size}: {valid_heads}"
            )


class TrainingPipeline:
    """
    Orchestrates the creation of a Semantic Model.

    Features beautiful Rich terminal output with step-based progress
    and a final summary table.
    """

    def __init__(self, quiet: bool = False):
        """
        Initialize the training pipeline.

        Args:
            quiet: If True, suppress Rich output (use plain logging only).
        """
        self.logger = get_logger(__name__)
        self.trainer = ModelTrainer()
        self.exporter = ONNXExporter()
        self.reporter = TrainingReporter(force_plain=quiet)
        self.quiet = quiet

    def run(self, config: TrainingConfig) -> str:
        """
        Run the full training pipeline.

        Args:
            config: Training configuration.

        Returns:
            Path to the final ONNX model directory.
        """
        input_path = Path(config.input_file).resolve()
        output_path = Path(config.output_dir).resolve()

        if not input_path.exists():
            raise FileNotFoundError(f"Input corpus not found: {input_path}")

        # Show pipeline header
        self.reporter.show_header(
            corpus_path=str(input_path),
            output_dir=str(output_path),
            architecture=config.architecture,
            epochs=config.epochs,
            vocab_size=config.vocab_size,
        )

        # Track step durations for summary
        step_durations: dict[str, tuple[str, float]] = {}
        pipeline_start = time.time()

        # Create a temporary workspace for intermediate artifacts
        # (tokenizer.json, pytorch checkpoints)
        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)

            # Step 1: Train Tokenizer
            self.reporter.step_start(1, 3, "Training Tokenizer")
            step1_start = time.time()

            tokenizer_path = self.trainer.train_tokenizer(
                corpus_path=str(input_path),
                output_dir=str(work_dir),
                vocab_size=config.vocab_size,
                min_frequency=config.min_frequency,
                word_boundary_aware=config.word_boundary_aware,
            )

            step1_duration = time.time() - step1_start
            self.reporter.step_complete(
                1, 3, "Tokenizer trained", f"vocab_size={config.vocab_size:,}"
            )
            step_durations["1. Tokenizer Training"] = ("complete", step1_duration)

            # Step 2: Train Model
            self.reporter.step_start(2, 3, "Training Language Model")
            step2_start = time.time()

            pytorch_model_dir = self.trainer.train_model(
                corpus_path=str(input_path),
                tokenizer_path=tokenizer_path,
                output_dir=str(work_dir),
                epochs=config.epochs,
                batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                hidden_size=config.hidden_size,
                num_hidden_layers=config.num_layers,
                num_attention_heads=config.num_heads,
                max_length=config.max_length,
                architecture=ModelArchitecture.from_string(config.architecture),
                resume_from_checkpoint=config.resume_from_checkpoint,
                warmup_ratio=config.warmup_ratio,
                weight_decay=config.weight_decay,
                save_metrics=config.save_metrics,
                whole_word_masking=config.whole_word_masking,
                pos_file=config.pos_file,
                denoising_ratio=config.denoising_ratio,
                fp16=config.fp16,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                reporter=self.reporter,
                confusable_masking=config.confusable_masking,
                confusable_mask_ratio=config.confusable_mask_ratio,
                confusable_words_file=config.confusable_words_file,
                lr_scheduler_type=config.lr_scheduler_type,
                corruption_ratio=config.corruption_ratio,
                mlm_probability=config.mlm_probability,
                embedding_surgery=config.embedding_surgery,
                embedding_warmup_steps=config.embedding_warmup_steps,
                embedding_lr=config.embedding_lr,
            )

            step2_duration = time.time() - step2_start
            self.reporter.step_complete(
                2, 3, "Model trained", f"epochs={config.epochs}, architecture={config.architecture}"
            )
            step_durations["2. Model Training"] = ("complete", step2_duration)

            # Copy checkpoints BEFORE exporting to ONNX to prevent data loss
            if config.keep_checkpoints:
                checkpoint_dest = output_path / "pytorch_source"
                if not checkpoint_dest.exists():
                    shutil.copytree(pytorch_model_dir, checkpoint_dest)
                self.reporter.info(f"PyTorch source preserved at {checkpoint_dest}")

            # Step 3: Export to ONNX
            self.reporter.step_start(3, 3, "Exporting to ONNX")
            step3_start = time.time()

            self.exporter.export(
                model_dir=pytorch_model_dir,
                output_dir=str(output_path),
                quantize=True,
            )

            step3_duration = time.time() - step3_start
            self.reporter.step_complete(3, 3, "ONNX export complete", "quantized=True")
            step_durations["3. ONNX Export"] = ("complete", step3_duration)

        # Show summary
        total_duration = time.time() - pipeline_start
        self.reporter.show_summary(step_durations, total_duration, str(output_path))

        return str(output_path)

    def run_streaming(self, config: TrainingConfig) -> str:
        """Run the training pipeline in streaming mode for large corpora.

        Same steps as run() but uses LineByLineIterableDataset to keep
        memory constant regardless of corpus size.

        When ``config.checkpoint_dir`` is set, all intermediate artifacts
        (tokenizer, model checkpoints) are written to that directory so they
        survive job restarts.  On resume, completed steps are automatically
        skipped.

        Args:
            config: Training configuration (streaming field is ignored here;
                    this method always streams).

        Returns:
            Path to the final ONNX model directory.
        """
        input_path = Path(config.input_file).resolve()
        output_path = Path(config.output_dir).resolve()

        if not input_path.exists():
            raise FileNotFoundError(f"Input corpus not found: {input_path}")

        self.reporter.show_header(
            corpus_path=str(input_path),
            output_dir=str(output_path),
            architecture=config.architecture,
            epochs=config.epochs,
            vocab_size=config.vocab_size,
        )

        step_durations: dict[str, tuple[str, float]] = {}
        pipeline_start = time.time()

        # Use persistent work_dir when checkpoint_dir is set so artifacts
        # survive job restarts.  Otherwise fall back to output_dir/_work.
        if config.checkpoint_dir:
            work_dir = Path(config.checkpoint_dir)
        else:
            work_dir = output_path / "_work"
        work_dir.mkdir(parents=True, exist_ok=True)

        # --- Step 1: Train Tokenizer + Count Lines (parallel) -------------
        # Tokenizer training and line counting both read the corpus
        # independently. Running them in parallel saves a full pass over
        # the (potentially huge) corpus file.
        def _count_lines(path: str) -> int:
            count = 0
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        count += 1
            return count

        tokenizer_dir = work_dir / "tokenizer"
        tokenizer_path = tokenizer_dir / "tokenizer.json"

        # DDP: only rank 0 trains tokenizer; other ranks wait for the file.
        is_main = int(os.environ.get("RANK", "0")) == 0

        self.reporter.step_start(1, 3, "Training Tokenizer + Counting Lines")
        step1_start = time.time()

        # Start line counting in background thread immediately
        self.logger.info("Starting parallel: tokenizer training + line counting")
        with ThreadPoolExecutor(max_workers=2) as executor:
            line_count_future = executor.submit(_count_lines, str(input_path))

            if tokenizer_path.exists():
                self.logger.info(f"Resuming: tokenizer found at {tokenizer_path}")
                tokenizer_path_str = str(tokenizer_path)
            elif is_main:
                tokenizer_dir.mkdir(parents=True, exist_ok=True)
                tokenizer_path_str = self.trainer.train_tokenizer(
                    corpus_path=str(input_path),
                    output_dir=str(tokenizer_dir),
                    vocab_size=config.vocab_size,
                    min_frequency=config.min_frequency,
                    word_boundary_aware=config.word_boundary_aware,
                )
            else:
                # Non-main ranks: poll until rank 0 finishes tokenizer training.
                # Can't use dist.barrier() here because rank 0 is in a different
                # code branch and won't hit the same barrier call.
                self.logger.info("Waiting for rank 0 to train tokenizer...")
                import time as _time

                while not tokenizer_path.exists():
                    _time.sleep(2)
                tokenizer_path_str = str(tokenizer_path)

            line_count = line_count_future.result()

        self.logger.info(f"Corpus: {line_count:,} non-empty lines")

        step1_duration = time.time() - step1_start
        self.reporter.step_complete(
            1,
            3,
            "Tokenizer trained + lines counted",
            f"vocab_size={config.vocab_size:,}, lines={line_count:,}",
        )
        step_durations["1. Tokenizer + Line Count"] = ("complete", step1_duration)

        # --- Step 2: Train model (streaming) ------------------------------
        self.reporter.step_start(2, 3, "Training Language Model (streaming)")
        step2_start = time.time()

        effective_batch = config.batch_size * config.gradient_accumulation_steps
        steps_per_epoch = max(1, line_count // effective_batch)
        max_steps = steps_per_epoch * config.epochs
        if config.max_steps is not None:
            max_steps = min(max_steps, config.max_steps)
            self.logger.info(f"max_steps capped to {max_steps:,} (from config)")
        self.logger.info(
            f"max_steps={max_steps:,} ({steps_per_epoch:,} steps/epoch x {config.epochs} epochs)"
        )

        # Auto-detect training checkpoint for resume.
        # Training checkpoints (with trainer_state.json) take priority over
        # pre-trained model paths because they represent more recent progress.
        # This handles the embedding surgery resume case: config has the v2.1
        # pretrained model path, but checkpoint_dir has Phase 2 checkpoints
        # from an interrupted run.
        resume_path = config.resume_from_checkpoint
        ckpt_dir = work_dir / "checkpoints"
        if ckpt_dir.exists():
            checkpoints = sorted(
                ckpt_dir.glob("checkpoint-*"),
                key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else 0,
            )
            if checkpoints:
                latest_ckpt = str(checkpoints[-1])
                # Only use if it's a real training checkpoint (has trainer_state.json)
                trainer_state = Path(latest_ckpt) / "trainer_state.json"
                if trainer_state.exists():
                    self.logger.info(
                        f"Found training checkpoint: {latest_ckpt} "
                        f"(overrides resume_from_checkpoint={resume_path})"
                    )
                    resume_path = latest_ckpt

        pytorch_model_dir = self.trainer.train_model(
            corpus_path=str(input_path),
            tokenizer_path=tokenizer_path_str,
            output_dir=str(work_dir),
            epochs=config.epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_layers,
            num_attention_heads=config.num_heads,
            max_length=config.max_length,
            architecture=ModelArchitecture.from_string(config.architecture),
            resume_from_checkpoint=resume_path,
            warmup_ratio=config.warmup_ratio,
            weight_decay=config.weight_decay,
            save_metrics=config.save_metrics,
            streaming=True,
            max_steps=max_steps,
            checkpoint_dir=str(ckpt_dir) if config.checkpoint_dir else None,
            whole_word_masking=config.whole_word_masking,
            pos_file=config.pos_file,
            denoising_ratio=config.denoising_ratio,
            fp16=config.fp16,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            reporter=self.reporter,
            confusable_masking=config.confusable_masking,
            confusable_mask_ratio=config.confusable_mask_ratio,
            confusable_words_file=config.confusable_words_file,
            lr_scheduler_type=config.lr_scheduler_type,
            corruption_ratio=config.corruption_ratio,
            mlm_probability=config.mlm_probability,
            embedding_surgery=config.embedding_surgery,
            embedding_warmup_steps=config.embedding_warmup_steps,
            embedding_lr=config.embedding_lr,
        )

        step2_duration = time.time() - step2_start
        self.reporter.step_complete(
            2,
            3,
            "Model trained (streaming)",
            f"epochs={config.epochs}, architecture={config.architecture}",
        )
        step_durations["2. Model Training"] = ("complete", step2_duration)

        # DDP: only rank 0 copies checkpoints and exports ONNX
        if is_main:
            if config.keep_checkpoints:
                checkpoint_dest = output_path / "pytorch_source"
                if not checkpoint_dest.exists():
                    shutil.copytree(pytorch_model_dir, checkpoint_dest)
                    self.reporter.info(f"PyTorch source preserved at {checkpoint_dest}")

            # --- Step 3: Export to ONNX ------------------------------------
            self.reporter.step_start(3, 3, "Exporting to ONNX")
            step3_start = time.time()

            self.exporter.export(
                model_dir=pytorch_model_dir,
                output_dir=str(output_path),
                quantize=True,
            )

            step3_duration = time.time() - step3_start
            self.reporter.step_complete(3, 3, "ONNX export complete", "quantized=True")
            step_durations["3. ONNX Export"] = ("complete", step3_duration)

        total_duration = time.time() - pipeline_start
        if is_main:
            self.reporter.show_summary(step_durations, total_duration, str(output_path))

        return str(output_path)
