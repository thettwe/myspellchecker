"""
Trainer module for Semantic Models.

Handles training of:
1. Byte-Level BPE Tokenizers (via tokenizers)
2. Masked Language Models (RoBERTa or BERT) (via transformers)

Features:
- Model architecture selection (RoBERTa, BERT)
- Resume training from checkpoints
- Learning rate scheduling
- Training metrics callback
- Training data validation

Example:
    >>> from myspellchecker.training import ModelTrainer, ModelArchitecture
    >>> trainer = ModelTrainer()
    >>> trainer.train_model(
    ...     corpus_path="corpus.txt",
    ...     tokenizer_path="tokenizer.json",
    ...     output_dir="./model",
    ...     architecture=ModelArchitecture.ROBERTA,
    ...     resume_from_checkpoint=None,  # Or path to checkpoint
    ... )
"""

import json
import os
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from myspellchecker.core.constants import DEFAULT_TOKENIZER_FILE
from myspellchecker.training.constants import (
    DEFAULT_EMBEDDING_LR,
    DEFAULT_EMBEDDING_WARMUP_STEPS,
    DEFAULT_MLM_PROBABILITY,
    DEFAULT_SAVE_TOTAL_LIMIT,
    IGNORE_LABEL_INDEX,
    LARGE_FILE_WARNING_MB,
    MIN_TRAINING_LINES,
    SPECIAL_TOKENS,
    compute_save_steps,
    get_dataloader_workers,
)
from myspellchecker.utils.logging_utils import get_logger

try:
    import torch
    from tokenizers import ByteLevelBPETokenizer
    from transformers import (
        BertConfig,
        BertForMaskedLM,
        DataCollatorForLanguageModeling,
        PreTrainedTokenizerFast,
        RobertaConfig,
        RobertaForMaskedLM,
        Trainer,
        TrainerCallback,
        TrainerControl,
        TrainerState,
        TrainingArguments,
    )

    TRAINING_AVAILABLE = True
except ImportError:
    # Training dependencies not installed
    torch = None  # type: ignore
    ByteLevelBPETokenizer = None  # type: ignore
    RobertaConfig = None  # type: ignore
    BertConfig = None  # type: ignore
    PreTrainedTokenizerFast = None  # type: ignore
    TrainerCallback = object  # type: ignore
    TRAINING_AVAILABLE = False

# Type checking import for tokenizer type hint
if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerFast as TokenizerType

    from myspellchecker.training.reporter import TrainingReporter


class ModelArchitecture(str, Enum):
    """Supported model architectures for training."""

    ROBERTA = "roberta"
    BERT = "bert"

    @classmethod
    def from_string(cls, value: str) -> "ModelArchitecture":
        """Create from string value."""
        value_lower = value.lower()
        for member in cls:
            if member.value == value_lower:
                return member
        raise ValueError(f"Unknown architecture: {value}. Supported: {[m.value for m in cls]}")


class TrainingMetricsCallback(TrainerCallback if TRAINING_AVAILABLE else object):  # type: ignore[misc]
    """
    Callback for structured training metrics logging.

    Logs metrics to a JSON file for later analysis and visualization.
    """

    def __init__(self, output_dir: str):
        """Initialize the callback."""
        self.output_dir = output_dir
        self.metrics_file = os.path.join(output_dir, "training_metrics.json")
        self.metrics: List[Dict[str, Any]] = []
        self.logger = get_logger(__name__)

    def on_log(
        self,
        args: "TrainingArguments",
        state: "TrainerState",
        control: "TrainerControl",
        logs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Called when the trainer logs metrics."""
        if logs is None:
            return

        # Add step and epoch info
        metric_entry = {
            "step": state.global_step,
            "epoch": state.epoch,
            **logs,
        }
        self.metrics.append(metric_entry)

        # Write to file (append-friendly format)
        self._save_metrics()

    def on_train_end(
        self,
        args: "TrainingArguments",
        state: "TrainerState",
        control: "TrainerControl",
        **kwargs: Any,
    ) -> None:
        """Called when training ends."""
        self._save_metrics()
        self.logger.info(f"Training metrics saved to {self.metrics_file}")

    def _save_metrics(self) -> None:
        """Save metrics to JSON file."""
        os.makedirs(self.output_dir, exist_ok=True)
        with open(self.metrics_file, "w", encoding="utf-8") as f:
            json.dump(self.metrics, f, indent=2, ensure_ascii=False)


class TokenizerSaveCallback(TrainerCallback if TRAINING_AVAILABLE else object):  # type: ignore[misc]
    """Save tokenizer into each checkpoint directory for self-contained resume."""

    def __init__(self, tokenizer: "TokenizerType"):
        self.tokenizer = tokenizer

    def on_save(
        self,
        args: "TrainingArguments",
        state: "TrainerState",
        control: "TrainerControl",
        **kwargs: Any,
    ) -> None:
        if args.output_dir is None:
            return
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if os.path.isdir(checkpoint_path):
            self.tokenizer.save_pretrained(checkpoint_path)
            _fix_tokenizer_config(checkpoint_path)


class WholeWordMaskCollator:
    """Data collator for Whole Word Masking (WWM) with optional POS weighting.

    Instead of masking random BPE tokens, this collator:
    1. Groups BPE tokens by word (using word_ids from tokenizer)
    2. Selects words to mask (with optional POS-based weighting)
    3. Masks ALL tokens of selected words together

    POS Weight Defaults (higher weight = more likely to be masked):
        noun (n): 0.35 — content words are most informative
        verb (v): 0.25 — action words carry semantic meaning
        adj:      0.15 — modifiers are useful context
        part:     0.15 — particles critical for confusable detection (к/ကို, မှာ/မှ)
        other:    0.10 — punctuation, etc.
    """

    DEFAULT_POS_WEIGHTS: Dict[str, float] = {
        "n": 0.35,
        "v": 0.25,
        "adj": 0.15,
        "part": 0.15,
    }
    DEFAULT_OTHER_WEIGHT: float = 0.10

    def __init__(
        self,
        tokenizer: "TokenizerType",
        mlm_probability: float = 0.15,
        pos_weights: Optional[Dict[str, float]] = None,
    ):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.pos_weights = pos_weights or self.DEFAULT_POS_WEIGHTS
        self.default_weight = self.DEFAULT_OTHER_WEIGHT
        self.mask_token_id = tokenizer.mask_token_id
        self.vocab_size = tokenizer.vocab_size
        # Pre-compute special token ids for masking exclusion
        self._special_ids: set = set()
        for attr in (
            "bos_token_id",
            "eos_token_id",
            "pad_token_id",
            "cls_token_id",
            "sep_token_id",
        ):
            tid = getattr(tokenizer, attr, None)
            if tid is not None:
                self._special_ids.add(tid)

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, "torch.Tensor"]:
        input_ids = torch.stack([e["input_ids"] for e in examples])
        attention_mask = torch.stack([e["attention_mask"] for e in examples])
        labels = input_ids.clone()

        for i in range(len(examples)):
            word_ids = examples[i].get("word_ids")
            pos_tags = examples[i].get("pos_tags")

            if word_ids is not None:
                masked_indices = self._whole_word_mask(
                    input_ids[i], word_ids, pos_tags, attention_mask[i]
                )
            else:
                # Fall back to standard random token masking
                prob = torch.full(labels[i].shape, self.mlm_probability)
                special_mask = torch.tensor(
                    [t.item() in self._special_ids for t in input_ids[i]], dtype=torch.bool
                )
                prob.masked_fill_(special_mask, value=0.0)
                prob.masked_fill_(attention_mask[i] == 0, value=0.0)
                masked_indices = torch.bernoulli(prob).bool()

            # Labels: IGNORE_LABEL_INDEX for non-masked (ignored in loss)
            labels[i, ~masked_indices] = IGNORE_LABEL_INDEX

            # 80% → [MASK], 10% → random token, 10% → keep original
            indices_replaced = (
                torch.bernoulli(torch.full(labels[i].shape, 0.8)).bool() & masked_indices
            )
            input_ids[i, indices_replaced] = self.mask_token_id

            indices_random = (
                torch.bernoulli(torch.full(labels[i].shape, 0.5)).bool()
                & masked_indices
                & ~indices_replaced
            )
            random_words = torch.randint(self.vocab_size, labels[i].shape, dtype=torch.long)
            input_ids[i, indices_random] = random_words[indices_random]

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    def _whole_word_mask(
        self,
        input_ids: "torch.Tensor",
        word_ids: List[Optional[int]],
        pos_tags: Optional[List[str]],
        attention_mask: "torch.Tensor",
    ) -> "torch.Tensor":
        """Select whole words to mask, using POS weights if available."""
        # Build word groups: word_id → list of token positions
        word_groups: Dict[int, List[int]] = {}
        for pos, wid in enumerate(word_ids):
            if (
                wid is not None
                and attention_mask[pos] == 1
                and input_ids[pos].item() not in self._special_ids
            ):
                word_groups.setdefault(wid, []).append(pos)

        if not word_groups:
            return torch.zeros_like(input_ids, dtype=torch.bool)

        word_ids_sorted = sorted(word_groups.keys())

        # Compute selection weight per word
        weights = []
        for wid in word_ids_sorted:
            if pos_tags and wid < len(pos_tags):
                w = self.pos_weights.get(pos_tags[wid], self.default_weight)
            else:
                w = 1.0  # uniform when no POS tags
            weights.append(w)

        weights_t = torch.tensor(weights, dtype=torch.float)
        weights_t = weights_t / weights_t.sum()

        # Select ~mlm_probability fraction of words
        num_to_mask = max(1, round(len(word_ids_sorted) * self.mlm_probability))
        num_to_mask = min(num_to_mask, len(word_ids_sorted))
        selected = torch.multinomial(weights_t, num_to_mask, replacement=False)

        # Mark all tokens of selected words
        masked = torch.zeros_like(input_ids, dtype=torch.bool)
        for idx in selected:
            wid = word_ids_sorted[idx.item()]
            for token_pos in word_groups[wid]:
                masked[token_pos] = True

        return masked


class ConfusableAwareMaskCollator(WholeWordMaskCollator):
    """Data collator that biases masking toward confusable words.

    Splits the masking budget into two phases:
    1. Select from confusable words (up to confusable_mask_ratio of budget)
    2. Fill remaining budget from non-confusable words

    This forces the model to predict confusable words from context 3-5x more
    often than random masking, building stronger contextual discrimination
    between near-homophones (e.g., ခက်/ခတ်, တက်/တတ်, ဖူး/ဘူး).

    Corruption augmentation (corruption_ratio > 0):
    A fraction of masked confusable positions are replaced with a confusable
    VARIANT instead of [MASK]. The label remains the CORRECT word. This teaches
    the model to predict the correct form given a wrong-but-plausible variant
    in context, directly training the detection/correction signal.
    """

    def __init__(
        self,
        tokenizer: "TokenizerType",
        confusable_words: set,
        confusable_mask_ratio: float = 0.3,
        mlm_probability: float = 0.15,
        pos_weights: Optional[Dict[str, float]] = None,
        confusable_pairs: Optional[Dict[str, List[str]]] = None,
        corruption_ratio: float = 0.0,
    ):
        super().__init__(tokenizer, mlm_probability, pos_weights)
        self.confusable_words = confusable_words
        self.confusable_mask_ratio = confusable_mask_ratio
        # word → list of confusable variants for corruption augmentation
        self.confusable_pairs = confusable_pairs or {}
        self.corruption_ratio = corruption_ratio

    @classmethod
    def load_confusable_words(cls, homophones_path: Optional[str] = None) -> set:
        """Load confusable word set from homophones YAML or JSON.

        Extracts all words (both sides of every pair) into a flat set.
        Falls back to the bundled homophones.yaml if no path is given.

        Supports two formats:
        - YAML: ``homophones.yaml`` with ``homophones`` key (legacy, ~200 words)
        - JSON: export from production DB with ``confusable_words`` list (~34K words)
        """
        if homophones_path is None:
            homophones_path = str(Path(__file__).parent.parent / "rules" / "homophones.yaml")

        words: set = set()
        _log = get_logger(__name__)
        try:
            if homophones_path.endswith(".json"):
                with open(homophones_path, encoding="utf-8") as f:
                    data = json.load(f)
                words = set(data.get("confusable_words", []))
            else:
                import yaml

                try:
                    with open(homophones_path, encoding="utf-8") as f:
                        data = yaml.safe_load(f)
                except yaml.YAMLError as exc:
                    _log.warning(
                        "Malformed YAML in %s: %s — confusable masking disabled",
                        homophones_path,
                        exc,
                    )
                    return words
                homophones = (data or {}).get("homophones", {})
                for word, variants in homophones.items():
                    words.add(word)
                    if isinstance(variants, list):
                        words.update(variants)
        except (OSError, ValueError, KeyError, ImportError) as exc:
            _log.warning(
                "Failed to load homophones from %s: %s — confusable masking disabled",
                homophones_path,
                exc,
            )

        return words

    @classmethod
    def load_confusable_pairs(cls, homophones_path: Optional[str] = None) -> Dict[str, List[str]]:
        """Load confusable word→variants mapping from homophones YAML or JSON.

        Returns a dict where each word maps to its confusable variants.
        Both directions are included (word→variants AND variant→[word]).
        Used for corruption augmentation: replacing a correct word with
        a confusable variant during training.

        Supports two formats:
        - YAML: ``homophones.yaml`` with ``homophones`` key (legacy)
        - JSON: export from production DB with ``confusable_pairs`` dict
        """
        if homophones_path is None:
            homophones_path = str(Path(__file__).parent.parent / "rules" / "homophones.yaml")

        pairs: Dict[str, List[str]] = {}
        try:
            if homophones_path.endswith(".json"):
                with open(homophones_path, encoding="utf-8") as f:
                    data = json.load(f)
                raw = data.get("confusable_pairs", {})
                pairs = {k: list(v) for k, v in raw.items()}
            else:
                import yaml

                with open(homophones_path, encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                homophones = data.get("homophones", {})
                for word, variants in homophones.items():
                    if isinstance(variants, list):
                        pairs.setdefault(word, []).extend(variants)
                        for v in variants:
                            pairs.setdefault(v, []).append(word)
        except Exception:
            get_logger(__name__).warning(
                "Failed to load confusable pairs from %s",
                homophones_path,
            )

        return pairs

    def _whole_word_mask(
        self,
        input_ids: "torch.Tensor",
        word_ids: List[Optional[int]],
        pos_tags: Optional[List[str]],
        attention_mask: "torch.Tensor",
    ) -> "torch.Tensor":
        """Select whole words to mask, biasing toward confusable words."""
        # Build word groups: word_id → list of token positions
        word_groups: Dict[int, List[int]] = {}
        for pos, wid in enumerate(word_ids):
            if (
                wid is not None
                and attention_mask[pos] == 1
                and input_ids[pos].item() not in self._special_ids
            ):
                word_groups.setdefault(wid, []).append(pos)

        if not word_groups:
            return torch.zeros_like(input_ids, dtype=torch.bool)

        word_ids_sorted = sorted(word_groups.keys())

        # Classify words as confusable or not by decoding tokens
        confusable_indices: List[int] = []
        non_confusable_indices: List[int] = []

        for i, wid in enumerate(word_ids_sorted):
            token_ids = [input_ids[p].item() for p in word_groups[wid]]
            decoded = (
                self.tokenizer.decode(token_ids, skip_special_tokens=True).strip().replace(" ", "")
            )

            if decoded in self.confusable_words:
                confusable_indices.append(i)
            else:
                non_confusable_indices.append(i)

        # Compute total masking budget
        num_to_mask = max(1, round(len(word_ids_sorted) * self.mlm_probability))
        num_to_mask = min(num_to_mask, len(word_ids_sorted))

        selected_word_indices: List[int] = []

        # Phase 1: Select from confusable words
        if confusable_indices:
            confusable_budget = min(
                len(confusable_indices),
                max(1, round(num_to_mask * self.confusable_mask_ratio)),
            )
            c_weights = self._compute_group_weights(confusable_indices, word_ids_sorted, pos_tags)
            c_selected = torch.multinomial(c_weights, confusable_budget, replacement=False)
            selected_word_indices.extend(confusable_indices[idx.item()] for idx in c_selected)

        # Phase 2: Fill remaining from non-confusable words
        remaining = num_to_mask - len(selected_word_indices)
        if remaining > 0 and non_confusable_indices:
            fill_count = min(remaining, len(non_confusable_indices))
            nc_weights = self._compute_group_weights(
                non_confusable_indices, word_ids_sorted, pos_tags
            )
            nc_selected = torch.multinomial(nc_weights, fill_count, replacement=False)
            selected_word_indices.extend(non_confusable_indices[idx.item()] for idx in nc_selected)

        # Phase 3: If budget remains (e.g., all words confusable), fill from confusable pool
        remaining = num_to_mask - len(selected_word_indices)
        if remaining > 0 and confusable_indices:
            already_selected = set(selected_word_indices)
            available = [i for i in confusable_indices if i not in already_selected]
            if available:
                fill_count = min(remaining, len(available))
                extra_weights = self._compute_group_weights(available, word_ids_sorted, pos_tags)
                extra_selected = torch.multinomial(extra_weights, fill_count, replacement=False)
                selected_word_indices.extend(available[idx.item()] for idx in extra_selected)

        # Mark all tokens of selected words
        masked = torch.zeros_like(input_ids, dtype=torch.bool)
        for word_idx in selected_word_indices:
            wid = word_ids_sorted[word_idx]
            for token_pos in word_groups[wid]:
                masked[token_pos] = True

        return masked

    def _compute_group_weights(
        self,
        indices: List[int],
        word_ids_sorted: List[int],
        pos_tags: Optional[List[str]],
    ) -> "torch.Tensor":
        """Compute normalized POS-based weights for a subset of word indices."""
        weights = []
        for i in indices:
            wid = word_ids_sorted[i]
            if pos_tags and wid < len(pos_tags):
                w = self.pos_weights.get(pos_tags[wid], self.default_weight)
            else:
                w = 1.0
            weights.append(w)
        weights_t = torch.tensor(weights, dtype=torch.float)
        total = weights_t.sum()
        if total == 0:
            weights_t = torch.ones_like(weights_t)
            total = weights_t.sum()
        return weights_t / total

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, "torch.Tensor"]:
        """Collate with optional corruption augmentation.

        After standard MLM masking (80% [MASK], 10% random, 10% keep),
        replaces a fraction of confusable masked words with their confusable
        variant's tokens. Labels remain the original correct tokens, teaching
        the model to correct wrong-but-plausible words from context.
        """
        import random as _random

        # Use parent's __call__ for standard masking
        batch = super().__call__(examples)

        if self.corruption_ratio <= 0 or not self.confusable_pairs:
            return batch

        input_ids = batch["input_ids"]
        labels = batch["labels"]

        for i in range(len(examples)):
            word_ids = examples[i].get("word_ids")
            if word_ids is None:
                continue

            # Rebuild word groups to identify confusable masked words
            word_groups: Dict[int, List[int]] = {}
            for pos, wid in enumerate(word_ids):
                if wid is not None:
                    word_groups.setdefault(wid, []).append(pos)

            for _wid, positions in word_groups.items():
                # Check if this word is masked (label != IGNORE_LABEL_INDEX)
                if labels[i, positions[0]].item() == IGNORE_LABEL_INDEX:
                    continue

                # Decode original word from labels (which has the original tokens)
                original_tokens = [labels[i, p].item() for p in positions]
                decoded = (
                    self.tokenizer.decode(original_tokens, skip_special_tokens=True)
                    .strip()
                    .replace(" ", "")
                )

                if decoded not in self.confusable_pairs:
                    continue

                # With corruption_ratio probability, replace with variant tokens
                if _random.random() >= self.corruption_ratio:
                    continue

                variants = self.confusable_pairs[decoded]
                variant = _random.choice(variants)

                # Tokenize the variant
                variant_encoding = self.tokenizer(
                    variant,
                    add_special_tokens=False,
                    return_attention_mask=False,
                )
                variant_token_ids = variant_encoding["input_ids"]

                # Replace masked positions with variant tokens
                # If variant has different number of tokens, use min length
                replace_len = min(len(positions), len(variant_token_ids))
                for j in range(replace_len):
                    input_ids[i, positions[j]] = variant_token_ids[j]

                # If variant is shorter, keep remaining positions as [MASK]
                # If variant is longer, extra tokens are dropped (acceptable)

        batch["input_ids"] = input_ids
        return batch


class LineByLineDataset(torch.utils.data.Dataset if torch else object):  # type: ignore[misc]
    """
    Custom dataset that reads a text file line by line.

    Replaces the deprecated LineByLineTextDataset from transformers.
    """

    def __init__(
        self,
        tokenizer: "TokenizerType",
        file_path: str,
        block_size: int,
        include_word_ids: bool = False,
        pos_file_path: Optional[str] = None,
        denoising_ratio: float = 0.0,
    ):
        """
        Initialize the dataset.

        Args:
            tokenizer: HuggingFace tokenizer to encode the text.
            file_path: Path to the text file (one sentence per line).
            block_size: Maximum sequence length.
            include_word_ids: If True, capture word_ids for Whole Word Masking.
            pos_file_path: Optional path to POS sidecar JSONL (one JSON per line).
            denoising_ratio: Fraction of lines to corrupt with SyntheticErrorGenerator
                before tokenization (0.0 = disabled, 0.2 = 20% corrupted).
        """
        import random

        logger = get_logger(__name__)

        if not Path(file_path).exists():
            raise FileNotFoundError(f"Corpus file not found: {file_path}")

        # Check file size and warn for large files
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if file_size_mb > LARGE_FILE_WARNING_MB:
            logger.warning(
                f"Large corpus detected ({file_size_mb:.1f}MB). "
                "This may consume significant memory. "
                "Consider splitting into smaller files for memory efficiency."
            )

        with open(file_path, encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        if not lines:
            raise ValueError(f"No valid lines found in corpus: {file_path}")

        # Apply denoising corruption before tokenization
        if denoising_ratio > 0:
            from myspellchecker.training.generator import SyntheticErrorGenerator

            gen = SyntheticErrorGenerator(seed=42)
            rng = random.Random(42)
            corrupted_count = 0
            for i in range(len(lines)):
                if rng.random() < denoising_ratio:
                    result = gen.generate_one(lines[i])
                    if result is not None:
                        lines[i] = result[0]  # corrupted_sentence
                        corrupted_count += 1
            logger.info(f"Denoising: corrupted {corrupted_count:,}/{len(lines):,} lines")

        # Load POS sidecar if provided
        self.pos_data: Optional[List[List[str]]] = None
        if pos_file_path:
            with open(pos_file_path, encoding="utf-8") as f:
                self.pos_data = [json.loads(line)["pos"] for line in f if line.strip()]
            if len(self.pos_data) != len(lines):
                logger.warning(
                    f"POS sidecar has {len(self.pos_data)} entries but corpus has "
                    f"{len(lines)} lines. POS weighting will be disabled for mismatched lines."
                )

        # Tokenize all lines at once for efficiency
        batch_encoding = tokenizer(
            lines,
            add_special_tokens=True,
            truncation=True,
            max_length=block_size,
            padding="max_length",
            return_attention_mask=True,
        )
        self.input_ids = batch_encoding["input_ids"]
        self.attention_masks = batch_encoding["attention_mask"]

        # Capture word_ids for Whole Word Masking
        self.word_ids_list: Optional[List[List[Optional[int]]]] = None
        if include_word_ids:
            self.word_ids_list = [batch_encoding.word_ids(i) for i in range(len(lines))]

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item: Dict[str, Any] = {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_masks[idx], dtype=torch.long),
        }
        if self.word_ids_list is not None:
            item["word_ids"] = self.word_ids_list[idx]
        if self.pos_data is not None and idx < len(self.pos_data):
            item["pos_tags"] = self.pos_data[idx]
        return item


class LineByLineIterableDataset(torch.utils.data.IterableDataset if torch else object):  # type: ignore[misc]
    """Streaming line-by-line dataset for large corpora.

    Reads and tokenizes one line at a time during iteration.
    Memory stays constant regardless of corpus size.
    """

    def __init__(
        self,
        tokenizer: "TokenizerType",
        file_path: Union[str, List[str]],
        block_size: int,
        include_word_ids: bool = False,
        pos_file_path: Optional[str] = None,
        denoising_ratio: float = 0.0,
    ):
        if isinstance(file_path, str):
            self.file_paths = [file_path]
        else:
            self.file_paths = list(file_path)
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.include_word_ids = include_word_ids
        self.pos_file_path = pos_file_path
        self.denoising_ratio = denoising_ratio
        self._error_generator: Any = None
        self._denoise_rng: Any = None

    def _get_error_generator(self):
        """Lazily create error generator for denoising (import inside worker)."""
        if self._error_generator is None:
            import random

            from myspellchecker.training.generator import SyntheticErrorGenerator

            self._error_generator = SyntheticErrorGenerator(seed=random.randint(0, 2**31))
            self._denoise_rng = random.Random()
        return self._error_generator

    def _maybe_corrupt(self, text: str) -> str:
        """With probability denoising_ratio, corrupt the text using SyntheticErrorGenerator."""
        if self.denoising_ratio <= 0:
            return text
        gen = self._get_error_generator()
        if self._denoise_rng.random() < self.denoising_ratio:
            result = gen.generate_one(text)
            if result is not None:
                corrupted_sentence, _, _ = result
                return str(corrupted_sentence)
        return text

    def __iter__(self):
        # Shard lines across DDP ranks AND DataLoader workers.
        # Under torchrun, RANK/WORLD_SIZE identify the DDP process.
        # Within each process, worker_info splits across DataLoader workers.
        # Total shards = world_size * num_workers; each shard gets unique lines.
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0

        # DDP rank sharding (torchrun sets RANK and WORLD_SIZE)
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        total_shards = world_size * num_workers
        shard_id = rank * num_workers + worker_id

        # Open POS sidecar if provided (only for the first file_path)
        pos_iter = None
        if self.pos_file_path:
            pos_iter = open(self.pos_file_path, encoding="utf-8")

        try:
            line_idx = 0
            for path in self.file_paths:
                with open(path, encoding="utf-8") as f:
                    for line in f:
                        # Read POS line in sync (even if we skip this corpus line)
                        pos_line = None
                        if pos_iter is not None:
                            pos_raw = pos_iter.readline()
                            if pos_raw.strip():
                                pos_line = pos_raw

                        line = line.strip()
                        if not line:
                            continue
                        # Round-robin across DDP ranks * DataLoader workers
                        if line_idx % total_shards != shard_id:
                            line_idx += 1
                            continue
                        line_idx += 1

                        # Apply denoising corruption before tokenization
                        text = self._maybe_corrupt(line)

                        encoding = self.tokenizer(
                            text,
                            add_special_tokens=True,
                            truncation=True,
                            max_length=self.block_size,
                            padding="max_length",
                            return_attention_mask=True,
                        )
                        item: Dict[str, Any] = {
                            "input_ids": torch.tensor(encoding["input_ids"], dtype=torch.long),
                            "attention_mask": torch.tensor(
                                encoding["attention_mask"], dtype=torch.long
                            ),
                        }
                        if self.include_word_ids:
                            item["word_ids"] = encoding.word_ids()
                        if pos_line is not None:
                            item["pos_tags"] = json.loads(pos_line)["pos"]
                        yield item
        finally:
            if pos_iter is not None:
                pos_iter.close()


def _fix_tokenizer_config(model_dir: str) -> None:
    """Fix tokenizer_config.json written by save_pretrained().

    HuggingFace tokenizers writes ``"tokenizer_class": "TokenizersBackend"``
    which causes ``AutoTokenizer.from_pretrained()`` to silently fall back to
    a slow tokenizer.  Replace it with the correct class name.
    """
    cfg_path = os.path.join(model_dir, "tokenizer_config.json")
    if not os.path.isfile(cfg_path):
        return
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    if cfg.get("tokenizer_class") in (None, "TokenizersBackend"):
        cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)


class ModelTrainer:
    """
    Wrapper class for training Custom Semantic Models.
    """

    def __init__(self):
        self.logger = get_logger(__name__)
        self._check_dependencies()

    def _check_dependencies(self):
        if torch is None or ByteLevelBPETokenizer is None:
            raise ImportError(
                "Training requires 'transformers', 'tokenizers', and 'torch'. "
                "Please install them using: pip install 'myspellchecker[train]'"
            )

    def train_tokenizer(
        self,
        corpus_path: str,
        output_dir: str,
        vocab_size: int = 30_000,
        min_frequency: int = 2,
        word_boundary_aware: bool = False,
    ) -> str:
        """
        Train a Byte-Level BPE tokenizer on the corpus.

        Args:
            corpus_path: Path to the text file.
            output_dir: Directory to save the tokenizer.json.
            vocab_size: Maximum vocabulary size.
            min_frequency: Minimum frequency for a token to be included.
            word_boundary_aware: If True, use WhitespaceSplit pre-tokenizer so
                BPE merges never cross word boundaries. Requires the corpus to
                be pre-segmented with spaces between words (e.g., via myword).

        Returns:
            Path to the saved tokenizer.json.
        """
        self.logger.info(f"Training Tokenizer on {corpus_path}...")
        os.makedirs(output_dir, exist_ok=True)

        if word_boundary_aware:
            tokenizer_path = self._train_word_boundary_tokenizer(
                corpus_path, output_dir, vocab_size, min_frequency
            )
        else:
            tokenizer_path = self._train_byte_level_tokenizer(
                corpus_path, output_dir, vocab_size, min_frequency
            )

        self.logger.info(f"Tokenizer saved to {tokenizer_path}")
        return tokenizer_path

    def _train_byte_level_tokenizer(
        self,
        corpus_path: str,
        output_dir: str,
        vocab_size: int,
        min_frequency: int,
    ) -> str:
        """Train standard ByteLevel BPE (no word boundary constraints)."""
        tokenizer = ByteLevelBPETokenizer()
        tokenizer.train(
            files=[corpus_path],
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=SPECIAL_TOKENS,
        )
        tokenizer_path = os.path.join(output_dir, DEFAULT_TOKENIZER_FILE)
        tokenizer.save(tokenizer_path)
        return tokenizer_path

    def _train_word_boundary_tokenizer(
        self,
        corpus_path: str,
        output_dir: str,
        vocab_size: int,
        min_frequency: int,
    ) -> str:
        """Train BPE with WhitespaceSplit pre-tokenizer for word-boundary awareness.

        Uses the lower-level `tokenizers` API to configure a pre-tokenizer that
        splits on whitespace BEFORE BPE. This ensures BPE merges only happen
        within words, never across word boundaries. The corpus must be
        pre-segmented with spaces between Myanmar words (e.g., via myword).

        Result: each Myanmar word maps to 1-3 BPE tokens, all contained within
        the word boundary. This aligns with inference where myword segments
        input text before tokenization.
        """
        from tokenizers import Tokenizer, models, pre_tokenizers, trainers

        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=SPECIAL_TOKENS,
        )

        self.logger.info(
            "Training word-boundary BPE (WhitespaceSplit pre-tokenizer) — "
            "merges will never cross word boundaries"
        )
        tokenizer.train(files=[corpus_path], trainer=trainer)

        tokenizer_path = os.path.join(output_dir, DEFAULT_TOKENIZER_FILE)
        tokenizer.save(tokenizer_path)
        return tokenizer_path

    def train_model(
        self,
        corpus_path: str,
        tokenizer_path: str,
        output_dir: str,
        epochs: int = 5,
        batch_size: int = 16,
        max_length: int = 128,
        learning_rate: float = 5e-5,
        hidden_size: int = 256,
        num_hidden_layers: int = 4,
        num_attention_heads: int = 4,
        architecture: ModelArchitecture = ModelArchitecture.ROBERTA,
        resume_from_checkpoint: Optional[str] = None,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        save_metrics: bool = True,
        max_steps: Optional[int] = None,
        save_steps: Optional[int] = None,
        streaming: bool = False,
        checkpoint_dir: Optional[str] = None,
        whole_word_masking: bool = False,
        pos_file: Optional[str] = None,
        denoising_ratio: float = 0.0,
        fp16: bool = False,
        gradient_accumulation_steps: int = 1,
        reporter: "TrainingReporter | None" = None,
        confusable_masking: bool = False,
        confusable_mask_ratio: float = 0.3,
        confusable_words_file: Optional[str] = None,
        lr_scheduler_type: str = "linear",
        corruption_ratio: float = 0.0,
        mlm_probability: float = DEFAULT_MLM_PROBABILITY,
        embedding_surgery: bool = False,
        embedding_warmup_steps: int = DEFAULT_EMBEDDING_WARMUP_STEPS,
        embedding_lr: float = DEFAULT_EMBEDDING_LR,
    ) -> str:
        """
        Train a Masked Language Model (RoBERTa or BERT).

        Args:
            corpus_path: Path to training data.
            tokenizer_path: Path to the trained tokenizer.json.
            output_dir: Directory to save the model.
            epochs: Number of training epochs.
            batch_size: Batch size per device.
            max_length: Max sequence length (context window).
            learning_rate: Learning rate.
            hidden_size: Size of hidden layers (smaller = faster).
            num_hidden_layers: Number of transformer layers.
            num_attention_heads: Number of attention heads.
            architecture: Model architecture (ROBERTA or BERT).
            resume_from_checkpoint: Path to checkpoint directory to resume from.
            warmup_ratio: Ratio of training steps for learning rate warmup.
            weight_decay: Weight decay for optimizer (AdamW).
            save_metrics: Whether to save training metrics to JSON file.
            reporter: Optional TrainingReporter for Rich progress display.

        Returns:
            Path to the saved model directory.

        Raises:
            ValueError: If training data is insufficient or architecture unsupported.
            FileNotFoundError: If corpus or tokenizer files don't exist.

        Example:
            >>> trainer = ModelTrainer()
            >>> model_path = trainer.train_model(
            ...     corpus_path="corpus.txt",
            ...     tokenizer_path="tokenizer.json",
            ...     output_dir="./output",
            ...     architecture=ModelArchitecture.ROBERTA,
            ...     resume_from_checkpoint="./output/checkpoints/checkpoint-500",
            ... )
        """
        self.logger.info("Initializing Model Training...")
        os.makedirs(output_dir, exist_ok=True)

        # Handle architecture as string
        if isinstance(architecture, str):
            architecture = ModelArchitecture.from_string(architecture)

        self.logger.info(f"Architecture: {architecture.value}")

        # 1. Load Tokenizer into Transformers wrapper
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=tokenizer_path,
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
            sep_token="</s>",
            pad_token="<pad>",
            cls_token="<s>",
            mask_token="<mask>",
        )

        # 2. Define Model Architecture (or load from pre-trained)
        # Auto-detect: if resume_from_checkpoint has trainer_state.json,
        # it's a training checkpoint (HF Trainer resume). Otherwise, it's
        # a pre-trained model export (load weights, new optimizer).
        pretrained_path = None
        training_checkpoint = None
        if resume_from_checkpoint and os.path.isdir(resume_from_checkpoint):
            trainer_state = os.path.join(resume_from_checkpoint, "trainer_state.json")
            if os.path.isfile(trainer_state):
                training_checkpoint = resume_from_checkpoint
                self.logger.info(f"Will resume training from checkpoint: {resume_from_checkpoint}")
            else:
                pretrained_path = resume_from_checkpoint
                self.logger.info(f"Loading pre-trained model from: {resume_from_checkpoint}")

        if pretrained_path:
            if architecture == ModelArchitecture.BERT:
                model = BertForMaskedLM.from_pretrained(pretrained_path)
            else:
                model = RobertaForMaskedLM.from_pretrained(pretrained_path)
            self.logger.info(
                f"Loaded pre-trained {architecture.value} model "
                f"({sum(p.numel() for p in model.parameters()):,} params)"
            )
            # Embedding surgery: resize for new tokenizer vocab
            old_vocab_size = model.config.vocab_size
            new_vocab_size = tokenizer.vocab_size
            if embedding_surgery and old_vocab_size != new_vocab_size:
                self.logger.info(
                    f"Embedding surgery: resizing {old_vocab_size} → {new_vocab_size} tokens"
                )
                model.resize_token_embeddings(new_vocab_size)
        else:
            model = self._create_model(
                architecture=architecture,
                vocab_size=tokenizer.vocab_size,
                max_length=max_length,
                hidden_size=hidden_size,
                num_hidden_layers=num_hidden_layers,
                num_attention_heads=num_attention_heads,
            )

        # 3. Prepare Dataset
        if streaming:
            dataset = LineByLineIterableDataset(
                tokenizer=tokenizer,
                file_path=corpus_path,
                block_size=max_length,
                include_word_ids=whole_word_masking,
                pos_file_path=pos_file if whole_word_masking else None,
                denoising_ratio=denoising_ratio,
            )
            self.logger.info("Using streaming dataset (IterableDataset)")
        else:
            dataset = LineByLineDataset(
                tokenizer=tokenizer,
                file_path=corpus_path,
                block_size=max_length,
                include_word_ids=whole_word_masking,
                pos_file_path=pos_file if whole_word_masking else None,
                denoising_ratio=denoising_ratio,
            )

            # Validate minimum dataset size (only for map-style dataset)
            if len(dataset) < MIN_TRAINING_LINES:
                raise ValueError(
                    f"Insufficient training data: {len(dataset)} lines. "
                    f"Minimum required: {MIN_TRAINING_LINES}. "
                    "Please provide a larger corpus for meaningful model training."
                )

            self.logger.info(f"Training dataset: {len(dataset)} lines")

        data_collator: Any
        if whole_word_masking and confusable_masking:
            confusable_words = ConfusableAwareMaskCollator.load_confusable_words(
                confusable_words_file
            )
            confusable_pairs: Dict[str, List[str]] = {}
            if corruption_ratio > 0:
                confusable_pairs = ConfusableAwareMaskCollator.load_confusable_pairs(
                    confusable_words_file
                )
            data_collator = ConfusableAwareMaskCollator(
                tokenizer=tokenizer,
                confusable_words=confusable_words,
                confusable_mask_ratio=confusable_mask_ratio,
                mlm_probability=mlm_probability,
                confusable_pairs=confusable_pairs,
                corruption_ratio=corruption_ratio,
            )
            self.logger.info(
                f"Using Confusable-Aware Masking collator "
                f"(ratio={confusable_mask_ratio}, words={len(confusable_words)}, "
                f"corruption={corruption_ratio}, mlm_prob={mlm_probability})"
            )
        elif whole_word_masking:
            data_collator = WholeWordMaskCollator(
                tokenizer=tokenizer,
                mlm_probability=mlm_probability,
            )
            self.logger.info(
                f"Using Whole Word Masking collator "
                f"(POS-weighted={pos_file is not None}, mlm_prob={mlm_probability})"
            )
        else:
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer, mlm=True, mlm_probability=mlm_probability
            )

        # 4. Setup Training Arguments with learning rate scheduler
        if torch.cuda.is_available():
            self.logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
        else:
            self.logger.info("GPU not available, training on CPU (this may be slow)")

        if checkpoint_dir:
            ckpt_dir = checkpoint_dir
        else:
            ckpt_dir = os.path.join(output_dir, "checkpoints")

        # Scale save_steps and logging_steps for large runs
        if save_steps is None:
            save_steps, logging_steps = compute_save_steps(max_steps)
        else:
            logging_steps = max(50, save_steps // 10)

        num_workers = get_dataloader_workers(torch.cuda.is_available())

        use_fp16 = fp16 and torch.cuda.is_available()
        if use_fp16:
            self.logger.info("Using fp16 mixed precision training")

        training_args = TrainingArguments(
            output_dir=ckpt_dir,
            num_train_epochs=epochs,
            max_steps=max_steps if max_steps is not None else -1,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            save_steps=save_steps,
            save_total_limit=DEFAULT_SAVE_TOTAL_LIMIT,
            prediction_loss_only=True,
            learning_rate=learning_rate,
            warmup_ratio=warmup_ratio,
            weight_decay=weight_decay,
            fp16=use_fp16,
            use_cpu=not torch.cuda.is_available(),
            logging_steps=logging_steps,
            logging_dir=os.path.join(output_dir, "logs"),
            lr_scheduler_type=lr_scheduler_type,
            load_best_model_at_end=False,
            report_to=["none"],
            dataloader_num_workers=num_workers,
            dataloader_pin_memory=torch.cuda.is_available(),
        )

        # 5. Setup Callbacks
        callbacks: list[Any] = []
        if save_metrics:
            callbacks.append(TrainingMetricsCallback(output_dir))
        callbacks.append(TokenizerSaveCallback(tokenizer))
        if reporter is not None:
            from myspellchecker.training.reporter import RichProgressCallback

            callbacks.append(RichProgressCallback(reporter))

        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset,
            callbacks=callbacks,
        )

        # 6a. Embedding surgery Phase 1: freeze body, warm up new embeddings
        if (
            embedding_surgery
            and pretrained_path
            and not training_checkpoint
            and old_vocab_size != new_vocab_size
        ):
            self.logger.info(
                f"Phase 1: Warming up new embeddings "
                f"({embedding_warmup_steps} steps, lr={embedding_lr})"
            )
            # Freeze all except embeddings and LM head
            for name, param in model.named_parameters():
                is_embedding = "embeddings" in name or "lm_head" in name
                param.requires_grad = is_embedding

            frozen = sum(1 for _, p in model.named_parameters() if not p.requires_grad)
            trainable = sum(1 for _, p in model.named_parameters() if p.requires_grad)
            self.logger.info(f"Frozen: {frozen} params, Trainable: {trainable} params")

            warmup_args = TrainingArguments(
                output_dir=os.path.join(ckpt_dir, "embedding_warmup"),
                max_steps=embedding_warmup_steps,
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                learning_rate=embedding_lr,
                warmup_ratio=0.05,
                weight_decay=0.0,
                fp16=use_fp16,
                use_cpu=not torch.cuda.is_available(),
                logging_steps=max(50, embedding_warmup_steps // 20),
                save_steps=embedding_warmup_steps,
                save_total_limit=1,
                prediction_loss_only=True,
                lr_scheduler_type="cosine",
                report_to=["none"],
                dataloader_num_workers=num_workers,
                dataloader_pin_memory=torch.cuda.is_available(),
            )
            warmup_trainer = Trainer(
                model=model,
                args=warmup_args,
                data_collator=data_collator,
                train_dataset=dataset,
                callbacks=[TokenizerSaveCallback(tokenizer)],
            )
            warmup_trainer.train()
            self.logger.info("Phase 1 complete. Unfreezing all parameters for Phase 2.")

            # Unfreeze all for Phase 2
            for param in model.parameters():
                param.requires_grad = True

            # Recreate trainer with fresh optimizer for Phase 2
            trainer = Trainer(
                model=model,
                args=training_args,
                data_collator=data_collator,
                train_dataset=dataset,
                callbacks=callbacks,
            )

        # 6b. Train (Phase 2 if surgery, or standard training)
        if training_checkpoint:
            self.logger.info(f"Resuming from training checkpoint: {training_checkpoint}")
            trainer.train(resume_from_checkpoint=training_checkpoint)
        else:
            if embedding_surgery and pretrained_path and old_vocab_size != new_vocab_size:
                self.logger.info("Phase 2: Full fine-tuning with all parameters...")
            elif pretrained_path:
                self.logger.info("Fine-tuning from pre-trained model (new optimizer)...")
            else:
                self.logger.info("Starting Training (this may take a while)...")
            trainer.train()

        # 7. Save Final Model (rank 0 only under DDP; Trainer.save_model handles this)
        final_output = os.path.join(output_dir, "pytorch_model")
        trainer.save_model(final_output)
        is_main = int(os.environ.get("RANK", "0")) == 0
        if is_main:
            tokenizer.save_pretrained(final_output)
            _fix_tokenizer_config(final_output)

        self.logger.info(f"PyTorch Model saved to {final_output}")
        return final_output

    def _create_model(
        self,
        architecture: ModelArchitecture,
        vocab_size: int,
        max_length: int,
        hidden_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
    ) -> "torch.nn.Module":
        """
        Create a model based on the specified architecture.

        Args:
            architecture: Model architecture type.
            vocab_size: Vocabulary size from tokenizer.
            max_length: Maximum sequence length.
            hidden_size: Size of hidden layers.
            num_hidden_layers: Number of transformer layers.
            num_attention_heads: Number of attention heads.

        Returns:
            Initialized model ready for training.
        """
        if architecture == ModelArchitecture.ROBERTA:
            config = RobertaConfig(
                vocab_size=vocab_size,
                max_position_embeddings=max_length + 2,
                num_attention_heads=num_attention_heads,
                num_hidden_layers=num_hidden_layers,
                hidden_size=hidden_size,
                type_vocab_size=1,
            )
            model = RobertaForMaskedLM(config)
            self.logger.info(f"Created RoBERTa model: {config}")
        elif architecture == ModelArchitecture.BERT:
            config = BertConfig(  # type: ignore[assignment]
                vocab_size=vocab_size,
                max_position_embeddings=max_length + 2,
                num_attention_heads=num_attention_heads,
                num_hidden_layers=num_hidden_layers,
                hidden_size=hidden_size,
                type_vocab_size=2,
            )
            model = BertForMaskedLM(config)  # type: ignore[assignment]
            self.logger.info(f"Created BERT model: {config}")
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

        return model
