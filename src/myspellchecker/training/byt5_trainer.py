"""ByT5 fine-tuning for Myanmar spell correction.

Fine-tunes google/byt5-small on seq2seq correction pairs.

Usage:
    # Basic training (auto-detects MPS/CUDA/CPU)
    python -m myspellchecker.training.byt5_trainer \
        --data-dir data/byt5_training \
        --output-dir models/byt5-myanmar-spell

    # With custom hyperparameters
    python -m myspellchecker.training.byt5_trainer \
        --data-dir data/byt5_training \
        --output-dir models/byt5-myanmar-spell \
        --epochs 10 \
        --batch-size 8 \
        --lr 3e-4 \
        --max-input-length 512 \
        --max-target-length 512

    # Resume from checkpoint
    python -m myspellchecker.training.byt5_trainer \
        --data-dir data/byt5_training \
        --output-dir models/byt5-myanmar-spell \
        --resume
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

MODEL_NAME = "google/byt5-small"


class CorrectionDataset(Dataset):
    """Seq2seq correction dataset from JSONL files.

    Each line is a JSON object with "input" and "output" fields,
    where "input" starts with the task prefix "correct: ".
    """

    def __init__(
        self,
        path: Path,
        tokenizer: AutoTokenizer,
        max_input_length: int = 512,
        max_target_length: int = 512,
    ):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.examples: list[dict[str, str]] = []

        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    self.examples.append(
                        {
                            "input": entry["input"],
                            "output": entry["output"],
                        }
                    )
                except (json.JSONDecodeError, KeyError):
                    continue

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        example = self.examples[idx]

        model_inputs = self.tokenizer(
            example["input"],
            max_length=self.max_input_length,
            truncation=True,
            padding=False,
        )

        labels = self.tokenizer(
            example["output"],
            max_length=self.max_target_length,
            truncation=True,
            padding=False,
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


def detect_device() -> str:
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings.

    Uses O(min(m,n)) space with two-row DP.
    """
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    if not s2:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            curr_row.append(
                min(
                    curr_row[j] + 1,  # insertion
                    prev_row[j + 1] + 1,  # deletion
                    prev_row[j] + cost,  # substitution
                )
            )
        prev_row = curr_row

    return prev_row[-1]


def compute_metrics_fn(tokenizer):
    """Return a compute_metrics function for the trainer."""
    import numpy as np

    def compute_metrics(eval_preds):
        preds, labels = eval_preds

        # Replace -100 with pad token id for decoding
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Exact match accuracy
        exact_match = sum(
            p.strip() == ref.strip() for p, ref in zip(decoded_preds, decoded_labels, strict=True)
        ) / len(decoded_preds)

        # Character Error Rate via Levenshtein edit distance
        total_chars = 0
        total_errors = 0
        for pred, label in zip(decoded_preds, decoded_labels, strict=True):
            pred_s = pred.strip()
            label_s = label.strip()
            total_chars += max(len(label_s), 1)
            total_errors += _levenshtein_distance(pred_s, label_s)

        cer = total_errors / max(total_chars, 1)

        return {
            "exact_match": round(exact_match, 4),
            "cer": round(cer, 4),
        }

    return compute_metrics


def _get_last_checkpoint(output_dir: Path) -> str | None:
    """Find the latest checkpoint by step number (not lexicographic order)."""
    checkpoints = list(output_dir.glob("checkpoint-*"))
    if not checkpoints:
        return None
    # Sort by step number extracted from directory name (B4 fix)
    checkpoints.sort(key=lambda p: int(p.name.split("-")[-1]))
    return str(checkpoints[-1])


def main():
    parser = argparse.ArgumentParser(description="Fine-tune ByT5 for Myanmar spell correction")
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory with train/dev/test JSONL",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for model",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_NAME,
        help=f"Base model (default: {MODEL_NAME})",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="Eval batch size (default: same as --batch-size)",
    )
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-input-length", type=int, default=256)
    parser.add_argument("--max-target-length", type=int, default=256)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--early-stopping",
        type=int,
        default=3,
        help="Patience (0=disabled)",
    )
    parser.add_argument("--beam-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Verify data directory
    train_path = args.data_dir / "train.jsonl"
    dev_path = args.data_dir / "dev.jsonl"
    if not train_path.exists():
        print(f"Error: Training data not found: {train_path}", file=sys.stderr)
        sys.exit(1)
    if not dev_path.exists():
        print(f"Error: Dev data not found: {dev_path}", file=sys.stderr)
        sys.exit(1)

    eval_batch_size = args.eval_batch_size or args.batch_size

    device = detect_device()
    print(f"{'=' * 60}")
    print("  ByT5 Fine-Tuning for Myanmar Spell Correction")
    print(f"{'=' * 60}")
    print(f"  Model: {args.model}")
    print(f"  Device: {device}")
    print(f"  Data: {args.data_dir}")
    print(f"  Output: {args.output_dir}")
    print(
        f"  Epochs: {args.epochs}, Batch: {args.batch_size}"
        f" (eval: {eval_batch_size}), LR: {args.lr}"
    )
    print(f"  Max lengths: input={args.max_input_length}, target={args.max_target_length}")
    print(f"  Gradient checkpointing: {args.gradient_checkpointing}")

    # Load tokenizer and model
    print(f"\n  Loading tokenizer and model from {args.model}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    print(
        f"  Loaded in {time.time() - t0:.1f}s "
        f"({sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params)"
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("  Gradient checkpointing: enabled")

    # Load datasets
    print("\n  Loading datasets...")
    train_dataset = CorrectionDataset(
        train_path,
        tokenizer,
        max_input_length=args.max_input_length,
        max_target_length=args.max_target_length,
    )
    dev_dataset = CorrectionDataset(
        dev_path,
        tokenizer,
        max_input_length=args.max_input_length,
        max_target_length=args.max_target_length,
    )
    print(f"  Train: {len(train_dataset)} examples")
    print(f"  Dev: {len(dev_dataset)} examples")

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        max_length=args.max_input_length,
        pad_to_multiple_of=8,
    )

    # Effective batch size
    effective_batch = args.batch_size * args.gradient_accumulation
    steps_per_epoch = len(train_dataset) // effective_batch
    total_steps = steps_per_epoch * args.epochs
    print(f"  Effective batch size: {effective_batch}")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Total steps: {total_steps}")

    # Training arguments
    use_fp16 = args.fp16 and device == "cuda"

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        fp16=use_fp16,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=args.max_target_length,
        generation_num_beams=args.beam_size,
        logging_steps=100,
        logging_first_step=True,
        report_to="none",
        seed=args.seed,
        dataloader_num_workers=0 if device == "mps" else 2,
        remove_unused_columns=False,
    )

    # Callbacks
    callbacks = []
    if args.early_stopping > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping))

    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn(tokenizer),
        callbacks=callbacks,
    )

    # Train
    print("\n  Starting training...")
    resume_checkpoint = None
    if args.resume:
        resume_checkpoint = _get_last_checkpoint(args.output_dir)
        if resume_checkpoint:
            print(f"  Resuming from: {resume_checkpoint}")

    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)

    # Save final model
    print(f"\n  Saving final model to {args.output_dir}...")
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    # W2: Clean up checkpoint directories to avoid bloating model.tar.gz
    for ckpt_dir in args.output_dir.glob("checkpoint-*"):
        shutil.rmtree(ckpt_dir)
        print(f"  Cleaned up: {ckpt_dir.name}")

    # Save training metrics
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # Final evaluation
    print("\n  Running final evaluation...")
    eval_metrics = trainer.evaluate()
    eval_metrics["eval_samples"] = len(dev_dataset)
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    print(f"\n{'=' * 60}")
    print("  Training complete!")
    print(f"  Final eval CER: {eval_metrics.get('eval_cer', 'N/A')}")
    print(f"  Final eval exact match: {eval_metrics.get('eval_exact_match', 'N/A')}")
    print(f"  Model saved to: {args.output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
