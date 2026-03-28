"""Reranker MLP trainer.

Trains a small feature-based MLP with cross-entropy (listwise) loss to re-rank
spell-checker suggestion lists. This module is used exclusively during the
**offline training pipeline** -- it is NOT used at inference time.

Pipeline context:
    - **Upstream**: Reads JSONL training data produced by
      ``RerankerDataGenerator`` in ``reranker_data.py``. Each JSONL line
      contains 20 features per candidate plus a gold correction index.
    - **Downstream**: Exports an ONNX model (``reranker.onnx``) plus a
      ``reranker.onnx.stats.json`` file with per-feature normalization
      statistics (mean/std computed via Welford's online algorithm).
    - **At inference**: The exported ONNX model is loaded by
      ``algorithms/ranker.py`` to re-score suggestions. This trainer module
      itself is not involved at inference time.

Architecture:
    - ``RerankerMLP``: Linear(20, 64) -> ReLU -> Dropout(0.1) -> Linear(64, 1).
      Input shape: (batch, num_candidates, 20). Output: scalar score per candidate.
      Total parameters: ~5K. Inference overhead: negligible (~0ms on CPU).
    - Training uses listwise cross-entropy loss (scores over candidates treated
      as a classification problem with the gold index as the target).
    - Early stopping on validation Top-1 accuracy with configurable patience.
    - Stratified train/val split by error_type when no separate validation set
      is provided.

Key classes:
    - ``RerankerTrainer``: Orchestrates data loading, normalization, training
      loop, early stopping, ONNX export, and metrics saving.
    - ``RerankerMLP``: PyTorch ``nn.Module`` for the scoring MLP.
    - ``RerankerDataset``: PyTorch ``Dataset`` that loads JSONL examples,
      pads candidates to a fixed width, and constructs feature/mask tensors.
    - ``TrainingMetrics``: Dataclass collecting loss curves, Top-1, and MRR
      per epoch.

Dependencies:
    Requires PyTorch (``pip install 'myspellchecker[train]'``). Raises
    ``ImportError`` at init time if torch is not available.

Usage (CLI):
    python -m myspellchecker.training.reranker_trainer \\
        --train data/reranker_training_100k.jsonl \\
        --output models/reranker-v1/ \\
        --epochs 20 --lr 1e-3 --batch-size 64

Usage (API):
    >>> from myspellchecker.training.reranker_trainer import RerankerTrainer
    >>> trainer = RerankerTrainer("data/train.jsonl")
    >>> metrics = trainer.train(epochs=20)
    >>> trainer.export_onnx("models/reranker-v1/reranker.onnx")
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from myspellchecker.training.reranker_data import FEATURE_NAMES, NUM_FEATURES
from myspellchecker.utils.logging_utils import get_logger

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset

    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]
    Dataset = object  # type: ignore[assignment]
    TORCH_AVAILABLE = False

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_HIDDEN_DIM = 64
DEFAULT_DROPOUT = 0.1
DEFAULT_MAX_CANDIDATES = 20
DEFAULT_EPOCHS = 20
DEFAULT_LR = 1e-3
DEFAULT_BATCH_SIZE = 64
DEFAULT_PATIENCE = 5
DEFAULT_VAL_RATIO = 0.2
ONNX_OPSET_VERSION = 18


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class TrainingMetrics:
    """Metrics collected during a training run."""

    train_losses: list[float] = field(default_factory=list)
    val_top1_accuracies: list[float] = field(default_factory=list)
    val_mrrs: list[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_top1: float = 0.0
    best_val_mrr: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_losses": self.train_losses,
            "val_top1_accuracies": self.val_top1_accuracies,
            "val_mrrs": self.val_mrrs,
            "best_epoch": self.best_epoch,
            "best_val_top1": self.best_val_top1,
            "best_val_mrr": self.best_val_mrr,
        }


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class RerankerMLP(nn.Module if TORCH_AVAILABLE else object):  # type: ignore[misc]
    """Small MLP for suggestion reranking.

    Architecture: Linear(NUM_FEATURES, 64) -> ReLU -> Dropout -> Linear(64, 1)

    Input:  (batch, num_candidates, NUM_FEATURES)  # NUM_FEATURES=20
    Output: (batch, num_candidates) -- scalar score per candidate
    """

    def __init__(
        self,
        input_dim: int = NUM_FEATURES,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        dropout: float = DEFAULT_DROPOUT,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Score each candidate.

        Args:
            x: Feature tensor of shape (batch, num_candidates, input_dim).

        Returns:
            Scores of shape (batch, num_candidates).
        """
        # (batch, num_candidates, 1) -> (batch, num_candidates)
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class RerankerDataset(Dataset if TORCH_AVAILABLE else object):  # type: ignore[misc]
    """Dataset for reranker training data.

    Reads JSONL, pads candidates to ``max_candidates``, and constructs
    feature tensors plus masks.
    """

    def __init__(
        self,
        examples: list[dict[str, Any]],
        max_candidates: int = DEFAULT_MAX_CANDIDATES,
    ):
        self.max_candidates = max_candidates
        self.features: list[torch.Tensor] = []
        self.gold_indices: list[int] = []
        self.masks: list[torch.Tensor] = []

        for ex in examples:
            feats = ex["features"]
            gold_idx = ex["gold_index"]
            n_cands = len(feats)

            # Clamp to max_candidates
            if n_cands > max_candidates:
                # If gold falls beyond max_candidates, skip
                if gold_idx >= max_candidates:
                    continue
                feats = feats[:max_candidates]
                n_cands = max_candidates

            # Pad features to (max_candidates, NUM_FEATURES)
            padded = feats + [[0.0] * NUM_FEATURES] * (max_candidates - n_cands)
            mask = [1.0] * n_cands + [0.0] * (max_candidates - n_cands)

            self.features.append(torch.tensor(padded, dtype=torch.float32))
            self.gold_indices.append(gold_idx)
            self.masks.append(torch.tensor(mask, dtype=torch.bool))

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> dict[str, "torch.Tensor"]:
        return {
            "features": self.features[idx],
            "gold_index": torch.tensor(self.gold_indices[idx], dtype=torch.long),
            "mask": self.masks[idx],
        }


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class RerankerTrainer:
    """Trains a RerankerMLP on JSONL data with early stopping.

    Args:
        train_path: Path to JSONL training data.
        val_path: Optional path to JSONL validation data.  When ``None``,
            splits training data using ``val_ratio``.
        val_ratio: Fraction of training data to use as validation
            (only when ``val_path`` is ``None``).
        max_candidates: Fixed candidate-list length for padding.
        hidden_dim: MLP hidden layer width.
        dropout: Dropout probability.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        train_path: str,
        val_path: str | None = None,
        val_ratio: float = DEFAULT_VAL_RATIO,
        max_candidates: int = DEFAULT_MAX_CANDIDATES,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        dropout: float = DEFAULT_DROPOUT,
        seed: int = 42,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError(
                "RerankerTrainer requires PyTorch. "
                "Install with: pip install 'myspellchecker[train]'"
            )

        self.max_candidates = max_candidates
        self.seed = seed
        torch.manual_seed(seed)

        # Load data
        train_examples = self._load_jsonl(train_path)
        logger.info(
            "Loaded %d training examples from %s",
            len(train_examples),
            train_path,
        )

        if val_path is not None:
            val_examples = self._load_jsonl(val_path)
            logger.info(
                "Loaded %d validation examples from %s",
                len(val_examples),
                val_path,
            )
        else:
            train_examples, val_examples = self._split_data(train_examples, val_ratio, seed)
            logger.info(
                "Split data: %d train, %d val (ratio=%.2f)",
                len(train_examples),
                len(val_examples),
                val_ratio,
            )

        self.train_dataset = RerankerDataset(train_examples, max_candidates)
        self.val_dataset = RerankerDataset(val_examples, max_candidates)

        # Compute normalization stats from training data
        self.feature_means, self.feature_stds = self._compute_norm_stats(self.train_dataset)
        logger.info("Computed normalization stats for %d features", NUM_FEATURES)

        # Apply normalization
        self._normalize_dataset(self.train_dataset)
        self._normalize_dataset(self.val_dataset)

        # Build model
        self.model = RerankerMLP(
            input_dim=NUM_FEATURES,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        param_count = sum(p.numel() for p in self.model.parameters())
        logger.info(
            "RerankerMLP: %d parameters (hidden_dim=%d, dropout=%.2f)",
            param_count,
            hidden_dim,
            dropout,
        )

    @staticmethod
    def _load_jsonl(path: str) -> list[dict[str, Any]]:
        """Load examples from a JSONL file."""
        examples: list[dict[str, Any]] = []
        with open(path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    examples.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning("Skipping malformed line %d: %s", line_num, e)
        return examples

    @staticmethod
    def _split_data(
        examples: list[dict[str, Any]],
        val_ratio: float,
        seed: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Split examples into train and validation sets.

        Stratifies by ``error_type`` when available to ensure the
        validation set reflects the distribution of error categories.
        """
        import random

        rng = random.Random(seed)

        # Group by error_type for stratified split
        by_type: dict[str, list[dict[str, Any]]] = {}
        for ex in examples:
            etype = ex.get("error_type", "_unknown")
            by_type.setdefault(etype, []).append(ex)

        train_out: list[dict[str, Any]] = []
        val_out: list[dict[str, Any]] = []

        for _etype, group in sorted(by_type.items()):
            rng.shuffle(group)
            if len(group) > 1:
                n_val = min(max(1, int(len(group) * val_ratio)), len(group) - 1)
            else:
                n_val = 0
            val_out.extend(group[:n_val])
            train_out.extend(group[n_val:])

        rng.shuffle(train_out)
        rng.shuffle(val_out)
        return train_out, val_out

    @staticmethod
    def _compute_norm_stats(
        dataset: RerankerDataset,
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        """Compute per-feature mean and std from training data.

        Uses Welford's online algorithm to avoid numerical instability.
        """
        n = 0
        mean = torch.zeros(NUM_FEATURES, dtype=torch.float64)
        m2 = torch.zeros(NUM_FEATURES, dtype=torch.float64)

        for i in range(len(dataset)):
            feats = dataset.features[i]  # (max_cands, NUM_FEATURES)
            mask = dataset.masks[i]  # (max_cands,)
            # Only use real (non-padding) candidates
            real_feats = feats[mask]  # (n_real, NUM_FEATURES)
            for row in real_feats:
                n += 1
                delta = row.double() - mean
                mean += delta / n
                delta2 = row.double() - mean
                m2 += delta * delta2

        if n < 2:
            return mean.float(), torch.ones(NUM_FEATURES, dtype=torch.float32)

        variance = m2 / (n - 1)
        std = torch.sqrt(variance).float()
        # Avoid division by zero: clamp std to a small positive value
        std = torch.clamp(std, min=1e-8)
        return mean.float(), std

    def _normalize_dataset(self, dataset: RerankerDataset) -> None:
        """Normalize features in-place using stored mean/std."""
        for i in range(len(dataset)):
            dataset.features[i] = (dataset.features[i] - self.feature_means) / self.feature_stds

    def train(
        self,
        epochs: int = DEFAULT_EPOCHS,
        lr: float = DEFAULT_LR,
        batch_size: int = DEFAULT_BATCH_SIZE,
        patience: int = DEFAULT_PATIENCE,
    ) -> TrainingMetrics:
        """Train the reranker with early stopping on validation Top-1.

        Args:
            epochs: Maximum number of training epochs.
            lr: Learning rate for Adam optimizer.
            batch_size: Mini-batch size.
            patience: Stop after this many epochs without improvement.

        Returns:
            ``TrainingMetrics`` with loss curve and results.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Training on device: %s", device)

        self.model.to(device)

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        metrics = TrainingMetrics()
        best_state: dict[str, Any] | None = None
        epochs_without_improvement = 0

        for epoch in range(1, epochs + 1):
            # --- Train ---
            self.model.train()
            total_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                feats = batch["features"].to(device)
                gold = batch["gold_index"].to(device)
                mask = batch["mask"].to(device)

                scores = self.model(feats)
                # Mask out padding candidates with -inf
                scores = scores.masked_fill(~mask, float("-inf"))
                loss = F.cross_entropy(scores, gold)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)
            metrics.train_losses.append(avg_loss)

            # --- Validate ---
            val_top1, val_mrr = self._evaluate_dataset(self.val_dataset, device=device)
            metrics.val_top1_accuracies.append(val_top1)
            metrics.val_mrrs.append(val_mrr)

            logger.info(
                "Epoch %d/%d -- loss=%.4f, val_top1=%.4f, val_mrr=%.4f",
                epoch,
                epochs,
                avg_loss,
                val_top1,
                val_mrr,
            )

            # --- Early stopping ---
            if val_top1 > metrics.best_val_top1:
                metrics.best_val_top1 = val_top1
                metrics.best_val_mrr = val_mrr
                metrics.best_epoch = epoch
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    logger.info(
                        "Early stopping at epoch %d (patience=%d). "
                        "Best epoch=%d, top1=%.4f, mrr=%.4f",
                        epoch,
                        patience,
                        metrics.best_epoch,
                        metrics.best_val_top1,
                        metrics.best_val_mrr,
                    )
                    break

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(device)
            logger.info("Restored best model from epoch %d", metrics.best_epoch)

        return metrics

    @torch.no_grad()
    def _evaluate_dataset(
        self,
        dataset: RerankerDataset,
        batch_size: int = DEFAULT_BATCH_SIZE,
        device: "torch.device | None" = None,
    ) -> tuple[float, float]:
        """Compute Top-1 accuracy and MRR on a dataset.

        Args:
            dataset: ``RerankerDataset`` to evaluate on.
            batch_size: Batch size for forward pass.
            device: Device to run on.

        Returns:
            Tuple of (top1_accuracy, mrr).
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.eval()
        self.model.to(device)

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        correct = 0
        total = 0
        reciprocal_rank_sum = 0.0

        for batch in loader:
            feats = batch["features"].to(device)
            gold = batch["gold_index"].to(device)
            mask = batch["mask"].to(device)

            scores = self.model(feats)
            scores = scores.masked_fill(~mask, float("-inf"))

            # Top-1
            preds = scores.argmax(dim=-1)
            correct += (preds == gold).sum().item()
            total += gold.size(0)

            # MRR: sort descending, find rank of gold candidate
            sorted_indices = scores.argsort(dim=-1, descending=True)
            for i in range(gold.size(0)):
                gold_idx = gold[i].item()
                rank_pos = (sorted_indices[i] == gold_idx).nonzero(as_tuple=True)
                if rank_pos[0].numel() > 0:
                    rank = rank_pos[0][0].item() + 1  # 1-based
                    reciprocal_rank_sum += 1.0 / rank

        top1 = correct / max(total, 1)
        mrr = reciprocal_rank_sum / max(total, 1)
        return top1, mrr

    def export_onnx(self, output_path: str) -> str:
        """Export the trained model to ONNX with normalization stats.

        Saves:
          - ``output_path``: The ONNX model file.
          - ``<output_path>.stats.json``: Feature normalization stats.

        Args:
            output_path: Path for the ONNX model file.

        Returns:
            Path to the saved ONNX file.
        """
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        self.model.eval()
        self.model.cpu()

        # Dummy input: (1, max_candidates, NUM_FEATURES)
        dummy = torch.randn(1, self.max_candidates, NUM_FEATURES)

        torch.onnx.export(
            self.model,
            dummy,
            str(output),
            input_names=["features"],
            output_names=["scores"],
            dynamic_axes={
                "features": {0: "batch", 1: "candidates"},
                "scores": {0: "batch", 1: "candidates"},
            },
            opset_version=ONNX_OPSET_VERSION,
        )

        # Save normalization stats alongside model
        stats_path = str(output) + ".stats.json"
        stats = {
            "feature_names": FEATURE_NAMES,
            "feature_means": self.feature_means.tolist(),
            "feature_stds": self.feature_stds.tolist(),
            "max_candidates": self.max_candidates,
            "num_features": NUM_FEATURES,
        }
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        # Report model size
        model_size = output.stat().st_size
        logger.info(
            "ONNX model exported to %s (%.1f KB)",
            output,
            model_size / 1024,
        )
        logger.info("Normalization stats saved to %s", stats_path)

        return str(output)

    def save_metrics(self, metrics: TrainingMetrics, output_path: str) -> None:
        """Save training metrics to JSON for analysis."""
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(metrics.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info("Training metrics saved to %s", output)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """CLI entry point for reranker training."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Train a reranker MLP on JSONL training data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--train",
        required=True,
        help="Path to JSONL training data.",
    )
    parser.add_argument(
        "--val",
        default=None,
        help=("Path to JSONL validation data (optional; auto-split if omitted)."),
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for ONNX model and stats.",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--max-candidates", type=int, default=DEFAULT_MAX_CANDIDATES)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    from myspellchecker.utils.logging_utils import configure_logging

    configure_logging(level="INFO")

    trainer = RerankerTrainer(
        train_path=args.train,
        val_path=args.val,
        max_candidates=args.max_candidates,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        seed=args.seed,
    )

    metrics = trainer.train(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        patience=args.patience,
    )

    # Save outputs
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = str(output_dir / "reranker.onnx")
    trainer.export_onnx(onnx_path)
    trainer.save_metrics(metrics, str(output_dir / "training_metrics.json"))

    # Summary
    print("\nTraining complete.")
    print(f"  Best epoch:    {metrics.best_epoch}")
    print(f"  Best val Top1: {metrics.best_val_top1:.4f}")
    print(f"  Best val MRR:  {metrics.best_val_mrr:.4f}")
    print(f"  ONNX model:    {onnx_path}")
    onnx_size = os.path.getsize(onnx_path)
    print(f"  Model size:    {onnx_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
