"""Reranker MLP trainer.

Trains a small feature-based MLP with cross-entropy (listwise) loss to re-rank
spell-checker suggestion lists. This module is used exclusively during the
**offline training pipeline** -- it is NOT used at inference time.

Pipeline context:
    - **Upstream**: Reads JSONL training data produced by
      ``RerankerDataGenerator`` in ``reranker_data.py``. Each JSONL line
      contains 19 features (v2) per candidate plus a gold correction index.
    - **Downstream**: Exports an ONNX model (``reranker.onnx``) plus a
      ``reranker.onnx.stats.json`` file with per-feature normalization
      statistics (mean/std computed via Welford's online algorithm).
    - **At inference**: The exported ONNX model is loaded by
      ``algorithms/ranker.py`` to re-score suggestions. This trainer module
      itself is not involved at inference time.

Architecture:
    - ``RerankerMLP``: Linear(input_dim, 64) -> ReLU -> Dropout(0.1) -> Linear(64, 1).
      Input shape: (batch, num_candidates, input_dim). Output: scalar score per candidate.
      Auto-detects feature dimension from training data (v2: 19 features).
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

from myspellchecker.training.reranker_data import (
    FEATURE_NAMES,
    MLP_CROSS_FEATURE_NAMES,
    MLP_CROSS_FEATURES,
    NUM_FEATURES,
    ORIGINAL_RANK_INDEX,
)
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

    Architecture: Linear(input_dim, hidden_dim) -> ReLU -> Dropout -> Linear(hidden_dim, 1)

    Input:  (batch, num_candidates, input_dim)  # v2: 19 features
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

    When ``drop_original_rank=True``, removes feature 14 (``original_rank``)
    to prevent label leakage in MLP training.

    When ``add_cross_features=True``, appends MLP-specific cross-features
    computed from the base 19 features.
    """

    def __init__(
        self,
        examples: list[dict[str, Any]],
        max_candidates: int = DEFAULT_MAX_CANDIDATES,
        drop_original_rank: bool = False,
        add_cross_features: bool = False,
    ):
        self.max_candidates = max_candidates
        self.features: list[torch.Tensor] = []
        self.gold_indices: list[int] = []
        self.masks: list[torch.Tensor] = []
        self.drop_original_rank = drop_original_rank
        self.add_cross_features = add_cross_features

        # Detect base feature dimension from data (supports both v1=20 and v2=19)
        base_num_features = NUM_FEATURES
        if examples and examples[0].get("features"):
            first_feats = examples[0]["features"]
            if first_feats:
                base_num_features = len(first_feats[0])

        # Compute final feature dimension after transforms
        self.num_features = base_num_features
        if drop_original_rank:
            self.num_features -= 1
        if add_cross_features:
            self.num_features += len(MLP_CROSS_FEATURES)

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

            # Apply MLP feature transforms
            if drop_original_rank or add_cross_features:
                feats = [self._transform_feature_vec(fv, base_num_features) for fv in feats]

            # Pad features to (max_candidates, num_features)
            padded = feats + [[0.0] * self.num_features] * (max_candidates - n_cands)
            mask = [1.0] * n_cands + [0.0] * (max_candidates - n_cands)

            self.features.append(torch.tensor(padded, dtype=torch.float32))
            self.gold_indices.append(gold_idx)
            self.masks.append(torch.tensor(mask, dtype=torch.bool))

    def _transform_feature_vec(
        self,
        fv: list[float],
        base_dim: int,
    ) -> list[float]:
        """Apply MLP-specific feature transforms to a single candidate vector.

        1. Drop ``original_rank`` (index 14) if configured.
        2. Append cross-features computed from base features.
        """
        # Work on a copy to avoid mutating the input
        base = list(fv[:base_dim])

        # Compute cross-features BEFORE dropping (indices refer to base layout)
        cross_vals: list[float] = []
        if self.add_cross_features:
            for _name, left_idx, right_idx in MLP_CROSS_FEATURES:
                if right_idx == -1:
                    # Special: mlm_logit * (ngram_left + ngram_right)
                    cross_vals.append(base[left_idx] * (base[8] + base[9]))
                else:
                    cross_vals.append(base[left_idx] * base[right_idx])

        # Drop original_rank after computing crosses (crosses don't use it)
        if self.drop_original_rank and ORIGINAL_RANK_INDEX < len(base):
            del base[ORIGINAL_RANK_INDEX]

        return base + cross_vals

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

    MLP-specific optimizations (enabled by default):
      - Drops ``original_rank`` feature to prevent label leakage.
      - Adds 5 cross-features for explicit interaction modeling.
      - Uses weighted sampling to oversample disagreement cases (gold_index > 0).

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
        drop_original_rank: Remove feature 14 (label leakage).
        add_cross_features: Append MLP-specific interaction features.
        oversample_weight: Weight multiplier for examples where
            gold_index > 0 (pipeline disagrees). Set to 1.0 to disable.
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
        drop_original_rank: bool = True,
        add_cross_features: bool = True,
        oversample_weight: float = 2.0,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError(
                "RerankerTrainer requires PyTorch. "
                "Install with: pip install 'myspellchecker[train]'"
            )

        self.max_candidates = max_candidates
        self.seed = seed
        self.drop_original_rank = drop_original_rank
        self.add_cross_features = add_cross_features
        self.oversample_weight = oversample_weight
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

        # MLP-specific transforms: drop original_rank, add cross-features
        self.train_dataset = RerankerDataset(
            train_examples,
            max_candidates,
            drop_original_rank=drop_original_rank,
            add_cross_features=add_cross_features,
        )
        self.val_dataset = RerankerDataset(
            val_examples,
            max_candidates,
            drop_original_rank=drop_original_rank,
            add_cross_features=add_cross_features,
        )

        if drop_original_rank:
            logger.info(
                "MLP: dropped original_rank (feature %d) to prevent label leakage",
                ORIGINAL_RANK_INDEX,
            )
        if add_cross_features:
            logger.info(
                "MLP: added %d cross-features: %s",
                len(MLP_CROSS_FEATURES),
                MLP_CROSS_FEATURE_NAMES,
            )

        # Compute sample weights for oversampling disagreement cases.
        # Use gold_indices from the dataset (after filtering), not raw examples,
        # to ensure weights and dataset have the same length.
        self._sample_weights: list[float] | None = None
        if oversample_weight > 1.0:
            gold_indices = self.train_dataset.gold_indices
            n_agree = sum(1 for gi in gold_indices if gi == 0)
            n_disagree = len(gold_indices) - n_agree
            self._sample_weights = [oversample_weight if gi > 0 else 1.0 for gi in gold_indices]
            logger.info(
                "MLP: weighted sampling — %d agree (1.0x), %d disagree (%.1fx)",
                n_agree,
                n_disagree,
                oversample_weight,
            )

        # Detect feature dimension from training data (supports v1=20 and v2=19)
        self.detected_dim = self.train_dataset.num_features

        # Compute normalization stats from training data
        self.feature_means, self.feature_stds = self._compute_norm_stats(self.train_dataset)
        logger.info("Computed normalization stats for %d features", self.detected_dim)

        # Apply normalization
        self._normalize_dataset(self.train_dataset)
        self._normalize_dataset(self.val_dataset)

        # Build model — use detected feature dimension from data
        self.model = RerankerMLP(
            input_dim=self.detected_dim,
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
        feat_dim = dataset.num_features
        n = 0
        mean = torch.zeros(feat_dim, dtype=torch.float64)
        m2 = torch.zeros(feat_dim, dtype=torch.float64)

        for i in range(len(dataset)):
            feats = dataset.features[i]  # (max_cands, feat_dim)
            mask = dataset.masks[i]  # (max_cands,)
            # Only use real (non-padding) candidates
            real_feats = feats[mask]  # (n_real, feat_dim)
            for row in real_feats:
                n += 1
                delta = row.double() - mean
                mean += delta / n
                delta2 = row.double() - mean
                m2 += delta * delta2

        if n < 2:
            return mean.float(), torch.ones(feat_dim, dtype=torch.float32)

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

        # Use weighted sampling if oversampling is enabled
        if self._sample_weights is not None:
            from torch.utils.data import WeightedRandomSampler

            sampler = WeightedRandomSampler(
                weights=self._sample_weights,
                num_samples=len(self._sample_weights),
                replacement=True,
            )
            train_loader = DataLoader(
                self.train_dataset,
                batch_size=batch_size,
                sampler=sampler,
                drop_last=False,
            )
        else:
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

        # Dummy input: (1, max_candidates, detected_dim)
        dummy = torch.randn(1, self.max_candidates, self.detected_dim)

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
        # Build feature names list reflecting MLP transforms
        mlp_feature_names = list(FEATURE_NAMES)
        if self.drop_original_rank:
            mlp_feature_names.pop(ORIGINAL_RANK_INDEX)
        if self.add_cross_features:
            mlp_feature_names.extend(MLP_CROSS_FEATURE_NAMES)

        stats_path = str(output) + ".stats.json"
        stats = {
            "feature_names": mlp_feature_names,
            "feature_means": self.feature_means.tolist(),
            "feature_stds": self.feature_stds.tolist(),
            "max_candidates": self.max_candidates,
            "num_features": self.detected_dim,
            "model_type": "mlp",
            "feature_schema": "mlp_v3",
            "drop_original_rank": self.drop_original_rank,
            "cross_features": MLP_CROSS_FEATURE_NAMES if self.add_cross_features else [],
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
    parser.add_argument(
        "--no-drop-rank",
        action="store_true",
        help="Keep original_rank feature (default: drop for MLP).",
    )
    parser.add_argument(
        "--no-cross-features",
        action="store_true",
        help="Skip MLP cross-feature generation.",
    )
    parser.add_argument(
        "--oversample-weight",
        type=float,
        default=2.0,
        help="Weight for oversampling gold_index>0 examples.",
    )

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
        drop_original_rank=not args.no_drop_rank,
        add_cross_features=not args.no_cross_features,
        oversample_weight=args.oversample_weight,
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
