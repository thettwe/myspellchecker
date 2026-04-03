"""LightGBM LambdaRank reranker trainer.

Trains a LightGBM ranking model (LambdaMART) on the same JSONL training
data produced by ``reranker_data.py``.  Unlike the MLP trainer, this uses
a proper ranking objective (``lambdarank``) that directly optimizes NDCG,
which is better aligned with the spell checker's goal of getting the
correct suggestion to rank 1.

Advantages over MLP:
  - Learns sharp conditional rules (IF-THEN) instead of smooth averages
  - Handles heterogeneous features without normalization
  - Built-in feature importance (SHAP)
  - LambdaRank loss directly optimizes ranking metrics

Usage (CLI):
    python -m myspellchecker.training.reranker_trainer_lgbm \\
        --train data/reranker_training_v4.jsonl \\
        --output data/reranker-lgbm/ \\
        --n-estimators 200 --num-leaves 31

Usage (API):
    >>> from myspellchecker.training.reranker_trainer_lgbm import train_lgbm_ranker
    >>> metrics = train_lgbm_ranker(
    ...     train_path="data/reranker_training_v4.jsonl",
    ...     output_dir="data/reranker-lgbm/",
    ... )
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from myspellchecker.training.reranker_data import FEATURE_NAMES
from myspellchecker.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _load_ranking_data(
    jsonl_path: str,
    max_candidates: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load JSONL training data into LightGBM ranking format.

    Returns:
        Tuple of (features, labels, group_sizes) where:
        - features: (N_total_candidates, num_features) float32
        - labels: (N_total_candidates,) float32 — relevance labels
        - group_sizes: (N_queries,) int32 — candidates per query
    """
    all_features: list[list[float]] = []
    all_labels: list[float] = []
    group_sizes: list[int] = []

    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                example = json.loads(line)
            except json.JSONDecodeError:
                continue

            feats = example.get("features", [])
            gold_idx = example.get("gold_index", -1)
            if not feats or gold_idx < 0:
                continue

            n_cands = min(len(feats), max_candidates)
            if gold_idx >= n_cands:
                continue

            group_sizes.append(n_cands)
            for i in range(n_cands):
                all_features.append(feats[i])
                # Relevance: gold candidate gets label 1, others get 0
                all_labels.append(1.0 if i == gold_idx else 0.0)

    features = np.array(all_features, dtype=np.float32)
    labels = np.array(all_labels, dtype=np.float32)
    groups = np.array(group_sizes, dtype=np.int32)

    logger.info(
        "Loaded %d queries, %d total candidates, %d features from %s",
        len(groups),
        len(labels),
        features.shape[1] if features.ndim == 2 else 0,
        jsonl_path,
    )
    return features, labels, groups


def train_lgbm_ranker(
    train_path: str,
    output_dir: str,
    val_path: str | None = None,
    val_ratio: float = 0.2,
    n_estimators: int = 200,
    num_leaves: int = 31,
    learning_rate: float = 0.05,
    max_depth: int = -1,
    colsample_bytree: float = 0.8,
    subsample: float = 0.8,
    min_child_samples: int = 20,
    seed: int = 42,
    max_candidates: int = 20,
) -> dict:
    """Train a LightGBM LambdaRank model and export to ONNX.

    Args:
        train_path: Path to JSONL training data.
        output_dir: Directory for model outputs.
        val_path: Optional separate validation JSONL.
        val_ratio: Fraction for validation split (if no val_path).
        n_estimators: Number of boosting rounds.
        num_leaves: Max leaves per tree.
        learning_rate: Boosting learning rate.
        max_depth: Max tree depth (-1 = no limit).
        colsample_bytree: Feature sampling ratio per tree.
        subsample: Row sampling ratio per tree.
        min_child_samples: Min samples per leaf.
        seed: Random seed.
        max_candidates: Max candidates per query.

    Returns:
        Dict with training metrics.
    """
    import lightgbm as lgb

    # Load data
    features, labels, groups = _load_ranking_data(train_path, max_candidates)

    if val_path:
        val_features, val_labels, val_groups = _load_ranking_data(val_path, max_candidates)
    else:
        # Split by queries (not by individual candidates)
        rng = np.random.RandomState(seed)
        n_queries = len(groups)
        indices = rng.permutation(n_queries)
        split_idx = int(n_queries * (1 - val_ratio))
        train_query_idx = sorted(indices[:split_idx])
        val_query_idx = sorted(indices[split_idx:])

        # Convert query indices to candidate indices
        cumsum = np.cumsum(groups)
        starts = np.concatenate([[0], cumsum[:-1]])

        train_cand_idx = np.concatenate([np.arange(starts[q], cumsum[q]) for q in train_query_idx])
        val_cand_idx = np.concatenate([np.arange(starts[q], cumsum[q]) for q in val_query_idx])

        val_features = features[val_cand_idx]
        val_labels = labels[val_cand_idx]
        val_groups = groups[val_query_idx]

        features = features[train_cand_idx]
        labels = labels[train_cand_idx]
        groups = groups[train_query_idx]

    logger.info(
        "Train: %d queries, %d candidates. Val: %d queries, %d candidates",
        len(groups),
        len(labels),
        len(val_groups),
        len(val_labels),
    )

    # Train
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [1, 3, 5],
        "n_estimators": n_estimators,
        "num_leaves": num_leaves,
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "colsample_bytree": colsample_bytree,
        "subsample": subsample,
        "min_child_samples": min_child_samples,
        "seed": seed,
        "verbose": -1,
    }

    callbacks = [
        lgb.log_evaluation(period=10),
        lgb.early_stopping(stopping_rounds=20, verbose=True),
    ]

    model = lgb.LGBMRanker(**params)
    model.fit(
        features,
        labels,
        group=groups.tolist(),
        eval_set=[(val_features, val_labels)],
        eval_group=[val_groups.tolist()],
        eval_metric="ndcg",
        eval_at=[1, 3, 5],
        callbacks=callbacks,
    )

    best_iteration = model.best_iteration_
    logger.info("Best iteration: %d", best_iteration)

    # Feature importance
    importance = model.feature_importances_
    feat_names = FEATURE_NAMES[: features.shape[1]]
    importance_pairs = sorted(
        zip(feat_names, importance, strict=False), key=lambda x: x[1], reverse=True
    )
    logger.info("Feature importance (top 10):")
    for name, imp in importance_pairs[:10]:
        logger.info("  %s: %d", name, imp)

    # Compute Top-1 and MRR on validation
    val_scores = model.predict(val_features)
    val_top1, val_mrr = _compute_ranking_metrics(val_scores, val_labels, val_groups)
    logger.info("Val Top-1: %.4f, Val MRR: %.4f", val_top1, val_mrr)

    # Save outputs
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Save native LightGBM model
    native_path = out_path / "reranker.lgbm"
    model.booster_.save_model(str(native_path))
    logger.info("Native model saved to %s", native_path)

    # Export to ONNX
    onnx_path = out_path / "reranker.onnx"
    _export_onnx(model, features.shape[1], str(onnx_path))

    # Save stats JSON (for compatibility with NeuralReranker loader)
    stats = {
        "feature_names": feat_names,
        "feature_means": [],  # GBTs don't need normalization
        "feature_stds": [],
        "max_candidates": max_candidates,
        "num_features": features.shape[1],
        "model_type": "lightgbm_lambdarank",
        "feature_schema": "gbt_v1",
        "drop_original_rank": False,
        "cross_features": [],
        "best_iteration": best_iteration,
        "n_estimators": n_estimators,
        "num_leaves": num_leaves,
    }
    stats_path = out_path / "reranker.onnx.stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # Save metrics
    metrics = {
        "val_top1": val_top1,
        "val_mrr": val_mrr,
        "best_iteration": best_iteration,
        "feature_importance": {name: int(imp) for name, imp in importance_pairs},
    }
    metrics_path = out_path / "training_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("\nTraining complete.")
    print(f"  Best iteration: {best_iteration}")
    print(f"  Val Top-1:      {val_top1:.4f}")
    print(f"  Val MRR:        {val_mrr:.4f}")
    print(f"  ONNX model:     {onnx_path}")
    print(f"  Model size:     {onnx_path.stat().st_size / 1024:.1f} KB")

    return metrics


def _compute_ranking_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
) -> tuple[float, float]:
    """Compute Top-1 accuracy and MRR from predicted scores."""
    offset = 0
    top1_correct = 0
    reciprocal_rank_sum = 0.0
    n_queries = len(groups)

    for g in groups:
        g_scores = scores[offset : offset + g]
        g_labels = labels[offset : offset + g]

        ranked_indices = np.argsort(-g_scores)
        gold_positions = np.where(g_labels[ranked_indices] > 0)[0]

        if len(gold_positions) > 0:
            rank = gold_positions[0]  # 0-indexed
            if rank == 0:
                top1_correct += 1
            reciprocal_rank_sum += 1.0 / (rank + 1)

        offset += g

    top1 = top1_correct / max(n_queries, 1)
    mrr = reciprocal_rank_sum / max(n_queries, 1)
    return top1, mrr


def _export_onnx(model, num_features: int, output_path: str) -> None:
    """Export LightGBM model to ONNX."""
    try:
        from onnxmltools import convert_lightgbm
        from onnxmltools.convert.common.data_types import FloatTensorType

        onnx_model = convert_lightgbm(
            model,
            initial_types=[("features", FloatTensorType([None, num_features]))],
            target_opset=15,
        )

        import onnx

        onnx.save(onnx_model, output_path)
        logger.info(
            "ONNX model exported to %s (%.1f KB)",
            output_path,
            os.path.getsize(output_path) / 1024,
        )
    except Exception as e:
        logger.warning("ONNX export failed: %s. Native model still available.", e)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train LightGBM LambdaRank reranker")
    parser.add_argument("--train", required=True, help="Training JSONL path")
    parser.add_argument("--val", default=None, help="Validation JSONL path")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_lgbm_ranker(
        train_path=args.train,
        output_dir=args.output,
        val_path=args.val,
        n_estimators=args.n_estimators,
        num_leaves=args.num_leaves,
        learning_rate=args.lr,
        max_depth=args.max_depth,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
