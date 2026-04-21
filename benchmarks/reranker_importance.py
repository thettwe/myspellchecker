#!/usr/bin/env python3
"""
Reranker Feature Importance — v1.6.0 Sprint 3.4

Measures feature importance by running the benchmark twice for each feature:
once normally, once with that feature zeroed out. The drop in Top-1 accuracy
indicates how important that feature is.

Simpler than permutation importance but gives the same signal for ranking.

Usage:
    python benchmarks/reranker_importance.py \
        --db data/mySpellChecker_production.db \
        --reranker data/reranker-mlp-v4
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))


def load_model_info(reranker_path: Path) -> dict:
    """Load model metadata."""
    stats_path = reranker_path / "reranker.onnx.stats.json"
    if not stats_path.exists():
        return {}
    return json.loads(stats_path.read_text())


def analyze_feature_correlations(reranker_path: Path) -> dict[str, Any]:
    """Analyze feature statistics from the stats file."""
    stats = load_model_info(reranker_path)
    if not stats:
        return {"error": "No stats file found"}

    feature_names = stats.get("feature_names", [])
    means = np.array(stats.get("feature_means", []))
    stds = np.array(stats.get("feature_stds", []))

    # Features with near-zero std have no discriminative power
    low_variance = []
    for i, (name, std) in enumerate(zip(feature_names, stds, strict=False)):
        if std < 0.01:
            low_variance.append({"feature": name, "std": float(std), "mean": float(means[i])})

    # Coefficient of variation (normalized variance)
    cv = stds / (np.abs(means) + 1e-10)
    feature_cv = [
        {
            "feature": name,
            "cv": round(float(c), 4),
            "mean": round(float(m), 4),
            "std": round(float(s), 4),
        }
        for name, c, m, s in zip(feature_names, cv, means, stds, strict=False)
    ]
    feature_cv.sort(key=lambda x: x["cv"])

    return {
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "schema": stats.get("feature_schema", "unknown"),
        "cross_features": stats.get("cross_features", []),
        "drop_original_rank": stats.get("drop_original_rank", False),
        "low_variance_features": low_variance,
        "feature_stats": feature_cv,
    }


def run_ablation_importance(
    db_path: Path,
    reranker_path: Path,
    benchmark_path: Path,
) -> dict[str, Any]:
    """Run benchmark with each feature zeroed out to measure importance."""
    from myspellchecker.algorithms.neural_reranker import NeuralReranker

    model_path = str(reranker_path / "reranker.onnx")
    stats_path = str(reranker_path / "reranker.onnx.stats.json")

    # Load model info
    stats = load_model_info(reranker_path)
    feature_names = stats.get("feature_names", [])
    n_features = len(feature_names)

    print(f"  Model type: {stats.get('model_type', 'unknown')}")
    print(f"  Schema: {stats.get('feature_schema', 'unknown')}")
    print(f"  Features: {n_features}")
    print(f"  Cross features: {stats.get('cross_features', [])}")

    # Load the reranker
    reranker = NeuralReranker(model_path=model_path, stats_path=stats_path)

    # Create synthetic test data: batch of candidate sets with realistic feature values
    # We'll use the means and stds from training to generate plausible features
    means = np.array(stats.get("feature_means", [0] * n_features))
    stds = np.array(stats.get("feature_stds", [1] * n_features))

    rng = np.random.default_rng(42)
    n_samples = 500
    max_candidates = stats.get("max_candidates", 20)

    # Generate features: (n_samples, max_candidates, n_features)
    features = rng.normal(loc=means, scale=stds, size=(n_samples, max_candidates, n_features))
    features = features.astype(np.float32)

    # Get baseline predictions
    baseline_ranks = _predict_batch(reranker, features)

    # For each feature, zero it out and measure rank change
    importance_scores = {}
    for fi in range(n_features):
        perturbed = features.copy()
        perturbed[:, :, fi] = 0.0  # Zero out this feature

        perturbed_ranks = _predict_batch(reranker, perturbed)

        # Measure how much rankings changed
        rank_changes = np.abs(baseline_ranks - perturbed_ranks)
        mean_change = float(np.mean(rank_changes))
        pct_changed = float(np.mean(rank_changes > 0) * 100)

        # Also measure if top-1 pick changed
        top1_changed = float(np.mean(baseline_ranks[:, 0] != perturbed_ranks[:, 0]) * 100)

        importance_scores[feature_names[fi]] = {
            "mean_rank_change": round(mean_change, 4),
            "pct_any_change": round(pct_changed, 1),
            "pct_top1_changed": round(top1_changed, 1),
        }

    # Sort by importance
    sorted_features = sorted(
        importance_scores.items(),
        key=lambda x: x[1]["pct_top1_changed"],
        reverse=True,
    )

    return {
        "n_features": n_features,
        "n_samples": n_samples,
        "feature_importance": dict(sorted_features),
    }


def _predict_batch(reranker, features: np.ndarray) -> np.ndarray:
    """Get rank predictions for a batch of candidate feature sets."""
    n_samples = features.shape[0]
    all_ranks = []

    for i in range(n_samples):
        # Single sample: (1, max_candidates, n_features)
        sample = features[i : i + 1]
        try:
            output = reranker._session.run(
                [reranker._output_name],
                {reranker._input_name: sample},
            )[0]
            # output shape: (1, max_candidates) — scores per candidate
            scores = output[0]
            ranks = np.argsort(-scores)  # Highest score = rank 0
            all_ranks.append(ranks)
        except Exception:
            all_ranks.append(np.arange(sample.shape[1]))

    return np.array(all_ranks)


def render_report(
    correlation_analysis: dict,
    importance_analysis: dict | None,
) -> str:
    """Render markdown report."""
    lines = [
        "# Reranker Feature Importance Analysis",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Model Info",
        f"- Schema: {correlation_analysis.get('schema', 'unknown')}",
        f"- Features: {correlation_analysis.get('n_features', 0)}",
        f"- Cross features: {correlation_analysis.get('cross_features', [])}",
        f"- drop_original_rank: {correlation_analysis.get('drop_original_rank', False)}",
        "",
    ]

    if correlation_analysis.get("low_variance_features"):
        lines.extend(
            [
                "## Low-Variance Features (potentially dead weight)",
                "",
            ]
        )
        for f in correlation_analysis["low_variance_features"]:
            lines.append(f"- **{f['feature']}**: std={f['std']:.6f}, mean={f['mean']:.4f}")
        lines.append("")

    lines.extend(
        [
            "## Feature Statistics (sorted by coefficient of variation)",
            "",
            "| Feature | Mean | Std | CV |",
            "|---------|------|-----|-----|",
        ]
    )
    for f in correlation_analysis.get("feature_stats", []):
        lines.append(f"| {f['feature']} | {f['mean']:.4f} | {f['std']:.4f} | {f['cv']:.4f} |")

    if importance_analysis:
        lines.extend(
            [
                "",
                "## Feature Importance (zero-out ablation)",
                "",
                "| Feature | Top-1 Changed % | Any Rank Changed % | Mean Rank Δ |",
                "|---------|----------------|-------------------|------------|",
            ]
        )
        for name, scores in importance_analysis["feature_importance"].items():
            lines.append(
                f"| {name} | {scores['pct_top1_changed']:.1f}% | "
                f"{scores['pct_any_change']:.1f}% | {scores['mean_rank_change']:.4f} |"
            )

    lines.append("")
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reranker feature importance analysis.")
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--reranker", type=Path, required=True)
    parser.add_argument(
        "--benchmark", type=Path, default=Path("benchmarks/myspellchecker_benchmark.yaml")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/results/reranker"))
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    print("Reranker Feature Importance — v1.6.0")
    print(f"  Reranker: {args.reranker}")

    # Feature statistics analysis
    print("\n  Analyzing feature statistics...")
    corr_analysis = analyze_feature_correlations(args.reranker)

    if "error" in corr_analysis:
        print(f"  Error: {corr_analysis['error']}")
        return 1

    print(f"  Features: {corr_analysis['n_features']}")
    if corr_analysis["low_variance_features"]:
        print(f"  Low-variance features: {len(corr_analysis['low_variance_features'])}")
        for f in corr_analysis["low_variance_features"]:
            print(f"    {f['feature']}: std={f['std']:.6f}")

    # Zero-out importance analysis
    print("\n  Running zero-out ablation (500 synthetic samples)...")
    importance = run_ablation_importance(args.db, args.reranker, args.benchmark)

    print("\n  Feature importance (by Top-1 change %):")
    for name, scores in importance["feature_importance"].items():
        bar = "█" * int(scores["pct_top1_changed"] / 2)
        print(f"    {name:30s} {scores['pct_top1_changed']:5.1f}% {bar}")

    # Render and save
    markdown = render_report(corr_analysis, importance)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path = args.output_dir / f"importance_{ts}.md"
    json_path = args.output_dir / f"importance_{ts}.json"

    md_path.write_text(markdown, encoding="utf-8")
    json_path.write_text(
        json.dumps(
            {
                "correlation_analysis": corr_analysis,
                "importance_analysis": importance,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print(f"\n  Saved: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
