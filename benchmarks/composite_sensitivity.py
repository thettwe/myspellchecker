#!/usr/bin/env python3
"""
Composite Score Sensitivity Analysis — v1.6.0 Sprint 1.4

Analyzes the stability of the composite formula by:
1. Weight perturbation: vary each weight by +/-0.05, check if rankings change
2. Component correlation: compute correlation matrix between F1, MRR, FPR, Top1, latency
3. Effective weight analysis: account for correlations to find true weighting

Usage:
    python benchmarks/composite_sensitivity.py \
        --report benchmarks/results/latest_run.json

    # Or run fresh:
    python benchmarks/composite_sensitivity.py \
        --db data/mySpellChecker_production.db
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

# Default weights
DEFAULT_WEIGHTS = {
    "f1": 0.30,
    "mrr": 0.25,
    "inv_fpr": 0.20,  # (1-FPR)
    "top1": 0.15,
    "inv_latency": 0.10,  # (1-latency_norm)
}


def extract_per_sentence_components(sentences: list[dict]) -> dict[str, np.ndarray]:
    """Extract per-sentence metric components for correlation analysis."""
    n = len(sentences)
    # Per-sentence F1 proxy (binary: had error, detected or not)
    f1_proxy = np.zeros(n)
    fpr_proxy = np.zeros(n)
    mrr_proxy = np.zeros(n)
    top1_proxy = np.zeros(n)
    latency = np.zeros(n)

    for i, s in enumerate(sentences):
        tp = s.get("true_positives", 0)
        fp = s.get("false_positives", 0)
        fn = s.get("false_negatives", 0)

        # Per-sentence precision/recall/F1
        p = tp / (tp + fp) if (tp + fp) > 0 else 1.0  # No detections on clean = perfect
        r = tp / (tp + fn) if (tp + fn) > 0 else 1.0  # No expected errors = perfect
        f1_proxy[i] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        # FPR: 1 if has FP on clean sentence, 0 otherwise
        if s.get("is_clean", False):
            fpr_proxy[i] = 1.0 if fp > 0 else 0.0
        else:
            fpr_proxy[i] = np.nan  # N/A for error sentences

        # MRR: average reciprocal rank for this sentence
        ranks = []
        for m in s.get("span_matches", []):
            if m.get("matched"):
                rank = m.get("rank_of_correct")
                ranks.append(1.0 / rank if rank else 0.0)
        mrr_proxy[i] = np.mean(ranks) if ranks else np.nan

        # Top1
        top1_hits = sum(1 for m in s.get("span_matches", []) if m.get("matched") and m.get("top1_correct"))
        total_matches = sum(1 for m in s.get("span_matches", []) if m.get("matched"))
        top1_proxy[i] = top1_hits / total_matches if total_matches > 0 else np.nan

        latency[i] = s.get("latency_ms", 0)

    return {
        "f1": f1_proxy,
        "fpr": fpr_proxy,
        "mrr": mrr_proxy,
        "top1": top1_proxy,
        "latency": latency,
    }


def compute_correlation_matrix(components: dict[str, np.ndarray]) -> dict[str, Any]:
    """Compute pairwise correlations between metric components."""
    keys = ["f1", "mrr", "top1"]  # Only use components with mostly valid data
    n = len(components["f1"])

    # Build matrix of valid observations
    valid_mask = np.ones(n, dtype=bool)
    for k in keys:
        valid_mask &= ~np.isnan(components[k])

    matrix = {}
    for k1 in keys:
        matrix[k1] = {}
        for k2 in keys:
            v1 = components[k1][valid_mask]
            v2 = components[k2][valid_mask]
            if len(v1) > 1:
                corr = float(np.corrcoef(v1, v2)[0, 1])
            else:
                corr = 0.0
            matrix[k1][k2] = round(corr, 4)

    return matrix


def weight_perturbation_analysis(
    aggregate_metrics: dict[str, float],
    perturbation: float = 0.05,
) -> dict[str, Any]:
    """Vary each weight by +/-perturbation and measure composite change."""
    base_composite = compute_composite(aggregate_metrics, DEFAULT_WEIGHTS)

    results = {}
    weight_keys = list(DEFAULT_WEIGHTS.keys())

    for target_key in weight_keys:
        variations = {}
        for delta in [-perturbation, +perturbation]:
            # Create perturbed weights (renormalize to sum=1)
            perturbed = dict(DEFAULT_WEIGHTS)
            perturbed[target_key] += delta

            # Renormalize
            total = sum(perturbed.values())
            perturbed = {k: v / total for k, v in perturbed.items()}

            new_composite = compute_composite(aggregate_metrics, perturbed)
            variations[f"{delta:+.2f}"] = {
                "composite": round(new_composite, 5),
                "delta": round(new_composite - base_composite, 5),
            }

        results[target_key] = {
            "base_weight": DEFAULT_WEIGHTS[target_key],
            "variations": variations,
            "sensitivity": round(
                abs(variations[f"+{perturbation:.2f}"]["delta"])
                + abs(variations[f"-{perturbation:.2f}"]["delta"]),
                5,
            ),
        }

    return {
        "base_composite": round(base_composite, 5),
        "perturbation_size": perturbation,
        "per_weight": results,
        "sensitivity_ranking": sorted(
            results.keys(), key=lambda k: results[k]["sensitivity"], reverse=True
        ),
    }


def compute_composite(metrics: dict[str, float], weights: dict[str, float]) -> float:
    """Compute composite score with given weights."""
    return (
        weights["f1"] * metrics["f1"]
        + weights["mrr"] * metrics["mrr"]
        + weights["inv_fpr"] * (1.0 - metrics["fpr"])
        + weights["top1"] * metrics["top1_accuracy"]
        + weights["inv_latency"] * (1.0 - metrics.get("latency_norm", 0.3))
    )


def analyze_latency_discriminative_power(sentences: list[dict]) -> dict[str, Any]:
    """Check if the latency component has any discriminative power."""
    latencies = [s.get("latency_ms", 0) for s in sentences]
    p95 = np.percentile(latencies, 95)
    latency_norm = min(p95 / 500.0, 1.0)
    latency_contribution = 0.10 * (1.0 - latency_norm)

    return {
        "p95_ms": round(float(p95), 1),
        "latency_normalized": round(latency_norm, 4),
        "composite_contribution": round(latency_contribution, 4),
        "is_discriminative": bool(latency_norm < 0.9),  # Only discriminative if p95 < 450ms
        "note": "Latency component is near-constant and adds ~fixed offset to composite"
        if latency_norm < 0.5
        else "Latency is penalizing composite significantly",
    }


def render_report(
    perturbation: dict[str, Any],
    correlations: dict[str, Any],
    latency_analysis: dict[str, Any],
) -> str:
    """Render markdown report."""
    lines = [
        "# Composite Score Sensitivity Analysis",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Current Formula",
        "",
        "`composite = 0.30*F1 + 0.25*MRR + 0.20*(1-FPR) + 0.15*Top1 + 0.10*(1-latency_norm)`",
        "",
        f"Base composite: **{perturbation['base_composite']:.4f}**",
        "",
        "## Weight Perturbation (+/-0.05)",
        "",
        "| Component | Weight | -0.05 Δ | +0.05 Δ | Sensitivity |",
        "|-----------|--------|---------|---------|-------------|",
    ]

    for key in perturbation["sensitivity_ranking"]:
        data = perturbation["per_weight"][key]
        neg = data["variations"]["-0.05"]["delta"]
        pos = data["variations"]["+0.05"]["delta"]
        lines.append(
            f"| {key} | {data['base_weight']:.2f} | {neg:+.5f} | {pos:+.5f} | {data['sensitivity']:.5f} |"
        )

    lines.extend([
        "",
        f"**Most sensitive to**: `{perturbation['sensitivity_ranking'][0]}`",
        f"**Least sensitive to**: `{perturbation['sensitivity_ranking'][-1]}`",
        "",
        "## Component Correlations (per-sentence)",
        "",
        "| | F1 | MRR | Top1 |",
        "|---|---|---|---|",
    ])

    for k1 in ["f1", "mrr", "top1"]:
        row = f"| {k1} |"
        for k2 in ["f1", "mrr", "top1"]:
            corr = correlations.get(k1, {}).get(k2, 0)
            row += f" {corr:.3f} |"
        lines.append(row)

    lines.extend([
        "",
        "High correlation (>0.8) between components means they double-count the same signal.",
        "",
        "## Latency Component Analysis",
        "",
        f"- p95 latency: {latency_analysis['p95_ms']:.0f}ms",
        f"- Normalized: {latency_analysis['latency_normalized']:.3f}",
        f"- Composite contribution: {latency_analysis['composite_contribution']:.4f} (fixed)",
        f"- Discriminative: {'Yes' if latency_analysis['is_discriminative'] else 'No (near-constant)'}",
        f"- {latency_analysis['note']}",
        "",
    ])

    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Composite score sensitivity analysis.")
    parser.add_argument("--report", type=Path, default=None, help="Existing benchmark JSON.")
    parser.add_argument("--db", type=Path, default=None, help="Database for fresh run.")
    parser.add_argument("--benchmark", type=Path, default=Path("benchmarks/myspellchecker_benchmark.yaml"))
    parser.add_argument("--semantic", type=Path, default=None)
    parser.add_argument("--reranker", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/results/sensitivity"))
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    if args.report and args.report.exists():
        report = json.loads(args.report.read_text(encoding="utf-8"))
    elif args.db and args.db.exists():
        from benchmarks.run_benchmark import run_benchmark

        kwargs: dict[str, Any] = {
            "benchmark_path": args.benchmark,
            "db_path": args.db,
            "level": "word",
            "warmup": 1,
            "enable_fusion": True,
        }
        if args.semantic and args.semantic.exists():
            kwargs["semantic_path"] = args.semantic
        if args.reranker and args.reranker.exists():
            kwargs["reranker_path"] = args.reranker
        report = run_benchmark(**kwargs)
    else:
        print("Error: Provide --report or --db")
        return 1

    # Get per-sentence data
    sentences = report.get("per_sentence_results", report.get("sentence_results", []))
    if not sentences:
        print("Error: No per-sentence results in report")
        return 1

    # Extract aggregate metrics
    overall = report["overall_metrics"]
    det = overall["detection"]
    sug = overall["suggestions"]
    fpr_data = overall["false_positive_rate"]
    latency_data = overall.get("latency_ms", overall.get("latency", {}))
    p95 = latency_data.get("p95", latency_data.get("p95_ms", 150))

    aggregate_metrics = {
        "f1": float(det["f1"]),
        "mrr": float(sug["mrr"]),
        "fpr": float(fpr_data["rate"]),
        "top1_accuracy": float(sug["top1_accuracy"]),
        "latency_norm": min(p95 / 500.0, 1.0),
    }

    print(f"Aggregate metrics: F1={aggregate_metrics['f1']:.4f} MRR={aggregate_metrics['mrr']:.4f} "
          f"FPR={aggregate_metrics['fpr']:.4f} Top1={aggregate_metrics['top1_accuracy']:.4f}")

    # 1. Weight perturbation
    perturbation = weight_perturbation_analysis(aggregate_metrics)

    # 2. Component correlations
    components = extract_per_sentence_components(sentences)
    correlations = compute_correlation_matrix(components)

    # 3. Latency analysis
    latency_analysis = analyze_latency_discriminative_power(sentences)

    # Render
    markdown = render_report(perturbation, correlations, latency_analysis)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path = args.output_dir / f"sensitivity_{ts}.md"
    json_path = args.output_dir / f"sensitivity_{ts}.json"

    md_path.write_text(markdown, encoding="utf-8")
    json_path.write_text(
        json.dumps({
            "perturbation": perturbation,
            "correlations": correlations,
            "latency_analysis": latency_analysis,
            "aggregate_metrics": aggregate_metrics,
        }, indent=2),
        encoding="utf-8",
    )

    print(f"\nSaved: {md_path}")
    print(f"\nSaved: {json_path}")
    print()
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
