#!/usr/bin/env python3
"""
Benchmark Statistical Power Analysis — v1.6.0 Sprint 1.2

Determines the minimum detectable effect size for the composite score
via bootstrap resampling. Answers: "Can we trust 0.004 composite deltas?"

Also computes per-component (F1, MRR, FPR, Top1) confidence intervals
and reports whether prior v1.5 sprint improvements were within noise.

Usage:
    python benchmarks/benchmark_power.py \
        --report benchmarks/results/latest_run.json

    # Or run fresh benchmark and analyze:
    python benchmarks/benchmark_power.py \
        --db data/mySpellChecker_production.db \
        --semantic data/semantic-v2.3-final \
        --reranker data/reranker-mlp-v4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))


def load_per_sentence_results(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract per-sentence TP/FP/FN/suggestion data from a benchmark report."""
    return report.get("per_sentence_results", report.get("sentence_results", []))


def compute_metrics_from_sentences(sentences: list[dict]) -> dict[str, float]:
    """Compute aggregate metrics from a list of per-sentence results."""
    total_tp = sum(s.get("tp", s.get("true_positives", 0)) for s in sentences)
    total_fp = sum(s.get("fp", s.get("false_positives", 0)) for s in sentences)
    total_fn = sum(s.get("fn", s.get("false_negatives", 0)) for s in sentences)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # MRR from matches
    reciprocal_ranks = []
    for s in sentences:
        for match in s.get("matches", s.get("span_matches", [])):
            if match.get("detected", match.get("matched", False)):
                rank = match.get("rank", match.get("rank_of_correct"))
                if rank is not None:
                    reciprocal_ranks.append(1.0 / rank)
                else:
                    reciprocal_ranks.append(0.0)
    mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0

    # Top1
    top1_correct = sum(
        1 for s in sentences for m in s.get("matches", s.get("span_matches", []))
        if m.get("detected", m.get("matched")) and m.get("top1_correct")
    )
    total_matched = sum(
        1 for s in sentences for m in s.get("matches", s.get("span_matches", []))
        if m.get("detected", m.get("matched"))
    )
    top1_acc = top1_correct / total_matched if total_matched > 0 else 0.0

    # FPR
    clean_sentences = [s for s in sentences if s.get("is_clean", False)]
    clean_fp = sum(s.get("fp", s.get("false_positives", 0)) for s in clean_sentences)
    fpr = clean_fp / len(clean_sentences) if clean_sentences else 0.0

    # Latency
    latencies = sorted(s.get("latency_ms", 0) for s in sentences)
    n = len(latencies)
    p95 = latencies[int(n * 0.95)] if n > 0 else 0.0
    latency_norm = min(p95 / 500.0, 1.0)

    # Composite
    composite = 0.30 * f1 + 0.25 * mrr + 0.20 * (1.0 - fpr) + 0.15 * top1_acc + 0.10 * (1.0 - latency_norm)

    return {
        "f1": f1,
        "mrr": float(mrr),
        "fpr": fpr,
        "top1_accuracy": top1_acc,
        "composite": composite,
        "precision": precision,
        "recall": recall,
    }


def bootstrap_ci(
    sentences: list[dict],
    n_bootstrap: int = 10000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Compute bootstrap confidence intervals for all metrics."""
    rng = np.random.default_rng(seed)
    n = len(sentences)

    metrics_keys = ["f1", "mrr", "fpr", "top1_accuracy", "composite"]
    bootstrap_samples: dict[str, list[float]] = {k: [] for k in metrics_keys}

    print(f"  Running {n_bootstrap:,} bootstrap resamples on {n} sentences...")
    start = time.perf_counter()

    for i in range(n_bootstrap):
        # Resample with replacement
        indices = rng.integers(0, n, size=n)
        sample = [sentences[idx] for idx in indices]
        metrics = compute_metrics_from_sentences(sample)
        for k in metrics_keys:
            bootstrap_samples[k].append(metrics[k])

    elapsed = time.perf_counter() - start
    print(f"  Done in {elapsed:.1f}s")

    # Compute CIs
    alpha = 1.0 - ci_level
    results = {}
    for k in metrics_keys:
        values = np.array(bootstrap_samples[k])
        lo = float(np.percentile(values, alpha / 2 * 100))
        hi = float(np.percentile(values, (1 - alpha / 2) * 100))
        mean = float(np.mean(values))
        std = float(np.std(values))
        results[k] = {
            "mean": round(mean, 5),
            "std": round(std, 5),
            "ci_lower": round(lo, 5),
            "ci_upper": round(hi, 5),
            "ci_width": round(hi - lo, 5),
            "min_detectable_effect": round((hi - lo) / 2, 5),
        }

    return results


def run_fresh_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """Run a fresh benchmark and return the full report with per-sentence data."""
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

    return run_benchmark(**kwargs)


def render_report(
    ci_results: dict[str, dict[str, float]],
    point_estimates: dict[str, float],
) -> str:
    """Render markdown report."""
    lines = [
        "# Benchmark Statistical Power Analysis",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Bootstrap Confidence Intervals (95%, 10K resamples)",
        "",
        "| Metric | Point Est. | Mean | Std | 95% CI | CI Width | Min Detectable |",
        "|--------|-----------|------|-----|--------|----------|----------------|",
    ]

    for metric, ci in ci_results.items():
        pe = point_estimates.get(metric, ci["mean"])
        lines.append(
            f"| {metric} | {pe:.4f} | {ci['mean']:.4f} | {ci['std']:.4f} | "
            f"[{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}] | "
            f"{ci['ci_width']:.4f} | {ci['min_detectable_effect']:.4f} |"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "- **CI Width**: The range of plausible values for the metric given sampling variation",
        "- **Min Detectable Effect**: Smallest improvement we can confidently distinguish from noise",
        "- If a sprint improvement was smaller than the Min Detectable Effect, it may be noise",
        "",
        "## v1.5 Sprint Improvements vs Statistical Power",
        "",
    ])

    composite_mde = ci_results["composite"]["min_detectable_effect"]
    lines.append(f"Composite Min Detectable Effect: **{composite_mde:.4f}**")
    lines.append("")

    known_deltas = [
        ("Sprint I-2 (curated pairs)", 0.0041),
        ("Sprint I-1 (fusion fix)", 0.002),
        ("Sprint A+B (meta-classifier)", 0.008),
    ]

    for name, delta in known_deltas:
        status = "DETECTABLE" if delta > composite_mde else "WITHIN NOISE"
        lines.append(f"- {name}: Δ={delta:.4f} → **{status}**")

    lines.append("")
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark statistical power analysis via bootstrap resampling."
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Path to existing benchmark JSON report (with per-sentence results).",
    )
    parser.add_argument("--db", type=Path, default=None, help="Database path (for fresh run).")
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("benchmarks/myspellchecker_benchmark.yaml"),
    )
    parser.add_argument("--semantic", type=Path, default=None)
    parser.add_argument("--reranker", type=Path, default=None)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/results/power_analysis"),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    if args.report and args.report.exists():
        print(f"Loading existing report: {args.report}")
        report = json.loads(args.report.read_text(encoding="utf-8"))
    elif args.db and args.db.exists():
        print("Running fresh benchmark...")
        report = run_fresh_benchmark(args)
    else:
        print("Error: Provide either --report (existing JSON) or --db (for fresh run)")
        return 1

    # Extract per-sentence data
    sentences = load_per_sentence_results(report)
    if not sentences:
        print("Error: No per-sentence results found in report.")
        print("  Available keys:", list(report.keys()))
        return 1

    print(f"  Found {len(sentences)} sentence results")

    # Compute point estimates
    point_estimates = compute_metrics_from_sentences(sentences)
    print(f"  Point estimates: composite={point_estimates['composite']:.4f}")

    # Bootstrap CI
    ci_results = bootstrap_ci(sentences, n_bootstrap=args.n_bootstrap)

    # Render and save
    markdown = render_report(ci_results, point_estimates)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path = args.output_dir / f"power_analysis_{ts}.md"
    json_path = args.output_dir / f"power_analysis_{ts}.json"

    md_path.write_text(markdown, encoding="utf-8")
    json_path.write_text(
        json.dumps({"ci_results": ci_results, "point_estimates": point_estimates}, indent=2),
        encoding="utf-8",
    )

    print(f"\nSaved: {md_path}")
    print(f"Saved: {json_path}")
    print()
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
