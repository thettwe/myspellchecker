#!/usr/bin/env python3
"""
Strategy ablation report — parses benchmark JSON to produce per-strategy TP/FP/FN metrics.

Usage:
    python benchmarks/ablation_report.py benchmarks/results/benchmark_*.json
    python benchmarks/ablation_report.py --compare baseline.json no_fast_path.json
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def analyze_single(report: dict) -> dict:
    """Extract per-strategy metrics from a benchmark result file."""
    strategy_tp: Counter = Counter()
    strategy_fp_clean: Counter = Counter()
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for sr in report["per_sentence_results"]:
        total_tp += sr.get("tp", 0)
        total_fp += sr.get("fp", 0)
        total_fn += sr.get("fn", 0)

        # Count TP per strategy from matched detections
        for m in sr.get("matches", []):
            if m.get("detected"):
                strategy = m.get("source_strategy", "unknown")
                strategy_tp[strategy] += 1

        # Count FP per strategy from clean sentence details
        if sr.get("is_clean") and sr.get("false_positive_details"):
            for fpd in sr["false_positive_details"]:
                strategy = fpd.get("source_strategy", "unknown")
                strategy_fp_clean[strategy] += 1

    # Build combined table
    all_strategies = sorted(set(strategy_tp.keys()) | set(strategy_fp_clean.keys()))
    result = {
        "overall": {
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "precision": total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0,
            "recall": total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0,
        },
        "per_strategy": {},
        "fn_reasons": report.get("fn_reason_telemetry", {}).get("histogram", {}),
    }
    f1_denom = result["overall"]["precision"] + result["overall"]["recall"]
    result["overall"]["f1"] = (
        2 * result["overall"]["precision"] * result["overall"]["recall"] / f1_denom
        if f1_denom > 0
        else 0
    )

    for s in all_strategies:
        tp = strategy_tp.get(s, 0)
        fp_clean = strategy_fp_clean.get(s, 0)
        result["per_strategy"][s] = {
            "tp": tp,
            "fp_on_clean": fp_clean,
        }

    return result


def print_report(analysis: dict, label: str = "Baseline") -> None:
    """Pretty-print the ablation report."""
    ov = analysis["overall"]
    print(f"\n{'=' * 80}")
    print(f"  {label}")
    print(f"{'=' * 80}")
    print(
        f"  TP: {ov['tp']}  FP: {ov['fp']}  FN: {ov['fn']}  "
        f"P: {ov['precision']:.4f}  R: {ov['recall']:.4f}  F1: {ov['f1']:.4f}"
    )

    fn_reasons = analysis.get("fn_reasons", {})
    if fn_reasons:
        print("\n  FN Reasons:")
        for reason, count in sorted(fn_reasons.items(), key=lambda x: -x[1]):
            print(f"    {reason}: {count}")

    strategies = analysis["per_strategy"]
    if strategies:
        print(f"\n  {'Strategy':<45} {'TP':>5} {'FP(clean)':>10} {'Net':>6}")
        print(f"  {'─' * 45} {'─' * 5} {'─' * 10} {'─' * 6}")
        sorted_strats = sorted(strategies.items(), key=lambda x: -x[1]["tp"])
        for name, metrics in sorted_strats:
            tp = metrics["tp"]
            fp = metrics["fp_on_clean"]
            net = tp - fp
            marker = " **" if fp > tp else ""
            print(f"  {name:<45} {tp:>5} {fp:>10} {net:>+6}{marker}")
        total_tp = sum(m["tp"] for m in strategies.values())
        total_fp = sum(m["fp_on_clean"] for m in strategies.values())
        print(f"  {'─' * 45} {'─' * 5} {'─' * 10} {'─' * 6}")
        print(f"  {'TOTAL':<45} {total_tp:>5} {total_fp:>10} {total_tp - total_fp:>+6}")


def print_comparison(baseline: dict, variant: dict, label: str = "Variant") -> None:
    """Compare two benchmark results side by side."""
    b = baseline["overall"]
    v = variant["overall"]
    print(f"\n{'=' * 80}")
    print(f"  COMPARISON: Baseline vs {label}")
    print(f"{'=' * 80}")
    print(f"  {'Metric':<15} {'Baseline':>10} {label:>15} {'Delta':>10}")
    print(f"  {'─' * 15} {'─' * 10} {'─' * 15} {'─' * 10}")
    for key in ["tp", "fp", "fn", "precision", "recall", "f1"]:
        bv = b[key]
        vv = v[key]
        delta = vv - bv
        if isinstance(bv, float):
            print(f"  {key:<15} {bv:>10.4f} {vv:>15.4f} {delta:>+10.4f}")
        else:
            print(f"  {key:<15} {bv:>10} {vv:>15} {delta:>+10}")

    # Per-strategy delta
    b_strats = baseline["per_strategy"]
    v_strats = variant["per_strategy"]
    all_names = sorted(set(b_strats.keys()) | set(v_strats.keys()))
    hdr = f"{'B-TP':>5} {'V-TP':>5} {'dTP':>5} {'B-FP':>5} {'V-FP':>5} {'dFP':>5}"
    print(f"\n  {'Strategy':<40} {hdr}")
    print(f"  {'─' * 40} {'─' * 5} {'─' * 5} {'─' * 5} {'─' * 5} {'─' * 5} {'─' * 5}")
    for name in all_names:
        bm = b_strats.get(name, {"tp": 0, "fp_on_clean": 0})
        vm = v_strats.get(name, {"tp": 0, "fp_on_clean": 0})
        dtp = vm["tp"] - bm["tp"]
        dfp = vm["fp_on_clean"] - bm["fp_on_clean"]
        if dtp != 0 or dfp != 0:
            print(
                f"  {name:<40} {bm['tp']:>5} {vm['tp']:>5} {dtp:>+5} "
                f"{bm['fp_on_clean']:>5} {vm['fp_on_clean']:>5} {dfp:>+5}"
            )


def main():
    parser = argparse.ArgumentParser(description="Strategy ablation analysis")
    parser.add_argument("files", nargs="+", type=Path, help="Benchmark result JSON files")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare first file (baseline) vs subsequent files",
    )
    args = parser.parse_args()

    analyses = []
    for f in args.files:
        if not f.exists():
            print(f"Error: {f} not found", file=sys.stderr)
            sys.exit(1)
        with open(f) as fh:
            report = json.load(fh)
        analyses.append((f.stem, analyze_single(report)))

    if args.compare and len(analyses) >= 2:
        _, baseline = analyses[0]
        print_report(baseline, label=f"Baseline: {analyses[0][0]}")
        for label, variant in analyses[1:]:
            print_report(variant, label=f"Variant: {label}")
            print_comparison(baseline, variant, label=label)
    else:
        for label, analysis in analyses:
            print_report(analysis, label=label)


if __name__ == "__main__":
    main()
