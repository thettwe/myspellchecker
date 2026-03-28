#!/usr/bin/env python3
"""
Compare two benchmark run JSON artifacts.

Outputs:
- metric deltas (detection + suggestion + clean FPR)
- miss bucket deltas (rerank, missing-candidate, missed-detection)
- per-system-type miss distribution
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _extract_metric_snapshot(report: dict[str, Any]) -> dict[str, float]:
    overall = report["overall_metrics"]
    det = overall["detection"]
    sug = overall["suggestions"]
    fpr = overall["false_positive_rate"]
    return {
        "precision": float(det["precision"]),
        "recall": float(det["recall"]),
        "f1": float(det["f1"]),
        "top1_accuracy": float(sug["top1_accuracy"]),
        "top3_accuracy": float(sug["top3_accuracy"]),
        "top5_accuracy": float(sug["top5_accuracy"]),
        "mrr": float(sug["mrr"]),
        "clean_fpr": float(fpr["rate"]),
    }


def _extract_miss_buckets(report: dict[str, Any]) -> dict[str, Any]:
    rerank = 0
    missing_candidate = 0
    missed_detection = 0

    rerank_types: Counter[str] = Counter()
    missing_types: Counter[str] = Counter()
    missed_types: Counter[str] = Counter()

    for sentence in report.get("per_sentence_results", []):
        for match in sentence.get("matches", []) or []:
            detected = bool(match.get("detected"))
            system_type = str(match.get("system_type", "undetected"))

            if not detected:
                missed_detection += 1
                missed_types["undetected"] += 1
                continue

            # top1_correct only exists in report when gold_correction is truthy.
            if "top1_correct" not in match:
                continue

            if match.get("top1_correct") is True:
                continue

            rank = match.get("rank")
            if rank is None:
                missing_candidate += 1
                missing_types[system_type] += 1
            elif int(rank) > 1:
                rerank += 1
                rerank_types[system_type] += 1

    return {
        "counts": {
            "rerank": rerank,
            "missing_candidate": missing_candidate,
            "missed_detection": missed_detection,
            "total": rerank + missing_candidate + missed_detection,
        },
        "by_system_type": {
            "rerank": dict(sorted(rerank_types.items(), key=lambda x: (-x[1], x[0]))),
            "missing_candidate": dict(sorted(missing_types.items(), key=lambda x: (-x[1], x[0]))),
            "missed_detection": dict(sorted(missed_types.items(), key=lambda x: (-x[1], x[0]))),
        },
    }


def _delta(cur: dict[str, float], base: dict[str, float]) -> dict[str, float]:
    return {k: round(cur[k] - base[k], 4) for k in cur}


def _format_pp(x: float) -> str:
    return f"{x * 100:.2f}%"


def build_comparison(baseline: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    base_metrics = _extract_metric_snapshot(baseline)
    curr_metrics = _extract_metric_snapshot(current)
    base_miss = _extract_miss_buckets(baseline)
    curr_miss = _extract_miss_buckets(current)

    miss_delta = {
        key: curr_miss["counts"][key] - base_miss["counts"][key]
        for key in ("rerank", "missing_candidate", "missed_detection", "total")
    }

    return {
        "baseline": {
            "file": baseline.get("_file"),
            "metrics": base_metrics,
            "miss_buckets": base_miss,
        },
        "current": {
            "file": current.get("_file"),
            "metrics": curr_metrics,
            "miss_buckets": curr_miss,
        },
        "delta": {
            "metrics": _delta(curr_metrics, base_metrics),
            "miss_buckets": miss_delta,
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    b = report["baseline"]
    c = report["current"]
    d = report["delta"]
    lines = []
    lines.append("# Benchmark Comparison")
    lines.append("")
    lines.append(f"- Baseline: `{b['file']}`")
    lines.append(f"- Current: `{c['file']}`")
    lines.append("")
    lines.append("## Metric Deltas")
    lines.append("")
    lines.append("| Metric | Baseline | Current | Delta |")
    lines.append("|---|---:|---:|---:|")
    for key in (
        "precision",
        "recall",
        "f1",
        "top1_accuracy",
        "top3_accuracy",
        "top5_accuracy",
        "mrr",
    ):
        lines.append(
            f"| {key} | {_format_pp(b['metrics'][key])} | {_format_pp(c['metrics'][key])} | "
            f"{d['metrics'][key] * 100:+.2f}pp |"
        )
    lines.append(
        f"| clean_fpr | {_format_pp(b['metrics']['clean_fpr'])} | "
        f"{_format_pp(c['metrics']['clean_fpr'])} | "
        f"{d['metrics']['clean_fpr'] * 100:+.2f}pp |"
    )
    lines.append("")
    lines.append("## Miss Buckets")
    lines.append("")
    lines.append("| Bucket | Baseline | Current | Delta |")
    lines.append("|---|---:|---:|---:|")
    for key in ("rerank", "missing_candidate", "missed_detection", "total"):
        lines.append(
            f"| {key} | {b['miss_buckets']['counts'][key]} | {c['miss_buckets']['counts'][key]} | "
            f"{d['miss_buckets'][key]:+d} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two benchmark run JSON files.")
    parser.add_argument("--baseline", required=True, help="Baseline benchmark JSON path.")
    parser.add_argument("--current", required=True, help="Current benchmark JSON path.")
    parser.add_argument(
        "--output-json", help="Optional output path for machine-readable comparison JSON."
    )
    parser.add_argument("--output-md", help="Optional output path for markdown summary.")
    args = parser.parse_args()

    baseline_path = Path(args.baseline)
    current_path = Path(args.current)
    baseline = _load_json(baseline_path)
    current = _load_json(current_path)
    baseline["_file"] = str(baseline_path)
    current["_file"] = str(current_path)

    comparison = build_comparison(baseline, current)
    markdown = render_markdown(comparison)
    print(markdown, end="")

    if args.output_json:
        out_json = Path(args.output_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.output_md:
        out_md = Path(args.output_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(markdown, encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
