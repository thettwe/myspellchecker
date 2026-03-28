#!/usr/bin/env python3
"""
Audit targeted rerank rule telemetry across benchmark reports.

This script aggregates `rerank_rule_telemetry` from one or more benchmark
run JSON artifacts and ranks rules by practical risk/opportunity:
- high-fire + low-win-rate (prune/generalize candidates)
- low-fire + high-win-rate (high-value rare rules to keep/refactor carefully)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class RuleStats:
    """Aggregated telemetry for one rule id."""

    rule_id: str
    group: str
    fires: int = 0
    top1_changes: int = 0
    runs_seen: int = 0

    @property
    def win_rate(self) -> float:
        return (self.top1_changes / self.fires) if self.fires > 0 else 0.0


RuleTelemetryAgg = tuple[dict[str, RuleStats], dict[str, int]]


def _load_report(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _rule_group(rule_id: str) -> str:
    if ":" not in rule_id:
        return "unknown"
    return rule_id.split(":", 1)[0]


def aggregate_rule_telemetry(
    report_paths: list[Path],
) -> RuleTelemetryAgg:
    """Aggregate rerank telemetry from benchmark reports."""
    aggregated: dict[str, RuleStats] = {}
    input_stats = {
        "reports_scanned": len(report_paths),
        "reports_with_telemetry_key": 0,
        "reports_with_nonempty_telemetry": 0,
    }

    for path in report_paths:
        report = _load_report(path)
        telemetry = report.get("rerank_rule_telemetry", {})
        if not isinstance(telemetry, dict):
            continue
        input_stats["reports_with_telemetry_key"] += 1
        if telemetry:
            input_stats["reports_with_nonempty_telemetry"] += 1
        for rule_id, stats in telemetry.items():
            if not isinstance(rule_id, str) or not isinstance(stats, dict):
                continue
            rule = aggregated.setdefault(
                rule_id,
                RuleStats(rule_id=rule_id, group=_rule_group(rule_id)),
            )
            rule.fires += int(stats.get("fires", 0))
            rule.top1_changes += int(stats.get("top1_changes", 0))
            rule.runs_seen += 1

    return aggregated, input_stats


def classify_rules(
    aggregated: dict[str, RuleStats],
    *,
    high_fire_min: int,
    low_win_max: float,
    low_fire_max: int,
    high_win_min: float,
) -> dict[str, list[RuleStats]]:
    """Classify rules into risk/opportunity buckets."""
    high_fire_low_win: list[RuleStats] = []
    low_fire_high_win: list[RuleStats] = []
    balanced: list[RuleStats] = []

    for rule in aggregated.values():
        if rule.fires >= high_fire_min and rule.win_rate <= low_win_max:
            high_fire_low_win.append(rule)
        elif rule.fires <= low_fire_max and rule.win_rate >= high_win_min and rule.top1_changes > 0:
            low_fire_high_win.append(rule)
        else:
            balanced.append(rule)

    high_fire_low_win.sort(key=lambda r: (-r.fires, r.win_rate, r.rule_id))
    low_fire_high_win.sort(key=lambda r: (r.fires, -r.win_rate, -r.top1_changes, r.rule_id))
    balanced.sort(key=lambda r: (-r.top1_changes, -r.fires, r.rule_id))

    return {
        "high_fire_low_win": high_fire_low_win,
        "low_fire_high_win": low_fire_high_win,
        "balanced": balanced,
    }


def build_report(
    report_paths: list[Path],
    *,
    high_fire_min: int,
    low_win_max: float,
    low_fire_max: int,
    high_win_min: float,
) -> dict[str, Any]:
    aggregated, input_stats = aggregate_rule_telemetry(report_paths)
    buckets = classify_rules(
        aggregated,
        high_fire_min=high_fire_min,
        low_win_max=low_win_max,
        low_fire_max=low_fire_max,
        high_win_min=high_win_min,
    )

    by_group: dict[str, dict[str, int | float]] = {}
    for rule in aggregated.values():
        group = by_group.setdefault(
            rule.group,
            {"rules": 0, "fires": 0, "top1_changes": 0, "win_rate": 0.0},
        )
        group["rules"] = int(group["rules"]) + 1
        group["fires"] = int(group["fires"]) + rule.fires
        group["top1_changes"] = int(group["top1_changes"]) + rule.top1_changes

    for group, stats in by_group.items():
        fires = int(stats["fires"])
        top1_changes = int(stats["top1_changes"])
        stats["win_rate"] = round((top1_changes / fires), 4) if fires > 0 else 0.0
        by_group[group] = stats

    return {
        "inputs": [str(p) for p in report_paths],
        "thresholds": {
            "high_fire_min": high_fire_min,
            "low_win_max": low_win_max,
            "low_fire_max": low_fire_max,
            "high_win_min": high_win_min,
        },
        "summary": {
            "reports_scanned": input_stats["reports_scanned"],
            "reports_with_telemetry_key": input_stats["reports_with_telemetry_key"],
            "reports_with_nonempty_telemetry": input_stats["reports_with_nonempty_telemetry"],
            "total_rules": len(aggregated),
            "high_fire_low_win": len(buckets["high_fire_low_win"]),
            "low_fire_high_win": len(buckets["low_fire_high_win"]),
            "balanced": len(buckets["balanced"]),
            "total_fires": sum(r.fires for r in aggregated.values()),
            "total_top1_changes": sum(r.top1_changes for r in aggregated.values()),
        },
        "by_group": dict(
            sorted(by_group.items(), key=lambda item: (-int(item[1]["fires"]), item[0]))
        ),
        "high_fire_low_win": [
            asdict(r) | {"win_rate": round(r.win_rate, 4)} for r in buckets["high_fire_low_win"]
        ],
        "low_fire_high_win": [
            asdict(r) | {"win_rate": round(r.win_rate, 4)} for r in buckets["low_fire_high_win"]
        ],
        "balanced_top": [
            asdict(r) | {"win_rate": round(r.win_rate, 4)} for r in buckets["balanced"][:20]
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Targeted Rule Telemetry Audit")
    lines.append("")
    lines.append("## Inputs")
    for p in report["inputs"]:
        lines.append(f"- `{p}`")
    lines.append("")
    lines.append("## Summary")
    s = report["summary"]
    lines.append(f"- Reports scanned: `{s['reports_scanned']}`")
    lines.append(f"- Reports with telemetry key: `{s['reports_with_telemetry_key']}`")
    lines.append(f"- Reports with non-empty telemetry: `{s['reports_with_nonempty_telemetry']}`")
    lines.append(f"- Total rules: `{s['total_rules']}`")
    lines.append(f"- Total fires: `{s['total_fires']}`")
    lines.append(f"- Total top1 changes: `{s['total_top1_changes']}`")
    lines.append(f"- High-fire low-win: `{s['high_fire_low_win']}`")
    lines.append(f"- Low-fire high-win: `{s['low_fire_high_win']}`")
    lines.append("")

    lines.append("## High-Fire Low-Win (Review/Prune Candidates)")
    lines.append("")
    lines.append("| Rule ID | Group | Fires | Top1Δ | Win Rate | Runs |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for row in report["high_fire_low_win"][:40]:
        lines.append(
            f"| {row['rule_id']} | {row['group']} | {row['fires']} | {row['top1_changes']} | "
            f"{row['win_rate']:.2%} | {row['runs_seen']} |"
        )
    if not report["high_fire_low_win"]:
        lines.append("| _none_ | - | 0 | 0 | 0.00% | 0 |")
    lines.append("")

    lines.append("## Low-Fire High-Win (Keep/Generalize Carefully)")
    lines.append("")
    lines.append("| Rule ID | Group | Fires | Top1Δ | Win Rate | Runs |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for row in report["low_fire_high_win"][:40]:
        lines.append(
            f"| {row['rule_id']} | {row['group']} | {row['fires']} | {row['top1_changes']} | "
            f"{row['win_rate']:.2%} | {row['runs_seen']} |"
        )
    if not report["low_fire_high_win"]:
        lines.append("| _none_ | - | 0 | 0 | 0.00% | 0 |")
    lines.append("")

    lines.append("## Rule Groups")
    lines.append("")
    lines.append("| Group | Rules | Fires | Top1Δ | Win Rate |")
    lines.append("|---|---:|---:|---:|---:|")
    for group, stats in report["by_group"].items():
        lines.append(
            f"| {group} | {stats['rules']} | {stats['fires']} | {stats['top1_changes']} | "
            f"{stats['win_rate']:.2%} |"
        )

    return "\n".join(lines) + "\n"


def _resolve_report_paths(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []

    if args.reports:
        paths.extend(Path(p) for p in args.reports)
    if args.report_glob:
        base = Path(".")
        paths.extend(sorted(base.glob(args.report_glob)))

    unique: list[Path] = []
    seen: set[Path] = set()
    for p in paths:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        unique.append(p)

    if not unique:
        raise ValueError("No report paths provided. Use --reports or --report-glob.")
    return unique


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit targeted rerank rule telemetry.")
    parser.add_argument(
        "--reports",
        nargs="+",
        help="Benchmark run JSON report paths.",
    )
    parser.add_argument(
        "--report-glob",
        help="Optional glob pattern for report paths (e.g. benchmarks/results/*.json).",
    )
    parser.add_argument(
        "--high-fire-min",
        type=int,
        default=5,
        help="Minimum fires for high-fire classification.",
    )
    parser.add_argument(
        "--low-win-max",
        type=float,
        default=0.25,
        help="Maximum win rate for high-fire low-win classification.",
    )
    parser.add_argument(
        "--low-fire-max",
        type=int,
        default=3,
        help="Maximum fires for low-fire classification.",
    )
    parser.add_argument(
        "--high-win-min",
        type=float,
        default=0.75,
        help="Minimum win rate for low-fire high-win classification.",
    )
    parser.add_argument("--output-json", help="Optional output path for JSON report.")
    parser.add_argument("--output-md", help="Optional output path for markdown report.")
    args = parser.parse_args()

    report_paths = _resolve_report_paths(args)
    report = build_report(
        report_paths,
        high_fire_min=args.high_fire_min,
        low_win_max=args.low_win_max,
        low_fire_max=args.low_fire_max,
        high_win_min=args.high_win_min,
    )
    markdown = render_markdown(report)
    print(markdown, end="")

    if args.output_json:
        out_json = Path(args.output_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.output_md:
        out_md = Path(args.output_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(markdown, encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
