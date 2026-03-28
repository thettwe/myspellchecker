#!/usr/bin/env python3
"""
Run targeted-rule ablation matrix and generate a consolidated report.

Matrix (default):
- default
- no_hint
- no_inject
- no_grammar_tpl
- all_off

Each case is executed on benchmark and holdout datasets. The script writes:
1) raw benchmark JSONs per run
2) consolidated ablation JSON
3) consolidated ablation markdown summary
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from benchmarks.compare_runs import build_comparison  # noqa: E402
from benchmarks.run_benchmark import run_benchmark  # noqa: E402


@dataclass(frozen=True)
class AblationCase:
    """One targeted-rule toggle configuration."""

    case_id: str
    label: str
    targeted_rerank_hints: bool
    targeted_candidate_injections: bool
    targeted_grammar_completion_templates: bool


ABLATION_CASES: tuple[AblationCase, ...] = (
    AblationCase(
        case_id="default",
        label="Default",
        targeted_rerank_hints=True,
        targeted_candidate_injections=True,
        targeted_grammar_completion_templates=True,
    ),
    AblationCase(
        case_id="no_hint",
        label="No Targeted Rerank Hints",
        targeted_rerank_hints=False,
        targeted_candidate_injections=True,
        targeted_grammar_completion_templates=True,
    ),
    AblationCase(
        case_id="no_inject",
        label="No Targeted Candidate Injections",
        targeted_rerank_hints=True,
        targeted_candidate_injections=False,
        targeted_grammar_completion_templates=True,
    ),
    AblationCase(
        case_id="no_grammar_tpl",
        label="No Grammar Templates",
        targeted_rerank_hints=True,
        targeted_candidate_injections=True,
        targeted_grammar_completion_templates=False,
    ),
    AblationCase(
        case_id="all_off",
        label="All Targeted Rules Off",
        targeted_rerank_hints=False,
        targeted_candidate_injections=False,
        targeted_grammar_completion_templates=False,
    ),
)

ABLATION_CASE_MAP: dict[str, AblationCase] = {case.case_id: case for case in ABLATION_CASES}


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


def _pp(x: float) -> str:
    return f"{x * 100:.2f}%"


def resolve_case_order(case_ids: list[str] | None) -> list[AblationCase]:
    """Resolve case order with default baseline guaranteed at index 0."""
    if not case_ids:
        return list(ABLATION_CASES)

    ordered_ids: list[str] = []
    if "default" not in case_ids:
        ordered_ids.append("default")
    for case_id in case_ids:
        if case_id not in ordered_ids:
            ordered_ids.append(case_id)

    unknown = [case_id for case_id in ordered_ids if case_id not in ABLATION_CASE_MAP]
    if unknown:
        raise ValueError(f"Unknown case ids: {unknown}")

    return [ABLATION_CASE_MAP[case_id] for case_id in ordered_ids]


def run_case(
    *,
    case: AblationCase,
    dataset_path: Path,
    dataset_tag: str,
    db_path: Path,
    level: str,
    warmup: int,
    detector_path: Path | None,
    confidence: float,
    semantic_path: Path | None,
    enable_ner: bool,
    output_dir: Path,
    timestamp: str,
) -> tuple[Path, dict[str, Any]]:
    """Run one ablation case and persist its benchmark JSON."""
    report = run_benchmark(
        benchmark_path=dataset_path,
        db_path=db_path,
        level=level,
        warmup=warmup,
        detector_path=detector_path,
        confidence=confidence,
        semantic_path=semantic_path,
        enable_ner=enable_ner,
        targeted_rerank_hints=case.targeted_rerank_hints,
        targeted_candidate_injections=case.targeted_candidate_injections,
        targeted_grammar_completion_templates=case.targeted_grammar_completion_templates,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (
        f"ablation_{db_path.stem}_{dataset_tag}_{level}_{case.case_id}_{timestamp}.json"
    )
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path, report


def build_dataset_section(
    *,
    dataset_name: str,
    dataset_path: Path,
    case_order: list[AblationCase],
    case_results: dict[str, tuple[Path, dict[str, Any]]],
) -> dict[str, Any]:
    """Build one dataset section with per-case metrics and deltas vs default."""
    default_case_id = case_order[0].case_id
    baseline_path, baseline_report = case_results[default_case_id]

    runs: dict[str, dict[str, Any]] = {}
    for case in case_order:
        run_path, run_report = case_results[case.case_id]
        run_cfg = run_report.get("config", {})
        runs[case.case_id] = {
            "label": case.label,
            "file": str(run_path),
            "metrics": _extract_metric_snapshot(run_report),
            "toggles": {
                "targeted_rerank_hints": bool(run_cfg.get("targeted_rerank_hints", True)),
                "targeted_candidate_injections": bool(
                    run_cfg.get("targeted_candidate_injections", True)
                ),
                "targeted_grammar_completion_templates": bool(
                    run_cfg.get("targeted_grammar_completion_templates", True)
                ),
            },
        }

    deltas_vs_default: dict[str, dict[str, Any]] = {}
    baseline_with_file = dict(baseline_report)
    baseline_with_file["_file"] = str(baseline_path)
    for case in case_order[1:]:
        run_path, run_report = case_results[case.case_id]
        current_with_file = dict(run_report)
        current_with_file["_file"] = str(run_path)
        comparison = build_comparison(baseline_with_file, current_with_file)
        deltas_vs_default[case.case_id] = comparison["delta"]

    return {
        "dataset": dataset_name,
        "path": str(dataset_path),
        "default_case": default_case_id,
        "runs": runs,
        "deltas_vs_default": deltas_vs_default,
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Render consolidated markdown report."""
    lines: list[str] = []
    lines.append("# Ablation Matrix Report")
    lines.append("")
    lines.append(f"- Generated: `{report['generated_at']}`")
    lines.append(f"- DB: `{report['inputs']['db']}`")
    lines.append(f"- Level: `{report['inputs']['level']}`")
    lines.append(f"- Warmup: `{report['inputs']['warmup']}`")
    lines.append("")

    for dataset_name, section in report["datasets"].items():
        lines.append(f"## Dataset: {dataset_name}")
        lines.append("")
        lines.append(f"- Path: `{section['path']}`")
        lines.append("")
        lines.append("| Case | Top1 | ΔTop1 | F1 | Clean FPR | ΔRerank | ΔMissing | ΔTotal |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")

        default_case = section["default_case"]
        default_metrics = section["runs"][default_case]["metrics"]
        lines.append(
            f"| {default_case} | {_pp(default_metrics['top1_accuracy'])} | - | "
            f"{_pp(default_metrics['f1'])} | {_pp(default_metrics['clean_fpr'])} | - | - | - |"
        )

        for case_id, delta in section["deltas_vs_default"].items():
            run_metrics = section["runs"][case_id]["metrics"]
            miss_delta = delta["miss_buckets"]
            lines.append(
                f"| {case_id} | {_pp(run_metrics['top1_accuracy'])} | "
                f"{delta['metrics']['top1_accuracy'] * 100:+.2f}pp | "
                f"{_pp(run_metrics['f1'])} | {_pp(run_metrics['clean_fpr'])} | "
                f"{miss_delta['rerank']:+d} | {miss_delta['missing_candidate']:+d} | "
                f"{miss_delta['total']:+d} |"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


def run_ablation_matrix(args: argparse.Namespace) -> dict[str, Any]:
    """Execute ablation runs and assemble a consolidated report."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    case_order = resolve_case_order(args.cases)
    datasets: list[tuple[str, Path]] = [("benchmark", args.benchmark)]
    if not args.skip_holdout:
        datasets.append(("holdout", args.holdout))

    runs_dir = args.output_dir / f"runs_{timestamp}"
    runs_dir.mkdir(parents=True, exist_ok=True)

    dataset_sections: dict[str, Any] = {}
    for dataset_name, dataset_path in datasets:
        case_results: dict[str, tuple[Path, dict[str, Any]]] = {}
        for case in case_order:
            print(f"[{dataset_name}] running {case.case_id} ...")
            out_path, report = run_case(
                case=case,
                dataset_path=dataset_path,
                dataset_tag=dataset_name,
                db_path=args.db,
                level=args.level,
                warmup=args.warmup,
                detector_path=args.detector,
                confidence=args.confidence,
                semantic_path=args.semantic,
                enable_ner=args.ner,
                output_dir=runs_dir,
                timestamp=timestamp,
            )
            case_results[case.case_id] = (out_path, report)
            print(f"[{dataset_name}] done {case.case_id}: {out_path.name}")

        dataset_sections[dataset_name] = build_dataset_section(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            case_order=case_order,
            case_results=case_results,
        )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "db": str(args.db),
            "benchmark": str(args.benchmark),
            "holdout": None if args.skip_holdout else str(args.holdout),
            "skip_holdout": args.skip_holdout,
            "level": args.level,
            "warmup": args.warmup,
            "detector": str(args.detector) if args.detector else None,
            "confidence": args.confidence,
            "semantic": str(args.semantic) if args.semantic else None,
            "ner": args.ner,
            "output_dir": str(args.output_dir),
        },
        "cases": [asdict(case) for case in case_order],
        "datasets": dataset_sections,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ablation matrix on benchmark + holdout.")
    parser.add_argument("--db", type=Path, required=True, help="Path to spell checker database.")
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("benchmarks/myspellchecker_benchmark.yaml"),
        help="Primary benchmark YAML path.",
    )
    parser.add_argument(
        "--holdout",
        type=Path,
        default=None,
        help="Holdout benchmark YAML path.",
    )
    parser.add_argument(
        "--skip-holdout",
        action="store_true",
        default=False,
        help="Skip holdout execution and report only primary benchmark.",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=sorted(ABLATION_CASE_MAP),
        help="Optional subset of case ids. 'default' is auto-included if omitted.",
    )
    parser.add_argument(
        "--level",
        choices=["syllable", "word"],
        default="word",
        help="Validation level for all runs.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Warmup runs before each benchmark invocation.",
    )
    parser.add_argument(
        "--detector",
        type=Path,
        default=None,
        metavar="PATH",
        help="Optional detector ONNX path (same semantics as run_benchmark.py).",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.7,
        help="Detector confidence threshold when --detector is used.",
    )
    parser.add_argument(
        "--semantic",
        type=Path,
        default=None,
        metavar="PATH",
        help="Optional semantic ONNX path (same semantics as run_benchmark.py).",
    )
    parser.add_argument(
        "--ner",
        action="store_true",
        default=False,
        help="Enable NER suppression during runs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/results/ablation"),
        help="Output directory for run artifacts and consolidated report.",
    )
    parser.add_argument(
        "--output-json", type=Path, help="Optional explicit consolidated JSON path."
    )
    parser.add_argument(
        "--output-md", type=Path, help="Optional explicit consolidated markdown path."
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    if not args.db.exists():
        raise FileNotFoundError(f"Database not found: {args.db}")
    if not args.benchmark.exists():
        raise FileNotFoundError(f"Benchmark file not found: {args.benchmark}")
    if args.holdout is None:
        args.skip_holdout = True
    elif not args.skip_holdout and not args.holdout.exists():
        raise FileNotFoundError(
            f"Holdout file not found: {args.holdout}. Pass --skip-holdout to skip."
        )

    report = run_ablation_matrix(args)
    markdown = render_markdown(report)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_json = args.output_json or (args.output_dir / f"ablation_report_{ts}.json")
    output_md = args.output_md or (args.output_dir / f"ablation_report_{ts}.md")

    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md.write_text(markdown, encoding="utf-8")

    print(f"\nSaved consolidated JSON: {output_json}")
    print(f"Saved consolidated Markdown: {output_md}")
    print()
    print(markdown, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
