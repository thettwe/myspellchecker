#!/usr/bin/env python3
"""
Component Ablation Matrix — v1.6.0 Sprint 1.1

Tests the net contribution of each major subsystem by running the benchmark
with one component disabled at a time.

Configurations:
  1. baseline        — All components enabled (production config)
  2. no_mlm          — Disable SemanticStrategy + ConfusableSemanticStrategy
  3. no_reranker     — Disable NeuralReranker
  4. no_ngram        — Disable N-gram context checking (use_context_checker=False)
  5. no_grammar      — Disable grammar rules (SyntacticRuleStrategy)
  6. mlm_only        — Enable MLM but disable reranker (isolates MLM contribution)

Usage:
    python benchmarks/run_component_ablation.py \
        --db data/mySpellChecker_production.db \
        --semantic data/semantic-v2.3-final \
        --reranker data/reranker-mlp-v4

    # Quick single-config test:
    python benchmarks/run_component_ablation.py \
        --db data/mySpellChecker_production.db \
        --cases baseline no_mlm
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))

from benchmarks.run_benchmark import run_benchmark  # noqa: E402


@dataclass(frozen=True)
class ComponentCase:
    """One component ablation configuration."""

    case_id: str
    label: str
    # What to disable
    disable_semantic: bool = False
    disable_reranker: bool = False
    disable_ngram: bool = False
    disable_grammar: bool = False


COMPONENT_CASES: tuple[ComponentCase, ...] = (
    ComponentCase(
        case_id="baseline",
        label="Baseline (all enabled)",
    ),
    ComponentCase(
        case_id="no_mlm",
        label="No MLM (semantic + confusable_semantic disabled)",
        disable_semantic=True,
    ),
    ComponentCase(
        case_id="no_reranker",
        label="No Neural Reranker",
        disable_reranker=True,
    ),
    ComponentCase(
        case_id="no_ngram",
        label="No N-gram Context",
        disable_ngram=True,
    ),
    ComponentCase(
        case_id="no_grammar",
        label="No Grammar Rules",
        disable_grammar=True,
    ),
    ComponentCase(
        case_id="mlm_only",
        label="MLM enabled, no reranker (isolate MLM)",
        disable_reranker=True,
    ),
)

CASE_MAP: dict[str, ComponentCase] = {c.case_id: c for c in COMPONENT_CASES}


def _extract_metrics(report: dict[str, Any]) -> dict[str, float]:
    """Extract key metrics from a benchmark report."""
    overall = report["overall_metrics"]
    det = overall["detection"]
    sug = overall["suggestions"]
    fpr_data = overall["false_positive_rate"]
    latency = overall.get("latency_ms", overall.get("latency", {}))
    return {
        "precision": float(det["precision"]),
        "recall": float(det["recall"]),
        "f1": float(det["f1"]),
        "top1_accuracy": float(sug["top1_accuracy"]),
        "top3_accuracy": float(sug.get("top3_accuracy", 0)),
        "mrr": float(sug["mrr"]),
        "fpr": float(fpr_data["rate"]),
        "composite": float(overall["composite_score"]),
        "tp": int(det["true_positives"]),
        "fp": int(fpr_data.get("total_fp_on_clean", fpr_data.get("false_positives", 0))),
        "fn": int(det["false_negatives"]),
        "p95_ms": float(latency.get("p95", latency.get("p95_ms", 0))),
    }


def run_case(
    case: ComponentCase,
    *,
    benchmark_path: Path,
    db_path: Path,
    semantic_path: Path | None,
    reranker_path: Path | None,
) -> dict[str, Any]:
    """Run one ablation case and return the full report."""
    kwargs: dict[str, Any] = {
        "benchmark_path": benchmark_path,
        "db_path": db_path,
        "level": "word",
        "warmup": 1,
        "enable_fusion": True,
        "fusion_threshold": 0.5,
    }

    # Semantic / MLM
    if semantic_path and not case.disable_semantic:
        kwargs["semantic_path"] = semantic_path
        kwargs["no_confusable_semantic"] = False
    else:
        kwargs["semantic_path"] = None
        kwargs["no_confusable_semantic"] = True

    # Reranker
    if reranker_path and not case.disable_reranker:
        kwargs["reranker_path"] = reranker_path
    else:
        kwargs["reranker_path"] = None

    # N-gram context — handled via config override after SpellChecker init
    # We pass a flag and patch inside run_benchmark... but run_benchmark doesn't
    # have this toggle. We need to work around it.
    # Actually, use_context_checker is on SpellCheckerConfig.
    # The cleanest approach: for no_ngram, we call run_benchmark with a monkey-patch.

    if case.disable_ngram or case.disable_grammar:
        # For these cases, we need to directly instantiate the checker with modified config
        return _run_with_config_override(case, **kwargs)

    return run_benchmark(**kwargs)


def _run_with_config_override(
    case: ComponentCase,
    *,
    benchmark_path: Path,
    db_path: Path,
    level: str = "word",
    warmup: int = 1,
    semantic_path: Path | None = None,
    no_confusable_semantic: bool = True,
    reranker_path: Path | None = None,
    enable_fusion: bool = True,
    fusion_threshold: float = 0.5,
    **_extra: Any,
) -> dict[str, Any]:
    """Run benchmark with SpellCheckerConfig overrides for ngram/grammar toggling."""
    from benchmarks.run_benchmark import (
        SentenceResult,
        compute_db_hash,
        load_benchmark,
        match_errors,
    )
    from myspellchecker import SpellChecker
    from myspellchecker.core.config.algorithm_configs import (
        NeuralRerankerConfig,
        SemanticConfig,
    )
    from myspellchecker.core.config.main import SpellCheckerConfig
    from myspellchecker.core.constants import ValidationLevel
    from myspellchecker.providers.sqlite import SQLiteProvider

    benchmark = load_benchmark(benchmark_path)
    sentences = benchmark["sentences"]

    config_kwargs: dict[str, Any] = {}

    # Semantic
    if semantic_path is not None:
        sem_model_file = (
            semantic_path if semantic_path.suffix == ".onnx" else semantic_path / "model.onnx"
        )
        sem_tokenizer_dir = sem_model_file.parent
        config_kwargs["semantic"] = SemanticConfig(
            model_path=str(sem_model_file),
            tokenizer_path=str(sem_tokenizer_dir),
        )

    # Reranker
    if reranker_path is not None:
        model_file = reranker_path / "reranker.onnx"
        stats_file = reranker_path / "reranker.onnx.stats.json"
        if model_file.exists():
            config_kwargs["neural_reranker"] = NeuralRerankerConfig(
                enabled=True,
                model_path=str(model_file),
                stats_path=str(stats_file) if stats_file.exists() else None,
            )

    config = SpellCheckerConfig(**config_kwargs)

    # Apply overrides
    if case.disable_ngram:
        config.use_context_checker = False

    if case.disable_grammar:
        # Disable grammar by setting all thresholds to maximum (1.0) — nothing will pass
        config.grammar_engine.default_confidence_threshold = 1.0
        config.grammar_engine.exact_match_confidence = 1.0
        config.grammar_engine.high_confidence = 1.0
        config.grammar_engine.medium_confidence = 1.0
        config.grammar_engine.pos_sequence_confidence = 1.0

    if enable_fusion:
        config.validation.use_candidate_fusion = True
        config.validation.fusion_confidence_threshold = fusion_threshold

    if not no_confusable_semantic and semantic_path:
        config.validation.use_confusable_semantic = True

    # Initialize checker
    provider = SQLiteProvider(database_path=str(db_path))
    checker = SpellChecker(config=config, provider=provider)
    val_level = ValidationLevel.WORD if level == "word" else ValidationLevel.SYLLABLE

    # Warmup
    warmup_text = "ကျွန်တော် ကျန်းမာပါတယ်"
    for _ in range(warmup):
        checker.check(warmup_text, level=val_level)

    # Run benchmark (simplified — mirrors core logic from run_benchmark)
    results: list[SentenceResult] = []
    total_start = time.perf_counter()

    # Scope filtering — match run_benchmark default (spelling only)
    _scope_set = {"spelling"}

    def _in_scope(err: dict) -> bool:
        return err.get("scope", "spelling") in _scope_set

    for sentence in sentences:
        input_text = sentence["input"]
        is_clean = sentence["is_clean"]
        all_gold_errors = sentence.get("expected_errors", [])
        gold_errors = [e for e in all_gold_errors if _in_scope(e)]
        out_of_scope = [e for e in all_gold_errors if not _in_scope(e)]
        # Sentences with no in-scope errors are treated as clean for FPR
        if not gold_errors and out_of_scope:
            is_clean = True

        start = time.perf_counter()
        response = checker.check(input_text, level=val_level)
        elapsed_ms = (time.perf_counter() - start) * 1000

        system_errors = []
        for err in response.errors:
            sys_err = {
                "text": err.text,
                "position": err.position,
                "suggestions": [s.text if hasattr(s, "text") else str(s) for s in err.suggestions]
                if err.suggestions
                else [],
                "error_type": err.error_type if hasattr(err, "error_type") else "unknown",
                "source_strategy": err.source_strategy
                if hasattr(err, "source_strategy")
                else "unknown",
            }
            system_errors.append(sys_err)

        # Use ALL gold errors for matching (out-of-scope absorb FPs)
        span_matches, tp, fp, fn = match_errors(all_gold_errors, system_errors)
        # But only count in-scope misses as FN
        fn = len(gold_errors) - tp

        result = SentenceResult(
            sentence_id=sentence.get("id", ""),
            input_text=input_text,
            is_clean=is_clean,
            difficulty_tier=sentence.get("difficulty_tier"),
            domain=sentence.get("domain", "unknown"),
            latency_ms=elapsed_ms,
            expected_error_count=len(gold_errors),
            detected_error_count=len(system_errors),
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            span_matches=span_matches,
            system_errors=system_errors,
        )
        results.append(result)

    total_elapsed = (time.perf_counter() - total_start) * 1000

    # Compute aggregate metrics
    total_tp = sum(r.true_positives for r in results)
    total_fp = sum(r.false_positives for r in results)
    total_fn = sum(r.false_negatives for r in results)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Suggestion metrics
    all_matches = [m for r in results for m in r.span_matches if m.matched]
    top1_correct = sum(1 for m in all_matches if m.top1_correct)
    top1_acc = top1_correct / len(all_matches) if all_matches else 0.0
    top3_correct = sum(1 for m in all_matches if m.top3_correct)
    top3_acc = top3_correct / len(all_matches) if all_matches else 0.0

    reciprocal_ranks = []
    for m in all_matches:
        if m.rank_of_correct is not None:
            reciprocal_ranks.append(1.0 / m.rank_of_correct)
        else:
            reciprocal_ranks.append(0.0)
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0

    # FPR
    clean_results = [r for r in results if r.is_clean]
    clean_fp = sum(r.false_positives for r in clean_results)
    clean_total = len(clean_results)
    fpr = clean_fp / clean_total if clean_total > 0 else 0.0

    # Latency
    latencies = sorted(r.latency_ms for r in results)
    n = len(latencies)
    p50 = latencies[int(n * 0.5)] if n > 0 else 0.0
    p95 = latencies[int(n * 0.95)] if n > 0 else 0.0
    p99 = latencies[int(n * 0.99)] if n > 0 else 0.0

    # Composite
    latency_normalized = min(p95 / 500.0, 1.0)
    composite = (
        0.30 * f1
        + 0.25 * mrr
        + 0.20 * (1.0 - fpr)
        + 0.15 * top1_acc
        + 0.10 * (1.0 - latency_normalized)
    )

    return {
        "benchmark_version": benchmark.get("version", "1.0.0"),
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "case_id": case.case_id,
            "disable_semantic": case.disable_semantic,
            "disable_reranker": case.disable_reranker,
            "disable_ngram": case.disable_ngram,
            "disable_grammar": case.disable_grammar,
        },
        "database": {
            "name": db_path.name,
            "hash": compute_db_hash(db_path),
        },
        "overall_metrics": {
            "detection": {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "true_positives": total_tp,
                "false_negatives": total_fn,
            },
            "suggestions": {
                "top1_accuracy": round(top1_acc, 4),
                "top3_accuracy": round(top3_acc, 4),
                "mrr": round(mrr, 4),
            },
            "false_positive_rate": {
                "rate": round(fpr, 4),
                "false_positives": clean_fp,
                "clean_sentences": clean_total,
            },
            "latency": {
                "p50_ms": round(p50, 1),
                "p95_ms": round(p95, 1),
                "p99_ms": round(p99, 1),
                "total_ms": round(total_elapsed, 1),
            },
            "composite_score": round(composite, 4),
        },
    }


def run_ablation_matrix(args: argparse.Namespace) -> dict[str, Any]:
    """Run all ablation cases and produce a consolidated report."""
    cases = [CASE_MAP[c] for c in args.cases] if args.cases else list(COMPONENT_CASES)

    # Validate paths
    semantic_path = args.semantic if args.semantic and args.semantic.exists() else None
    reranker_path = args.reranker if args.reranker and args.reranker.exists() else None

    if not semantic_path:
        # Remove cases that require semantic to show difference
        cases = [c for c in cases if c.case_id not in ("no_mlm", "mlm_only")]
        print("  [WARN] No semantic model provided — skipping no_mlm and mlm_only cases")

    if not reranker_path:
        cases = [c for c in cases if c.case_id not in ("no_reranker", "mlm_only")]
        print("  [WARN] No reranker model provided — skipping no_reranker and mlm_only cases")

    results: dict[str, dict[str, Any]] = {}
    for case in cases:
        print(f"\n{'=' * 60}")
        print(f"  Running: {case.label}")
        print(f"{'=' * 60}")
        start = time.perf_counter()

        report = run_case(
            case,
            benchmark_path=args.benchmark,
            db_path=args.db,
            semantic_path=semantic_path,
            reranker_path=reranker_path,
        )
        elapsed = time.perf_counter() - start
        metrics = _extract_metrics(report)
        results[case.case_id] = {
            "label": case.label,
            "metrics": metrics,
            "elapsed_s": round(elapsed, 1),
        }
        print(f"  Done in {elapsed:.1f}s — composite={metrics['composite']:.4f}")

    # Compute deltas vs baseline
    baseline_metrics = results.get("baseline", {}).get("metrics", {})
    if baseline_metrics:
        for case_id, result in results.items():
            if case_id == "baseline":
                result["delta"] = {}
                continue
            delta = {}
            for key in ("f1", "mrr", "fpr", "top1_accuracy", "composite", "tp", "fp", "fn"):
                if key in result["metrics"] and key in baseline_metrics:
                    delta[key] = round(result["metrics"][key] - baseline_metrics[key], 4)
            result["delta"] = delta

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "db": str(args.db),
            "benchmark": str(args.benchmark),
            "semantic": str(semantic_path) if semantic_path else None,
            "reranker": str(reranker_path) if reranker_path else None,
        },
        "results": results,
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Render ablation results as markdown table."""
    lines = [
        "# Component Ablation Matrix",
        "",
        f"Generated: {report['generated_at']}",
        f"DB: `{report['inputs']['db']}`",
        f"Semantic: `{report['inputs']['semantic']}`",
        f"Reranker: `{report['inputs']['reranker']}`",
        "",
        "## Results",
        "",
        "| Config | Composite | F1 | FPR | MRR | Top1 | TP | FP | FN | p95ms |",
        "|--------|-----------|----|----|-----|------|----|----|----|----|",
    ]

    for _case_id, result in report["results"].items():
        m = result["metrics"]
        lines.append(
            f"| {result['label'][:30]} | {m['composite']:.4f} | "
            f"{m['f1'] * 100:.1f}% | {m['fpr'] * 100:.1f}% | "
            f"{m['mrr']:.4f} | {m['top1_accuracy'] * 100:.1f}% | "
            f"{m['tp']} | {m['fp']} | {m['fn']} | {m['p95_ms']:.0f} |"
        )

    # Delta table
    lines.extend(
        [
            "",
            "## Deltas vs Baseline",
            "",
            "| Config | ΔComposite | ΔF1 | ΔFPR | ΔMRR | ΔTP | ΔFP | ΔFN |",
            "|--------|-----------|-----|------|------|-----|-----|-----|",
        ]
    )

    for case_id, result in report["results"].items():
        if case_id == "baseline":
            continue
        d = result.get("delta", {})
        if not d:
            continue
        lines.append(
            f"| {result['label'][:30]} | {d.get('composite', 0):+.4f} | "
            f"{d.get('f1', 0) * 100:+.1f}pp | {d.get('fpr', 0) * 100:+.1f}pp | "
            f"{d.get('mrr', 0):+.4f} | {d.get('tp', 0):+d} | "
            f"{d.get('fp', 0):+d} | {d.get('fn', 0):+d} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- **Positive ΔComposite when component removed** → component is hurting overall",
            "- **Negative ΔComposite when component removed** → component is helping",
            "- **Large ΔFPR** → component is a major source of false positives",
            "- **Large ΔFN** → component is critical for detection",
            "",
        ]
    )

    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Component Ablation Matrix — measure net contribution of each subsystem."
    )
    parser.add_argument("--db", type=Path, required=True, help="Path to production database.")
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("benchmarks/myspellchecker_benchmark.yaml"),
        help="Benchmark YAML path.",
    )
    parser.add_argument(
        "--semantic",
        type=Path,
        default=None,
        help="Semantic model directory (contains model.onnx + tokenizer.json).",
    )
    parser.add_argument(
        "--reranker",
        type=Path,
        default=None,
        help="Reranker model directory (contains reranker.onnx).",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=sorted(CASE_MAP),
        default=None,
        help="Optional subset of cases to run. Default: all.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/results/component_ablation"),
        help="Output directory.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    if not args.db.exists():
        print(f"Error: Database not found: {args.db}", file=sys.stderr)
        return 1
    if not args.benchmark.exists():
        print(f"Error: Benchmark not found: {args.benchmark}", file=sys.stderr)
        return 1

    print("Component Ablation Matrix — v1.6.0")
    print(f"  DB: {args.db}")
    print(f"  Benchmark: {args.benchmark} ")
    print(f"  Semantic: {args.semantic}")
    print(f"  Reranker: {args.reranker}")

    report = run_ablation_matrix(args)
    markdown = render_markdown(report)

    # Save outputs
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = args.output_dir / f"component_ablation_{ts}.json"
    md_path = args.output_dir / f"component_ablation_{ts}.md"

    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(markdown, encoding="utf-8")

    print(f"\nSaved: {json_path}")
    print(f"Saved: {md_path}")
    print()
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
