from __future__ import annotations

from pathlib import Path

import yaml

from benchmarks.audit_targeted_rules import build_report
from benchmarks.compare_runs import build_comparison
from benchmarks.run_ablation import (
    ABLATION_CASE_MAP,
    build_dataset_section,
    resolve_case_order,
)
from benchmarks.run_ablation import (
    render_markdown as render_ablation_markdown,
)


def _write_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def test_build_comparison_splits_miss_buckets() -> None:
    baseline = {
        "_file": "baseline.json",
        "overall_metrics": {
            "detection": {"precision": 0.9, "recall": 0.8, "f1": 0.85},
            "suggestions": {
                "top1_accuracy": 0.7,
                "top3_accuracy": 0.8,
                "top5_accuracy": 0.85,
                "mrr": 0.75,
            },
            "false_positive_rate": {"rate": 0.01},
        },
        "per_sentence_results": [
            {
                "id": "B-1",
                "matches": [
                    {
                        "gold_id": "B-1-E1",
                        "detected": True,
                        "system_type": "invalid_syllable",
                        "top1_correct": False,
                        "rank": 2,
                    },
                    {
                        "gold_id": "B-1-E2",
                        "detected": True,
                        "system_type": "syntax_error",
                        "top1_correct": False,
                        "rank": None,
                    },
                    {"gold_id": "B-1-E3", "detected": False},
                ],
            }
        ],
    }
    current = {
        "_file": "current.json",
        "overall_metrics": {
            "detection": {"precision": 0.92, "recall": 0.84, "f1": 0.88},
            "suggestions": {
                "top1_accuracy": 0.78,
                "top3_accuracy": 0.86,
                "top5_accuracy": 0.9,
                "mrr": 0.8,
            },
            "false_positive_rate": {"rate": 0.01},
        },
        "per_sentence_results": [
            {
                "id": "C-1",
                "matches": [
                    {
                        "gold_id": "C-1-E1",
                        "detected": True,
                        "system_type": "invalid_syllable",
                        "top1_correct": True,
                        "rank": 1,
                    },
                    {
                        "gold_id": "C-1-E2",
                        "detected": True,
                        "system_type": "syntax_error",
                        "top1_correct": False,
                        "rank": None,
                    },
                ],
            }
        ],
    }

    result = build_comparison(baseline, current)
    assert result["baseline"]["miss_buckets"]["counts"]["rerank"] == 1
    assert result["baseline"]["miss_buckets"]["counts"]["missing_candidate"] == 1
    assert result["baseline"]["miss_buckets"]["counts"]["missed_detection"] == 1
    assert result["current"]["miss_buckets"]["counts"]["rerank"] == 0
    assert result["delta"]["miss_buckets"]["rerank"] == -1


def test_audit_targeted_rules_classifies_fire_win_buckets(tmp_path: Path) -> None:
    report1 = tmp_path / "r1.json"
    report2 = tmp_path / "r2.json"

    report1.write_text(
        """
{
  "rerank_rule_telemetry": {
    "literal_hint:confusable_error:ဂစား": {"fires": 10, "top1_changes": 1, "top1_change_rate": 0.1},
    "literal_hint:question_structure:ရဲ့လဲ": {"fires": 2, "top1_changes": 2, "top1_change_rate": 1.0},
    "distance_rerank:invalid_syllable": {"fires": 8, "top1_changes": 6, "top1_change_rate": 0.75}
  }
}
""".strip(),
        encoding="utf-8",
    )
    report2.write_text(
        """
{
  "rerank_rule_telemetry": {
    "literal_hint:confusable_error:ဂစား": {"fires": 6, "top1_changes": 0, "top1_change_rate": 0.0},
    "literal_hint:question_structure:ရဲ့လဲ": {"fires": 1, "top1_changes": 1, "top1_change_rate": 1.0},
    "distance_rerank:invalid_syllable": {"fires": 4, "top1_changes": 2, "top1_change_rate": 0.5}
  }
}
""".strip(),
        encoding="utf-8",
    )

    audit = build_report(
        [report1, report2],
        high_fire_min=10,
        low_win_max=0.2,
        low_fire_max=3,
        high_win_min=0.9,
    )

    assert audit["summary"]["reports_scanned"] == 2
    assert audit["summary"]["reports_with_telemetry_key"] == 2
    assert audit["summary"]["reports_with_nonempty_telemetry"] == 2

    hf = [row["rule_id"] for row in audit["high_fire_low_win"]]
    lf = [row["rule_id"] for row in audit["low_fire_high_win"]]
    assert "literal_hint:confusable_error:ဂစား" in hf
    assert "literal_hint:question_structure:ရဲ့လဲ" in lf


def test_run_ablation_resolve_case_order_includes_default() -> None:
    resolved = resolve_case_order(["no_hint", "no_inject"])
    assert resolved[0].case_id == "default"
    assert [case.case_id for case in resolved] == ["default", "no_hint", "no_inject"]


def test_run_ablation_build_dataset_section_computes_deltas(tmp_path: Path) -> None:
    baseline_report = {
        "overall_metrics": {
            "detection": {"precision": 0.9, "recall": 0.8, "f1": 0.85},
            "suggestions": {
                "top1_accuracy": 0.7,
                "top3_accuracy": 0.8,
                "top5_accuracy": 0.85,
                "mrr": 0.75,
            },
            "false_positive_rate": {"rate": 0.01},
        },
        "config": {
            "targeted_rerank_hints": True,
            "targeted_candidate_injections": True,
            "targeted_grammar_completion_templates": True,
        },
        "per_sentence_results": [
            {
                "id": "B-1",
                "matches": [
                    {
                        "gold_id": "B-1-E1",
                        "detected": True,
                        "system_type": "invalid_syllable",
                        "top1_correct": False,
                        "rank": 2,
                    }
                ],
            }
        ],
    }
    no_hint_report = {
        "overall_metrics": {
            "detection": {"precision": 0.9, "recall": 0.8, "f1": 0.85},
            "suggestions": {
                "top1_accuracy": 0.62,
                "top3_accuracy": 0.78,
                "top5_accuracy": 0.84,
                "mrr": 0.7,
            },
            "false_positive_rate": {"rate": 0.01},
        },
        "config": {
            "targeted_rerank_hints": False,
            "targeted_candidate_injections": True,
            "targeted_grammar_completion_templates": True,
        },
        "per_sentence_results": [
            {
                "id": "C-1",
                "matches": [
                    {
                        "gold_id": "C-1-E1",
                        "detected": True,
                        "system_type": "invalid_syllable",
                        "top1_correct": False,
                        "rank": None,
                    }
                ],
            }
        ],
    }

    case_order = [ABLATION_CASE_MAP["default"], ABLATION_CASE_MAP["no_hint"]]
    case_results = {
        "default": (tmp_path / "default.json", baseline_report),
        "no_hint": (tmp_path / "no_hint.json", no_hint_report),
    }
    section = build_dataset_section(
        dataset_name="benchmark",
        dataset_path=tmp_path / "benchmark.yaml",
        case_order=case_order,
        case_results=case_results,
    )

    assert section["default_case"] == "default"
    assert section["runs"]["default"]["metrics"]["top1_accuracy"] == 0.7
    assert section["runs"]["no_hint"]["toggles"]["targeted_rerank_hints"] is False
    assert section["deltas_vs_default"]["no_hint"]["metrics"]["top1_accuracy"] == -0.08
    assert section["deltas_vs_default"]["no_hint"]["miss_buckets"]["rerank"] == -1
    assert section["deltas_vs_default"]["no_hint"]["miss_buckets"]["missing_candidate"] == 1


def test_run_ablation_render_markdown_contains_dataset_tables() -> None:
    report = {
        "generated_at": "2026-03-08T00:00:00+00:00",
        "inputs": {"db": "data/db.db", "level": "word", "warmup": 0},
        "datasets": {
            "benchmark": {
                "path": "benchmarks/myspellchecker_benchmark.yaml",
                "default_case": "default",
                "runs": {
                    "default": {
                        "metrics": {
                            "top1_accuracy": 0.9,
                            "f1": 0.88,
                            "clean_fpr": 0.01,
                        }
                    },
                    "no_hint": {
                        "metrics": {
                            "top1_accuracy": 0.8,
                            "f1": 0.88,
                            "clean_fpr": 0.01,
                        }
                    },
                },
                "deltas_vs_default": {
                    "no_hint": {
                        "metrics": {"top1_accuracy": -0.1},
                        "miss_buckets": {"rerank": 5, "missing_candidate": 1, "total": 6},
                    }
                },
            }
        },
    }

    md = render_ablation_markdown(report)
    assert "Ablation Matrix Report" in md
    assert "Dataset: benchmark" in md
    assert "| default |" in md
    assert "| no_hint |" in md
