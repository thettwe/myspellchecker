"""Benchmark YAML hygiene invariants.

Enforces the schema rules that bench-hygiene-v17 introduced so that regressions
(empty gold, missing fields, inconsistent skip-flags) fail in CI rather than
silently corrupting FN/TP accounting.

Invariants (as of bhv17-root-cause-01):
1. Every `expected_errors` entry with empty `gold_correction` MUST carry
   `detection_only: true`. Empty-gold rows are annotations that cannot be
   scored against a correction target.
2. Every `expected_errors` entry MUST have the core identity fields
   (`error_id`, `error_type`, `span`, `erroneous_text`).
3. `detection_layer` and `detection_only` are the two skip-flags benchmark
   runners must respect. A runner that ignores either will double-count the
   flagged rows as false-negatives.

Runs as part of the standard pytest suite; pre-commit gates execute pytest
before accepting a commit.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH_YAML = REPO_ROOT / "benchmarks" / "myspellchecker_benchmark.yaml"


@pytest.fixture(scope="module")
def benchmark_data() -> dict:
    with BENCH_YAML.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _is_empty_gold(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def test_empty_gold_rows_have_detection_only(benchmark_data: dict) -> None:
    offenders: list[str] = []
    for sentence in benchmark_data.get("sentences", []):
        for error in sentence.get("expected_errors", []):
            if _is_empty_gold(error.get("gold_correction")) and not error.get("detection_only"):
                offenders.append(
                    f"{error.get('error_id', '<no-id>')} (sentence {sentence.get('id')})"
                )
    assert not offenders, (
        "Benchmark YAML contains empty-gold rows without `detection_only: true`. "
        "Such rows cannot be scored against a correction target and silently inflate "
        "FN counts. Remediate per bhv17-m6-empty-gold-01 pattern. Offenders: "
        + ", ".join(offenders)
    )


def test_expected_errors_have_required_fields(benchmark_data: dict) -> None:
    required = {"error_id", "error_type", "span", "erroneous_text"}
    offenders: list[str] = []
    for sentence in benchmark_data.get("sentences", []):
        for error in sentence.get("expected_errors", []):
            missing = required - set(error.keys())
            if missing:
                offenders.append(
                    f"{error.get('error_id', '<no-id>')} "
                    f"(sentence {sentence.get('id')}): missing {sorted(missing)}"
                )
    assert not offenders, (
        "Benchmark YAML has error rows missing required identity fields. " + "; ".join(offenders)
    )


def test_benchmark_version_is_populated(benchmark_data: dict) -> None:
    version = benchmark_data.get("version")
    assert isinstance(version, str) and version.strip(), (
        "Benchmark YAML must carry a non-empty `version:` field. Every change to "
        "the YAML bumps the version (feedback_single_benchmark_file)."
    )
