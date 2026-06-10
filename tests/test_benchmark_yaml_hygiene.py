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


def test_span_text_matches_erroneous_text(benchmark_data: dict) -> None:
    """G4 (v1.9 granularity rules): the annotated span must slice exactly to
    `erroneous_text`, otherwise span-based TP matching and `gold_correction`
    splicing silently diverge."""
    offenders: list[str] = []
    for sentence in benchmark_data.get("sentences", []):
        text = sentence.get("input") or ""
        for error in sentence.get("expected_errors", []):
            span = error.get("span") or {}
            start, end = span.get("start"), span.get("end")
            expected = error.get("erroneous_text")
            if start is None or end is None or expected is None:
                continue
            if text[start:end] != expected:
                offenders.append(
                    f"{error.get('error_id', '<no-id>')} (sentence {sentence.get('id')})"
                )
    assert not offenders, (
        "Benchmark YAML has spans that do not slice to their `erroneous_text`: "
        + ", ".join(offenders)
    )


CANONICAL_SUBTYPES = {
    "consonant_substitution",
    "vowel_medial_substitution",
    "broken_compound",
    "compound_confusion",
    "tone_mark_error",
    "aukmyit_confusion",
    "loan_word_misspelling",
    "real_word_confusion",
    "non_word_typo",
    "homophone_confusion",
    "zawgyi_conversion_error",
    "zawgyi_encoding",
    "word_boundary",
    "missing_visarga",
    "missing_asat",
    "register_mismatch",
    "particle_misuse",
    "syllable_error",
    "hidden_compound_typo",
    "synonym_substitution",
    "stacking_error",
    "verb_tense_agreement",
    "spacing",
    "word_order",
    "collocation_error",
    "ngram_unlikely",
    "colloquial_in_formal",
    "classifier_error",
    "semantic_error",
    "negation_error",
    "aspect_error",
    "merged_word",
    "question_structure",
    "zero_width_chars",
    "missing_information",
    "missing_word",
    "incomplete_sentence",
    "invalid_syllable",
}


def test_error_subtype_in_canonical_vocabulary(benchmark_data: dict) -> None:
    """G6 (v1.9 granularity rules): `error_subtype` is a closed vocabulary.
    The 62-label long tail was consolidated in bp-03 (2026-06-10); new labels
    require a deliberate vocabulary addition here, not ad-hoc invention."""
    offenders: list[str] = []
    for sentence in benchmark_data.get("sentences", []):
        for error in sentence.get("expected_errors", []):
            subtype = error.get("error_subtype")
            if subtype is not None and subtype not in CANONICAL_SUBTYPES:
                offenders.append(f"{error.get('error_id', '<no-id>')}: {subtype}")
    assert not offenders, (
        "Benchmark YAML uses non-canonical `error_subtype` labels (G6 closed "
        "vocabulary, bp-03 2026-06-10): " + ", ".join(sorted(set(offenders)))
    )
