"""Benchmark annotation integrity checks (Sprint F).

Sprint E had to manually fix 4 benchmark entries where single-token error
subtypes (aukmyit_confusion, compound_confusion) had multi-token gold
corrections. This test suite codifies the invariants so future annotation
bugs are caught at CI time.

Two checks:

1. ``test_single_token_subtypes_have_single_token_spans`` — every error whose
   subtype is in ``SINGLE_TOKEN_SUBTYPES`` must have a span that does NOT
   contain whitespace, AND a ``gold_correction`` that does NOT contain
   whitespace.
2. ``test_span_matches_erroneous_text`` — every expected error's
   ``input[span.start:span.end]`` must equal ``erroneous_text``. Prevents
   off-by-one span bugs that would silently mis-measure detection TP.

The subtype list is deliberately narrow: only strict single-edit subtypes
(aukmyit, consonant substitution, visarga, asat, tone, medial, kinzi,
stacking). Homophone and compound confusions are EXCLUDED because they can
legitimately span multiple tokens (per Sprint F debate gate finding R3).
"""

from __future__ import annotations

from pathlib import Path

import yaml

BENCHMARK_PATH = Path(__file__).parent.parent / "benchmarks" / "myspellchecker_benchmark.yaml"

# Strict single-edit error subtypes: the gold correction differs from the
# erroneous text by a single character-level edit (substitution, insertion,
# deletion, diacritic). Multi-token gold corrections are inappropriate here.
#
# Excluded on purpose (can be multi-token):
#   - homophone_confusion   (whole-word replacements)
#   - compound_confusion    (compounds can span syllables/tokens)
#   - loan_word_misspelling (loanwords vary in structure)
#   - non_word_typo         (arbitrary misspellings)
#   - real_word_confusion   (word-for-word)
#   - register_mismatch, particle_misuse, word_boundary, word_order,
#     verb_tense_agreement, classifier_error, collocation_error (all
#     structural/grammatical, not single-edit)
SINGLE_TOKEN_SUBTYPES: frozenset[str] = frozenset(
    {
        "aukmyit_confusion",  # visarga (့) drop
        "consonant_substitution",  # e.g., aspiration pair ထ ↔ တ
        "tone_confusion",  # tone mark swap
        "medial_confusion",  # medial ြ ↔ ျ
        "visarga_confusion",  # visarga diacritic confusion
        "asat_confusion",  # asat (်) position
        "missing_visarga",  # ့ required but absent
        "missing_asat",  # ် required but absent
        "kinzi_confusion",  # င်္ stacking
        "tone_mark_error",  # tone diacritic mis-placement
        "stacking_error",  # consonant stack
        "vowel_medial_substitution",  # single-edit vowel+medial swap
    }
)


def _load_benchmark() -> dict:
    with open(BENCHMARK_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_single_token_subtypes_have_single_token_spans() -> None:
    """Single-edit error subtypes must not span multiple whitespace-separated tokens.

    Catches the Sprint E bug class: an error annotated as (e.g.) aukmyit_confusion
    with a multi-token gold correction that the token-level homophone/word
    strategy can never match at top-1.
    """
    benchmark = _load_benchmark()
    violations: list[str] = []
    for sentence in benchmark.get("sentences", []):
        sid = sentence["id"]
        input_text = sentence.get("input", "")
        for err in sentence.get("expected_errors", []):
            subtype = err.get("error_subtype", "")
            if subtype not in SINGLE_TOKEN_SUBTYPES:
                continue
            span = err.get("span", {})
            start = span.get("start", 0)
            end = span.get("end", 0)
            span_text = input_text[start:end]
            gold = err.get("gold_correction", "")
            if " " in span_text:
                violations.append(
                    f"{sid}/{err.get('error_id', '?')}: subtype={subtype!r} "
                    f"span contains whitespace: {span_text!r}"
                )
            if " " in gold:
                violations.append(
                    f"{sid}/{err.get('error_id', '?')}: subtype={subtype!r} "
                    f"gold_correction contains whitespace: {gold!r}"
                )
    assert not violations, "Benchmark annotation violations:\n  " + "\n  ".join(violations)


def test_span_matches_erroneous_text() -> None:
    """Every expected_error's ``input[span.start:span.end]`` must equal ``erroneous_text``.

    Catches off-by-one span bugs and ensures that detection overlap matching
    in the benchmark runner compares what the annotator actually meant.
    """
    benchmark = _load_benchmark()
    violations: list[str] = []
    for sentence in benchmark.get("sentences", []):
        sid = sentence["id"]
        input_text = sentence.get("input", "")
        for err in sentence.get("expected_errors", []):
            span = err.get("span", {})
            start = span.get("start", 0)
            end = span.get("end", 0)
            span_text = input_text[start:end]
            erroneous = err.get("erroneous_text", "")
            if span_text != erroneous:
                violations.append(
                    f"{sid}/{err.get('error_id', '?')}: "
                    f"span text {span_text!r} != erroneous_text {erroneous!r}"
                )
    assert not violations, "Span/erroneous_text mismatch:\n  " + "\n  ".join(violations)
