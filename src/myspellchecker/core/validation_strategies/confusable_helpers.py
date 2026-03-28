"""
Helper utilities for ConfusableSemanticStrategy.

Contains pair evaluation, threshold calculation, medial/tone checks,
and boundary detection logic extracted from the main strategy class.
"""

from __future__ import annotations

from myspellchecker.core.constants import (
    CONFUSABLE_EXEMPT_SUFFIX_PAIRS,
    PARTICLE_CONFUSABLES,
)

# ── Myanmar medial characters ──────────────────────────────────────────

MEDIAL_YA_PIN = "\u103b"  # ျ
MEDIAL_YA_YIT = "\u103c"  # ြ

# Boundary chars used for token-boundary detection.
_BOUNDARY_CHARS = " \t\n\u200b\u104a\u104b"  # space, ZWSP, Myanmar punctuation


# ── Pair evaluation helpers (all static / pure) ────────────────────────


def is_medial_confusable(word: str, variant: str) -> bool:
    """Check if the difference is a medial swap, insertion, or deletion.

    Covers:
    - ျ↔ြ swap (e.g., ပျော်↔ပြော်)
    - Medial insertion/deletion (e.g., ပီး↔ပြီး, ရေ့↔ရှေ့, မာ↔မှာ)
    """
    # 1. Medial swap check (ျ↔ြ)
    word_has_yit = MEDIAL_YA_YIT in word
    word_has_pin = MEDIAL_YA_PIN in word
    var_has_yit = MEDIAL_YA_YIT in variant
    var_has_pin = MEDIAL_YA_PIN in variant

    if word_has_yit and not word_has_pin and var_has_pin and not var_has_yit:
        return word.replace(MEDIAL_YA_YIT, MEDIAL_YA_PIN) == variant
    if word_has_pin and not word_has_yit and var_has_yit and not var_has_pin:
        return word.replace(MEDIAL_YA_PIN, MEDIAL_YA_YIT) == variant

    # 2. Medial insertion/deletion check (ျ, ြ, ွ, ှ)
    _medials = (MEDIAL_YA_PIN, MEDIAL_YA_YIT, "\u103d", "\u103e")
    for m in _medials:
        # Variant has medial that word doesn't -> insertion
        if m in variant and m not in word:
            if variant.replace(m, "", 1) == word:
                return True
        # Word has medial that variant doesn't -> deletion
        if m in word and m not in variant:
            if word.replace(m, "", 1) == variant:
                return True

    return False


def is_medial_deletion(word: str, variant: str) -> bool:
    """Check if the confusable is a medial deletion (not swap).

    Returns True when one word has a medial character that the other
    lacks and removing it makes them identical.  Returns False for
    medial swaps (ျ↔ြ).
    """
    _medials = (MEDIAL_YA_PIN, MEDIAL_YA_YIT, "\u103d", "\u103e")
    for m in _medials:
        if m in variant and m not in word:
            if variant.replace(m, "", 1) == word:
                return True
        if m in word and m not in variant:
            if word.replace(m, "", 1) == variant:
                return True
    return False


def is_visarga_only_pair(word: str, variant: str) -> bool:
    """Check if difference is only visarga (း U+1038) addition/removal."""
    visarga = "\u1038"
    return word + visarga == variant or variant + visarga == word


def is_tone_marker_only_pair(word: str, variant: str) -> bool:
    """Check if difference is only a tone marker addition/removal.

    Covers visarga (း U+1038) and aukmyit (့ U+1037).
    In Myanmar, these tone markers change morpheme identity
    (e.g., ပြီ completion vs ပြီး connective, လို want vs လို့ because).
    """
    visarga = "\u1038"
    aukmyit = "\u1037"
    return (
        word + visarga == variant
        or variant + visarga == word
        or word + aukmyit == variant
        or variant + aukmyit == word
    )


def is_particle_confusable(word: str, variant: str) -> bool:
    """Check if word-variant pair is a particle confusable."""
    variants = PARTICLE_CONFUSABLES.get(word, [])
    return variant in variants


def is_exempt_suffix_pair(word: str, variant: str) -> bool:
    """Check if word-variant pair differs only by an exempt suffix.

    For example, words ending in သည် vs သည့် differ only in syntactic
    function (declarative vs attributive) -- the MLM cannot reliably
    distinguish them, so we skip the variant.
    """
    for suffix_a, suffix_b in CONFUSABLE_EXEMPT_SUFFIX_PAIRS:
        if word.endswith(suffix_a) and variant.endswith(suffix_b):
            # Check that the stems match (everything before the suffix)
            if word[: -len(suffix_a)] == variant[: -len(suffix_b)]:
                return True
    return False


def is_db_suppressed(word: str, variant: str, provider: object) -> bool:
    """Check if word-variant pair is suppressed in the DB confusable_pairs table."""
    if hasattr(provider, "is_confusable_suppressed"):
        result = provider.is_confusable_suppressed(word, variant)
        if result is True:
            return True
        result = provider.is_confusable_suppressed(variant, word)
        if result is True:
            return True
    return False


# ── Token boundary helpers ─────────────────────────────────────────────


def find_all_positions(sentence: str, target_word: str) -> list[int]:
    """Find all non-overlapping occurrences of *target_word* in *sentence*."""
    if not sentence or not target_word:
        return []

    positions: list[int] = []
    start = 0
    step = len(target_word)
    while True:
        pos = sentence.find(target_word, start)
        if pos == -1:
            break
        positions.append(pos)
        start = pos + step
    return positions


def is_token_boundary(sentence: str, start: int, end: int) -> bool:
    """Check whether [start:end] is bounded by non-word-like characters.

    Uses space/punctuation boundary detection instead of ``isalnum()``,
    which misclassifies Myanmar dependent characters and causes the
    non-boundary penalty to fire on virtually all Myanmar words.
    """
    if start < 0 or end > len(sentence) or start >= end:
        return False
    left_ok = start == 0 or sentence[start - 1] in _BOUNDARY_CHARS
    right_ok = end == len(sentence) or sentence[end] in _BOUNDARY_CHARS
    return left_ok and right_ok


def is_occurrence_token_boundary(
    sentence: str,
    word: str,
    occurrence: int,
    semantic_checker: object,
) -> bool:
    """Check boundary status of the exact occurrence used for masking."""
    if not sentence or not word:
        return False

    positions: list[int] = []
    find_positions_fn = getattr(semantic_checker, "_find_all_word_positions", None)
    if callable(find_positions_fn):
        try:
            raw = find_positions_fn(sentence, word)
            if isinstance(raw, list):
                positions = [int(pos) for pos in raw]
        except (RuntimeError, ValueError, TypeError, AttributeError):
            positions = []

    if not positions:
        positions = find_all_positions(sentence, word)
    if occurrence < 0 or occurrence >= len(positions):
        return False

    start = positions[occurrence]
    end = start + len(word)
    boundary_fn = getattr(semantic_checker, "_is_token_boundary", None)
    if callable(boundary_fn):
        try:
            return bool(boundary_fn(sentence, start, end))
        except (RuntimeError, ValueError, TypeError, AttributeError):
            return is_token_boundary(sentence, start, end)
    return is_token_boundary(sentence, start, end)


# ── Threshold logic ────────────────────────────────────────────────────


def cap_threshold(threshold: float, max_threshold: float) -> float:
    """Apply max_threshold cap if configured (non-zero)."""
    if max_threshold > 0:
        return min(threshold, max_threshold)
    return threshold
