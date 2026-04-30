"""
Correction Utilities.

This module contains utility functions for text correction,
extracted from SpellChecker for better testability and reusability.
"""

from __future__ import annotations

import bisect

from myspellchecker.core.constants import (
    ET_ASPECT_ADVERB_CONFLICT,
    ET_BROKEN_COMPOUND,
    ET_BROKEN_STACKING,
    ET_BROKEN_VIRAMA,
    ET_CLASSIFIER_ERROR,
    ET_COLLOCATION_ERROR,
    ET_COLLOQUIAL_CONTRACTION,
    ET_CONFUSABLE_ERROR,
    ET_DANGLING_PARTICLE,
    ET_DANGLING_WORD,
    ET_DUPLICATE_PUNCTUATION,
    ET_HA_HTOE_CONFUSION,
    ET_INCOMPLETE_STACKING,
    ET_LEADING_VOWEL_E,
    ET_MEDIAL_CONFUSION,
    ET_MEDIAL_ORDER_ERROR,
    ET_MERGED_SFP_CONJUNCTION,
    ET_MISSING_ASAT,
    ET_MISSING_CONJUNCTION,
    ET_MISSING_PUNCTUATION,
    ET_NEGATION_SFP_MISMATCH,
    ET_PARTICLE_CONFUSION,
    ET_PARTICLE_MISUSE,
    ET_REGISTER_MIXING,
    ET_SEMANTIC_ERROR,
    ET_SYLLABLE,
    ET_TENSE_MISMATCH,
    ET_VOWEL_AFTER_ASAT,
    ET_WRONG_PUNCTUATION,
    ActionType,
)
from myspellchecker.core.response import Error, SyllableError

__all__ = [
    "build_orig_to_norm_map",
    "filter_syllable_errors_in_valid_words",
    "generate_corrected_text",
    "remap_pre_norm_error",
]

# Characters removed by normalize_c.remove_zero_width_chars (ZWSP, ZWNJ, ZWJ, BOM)
_NORM_ZERO_WIDTH = frozenset("\u200b\u200c\u200d\ufeff")


def build_orig_to_norm_map(original: str) -> list[int]:
    """Build position map from original-text offsets to normalized-text offsets.

    When normalization removes zero-width characters (ZWSP, ZWNJ, ZWJ, BOM),
    pre-normalization error positions (which reference the original text) become
    invalid for ``generate_corrected_text`` which operates on normalized text.

    This function produces a list where ``mapping[orig_pos]`` gives the
    corresponding position in the normalized text. A sentinel entry at
    ``mapping[len(original)]`` maps past-the-end.

    Args:
        original: The raw input text before normalization.

    Returns:
        A list of length ``len(original) + 1`` mapping original positions to
        normalized positions.
    """
    mapping: list[int] = []
    norm_pos = 0
    for ch in original:
        mapping.append(norm_pos)
        if ch not in _NORM_ZERO_WIDTH:
            norm_pos += 1
    mapping.append(norm_pos)  # sentinel for past-the-end
    return mapping


def remap_pre_norm_error(error: Error, offset_map: list[int]) -> None:
    """Remap a pre-normalization error's position to normalized-text coordinates.

    Mutates ``error.position`` in place so that it aligns with the normalized
    text used by ``generate_corrected_text``.

    Args:
        error: An error whose ``position`` references the original (raw) text.
        offset_map: The mapping produced by :func:`build_orig_to_norm_map`.
    """
    old_pos = error.position
    if 0 <= old_pos < len(offset_map):
        error.position = offset_map[old_pos]


# High-value suffix/root-cause syllable fixes that should survive filtering
# even when the segmenter splits them into valid dictionary words.
_PRESERVE_SYLLABLE_SUGGESTION_PAIRS: frozenset[tuple[str, str]] = frozenset(
    {
        ("တယ", "တယ်"),
        ("မယ", "မယ်"),
        ("လို", "လို့"),
    }
)

# Error types from direct text-search detectors that should be preserved
# even when the flagged text happens to be a valid dictionary word.
# Shared with error_suppression.py to guard against two-stage removal
# (POS suppressed by syllable error, then syllable removed by valid-word filter).
_PRESERVE_ERROR_TYPES: frozenset[str] = frozenset(
    {
        ET_COLLOQUIAL_CONTRACTION,
        ET_PARTICLE_CONFUSION,
        ET_DANGLING_PARTICLE,
        ET_REGISTER_MIXING,
        ET_DANGLING_WORD,
        ET_MISSING_CONJUNCTION,
        ET_TENSE_MISMATCH,
        ET_MEDIAL_CONFUSION,
        ET_HA_HTOE_CONFUSION,
        ET_VOWEL_AFTER_ASAT,
        ET_BROKEN_VIRAMA,
        ET_BROKEN_STACKING,
        ET_CONFUSABLE_ERROR,
        ET_CLASSIFIER_ERROR,
        ET_BROKEN_COMPOUND,
        ET_MEDIAL_ORDER_ERROR,
        ET_SEMANTIC_ERROR,
        ET_LEADING_VOWEL_E,
        ET_INCOMPLETE_STACKING,
        ET_NEGATION_SFP_MISMATCH,
        ET_MERGED_SFP_CONJUNCTION,
        ET_ASPECT_ADVERB_CONFLICT,
        ET_MISSING_ASAT,
        ET_PARTICLE_MISUSE,
        ET_COLLOCATION_ERROR,
        ET_MISSING_PUNCTUATION,
        ET_DUPLICATE_PUNCTUATION,
        ET_WRONG_PUNCTUATION,
    }
)


def generate_corrected_text(text: str, errors: list[Error]) -> str:
    """
    Generate corrected text by applying error suggestions.

    This function processes errors in position order and applies the top
    suggestion for each error. Overlapping errors are handled as follows:

    **Overlapping Error Resolution:**
    When two errors overlap (e.g., Error A at char 0-5 and Error B at char 2-7),
    only the first error (by position) is corrected. Subsequent overlapping
    errors are skipped to prevent conflicting corrections.

    This "first-wins" strategy ensures:
    1. Consistent, reproducible results
    2. No conflicting text modifications
    3. Safe handling of complex multi-word error patterns

    Note:
        For complex overlapping errors, users should review the `errors` list
        in the Response object to see all detected issues, including those
        that were not auto-corrected due to overlap.

    Args:
        text: Original input text.
        errors: List of Error objects with suggestions.

    Returns:
        Corrected text with top suggestions applied for non-overlapping errors.

    Example:
        >>> from myspellchecker.core.response import SyllableError
        >>> errors = [SyllableError(position=0, text="abc", suggestions=["xyz"])]
        >>> generate_corrected_text("abc def", errors)
        'xyz def'
    """
    if not errors:
        return text

    # Sort errors by position ascending for forward processing
    sorted_errors = sorted(errors, key=lambda e: e.position)

    # Build result in forward order (more efficient - no reversal needed)
    segments: list[str] = []
    current_pos = 0

    for error in sorted_errors:
        if not error.suggestions:
            continue

        # Skip advisory-only errors — INFORM means "notify, don't auto-correct"
        if error.action == ActionType.INFORM:
            continue

        top_suggestion = error.suggestions[0]
        start_pos = error.position
        end_pos = start_pos + len(error.text)

        # Skip overlapping errors (already handled by a previous correction)
        if start_pos < current_pos:
            continue

        # Append text before the error
        if current_pos < start_pos:
            segments.append(text[current_pos:start_pos])

        # Append correction
        segments.append(top_suggestion)

        current_pos = end_pos

    # Append remaining text at the end
    if current_pos < len(text):
        segments.append(text[current_pos:])

    return "".join(segments)


def filter_syllable_errors_in_valid_words(
    text: str,
    errors: list[Error],
    words: list[str],
    validity_map: dict[str, bool],
) -> list[Error]:
    """
    Filter out syllable errors that occur within valid words.

    Performance: O(n log m) where n = errors, m = valid words.
    Uses binary search instead of O(n*m) linear scan.

    Args:
        text: Original text being validated.
        errors: List of errors to filter.
        words: Pre-segmented words from text.
        validity_map: Map of word -> is_valid (from bulk validity check).

    Returns:
        Filtered list of errors (syllable errors inside valid words removed).

    Example:
        >>> text = "မြန်မာနိုင်ငံ"
        >>> errors = [SyllableError(position=0, text="မြန်", ...)]
        >>> words = ["မြန်မာ", "နိုင်ငံ"]
        >>> validity_map = {"မြန်မာ": True, "နိုင်ငံ": True}
        >>> # Syllable error at position 0 is inside valid word "မြန်မာ"
        >>> filtered = filter_syllable_errors_in_valid_words(text, errors, words, validity_map)
        >>> len(filtered)  # Error removed
        0
    """
    if not errors:
        return []

    # Build valid word ranges using batch results
    valid_word_ranges: list[tuple[int, int]] = []
    cursor = 0
    for w in words:
        w_pos = text.find(w, cursor)
        if w_pos != -1:
            if validity_map.get(w, False):
                valid_word_ranges.append((w_pos, w_pos + len(w)))
            cursor = w_pos + len(w)

    if not valid_word_ranges:
        # No valid words, nothing to filter
        return list(errors)

    # Sort ranges by start position for binary search
    # (usually already sorted, but ensure it for correctness)
    valid_word_ranges.sort(key=lambda r: r[0])

    # Build compound ranges from adjacent word pairs that form valid compounds.
    # This handles cases like ဂေဟ+စနစ် where each part is valid individually
    # AND their concatenation is also a valid word in the dictionary.
    compound_ranges: list[tuple[int, int]] = []
    word_positions: list[tuple[int, int]] = []  # (start, end) for each word
    cursor2 = 0
    for w in words:
        w_pos = text.find(w, cursor2)
        if w_pos != -1:
            word_positions.append((w_pos, w_pos + len(w)))
            cursor2 = w_pos + len(w)
    for i in range(len(word_positions) - 1):
        _, end_i = word_positions[i]
        start_j, end_j = word_positions[i + 1]
        if end_i == start_j:
            compound = text[word_positions[i][0] : end_j]
            if validity_map.get(compound, False):
                compound_ranges.append((word_positions[i][0], end_j))

    all_ranges = valid_word_ranges + compound_ranges
    all_ranges.sort(key=lambda r: r[0])
    all_starts = [r[0] for r in all_ranges]

    def is_inside_valid_word(err_start: int, err_end: int) -> bool:
        """Check if error range is inside any valid word or compound using binary search.

        Uses bisect to find candidate range in O(log m) time.
        """
        idx = bisect.bisect_right(all_starts, err_start)
        if idx > 0:
            prev_start, prev_end = all_ranges[idx - 1]
            if err_start >= prev_start and err_end <= prev_end:
                return True
        return False

    # Filter syllable errors inside valid words - O(n log m)
    filtered_errors: list[Error] = []
    for err in errors:
        if isinstance(err, SyllableError) and err.text:
            top_suggestion = err.suggestions[0] if err.suggestions else ""
            preserve_suffix_root_cause = (
                err.error_type == ET_SYLLABLE
                and (err.text, top_suggestion) in _PRESERVE_SYLLABLE_SUGGESTION_PAIRS
            )
            # Always keep errors from text-search detectors
            if err.error_type in _PRESERVE_ERROR_TYPES or preserve_suffix_root_cause:
                filtered_errors.append(err)
            else:
                err_end = err.position + len(err.text)
                if not is_inside_valid_word(err.position, err_end):
                    filtered_errors.append(err)
        elif isinstance(err, SyllableError):
            # SyllableError with None text - include as-is (can't determine range)
            filtered_errors.append(err)
        else:
            filtered_errors.append(err)

    return filtered_errors


def has_confident_symspell_candidate(
    symspell: object | None,
    word: str,
    max_ed: float,
    min_freq: int,
) -> bool:
    """Return True if SymSpell has a top-1 candidate clearing the confidence gate.

    Used by both the pre-validation skip rule (WordValidator) and the
    post-validation compound-split suppression (ErrorSuppressionMixin).
    """
    if symspell is None:
        return False
    try:
        candidates = symspell.lookup(word, level="word", max_suggestions=1)  # type: ignore[union-attr]
    except (RuntimeError, ValueError, KeyError):
        return False
    if not candidates:
        return False
    top = candidates[0]
    term = getattr(top, "term", None)
    if term is None or term == word:
        return False
    if float(getattr(top, "edit_distance", 99)) > max_ed:
        return False
    if int(getattr(top, "frequency", 0) or 0) < min_freq:
        return False
    return True
