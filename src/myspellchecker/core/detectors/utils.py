"""Shared utilities for text-level detector mixins."""

from __future__ import annotations

from collections.abc import Iterator

from myspellchecker.core.constants import ET_SYLLABLE, ET_WORD
from myspellchecker.core.response import Error

# Error types that are safe to replace with a more specific text-level
# detector error.  These are generic L1/L2 errors that would be removed
# by ``filter_syllable_errors_in_valid_words`` anyway (they are NOT in
# ``_PRESERVE_ERROR_TYPES``), so replacing them with a preserved type
# is strictly better.
_REPLACEABLE_ERROR_TYPES: frozenset[str] = frozenset(
    {
        ET_SYLLABLE,  # "invalid_syllable"
        ET_WORD,  # "invalid_word"
    }
)


def get_existing_positions(
    errors: list[Error],
    ignore_types: frozenset[str] | None = None,
) -> set[int]:
    """Get set of positions already covered by existing errors.

    Args:
        errors: Current error list.
        ignore_types: Error types to exclude from the blocking set.
            When provided, errors with these types are treated as if
            they don't exist, allowing downstream detectors to re-examine
            those positions. Used to prevent L1 syllable errors from
            masking L3 grammar pattern detection.
    """
    if ignore_types:
        return {e.position for e in errors if e.error_type not in ignore_types}
    return {e.position for e in errors}


def try_replace_syllable_error(
    errors: list[Error],
    position: int,
    new_error: Error,
) -> bool:
    """Replace a replaceable error at *position* with *new_error*.

    If there is an ``invalid_syllable`` or ``invalid_word`` error at the
    given position, replace it in-place with *new_error* and return True.

    If there is a non-replaceable (preserved) error at that position,
    return False (caller should skip).

    This prevents the D5 bug where a text-level detector skips a position
    because ``invalid_syllable`` is already there, then
    ``filter_syllable_errors_in_valid_words`` removes the
    ``invalid_syllable`` (it's not in ``_PRESERVE_ERROR_TYPES``),
    leaving no error at all.
    """
    for i, e in enumerate(errors):
        if e.position != position:
            continue
        if e.error_type in _REPLACEABLE_ERROR_TYPES:
            errors[i] = new_error
            return True
        # Position occupied by a preserved error type -- don't replace
        return False
    # No error at this position (shouldn't happen if caller checked
    # existing_positions, but handle gracefully)
    return False


def get_tokenized(host: object, text: str):
    """Get pre-computed TokenizedText or build on-the-fly.

    During normal ``check()`` flow, ``_current_tokenized`` is set by
    ``_run_validation_layers``.  When a detector is called directly
    (e.g. in unit tests via ``SpellChecker.__new__()``), we fall back
    to building the tokenization from the text argument.
    """
    from myspellchecker.core.detectors.tokenized_text import TokenizedText

    # Prefer thread-local (safe under check_batch_async concurrency)
    tl = getattr(host, "_thread_local", None)
    if tl is not None:
        tokenized = getattr(tl, "current_tokenized", None)
        if tokenized is not None and tokenized.raw == text:
            return tokenized
    # Fallback for direct detector calls in unit tests
    tokenized = getattr(host, "_current_tokenized", None)
    if tokenized is not None and tokenized.raw == text:
        return tokenized
    return TokenizedText.from_text(text)


def iter_occurrences(text: str, pattern: str) -> Iterator[tuple[int, int]]:
    """Yield (start, end) positions of all non-overlapping occurrences of *pattern* in *text*."""
    start = 0
    while True:
        idx = text.find(pattern, start)
        if idx == -1:
            break
        end = idx + len(pattern)
        yield idx, end
        start = end
