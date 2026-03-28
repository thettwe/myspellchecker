"""Collocation error detection mixin for PostNormalizationDetectorsMixin.

Provides ``_detect_collocation_errors`` which detects wrong word partners
in fixed phrases using context-driven rules loaded from
``rules/collocations.yaml``.

Example: "ချိုသာသော နမူနာ" (sweet example) should be
"ရှင်းလင်းသော နမူနာ" (clear example).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from myspellchecker.core.constants import (
    ET_COLLOCATION_ERROR,
    ET_CONFUSABLE_ERROR,
    ET_CONTEXT_PROBABILITY,
    ET_SYLLABLE,
    ET_WORD,
)

if TYPE_CHECKING:
    from myspellchecker.providers.base import DictionaryProvider
from myspellchecker.core.detector_data import (
    TEXT_DETECTOR_CONFIDENCES,
)
from myspellchecker.core.detectors.utils import (
    get_existing_positions,
    iter_occurrences,
)
from myspellchecker.core.response import Error, SyllableError

# Error types that collocation rules can override.  Collocation rules are
# manually curated and highly precise, so they should take priority over
# generic or statistical error types at the same position.
_COLLOCATION_REPLACEABLE: frozenset[str] = frozenset(
    {
        ET_SYLLABLE,  # "invalid_syllable"
        ET_WORD,  # "invalid_word"
        ET_CONFUSABLE_ERROR,  # "confusable_error"
        ET_CONTEXT_PROBABILITY,  # "context_probability"
    }
)


class CollocationDetectionMixin:
    """Mixin providing collocation error detection.

    Detects wrong word partners in collocations by checking nearby
    context words within a configurable window.

    The ``_COLLOCATION_RULES`` list is an empty default that is
    overridden at init time by YAML-loaded values from
    ``rules/collocations.yaml``.
    """

    # --- Type stubs for attributes provided by SpellChecker ---
    provider: "DictionaryProvider"

    # Collocation rules: list of dicts, each with:
    #   wrong_word: str
    #   correct_word: str
    #   context_words: tuple[str, ...]
    #   direction: str ("left" | "right" | "both")
    #   window: int
    _COLLOCATION_RULES: list[dict[str, Any]] = []

    def _detect_collocation_errors(self, text: str, errors: list[Error]) -> None:
        """Detect wrong word partners in collocations.

        For each collocation rule, scans the text for occurrences of the
        wrong word.  When found, checks whether any of the specified
        context words appear within the configured window (measured in
        space-separated tokens) in the specified direction.  If a context
        word matches, the wrong word is flagged as a collocation error.
        """
        if not self._COLLOCATION_RULES:
            return

        from myspellchecker.core.detectors.utils import get_tokenized

        tokenized = get_tokenized(self, text)
        existing_positions = get_existing_positions(errors)
        tokens = tokenized.tokens
        token_starts = tokenized.positions

        for rule in self._COLLOCATION_RULES:
            wrong: str = rule["wrong_word"]
            correct: str = rule["correct_word"]
            context_words: tuple[str, ...] = rule["context_words"]
            direction: str = rule["direction"]
            window: int = rule["window"]

            for occ_start, _occ_end in iter_occurrences(text, wrong):
                # Find which token index this occurrence belongs to
                token_idx = -1
                for ti, ts in enumerate(token_starts):
                    if ts == occ_start:
                        token_idx = ti
                        break
                    # Handle multi-token wrong words: match if occurrence
                    # starts at this token's position
                    if ts <= occ_start < ts + len(tokens[ti]):
                        token_idx = ti
                        break

                if token_idx == -1:
                    # Occurrence not aligned to a token boundary -- skip
                    continue

                # Check context in the specified direction
                matched = False

                if direction in ("left", "both"):
                    left_start = max(0, token_idx - window)
                    for ci in range(left_start, token_idx):
                        tok = tokens[ci]
                        if any(cw in tok for cw in context_words):
                            matched = True
                            break

                if not matched and direction in ("right", "both"):
                    right_end = min(len(tokens), token_idx + 1 + window)
                    # Skip tokens that are part of the wrong_word itself
                    # (for multi-token wrong words)
                    wrong_token_count = len(wrong.split())
                    scan_start = token_idx + wrong_token_count
                    for ci in range(scan_start, right_end):
                        tok = tokens[ci]
                        if any(cw in tok for cw in context_words):
                            matched = True
                            break

                if not matched:
                    continue

                new_err = SyllableError(
                    text=wrong,
                    position=occ_start,
                    suggestions=[correct],
                    confidence=TEXT_DETECTOR_CONFIDENCES.get("collocation_error", 0.85),
                    error_type=ET_COLLOCATION_ERROR,
                )

                if occ_start in existing_positions:
                    # Context confirmed — replace a generic/statistical error
                    # at this position with the more precise collocation error.
                    _replace_collocation(errors, occ_start, new_err)
                else:
                    errors.append(new_err)
                    existing_positions.add(occ_start)


def _replace_collocation(errors: list[Error], position: int, new_error: Error) -> None:
    """Replace a replaceable error at *position* with a collocation error.

    Collocation rules are manually curated and context-confirmed, so they
    take priority over generic (invalid_syllable, invalid_word) and
    statistical (confusable_error, context_probability) errors.
    """
    for i, e in enumerate(errors):
        if e.position != position:
            continue
        if e.error_type in _COLLOCATION_REPLACEABLE:
            errors[i] = new_error
            return
        # Position occupied by a higher-priority preserved type — keep it.
        return
