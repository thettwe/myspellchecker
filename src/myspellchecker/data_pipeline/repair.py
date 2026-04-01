"""
Segmentation repair module.

This module provides logic to repair incorrectly segmented Myanmar words,
specifically fixing cases where syllables are split across word boundaries
(e.g., "ကျော" + "င်း" -> "ကျောင်း").
"""

from __future__ import annotations

import re

from ..core.constants import (
    CONSONANTS,
    INDEPENDENT_VOWELS,
    MYANMAR_NUMERALS,
)
from ..core.syllable_rules import SyllableRuleValidator
from .config import runtime_flags as _flags

__all__ = [
    "SegmentationRepair",
    "set_allow_extended_myanmar",
]

# ---------------------------------------------------------------------------
# Shared runtime flags (replaces per-module globals)
# ---------------------------------------------------------------------------


def set_allow_extended_myanmar(allow: bool) -> None:
    """Set whether to allow extended Myanmar characters in validation.

    .. deprecated::
        Set ``PipelineConfig.allow_extended_myanmar`` instead and pass the
        config to ``Pipeline``.  This function will be removed in a future
        release.
    """
    import warnings

    warnings.warn(
        "set_allow_extended_myanmar() is deprecated. "
        "Use PipelineConfig.allow_extended_myanmar instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    _flags.allow_extended_myanmar = allow


class SegmentationRepair:
    """Repairs invalid word segmentations by merging broken syllables."""

    def __init__(self):
        self.validator = SyllableRuleValidator(allow_extended_myanmar=_flags.allow_extended_myanmar)

        # Regex to detect if a string STARTS with a valid base character
        # (Consonant, Independent Vowel, or Myanmar Numeral).
        # Myanmar numerals are valid standalone tokens (e.g., "၁၂၃")
        self.valid_start_pattern = re.compile(
            rf"^[{''.join(CONSONANTS)}{''.join(INDEPENDENT_VOWELS)}{''.join(MYANMAR_NUMERALS)}]"
        )

        # Regex to detect if a string starts with a Myanmar numeral
        # Used to skip syllable validation for numeral tokens
        self.numeral_start_pattern = re.compile(rf"^[{''.join(MYANMAR_NUMERALS)}]")

        # Explicit list of tokens that are often valid syllables but
        # in practice are almost always fragments of a larger word
        # when incorrectly segmented.
        self.suspicious_fragments = {
            "င်း",  # Part of ောင်း, ိုင်း
            "င့်",  # Part of ောင့်
            "န့်",  # Part of ောင့်
        }

        # Characters that indicate a syllable is "closed" or complete.
        # We generally should not append a rhyme fragment to these.
        self.closing_chars = {"း", "့", "်"}

        # Fragments that would cause double-ending when merged after closed syllable
        # These create invalid patterns like တွင်င်း, သော်င်း, သည်င်း
        self.double_ending_fragments = {
            "င်း",
            "င့်",
            "န့်",
        }

    def _would_create_double_ending(self, prev_token: str, current_token: str) -> bool:
        """
        Check if merging prev_token + current_token would create a double-ending pattern.

        Double-ending patterns occur when a closed syllable (ending in ်, း, or ့)
        is merged with a fragment like င်း, resulting in patterns like တွင်င်း.

        Args:
            prev_token: The previous token (e.g., "တွင်")
            current_token: The current token to potentially merge (e.g., "င်း")

        Returns:
            True if merging would create an invalid double-ending pattern
        """
        if not prev_token or not current_token:
            return False

        # Check if prev ends with closing char AND current is a double-ending fragment
        if prev_token[-1] in self.closing_chars and current_token in self.double_ending_fragments:
            return True

        return False

    def repair(self, tokens: list[str]) -> list[str]:
        """
        Repair a list of tokens by merging invalid fragments into previous words.

        Args:
            tokens: List of word tokens (e.g. ["ကျော", "င်း", "သည်"])

        Returns:
            List of repaired tokens (e.g. ["ကျောင်း", "သည်"])
        """
        if not tokens:
            return []

        repaired = [tokens[0]]

        for i in range(1, len(tokens)):
            current_token = tokens[i]
            prev_token = repaired[-1]

            # Condition 1: Starts with a dependent sign (invalid start)
            is_invalid_start = not self.valid_start_pattern.match(current_token)

            # Condition 2: Is explicitly known as a problematic fragment
            is_suspicious = current_token in self.suspicious_fragments

            # Condition 3: Is not a valid syllable structure (expensive check, do last)
            # Only check if it passed the first two checks but might still be weird
            # SKIP for numeral tokens - they are valid standalone and don't follow
            # syllable structure rules
            is_invalid_syllable = False
            is_numeral_token = self.numeral_start_pattern.match(current_token) is not None
            if not is_invalid_start and not is_suspicious and not is_numeral_token:
                is_invalid_syllable = not self.validator.validate(current_token)

            if is_invalid_start or is_suspicious or is_invalid_syllable:
                # Prevent double-ending merges
                # Don't merge if it would create patterns like တွင်င်း, သော်င်း, etc.
                if self._would_create_double_ending(prev_token, current_token):
                    repaired.append(current_token)
                    continue

                # Check if previous token is already "closed"
                # If so, appending another rhyme fragment is almost certainly wrong.
                # e.g. "ပြီး" (ends with း) + "င်း" -> Reject
                prev_is_closed = prev_token and prev_token[-1] in self.closing_chars

                if prev_is_closed:
                    repaired.append(current_token)
                    continue

                # Attempt Merge
                candidate = prev_token + current_token

                # CRITICAL CHECK: Only merge if the result is a VALID syllable
                # This prevents "သည်" + "င်း" -> "သည်င်း" (Invalid)
                # But allows "ကျော" + "င်း" -> "ကျောင်း" (Valid)
                if self.validator.validate(candidate):
                    repaired[-1] = candidate
                else:
                    # If merge creates garbage, don't merge.
                    # Treat 'current_token' as a separate word
                    # (which will likely be filtered out later)
                    repaired.append(current_token)
            else:
                # It's a valid starter, so it's a new word
                repaired.append(current_token)

        return repaired
