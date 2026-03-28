"""
Regex-based Myanmar text segmentation.

This module provides a pure-Python, rule-based segmenter for Myanmar text,
primarily focused on syllable segmentation using regular expressions.
"""

from __future__ import annotations

import re

from myspellchecker.core.constants import (
    KINZI_SEQUENCE,
    SENTENCE_SEPARATOR,
)
from myspellchecker.core.exceptions import TokenizationError
from myspellchecker.core.syllable_rules import SyllableRuleValidator
from myspellchecker.segmenters.base import Segmenter
from myspellchecker.text.normalize import normalize

__all__ = [
    "RegexSegmenter",
]

# Try import Cython segmenter
try:
    from myspellchecker.text.normalize_c import segment_syllables_c

    _HAS_CYTHON_SEGMENTER = True
except ImportError:
    _HAS_CYTHON_SEGMENTER = False


class RegexSegmenter(Segmenter):
    """
    Pure-Python, rule-based Myanmar text segmenter.

    This segmenter uses an iterative approach combined with `SyllableRuleValidator`
    to accurately segment Myanmar text into syllables. It is a lightweight,
    zero-dependency alternative.

    Args:
        allow_extended_myanmar: When True, include extended Myanmar blocks in
            Myanmar character detection. When False (default), only core Burmese
            block (U+1000-U+104F minus non-standard) is treated as Myanmar.

    Notes:
        - Syllable segmentation is rule-based and aims for high accuracy.
        - Word segmentation is a placeholder, returning syllables for now.
        - Consonant/starter patterns are core Burmese only; extended blocks have
          different orthographic rules.
    """

    def __init__(self, allow_extended_myanmar: bool = False):
        self.syllable_validator = SyllableRuleValidator(
            allow_extended_myanmar=allow_extended_myanmar
        )
        self._allow_extended_myanmar = allow_extended_myanmar

        # --- Sylbreak Logic (Adapted) ---
        # Pattern 1: Myanmar Syllable Start
        # Consonant (1000-1021)
        # Logic:
        # - (?<!(?<!\u103a)\u1039): NOT preceded by a stacking Virama (Virama NOT preceded by Asat).
        #   This allows breaking after Kinzi (Asat + Virama) but keeps stacks glued.
        # - (?!\u103a): NOT followed by Asat (Killer).
        #   This matches Cython behavior — Virama is handled by the lookbehind,
        #   so the lookahead only needs to check Asat.
        p_my_cons = r"(?<!(?<!\u103a)\u1039)[\u1000-\u1021](?!\u103a)"

        # Pattern 2: Independent Vowels, Digits, Symbols (Start new syllable)
        # 1022-102A (Indep Vowels incl. Shan A), 103F (Great Sa), 104C-104F (Symbols),
        # 1040-1049 (Digits), 104A-104B (Punct - ၊ and ။)
        # Note: U+1022 is included to prevent silent character dropping even though
        # it's out of scope for Burmese — it will be filtered by SyllableRuleValidator.
        p_other_starters = r"[\u1022-\u102a\u103f\u104c-\u104f\u1040-\u1049\u104a\u104b]"

        # Pattern 3: Non-Myanmar (English, punctuation, whitespace)
        # We group consecutive non-Myanmar chars to avoid over-fragmentation
        # Build pattern based on allow_extended_myanmar scope
        if allow_extended_myanmar:
            # Include core + Extended Core + Extended-A + Extended-B
            p_non_myanmar = r"[^\u1000-\u109F\uAA60-\uAA7F\uA9E0-\uA9FF]+"
        else:
            # Strict Burmese: U+1000-104F only (excludes Extended Core U+1050-109F)
            # Non-standard core chars (U+1022, U+1028, U+1033-U+1035) are still in range
            # but will be filtered by SyllableRuleValidator later
            p_non_myanmar = r"[^\u1000-\u104F]+"

        # Combined Pattern
        full_pattern = f"({p_my_cons}|{p_other_starters}|{p_non_myanmar})"

        self.regex = re.compile(full_pattern)

        # Note: Kinzi coda post-merge is applied in _merge_kinzi_codas() after
        # both the Cython and Python segmentation paths. See that method for details.

    def _validate_input(self, text: str) -> None:
        """
        Validate input text.
        """
        if not isinstance(text, str):
            raise TypeError(f"Expected str, got {type(text).__name__}")

        if not text or not text.strip():
            raise TokenizationError("Text cannot be empty or whitespace-only")

    def segment_syllables(self, text: str, normalize_text: bool = False) -> list[str]:
        """
        Segment Myanmar text into syllables using Sylbreak logic.

        This method uses a regex-based substitution strategy (ported from Sylbreak)
        to identify syllable boundaries efficiently. It correctly handles:
        - Standard Myanmar syllables
        - Stacked consonants (Virama)
        - Kinzi sequences (Nga+Asat+Virama)
        - Non-Myanmar chunks (English, numbers) - grouped for better tokens

        Args:
            text: Raw Myanmar text (Unicode).
            normalize_text: Whether to normalize text before segmentation.
                           Defaults to False for performance in bulk pipelines.

        Returns:
            List of Myanmar syllables and other tokens.
        """
        self._validate_input(text)

        # Normalize text (handles reordering for edge cases like ကျွန်ုပ်)
        cleaned_text = normalize(text) if normalize_text else text
        if not cleaned_text:
            return []

        # OPTIMIZATION: Use fast C++ implementation if available
        if _HAS_CYTHON_SEGMENTER:
            parts = segment_syllables_c(cleaned_text, self._allow_extended_myanmar)
            return self._merge_kinzi_codas(parts)

        # U+FFFF is a Unicode noncharacter, safe as internal separator
        sep = "\uffff"

        # Substitute: Insert separator BEFORE the match
        # Note: Sylbreak does s/.../SEP$1/g which puts SEP before the match content
        # We use a lambda to ensure robust insertion
        segmented_text = self.regex.sub(lambda m: sep + m.group(0), cleaned_text)

        # Split and filter empty strings
        # Note: If text starts with a match, we get an empty string at index 0
        parts = [s for s in segmented_text.split(sep) if s]

        return self._merge_kinzi_codas(parts)

    @staticmethod
    def _merge_kinzi_codas(parts: list[str]) -> list[str]:
        """Merge Kinzi-coda segments with the following segment.

        The nested lookbehind in the syllable regex correctly identifies phonetic
        boundaries at Kinzi (Nga+Asat+Virama), but produces coda segments like
        "သင်္" that end with virama and fail syllable validation. This method
        re-joins them with the next segment so the full Kinzi construct (e.g.
        "သင်္ဘော") is returned as one syllable.
        """
        if not parts:
            return parts

        merged: list[str] = []
        i = 0
        while i < len(parts):
            if parts[i].endswith(KINZI_SEQUENCE) and i + 1 < len(parts):
                # Merge Kinzi coda with next segment; continue merging if
                # the result still ends with a Kinzi suffix (consecutive codas)
                combined = parts[i] + parts[i + 1]
                i += 2
                while combined.endswith(KINZI_SEQUENCE) and i < len(parts):
                    combined += parts[i]
                    i += 1
                merged.append(combined)
            else:
                merged.append(parts[i])
                i += 1
        return merged

    def segment_words(self, text: str, normalize_text: bool = False) -> list[str]:
        """
        Word segmentation is not implemented in RegexSegmenter.

        RegexSegmenter is designed for syllable-level segmentation only.
        For word segmentation, use DefaultSegmenter with a dictionary-based
        engine (e.g., 'myword') that can perform proper word boundary detection.

        Args:
            text: Input text (unused, validates only).
            normalize_text: Whether to normalize (unused).

        Raises:
            NotImplementedError: Always raised. Use DefaultSegmenter for word segmentation.

        Example:
            >>> from myspellchecker.segmenters import DefaultSegmenter
            >>> segmenter = DefaultSegmenter(word_engine='myword')
            >>> words = segmenter.segment_words(text)
        """
        self._validate_input(text)
        raise NotImplementedError(
            "RegexSegmenter does not support word segmentation. "
            "Use DefaultSegmenter with engine='myword' for word segmentation, "
            "or use segment_syllables() for syllable-level tokenization."
        )

    def segment_sentences(self, text: str) -> list[str]:
        """
        Segment Myanmar text into sentences using basic rules.

        The separator (။) is preserved at the end of each sentence.
        """
        self._validate_input(text)
        parts = text.split(SENTENCE_SEPARATOR)
        sentences = []
        for i, part in enumerate(parts):
            part = part.strip()
            if part:
                if i < len(parts) - 1:
                    sentences.append(part + SENTENCE_SEPARATOR)
                else:
                    sentences.append(part)
        return sentences if sentences else [text]
