"""
Syllable tokenizer for Myanmar (Burmese) text.

Uses regex-based syllable breaking rules based on Myanmar script structure.
"""

from __future__ import annotations

import re

from myspellchecker.core.constants import KINZI_SEQUENCE

# Unicode noncharacter used as internal separator during segmentation.
# U+FFFF is guaranteed never to appear in valid Unicode text.
_SEP = "\uffff"


class SyllableTokenizer:
    """
    Syllable Tokenizer using Sylbreak rules for Myanmar language.

    This tokenizer breaks Myanmar text into syllables using regex patterns
    that identify syllable boundaries based on consonants, vowels, and
    diacritical marks.

    Example:
        >>> tokenizer = SyllableTokenizer()
        >>> tokenizer.tokenize("မြန်မာနိုင်ငံ")
        ['မြန်', 'မာ', 'နိုင်', 'ငံ']
    """

    def __init__(self) -> None:
        # Pattern identifies characters that START a new syllable.
        #
        # Alternative 1: single Myanmar character that can open a syllable
        #   [က-ဪ]  U+1000-U+102A: consonants and independent vowels
        #   ဿ      U+103F: Great Sa
        #   [၊-၏]  U+104A-U+104F: punctuation and logographic particles
        #
        # Alternative 2: run of Myanmar numerals (grouped, not split per digit)
        #   [၀-၉]+  U+1040-U+1049
        #
        # Alternative 3: run of non-Myanmar characters (ASCII, English, spaces …)
        #   [^က-၏]+  anything outside U+1000-U+104F
        #
        # Lookbehind (?<!္): stacked consonants (after virama U+1039) are not
        #   new-syllable starters; leave them attached to their cluster.
        #
        # Lookahead (?![ှျ]?[့္်]): a consonant followed by an optional
        #   ha-htoe/ya-pin medial then asat/virama/dot-below is a coda (final)
        #   consonant, not the onset of a new syllable — no break.
        self._break_pattern: re.Pattern[str] = re.compile(
            r"(?:(?<!္)([က-ဪဿ၊-၏]|[၀-၉]+|[^က-၏]+)(?![ှျ]?[့္်]))"
        )

    def tokenize(self, text: str) -> list[str]:
        """
        Tokenize Myanmar text into syllables.

        Args:
            text: Input Myanmar text string.

        Returns:
            List of syllables. Non-Myanmar runs (English, spaces) and Myanmar
            numeral runs are each returned as a single token.
        """
        if not text:
            return []

        segmented = self._break_pattern.sub(lambda m: _SEP + m.group(0), text)
        parts = [s.strip() for s in segmented.split(_SEP) if s.strip()]
        return self._merge_kinzi_codas(parts)

    @staticmethod
    def _merge_kinzi_codas(parts: list[str]) -> list[str]:
        """
        Merge trailing kinzi fragments with following syllables.

        The regex splitter may emit fragments that end with င်္, such as "သင်္".
        These should be attached to the following syllable.

        Note: the (?<!္) lookbehind in the current regex already keeps kinzi
        clusters intact at the pattern level, so this method rarely finds
        anything to merge. It is retained as a safety net for edge cases.
        """
        if not parts:
            return parts

        merged: list[str] = []
        i = 0
        while i < len(parts):
            current = parts[i]
            if current.endswith(KINZI_SEQUENCE) and i + 1 < len(parts):
                combined = current + parts[i + 1]
                i += 2
                while combined.endswith(KINZI_SEQUENCE) and i < len(parts):
                    combined += parts[i]
                    i += 1
                merged.append(combined)
            else:
                merged.append(current)
                i += 1
        return merged


__all__ = ["SyllableTokenizer"]
