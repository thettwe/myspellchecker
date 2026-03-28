"""
Syllable tokenizer for Myanmar (Burmese) text.

Uses regex-based syllable breaking rules based on Myanmar script structure.
"""

from __future__ import annotations

import re

from myspellchecker.core.constants import KINZI_SEQUENCE


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
        # Myanmar consonant range
        self._my_consonant = r"က-အ"
        # English alphanumeric
        self._en_char = r"a-zA-Z0-9"
        # Other Myanmar characters and punctuation
        # Note: ၌၍၎၏ are logographic particles (U+104C-104F) - all should be standalone
        self._other_char = r"ဣဤဥဦဧဩဪဿ၌၍၎၏၀-၉၊။!-/:-@\\[-`{-~\\s"
        # Stacked consonant marker (္)
        self._ss_symbol = "္"
        # Asat/Virama marker (်)
        self._a_that = "်"

        # Build the syllable break pattern
        pattern = (
            # negative-lookbehind: not preceded by Myanmar consonant + stacked marker
            rf"((?<![{self._my_consonant}]{self._ss_symbol})["
            rf"{self._my_consonant}"  # any Burmese consonant
            rf"](?![{self._a_that}{self._ss_symbol}])"  # not followed by virama
            rf"|[{self._en_char}{self._other_char}])"
        )
        self._break_pattern: re.Pattern[str] = re.compile(pattern)

    def tokenize(self, text: str) -> list[str]:
        """
        Tokenize Myanmar text into syllables.

        Args:
            text: Input Myanmar text string.

        Returns:
            List of syllables.
        """
        if not text:
            return []

        # Insert space before each syllable boundary
        lined_text = re.sub(self._break_pattern, r" \1", text)
        parts = lined_text.split()
        return self._merge_kinzi_codas(parts)

    @staticmethod
    def _merge_kinzi_codas(parts: list[str]) -> list[str]:
        """
        Merge trailing kinzi fragments with following syllables.

        The regex splitter may emit fragments that end with င်္, such as "သင်္".
        These should be attached to the following syllable.
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
