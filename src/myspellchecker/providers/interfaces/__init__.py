"""
Provider Protocol Interfaces.

This module defines minimal Protocol interfaces for dictionary/database access.
These interfaces follow the Interface Segregation Principle - components depend
only on the methods they actually use.

Key Protocols:
- SyllableRepository: Syllable validation and frequency
- WordRepository: Word validation and frequency
- NgramRepository: N-gram probability lookups
- POSRepository: POS tagging data access

Benefits:
- Validators depend on minimal interfaces, not monolithic DictionaryProvider
- Easy to create test doubles
- Clear dependency contracts
- Type-safe with Protocol (PEP 544)
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol


class SyllableRepository(Protocol):
    """
    Minimal interface for syllable validation and frequency.

    Used by SyllableValidator to check syllable validity and get frequencies
    without depending on full DictionaryProvider.
    """

    def is_valid_syllable(self, syllable: str) -> bool:
        """
        Check if syllable is valid.

        Args:
            syllable: Syllable to check

        Returns:
            True if valid, False otherwise
        """
        ...

    def get_syllable_frequency(self, syllable: str) -> int:
        """
        Get frequency count for syllable.

        Args:
            syllable: Syllable to query

        Returns:
            Frequency count (0 if not found)
        """
        ...

    def get_all_syllables(self) -> Iterator[tuple[str, int]]:
        """
        Get all syllables with frequencies.

        Returns:
            Iterator of (syllable, frequency) tuples
        """
        ...


class WordRepository(Protocol):
    """
    Minimal interface for word validation and frequency.

    Used by WordValidator to check word validity and get frequencies.
    """

    def is_valid_word(self, word: str) -> bool:
        """
        Check if word is valid.

        Args:
            word: Word to check

        Returns:
            True if valid, False otherwise
        """
        ...

    def is_valid_words_bulk(self, words: list[str]) -> dict[str, bool]:
        """
        Check validity of multiple words in a single operation.

        Args:
            words: List of words to validate.

        Returns:
            Dictionary mapping word to validity.
        """
        ...

    def is_valid_vocabulary(self, word: str) -> bool:
        """
        Check if word is curated vocabulary (stricter than is_valid_word).

        Args:
            word: Word to check.

        Returns:
            True if curated vocabulary, False otherwise.
        """
        ...

    def get_word_frequency(self, word: str) -> int:
        """
        Get frequency count for word.

        Args:
            word: Word to query

        Returns:
            Frequency count (0 if not found)
        """
        ...

    def get_all_words(self) -> Iterator[tuple[str, int]]:
        """
        Get all words with frequencies.

        Returns:
            Iterator of (word, frequency) tuples
        """
        ...


class NgramRepository(Protocol):
    """
    Minimal interface for N-gram probability lookups.

    Used by ContextValidator and NgramContextChecker for context validation.
    """

    def get_bigram_probability(self, w1: str, w2: str) -> float:
        """
        Get bigram probability P(w2|w1).

        Args:
            w1: First word
            w2: Second word

        Returns:
            Probability (0.0 if not found)
        """
        ...

    def get_trigram_probability(self, w1: str, w2: str, w3: str) -> float:
        """
        Get trigram probability P(w3|w1,w2).

        Args:
            w1: First word
            w2: Second word
            w3: Third word

        Returns:
            Probability (0.0 if not found)
        """
        ...

    def get_fourgram_probability(self, w1: str, w2: str, w3: str, w4: str) -> float:
        """
        Get fourgram probability P(w4|w1,w2,w3).

        Args:
            w1: First word
            w2: Second word
            w3: Third word
            w4: Fourth word

        Returns:
            Probability (0.0 if not found)
        """
        ...

    def get_fivegram_probability(self, w1: str, w2: str, w3: str, w4: str, w5: str) -> float:
        """
        Get fivegram probability P(w5|w1,w2,w3,w4).

        Args:
            w1: First word
            w2: Second word
            w3: Third word
            w4: Fourth word
            w5: Fifth word

        Returns:
            Probability (0.0 if not found)
        """
        ...


# Re-export all protocols
__all__ = [
    "SyllableRepository",
    "WordRepository",
    "NgramRepository",
]
