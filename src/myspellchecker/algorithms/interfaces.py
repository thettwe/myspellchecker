"""
Algorithm Protocol Interfaces.

This module defines minimal Protocol interfaces for algorithm data access.
Algorithms declare exactly what they need instead of depending on the full
DictionaryProvider interface.

Design Pattern:
    - **Interface Segregation**: Small, focused protocols instead of large interfaces
    - **Dependency Inversion**: Algorithms depend on abstractions, not concretions
    - **Protocol Pattern**: PEP 544 structural subtyping for duck-typed compatibility

Key Protocols:
    - DictionaryLookup: Basic dictionary validation (is_valid_syllable, is_valid_word)
    - FrequencySource: Frequency-based ranking (get_*_frequency, get_all_*)
    - BigramSource: Bigram probability lookups (get_bigram_probability)
    - TrigramSource: Trigram probability lookups (get_trigram_probability)
    - POSRepository: POS transition probabilities (unigram, bigram, trigram)

Protocol Method Return Types:
    - is_valid_*: bool - True if term exists in dictionary
    - get_*_frequency: int - Frequency count (0 if not found)
    - get_*_probability: float - Probability value (0.0 if not found)
    - get_all_*: Iterator[tuple[str, int]] - (term, frequency) pairs
    - get_top_continuations: list[tuple[str, float]] - (word, probability) pairs

Benefits:
    - Algorithms can be used standalone without database
    - Easy to create in-memory test doubles
    - Clear dependency contracts
    - Type-safe with Protocol (PEP 544)
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol


class DictionaryLookup(Protocol):
    """
    Minimal interface for dictionary validation.

    Used by SymSpell and other algorithms that need basic word/syllable validation.
    """

    def is_valid_syllable(self, syllable: str) -> bool:
        """Check if syllable exists in dictionary."""
        ...

    def is_valid_word(self, word: str) -> bool:
        """Check if word exists in dictionary."""
        ...

    def get_syllable_frequency(self, syllable: str) -> int:
        """Get frequency count for syllable."""
        ...

    def get_word_frequency(self, word: str) -> int:
        """Get frequency count for word."""
        ...


class FrequencySource(Protocol):
    """
    Minimal interface for frequency-based ranking.

    Used by SymSpell for ranking suggestions by frequency.
    """

    def get_syllable_frequency(self, syllable: str) -> int:
        """Get frequency count for syllable."""
        ...

    def get_word_frequency(self, word: str) -> int:
        """Get frequency count for word."""
        ...

    def get_all_syllables(self) -> Iterator[tuple[str, int]]:
        """Get all syllables with frequencies for index building."""
        ...

    def get_all_words(self) -> Iterator[tuple[str, int]]:
        """Get all words with frequencies for index building."""
        ...


class BigramSource(Protocol):
    """
    Minimal interface for bigram probabilities.

    Used by NgramContextChecker for context validation.
    """

    def get_bigram_probability(self, w1: str, w2: str) -> float:
        """Get bigram probability P(w2|w1)."""
        ...

    def get_top_continuations(self, prev_word: str, limit: int = 10) -> list[tuple[str, float]]:
        """Get top N most likely continuations for prev_word."""
        ...


class TrigramSource(Protocol):
    """
    Minimal interface for trigram probabilities.

    Used by NgramContextChecker for deeper context validation.
    """

    def get_trigram_probability(self, w1: str, w2: str, w3: str) -> float:
        """Get trigram probability P(w3|w1,w2)."""
        ...


class POSRepository(Protocol):
    """
    Minimal interface for Part-of-Speech probability operations.

    Used by POS taggers (Viterbi) and grammar checkers for transition
    probabilities and POS-based validation.
    """

    def get_pos_unigram_probabilities(self) -> dict[str, float]:
        """
        Get POS tag unigram probabilities P(tag).

        Returns:
            Dictionary mapping POS tag -> probability
        """
        ...

    def get_pos_bigram_probabilities(self) -> dict[tuple[str, str], float]:
        """
        Get POS tag bigram (transition) probabilities P(tag2|tag1).

        Returns:
            Dictionary mapping (tag1, tag2) -> probability
        """
        ...

    def get_pos_trigram_probabilities(self) -> dict[tuple[str, str, str], float]:
        """
        Get POS tag trigram probabilities P(tag3|tag1,tag2).

        Returns:
            Dictionary mapping (tag1, tag2, tag3) -> probability
        """
        ...


# Re-export all protocols
__all__ = [
    "DictionaryLookup",
    "FrequencySource",
    "BigramSource",
    "TrigramSource",
    "POSRepository",
]
