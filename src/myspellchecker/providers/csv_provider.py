"""
CSV-based dictionary provider implementation.

This module provides a CSV-file backend for dictionary data,
useful for spreadsheet-based dictionary management and easy editing.
"""

from __future__ import annotations

import csv
from pathlib import Path

from myspellchecker.core.constants import (
    CSV_HEADER_POS,
    CSV_HEADER_WORD1,
    CSV_HEADER_WORD2,
    DATA_KEY_FREQUENCY,
    DATA_KEY_PROBABILITY,
    DATA_KEY_SYLLABLE,
    DATA_KEY_SYLLABLE_COUNT,
    DATA_KEY_WORD,
    DEFAULT_FILE_ENCODING,
    STATS_KEY_BIGRAM_COUNT,
    STATS_KEY_SYLLABLE_COUNT,
    STATS_KEY_WORD_COUNT,
)
from myspellchecker.core.exceptions import ProviderError

from .memory import MemoryProvider

__all__ = [
    "CSVProvider",
]


class CSVProvider(MemoryProvider):
    """
    CSV-file-based dictionary provider.

    This provider reads dictionary data from CSV files and stores them in memory,
    inheriting from MemoryProvider for fast lookups.

    CSV Format:
        **syllables.csv:**
        syllable,frequency
        မြန်,15432
        ...

        **words.csv:**
        word,frequency
        မြန်မာ,8752
        ...

        **bigrams.csv:**
        word1,word2,probability
        သူ,သွား,0.234
        ...

    Note: Syllable counts in words.csv are currently ignored to maintain
    compatibility with MemoryProvider structure.

    Example:
        >>> from myspellchecker.providers import CSVProvider
        >>> provider = CSVProvider(syllables_csv="syllables.csv")
    """

    def __init__(
        self,
        syllables_csv: str | None = None,
        words_csv: str | None = None,
        bigrams_csv: str | None = None,
        encoding: str = DEFAULT_FILE_ENCODING,
    ) -> None:
        """
        Initialize CSV provider.

        Args:
            syllables_csv: Path to syllables CSV file.
            words_csv: Path to words CSV file.
            bigrams_csv: Path to bigrams CSV file.
            encoding: CSV file encoding.
        """
        super().__init__()

        self.syllables_csv = Path(syllables_csv) if syllables_csv else None
        self.words_csv = Path(words_csv) if words_csv else None
        self.bigrams_csv = Path(bigrams_csv) if bigrams_csv else None
        self.encoding = encoding

        # Extra metadata not supported by base MemoryProvider
        self._word_syllable_counts: dict[str, int] = {}

        # Load data
        if self.syllables_csv:
            self._load_syllables()
        if self.words_csv:
            self._load_words()
        if self.bigrams_csv:
            self._load_bigrams()

    def _load_syllables(self) -> None:
        if not self.syllables_csv or not self.syllables_csv.exists():
            raise FileNotFoundError(f"Syllables CSV not found: {self.syllables_csv}")

        with open(self.syllables_csv, "r", encoding=self.encoding) as f:
            reader = csv.DictReader(f)
            try:
                for row in reader:
                    self.add_syllable(row[DATA_KEY_SYLLABLE].strip(), int(row[DATA_KEY_FREQUENCY]))
            except KeyError as e:
                raise ProviderError(f"Invalid Syllables CSV format. Missing column: {e}") from e

    def _load_words(self) -> None:
        if not self.words_csv or not self.words_csv.exists():
            raise FileNotFoundError(f"Words CSV not found: {self.words_csv}")

        with open(self.words_csv, "r", encoding=self.encoding) as f:
            reader = csv.DictReader(f)
            try:
                for row in reader:
                    word = row[DATA_KEY_WORD].strip()
                    self.add_word(word, int(row[DATA_KEY_FREQUENCY]))
                    if DATA_KEY_SYLLABLE_COUNT in row:
                        self._word_syllable_counts[word] = int(row[DATA_KEY_SYLLABLE_COUNT])
                    if CSV_HEADER_POS in row and row[CSV_HEADER_POS]:
                        self.add_word_pos(word, row[CSV_HEADER_POS].strip())
            except KeyError as e:
                raise ProviderError(f"Invalid Words CSV format. Missing column: {e}") from e

    def _load_bigrams(self) -> None:
        if not self.bigrams_csv or not self.bigrams_csv.exists():
            raise FileNotFoundError(f"Bigrams CSV not found: {self.bigrams_csv}")

        with open(self.bigrams_csv, "r", encoding=self.encoding) as f:
            reader = csv.DictReader(f)
            try:
                for row in reader:
                    self.add_bigram(
                        row[CSV_HEADER_WORD1].strip(),
                        row[CSV_HEADER_WORD2].strip(),
                        float(row[DATA_KEY_PROBABILITY]),
                    )
            except KeyError as e:
                raise ProviderError(f"Invalid Bigrams CSV format. Missing column: {e}") from e

    def get_word_syllable_count(self, word: str) -> int | None:
        """
        Get syllable count for a word.

        Args:
            word: Myanmar word.

        Returns:
            Syllable count, or None if word not found.
        """
        if not word:
            return None
        return self._word_syllable_counts.get(word)

    @property
    def supports_syllable_count(self) -> bool:
        """Whether this provider supports syllable count retrieval."""
        return True

    def get_statistics(self) -> dict:
        """Get dictionary statistics."""
        return {
            STATS_KEY_SYLLABLE_COUNT: self.get_syllable_count(),
            STATS_KEY_WORD_COUNT: self.get_word_count(),
            STATS_KEY_BIGRAM_COUNT: self.get_bigram_count(),
            "syllables_csv": str(self.syllables_csv) if self.syllables_csv else None,
            "words_csv": str(self.words_csv) if self.words_csv else None,
            "bigrams_csv": str(self.bigrams_csv) if self.bigrams_csv else None,
        }

    def __repr__(self) -> str:
        csv_files = []
        if self.syllables_csv:
            csv_files.append(f"syllables='{self.syllables_csv.name}'")
        if self.words_csv:
            csv_files.append(f"words='{self.words_csv.name}'")
        if self.bigrams_csv:
            csv_files.append(f"bigrams='{self.bigrams_csv.name}'")
        return f"CSVProvider({', '.join(csv_files)})"
