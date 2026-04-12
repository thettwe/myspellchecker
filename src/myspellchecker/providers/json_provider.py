"""
JSON-based dictionary provider implementation.

This module provides a JSON-file backend for dictionary data,
useful for simple deployments and easy data inspection/editing.
"""

from __future__ import annotations

import json
from pathlib import Path

from myspellchecker.core.constants import (
    BIGRAM_SEPARATOR,
    DATA_KEY_BIGRAMS,
    DATA_KEY_FREQUENCY,
    DATA_KEY_POS,
    DATA_KEY_SYLLABLE_COUNT,
    DATA_KEY_SYLLABLES,
    DATA_KEY_WORDS,
    DEFAULT_FILE_ENCODING,
    STATS_KEY_BIGRAM_COUNT,
    STATS_KEY_SYLLABLE_COUNT,
    STATS_KEY_WORD_COUNT,
)
from myspellchecker.core.exceptions import ProviderError
from myspellchecker.providers.memory import MemoryProvider

__all__ = [
    "JSONProvider",
]


class JSONProvider(MemoryProvider):
    """
    JSON-file-based dictionary provider.

    This provider reads dictionary data from JSON files, providing a simple
    alternative to SQLite for smaller dictionaries or when human-readable
    data format is preferred.

    JSON Format:
        {
            "syllables": {
                "မြန်": 15432,
                "မာ": 12341,
                ...
            },
            "words": {
                "မြန်မာ": {"frequency": 8752, "syllable_count": 2},
                "နိုင်ငံ": {"frequency": 12341, "syllable_count": 2},
                ...
            },
            "bigrams": {
                "သူ|သွား": 0.234,
                "သူ|ဘယ်": 0.012,
                ...
            }
        }

    Thread Safety:
        This provider is thread-safe for read operations after initialization.
        Data is loaded into memory once and accessed via thread-safe dicts.

    Example:
        >>> from myspellchecker.providers import JSONProvider
        >>>
        >>> # Load from JSON file
        >>> provider = JSONProvider("dictionary.json")
        >>>
        >>> # Check syllable
        >>> provider.is_valid_syllable("မြန်")
        True
        >>>
        >>> # Get frequency
        >>> freq = provider.get_syllable_frequency("မြန်")
    """

    def __init__(self, json_path: str) -> None:
        """
        Initialize JSON provider.

        Args:
            json_path: Path to JSON dictionary file.

        Raises:
            FileNotFoundError: If JSON file doesn't exist.
            json.JSONDecodeError: If JSON file is malformed.
            ValueError: If JSON structure is invalid.

        Example:
            >>> provider = JSONProvider("dictionary.json")
        """
        super().__init__()
        self.json_path = Path(json_path)

        # Validate file exists
        if not self.json_path.exists():
            raise FileNotFoundError(
                f"JSON file not found: {self.json_path}\n"
                f"Please provide a valid JSON dictionary file."
            )

        self._word_syllable_counts: dict[str, int] = {}

        # Load data
        self._load_data()

    def _load_data(self) -> None:
        """
        Load dictionary data from JSON file.

        Raises:
            json.JSONDecodeError: If JSON is malformed.
            ValueError: If JSON structure is invalid.
        """
        try:
            with open(self.json_path, encoding=DEFAULT_FILE_ENCODING) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Invalid JSON in {self.json_path}: {e.msg}", e.doc, e.pos
            ) from e

        # Validate structure
        if not isinstance(data, dict):
            raise ProviderError(f"JSON root must be an object/dict in {self.json_path}")

        # Load syllables (syllable -> frequency)
        syllables_data = data.get(DATA_KEY_SYLLABLES, {})
        if not isinstance(syllables_data, dict):
            raise ProviderError("'syllables' must be a dictionary")

        for syllable, freq in syllables_data.items():
            self.add_syllable(syllable, int(freq))

        # Load words (word -> {frequency, syllable_count})
        words_data = data.get(DATA_KEY_WORDS, {})
        if not isinstance(words_data, dict):
            raise ProviderError("'words' must be a dictionary")

        for word, details in words_data.items():
            if isinstance(details, dict):
                self.add_word(word, int(details.get(DATA_KEY_FREQUENCY, 0)))
                self._word_syllable_counts[word] = int(details.get(DATA_KEY_SYLLABLE_COUNT, 0))
                if DATA_KEY_POS in details:
                    self.add_word_pos(word, details[DATA_KEY_POS])

        # Load bigrams (word1|word2 -> probability)
        bigrams_raw = data.get(DATA_KEY_BIGRAMS, {})
        if not isinstance(bigrams_raw, dict):
            raise ProviderError("'bigrams' must be a dictionary")

        # Parse bigrams (convert "word1|word2" to (word1, word2))
        for bigram_key, prob in bigrams_raw.items():
            parts = bigram_key.split(BIGRAM_SEPARATOR)
            if len(parts) == 2:
                word1, word2 = parts
                self.add_bigram(word1, word2, float(prob))

    def get_word_syllable_count(self, word: str) -> int | None:
        """
        Get syllable count for a word.

        Args:
            word: Myanmar word (Unicode string).

        Returns:
            Syllable count, or None if word not found.

        Example:
            >>> provider = JSONProvider("dictionary.json")
            >>> provider.get_word_syllable_count("မြန်မာ")
            2
            >>> provider.get_word_syllable_count("unknown")
            None
        """
        if not word:
            return None

        count = self._word_syllable_counts.get(word)
        return count if count is not None else None

    @property
    def supports_syllable_count(self) -> bool:
        """Whether this provider supports syllable count retrieval."""
        return True

    def get_statistics(self) -> dict:
        """
        Get dictionary statistics.

        Returns:
            Dictionary with statistics:
                - syllable_count: Number of syllables
                - word_count: Number of words
                - bigram_count: Number of bigrams
                - json_path: Path to JSON file

        Example:
            >>> provider = JSONProvider("dictionary.json")
            >>> stats = provider.get_statistics()
            >>> print(f"Syllables: {stats['syllable_count']:,}")
        """
        return {
            STATS_KEY_SYLLABLE_COUNT: self.get_syllable_count(),
            STATS_KEY_WORD_COUNT: self.get_word_count(),
            STATS_KEY_BIGRAM_COUNT: self.get_bigram_count(),
            "json_path": str(self.json_path),
        }

    def __repr__(self) -> str:
        """String representation of provider."""
        return f"JSONProvider(json_path='{self.json_path}')"
