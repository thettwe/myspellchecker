"""Comprehensive tests for providers/base.py.

Tests cover:
- Default bulk operation implementations
- Existence check methods (has_syllable, has_word, __contains__)
- Syllable count methods
- Factory method create()
"""

from typing import Dict, Iterator, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import pytest

from myspellchecker.core.exceptions import ProviderError
from myspellchecker.providers.base import DictionaryProvider


class ConcreteProvider(DictionaryProvider):
    """Concrete implementation for testing abstract base class."""

    def __init__(self):
        """Initialize with test data."""
        self._syllables = {"မြန်": True, "မာ": True, "ကား": True}
        self._words = {"မြန်မာ": True, "ကား": True}
        self._syllable_freq = {"မြန်": 100, "မာ": 50, "ကား": 30}
        self._word_freq = {"မြန်မာ": 80, "ကား": 25}
        self._word_pos = {"မြန်မာ": "N", "ကား": "N", "သွား": "V"}

    def is_valid_syllable(self, syllable: str) -> bool:
        return syllable in self._syllables

    def is_valid_word(self, word: str) -> bool:
        return word in self._words

    def get_syllable_frequency(self, syllable: str) -> int:
        return self._syllable_freq.get(syllable, 0)

    def get_word_frequency(self, word: str) -> int:
        return self._word_freq.get(word, 0)

    def get_word_pos(self, word: str) -> Optional[str]:
        return self._word_pos.get(word)

    def get_bigram_probability(self, prev_word: str, current_word: str) -> float:
        return 0.5

    def get_trigram_probability(self, w1: str, w2: str, w3: str) -> float:
        return 0.1

    def get_fourgram_probability(self, w1: str, w2: str, w3: str, w4: str) -> float:
        return 0.0

    def get_fivegram_probability(self, w1: str, w2: str, w3: str, w4: str, w5: str) -> float:
        return 0.0

    def get_top_continuations(self, prev_word: str, limit: int = 20) -> List[Tuple[str, float]]:
        return [("word1", 0.5), ("word2", 0.3)]

    def get_all_syllables(self) -> Iterator[Tuple[str, int]]:
        for s, f in self._syllable_freq.items():
            yield (s, f)

    def get_all_words(self) -> Iterator[Tuple[str, int]]:
        for w, f in self._word_freq.items():
            yield (w, f)

    def get_pos_unigram_probabilities(self) -> Dict[str, float]:
        return {"N": 0.4, "V": 0.3}

    def get_pos_bigram_probabilities(self) -> Dict[Tuple[str, str], float]:
        return {("N", "V"): 0.3, ("V", "N"): 0.2}

    def get_pos_trigram_probabilities(self) -> Dict[Tuple[str, str, str], float]:
        return {("N", "V", "N"): 0.1}


class TestBulkOperations:
    """Tests for default bulk operation implementations."""

    @pytest.fixture
    def provider(self):
        return ConcreteProvider()

    def test_is_valid_syllables_bulk(self, provider):
        """Test bulk syllable validation with mix of valid and invalid."""
        result = provider.is_valid_syllables_bulk(["မြန်", "xyz", "ကား"])
        assert result["မြန်"] is True
        assert result["xyz"] is False
        assert result["ကား"] is True

    def test_is_valid_syllables_bulk_empty(self, provider):
        """Test bulk syllable validation with empty list."""
        assert provider.is_valid_syllables_bulk([]) == {}

    def test_is_valid_words_bulk(self, provider):
        """Test bulk word validation with mix of valid and invalid."""
        result = provider.is_valid_words_bulk(["မြန်မာ", "unknown", "ကား"])
        assert result["မြန်မာ"] is True
        assert result["unknown"] is False
        assert result["ကား"] is True

    def test_get_syllable_frequencies_bulk(self, provider):
        """Test bulk syllable frequency retrieval."""
        result = provider.get_syllable_frequencies_bulk(["မြန်", "xyz"])
        assert result["မြန်"] == 100
        assert result["xyz"] == 0

    def test_get_word_frequencies_bulk(self, provider):
        """Test bulk word frequency retrieval."""
        result = provider.get_word_frequencies_bulk(["မြန်မာ", "unknown"])
        assert result["မြန်မာ"] == 80
        assert result["unknown"] == 0

    def test_get_word_pos_bulk(self, provider):
        """Test bulk POS tag retrieval."""
        result = provider.get_word_pos_bulk(["မြန်မာ", "သွား", "unknown"])
        assert result["မြန်မာ"] == "N"
        assert result["သွား"] == "V"
        assert result["unknown"] is None

    def test_bulk_with_duplicates(self, provider):
        """Test bulk operations with duplicate inputs."""
        result = provider.is_valid_syllables_bulk(["မြန်", "မြန်", "မာ"])
        assert "မြန်" in result
        assert "မာ" in result

    def test_bulk_preserves_order(self, provider):
        """Test that bulk operations preserve input order."""
        syllables = ["ကား", "မြန်", "မာ"]
        result = provider.is_valid_syllables_bulk(syllables)
        assert list(result.keys()) == syllables


class TestExistenceChecks:
    """Tests for has_syllable, has_word, and __contains__."""

    @pytest.fixture
    def provider(self):
        return ConcreteProvider()

    def test_has_syllable(self, provider):
        """Test has_syllable for existing and non-existing."""
        assert provider.has_syllable("မြန်") is True
        assert provider.has_syllable("xyz") is False
        assert provider.has_syllable("") is False

    def test_has_word(self, provider):
        """Test has_word for existing and non-existing."""
        assert provider.has_word("မြန်မာ") is True
        assert provider.has_word("unknown") is False
        assert provider.has_word("") is False

    def test_contains(self, provider):
        """Test __contains__ checks syllable and word."""
        assert "မြန်" in provider
        assert "မြန်မာ" in provider
        assert "xyz" not in provider
        assert "" not in provider

    def test_syllable_count_defaults(self, provider):
        """Test default syllable count methods."""
        assert provider.get_word_syllable_count("မြန်မာ") is None
        assert provider.supports_syllable_count is False


class TestFactory:
    """Tests for factory method create()."""

    def test_create_memory_provider(self):
        """Test creating Memory provider via factory."""
        provider = DictionaryProvider.create(
            "memory",
            syllables={"မြန်": 100, "မာ": 50},
            words={"မြန်မာ": 80},
        )
        assert provider.is_valid_syllable("မြန်")
        assert provider.get_syllable_frequency("မြန်") == 100

    def test_create_case_insensitive(self):
        """Test that provider_type is case insensitive."""
        with patch("myspellchecker.providers.memory.MemoryProvider") as mock_memory:
            mock_memory.return_value = MagicMock()
            DictionaryProvider.create("MEMORY")
            mock_memory.assert_called()

    def test_create_unknown_provider_raises_error(self):
        """Test that unknown provider type raises ValueError."""
        with pytest.raises(ProviderError) as exc_info:
            DictionaryProvider.create("unknown_provider")
        assert "Unknown provider type" in str(exc_info.value)

    def test_create_sqlite_provider(self):
        """Test creating SQLite provider via factory (mocked)."""
        with patch("myspellchecker.providers.sqlite.SQLiteProvider") as mock_sqlite:
            mock_sqlite.return_value = MagicMock()
            result = DictionaryProvider.create("sqlite", database_path="test.db")
            mock_sqlite.assert_called_once_with(database_path="test.db")
            assert result == mock_sqlite.return_value
