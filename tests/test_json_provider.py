"""
Unit tests for JSONProvider.
"""

import json
from pathlib import Path

import pytest

from myspellchecker.core.constants import (
    STATS_KEY_BIGRAM_COUNT,
    STATS_KEY_SYLLABLE_COUNT,
    STATS_KEY_WORD_COUNT,
)
from myspellchecker.core.exceptions import ProviderError
from myspellchecker.providers import JSONProvider


@pytest.fixture
def json_file(tmp_path):
    """Create a temporary JSON dictionary file for testing."""
    dict_file = tmp_path / "dictionary.json"

    data = {
        "syllables": {"မြန်": 1000, "မာ": 800, "က": 500},
        "words": {
            "မြန်မာ": {"syllable_count": 2, "frequency": 500},
            "ကမ္ဘာ": {"syllable_count": 2, "frequency": 300},
        },
        "bigrams": {"မြန်မာ|နိုင်ငံ": 0.5, "ကမ္ဘာ|ကြီး": 0.3},
    }

    with open(dict_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    return str(dict_file)


class TestJSONProvider:
    """Test JSONProvider implementation."""

    def test_init_with_valid_file(self, json_file):
        """Test initialization with valid JSON file."""
        provider = JSONProvider(json_path=json_file)

        assert provider.json_path == Path(json_file)

        # Verify data loaded using public count methods
        assert provider.get_syllable_count() == 3
        assert provider.get_word_count() == 2
        assert provider.get_bigram_count() == 2

    def test_init_missing_file(self, tmp_path):
        """Test initialization with missing file raises FileNotFoundError."""
        missing_file = tmp_path / "missing.json"

        with pytest.raises(FileNotFoundError):
            JSONProvider(json_path=str(missing_file))

    def test_init_malformed_json(self, tmp_path):
        """Test initialization with malformed JSON."""
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("{invalid json", encoding="utf-8")

        with pytest.raises(json.JSONDecodeError):
            JSONProvider(json_path=str(bad_file))

    def test_init_invalid_structure(self, tmp_path):
        """Test initialization with invalid JSON structure."""
        # Root not a dict
        bad_file1 = tmp_path / "bad1.json"
        bad_file1.write_text("[]", encoding="utf-8")
        with pytest.raises(ProviderError, match="root must be an object"):
            JSONProvider(json_path=str(bad_file1))

        # Syllables not a dict
        bad_file2 = tmp_path / "bad2.json"
        bad_file2.write_text('{"syllables": []}', encoding="utf-8")
        with pytest.raises(ProviderError, match="'syllables' must be a dictionary"):
            JSONProvider(json_path=str(bad_file2))

    def test_is_valid_syllable(self, json_file):
        """Test syllable validation."""
        provider = JSONProvider(json_path=json_file)

        assert provider.is_valid_syllable("မြန်") is True
        assert provider.is_valid_syllable("xyz") is False
        assert provider.is_valid_syllable("") is False
        assert provider.is_valid_syllable(None) is False

    def test_is_valid_word(self, json_file):
        """Test word validation."""
        provider = JSONProvider(json_path=json_file)

        assert provider.is_valid_word("မြန်မာ") is True
        assert provider.is_valid_word("xyz") is False
        assert provider.is_valid_word("") is False
        assert provider.is_valid_word(None) is False

    def test_get_syllable_frequency(self, json_file):
        """Test syllable frequency retrieval."""
        provider = JSONProvider(json_path=json_file)

        assert provider.get_syllable_frequency("မြန်") == 1000
        assert provider.get_syllable_frequency("xyz") == 0
        assert provider.get_syllable_frequency(None) == 0

    def test_get_word_frequency(self, json_file):
        """Test word frequency retrieval."""
        provider = JSONProvider(json_path=json_file)

        assert provider.get_word_frequency("မြန်မာ") == 500
        assert provider.get_word_frequency("xyz") == 0
        assert provider.get_word_frequency(None) == 0

    def test_get_word_syllable_count(self, json_file):
        """Test word syllable count retrieval."""
        provider = JSONProvider(json_path=json_file)

        # Known word returns syllable count
        assert provider.get_word_syllable_count("မြန်မာ") == 2

        # Unknown word returns None (API-004 change)
        assert provider.get_word_syllable_count("xyz") is None
        assert provider.get_word_syllable_count(None) is None

        # Provider supports syllable count
        assert provider.supports_syllable_count is True

    def test_get_bigram_probability(self, json_file):
        """Test bigram probability retrieval."""
        provider = JSONProvider(json_path=json_file)

        assert provider.get_bigram_probability("မြန်မာ", "နိုင်ငံ") == 0.5
        assert provider.get_bigram_probability("unknown", "word") == 0.0
        assert provider.get_bigram_probability(None, "word") == 0.0

    def test_get_top_continuations(self, json_file):
        """Test getting top continuations."""
        provider = JSONProvider(json_path=json_file)

        continuations = provider.get_top_continuations("မြန်မာ")
        assert len(continuations) == 1
        assert continuations[0] == ("နိုင်ငံ", 0.5)

        assert provider.get_top_continuations("unknown") == []
        assert provider.get_top_continuations(None) == []

    def test_get_statistics(self, json_file):
        """Test statistics retrieval."""
        provider = JSONProvider(json_path=json_file)

        stats = provider.get_statistics()
        assert stats[STATS_KEY_SYLLABLE_COUNT] == 3
        assert stats[STATS_KEY_WORD_COUNT] == 2
        assert stats[STATS_KEY_BIGRAM_COUNT] == 2
        assert stats["json_path"] == json_file

    def test_repr(self, json_file):
        """Test string representation."""
        provider = JSONProvider(json_path=json_file)
        assert "JSONProvider" in repr(provider)
        assert "dictionary.json" in repr(provider)
