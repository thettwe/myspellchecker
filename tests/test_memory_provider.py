"""
Tests for MemoryProvider implementation.

Tests cover: initialization, CRUD operations, validation edge cases,
bulk loading, error handling, and integration workflows.
"""

import pytest

from myspellchecker.providers import MemoryProvider


class TestMemoryProviderInitialization:
    """Test MemoryProvider initialization."""

    def test_empty_initialization(self):
        """Test creating empty MemoryProvider."""
        provider = MemoryProvider()
        assert provider.get_syllable_count() == 0
        assert provider.get_word_count() == 0
        assert provider.get_bigram_count() == 0

    def test_initialization_with_all_data(self):
        """Test initialization with syllables, words, and bigrams."""
        syllables = {"မြန်": 1500, "မာ": 2300}
        words = {"မြန်မာ": 850}
        bigrams = {("သူ", "သွား"): 0.234}

        provider = MemoryProvider(syllables=syllables, words=words, bigrams=bigrams)

        assert provider.get_syllable_count() == 2
        assert provider.is_valid_syllable("မြန်")
        assert provider.get_syllable_frequency("မာ") == 2300
        assert provider.get_word_count() == 1
        assert provider.is_valid_word("မြန်မာ")
        assert provider.get_word_frequency("မြန်မာ") == 850
        assert provider.get_bigram_count() == 1
        assert provider.get_bigram_probability("သူ", "သွား") == 0.234


class TestValidationAndFrequency:
    """Test validation and frequency retrieval for syllables, words, and bigrams."""

    def test_valid_and_invalid_syllables(self):
        """Test syllable validation for existing and non-existent entries."""
        provider = MemoryProvider(syllables={"မြန်": 100})
        assert provider.is_valid_syllable("မြန်") is True
        assert provider.is_valid_syllable("xyz") is False
        assert provider.is_valid_syllable("") is False

    def test_syllable_frequency(self):
        """Test frequency retrieval for existing and non-existent syllables."""
        provider = MemoryProvider(syllables={"မြန်": 1500})
        assert provider.get_syllable_frequency("မြန်") == 1500
        assert provider.get_syllable_frequency("xyz") == 0

    def test_zero_frequency_syllable_is_valid(self):
        """Test syllable with explicit zero frequency is still valid."""
        provider = MemoryProvider(syllables={"rare": 0})
        assert provider.is_valid_syllable("rare") is True
        assert provider.get_syllable_frequency("rare") == 0

    def test_valid_and_invalid_words(self):
        """Test word validation for existing and non-existent entries."""
        provider = MemoryProvider(words={"မြန်မာ": 850})
        assert provider.is_valid_word("မြန်မာ") is True
        assert provider.is_valid_word("invalid") is False
        assert provider.is_valid_word("") is False

    def test_word_frequency(self):
        """Test frequency retrieval for existing and non-existent words."""
        provider = MemoryProvider(words={"မြန်မာ": 850})
        assert provider.get_word_frequency("မြန်မာ") == 850
        assert provider.get_word_frequency("unknown") == 0

    def test_bigram_probability(self):
        """Test probability retrieval for existing and non-existent bigrams."""
        bigrams = {("သူ", "သွား"): 0.234}
        provider = MemoryProvider(bigrams=bigrams)

        assert provider.get_bigram_probability("သူ", "သွား") == 0.234
        assert provider.get_bigram_probability("abc", "xyz") == 0.0

    def test_bigram_order_matters(self):
        """Test that bigram order is significant."""
        bigrams = {("သူ", "သွား"): 0.234, ("သွား", "သူ"): 0.156}
        provider = MemoryProvider(bigrams=bigrams)

        assert provider.get_bigram_probability("သူ", "သွား") == 0.234
        assert provider.get_bigram_probability("သွား", "သူ") == 0.156


class TestDynamicDataManipulation:
    """Test adding/removing data dynamically."""

    def test_add_and_remove_syllable(self):
        """Test adding and removing syllables."""
        provider = MemoryProvider()

        provider.add_syllable("သူ", frequency=1000)
        assert provider.is_valid_syllable("သူ") is True
        assert provider.get_syllable_frequency("သူ") == 1000
        assert provider.get_syllable_count() == 1

        assert provider.remove_syllable("သူ") is True
        assert provider.is_valid_syllable("သူ") is False
        assert provider.remove_syllable("သူ") is False

    def test_add_syllable_default_frequency_and_overwrite(self):
        """Test default frequency and overwrite behavior."""
        provider = MemoryProvider()
        provider.add_syllable("သူ")
        assert provider.get_syllable_frequency("သူ") == 1

        provider.add_syllable("သူ", frequency=500)
        assert provider.get_syllable_frequency("သူ") == 500

    def test_add_syllable_validation_errors(self):
        """Test that invalid inputs raise ValueError."""
        provider = MemoryProvider()
        with pytest.raises(ValueError, match="Syllable cannot be empty"):
            provider.add_syllable("")
        with pytest.raises(ValueError, match="Frequency must be non-negative"):
            provider.add_syllable("သူ", frequency=-10)

    def test_add_and_remove_word(self):
        """Test adding and removing words."""
        provider = MemoryProvider()

        provider.add_word("မြန်မာ", frequency=850)
        assert provider.is_valid_word("မြန်မာ") is True
        assert provider.get_word_frequency("မြန်မာ") == 850

        assert provider.remove_word("မြန်မာ") is True
        assert provider.is_valid_word("မြန်မာ") is False
        assert provider.remove_word("မြန်မာ") is False

    def test_add_word_validation_errors(self):
        """Test that invalid word inputs raise ValueError."""
        provider = MemoryProvider()
        with pytest.raises(ValueError, match="Word cannot be empty"):
            provider.add_word("")
        with pytest.raises(ValueError, match="Frequency must be non-negative"):
            provider.add_word("word", frequency=-5)

    def test_add_and_remove_bigram(self):
        """Test adding and removing bigrams."""
        provider = MemoryProvider()

        provider.add_bigram("သူ", "သွား", 0.234)
        assert provider.get_bigram_probability("သူ", "သွား") == 0.234

        assert provider.remove_bigram("သူ", "သွား") is True
        assert provider.get_bigram_probability("သူ", "သွား") == 0.0
        assert provider.remove_bigram("သူ", "သွား") is False

    def test_add_bigram_validation_errors(self):
        """Test that invalid bigram inputs raise ValueError."""
        provider = MemoryProvider()
        with pytest.raises(ValueError, match="Words cannot be empty"):
            provider.add_bigram("", "word", 0.5)
        with pytest.raises(ValueError, match="Probability must be in range"):
            provider.add_bigram("word1", "word2", -0.1)
        with pytest.raises(ValueError, match="Probability must be in range"):
            provider.add_bigram("word1", "word2", 1.5)

    def test_clear_populated_provider(self):
        """Test clearing a populated provider."""
        provider = MemoryProvider(
            syllables={"သူ": 100}, words={"မြန်မာ": 50}, bigrams={("သူ", "သွား"): 0.234}
        )
        provider.clear()

        assert provider.get_syllable_count() == 0
        assert provider.get_word_count() == 0
        assert provider.get_bigram_count() == 0


class TestBulkLoading:
    """Test bulk loading methods."""

    def test_load_from_lists_all_data(self):
        """Test bulk loading all types of data."""
        provider = MemoryProvider()

        provider.load_from_lists(
            syllable_list=[("မြန်", 1500), ("မာ", 2300)],
            word_list=[("မြန်မာ", 850)],
            bigram_list=[("သူ", "သွား", 0.234)],
        )

        assert provider.get_syllable_count() == 2
        assert provider.get_word_count() == 1
        assert provider.get_bigram_count() == 1
        assert provider.get_syllable_frequency("မြန်") == 1500

    def test_load_from_lists_extends_existing_data(self):
        """Test that load_from_lists extends rather than replaces data."""
        provider = MemoryProvider(syllables={"သူ": 100})

        provider.load_from_lists(syllable_list=[("မြန်", 1500)])

        assert provider.get_syllable_count() == 2
        assert provider.is_valid_syllable("သူ") is True
        assert provider.is_valid_syllable("မြန်") is True

    def test_load_from_lists_duplicate_overwrites(self):
        """Test that duplicate entries overwrite previous values."""
        provider = MemoryProvider(syllables={"မြန်": 100})

        provider.load_from_lists(syllable_list=[("မြန်", 1500)])
        assert provider.get_syllable_frequency("မြန်") == 1500

    def test_load_from_lists_empty_and_none(self):
        """Test loading with empty lists and None parameters."""
        provider = MemoryProvider(syllables={"သူ": 100})

        provider.load_from_lists(syllable_list=[], word_list=None, bigram_list=None)
        assert provider.get_syllable_count() == 1


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    def test_complete_dictionary_workflow(self):
        """Test a complete dictionary creation and usage workflow."""
        provider = MemoryProvider()

        provider.add_syllable("မြန်", 1500)
        provider.add_syllable("မာ", 2300)
        provider.add_word("မြန်မာ", 850)
        provider.add_bigram("မြန်မာ", "နိုင်ငံ", 0.45)

        assert provider.is_valid_syllable("မြန်") is True
        assert provider.is_valid_syllable("မျန်") is False
        assert provider.is_valid_word("မြန်မာ") is True
        assert provider.is_valid_word("မြန်စာ") is False
        assert provider.get_bigram_probability("မြန်မာ", "နိုင်ငံ") == 0.45
        assert provider.get_syllable_frequency("မာ") > provider.get_syllable_frequency("မြန်")

    def test_dictionary_update_workflow(self):
        """Test updating an existing dictionary."""
        provider = MemoryProvider(syllables={"သူ": 100}, words={"သူတို့": 50})

        provider.add_syllable("သူ", 500)
        assert provider.get_syllable_frequency("သူ") == 500

        provider.add_word("ကျွန်တော်", 400)
        provider.remove_word("သူတို့")

        assert provider.is_valid_word("သူတို့") is False
        assert provider.is_valid_word("ကျွန်တော်") is True
