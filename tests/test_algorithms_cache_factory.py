from unittest.mock import Mock, patch

import pytest

from myspellchecker.algorithms.cache import (
    CachedBigramSource,
    CachedDictionaryLookup,
    CachedFrequencySource,
    CachedPOSRepository,
    CachedTrigramSource,
    with_cache,
)
from myspellchecker.algorithms.factory import AlgorithmFactory
from myspellchecker.algorithms.interfaces import (
    BigramSource,
    DictionaryLookup,
    FrequencySource,
    POSRepository,
    TrigramSource,
)
from myspellchecker.core.config import SemanticConfig, SymSpellConfig
from myspellchecker.providers import DictionaryProvider


class TestCachedDictionaryLookup:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.mock_provider = Mock(spec=DictionaryLookup)
        self.cached_lookup = CachedDictionaryLookup(
            self.mock_provider, syllable_cache_size=10, word_cache_size=10
        )

    def test_is_valid_syllable_caching(self):
        self.mock_provider.is_valid_syllable.return_value = True

        # First call - cache miss
        result1 = self.cached_lookup.is_valid_syllable("syllable")
        assert result1
        self.mock_provider.is_valid_syllable.assert_called_once_with("syllable")

        # Second call - cache hit
        result2 = self.cached_lookup.is_valid_syllable("syllable")
        assert result2
        self.mock_provider.is_valid_syllable.assert_called_once()  # Count shouldn't increase

    def test_is_valid_word_caching(self):
        self.mock_provider.is_valid_word.return_value = True

        self.cached_lookup.is_valid_word("word")
        self.mock_provider.is_valid_word.assert_called_once_with("word")

        self.cached_lookup.is_valid_word("word")
        self.mock_provider.is_valid_word.assert_called_once()

    def test_get_syllable_frequency_caching(self):
        self.mock_provider.get_syllable_frequency.return_value = 50

        freq1 = self.cached_lookup.get_syllable_frequency("syllable")
        assert freq1 == 50

        freq2 = self.cached_lookup.get_syllable_frequency("syllable")
        assert freq2 == 50
        self.mock_provider.get_syllable_frequency.assert_called_once()

    def test_get_word_frequency_caching(self):
        self.mock_provider.get_word_frequency.return_value = 100

        freq1 = self.cached_lookup.get_word_frequency("word")
        assert freq1 == 100

        freq2 = self.cached_lookup.get_word_frequency("word")
        assert freq2 == 100
        self.mock_provider.get_word_frequency.assert_called_once()

    def test_cache_management(self):
        # Call methods to populate cache
        self.cached_lookup.is_valid_syllable("syl")
        self.cached_lookup.is_valid_word("word")

        # Check info
        stats = self.cached_lookup.cache_info()
        assert "syllable_validation" in stats
        assert "word_validation" in stats

        # Clear cache
        self.cached_lookup.cache_clear()

        # Verify next calls hit provider again
        self.cached_lookup.is_valid_syllable("syl")
        assert self.mock_provider.is_valid_syllable.call_count == 2


class TestCachedFrequencySource:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.mock_provider = Mock(spec=FrequencySource)
        self.cached_source = CachedFrequencySource(self.mock_provider, cache_size=10)

    def test_frequency_caching(self):
        self.mock_provider.get_syllable_frequency.return_value = 10
        self.mock_provider.get_word_frequency.return_value = 20

        assert self.cached_source.get_syllable_frequency("syl") == 10
        assert self.cached_source.get_syllable_frequency("syl") == 10
        self.mock_provider.get_syllable_frequency.assert_called_once()

        assert self.cached_source.get_word_frequency("word") == 20
        assert self.cached_source.get_word_frequency("word") == 20
        self.mock_provider.get_word_frequency.assert_called_once()

    def test_iteration_delegation(self):
        # Iteration methods should just delegate, not cache
        self.mock_provider.get_all_syllables.return_value = iter([("syl", 1)])
        self.mock_provider.get_all_words.return_value = iter([("word", 1)])

        list(self.cached_source.get_all_syllables())
        self.mock_provider.get_all_syllables.assert_called_once()

        list(self.cached_source.get_all_words())
        self.mock_provider.get_all_words.assert_called_once()


class TestCachedBigramSource:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.mock_provider = Mock(spec=BigramSource)
        self.cached_source = CachedBigramSource(self.mock_provider, cache_size=10)

    def test_probability_caching(self):
        self.mock_provider.get_bigram_probability.return_value = 0.5

        prob1 = self.cached_source.get_bigram_probability("w1", "w2")
        prob2 = self.cached_source.get_bigram_probability("w1", "w2")

        assert prob1 == 0.5
        assert prob2 == 0.5
        self.mock_provider.get_bigram_probability.assert_called_once()

    def test_continuations_caching(self):
        expected = [("next", 0.5)]
        self.mock_provider.get_top_continuations.return_value = expected

        # The cache stores a tuple, but the method returns a list
        result1 = self.cached_source.get_top_continuations("prev", 5)
        result2 = self.cached_source.get_top_continuations("prev", 5)

        assert result1 == expected
        assert result2 == expected
        assert result1 is not result2  # Should be different list objects
        self.mock_provider.get_top_continuations.assert_called_once()


class TestCachedTrigramSource:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.mock_provider = Mock(spec=TrigramSource)
        self.cached_source = CachedTrigramSource(self.mock_provider, cache_size=10)

    def test_probability_caching(self):
        self.mock_provider.get_trigram_probability.return_value = 0.1

        prob1 = self.cached_source.get_trigram_probability("w1", "w2", "w3")
        prob2 = self.cached_source.get_trigram_probability("w1", "w2", "w3")

        assert prob1 == 0.1
        assert prob2 == 0.1
        self.mock_provider.get_trigram_probability.assert_called_once()


class TestCachedPOSRepository:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.mock_provider = Mock(spec=POSRepository)
        self.cached_repo = CachedPOSRepository(self.mock_provider)

    def test_unigram_caching(self):
        data = {"NOUN": 0.1}
        self.mock_provider.get_pos_unigram_probabilities.return_value = data

        res1 = self.cached_repo.get_pos_unigram_probabilities()
        res2 = self.cached_repo.get_pos_unigram_probabilities()

        assert res1 == data
        assert res1 is res2
        self.mock_provider.get_pos_unigram_probabilities.assert_called_once()

    def test_bigram_caching(self):
        data = {("NOUN", "VERB"): 0.1}
        self.mock_provider.get_pos_bigram_probabilities.return_value = data

        res1 = self.cached_repo.get_pos_bigram_probabilities()
        res2 = self.cached_repo.get_pos_bigram_probabilities()

        assert res1 == data
        assert res1 is res2
        self.mock_provider.get_pos_bigram_probabilities.assert_called_once()

    def test_trigram_caching(self):
        data = {("NOUN", "VERB", "PART"): 0.1}
        self.mock_provider.get_pos_trigram_probabilities.return_value = data

        res1 = self.cached_repo.get_pos_trigram_probabilities()
        res2 = self.cached_repo.get_pos_trigram_probabilities()

        assert res1 == data
        assert res1 is res2
        self.mock_provider.get_pos_trigram_probabilities.assert_called_once()


class TestAlgorithmFactory:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.mock_provider = Mock(spec=DictionaryProvider)
        self.factory = AlgorithmFactory(self.mock_provider, enable_caching=True)

    def test_init_with_caching(self):
        assert isinstance(self.factory.dict_source, CachedDictionaryLookup)
        assert isinstance(self.factory.freq_source, CachedFrequencySource)
        assert isinstance(self.factory.bigram_source, CachedBigramSource)

    def test_init_without_caching(self):
        factory = AlgorithmFactory(self.mock_provider, enable_caching=False)
        assert factory.dict_source is self.mock_provider
        assert factory.freq_source is self.mock_provider

    @patch("myspellchecker.algorithms.symspell.SymSpell")
    def test_create_symspell(self, mock_symspell_cls):
        config = SymSpellConfig(prefix_length=5)
        self.factory.create_symspell(config)

        mock_symspell_cls.assert_called_once()
        _, kwargs = mock_symspell_cls.call_args
        assert kwargs["prefix_length"] == 5
        assert kwargs["provider"] == self.mock_provider

    def test_create_semantic_checker(self):
        # Should return None if no model configured
        config = SemanticConfig()  # No model path
        assert self.factory.create_semantic_checker(config) is None

    def test_with_cache_helper(self):
        cached = with_cache(self.mock_provider)
        assert isinstance(cached, CachedDictionaryLookup)
