"""Tests for base classes targeting uncovered lines."""

import pytest


class TestSegmenterBase:
    """Tests for segmenters/base.py abstract class."""

    def test_segmenter_cannot_be_instantiated(self):
        """Test that Segmenter ABC cannot be directly instantiated."""
        from myspellchecker.segmenters.base import Segmenter

        with pytest.raises(TypeError):
            Segmenter()

    def test_segment_and_tag_not_implemented(self):
        """Test segment_and_tag raises NotImplementedError."""
        from myspellchecker.segmenters.base import Segmenter

        class MinimalSegmenter(Segmenter):
            def segment_syllables(self, text: str):
                return [text]

            def segment_words(self, text: str):
                return [text]

            def segment_sentences(self, text: str):
                return [text]

        segmenter = MinimalSegmenter()
        with pytest.raises(NotImplementedError) as exc_info:
            segmenter.segment_and_tag("test")

        assert "MinimalSegmenter" in str(exc_info.value)

    def test_minimal_segmenter_methods(self):
        """Test minimal segmenter implementation."""
        from myspellchecker.segmenters.base import Segmenter

        class MinimalSegmenter(Segmenter):
            def segment_syllables(self, text: str):
                return list(text)

            def segment_words(self, text: str):
                return text.split()

            def segment_sentences(self, text: str):
                return [text]

        segmenter = MinimalSegmenter()
        assert segmenter.segment_syllables("abc") == ["a", "b", "c"]
        assert segmenter.segment_words("a b c") == ["a", "b", "c"]
        assert segmenter.segment_sentences("test") == ["test"]


class TestDictionaryProviderBase:
    """Tests for providers/base.py abstract class."""

    def test_dictionary_provider_cannot_be_instantiated(self):
        """Test that DictionaryProvider ABC cannot be directly instantiated."""
        from myspellchecker.providers.base import DictionaryProvider

        with pytest.raises(TypeError):
            DictionaryProvider()

    def test_minimal_provider_bulk_operations(self):
        """Test default bulk operation implementations."""
        from myspellchecker.providers.base import DictionaryProvider

        class MinimalProvider(DictionaryProvider):
            def __init__(self):
                self._data = {"valid": True, "test": True}
                self._freq = {"valid": 100, "test": 50}
                self._pos = {"valid": "N", "test": "V"}

            def is_valid_syllable(self, syllable: str) -> bool:
                return syllable in self._data

            def is_valid_word(self, word: str) -> bool:
                return word in self._data

            def get_syllable_frequency(self, syllable: str) -> int:
                return self._freq.get(syllable, 0)

            def get_word_frequency(self, word: str) -> int:
                return self._freq.get(word, 0)

            def get_word_pos(self, word: str):
                return self._pos.get(word)

            def get_bigram_probability(self, prev_word: str, current_word: str) -> float:
                return 0.0

            def get_trigram_probability(self, w1: str, w2: str, w3: str) -> float:
                return 0.0

            def get_fourgram_probability(self, w1: str, w2: str, w3: str, w4: str) -> float:
                return 0.0

            def get_fivegram_probability(
                self, w1: str, w2: str, w3: str, w4: str, w5: str
            ) -> float:
                return 0.0

            def get_pos_unigram_probabilities(self):
                return {}

            def get_pos_bigram_probabilities(self):
                return {}

            def get_pos_trigram_probabilities(self):
                return {}

            def get_top_continuations(self, prev_word: str, limit: int = 20):
                return []

            def get_all_syllables(self):
                return iter([("valid", 100), ("test", 50)])

            def get_all_words(self):
                return iter([("valid", 100), ("test", 50)])

        provider = MinimalProvider()

        assert provider.is_valid_syllables_bulk(["valid", "invalid"]) == {
            "valid": True,
            "invalid": False,
        }
        assert provider.get_word_frequencies_bulk(["valid", "test"]) == {"valid": 100, "test": 50}
        assert provider.get_word_pos_bulk(["valid", "unknown"]) == {"valid": "N", "unknown": None}


class TestCompoundCheckerBehavior:
    """Tests for grammar/checkers/compound.py behavior."""

    def test_compound_checker_analyze_word(self):
        """Test analyze_word with non-compound and reduplication patterns."""
        from myspellchecker.grammar.checkers.compound import CompoundChecker

        checker = CompoundChecker()

        result = checker.analyze_word("x")
        assert result["is_compound"] is False

        result = checker.analyze_word("abab")
        assert result["is_compound"] is True
        assert result["is_reduplication"] is True

    def test_get_compound_checker_singleton(self):
        """Test get_compound_checker returns singleton."""
        from myspellchecker.grammar.checkers.compound import get_compound_checker

        assert get_compound_checker() is get_compound_checker()

    def test_module_level_is_reduplication(self):
        """Test module-level is_reduplication function."""
        from myspellchecker.grammar.checkers.compound import is_reduplication

        assert is_reduplication("abab") is True
        assert is_reduplication("abc") is False


class TestPipelineConfig:
    """Tests for data_pipeline/config.py."""

    def test_pipeline_config_defaults(self):
        """Test PipelineConfig default values."""
        from myspellchecker.data_pipeline.config import PipelineConfig

        config = PipelineConfig()
        assert config.batch_size == 10000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
