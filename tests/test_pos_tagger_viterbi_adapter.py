"""Tests for ViterbiPOSTaggerAdapter to boost coverage."""

from unittest.mock import Mock

import pytest


class TestViterbiPOSTaggerAdapter:
    """Test ViterbiPOSTaggerAdapter class."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock provider."""
        provider = Mock()
        provider.get_word_pos = Mock(return_value=None)
        provider.get_pos_bigram_probabilities = Mock(return_value={("N", "V"): 0.5})
        provider.get_pos_trigram_probabilities = Mock(return_value={("N", "V", "N"): 0.3})
        provider.get_pos_unigram_probabilities = Mock(return_value={"N": 0.5, "V": 0.3})
        return provider

    def test_adapter_init_with_provider_probs(self, mock_provider):
        """Test adapter loads probabilities from provider."""
        from myspellchecker.algorithms.pos_tagger_viterbi import ViterbiPOSTaggerAdapter

        ViterbiPOSTaggerAdapter(provider=mock_provider)

        # Check that provider methods were called
        mock_provider.get_pos_bigram_probabilities.assert_called_once()
        mock_provider.get_pos_trigram_probabilities.assert_called_once()
        mock_provider.get_pos_unigram_probabilities.assert_called_once()

    def test_adapter_init_with_explicit_probs(self, mock_provider):
        """Test adapter with explicit probabilities."""
        from myspellchecker.algorithms.pos_tagger_viterbi import ViterbiPOSTaggerAdapter

        bigram = {("N", "V"): 0.6}
        trigram = {("N", "V", "ADJ"): 0.4}
        unigram = {"N": 0.5}

        ViterbiPOSTaggerAdapter(
            provider=mock_provider,
            pos_bigram_probs=bigram,
            pos_trigram_probs=trigram,
            pos_unigram_probs=unigram,
        )

        # Provider methods should not be called when probs are provided
        mock_provider.get_pos_bigram_probabilities.assert_not_called()
        mock_provider.get_pos_trigram_probabilities.assert_not_called()
        mock_provider.get_pos_unigram_probabilities.assert_not_called()

    def test_adapter_tag_word_empty(self, mock_provider):
        """Test tag_word with empty string."""
        from myspellchecker.algorithms.pos_tagger_viterbi import ViterbiPOSTaggerAdapter

        adapter = ViterbiPOSTaggerAdapter(provider=mock_provider)
        result = adapter.tag_word("")
        assert result == adapter.unknown_tag

    def test_adapter_tag_word_valid(self, mock_provider):
        """Test tag_word with valid word."""
        from myspellchecker.algorithms.pos_tagger_viterbi import ViterbiPOSTaggerAdapter

        adapter = ViterbiPOSTaggerAdapter(provider=mock_provider)
        result = adapter.tag_word("test")
        assert isinstance(result, str)

    def test_adapter_tag_sequence_empty(self, mock_provider):
        """Test tag_sequence with empty list."""
        from myspellchecker.algorithms.pos_tagger_viterbi import ViterbiPOSTaggerAdapter

        adapter = ViterbiPOSTaggerAdapter(provider=mock_provider)
        result = adapter.tag_sequence([])
        assert result == []

    def test_adapter_tag_sequence_valid(self, mock_provider):
        """Test tag_sequence with valid words."""
        from myspellchecker.algorithms.pos_tagger_viterbi import ViterbiPOSTaggerAdapter

        adapter = ViterbiPOSTaggerAdapter(provider=mock_provider)
        result = adapter.tag_sequence(["word1", "word2"])
        assert len(result) == 2
        assert all(isinstance(t, str) for t in result)

    def test_adapter_tag_word_with_confidence(self, mock_provider):
        """Test tag_word_with_confidence."""
        from myspellchecker.algorithms.pos_tagger_viterbi import ViterbiPOSTaggerAdapter

        adapter = ViterbiPOSTaggerAdapter(provider=mock_provider)
        result = adapter.tag_word_with_confidence("test")

        assert result.word == "test"
        assert isinstance(result.tag, str)
        assert result.confidence == 1.0
        assert "method" in result.metadata
        assert result.metadata["method"] == "viterbi"

    def test_adapter_tag_sequence_with_confidence_empty(self, mock_provider):
        """Test tag_sequence_with_confidence with empty list."""
        from myspellchecker.algorithms.pos_tagger_viterbi import ViterbiPOSTaggerAdapter

        adapter = ViterbiPOSTaggerAdapter(provider=mock_provider)
        result = adapter.tag_sequence_with_confidence([])
        assert result == []

    def test_adapter_tag_sequence_with_confidence_valid(self, mock_provider):
        """Test tag_sequence_with_confidence with valid words."""
        from myspellchecker.algorithms.pos_tagger_viterbi import ViterbiPOSTaggerAdapter

        adapter = ViterbiPOSTaggerAdapter(provider=mock_provider)
        result = adapter.tag_sequence_with_confidence(["word1", "word2"])

        assert len(result) == 2
        for i, pred in enumerate(result):
            assert 0.0 <= pred.confidence <= 1.0
            assert pred.metadata["method"] in ("viterbi", "viterbi_marginal")
            assert pred.metadata["position"] == i

    def test_adapter_tagger_type_property(self, mock_provider):
        """Test tagger_type property."""
        from myspellchecker.algorithms.pos_tagger_base import TaggerType
        from myspellchecker.algorithms.pos_tagger_viterbi import ViterbiPOSTaggerAdapter

        adapter = ViterbiPOSTaggerAdapter(provider=mock_provider)
        assert adapter.tagger_type == TaggerType.VITERBI

    def test_adapter_supports_batch_property(self, mock_provider):
        """Test supports_batch property."""
        from myspellchecker.algorithms.pos_tagger_viterbi import ViterbiPOSTaggerAdapter

        adapter = ViterbiPOSTaggerAdapter(provider=mock_provider)
        assert adapter.supports_batch is False

    def test_adapter_is_fork_safe_property(self, mock_provider):
        """Test is_fork_safe property."""
        from myspellchecker.algorithms.pos_tagger_viterbi import ViterbiPOSTaggerAdapter

        adapter = ViterbiPOSTaggerAdapter(provider=mock_provider)
        assert adapter.is_fork_safe is True


class TestViterbiPOSTaggerAdapterNoProviderMethods:
    """Test adapter when provider doesn't have probability methods."""

    @pytest.fixture
    def basic_provider(self):
        """Create a provider without probability methods."""
        provider = Mock(spec=["get_word_pos"])
        provider.get_word_pos = Mock(return_value=None)
        return provider

    def test_adapter_fallback_empty_probs(self, basic_provider):
        """Test adapter uses empty dicts when provider doesn't support probs."""
        from myspellchecker.algorithms.pos_tagger_viterbi import ViterbiPOSTaggerAdapter

        adapter = ViterbiPOSTaggerAdapter(provider=basic_provider)
        # Should not raise - uses empty dicts as fallback
        assert adapter is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
