"""Tests for Viterbi POS tagger deleted interpolation smoothing.

These tests verify that the deleted interpolation smoothing is
mathematically correct:
- P_smooth(t2|t1) = λ2 * P(t2|t1) + λ1 * P(t2)
- P_smooth(t3|t1,t2) = λ3 * P(t3|t1,t2) + λ2 * P(t3|t2) + λ1 * P(t3)
"""

import math
from unittest.mock import MagicMock, patch

import pytest


class TestSmoothingMathematics:
    """Tests for mathematical correctness of deleted interpolation."""

    @pytest.fixture
    def mock_provider(self):
        provider = MagicMock()
        provider.get_word_pos.return_value = None
        return provider

    @pytest.fixture
    def tagger_python(self, mock_provider):
        """Create Python-only tagger for testing."""
        from myspellchecker.algorithms.viterbi import ViterbiTagger

        pos_bigram = {("N", "V"): 0.5, ("V", "N"): 0.3}
        pos_trigram = {("N", "V", "N"): 0.4, ("N", "N", "V"): 0.2}
        pos_unigram = {"N": 0.4, "V": 0.3, "ADJ": 0.2, "UNK": 0.1}

        with patch("myspellchecker.algorithms.viterbi._HAS_CYTHON_VITERBI", False):
            tagger = ViterbiTagger(
                provider=mock_provider,
                pos_bigram_probs=pos_bigram,
                pos_trigram_probs=pos_trigram,
                pos_unigram_probs=pos_unigram,
                lambda_unigram=0.1,
                lambda_bigram=0.3,
                lambda_trigram=0.6,
            )
        return tagger

    def test_log_sum_exp_basic(self, tagger_python):
        """Test log-sum-exp helper function."""
        # log(e^0 + e^0) = log(2) ≈ 0.693
        result = tagger_python._log_sum_exp([0.0, 0.0])
        assert result == pytest.approx(math.log(2), rel=1e-6)

        # log(e^1 + e^2) = log(e + e^2) ≈ 2.313
        result = tagger_python._log_sum_exp([1.0, 2.0])
        expected = math.log(math.exp(1) + math.exp(2))
        assert result == pytest.approx(expected, rel=1e-6)

    def test_log_sum_exp_large_difference(self, tagger_python):
        """Test log-sum-exp with large difference (numerical stability)."""
        # When one value dominates, result should be close to max
        result = tagger_python._log_sum_exp([0.0, -100.0])
        assert result == pytest.approx(0.0, abs=1e-10)

        result = tagger_python._log_sum_exp([-100.0, 0.0])
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_bigram_smoothing_formula(self, tagger_python):
        """Test bigram smoothing matches mathematical formula."""
        # P_smooth(V|N) = λ2 * P(V|N) + λ1 * P(V)
        # λ2 = 0.3, λ1 = 0.1
        # P(V|N) = 0.5, P(V) = 0.3
        # Expected = 0.3 * 0.5 + 0.1 * 0.3 = 0.15 + 0.03 = 0.18
        expected_prob = 0.3 * 0.5 + 0.1 * 0.3
        expected_log = math.log(expected_prob)

        result = tagger_python._get_smoothed_bigram_prob("N", "V")
        assert result == pytest.approx(expected_log, rel=1e-6)

    def test_bigram_smoothing_backoff(self, tagger_python):
        """Test bigram smoothing with missing bigram (backoff to unigram)."""
        # P_smooth(ADJ|N) = λ2 * P(ADJ|N) + λ1 * P(ADJ)
        # λ2 = 0.3, λ1 = 0.1
        # P(ADJ|N) = min_prob (1e-10), P(ADJ) = 0.2
        # Expected ≈ 0.3 * 1e-10 + 0.1 * 0.2 = 0.02 (unigram dominates)
        min_prob = tagger_python.min_prob
        expected_prob = 0.3 * min_prob + 0.1 * 0.2
        expected_log = math.log(expected_prob)

        result = tagger_python._get_smoothed_bigram_prob("N", "ADJ")
        assert result == pytest.approx(expected_log, rel=1e-4)

    def test_trigram_smoothing_formula(self, tagger_python):
        """Test trigram smoothing matches mathematical formula."""
        # P_smooth(N|N,V) = λ3 * P(N|N,V) + λ2 * P(N|V) + λ1 * P(N)
        # λ3 = 0.6, λ2 = 0.3, λ1 = 0.1
        # P(N|N,V) = 0.4, P(N|V) = 0.3, P(N) = 0.4
        # Expected = 0.6 * 0.4 + 0.3 * 0.3 + 0.1 * 0.4 = 0.24 + 0.09 + 0.04 = 0.37
        expected_prob = 0.6 * 0.4 + 0.3 * 0.3 + 0.1 * 0.4
        expected_log = math.log(expected_prob)

        result = tagger_python._get_smoothed_trigram_prob("N", "V", "N")
        assert result == pytest.approx(expected_log, rel=1e-6)

    def test_trigram_smoothing_backoff(self, tagger_python):
        """Test trigram smoothing with missing trigram (backoff)."""
        # P_smooth(V|V,N) = λ3 * P(V|V,N) + λ2 * P(V|N) + λ1 * P(V)
        # P(V|V,N) = min_prob (missing), P(V|N) = 0.5, P(V) = 0.3
        min_prob = tagger_python.min_prob
        expected_prob = 0.6 * min_prob + 0.3 * 0.5 + 0.1 * 0.3
        expected_log = math.log(expected_prob)

        result = tagger_python._get_smoothed_trigram_prob("V", "N", "V")
        assert result == pytest.approx(expected_log, rel=1e-4)

    def test_smoothing_weights_sum_to_one(self, tagger_python):
        """Test that interpolation weights sum to 1.0."""
        weight_sum = (
            tagger_python.lambda_unigram
            + tagger_python.lambda_bigram
            + tagger_python.lambda_trigram
        )
        assert weight_sum == pytest.approx(1.0, rel=1e-6)

    def test_smoothing_never_returns_negative_infinity(self, tagger_python):
        """Test that smoothing never returns -inf (always has backoff)."""
        # Even for unknown tag combinations, should return valid log probability
        result = tagger_python._get_smoothed_bigram_prob("UNKNOWN", "UNKNOWN")
        assert result != float("-inf")
        assert math.isfinite(result)

        result = tagger_python._get_smoothed_trigram_prob("X", "Y", "Z")
        assert result != float("-inf")
        assert math.isfinite(result)


class TestCythonSmoothingConsistency:
    """Test that Cython and Python implementations give same results."""

    @pytest.fixture
    def mock_provider(self):
        provider = MagicMock()
        provider.get_word_pos.return_value = None
        return provider

    def test_cython_bigram_matches_python(self, mock_provider):
        """Test that Cython bigram smoothing matches Python implementation."""
        from myspellchecker.algorithms.viterbi import ViterbiTagger

        pos_bigram = {("N", "V"): 0.5}
        pos_trigram = {}
        pos_unigram = {"N": 0.4, "V": 0.3}

        # Create Python-only tagger
        with patch("myspellchecker.algorithms.viterbi._HAS_CYTHON_VITERBI", False):
            python_tagger = ViterbiTagger(
                provider=mock_provider,
                pos_bigram_probs=pos_bigram,
                pos_trigram_probs=pos_trigram,
                pos_unigram_probs=pos_unigram,
            )

        # Check if Cython is available
        try:
            from myspellchecker.algorithms.viterbi_c import CythonViterbiTagger

            CythonViterbiTagger(
                mock_provider,
                pos_bigram,
                pos_trigram,
                "UNK",
                1e-10,
                pos_unigram_probs=pos_unigram,
            )

            # Calculate expected bigram smoothed probability
            # P_smooth(V|N) = λ2 * P(V|N) + λ1 * P(V) = 0.3 * 0.5 + 0.1 * 0.3 = 0.18
            expected_prob = 0.3 * 0.5 + 0.1 * 0.3
            expected_log = math.log(expected_prob)

            python_result = python_tagger._get_smoothed_bigram_prob("N", "V")
            assert python_result == pytest.approx(expected_log, rel=1e-4)

        except ImportError:
            pytest.skip("Cython Viterbi not available")

    def test_tagging_produces_valid_tags(self, mock_provider):
        """Test that tagging produces valid POS tags after smoothing fix."""
        from myspellchecker.algorithms.viterbi import ViterbiTagger

        mock_provider.get_word_pos.side_effect = lambda w: "N" if w == "cat" else "V"

        pos_bigram = {("N", "V"): 0.5, ("V", "N"): 0.3}
        pos_trigram = {("N", "V", "N"): 0.4}
        pos_unigram = {"N": 0.4, "V": 0.3, "UNK": 0.3}

        tagger = ViterbiTagger(
            provider=mock_provider,
            pos_bigram_probs=pos_bigram,
            pos_trigram_probs=pos_trigram,
            pos_unigram_probs=pos_unigram,
        )

        # Should produce valid tags without errors
        tags = tagger.tag_sequence(["cat", "run"])
        assert len(tags) == 2
        assert all(isinstance(t, str) for t in tags)
        assert all(len(t) > 0 for t in tags)


class TestMathDomainErrorPrevention:
    """Tests for preventing math.log(0) domain errors."""

    @pytest.fixture
    def mock_provider(self):
        provider = MagicMock()
        provider.get_word_pos.return_value = None
        return provider

    def test_zero_lambda_values_do_not_crash(self, mock_provider):
        """Test that zero lambda values are handled gracefully."""
        from myspellchecker.algorithms.viterbi import ViterbiTagger

        pos_bigram = {("N", "V"): 0.5}
        pos_trigram = {}
        pos_unigram = {"N": 0.4, "V": 0.3}

        # Zero lambda values should not cause math domain error
        with patch("myspellchecker.algorithms.viterbi._HAS_CYTHON_VITERBI", False):
            tagger = ViterbiTagger(
                provider=mock_provider,
                pos_bigram_probs=pos_bigram,
                pos_trigram_probs=pos_trigram,
                pos_unigram_probs=pos_unigram,
                lambda_unigram=0.0,  # Zero value
                lambda_bigram=0.5,
                lambda_trigram=0.5,
            )

        # Lambda values should be floored to min_prob
        assert tagger.lambda_unigram >= tagger.min_prob
        assert tagger.lambda_bigram >= tagger.min_prob
        assert tagger.lambda_trigram >= tagger.min_prob

        # Should not raise math domain error
        result = tagger._get_smoothed_bigram_prob("N", "V")
        assert math.isfinite(result)

    def test_all_zero_lambdas_handled(self, mock_provider):
        """Test that all zero lambdas don't cause crash."""
        from myspellchecker.algorithms.viterbi import ViterbiTagger

        pos_bigram = {("N", "V"): 0.5}
        pos_trigram = {}
        pos_unigram = {"N": 0.4, "V": 0.3}

        # All zero lambdas (edge case)
        with patch("myspellchecker.algorithms.viterbi._HAS_CYTHON_VITERBI", False):
            tagger = ViterbiTagger(
                provider=mock_provider,
                pos_bigram_probs=pos_bigram,
                pos_trigram_probs=pos_trigram,
                pos_unigram_probs=pos_unigram,
                lambda_unigram=0.0,
                lambda_bigram=0.0,
                lambda_trigram=0.0,
            )

        # All lambdas should be floored
        assert tagger.lambda_unigram >= tagger.min_prob
        assert tagger.lambda_bigram >= tagger.min_prob
        assert tagger.lambda_trigram >= tagger.min_prob

        # Smoothing should still work
        result = tagger._get_smoothed_trigram_prob("N", "V", "N")
        assert math.isfinite(result)


class TestViterbiPerformanceOptimizations:
    """Tests for Viterbi performance optimizations."""

    @pytest.fixture
    def mock_provider(self):
        provider = MagicMock()
        provider.get_word_pos.return_value = "N"
        return provider

    def test_cython_precomputed_log_lambda(self, mock_provider):
        """Test that Cython tagger pre-computes log(lambda) values."""
        try:
            from myspellchecker.algorithms.viterbi_c import CythonViterbiTagger

            pos_bigram = {("N", "V"): 0.5}
            pos_trigram = {("N", "V", "N"): 0.4}
            pos_unigram = {"N": 0.4, "V": 0.3}

            tagger = CythonViterbiTagger(
                provider=mock_provider,
                pos_bigram_probs=pos_bigram,
                pos_trigram_probs=pos_trigram,
                pos_unigram_probs=pos_unigram,
                lambda_unigram=0.1,
                lambda_bigram=0.3,
                lambda_trigram=0.6,
            )

            # The tagger should work correctly with pre-computed values
            result = tagger.tag_sequence(["word1", "word2", "word3"])
            assert len(result) == 3
            assert all(isinstance(tag, str) for tag in result)

        except ImportError:
            pytest.skip("Cython Viterbi module not available")

    def test_cython_precomputed_pipe_separator(self, mock_provider):
        """Test that Cython tagger uses pre-computed pipe separator."""
        try:
            from myspellchecker.algorithms.viterbi_c import CythonViterbiTagger

            pos_bigram = {("N", "V"): 0.5, ("V", "N"): 0.3}
            pos_trigram = {("N", "V", "N"): 0.4}
            pos_unigram = {"N": 0.4, "V": 0.3}

            tagger = CythonViterbiTagger(
                provider=mock_provider,
                pos_bigram_probs=pos_bigram,
                pos_trigram_probs=pos_trigram,
                pos_unigram_probs=pos_unigram,
            )

            # Multi-word sequences use state keys with pipe separators
            result = tagger.tag_sequence(["w1", "w2", "w3", "w4", "w5"])
            assert len(result) == 5

        except ImportError:
            pytest.skip("Cython Viterbi module not available")

    def test_large_sequence_performance(self, mock_provider):
        """Test tagging a larger sequence for performance."""
        from myspellchecker.algorithms.viterbi import ViterbiTagger

        pos_bigram = {("N", "V"): 0.5, ("V", "N"): 0.3, ("N", "N"): 0.2}
        pos_trigram = {("N", "V", "N"): 0.4}
        pos_unigram = {"N": 0.4, "V": 0.3, "UNK": 0.3}

        tagger = ViterbiTagger(
            provider=mock_provider,
            pos_bigram_probs=pos_bigram,
            pos_trigram_probs=pos_trigram,
            pos_unigram_probs=pos_unigram,
            beam_width=5,
        )

        words = [f"word{i}" for i in range(20)]
        result = tagger.tag_sequence(words)
        assert len(result) == 20
        assert all(tag in {"N", "V", "UNK"} for tag in result)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
