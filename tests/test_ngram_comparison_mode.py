"""Tests for n-gram comparison mode (compare_contextual_probability).

Tests cover:
1. NgramContextChecker.compare_contextual_probability() — core comparison logic
2. NgramContextChecker._compute_bidirectional_prob() — probability computation

The ConfusableVariantStrategy and NgramContextValidationStrategy comparison
mode tests have been removed — that logic now lives in
NgramContextChecker.check_word_in_context().
"""

from unittest.mock import MagicMock

import pytest

from myspellchecker.algorithms.ngram_context_checker import NgramContextChecker
from myspellchecker.core.config import NgramContextConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_provider():
    """Create a mock provider with default zero returns."""
    provider = MagicMock()
    provider.get_bigram_probability.return_value = 0.0
    provider.get_trigram_probability.return_value = 0.0
    provider.get_fourgram_probability.return_value = 0.0
    provider.get_fivegram_probability.return_value = 0.0
    provider.get_word_frequency.return_value = 0
    provider.get_word_pos.return_value = None
    provider.get_top_continuations.return_value = []
    provider.get_metadata.return_value = None
    return provider


@pytest.fixture
def checker(mock_provider):
    """Create an NgramContextChecker with default config."""
    cfg = NgramContextConfig(
        bigram_threshold=0.01,
        trigram_threshold=0.005,
        right_context_threshold=0.1,
        min_unigram_threshold=10,
        unigram_denominator=1_000_000.0,
    )
    return NgramContextChecker(provider=mock_provider, config=cfg)


# ===========================================================================
# Tests for _compute_bidirectional_prob
# ===========================================================================


class TestComputeBidirectionalProb:
    """Tests for NgramContextChecker._compute_bidirectional_prob()."""

    def test_no_context_returns_unigram_fallback(self, checker, mock_provider):
        """With no prev/next words, falls back to unigram probability."""
        mock_provider.get_word_frequency.return_value = 5000
        prob = checker._compute_bidirectional_prob("word", [], [])
        # 5000 / 1_000_000 = 0.005
        assert prob == pytest.approx(0.005)

    def test_no_context_zero_freq_returns_zero(self, checker, mock_provider):
        """With no context and zero frequency, returns 0."""
        mock_provider.get_word_frequency.return_value = 0
        prob = checker._compute_bidirectional_prob("word", [], [])
        assert prob == 0.0

    def test_left_bigram_only(self, checker, mock_provider):
        """With only left bigram context."""
        mock_provider.get_bigram_probability.side_effect = lambda w1, w2: (
            0.05 if (w1, w2) == ("prev", "word") else 0.0
        )
        prob = checker._compute_bidirectional_prob("word", ["prev"], [])
        assert prob == pytest.approx(0.05)

    def test_right_bigram_only(self, checker, mock_provider):
        """With only right bigram context."""
        mock_provider.get_bigram_probability.side_effect = lambda w1, w2: (
            0.03 if (w1, w2) == ("word", "next") else 0.0
        )
        prob = checker._compute_bidirectional_prob("word", [], ["next"])
        assert prob == pytest.approx(0.03)

    def test_both_directions_averaged(self, checker, mock_provider):
        """With both left and right context, probabilities are averaged."""
        mock_provider.get_bigram_probability.side_effect = lambda w1, w2: {
            ("prev", "word"): 0.04,
            ("word", "next"): 0.06,
        }.get((w1, w2), 0.0)
        prob = checker._compute_bidirectional_prob("word", ["prev"], ["next"])
        assert prob == pytest.approx(0.05)  # (0.04 + 0.06) / 2

    def test_trigram_preferred_over_bigram_left(self, checker, mock_provider):
        """Trigram is preferred over bigram for left context."""
        mock_provider.get_trigram_probability.side_effect = lambda w1, w2, w3: (
            0.08 if (w1, w2, w3) == ("pp", "prev", "word") else 0.0
        )
        mock_provider.get_bigram_probability.return_value = 0.02
        prob = checker._compute_bidirectional_prob("word", ["pp", "prev"], [])
        assert prob == pytest.approx(0.08)

    def test_trigram_fallback_to_bigram_left(self, checker, mock_provider):
        """Falls back to bigram when trigram is zero."""
        mock_provider.get_trigram_probability.return_value = 0.0
        mock_provider.get_bigram_probability.side_effect = lambda w1, w2: (
            0.03 if (w1, w2) == ("prev", "word") else 0.0
        )
        prob = checker._compute_bidirectional_prob("word", ["pp", "prev"], [])
        assert prob == pytest.approx(0.03)

    def test_trigram_preferred_over_bigram_right(self, checker, mock_provider):
        """Trigram is preferred over bigram for right context."""
        mock_provider.get_trigram_probability.side_effect = lambda w1, w2, w3: (
            0.07 if (w1, w2, w3) == ("word", "next", "nn") else 0.0
        )
        mock_provider.get_bigram_probability.return_value = 0.01
        prob = checker._compute_bidirectional_prob("word", [], ["next", "nn"])
        assert prob == pytest.approx(0.07)


# ===========================================================================
# Tests for compare_contextual_probability
# ===========================================================================


class TestCompareContextualProbability:
    """Tests for NgramContextChecker.compare_contextual_probability()."""

    def test_no_alternatives_returns_none(self, checker):
        """Empty alternatives list returns (None, 0.0)."""
        result = checker.compare_contextual_probability(
            word="word", alternatives=[], prev_words=["prev"], next_words=["next"]
        )
        assert result == (None, 0.0)

    def test_alternative_same_as_word_is_skipped(self, checker, mock_provider):
        """Alternative identical to word is skipped."""
        mock_provider.get_bigram_probability.return_value = 0.01
        result = checker.compare_contextual_probability(
            word="word", alternatives=["word"], prev_words=["prev"], next_words=[]
        )
        assert result == (None, 0.0)

    def test_alternative_much_better_is_returned(self, checker, mock_provider):
        """Alternative with >= min_ratio probability is returned."""
        mock_provider.get_bigram_probability.side_effect = lambda w1, w2: {
            ("prev", "word"): 0.001,
            ("prev", "better"): 0.01,
        }.get((w1, w2), 0.0)
        mock_provider.get_word_frequency.return_value = 100

        result = checker.compare_contextual_probability(
            word="word",
            alternatives=["better"],
            prev_words=["prev"],
            next_words=[],
            min_ratio=5.0,
        )
        assert result[0] == "better"
        assert result[1] >= 5.0  # ratio = 0.01 / 0.001 = 10

    def test_alternative_not_enough_better_returns_none(self, checker, mock_provider):
        """Alternative below min_ratio returns None."""
        mock_provider.get_bigram_probability.side_effect = lambda w1, w2: {
            ("prev", "word"): 0.005,
            ("prev", "slightly_better"): 0.01,
        }.get((w1, w2), 0.0)
        mock_provider.get_word_frequency.return_value = 100

        result = checker.compare_contextual_probability(
            word="word",
            alternatives=["slightly_better"],
            prev_words=["prev"],
            next_words=[],
            min_ratio=5.0,
        )
        assert result == (None, 0.0)

    def test_high_freq_word_uses_stricter_ratio(self, checker, mock_provider):
        """High-frequency words use high_freq_min_ratio."""
        # word has freq >= 10000 (high_freq_threshold)
        mock_provider.get_word_frequency.side_effect = lambda w: {
            "common_word": 15000,
            "alt": 200,
        }.get(w, 0)
        mock_provider.get_bigram_probability.side_effect = lambda w1, w2: {
            ("prev", "common_word"): 0.001,
            ("prev", "alt"): 0.008,  # 8x better, but high_freq needs 10x
        }.get((w1, w2), 0.0)

        result = checker.compare_contextual_probability(
            word="common_word",
            alternatives=["alt"],
            prev_words=["prev"],
            next_words=[],
            min_ratio=5.0,
            high_freq_threshold=10000,
            high_freq_min_ratio=10.0,
        )
        assert result == (None, 0.0)

    def test_high_freq_word_passes_with_enough_ratio(self, checker, mock_provider):
        """High-frequency word passes when ratio exceeds high_freq_min_ratio."""
        mock_provider.get_word_frequency.side_effect = lambda w: {
            "common_word": 15000,
            "alt": 200,
        }.get(w, 0)
        mock_provider.get_bigram_probability.side_effect = lambda w1, w2: {
            ("prev", "common_word"): 0.001,
            ("prev", "alt"): 0.015,  # 15x better — exceeds 10x
        }.get((w1, w2), 0.0)

        result = checker.compare_contextual_probability(
            word="common_word",
            alternatives=["alt"],
            prev_words=["prev"],
            next_words=[],
            min_ratio=5.0,
            high_freq_threshold=10000,
            high_freq_min_ratio=10.0,
        )
        assert result[0] == "alt"
        assert result[1] >= 10.0

    def test_both_zero_prob_returns_none(self, checker, mock_provider):
        """Both word and alternative with zero prob returns None."""
        mock_provider.get_bigram_probability.return_value = 0.0
        mock_provider.get_word_frequency.return_value = 0

        result = checker.compare_contextual_probability(
            word="word", alternatives=["alt"], prev_words=["prev"], next_words=[]
        )
        assert result == (None, 0.0)

    def test_word_zero_alt_nonzero_returns_alt(self, checker, mock_provider):
        """Word with zero prob but alt with nonzero returns alt (infinite ratio)."""
        mock_provider.get_bigram_probability.side_effect = lambda w1, w2: {
            ("prev", "alt"): 0.005,
        }.get((w1, w2), 0.0)
        mock_provider.get_word_frequency.return_value = 0

        result = checker.compare_contextual_probability(
            word="word",
            alternatives=["alt"],
            prev_words=["prev"],
            next_words=[],
            min_ratio=5.0,
        )
        assert result[0] == "alt"

    def test_both_below_min_meaningful_prob_returns_none(self, checker, mock_provider):
        """When both word and alt have near-zero prob, returns None."""
        mock_provider.get_bigram_probability.side_effect = lambda w1, w2: {
            ("prev", "word"): 1e-10,
            ("prev", "alt"): 1e-9,
        }.get((w1, w2), 0.0)
        mock_provider.get_word_frequency.return_value = 0

        result = checker.compare_contextual_probability(
            word="word",
            alternatives=["alt"],
            prev_words=["prev"],
            next_words=[],
            min_meaningful_prob=1e-7,
        )
        assert result == (None, 0.0)

    def test_best_of_multiple_alternatives(self, checker, mock_provider):
        """Returns the best alternative among multiple candidates."""
        mock_provider.get_bigram_probability.side_effect = lambda w1, w2: {
            ("prev", "word"): 0.001,
            ("prev", "good"): 0.01,
            ("prev", "better"): 0.02,
            ("prev", "best"): 0.05,
        }.get((w1, w2), 0.0)
        mock_provider.get_word_frequency.return_value = 100

        result = checker.compare_contextual_probability(
            word="word",
            alternatives=["good", "better", "best"],
            prev_words=["prev"],
            next_words=[],
            min_ratio=5.0,
        )
        assert result[0] == "best"  # 50x ratio, highest
        assert result[1] >= 5.0

    def test_bidirectional_context_used(self, checker, mock_provider):
        """Both left and right context are used in comparison."""
        mock_provider.get_bigram_probability.side_effect = lambda w1, w2: {
            ("prev", "word"): 0.002,
            ("word", "next"): 0.002,
            ("prev", "alt"): 0.01,
            ("alt", "next"): 0.02,
        }.get((w1, w2), 0.0)
        mock_provider.get_word_frequency.return_value = 100

        result = checker.compare_contextual_probability(
            word="word",
            alternatives=["alt"],
            prev_words=["prev"],
            next_words=["next"],
            min_ratio=5.0,
        )
        # word prob = (0.002 + 0.002) / 2 = 0.002
        # alt prob = (0.01 + 0.02) / 2 = 0.015
        # ratio = 0.015 / 0.002 = 7.5 >= 5.0
        assert result[0] == "alt"
        assert result[1] == pytest.approx(7.5)
