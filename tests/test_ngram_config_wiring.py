"""Tests for NgramContextConfig field defaults and builder wiring.

Verifies that:
1. New config field defaults match NgramContextChecker constructor defaults exactly.
2. Removed fields (context_window_size, adaptive_window) raise ValidationError.
3. Config values propagate through build_ngram_context_checker() to the checker instance.
4. smoothing_strategy string is converted to SmoothingStrategy enum correctly.
"""

from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from myspellchecker.algorithms.ngram_context_checker import (
    NgramContextChecker,
    SmoothingStrategy,
)
from myspellchecker.core.config import NgramContextConfig, SpellCheckerConfig
from myspellchecker.core.factories.builders import build_ngram_context_checker

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_provider():
    """Create a mock provider with required methods."""
    provider = MagicMock()
    provider.get_pos_unigram_probabilities.return_value = {}
    provider.get_pos_bigram_probabilities.return_value = {}
    provider.get_metadata.return_value = None
    return provider


# ---------------------------------------------------------------------------
# 1. Config defaults match constructor defaults
# ---------------------------------------------------------------------------


class TestConfigDefaultsMatchConstructor:
    """New NgramContextConfig field defaults must match NgramContextChecker constructor defaults."""

    def test_use_smoothing_default(self):
        config = NgramContextConfig()
        assert config.use_smoothing is True

    def test_smoothing_strategy_default(self):
        config = NgramContextConfig()
        assert config.smoothing_strategy == "stupid_backoff"

    def test_backoff_weight_default(self):
        config = NgramContextConfig()
        assert config.backoff_weight == 0.4

    def test_add_k_smoothing_default(self):
        config = NgramContextConfig()
        assert config.add_k_smoothing == 0.0

    def test_unigram_denominator_default(self):
        config = NgramContextConfig()
        assert config.unigram_denominator == 500_000_000.0

    def test_unigram_prob_cap_default(self):
        config = NgramContextConfig()
        assert config.unigram_prob_cap == 0.1

    def test_min_unigram_threshold_default(self):
        config = NgramContextConfig()
        assert config.min_unigram_threshold == 5

    def test_right_context_threshold_default(self):
        config = NgramContextConfig()
        assert config.right_context_threshold == 0.001


# ---------------------------------------------------------------------------
# 2. Removed fields raise ValidationError
# ---------------------------------------------------------------------------


class TestRemovedFieldsRejected:
    """Fields removed from NgramContextConfig must raise ValidationError (extra='forbid')."""

    def test_context_window_size_rejected(self):
        with pytest.raises(ValidationError, match="context_window_size"):
            NgramContextConfig(context_window_size=2)

    def test_adaptive_window_rejected(self):
        with pytest.raises(ValidationError, match="adaptive_window"):
            NgramContextConfig(adaptive_window=True)


# ---------------------------------------------------------------------------
# 3. Config → builder → checker wiring
# ---------------------------------------------------------------------------


class TestBuilderWiring:
    """Config values must propagate through build_ngram_context_checker() to the checker."""

    def _build_checker(self, **ngram_overrides) -> NgramContextChecker:
        """Helper to build a checker with custom ngram config."""
        ngram_config = NgramContextConfig(**ngram_overrides)
        config = SpellCheckerConfig(
            use_context_checker=True,
            ngram_context=ngram_config,
        )
        provider = _make_mock_provider()
        checker = build_ngram_context_checker(provider, config)
        assert checker is not None
        return checker

    def test_backoff_weight_wired(self):
        checker = self._build_checker(backoff_weight=0.6)
        assert checker.backoff_weight == 0.6

    def test_use_smoothing_wired(self):
        checker = self._build_checker(use_smoothing=False)
        assert checker.use_smoothing is False

    def test_smoothing_strategy_wired(self):
        checker = self._build_checker(smoothing_strategy="add_k")
        assert checker.smoothing_strategy == SmoothingStrategy.ADD_K

    def test_add_k_smoothing_wired(self):
        checker = self._build_checker(add_k_smoothing=0.05)
        assert checker.add_k_smoothing == 0.05

    def test_unigram_denominator_wired(self):
        checker = self._build_checker(unigram_denominator=500000.0)
        # The checker may use a dynamic denominator from provider metadata,
        # but since our mock returns None, it should use the config value.
        assert checker.unigram_denominator == 500000.0

    def test_unigram_prob_cap_wired(self):
        checker = self._build_checker(unigram_prob_cap=0.05)
        assert checker.unigram_prob_cap == 0.05

    def test_min_unigram_threshold_wired(self):
        checker = self._build_checker(min_unigram_threshold=10)
        assert checker.min_unigram_threshold == 10

    def test_right_context_threshold_wired(self):
        checker = self._build_checker(right_context_threshold=0.005)
        assert checker.right_context_threshold == 0.005

    def test_backoff_floor_multiplier_wired(self):
        checker = self._build_checker(backoff_floor_multiplier=0.2)
        assert checker.backoff_floor_multiplier == 0.2

    def test_default_checker_has_correct_smoothing_strategy(self):
        """Default config should produce STUPID_BACKOFF strategy."""
        checker = self._build_checker()
        assert checker.smoothing_strategy == SmoothingStrategy.STUPID_BACKOFF


# ---------------------------------------------------------------------------
# 4. smoothing_strategy string → SmoothingStrategy enum conversion
# ---------------------------------------------------------------------------


class TestSmoothingStrategyConversion:
    """smoothing_strategy string in config must be converted to SmoothingStrategy enum."""

    @pytest.mark.parametrize(
        "strategy_str,expected_enum",
        [
            ("none", SmoothingStrategy.NONE),
            ("stupid_backoff", SmoothingStrategy.STUPID_BACKOFF),
            ("add_k", SmoothingStrategy.ADD_K),
        ],
    )
    def test_string_to_enum(self, strategy_str, expected_enum):
        config = NgramContextConfig(smoothing_strategy=strategy_str)
        provider = _make_mock_provider()
        full_config = SpellCheckerConfig(
            use_context_checker=True,
            ngram_context=config,
        )
        checker = build_ngram_context_checker(provider, full_config)
        assert checker is not None
        assert checker.smoothing_strategy == expected_enum

    def test_invalid_strategy_raises(self):
        """Invalid smoothing_strategy should raise ValueError during SmoothingStrategy()."""
        config = NgramContextConfig()
        # Override after creation to bypass pydantic validation
        object.__setattr__(config, "smoothing_strategy", "invalid_strategy")
        provider = _make_mock_provider()
        full_config = SpellCheckerConfig(
            use_context_checker=True,
            ngram_context=config,
        )
        with pytest.raises(ValueError, match="invalid_strategy"):
            build_ngram_context_checker(provider, full_config)
