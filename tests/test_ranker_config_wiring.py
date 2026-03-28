"""Tests for RankerConfig calibration fields and suggestion-builder wiring."""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from myspellchecker.core.config import NgramContextConfig, RankerConfig, SpellCheckerConfig
from myspellchecker.core.factories.builders import build_suggestion_strategy


def test_ranker_config_new_defaults():
    """New ranker calibration defaults should be stable."""
    config = RankerConfig()
    assert config.context_strategy_score_weight == 0.7
    assert config.strategy_score_cap == 10.0
    assert config.enable_targeted_rerank_hints is True
    assert config.enable_targeted_candidate_injections is True
    assert config.enable_targeted_grammar_completion_templates is True


def test_ranker_config_context_strategy_weight_validation():
    """context_strategy_score_weight must stay within [0.0, 1.0]."""
    with pytest.raises(ValidationError):
        RankerConfig(context_strategy_score_weight=1.1)
    with pytest.raises(ValidationError):
        RankerConfig(context_strategy_score_weight=-0.1)


def test_ranker_config_strategy_score_cap_validation():
    """strategy_score_cap must be strictly positive."""
    with pytest.raises(ValidationError):
        RankerConfig(strategy_score_cap=0.0)


def test_builder_propagates_rerank_weights_to_context_strategy():
    """build_suggestion_strategy should pass configured rerank weights."""
    config = SpellCheckerConfig(
        use_context_checker=True,
        ngram_context=NgramContextConfig(
            rerank_left_weight=0.77,
            rerank_right_weight=0.22,
        ),
    )

    symspell = MagicMock()
    provider = MagicMock()
    provider.is_valid_word.return_value = True
    provider.is_valid_syllable.return_value = True
    context_checker = MagicMock()

    with patch(
        "myspellchecker.algorithms.suggestion_strategy.ContextSuggestionStrategy"
    ) as mock_ctx:
        mock_ctx.return_value = MagicMock()
        build_suggestion_strategy(
            symspell=symspell,
            provider=provider,
            config=config,
            context_checker=context_checker,
        )

        mock_ctx.assert_called_once()
        _, kwargs = mock_ctx.call_args
        assert kwargs["left_weight"] == 0.77
        assert kwargs["right_weight"] == 0.22
