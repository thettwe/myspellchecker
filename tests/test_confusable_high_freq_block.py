"""Unit tests for the curated/near-synonym high-frequency hard block (fph-03).

The curated path of ``ConfusableSemanticStrategy`` uses a flat low threshold
and — unlike the default path's ``high_freq_logit_diff`` bar — ignored word
frequency entirely. Context-dependent curated near-synonym pairs (ပေး↔ထား
"give↔place", သွား↔နေ "go↔stay") therefore fired on correct usage of
ultra-common verbs, producing clean-sentence false positives.

``curated_high_freq_hard_block_threshold`` (default 500_000) hard-blocks
curated and near-synonym flagging for words at/above that frequency, which are
virtually never real-word-confusion errors. These tests pin the gate logic via
``_find_best_variant`` directly so they need no MLM.
"""

from __future__ import annotations

from myspellchecker.core.config.algorithm_configs import ConfusableSemanticConfig
from myspellchecker.core.validation_strategies.confusable_semantic_strategy import (
    ConfusableSemanticStrategy,
)


class _Provider:
    """Minimal provider exposing only get_word_frequency."""

    def __init__(self, freq: int) -> None:
        self._freq = freq

    def get_word_frequency(self, word: str) -> int:
        return self._freq


def _strategy(freq: int, **cfg_kwargs) -> ConfusableSemanticStrategy:
    config = ConfusableSemanticConfig(**cfg_kwargs)
    return ConfusableSemanticStrategy(
        semantic_checker=None,  # not touched by _find_best_variant
        provider=_Provider(freq),
        config=config,
    )


def _best(strategy: ConfusableSemanticStrategy, word: str, variant: str, freq: int):
    """Run _find_best_variant on a single curated variant with no CMS context."""
    # logit_diff = variant_score - current_score = 5.0, well over the curated
    # threshold (2.0); empty prev/next words disable CMS reduction.
    return strategy._find_best_variant(
        word=word,
        valid_variants={variant},
        pred_map={word: 0.0, variant: 5.0},
        explicit_scores={word: 0.0, variant: 5.0},
        current_score=0.0,
        current_in_topk=True,
        is_high_freq=freq >= 50_000,
        word_freq=freq,
        is_sentence_final=False,
        is_boundary_occurrence=True,
        curated_variants={variant},
        prev_word="",
        next_word="",
    )


def test_curated_high_freq_word_is_hard_blocked() -> None:
    """ပေး (freq 1.05M ≥ 500k) curated→ထား is blocked despite a large logit diff."""
    strat = _strategy(freq=1_049_204)
    assert _best(strat, "ပေး", "ထား", 1_049_204) is None


def test_curated_mid_freq_word_still_flags() -> None:
    """A mid-frequency word (40k < 500k) is below the block and still flags."""
    strat = _strategy(freq=40_000)
    assert _best(strat, "ကင်ပွန်း", "ခင်ပွန်း", 40_000) == "ခင်ပွန်း"


def test_block_threshold_boundary_is_inclusive() -> None:
    """Frequency exactly at the threshold is blocked (>=)."""
    strat = _strategy(freq=500_000)
    assert _best(strat, "ပေး", "ထား", 500_000) is None


def test_block_disabled_when_threshold_zero() -> None:
    """Setting the threshold to 0 disables the block (legacy behaviour)."""
    strat = _strategy(freq=1_049_204, curated_high_freq_hard_block_threshold=0)
    assert _best(strat, "ပေး", "ထား", 1_049_204) == "ထား"


def test_near_synonym_high_freq_word_is_hard_blocked() -> None:
    """The same block applies to the near-synonym branch."""
    strat = _strategy(freq=1_049_204)
    result = strat._find_best_variant(
        word="ပေး",
        valid_variants={"ထား"},
        pred_map={"ပေး": 0.0, "ထား": 5.0},
        explicit_scores={"ပေး": 0.0, "ထား": 5.0},
        current_score=0.0,
        current_in_topk=True,
        is_high_freq=True,
        word_freq=1_049_204,
        is_sentence_final=False,
        is_boundary_occurrence=True,
        near_synonym_variants={"ထား"},
        prev_word="",
        next_word="",
    )
    assert result is None
