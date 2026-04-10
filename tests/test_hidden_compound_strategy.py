"""Tests for HiddenCompoundStrategy (Sprint A scaffold).

Sprint A verifies only the strategy loads, registers, and no-ops.
Detection logic tests land in Sprint B.

See Workstreams/v1.5.0/hidden-compound-typo-plan.md for the full plan.
"""

from __future__ import annotations

import pytest

from myspellchecker.core.config import SpellCheckerConfig
from myspellchecker.core.constants import ET_HIDDEN_COMPOUND_TYPO, ErrorType
from myspellchecker.core.factories.builders import build_context_validation_strategies
from myspellchecker.core.validation_strategies import HiddenCompoundStrategy
from myspellchecker.core.validation_strategies.base import ValidationContext
from myspellchecker.providers.memory import MemoryProvider


@pytest.fixture
def memory_provider() -> MemoryProvider:
    """Build a small in-memory provider with the canonical BM-005 vocabulary."""
    provider = MemoryProvider()
    provider.add_word("ခုန်", frequency=9032)
    provider.add_word("ကုန်", frequency=80914)
    provider.add_word("ကျ", frequency=0)
    provider.add_word("စရိတ်", frequency=13958)
    # Valid but freq=0 dict entry — the subsumed-compound case
    provider.add_word("ကုန်ကျ", frequency=0)
    # The correct compound
    provider.add_word("ကုန်ကျစရိတ်", frequency=27677)
    return provider


class TestHiddenCompoundStrategyScaffold:
    """Sprint A: minimal scaffold. No detection logic yet."""

    def test_error_type_constant_exists(self) -> None:
        assert ET_HIDDEN_COMPOUND_TYPO == "hidden_compound_typo"
        assert ErrorType.HIDDEN_COMPOUND_TYPO.value == "hidden_compound_typo"

    def test_strategy_instantiates_with_defaults(self, memory_provider: MemoryProvider) -> None:
        strategy = HiddenCompoundStrategy(provider=memory_provider, hasher=None)
        assert strategy.priority() == 23
        assert strategy.enabled is False
        assert strategy.max_token_syllables == 3
        assert strategy.confidence_floor == 0.75

    def test_strategy_instantiates_with_config_overrides(
        self, memory_provider: MemoryProvider
    ) -> None:
        strategy = HiddenCompoundStrategy(
            provider=memory_provider,
            hasher=None,
            enabled=True,
            max_token_syllables=4,
            confidence_floor=0.85,
            compound_min_frequency=50,
        )
        assert strategy.enabled is True
        assert strategy.max_token_syllables == 4
        assert strategy.confidence_floor == 0.85
        assert strategy.compound_min_frequency == 50

    def test_validate_returns_empty_when_disabled(self, memory_provider: MemoryProvider) -> None:
        """Disabled strategy must return [] even with a well-formed context."""
        strategy = HiddenCompoundStrategy(provider=memory_provider, hasher=None, enabled=False)
        ctx = ValidationContext(
            sentence="ခုန်ကျစရိတ်",
            words=["ခုန်", "ကျ", "စရိတ်"],
            word_positions=[0, 4, 6],
        )
        assert strategy.validate(ctx) == []

    def test_validate_returns_empty_when_enabled_sprint_a_scaffold(
        self, memory_provider: MemoryProvider
    ) -> None:
        """Sprint A scaffold: enabled strategy still no-ops (detection comes in Sprint B)."""
        strategy = HiddenCompoundStrategy(provider=memory_provider, hasher=None, enabled=True)
        ctx = ValidationContext(
            sentence="ခုန်ကျစရိတ်",
            words=["ခုန်", "ကျ", "စရိတ်"],
            word_positions=[0, 4, 6],
        )
        assert strategy.validate(ctx) == []

    def test_validate_empty_context(self, memory_provider: MemoryProvider) -> None:
        strategy = HiddenCompoundStrategy(provider=memory_provider, hasher=None, enabled=True)
        ctx = ValidationContext(sentence="", words=[], word_positions=[])
        assert strategy.validate(ctx) == []

    def test_repr_contains_priority_and_enabled(self, memory_provider: MemoryProvider) -> None:
        strategy = HiddenCompoundStrategy(provider=memory_provider, hasher=None, enabled=True)
        text = repr(strategy)
        assert "HiddenCompoundStrategy" in text
        assert "priority=23" in text
        assert "enabled=True" in text


class TestHiddenCompoundStrategyRegistration:
    """Verify factory registration under config flag."""

    def test_strategy_not_registered_by_default(self, memory_provider: MemoryProvider) -> None:
        config = SpellCheckerConfig()
        assert config.validation.use_hidden_compound_detection is False
        strategies = build_context_validation_strategies(config=config, provider=memory_provider)
        names = [s.__class__.__name__ for s in strategies]
        assert "HiddenCompoundStrategy" not in names

    def test_strategy_registered_when_flag_enabled(self, memory_provider: MemoryProvider) -> None:
        config = SpellCheckerConfig()
        config.validation.use_hidden_compound_detection = True
        strategies = build_context_validation_strategies(config=config, provider=memory_provider)
        hidden = [s for s in strategies if s.__class__.__name__ == "HiddenCompoundStrategy"]
        assert len(hidden) == 1
        assert hidden[0].priority() == 23
        assert hidden[0].enabled is True

    def test_strategy_placed_before_statistical_confusable_and_broken_compound(
        self, memory_provider: MemoryProvider
    ) -> None:
        config = SpellCheckerConfig()
        config.validation.use_hidden_compound_detection = True
        strategies = build_context_validation_strategies(config=config, provider=memory_provider)
        # Sort by priority — what the ContextValidator will do at execution
        ordered = sorted(strategies, key=lambda s: s.priority())
        names_by_priority = [(s.priority(), s.__class__.__name__) for s in ordered]
        hidden_idx = next(
            i for i, (_, n) in enumerate(names_by_priority) if n == "HiddenCompoundStrategy"
        )
        following_priorities = {n for _, n in names_by_priority[hidden_idx + 1 :]}
        # These must appear after HiddenCompound in priority order
        if any(n == "StatisticalConfusableStrategy" for _, n in names_by_priority):
            assert "StatisticalConfusableStrategy" in following_priorities
        if any(n == "BrokenCompoundStrategy" for _, n in names_by_priority):
            assert "BrokenCompoundStrategy" in following_priorities


class TestProviderVocabularyDefault:
    """Sprint A.2: verify the provider base default for is_valid_vocabulary."""

    def test_memory_provider_inherits_default(self) -> None:
        provider = MemoryProvider()
        provider.add_word("မြန်မာ", frequency=1000)
        # Default falls back to is_valid_word
        assert provider.is_valid_vocabulary("မြန်မာ") is True
        assert provider.is_valid_vocabulary("nonexistent") is False

    def test_memory_provider_bulk_default(self) -> None:
        provider = MemoryProvider()
        provider.add_word("က", frequency=1)
        provider.add_word("ခ", frequency=1)
        result = provider.is_valid_vocabulary_bulk(["က", "ခ", "xyz"])
        assert result == {"က": True, "ခ": True, "xyz": False}
