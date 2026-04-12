"""Tests for SyllableWindowOOVStrategy."""

from __future__ import annotations

import pytest

from myspellchecker.algorithms.symspell import SymSpell
from myspellchecker.core.config import SpellCheckerConfig
from myspellchecker.core.constants import ET_SYLLABLE_WINDOW_OOV, ErrorType
from myspellchecker.core.factories.builders import build_context_validation_strategies
from myspellchecker.core.validation_strategies import SyllableWindowOOVStrategy
from myspellchecker.core.validation_strategies.base import ValidationContext
from myspellchecker.providers.memory import MemoryProvider


@pytest.fixture
def hidden_compound_provider() -> MemoryProvider:
    """Provider seeded with a hidden-compound example: ခုန်ကျစရိတ် → ကုန်ကျစရိတ်."""
    provider = MemoryProvider()
    provider.add_word("ခုန်", frequency=9032)
    provider.add_word("ကျ", frequency=100)
    provider.add_word("စရိတ်", frequency=13958)
    provider.add_word("ကုန်", frequency=80914)
    provider.add_word("ကုန်ကျစရိတ်", frequency=27677)
    return provider


@pytest.fixture
def hidden_compound_symspell(hidden_compound_provider: MemoryProvider) -> SymSpell:
    ss = SymSpell(
        hidden_compound_provider,
        max_edit_distance=2,
        prefix_length=7,
        count_threshold=50,
    )
    ss.build_index(["word"])
    return ss


class TestSyllableWindowOOVScaffold:
    def test_error_type_constant_exists(self) -> None:
        assert ET_SYLLABLE_WINDOW_OOV == "syllable_window_oov"
        assert ErrorType.SYLLABLE_WINDOW_OOV.value == "syllable_window_oov"

    def test_strategy_priority_is_22(self, hidden_compound_provider: MemoryProvider) -> None:
        s = SyllableWindowOOVStrategy(provider=hidden_compound_provider, symspell=None)
        assert s.priority() == 22

    def test_strategy_instantiates_with_defaults(
        self, hidden_compound_provider: MemoryProvider
    ) -> None:
        s = SyllableWindowOOVStrategy(provider=hidden_compound_provider, symspell=None)
        assert s.enabled is True
        assert s.window_sizes == (2, 3, 4)
        assert s.min_frequency == 50
        assert s.confidence_floor == 0.70

    def test_repr_contains_priority(self, hidden_compound_provider: MemoryProvider) -> None:
        s = SyllableWindowOOVStrategy(provider=hidden_compound_provider, symspell=None)
        text = repr(s)
        assert "SyllableWindowOOVStrategy" in text
        assert "priority=22" in text


class TestSyllableWindowOOVDetection:
    def test_detects_hidden_compound_typo(
        self,
        hidden_compound_provider: MemoryProvider,
        hidden_compound_symspell: SymSpell,
    ) -> None:
        """``ခုန်ကျစရိတ်`` segmented as ``['ခုန်', 'ကျ', 'စရိတ်']`` is each valid
        individually but joined forms an OOV string whose SymSpell near-match
        is the high-frequency ``ကုန်ကျစရိတ်``."""
        s = SyllableWindowOOVStrategy(
            provider=hidden_compound_provider,
            symspell=hidden_compound_symspell,
            enabled=True,
        )
        ctx = ValidationContext(
            sentence="ခုန်ကျစရိတ်",
            words=["ခုန်", "ကျ", "စရိတ်"],
            word_positions=[0, 4, 6],
            full_text="ခုန်ကျစရိတ်",
        )
        errors = s.validate(ctx)
        assert len(errors) >= 1
        err = errors[0]
        assert err.error_type == ET_SYLLABLE_WINDOW_OOV
        assert err.position == 0
        assert any("ကုန်ကျစရိတ်" in str(s) for s in err.suggestions)
        assert err.confidence >= 0.70

    def test_returns_empty_when_disabled(
        self, hidden_compound_provider: MemoryProvider, hidden_compound_symspell: SymSpell
    ) -> None:
        s = SyllableWindowOOVStrategy(
            provider=hidden_compound_provider, symspell=hidden_compound_symspell, enabled=False
        )
        ctx = ValidationContext(
            sentence="ခုန်ကျစရိတ်",
            words=["ခုန်", "ကျ", "စရိတ်"],
            word_positions=[0, 4, 6],
            full_text="ခုန်ကျစရိတ်",
        )
        assert s.validate(ctx) == []

    def test_returns_empty_when_symspell_is_none(
        self, hidden_compound_provider: MemoryProvider
    ) -> None:
        s = SyllableWindowOOVStrategy(
            provider=hidden_compound_provider, symspell=None, enabled=True
        )
        ctx = ValidationContext(
            sentence="ခုန်ကျစရိတ်",
            words=["ခုန်", "ကျ", "စရိတ်"],
            word_positions=[0, 4, 6],
            full_text="ခုန်ကျစရိတ်",
        )
        assert s.validate(ctx) == []

    def test_returns_empty_on_empty_context(
        self, hidden_compound_provider: MemoryProvider, hidden_compound_symspell: SymSpell
    ) -> None:
        s = SyllableWindowOOVStrategy(
            provider=hidden_compound_provider, symspell=hidden_compound_symspell, enabled=True
        )
        ctx = ValidationContext(sentence="", words=[], word_positions=[])
        assert s.validate(ctx) == []

    def test_skips_valid_compound(
        self, hidden_compound_provider: MemoryProvider, hidden_compound_symspell: SymSpell
    ) -> None:
        """A single valid word covering the whole span produces no error."""
        s = SyllableWindowOOVStrategy(
            provider=hidden_compound_provider, symspell=hidden_compound_symspell, enabled=True
        )
        ctx = ValidationContext(
            sentence="ကုန်ကျစရိတ်",
            words=["ကုန်ကျစရိတ်"],
            word_positions=[0],
            full_text="ကုန်ကျစရိတ်",
        )
        assert s.validate(ctx) == []

    def test_skips_existing_errors_at_position(
        self, hidden_compound_provider: MemoryProvider, hidden_compound_symspell: SymSpell
    ) -> None:
        """If upstream already flagged a position, the strategy skips it."""
        s = SyllableWindowOOVStrategy(
            provider=hidden_compound_provider, symspell=hidden_compound_symspell, enabled=True
        )
        ctx = ValidationContext(
            sentence="ခုန်ကျစရိတ်",
            words=["ခုန်", "ကျ", "စရိတ်"],
            word_positions=[0, 4, 6],
            full_text="ခုန်ကျစရိတ်",
            existing_errors={0: "some_upstream_error"},
        )
        errors = s.validate(ctx)
        assert all(e.position != 0 for e in errors)


class TestSyllableWindowFPRSafeguards:
    def test_skip_names_avoids_proper_noun_false_positive(
        self, hidden_compound_provider: MemoryProvider, hidden_compound_symspell: SymSpell
    ) -> None:
        """Windows overlapping name-masked words should not fire."""
        s = SyllableWindowOOVStrategy(
            provider=hidden_compound_provider,
            symspell=hidden_compound_symspell,
            enabled=True,
            skip_names=True,
        )
        ctx = ValidationContext(
            sentence="ခုန်ကျစရိတ်",
            words=["ခုန်", "ကျ", "စရိတ်"],
            word_positions=[0, 4, 6],
            is_name_mask=[True, False, False],
            full_text="ခုန်ကျစရိတ်",
        )
        assert s.validate(ctx) == []

    def test_require_valid_source_words_skips_unknown_segments(
        self, hidden_compound_symspell: SymSpell
    ) -> None:
        """Windows that span an OOV source word should be skipped."""
        # Provider knows the compound but NOT all the source words.
        provider = MemoryProvider()
        provider.add_word("ကျ", frequency=100)
        provider.add_word("စရိတ်", frequency=13958)
        provider.add_word("ကုန်ကျစရိတ်", frequency=27677)

        ss = SymSpell(provider, max_edit_distance=2, count_threshold=50)
        ss.build_index(["word"])

        s = SyllableWindowOOVStrategy(
            provider=provider,
            symspell=ss,
            enabled=True,
            require_valid_source_words=True,
        )
        ctx = ValidationContext(
            sentence="ခုန်ကျစရိတ်",
            words=["ခုန်", "ကျ", "စရိတ်"],
            word_positions=[0, 4, 6],
            full_text="ခုန်ကျစရိတ်",
        )
        assert s.validate(ctx) == []

    def test_require_typo_prone_char(
        self, hidden_compound_provider: MemoryProvider, hidden_compound_symspell: SymSpell
    ) -> None:
        """Non-Myanmar tokens never contain a typo-prone character."""
        s = SyllableWindowOOVStrategy(
            provider=hidden_compound_provider,
            symspell=hidden_compound_symspell,
            enabled=True,
            require_typo_prone=True,
        )
        ctx = ValidationContext(
            sentence="ABC",
            words=["ABC"],
            word_positions=[0],
            full_text="ABC",
        )
        assert s.validate(ctx) == []

    def test_confidence_floor_rejects_weak_matches(
        self, hidden_compound_provider: MemoryProvider, hidden_compound_symspell: SymSpell
    ) -> None:
        """A floor near 1.0 suppresses every emission."""
        s = SyllableWindowOOVStrategy(
            provider=hidden_compound_provider,
            symspell=hidden_compound_symspell,
            enabled=True,
            confidence_floor=0.999,
        )
        ctx = ValidationContext(
            sentence="ခုန်ကျစရိတ်",
            words=["ခုန်", "ကျ", "စရိတ်"],
            word_positions=[0, 4, 6],
            full_text="ခုန်ကျစရိတ်",
        )
        assert s.validate(ctx) == []


class TestSyllableWindowRegistration:
    def test_strategy_not_registered_by_default(
        self, hidden_compound_provider: MemoryProvider, hidden_compound_symspell: SymSpell
    ) -> None:
        """The strategy is disabled by default and is not added to the chain."""
        config = SpellCheckerConfig()
        assert config.validation.use_syllable_window_oov is False
        strategies = build_context_validation_strategies(
            config=config,
            provider=hidden_compound_provider,
            symspell=hidden_compound_symspell,
        )
        names = [s.__class__.__name__ for s in strategies]
        assert "SyllableWindowOOVStrategy" not in names

    def test_strategy_registered_when_explicitly_enabled(
        self, hidden_compound_provider: MemoryProvider, hidden_compound_symspell: SymSpell
    ) -> None:
        config = SpellCheckerConfig()
        config.validation.use_syllable_window_oov = True
        strategies = build_context_validation_strategies(
            config=config,
            provider=hidden_compound_provider,
            symspell=hidden_compound_symspell,
        )
        names = [s.__class__.__name__ for s in strategies]
        assert "SyllableWindowOOVStrategy" in names

    def test_strategy_not_registered_without_symspell(
        self, hidden_compound_provider: MemoryProvider
    ) -> None:
        config = SpellCheckerConfig()
        strategies = build_context_validation_strategies(
            config=config, provider=hidden_compound_provider, symspell=None
        )
        names = [s.__class__.__name__ for s in strategies]
        assert "SyllableWindowOOVStrategy" not in names

    def test_strategy_disabled_via_config(
        self, hidden_compound_provider: MemoryProvider, hidden_compound_symspell: SymSpell
    ) -> None:
        config = SpellCheckerConfig()
        config.validation.use_syllable_window_oov = False
        strategies = build_context_validation_strategies(
            config=config,
            provider=hidden_compound_provider,
            symspell=hidden_compound_symspell,
        )
        names = [s.__class__.__name__ for s in strategies]
        assert "SyllableWindowOOVStrategy" not in names

    def test_priority_runs_before_hidden_compound(
        self, hidden_compound_provider: MemoryProvider, hidden_compound_symspell: SymSpell
    ) -> None:
        config = SpellCheckerConfig()
        config.validation.use_syllable_window_oov = True
        strategies = build_context_validation_strategies(
            config=config,
            provider=hidden_compound_provider,
            symspell=hidden_compound_symspell,
        )
        strategies = sorted(strategies, key=lambda s: s.priority())
        names_by_priority = [(s.priority(), s.__class__.__name__) for s in strategies]
        sw_idx = next(
            i for i, (_, n) in enumerate(names_by_priority) if n == "SyllableWindowOOVStrategy"
        )
        hc_idx = next(
            (i for i, (_, n) in enumerate(names_by_priority) if n == "HiddenCompoundStrategy"),
            None,
        )
        if hc_idx is not None:
            assert sw_idx < hc_idx
