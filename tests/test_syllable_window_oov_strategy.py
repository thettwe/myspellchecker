"""Tests for SyllableWindowOOVStrategy (Sprint I-1).

Target FNs from `scripts/syllable_window_oracle.py`: multi-syllable typos
the segmenter decomposes into valid syllables (e.g. BM-100, BM-326, BM-375,
BM-376). The strategy recovers ~34/185 clean-sentence FNs.

See Workstreams/v1.5.0/sprint-i-1-syllable-window-detector.md for the plan.
"""

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
def bm005_provider() -> MemoryProvider:
    """Provider seeded with BM-005-style compound: ခုန်ကျစရိတ် → ကုန်ကျစရိတ်."""
    provider = MemoryProvider()
    # Individually-valid syllables (segmenter sees these as valid words)
    provider.add_word("ခုန်", frequency=9032)
    provider.add_word("ကျ", frequency=100)
    provider.add_word("စရိတ်", frequency=13958)
    provider.add_word("ကုန်", frequency=80914)
    # The correct compound
    provider.add_word("ကုန်ကျစရိတ်", frequency=27677)
    return provider


@pytest.fixture
def bm005_symspell(bm005_provider: MemoryProvider) -> SymSpell:
    ss = SymSpell(
        bm005_provider,
        max_edit_distance=2,
        prefix_length=7,
        count_threshold=50,
    )
    ss.build_index(["word"])
    return ss


class TestSyllableWindowOOVScaffold:
    """Basic strategy construction and priority."""

    def test_error_type_constant_exists(self) -> None:
        assert ET_SYLLABLE_WINDOW_OOV == "syllable_window_oov"
        assert ErrorType.SYLLABLE_WINDOW_OOV.value == "syllable_window_oov"

    def test_strategy_priority_is_22(self, bm005_provider: MemoryProvider) -> None:
        s = SyllableWindowOOVStrategy(provider=bm005_provider, symspell=None)
        assert s.priority() == 22

    def test_strategy_instantiates_with_defaults(
        self, bm005_provider: MemoryProvider
    ) -> None:
        s = SyllableWindowOOVStrategy(provider=bm005_provider, symspell=None)
        assert s.enabled is True
        assert s.window_sizes == (2, 3, 4)
        assert s.min_frequency == 50
        assert s.confidence_floor == 0.70

    def test_repr_contains_priority(self, bm005_provider: MemoryProvider) -> None:
        s = SyllableWindowOOVStrategy(provider=bm005_provider, symspell=None)
        text = repr(s)
        assert "SyllableWindowOOVStrategy" in text
        assert "priority=22" in text


class TestSyllableWindowOOVDetection:
    """Detection on canonical hidden-compound examples."""

    def test_bm005_canonical_detects_compound_typo(
        self, bm005_provider: MemoryProvider, bm005_symspell: SymSpell
    ) -> None:
        """Canonical Sprint I-1 example:

        Input ``ခုန်ကျစရိတ်`` is segmented as ``['ခုန်', 'ကျ', 'စရိတ်']``.
        Each token is valid individually, but joined they form an OOV string
        whose SymSpell near-match is the high-frequency ``ကုန်ကျစရိတ်``.
        """
        s = SyllableWindowOOVStrategy(
            provider=bm005_provider,
            symspell=bm005_symspell,
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
        self, bm005_provider: MemoryProvider, bm005_symspell: SymSpell
    ) -> None:
        s = SyllableWindowOOVStrategy(
            provider=bm005_provider, symspell=bm005_symspell, enabled=False
        )
        ctx = ValidationContext(
            sentence="ခုန်ကျစရိတ်",
            words=["ခုန်", "ကျ", "စရိတ်"],
            word_positions=[0, 4, 6],
            full_text="ခုန်ကျစရိတ်",
        )
        assert s.validate(ctx) == []

    def test_returns_empty_when_symspell_is_none(
        self, bm005_provider: MemoryProvider
    ) -> None:
        s = SyllableWindowOOVStrategy(
            provider=bm005_provider, symspell=None, enabled=True
        )
        ctx = ValidationContext(
            sentence="ခုန်ကျစရိတ်",
            words=["ခုန်", "ကျ", "စရိတ်"],
            word_positions=[0, 4, 6],
            full_text="ခုန်ကျစရိတ်",
        )
        assert s.validate(ctx) == []

    def test_returns_empty_on_empty_context(
        self, bm005_provider: MemoryProvider, bm005_symspell: SymSpell
    ) -> None:
        s = SyllableWindowOOVStrategy(
            provider=bm005_provider, symspell=bm005_symspell, enabled=True
        )
        ctx = ValidationContext(sentence="", words=[], word_positions=[])
        assert s.validate(ctx) == []

    def test_skips_valid_compound(
        self, bm005_provider: MemoryProvider, bm005_symspell: SymSpell
    ) -> None:
        """If the joined window is a valid dictionary word, nothing fires."""
        s = SyllableWindowOOVStrategy(
            provider=bm005_provider, symspell=bm005_symspell, enabled=True
        )
        ctx = ValidationContext(
            sentence="ကုန်ကျစရိတ်",
            words=["ကုန်ကျစရိတ်"],
            word_positions=[0],
            full_text="ကုန်ကျစရိတ်",
        )
        # Single word that's valid and is itself the target compound — no
        # window should fire.
        assert s.validate(ctx) == []

    def test_skips_existing_errors_at_position(
        self, bm005_provider: MemoryProvider, bm005_symspell: SymSpell
    ) -> None:
        """If upstream already flagged pos 0, the strategy skips."""
        s = SyllableWindowOOVStrategy(
            provider=bm005_provider, symspell=bm005_symspell, enabled=True
        )
        ctx = ValidationContext(
            sentence="ခုန်ကျစရိတ်",
            words=["ခုန်", "ကျ", "စရိတ်"],
            word_positions=[0, 4, 6],
            full_text="ခုန်ကျစရိတ်",
            existing_errors={0: "some_upstream_error"},
        )
        errors = s.validate(ctx)
        # Position 0 should be skipped — may still fire at other offsets.
        assert all(e.position != 0 for e in errors)


class TestSyllableWindowFPRSafeguards:
    """Debate-gate-driven FPR mitigations."""

    def test_skip_names_avoids_proper_noun_false_positive(
        self, bm005_provider: MemoryProvider, bm005_symspell: SymSpell
    ) -> None:
        """Windows overlapping name-masked words should not fire.

        This covers the top FPR risk (proper nouns) surfaced in the debate
        gate by codex+gemini.
        """
        s = SyllableWindowOOVStrategy(
            provider=bm005_provider,
            symspell=bm005_symspell,
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
        errors = s.validate(ctx)
        # The only recoverable window starts at position 0, which overlaps a
        # name-masked word. Expect no emission.
        assert errors == []

    def test_require_valid_source_words_skips_unknown_segments(
        self, bm005_symspell: SymSpell
    ) -> None:
        """When a source word is NOT individually valid, skip.

        This avoids doubling up on sentences where upstream syllable/word
        validators should have already caught the error.
        """
        # Build a provider that knows compound but NOT all the source words.
        provider = MemoryProvider()
        provider.add_word("ကျ", frequency=100)
        provider.add_word("စရိတ်", frequency=13958)
        provider.add_word("ကုန်ကျစရိတ်", frequency=27677)
        # Note: ခုန် is NOT in provider

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
        # ခုန် is invalid → the require_valid_source_words guard should
        # skip any window that spans it.
        assert s.validate(ctx) == []

    def test_require_typo_prone_char(
        self, bm005_provider: MemoryProvider, bm005_symspell: SymSpell
    ) -> None:
        """If the strict typo-prone filter is on, non-Myanmar tokens skip."""
        s = SyllableWindowOOVStrategy(
            provider=bm005_provider,
            symspell=bm005_symspell,
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
        self, bm005_provider: MemoryProvider, bm005_symspell: SymSpell
    ) -> None:
        """An impossibly high floor suppresses all emissions."""
        s = SyllableWindowOOVStrategy(
            provider=bm005_provider,
            symspell=bm005_symspell,
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
    """Verify factory registration under config flag."""

    def test_strategy_disabled_by_default_in_v140(
        self, bm005_provider: MemoryProvider, bm005_symspell: SymSpell
    ) -> None:
        """v1.4.0 default: SW is DISABLED. Implementation kept for future sprint.

        Sprint I-1.5 confirmed: even with aggressive gating, SW's empirical
        recall on the production benchmark is neutral and the per-window
        SymSpell scan adds ~90ms p95 latency. Implementation is preserved
        and re-enable is a one-line config flip when a future sprint
        addresses the precision/latency tradeoff.
        """
        config = SpellCheckerConfig()
        assert config.validation.use_syllable_window_oov is False
        strategies = build_context_validation_strategies(
            config=config, provider=bm005_provider, symspell=bm005_symspell
        )
        names = [s.__class__.__name__ for s in strategies]
        assert "SyllableWindowOOVStrategy" not in names

    def test_strategy_registered_when_explicitly_enabled(
        self, bm005_provider: MemoryProvider, bm005_symspell: SymSpell
    ) -> None:
        """Explicit opt-in wires the strategy into the builder."""
        config = SpellCheckerConfig()
        config.validation.use_syllable_window_oov = True
        strategies = build_context_validation_strategies(
            config=config, provider=bm005_provider, symspell=bm005_symspell
        )
        names = [s.__class__.__name__ for s in strategies]
        assert "SyllableWindowOOVStrategy" in names

    def test_strategy_not_registered_without_symspell(
        self, bm005_provider: MemoryProvider
    ) -> None:
        """Without symspell, the strategy cannot be instantiated."""
        config = SpellCheckerConfig()
        strategies = build_context_validation_strategies(
            config=config, provider=bm005_provider, symspell=None
        )
        names = [s.__class__.__name__ for s in strategies]
        assert "SyllableWindowOOVStrategy" not in names

    def test_strategy_disabled_via_config(
        self, bm005_provider: MemoryProvider, bm005_symspell: SymSpell
    ) -> None:
        config = SpellCheckerConfig()
        config.validation.use_syllable_window_oov = False
        strategies = build_context_validation_strategies(
            config=config, provider=bm005_provider, symspell=bm005_symspell
        )
        names = [s.__class__.__name__ for s in strategies]
        assert "SyllableWindowOOVStrategy" not in names

    def test_strategy_priority_order_before_hidden_compound(
        self, bm005_provider: MemoryProvider, bm005_symspell: SymSpell
    ) -> None:
        """SW (22) must run BEFORE HC (23) in the strategy list."""
        config = SpellCheckerConfig()
        config.validation.use_syllable_window_oov = True  # opt-in for this test
        strategies = build_context_validation_strategies(
            config=config, provider=bm005_provider, symspell=bm005_symspell
        )
        # Sort by priority to match ContextValidator behavior
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
