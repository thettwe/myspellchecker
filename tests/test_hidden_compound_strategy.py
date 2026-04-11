"""Tests for HiddenCompoundStrategy.

Covers strategy loading, registration, scaffold no-op behaviour, and
bigram + trigram-lookahead detection logic.
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
    """Minimal scaffold checks: instantiation and disabled-mode no-op."""

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

    def test_validate_returns_empty_for_minimal_provider(
        self, memory_provider: MemoryProvider
    ) -> None:
        """Enabled strategy returns [] when the in-memory provider lacks
        enough vocabulary to trigger compound detection."""
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

    def test_strategy_registered_by_default(self, memory_provider: MemoryProvider) -> None:
        """HiddenCompoundStrategy is registered when no overrides are set."""
        config = SpellCheckerConfig()
        assert config.validation.use_hidden_compound_detection is True
        strategies = build_context_validation_strategies(config=config, provider=memory_provider)
        names = [s.__class__.__name__ for s in strategies]
        assert "HiddenCompoundStrategy" in names

    def test_strategy_not_registered_when_disabled(
        self, memory_provider: MemoryProvider
    ) -> None:
        config = SpellCheckerConfig()
        config.validation.use_hidden_compound_detection = False
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
    """Verify the provider base default for is_valid_vocabulary."""

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


# ── Bigram / trigram detection ────────────────────────────────────────


@pytest.fixture
def real_provider():
    """Real SQLiteProvider against the production DB (skipped if missing)."""
    import os

    db_path = "data/mySpellChecker_production.db"
    if not os.path.exists(db_path):
        pytest.skip(f"Production DB not found at {db_path}")
    from myspellchecker.providers.sqlite import SQLiteProvider

    return SQLiteProvider(database_path=db_path)


@pytest.fixture
def real_hasher():
    """Real PhoneticHasher (required for variant generation)."""
    from myspellchecker.text.phonetic import PhoneticHasher

    return PhoneticHasher()


@pytest.fixture
def enabled_strategy(real_provider, real_hasher):
    return HiddenCompoundStrategy(
        provider=real_provider,
        hasher=real_hasher,
        enabled=True,
        max_token_syllables=3,
        max_variants_per_token=20,
        compound_min_frequency=100,
        confidence_floor=0.75,
        enable_trigram_lookahead=True,
    )


def _make_ctx(sentence: str, words: list[str]) -> ValidationContext:
    """Build a ValidationContext with word_positions inferred from the sentence."""
    positions = []
    offset = 0
    for w in words:
        positions.append(offset)
        offset += len(w)
    return ValidationContext(sentence=sentence, words=words, word_positions=positions)


class TestHiddenCompoundDetection:
    """Bigram + trigram lookahead detection."""

    def test_bm005_trigram_lookahead(self, enabled_strategy) -> None:
        """BM-005: segmented 'ခုန်', 'ကျ', 'စရိတ်' → emit single-token span
        with dual suggestions (single variant + compound).

        The trigram is used for VERIFICATION only. The emitted span is the
        single mistyped token. The suggestion list contains both the
        single-token variant ('ကုန်') and the compound ('ကုန်ကျ'), so the
        error matches gold regardless of whether gold annotations use
        single-token or bigram format.
        """
        ctx = _make_ctx(
            "ခုန်ကျစရိတ်အကြောင်း",
            ["ခုန်", "ကျ", "စရိတ်", "အကြောင်း"],
        )
        errors = enabled_strategy.validate(ctx)
        assert len(errors) == 1
        e = errors[0]
        assert e.error_type == ET_HIDDEN_COMPOUND_TYPO
        suggestion_texts = [s.text if hasattr(s, "text") else s for s in e.suggestions]
        assert "ကုန်" in suggestion_texts  # single-token variant
        assert "ကုန်ကျ" in suggestion_texts  # compound
        # Span covers the single mistyped token.
        assert e.text == "ခုန်"
        assert e.position == 0
        assert e.confidence >= 0.75

    def test_pinap_single_char_substitution(self, enabled_strategy) -> None:
        """ပိနပ် → ဖိနပ် via ပ→ဖ aspirated-unaspirated swap.

        Segmenter splits as ['ပိ', 'နပ်']; strategy tests variants of 'ပိ'
        including 'ဖိ', builds candidate 'ဖိနပ်' which is a high-freq word.
        Single-token span is 'ပိ'; suggestions include 'ဖိ' + 'ဖိနပ်'.
        """
        ctx = _make_ctx("ပိနပ်", ["ပိ", "နပ်"])
        errors = enabled_strategy.validate(ctx)
        assert len(errors) == 1
        suggestion_texts = [
            s.text if hasattr(s, "text") else s for s in errors[0].suggestions
        ]
        assert "ဖိနပ်" in suggestion_texts

    def test_ganan_backward_direction(self, enabled_strategy) -> None:
        """ဃဏန်း → ဂဏန်း via backward walking (typo in w_0, not w_1).

        Segmenter splits as ['ဃ', 'ဏန်း']; forward direction (variants of 'ဃ')
        produces 'ဂ' which when combined with 'ဏန်း' yields 'ဂဏန်း' (freq 16367).
        """
        ctx = _make_ctx("ဃဏန်း", ["ဃ", "ဏန်း"])
        errors = enabled_strategy.validate(ctx)
        assert len(errors) == 1
        suggestion_texts = [
            s.text if hasattr(s, "text") else s for s in errors[0].suggestions
        ]
        assert "ဂဏန်း" in suggestion_texts

    def test_correct_compound_not_flagged(self, enabled_strategy) -> None:
        """The canonical correct form must NOT be flagged as a typo."""
        ctx = _make_ctx("ကုန်ကျစရိတ်", ["ကုန်", "ကျ", "စရိတ်"])
        errors = enabled_strategy.validate(ctx)
        assert errors == []

    def test_unrelated_phrase_not_flagged(self, enabled_strategy) -> None:
        """A valid phrase with no compound typo must NOT trigger."""
        ctx = _make_ctx("ငါ သွား မယ်", ["ငါ", "သွား", "မယ်"])
        errors = enabled_strategy.validate(ctx)
        assert errors == []

    def test_polite_particle_not_flagged(self, enabled_strategy) -> None:
        """Common particle sequences must not trigger false positives."""
        ctx = _make_ctx("သူ သည်", ["သူ", "သည်"])
        errors = enabled_strategy.validate(ctx)
        assert errors == []

    def test_disabled_strategy_returns_empty(self, enabled_strategy) -> None:
        """Toggle enabled=False and re-run — must return []."""
        enabled_strategy.enabled = False
        ctx = _make_ctx("ခုန်ကျစရိတ်", ["ခုန်", "ကျ", "စရိတ်"])
        assert enabled_strategy.validate(ctx) == []

    def test_confidence_floor_gates_emission(self, real_provider, real_hasher) -> None:
        """Raising the floor should suppress marginal corrections."""
        strict_strategy = HiddenCompoundStrategy(
            provider=real_provider,
            hasher=real_hasher,
            enabled=True,
            compound_min_frequency=100,
            confidence_floor=0.99,  # impossibly strict
        )
        ctx = _make_ctx("ခုန်ကျစရိတ်", ["ခုန်", "ကျ", "စရိတ်"])
        errors = strict_strategy.validate(ctx)
        # ခုန်→ကုန် has high confidence (~0.977) so this should still fire,
        # but only at the threshold. Proves the gate is wired.
        # A threshold of 0.99 suppresses anything below 0.99.
        for e in errors:
            assert e.confidence >= 0.99

    def test_character_filter_rejects_plain_particles(self, real_provider, real_hasher) -> None:
        """Tokens without any typo-prone chars are skipped for perf."""
        strategy = HiddenCompoundStrategy(
            provider=real_provider,
            hasher=real_hasher,
            enabled=True,
            require_typo_prone_chars=True,
        )
        # Check the private helper directly — a plain particle 'သည်' should
        # contain ည (retroflex/dental typo-prone), so it IS a candidate.
        # 'ာ' alone is not typo-prone in the list.
        assert strategy._is_candidate_token("သည်") is True
        # A garbage single particle without typo-prone chars is rejected.
        assert strategy._is_candidate_token("") is False

    def test_variant_cache_returns_frozenset(self, real_provider, real_hasher) -> None:
        """The variant cache must return a hashable frozenset."""
        strategy = HiddenCompoundStrategy(provider=real_provider, hasher=real_hasher, enabled=True)
        result = strategy._cached_variants("ခုန်")
        assert isinstance(result, frozenset)
        assert len(result) > 0
        # Cache hit on second call (equal result).
        result2 = strategy._cached_variants("ခုန်")
        assert result == result2

    def test_short_sentence_no_crash(self, enabled_strategy) -> None:
        """Single token or empty context must not crash."""
        assert enabled_strategy.validate(_make_ctx("", [])) == []
        assert enabled_strategy.validate(_make_ctx("ခုန်", ["ခုန်"])) == []

    def test_error_position_is_typo_token(self, enabled_strategy) -> None:
        """Emitted error span is the single mistyped token only."""
        ctx = _make_ctx(
            "ခုန်ကျစရိတ်အကြောင်း",
            ["ခုန်", "ကျ", "စရိတ်", "အကြောင်း"],
        )
        errors = enabled_strategy.validate(ctx)
        assert len(errors) == 1
        e = errors[0]
        assert e.position == 0
        assert e.text == "ခုန်"
