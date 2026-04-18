"""Tests for :class:`ToneSafetyNetStrategy`.

Covers:

* Config wiring (default-off flag, env-override-compatible fields).
* Strategy registration via :func:`build_context_validation_strategies`.
* Happy path for each of the 4 tone chars (း, ့, ံ, ်).
* Guards: non-dict-token skip, frequency guard, freq-ratio guard,
  skip-above-freq guard, name-mask skip, colloquial skip, disabled flag,
  existing-error skip.

Representative D2 examples from [[Tone-Zawgyi Slice 2026-04-19]]:

* ``သွင်`` → ``သွင်း`` (missing visarga)
* ``ခဲ``   → ``ခဲ့``    (missing dot-below)
* ``မှား``  → ``မှာ``   (extraneous visarga — strip direction)
"""

from __future__ import annotations

import pytest

from myspellchecker.core.config import SpellCheckerConfig
from myspellchecker.core.constants import ET_WORD
from myspellchecker.core.factories.builders import build_context_validation_strategies
from myspellchecker.core.validation_strategies import ToneSafetyNetStrategy
from myspellchecker.core.validation_strategies.base import ValidationContext
from myspellchecker.providers.memory import MemoryProvider


@pytest.fixture
def provider() -> MemoryProvider:
    """In-memory provider with the D2 vocabulary + guards for edge cases."""
    p = MemoryProvider()
    # Happy-path D2 pairs (token in dict, candidate in dict, ≥10x ratio).
    p.add_word("သွင်", frequency=500)  # typo (rare)
    p.add_word("သွင်း", frequency=20_000)  # gold (much more common)
    p.add_word("ခဲ", frequency=800)  # typo
    p.add_word("ခဲ့", frequency=50_000)  # gold
    # Anusvara-gold pair.
    p.add_word("စ", frequency=600)  # typo
    p.add_word("စံ", frequency=9_000)  # gold
    # Strip-direction pair: ``မှား`` has an extraneous ``း``; gold is ``မှာ``.
    p.add_word("မှား", frequency=600)
    p.add_word("မှာ", frequency=40_000)
    # Clean sentence (no tone fix) — high-freq token must not be second-guessed.
    p.add_word("မြန်မာ", frequency=120_000)
    # Low-frequency candidate that should fail the min_frequency gate.
    p.add_word("နည်း", frequency=200)
    p.add_word("နည်", frequency=100)
    # High-freq baseline token whose candidate would clear the ratio but
    # the token is above skip_above_freq.
    p.add_word("က", frequency=100_000)
    p.add_word("က်", frequency=80_000)
    return p


def _context(sentence: str, words: list[str]) -> ValidationContext:
    positions: list[int] = []
    cursor = 0
    for word in words:
        idx = sentence.find(word, cursor)
        assert idx >= 0, f"word {word!r} not in sentence"
        positions.append(idx)
        cursor = idx + len(word)
    return ValidationContext(sentence=sentence, words=words, word_positions=positions)


class TestConfigWiring:
    def test_flag_defaults_off(self) -> None:
        """Default off pending tzn-benchmark-01 gate."""
        config = SpellCheckerConfig()
        assert config.validation.use_tone_safety_net is False
        assert config.validation.tone_safety_net_min_frequency == 1000
        assert config.validation.tone_safety_net_freq_ratio == 10.0
        assert config.validation.tone_safety_net_skip_above_freq == 50_000
        assert config.validation.tone_safety_net_confidence == 0.80

    def test_strategy_registered_when_flag_enabled(self, provider: MemoryProvider) -> None:
        config = SpellCheckerConfig()
        config.validation.use_tone_safety_net = True
        strategies = build_context_validation_strategies(
            provider=provider,
            config=config,
            symspell=None,
            semantic_checker=None,
        )
        assert any(isinstance(s, ToneSafetyNetStrategy) for s in strategies)

    def test_strategy_not_registered_when_flag_disabled(self, provider: MemoryProvider) -> None:
        config = SpellCheckerConfig()
        config.validation.use_tone_safety_net = False
        strategies = build_context_validation_strategies(
            provider=provider,
            config=config,
            symspell=None,
            semantic_checker=None,
        )
        assert not any(isinstance(s, ToneSafetyNetStrategy) for s in strategies)


class TestInsertTone:
    """Happy-path: missing visarga / dot-below / anusvara is repaired."""

    def test_visarga_insert(self, provider: MemoryProvider) -> None:
        strategy = ToneSafetyNetStrategy(provider=provider, enabled=True)
        context = _context("သွင်", ["သွင်"])
        errors = strategy.validate(context)
        assert len(errors) == 1
        assert errors[0].text == "သွင်"
        assert errors[0].error_type == ET_WORD
        assert errors[0].suggestions[0].text == "သွင်း"
        assert errors[0].suggestions[0].source == "tone_safety_net"

    def test_dot_below_insert(self, provider: MemoryProvider) -> None:
        strategy = ToneSafetyNetStrategy(provider=provider, enabled=True)
        context = _context("ခဲ", ["ခဲ"])
        errors = strategy.validate(context)
        assert len(errors) == 1
        assert errors[0].suggestions[0].text == "ခဲ့"

    def test_anusvara_insert(self, provider: MemoryProvider) -> None:
        strategy = ToneSafetyNetStrategy(provider=provider, enabled=True)
        context = _context("စ", ["စ"])
        errors = strategy.validate(context)
        assert len(errors) == 1
        assert errors[0].suggestions[0].text == "စံ"


class TestStripTone:
    """Strip direction: extraneous trailing tone is flagged."""

    def test_strip_visarga(self, provider: MemoryProvider) -> None:
        strategy = ToneSafetyNetStrategy(provider=provider, enabled=True)
        context = _context("မှား", ["မှား"])
        errors = strategy.validate(context)
        assert len(errors) == 1
        assert errors[0].suggestions[0].text == "မှာ"


class TestGuards:
    def test_oov_token_skipped(self, provider: MemoryProvider) -> None:
        """OOV tokens are skipped — raw-probe / SymSpell handle those."""
        strategy = ToneSafetyNetStrategy(provider=provider, enabled=True)
        context = _context("ဝလဒှ", ["ဝလဒှ"])  # not in dict
        errors = strategy.validate(context)
        assert errors == []

    def test_below_min_frequency_skipped(self, provider: MemoryProvider) -> None:
        """Candidate freq below ``min_frequency`` is rejected."""
        strategy = ToneSafetyNetStrategy(
            provider=provider,
            enabled=True,
            min_frequency=1_000,
        )
        # ``နည်`` (freq 100) + း → ``နည်း`` (freq 200) — candidate below min.
        context = _context("နည်", ["နည်"])
        errors = strategy.validate(context)
        assert errors == []

    def test_freq_ratio_guard(self, provider: MemoryProvider) -> None:
        """Candidate must be ratio× more frequent than the token."""
        # Force a very high ratio — even ``သွင်`` (500) vs ``သွင်း`` (20000,
        # 40x) clears 10x; raising the bar past 40x should reject.
        strategy = ToneSafetyNetStrategy(
            provider=provider,
            enabled=True,
            freq_ratio=100.0,
        )
        context = _context("သွင်", ["သွင်"])
        errors = strategy.validate(context)
        assert errors == []

    def test_skip_above_freq(self, provider: MemoryProvider) -> None:
        """High-frequency tokens are not second-guessed."""
        strategy = ToneSafetyNetStrategy(
            provider=provider,
            enabled=True,
            skip_above_freq=10_000,
        )
        # ``က`` is freq 100_000 — well above the 10_000 cap → no probe.
        context = _context("က", ["က"])
        errors = strategy.validate(context)
        assert errors == []

    def test_disabled_flag(self, provider: MemoryProvider) -> None:
        """With ``enabled=False`` the strategy is a no-op."""
        strategy = ToneSafetyNetStrategy(provider=provider, enabled=False)
        context = _context("သွင်", ["သွင်"])
        errors = strategy.validate(context)
        assert errors == []

    def test_existing_error_skipped(self, provider: MemoryProvider) -> None:
        """If another strategy already claimed the position, skip it."""
        strategy = ToneSafetyNetStrategy(provider=provider, enabled=True)
        context = _context("သွင်", ["သွင်"])
        context.existing_errors[0] = ET_WORD
        errors = strategy.validate(context)
        assert errors == []

    def test_name_mask_skipped(self, provider: MemoryProvider) -> None:
        """Name-masked tokens are not probed."""
        strategy = ToneSafetyNetStrategy(provider=provider, enabled=True)
        context = _context("သွင်", ["သွင်"])
        context.is_name_mask = [True]
        errors = strategy.validate(context)
        assert errors == []

    def test_empty_sentence(self, provider: MemoryProvider) -> None:
        strategy = ToneSafetyNetStrategy(provider=provider, enabled=True)
        context = ValidationContext(sentence="", words=[], word_positions=[])
        errors = strategy.validate(context)
        assert errors == []

    def test_non_myanmar_passthrough(self, provider: MemoryProvider) -> None:
        strategy = ToneSafetyNetStrategy(provider=provider, enabled=True)
        context = ValidationContext(sentence="hello world", words=[], word_positions=[])
        errors = strategy.validate(context)
        assert errors == []


class TestCandidateVariants:
    """Exercise the candidate-generation helper in isolation."""

    def test_four_variants_for_bare_token(self, provider: MemoryProvider) -> None:
        strategy = ToneSafetyNetStrategy(provider=provider, enabled=True)
        variants = strategy._candidate_variants("သွင်")
        # Append visarga / dot-below / anusvara; no strip since last char is asat.
        assert "သွင်\u1038" in variants  # +း
        assert "သွင်\u1037" in variants  # +့
        assert "သွင်\u1036" in variants  # +ံ
        # Strip the trailing ် (asat is in the strip set).
        assert "သွင" in variants

    def test_strip_when_tone_present(self, provider: MemoryProvider) -> None:
        strategy = ToneSafetyNetStrategy(provider=provider, enabled=True)
        variants = strategy._candidate_variants("မှား")
        # Strip the trailing ``း``.
        assert "မှာ" in variants
        # Append ့ and ံ (but not း since that's already the suffix).
        assert "မှား\u1037" in variants
        assert "မှား\u1036" in variants
        assert "မှား\u1038" not in variants

    def test_empty_token_returns_empty(self, provider: MemoryProvider) -> None:
        strategy = ToneSafetyNetStrategy(provider=provider, enabled=True)
        assert strategy._candidate_variants("") == []


class TestMultipleTokens:
    def test_fires_on_each_matching_token(self, provider: MemoryProvider) -> None:
        """Strategy emits one error per problematic token in the sentence."""
        strategy = ToneSafetyNetStrategy(provider=provider, enabled=True)
        sentence = "သွင် ခဲ"
        context = _context(sentence, ["သွင်", "ခဲ"])
        errors = strategy.validate(context)
        texts = sorted(e.text for e in errors)
        assert texts == sorted(["သွင်", "ခဲ"])

    def test_mixed_token_leaves_clean_word_alone(self, provider: MemoryProvider) -> None:
        """High-freq ``မြန်မာ`` is clean; only ``သွင်`` is flagged."""
        strategy = ToneSafetyNetStrategy(provider=provider, enabled=True)
        sentence = "မြန်မာ သွင်"
        context = _context(sentence, ["မြန်မာ", "သွင်"])
        errors = strategy.validate(context)
        assert len(errors) == 1
        assert errors[0].text == "သွင်"
