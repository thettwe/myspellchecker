"""Tests for :class:`CrossWhitespaceProbeStrategy`.

Covers:
* Happy path: space-insertion compound detected (e.g., "လူ သွား" → "လူသွား").
* Multi-part: 3-part compound (e.g., "လူ သွား လမ်း" → pairwise detection).
* Guards: parts not in dict, concat not in dict, concat low freq,
  part too long, concat too long, name-masked overlap, already-claimed position.
* Config wiring and builder registration.
"""

from __future__ import annotations

import pytest

from myspellchecker.core.config import SpellCheckerConfig
from myspellchecker.core.constants import ET_WORD
from myspellchecker.core.factories.builders import build_context_validation_strategies
from myspellchecker.core.validation_strategies.base import ValidationContext
from myspellchecker.core.validation_strategies.cross_whitespace_probe_strategy import (
    CrossWhitespaceProbeStrategy,
)
from myspellchecker.providers.memory import MemoryProvider


@pytest.fixture
def provider() -> MemoryProvider:
    p = MemoryProvider()
    p.add_word("လူ", frequency=200_000)
    p.add_word("သွား", frequency=180_000)
    p.add_word("လမ်း", frequency=150_000)
    p.add_word("လူသွား", frequency=672)
    p.add_word("လူသွားလမ်း", frequency=653)
    p.add_word("အိမ်ခြံ", frequency=5_000)
    p.add_word("မြေ", frequency=80_000)
    p.add_word("အိမ်ခြံမြေ", frequency=9_955)
    p.add_word("ကောင်း", frequency=300_000)
    p.add_word("မွန်", frequency=50_000)
    # concat of ကောင်း + မွန် NOT in dict → should not fire
    p.add_word("မြန်မာ", frequency=120_000)
    return p


def _context(sentence: str, words: list[str] | None = None) -> ValidationContext:
    if words is None:
        words = sentence.split()
    positions: list[int] = []
    cursor = 0
    for word in words:
        idx = sentence.find(word, cursor)
        assert idx >= 0, f"word {word!r} not in sentence at cursor {cursor}"
        positions.append(idx)
        cursor = idx + len(word)
    return ValidationContext(sentence=sentence, words=words, word_positions=positions)


def _make_strategy(provider: MemoryProvider, **overrides) -> CrossWhitespaceProbeStrategy:
    defaults = {
        "enabled": True,
        "min_concat_freq": 50,
        "max_part_length": 30,
        "max_concat_length": 40,
        "confidence": 0.90,
    }
    defaults.update(overrides)
    return CrossWhitespaceProbeStrategy(provider=provider, **defaults)


class TestHappyPath:
    def test_two_part_compound_detected(self, provider: MemoryProvider) -> None:
        strategy = _make_strategy(provider)
        ctx = _context("အိမ်ခြံ မြေ")
        errors = strategy.validate(ctx)
        assert len(errors) == 1
        assert errors[0].text == "အိမ်ခြံ မြေ"
        assert errors[0].suggestions[0].text == "အိမ်ခြံမြေ"
        assert errors[0].error_type == ET_WORD

    def test_three_part_pairwise_detection(self, provider: MemoryProvider) -> None:
        strategy = _make_strategy(provider)
        ctx = _context("လူ သွား လမ်း")
        errors = strategy.validate(ctx)
        assert len(errors) >= 1
        suggestions = [e.suggestions[0].text for e in errors]
        assert "လူသွား" in suggestions

    def test_confidence_and_source(self, provider: MemoryProvider) -> None:
        strategy = _make_strategy(provider, confidence=0.88)
        ctx = _context("အိမ်ခြံ မြေ")
        errors = strategy.validate(ctx)
        assert errors[0].confidence == 0.88
        assert errors[0].suggestions[0].source == "cross_whitespace_probe"

    def test_position_claimed_in_context(self, provider: MemoryProvider) -> None:
        strategy = _make_strategy(provider)
        ctx = _context("အိမ်ခြံ မြေ")
        strategy.validate(ctx)
        assert 0 in ctx.existing_errors


class TestGuards:
    def test_concat_not_in_dict_skipped(self, provider: MemoryProvider) -> None:
        strategy = _make_strategy(provider)
        ctx = _context("ကောင်း မွန်")
        errors = strategy.validate(ctx)
        assert len(errors) == 0

    def test_concat_low_freq_skipped(self, provider: MemoryProvider) -> None:
        provider.add_word("ကောင်းမွန်", frequency=10)
        strategy = _make_strategy(provider, min_concat_freq=50)
        ctx = _context("ကောင်း မွန်")
        errors = strategy.validate(ctx)
        assert len(errors) == 0

    def test_part_not_in_dict_skipped(self, provider: MemoryProvider) -> None:
        provider.add_word("ထွတ်ချိန်", frequency=5_000)
        strategy = _make_strategy(provider)
        ctx = _context("ထွတ် ချိန်")
        errors = strategy.validate(ctx)
        assert len(errors) == 0

    def test_part_too_long_skipped(self, provider: MemoryProvider) -> None:
        strategy = _make_strategy(provider, max_part_length=3)
        ctx = _context("အိမ်ခြံ မြေ")
        errors = strategy.validate(ctx)
        assert len(errors) == 0

    def test_concat_too_long_skipped(self, provider: MemoryProvider) -> None:
        strategy = _make_strategy(provider, max_concat_length=5)
        ctx = _context("အိမ်ခြံ မြေ")
        errors = strategy.validate(ctx)
        assert len(errors) == 0

    def test_already_claimed_position_skipped(self, provider: MemoryProvider) -> None:
        strategy = _make_strategy(provider)
        ctx = _context("အိမ်ခြံ မြေ")
        ctx.existing_errors[0] = ET_WORD
        errors = strategy.validate(ctx)
        assert len(errors) == 0

    def test_name_masked_skipped(self, provider: MemoryProvider) -> None:
        strategy = _make_strategy(provider)
        ctx = _context("အိမ်ခြံ မြေ")
        ctx.is_name_mask = [True, True]
        errors = strategy.validate(ctx)
        assert len(errors) == 0

    def test_disabled_returns_empty(self, provider: MemoryProvider) -> None:
        strategy = _make_strategy(provider, enabled=False)
        ctx = _context("အိမ်ခြံ မြေ")
        errors = strategy.validate(ctx)
        assert len(errors) == 0

    def test_single_token_no_error(self, provider: MemoryProvider) -> None:
        strategy = _make_strategy(provider)
        ctx = _context("မြန်မာ")
        errors = strategy.validate(ctx)
        assert len(errors) == 0

    def test_non_myanmar_gap_skipped(self, provider: MemoryProvider) -> None:
        strategy = _make_strategy(provider)
        ctx = _context("အိမ်ခြံ123မြေ", words=["အိမ်ခြံ123မြေ"])
        errors = strategy.validate(ctx)
        assert len(errors) == 0


class TestConfig:
    def test_default_enabled(self) -> None:
        cfg = SpellCheckerConfig()
        assert cfg.validation.use_cross_whitespace_probe is True

    def test_builder_registers_strategy(self) -> None:
        cfg = SpellCheckerConfig()
        cfg.validation.use_cross_whitespace_probe = True
        p = MemoryProvider()
        strategies = build_context_validation_strategies(cfg, p, symspell=None)
        types = [type(s).__name__ for s in strategies]
        assert "CrossWhitespaceProbeStrategy" in types

    def test_builder_skips_when_disabled(self) -> None:
        cfg = SpellCheckerConfig()
        cfg.validation.use_cross_whitespace_probe = False
        p = MemoryProvider()
        strategies = build_context_validation_strategies(cfg, p, symspell=None)
        types = [type(s).__name__ for s in strategies]
        assert "CrossWhitespaceProbeStrategy" not in types
