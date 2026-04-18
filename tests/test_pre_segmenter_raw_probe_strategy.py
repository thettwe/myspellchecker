"""Tests for :class:`PreSegmenterRawProbeStrategy`.

Covers:

* Config wiring (default-off flag, env override plumbing).
* Strategy registration via :func:`build_context_validation_strategies`.
* Happy path (BM-065-style: missing asat on a compound whose pieces the
  segmenter over-splits into individually-valid words).
* Guards: dict-word skip, long-token skip, name-mask skip, colloquial skip,
  length-diff guard, frequency guard, edit-distance guard.
"""

from __future__ import annotations

from typing import Any

import pytest

from myspellchecker.core.config import SpellCheckerConfig
from myspellchecker.core.constants import ET_WORD
from myspellchecker.core.factories.builders import build_context_validation_strategies
from myspellchecker.core.response import Suggestion
from myspellchecker.core.validation_strategies import PreSegmenterRawProbeStrategy
from myspellchecker.core.validation_strategies.base import ValidationContext
from myspellchecker.providers.memory import MemoryProvider


class _FakeSymSpell:
    """Minimal SymSpell stand-in that returns a fixed candidate list.

    The real :class:`SymSpell` requires a built index; strategies only use
    :meth:`lookup`, so a fake is enough to exercise the probe's decision
    tree in isolation.
    """

    def __init__(self, candidates: dict[str, list[str]]) -> None:
        self._candidates = candidates
        self.calls: list[tuple[str, str, int]] = []

    def lookup(
        self,
        term: str,
        level: str = "syllable",
        max_suggestions: int = 5,
        include_known: bool = False,
        use_phonetic: bool = False,
    ) -> list[Suggestion]:
        self.calls.append((term, level, max_suggestions))
        suggestions = self._candidates.get(term, [])
        return [Suggestion(text=s, source="symspell") for s in suggestions[:max_suggestions]]


@pytest.fixture
def provider() -> MemoryProvider:
    """In-memory provider with the BM-065 vocabulary + a few guard helpers."""
    p = MemoryProvider()
    # BM-065 canonical: the typo 'စွမ်းဆောင်ရည' lacks the final asat.
    p.add_word("စွမ်းဆောင်ရည်", frequency=48_971)
    p.add_word("စွမ်းဆောင်", frequency=12_000)
    p.add_word("ရ", frequency=6_000)
    p.add_word("ည", frequency=900)
    # Separate, high-frequency word used in the "already valid" skip test.
    p.add_word("မြန်မာ", frequency=120_000)
    # Low-freq candidate to exercise the frequency guard.
    p.add_word("ရည်", frequency=40)
    return p


@pytest.fixture
def sym_hit() -> _FakeSymSpell:
    """Fake SymSpell that maps the typo to the gold compound."""
    return _FakeSymSpell({"စွမ်းဆောင်ရည": ["စွမ်းဆောင်ရည်"]})


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
        config = SpellCheckerConfig()
        assert config.validation.use_pre_segmenter_raw_probe is False
        assert config.validation.pre_segmenter_raw_probe_max_ed == 2
        assert config.validation.pre_segmenter_raw_probe_min_freq == 100
        assert config.validation.pre_segmenter_raw_probe_max_length == 15

    def test_strategy_not_registered_when_disabled(self, provider: MemoryProvider) -> None:
        config = SpellCheckerConfig()
        strategies = build_context_validation_strategies(config=config, provider=provider)
        names = [s.__class__.__name__ for s in strategies]
        assert "PreSegmenterRawProbeStrategy" not in names

    def test_strategy_not_registered_without_symspell(self, provider: MemoryProvider) -> None:
        """Flag on but no SymSpell → strategy is skipped (cannot probe)."""
        config = SpellCheckerConfig()
        config.validation.use_pre_segmenter_raw_probe = True
        strategies = build_context_validation_strategies(
            config=config, provider=provider, symspell=None
        )
        names = [s.__class__.__name__ for s in strategies]
        assert "PreSegmenterRawProbeStrategy" not in names

    def test_strategy_registered_when_enabled(
        self, provider: MemoryProvider, sym_hit: _FakeSymSpell
    ) -> None:
        config = SpellCheckerConfig()
        config.validation.use_pre_segmenter_raw_probe = True
        strategies = build_context_validation_strategies(
            config=config,
            provider=provider,
            symspell=sym_hit,  # type: ignore[arg-type]
        )
        hits = [s for s in strategies if isinstance(s, PreSegmenterRawProbeStrategy)]
        assert len(hits) == 1
        assert hits[0].priority() == 23
        assert hits[0].enabled is True


class TestProbeBehaviour:
    def test_happy_path_bm065(self, provider: MemoryProvider, sym_hit: _FakeSymSpell) -> None:
        strategy = PreSegmenterRawProbeStrategy(
            symspell=sym_hit,  # type: ignore[arg-type]
            provider=provider,
            enabled=True,
        )
        ctx = _context("စွမ်းဆောင်ရည", ["စွမ်းဆောင်", "ရ", "ည"])
        errors = strategy.validate(ctx)
        assert len(errors) == 1
        err = errors[0]
        assert err.error_type == ET_WORD
        assert err.text == "စွမ်းဆောင်ရည"
        assert err.position == 0
        assert [str(s) for s in err.suggestions] == ["စွမ်းဆောင်ရည်"]
        # Strategy marks the position so lower-priority strategies can defer.
        assert ctx.existing_errors[0] == ET_WORD

    def test_disabled_returns_empty(self, provider: MemoryProvider, sym_hit: _FakeSymSpell) -> None:
        strategy = PreSegmenterRawProbeStrategy(
            symspell=sym_hit,  # type: ignore[arg-type]
            provider=provider,
            enabled=False,
        )
        ctx = _context("စွမ်းဆောင်ရည", ["စွမ်းဆောင်", "ရ", "ည"])
        assert strategy.validate(ctx) == []

    def test_valid_dict_word_skipped(self, provider: MemoryProvider) -> None:
        """Tokens that are already dict words must NOT be probed."""
        sym = _FakeSymSpell({"မြန်မာ": ["မြန်မာ့"]})
        strategy = PreSegmenterRawProbeStrategy(
            symspell=sym,  # type: ignore[arg-type]
            provider=provider,
            enabled=True,
        )
        ctx = _context("မြန်မာ", ["မြန်မာ"])
        assert strategy.validate(ctx) == []
        # SymSpell.lookup was never called — the guard short-circuits earlier.
        assert sym.calls == []

    def test_long_token_skipped(self, provider: MemoryProvider, sym_hit: _FakeSymSpell) -> None:
        strategy = PreSegmenterRawProbeStrategy(
            symspell=sym_hit,  # type: ignore[arg-type]
            provider=provider,
            enabled=True,
            max_token_length=5,  # shorter than BM-065's 10-char typo
        )
        ctx = _context("စွမ်းဆောင်ရည", ["စွမ်းဆောင်", "ရ", "ည"])
        assert strategy.validate(ctx) == []

    def test_name_mask_skipped(self, provider: MemoryProvider, sym_hit: _FakeSymSpell) -> None:
        strategy = PreSegmenterRawProbeStrategy(
            symspell=sym_hit,  # type: ignore[arg-type]
            provider=provider,
            enabled=True,
        )
        ctx = _context("စွမ်းဆောင်ရည", ["စွမ်းဆောင်", "ရ", "ည"])
        # Flag the first segmented word as a proper name — the whole raw span
        # overlaps it, so the probe should skip.
        ctx.is_name_mask = [True, False, False]
        assert strategy.validate(ctx) == []

    def test_colloquial_variant_skipped(self, provider: MemoryProvider) -> None:
        """Known colloquial variants are respected by the probe."""
        sym = _FakeSymSpell({"ကျနော်": ["ကျွန်တော်"]})
        strategy = PreSegmenterRawProbeStrategy(
            symspell=sym,  # type: ignore[arg-type]
            provider=provider,
            enabled=True,
        )
        ctx = _context("ကျနော်", ["ကျနော်"])
        assert strategy.validate(ctx) == []

    def test_frequency_guard(self, provider: MemoryProvider) -> None:
        """Candidate with freq below `min_frequency` is rejected."""
        sym = _FakeSymSpell({"ရညး": ["ရည်"]})  # candidate has freq 40 in fixture
        strategy = PreSegmenterRawProbeStrategy(
            symspell=sym,  # type: ignore[arg-type]
            provider=provider,
            enabled=True,
            min_frequency=100,
        )
        ctx = _context("ရညး", ["ရညး"])
        assert strategy.validate(ctx) == []

    def test_length_diff_guard(self, provider: MemoryProvider) -> None:
        """Candidate with |len(cand)-len(token)| > max_length_diff is rejected."""
        # Same provider has စွမ်းဆောင် (9 chars) + စွမ်းဆောင်ရည် (12).
        # Token 'စွ' (2 chars) → candidate 'စွမ်းဆောင်ရည်' (12) is len_diff=10.
        sym = _FakeSymSpell({"စွ": ["စွမ်းဆောင်ရည်"]})
        strategy = PreSegmenterRawProbeStrategy(
            symspell=sym,  # type: ignore[arg-type]
            provider=provider,
            enabled=True,
            max_length_diff=2,
        )
        ctx = _context("စွ", ["စွ"])
        assert strategy.validate(ctx) == []

    def test_existing_error_skipped(self, provider: MemoryProvider, sym_hit: _FakeSymSpell) -> None:
        """Positions already claimed by higher-priority strategies are skipped."""
        strategy = PreSegmenterRawProbeStrategy(
            symspell=sym_hit,  # type: ignore[arg-type]
            provider=provider,
            enabled=True,
        )
        ctx = _context("စွမ်းဆောင်ရည", ["စွမ်းဆောင်", "ရ", "ည"])
        ctx.existing_errors[0] = "some_prior_error"
        assert strategy.validate(ctx) == []

    def test_empty_sentence(self, provider: MemoryProvider, sym_hit: _FakeSymSpell) -> None:
        strategy = PreSegmenterRawProbeStrategy(
            symspell=sym_hit,  # type: ignore[arg-type]
            provider=provider,
            enabled=True,
        )
        ctx = ValidationContext(sentence="", words=[], word_positions=[])
        assert strategy.validate(ctx) == []

    def test_absolute_position_uses_sentence_base(
        self, provider: MemoryProvider, sym_hit: _FakeSymSpell
    ) -> None:
        """When the sentence is offset into a larger document the error
        position is the absolute offset, not the sentence-local one."""
        strategy = PreSegmenterRawProbeStrategy(
            symspell=sym_hit,  # type: ignore[arg-type]
            provider=provider,
            enabled=True,
        )
        sentence = "စွမ်းဆောင်ရည"
        words = ["စွမ်းဆောင်", "ရ", "ည"]
        # Simulate a document offset of 50 — sentence_base = 50, so the emitted
        # position should be 50, not 0.
        base = 50
        ctx = ValidationContext(
            sentence=sentence,
            words=words,
            word_positions=[base + sentence.find(w) for w in words],
        )
        errors = strategy.validate(ctx)
        assert len(errors) == 1
        assert errors[0].position == base


class TestRepr:
    def test_repr_contains_key_fields(
        self, provider: MemoryProvider, sym_hit: _FakeSymSpell
    ) -> None:
        strategy = PreSegmenterRawProbeStrategy(
            symspell=sym_hit,  # type: ignore[arg-type]
            provider=provider,
            enabled=True,
        )
        text = repr(strategy)
        assert "PreSegmenterRawProbeStrategy" in text
        assert "priority=23" in text
        assert "enabled=True" in text


class TestSymSpellInvocation:
    """The probe must call SymSpell with level='word' (production code paths
    rely on word-level indexing, not syllable-level)."""

    def test_lookup_uses_word_level(self, provider: MemoryProvider, sym_hit: _FakeSymSpell) -> None:
        strategy = PreSegmenterRawProbeStrategy(
            symspell=sym_hit,  # type: ignore[arg-type]
            provider=provider,
            enabled=True,
        )
        ctx = _context("စွမ်းဆောင်ရည", ["စွမ်းဆောင်", "ရ", "ည"])
        strategy.validate(ctx)
        assert sym_hit.calls, "SymSpell.lookup should have been called"
        assert all(level == "word" for _, level, _ in sym_hit.calls)


class TestLookupExceptionsGraceful:
    """When SymSpell raises an exception the strategy must return [] rather
    than propagate."""

    def test_runtime_error_suppressed(self, provider: MemoryProvider) -> None:
        class _Raising:
            def lookup(self, *_args: Any, **_kwargs: Any) -> list[Suggestion]:
                raise RuntimeError("boom")

        strategy = PreSegmenterRawProbeStrategy(
            symspell=_Raising(),  # type: ignore[arg-type]
            provider=provider,
            enabled=True,
        )
        ctx = _context("စွမ်းဆောင်ရည", ["စွမ်းဆောင်", "ရ", "ည"])
        assert strategy.validate(ctx) == []
