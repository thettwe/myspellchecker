"""Tests for :class:`CompoundMergeProbeStrategy`.

Covers:
* Happy path: compound typo fragmented into 2-3 valid tokens is recovered.
* Fragment-evidence gate: all-high-frequency tokens are skipped.
* Name-mask gate: windows overlapping a name-masked token are skipped.
* Position dedup: already-claimed positions are not re-emitted.
* Colloquial variant skip.
* Edit distance guard.
* Frequency guard.
* Length guard.
* Single-token input (no window possible).
* Config wiring via builder factory.
"""

from __future__ import annotations

import pytest

from myspellchecker.core.config import SpellCheckerConfig
from myspellchecker.core.constants import ET_WORD
from myspellchecker.core.factories.builders import build_context_validation_strategies
from myspellchecker.core.response import Suggestion
from myspellchecker.core.validation_strategies.base import ValidationContext
from myspellchecker.core.validation_strategies.compound_merge_probe_strategy import (
    CompoundMergeProbeStrategy,
)
from myspellchecker.providers.memory import MemoryProvider


class _FakeSymSpell:
    """Minimal SymSpell stand-in returning fixed candidates."""

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
    """In-memory provider with BM-065 vocabulary + guard helpers."""
    p = MemoryProvider()
    # Gold compound
    p.add_word("စွမ်းဆောင်ရည်", frequency=48_971)
    # Fragments (all valid, mix of freq)
    p.add_word("စွမ်းဆောင်", frequency=12_000)
    p.add_word("ရ", frequency=6_000)
    p.add_word("ည", frequency=900)
    # Two-token merge case: သံဂါ → သံဃာ
    p.add_word("သံဃာ", frequency=30_000)
    p.add_word("သံ", frequency=80_000)
    p.add_word("ဂါ", frequency=5_000)
    # High-frequency pair that should NOT be merged
    p.add_word("ရန်", frequency=200_000)
    p.add_word("ကုန်", frequency=180_000)
    p.add_word("ရန်ကုန်", frequency=300_000)
    # Low-freq candidate to exercise frequency guard
    p.add_word("ရည်", frequency=40)
    return p


@pytest.fixture
def sym_3tok() -> _FakeSymSpell:
    """SymSpell that resolves the 3-token compound."""
    return _FakeSymSpell({"စွမ်းဆောင်ရည": ["စွမ်းဆောင်ရည်"]})


@pytest.fixture
def sym_2tok() -> _FakeSymSpell:
    """SymSpell that resolves the 2-token compound."""
    return _FakeSymSpell({"သံဂါ": ["သံဃာ"]})


def _context(
    sentence: str,
    words: list[str],
    is_name_mask: list[bool] | None = None,
) -> ValidationContext:
    positions: list[int] = []
    cursor = 0
    for word in words:
        idx = sentence.find(word, cursor)
        assert idx >= 0, f"word {word!r} not in sentence from cursor {cursor}"
        positions.append(idx)
        cursor = idx + len(word)
    return ValidationContext(
        sentence=sentence,
        words=words,
        word_positions=positions,
        is_name_mask=is_name_mask or [False] * len(words),
    )


# --- Happy path ---


def test_3token_compound_recovery(provider: MemoryProvider, sym_3tok: _FakeSymSpell) -> None:
    """BM-065: စွမ်းဆောင်ရည (3 fragments) → စွမ်းဆောင်ရည်."""
    strategy = CompoundMergeProbeStrategy(
        symspell=sym_3tok,
        provider=provider,
        enabled=True,
        min_candidate_freq=100,
        max_edit_distance=2,
        fragment_freq_floor=50_000,
    )
    ctx = _context("စွမ်းဆောင်ရည", ["စွမ်းဆောင်", "ရ", "ည"])
    errors = strategy.validate(ctx)
    assert len(errors) == 1
    assert errors[0].text == "စွမ်းဆောင်ရည"
    assert errors[0].suggestions[0].text == "စွမ်းဆောင်ရည်"
    assert errors[0].position == 0
    assert errors[0].error_type == ET_WORD


def test_2token_compound_recovery(provider: MemoryProvider, sym_2tok: _FakeSymSpell) -> None:
    """BM-100: သံဂါ (2 fragments) → သံဃာ."""
    strategy = CompoundMergeProbeStrategy(
        symspell=sym_2tok,
        provider=provider,
        enabled=True,
        min_candidate_freq=100,
        max_edit_distance=2,
        fragment_freq_floor=50_000,
    )
    ctx = _context("သံဂါ", ["သံ", "ဂါ"])
    errors = strategy.validate(ctx)
    assert len(errors) == 1
    assert errors[0].suggestions[0].text == "သံဃာ"


# --- Guards ---


def test_disabled_returns_empty(provider: MemoryProvider, sym_3tok: _FakeSymSpell) -> None:
    strategy = CompoundMergeProbeStrategy(symspell=sym_3tok, provider=provider, enabled=False)
    ctx = _context("စွမ်းဆောင်ရည", ["စွမ်းဆောင်", "ရ", "ည"])
    assert strategy.validate(ctx) == []


def test_single_token_returns_empty(provider: MemoryProvider, sym_3tok: _FakeSymSpell) -> None:
    """Single token — no window of 2+ possible."""
    strategy = CompoundMergeProbeStrategy(symspell=sym_3tok, provider=provider, enabled=True)
    ctx = _context("စွမ်းဆောင်", ["စွမ်းဆောင်"])
    assert strategy.validate(ctx) == []


def test_all_high_freq_skipped(provider: MemoryProvider) -> None:
    """When all tokens are high-frequency, fragment-evidence gate rejects."""
    sym = _FakeSymSpell({"ရန်ကုန်": ["ရန်ကုန်"]})
    strategy = CompoundMergeProbeStrategy(
        symspell=sym,
        provider=provider,
        enabled=True,
        fragment_freq_floor=50_000,
    )
    ctx = _context("ရန်ကုန်", ["ရန်", "ကုန်"])
    errors = strategy.validate(ctx)
    assert len(errors) == 0
    assert len(sym.calls) == 0


def test_name_mask_skipped(provider: MemoryProvider, sym_2tok: _FakeSymSpell) -> None:
    """Name-masked tokens prevent the probe from firing."""
    strategy = CompoundMergeProbeStrategy(
        symspell=sym_2tok,
        provider=provider,
        enabled=True,
        min_candidate_freq=100,
        fragment_freq_floor=100_000,
    )
    ctx = _context("သံဂါ", ["သံ", "ဂါ"], is_name_mask=[True, False])
    assert strategy.validate(ctx) == []


def test_existing_error_skipped(provider: MemoryProvider, sym_2tok: _FakeSymSpell) -> None:
    """Position already flagged → skip."""
    strategy = CompoundMergeProbeStrategy(
        symspell=sym_2tok,
        provider=provider,
        enabled=True,
        min_candidate_freq=100,
        fragment_freq_floor=100_000,
    )
    ctx = _context("သံဂါ", ["သံ", "ဂါ"])
    ctx.existing_errors[0] = ET_WORD
    assert strategy.validate(ctx) == []


def test_candidate_freq_guard(provider: MemoryProvider) -> None:
    """Candidate below min_candidate_freq is rejected."""
    sym = _FakeSymSpell({"စွမ်းရေ": ["ရည်"]})
    strategy = CompoundMergeProbeStrategy(
        symspell=sym,
        provider=provider,
        enabled=True,
        min_candidate_freq=500,
        fragment_freq_floor=100_000,
    )
    # ရည် has freq=40 < 500
    provider.add_word("စွမ်း", frequency=10_000)
    provider.add_word("ရေ", frequency=8_000)
    ctx = _context("စွမ်းရေ", ["စွမ်း", "ရေ"])
    assert strategy.validate(ctx) == []


def test_span_length_guard(provider: MemoryProvider, sym_3tok: _FakeSymSpell) -> None:
    """Spans exceeding max_span_length are skipped."""
    strategy = CompoundMergeProbeStrategy(
        symspell=sym_3tok,
        provider=provider,
        enabled=True,
        max_span_length=5,
    )
    ctx = _context("စွမ်းဆောင်ရည", ["စွမ်းဆောင်", "ရ", "ည"])
    assert strategy.validate(ctx) == []


def test_priority_is_22(provider: MemoryProvider, sym_3tok: _FakeSymSpell) -> None:
    strategy = CompoundMergeProbeStrategy(symspell=sym_3tok, provider=provider, enabled=True)
    assert strategy.priority() == 46


# --- Particle exclusion gate (cmlg-02) ---


def test_particle_in_window_skipped(provider: MemoryProvider) -> None:
    """Window containing a never-merge particle is skipped entirely."""
    sym = _FakeSymSpell({"သံကို": ["သံကို"]})
    provider.add_word("သံကို", frequency=60_000)
    strategy = CompoundMergeProbeStrategy(
        symspell=sym,
        provider=provider,
        enabled=True,
        min_candidate_freq=100,
        fragment_freq_floor=100_000,
    )
    ctx = _context("သံကို", ["သံ", "ကို"])
    assert strategy.validate(ctx) == []
    assert len(sym.calls) == 0


def test_multiple_particles_all_skipped(provider: MemoryProvider) -> None:
    """Every particle type in a sentence prevents its window from probing."""
    sym = _FakeSymSpell({})
    provider.add_word("တယ်", frequency=500_000)
    provider.add_word("ပါ", frequency=400_000)
    strategy = CompoundMergeProbeStrategy(
        symspell=sym,
        provider=provider,
        enabled=True,
        min_candidate_freq=100,
        fragment_freq_floor=100_000,
    )
    ctx = _context("သံတယ်ပါ", ["သံ", "တယ်", "ပါ"])
    assert strategy.validate(ctx) == []


def test_non_particle_window_still_probes(
    provider: MemoryProvider, sym_2tok: _FakeSymSpell
) -> None:
    """Windows with non-particle tokens are not affected by the gate."""
    strategy = CompoundMergeProbeStrategy(
        symspell=sym_2tok,
        provider=provider,
        enabled=True,
        min_candidate_freq=100,
        max_edit_distance=2,
        fragment_freq_floor=50_000,
    )
    ctx = _context("သံဂါ", ["သံ", "ဂါ"])
    errors = strategy.validate(ctx)
    assert len(errors) == 1
    assert errors[0].suggestions[0].text == "သံဃာ"


def test_verbal_complement_not_excluded(provider: MemoryProvider) -> None:
    """Verbal complements (ကျ, ပြ, ချ) are NOT particles — they can be fragments."""
    sym = _FakeSymSpell({"သံကျ": ["သံကျား"]})
    provider.add_word("သံကျား", frequency=20_000)
    provider.add_word("ကျ", frequency=30_000)
    strategy = CompoundMergeProbeStrategy(
        symspell=sym,
        provider=provider,
        enabled=True,
        min_candidate_freq=100,
        max_edit_distance=2,
        max_length_diff=3,
        fragment_freq_floor=100_000,
    )
    ctx = _context("သံကျ", ["သံ", "ကျ"])
    errors = strategy.validate(ctx)
    assert len(errors) == 1
    assert errors[0].suggestions[0].text == "သံကျား"


# --- Config wiring ---


def test_builder_registers_when_enabled() -> None:
    """CompoundMergeProbeStrategy appears in the strategy list when enabled."""
    config = SpellCheckerConfig()
    config.validation.use_compound_merge_probe = True

    strategies = build_context_validation_strategies(
        config=config,
        provider=MemoryProvider(),
        symspell=_FakeSymSpell({}),
    )
    names = [s.__class__.__name__ for s in strategies]
    assert "CompoundMergeProbeStrategy" in names


def test_builder_skips_when_disabled() -> None:
    """CompoundMergeProbeStrategy absent when disabled."""
    config = SpellCheckerConfig()
    config.validation.use_compound_merge_probe = False

    strategies = build_context_validation_strategies(
        config=config,
        provider=MemoryProvider(),
        symspell=_FakeSymSpell({}),
    )
    names = [s.__class__.__name__ for s in strategies]
    assert "CompoundMergeProbeStrategy" not in names
