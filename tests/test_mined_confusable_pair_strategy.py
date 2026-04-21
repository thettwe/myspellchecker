"""Tests for :class:`MinedConfusablePairStrategy`.

Covers:

* Constructor disables the strategy cleanly when ``enabled=False``,
  when ``backend='mlm'`` but ``semantic_checker is None``, and when the
  pair YAML is missing.
* YAML loading filters entries below ``low_freq_min`` and builds a
  symmetric partner map (either direction is a candidate).
* ``validate`` guards: skips empty contexts, name-masked tokens,
  short ``word_positions`` lists, tokens with no partner, tokens whose
  partners fail the frequency-ratio gate.
* Happy path: emits a ``WordError`` with the partner as suggestion
  when the MLM margin exceeds ``mlm_margin``.
* Margin gate: no error when partner MLM score is below threshold.
* Frequency cache is populated after a lookup.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from myspellchecker.core.constants import ET_CONFUSABLE_ERROR
from myspellchecker.core.response import WordError
from myspellchecker.core.validation_strategies.base import ValidationContext
from myspellchecker.core.validation_strategies.mined_confusable_pair_strategy import (
    MinedConfusablePairStrategy,
)
from myspellchecker.providers.memory import MemoryProvider


class _FakeSemantic:
    """Minimal stand-in for :class:`SemanticChecker` exposing
    ``score_mask_candidates`` with canned scores."""

    def __init__(self, scores_by_word: dict[str, dict[str, float]]) -> None:
        self._scores_by_word = scores_by_word

    def score_mask_candidates(
        self, sentence: str, word: str, candidates: list[str]
    ) -> dict[str, float]:
        return {c: self._scores_by_word.get(word, {}).get(c, 0.0) for c in candidates}


@pytest.fixture
def pair_yaml(tmp_path: Path) -> Path:
    """Write a minimal pair YAML with 2 real pairs + 1 that fails ``low_freq_min``."""
    path = tmp_path / "mined_pairs.yaml"
    data: dict[str, Any] = {
        "version": "1.0",
        "pairs": [
            {
                "high": "ရေး",
                "low": "လေး",
                "high_freq": 957_536,
                "low_freq": 648_208,
            },
            {
                "high": "သည်",
                "low": "သည့်",
                "high_freq": 4_845_813,
                "low_freq": 642_782,
            },
            {
                # Filtered out: low_freq < low_freq_min=100.
                "high": "အ",
                "low": "အာ",
                "high_freq": 9_000,
                "low_freq": 50,
            },
        ],
    }
    path.write_text(yaml.safe_dump(data, allow_unicode=True))
    return path


@pytest.fixture
def provider() -> MemoryProvider:
    p = MemoryProvider()
    p.add_word("ရေး", frequency=957_536)
    p.add_word("လေး", frequency=648_208)
    p.add_word("သည်", frequency=4_845_813)
    p.add_word("သည့်", frequency=642_782)
    # Low-frequency current word that should pass the freq-ratio gate
    # when we target ``ရေး`` as the partner (957k > 2 * 300).
    p.add_word("လေးရာ", frequency=300)
    return p


def _context(sentence: str, words: list[str]) -> ValidationContext:
    positions: list[int] = []
    cursor = 0
    for w in words:
        idx = sentence.find(w, cursor)
        assert idx >= 0, f"word {w!r} not in sentence"
        positions.append(idx)
        cursor = idx + len(w)
    return ValidationContext(sentence=sentence, words=words, word_positions=positions)


class TestConstructor:
    def test_disabled_flag_skips_loading(self, provider: MemoryProvider, pair_yaml: Path) -> None:
        strat = MinedConfusablePairStrategy(
            provider=provider,
            semantic_checker=_FakeSemantic({}),
            enabled=False,
            yaml_path=pair_yaml,
        )
        assert strat.enabled is False
        assert strat._partner_map == {}

    def test_mlm_backend_without_semantic_checker_disables(
        self, provider: MemoryProvider, pair_yaml: Path
    ) -> None:
        strat = MinedConfusablePairStrategy(
            provider=provider,
            semantic_checker=None,
            yaml_path=pair_yaml,
            backend="mlm",
        )
        assert strat.enabled is False

    def test_missing_yaml_disables(self, provider: MemoryProvider, tmp_path: Path) -> None:
        strat = MinedConfusablePairStrategy(
            provider=provider,
            semantic_checker=_FakeSemantic({}),
            yaml_path=tmp_path / "does_not_exist.yaml",
        )
        assert strat.enabled is False


class TestPartnerMap:
    def test_symmetric_and_low_freq_filter(self, provider: MemoryProvider, pair_yaml: Path) -> None:
        strat = MinedConfusablePairStrategy(
            provider=provider,
            semantic_checker=_FakeSemantic({}),
            yaml_path=pair_yaml,
        )
        # Both directions get entries.
        assert "ရေး" in strat._partner_map
        assert "လေး" in strat._partner_map
        assert ("လေး", 648_208) in strat._partner_map["ရေး"]
        assert ("ရေး", 957_536) in strat._partner_map["လေး"]
        # Entry below low_freq_min is filtered out.
        assert "အ" not in strat._partner_map


class TestValidateGuards:
    def test_empty_words_returns_empty(self, provider: MemoryProvider, pair_yaml: Path) -> None:
        strat = MinedConfusablePairStrategy(
            provider=provider,
            semantic_checker=_FakeSemantic({}),
            yaml_path=pair_yaml,
        )
        ctx = ValidationContext(sentence="", words=[], word_positions=[])
        assert strat.validate(ctx) == []

    def test_name_mask_skips(self, provider: MemoryProvider, pair_yaml: Path) -> None:
        scores = {"ရေး": {"လေး": 10.0, "ရေး": 0.0}}
        strat = MinedConfusablePairStrategy(
            provider=provider,
            semantic_checker=_FakeSemantic(scores),
            yaml_path=pair_yaml,
        )
        ctx = _context("ဦး ရေး ပါ", ["ဦး", "ရေး", "ပါ"])
        ctx.is_name_mask = [False, True, False]
        assert strat.validate(ctx) == []

    def test_token_without_partner_skipped(self, provider: MemoryProvider, pair_yaml: Path) -> None:
        strat = MinedConfusablePairStrategy(
            provider=provider,
            semantic_checker=_FakeSemantic({}),
            yaml_path=pair_yaml,
        )
        ctx = _context("အခုပဲ", ["အခုပဲ"])
        assert strat.validate(ctx) == []

    def test_freq_ratio_gate_suppresses(self, provider: MemoryProvider, pair_yaml: Path) -> None:
        """Partner must have freq >= ratio × current. When current is
        very high, no partner survives the gate."""
        scores = {"ရေး": {"လေး": 10.0, "ရေး": 0.0}}
        # ``ရေး`` has frequency 957k; its only partner ``လေး`` is at 648k,
        # which fails ``648k >= 2.0 × 957k``.
        strat = MinedConfusablePairStrategy(
            provider=provider,
            semantic_checker=_FakeSemantic(scores),
            yaml_path=pair_yaml,
            freq_ratio=2.0,
        )
        ctx = _context("ရေး ပါ", ["ရေး", "ပါ"])
        assert strat.validate(ctx) == []


class TestValidateHappyPath:
    def test_emits_word_error_when_margin_exceeded(
        self, provider: MemoryProvider, pair_yaml: Path
    ) -> None:
        # ``လေးရာ`` — unknown word, but shares the partner map via ``လေး``?
        # Not directly useful. Use ``လေး`` as current with ``ရေး`` as partner.
        # provider freqs: ရေး=957k, လေး=648k. ratio 957k >= 2.0 * 648k → True.
        scores = {"လေး": {"ရေး": 15.0, "လေး": 0.0}}  # margin 15 > default mlm_margin 2.5
        strat = MinedConfusablePairStrategy(
            provider=provider,
            semantic_checker=_FakeSemantic(scores),
            yaml_path=pair_yaml,
            mlm_margin=2.5,
            freq_ratio=1.0,
        )
        ctx = _context("သူ လေး တယ်", ["သူ", "လေး", "တယ်"])
        errors = strat.validate(ctx)
        assert len(errors) == 1
        err = errors[0]
        assert isinstance(err, WordError)
        assert err.error_type == ET_CONFUSABLE_ERROR
        assert err.text == "လေး"
        assert err.suggestions[0].text == "ရေး"
        # Confidence is clamped to 1.0 and proportional to margin/10.
        assert 0.0 < err.confidence <= 1.0

    def test_margin_below_threshold_no_error(
        self, provider: MemoryProvider, pair_yaml: Path
    ) -> None:
        scores = {"လေး": {"ရေး": 1.0, "လေး": 0.0}}  # margin 1 < mlm_margin 2.5
        strat = MinedConfusablePairStrategy(
            provider=provider,
            semantic_checker=_FakeSemantic(scores),
            yaml_path=pair_yaml,
            mlm_margin=2.5,
            freq_ratio=1.0,
        )
        ctx = _context("သူ လေး တယ်", ["သူ", "လေး", "တယ်"])
        assert strat.validate(ctx) == []


class TestFreqCache:
    def test_cache_populated_after_lookup(self, provider: MemoryProvider, pair_yaml: Path) -> None:
        strat = MinedConfusablePairStrategy(
            provider=provider,
            semantic_checker=_FakeSemantic({}),
            yaml_path=pair_yaml,
            freq_ratio=1.0,
        )
        assert strat._freq_cache == {}
        # Drive a validate() call so the cache gets populated via ``_unigram_freq``.
        ctx = _context("သူ လေး တယ်", ["သူ", "လေး", "တယ်"])
        strat.validate(ctx)
        assert "လေး" in strat._freq_cache
        assert strat._freq_cache["လေး"] == 648_208
