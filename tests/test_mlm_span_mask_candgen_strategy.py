"""Tests for :class:`MLMSpanMaskCandGenStrategy`.

Covers:

* Config wiring (default-off flag, env-override-compatible fields).
* Strategy registration via :func:`build_context_validation_strategies`.
* Happy path (real-word confusion: token and candidate both in dict).
* Guards: margin gate, ED-gate, disabled-flag no-op, ONNX-unavailable
  graceful degradation, existing-error skip, name-mask skip, high-freq
  skip, short-token skip.
"""

from __future__ import annotations

import pytest

from myspellchecker.core.config import SpellCheckerConfig
from myspellchecker.core.constants import ET_WORD
from myspellchecker.core.factories.builders import build_context_validation_strategies
from myspellchecker.core.validation_strategies import MLMSpanMaskCandGenStrategy
from myspellchecker.core.validation_strategies.base import ValidationContext
from myspellchecker.providers.memory import MemoryProvider


class _FakeSemanticChecker:
    """Minimal ``SemanticChecker`` stand-in for deterministic probe tests.

    ``predict_mask`` returns a canned top-K. ``score_mask_candidates``
    returns a canned {word: score} map. Each call is recorded.
    """

    def __init__(
        self,
        predictions: dict[tuple[str, str], list[tuple[str, float]]],
        scores: dict[tuple[str, str], dict[str, float]],
    ) -> None:
        self._predictions = predictions
        self._scores = scores
        self.predict_calls: list[tuple[str, str, int]] = []
        self.score_calls: list[tuple[str, str, tuple[str, ...]]] = []

    def predict_mask(
        self,
        sentence: str,
        target_word: str,
        top_k: int | None = None,
        occurrence: int = 0,
    ) -> list[tuple[str, float]]:
        self.predict_calls.append((sentence, target_word, int(top_k or 0)))
        return self._predictions.get((sentence, target_word), [])[: (top_k or 10)]

    def score_mask_candidates(
        self,
        sentence: str,
        target_word: str,
        candidates: list[str],
        occurrence: int = 0,
    ) -> dict[str, float]:
        self.score_calls.append((sentence, target_word, tuple(candidates)))
        base = self._scores.get((sentence, target_word), {})
        return {c: base[c] for c in candidates if c in base}


@pytest.fixture
def provider() -> MemoryProvider:
    p = MemoryProvider()
    # Real-word confusion pair: both forms are valid dict words.
    p.add_word("ကြောင်း", frequency=30_000)  # typo / ambiguous in context
    p.add_word("ကျောင်း", frequency=60_000)  # intended "school"
    # High-freq token used to exercise the skip_above_freq guard.
    p.add_word("သည်", frequency=200_000)
    # Another low-freq token for the multi-fire test.
    p.add_word("စာ", frequency=5_000)
    p.add_word("စား", frequency=40_000)
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
        config = SpellCheckerConfig()
        assert config.validation.use_mlm_span_mask_candgen is False
        assert config.validation.mlm_candgen_top_k == 10
        assert config.validation.mlm_candgen_margin == 2.0
        assert config.validation.mlm_candgen_max_ed == 2
        assert config.validation.mlm_candgen_skip_above_freq == 50_000
        assert config.validation.mlm_candgen_min_token_length == 2
        assert config.validation.mlm_candgen_confidence == 0.75

    def test_not_registered_without_semantic_checker(self, provider: MemoryProvider) -> None:
        """MLM requires the semantic checker; None => not registered."""
        config = SpellCheckerConfig()
        config.validation.use_mlm_span_mask_candgen = True
        strategies = build_context_validation_strategies(
            provider=provider,
            config=config,
            symspell=None,
            semantic_checker=None,
        )
        assert not any(isinstance(s, MLMSpanMaskCandGenStrategy) for s in strategies)


class TestHappyPath:
    def test_fires_when_margin_cleared(self, provider: MemoryProvider) -> None:
        """ကျောင်း wins over ကြောင်း by a logit margin > 2.0."""
        sentence = "သူ ကြောင်း သွားတယ်"  # "he went to [X]"
        fake = _FakeSemanticChecker(
            predictions={
                (sentence, "ကြောင်း"): [("ကျောင်း", 8.5), ("ကြောင်း", 5.0)],
            },
            scores={
                (sentence, "ကြောင်း"): {"ကြောင်း": 5.0, "ကျောင်း": 8.5},
            },
        )
        strategy = MLMSpanMaskCandGenStrategy(
            semantic_checker=fake, provider=provider, enabled=True
        )
        context = _context(sentence, ["သူ", "ကြောင်း", "သွားတယ်"])
        errors = strategy.validate(context)
        assert len(errors) == 1
        assert errors[0].text == "ကြောင်း"
        assert errors[0].error_type == ET_WORD
        assert errors[0].suggestions[0].text == "ကျောင်း"
        assert errors[0].suggestions[0].source == "mlm_span_mask_candgen"


class TestGuards:
    def test_margin_not_met(self, provider: MemoryProvider) -> None:
        """Margin < 2.0 → no emission."""
        sentence = "သူ ကြောင်း သွားတယ်"
        fake = _FakeSemanticChecker(
            predictions={(sentence, "ကြောင်း"): [("ကျောင်း", 6.0)]},
            scores={(sentence, "ကြောင်း"): {"ကြောင်း": 5.0, "ကျောင်း": 6.0}},
        )
        strategy = MLMSpanMaskCandGenStrategy(
            semantic_checker=fake, provider=provider, enabled=True, margin=2.0
        )
        context = _context(sentence, ["သူ", "ကြောင်း", "သွားတယ်"])
        assert strategy.validate(context) == []

    def test_edit_distance_exceeded(self, provider: MemoryProvider) -> None:
        """A high-scoring candidate with ED > max_edit_distance is rejected."""
        sentence = "သူ စာ ဖတ်တယ်"
        # 'ဘုန်းကြီး' has very different characters from 'စာ' — ED > 2.
        provider.add_word("ဘုန်းကြီး", frequency=20_000)
        fake = _FakeSemanticChecker(
            predictions={(sentence, "စာ"): [("ဘုန်းကြီး", 10.0)]},
            scores={(sentence, "စာ"): {"စာ": 3.0, "ဘုန်းကြီး": 10.0}},
        )
        strategy = MLMSpanMaskCandGenStrategy(
            semantic_checker=fake, provider=provider, enabled=True
        )
        context = _context(sentence, ["သူ", "စာ", "ဖတ်တယ်"])
        assert strategy.validate(context) == []

    def test_disabled_flag(self, provider: MemoryProvider) -> None:
        fake = _FakeSemanticChecker(predictions={}, scores={})
        strategy = MLMSpanMaskCandGenStrategy(
            semantic_checker=fake, provider=provider, enabled=False
        )
        context = _context("သူ ကြောင်း", ["သူ", "ကြောင်း"])
        assert strategy.validate(context) == []

    def test_semantic_checker_none(self, provider: MemoryProvider) -> None:
        """Graceful degradation when the ONNX backend is unavailable."""
        strategy = MLMSpanMaskCandGenStrategy(
            semantic_checker=None, provider=provider, enabled=True
        )
        context = _context("သူ ကြောင်း", ["သူ", "ကြောင်း"])
        assert strategy.validate(context) == []

    def test_existing_error_skipped(self, provider: MemoryProvider) -> None:
        """Position already claimed by a higher-priority strategy is skipped."""
        sentence = "သူ ကြောင်း သွားတယ်"
        fake = _FakeSemanticChecker(
            predictions={(sentence, "ကြောင်း"): [("ကျောင်း", 8.5)]},
            scores={(sentence, "ကြောင်း"): {"ကြောင်း": 5.0, "ကျောင်း": 8.5}},
        )
        strategy = MLMSpanMaskCandGenStrategy(
            semantic_checker=fake, provider=provider, enabled=True
        )
        context = _context(sentence, ["သူ", "ကြောင်း", "သွားတယ်"])
        # Mark the ကြောင်း position as already having an error.
        context.existing_errors[sentence.find("ကြောင်း")] = ET_WORD
        assert strategy.validate(context) == []

    def test_skip_above_freq(self, provider: MemoryProvider) -> None:
        """Very high-frequency tokens are not probed."""
        sentence = "သည်"  # freq 200k, above default 50k cap
        fake = _FakeSemanticChecker(predictions={}, scores={})
        strategy = MLMSpanMaskCandGenStrategy(
            semantic_checker=fake, provider=provider, enabled=True
        )
        context = _context(sentence, ["သည်"])
        assert strategy.validate(context) == []
        # predict_mask must not have been called.
        assert fake.predict_calls == []

    def test_oov_token_skipped(self, provider: MemoryProvider) -> None:
        """OOV tokens belong to SymSpell / raw-probe, not MLM-candgen."""
        sentence = "ဝလဒှ မြန်မာ"  # ဝလဒှ is OOV
        fake = _FakeSemanticChecker(predictions={}, scores={})
        strategy = MLMSpanMaskCandGenStrategy(
            semantic_checker=fake, provider=provider, enabled=True
        )
        context = _context(sentence, ["ဝလဒှ", "မြန်မာ"])
        strategy.validate(context)
        assert ("ဝလဒှ" in t for _, t, _ in fake.predict_calls) and all(
            t != "ဝလဒှ" for _, t, _ in fake.predict_calls
        )

    def test_short_token_skipped(self, provider: MemoryProvider) -> None:
        """Single-character tokens are below min_token_length."""
        sentence = "က"
        provider.add_word("က", frequency=1_000)
        fake = _FakeSemanticChecker(
            predictions={(sentence, "က"): [("ကျ", 10.0)]},
            scores={(sentence, "က"): {"က": 2.0, "ကျ": 10.0}},
        )
        provider.add_word("ကျ", frequency=5_000)
        strategy = MLMSpanMaskCandGenStrategy(
            semantic_checker=fake,
            provider=provider,
            enabled=True,
            min_token_length=2,
        )
        context = _context(sentence, ["က"])
        assert strategy.validate(context) == []

    def test_name_mask_skipped(self, provider: MemoryProvider) -> None:
        sentence = "ကြောင်း"
        fake = _FakeSemanticChecker(
            predictions={(sentence, "ကြောင်း"): [("ကျောင်း", 8.5)]},
            scores={(sentence, "ကြောင်း"): {"ကြောင်း": 5.0, "ကျောင်း": 8.5}},
        )
        strategy = MLMSpanMaskCandGenStrategy(
            semantic_checker=fake, provider=provider, enabled=True
        )
        context = _context(sentence, ["ကြောင်း"])
        context.is_name_mask = [True]
        assert strategy.validate(context) == []

    def test_empty_sentence(self, provider: MemoryProvider) -> None:
        fake = _FakeSemanticChecker(predictions={}, scores={})
        strategy = MLMSpanMaskCandGenStrategy(
            semantic_checker=fake, provider=provider, enabled=True
        )
        context = ValidationContext(sentence="", words=[], word_positions=[])
        assert strategy.validate(context) == []

    def test_predict_mask_exception_graceful(self, provider: MemoryProvider) -> None:
        """ONNX runtime errors become a no-op, not a crash."""

        class _BrokenChecker:
            def predict_mask(self, *args, **kwargs):  # noqa: D401
                raise RuntimeError("ONNX session died")

            def score_mask_candidates(self, *args, **kwargs):
                return {}

        strategy = MLMSpanMaskCandGenStrategy(
            semantic_checker=_BrokenChecker(), provider=provider, enabled=True
        )
        context = _context("ကြောင်း", ["ကြောင်း"])
        assert strategy.validate(context) == []
