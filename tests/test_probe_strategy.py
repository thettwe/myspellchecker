"""Tests for the v1.7.x probe-based neural strategies.

Covers:
- ProbeInferenceEngine load/score happy path (skipped if model not available)
- ProbeValidationStrategy + ProbeBoostedCompoundStrategy fire/no-fire behavior
- Config flag wiring through builders

Skips entire module if `models/probe-syllable-span-v1/` is not present
(which is the default state — model is gitignored, downloaded at deploy
time).
"""

from __future__ import annotations

from pathlib import Path

import pytest

PROBE_MODEL_PATH = Path("models/probe-syllable-span-v1")
PROBE_AVAILABLE = (
    PROBE_MODEL_PATH.exists()
    and (PROBE_MODEL_PATH / "head.pt").exists()
    and (PROBE_MODEL_PATH / "config.json").exists()
)

pytestmark = pytest.mark.skipif(
    not PROBE_AVAILABLE,
    reason=f"Probe model not present at {PROBE_MODEL_PATH}; skip integration tests.",
)


@pytest.fixture(scope="module")
def probe_engine():
    from myspellchecker.algorithms.probe.syllable_span_probe import (
        ProbeInferenceEngine,
    )

    return ProbeInferenceEngine(str(PROBE_MODEL_PATH))


@pytest.fixture(scope="module")
def provider():
    from myspellchecker.providers.sqlite import SQLiteProvider

    db_path = Path("data/mySpellChecker_production.db")
    if not db_path.exists():
        pytest.skip(f"Production DB not present at {db_path}")
    return SQLiteProvider(database_path=str(db_path))


def _build_context(text: str, provider):
    """Construct a ValidationContext from text using the production segmenter."""
    from myspellchecker.core.constants.myanmar_constants import contains_myanmar
    from myspellchecker.core.spellchecker import SpellChecker
    from myspellchecker.core.validation_strategies.base import ValidationContext
    from myspellchecker.core.validators.base import Validator

    checker = SpellChecker(provider=provider)
    words = checker.segmenter.segment_words(text)
    cursor = 0
    raw_positions = []
    for w in words:
        idx = text.find(w, cursor)
        if idx == -1:
            idx = cursor
        raw_positions.append(idx)
        cursor = idx + len(w)
    fwords: list[str] = []
    fpos: list[int] = []
    for w, p in zip(words, raw_positions, strict=False):
        if (
            w.strip()
            and not Validator.is_punctuation(w)
            and contains_myanmar(w, allow_extended=False)
        ):
            fwords.append(w)
            fpos.append(p)
    return ValidationContext(
        sentence=text,
        words=fwords,
        word_positions=fpos,
        is_name_mask=[False] * len(fwords),
    )


def test_inference_engine_loads_and_scores(probe_engine):
    """Engine returns per-syllable probabilities and matching syllable spans."""
    text = "မြန်မာနိုင်ငံ"
    probs, spans = probe_engine.score_sentence(text)
    assert len(probs) == len(spans)
    assert all(0.0 <= p <= 1.0 for p in probs)


def test_probe_strategy_fires_on_broken_compound(probe_engine, provider):
    """ProbeValidationStrategy emits an error on a known broken-compound FN."""
    from myspellchecker.core.validation_strategies.probe_strategy import (
        ProbeValidationStrategy,
    )

    strategy = ProbeValidationStrategy(engine=probe_engine, threshold=0.75, max_existing_errors=100)
    # BM-EXP-E003 — broken_compound FN
    text = "Delos သည် သေးငယ်သော စတိုးဆိုင်ဖြစ်ပြီး ထိုနေ ရာကို မသွားတော့ပါ။"
    ctx = _build_context(text, provider)
    errors = strategy.validate(ctx)
    assert len(errors) >= 1
    assert any(e.text == "နေ" for e in errors), (
        f"Expected fire on 'နေ', got: {[(e.position, e.text) for e in errors]}"
    )


def test_probe_strategy_silent_on_clean_text(probe_engine, provider):
    """Probe should not over-fire on clean Myanmar sentences."""
    from myspellchecker.core.validation_strategies.probe_strategy import (
        ProbeValidationStrategy,
    )

    strategy = ProbeValidationStrategy(engine=probe_engine, threshold=0.75, max_existing_errors=100)
    # Short clean sentence
    text = "သူသည်ကျောင်းသို့သွားသည်။"
    ctx = _build_context(text, provider)
    errors = strategy.validate(ctx)
    # We tolerate up to 1 false fire on a single clean sentence (the model
    # trained without per-sentence calibration); production aggregate FPR
    # is the real gate.
    assert len(errors) <= 1


def test_probe_boosted_compound_fires_with_dict_gate(probe_engine, provider):
    """ProbeBoostedCompoundStrategy fires on whitespace + dict hit."""
    from myspellchecker.core.validation_strategies.probe_boosted_compound_strategy import (
        ProbeBoostedCompoundStrategy,
    )

    strategy = ProbeBoostedCompoundStrategy(
        engine=probe_engine,
        provider=provider,
        threshold=0.7,
        compound_min_freq=50,
        max_existing_errors=100,
    )
    text = "Delos သည် သေးငယ်သော စတိုးဆိုင်ဖြစ်ပြီး ထိုနေ ရာကို မသွားတော့ပါ။"
    ctx = _build_context(text, provider)
    errors = strategy.validate(ctx)
    assert len(errors) == 1
    err = errors[0]
    assert err.text == "နေ ရာ"
    assert err.suggestions == ["နေရာ"]
    assert err.error_type == "broken_compound"
    assert err.source_strategy == "ProbeBoostedCompoundStrategy"


def test_probe_boosted_compound_silent_when_compound_not_in_dict(probe_engine, provider):
    """ProbeBoostedCompoundStrategy must NOT fire if merged form is not a known word."""
    from myspellchecker.core.validation_strategies.probe_boosted_compound_strategy import (
        ProbeBoostedCompoundStrategy,
    )

    strategy = ProbeBoostedCompoundStrategy(
        engine=probe_engine,
        provider=provider,
        threshold=0.7,
        compound_min_freq=50,
        max_existing_errors=100,
    )
    # Two unrelated words separated by space — concatenation is gibberish
    text = "ရှေ့ ပိတ်"
    ctx = _build_context(text, provider)
    errors = strategy.validate(ctx)
    # Either dict-gate rejected or probe didn't fire — both acceptable
    if errors:
        # If anything fires, it must be a real dict word
        for e in errors:
            assert provider.is_valid_word(e.suggestions[0])


def test_config_flags_wire_through_builder(provider):
    """Enabling probe flags via config registers both strategies."""
    from myspellchecker.core.config.main import SpellCheckerConfig
    from myspellchecker.core.spellchecker import SpellChecker

    config = SpellCheckerConfig()
    config.validation.use_probe_corrector = True
    config.validation.use_probe_compound = True
    config.validation.probe_model_path = str(PROBE_MODEL_PATH)
    checker = SpellChecker(config=config, provider=provider)
    classes = {s.__class__.__name__ for s in checker.context_validator.strategies}
    assert "ProbeValidationStrategy" in classes
    assert "ProbeBoostedCompoundStrategy" in classes


def test_not_registered_without_model_path(provider, caplog):
    """Probe flags are default-on since v1.9.0, but without `probe_model_path`
    the strategies degrade gracefully: none registered, and a warning makes
    the inert state visible (a silently-inert probe once masqueraded as a
    benchmark nondeterminism bug)."""
    import logging

    from myspellchecker.core.config.main import SpellCheckerConfig
    from myspellchecker.core.spellchecker import SpellChecker

    config = SpellCheckerConfig()
    assert config.validation.use_probe_corrector is True
    assert config.validation.use_probe_compound is True
    assert config.validation.use_probe_segmenter_rescue is True
    assert config.validation.probe_model_path is None
    with caplog.at_level(logging.WARNING):
        checker = SpellChecker(config=config, provider=provider)
    classes = {s.__class__.__name__ for s in checker.context_validator.strategies}
    assert "ProbeValidationStrategy" not in classes
    assert "ProbeBoostedCompoundStrategy" not in classes
    assert "ProbeSegmenterRescueStrategy" not in classes
    assert any("probe_model_path is not set" in r.message for r in caplog.records)


def test_probe_segmenter_rescue_runs_without_error(probe_engine, provider):
    """ProbeSegmenterRescueStrategy runs and returns a list (any cardinality)."""
    from myspellchecker.core.spellchecker import SpellChecker
    from myspellchecker.core.validation_strategies.probe_segmenter_rescue_strategy import (
        ProbeSegmenterRescueStrategy,
    )

    symspell = SpellChecker(provider=provider).symspell
    strategy = ProbeSegmenterRescueStrategy(
        engine=probe_engine,
        provider=provider,
        symspell=symspell,
        threshold=0.75,
        min_freq=2000,
        max_existing_errors=100,
    )
    text = "သံဂါအတွက် သူ စျေးကို စောစော သွားခဲ့တယ် လို့ မေမေက ပြောတယ်။"
    ctx = _build_context(text, provider)
    errors = strategy.validate(ctx)
    # The strategy may or may not fire on this specific sentence; only requirement
    # is that it runs without exception and returns a list of errors. Any errors
    # emitted must have non-empty suggestions and ed=1 dict-valid candidates.
    assert isinstance(errors, list)
    for e in errors:
        assert e.error_type == "broken_compound"
        assert e.source_strategy == "ProbeSegmenterRescueStrategy"
        assert len(e.suggestions) >= 1
        for s in e.suggestions:
            cand = s.text if hasattr(s, "text") else str(s)
            assert provider.is_valid_word(cand), f"Emitted suggestion {cand!r} is not in dict"


def test_probe_segmenter_rescue_silent_on_pali_stacking(probe_engine, provider):
    """ProbeSegmenterRescueStrategy must NOT fire on Pali stacking sequences."""
    from myspellchecker.core.spellchecker import SpellChecker
    from myspellchecker.core.validation_strategies.probe_segmenter_rescue_strategy import (
        ProbeSegmenterRescueStrategy,
    )

    # Use production-configured SymSpell (matches the shipping pipeline)
    symspell = SpellChecker(provider=provider).symspell

    strategy = ProbeSegmenterRescueStrategy(
        engine=probe_engine,
        provider=provider,
        symspell=symspell,
        threshold=0.75,
        min_freq=2000,
        max_existing_errors=100,
    )
    # Word containing Pali virama — should be excluded by linguistic gate
    text = "ဗုဒ္ဓဘာသာ ဆိုသည်မှာ"  # has ္ (virama)
    ctx = _build_context(text, provider)
    errors = strategy.validate(ctx)
    # The strategy should not flag any Pali-stacking adjacency
    for e in errors:
        assert "္" not in e.text, f"Should not fire on virama-containing text: {e.text}"


def test_all_three_probe_flags_register(provider):
    """Enabling all three probe flags registers all three strategies."""
    from myspellchecker.core.config.main import SpellCheckerConfig
    from myspellchecker.core.spellchecker import SpellChecker

    config = SpellCheckerConfig()
    config.validation.use_probe_corrector = True
    config.validation.use_probe_compound = True
    config.validation.use_probe_segmenter_rescue = True
    config.validation.probe_model_path = str(PROBE_MODEL_PATH)
    checker = SpellChecker(config=config, provider=provider)
    classes = {s.__class__.__name__ for s in checker.context_validator.strategies}
    assert "ProbeValidationStrategy" in classes
    assert "ProbeBoostedCompoundStrategy" in classes
    assert "ProbeSegmenterRescueStrategy" in classes
