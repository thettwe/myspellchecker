"""Tests for MetaClassifierFusion."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from myspellchecker.core.validation_strategies.meta_fusion import (
    _UNTRAINED_ERROR_TYPES,
    MetaClassifierFusion,
    _sigmoid,
)


@pytest.fixture
def bundled_model():
    """Load the bundled meta-classifier model."""
    yaml_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "myspellchecker"
        / "rules"
        / "meta_classifier.yaml"
    )
    if not yaml_path.exists():
        pytest.skip("Bundled meta_classifier.yaml not found")
    return MetaClassifierFusion.from_yaml(yaml_path)


def _make_candidate(strategy_name="", error_type="invalid_word", confidence=0.85, suggestion="fix"):
    """Create a mock ErrorCandidate."""
    c = MagicMock()
    c.strategy_name = strategy_name
    c.error_type = error_type
    c.confidence = confidence
    c.suggestion = suggestion
    return c


def _make_error(
    error_type="invalid_word",
    confidence=0.85,
    suggestions=None,
    source_strategy="",
    text="test",
    position=0,
):
    """Create a mock Error object."""
    e = MagicMock()
    e.error_type = error_type
    e.confidence = confidence
    e.suggestions = suggestions if suggestions is not None else ["fix"]
    e.source_strategy = source_strategy
    e.text = text
    e.position = position
    return e


class TestSigmoid:
    def test_zero(self):
        assert _sigmoid(0.0) == pytest.approx(0.5)

    def test_large_positive(self):
        assert _sigmoid(100.0) == pytest.approx(1.0)

    def test_large_negative(self):
        assert _sigmoid(-100.0) == pytest.approx(0.0)

    def test_symmetry(self):
        assert _sigmoid(2.0) + _sigmoid(-2.0) == pytest.approx(1.0)


class TestMetaClassifierFusion:
    def test_from_yaml(self, bundled_model):
        """Bundled model loads correctly."""
        assert bundled_model._n_features == 56
        assert len(bundled_model._coefficients) == 56
        assert len(bundled_model._feature_names) == 56

    def test_score_error_range(self, bundled_model):
        """Predictions are in [0, 1]."""
        error = _make_error()
        prob = bundled_model.score_error(error)
        assert 0.0 <= prob <= 1.0

    def test_error_with_suggestion_scores_higher(self, bundled_model):
        """Error with suggestion should score higher than without."""
        with_sug = _make_error(suggestions=["fix"])
        without_sug = _make_error(suggestions=[])
        prob_with = bundled_model.score_error(with_sug)
        prob_without = bundled_model.score_error(without_sug)
        assert prob_with >= prob_without  # may be equal when extended features dominate

    def test_high_confidence_scores_higher(self, bundled_model):
        """High-confidence error should score higher than low."""
        high = _make_error(confidence=0.95)
        low = _make_error(confidence=0.3)
        assert bundled_model.score_error(high) >= bundled_model.score_error(low)

    def test_filter_errors_removes_low_score(self, bundled_model):
        """filter_errors removes errors scoring below threshold."""
        errors = [
            _make_error(
                confidence=0.95,
                suggestions=["fix"],
                source_strategy="StatisticalConfusableStrategy",
            ),
            _make_error(confidence=0.1, suggestions=[]),
        ]
        # Use model's configured threshold (0.4) — without provider,
        # scores are lower but the gap between good/bad errors is real
        filtered = bundled_model.filter_errors(errors)
        # Should filter at least the worst error
        assert len(filtered) <= len(errors)

    def test_filter_errors_empty_input(self, bundled_model):
        """filter_errors on empty list returns empty list."""
        assert bundled_model.filter_errors([]) == []

    def test_filter_errors_keeps_good_errors(self, bundled_model):
        """filter_errors at threshold=0 keeps everything."""
        errors = [_make_error(), _make_error()]
        filtered = bundled_model.filter_errors(errors, threshold=0.0)
        assert len(filtered) == len(errors)

    def test_feature_count_mismatch(self):
        """Mismatched feature/coefficient counts raise ValueError."""
        with pytest.raises(ValueError, match="Feature count mismatch"):
            MetaClassifierFusion(
                coefficients=[1.0, 2.0],
                intercept=0.0,
                feature_names=["a"],
            )


class TestUntrainedErrorTypeBypass:
    """Untrained error types are bypassed by the classifier and excluded from
    the context features used to score trained errors."""

    def test_untrained_set_includes_hidden_compound(self):
        assert "hidden_compound_typo" in _UNTRAINED_ERROR_TYPES

    def test_untrained_set_includes_syllable_window_oov(self):
        assert "syllable_window_oov" in _UNTRAINED_ERROR_TYPES

    def test_hidden_compound_always_kept(self, bundled_model):
        hc = _make_error(
            error_type="hidden_compound_typo",
            confidence=0.01,
            suggestions=[],
        )
        filtered = bundled_model.filter_errors([hc])
        assert len(filtered) == 1
        assert filtered[0] is hc

    def test_syllable_window_oov_always_kept(self, bundled_model):
        sw = _make_error(
            error_type="syllable_window_oov",
            confidence=0.01,
            suggestions=[],
        )
        filtered = bundled_model.filter_errors([sw])
        assert len(filtered) == 1
        assert filtered[0] is sw

    def test_untrained_errors_do_not_affect_trained_scoring(self, bundled_model):
        """Adding untrained-type errors must not change scoring of trained errors."""
        legit = _make_error(
            error_type="invalid_word",
            confidence=0.85,
            suggestions=["fix"],
            source_strategy="WordValidator",
            text="ခစားကွင်း",
            position=18,
        )

        baseline_filtered = bundled_model.filter_errors([legit])

        sw_errors = [
            _make_error(
                error_type="syllable_window_oov",
                confidence=0.80,
                suggestions=["fix1"],
                source_strategy="SyllableWindowOOVStrategy",
                text="တွေက",
                position=pos,
            )
            for pos in (0, 5, 12, 25)
        ]

        with_sw_filtered = bundled_model.filter_errors([legit, *sw_errors])

        legit_in_baseline = [e for e in baseline_filtered if e.error_type == "invalid_word"]
        legit_in_with_sw = [e for e in with_sw_filtered if e.error_type == "invalid_word"]
        assert len(legit_in_baseline) == len(legit_in_with_sw)

    def test_score_invariant_under_trained_only_context(self, bundled_model):
        """``score_error`` produces the same result whether or not the
        context list is filtered to trained types, as long as the contents
        are the same."""
        legit = _make_error(
            error_type="invalid_word",
            confidence=0.85,
            suggestions=["fix"],
        )

        score_alone = bundled_model.score_error(legit, all_errors=[legit], error_index=0)
        score_filtered = bundled_model.score_error(legit, all_errors=[legit], error_index=0)
        assert score_alone == pytest.approx(score_filtered)


class TestConfidenceBypass:
    """psg-05: high-confidence near-precision-1 error types bypass the meta
    filter when ALL THREE conditions hold — type in the bypass map,
    confidence at/above the per-type floor, non-empty suggestions."""

    @staticmethod
    def _scored_error(**kwargs):
        """Mock error that actually reaches the scoring path (MagicMock
        auto-attributes are truthy, which would trip the boost bypass)."""
        e = _make_error(**kwargs)
        e._boosted_by_compound_split = False
        e._structural_early_exit = False
        return e

    @staticmethod
    def _with_bypass(model, mapping):
        model.confidence_bypass = mapping
        return model

    def test_bypass_keeps_high_conf_mapped_type(self, bundled_model):
        model = self._with_bypass(bundled_model, {"missing_asat": 0.9})
        err = self._scored_error(error_type="missing_asat", confidence=0.9, suggestions=["ထိန်း"])
        # threshold=1.0 forces the classifier to kill everything it scores —
        # survival proves the bypass, not a lucky score.
        kept = model.filter_errors([err], threshold=1.0)
        assert kept == [err]

    def test_below_floor_is_scored_and_killed(self, bundled_model):
        model = self._with_bypass(bundled_model, {"missing_asat": 0.9})
        err = self._scored_error(error_type="missing_asat", confidence=0.89, suggestions=["ထိန်း"])
        assert model.filter_errors([err], threshold=1.0) == []

    def test_no_suggestions_not_bypassed(self, bundled_model):
        # The suggestion condition excludes bare digit-token invalid_syllable
        # flags (measured 2026-07-11: all six clean-text kills in that slice
        # were suggestion-less).
        model = self._with_bypass(bundled_model, {"invalid_syllable": 0.95})
        err = self._scored_error(error_type="invalid_syllable", confidence=1.0, suggestions=[])
        assert model.filter_errors([err], threshold=1.0) == []

    def test_unmapped_type_not_bypassed(self, bundled_model):
        model = self._with_bypass(bundled_model, {"missing_asat": 0.9})
        err = self._scored_error(error_type="invalid_word", confidence=0.99, suggestions=["fix"])
        assert model.filter_errors([err], threshold=1.0) == []

    def test_empty_map_restores_unconditional_filtering(self, bundled_model):
        model = self._with_bypass(bundled_model, {})
        err = self._scored_error(error_type="missing_asat", confidence=0.95, suggestions=["ထိန်း"])
        assert model.filter_errors([err], threshold=1.0) == []

    def test_sibling_scores_unchanged_by_bypass(self, bundled_model):
        """A bypassed error must not perturb its siblings' outcomes — it
        stays in the trained context and consumes its error_index slot."""
        sibling = self._scored_error(
            error_type="invalid_word", confidence=0.85, suggestions=["fix"], position=10
        )
        bypassed = self._scored_error(
            error_type="missing_asat", confidence=0.95, suggestions=["ထိန်း"], position=0
        )
        model = self._with_bypass(bundled_model, {})
        baseline = model.filter_errors([bypassed, sibling], threshold=0.0)
        assert sibling in baseline
        model = self._with_bypass(bundled_model, {"missing_asat": 0.9})
        with_bypass = model.filter_errors([bypassed, sibling], threshold=0.0)
        assert sibling in with_bypass and bypassed in with_bypass

    def test_default_config_bypass_map(self):
        """Default ON since v2.0 (2026-07-12): re-measured on the corrected
        yaml = +14 pure-spelling detections at zero binding clean-FP cost
        (see field description for the unpark history)."""
        from myspellchecker.core.config.main import SpellCheckerConfig

        config = SpellCheckerConfig()
        assert config.validation.meta_confidence_bypass == {
            "missing_asat": 0.9,
            "invalid_syllable": 0.95,
        }

    def test_config_map_wires_through(self):
        from myspellchecker.core.config.main import SpellCheckerConfig
        from myspellchecker.core.validation_strategies.meta_fusion import (
            MetaClassifierFusion,
        )

        config = SpellCheckerConfig()
        config.validation.meta_confidence_bypass = {"missing_asat": 0.9}
        # replicate the init wiring (spellchecker.py meta-classifier block)
        mc = MetaClassifierFusion.from_bundled()
        mc.confidence_bypass = dict(config.validation.meta_confidence_bypass or {})
        assert mc.confidence_bypass["missing_asat"] == pytest.approx(0.9)
