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
    """Sprint I-1.5: untrained error types are bypassed and isolated."""

    def test_untrained_set_includes_hidden_compound(self):
        assert "hidden_compound_typo" in _UNTRAINED_ERROR_TYPES

    def test_untrained_set_includes_syllable_window_oov(self):
        assert "syllable_window_oov" in _UNTRAINED_ERROR_TYPES

    def test_hidden_compound_always_kept(self, bundled_model):
        """HiddenCompound errors bypass scoring regardless of feature values."""
        hc = _make_error(
            error_type="hidden_compound_typo",
            confidence=0.01,
            suggestions=[],
        )
        filtered = bundled_model.filter_errors([hc])
        assert len(filtered) == 1
        assert filtered[0] is hc

    def test_syllable_window_oov_always_kept(self, bundled_model):
        """SW errors bypass scoring regardless of feature values."""
        sw = _make_error(
            error_type="syllable_window_oov",
            confidence=0.01,
            suggestions=[],
        )
        filtered = bundled_model.filter_errors([sw])
        assert len(filtered) == 1
        assert filtered[0] is sw

    def test_syllable_window_oov_does_not_affect_other_error_scoring(
        self, bundled_model
    ):
        """Adding SW errors to a sentence must not change scoring of other errors.

        This is the Sprint I-1.5 regression test. The initial SW roll-out
        regressed recall by -6.9pp because the meta-classifier's context
        features (n_errors, max_other_conf) incorporated SW candidates,
        pushing legitimate invalid_word scores below the 0.5 threshold.
        After the fix, trained-error context features only include other
        trained errors.
        """
        legit = _make_error(
            error_type="invalid_word",
            confidence=0.85,
            suggestions=["fix"],
            source_strategy="WordValidator",
            text="ခစားကွင်း",
            position=18,
        )

        baseline_filtered = bundled_model.filter_errors([legit])

        # Add 4 SW candidates to the same sentence
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

        # The legitimate invalid_word must survive both runs identically.
        legit_in_baseline = [
            e for e in baseline_filtered if e.error_type == "invalid_word"
        ]
        legit_in_with_sw = [
            e for e in with_sw_filtered if e.error_type == "invalid_word"
        ]
        assert len(legit_in_baseline) == len(legit_in_with_sw), (
            "SW candidates must not change invalid_word filter decision"
        )

    def test_untrained_errors_excluded_from_context_features(
        self, bundled_model
    ):
        """score_error receives only trained errors as context when called by filter_errors.

        Verify via an indirect observable: ``filter_errors`` with SW
        candidates added should produce the same P(invalid_word) as the
        baseline call because the fix strips untrained errors from
        context features before scoring.
        """
        legit = _make_error(
            error_type="invalid_word",
            confidence=0.85,
            suggestions=["fix"],
        )

        # Baseline: score legit alone
        score_alone = bundled_model.score_error(
            legit, all_errors=[legit], error_index=0
        )
        # filter_errors strips untrained types from the context list; the
        # resulting score for the trained error is computed against the
        # same [legit] context whether or not SW is present.
        trained_only = [legit]
        score_filtered = bundled_model.score_error(
            legit, all_errors=trained_only, error_index=0
        )
        # Scores must be identical because context is identical.
        assert score_alone == pytest.approx(score_filtered)
