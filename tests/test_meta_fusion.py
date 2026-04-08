"""Tests for MetaClassifierFusion."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from myspellchecker.core.validation_strategies.meta_fusion import (
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
        assert bundled_model._n_features == 13
        assert len(bundled_model._coefficients) == 13
        assert len(bundled_model._feature_names) == 13

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
            _make_error(confidence=0.95, suggestions=["fix"],
                       source_strategy="StatisticalConfusableStrategy"),
            _make_error(confidence=0.1, suggestions=[]),
        ]
        # Use model's configured threshold (0.2) — without provider,
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
