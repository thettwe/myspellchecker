"""Unit tests for score calibration (Component 1)."""

import pytest

from myspellchecker.core.calibration import (
    DEFAULT_RELIABILITY,
    STRATEGY_RELIABILITY,
    CalibrationData,
    StrategyCalibrator,
)


class TestCalibrationData:
    """Tests for piecewise-linear calibration."""

    def test_identity_calibration(self):
        """Default identity mapping: raw == calibrated."""
        cal = CalibrationData()
        assert cal.calibrate(0.0) == 0.0
        assert cal.calibrate(0.5) == 0.5
        assert cal.calibrate(1.0) == 1.0

    def test_identity_midpoint(self):
        cal = CalibrationData()
        assert cal.calibrate(0.75) == pytest.approx(0.75)

    def test_custom_mapping(self):
        """Piecewise linear with 3 breakpoints."""
        cal = CalibrationData(
            x_thresholds=[0.0, 0.5, 1.0],
            y_thresholds=[0.0, 0.8, 1.0],
        )
        assert cal.calibrate(0.0) == 0.0
        assert cal.calibrate(0.5) == pytest.approx(0.8)
        assert cal.calibrate(1.0) == 1.0
        # Midpoint between first and second breakpoint
        assert cal.calibrate(0.25) == pytest.approx(0.4)
        # Midpoint between second and third breakpoint
        assert cal.calibrate(0.75) == pytest.approx(0.9)

    def test_clamps_below_range(self):
        cal = CalibrationData(
            x_thresholds=[0.2, 0.8],
            y_thresholds=[0.1, 0.9],
        )
        assert cal.calibrate(0.0) == 0.1
        assert cal.calibrate(0.1) == 0.1

    def test_clamps_above_range(self):
        cal = CalibrationData(
            x_thresholds=[0.2, 0.8],
            y_thresholds=[0.1, 0.9],
        )
        assert cal.calibrate(1.0) == 0.9

    def test_four_breakpoints(self):
        cal = CalibrationData(
            x_thresholds=[0.0, 0.3, 0.7, 1.0],
            y_thresholds=[0.0, 0.2, 0.9, 1.0],
        )
        assert cal.calibrate(0.0) == 0.0
        assert cal.calibrate(0.3) == pytest.approx(0.2)
        assert cal.calibrate(0.7) == pytest.approx(0.9)
        assert cal.calibrate(1.0) == 1.0
        # Between 0.3 and 0.7: linear interpolation
        assert cal.calibrate(0.5) == pytest.approx(0.55)

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="must match"):
            CalibrationData(x_thresholds=[0.0, 1.0], y_thresholds=[0.0])

    def test_too_few_breakpoints_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            CalibrationData(x_thresholds=[0.5], y_thresholds=[0.5])

    def test_unsorted_x_thresholds_raises(self):
        with pytest.raises(ValueError, match="sorted in ascending order"):
            CalibrationData(x_thresholds=[0.8, 0.2], y_thresholds=[0.9, 0.1])


class TestStrategyCalibrator:
    """Tests for the strategy calibrator."""

    def test_unknown_strategy_identity(self):
        cal = StrategyCalibrator()
        assert cal.calibrate("UnknownStrategy", 0.7) == pytest.approx(0.7)

    def test_known_strategy_with_custom_data(self):
        data = {
            "TestStrategy": CalibrationData(
                x_thresholds=[0.0, 0.5, 1.0],
                y_thresholds=[0.0, 0.3, 1.0],
            )
        }
        cal = StrategyCalibrator(calibration_data=data)
        assert cal.calibrate("TestStrategy", 0.5) == pytest.approx(0.3)
        # Unknown falls back to identity
        assert cal.calibrate("Other", 0.5) == pytest.approx(0.5)

    def test_reliability_defaults(self):
        cal = StrategyCalibrator()
        assert cal.get_reliability("OrthographyValidationStrategy") == 0.90
        assert cal.get_reliability("ConfusableSemanticStrategy") == 0.70
        assert cal.get_reliability("UnknownStrategy") == DEFAULT_RELIABILITY

    def test_custom_reliability(self):
        cal = StrategyCalibrator(reliability={"MyStrategy": 0.99})
        assert cal.get_reliability("MyStrategy") == 0.99
        assert cal.get_reliability("Other") == DEFAULT_RELIABILITY

    def test_all_known_strategies_have_reliability(self):
        """Every strategy in the tier system has a reliability weight."""
        from myspellchecker.core.validation_strategies.arbiter import STRATEGY_TIER

        for strategy_name in STRATEGY_TIER:
            assert strategy_name in STRATEGY_RELIABILITY, (
                f"{strategy_name} missing from STRATEGY_RELIABILITY"
            )
