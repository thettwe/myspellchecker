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

    def test_y_thresholds_out_of_range_raises(self):
        with pytest.raises(ValueError, match="y_thresholds values must be in"):
            CalibrationData(x_thresholds=[0.0, 1.0], y_thresholds=[0.0, 1.5])

    def test_y_thresholds_negative_raises(self):
        with pytest.raises(ValueError, match="y_thresholds values must be in"):
            CalibrationData(x_thresholds=[0.0, 1.0], y_thresholds=[-0.1, 0.5])


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


class TestFromYaml:
    """Tests for StrategyCalibrator.from_yaml."""

    def test_load_valid_yaml(self, tmp_path):
        yaml_file = tmp_path / "cal.yaml"
        yaml_file.write_text(
            "calibrations:\n"
            "  TestStrategy:\n"
            "    x_thresholds: [0.0, 0.5, 1.0]\n"
            "    y_thresholds: [0.0, 0.3, 1.0]\n"
            "reliability_weights:\n"
            "  TestStrategy: 0.8\n"
        )
        cal = StrategyCalibrator.from_yaml(str(yaml_file))
        assert cal.calibrate("TestStrategy", 0.5) == pytest.approx(0.3)
        assert cal.get_reliability("TestStrategy") == 0.8

    def test_missing_file_returns_bootstrap(self):
        cal = StrategyCalibrator.from_yaml("/nonexistent/path.yaml")
        # Falls back to bootstrap defaults
        assert cal.get_reliability("OrthographyValidationStrategy") == 0.90

    def test_empty_yaml_returns_bootstrap(self, tmp_path):
        yaml_file = tmp_path / "empty.yaml"
        yaml_file.write_text("")
        cal = StrategyCalibrator.from_yaml(str(yaml_file))
        assert cal.get_reliability("OrthographyValidationStrategy") == 0.90

    def test_missing_reliability_section_uses_bootstrap(self, tmp_path):
        yaml_file = tmp_path / "no_weights.yaml"
        yaml_file.write_text(
            "calibrations:\n"
            "  TestStrategy:\n"
            "    x_thresholds: [0.0, 1.0]\n"
            "    y_thresholds: [0.0, 1.0]\n"
        )
        cal = StrategyCalibrator.from_yaml(str(yaml_file))
        # Should fall back to bootstrap, not empty dict
        assert cal.get_reliability("OrthographyValidationStrategy") == 0.90

    def test_mismatched_thresholds_skipped(self, tmp_path):
        yaml_file = tmp_path / "bad.yaml"
        yaml_file.write_text(
            "calibrations:\n"
            "  BadStrategy:\n"
            "    x_thresholds: [0.0, 0.5, 1.0]\n"
            "    y_thresholds: [0.0, 1.0]\n"
            "  GoodStrategy:\n"
            "    x_thresholds: [0.0, 1.0]\n"
            "    y_thresholds: [0.0, 1.0]\n"
        )
        cal = StrategyCalibrator.from_yaml(str(yaml_file))
        # BadStrategy skipped (length mismatch), GoodStrategy loaded
        assert cal.calibrate("BadStrategy", 0.5) == pytest.approx(0.5)  # identity
        assert cal.calibrate("GoodStrategy", 0.5) == pytest.approx(0.5)

    def test_unsorted_thresholds_skipped(self, tmp_path):
        yaml_file = tmp_path / "unsorted.yaml"
        yaml_file.write_text(
            "calibrations:\n"
            "  UnsortedStrategy:\n"
            "    x_thresholds: [1.0, 0.0]\n"
            "    y_thresholds: [1.0, 0.0]\n"
        )
        cal = StrategyCalibrator.from_yaml(str(yaml_file))
        # Unsorted skipped, falls back to identity
        assert cal.calibrate("UnsortedStrategy", 0.5) == pytest.approx(0.5)
