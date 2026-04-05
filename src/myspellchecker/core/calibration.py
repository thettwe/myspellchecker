"""Score calibration for multi-strategy confidence fusion.

Maps raw strategy confidence scores to calibrated probabilities so that
scores from different strategies are comparable on a common scale.

v1.3.0: bootstrapped with strategy-specific reliability weights and
identity calibration.  Future versions will use labeled benchmark data
to train per-strategy isotonic regression calibrators.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Per-strategy reliability weights (lambda_i).
# Estimated precision of each strategy based on tier hierarchy and
# observed accuracy characteristics.  These bootstrap values will be
# refined with labeled benchmark data in future versions.
STRATEGY_RELIABILITY: dict[str, float] = {
    # Tier 1: Deterministic -- high precision on valid inputs
    "ToneValidationStrategy": 0.85,
    "OrthographyValidationStrategy": 0.90,
    # Tier 2: Structural -- moderate precision
    "SyntacticValidationStrategy": 0.70,
    "StatisticalConfusableStrategy": 0.75,
    "BrokenCompoundStrategy": 0.80,
    # Tier 3: Contextual -- precision varies with context quality
    "POSSequenceValidationStrategy": 0.60,
    "QuestionStructureValidationStrategy": 0.55,
    "HomophoneValidationStrategy": 0.65,
    "NgramContextValidationStrategy": 0.65,
    # Tier 4: Neural -- highest accuracy but also highest FPR risk
    "ConfusableSemanticStrategy": 0.70,
    "SemanticValidationStrategy": 0.65,
}

DEFAULT_RELIABILITY: float = 0.5


@dataclass
class CalibrationData:
    """Piecewise-linear calibration mapping for a single strategy.

    Stores breakpoints ``(x_thresholds, y_thresholds)`` for piecewise
    linear interpolation.  When no labeled data is available, defaults
    to identity mapping (``raw == calibrated``).

    Both lists must be the same length and sorted in ascending order on
    ``x_thresholds``.
    """

    x_thresholds: list[float] = field(default_factory=lambda: [0.0, 1.0])
    y_thresholds: list[float] = field(default_factory=lambda: [0.0, 1.0])

    def __post_init__(self) -> None:
        if len(self.x_thresholds) != len(self.y_thresholds):
            raise ValueError(
                f"x_thresholds length ({len(self.x_thresholds)}) must match "
                f"y_thresholds length ({len(self.y_thresholds)})"
            )
        if len(self.x_thresholds) < 2:
            raise ValueError("CalibrationData requires at least 2 breakpoints")
        if self.x_thresholds != sorted(self.x_thresholds):
            raise ValueError("x_thresholds must be sorted in ascending order")

    def calibrate(self, raw_score: float) -> float:
        """Map a raw confidence score to a calibrated probability.

        Uses piecewise linear interpolation between breakpoints.
        Clamps to boundary values outside the defined range.
        """
        if raw_score <= self.x_thresholds[0]:
            return self.y_thresholds[0]
        if raw_score >= self.x_thresholds[-1]:
            return self.y_thresholds[-1]

        for i in range(len(self.x_thresholds) - 1):
            x0, x1 = self.x_thresholds[i], self.x_thresholds[i + 1]
            if x0 <= raw_score <= x1:
                if x1 == x0:
                    return self.y_thresholds[i]
                t = (raw_score - x0) / (x1 - x0)
                return self.y_thresholds[i] + t * (self.y_thresholds[i + 1] - self.y_thresholds[i])

        return raw_score  # Unreachable with valid breakpoints


# Identity calibration singleton (raw == calibrated).
_IDENTITY = CalibrationData()


class StrategyCalibrator:
    """Calibrate confidence scores across strategies.

    Holds per-strategy ``CalibrationData`` and reliability weights.
    Unknown strategies fall back to identity calibration and default
    reliability.

    Example::

        cal = StrategyCalibrator()
        raw = 0.9
        calibrated = cal.calibrate("OrthographyValidationStrategy", raw)
        reliability = cal.get_reliability("OrthographyValidationStrategy")
    """

    def __init__(
        self,
        calibration_data: dict[str, CalibrationData] | None = None,
        reliability: dict[str, float] | None = None,
    ):
        self._calibration_data: dict[str, CalibrationData] = calibration_data or {}
        self._reliability: dict[str, float] = (
            reliability if reliability is not None else dict(STRATEGY_RELIABILITY)
        )

    def calibrate(self, strategy_name: str, raw_score: float) -> float:
        """Return calibrated confidence for *strategy_name*'s raw score."""
        data = self._calibration_data.get(strategy_name, _IDENTITY)
        return data.calibrate(raw_score)

    def get_reliability(self, strategy_name: str) -> float:
        """Return the reliability weight (lambda) for *strategy_name*."""
        return self._reliability.get(strategy_name, DEFAULT_RELIABILITY)

    @classmethod
    def from_yaml(cls, path: str) -> "StrategyCalibrator":
        """Load calibration data and reliability weights from a YAML file.

        The YAML schema must have ``calibrations`` (per-strategy breakpoints)
        and ``reliability_weights`` (per-strategy lambda) keys, as produced
        by ``scripts/train_calibrators.py``.
        """
        import yaml  # noqa: PLC0415

        with open(path) as f:
            data = yaml.safe_load(f)

        cal_data: dict[str, CalibrationData] = {}
        for name, entry in data.get("calibrations", {}).items():
            x = entry.get("x_thresholds")
            y = entry.get("y_thresholds")
            if x and y and len(x) >= 2:
                cal_data[name] = CalibrationData(
                    x_thresholds=x, y_thresholds=y
                )

        weights = data.get("reliability_weights", {})
        return cls(calibration_data=cal_data, reliability=weights)
