"""Learned meta-classifier fusion for error detection.

Replaces hand-tuned Noisy-OR fusion with a logistic regression model
that predicts P(true_error) from per-position strategy features.

The model coefficients are loaded from a YAML file (no sklearn needed
at runtime).  Inference is a single dot product + sigmoid.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from myspellchecker.core.validation_strategies.arbiter import (
    STRATEGY_TIER,
    select_winner,
)
from myspellchecker.utils.logging_utils import get_logger

if TYPE_CHECKING:
    from myspellchecker.core.validation_strategies.base import ErrorCandidate

logger = get_logger(__name__)

# Canonical strategy order matching the training feature extraction.
_STRATEGY_NAMES = [
    "ToneValidationStrategy",
    "OrthographyValidationStrategy",
    "SyntacticValidationStrategy",
    "StatisticalConfusableStrategy",
    "BrokenCompoundStrategy",
    "POSSequenceValidationStrategy",
    "QuestionStructureValidationStrategy",
    "HomophoneValidationStrategy",
    "NgramContextValidationStrategy",
    "ConfusableCompoundClassifierStrategy",
    "ConfusableSemanticStrategy",
    "SemanticValidationStrategy",
]

_STRATEGY_SHORT_NAMES = [
    s.replace("ValidationStrategy", "").replace("Strategy", "")
    for s in _STRATEGY_NAMES
]


_ERROR_TYPE_ENCODING = {
    "invalid_word": 1, "invalid_syllable": 2, "confusable_error": 3,
    "medial_confusion": 4, "pos_sequence_error": 5, "context_probability": 6,
    "broken_compound": 7, "tone_ambiguity": 8, "homophone_error": 9,
    "semantic_error": 10, "particle_confusion": 11, "syntax_error": 12,
    "register_mixing": 13, "question_structure": 14, "missing_asat": 15,
    "collocation_error": 16, "ha_htoe_confusion": 17, "medial_order_error": 18,
    "aspect_adverb_conflict": 19, "dangling_word": 20,
    "merged_sfp_conjunction": 21, "tense_mismatch": 22,
    "missing_conjunction": 23,
}

_STRATEGY_ENCODING = {
    "": 0, "ToneValidationStrategy": 1, "OrthographyValidationStrategy": 2,
    "SyntacticValidationStrategy": 3, "StatisticalConfusableStrategy": 4,
    "BrokenCompoundStrategy": 5, "POSSequenceValidationStrategy": 6,
    "QuestionStructureValidationStrategy": 7, "HomophoneValidationStrategy": 8,
    "NgramContextValidationStrategy": 9,
    "ConfusableCompoundClassifierStrategy": 10,
    "ConfusableSemanticStrategy": 11, "SemanticValidationStrategy": 12,
}


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


class MetaClassifierFusion:
    """Learned meta-classifier fusion replacing Noisy-OR.

    Loads logistic regression coefficients from YAML and predicts
    P(true_error) per position using strategy-level features.

    Same input/output contract as ``arbiter.fuse_all_candidates()``.
    """

    def __init__(
        self,
        coefficients: list[float],
        intercept: float,
        feature_names: list[str],
        threshold: float = 0.5,
    ) -> None:
        self._coefficients = coefficients
        self._intercept = intercept
        self._feature_names = feature_names
        self._threshold = threshold
        self._n_features = len(coefficients)

        if len(feature_names) != len(coefficients):
            raise ValueError(
                f"Feature count mismatch: {len(feature_names)} names vs "
                f"{len(coefficients)} coefficients"
            )

    @classmethod
    def from_yaml(cls, path: str | Path) -> MetaClassifierFusion:
        """Load meta-classifier from YAML (produced by train_meta_classifier.py)."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Meta-classifier YAML not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        return cls(
            coefficients=data["coefficients"],
            intercept=data["intercept"],
            feature_names=data["features"],
            threshold=data.get("threshold", 0.5),
        )

    @classmethod
    def from_bundled(cls) -> MetaClassifierFusion:
        """Load from the bundled rules/meta_classifier.yaml."""
        bundled = Path(__file__).resolve().parents[2] / "rules" / "meta_classifier.yaml"
        return cls.from_yaml(bundled)

    def _extract_features(
        self, candidates: list[ErrorCandidate]
    ) -> list[float]:
        """Extract feature vector from candidates at a single position."""
        # Strategy binary + confidence features
        fired = {s: 0.0 for s in _STRATEGY_NAMES}
        conf = {s: 0.0 for s in _STRATEGY_NAMES}

        for c in candidates:
            if c.strategy_name in fired:
                fired[c.strategy_name] = 1.0
                conf[c.strategy_name] = max(conf[c.strategy_name], c.confidence)

        # Aggregate features
        named = [c for c in candidates if c.strategy_name in STRATEGY_TIER]
        agreement_count = len(named)
        all_confs = [c.confidence for c in candidates]
        max_confidence = max(all_confs) if all_confs else 0.0
        all_tiers = [STRATEGY_TIER.get(c.strategy_name, 2) for c in named]
        max_tier = max(all_tiers) if all_tiers else 0

        error_types = {c.error_type for c in candidates}
        error_type_count = len(error_types)
        has_suggestion = 1.0 if any(c.suggestion for c in candidates) else 0.0

        from collections import Counter

        et_counts = Counter(c.error_type for c in candidates)
        dominant = et_counts.most_common(1)[0][0] if et_counts else ""
        dominant_encoded = float(_ERROR_TYPE_ENCODING.get(dominant, 0))

        # Build feature vector in canonical order
        features: list[float] = []
        for s in _STRATEGY_NAMES:
            features.append(fired[s])
        for s in _STRATEGY_NAMES:
            features.append(conf[s])
        features.extend([
            float(agreement_count),
            max_confidence,
            float(max_tier),
            float(error_type_count),
            has_suggestion,
            dominant_encoded,
        ])

        return features

    def predict_proba(self, candidates: list[ErrorCandidate]) -> float:
        """Predict P(true_error) for candidates at a single position."""
        features = self._extract_features(candidates)
        if len(features) != self._n_features:
            logger.warning(
                "Feature count mismatch: expected %d, got %d; falling back to 0.5",
                self._n_features,
                len(features),
            )
            return 0.5

        logit = self._intercept
        for i, f in enumerate(features):
            logit += self._coefficients[i] * f
        return _sigmoid(logit)

    def score_error(self, error: object, provider: object | None = None) -> float:
        """Score a single Error object. Returns P(true_error).

        Works with Error objects from the validation pipeline (not ErrorCandidate).
        Extracts features from the error's attributes. When a provider is given,
        enriches with word frequency data.
        """
        confidence = getattr(error, "confidence", 0.0)
        error_type = getattr(error, "error_type", "")
        suggestions = getattr(error, "suggestions", []) or []
        source_strategy = getattr(error, "source_strategy", "") or ""
        text = getattr(error, "text", "") or ""

        # Base 8 features (always available)
        base_features = [
            round(confidence, 4),
            float(_ERROR_TYPE_ENCODING.get(error_type, 0)),
            1.0 if suggestions else 0.0,
            float(min(len(suggestions), 10)),
            1.0 if source_strategy else 0.0,
            float(min(len(text), 30)),
            1.0 if confidence >= 0.85 else 0.0,
            float(_STRATEGY_ENCODING.get(source_strategy, 0)),
        ]

        # Extended features (when model expects them)
        if self._n_features > 8:
            word_freq = 0
            top_suggestion_freq = 0
            try:
                word_freq = provider.get_word_frequency(text) or 0
                if suggestions:
                    sug_text = suggestions[0].text if hasattr(suggestions[0], "text") else str(suggestions[0])
                    top_suggestion_freq = provider.get_word_frequency(sug_text) or 0
            except Exception:
                pass

            import math
            log_word_freq = math.log1p(word_freq)
            log_sug_freq = math.log1p(top_suggestion_freq)
            freq_ratio = log_sug_freq / log_word_freq if log_word_freq > 0 else 0.0
            is_in_dict = 1.0 if word_freq > 0 else 0.0
            is_high_freq = 1.0 if word_freq >= 5000 else 0.0

            base_features.extend([
                log_word_freq,
                log_sug_freq,
                min(freq_ratio, 10.0),
                is_in_dict,
                is_high_freq,
            ])

        if len(base_features) != self._n_features:
            # Feature count mismatch — use only what we can
            if len(base_features) > self._n_features:
                base_features = base_features[: self._n_features]
            else:
                return 0.5  # fallback

        logit = self._intercept
        for i, f in enumerate(base_features):
            logit += self._coefficients[i] * f
        return _sigmoid(logit)

    def filter_errors(
        self,
        errors: list,
        threshold: float | None = None,
        provider: object | None = None,
    ) -> list:
        """Filter errors by meta-classifier score.

        Removes errors that the classifier predicts are likely false positives.

        Args:
            errors: List of Error objects from the validation pipeline.
            threshold: Minimum P(true_error) to keep. Defaults to model threshold.
            provider: Optional DictionaryProvider for word frequency features.

        Returns:
            Filtered list of errors (same type as input).
        """
        if threshold is None:
            threshold = self._threshold

        kept = []
        for error in errors:
            prob = self.score_error(error, provider=provider)
            if prob >= threshold:
                kept.append(error)
            else:
                logger.debug(
                    "meta_filter: suppressed %s at pos=%s (prob=%.3f < %.3f)",
                    getattr(error, "error_type", "?"),
                    getattr(error, "position", "?"),
                    prob,
                    threshold,
                )
        return kept

    def fuse_all_candidates(
        self,
        error_candidates: dict[int, list[ErrorCandidate]],
        threshold: float | None = None,
    ) -> dict[int, tuple[float, ErrorCandidate]]:
        """Fuse candidates at all positions using the meta-classifier.

        Same contract as ``arbiter.fuse_all_candidates()``.

        Args:
            error_candidates: Position → list of ErrorCandidates.
            threshold: Minimum probability to include. Defaults to model threshold.

        Returns:
            Position → (probability, winning ErrorCandidate).
        """
        if threshold is None:
            threshold = self._threshold

        result: dict[int, tuple[float, ErrorCandidate]] = {}

        for pos, candidates in error_candidates.items():
            if not candidates:
                continue

            prob = self.predict_proba(candidates)
            if prob >= threshold:
                winner = select_winner(candidates)
                result[pos] = (prob, winner)

        return result
