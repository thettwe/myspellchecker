"""Learned meta-classifier fusion for error detection.

Replaces hand-tuned Noisy-OR fusion with a logistic regression model
that predicts P(true_error) from per-position strategy features.

The model coefficients are loaded from a YAML file (no sklearn needed
at runtime).  Inference is a single dot product + sigmoid.
"""

from __future__ import annotations

import math
from pathlib import Path

import yaml

from myspellchecker.utils.logging_utils import get_logger

logger = get_logger(__name__)


_ERROR_TYPE_ENCODING = {
    "invalid_word": 1,
    "invalid_syllable": 2,
    "confusable_error": 3,
    "medial_confusion": 4,
    "pos_sequence_error": 5,
    "context_probability": 6,
    "broken_compound": 7,
    "tone_ambiguity": 8,
    "homophone_error": 9,
    "semantic_error": 10,
    "particle_confusion": 11,
    "syntax_error": 12,
    "register_mixing": 13,
    "question_structure": 14,
    "missing_asat": 15,
    "collocation_error": 16,
    "ha_htoe_confusion": 17,
    "medial_order_error": 18,
    "aspect_adverb_conflict": 19,
    "dangling_word": 20,
    "merged_sfp_conjunction": 21,
    "tense_mismatch": 22,
    "missing_conjunction": 23,
}

# Error types absent from the trained 23-dim one-hot vector.
# These are bypassed by filter_errors (trusted via strategy confidence) and
# stripped from the context features passed to score_error for trained errors.
_UNTRAINED_ERROR_TYPES: frozenset[str] = frozenset(
    {
        "hidden_compound_typo",
        "syllable_window_oov",
    }
)

# Strategy sources that enforce their own confidence gates and should
# bypass the meta-classifier regardless of their error_type.
_BYPASS_META_STRATEGIES: frozenset[str] = frozenset(
    {
        "LoanWordValidationStrategy",
        "VisargaStrategy",
    }
)

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

_KNOWN_PARTICLES = frozenset(
    {
        "က",
        "ကို",
        "မှာ",
        "တွင်",
        "၌",
        "သို့",
        "မှ",
        "နှင့်",
        "နဲ့",
        "သည်",
        "တယ်",
        "ပါ",
        "ပြီ",
        "ပြီး",
        "လျှင်",
        "လျှင့်",
        "လို့",
        "ရင်",
        "ကတည်းက",
        "တော့",
        "ပဲ",
        "ပေါ့",
        "ဘူး",
        "ဘဲ",
        "လား",
        "မလား",
        "သလား",
        "ပါသလား",
        "နော်",
        "လေ",
        "ခဲ့",
        "နေ",
        "ထား",
        "တတ်",
        "တတ်တယ်",
    }
)

_KNOWN_SUFFIXES = frozenset(
    {
        "များ",
        "တွေ",
        "တို့",
        "ခြင်း",
        "မှု",
        "ခု",
        "ယောက်",
        "ကောင်",
        "စင်",
        "ခွက်",
        "လုံး",
        "ပါး",
        "စု",
        "ဦး",
    }
)

_ERROR_TYPE_PRECISION = {
    "invalid_word": 0.46,
    "invalid_syllable": 0.80,
    "confusable_error": 0.54,
    "medial_confusion": 0.92,
    "pos_sequence_error": 0.20,
    "context_probability": 0.10,
    "broken_compound": 0.38,
    "tone_ambiguity": 0.09,
    "homophone_error": 0.38,
    "semantic_error": 0.36,
    "particle_confusion": 0.38,
    "syntax_error": 0.38,
    "register_mixing": 0.06,
    "question_structure": 0.20,
    "missing_asat": 0.67,
    "collocation_error": 0.12,
    "ha_htoe_confusion": 0.67,
    "medial_order_error": 1.00,
    "aspect_adverb_conflict": 0.50,
    "dangling_word": 0.08,
    "merged_sfp_conjunction": 0.05,
    "tense_mismatch": 0.42,
    "missing_conjunction": 0.05,
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

    def score_error(
        self,
        error: object,
        provider: object | None = None,
        all_errors: list | None = None,
        error_index: int = 0,
        normalized_text: str = "",
    ) -> float:
        """Score a single Error object. Returns P(true_error).

        Supports both 13-feature (legacy) and 41-feature (one-hot + context)
        models. Feature vector is matched to model expectations at runtime.
        """
        confidence = getattr(error, "confidence", 0.0)
        error_type = getattr(error, "error_type", "")
        suggestions = getattr(error, "suggestions", []) or []
        source_strategy = getattr(error, "source_strategy", "") or ""
        text = getattr(error, "text", "") or ""
        position = getattr(error, "position", 0)

        # --- Base features (6) ---
        features: list[float] = [
            round(confidence, 4),
            1.0 if suggestions else 0.0,
            float(min(len(suggestions), 10)),
            1.0 if source_strategy else 0.0,
            float(min(len(text), 30)),
            1.0 if confidence >= 0.85 else 0.0,
        ]

        # --- Frequency features (5) ---
        word_freq = 0
        top_sug_freq = 0
        try:
            if provider is not None:
                word_freq = provider.get_word_frequency(text) or 0
                if suggestions:
                    sug_text = (
                        suggestions[0].text
                        if hasattr(suggestions[0], "text")
                        else str(suggestions[0])
                    )
                    top_sug_freq = provider.get_word_frequency(sug_text) or 0
        except Exception:
            pass

        log_wf = math.log1p(word_freq)
        log_sf = math.log1p(top_sug_freq)
        features.extend(
            [
                log_wf,
                log_sf,
                min(log_sf / log_wf if log_wf > 0 else 0.0, 10.0),
                1.0 if word_freq > 0 else 0.0,
                1.0 if word_freq >= 5000 else 0.0,
            ]
        )

        # --- Source strategy binary flags (12) ---
        for sname in _STRATEGY_NAMES:
            features.append(1.0 if source_strategy == sname else 0.0)

        # --- One-hot error type (23) ---
        etype_id = _ERROR_TYPE_ENCODING.get(error_type, 0)
        for i in range(1, 24):
            features.append(1.0 if etype_id == i else 0.0)

        # --- Context features (6) ---
        errors = all_errors or []
        n_errors = len(errors)
        features.append(float(n_errors))
        features.append(float(sum(1 for e in errors if getattr(e, "error_type", "") == error_type)))
        other_confs = [
            getattr(e, "confidence", 0.0)
            for e in errors
            if getattr(e, "error_type", "") != error_type
        ]
        features.append(round(max(other_confs), 4) if other_confs else 0.0)
        prev_pos = [
            getattr(e, "position", 0) for e in errors if getattr(e, "position", 0) < position
        ]
        features.append(float(position - max(prev_pos)) if prev_pos else -1.0)
        features.append(round(position / len(normalized_text), 4) if normalized_text else 0.5)
        features.append(round(error_index / n_errors, 4) if n_errors > 0 else 0.5)

        # --- Morphological features (3) ---
        features.append(1.0 if text in _KNOWN_PARTICLES else 0.0)
        features.append(1.0 if text in _KNOWN_SUFFIXES else 0.0)
        syl_count = sum(1 for ch in text if 0x1000 <= ord(ch) <= 0x1021)
        features.append(float(min(syl_count, 10)))

        # --- Error-type precision baseline (1) ---
        features.append(_ERROR_TYPE_PRECISION.get(error_type, 0.2))

        if len(features) != self._n_features:
            return 0.5

        logit = self._intercept
        for i, f in enumerate(features):
            logit += self._coefficients[i] * f
        return _sigmoid(logit)

    def filter_errors(
        self,
        errors: list,
        threshold: float | None = None,
        provider: object | None = None,
        normalized_text: str = "",
    ) -> list:
        """Filter errors by meta-classifier score.

        Errors with types in ``_UNTRAINED_ERROR_TYPES`` are kept unconditionally;
        their emitting strategies enforce their own confidence gates and the
        classifier has no slot for them in its one-hot vector.

        Trained errors are scored against a context restricted to other trained
        errors so untrained types do not inflate ``n_errors`` / ``max_other_conf``.
        """
        if threshold is None:
            threshold = self._threshold

        trained_errors = [
            e
            for e in errors
            if getattr(e, "error_type", "") not in _UNTRAINED_ERROR_TYPES
            and getattr(e, "source_strategy", "") not in _BYPASS_META_STRATEGIES
        ]
        trained_count = len(trained_errors)

        kept = []
        trained_idx = 0
        for error in errors:
            error_type = getattr(error, "error_type", "")
            if error_type in _UNTRAINED_ERROR_TYPES:
                kept.append(error)
                continue

            # Strategies with built-in confidence gates bypass the meta-classifier.
            source = getattr(error, "source_strategy", "")
            if source in _BYPASS_META_STRATEGIES:
                kept.append(error)
                continue

            prob = self.score_error(
                error,
                provider=provider,
                all_errors=trained_errors,
                error_index=trained_idx if trained_count else 0,
                normalized_text=normalized_text,
            )
            trained_idx += 1

            if prob >= threshold:
                kept.append(error)
            else:
                logger.debug(
                    "meta_filter: suppressed %s at pos=%s (prob=%.3f < %.3f)",
                    error_type or "?",
                    getattr(error, "position", "?"),
                    prob,
                    threshold,
                )
        return kept
