"""Visarga/aukmyit/asat correction strategy.

Detects words with missing or extra visarga (း), aukmyit (့), or
asat (်) and suggests the correct form.

Two detection paths:

1. **Tier 1 (curated table)**: Word is in ``visarga_corrections.yaml``
   — fire immediately with high confidence.

2. **Generative variant**: For every word, toggle visarga/aukmyit/asat
   to produce candidate corrections.  If a variant has much higher
   frequency in the production DB, flag as error.  Uses frequency
   ratio gating and optional bigram context (same pattern as
   LoanWordValidationStrategy).

Priority: 16 (after Orthography at 15, before LoanWord at 18).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from myspellchecker.core.constants import ET_CONFUSABLE_ERROR
from myspellchecker.core.response import Error, WordError
from myspellchecker.core.validation_strategies.base import (
    ValidationContext,
    ValidationStrategy,
)
from myspellchecker.utils.logging_utils import get_logger

if TYPE_CHECKING:
    from myspellchecker.providers.interfaces import WordRepository

logger = get_logger(__name__)

_RULES_DIR = Path(__file__).resolve().parents[2] / "rules"

# မှု / မူ confusion pair (very common in compound suffixes).
_MHU = "မှု"
_MU = "မူ"

# Frequency-ratio threshold for the generative မှု/မူ swap path.
_FREQ_RATIO_THRESHOLD = 10.0


@lru_cache(maxsize=1)
def _load_visarga_corrections() -> dict[str, dict]:
    """Load visarga correction table from YAML."""
    import yaml

    yaml_path = _RULES_DIR / "visarga_corrections.yaml"
    if not yaml_path.exists():
        logger.warning("visarga_corrections.yaml not found at %s", yaml_path)
        return {}

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    if not data or not isinstance(data, dict):
        return {}

    corrections: dict[str, dict] = {}
    for item in data.get("corrections", []):
        incorrect = item.get("incorrect")
        correct = item.get("correct")
        if not incorrect or not correct:
            continue
        corrections[incorrect] = {
            "correct": correct,
            "error_type": item.get("error_type", "missing_visarga"),
            "confidence": item.get("confidence", 0.90),
        }

    logger.debug("VisargaStrategy: loaded %d correction pairs", len(corrections))
    return corrections


class VisargaStrategy(ValidationStrategy):
    """Detect visarga/aukmyit/asat errors via curated table + generative variants.

    Path 1 (Tier 1): Curated YAML table — unconditional, high confidence.
    Path 2 (Generative): Toggle visarga/aukmyit/asat, check frequency
    ratio against the production DB.

    Priority: 16
    """

    _PRIORITY = 16

    def __init__(
        self,
        provider: WordRepository,
        confidence: float = 0.92,
    ) -> None:
        self.provider = provider
        self._confidence = confidence
        self._corrections: dict[str, dict] | None = None

    def _ensure_loaded(self) -> None:
        if self._corrections is not None:
            return
        self._corrections = _load_visarga_corrections()

    def validate(self, context: ValidationContext) -> list[Error]:
        self._ensure_loaded()
        assert self._corrections is not None

        errors: list[Error] = []

        for i, word in enumerate(context.words):
            pos_i = context.word_positions[i]

            if pos_i in context.existing_errors:
                continue
            if context.is_name_mask[i]:
                continue

            # Path 1: Curated table lookup.
            entry = self._corrections.get(word)
            if entry:
                correct = entry["correct"]
                conf = min(entry["confidence"], self._confidence)
                errors.append(self._make_error(word, pos_i, correct, conf, context))
                logger.debug(
                    "visarga_tier1: %s -> %s conf=%.2f",
                    word, correct, conf,
                )
                continue

            # Path 2: Generative variant detection.
            result = self._check_generative(word, context, i)
            if result:
                variant, score, vtype = result
                conf = min(score, self._confidence)
                errors.append(self._make_error(word, pos_i, variant, conf, context))
                logger.debug(
                    "visarga_generative: %s -> %s conf=%.2f type=%s",
                    word, variant, conf, vtype,
                )

        return errors

    def _make_error(
        self,
        word: str,
        position: int,
        correct: str,
        confidence: float,
        context: ValidationContext,
    ) -> WordError:
        """Create error and claim position."""
        context.existing_errors[position] = ET_CONFUSABLE_ERROR
        context.existing_confidences[position] = confidence
        context.existing_suggestions[position] = [correct]
        return WordError(
            text=word,
            position=position,
            error_type=ET_CONFUSABLE_ERROR,
            suggestions=[correct],
            confidence=confidence,
        )

    def _check_generative(
        self,
        word: str,
        context: ValidationContext,
        idx: int,
    ) -> tuple[str, float, str] | None:
        """Generate visarga/aukmyit/asat variants and check frequency ratios.

        Only the မှု/မူ swap uses the generative path — visarga/aukmyit/asat
        toggling is handled by the curated table (Tier 1) because the
        single-character toggle is too broad and SymSpell already catches
        most OOV cases via edit distance.

        Returns (best_variant, score, variant_type) or None.
        """
        word_freq = self.provider.get_word_frequency(word) or 0

        # Skip very high-frequency words.
        if word_freq >= 50_000:
            return None

        # Only generate မှု/မူ swap variants.  Visarga/aukmyit/asat
        # toggling is too noisy and mostly redundant with SymSpell.
        variants: list[tuple[str, str]] = []
        if _MU in word:
            variants.append((word.replace(_MU, _MHU, 1), "mhu_mu_confusion"))
        if _MHU in word:
            variants.append((word.replace(_MHU, _MU, 1), "mhu_mu_confusion"))

        if not variants:
            return None

        best: tuple[str, float, str] | None = None
        best_score = 0.0

        for variant, vtype in variants:
            var_freq = self.provider.get_word_frequency(variant) or 0
            if var_freq == 0:
                continue

            # Word is OOV → variant in DB → strong signal.
            if word_freq == 0:
                score = 0.88
                if score > best_score:
                    best_score = score
                    best = (variant, score, vtype)
                continue

            freq_ratio = var_freq / max(word_freq, 1)

            # Require strong frequency evidence for မှု/မူ swap.
            if freq_ratio >= _FREQ_RATIO_THRESHOLD:
                score = min(0.85, 0.70 + 0.05 * min(freq_ratio / 10, 3.0))
                if score > best_score:
                    best_score = score
                    best = (variant, score, vtype)

        return best

    def priority(self) -> int:
        return self._PRIORITY
