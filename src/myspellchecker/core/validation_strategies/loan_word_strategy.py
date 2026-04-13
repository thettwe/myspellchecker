"""Loan word transliteration error detection strategy.

Detects words that are known loan word variants (non-standard
transliterations) and suggests the standard form.  Unlike SymSpell-level
detection, this strategy runs on ALL words — including those that are
valid in the production DB — because many common misspellings of loan
words (e.g. ဗွီဒီယို for "video") accumulate corpus frequency and
appear as "valid" words.

Uses bidirectional bigram context to disambiguate when the variant is
a high-frequency word.  For Tier 1 corrections (incorrect form is never
valid Myanmar), fires unconditionally.

Priority: 22 (structural phase, after SyntacticRule at 20, before
StatisticalConfusable at 24).
"""

from __future__ import annotations

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

# Minimum confidence for unconditional corrections (Tier 1).
_TIER1_CONFIDENCE = 0.92

# Minimum frequency ratio (standard/variant) to flag without bigram context.
_FREQ_RATIO_THRESHOLD = 3.0

# Bigram ratio threshold to flag a variant when both forms are common.
_BIGRAM_RATIO_THRESHOLD = 2.0

# Maximum word frequency for a variant to be flagged without bigram context.
# Variants above this threshold require bigram evidence to avoid FPs on
# extremely common words that happen to also be loan word variants.
_HIGH_FREQ_GUARD = 50_000


class LoanWordValidationStrategy(ValidationStrategy):
    """Detect loan word transliteration errors via variant lookup.

    Two detection paths:

    1. **Tier 1 (unconditional)**: Word is in ``loan_word_corrections``
       table — fire immediately with high confidence.  The incorrect
       form is never a valid Myanmar word.

    2. **Variant lookup**: Word is a known non-standard variant in
       ``loan_words.yaml``.  Compare frequency of variant vs standard
       form, and optionally check bigram context.

    Priority: 22
    """

    _PRIORITY = 22

    def __init__(
        self,
        provider: WordRepository,
        confidence: float = 0.85,
    ) -> None:
        self.provider = provider
        self._confidence = confidence

        # Lazy-load loan word data on first use.
        self._variant_to_standard: dict[str, set[str]] | None = None
        self._tier1_corrections: dict[str, dict] | None = None

    def _ensure_loaded(self) -> None:
        """Lazy-load loan word lookup tables on first call."""
        if self._variant_to_standard is not None:
            return

        from myspellchecker.core.loan_word_variants import _load_loan_word_data

        v2s, _ = _load_loan_word_data()
        self._variant_to_standard = dict(v2s)

        # Also load Tier 1 correction table for unconditional matches.
        try:
            from myspellchecker.grammar.config import get_grammar_config

            self._tier1_corrections = dict(get_grammar_config().loan_word_corrections)
        except Exception:
            self._tier1_corrections = {}

        logger.debug(
            "LoanWordValidationStrategy: %d variants, %d tier1 corrections",
            len(self._variant_to_standard),
            len(self._tier1_corrections or {}),
        )

    def validate(self, context: ValidationContext) -> list[Error]:
        """Check each word for loan word transliteration errors."""
        self._ensure_loaded()
        assert self._variant_to_standard is not None
        assert self._tier1_corrections is not None

        errors: list[Error] = []

        for i, word in enumerate(context.words):
            pos_i = context.word_positions[i]

            # Skip positions already flagged by higher-priority strategies.
            if pos_i in context.existing_errors:
                continue
            if context.is_name_mask[i]:
                continue

            # Path 1: Tier 1 unconditional correction.
            tier1 = self._tier1_corrections.get(word)
            if tier1:
                correct = tier1["correct"]
                conf = tier1.get("confidence", _TIER1_CONFIDENCE)
                error = WordError(
                    text=word,
                    position=pos_i,
                    error_type=ET_CONFUSABLE_ERROR,
                    suggestions=[correct],
                    confidence=conf,
                )
                errors.append(error)
                context.existing_errors[pos_i] = ET_CONFUSABLE_ERROR
                context.existing_confidences[pos_i] = conf
                context.existing_suggestions[pos_i] = [correct]
                logger.debug(
                    "loan_word_tier1: %s -> %s conf=%.2f",
                    word,
                    correct,
                    conf,
                )
                continue

            # Path 2: Known variant from loan_words.yaml.
            standards = self._variant_to_standard.get(word)
            if not standards:
                continue

            best_standard = None
            best_score = 0.0

            for std in standards:
                score = self._score_replacement(word, std, context, i)
                if score > best_score:
                    best_score = score
                    best_standard = std

            if best_standard and best_score > 0:
                conf = min(best_score, self._confidence)
                error = WordError(
                    text=word,
                    position=pos_i,
                    error_type=ET_CONFUSABLE_ERROR,
                    suggestions=[best_standard],
                    confidence=conf,
                )
                errors.append(error)
                context.existing_errors[pos_i] = ET_CONFUSABLE_ERROR
                context.existing_confidences[pos_i] = conf
                context.existing_suggestions[pos_i] = [best_standard]
                logger.debug(
                    "loan_word_variant: %s -> %s score=%.2f conf=%.2f",
                    word,
                    best_standard,
                    best_score,
                    conf,
                )

        return errors

    def _score_replacement(
        self,
        variant: str,
        standard: str,
        context: ValidationContext,
        idx: int,
    ) -> float:
        """Score how strongly the standard form should replace the variant.

        Returns a value in [0, 1] where 0 = don't replace, 1 = definitely replace.
        Uses frequency ratio and bigram context.
        """
        # Get frequencies.
        var_freq = self.provider.get_word_frequency(variant) or 0
        std_freq = self.provider.get_word_frequency(standard) or 0

        # Standard form not in DB → can't suggest it.
        if std_freq == 0:
            return 0.0

        # Variant not in DB (OOV) → definitely replace with standard.
        if var_freq == 0:
            return 0.90

        # Both in DB. Use frequency ratio.
        freq_ratio = std_freq / max(var_freq, 1)

        # High-frequency variant guard: very common words need strong evidence.
        if var_freq >= _HIGH_FREQ_GUARD:
            # Require bigram support for high-frequency variants.
            bigram_ratio = self._compute_bigram_ratio(variant, standard, context, idx)
            if bigram_ratio >= _BIGRAM_RATIO_THRESHOLD:
                return 0.80
            return 0.0  # Not enough evidence.

        # Standard is much more frequent → likely the intended form.
        if freq_ratio >= _FREQ_RATIO_THRESHOLD:
            return 0.85

        # Close frequencies → use bigram context for disambiguation.
        bigram_ratio = self._compute_bigram_ratio(variant, standard, context, idx)
        if bigram_ratio >= _BIGRAM_RATIO_THRESHOLD:
            return 0.75

        return 0.0  # Ambiguous, don't flag.

    def _compute_bigram_ratio(
        self,
        variant: str,
        standard: str,
        context: ValidationContext,
        idx: int,
    ) -> float:
        """Compute bidirectional bigram ratio favoring standard form.

        Returns max(left_ratio, right_ratio) where:
        - left_ratio = P(standard|prev) / P(variant|prev)
        - right_ratio = P(next|standard) / P(next|variant)
        """
        if not hasattr(self.provider, "get_bigram_probability"):
            return 0.0

        prev_word = context.words[idx - 1] if idx > 0 else ""
        next_word = context.words[idx + 1] if idx + 1 < len(context.words) else ""

        best = 0.0

        if prev_word:
            p_var = self.provider.get_bigram_probability(prev_word, variant)
            p_std = self.provider.get_bigram_probability(prev_word, standard)
            if p_std > 0 and p_var > 0:
                best = max(best, p_std / p_var)
            elif p_std > 0 and p_var == 0:
                best = max(best, 10.0)

        if next_word:
            p_var = self.provider.get_bigram_probability(variant, next_word)
            p_std = self.provider.get_bigram_probability(standard, next_word)
            if p_std > 0 and p_var > 0:
                best = max(best, p_std / p_var)
            elif p_std > 0 and p_var == 0:
                best = max(best, 10.0)

        return best

    def priority(self) -> int:
        return self._PRIORITY
