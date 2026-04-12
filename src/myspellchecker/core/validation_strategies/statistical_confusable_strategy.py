"""Statistical confusable detection strategy.

Uses bidirectional bigram ratio comparison to detect confusable word
pairs. When the context strongly prefers the variant over the current
word, flags it as a confusable error.

No neural models — pure database lookups. ~0.3ms per confusable word.

Priority: 24 (within structural phase, before fast-path cutoff of 25)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from myspellchecker.core.constants import ET_CONFUSABLE_ERROR, ET_HOMOPHONE_ERROR
from myspellchecker.core.response import Error, WordError
from myspellchecker.core.validation_strategies.base import (
    ValidationContext,
    ValidationStrategy,
)
from myspellchecker.utils.logging_utils import get_logger

if TYPE_CHECKING:
    from myspellchecker.providers.interfaces import WordRepository

logger = get_logger(__name__)


class StatisticalConfusableStrategy(ValidationStrategy):
    """Detect confusable word pairs using bigram ratio comparison.

    For each word that has a known confusable variant (from
    confusable_pairs.yaml), compares bidirectional bigram
    probabilities: P(variant|prev) vs P(word|prev) and
    P(next|variant) vs P(next|word). When the ratio exceeds
    the threshold, the variant is the likely intended word.

    Independent of MLM and error budget.

    Priority: 24 (within structural phase to avoid fast-path skip)
    """

    # Within the structural phase (cutoff=25) so the fast-path optimisation
    # does not skip this strategy on otherwise-clean sentences.
    _PRIORITY = 24

    # Curated-pair promotion thresholds. A detection whose (word, variant)
    # is also in the homophone map and whose ratio clears the floor below
    # is emitted as ``homophone_error`` with a boosted confidence so it
    # can clear the downstream output filter for ``confusable_error``.
    _CURATED_PAIR_MIN_RATIO: float = 10.0
    _CURATED_PAIR_TARGET_CONFIDENCE: float = 0.80

    def __init__(
        self,
        provider: WordRepository,
        confusable_map: dict[str, set[str]],
        threshold: float = 5.0,
        confidence: float = 0.85,
        homophone_map: dict[str, set[str]] | None = None,
    ):
        self.provider = provider
        self._confusable_map = confusable_map
        self._threshold = threshold
        self._confidence = confidence
        self._all_confusable_words = frozenset(confusable_map.keys())
        self._homophone_map: dict[str, frozenset[str]] = (
            {k: frozenset(v) for k, v in homophone_map.items()} if homophone_map else {}
        )
        logger.debug(
            "StatisticalConfusableStrategy: %d words, threshold=%.1f, homophone_pairs=%d",
            len(self._all_confusable_words),
            threshold,
            len(self._homophone_map),
        )

    def validate(self, context: ValidationContext) -> list[Error]:
        """Check each word for confusable errors via bigram ratios."""
        if len(context.words) < 2:
            return []

        if not hasattr(self.provider, "get_bigram_probability"):
            return []

        errors: list[Error] = []

        for i, word in enumerate(context.words):
            pos_i = context.word_positions[i]

            if pos_i in context.existing_errors:
                continue
            if context.is_name_mask[i]:
                continue

            if word not in self._all_confusable_words:
                continue

            variants = self._confusable_map[word]
            prev_word = context.words[i - 1] if i > 0 else ""
            next_word = context.words[i + 1] if i + 1 < len(context.words) else ""

            best_variant = None
            best_ratio = 0.0

            for variant in variants:
                ratio = self._compute_bigram_ratio(
                    word,
                    variant,
                    prev_word,
                    next_word,
                )
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_variant = variant

            if best_variant and best_ratio >= self._threshold:
                conf = self._ratio_to_confidence(best_ratio)

                # Curated-pair promotion: a detection that is also in the
                # homophone map carries independent dictionary evidence on
                # top of the bigram-ratio signal. Promote it from a generic
                # confusable_error (downstream output filter requires
                # conf >= 0.75) to a homophone_error with a boosted
                # confidence so it clears the filter.
                curated = self._is_curated_homophone_pair(word, best_variant)
                if curated and best_ratio >= self._CURATED_PAIR_MIN_RATIO:
                    conf = max(conf, self._CURATED_PAIR_TARGET_CONFIDENCE)
                    error_type = ET_HOMOPHONE_ERROR
                else:
                    error_type = ET_CONFUSABLE_ERROR

                error = WordError(
                    text=word,
                    position=pos_i,
                    error_type=error_type,
                    suggestions=[best_variant],
                    confidence=conf,
                )
                errors.append(error)
                context.existing_errors[pos_i] = error_type
                context.existing_confidences[pos_i] = conf
                context.existing_suggestions[pos_i] = [best_variant]

                logger.debug(
                    "statistical_confusable: %s→%s ratio=%.1f conf=%.2f "
                    "curated=%s type=%s prev=%s next=%s",
                    word,
                    best_variant,
                    best_ratio,
                    conf,
                    curated,
                    error_type,
                    prev_word,
                    next_word,
                )

        return errors

    def _is_curated_homophone_pair(self, word: str, variant: str) -> bool:
        """Return True if (word, variant) is in the curated homophone map."""
        if not self._homophone_map:
            return False
        alts = self._homophone_map.get(word)
        if alts and variant in alts:
            return True
        return False

    def _compute_bigram_ratio(
        self,
        word: str,
        variant: str,
        prev_word: str,
        next_word: str,
    ) -> float:
        """Compute bidirectional bigram ratio favoring variant.

        Returns max(left_ratio, right_ratio) where:
        - left_ratio = P(variant|prev) / P(word|prev)
        - right_ratio = P(next|variant) / P(next|word)

        A ratio > 1.0 means the variant fits the context better.
        """
        best = 0.0

        if prev_word:
            p_word = self.provider.get_bigram_probability(
                prev_word,
                word,
            )
            p_variant = self.provider.get_bigram_probability(
                prev_word,
                variant,
            )
            if p_variant > 0:
                if p_word > 0:
                    ratio = p_variant / p_word
                else:
                    # Variant attested, word never attested in this context.
                    ratio = 1e6
                best = max(best, ratio)

        if next_word:
            p_word = self.provider.get_bigram_probability(
                word,
                next_word,
            )
            p_variant = self.provider.get_bigram_probability(
                variant,
                next_word,
            )
            if p_variant > 0:
                if p_word > 0:
                    ratio = p_variant / p_word
                else:
                    ratio = 1e6
                best = max(best, ratio)

        return best

    def _ratio_to_confidence(self, ratio: float) -> float:
        """Map a bigram ratio to a confidence score in ``[0.5, self._confidence]``.

        Uses ``0.5 + 0.45 * min(log10(ratio) / 6.0, 1.0)``. The ``/6.0``
        divisor is intentionally conservative: it produces ~0.63 for a
        ratio of 50 and ~0.95 only for ratios approaching 10⁶, which keeps
        mid-range emissions below the downstream output filter unless the
        curated-pair promotion path overrides them in :meth:`validate`.
        """
        if ratio <= 0:
            return 0.5
        log_ratio = math.log10(max(ratio, 1.0))
        conf = 0.5 + 0.45 * min(log_ratio / 6.0, 1.0)
        return min(conf, self._confidence)

    def priority(self) -> int:
        return self._PRIORITY

    def __repr__(self) -> str:
        return (
            f"StatisticalConfusableStrategy("
            f"priority={self._PRIORITY}, "
            f"words={len(self._all_confusable_words)}, "
            f"threshold={self._threshold})"
        )
