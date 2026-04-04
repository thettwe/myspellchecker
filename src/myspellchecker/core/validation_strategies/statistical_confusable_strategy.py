"""Statistical confusable detection strategy.

Uses bidirectional bigram ratio comparison to detect confusable word
pairs. When the context strongly prefers the variant over the current
word, flags it as a confusable error.

No neural models — pure database lookups. ~0.3ms per confusable word.

Priority: 47 (after Homophone 45, before ConfusableSemantic 48)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from myspellchecker.core.response import Error, WordError
from myspellchecker.core.validation_strategies.base import (
    ValidationContext,
    ValidationStrategy,
)
from myspellchecker.utils.logging_utils import get_logger

if TYPE_CHECKING:
    from myspellchecker.providers.interfaces import WordRepository

logger = get_logger(__name__)

# Error type for confusable detections
_ERROR_TYPE = "confusable_error"


class StatisticalConfusableStrategy(ValidationStrategy):
    """Detect confusable word pairs using bigram ratio comparison.

    For each word that has a known confusable variant (from
    confusable_pairs.yaml), compares bidirectional bigram
    probabilities: P(variant|prev) vs P(word|prev) and
    P(next|variant) vs P(next|word). When the ratio exceeds
    the threshold, the variant is the likely intended word.

    Independent of MLM and error budget.

    Priority: 47
    """

    # Priority 24: within the structural phase (cutoff=25) so the
    # fast-path optimization does NOT skip this strategy on clean
    # sentences. Confusable errors are "structurally clean" text
    # that contains the wrong valid word — they MUST be checked
    # even when no structural errors are found.
    _PRIORITY = 24

    def __init__(
        self,
        provider: WordRepository,
        confusable_map: dict[str, set[str]],
        threshold: float = 5.0,
        confidence: float = 0.85,
    ):
        self.provider = provider
        self._confusable_map = confusable_map
        self._threshold = threshold
        self._confidence = confidence
        self._all_confusable_words = frozenset(confusable_map.keys())
        logger.info(
            "StatisticalConfusableStrategy: %d words, threshold=%.1f",
            len(self._all_confusable_words),
            threshold,
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

            # Skip already flagged — UNLESS the existing error is a
            # POS sequence error (which might actually be a confusable
            # that the POS tagger couldn't resolve). Confusable detection
            # can provide a better suggestion than "wrong POS".
            if pos_i in context.existing_errors:
                existing_type = context.existing_errors[pos_i]
                if existing_type != "pos_sequence_error":
                    continue
            if context.is_name_mask[i]:
                continue

            # Only check words in confusable map
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
                error = WordError(
                    text=word,
                    position=pos_i,
                    error_type=_ERROR_TYPE,
                    suggestions=[best_variant],
                    confidence=conf,
                )
                errors.append(error)
                context.existing_errors[pos_i] = _ERROR_TYPE
                context.existing_confidences[pos_i] = conf

                logger.debug(
                    "statistical_confusable: %s→%s ratio=%.1f conf=%.2f prev=%s next=%s",
                    word,
                    best_variant,
                    best_ratio,
                    conf,
                    prev_word,
                    next_word,
                )

        return errors

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
                    # Word has zero prob, variant has nonzero
                    # → strong signal (use a large but finite ratio)
                    ratio = p_variant * 1e6
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
                    ratio = p_variant * 1e6
                best = max(best, ratio)

        return best

    def _ratio_to_confidence(self, ratio: float) -> float:
        """Map bigram ratio to confidence score [0.5, 0.95]."""
        # sigmoid-like scaling: ratio=5 → 0.70, ratio=50 → 0.88,
        # ratio=1000+ → 0.95
        if ratio <= 0:
            return 0.5
        log_ratio = math.log10(max(ratio, 1.0))
        # Scale: log10(5)=0.7→0.70, log10(50)=1.7→0.88, log10(1e6)=6→0.95
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
