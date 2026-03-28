"""Context-aware suggestion re-ranking strategy.

Re-ranks existing suggestions using bidirectional N-gram context
probabilities with configurable left/right weights.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from myspellchecker.algorithms.ranker import SuggestionData
from myspellchecker.algorithms.suggestion_strategy import (
    BaseSuggestionStrategy,
    SuggestionContext,
    SuggestionResult,
    SuggestionStrategy,
)

if TYPE_CHECKING:
    from myspellchecker.algorithms.ngram_context_checker import NgramContextChecker


class ContextSuggestionStrategy(BaseSuggestionStrategy):
    """Strategy that re-ranks suggestions using bidirectional N-gram context.

    This strategy takes existing suggestions and re-ranks them based on
    contextual probability using both left context (prev_words) and right
    context (next_words) with configurable weights.

    Note: This strategy works best when composed with other strategies
    that generate the initial candidates.

    Example:
        >>> strategy = ContextSuggestionStrategy(
        ...     context_checker=checker,
        ...     left_weight=0.5,
        ...     right_weight=0.3,
        ... )
        >>> result = strategy.suggest("typo", context_with_prev_and_next_words)
    """

    def __init__(
        self,
        context_checker: "NgramContextChecker",
        base_strategy: "SuggestionStrategy | None" = None,
        max_suggestions: int = 5,
        left_weight: float = 0.5,
        right_weight: float = 0.3,
    ):
        """Initialize context strategy.

        Args:
            context_checker: NgramContextChecker for probability lookup.
            base_strategy: Optional base strategy to get initial candidates.
            max_suggestions: Maximum suggestions to return.
            left_weight: Weight for left context (prev_words) probability.
            right_weight: Weight for right context (next_words) probability.
        """
        super().__init__(max_suggestions=max_suggestions)
        self._context_checker = context_checker
        self._base_strategy = base_strategy
        self._left_weight = left_weight
        self._right_weight = right_weight

    @property
    def name(self) -> str:
        """Return the strategy identifier."""
        return "context"

    def supports_context(self) -> bool:
        """This strategy requires context to be effective."""
        return True

    def suggest(
        self,
        term: str,
        context: SuggestionContext | None = None,
    ) -> SuggestionResult:
        """Generate or re-rank suggestions using bidirectional context.

        If a base_strategy is provided, gets candidates from it first.
        Otherwise, returns empty result (context alone cannot generate).

        Uses both left context P(candidate|prev_word) and right context
        P(next_word|candidate) with configurable weights for re-ranking.

        Args:
            term: Term to generate suggestions for.
            context: Context information with prev_words/next_words.

        Returns:
            SuggestionResult with context-scored suggestions.
        """
        # Get base candidates if we have a base strategy
        if self._base_strategy:
            base_result = self._base_strategy.suggest(term, context)
            candidates = list(base_result.suggestions)
        else:
            # No base strategy - we can't generate candidates from context alone
            return self._create_result([])

        # If no context, return base results as-is
        if not context:
            return self._create_result(candidates)

        # Extract context words -- keep full lists for n-gram lookup, single
        # words for POS scoring which only needs immediate neighbors.
        prev_words = context.prev_words
        next_words = context.next_words
        prev_word = prev_words[-1] if prev_words else None
        next_word = next_words[0] if next_words else None

        # If no context words at all, return base results
        if not prev_word and not next_word:
            return self._create_result(candidates)

        # Phase 1: Collect raw n-gram probabilities for ALL candidates.
        # Uses trigram when 2+ context words are available, bigram otherwise.
        # This enables relative scoring -- "how much better is this candidate
        # vs others?" rather than absolute scoring that penalizes candidates
        # with zero n-gram coverage.
        raw_probs: list[tuple[float, float]] = []  # (left_prob, right_prob)
        for suggestion in candidates:
            left_prob = 0.0
            if prev_words:
                left_prob = self._context_checker.get_best_left_probability(
                    prev_words, suggestion.term
                )
            right_prob = 0.0
            if next_words:
                right_prob = self._context_checker.get_best_right_probability(
                    suggestion.term, next_words
                )
            raw_probs.append((left_prob, right_prob))

        # Phase 2: Compute relative context scores.
        # Use the median nonzero probability as the neutral baseline.
        # Candidates with zero bigram data get the median (neutral -- "no opinion"),
        # not a penalty floor. Candidates with real data get their actual score.
        all_left = [lp for lp, _ in raw_probs if lp > 1e-10]
        all_right = [rp for _, rp in raw_probs if rp > 1e-10]
        from statistics import median as _median

        median_left = _median(all_left) if all_left else 0.0
        median_right = _median(all_right) if all_right else 0.0

        scored_suggestions = []
        for suggestion, (left_prob, right_prob) in zip(candidates, raw_probs, strict=False):
            if suggestion.strategy_score is not None:
                base_score = suggestion.strategy_score
            else:
                base_score = float(suggestion.edit_distance)

            # Use actual probability when available, median when zero (neutral)
            effective_left = left_prob if left_prob > 1e-10 else median_left
            effective_right = right_prob if right_prob > 1e-10 else median_right

            # Convert to log scale for scoring
            left_log = math.log(effective_left) if effective_left > 1e-10 else 0.0
            right_log = math.log(effective_right) if effective_right > 1e-10 else 0.0

            context_adjustment = self._left_weight * left_log + self._right_weight * right_log
            context_score = base_score - context_adjustment

            # POS fit
            pos_fit = 0.0
            if prev_word and hasattr(self._context_checker, "calculate_pos_context_score"):
                pos_fit = self._context_checker.calculate_pos_context_score(
                    prev_word, suggestion.term, next_word
                )

            scored_suggestions.append(
                SuggestionData(
                    term=suggestion.term,
                    edit_distance=suggestion.edit_distance,
                    frequency=suggestion.frequency,
                    phonetic_score=suggestion.phonetic_score,
                    syllable_distance=suggestion.syllable_distance,
                    weighted_distance=suggestion.weighted_distance,
                    is_nasal_variant=suggestion.is_nasal_variant,
                    has_same_nasal_ending=suggestion.has_same_nasal_ending,
                    source="context",
                    confidence=min(1.0, suggestion.confidence + 0.1),
                    strategy_score=context_score,
                    pos_fit_score=pos_fit,
                    score_breakdown={
                        "base_score": base_score,
                        "left_prob": left_prob,
                        "right_prob": right_prob,
                        "effective_left": effective_left,
                        "effective_right": effective_right,
                        "context_adjustment": context_adjustment,
                        "pos_fit": pos_fit,
                        "left_context_depth": len(prev_words),
                        "right_context_depth": len(next_words),
                    },
                )
            )

        # Sort by strategy_score (lower is better)
        def sort_key(s: SuggestionData) -> float:
            """Return strategy score for sorting, defaulting to inf if absent."""
            return s.strategy_score if s.strategy_score is not None else float("inf")

        scored_suggestions.sort(key=sort_key)

        return self._create_result(
            scored_suggestions,
            metadata={
                "prev_word": prev_word,
                "next_word": next_word,
                "left_weight": self._left_weight,
                "right_weight": self._right_weight,
                "left_context_depth": len(prev_words),
                "right_context_depth": len(next_words),
            },
        )
