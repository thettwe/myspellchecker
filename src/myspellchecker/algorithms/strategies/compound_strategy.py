"""Compound suggestion strategy.

Generates suggestions using SymSpell's lookup_compound for multi-word
corrections, handling word boundary ambiguity in Myanmar text.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from myspellchecker.algorithms.ranker import SuggestionData
from myspellchecker.algorithms.suggestion_strategy import (
    BaseSuggestionStrategy,
    SuggestionContext,
    SuggestionResult,
)

if TYPE_CHECKING:
    from myspellchecker.algorithms.symspell import SymSpell


class CompoundSuggestionStrategy(BaseSuggestionStrategy):
    """Strategy that generates compound suggestions using SymSpell lookup_compound.

    This strategy handles multi-word corrections by finding optimal segmentations
    and corrections for compound terms. Particularly useful for Myanmar text where
    word boundaries can be ambiguous.

    Example:
        >>> strategy = CompoundSuggestionStrategy(symspell=symspell_instance)
        >>> result = strategy.suggest("compoundterm")
        >>> for s in result.suggestions:
        ...     print(f"{s.term}: source={s.source}")
    """

    def __init__(
        self,
        symspell: "SymSpell",
        max_suggestions: int = 5,
        max_edit_distance: int = 2,
        include_spaced_variants: bool = True,
        spacing_penalty: float = 0.3,
    ):
        """Initialize compound strategy.

        Args:
            symspell: SymSpell instance for compound lookup.
            max_suggestions: Maximum suggestions to return.
            max_edit_distance: Maximum edit distance per word in compound.
            include_spaced_variants: Whether to include space-separated forms.
            spacing_penalty: Score penalty for spaced variants (added to strategy_score).
        """
        super().__init__(max_suggestions=max_suggestions, max_edit_distance=max_edit_distance)
        self._symspell = symspell
        self._include_spaced_variants = include_spaced_variants
        self._spacing_penalty = spacing_penalty

    @property
    def name(self) -> str:
        """Return the strategy identifier."""
        return "compound"

    def suggest(
        self,
        term: str,
        context: SuggestionContext | None = None,
    ) -> SuggestionResult:
        """Generate compound suggestions for a term.

        Args:
            term: Term to find compound corrections for.
            context: Optional context (not used for compound lookup).

        Returns:
            SuggestionResult with compound suggestions.
        """
        if not term or not term.strip():
            return self._create_result([])

        # Get compound suggestions from SymSpell
        compound_results = self._symspell.lookup_compound(
            term,
            max_suggestions=self._max_suggestions * 2,  # Fetch extra for filtering
            max_edit_distance=self._max_edit_distance,
        )

        if not compound_results:
            return self._create_result([])

        suggestions: list[SuggestionData] = []
        seen_terms: set[str] = set()

        for word_list, total_distance, total_frequency in compound_results:
            if not word_list:
                continue

            # Joined form (no spaces)
            joined_term = "".join(word_list)

            # Skip if identical to original input
            if joined_term == term:
                continue

            # Add joined form
            if joined_term not in seen_terms:
                suggestions.append(
                    SuggestionData(
                        term=joined_term,
                        edit_distance=total_distance,
                        frequency=total_frequency,
                        source="compound",
                        confidence=self._distance_to_confidence(total_distance),
                        strategy_score=float(total_distance),
                        score_breakdown={
                            "total_distance": float(total_distance),
                            "word_count": float(len(word_list)),
                        },
                    )
                )
                seen_terms.add(joined_term)

            # Optionally add spaced form if different
            if self._include_spaced_variants and len(word_list) > 1:
                spaced_term = " ".join(word_list)
                if spaced_term != joined_term and spaced_term not in seen_terms:
                    # Spaced variants get a small penalty
                    spaced_score = float(total_distance) + self._spacing_penalty
                    suggestions.append(
                        SuggestionData(
                            term=spaced_term,
                            edit_distance=total_distance,
                            frequency=total_frequency,
                            source="compound",
                            confidence=self._distance_to_confidence(total_distance) * 0.9,
                            strategy_score=spaced_score,
                            score_breakdown={
                                "total_distance": float(total_distance),
                                "spacing_penalty": self._spacing_penalty,
                                "word_count": float(len(word_list)),
                            },
                        )
                    )
                    seen_terms.add(spaced_term)

        # Sort by strategy_score (lower is better)
        suggestions.sort(key=lambda s: s.strategy_score or float("inf"))

        return self._create_result(
            suggestions,
            metadata={"original": term, "include_spaced": self._include_spaced_variants},
        )

    def _distance_to_confidence(self, distance: int) -> float:
        """Convert edit distance to confidence score.

        Lower distance = higher confidence.
        """
        if distance <= 0:
            return 1.0
        elif distance == 1:
            return 0.85
        elif distance == 2:
            return 0.70
        else:
            return max(0.3, 1.0 - (distance * 0.2))
