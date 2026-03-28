"""SymSpell-based suggestion strategy.

Generates spelling correction candidates using the SymSpell algorithm
with edit-distance based lookup and phonetic matching.
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


class SymSpellSuggestionStrategy(BaseSuggestionStrategy):
    """Strategy that generates suggestions using SymSpell algorithm.

    This strategy wraps SymSpell.lookup to provide edit-distance based
    suggestions with proper source attribution.

    Example:
        >>> strategy = SymSpellSuggestionStrategy(symspell=symspell_instance)
        >>> result = strategy.suggest("typo")
        >>> for s in result.suggestions:
        ...     print(f"{s.term}: source={s.source}, confidence={s.confidence}")
    """

    def __init__(
        self,
        symspell: "SymSpell",
        max_suggestions: int = 10,
        max_edit_distance: int = 2,
        use_phonetic: bool = True,
        validation_level: str = "word",
    ):
        """Initialize SymSpell strategy.

        Args:
            symspell: SymSpell instance for suggestion lookup.
            max_suggestions: Maximum suggestions to fetch (fetches 2x for filtering).
            max_edit_distance: Maximum edit distance for candidates.
            use_phonetic: Whether to include phonetic matches.
            validation_level: Validation level ("syllable" or "word").
        """
        super().__init__(max_suggestions=max_suggestions, max_edit_distance=max_edit_distance)
        self._symspell = symspell
        self._use_phonetic = use_phonetic
        self._validation_level = validation_level

    @property
    def name(self) -> str:
        """Return the strategy identifier."""
        return "symspell"

    def suggest(
        self,
        term: str,
        context: SuggestionContext | None = None,
    ) -> SuggestionResult:
        """Generate suggestions using SymSpell lookup.

        Args:
            term: Term to generate suggestions for.
            context: Optional context (not used by this strategy).

        Returns:
            SuggestionResult with SymSpell-sourced suggestions.
        """
        max_sugg = context.max_suggestions if context else self._max_suggestions
        # Fetch 2x to allow for filtering/re-ranking
        lookup_count = max_sugg * 2

        symspell_results = self._symspell.lookup(
            term,
            level=self._validation_level,
            max_suggestions=lookup_count,
            use_phonetic=self._use_phonetic,
        )

        # Convert to SuggestionData with source attribution
        # Pass through all SymSpell-specific fields for comprehensive ranking
        suggestions = []
        for result in symspell_results:
            suggestions.append(
                SuggestionData(
                    term=result.term,
                    edit_distance=result.edit_distance,
                    frequency=result.frequency,
                    phonetic_score=result.phonetic_score,
                    syllable_distance=result.syllable_distance,
                    weighted_distance=result.weighted_distance,
                    is_nasal_variant=result.is_nasal_variant,
                    has_same_nasal_ending=result.has_same_nasal_ending,
                    source="symspell",
                    confidence=self._score_to_confidence(result.score),
                    strategy_score=result.score,  # Pass SymSpell score for optional blending
                )
            )

        return self._create_result(suggestions, max_suggestions=max_sugg)

    def _score_to_confidence(self, score: float) -> float:
        """Convert SymSpell score to confidence (0.0-1.0).

        Lower scores are better in SymSpell, so we invert.
        """
        # Score typically ranges from 0 (perfect match) to ~10+ (poor match)
        # Map to confidence: 0 -> 1.0, 5 -> 0.5, 10+ -> ~0.0
        if score <= 0:
            return 1.0
        return max(0.0, 1.0 - (score / 10.0))
