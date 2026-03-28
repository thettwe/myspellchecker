"""Morphology-based suggestion strategy.

Generates suggestions by decomposing OOV words into root + suffixes,
finding similar roots via SymSpell, and reconstructing with original suffixes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from myspellchecker.algorithms.ranker import SuggestionData
from myspellchecker.algorithms.suggestion_strategy import (
    BaseSuggestionStrategy,
    SuggestionContext,
    SuggestionResult,
)

if TYPE_CHECKING:
    from myspellchecker.algorithms.symspell import SymSpell


class MorphologySuggestionStrategy(BaseSuggestionStrategy):
    """Strategy that generates suggestions using morphological analysis.

    This strategy analyzes OOV words by decomposing them into root + suffixes,
    finding similar roots, and reconstructing with the original suffixes.

    Example:
        >>> strategy = MorphologySuggestionStrategy(
        ...     symspell=symspell,
        ...     dictionary_check=provider.is_valid,
        ... )
        >>> result = strategy.suggest("wordhere")  # Inflected verb
    """

    def __init__(
        self,
        symspell: "SymSpell",
        dictionary_check: "Callable[[str], bool]",
        max_suggestions: int = 5,
        use_phonetic: bool = True,
        allow_extended_myanmar: bool = False,
    ):
        """Initialize morphology strategy.

        Args:
            symspell: SymSpell instance for root lookup.
            dictionary_check: Function to check if a word is valid.
            max_suggestions: Maximum suggestions to return.
            use_phonetic: Whether to use phonetic matching for roots.
            allow_extended_myanmar: Whether to allow Extended Myanmar chars in validation.
        """
        super().__init__(max_suggestions=max_suggestions)
        self._symspell = symspell
        self._dictionary_check = dictionary_check
        self._use_phonetic = use_phonetic
        self._allow_extended_myanmar = allow_extended_myanmar

    @property
    def name(self) -> str:
        """Return the strategy identifier."""
        return "morphology"

    def suggest(
        self,
        term: str,
        context: SuggestionContext | None = None,
    ) -> SuggestionResult:
        """Generate suggestions via morphological analysis.

        Args:
            term: Term to analyze and generate suggestions for.
            context: Optional context (not used by this strategy).

        Returns:
            SuggestionResult with morphology-derived suggestions.
        """
        from myspellchecker.text.morphology import analyze_word
        from myspellchecker.text.validator import validate_word

        suggestions: list[SuggestionData] = []

        # Analyze word morphology
        analysis = analyze_word(term, dictionary_check=self._dictionary_check)

        # Only proceed if we recovered a valid root different from original
        if not (
            analysis.root
            and analysis.root != analysis.original
            and analysis.suffixes
            and self._dictionary_check(analysis.root)
        ):
            return self._create_result(suggestions)

        # Strategy 1: Find similar roots and reconstruct
        similar_roots = self._symspell.lookup(
            analysis.root,
            level="word",
            max_suggestions=3,
            use_phonetic=self._use_phonetic,
        )

        for result in similar_roots:
            if result.term == analysis.root:
                continue  # Skip if same as recovered root

            # Reconstruct: corrected_root + original_suffixes
            reconstructed = result.term + "".join(analysis.suffixes)

            if validate_word(reconstructed, allow_extended_myanmar=self._allow_extended_myanmar):
                suggestions.append(
                    SuggestionData(
                        term=reconstructed,
                        edit_distance=result.edit_distance,
                        frequency=result.frequency,
                        phonetic_score=result.phonetic_score,
                        syllable_distance=result.syllable_distance,
                        weighted_distance=result.weighted_distance,
                        source="morphology",
                        confidence=0.85,  # High confidence for morphological matches
                        strategy_score=result.score + 0.5,  # Slight penalty for reconstruction
                    )
                )

        # Strategy 2: Add the recovered root itself if valid
        if self._dictionary_check(analysis.root):
            suggestions.insert(
                0,
                SuggestionData(
                    term=analysis.root,
                    edit_distance=len(analysis.suffixes),  # Approximate
                    frequency=0,  # Unknown
                    source="morphology",
                    confidence=0.90,  # Higher confidence for direct root
                    strategy_score=0.5,  # Good score for valid root
                ),
            )

        return self._create_result(
            suggestions,
            metadata={"root": analysis.root, "suffixes": analysis.suffixes},
        )
