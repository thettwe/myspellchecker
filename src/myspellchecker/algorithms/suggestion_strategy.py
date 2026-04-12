"""
Suggestion Strategy Protocol for spell correction.

This module defines the SuggestionStrategy protocol for implementing different
suggestion generation strategies. Unlike SuggestionRanker (which handles ranking),
SuggestionStrategy defines the interface for the entire suggestion generation
pipeline.

The strategy pattern allows swapping different suggestion algorithms:
- Edit distance-based (SymSpell)
- Context-aware (N-gram)
- Phonetic similarity
- Semantic similarity
- Hybrid approaches

Example:
    >>> from myspellchecker.algorithms.suggestion_strategy import (
    ...     SuggestionStrategy,
    ...     SuggestionResult,
    ... )
    >>>
    >>> class CustomStrategy(SuggestionStrategy):
    ...     def suggest(self, term: str, context: SuggestionContext) -> SuggestionResult:
    ...         # Custom implementation
    ...         pass
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import (
    Protocol,
    runtime_checkable,
)

from myspellchecker.algorithms.ranker import SuggestionData, SuggestionRanker


@dataclass
class SuggestionContext:
    """Context information for generating suggestions.

    This encapsulates all contextual information that strategies might
    use to generate more relevant suggestions.

    Attributes:
        prev_words: Previous words for left context.
        next_words: Following words for right context.
        sentence: Full sentence containing the term (optional).
        position: Position of the term in the sentence.
        max_suggestions: Maximum number of suggestions to return.
        max_edit_distance: Maximum edit distance for candidates.
        include_self: Whether to include the input term if valid.
    """

    prev_words: list[str] = field(default_factory=list)
    next_words: list[str] = field(default_factory=list)
    sentence: str | None = None
    position: int = 0
    max_suggestions: int = 5
    max_edit_distance: int = 2
    include_self: bool = False


@dataclass
class SuggestionResult:
    """Result of a suggestion generation operation.

    Attributes:
        suggestions: List of suggestion candidates, sorted by relevance.
        strategy_name: Name of the strategy that generated these.
        metadata: Optional strategy-specific metadata.
        is_truncated: True if results were truncated to max_suggestions.
    """

    suggestions: list[SuggestionData]
    strategy_name: str
    metadata: dict = field(default_factory=dict)
    is_truncated: bool = False

    def __len__(self) -> int:
        return len(self.suggestions)

    def __bool__(self) -> bool:
        return len(self.suggestions) > 0

    @property
    def best(self) -> SuggestionData | None:
        """Get the best (first) suggestion."""
        return self.suggestions[0] if self.suggestions else None

    @property
    def terms(self) -> list[str]:
        """Get just the suggestion terms."""
        return [s.term for s in self.suggestions]


@runtime_checkable
class SuggestionStrategy(Protocol):
    """Protocol defining the interface for suggestion strategies.

    Implementations generate spelling correction suggestions using
    different algorithms and data sources. The strategy pattern allows
    the spell checker to swap algorithms at runtime.

    Key methods:
        suggest: Generate suggestions for a single term
        suggest_batch: Generate suggestions for multiple terms (optional optimization)
        supports_context: Check if strategy uses contextual information
        name: Strategy identifier for logging and debugging

    Example:
        >>> class MyStrategy:
        ...     @property
        ...     def name(self) -> str:
        ...         return "custom"
        ...
        ...     def suggest(self, term: str, context: SuggestionContext) -> SuggestionResult:
        ...         suggestions = self._generate_candidates(term)
        ...         return SuggestionResult(suggestions=suggestions, strategy_name=self.name)
        ...
        ...     def supports_context(self) -> bool:
        ...         return False
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the strategy name for identification."""
        ...

    @abstractmethod
    def suggest(
        self,
        term: str,
        context: SuggestionContext | None = None,
    ) -> SuggestionResult:
        """Generate suggestions for a term.

        Args:
            term: The potentially misspelled term to correct.
            context: Optional context for context-aware suggestions.

        Returns:
            SuggestionResult containing ranked suggestions.
        """
        ...

    def suggest_batch(
        self,
        terms: Sequence[str],
        contexts: Sequence[SuggestionContext] | None = None,
    ) -> list[SuggestionResult]:
        """Generate suggestions for multiple terms.

        Default implementation calls suggest() for each term.
        Override for batch-optimized strategies.

        Args:
            terms: List of terms to generate suggestions for.
            contexts: Optional contexts for each term (must match length).

        Returns:
            List of SuggestionResults, one per term.
        """
        results = []
        for i, term in enumerate(terms):
            ctx = contexts[i] if contexts and i < len(contexts) else None
            results.append(self.suggest(term, ctx))
        return results

    def supports_context(self) -> bool:
        """Check if this strategy uses contextual information.

        Returns:
            True if strategy uses prev_words/next_words from context.
        """
        return False


class BaseSuggestionStrategy:
    """Base class providing common functionality for strategies.

    Subclass this for convenience methods and default implementations.
    """

    def __init__(self, max_suggestions: int = 5, max_edit_distance: int = 2):
        """Initialize the base strategy.

        Args:
            max_suggestions: Default maximum suggestions to return.
            max_edit_distance: Default maximum edit distance.
        """
        self._max_suggestions = max_suggestions
        self._max_edit_distance = max_edit_distance

    @property
    def name(self) -> str:
        """Return strategy name (override in subclasses)."""
        return "base"

    def supports_context(self) -> bool:
        """Default: no context support."""
        return False

    def suggest(
        self,
        term: str,
        context: SuggestionContext | None = None,
    ) -> SuggestionResult:
        """Generate suggestions (must be overridden)."""
        raise NotImplementedError("Subclasses must implement suggest()")

    def suggest_batch(
        self,
        terms: Sequence[str],
        contexts: Sequence[SuggestionContext] | None = None,
    ) -> list[SuggestionResult]:
        """Default batch implementation."""
        results = []
        for i, term in enumerate(terms):
            ctx = contexts[i] if contexts and i < len(contexts) else None
            results.append(self.suggest(term, ctx))
        return results

    def _create_result(
        self,
        suggestions: list[SuggestionData],
        max_suggestions: int | None = None,
        metadata: dict | None = None,
    ) -> SuggestionResult:
        """Create a SuggestionResult with common handling.

        Args:
            suggestions: List of suggestions to include.
            max_suggestions: Limit results (uses default if None).
            metadata: Optional metadata to include.

        Returns:
            SuggestionResult with truncation handled.
        """
        limit = max_suggestions or self._max_suggestions
        is_truncated = len(suggestions) > limit
        truncated = suggestions[:limit]

        return SuggestionResult(
            suggestions=truncated,
            strategy_name=self.name,
            metadata=metadata or {},
            is_truncated=is_truncated,
        )


class CompositeSuggestionStrategy(BaseSuggestionStrategy):
    """Combines multiple strategies and merges their results.

    This strategy aggregates suggestions from multiple sources and
    provides unified ranking using a SuggestionRanker.

    Useful for combining edit-distance, phonetic, and context-aware
    strategies into a single suggestion source.

    Example:
        >>> from myspellchecker.algorithms.ranker import UnifiedRanker
        >>>
        >>> composite = CompositeSuggestionStrategy(
        ...     strategies=[symspell_strategy, context_strategy],
        ...     ranker=UnifiedRanker(),
        ... )
        >>> result = composite.suggest("typo", context)
    """

    def __init__(
        self,
        strategies: list["SuggestionStrategy"],
        ranker: SuggestionRanker | None = None,
        max_suggestions: int = 5,
        deduplicate: bool = True,
    ):
        """Initialize composite strategy.

        Args:
            strategies: List of strategies to combine.
            ranker: Ranker for unified scoring (uses UnifiedRanker if None).
            max_suggestions: Maximum suggestions in final result.
            deduplicate: Whether to remove duplicate terms.
        """
        super().__init__(max_suggestions=max_suggestions)
        self._strategies = strategies
        self._deduplicate = deduplicate

        # Import UnifiedRanker here to avoid circular dependency
        from myspellchecker.algorithms.ranker import UnifiedRanker

        self._ranker: SuggestionRanker = ranker or UnifiedRanker()
        # Store type reference for isinstance check in suggest()
        self._UnifiedRanker = UnifiedRanker

    @property
    def name(self) -> str:
        """Return strategy identifier string."""
        return "composite"

    def supports_context(self) -> bool:
        """True if any sub-strategy supports context."""
        return any(s.supports_context() for s in self._strategies)

    def suggest(
        self,
        term: str,
        context: SuggestionContext | None = None,
    ) -> SuggestionResult:
        """Generate suggestions from all strategies and merge.

        Args:
            term: Term to generate suggestions for.
            context: Optional context information.

        Returns:
            Merged and ranked SuggestionResult.
        """
        all_suggestions: list[SuggestionData] = []
        strategy_names: list[str] = []

        for strategy in self._strategies:
            result = strategy.suggest(term, context)
            all_suggestions.extend(result.suggestions)
            strategy_names.append(strategy.name)

        # Rank and deduplicate using ranker
        # Note: _UnifiedRanker stored in __init__ to avoid duplicate import
        if isinstance(self._ranker, self._UnifiedRanker):
            ranked = self._ranker.rank_suggestions(
                all_suggestions,
                deduplicate=self._deduplicate,
                error_length=len(term),
            )
        else:
            # For other rankers, do manual dedup and sort
            if self._deduplicate:
                ranked = self._deduplicate_suggestions(all_suggestions)
            else:
                ranked = all_suggestions
            ranked.sort(key=lambda s: self._ranker.score(s))

        return self._create_result(
            suggestions=ranked,
            metadata={"strategies": strategy_names},
        )

    def _deduplicate_suggestions(
        self,
        suggestions: list[SuggestionData],
    ) -> list[SuggestionData]:
        """Remove duplicate terms, keeping best candidate.

        Uses source-agnostic deduplication (raw confidence comparison),
        with tie-breaks on strategy_score, edit_distance, and frequency.
        """
        from myspellchecker.algorithms.dedup import deduplicate_suggestions

        return deduplicate_suggestions(suggestions)


# --------------------------------------------------------------------------- #
# Backwards-compatible re-exports: concrete strategies moved to
# myspellchecker.algorithms.strategies.*  but existing imports from
# myspellchecker.algorithms.suggestion_strategy still work.
# --------------------------------------------------------------------------- #
from myspellchecker.algorithms.strategies.compound_strategy import (  # noqa: E402
    CompoundSuggestionStrategy,
)
from myspellchecker.algorithms.strategies.context_strategy import (  # noqa: E402
    ContextSuggestionStrategy,
)
from myspellchecker.algorithms.strategies.morphology_strategy import (  # noqa: E402
    MorphologySuggestionStrategy,
)
from myspellchecker.algorithms.strategies.symspell_strategy import (  # noqa: E402
    SymSpellSuggestionStrategy,
)

__all__ = [
    # Base types (defined here)
    "SuggestionContext",
    "SuggestionResult",
    "SuggestionStrategy",
    "BaseSuggestionStrategy",
    "CompositeSuggestionStrategy",
    # Re-exported concrete strategies
    "SymSpellSuggestionStrategy",
    "MorphologySuggestionStrategy",
    "CompoundSuggestionStrategy",
    "ContextSuggestionStrategy",
]
