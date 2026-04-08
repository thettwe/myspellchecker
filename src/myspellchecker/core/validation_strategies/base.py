"""
Base classes for validation strategies.

This module defines the abstract base class and context for validation strategies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from myspellchecker.core.response import Error


@dataclass
class ErrorCandidate:
    """A candidate error emitted by a validation strategy.

    Multiple strategies may produce candidates for the same position.
    The arbiter selects the best candidate when conflicts exist.
    """

    strategy_name: str
    error_type: str
    confidence: float
    suggestion: str | None = None
    evidence: str = ""
    word_indices: tuple[int, ...] = ()


@dataclass
class ValidationContext:
    """
    Shared context for validation strategies.

    This context is passed to each strategy in the validation chain,
    allowing strategies to access sentence-level information and track
    errors found by previous strategies.

    Attributes:
        sentence: Full sentence being validated
        words: List of words in sentence
        word_positions: Character position of each word in sentence
        is_name_mask: Boolean mask indicating which words are proper names
        existing_errors: Map of word positions to error types for positions
            where errors have been found. Supports ``pos in existing_errors``
            membership checks (same as the former ``set``).
        existing_suggestions: Suggestions produced by the strategy that
            first flagged each position. Populated alongside existing_errors.
        existing_confidences: Confidence scores of the first-flagged errors.
        sentence_type: Type of sentence (statement, question, command, etc.)
        pos_tags: POS tags for each word (if available)
    """

    sentence: str
    words: list[str]
    word_positions: list[int]
    is_name_mask: list[bool] = field(default_factory=list)
    existing_errors: dict[int, str] = field(default_factory=dict)
    existing_suggestions: dict[int, list[str]] = field(default_factory=dict)
    existing_confidences: dict[int, float] = field(default_factory=dict)
    sentence_type: str = "statement"
    pos_tags: list[str] = field(default_factory=list)
    full_text: str = ""
    global_error_count: int = 0
    error_candidates: dict[int, list[ErrorCandidate]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate consistency of parallel lists."""
        n = len(self.words)
        if len(self.word_positions) != n:
            raise ValueError(
                f"word_positions length ({len(self.word_positions)}) must match words length ({n})"
            )
        # Auto-fill is_name_mask if empty (default_factory gives [])
        if not self.is_name_mask:
            self.is_name_mask = [False] * n
        elif len(self.is_name_mask) != n:
            raise ValueError(
                f"is_name_mask length ({len(self.is_name_mask)}) must match words length ({n})"
            )


class ValidationStrategy(ABC):
    """
    Abstract base class for validation strategies.

    Each strategy implements a specific validation concern (e.g., tone
    disambiguation, grammar rules, POS sequence checking) and can be
    composed with other strategies to build a complete validation pipeline.

    Strategies are executed in priority order (lower priority runs first).

    Example:
        >>> class MyStrategy(ValidationStrategy):
        ...     def priority(self) -> int:
        ...         return 50
        ...
        ...     def validate(self, context: ValidationContext) -> list[Error]:
        ...         errors = []
        ...         for i, word in enumerate(context.words):
        ...             if is_invalid(word):
        ...                 errors.append(create_error(word, i))
        ...         return errors
    """

    @abstractmethod
    def validate(self, context: ValidationContext) -> list[Error]:
        """
        Validate words in context and return errors.

        Args:
            context: Validation context with sentence information

        Returns:
            List of errors found by this strategy

        Example:
            >>> context = ValidationContext(
            ...     sentence="သူ သွား ကျောင်း",
            ...     words=["သူ", "သွား", "ကျောင်း"],
            ...     word_positions=[0, 6, 15]
            ... )
            >>> errors = strategy.validate(context)
        """
        raise NotImplementedError

    @abstractmethod
    def priority(self) -> int:
        """
        Return strategy execution priority.

        Lower values run first. Current strategy priorities:
        - 10: ToneValidation (tone mark disambiguation)
        - 15: Orthography (medial order and compatibility)
        - 20: SyntacticRule (grammar rule checking)
        - 24: StatisticalConfusable (bigram-based confusable detection)
        - 25: BrokenCompound (broken compound detection)
        - 30: POSSequence (POS sequence validation)
        - 40: Question (question structure validation)
        - 45: Homophone (sound-alike detection)
        - 47: ConfusableCompoundClassifier (MLP-based compound detection)
        - 48: ConfusableSemantic (MLM-enhanced confusable detection)
        - 50: NgramContext (bigram/trigram probability)
        - 70: Semantic (AI-powered MLM validation, expensive, run last)

        Returns:
            Priority integer (lower runs first)

        Example:
            >>> strategy.priority()
            50
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        """String representation showing strategy name and priority."""
        return f"{self.__class__.__name__}(priority={self.priority()})"
