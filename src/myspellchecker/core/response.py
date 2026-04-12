"""
Response and Error data classes for spell checking results.

This module defines the data structures returned by the spell checker,
including various error types and response formatting methods.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from myspellchecker.core.constants import (
    ET_BROKEN_STACKING,
    ET_BROKEN_VIRAMA,
    ET_COLLOQUIAL_INFO,
    ET_COLLOQUIAL_VARIANT,
    ET_CONTEXT_PROBABILITY,
    ET_DUPLICATE_PUNCTUATION,
    ET_GRAMMAR,
    ET_HA_HTOE_CONFUSION,
    ET_INCOMPLETE_STACKING,
    ET_LEADING_VOWEL_E,
    ET_MEDIAL_COMPATIBILITY_ERROR,
    ET_MEDIAL_CONFUSION,
    ET_MEDIAL_ORDER_ERROR,
    ET_MISSING_ASAT,
    ET_PARTICLE_TYPO,
    ET_SYLLABLE,
    ET_VOWEL_AFTER_ASAT,
    ET_WORD,
    ET_ZAWGYI_ENCODING,
    ActionType,
)

__all__ = [
    "ContextError",
    "Error",
    "GrammarError",
    "Response",
    "Suggestion",
    "SyllableError",
    "WordError",
    "classify_action",
]

# Error types that are safe to auto-fix: deterministic, high-confidence,
# structural repairs where the correct form is unambiguous.
_AUTO_FIX_TYPES: frozenset[str] = frozenset(
    {
        ET_ZAWGYI_ENCODING,
        ET_PARTICLE_TYPO,
        ET_MEDIAL_CONFUSION,
        ET_MEDIAL_ORDER_ERROR,
        ET_MEDIAL_COMPATIBILITY_ERROR,
        ET_HA_HTOE_CONFUSION,
        ET_BROKEN_VIRAMA,
        ET_BROKEN_STACKING,
        ET_INCOMPLETE_STACKING,
        ET_MISSING_ASAT,
        ET_LEADING_VOWEL_E,
        ET_VOWEL_AFTER_ASAT,
        ET_DUPLICATE_PUNCTUATION,
    }
)

# Error types that are advisory only — present as informational, not errors.
_INFORM_TYPES: frozenset[str] = frozenset(
    {
        ET_COLLOQUIAL_INFO,
        ET_COLLOQUIAL_VARIANT,
    }
)

# Confidence threshold below which we downgrade to INFORM.
_INFORM_CONFIDENCE_THRESHOLD: float = 0.60


def _suggestion_text(s: "str | Suggestion") -> str:
    """Extract plain text from a Suggestion or string."""
    return str(s)


def _suggestion_detail(s: "str | Suggestion") -> dict[str, Any]:
    """Build a detail dict from a Suggestion or string."""
    if isinstance(s, Suggestion):
        return {"text": s.text, "confidence": s.confidence, "source": s.source}
    return {"text": str(s), "confidence": 0.0, "source": ""}


class Suggestion(str):
    """A correction suggestion with confidence score and source attribution.

    Inherits from :class:`str` so that every string operation (slicing,
    ``startswith``, ``len()``, etc.) works transparently.  The extra
    metadata is stored as instance attributes.

    Attributes:
        text: The suggestion text (same as ``str(self)``).
        confidence: Confidence score for this suggestion [0.0, 1.0].
        source: Origin of the suggestion (e.g. "symspell", "phonetic",
                "compound_resolver").
    """

    confidence: float
    source: str

    def __new__(
        cls,
        text: str = "",
        confidence: float = 0.0,
        source: str = "",
    ) -> "Suggestion":
        instance = super().__new__(cls, text)
        instance.confidence = confidence
        instance.source = source
        return instance

    @property
    def text(self) -> str:
        """The suggestion string (identical to ``str(self)``)."""
        return str.__str__(self)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return str.__eq__(self, other)
        return NotImplemented

    def __hash__(self) -> int:
        return str.__hash__(self)


def classify_action(error_type: str, confidence: float) -> ActionType:
    """
    Determine the recommended action for a detected error.

    Args:
        error_type: The error type string (from ErrorType enum).
        confidence: Confidence score [0.0, 1.0].

    Returns:
        ActionType.AUTO_FIX  — safe to apply silently
        ActionType.SUGGEST   — show to user for confirmation
        ActionType.INFORM    — advisory note only
    """
    if error_type in _INFORM_TYPES or confidence < _INFORM_CONFIDENCE_THRESHOLD:
        return ActionType.INFORM
    if error_type in _AUTO_FIX_TYPES:
        return ActionType.AUTO_FIX
    return ActionType.SUGGEST


@dataclass
class Error:
    """
    Base error class for spell checking errors.

    All error types inherit from this base class and provide common
    attributes for error reporting and suggestion display.

    Attributes:
        text: The original erroneous text (syllable or word).
        position: Character position in the normalized input text (0-indexed).
        error_type: Type of error ('invalid_syllable', 'invalid_word', 'context_probability').
        suggestions: List of correction suggestions, ranked by likelihood.
        confidence: Confidence score for this error detection [0.0, 1.0].
                   Higher values mean more certain this is an error.
    """

    text: str
    position: int
    suggestions: list[Suggestion]
    error_type: str
    confidence: float = 1.0
    source_strategy: str = ""

    def __post_init__(self) -> None:
        """Auto-convert plain strings to Suggestion objects."""
        self.suggestions = [
            s if isinstance(s, Suggestion) else Suggestion(text=s) for s in self.suggestions
        ]

    @property
    def end(self) -> int:
        """End position (exclusive) of the error span in the normalized input."""
        return self.position + len(self.text)

    @property
    def action(self) -> ActionType:
        """Recommended consumer action: auto_fix, suggest, or inform."""
        return classify_action(self.error_type, self.confidence)

    @property
    def severity(self) -> str:
        """Severity level derived from the recommended action.

        Returns:
            ``"error"`` when action is AUTO_FIX (high-confidence correction),
            ``"warning"`` when action is SUGGEST (uncertain correction),
            ``"info"`` when action is INFORM (advisory only).
        """
        _ACTION_TO_SEVERITY = {
            ActionType.AUTO_FIX: "error",
            ActionType.SUGGEST: "warning",
            ActionType.INFORM: "info",
        }
        return _ACTION_TO_SEVERITY.get(self.action, "warning")

    @property
    def message(self) -> str:
        """Human-readable, localized description of this error."""
        from myspellchecker.core.i18n import get_message

        return get_message(self.error_type)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert error to dictionary representation.

        Returns:
            Dictionary containing all error attributes.

        Example:
            >>> error = SyllableError("မျန်", 0, ["မြန်", "မျန်း"])
            >>> error.to_dict()
            {
                'text': 'မျန်',
                'position': 0,
                'error_type': 'invalid_syllable',
                'suggestions': ['မြန်', 'မျန်း'],
                'confidence': 1.0,
                'end': 4,
                'action': 'auto_fix',
                'severity': 'error',
                'message': 'Invalid syllable'
            }
        """
        d = asdict(self)
        d["suggestions"] = [_suggestion_text(s) for s in self.suggestions]
        d["suggestions_detail"] = [_suggestion_detail(s) for s in self.suggestions]
        d["end"] = self.end
        d["action"] = self.action.value
        d["severity"] = self.severity
        d["message"] = self.message
        return d

    @staticmethod
    def _add_action(d: dict[str, Any], error: "Error") -> dict[str, Any]:
        """Add computed fields and flatten suggestions. Used by subclasses."""
        d["suggestions"] = [_suggestion_text(s) for s in error.suggestions]
        d["suggestions_detail"] = [_suggestion_detail(s) for s in error.suggestions]
        d["end"] = error.end
        d["action"] = error.action.value
        d["severity"] = error.severity
        d["message"] = error.message
        return d

    def to_json(self, indent: int = 2) -> str:
        """
        Convert error to JSON string.

        Args:
            indent: JSON indentation level (default: 2 spaces).
                   Set to None for compact output.

        Returns:
            Pretty-printed JSON string with Unicode preserved.

        Example:
            >>> error = SyllableError("မျန်", 0, ["မြန်", "မျန်း"])
            >>> print(error.to_json())
            {
              "text": "မျန်",
              "position": 0,
              "error_type": "invalid_syllable",
              "suggestions": ["မြန်", "မျန်း"],
              "confidence": 1.0
            }
        """
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


@dataclass
class SyllableError(Error):
    """
    Invalid syllable error (Layer 1).

    Detected when a syllable is not found in the dictionary.
    Common causes: typos, wrong diacritics, encoding errors.

    Error types:
        - "invalid_syllable": Standard syllable validation error
        - "particle_typo": Known particle typo pattern (e.g., တယ → တယ်)
        - "medial_confusion": Ya-pin/Ya-yit confusion (e.g., ကြေးဇူး → ကျေးဇူး)

    Example:
        မျန် → မြန် (wrong diacritic: medial ya vs ra)
    """

    error_type: str = field(default=ET_SYLLABLE)


@dataclass
class WordError(Error):
    """
    Invalid word error (Layer 2).

    Detected when a word's syllables are valid individually,
    but their combination forms an invalid word.

    Error types:
        - "invalid_word": Standard word validation error
        - "colloquial_variant": Colloquial spelling (strict mode)
        - "colloquial_info": Colloquial spelling info note (lenient mode)

    Attributes:
        syllable_count: Number of syllables in the invalid word.
        error_type: Type of error (default: "invalid_word").

    Example:
        မြန်စာ (invalid combination, though မြန် and စာ are valid syllables)
    """

    syllable_count: int = 0
    error_type: str = field(default=ET_WORD)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert word error to dictionary representation.

        Returns:
            Dictionary containing all error attributes including syllable_count.

        Example:
            >>> error = WordError("မြန်စာ", 0, ["မြန်မာ", "မြန်"], syllable_count=2)
            >>> error.to_dict()
            {
                'text': 'မြန်စာ',
                'position': 0,
                'error_type': 'invalid_word',
                'suggestions': ['မြန်မာ', 'မြန်'],
                'confidence': 1.0,
                'syllable_count': 2,
                'action': 'suggest',
                'message': 'Invalid word'
            }
        """
        return self._add_action(asdict(self), self)


@dataclass
class ContextError(Error):
    """
    Context error - unlikely word sequence (Layer 3).

    Detected when the bigram probability P(current|previous) falls
    below the threshold, indicating an unusual or unlikely sequence.

    Attributes:
        probability: The bigram probability P(current|previous).
        prev_word: The previous word providing context.

    Example:
        "သူ ဘယ်" might be valid words but unlikely sequence
        (low P(ဘယ်|သူ) probability)
    """

    probability: float = 0.0
    prev_word: str = ""
    error_type: str = field(default=ET_CONTEXT_PROBABILITY)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert context error to dictionary representation.

        Returns:
            Dictionary containing all error attributes including probability and prev_word.

        Example:
            >>> error = ContextError(
            ...     "ဘယ်", 3, ["သွား", "သည်"],
            ...     probability=0.005, prev_word="သူ"
            ... )
            >>> error.to_dict()
            {
                'text': 'ဘယ်',
                'position': 3,
                'error_type': 'context_probability',
                'suggestions': ['သွား', 'သည်'],
                'confidence': 1.0,
                'probability': 0.005,
                'prev_word': 'သူ',
                'action': 'suggest',
                'message': 'Unlikely word sequence'
            }
        """
        return self._add_action(asdict(self), self)


@dataclass
class GrammarError(Error):
    """
    Base class for grammar-related errors.

    All grammar checker errors (ClassifierError, NegationError, AspectError,
    CompoundError, RegisterError) should extend this class to maintain a
    consistent interface with the spell checker's error hierarchy.

    Attributes:
        text: The erroneous word (mapped from 'word' in grammar checkers).
        position: Index of the error in the word list.
        error_type: Type of grammar error.
        suggestions: List containing the suggested correction.
        confidence: Confidence score (0.0-1.0).
        reason: Human-readable explanation of the error.

    Backward Compatibility:
        - word: Alias for 'text' (read-only property).
        - suggestion: Returns first suggestion (read-only property).

    Example:
        >>> error = GrammarError(
        ...     text="ယေက်",
        ...     position=2,
        ...     suggestions=["ယောက်"],
        ...     error_type="classifier_typo",
        ...     confidence=0.95,
        ...     reason="Classifier typo: ယေက် should be ယောက်"
        ... )
        >>> error.word  # Backward compatible alias
        'ယေက်'
        >>> error.suggestion  # First suggestion
        'ယောက်'
    """

    reason: str = ""
    error_type: str = field(default=ET_GRAMMAR)

    @property
    def word(self) -> str:
        """Alias for 'text' for backward compatibility."""
        return self.text

    @property
    def suggestion(self) -> str:
        """Return first suggestion for backward compatibility."""
        return self.suggestions[0] if self.suggestions else ""


@dataclass
class Response:
    """
    Spell checking response object.

    Contains the results of a spell checking operation, including
    original text, corrected text, errors, and metadata.

    Attributes:
        text: Original input text (unchanged).
        corrected_text: Auto-corrected text using top suggestions.
                       May be same as original if no errors or no auto-correct.
        has_errors: Quick boolean flag - True if any errors detected.
        level: Validation level used ('syllable' or 'word').
        errors: List of Error objects (SyllableError, WordError, ContextError).
        metadata: Additional metadata dict with processing information.
                 Keys: processing_time (seconds), total_errors, syllable_errors,
                 word_errors, context_errors, semantic_errors, layers_applied,
                 zawgyi_warning.
    """

    text: str
    corrected_text: str
    has_errors: bool
    level: str
    errors: list[Error]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """
        Convert response to dictionary representation.

        Converts all nested Error objects to dictionaries as well.

        Returns:
            Dictionary containing all response data.

        Example:
            >>> response = Response(
            ...     text="မျနမ်ာ",
            ...     corrected_text="မြနမ်ာ",
            ...     has_errors=True,
            ...     level="syllable",
            ...     errors=[SyllableError("မျန်", 0, ["မြန်"])],
            ...     metadata={"processing_time": 0.015}
            ... )
            >>> response.to_dict()
            {
                'text': 'မျနမ်ာ',
                'corrected_text': 'မြနမ်ာ',
                'has_errors': True,
                'level': 'syllable',
                'errors': [{'text': 'မျန်', ...}],
                'metadata': {'processing_time_ms': 15.2}
            }
        """
        return {
            "text": self.text,
            "corrected_text": self.corrected_text,
            "has_errors": self.has_errors,
            "level": self.level,
            "errors": [e.to_dict() for e in self.errors],
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        """
        Convert response to JSON string.

        Args:
            indent: JSON indentation level (default: 2 spaces).
                   Set to None for compact output.

        Returns:
            Pretty-printed JSON string with Unicode preserved.

        Example:
            >>> response = Response(...)
            >>> print(response.to_json())
            {
              "text": "မျနမ်ာ",
              "corrected_text": "မြနမ်ာ",
              "has_errors": true,
              "level": "syllable",
              "errors": [
                {
                  "text": "မျန်",
                  "position": 0,
                  "error_type": "invalid_syllable",
                  "suggestions": ["မြန်", "မျန်း"],
                  "confidence": 1.0
                }
              ],
              "metadata": {
                "processing_time": 0.015
              }
            }

        Notes:
            - Uses ensure_ascii=False to preserve Myanmar Unicode characters
            - Suitable for API responses or logging
        """
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def __str__(self) -> str:
        """
        String representation for debugging.

        Returns:
            Human-readable summary of the response.
        """
        error_count = len(self.errors)
        # Truncate text if longer than 20 characters, otherwise show full text
        text_display = f"'{self.text[:20]}...'" if len(self.text) > 20 else f"'{self.text}'"
        return (
            f"Response(text={text_display}, "
            f"has_errors={self.has_errors}, "
            f"error_count={error_count}, "
            f"level='{self.level}')"
        )
