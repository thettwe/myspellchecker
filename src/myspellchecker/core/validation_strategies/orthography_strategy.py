"""
Orthography Validation Strategy.

This strategy validates Myanmar orthographic rules at the word level,
including medial consonant ordering and compatibility checks.

Priority: 15 (runs early, after tone validation)
"""

from __future__ import annotations

from typing import Any

from myspellchecker.core.constants import ET_MEDIAL_COMPATIBILITY_ERROR, ET_MEDIAL_ORDER_ERROR
from myspellchecker.core.constants.myanmar_constants import (
    COMPATIBLE_HA,
    COMPATIBLE_RA,
    COMPATIBLE_WA,
    COMPATIBLE_YA,
)
from myspellchecker.core.response import Error, SyllableError
from myspellchecker.core.validation_strategies.base import ValidationContext, ValidationStrategy
from myspellchecker.grammar.patterns import (
    MEDIAL_ORDER_CORRECTIONS,
    get_medial_order_correction,
)
from myspellchecker.utils.logging_utils import get_logger

# Myanmar medial characters
MEDIAL_YA = "\u103b"  # ျ Ya-pin (subscript ya)
MEDIAL_RA = "\u103c"  # ြ Ya-yit (left-side ra)
MEDIAL_WA = "\u103d"  # ွ Wa-hswe
MEDIAL_HA = "\u103e"  # ှ Ha-htoe

# Medial to compatibility set mapping
MEDIAL_COMPATIBILITY = {
    MEDIAL_YA: COMPATIBLE_YA,
    MEDIAL_RA: COMPATIBLE_RA,
    MEDIAL_WA: COMPATIBLE_WA,
    MEDIAL_HA: COMPATIBLE_HA,
}


def _check_medial_compatibility(word: str) -> tuple[str, str, str] | None:
    """
    Check if medials in the word are compatible with their preceding consonant.

    Each medial is checked against the consonant it directly follows, not the
    word-initial consonant. For multi-syllable words like အကျိုး, the medial ျ
    belongs to က (second syllable), not အ (first syllable).

    Args:
        word: Myanmar word to check.

    Returns:
        Tuple of (medial, base_consonant, error_description) if incompatible,
        None if all medials are compatible.
    """
    current_consonant: str | None = None

    for char in word:
        # Track the most recent consonant (U+1000 to U+1021)
        if "\u1000" <= char <= "\u1021":
            current_consonant = char
        elif char in MEDIAL_COMPATIBILITY and current_consonant is not None:
            compatible_consonants = MEDIAL_COMPATIBILITY[char]
            if current_consonant not in compatible_consonants:
                medial_names = {
                    MEDIAL_YA: "Ya-pin (ျ)",
                    MEDIAL_RA: "Ya-yit (ြ)",
                    MEDIAL_WA: "Wa-hswe (ွ)",
                    MEDIAL_HA: "Ha-htoe (ှ)",
                }
                return (
                    char,
                    current_consonant,
                    f"{medial_names.get(char, char)} incompatible"
                    f" with consonant {current_consonant}",
                )

    return None


def _has_medial_order_violation(word: str) -> bool:
    """
    Check if word has medial order violation per UTN #11.

    Canonical order: Ya (ျ) < Ra (ြ) < Wa (ွ) < Ha (ှ)

    Args:
        word: Myanmar word to check.

    Returns:
        True if word has medial order violation.
    """
    for wrong_order in MEDIAL_ORDER_CORRECTIONS:
        if wrong_order in word:
            return True
    return False


def _medial_order_suggestions(word: str, reordered: str) -> list[str]:
    """
    Generate suggestion list for a medial order violation.

    Returns the reordered form plus stripped variants (removing each
    offending medial individually). The stripped variants handle cases
    where the word has an EXTRA medial that doesn't belong — e.g.,
    ငှျက် should become ငှက် (bird), not ငျှက် (reorder only).
    """
    medials = {MEDIAL_YA, MEDIAL_RA, MEDIAL_WA, MEDIAL_HA}
    suggestions: list[str] = [reordered]

    # Find which medials are involved in the violation
    for wrong_pair in MEDIAL_ORDER_CORRECTIONS:
        if wrong_pair in word:
            # Strip each medial in the violating pair individually
            for ch in wrong_pair:
                if ch in medials:
                    stripped = word.replace(ch, "", 1)
                    if stripped and stripped != word and stripped not in suggestions:
                        suggestions.append(stripped)
    return suggestions


class OrthographyValidationStrategy(ValidationStrategy):
    """
    Orthography validation strategy for Myanmar text.

    This strategy validates orthographic rules at the word level:
    1. Medial order per UTN #11: Ya (ျ) < Ra (ြ) < Wa (ွ) < Ha (ှ)
    2. Medial-consonant compatibility (e.g., Ha-htoe only with sonorants)

    Common violations detected:
    - Medial order: "ြျ" should be "ျြ" (Ra+Ya → Ya+Ra)
    - Compatibility: "ကှ" invalid (Ha incompatible with Ka, a stop)
    - Compatibility: "သြ" invalid (Ya-yit/ြ incompatible with Tha/သ)

    Linguistic basis:
    - Medial Ha (ှ) aspirates consonants, only meaningful for sonorants
    - Medial Ya-yit (ြ) cannot combine with Tha (သ), unlike Ya-pin (ျ)

    Priority: 15 (runs early, after tone validation)
    - Orthographic errors are fundamental and should be caught early
    - Provides correction suggestions for medial order errors

    Example:
        >>> strategy = OrthographyValidationStrategy()
        >>> context = ValidationContext(
        ...     sentence="ကွြေး လှပ",
        ...     words=["ကွြေး", "လှပ"],
        ...     word_positions=[0, 10]
        ... )
        >>> errors = strategy.validate(context)
        # Detects medial order violation in "ကွြေး" (should be "ကြွေး")
    """

    def __init__(self, confidence: float = 0.9, provider: Any | None = None):
        """
        Initialize orthography validation strategy.

        Args:
            confidence: Confidence score for orthographic violations (default: 0.9).
                       High confidence since these are rule-based, not statistical.
            provider: Optional DictionaryProvider for sorting suggestions by validity.
        """
        self.confidence = confidence
        self.provider = provider
        self.logger = get_logger(__name__)

    def validate(self, context: ValidationContext) -> list[Error]:
        """
        Validate words for orthographic errors.

        Args:
            context: Validation context with sentence information.

        Returns:
            List of SyllableError objects for orthographic violations.
        """
        if not context.words:
            return []

        errors: list[Error] = []

        for i, word in enumerate(context.words):
            # Skip if this word is a proper name
            if context.is_name_mask[i]:
                continue

            # Skip if this word already has an error from previous strategy
            position = context.word_positions[i]
            if position in context.existing_errors:
                continue

            # Check for medial order violations
            if _has_medial_order_violation(word):
                correction = get_medial_order_correction(word)
                if correction:
                    # Also generate medial-stripped suggestions. The reordered
                    # form may not be a valid word (e.g., ငှျက် → ငျှက် is
                    # still invalid; the real fix is ငှက် by stripping ya-yit).
                    suggestions = _medial_order_suggestions(word, correction)
                    # Sort by dictionary validity: valid words first
                    provider = self.provider
                    if provider and len(suggestions) > 1:
                        indexed = [(s, idx) for idx, s in enumerate(suggestions)]
                        indexed.sort(
                            key=lambda t: (
                                not provider.is_valid_word(t[0]),
                                t[1],
                            )
                        )
                        suggestions = [t[0] for t in indexed]
                    errors.append(
                        SyllableError(
                            text=word,
                            position=position,
                            error_type=ET_MEDIAL_ORDER_ERROR,
                            suggestions=suggestions,
                            confidence=self.confidence,
                        )
                    )
                    context.existing_errors[position] = ET_MEDIAL_ORDER_ERROR
                    context.existing_suggestions[position] = suggestions
                    context.existing_confidences[position] = self.confidence
                    continue  # Don't check compatibility if order is wrong

            # Check for medial-consonant compatibility violations
            compat_error = _check_medial_compatibility(word)
            if compat_error:
                medial, base, description = compat_error
                errors.append(
                    SyllableError(
                        text=word,
                        position=position,
                        error_type=ET_MEDIAL_COMPATIBILITY_ERROR,
                        suggestions=[],  # No automatic correction for compatibility errors
                        confidence=self.confidence,
                    )
                )
                context.existing_errors[position] = ET_MEDIAL_COMPATIBILITY_ERROR
                context.existing_suggestions[position] = []
                context.existing_confidences[position] = self.confidence

        return errors

    def priority(self) -> int:
        """
        Return strategy execution priority.

        Returns:
            15 (runs early - orthographic errors are fundamental)
        """
        return 15

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"OrthographyValidationStrategy(priority={self.priority()}, "
            f"confidence={self.confidence})"
        )
