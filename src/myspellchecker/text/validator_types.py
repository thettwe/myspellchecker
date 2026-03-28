"""Validation type definitions for Myanmar text ingestion.

This module defines the core types used by the validator subsystem:
``ValidationIssue`` (enum of detected issue kinds) and ``ValidationResult``
(dataclass returned by ``validate_text``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

__all__ = [
    "ValidationIssue",
    "ValidationResult",
]


class ValidationIssue(Enum):
    """
    Types of validation issues detected during data ingestion.

    These issues represent structural or encoding problems in Myanmar text
    that should be filtered out BEFORE entering the dictionary database.

    Note:
        This enum is distinct from ``core.constants.ErrorType`` which is used
        for spell-check errors detected in user input. The separation exists
        because:

        - ``ValidationIssue``: Data quality issues (Zawgyi, fragments, truncation)
        - ``ErrorType``: Spelling/grammar errors (typos, wrong particles)

    Categories:
        **Encoding Issues**: EXTENDED_MYANMAR, ZAWGYI_*
        **Structural Issues**: ASAT_BEFORE_VOWEL, SCRAMBLED_ORDER, INVALID_START
        **Segmentation Artifacts**: FRAGMENT_PATTERN, DOUBLE_ENDING, ASAT_INITIAL
        **Truncation Issues**: INCOMPLETE_WORD, COMPOUND_TRUNCATED, MISSING_E_VOWEL
        **Quality Filters**: PURE_NUMERAL, DOUBLED_CONSONANT, MIXED_LETTER_NUMERAL

    Example:
        >>> result = validate_text("ပျှော်")  # Invalid extra medial
        >>> result.issues[0][0]
        ValidationIssue.KNOWN_INVALID
    """

    EXTENDED_MYANMAR = "extended_myanmar"
    ZAWGYI_YA_ASAT = "zawgyi_ya_asat"
    ZAWGYI_YA_TERMINAL = "zawgyi_ya_terminal"
    ZAWGYI_YA_RA = "zawgyi_ya_ra"
    ASAT_BEFORE_VOWEL = "asat_before_vowel"
    INCOMPLETE_VOWEL = "incomplete_vowel"
    DIGIT_TONE = "digit_tone"
    SCRAMBLED_ORDER = "scrambled_order"
    INVALID_START = "invalid_start"
    DOUBLED_DIACRITIC = "doubled_diacritic"
    VIRAMA_AT_END = "virama_at_end"
    EMPTY_OR_WHITESPACE = "empty_or_whitespace"
    KNOWN_INVALID = "known_invalid"
    # Word quality issues (segmentation artifacts)
    FRAGMENT_PATTERN = "fragment_pattern"
    DOUBLE_ENDING = "double_ending"
    INCOMPLETE_WORD = "incomplete_word"
    MIXED_LETTER_NUMERAL = "mixed_letter_numeral"
    ASAT_INITIAL = "asat_initial"
    COMPOUND_TRUNCATED = "compound_truncated"
    MISSING_E_VOWEL = "missing_e_vowel"
    PURE_NUMERAL = "pure_numeral"
    DOUBLED_CONSONANT = "doubled_consonant"
    INVALID_VOWEL_SEQUENCE_SYLLABLE = "invalid_vowel_sequence_syllable"
    # Segmentation fragment issues (myword artifacts)
    BARE_CONSONANT_END = "bare_consonant_end"
    STACKED_CONSONANT_START = "stacked_consonant_start"
    MEDIAL_START = "medial_start"
    DEPENDENT_VOWEL_START = "dependent_vowel_start"
    GREAT_SA_START = "great_sa_start"
    ASAT_ANUSVARA_SEQUENCE = "asat_anusvara_sequence"
    DOUBLED_INDEPENDENT_VOWEL = "doubled_independent_vowel"


@dataclass
class ValidationResult:
    """
    Result of text validation during data ingestion.

    This dataclass contains the validation outcome including a boolean
    validity flag and a list of detected issues with descriptions.

    Attributes:
        is_valid: True if no validation issues were found, False otherwise.
        issues: List of (ValidationIssue, description) tuples describing
            each detected problem. Empty list if is_valid is True.
        cleaned_text: Optional cleaned version of the input text (reserved
            for future use in automatic correction/normalization).

    Example:
        >>> result = validate_text("ကျွန်ုပ်")
        >>> if not result.is_valid:
        ...     for issue, desc in result.issues:
        ...         print(f"{issue.value}: {desc}")
        asat_before_vowel: Asat before vowel: ်ု

    Note:
        The ``__bool__`` method allows direct boolean evaluation:
        >>> if validate_text("မြန်မာ"):
        ...     print("Valid text")
    """

    is_valid: bool
    issues: list[tuple[ValidationIssue, str]] = field(default_factory=list)
    cleaned_text: str | None = None

    def __bool__(self) -> bool:
        """Return validity status for boolean context."""
        return self.is_valid
