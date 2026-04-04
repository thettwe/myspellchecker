"""
Myanmar Text Validation Module.

This module provides comprehensive validation for Myanmar text to detect
and filter out invalid patterns before they enter the database.

Validation Hierarchy
--------------------
This module is part of a two-layer validation architecture:

1. **Data Ingestion Validation** (this module - ``text/validator.py``):
   - Validates text BEFORE it enters the dictionary database
   - Uses ``ValidationIssue`` enum for structural/encoding issues
   - Focus: Filtering garbage, Zawgyi artifacts, segmentation errors
   - Returns: ``ValidationResult`` with detailed issue information

2. **Spell-Check Validation** (``core/validators.py``):
   - Validates user input text DURING spell checking
   - Uses ``ErrorType`` enum from ``core/constants.py``
   - Focus: Detecting spelling errors, typos, grammar issues
   - Returns: ``list[Error]`` with suggestions for correction

Key Validation Categories (Ingestion)
-------------------------------------
1. Extended Myanmar/Shan character detection (U+1050-U+109F)
2. Zawgyi encoding artifact detection
3. Invalid character ordering validation
4. Malformed syllable structure detection
5. Incomplete vowel pattern detection
6. Segmentation artifact filtering (fragments, double-endings)
7. Truncation detection via frequency comparison

Usage Example
-------------
    >>> from myspellchecker.text.validator import validate_text, validate_word
    >>> # Full validation with issue details
    >>> result = validate_text("ကျွန်ုပ်")
    >>> result.is_valid
    False
    >>> result.issues[0][0]
    ValidationIssue.ASAT_BEFORE_VOWEL
    >>> # Quick boolean check
    >>> validate_word("မြန်မာ")
    True

API Naming Conventions
----------------------
- ``validate_text(text)`` - Full validation returning ``ValidationResult``
- ``validate_word(word)`` - Quick boolean validation for single words
- ``is_*`` functions - Boolean checks for specific patterns
- ``has_*`` functions - Boolean checks for presence of patterns
- ``get_*`` functions - Return data structures (lists, dicts)
- ``filter_*`` functions - Return filtered collections

See Also
--------
- ``myspellchecker.core.validators`` - Spell-check validators
- ``myspellchecker.core.constants.ErrorType`` - Spell-check error types
"""

from __future__ import annotations

# --- Check functions ---
from myspellchecker.text.validator_checks import (  # noqa: F401
    get_quality_issues,
    get_truncation_candidates,
    is_fragment_pattern,
    is_incomplete_word,
    is_quality_word,
    is_segmentation_fragment,
    is_truncated_word,
)

# --- Data sets ---
from myspellchecker.text.validator_data import (  # noqa: F401
    KNOWN_INVALID_WORDS,
    PALI_WHITELIST,
    VALID_PALI_BARE_ENDINGS,
)

# --- Patterns & character sets ---
from myspellchecker.text.validator_patterns import (  # noqa: F401
    ASAT_BEFORE_VOWEL_PATTERN,
    ASAT_INITIAL_PATTERN,
    BROKEN_VIRAMA_PATTERN,
    COMPOUND_TRUNCATED_ENDING_PATTERN,
    CONSECUTIVE_ASAT_PATTERN,
    DIGIT_TONE_PATTERN,
    DOUBLE_ENDING_PATTERN,
    DOUBLED_CONSONANT_PATTERN,
    DOUBLED_INITIAL_CONSONANT_PATTERN,
    DOUBLED_MEDIAL_PATTERN,
    DOUBLED_VOWEL_PATTERN,
    EXTENDED_MYANMAR_PATTERN,
    FRAGMENT_CONSONANT_ASAT_PATTERN,
    FRAGMENT_CONSONANT_TONE_ASAT_PATTERN,
    FRAGMENT_CONSONANT_TONE_PATTERN,
    INCOMPLETE_CONSONANT_MEDIAL_PATTERN,
    INCOMPLETE_MEDIAL_CONSONANT_END_PATTERN,
    INCOMPLETE_MEDIAL_END_PATTERN,
    INCOMPLETE_O_VOWEL_PATTERN,
    INCOMPLETE_STACKING_PATTERN,
    INVALID_TONE_SEQUENCE,
    INVALID_VOWEL_SEQUENCE,
    MISSING_E_CONSONANT_AA_NG_PATTERN,
    MISSING_E_MEDIAL_AA_NG_PATTERN,
    MIXED_LETTER_NUMERAL_PATTERN,
    MYANMAR_CONSONANTS,
    MYANMAR_DIGITS,
    MYANMAR_MEDIALS,
    MYANMAR_TONES,
    MYANMAR_VOWELS,
    NON_STANDARD_PATTERN,
    PURE_NUMERAL_PATTERN,
    SCRAMBLED_ASAT_PATTERN,
    VALID_STARTERS,
    VIRAMA_AT_END_PATTERN,
    VOWEL_BEFORE_ASAT_PATTERN,
    ZAWGYI_YA_ASAT_PATTERN,
    ZAWGYI_YA_TERMINAL_PATTERN,
)

__all__ = [
    # Types
    "ValidationIssue",
    "ValidationResult",
    # Check functions
    "get_quality_issues",
    "get_truncation_candidates",
    "is_fragment_pattern",
    "is_incomplete_word",
    "is_quality_word",
    "is_segmentation_fragment",
    "is_truncated_word",
    # Data sets
    "KNOWN_INVALID_WORDS",
    "PALI_WHITELIST",
    "VALID_PALI_BARE_ENDINGS",
    # Patterns & character sets
    "ASAT_BEFORE_VOWEL_PATTERN",
    "ASAT_INITIAL_PATTERN",
    "BROKEN_VIRAMA_PATTERN",
    "COMPOUND_TRUNCATED_ENDING_PATTERN",
    "CONSECUTIVE_ASAT_PATTERN",
    "DIGIT_TONE_PATTERN",
    "DOUBLE_ENDING_PATTERN",
    "DOUBLED_CONSONANT_PATTERN",
    "DOUBLED_INITIAL_CONSONANT_PATTERN",
    "DOUBLED_MEDIAL_PATTERN",
    "DOUBLED_VOWEL_PATTERN",
    "EXTENDED_MYANMAR_PATTERN",
    "FRAGMENT_CONSONANT_ASAT_PATTERN",
    "FRAGMENT_CONSONANT_TONE_ASAT_PATTERN",
    "FRAGMENT_CONSONANT_TONE_PATTERN",
    "INCOMPLETE_CONSONANT_MEDIAL_PATTERN",
    "INCOMPLETE_MEDIAL_CONSONANT_END_PATTERN",
    "INCOMPLETE_MEDIAL_END_PATTERN",
    "INCOMPLETE_O_VOWEL_PATTERN",
    "INCOMPLETE_STACKING_PATTERN",
    "INVALID_TONE_SEQUENCE",
    "INVALID_VOWEL_SEQUENCE",
    "MISSING_E_CONSONANT_AA_NG_PATTERN",
    "MISSING_E_MEDIAL_AA_NG_PATTERN",
    "MIXED_LETTER_NUMERAL_PATTERN",
    "MYANMAR_CONSONANTS",
    "MYANMAR_DIGITS",
    "MYANMAR_MEDIALS",
    "MYANMAR_TONES",
    "MYANMAR_VOWELS",
    "NON_STANDARD_PATTERN",
    "PURE_NUMERAL_PATTERN",
    "SCRAMBLED_ASAT_PATTERN",
    "VALID_STARTERS",
    "VIRAMA_AT_END_PATTERN",
    "VOWEL_BEFORE_ASAT_PATTERN",
    "ZAWGYI_YA_ASAT_PATTERN",
    "ZAWGYI_YA_TERMINAL_PATTERN",
    # Validation functions
    "validate_text",
    "validate_word",
]

# ============================================================================
# RE-EXPORTS: every name previously importable from this module is still
# importable -- downstream code and tests require zero changes.
# ============================================================================
# --- Types ---
from myspellchecker.text.validator_types import (  # noqa: F401
    ValidationIssue,
    ValidationResult,
)

# ============================================================================
# VALIDATION FUNCTIONS (primary API -- kept in this file)
# ============================================================================


def validate_text(text: str, *, allow_extended_myanmar: bool = False) -> ValidationResult:
    """
    Comprehensively validate Myanmar text for data ingestion.

    This is the main validation function that checks for all known
    invalid patterns in Myanmar text. It is designed for filtering
    data BEFORE it enters the dictionary database.

    The validation checks are performed in order of priority:
        1. Known invalid words (lexically verified blocklist)
        2. Phase 1 quality filters (pure numerals, doubled consonants)
        3. Extended Myanmar/Non-standard characters (if allow_extended_myanmar=False)
        4. Zawgyi encoding artifacts
        5. Character ordering issues
        6. Tone and vowel sequence issues
        7. Word quality checks (segmentation artifacts)

    Args:
        text: Myanmar text to validate. Can be a single word or
            multi-word text (whitespace-separated).
        allow_extended_myanmar: If True, allow Extended Myanmar characters
            for non-Burmese Myanmar-script languages, including:
            - Extended Core Block (U+1050-U+109F)
            - Extended-A (U+AA60-U+AA7F)
            - Extended-B (U+A9E0-U+A9FF)
            - Non-standard core chars (U+1022, U+1028, U+1033-U+1035)
            Default is False to enforce strict Burmese-only scope.

    Returns:
        ValidationResult containing:
            - is_valid: True if no issues found
            - issues: List of (ValidationIssue, description) tuples
            - cleaned_text: Reserved for future auto-correction

    Example:
        >>> result = validate_text("ကျွန်ုပ်")
        >>> result.is_valid
        False
        >>> result.issues[0][0]
        ValidationIssue.ASAT_BEFORE_VOWEL

        >>> # Check multi-word text
        >>> result = validate_text("မြန်မာ နိုင်ငံ")
        >>> result.is_valid
        True

        >>> # Allow extended Myanmar (for Shan/Mon/Karen text)
        >>> result = validate_text("\\uaa60text", allow_extended_myanmar=True)
        >>> # Extended chars won't be flagged

    Note:
        For quick boolean checks on single words, use ``validate_word()``
        which is optimized for that use case.

    See Also:
        - ``validate_word``: Quick boolean validation
        - ``is_quality_word``: Quality filter for dictionary inclusion
        - ``core.validators.SyllableValidator``: Spell-check validation
    """
    if not text or not text.strip():
        # Empty or whitespace-only text is valid (nothing to validate)
        return ValidationResult(
            is_valid=True,
            issues=[],
        )

    issues: list[tuple[ValidationIssue, str]] = []

    # Check for known invalid words first (lexically verified)
    # Check both the entire text and individual words (split by whitespace)
    if text in KNOWN_INVALID_WORDS:
        issues.append((ValidationIssue.KNOWN_INVALID, f"Known invalid word: {text}"))
        return ValidationResult(is_valid=False, issues=issues)

    # Also check each word if text contains whitespace
    if " " in text or "\t" in text or "\n" in text:
        words = text.split()
        for word in words:
            if word in KNOWN_INVALID_WORDS:
                issues.append((ValidationIssue.KNOWN_INVALID, f"Known invalid word: {word}"))
                # Don't return early - continue checking for other issues

    # ========================================================================
    # PHASE 1 QUALITY FILTERS
    # ========================================================================

    # Check for pure numeral sequences (not valid dictionary words)
    # Examples: ၆၉၀၀, ၁၆၄၂, ၅၀၀၀၀၀
    if PURE_NUMERAL_PATTERN.match(text):
        issues.append(
            (
                ValidationIssue.PURE_NUMERAL,
                f"Pure numeral sequence (not a word): {text}",
            )
        )
        return ValidationResult(is_valid=False, issues=issues)

    # Check for doubled consonant (2-char) fragments
    # Examples: ဆဆ, အအ, တတ, ညည
    if DOUBLED_CONSONANT_PATTERN.match(text):
        issues.append(
            (
                ValidationIssue.DOUBLED_CONSONANT,
                f"Doubled consonant fragment: {text}",
            )
        )
        return ValidationResult(is_valid=False, issues=issues)

    # Check for Extended Myanmar characters (only if not allowed)
    if not allow_extended_myanmar:
        match = EXTENDED_MYANMAR_PATTERN.search(text)
        if match:
            char = match.group(0)
            issues.append(
                (
                    ValidationIssue.EXTENDED_MYANMAR,
                    f"Extended Myanmar char: {char} (U+{ord(char):04X})",
                )
            )

    # Check for Non-Standard Characters (Mon/Shan in core block)
    # Skip this check if extended Myanmar is allowed (these chars are valid in extended mode)
    if not allow_extended_myanmar:
        match = NON_STANDARD_PATTERN.search(text)
        if match:
            char = match.group(0)
            issues.append(
                (
                    ValidationIssue.EXTENDED_MYANMAR,
                    f"Non-standard char: {char} (U+{ord(char):04X})",
                )
            )

    # Check for Zawgyi ya-as-asat patterns
    match = ZAWGYI_YA_ASAT_PATTERN.search(text)
    if match:
        issues.append((ValidationIssue.ZAWGYI_YA_ASAT, f"Zawgyi ya+tone: {match.group(0)}"))

    # Check for Zawgyi ya-terminal patterns
    match = ZAWGYI_YA_TERMINAL_PATTERN.search(text)
    if match:
        issues.append(
            (
                ValidationIssue.ZAWGYI_YA_TERMINAL,
                f"Zawgyi ya at end: {match.group(0)}",
            )
        )

    # Note: Ya+Ra (ျြ) check removed - it IS valid in Unicode Burmese per UTN #11
    # Example valid words: ကျြေး (crane), ပျြောင်း (praise)

    # Check for asat before vowel (invalid ordering)
    match = ASAT_BEFORE_VOWEL_PATTERN.search(text)
    if match:
        issues.append(
            (
                ValidationIssue.ASAT_BEFORE_VOWEL,
                f"Asat before vowel: {match.group(0)}",
            )
        )

    # Check for vowel before asat (invalid structure)
    match = VOWEL_BEFORE_ASAT_PATTERN.search(text)
    if match:
        issues.append(
            (
                ValidationIssue.INCOMPLETE_VOWEL,
                f"Asat after vowel (invalid): {match.group(0)}",
            )
        )

    # Check for digit + tone patterns
    match = DIGIT_TONE_PATTERN.search(text)
    if match:
        issues.append((ValidationIssue.DIGIT_TONE, f"Digit+tone: {match.group(0)}"))

    # Check for scrambled ordering
    match = SCRAMBLED_ASAT_PATTERN.search(text)
    if match:
        issues.append((ValidationIssue.SCRAMBLED_ORDER, f"Scrambled: {match.group(0)}"))

    # Check for incomplete O-vowel (suppress for Pali terms with virama stacking)
    match = INCOMPLETE_O_VOWEL_PATTERN.search(text)
    if match and "\u1039" not in text:
        issues.append((ValidationIssue.INCOMPLETE_VOWEL, f"Incomplete vowel: {match.group(0)}"))

    # Check for doubled diacritics
    match = DOUBLED_VOWEL_PATTERN.search(text)
    if match:
        issues.append((ValidationIssue.DOUBLED_DIACRITIC, f"Doubled vowel: {match.group(0)}"))

    match = DOUBLED_MEDIAL_PATTERN.search(text)
    if match:
        issues.append((ValidationIssue.DOUBLED_DIACRITIC, f"Doubled medial: {match.group(0)}"))

    # Check for invalid tone sequences
    match = INVALID_TONE_SEQUENCE.search(text)
    if match:
        issues.append((ValidationIssue.DOUBLED_DIACRITIC, f"Invalid tone seq: {match.group(0)}"))

    # Check for invalid vowel sequences
    match = INVALID_VOWEL_SEQUENCE.search(text)
    if match:
        issues.append(
            (
                ValidationIssue.INVALID_VOWEL_SEQUENCE_SYLLABLE,
                f"Invalid vowel seq: {match.group(0)}",
            )
        )

    # Check for virama at end
    match = VIRAMA_AT_END_PATTERN.search(text)
    if match:
        issues.append((ValidationIssue.VIRAMA_AT_END, "Virama at word end"))

    # Check for invalid start character
    # Only apply VALID_STARTERS check to core Burmese characters (U+1000-U+104F)
    # Extended Myanmar characters (Shan, Mon, Karen) have different starter rules
    # that we don't model, so skip invalid-start check for them
    first_char_is_core_burmese = False
    if text:
        # Core Burmese range: U+1000-U+104F (excluding non-standard chars)
        c = ord(text[0])
        first_char_is_core_burmese = 0x1000 <= c <= 0x104F

    # Only check VALID_STARTERS for core Burmese characters
    if first_char_is_core_burmese and text and text[0] not in VALID_STARTERS:
        # Allow some special cases (whitespace, punctuation)
        if text[0] not in {" ", "\t", "\n"}:
            issues.append(
                (
                    ValidationIssue.INVALID_START,
                    f"Invalid start: {text[0]} (U+{ord(text[0]):04X})",
                )
            )

    # ========================================================================
    # WORD QUALITY CHECKS (segmentation artifacts)
    # ========================================================================

    # Check for mixed letter + numeral (safety net for split_word_numeral_tokens)
    if MIXED_LETTER_NUMERAL_PATTERN.search(text):
        issues.append(
            (
                ValidationIssue.MIXED_LETTER_NUMERAL,
                f"Mixed letter+numeral (should be split): {text}",
            )
        )

    # Check for asat-initial fragments (segmentation errors)
    if ASAT_INITIAL_PATTERN.match(text):
        issues.append(
            (
                ValidationIssue.ASAT_INITIAL,
                f"Asat-initial fragment (segmentation error): {text}",
            )
        )

    # Check for compound words with truncated endings
    if COMPOUND_TRUNCATED_ENDING_PATTERN.search(text):
        issues.append(
            (
                ValidationIssue.COMPOUND_TRUNCATED,
                f"Compound word with truncated ending: {text}",
            )
        )

    # Check for missing ေ in ောင pattern (common typo)
    if MISSING_E_MEDIAL_AA_NG_PATTERN.search(text):
        issues.append(
            (
                ValidationIssue.MISSING_E_VOWEL,
                f"Missing ေ in ောင pattern (medial): {text}",
            )
        )
    elif MISSING_E_CONSONANT_AA_NG_PATTERN.match(text):
        issues.append(
            (
                ValidationIssue.MISSING_E_VOWEL,
                f"Missing ေ in ောင pattern (consonant): {text}",
            )
        )

    # Check for fragment patterns (consonant + asat/tone only)
    if FRAGMENT_CONSONANT_ASAT_PATTERN.match(text):
        issues.append((ValidationIssue.FRAGMENT_PATTERN, f"Fragment (consonant+asat): {text}"))
    elif FRAGMENT_CONSONANT_TONE_PATTERN.match(text):
        issues.append((ValidationIssue.FRAGMENT_PATTERN, f"Fragment (consonant+tone): {text}"))
    elif FRAGMENT_CONSONANT_TONE_ASAT_PATTERN.match(text):
        issues.append((ValidationIssue.FRAGMENT_PATTERN, f"Fragment (consonant+tone+asat): {text}"))

    # Check for double-ending pattern (word + fragment merged)
    match = DOUBLE_ENDING_PATTERN.search(text)
    if match:
        issues.append((ValidationIssue.DOUBLE_ENDING, f"Double-ending fragment: {text}"))

    # Check for incomplete word patterns

    # Ends with medial only (no vowel or asat)
    if INCOMPLETE_MEDIAL_END_PATTERN.search(text):
        issues.append((ValidationIssue.INCOMPLETE_WORD, f"Incomplete (ends with medial): {text}"))

    # Ends with incomplete stacking (virama + consonant but no closing)
    if INCOMPLETE_STACKING_PATTERN.search(text):
        issues.append(
            (
                ValidationIssue.INCOMPLETE_WORD,
                f"Incomplete (stacking without closing): {text}",
            )
        )

    # Ends with medial + bare consonant (no asat/vowel/tone on final consonant)
    if INCOMPLETE_MEDIAL_CONSONANT_END_PATTERN.search(text):
        issues.append(
            (
                ValidationIssue.INCOMPLETE_WORD,
                f"Incomplete (medial+consonant without closing): {text}",
            )
        )

    # Ends with consonant + medial but no vowel/asat
    # Only flag if the word is very short (likely a fragment)
    if len(text) <= 3 and INCOMPLETE_CONSONANT_MEDIAL_PATTERN.search(text):
        issues.append(
            (
                ValidationIssue.INCOMPLETE_WORD,
                f"Incomplete (consonant+medial only): {text}",
            )
        )

    return ValidationResult(is_valid=len(issues) == 0, issues=issues)


def validate_word(word: str, *, allow_extended_myanmar: bool = False) -> bool:
    """
    Quick boolean validation check for a single word.

    This is a convenience function optimized for boolean checks.
    For detailed issue information, use ``validate_text()`` instead.

    This function is imported and used by ``core.validators`` to filter
    SymSpell suggestions, ensuring only structurally valid words are
    returned to users.

    Args:
        word: Single Myanmar word to validate. Should not contain
            whitespace (use ``validate_text()`` for multi-word text).
        allow_extended_myanmar: If True, allow Extended Myanmar characters
            for non-Burmese Myanmar-script languages, including:
            - Extended Core Block (U+1050-U+109F)
            - Extended-A (U+AA60-U+AA7F)
            - Extended-B (U+A9E0-U+A9FF)
            - Non-standard core chars (U+1022, U+1028, U+1033-U+1035)
            Default is False to enforce strict Burmese-only scope.

    Returns:
        True if word is structurally valid Myanmar text, False otherwise.

    Example:
        >>> validate_word("မြန်မာ")
        True
        >>> validate_word("ကျွန်ုပ်")  # Invalid: asat before vowel
        False
        >>> validate_word("")  # Empty is not a valid word
        False
        >>> validate_word("   ")  # Whitespace-only is not a valid word
        False

    Note:
        **Empty string handling differs between functions:**

        - ``validate_word("")`` returns ``False`` (empty is not a valid word)
        - ``validate_text("")`` returns ``True`` (no errors found in empty text)

        This is intentional: ``validate_word`` checks if something IS a valid
        word, while ``validate_text`` checks if text CONTAINS errors. An empty
        string is not a valid word, but empty text has no errors.

    See Also:
        - ``validate_text``: Full validation with issue details
        - ``is_quality_word``: Additional quality checks for dictionary
        - ``core.validators.Validator._filter_suggestions``: Uses this function
    """
    # Empty word is not valid (intentionally different from validate_text)
    # validate_word: "Is this a valid word?" -> empty = No
    # validate_text: "Does this have errors?" -> empty = No errors = True
    if not word or not word.strip():
        return False

    # Delegate to validate_text for consistent validation logic
    return validate_text(word, allow_extended_myanmar=allow_extended_myanmar).is_valid
