"""Utility check functions for Myanmar text validation.

Public helpers consumed by the data pipeline, the quality filter, and tests:
``is_truncated_word``, ``get_truncation_candidates``, ``is_fragment_pattern``,
``is_incomplete_word``, ``is_segmentation_fragment``, ``is_quality_word``,
``get_quality_issues``, and their private helpers.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

from myspellchecker.text.validator_data import (
    _ALLOWED_SINGLE_CONSONANTS,
    _ASAT_ANUSVARA,
    _DEPENDENT_VOWELS,
    _GREAT_SA,
    _INDEPENDENT_VOWELS_SET,
    _STACKING_MARKER,
    _WORD_ALLOWED_NON_MYANMAR,
    MEDIALS_SET,
    PALI_WHITELIST,
    VALID_PALI_BARE_ENDINGS,
)
from myspellchecker.text.validator_patterns import (
    BROKEN_VIRAMA_PATTERN,
    CONSECUTIVE_ASAT_PATTERN,
    DOUBLE_ENDING_PATTERN,
    DOUBLED_CONSONANT_PATTERN,
    DOUBLED_INITIAL_CONSONANT_PATTERN,
    FRAGMENT_CONSONANT_ASAT_PATTERN,
    FRAGMENT_CONSONANT_TONE_ASAT_PATTERN,
    FRAGMENT_CONSONANT_TONE_PATTERN,
    INCOMPLETE_CONSONANT_MEDIAL_PATTERN,
    INCOMPLETE_MEDIAL_CONSONANT_END_PATTERN,
    INCOMPLETE_MEDIAL_END_PATTERN,
    INCOMPLETE_STACKING_PATTERN,
    MYANMAR_CONSONANTS,
    PURE_NUMERAL_PATTERN,
    VIRAMA_AT_END_PATTERN,
)

__all__ = [
    "get_quality_issues",
    "get_truncation_candidates",
    "is_fragment_pattern",
    "is_incomplete_word",
    "is_quality_word",
    "is_segmentation_fragment",
    "is_truncated_word",
]

# ============================================================================
# FREQUENCY-BASED TRUNCATION DETECTION
# ============================================================================


def is_truncated_word(
    word: str,
    get_frequency: Callable[[str], int],
    frequency_ratio_threshold: int = 100,
) -> tuple[bool, str | None]:
    """
    Detect if a word is likely truncated by comparing with potential complete forms.

    This function checks if a word ending with a bare consonant has a complete form
    (with common endings like \u103a, \u1038, etc.) that is significantly more frequent.
    This is a frequency-based heuristic that complements pattern-based detection.

    The detection algorithm:
        1. Skip words in the Pali/Sanskrit whitelist (valid bare endings)
        2. Check if word ends with a bare consonant
        3. Try appending common endings (\u103a, \u1038, \u1037, \u102f, etc.)
        4. Compare frequencies of truncated vs complete forms
        5. If ratio exceeds threshold, word is likely truncated

    Args:
        word: The word to check for truncation.
        get_frequency: A callable that takes a word string and returns its
            corpus frequency as an integer. Should return 0 if word doesn't
            exist in the corpus.
        frequency_ratio_threshold: Minimum ratio of (complete_freq / truncated_freq)
            to consider the word truncated. Higher values = stricter detection.
            Default: 100 (complete form must be 100x more frequent).

    Returns:
        Tuple of (is_truncated, suggested_complete_form) where:
            - is_truncated: True if word appears to be truncated
            - suggested_complete_form: The likely complete form, or None

    Example:
        >>> def mock_freq(w: str) -> int:
        ...     return {'ချိန': 83, 'ချိန\u103a': 179977}.get(w, 0)
        >>> is_truncated_word('ချိန', mock_freq)
        (True, 'ချိန\u103a')
        >>> is_truncated_word('ဒေသ', mock_freq)  # Pali word, valid bare ending
        (False, None)

    See Also:
        - ``VALID_PALI_BARE_ENDINGS``: Whitelist of valid bare-consonant words
        - ``get_truncation_candidates``: Batch analysis of word lists
    """
    # Skip if word is in Pali whitelist (valid bare consonant endings)
    if word in VALID_PALI_BARE_ENDINGS:
        return (False, None)

    # Only check words ending with bare consonants
    if not word:
        return (False, None)

    last_char = word[-1]
    if last_char not in MYANMAR_CONSONANTS:
        return (False, None)

    # Get frequency of the truncated form
    word_freq = get_frequency(word)
    if word_freq == 0:
        return (False, None)

    # Common endings to try (in order of likelihood)
    # ် (asat), ်း (asat+visarga), း (visarga), ့ (dot), ု, ို, ုံ, ိ, ီ, ေ
    COMMON_ENDINGS = ["်", "း", "့", "ု", "ို", "ုံ", "ိ", "ီ", "ေ", "်း"]

    best_complete = None
    best_ratio: float = 0

    for ending in COMMON_ENDINGS:
        complete_form = word + ending
        complete_freq = get_frequency(complete_form)

        if complete_freq > 0:
            ratio = complete_freq / word_freq if word_freq > 0 else float("inf")
            if ratio > best_ratio:
                best_ratio = ratio
                best_complete = complete_form

    # If best complete form has significantly higher frequency, word is truncated
    if best_ratio >= frequency_ratio_threshold:
        return (True, best_complete)

    return (False, None)


def get_truncation_candidates(
    words_with_freq: list[tuple[str, int]],
    get_frequency: Callable[[str], int],
    frequency_ratio_threshold: int = 100,
) -> list[tuple[str, int, str, int, float]]:
    """
    Analyze a list of words to find truncation candidates.

    This is a batch analysis function for identifying potentially truncated
    words in a corpus. Results are sorted by frequency ratio (highest first)
    to prioritize the most obvious truncations.

    Args:
        words_with_freq: List of (word, frequency) tuples to analyze.
            Typically obtained from corpus frequency data.
        get_frequency: A callable that takes a word string and returns
            its corpus frequency. Used to look up complete form frequencies.
        frequency_ratio_threshold: Minimum ratio of (complete_freq / truncated_freq)
            to include in results. Default: 100.

    Returns:
        List of tuples containing:
            - truncated_word (str): The potentially truncated word
            - truncated_freq (int): Frequency of the truncated form
            - complete_form (str): Suggested complete form
            - complete_freq (int): Frequency of the complete form
            - ratio (float): complete_freq / truncated_freq

        Results are sorted by ratio in descending order (most obvious
        truncations first).

    Example:
        >>> words = [('ချိန', 83), ('မြန်မာ', 1000000), ('တင', 12)]
        >>> def freq_lookup(w: str) -> int:
        ...     return {'ချိန': 83, 'ချိန\u103a': 179977, 'တင': 12, 'တင\u103a': 10452}.get(w, 0)
        >>> candidates = get_truncation_candidates(words, freq_lookup)
        >>> for trunc, t_freq, complete, c_freq, ratio in candidates:
        ...     print(f"{trunc} -> {complete} (ratio: {ratio:.1f}x)")
        ချိန -> ချိန\u103a (ratio: 2168.4x)
        တင -> တင\u103a (ratio: 871.0x)

    See Also:
        - ``is_truncated_word``: Single-word truncation detection
        - ``VALID_PALI_BARE_ENDINGS``: Whitelist of valid bare endings
    """
    candidates = []

    for word, freq in words_with_freq:
        is_truncated, complete_form = is_truncated_word(
            word, get_frequency, frequency_ratio_threshold
        )
        if is_truncated and complete_form:
            complete_freq = get_frequency(complete_form)
            ratio = complete_freq / freq if freq > 0 else float("inf")
            candidates.append((word, freq, complete_form, complete_freq, ratio))

    # Sort by ratio descending (most obvious truncations first)
    candidates.sort(key=lambda x: x[4], reverse=True)
    return candidates


# ============================================================================
# QUALITY FILTER FUNCTIONS
# ============================================================================


def is_fragment_pattern(word: str) -> tuple[bool, str | None]:
    """
    Check if word matches a fragment pattern that shouldn't be in dictionary.

    Fragment patterns are typically segmentation artifacts:
    - Consonant + asat only (e.g., ဉ\u103a, ည\u103a)
    - Consonant + tone only (e.g., မး, က\u1037)
    - Consonant + tone + asat (e.g., န\u1037\u103a)
    - Double-ending patterns (e.g., တွင်င်း)

    Args:
        word: Word to check

    Returns:
        Tuple of (is_fragment: bool, pattern_name: str | None)
    """
    if not word:
        return (False, None)

    # Check fragment patterns
    if FRAGMENT_CONSONANT_ASAT_PATTERN.match(word):
        return (True, "consonant_asat")

    if FRAGMENT_CONSONANT_TONE_PATTERN.match(word):
        return (True, "consonant_tone")

    if FRAGMENT_CONSONANT_TONE_ASAT_PATTERN.match(word):
        return (True, "consonant_tone_asat")

    # Check double-ending pattern
    if DOUBLE_ENDING_PATTERN.search(word):
        return (True, "double_ending")

    return (False, None)


def is_incomplete_word(word: str) -> tuple[bool, str | None]:
    """
    Check if word appears to be incomplete (truncated).

    Incomplete word patterns:
    - Ends with medial only (e.g., ကြ, မျ)
    - Ends with stacking marker (e.g., န\u1039)
    - Ends with incomplete stacking (e.g., န\u1039တ)
    - Ends with consonant + medial (e.g., မြန, အခွင)

    Args:
        word: Word to check

    Returns:
        Tuple of (is_incomplete: bool, pattern_name: str | None)
    """
    if not word:
        return (False, None)

    # Skip Pali whitelist words
    if word in VALID_PALI_BARE_ENDINGS:
        return (False, None)

    # Check incomplete patterns
    if INCOMPLETE_MEDIAL_END_PATTERN.search(word):
        return (True, "medial_end")

    if VIRAMA_AT_END_PATTERN.search(word):
        return (True, "virama_end")

    if INCOMPLETE_STACKING_PATTERN.search(word):
        return (True, "incomplete_stacking")

    if INCOMPLETE_CONSONANT_MEDIAL_PATTERN.search(word):
        return (True, "consonant_medial_end")

    if INCOMPLETE_MEDIAL_CONSONANT_END_PATTERN.search(word):
        return (True, "medial_consonant_end")

    return (False, None)


def is_segmentation_fragment(word: str) -> tuple[bool, str | None]:
    """
    Detect likely segmentation fragments from word segmenter errors.

    This function identifies words that are likely artifacts of incorrect
    word segmentation (e.g., from myword or CRF segmenters). These patterns
    represent incomplete or malformed words that should not be in a dictionary.

    Detection Rules (linguistically verified as safe):
        1. **BARE_CONSONANT_END**: Word ends with consonant without asat.
           Exception: Single-character interjections (အ, ဟ) are allowed.
        2. **STACKED_START**: Word starts with stacked consonant marker (\u1039).
           This is always invalid as \u1039 must follow a base consonant.
        3. **MEDIAL_START**: Word starts with a medial (ျ ြ ွ ှ).
           Medials must attach to a preceding consonant.
        4. **DEPENDENT_VOWEL_START**: Word starts with a dependent vowel sign.
           These must attach to a consonant.
        5. **GREAT_SA_START**: Word starts with Great Sa (ဿ, U+103F).
           This conjunct character only appears mid-word (e.g., ပြဿနာ).
        6. **ASAT_ANUSVARA_SEQUENCE**: Word contains \u103a\u1036 sequence.
           Phonetically impossible - asat closes syllable, anusvara needs vowel.
        7. **DOUBLED_INDEPENDENT_VOWEL**: Word is two identical independent vowels.
           e.g., ဤဤ, ဥဥ - these are OCR errors, not valid words.

    Args:
        word: Word to check for segmentation artifacts.

    Returns:
        Tuple of (is_fragment: bool, issue_type: str | None).
        issue_type is one of: "bare_consonant_end", "stacked_start",
        "medial_start", "dependent_vowel_start", "great_sa_start",
        "asat_anusvara_sequence", "doubled_independent_vowel",
        or None if valid.

    Example:
        >>> is_segmentation_fragment("ဒစ်ဂျစ်တယ")  # Missing final asat
        (True, 'bare_consonant_end')
        >>> is_segmentation_fragment("စ\u1039ဆာန\u103a")  # Orphaned stacked consonant
        (True, 'stacked_start')
        >>> is_segmentation_fragment("ြမန\u103aမာ")  # Starts with medial
        (True, 'medial_start')
        >>> is_segmentation_fragment("မြန\u103aမာ")  # Valid word
        (False, None)

    Note:
        These rules were verified by a linguistic analysis to ensure they
        do not incorrectly filter legitimate Myanmar words. The frequency-based
        "prefix of longer word" rule was intentionally excluded as it would
        incorrectly filter valid short words like စာ, လူ, ရေ, etc.
    """
    if not word:
        return (False, None)

    first_char = word[0]
    last_char = word[-1]

    # Rule 1: Bare consonant ending (without asat)
    # Check if last character is a consonant (U+1000-U+1021)
    if "\u1000" <= last_char <= "\u1021":
        # Exception: single-character interjections are valid
        if len(word) == 1 and first_char in _ALLOWED_SINGLE_CONSONANTS:
            pass  # Allow အ, ဟ as standalone
        # Exception: Pali loanwords that legitimately end with bare consonants
        elif word in PALI_WHITELIST or word in VALID_PALI_BARE_ENDINGS:
            pass  # Allow Pali loanwords
        # Exception: Words with virama stacking (U+1039) are Pali/Sanskrit compounds
        # that can legitimately end with a bare consonant (implicit vowel)
        elif "\u1039" in word:
            pass  # Allow Pali compound terms with virama stacking
        else:
            return (True, "bare_consonant_end")

    # Rule 2: Stacked consonant at start (္ at position 0 or 1)
    if first_char == _STACKING_MARKER:
        return (True, "stacked_start")
    if len(word) >= 2 and word[1] == _STACKING_MARKER:
        return (True, "stacked_start")

    # Rule 3: Medial at start (ျ ြ ွ ှ)
    if first_char in MEDIALS_SET:
        return (True, "medial_start")

    # Rule 4: Dependent vowel at start
    if first_char in _DEPENDENT_VOWELS:
        return (True, "dependent_vowel_start")

    # Rule 5: Great Sa at start (ဿ)
    # Great Sa is a conjunct that only appears mid-word (e.g., ပြဿနာ, မနုဿ)
    if first_char == _GREAT_SA:
        return (True, "great_sa_start")

    # Rule 6: Asat + Anusvara sequence (်ံ)
    # Phonetically impossible: asat closes the syllable, anusvara needs a vowel
    # Examples: အ်ံ, က်ံ - these are garbage/OCR errors
    if _ASAT_ANUSVARA in word:
        return (True, "asat_anusvara_sequence")

    # Rule 7: Doubled independent vowel (e.g., ဤဤ, ဥဥ)
    # Two identical independent vowels as a word is always an OCR error
    # Example: ဤဤ, ဥဥ - these are not valid Myanmar words
    if len(word) == 2 and word[0] == word[1] and word[0] in _INDEPENDENT_VOWELS_SET:
        return (True, "doubled_independent_vowel")

    return (False, None)


_regex_segmenter_lock = threading.Lock()


def _get_regex_segmenter():
    """Return a cached RegexSegmenter instance (lazy singleton, thread-safe)."""
    if not hasattr(_get_regex_segmenter, "_instance"):
        with _regex_segmenter_lock:
            if not hasattr(_get_regex_segmenter, "_instance"):
                from myspellchecker.segmenters.regex import RegexSegmenter

                _get_regex_segmenter._instance = RegexSegmenter()  # type: ignore[attr-defined]
    return _get_regex_segmenter._instance  # type: ignore[attr-defined]


def _has_myanmar_content(word: str) -> bool:
    """Check that word contains Myanmar content and no punctuation/foreign chars.

    Returns False if:
    - Word has no Myanmar characters at all (bare punctuation)
    - Word contains ASCII punctuation: ( ) [ ] { } , . ; : ! ? " ' / \\\\ = - + < >
    - Word contains Myanmar punctuation: ၊ (U+104A) ။ (U+104B)
    - Word contains other non-Myanmar characters (Latin, digits 0-9, etc.)

    Allows: Core Myanmar block U+1000-U+109F (excluding ၊ ။), zero-width chars.
    """
    has_myanmar = False
    for ch in word:
        cp = ord(ch)
        # Myanmar punctuation ၊ (U+104A) and ။ (U+104B) are NOT word characters
        if cp == 0x104A or cp == 0x104B:
            return False
        if 0x1000 <= cp <= 0x109F:
            has_myanmar = True
        elif ch in _WORD_ALLOWED_NON_MYANMAR:
            continue
        else:
            return False
    return has_myanmar


def is_quality_word(word: str, *, allow_extended_myanmar: bool = False) -> bool:
    """
    Check if word meets quality standards for dictionary inclusion.

    This comprehensive quality filter combines multiple checks to ensure
    only well-formed, meaningful words enter the dictionary database.
    It is more strict than ``validate_word()`` and is designed for the
    database packager's filtering stage.

    Quality Checks (in order):
        0. Punctuation/non-Myanmar rejection via ``_has_myanmar_content()``
        1. Basic structural validation via ``validate_word()``
        2. Fragment pattern detection (consonant + asat/tone only)
        3. Double-ending detection (merged segmentation artifacts)
        4. Incomplete word detection (truncated patterns)
        5. Pure numeral filtering (date/quantity strings)
        6. Doubled consonant filtering (segmentation artifacts)
        7. Excessive length filtering (7+ syllables = concatenation)
        8. ``<ENG>`` token filtering (pipeline placeholder leak)
        9. Doubled initial consonant filtering (segmentation artifacts)
        10. Consecutive asat filtering (encoding errors)
        11. Broken virama stacking filtering (encoding errors)

    Args:
        word: Single word to check for dictionary inclusion.
        allow_extended_myanmar: If True, allow Extended Myanmar characters
            for non-Burmese Myanmar-script languages, including:
            - Extended Core Block (U+1050-U+109F)
            - Extended-A (U+AA60-U+AA7F)
            - Extended-B (U+A9E0-U+A9FF)
            - Non-standard core chars (U+1022, U+1028, U+1033-U+1035)
            Default is False to enforce strict Burmese-only scope.

    Returns:
        True if word should be included in dictionary, False if it
        should be filtered out.

    Example:
        >>> is_quality_word("ကျောင်း")  # Valid word
        True
        >>> is_quality_word("ဉ\u103a")  # Fragment (consonant+asat only)
        False
        >>> is_quality_word("တွင်င်း")  # Double-ending artifact
        False
        >>> is_quality_word("၁၂၃၄")  # Pure numeral (not a word)
        False
        >>> is_quality_word("မြန")  # Incomplete (missing asat)
        False

    Note:
        This function is used in the data pipeline's ``database_packager.py``
        to filter entries. Words failing this check are logged but not
        included in the final dictionary.

    See Also:
        - ``validate_word``: Basic structural validation
        - ``is_fragment_pattern``: Fragment-specific detection
        - ``is_incomplete_word``: Truncation-specific detection
        - ``get_quality_issues``: Detailed issue reporting
    """
    # Import here to avoid circular dependency (validate_word lives in validator.py)
    from myspellchecker.text.validator import validate_word

    if not word or not word.strip():
        return False

    # 0. Reject words containing punctuation or non-Myanmar characters.
    # A quality word must contain at least one core Myanmar char (U+1000-U+109F)
    # and must not contain ASCII punctuation, Myanmar punctuation (၊ ။), or other
    # non-Myanmar characters. This catches segmenter artifacts like "(ပ)", "သည်၊",
    # bare "(", ",", etc.
    if not _has_myanmar_content(word):
        return False

    # 1. Basic structural validation
    if not validate_word(word, allow_extended_myanmar=allow_extended_myanmar):
        return False

    # 2. Check for fragment patterns
    is_fragment, _ = is_fragment_pattern(word)
    if is_fragment:
        return False

    # 3. Check for incomplete word patterns
    is_incomplete, _ = is_incomplete_word(word)
    if is_incomplete:
        return False

    # 4. Check for segmentation fragments (myword artifacts)
    is_seg_fragment, _ = is_segmentation_fragment(word)
    if is_seg_fragment:
        return False

    # 5. Pure numeral filter
    if PURE_NUMERAL_PATTERN.match(word):
        return False

    # 6. Doubled consonant filter
    if DOUBLED_CONSONANT_PATTERN.match(word):
        return False

    # 7. Excessive length filter
    # Myanmar words rarely exceed 5-6 syllables. Words with 7+
    # syllables are concatenated artifacts from segmentation errors.
    # NOTE: Reduplicated forms (same syllable repeated, e.g., ရှိရှိ, များများ)
    # are valid Myanmar adverbs/intensifiers and should NOT be filtered.
    syllables = _get_regex_segmenter().segment_syllables(word)
    if len(syllables) > 7:
        return False

    # 8. <ENG> token filter
    # The <ENG> placeholder from the pipeline should never enter
    # the dictionary.
    if "<ENG>" in word:
        return False

    # 9. Doubled initial consonant pattern
    # A word starting with the same consonant twice without stacking
    # (U+1039) is invalid when followed by a vowel/medial.
    # Excludes asat (U+103A) and virama (U+1039) — စစ်*, နန်း* are valid.
    if DOUBLED_INITIAL_CONSONANT_PATTERN.match(word):
        return False

    # 10. Consecutive asat (doubled ်) — always encoding error
    if CONSECUTIVE_ASAT_PATTERN.search(word):
        return False

    # 11. Broken virama stacking — U+1039 must be followed by consonant
    if BROKEN_VIRAMA_PATTERN.search(word):
        return False

    return True


def get_quality_issues(word: str, *, allow_extended_myanmar: bool = False) -> list[tuple[str, str]]:
    """
    Get list of quality issues for a word.

    Useful for debugging and generating quality reports.

    Args:
        word: Word to check
        allow_extended_myanmar: If True, allow Extended Myanmar characters
            for non-Burmese Myanmar-script languages, including:
            - Extended Core Block (U+1050-U+109F)
            - Extended-A (U+AA60-U+AA7F)
            - Extended-B (U+A9E0-U+A9FF)
            - Non-standard core chars (U+1022, U+1028, U+1033-U+1035)
            Default is False to enforce strict Burmese-only scope.

    Returns:
        List of (issue_type, description) tuples. Empty list if no issues.

    Example:
        >>> get_quality_issues("တွင်င်း")
        [('fragment', 'double_ending pattern detected')]
    """
    # Import here to avoid circular dependency (validate_text lives in validator.py)
    from myspellchecker.text.validator import validate_text

    issues: list[tuple[str, str]] = []

    if not word or not word.strip():
        issues.append(("empty", "Empty or whitespace-only"))
        return issues

    # Check basic validation
    result = validate_text(word, allow_extended_myanmar=allow_extended_myanmar)
    if not result.is_valid:
        for issue, desc in result.issues:
            issues.append((issue.value, desc))

    # Check fragment patterns
    is_fragment, pattern_name = is_fragment_pattern(word)
    if is_fragment:
        issues.append(("fragment", f"{pattern_name} pattern detected"))

    # Check incomplete patterns
    is_incomplete, pattern_name = is_incomplete_word(word)
    if is_incomplete:
        issues.append(("incomplete", f"{pattern_name} pattern detected"))

    # Check segmentation fragments
    is_seg_fragment, fragment_type = is_segmentation_fragment(word)
    if is_seg_fragment:
        issues.append(("segmentation_fragment", f"{fragment_type} detected"))

    # Pure numeral
    if PURE_NUMERAL_PATTERN.match(word):
        issues.append(("pure_numeral", "Pure numeral sequence"))

    # Doubled consonant
    if DOUBLED_CONSONANT_PATTERN.match(word):
        issues.append(("doubled_consonant", "Doubled consonant sequence"))

    return issues
