# distutils: language=c++
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=True
# cython: cdivision=True
"""
Cython-optimized segmentation repair module.

High-performance repair logic for fixing incorrectly segmented Myanmar words.
Uses C++ sets and pre-compiled patterns for maximum throughput.
"""

from libcpp.set cimport set as cpp_set
from libcpp.string cimport string
from libcpp.vector cimport vector
from cpython.unicode cimport PyUnicode_AsUTF8String, PyUnicode_DecodeUTF8

# Import is done lazily to avoid circular import issues

# C++ sets for O(1) lookups
cdef cpp_set[string] cpp_suspicious_fragments
cdef cpp_set[string] cpp_closing_chars
cdef cpp_set[string] cpp_valid_start_consonants
cdef cpp_set[string] cpp_valid_start_vowels
cdef cpp_set[string] cpp_double_ending_fragments

# Python validator instance (kept as Python object for complex validation)
cdef object _validator = None

# Myanmar Unicode ranges for valid start characters
# IMPORTANT: These constants MUST match core/constants/myanmar_constants.py
# to ensure consistent behavior between Python and Cython code paths.
# Consonants: U+1000-U+1020 + U+103F (Great Sa) (matches CONSONANTS in core/constants)
# Plus vowel carrier U+1021 (အ) which is needed for valid word starts
# Independent vowels: U+1023-U+1027, U+1029, U+102A
cdef list CONSONANTS = [
    'က', 'ခ', 'ဂ', 'ဃ', 'င', 'စ', 'ဆ', 'ဇ', 'ဈ', 'ဉ', 'ည',
    'ဋ', 'ဌ', 'ဍ', 'ဎ', 'ဏ', 'တ', 'ထ', 'ဒ', 'ဓ', 'န',
    'ပ', 'ဖ', 'ဗ', 'ဘ', 'မ', 'ယ', 'ရ', 'လ', 'ဝ', 'သ',
    'ဟ', 'ဠ', 'အ', 'ဿ'
]

cdef list INDEPENDENT_VOWELS = ['ဣ', 'ဤ', 'ဥ', 'ဦ', 'ဧ', 'ဩ', 'ဪ']

# Myanmar numerals (valid standalone tokens, should NOT be merged back)
cdef list NUMERALS = ['၀', '၁', '၂', '၃', '၄', '၅', '၆', '၇', '၈', '၉']

# C++ set for numerals
cdef cpp_set[string] cpp_numerals

# Suspicious fragments that are often incorrectly split
cdef list SUSPICIOUS_FRAGMENTS = ['င်း', 'င့်', 'န့်']

# Closing characters that indicate syllable completion
cdef list CLOSING_CHARS = ['း', '့', '်']

# Double-ending pattern suffix (would create invalid *်င်း, *းင်း, *့င်း)
# These fragments, when merged after a closed syllable, create invalid patterns
cdef list DOUBLE_ENDING_FRAGMENTS = ['င်း', 'င့်', 'န့်']

# Module-level scope flag for extended Myanmar support
# Can be set via set_allow_extended_myanmar() before processing
cdef bint _allow_extended_myanmar = False


def set_allow_extended_myanmar(bint allow):
    """Set whether to allow extended Myanmar characters in validation."""
    global _allow_extended_myanmar
    _allow_extended_myanmar = allow


cdef void _init_sets():
    """Initialize C++ sets with Myanmar character data."""
    global cpp_suspicious_fragments, cpp_closing_chars
    global cpp_valid_start_consonants, cpp_valid_start_vowels
    global cpp_numerals, cpp_double_ending_fragments

    cdef str s
    cdef bytes b

    cpp_suspicious_fragments.clear()
    for s in SUSPICIOUS_FRAGMENTS:
        b = s.encode('utf-8')
        cpp_suspicious_fragments.insert(string(b))

    cpp_closing_chars.clear()
    for s in CLOSING_CHARS:
        b = s.encode('utf-8')
        cpp_closing_chars.insert(string(b))

    cpp_valid_start_consonants.clear()
    for s in CONSONANTS:
        b = s.encode('utf-8')
        cpp_valid_start_consonants.insert(string(b))

    cpp_valid_start_vowels.clear()
    for s in INDEPENDENT_VOWELS:
        b = s.encode('utf-8')
        cpp_valid_start_vowels.insert(string(b))

    cpp_numerals.clear()
    for s in NUMERALS:
        b = s.encode('utf-8')
        cpp_numerals.insert(string(b))

    cpp_double_ending_fragments.clear()
    for s in DOUBLE_ENDING_FRAGMENTS:
        b = s.encode('utf-8')
        cpp_double_ending_fragments.insert(string(b))


cdef bint _is_suspicious_fragment(str token):
    """Check if token is a known suspicious fragment."""
    cdef bytes b = token.encode('utf-8')
    return cpp_suspicious_fragments.count(string(b)) > 0


cdef bint _is_double_ending_fragment(str token):
    """Check if token is a fragment that would cause double-ending when merged after closed syllable."""
    cdef bytes b = token.encode('utf-8')
    return cpp_double_ending_fragments.count(string(b)) > 0


cdef bint _would_create_double_ending(str prev_token, str current_token):
    """
    Check if merging prev_token + current_token would create a double-ending pattern.

    Double-ending patterns occur when a closed syllable (ending in ်, း, or ့)
    is merged with a fragment like င်း, resulting in patterns like တွင်င်း.

    Args:
        prev_token: The previous token (e.g., "တွင်")
        current_token: The current token to potentially merge (e.g., "င်း")

    Returns:
        True if merging would create an invalid double-ending pattern
    """
    if not prev_token or not current_token:
        return False

    # Check if prev ends with closing char AND current is a double-ending fragment
    if _has_closing_char(prev_token) and _is_double_ending_fragment(current_token):
        return True

    return False


cdef bint _has_closing_char(str token):
    """Check if token ends with a closing character."""
    if not token:
        return False

    cdef str last_char = token[-1]
    cdef bytes b = last_char.encode('utf-8')
    return cpp_closing_chars.count(string(b)) > 0


cdef bint _has_valid_start(str token):
    """Check if token starts with a valid base character (consonant, independent vowel, or numeral)."""
    if not token:
        return False

    # Get first character (handling Myanmar multi-byte)
    cdef str first_char = token[0]
    cdef bytes b = first_char.encode('utf-8')

    # Check if it's a valid starting character
    if cpp_valid_start_consonants.count(string(b)) > 0:
        return True
    if cpp_valid_start_vowels.count(string(b)) > 0:
        return True
    # Myanmar numerals are valid standalone tokens (e.g., "၁၂၃")
    if cpp_numerals.count(string(b)) > 0:
        return True

    return False


cdef bint _starts_with_numeral(str token):
    """Check if token starts with a Myanmar numeral."""
    if not token:
        return False

    cdef str first_char = token[0]
    cdef bytes b = first_char.encode('utf-8')
    return cpp_numerals.count(string(b)) > 0


cdef bint _validate_syllable(str token):
    """Validate syllable using the Python validator."""
    global _validator
    if _validator is None:
        # Lazy import to avoid circular dependencies
        from myspellchecker.core.syllable_rules import SyllableRuleValidator
        _validator = SyllableRuleValidator(
            allow_extended_myanmar=_allow_extended_myanmar
        )
    return _validator.validate(token)


cpdef list repair_batch(list tokens_list):
    """
    Repair a batch of token lists in a single call.

    Args:
        tokens_list: List of token lists (e.g., [["ကျော", "င်း"], ["သည်"]])

    Returns:
        List of repaired token lists
    """
    cdef list results = []
    cdef list tokens

    for tokens in tokens_list:
        results.append(repair(tokens))

    return results


cpdef list repair(list tokens):
    """
    Repair a list of tokens by merging invalid fragments into previous words.

    This is the Cython-optimized version of SegmentationRepair.repair().

    Args:
        tokens: List of word tokens (e.g., ["ကျော", "င်း", "သည်"])

    Returns:
        List of repaired tokens (e.g., ["ကျောင်း", "သည်"])
    """
    if not tokens:
        return []

    # Ensure sets are initialized
    if cpp_suspicious_fragments.empty():
        _init_sets()

    cdef list repaired = [tokens[0]]
    cdef int i
    cdef int n = len(tokens)
    cdef str current_token, prev_token, candidate
    cdef bint is_invalid_start, is_suspicious, is_invalid_syllable
    cdef bint prev_is_closed, is_numeral_token

    for i in range(1, n):
        current_token = tokens[i]
        prev_token = repaired[-1]

        # Condition 1: Invalid start (doesn't begin with consonant/vowel/numeral)
        is_invalid_start = not _has_valid_start(current_token)

        # Condition 2: Known suspicious fragment
        is_suspicious = _is_suspicious_fragment(current_token)

        # Condition 3: Invalid syllable structure (expensive, do last)
        # SKIP for numeral tokens - they are valid standalone and don't follow
        # syllable structure rules (e.g., "၁၉၅၃" is valid but not a syllable)
        is_invalid_syllable = False
        is_numeral_token = _starts_with_numeral(current_token)
        if not is_invalid_start and not is_suspicious and not is_numeral_token:
            is_invalid_syllable = not _validate_syllable(current_token)

        if is_invalid_start or is_suspicious or is_invalid_syllable:
            # Check if previous token is "closed"
            prev_is_closed = _has_closing_char(prev_token) if prev_token else False

            # Prevent double-ending merges
            # Don't merge if it would create patterns like တွင်င်း, သော်င်း, etc.
            if _would_create_double_ending(prev_token, current_token):
                repaired.append(current_token)
                continue

            if prev_is_closed:
                repaired.append(current_token)
                continue

            # Attempt merge
            candidate = prev_token + current_token

            # Only merge if result is valid
            if _validate_syllable(candidate):
                repaired[-1] = candidate
            else:
                repaired.append(current_token)
        else:
            # Valid starter - new word
            repaired.append(current_token)

    return repaired


def init_repair_module():
    """
    Pre-initialize the repair module.
    Call this in the parent process before forking workers.
    """
    global _validator
    _init_sets()
    # Lazy import to avoid circular dependencies
    from myspellchecker.core.syllable_rules import SyllableRuleValidator
    _validator = SyllableRuleValidator(
        allow_extended_myanmar=_allow_extended_myanmar
    )


# =============================================================================
# Test-only helper functions for constant sync verification
# =============================================================================
# These functions expose the embedded character lists for testing purposes.
# They allow automated tests to verify that Cython constants match the
# canonical Python constants in core/constants/myanmar_constants.py.
# DO NOT use these in production code paths.

def get_repair_consonants():
    """
    Return the consonant list used by repair module.

    Test-only: Used to verify sync with core/constants/myanmar_constants.py.

    Returns:
        set: The consonant characters used for valid start detection.
    """
    return set(CONSONANTS)


def get_repair_independent_vowels():
    """
    Return the independent vowel list used by repair module.

    Test-only: Used to verify sync with core/constants/myanmar_constants.py.

    Returns:
        set: The independent vowel characters used for valid start detection.
    """
    return set(INDEPENDENT_VOWELS)


def get_repair_numerals():
    """
    Return the numeral list used by repair module.

    Test-only: Used to verify sync with core/constants/myanmar_constants.py.

    Returns:
        set: The Myanmar numeral characters.
    """
    return set(NUMERALS)


class CythonSegmentationRepair:
    """
    Drop-in replacement for SegmentationRepair using Cython backend.

    This class provides the same interface as the pure Python version
    but delegates to the optimized Cython functions.
    """

    def __init__(self):
        """Initialize the repair instance."""
        # Ensure module is initialized
        init_repair_module()

    def repair(self, tokens: list) -> list:
        """
        Repair a list of tokens by merging invalid fragments.

        Args:
            tokens: List of word tokens

        Returns:
            List of repaired tokens
        """
        return repair(tokens)

    def repair_batch(self, tokens_list: list) -> list:
        """
        Repair multiple token lists in one call.

        Args:
            tokens_list: List of token lists

        Returns:
            List of repaired token lists
        """
        return repair_batch(tokens_list)
