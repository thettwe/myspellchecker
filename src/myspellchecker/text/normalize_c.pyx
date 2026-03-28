# distutils: language=c++
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""
Cython-optimized text normalization for Myanmar Unicode.

This module provides high-performance text normalization functions for
Myanmar script, handling Unicode normalization, zero-width character
removal, and character reordering.

Architecture Role
=================
This is **Layer 1** of the normalization hierarchy - the performance-critical
Cython/C++ implementation layer. It provides the building blocks used by
higher-level Python functions in ``normalize.py`` and the service layer in
``normalization_service.py``.

Normalization Hierarchy::

    Layer 3: NormalizationService (normalization_service.py)
        ↓ calls
    Layer 2: normalize.py (Python wrappers)
        ↓ calls
    Layer 1: normalize_c.pyx (this module - Cython/C++)

Public Functions
================
The following functions are exported and can be imported directly::

    from myspellchecker.text.normalize_c import (
        remove_zero_width_chars,
        reorder_myanmar_diacritics,
        get_myanmar_ratio,
        is_myanmar_string,
        clean_text_for_segmentation,
        segment_syllables_c,
    )

Key Features
============
- Fast UTF-8 codepoint decoding with inline optimization
- Zero-width character (ZWSP, ZWNJ, ZWJ, BOM) removal
- Myanmar Unicode character classification (consonants, vowels)
- Medial character reordering for canonical form
- GIL-free operations for parallel processing (where applicable)

Character Classes
=================
- Consonants: U+1000-U+1020 (excluding U+1021), U+103F, U+104E
- Independent Vowels: U+1021-U+102A (inclusive)
- Zero-Width: U+200B (ZWSP), U+200C (ZWNJ), U+200D (ZWJ), U+FEFF (BOM)

Diacritic Ordering (UTN #11 Compliant)
======================================
The reorder_myanmar_diacritics function sorts diacritics according to
Unicode Technical Note #11 (Myanmar Collation):

    Medials: Ya (103B) < Ra (103C) < Wa (103D) < Ha (103E)
    Vowels: E (1031) < Upper (102D/E/32) < Tall (102B/C) < Lower (102F/30)
    Finals: Asat (103A) < Anusvara (1036) < Dot (1037) < Visarga (1038)
    Virama (1039) always comes last for stacking

Performance
===========
- ~20x faster than pure Python implementation
- Optimized for Myanmar Unicode range (U+1000-U+109F)
- Memory-efficient in-place processing
- Uses C++ STL containers (unordered_set, vector) for speed

Compilation
===========
This module requires compilation. After modifying .pyx files::

    python setup.py build_ext --inplace

Edge Cases Handled
==================
- **Empty strings**: All functions safely return empty string/False/0.0
- **Truncated UTF-8**: Returns Unicode replacement character (U+FFFD)
- **Non-Myanmar text**: Passed through unchanged
- **Extended Myanmar**: Recognized when allow_extended=True, including:
  - Extended Core Block (U+1050-U+109F)
  - Extended-A (U+AA60-U+AA7F)
  - Extended-B (U+A9E0-U+A9FF)
  - Non-standard core chars (U+1022, U+1028, U+1033-U+1035)

Example
=======
>>> from myspellchecker.text import normalize_c
>>> text = "မြန်\\u200bမာ"  # with ZWSP
>>> normalize_c.remove_zero_width_chars(text)
'မြန်မာ'

See Also
========
- normalize.py: Python wrapper with higher-level functions
- normalization_service.py: Purpose-specific normalization service
- core/constants/myanmar_constants.py: Myanmar Unicode constant definitions
"""

from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.unordered_set cimport unordered_set
from libcpp.utility cimport pair
from libcpp cimport bool
from libc.stdint cimport uint32_t

cdef unordered_set[int] CONSONANTS
cdef unordered_set[int] INDEPENDENT_VOWELS
cdef unordered_set[int] ZERO_WIDTH_CHARS
cdef unordered_set[int] LEADING_PUNCT_CHARS

# Initialize constants
# See core/constants/myanmar_constants.py for canonical definitions
# Note: This uses C-level unordered_set for performance in hot paths
cdef void init_constants():
    # Consonants: U+1000 to U+1020 (exclusive of U+1021)
    # U+1021 (အ) is an independent vowel, not a consonant
    for i in range(0x1000, 0x1021):  # 0x1021 is NOT included
        CONSONANTS.insert(i)
    # Extra Consonants
    CONSONANTS.insert(0x103F)  # Great Sa (ဿ)
    # Note: U+104E (၎) is a symbol, NOT a consonant. Do not add to CONSONANTS.
    # See myanmar_constants.py for canonical consonant definitions.

    # Independent Vowels: U+1021 to U+102A (inclusive)
    # U+1021 (အ) serves as consonant carrier and independent vowel 'a'
    for i in range(0x1021, 0x102B):  # 0x102B is NOT included
        INDEPENDENT_VOWELS.insert(i)
        
    # Zero Width
    ZERO_WIDTH_CHARS.insert(0x200B) # ZWSP
    ZERO_WIDTH_CHARS.insert(0x200C) # ZWNJ
    ZERO_WIDTH_CHARS.insert(0x200D) # ZWJ
    ZERO_WIDTH_CHARS.insert(0xFEFF) # BOM

    # Leading Punctuation (ASCII brackets/quotes + Myanmar punctuation)
    # These are skipped when finding first significant character
    LEADING_PUNCT_CHARS.insert(ord('('))
    LEADING_PUNCT_CHARS.insert(ord(')'))
    LEADING_PUNCT_CHARS.insert(ord('['))
    LEADING_PUNCT_CHARS.insert(ord(']'))
    LEADING_PUNCT_CHARS.insert(ord('{'))
    LEADING_PUNCT_CHARS.insert(ord('}'))
    LEADING_PUNCT_CHARS.insert(ord('"'))
    LEADING_PUNCT_CHARS.insert(ord("'"))
    LEADING_PUNCT_CHARS.insert(ord('`'))
    LEADING_PUNCT_CHARS.insert(ord('<'))
    LEADING_PUNCT_CHARS.insert(ord('>'))
    # Myanmar punctuation U+104A-U+104F
    LEADING_PUNCT_CHARS.insert(0x104A)  # ၊ (comma)
    LEADING_PUNCT_CHARS.insert(0x104B)  # ။ (full stop)
    LEADING_PUNCT_CHARS.insert(0x104C)  # ၌ (locative)
    LEADING_PUNCT_CHARS.insert(0x104D)  # ၍ (completed)
    LEADING_PUNCT_CHARS.insert(0x104E)  # ၎ (aforementioned)
    LEADING_PUNCT_CHARS.insert(0x104F)  # ၏ (genitive)

init_constants()


# OPTIMIZATION: Inline UTF-8 decoder that returns (codepoint, byte_length) in one call
# This reduces the repeated if-else chain overhead when both values are needed
cdef inline pair[int, int] decode_utf8_char(const char* data, int pos, int max_len) noexcept nogil:
    """
    Decode a single UTF-8 character at position pos with full validation.
    Returns (codepoint, byte_length) pair.
    Issue #1240: Added continuation byte validation.
    """
    cdef unsigned char b0 = <unsigned char>data[pos]
    cdef unsigned char b1, b2, b3  # Declare at function scope
    cdef int cp, char_len

    if b0 < 0x80:
        return pair[int, int](b0, 1)
    elif (b0 & 0xE0) == 0xC0:
        if pos + 1 < max_len:
            # Validate continuation byte has pattern 10xxxxxx (Issue #1240)
            b1 = <unsigned char>data[pos + 1]
            if (b1 & 0xC0) != 0x80:
                return pair[int, int](0xFFFD, 1)  # Invalid continuation byte
            cp = ((b0 & 0x1F) << 6) | (b1 & 0x3F)
            return pair[int, int](cp, 2)
        return pair[int, int](0xFFFD, 1)  # Truncated: use Unicode replacement char
    elif (b0 & 0xF0) == 0xE0:
        if pos + 2 < max_len:
            # Validate continuation bytes (Issue #1240)
            b1 = <unsigned char>data[pos + 1]
            b2 = <unsigned char>data[pos + 2]
            if (b1 & 0xC0) != 0x80 or (b2 & 0xC0) != 0x80:
                return pair[int, int](0xFFFD, 1)  # Invalid continuation bytes
            cp = ((b0 & 0x0F) << 12) | ((b1 & 0x3F) << 6) | (b2 & 0x3F)
            return pair[int, int](cp, 3)
        return pair[int, int](0xFFFD, 1)  # Truncated: use Unicode replacement char
    else:  # 4-byte (rare for Myanmar)
        if pos + 3 < max_len:
            # Validate continuation bytes (Issue #1240)
            b1 = <unsigned char>data[pos + 1]
            b2 = <unsigned char>data[pos + 2]
            b3 = <unsigned char>data[pos + 3]
            if (b1 & 0xC0) != 0x80 or (b2 & 0xC0) != 0x80 or (b3 & 0xC0) != 0x80:
                return pair[int, int](0xFFFD, 1)  # Invalid continuation bytes
            cp = ((b0 & 0x07) << 18) | ((b1 & 0x3F) << 12) | ((b2 & 0x3F) << 6) | (b3 & 0x3F)
            return pair[int, int](cp, 4)
        return pair[int, int](0xFFFD, 1)  # Truncated: use Unicode replacement char


cdef bint is_myanmar_char(int cp):
    """Check if codepoint is any Myanmar character (all blocks)."""
    # Main Block
    if 0x1000 <= cp <= 0x109F: return True
    # Extended A
    if 0xAA60 <= cp <= 0xAA7F: return True
    # Extended B
    if 0xA9E0 <= cp <= 0xA9FF: return True
    return False


cdef bint is_myanmar_char_core(int cp):
    """
    Check if codepoint is core Burmese only (U+1000-U+104F).

    This excludes:
    - Extended Core Block (U+1050-U+109F): Shan, Mon, Karen characters
    - Extended-A (U+AA60-U+AA7F): Shan, Mon
    - Extended-B (U+A9E0-U+A9FF): Shan, Mon

    Used for strict Burmese scope in get_myanmar_ratio.

    Note: This is a pure Unicode block range check. It returns True for ALL
    codepoints in U+1000-U+104F, including non-standard chars (U+1022, U+1028,
    U+1033-U+1035). Use is_myanmar_char_scoped() for scope-aware checks.
    """
    return 0x1000 <= cp <= 0x104F


cdef bint is_non_standard_core(int cp):
    """
    Check if codepoint is a non-standard char within the core block.

    These are Mon/Shan characters that exist within U+1000-U+104F but are
    not used in standard Burmese:
    - U+1022: Shan Letter A
    - U+1028: Mon E
    - U+1033-U+1035: Mon vowels (II, U, UU)

    Used to exclude non-standard chars in strict Burmese mode.
    """
    return cp == 0x1022 or cp == 0x1028 or (0x1033 <= cp <= 0x1035)


cdef bint is_myanmar_char_scoped(int cp, bint allow_extended):
    """
    Check if codepoint is Myanmar based on allow_extended flag.

    When allow_extended=True: Includes all Myanmar blocks including non-standard
    When allow_extended=False: Only strict Burmese (U+1000-U+104F minus non-standard)

    Scope-aware character classification.
    Non-standard core chars excluded in strict mode.
    """
    # Non-standard core chars (Mon/Shan within U+1000-U+104F) only allowed when extended
    if is_non_standard_core(cp):
        return allow_extended

    # Standard core block chars always allowed
    if is_myanmar_char_core(cp):
        return True

    # Extended blocks only when allow_extended=True
    if not allow_extended:
        return False

    # Extended Core Block (Shan, Mon, Karen in main block)
    if 0x1050 <= cp <= 0x109F:
        return True
    # Extended-A (U+AA60-U+AA7F)
    if 0xAA60 <= cp <= 0xAA7F:
        return True
    # Extended-B (U+A9E0-U+A9FF)
    if 0xA9E0 <= cp <= 0xA9FF:
        return True
    return False

cdef int get_order_weight(int cp):
    """
    Return sorting weight for Myanmar diacritics.
    Matches ORDER_WEIGHTS in constants.py (multiplied by 10 for int precision).
    """
    # Medials - UTN #11 canonical order: Ya < Ra < Wa < Ha
    # Reference: https://unicode.org/notes/tn11/UTN11_4.pdf (Section: Diacritic storage order)
    # Slot order: Medial Y (slot 3) < Medial R (slot 4) < Medial W (slot 5) < Medial H (slot 6)
    if cp == 0x103B: return 100  # Ya (ျ) - slot 3, comes first
    if cp == 0x103C: return 110  # Ra (ြ) - slot 4, comes second
    if cp == 0x103D: return 120  # Wa (ွ) - slot 5
    if cp == 0x103E: return 130  # Ha (ှ) - slot 6
    
    # Vowels (20-22) -> 200-220
    if cp == 0x1031: return 200 # E
    
    # Upper Vowels (I, II, AI)
    if cp == 0x102D or cp == 0x102E or cp == 0x1032: return 210
    
    # A / AA (Tall A, AA)
    if cp == 0x102B or cp == 0x102C: return 214
    
    # Asat
    if cp == 0x103A: return 215
    
    # Lower Vowels (U, UU)
    if cp == 0x102F or cp == 0x1030: return 220
    
    # Finals/Tones (30-40) -> 300-400
    if cp == 0x1036: return 300 # Anusvara
    if cp == 0x1037: return 320 # Dot Below
    if cp == 0x1038: return 330 # Visarga
    if cp == 0x1039: return 400 # Virama
    
    # Default high weight
    return 999

# Comparator for sorting
cdef bool compare_diacritics(int a, int b):
    return get_order_weight(a) < get_order_weight(b)

cpdef str remove_zero_width_chars(str text):
    cdef string s = text.encode('utf-8')
    cdef string result
    result.reserve(s.length())

    cdef int i = 0
    cdef int n = s.length()
    cdef int cp
    cdef int char_len
    cdef pair[int, int] decoded  # Declare for safe UTF-8 decoding

    while i < n:
        # Use safe UTF-8 decoding helper with bounds checking
        # Prevents buffer overflow when reading continuation bytes (Issue #1236)
        decoded = decode_utf8_char(s.c_str(), i, n)
        cp = decoded.first
        char_len = decoded.second

        if ZERO_WIDTH_CHARS.count(cp) == 0:
            result.append(s.substr(i, char_len))

        i += char_len

    return result.decode('utf-8')

cpdef str reorder_myanmar_diacritics(str text):
    """
    Optimized C++ implementation of diacritic reordering.
    """
    if not text:
        return text

    cdef string s = text.encode('utf-8')
    cdef string result
    result.reserve(s.length())
    
    cdef int i = 0
    cdef int n = s.length()
    cdef int cp
    cdef int char_len
    cdef pair[int, int] decoded  # For safe UTF-8 decoding (outer loop)
    cdef pair[int, int] next_decoded  # For safe UTF-8 decoding (inner loop)

    # We need to process character by character
    # To reorder, we need to collect a syllable buffer

    # Vector to store diacritics (code point, original bytes)
    cdef vector[pair[int, string]] diacritics
    cdef string base_char_bytes

    # Helper to peek next char
    cdef int next_cp
    cdef int next_len
    
    while i < n:
        # Use safe UTF-8 decoding helper (bounds checking - Issue #1236)
        decoded = decode_utf8_char(s.c_str(), i, n)
        cp = decoded.first
        char_len = decoded.second

        # Check if Base Consonant
        if CONSONANTS.count(cp) or INDEPENDENT_VOWELS.count(cp):
            # Start of Syllable
            base_char_bytes = s.substr(i, char_len)
            i += char_len
            
            diacritics.clear()
            
            # Look ahead for diacritics
            while i < n:
                # Use safe UTF-8 decoding for next character (Issue #1236)
                next_decoded = decode_utf8_char(s.c_str(), i, n)
                next_cp = next_decoded.first
                next_len = next_decoded.second

                # Stop if next is base or non-myanmar
                if CONSONANTS.count(next_cp) or INDEPENDENT_VOWELS.count(next_cp) or not is_myanmar_char(next_cp):
                    break
                
                # Use substr for string assignment
                diacritics.push_back(pair[int, string](next_cp, string()))
                diacritics.back().second = s.substr(i, next_len)
                
                i += next_len
            
            # Sort Diacritics
            # Simple bubble sort or similar is fine for small N (diacritics usually < 5)
            # But std::sort requires a comparator that takes pairs.
            # Let's do simple manual bubble sort for tiny arrays
            for k in range(diacritics.size()):
                for j in range(0, diacritics.size() - k - 1):
                    if get_order_weight(diacritics[j].first) > get_order_weight(diacritics[j+1].first):
                        # Swap
                        temp = diacritics[j]
                        diacritics[j] = diacritics[j+1]
                        diacritics[j+1] = temp
            
            # Append Base
            result.append(base_char_bytes)
            # Append Sorted Diacritics
            for k in range(diacritics.size()):
                result.append(diacritics[k].second)
                
        else:
            # Not a base, just append
            result.append(s.substr(i, char_len))
            i += char_len
            
    return result.decode('utf-8')

cpdef bint is_myanmar_string(str text):
    """
    Check if the first character of the string is a Myanmar character.
    Fast C++ implementation to replace regex search.

    Note: This is unscoped and includes all Myanmar blocks.
    Use is_myanmar_string_scoped() for scope-aware checks.
    """
    if not text:
        return False

    cdef string s = text.encode('utf-8')
    if s.empty():
        return False

    # Decode first char
    cdef int cp
    cdef unsigned char unsigned_char = <unsigned char>s[0]

    if unsigned_char < 0x80:
        cp = unsigned_char
    elif (unsigned_char & 0xE0) == 0xC0:
        if s.length() < 2: return False
        cp = ((unsigned_char & 0x1F) << 6) | (<unsigned char>s[1] & 0x3F)
    elif (unsigned_char & 0xF0) == 0xE0:
        if s.length() < 3: return False
        cp = ((unsigned_char & 0x0F) << 12) | \
             ((<unsigned char>s[1] & 0x3F) << 6) | \
             (<unsigned char>s[2] & 0x3F)
    else:
        if s.length() < 4: return False
        cp = ((unsigned_char & 0x07) << 18) | \
             ((<unsigned char>s[1] & 0x3F) << 12) | \
             ((<unsigned char>s[2] & 0x3F) << 6) | \
             (<unsigned char>s[3] & 0x3F)

    return is_myanmar_char(cp)


cdef inline bint is_skip_char(int cp) noexcept nogil:
    """Check if codepoint should be skipped when finding first significant char.

    Skips:
    - ASCII whitespace and control chars (U+0000-U+0020)
    - Ideographic space (U+3000)
    - Zero-width characters (ZWSP, ZWNJ, ZWJ, BOM)
    - Leading punctuation (brackets, quotes, Myanmar punctuation)
    """
    # ASCII whitespace and control characters
    if cp <= 0x20:
        return True
    # Ideographic space
    if cp == 0x3000:
        return True
    # Zero-width characters
    if ZERO_WIDTH_CHARS.count(cp) > 0:
        return True
    # Leading punctuation
    if LEADING_PUNCT_CHARS.count(cp) > 0:
        return True
    return False


cpdef bint is_myanmar_string_scoped(str text, bint allow_extended=False):
    """
    Check if the first significant character is Myanmar (scope-aware).

    Skips leading whitespace, zero-width characters, and punctuation to find
    the first significant character, then checks if it's Myanmar.

    When allow_extended=True: Includes all Myanmar blocks including non-standard
    When allow_extended=False: Only strict Burmese (excludes non-standard core chars)

    Scope-aware string check for batch pipeline filtering.
    Uses first-significant-char semantics for consistent behavior.
    """
    if not text:
        return False

    cdef string s = text.encode('utf-8')
    cdef int length = <int>s.length()
    if length == 0:
        return False

    cdef const char* data = s.c_str()
    cdef int pos = 0
    cdef int cp
    cdef int char_len
    cdef pair[int, int] decoded

    # Iterate to find first significant character
    while pos < length:
        decoded = decode_utf8_char(data, pos, length)
        cp = decoded.first
        char_len = decoded.second

        # Skip insignificant characters
        if is_skip_char(cp):
            pos += char_len
            continue

        # Found significant char - check if Myanmar
        return is_myanmar_char_scoped(cp, allow_extended)

    # No significant character found
    return False

def get_myanmar_ratio(str text, bint allow_extended=True):
    """
    Calculate the ratio of Myanmar characters in the text.
    Ignores whitespace and common punctuation.

    Args:
        text: Input text string.
        allow_extended: If True (default), count all Myanmar blocks including:
            - Extended Core Block (U+1050-U+109F)
            - Extended-A (U+AA60-U+AA7F)
            - Extended-B (U+A9E0-U+A9FF)
            - Non-standard core chars (U+1022, U+1028, U+1033-U+1035)
            If False, only count core Burmese (U+1000-U+104F excluding non-standard).
            This provides scope-aware ratio for strict Burmese mode.

    Returns:
        Ratio of Myanmar characters (0.0 to 1.0).
    """
    if not text:
        return 0.0

    cdef string s = text.encode('utf-8')
    cdef int i = 0
    cdef int n = s.length()
    cdef int cp
    cdef int char_len
    cdef pair[int, int] decoded  # For safe UTF-8 decoding

    cdef int total_chars = 0
    cdef int myanmar_chars = 0

    while i < n:
        # Decode
        # Safe UTF-8 decode (Issue #1236)
        decoded = decode_utf8_char(s.c_str(), i, n)
        cp = decoded.first
        char_len = decoded.second

        i += char_len

        # Logic:
        # Skip Whitespace
        # cp == 32 (space), 9 (tab), 10 (newline), 13 (cr)
        if cp == 32 or cp == 9 or cp == 10 or cp == 13:
            continue

        # Skip Common Punctuation (Approximate check for speed)
        # ASCII Punctuation: 33-47, 58-64, 91-96, 123-126
        if (33 <= cp <= 47) or (58 <= cp <= 64) or (91 <= cp <= 96) or (123 <= cp <= 126):
            continue

        # Skip Myanmar Punctuation (104A-104F)
        if 0x104A <= cp <= 0x104F:
            continue

        total_chars += 1

        # Use scoped character check
        if is_myanmar_char_scoped(cp, allow_extended):
            myanmar_chars += 1

    if total_chars == 0:
        return 0.0

    return <double>myanmar_chars / <double>total_chars

cpdef str clean_text_for_segmentation(str text, bint allow_extended=False):
    """
    Remove spaces, punctuation, and non-Myanmar characters to prepare for Viterbi.
    This ensures we only segment valid Myanmar text.
    OPTIMIZED: Uses inline UTF-8 decoder helper.

    Args:
        text: Text to clean for segmentation.
        allow_extended: If True, include Extended Myanmar characters:
                       - Extended Core Block (U+1050-U+109F)
                       - Extended-A (U+AA60-U+AA7F)
                       - Extended-B (U+A9E0-U+A9FF)
                       - Non-standard core chars (U+1022, U+1028, U+1033-U+1035)
                       If False (default), only core Burmese characters are kept.
    """
    if not text:
        return text

    cdef string s = text.encode('utf-8')
    cdef string result
    result.reserve(s.length())

    cdef int i = 0
    cdef int n = s.length()
    cdef pair[int, int] decoded
    cdef int cp, char_len
    cdef const char* data = s.c_str()

    while i < n:
        # Use optimized inline decoder
        decoded = decode_utf8_char(data, i, n)
        cp = decoded.first
        char_len = decoded.second

        # Filter Logic:
        # Keep ONLY if it is a valid Myanmar character (scope-aware)
        # AND NOT Punctuation (104A-104F)
        # Use scoped character check
        if is_myanmar_char_scoped(cp, allow_extended) and not (0x104A <= cp <= 0x104F):
            result.append(s.substr(i, char_len))

        i += char_len

    return result.decode('utf-8')

cpdef list segment_syllables_c(str text, bint allow_extended=False):
    """
    Cython implementation of Sylbreak syllable segmentation.

    Args:
        text: Input text to segment.
        allow_extended: When True, extended Myanmar blocks are considered Myanmar.
            When False (default), only strict Burmese (U+1000-U+104F minus non-standard).

    Returns:
        List of syllable strings.
    """
    if not text:
        return []

    cdef string s = text.encode('utf-8')
    cdef vector[string] parts
    cdef int n = s.length()
    
    # We need a robust way to decode and peek chars.
    # To match regex logic:
    # 1. Myanmar Consonant: (?<!1039)[1000-1021](?![103a 1039])
    # 2. Other Starters: [1023-102a 103f 104c-104f 1040-1049 104b]
    # 3. Non-Myanmar: [^1000-109F]+
    
    cdef int i = 0
    cdef int prev_prev_cp = -1 # Needed for Kinzi check
    cdef int prev_cp = -1
    cdef int curr_cp = -1
    cdef int next_cp = -1
    cdef int curr_len = 0
    cdef int next_len = 0
    cdef pair[int, int] decoded  # For safe UTF-8 decoding
    cdef pair[int, int] next_decoded  # For safe UTF-8 decoding (lookahead)
    
    cdef int start_idx = 0
    
    # Helper to decode at index
    # We need to peek ahead.
    
    while i < n:
        # Decode current char if not already
        # (curr_cp and curr_len should be set from previous iteration's 'next')
        if curr_cp == -1:
            # First char
            # Safe UTF-8 decode (Issue #1236)
            decoded = decode_utf8_char(s.c_str(), i, n)
            curr_cp = decoded.first
            curr_len = decoded.second
        
        # Peek Next
        next_idx = i + curr_len
        if next_idx < n:
            # Safe UTF-8 decode for lookahead (Issue #1236)
            next_decoded = decode_utf8_char(s.c_str(), next_idx, n)
            next_cp = next_decoded.first
            next_len = next_decoded.second
        else:
            next_cp = -1 # EOF
            next_len = 0
            
        # --- Logic Checks ---
        is_break = False
        
        # 1. Non-Myanmar Chunk
        if not is_myanmar_char_scoped(curr_cp, allow_extended):
            # If we are starting a non-Myanmar chunk (and not at start of string)
            # Break here.
            # AND consume until we hit Myanmar char again.
            
            if i > start_idx:
                # Break before this non-Myanmar char
                parts.push_back(s.substr(start_idx, i - start_idx))
                start_idx = i
            
            # Fast forward through Non-Myanmar chars
            # Current char is already non-Myanmar.
            # Move i forward.
            i += curr_len
            prev_prev_cp = prev_cp
            prev_cp = curr_cp
            
            # Loop
            while i < n:
                # We need to decode to check is_myanmar_char
                # Reuse the peek logic essentially
                # Safe UTF-8 decode (Issue #1236)
                decoded = decode_utf8_char(s.c_str(), i, n)
                curr_cp = decoded.first
                curr_len = decoded.second
                
                if is_myanmar_char_scoped(curr_cp, allow_extended):
                    # Found Myanmar char, stop consuming
                    # Break happens at top of next loop iteration naturally?
                    # No, we just finished a non-Myanmar chunk.
                    # We should emit it now?
                    # Regex: `(NonMyanmar+)` is a token.
                    # So yes, emit [start_idx : i]
                    parts.push_back(s.substr(start_idx, i - start_idx))
                    start_idx = i
                    # Don't increment i here, let main loop handle the Myanmar char
                    # Reset next_cp/len logic for main loop state
                    # curr_cp is already set to the Myanmar char
                    next_cp = -1 # Force re-peek
                    break
                
                prev_prev_cp = prev_cp
                prev_cp = curr_cp
                i += curr_len
                
            continue # Continue main loop (handling the Myanmar char we stopped at, or EOF)

        # 2. Myanmar Consonant Rule
        # Regex: (?<!(?<!\u103a)\u1039)[\u1000-\u1021](?![\u103a])
        # Explanation:
        # - Consonant: 1000-1021
        # - NOT preceded by Virama (1039) UNLESS that Virama is preceded by Asat (103a) [Kinzi]
        #   Equivalent to: Break if (prev != 1039) OR (prev == 1039 AND prev_prev == 103a)
        # - NOT followed by Asat (103a)
        elif 0x1000 <= curr_cp <= 0x1021:
            # Check Lookbehind
            lookbehind_ok = True
            if prev_cp == 0x1039:
                # If preceded by Virama, it's a stack...
                # UNLESS it's a Kinzi (Asat + Virama)
                if prev_prev_cp != 0x103A:
                    lookbehind_ok = False # Standard stack, don't break
            
            # Check Lookahead
            lookahead_ok = True
            if next_cp == 0x103A: # Only check for Asat now
                lookahead_ok = False
                
            if lookbehind_ok and lookahead_ok:
                is_break = True
        
        # 3. Other Starters Rule
        # [1023-102a 103f 104c-104f 1040-1049 104a 104b]
        # Added: 104A (၊), 104B (။) to ensure punctuation splits
        elif (0x1023 <= curr_cp <= 0x102A) or \
             (curr_cp == 0x103F) or \
             (0x104C <= curr_cp <= 0x104F) or \
             (0x1040 <= curr_cp <= 0x1049) or \
             (curr_cp == 0x104A) or \
             (curr_cp == 0x104B):
             is_break = True
             
        
        if is_break and i > start_idx:
            parts.push_back(s.substr(start_idx, i - start_idx))
            start_idx = i
            
        # Advance
        prev_prev_cp = prev_cp
        prev_cp = curr_cp
        i += curr_len
        
        # Setup next iteration
        if next_cp != -1:
            curr_cp = next_cp
            curr_len = next_len
        else:
            curr_cp = -1 # Will force re-decode if not EOF, but here we are near EOF
            
    # Append remainder
    if i > start_idx:
        parts.push_back(s.substr(start_idx, i - start_idx))
        
    # Convert vector[string] to list[str]
    # Cython handles this usually, but let's be explicit
    py_parts = []
    for p in parts:
        py_parts.append(p.decode('utf-8'))
        
    return py_parts
