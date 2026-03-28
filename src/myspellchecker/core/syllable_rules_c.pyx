# cython: language_level=3
"""
Cython Implementation of Syllable Rule Validator for Myanmar (Burmese) script.

This module is the optimized Cython version of the SyllableRuleValidator.
It provides identical functionality to _SyllableRuleValidatorPython in
syllable_rules.py but with C-level performance optimizations.

Architecture Notes:
==========================

**Single Source of Truth**:
    All character sets (CONSONANTS, MEDIALS, VOWEL_SIGNS, TONE_MARKS, etc.)
    are imported from myspellchecker.core.constants.myanmar_constants.
    This ensures consistency with the Python implementation.

**Performance Optimizations**:
    - C-level lookup tables (is_consonant, is_medial, etc.) initialized at
      module load for O(1) character classification
    - Typed variables (uint32_t, Py_ssize_t, bint) for faster operations
    - Inline helper functions (check_table) for table lookups

**Consistency with Python Implementation**:
    All validation methods must produce identical results to the Python
    version in syllable_rules.py. When modifying validation logic, both
    files MUST be updated together.

**Build Requirements**:
    This file is compiled to a C extension during package installation.
    After modifying, rebuild with: python setup.py build_ext --inplace
"""
import re
from libc.stdint cimport uint32_t

# Define Py_UCS4 as uint32_t
ctypedef uint32_t Py_UCS4

# Import Python constants - SINGLE SOURCE OF TRUTH for all character sets
from myspellchecker.core.constants import (
    ANUSVARA_ALLOWED_VOWELS,
    ASAT,
    COMPATIBLE_HA,
    COMPATIBLE_RA,
    COMPATIBLE_WA,
    COMPATIBLE_YA,
    CONSONANTS,
    DOT_BELOW,
    GREAT_SA,
    INDEPENDENT_VOWELS,
    INVALID_E_COMBINATIONS,
    KINZI_VALID_FOLLOWERS,
    LOWER_VOWELS,
    MEDIAL_HA,
    MEDIAL_RA,
    MEDIAL_WA,
    MEDIAL_YA,
    MEDIALS,
    NGA,
    NON_STANDARD_CHARS,  # Added for single source of truth

    STACKING_EXCEPTIONS,
    STOP_FINALS,
    TONE_MARKS,
    UPPER_VOWELS,
    VALID_MEDIAL_SEQUENCES,
    VALID_PARTICLES,
    VALID_VOWEL_COMBINATIONS,
    VIRAMA,
    VISARGA,
    VOWEL_SIGNS,
    WET_MAPPING,
    ZERO_WIDTH_CHARS,
    get_myanmar_char_set,
)

# Constants
cdef uint32_t TALL_A = 0x102b
cdef uint32_t AA_VOWEL = 0x102c
cdef uint32_t ANUSVARA_CHAR = 0x1036
cdef uint32_t ASAT_CHAR = 0x103a
cdef uint32_t VIRAMA_CHAR = 0x1039
cdef uint32_t DOT_BELOW_CHAR = 0x1037
cdef uint32_t VISARGA_CHAR = 0x1038
cdef uint32_t NGA_CHAR = 0x1004
cdef uint32_t GREAT_SA_CHAR = 0x103f
cdef uint32_t MEDIAL_YA_CHAR = 0x103b
cdef uint32_t MEDIAL_RA_CHAR = 0x103c
cdef uint32_t MEDIAL_WA_CHAR = 0x103d
cdef uint32_t MEDIAL_HA_CHAR = 0x103e
cdef uint32_t VOWEL_U_CHAR = 0x102f
cdef uint32_t VOWEL_UU_CHAR = 0x1030
cdef uint32_t VOWEL_AI_CHAR = 0x1032
cdef uint32_t VOWEL_E_CHAR = 0x1031
cdef uint32_t NYA_CHAR = 0x100a

# Lookup Tables
cdef bint is_consonant[256]
cdef bint is_medial[256]
cdef bint is_vowel_sign[256]
cdef bint is_tone_mark[256]
cdef bint is_independent_vowel[256]
cdef bint is_upper_vowel[256]
cdef bint is_lower_vowel[256]
cdef bint is_stop_final[256]
cdef bint is_kinzi_valid_follower[256]
cdef bint is_compatible_ya[256]
cdef bint is_compatible_ra[256]
cdef bint is_compatible_wa[256]
cdef bint is_compatible_ha[256]
cdef bint is_anusvara_allowed_vowel[256]
cdef bint is_non_standard[256]

# Valid Combinations (Integer Sets)
VALID_VOWEL_COMBINATIONS_INT = set()

def init_constants():
    global is_consonant, is_medial, is_vowel_sign, is_tone_mark, is_independent_vowel
    global is_upper_vowel, is_lower_vowel, is_stop_final
    global is_kinzi_valid_follower, is_compatible_ya, is_compatible_ra, is_compatible_wa, is_compatible_ha
    global is_anusvara_allowed_vowel
    global is_non_standard
    global VALID_VOWEL_COMBINATIONS_INT

    cdef int i
    
    for i in range(256):
        is_consonant[i] = 0
        is_medial[i] = 0
        is_vowel_sign[i] = 0
        is_tone_mark[i] = 0
        is_independent_vowel[i] = 0
        is_upper_vowel[i] = 0
        is_lower_vowel[i] = 0
        is_stop_final[i] = 0
        is_kinzi_valid_follower[i] = 0
        is_compatible_ya[i] = 0
        is_compatible_ra[i] = 0
        is_compatible_wa[i] = 0
        is_compatible_ha[i] = 0
        is_anusvara_allowed_vowel[i] = 0
        is_non_standard[i] = 0

    for char in CONSONANTS:
        if 0x1000 <= ord(char) < 0x1100: is_consonant[ord(char) - 0x1000] = 1
    for char in MEDIALS:
        if 0x1000 <= ord(char) < 0x1100: is_medial[ord(char) - 0x1000] = 1
    for char in VOWEL_SIGNS:
        if 0x1000 <= ord(char) < 0x1100: is_vowel_sign[ord(char) - 0x1000] = 1
    for char in TONE_MARKS:
        if 0x1000 <= ord(char) < 0x1100: is_tone_mark[ord(char) - 0x1000] = 1
    for char in INDEPENDENT_VOWELS:
        if 0x1000 <= ord(char) < 0x1100: is_independent_vowel[ord(char) - 0x1000] = 1
    for char in UPPER_VOWELS:
        if 0x1000 <= ord(char) < 0x1100: is_upper_vowel[ord(char) - 0x1000] = 1
    for char in LOWER_VOWELS:
        if 0x1000 <= ord(char) < 0x1100: is_lower_vowel[ord(char) - 0x1000] = 1
    for char in STOP_FINALS:
        if 0x1000 <= ord(char) < 0x1100: is_stop_final[ord(char) - 0x1000] = 1
    for char in KINZI_VALID_FOLLOWERS:
        if 0x1000 <= ord(char) < 0x1100: is_kinzi_valid_follower[ord(char) - 0x1000] = 1
    for char in COMPATIBLE_YA:
        if 0x1000 <= ord(char) < 0x1100: is_compatible_ya[ord(char) - 0x1000] = 1
    for char in COMPATIBLE_RA:
        if 0x1000 <= ord(char) < 0x1100: is_compatible_ra[ord(char) - 0x1000] = 1
    for char in COMPATIBLE_WA:
        if 0x1000 <= ord(char) < 0x1100: is_compatible_wa[ord(char) - 0x1000] = 1
    for char in COMPATIBLE_HA:
        if 0x1000 <= ord(char) < 0x1100: is_compatible_ha[ord(char) - 0x1000] = 1
    for char in ANUSVARA_ALLOWED_VOWELS:
        if 0x1000 <= ord(char) < 0x1100: is_anusvara_allowed_vowel[ord(char) - 0x1000] = 1

    # Non-Standard Chars - use imported constant for single source of truth
    for char in NON_STANDARD_CHARS:
        if 0x1000 <= ord(char) < 0x1100: is_non_standard[ord(char) - 0x1000] = 1

    # Convert VALID_VOWEL_COMBINATIONS to int sets
    for combo in VALID_VOWEL_COMBINATIONS:
        int_set = frozenset(ord(c) for c in combo)
        VALID_VOWEL_COMBINATIONS_INT.add(int_set)

init_constants()

# Module-level string constants (avoid repeated f-string construction)
KINZI_SEQ_STR = f"{chr(NGA_CHAR)}{chr(ASAT_CHAR)}{chr(VIRAMA_CHAR)}"
ANUSVARA_ASAT_STR = f"{chr(ANUSVARA_CHAR)}{chr(ASAT_CHAR)}"

# Pre-compiled regex pattern for extracting medial sequences
# Computed at module level to avoid repeated compilation in each instance.
# This pattern matches one or more consecutive medial characters.
_MEDIAL_EXTRACTOR_PATTERN = re.compile(f"[{''.join(MEDIALS)}]+")

cdef inline bint check_table(bint* table, uint32_t c):
    if 0x1000 <= c < 0x1100:
        return table[<int>c - 0x1000]
    return 0

cdef class SyllableRuleValidator:
    # Default values must match DEFAULT_MAX_SYLLABLE_LENGTH and DEFAULT_CORRUPTION_THRESHOLD
    # from syllable_rules.py. Cython doesn't allow Python expressions in typed signatures.
    def __init__(self, int max_syllable_length=15, int corruption_threshold=3, bint strict=True, bint allow_extended_myanmar=False, stacking_pairs=None):
        self.max_syllable_length = max_syllable_length
        self.corruption_threshold = corruption_threshold
        self.strict = strict
        self.allow_extended_myanmar = allow_extended_myanmar
        # Cache valid character set for scope checking
        self._valid_myanmar_chars = get_myanmar_char_set(allow_extended_myanmar)
        # Use pre-compiled module-level pattern
        self._medial_extractor = _MEDIAL_EXTRACTOR_PATTERN
        # Configurable stacking pairs (defaults to hardcoded STACKING_EXCEPTIONS)
        self._stacking_pairs = stacking_pairs if stacking_pairs is not None else STACKING_EXCEPTIONS

    cpdef bint validate(self, str syllable):
        if not syllable: return False
        cdef Py_ssize_t length = len(syllable)
        if length > self.max_syllable_length: return False
        if syllable in VALID_PARTICLES: return True

        if not self._check_zero_width_chars(syllable): return False
        if not self._check_corruption(syllable): return False
        if not self._check_start_char(syllable): return False

        cdef uint32_t first_char = ord(syllable[0])
        if not (check_table(is_consonant, first_char) or check_table(is_independent_vowel, first_char)):
            return False

        # FIX: Do not return early for Independent Vowels
        # Just check validity constraints, then proceed to general checks
        if check_table(is_independent_vowel, first_char) and first_char != 0x1021:
            if not self._check_independent_vowel(syllable): return False

        if not self._check_structure_sanity(syllable): return False
        if not self._check_kinzi_pattern(syllable): return False
        if not self._check_asat_predecessor(syllable): return False
        if not self._check_unexpected_consonant(syllable): return False
        if not self._check_medial_compatibility(syllable): return False
        if not self._check_medial_vowel_compatibility(syllable): return False
        if not self._check_tone_rules(syllable): return False
        if not self._check_virama_usage(syllable): return False
        if not self._check_vowel_combinations(syllable): return False
        if not self._check_vowel_exclusivity(syllable): return False
        if not self._check_e_vowel_combinations(syllable): return False
        if not self._check_e_vowel_position(syllable): return False
        if not self._check_great_sa_rules(syllable): return False
        if not self._check_anusvara_compatibility(syllable): return False
        if not self._check_asat_count(syllable): return False
        if not self._check_double_diacritics(syllable): return False
        if not self._check_tall_a_exclusivity(syllable): return False
        if not self._check_tall_aa_after_medial_wa(syllable): return False
        if not self._check_dot_below_position(syllable): return False
        if not self._check_virama_count(syllable): return False
        if not self._check_anusvara_asat_conflict(syllable): return False
        if not self._check_asat_before_vowel(syllable): return False

        if self.strict:
            if not self._check_tone_strictness(syllable): return False
            if not self._check_tone_position(syllable): return False
            if not self._check_character_scope(syllable): return False
            if not self._check_diacritic_uniqueness(syllable): return False
            if not self._check_one_final_rule(syllable): return False
            if not self._check_strict_kinzi(syllable): return False

        cdef bint has_virama = False
        cdef uint32_t c
        for char in syllable:
            c = ord(char)
            if c == VIRAMA_CHAR:
                has_virama = True
                break
        
        if has_virama:
            if not self._check_virama_ordering(syllable): return False
            if not self._check_pat_sint_validity(syllable): return False

        return True

    cdef bint _check_zero_width_chars(self, str syllable):
        cdef uint32_t char
        for c in syllable:
            char = ord(c)
            if char == 0x200b or char == 0x200c or char == 0x200d or char == 0xfeff:
                return False
        return True

    cdef bint _check_corruption(self, str syllable):
        cdef Py_ssize_t i, j
        cdef Py_ssize_t length = len(syllable)
        cdef uint32_t char, d
        cdef bint is_same
        cdef int count

        if length > self.corruption_threshold:
            for i in range(length - self.corruption_threshold):
                char = ord(syllable[i])
                is_same = True
                for j in range(1, self.corruption_threshold + 1):
                    if ord(syllable[i+j]) != char:
                        is_same = False
                        break
                if is_same: return False

        # O(n) diacritic spam check using counting array
        cdef int diac_counts[256]
        cdef int idx2
        for i in range(256): diac_counts[i] = 0
        for c in syllable:
            char = ord(c)
            if (check_table(is_medial, char) or check_table(is_vowel_sign, char) or check_table(is_tone_mark, char)):
                if 0x1000 <= char < 0x1100:
                    idx2 = <int>char - 0x1000
                    diac_counts[idx2] += 1
                    if diac_counts[idx2] > 2: return False
        return True

    cdef bint _check_start_char(self, str syllable):
        cdef uint32_t first = ord(syllable[0])
        if check_table(is_consonant, first): return True
        if check_table(is_independent_vowel, first): return True
        return False

    cdef bint _check_independent_vowel(self, str syllable):
        cdef uint32_t char
        for c in syllable:
            char = ord(c)
            if check_table(is_medial, char): return False
            if check_table(is_vowel_sign, char): return False
        return True

    cdef bint _check_structure_sanity(self, str syllable):
        cdef Py_ssize_t length = len(syllable)
        cdef Py_ssize_t j
        cdef uint32_t curr_char, next_char

        # Note: YA+RA (ျြ) sequence IS valid in Unicode Burmese per UTN #11
        # Example words: ကျြေး (crane), ပျြောင်း (praise)
        # The earlier rejection was based on a misunderstanding of Zawgyi vs Unicode.
        # See VALID_MEDIAL_SEQUENCES in myanmar_constants.py which includes "ျြ".

        medial_sequences = self._medial_extractor.findall(syllable)
        for seq in medial_sequences:
            if seq not in VALID_MEDIAL_SEQUENCES: return False

        cdef Py_ssize_t first_medial_idx = -1
        cdef Py_ssize_t i
        cdef uint32_t char

        for i in range(length):
            if check_table(is_medial, ord(syllable[i])):
                first_medial_idx = i
                break
        
        # Build set of Kinzi Asat positions (Asat that is part of Nga+Asat+Virama)
        cdef set kinzi_asat_positions = set()
        cdef Py_ssize_t kinzi_idx = syllable.find(KINZI_SEQ_STR)
        while kinzi_idx != -1:
            kinzi_asat_positions.add(kinzi_idx + 1)
            kinzi_idx = syllable.find(KINZI_SEQ_STR, kinzi_idx + 1)

        if first_medial_idx != -1:
            for i in range(first_medial_idx):
                char = ord(syllable[i])
                if char == ASAT_CHAR and i in kinzi_asat_positions:
                    continue  # Skip Kinzi Asat
                if (check_table(is_vowel_sign, char) or check_table(is_tone_mark, char) or
                    char == ASAT_CHAR or char == VISARGA_CHAR):
                    return False

        if ord(syllable[length - 1]) != VISARGA_CHAR:
            for c in syllable:
                if ord(c) == VISARGA_CHAR: return False

        cdef Py_ssize_t e_idx = -1
        for i in range(length):
            if ord(syllable[i]) == VOWEL_E_CHAR:
                e_idx = i
                break
        
        if e_idx != -1:
            for i in range(length):
                char = ord(syllable[i])
                if char == TALL_A or char == AA_VOWEL:
                    if i < e_idx: return False

        cdef Py_ssize_t anu_idx = -1
        for i in range(length):
            if ord(syllable[i]) == ANUSVARA_CHAR:
                anu_idx = i
                break
        
        if anu_idx != -1:
            for i in range(length):
                char = ord(syllable[i])
                if (char == DOT_BELOW_CHAR or char == VISARGA_CHAR) and i < anu_idx:
                    return False
        return True

    cdef bint _check_kinzi_pattern(self, str syllable):
        cdef Py_ssize_t idx = syllable.find(KINZI_SEQ_STR)
        cdef uint32_t next_char
        cdef uint32_t prev_char
        if idx != -1:
            # Strict: Kinzi must be preceded by nothing OR consonant/independent vowel
            # Not by dependent vowel, medial, or tone marks
            if self.strict and idx > 0:
                prev_char = ord(syllable[idx - 1])
                # Valid predecessors: consonants and independent vowels
                if not (check_table(is_consonant, prev_char) or check_table(is_independent_vowel, prev_char)):
                    return False
            if idx + 3 >= len(syllable): return False
            next_char = ord(syllable[idx + 3])
            if self.strict:
                if not check_table(is_kinzi_valid_follower, next_char): return False
            else:
                if not (check_table(is_kinzi_valid_follower, next_char) or check_table(is_consonant, next_char)):
                    return False
        return True

    cdef bint _check_asat_predecessor(self, str syllable):
        cdef Py_ssize_t i
        cdef uint32_t char, prev_char
        cdef Py_ssize_t length = len(syllable)
        cdef bint has_e
        
        for i in range(length):
            if ord(syllable[i]) == ASAT_CHAR:
                if i == 0: return False
                prev_char = ord(syllable[i-1])
                
                if (check_table(is_consonant, prev_char) or check_table(is_independent_vowel, prev_char)):
                    if self.strict and prev_char == 0x1021: return False
                    continue
                
                if prev_char == AA_VOWEL or prev_char == TALL_A:
                    has_e = False
                    for c in syllable:
                        if ord(c) == VOWEL_E_CHAR:
                            has_e = True
                            break
                    if has_e: continue
                        
                return False
        return True

    cdef bint _check_unexpected_consonant(self, str syllable):
        """
        Check for multiple base consonants that are not properly stacked.

        Special case: Great Sa (U+103F) is a self-contained conjunct (doubled Sa)
        that can appear after vowels without virama stacking.
        """
        cdef int num_active_bases = 0
        cdef Py_ssize_t length = len(syllable)
        cdef Py_ssize_t i
        cdef uint32_t char
        cdef bint is_final, is_stacked, is_stacker

        for i in range(length):
            char = ord(syllable[i])
            if check_table(is_consonant, char) or check_table(is_independent_vowel, char):
                # Special case: Great Sa (U+103F) is a self-contained conjunct
                # It can follow vowels and is treated as a syllable extension
                # rather than a new base consonant
                if char == GREAT_SA_CHAR:
                    continue

                is_final = False
                if i + 1 < length and ord(syllable[i+1]) == ASAT_CHAR:
                    is_final = True
                elif char == NYA_CHAR and i + 1 < length and ord(syllable[i+1]) == DOT_BELOW_CHAR:
                    is_final = True

                is_stacked = False
                if i > 0 and ord(syllable[i-1]) == VIRAMA_CHAR:
                    is_stacked = True

                is_stacker = False
                if i + 1 < length and ord(syllable[i+1]) == VIRAMA_CHAR:
                    is_stacker = True

                if not is_final and not is_stacked and not is_stacker:
                    num_active_bases += 1
        return num_active_bases <= 1

    cdef uint32_t _get_medial_base_consonant(self, str syllable):
        """
        Find the true base consonant that medials attach to.
        Returns 0 if no valid base consonant found.
        """
        cdef Py_ssize_t length = len(syllable)
        cdef Py_ssize_t pos = 0
        cdef uint32_t base = 0
        cdef uint32_t ch

        if length == 0:
            return 0

        # Check for Kinzi-initial syllable (NGA + ASAT + VIRAMA)
        if length >= 3:
            if (ord(syllable[0]) == NGA_CHAR and
                ord(syllable[1]) == ASAT_CHAR and
                ord(syllable[2]) == VIRAMA_CHAR):
                # Base is the first consonant after the 3-char Kinzi sequence
                pos = 3
                while pos < length:
                    ch = ord(syllable[pos])
                    if check_table(is_consonant, ch):
                        return ch
                    pos += 1
                return 0

        # Rule 1b: Non-initial Kinzi (merged segment like အင်္ကျီ)
        cdef Py_ssize_t kinzi_pos = syllable.find(KINZI_SEQ_STR)
        if kinzi_pos != -1:
            pos = kinzi_pos + 3  # skip past Kinzi sequence
            while pos < length:
                ch = ord(syllable[pos])
                if check_table(is_consonant, ch):
                    return ch
                pos += 1
            return 0

        # Initial stacked consonants (C + VIRAMA + C ...)
        # Find the consonant after the last virama in the initial stacking chain
        pos = 0
        base = 0

        while pos < length:
            ch = ord(syllable[pos])
            if check_table(is_consonant, ch):
                base = ch
                pos += 1
                # Check if followed by VIRAMA (stacking continues)
                if pos < length and ord(syllable[pos]) == VIRAMA_CHAR:
                    pos += 1  # Skip virama, look for next consonant
                    continue
                else:
                    # No more stacking, we found the base
                    return base
            else:
                # Non-consonant character - stop looking
                break

        # If we tracked a base during stacking, return it
        if base != 0:
            return base

        # Fallback: return the first character if it's a consonant
        if length > 0:
            ch = ord(syllable[0])
            if check_table(is_consonant, ch):
                return ch

        return 0

    cdef bint _check_medial_compatibility(self, str syllable):
        cdef bint has_ya = False, has_ra = False, has_wa = False, has_ha = False
        cdef uint32_t char
        cdef uint32_t base_char

        for c in syllable:
            char = ord(c)
            if char == MEDIAL_YA_CHAR: has_ya = True
            elif char == MEDIAL_RA_CHAR: has_ra = True
            elif char == MEDIAL_WA_CHAR: has_wa = True
            elif char == MEDIAL_HA_CHAR: has_ha = True

        if not (has_ya or has_ra or has_wa or has_ha): return True

        # Get the true base consonant (handles Kinzi and stacked syllables)
        base_char = self._get_medial_base_consonant(syllable)

        # If no valid base consonant found and strict mode, fail the check
        if base_char == 0:
            return not self.strict

        if self.strict:
            if has_ya and not check_table(is_compatible_ya, base_char): return False
            if has_ra and not check_table(is_compatible_ra, base_char): return False
            if has_wa and not check_table(is_compatible_wa, base_char): return False
            if has_ha and not check_table(is_compatible_ha, base_char): return False
        return True

    cdef bint _check_medial_vowel_compatibility(self, str syllable):
        cdef list medials = []
        cdef list vowels = []
        cdef uint32_t char
        for c in syllable:
            char = ord(c)
            if check_table(is_medial, char): medials.append(char)
            if check_table(is_vowel_sign, char): vowels.append(char)
                
        if not medials: return True
            
        cdef bint has_ya = MEDIAL_YA_CHAR in medials
        cdef bint has_ra = MEDIAL_RA_CHAR in medials
        cdef bint has_wa = MEDIAL_WA_CHAR in medials
        
        # Note: Medial Ya + AI vowel (ျ + ဲ) was previously considered invalid,
        # but this combination does occur in Pali/Sanskrit loanwords and proper
        # nouns, so it is no longer rejected.
        for m in medials:
            if m == MEDIAL_WA_CHAR:
                # Wa + u/uu invalid (phonetically incompatible)
                if VOWEL_U_CHAR in vowels or VOWEL_UU_CHAR in vowels: return False
                
        if (has_ya and has_wa) or (has_ra and has_wa):
            if VOWEL_U_CHAR in vowels or VOWEL_UU_CHAR in vowels: return False
                
        if has_ya and len(vowels) > 2: return False
        return True

    cdef bint _check_tone_rules(self, str syllable):
        cdef bint has_dot = False
        cdef bint has_visarga = False
        cdef int dot_count = 0
        cdef int visarga_count = 0
        cdef Py_ssize_t i
        cdef uint32_t char, prev
        cdef Py_ssize_t length = len(syllable)

        for c in syllable:
            char = ord(c)
            if char == DOT_BELOW_CHAR:
                has_dot = True
                dot_count += 1
            if char == VISARGA_CHAR:
                has_visarga = True
                visarga_count += 1

        # Dot Below and Visarga are mutually exclusive
        if has_dot and has_visarga: return False

        # Each tone mark can only appear once per syllable
        # This check applies in both strict and non-strict modes
        if dot_count > 1: return False
        if visarga_count > 1: return False

        if not (has_dot or has_visarga): return True

        for i in range(1, length):
            if ord(syllable[i]) == ASAT_CHAR:
                prev = ord(syllable[i-1])
                if check_table(is_stop_final, prev): return False
        return True

    cdef bint _check_virama_usage(self, str syllable):
        cdef Py_ssize_t length = len(syllable)
        cdef Py_ssize_t i
        cdef uint32_t c, next_c
        if ord(syllable[length - 1]) == VIRAMA_CHAR: return False
        # Virama must be followed by a consonant (stacking)
        for i in range(length - 1):
            c = ord(syllable[i])
            if c == VIRAMA_CHAR:
                next_c = ord(syllable[i + 1])
                if not check_table(is_consonant, next_c):
                    return False
        return True

    cdef bint _check_vowel_combinations(self, str syllable):
        """
        Check for valid multiple vowel combinations (digraphs).

        Special case: Great Sa (U+103F) can have its own vowels, so vowels
        before and after Great Sa don't count as a single combination.
        """
        cdef set vowels_before = set()
        cdef set vowels_after = set()
        cdef set all_vowels = set()
        cdef uint32_t char
        cdef bint found_great_sa = False
        cdef Py_ssize_t gs_pos = -1
        cdef Py_ssize_t i

        # Check if Great Sa is present
        for i in range(len(syllable)):
            if ord(syllable[i]) == GREAT_SA_CHAR:
                found_great_sa = True
                gs_pos = i
                break

        if found_great_sa:
            # Split vowel collection at Great Sa
            for i in range(len(syllable)):
                char = ord(syllable[i])
                if check_table(is_vowel_sign, char):
                    if i < gs_pos:
                        vowels_before.add(char)
                    else:
                        vowels_after.add(char)

            # Check vowels before Great Sa
            if len(vowels_before) > 1:
                v_set = frozenset(vowels_before)
                if v_set not in VALID_VOWEL_COMBINATIONS_INT:
                    return False

            # Check vowels after Great Sa
            if len(vowels_after) > 1:
                v_set = frozenset(vowels_after)
                if v_set not in VALID_VOWEL_COMBINATIONS_INT:
                    return False

            return True

        # Standard check for syllables without Great Sa
        for c in syllable:
            char = ord(c)
            if check_table(is_vowel_sign, char):
                all_vowels.add(char)

        if len(all_vowels) > 1:
            v_set = frozenset(all_vowels)
            if v_set not in VALID_VOWEL_COMBINATIONS_INT:
                return False
        return True

    cdef bint _check_vowel_exclusivity(self, str syllable):
        cdef int upper_count = 0
        cdef int lower_count = 0
        cdef uint32_t char
        for c in syllable:
            char = ord(c)
            if check_table(is_upper_vowel, char): upper_count += 1
            if check_table(is_lower_vowel, char): lower_count += 1
        if upper_count > 1 or lower_count > 1: return False
        return True

    cdef bint _check_e_vowel_position(self, str syllable):
        """
        Check E vowel (U+1031) position validity.

        E vowel appears before the consonant visually but after in encoding.
        In Unicode encoding order, E vowel must:
        1. Not be at position 0 (floating diacritic)
        2. Come immediately after a consonant, medial, or independent vowel
        3. Not appear multiple times in a syllable

        Returns:
            True if E vowel position is valid, False otherwise.
        """
        cdef Py_ssize_t length = len(syllable)
        cdef Py_ssize_t e_idx = -1
        cdef int e_count = 0
        cdef uint32_t c, prev_c
        cdef Py_ssize_t i

        # Count E vowels and find position
        for i in range(length):
            c = ord(syllable[i])
            if c == VOWEL_E_CHAR:
                e_count += 1
                if e_idx == -1:
                    e_idx = i

        if e_count == 0:
            return True

        # E vowel should appear only once
        if e_count > 1:
            return False

        # E vowel at position 0 is invalid (floating diacritic)
        if e_idx == 0:
            return False

        # Check what precedes E vowel
        prev_c = ord(syllable[e_idx - 1])

        # Valid predecessors: consonants, medials, or independent vowels
        # Note: VIRAMA is not valid — E cannot directly follow Virama;
        # the consonant after Virama is the actual predecessor.
        if not (check_table(is_consonant, prev_c) or
                check_table(is_medial, prev_c) or
                check_table(is_independent_vowel, prev_c)):
            return False

        return True

    cdef bint _check_e_vowel_combinations(self, str syllable):
        """
        Check E vowel (U+1031) combinations.

        E vowel cannot combine with:
        - I (U+102D), II (U+102E), U (U+102F), UU (U+1030) - in INVALID_E_COMBINATIONS
        - Anusvara (U+1036) - phonotactically impossible in standard Myanmar

        Returns:
            True if E vowel combinations are valid, False otherwise.
        """
        cdef bint has_e = False
        for c in syllable:
            if ord(c) == VOWEL_E_CHAR:
                has_e = True
                break
        if has_e:
            for invalid in INVALID_E_COMBINATIONS:
                if invalid in syllable: return False
            # E + Anusvara is also invalid
            # Anusvara is classified as a tone mark, not a vowel sign,
            # so it's not caught by INVALID_E_COMBINATIONS
            for c in syllable:
                if ord(c) == ANUSVARA_CHAR:
                    return False
        return True

    cdef bint _check_great_sa_rules(self, str syllable):
        """
        Validate Great Sa (U+103F) usage patterns.

        Great Sa is a special conjunct used in Pali/Sanskrit loanwords with
        specific restrictions on medials, stacking, and position.
        """
        cdef bint has_great_sa = False
        cdef bint has_virama = False
        cdef int great_sa_count = 0
        cdef int great_sa_pos = -1
        cdef uint32_t char
        cdef Py_ssize_t i = 0
        cdef Py_ssize_t length = len(syllable)

        # First pass: check for Great Sa, medials, virama, and position
        for i in range(length):
            char = ord(syllable[i])
            if char == GREAT_SA_CHAR:
                has_great_sa = True
                great_sa_count += 1
                if great_sa_pos == -1:
                    great_sa_pos = i
            if char == VIRAMA_CHAR: has_virama = True

        if not has_great_sa:
            return True

        # Rule 1: Great Sa cannot take medials (only check after Great Sa position)
        cdef bint has_medial_after_gs = False
        for i in range(great_sa_pos, length):
            if check_table(is_medial, ord(syllable[i])):
                has_medial_after_gs = True
                break
        if has_medial_after_gs or has_virama:
            return False

        # Rule 3: Cannot have multiple Great Sa in one syllable
        if great_sa_count > 1:
            return False

        # Rule 3b: Great Sa + Asat is invalid
        if great_sa_pos + 1 < length and ord(syllable[great_sa_pos + 1]) == ASAT_CHAR:
            return False

        # Rule 4: Check what follows Great Sa
        if great_sa_pos + 1 < length:
            char = ord(syllable[great_sa_pos + 1])
            # Valid: vowel signs, tone marks, visarga, or consonants
            if not (check_table(is_vowel_sign, char) or
                    check_table(is_tone_mark, char) or
                    char == VISARGA_CHAR or
                    check_table(is_consonant, char)):
                return False

        # Rule 5: Great Sa should not be preceded by medials or non-asat tones
        if great_sa_pos > 0:
            char = ord(syllable[great_sa_pos - 1])
            if check_table(is_medial, char):
                return False
            if check_table(is_tone_mark, char):
                return False

        return True

    cdef bint _check_anusvara_compatibility(self, str syllable):
        cdef bint has_anusvara = False
        cdef uint32_t char
        for c in syllable:
            if ord(c) == ANUSVARA_CHAR:
                has_anusvara = True
                break
        if has_anusvara:
            for c in syllable:
                char = ord(c)
                if check_table(is_vowel_sign, char):
                    if not check_table(is_anusvara_allowed_vowel, char): return False
        return True

    cdef bint _check_asat_count(self, str syllable):
        cdef int count = 0
        for c in syllable:
            if ord(c) == ASAT_CHAR: count += 1
        
        if count <= 1: return True
        if count > 2: return False
        
        if KINZI_SEQ_STR not in syllable: return False
        return True

    cdef bint _check_double_diacritics(self, str syllable):
        cdef Py_ssize_t i
        cdef Py_ssize_t length = len(syllable)
        cdef uint32_t curr, next_c
        for i in range(length - 1):
            curr = ord(syllable[i])
            next_c = ord(syllable[i+1])
            if curr == next_c:
                if (check_table(is_medial, curr) or check_table(is_vowel_sign, curr) or check_table(is_tone_mark, curr)):
                    return False
        return True

    cdef bint _check_tall_a_exclusivity(self, str syllable):
        cdef bint has_tall = False
        cdef bint has_aa = False
        cdef uint32_t char
        for c in syllable:
            char = ord(c)
            if char == TALL_A: has_tall = True
            elif char == AA_VOWEL: has_aa = True
        if has_tall and has_aa: return False
        return True

    cdef bint _check_tall_aa_after_medial_wa(self, str syllable):
        """Reject Tall A (U+102B) after Medial Wa (U+103D)."""
        cdef bint has_wa = False
        cdef bint has_tall = False
        cdef uint32_t char
        for c in syllable:
            char = ord(c)
            if char == MEDIAL_WA_CHAR: has_wa = True
            elif char == TALL_A: has_tall = True
        if has_wa and has_tall: return False
        return True

    cdef bint _check_dot_below_position(self, str syllable):
        cdef Py_ssize_t idx = -1
        cdef Py_ssize_t i
        cdef Py_ssize_t length = len(syllable)
        cdef uint32_t char
        
        for i in range(length):
            if ord(syllable[i]) == DOT_BELOW_CHAR:
                idx = i
                break
        if idx != -1:
            for i in range(idx + 1, length):
                char = ord(syllable[i])
                if (check_table(is_vowel_sign, char) or check_table(is_medial, char) or char == ANUSVARA_CHAR):
                    return False
        return True

    cdef bint _check_virama_count(self, str syllable):
        cdef int count = 0
        for c in syllable:
            if ord(c) == VIRAMA_CHAR: count += 1
        if count <= 1: return True
        if count == 2 and KINZI_SEQ_STR in syllable: return True
        return False

    cdef bint _check_anusvara_asat_conflict(self, str syllable):
        if ANUSVARA_ASAT_STR in syllable: return False
        return True

    cdef bint _check_asat_before_vowel(self, str syllable):
        cdef Py_ssize_t length = len(syllable)
        cdef Py_ssize_t i
        for i in range(length - 1):
            if ord(syllable[i]) == ASAT_CHAR:
                if check_table(is_vowel_sign, ord(syllable[i+1])): return False
        return True

    cdef bint _check_virama_ordering(self, str syllable):
        if not self.strict: return True
        cdef Py_ssize_t v_idx = syllable.rfind(chr(VIRAMA_CHAR))
        if v_idx == -1: return True
        cdef Py_ssize_t m_idx = -1
        cdef Py_ssize_t i
        for i in range(len(syllable)):
            if check_table(is_medial, ord(syllable[i])):
                m_idx = i
                break
        if m_idx != -1 and v_idx > m_idx: return False
        return True

    cdef bint _check_pat_sint_validity(self, str syllable):
        if not self.strict: return True
        cdef Py_ssize_t v_idx = syllable.find(chr(VIRAMA_CHAR))
        if v_idx <= 0 or v_idx >= len(syllable) - 1: return False
        cdef uint32_t upper = ord(syllable[v_idx - 1])
        cdef uint32_t lower = ord(syllable[v_idx + 1])
        if upper == ASAT_CHAR:
            if v_idx >= 2 and ord(syllable[v_idx - 2]) == NGA_CHAR: return True
            return False
        if not check_table(is_consonant, upper): return False
        if not check_table(is_consonant, lower): return False
        # Convert to characters for stacking pairs lookup (set contains string tuples)
        cdef str u_str = chr(upper)
        cdef str l_str = chr(lower)
        if (u_str, l_str) in self._stacking_pairs: return True
        if upper == 0x101e and lower == 0x101e: return False
        if u_str not in WET_MAPPING or l_str not in WET_MAPPING: return False
        row1, col1 = WET_MAPPING[u_str]
        row2, col2 = WET_MAPPING[l_str]
        if row1 != row2: return False
        if col1 == 1:
            if col2 not in (1, 2): return False
        elif col1 == 3:
            if col2 not in (3, 4): return False
        return True

    cdef bint _check_tone_strictness(self, str syllable):
        cdef int count = 0
        cdef uint32_t char
        for c in syllable:
            char = ord(c)
            if char == DOT_BELOW_CHAR or char == VISARGA_CHAR: count += 1
        return count <= 1

    cdef bint _check_tone_position(self, str syllable):
        cdef Py_ssize_t length = len(syllable)
        cdef Py_ssize_t i
        cdef Py_ssize_t dot_idx = -1
        cdef Py_ssize_t vis_idx = -1
        cdef uint32_t char
        
        for i in range(length):
            char = ord(syllable[i])
            if char == DOT_BELOW_CHAR: dot_idx = i
            elif char == VISARGA_CHAR: vis_idx = i
            
        if dot_idx != -1 and dot_idx != length - 1: return False
        if vis_idx != -1 and vis_idx != length - 1: return False
        return True

    cdef bint _check_character_scope(self, str syllable):
        """
        Enforce Myanmar character scope based on allow_extended_myanmar setting.

        When allow_extended_myanmar is False (default):
            - Accepts only standard Burmese (U+1000-U+104F)
            - Rejects Extended Core (U+1050-U+109F), Extended-A/B, and non-standard chars

        When allow_extended_myanmar is True:
            - Accepts Extended Core (U+1050-U+109F), Extended-A (U+AA60-AA7F),
              Extended-B (U+A9E0-A9FF), and non-standard core chars
        """
        # Use cached valid character set for scope checking
        for char in syllable:
            if char not in self._valid_myanmar_chars:
                return False
        return True

    cdef bint _check_diacritic_uniqueness(self, str syllable):
        cdef int counts[256]
        cdef int i
        for i in range(256): counts[i] = 0
        cdef uint32_t c
        cdef int idx
        for char in syllable:
            c = ord(char)
            if check_table(is_medial, c) or check_table(is_vowel_sign, c):
                if 0x1000 <= c < 0x1100:
                    idx = c - 0x1000
                    counts[idx] += 1
                    if counts[idx] > 1: return False
        return True

    cdef bint _check_one_final_rule(self, str syllable):
        cdef bint has_anusvara = False
        cdef Py_ssize_t length = len(syllable)
        cdef Py_ssize_t i
        
        # Check Anusvara presence
        for i in range(length):
            if ord(syllable[i]) == ANUSVARA_CHAR:
                has_anusvara = True
                break
        
        if not has_anusvara: return True
        
        # If Anusvara present, check for non-Kinzi Asat
        for i in range(length):
            if ord(syllable[i]) == ASAT_CHAR:
                # Check next char for Virama (Kinzi)
                if i + 1 < length and ord(syllable[i+1]) == VIRAMA_CHAR:
                    continue # Kinzi, allowed
                return False # Regular Asat + Anusvara = Invalid
        return True

    cdef bint _check_strict_kinzi(self, str syllable):
        """
        Enforce Strict Kinzi: If Nga is stacked (Nga + Virama), it MUST be Kinzi (Asat + Nga + Virama)?
        Actually Kinzi is Nga + Asat + Virama + Base.
        The sequence is `Nga` (1004) `Asat` (103A) `Virama` (1039).
        If we see `Nga` (1004) followed by `Virama` (1039), it is missing Asat.
        """
        cdef Py_ssize_t length = len(syllable)
        cdef Py_ssize_t i
        
        for i in range(length - 1):
            if ord(syllable[i]) == NGA_CHAR and ord(syllable[i+1]) == VIRAMA_CHAR:
                # Found Nga + Virama
                # This is "Stacked Nga" without Asat.
                # In standard Burmese, this is usually a typo for Kinzi (missing Asat).
                # Unless it's Pali "Ganga" (Ga + Nga + Virama + Ga).
                # But Ganga is usually written with Kinzi in modern Burmese?
                # "ဂင်္ဂါ" -> Ga + (Nga+Virama) + Ga + Aa.
                # If we ban this, we ban Pali.
                # The issue was "Malformed Kinzi" like "င္ကေ".
                # If it's single syllable, it might be weird.
                # But strict Standard Burmese usually enforces Kinzi for Nga-stacking.
                return False
        return True
