"""
Syllable Rule Validator for Myanmar (Burmese) script.

This module implements algorithmic checks for Myanmar syllable structure.
It serves as a "validity filter" to detect impossible character combinations
without needing a dictionary lookup.

Architecture Overview
============================

The syllable validation system is designed with a clear separation of concerns:

**Single Source of Truth for Character Sets**:
    All character sets (CONSONANTS, MEDIALS, VOWELS, TONE_MARKS, etc.) are
    defined in `myspellchecker.core.constants.myanmar_constants` and exported
    via `myspellchecker.core.constants`. Both Python and Cython implementations
    import from the same source, ensuring consistency.

**Dual Implementation Pattern**:
    - `syllable_rules.py` - Pure Python implementation (_SyllableRuleValidatorPython)
    - `syllable_rules_c.pyx` - Cython implementation (SyllableRuleValidator)

    The Python implementation serves as:
    1. Fallback when Cython extensions are not compiled
    2. Reference implementation for testing and documentation
    3. Portable version for environments without C compiler

    At module load time, we try to import the Cython version; if unavailable,
    we fall back to Python (see bottom of this file).

**Validation Pipeline Integration**:
    1. `SyllableRuleValidator` - Algorithmic/structural validation (this module)
       - Checks phonotactic rules without dictionary access
       - Returns True/False for structural validity
       - Fast O(n) checks based on Myanmar orthographic rules

    2. `SyllableValidator` (in validators.py) - Full validation pipeline
       - Uses SyllableRuleValidator for structural checks
       - Adds dictionary lookup via SyllableRepository
       - Adds frequency threshold filtering
       - Generates suggestions via SymSpell
       - Returns Error objects with position and suggestions

**Validation Order** (22 checks in validate() method):
    1. Zero-width character rejection (encoding check)
    2. Corruption check (length, repetition)
    3. Start character check (must be consonant or independent vowel)
    4. Base character validation
    5. Independent vowel rules
    6. Structure sanity (medial sequences, ordering)
    7. Kinzi pattern validation
    8. Asat predecessor check
    9. Unexpected consonant detection
    10. Medial compatibility (consonant-medial phonotactics)
    11. Medial-vowel compatibility
    12. Tone rules (stop finals, conflicts)
    13. Virama usage check
    14. Vowel combinations (digraphs)
    15. Vowel exclusivity (upper vs lower slots)
    16. E vowel combinations
    17. Great Sa rules
    18. Anusvara compatibility
    19. Asat count
    20. Double diacritics
    21. Tall A / Aa exclusivity
    22. Dot below position

    Strict mode adds:
    - Virama count, Anusvara+Asat conflict, Asat before vowel
    - Tone strictness, position, character scope
    - Diacritic uniqueness, one final rule, strict kinzi

Rules are based on Myanmar phonotactics:
C(M)V(F)(T) - Consonant, Medial, Vowel, Final, Tone

Key validations:
1. Must start with a Consonant or Independent Vowel.
2. Diacritic ordering must be canonical (Medial < Vowel < Tone).
3. Medial compatibility (e.g., Ka+Ya valid, but some Cs reject Ya).
4. Invalid combinations (e.g., multiple vowels, incompatible tones).
"""

from __future__ import annotations

import re

from myspellchecker.core.constants import (
    ANUSVARA,
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
    STACKING_EXCEPTIONS,
    STOP_FINALS,
    TONE_MARKS,
    UPPER_VOWELS,
    VALID_MEDIAL_SEQUENCES,
    VALID_PARTICLES,
    VALID_VOWEL_COMBINATIONS,
    VIRAMA,
    VISARGA,
    VOWEL_E,
    VOWEL_SIGNS,
    VOWEL_U,
    VOWEL_UU,
    WET_MAPPING,
    ZERO_WIDTH_CHARS,
    get_myanmar_char_set,
)

__all__ = [
    "SyllableRuleValidator",
]

# Vowel constants for exclusivity checks
TALL_A = "\u102b"
AA_VOWEL = "\u102c"

# Syllable validation defaults
# Maximum syllable length accommodates longest known Myanmar syllables including
# merged Kinzi segments. Kinzi prefix adds 4 codepoints (C + Nga + Asat + Virama)
# before the base syllable: C(1) + Kinzi(3) + C(1) + Medials(4) + Vowel(2) +
# Tone(1) + Asat(1) + Tone(1) = ~14 max.
DEFAULT_MAX_SYLLABLE_LENGTH = 15

# Maximum consecutive identical characters allowed (minus one).
# 4+ consecutive identical characters are flagged as data corruption.
DEFAULT_CORRUPTION_THRESHOLD = 3

# Pre-compiled regex pattern for extracting medial sequences
# Computed at module level to avoid repeated compilation in each instance.
# This pattern matches one or more consecutive medial characters.
_MEDIAL_EXTRACTOR_PATTERN = re.compile(f"[{''.join(MEDIALS)}]+")


class _SyllableRuleValidatorPython:
    """
    Validates Myanmar syllable structure using phonotactic rules.

    Attributes:
        max_syllable_length: Maximum allowed character length for a valid syllable.
            Default is 15, which accommodates merged Kinzi segments where the Kinzi
            prefix adds 4 codepoints (C + Nga + Asat + Virama) before the base
            syllable (e.g., သင်္ချိုင်း = 11 codepoints).

        corruption_threshold: Number of consecutive identical characters allowed.
            Default is 3, meaning 4+ consecutive identical characters are flagged
            as data corruption. This catches encoding errors and repeated character
            glitches while allowing legitimate cases like some vowel combinations.
    """

    def __init__(
        self,
        max_syllable_length: int = DEFAULT_MAX_SYLLABLE_LENGTH,
        corruption_threshold: int = DEFAULT_CORRUPTION_THRESHOLD,
        strict: bool = True,
        allow_extended_myanmar: bool = False,
        stacking_pairs: set[tuple[str, str]] | None = None,
    ) -> None:
        """
        Initialize the syllable rule validator.

        Args:
            max_syllable_length: Maximum valid syllable length in characters.
                Typical Myanmar syllables are 2-6 characters. Merged Kinzi segments
                (e.g., သင်္ချိုင်း) can reach 11-14 characters. Default of 15
                accommodates known edge cases while rejecting likely corrupted data.
            corruption_threshold: Maximum allowed consecutive identical characters
                minus one. Default of 3 means sequences like "ကကကက" (4 identical)
                are rejected as likely corruption, while "ကကက" (3) is allowed.
            strict: If True, enforce strict Pali/Sanskrit stacking rules and
                canonical ordering. If False, allow for non-standard stacking often
                seen in loan words or modern transliterations. Default is True.
            allow_extended_myanmar: If True, accept characters from Extended Core
                (U+1050-109F), Extended-A (U+AA60-AA7F), Extended-B (U+A9E0-A9FF),
                and non-standard core chars (U+1022, U+1028, U+1033-U+1035).
                If False (default), only strict Burmese (U+1000-U+104F minus
                non-standard) is accepted.
            stacking_pairs: Optional set of valid (upper, lower) consonant pairs
                for virama stacking. If None, uses the hardcoded
                ``STACKING_EXCEPTIONS`` from myanmar_constants. Can be loaded
                from YAML via ``detection_rules.load_stacking_pairs()``.
        """
        # Use pre-compiled module-level pattern
        self._medial_extractor = _MEDIAL_EXTRACTOR_PATTERN
        self.max_syllable_length = max_syllable_length
        self.corruption_threshold = corruption_threshold
        self.strict = strict
        self.allow_extended_myanmar = allow_extended_myanmar
        self._stacking_pairs = stacking_pairs if stacking_pairs is not None else STACKING_EXCEPTIONS

    def validate(self, syllable: str) -> bool:
        """
        Check if a syllable is structurally valid.

        Delegates to four validation phases covering basic structure,
        medial compatibility, vowel/tone rules, and advanced checks
        (kinzi, stacking, strict-mode rules).

        Args:
            syllable: The syllable string to check.

        Returns:
            True if structurally valid, False otherwise.
        """
        if not syllable:
            return False

        # Early length check (consistent with Cython validate())
        if len(syllable) > self.max_syllable_length:
            return False

        if syllable in VALID_PARTICLES:
            return True

        if not self._validate_basic_structure(syllable):
            return False

        if not self._validate_medials(syllable):
            return False

        if not self._validate_vowels_tones(syllable):
            return False

        if not self._validate_advanced(syllable):
            return False

        return True

    def _validate_basic_structure(self, syllable: str) -> bool:
        """Validate encoding, corruption, start character, base character, and independent vowels.

        Checks:
            - Zero-width character rejection (encoding check)
            - Corruption (length, repetition)
            - Start character (must be consonant or independent vowel)
            - Base character validation
            - Independent vowel rules
            - Structure sanity (medial sequences, ordering)
            - Kinzi pattern validation
            - Asat predecessor check
            - Unexpected consonant detection
        """
        # Zero-width character rejection (encoding check)
        if not self._check_zero_width_chars(syllable):
            return False

        # Corruption & Basic Structure
        if not self._check_corruption(syllable):
            return False

        # Start Character (Floating Diacritics)
        if not self._check_start_char(syllable):
            return False

        # Base Character Check
        first_char = syllable[0]
        if first_char not in CONSONANTS and first_char not in INDEPENDENT_VOWELS:
            return False

        # Independent Vowel Rules
        # Exception: U+1021 (အ) behaves like a consonant carrier and should
        # go through normal validation, not the restricted independent vowel path
        if first_char in INDEPENDENT_VOWELS and first_char != "\u1021":
            if not self._check_independent_vowel(syllable):
                return False

        # Structure Sanity (Medials position, etc.)
        if not self._check_structure_sanity(syllable):
            return False

        # Kinzi Pattern Check
        if not self._check_kinzi_pattern(syllable):
            return False

        # Asat Predecessor Check
        if not self._check_asat_predecessor(syllable):
            return False

        # Unexpected Consonants
        if not self._check_unexpected_consonant(syllable):
            return False

        return True

    def _validate_medials(self, syllable: str) -> bool:
        """Validate medial compatibility and medial-vowel compatibility.

        Checks:
            - Medial compatibility (consonant-medial phonotactics)
            - Medial-vowel compatibility
        """
        # Medial Compatibility (Consonant-Medial)
        if not self._check_medial_compatibility(syllable):
            return False

        # Medial-Vowel Compatibility
        if not self._check_medial_vowel_compatibility(syllable):
            return False

        return True

    def _validate_vowels_tones(self, syllable: str) -> bool:
        """Validate vowel/tone combinations, virama usage, and diacritics.

        Checks:
            - Tone rules (stop finals, conflict)
            - Virama usage & stacking logic
            - Vowel combinations (digraphs)
            - Vowel exclusivity (upper vs lower slots)
            - E vowel combinations and position
            - Great Sa rules
            - Anusvara compatibility
            - Asat count
            - Duplicate diacritics
            - Tall A / Aa exclusivity (Phase 1)
            - Tall A after Medial Wa (Phase 1)
            - Dot below position (Phase 2)
            - Virama count (Phase 2)
            - Anusvara + Asat conflict (Phase 2)
            - Asat + Vowel conflict (Phase 3)
        """
        # Tone Rules (Stop finals, conflict)
        if not self._check_tone_rules(syllable):
            return False

        # Virama Usage & Stacking Logic
        if not self._check_virama_usage(syllable):
            return False

        # Vowel Combinations (Digraphs)
        if not self._check_vowel_combinations(syllable):
            return False

        # Vowel Exclusivity
        if not self._check_vowel_exclusivity(syllable):
            return False

        # E Vowel Combinations
        if not self._check_e_vowel_combinations(syllable):
            return False

        # E Vowel Position
        if not self._check_e_vowel_position(syllable):
            return False

        # Great Sa Rules
        if not self._check_great_sa_rules(syllable):
            return False

        # Anusvara Compatibility
        if not self._check_anusvara_compatibility(syllable):
            return False

        # Asat Count (Max 1 unless Kinzi)
        if not self._check_asat_count(syllable):
            return False

        # Duplicate Diacritics
        if not self._check_double_diacritics(syllable):
            return False

        # Tall A / Aa Exclusivity (Phase 1)
        if not self._check_tall_a_exclusivity(syllable):
            return False

        # Tall A after Medial Wa (Phase 1)
        # In standard Myanmar, ါ (U+102B) never follows medial ွ (U+103D).
        # Always use ာ (U+102C) after medial Wa.
        if not self._check_tall_aa_after_medial_wa(syllable):
            return False

        # Dot Below Position (Phase 2)
        if not self._check_dot_below_position(syllable):
            return False

        # Virama Count (Phase 2)
        if not self._check_virama_count(syllable):
            return False

        # Anusvara + Asat Conflict (Phase 2)
        if not self._check_anusvara_asat_conflict(syllable):
            return False

        # Asat + Vowel Conflict (Phase 3)
        if not self._check_asat_before_vowel(syllable):
            return False

        return True

    def _validate_advanced(self, syllable: str) -> bool:
        """Validate strict-mode rules, kinzi stacking, and virama ordering.

        Checks (strict mode only):
            - Strict tone rules (max 1 tone mark)
            - Strict tone position (must be final)
            - Character scope (core Myanmar only)
            - Strict diacritic uniqueness
            - One final rule (hygiene)
            - Strict kinzi rule

        Checks (when virama present):
            - Virama ordering (must be before medials)
            - Pat sint validity
        """
        if self.strict:
            # Strict Tone Rules (Max 1 tone mark)
            if not self._check_tone_strictness(syllable):
                return False

            # Strict Tone Position (Must be final)
            if not self._check_tone_position(syllable):
                return False

            # Character Scope (Core Myanmar only)
            if not self._check_character_scope(syllable):
                return False

            # Strict Diacritic Uniqueness
            if not self._check_diacritic_uniqueness(syllable):
                return False

            # One Final Rule (Hygiene)
            if not self._check_one_final_rule(syllable):
                return False

            # Strict Kinzi Rule (Nga + Virama without Asat is invalid)
            if not self._check_strict_kinzi(syllable):
                return False

        if VIRAMA in syllable:
            # Virama Ordering (Must be before Medials)
            if not self._check_virama_ordering(syllable):
                return False

            if not self._check_pat_sint_validity(syllable):
                return False

        return True

    def _check_start_char(self, syllable: str) -> bool:
        """
        Check if the syllable starts with a valid base character.
        Must be a Consonant or Independent Vowel.
        Prevents "floating diacritics" (e.g. starting with 1031, 102C, etc.)
        """
        first_char = syllable[0]
        # Note: Valid Particles (e.g. ၌) are handled before this.
        if first_char in CONSONANTS:
            return True
        if first_char in INDEPENDENT_VOWELS:
            return True

        # If it's a digit or symbol, we usually segment those out or handle them separately.
        # But structurally, a "syllable" shouldn't start with a diacritic.
        return False

    def _check_structure_sanity(self, syllable: str) -> bool:
        """
        Check structural sanity of syllable ordering.

        This validates:
        1. Medial sequences are valid combinations
        2. Vowels/tones don't appear before medials
        3. Visarga (if present) is at the end
        4. E vowel ordering with Tall A / Aa
        5. Anusvara position relative to tone marks

        Returns:
            True if structure is sane, False otherwise.
        """
        # Note: YA+RA (ျြ) sequence IS valid in Unicode Burmese per UTN #11.
        # These medial combinations are structurally valid but rare in modern
        # standard Burmese. They should not be rejected as invalid sequences.

        # Check medial sequences
        medial_sequences = self._medial_extractor.findall(syllable)
        for seq in medial_sequences:
            if seq not in VALID_MEDIAL_SEQUENCES:
                return False

        # Find first medial position
        first_medial_idx = -1
        for i, char in enumerate(syllable):
            if char in MEDIALS:
                first_medial_idx = i
                break

        # Vowels, tones, Asat, Visarga should not appear before first medial
        # Exception: Asat that is part of a Kinzi sequence (Nga+Asat+Virama) is allowed
        kinzi_seq = NGA + ASAT + VIRAMA
        kinzi_asat_positions: set[int] = set()
        kinzi_idx = syllable.find(kinzi_seq)
        while kinzi_idx != -1:
            kinzi_asat_positions.add(kinzi_idx + 1)  # position of Asat in Kinzi
            kinzi_idx = syllable.find(kinzi_seq, kinzi_idx + 1)

        if first_medial_idx != -1:
            for i in range(first_medial_idx):
                char = syllable[i]
                if char == ASAT and i in kinzi_asat_positions:
                    continue  # Skip Kinzi Asat
                if char in VOWEL_SIGNS or char in TONE_MARKS or char == ASAT or char == VISARGA:
                    return False

        # Visarga must be at the end if present (except as the last char)
        if syllable[-1] != VISARGA:
            if VISARGA in syllable:
                return False

        # E vowel ordering: Tall A / Aa should not appear before E vowel
        e_idx = syllable.find(VOWEL_E)
        if e_idx != -1:
            for i, char in enumerate(syllable):
                if char == TALL_A or char == AA_VOWEL:
                    if i < e_idx:
                        return False

        # Anusvara position: tone marks should not appear before Anusvara
        anu_idx = syllable.find(ANUSVARA)
        if anu_idx != -1:
            for i, char in enumerate(syllable):
                if (char == DOT_BELOW or char == VISARGA) and i < anu_idx:
                    return False

        return True

    def _check_asat_count(self, syllable: str) -> bool:
        """
        Check Asat count.
        Max 1 Asat allowed per syllable usually.
        Exception: Kinzi (Nga+Asat+Virama) + Final (C+Asat) = 2 Asats.
        """
        asat_count = syllable.count(ASAT)
        if asat_count <= 1:
            return True

        if asat_count > 2:
            return False

        # If 2 Asats, one MUST be part of a Kinzi sequence
        kinzi_seq = NGA + ASAT + VIRAMA
        if kinzi_seq not in syllable:
            return False

        return True

    def _check_virama_ordering(self, syllable: str) -> bool:
        """
        Check Virama (1039) position relative to Medials.
        Virama (Stacking) must occur BEFORE any Medial signs (103B-103E).
        """
        if not self.strict:
            return True

        # Find last Virama index
        v_idx = syllable.rfind(VIRAMA)

        # Find first Medial index
        m_idx = -1
        for i, char in enumerate(syllable):
            if char in MEDIALS:
                m_idx = i
                break

        if m_idx != -1 and v_idx > m_idx:
            return False

        return True

    def _check_anusvara_compatibility(self, syllable: str) -> bool:
        """
        Check Anusvara (1036) vowel compatibility.
        Anusvara is restrictive and cannot combine with most vowel signs
        (e.g. Aa, Tall A, E, AI, II, UU).
        It generally only works with Inherent vowel, I (102D), or U (102F).
        """
        if "\u1036" not in syllable:
            return True

        # Get all vowels in the syllable
        vowels = [c for c in syllable if c in VOWEL_SIGNS]

        for vowel in vowels:
            if vowel not in ANUSVARA_ALLOWED_VOWELS:
                return False

        return True

    def _check_kinzi_pattern(self, syllable: str) -> bool:
        """
        Validate kinzi sequences: Nga+Asat+Virama must be followed by valid consonant.

        The kinzi pattern (င်္) is used for nasalization in Pali/Sanskrit loanwords.
        Only certain consonants can validly follow the kinzi sequence.

        In strict mode, kinzi must also be preceded by nothing OR a consonant/
        independent vowel - not by a dependent vowel, medial, or tone mark.
        This prevents invalid placements like vowel+kinzi+consonant.
        """
        kinzi_seq = NGA + ASAT + VIRAMA
        if kinzi_seq in syllable:
            idx = syllable.find(kinzi_seq)

            # Strict: Kinzi must be preceded by nothing OR consonant/independent vowel
            # Not by dependent vowel, medial, or tone marks
            if self.strict and idx > 0:
                prev_char = syllable[idx - 1]
                # Valid predecessors: consonants and independent vowels
                if prev_char not in CONSONANTS and prev_char not in INDEPENDENT_VOWELS:
                    return False

            if idx + 3 >= len(syllable):
                return False
            next_char = syllable[idx + 3]

            if self.strict:
                # Strict: Must be in KINZI_VALID_FOLLOWERS
                if next_char not in KINZI_VALID_FOLLOWERS:
                    return False
            else:
                # Lenient: Fall back to CONSONANTS for broader compatibility
                # (Original logic allowed any consonant)
                if next_char not in KINZI_VALID_FOLLOWERS and next_char not in CONSONANTS:
                    return False
        return True

    def _check_vowel_exclusivity(self, syllable: str) -> bool:
        """
        Check for mutually exclusive vowel slots (Upper vs Upper, Lower vs Lower).
        """
        uppers = [c for c in syllable if c in UPPER_VOWELS]
        lowers = [c for c in syllable if c in LOWER_VOWELS]

        if len(uppers) > 1:
            return False
        if len(lowers) > 1:
            return False
        return True

    def _check_great_sa_rules(self, syllable: str) -> bool:
        """
        Validate Great Sa (U+103F) usage patterns.

        Great Sa is a special conjunct form of doubled Sa (သ+သ) used primarily
        in Pali/Sanskrit loanwords. It has specific usage restrictions:

        1. Cannot take medials (ျ ြ ွ ှ)
        2. Cannot stack via virama (it's already a conjunct)
        3. Cannot be followed by another Great Sa
        4. Must be followed by valid vowel signs, tones, or asat (if any)
        5. Typically not syllable-initial in native Myanmar words

        Returns:
            True if Great Sa usage is valid, False otherwise.
        """
        if GREAT_SA not in syllable:
            return True

        # Find position of Great Sa
        pos = syllable.index(GREAT_SA)

        # Rule 1: Great Sa cannot take medials (check after Great Sa position only)
        if any(c in MEDIALS for c in syllable[pos:]):
            return False

        # Rule 2: Great Sa cannot stack via virama (it's already a stack)
        if VIRAMA in syllable:
            return False

        # Rule 3: Cannot have multiple Great Sa in one syllable
        if syllable.count(GREAT_SA) > 1:
            return False

        # Rule 3b: Great Sa + Asat is invalid (Great Sa is already doubled Sa;
        # killing it makes no linguistic sense in standard Burmese or Pali)
        if pos + 1 < len(syllable) and syllable[pos + 1] == ASAT:
            return False

        # Rule 4: Check what follows Great Sa (if anything)
        if pos + 1 < len(syllable):
            next_char = syllable[pos + 1]
            # Great Sa can be followed by: vowel signs, tone marks, or visarga
            valid_followers = VOWEL_SIGNS | TONE_MARKS | {VISARGA}
            if next_char not in valid_followers:
                # Also allow if followed by another consonant (rare but valid)
                if next_char not in CONSONANTS:
                    return False

        # Rule 5: Great Sa should not be preceded by certain characters
        if pos > 0:
            prev_char = syllable[pos - 1]
            # Great Sa typically follows vowel signs or is at syllable start
            # It should not follow medials or tone marks
            if prev_char in MEDIALS:
                return False
            if prev_char in TONE_MARKS:
                return False

        return True

    def _check_e_vowel_combinations(self, syllable: str) -> bool:
        """
        Check E vowel (U+1031) combinations.

        E vowel cannot combine with:
        - I (U+102D), II (U+102E), U (U+102F), UU (U+1030) - in INVALID_E_COMBINATIONS
        - Anusvara (U+1036) - phonotactically impossible in standard Myanmar

        Returns:
            True if E vowel combinations are valid, False otherwise.
        """
        if VOWEL_E in syllable:
            # Check standard invalid combinations
            for invalid in INVALID_E_COMBINATIONS:
                if invalid in syllable:
                    return False
            # E + Anusvara is also invalid
            # Anusvara is classified as a tone mark, not a vowel sign,
            # so it's not caught by INVALID_E_COMBINATIONS
            if ANUSVARA in syllable:
                return False
        return True

    def _check_e_vowel_position(self, syllable: str) -> bool:
        """
        Check E vowel (U+1031) position validity.

        E vowel appears before the consonant visually but after in encoding.
        In Unicode encoding order, E vowel must:
        1. Not be at position 0 (already checked by _check_start_char)
        2. Come immediately after a consonant, medial, or independent vowel
        3. Not appear multiple times in a syllable

        Valid examples:
            ကေ (Ka + E) - E after consonant
            ကြေ (Ka + medial Ya + E) - E after medial
            က္ကေ (Ka + virama + Ka + E) - E after stacked consonant

        Invalid examples:
            ေက (E + Ka) - E at start (floating diacritic)
            ကေေ (Ka + E + E) - Multiple E vowels
            ကိေ (Ka + I + E) - E after another vowel

        Returns:
            True if E vowel position is valid, False otherwise.
        """
        if VOWEL_E not in syllable:
            return True

        # Check: E vowel should appear only once
        if syllable.count(VOWEL_E) > 1:
            return False

        # Find E vowel position
        e_idx = syllable.find(VOWEL_E)

        # E vowel at position 0 is invalid (floating diacritic)
        # This should already be caught by _check_start_char
        if e_idx == 0:
            return False

        # Check what precedes E vowel
        prev_char = syllable[e_idx - 1]

        # Valid predecessors: consonants, medials, or independent vowels
        # (for rare cases like ဥ + E in archaic spellings)
        # Note: VIRAMA is not valid because E cannot directly follow Virama —
        # the consonant after Virama is the actual predecessor (e.g. က္ကေ is valid
        # because E follows Ka, not Virama).
        valid_predecessors = CONSONANTS | MEDIALS | INDEPENDENT_VOWELS

        if prev_char not in valid_predecessors:
            # E vowel after another vowel sign or tone mark is invalid
            return False

        return True

    def _check_asat_predecessor(self, syllable: str) -> bool:
        """
        Check that Asat (103A) is always preceded by a Consonant.
        It cannot follow Vowels, Medials, or other Tones.
        """
        if ASAT not in syllable:
            return True

        # Find all occurrences (though usually only 1 or 2 in Kinzi)
        for i, char in enumerate(syllable):
            if char == ASAT:
                if i == 0:
                    return False  # Asat at start

                prev_char = syllable[i - 1]

                # Allowed: Consonants or Independent Vowels
                if prev_char in CONSONANTS or prev_char in INDEPENDENT_VOWELS:
                    # Strict Mode: Reject U+1021 (Ah) + Asat (Non-standard)
                    if self.strict and prev_char == "\u1021":
                        return False
                    continue

                # Allowed: Aa (102C) or Tall A (102B) - for 'Aw' vowel (e.g. ပျော်)
                # BUT only if E vowel (1031) is also present in the syllable (e.g. ေ...ာ်)
                if prev_char == "\u102c" or prev_char == "\u102b":
                    if VOWEL_E in syllable:
                        continue

                # Allowed: Dot Below (1037) or Anusvara (1036) before Asat
                # in pre-normalization text (e.g. ည့် before reorder to ည့်)
                if prev_char == DOT_BELOW or prev_char == ANUSVARA:
                    continue

                # Invalid: Other Vowels, Medials, Tones, other Asat
                return False

        return True

    def _check_vowel_combinations(self, syllable: str) -> bool:
        """
        Check for valid multiple vowel combinations (digraphs).

        Special case: Great Sa (U+103F) can have its own vowels, so vowels
        before and after Great Sa don't count as a single combination.
        """
        # Check if Great Sa is present - it can carry its own vowels
        if GREAT_SA in syllable:
            # Split syllable at Great Sa and check each part separately
            gs_pos = syllable.index(GREAT_SA)
            before_gs = syllable[:gs_pos]
            after_gs = syllable[gs_pos + 1 :]

            # Check vowels before Great Sa
            vowels_before = [c for c in before_gs if c in VOWEL_SIGNS]
            if len(vowels_before) > 1:
                v_set = frozenset(vowels_before)
                if v_set not in VALID_VOWEL_COMBINATIONS:
                    return False

            # Check vowels after Great Sa
            vowels_after = [c for c in after_gs if c in VOWEL_SIGNS]
            if len(vowels_after) > 1:
                v_set = frozenset(vowels_after)
                if v_set not in VALID_VOWEL_COMBINATIONS:
                    return False

            return True

        # Standard check for syllables without Great Sa
        # Use a set to collect unique vowels (consistent with Cython implementation)
        vowels = {c for c in syllable if c in VOWEL_SIGNS}

        if len(vowels) > 1:
            v_set = frozenset(vowels)
            if v_set not in VALID_VOWEL_COMBINATIONS:
                return False

        return True

    def _check_double_diacritics(self, syllable: str) -> bool:
        """Check for duplicate consecutive diacritics."""
        # Check Vowels, Medials, Tones
        # We iterate and check if current == next and both are diacritics
        all_diacritics = MEDIALS.union(VOWEL_SIGNS).union(TONE_MARKS)

        for i in range(len(syllable) - 1):
            if syllable[i] == syllable[i + 1]:
                if syllable[i] in all_diacritics:
                    return False
        return True

    def _get_medial_base_consonant(self, syllable: str) -> str | None:
        """
        Find the true base consonant that medials attach to.

        Medials phonotactically attach to the "base" consonant, not the initial
        character. For Kinzi-initial or stacked-initial syllables, the base is
        the consonant after the Kinzi or stacking sequence.

        Resolution rules:
        1. If syllable starts with Kinzi (NGA + ASAT + VIRAMA), the base is
           the next consonant after the 3-codepoint Kinzi sequence.
        2. If syllable starts with initial stacked consonants (C + VIRAMA + C ...),
           the base is the consonant immediately after the last virama in the
           initial stacking chain.
        3. Otherwise, the base is the first consonant in the syllable.

        Args:
            syllable: The syllable string to analyze.

        Returns:
            The base consonant character that medials attach to, or None if
            no valid base consonant can be determined.
        """
        if not syllable:
            return None

        kinzi_seq = NGA + ASAT + VIRAMA

        # Rule 1: Kinzi-initial syllable
        if syllable.startswith(kinzi_seq):
            # Base is the consonant after Kinzi (3 characters)
            remaining = syllable[3:]
            if remaining:
                # The first consonant after Kinzi is the base
                for ch in remaining:
                    if ch in CONSONANTS:
                        return ch
            return None

        # Rule 1b: Non-initial Kinzi (merged segment like အင်္ကျီ)
        # The medial base is the consonant immediately after the Kinzi sequence
        kinzi_pos = syllable.find(kinzi_seq)
        if kinzi_pos != -1:
            after_kinzi = kinzi_pos + len(kinzi_seq)
            for i in range(after_kinzi, len(syllable)):
                if syllable[i] in CONSONANTS:
                    return syllable[i]
            return None

        # Rule 2: Initial stacked consonants (C + VIRAMA + C ...)
        # Find the consonant after the last virama in the initial stacking chain
        pos = 0
        base = None

        while pos < len(syllable):
            ch = syllable[pos]
            if ch in CONSONANTS:
                base = ch
                pos += 1
                # Check if followed by VIRAMA (stacking continues)
                if pos < len(syllable) and syllable[pos] == VIRAMA:
                    pos += 1  # Skip virama, look for next consonant
                    continue
                else:
                    # No more stacking, we found the base
                    return base
            else:
                # Non-consonant character - stop looking
                break

        # Rule 3: First consonant is the base (simple case)
        if base:
            return base

        # Fallback: return the first character if it's a consonant
        if syllable[0] in CONSONANTS:
            return syllable[0]

        return None

    def _check_medial_compatibility(self, syllable: str) -> bool:
        """
        Check consonant-medial compatibility.

        Myanmar medials have phonotactic constraints on which consonants they can
        attach to. This check validates that the BASE consonant (not necessarily
        the first character) is compatible with any medials present.

        For Kinzi-initial or stacked-initial syllables, the base consonant is
        resolved using _get_medial_base_consonant().

        Compatibility rules (in strict mode):
        - Medial Ya (ျ): Compatible with Ka-group, Pa-group, and Tha
        - Medial Ra (ြ): Compatible with Ka-group and Pa-group
        - Medial Wa (ွ): Broadly compatible (most consonants)
        - Medial Ha (ှ): Compatible only with sonorants (nasals, liquids, glides)

        Args:
            syllable: The syllable string to check.

        Returns:
            True if consonant-medial compatibility is valid, False otherwise.
        """
        if not syllable:
            return True

        # Check for presence of each medial
        has_ya = MEDIAL_YA in syllable
        has_ra = MEDIAL_RA in syllable
        has_wa = MEDIAL_WA in syllable
        has_ha = MEDIAL_HA in syllable

        # If no medials, no compatibility check needed
        if not (has_ya or has_ra or has_wa or has_ha):
            return True

        # Get the true base consonant (handles Kinzi and stacked syllables)
        base_char = self._get_medial_base_consonant(syllable)

        # If no valid base consonant found and strict mode, fail the check
        if base_char is None:
            return not self.strict

        # In strict mode, validate against compatibility sets
        if self.strict:
            if has_ya and base_char not in COMPATIBLE_YA:
                return False
            if has_ra and base_char not in COMPATIBLE_RA:
                return False
            if has_wa and base_char not in COMPATIBLE_WA:
                return False
            if has_ha and base_char not in COMPATIBLE_HA:
                return False

        return True

    def _check_medial_vowel_compatibility(self, syllable: str) -> bool:
        """
        Check for impossible medial-vowel combinations.

        Myanmar has complex phonotactic constraints on medial+vowel combinations:

        1. Medial-Ya (ျ U+103B):
           - Cannot combine with ဲ (ai vowel, U+1032) - phonetically incompatible

        2. Medial-Ra (ြ U+103C):
           - Generally permissive
           - Warn/rare: with ု/ူ (u vowels)

        3. Medial-Wa (ွ U+103D):
           - Cannot combine with ု/ူ (u vowels) - occupy same phonetic space

        4. Medial-Ha (ှ U+103E):
           - Generally permissive, context-dependent

        5. Combined medials (ျွ, ြွ, etc.):
           - Ya+Wa (ျွ) cannot combine with ု/ူ
           - Ra+Wa (ြွ) cannot combine with ု/ူ
        """
        # Extract medials present in syllable
        medials_in_syllable = [c for c in syllable if c in MEDIALS]
        vowels_in_syllable = [c for c in syllable if c in VOWEL_SIGNS]

        if not medials_in_syllable:
            return True  # No medials, no constraint

        # Single medial constraints
        # Note: Medial Ya + AI vowel (ျ + ဲ) was previously considered invalid,
        # but this combination does occur in Pali/Sanskrit loanwords and proper
        # nouns, so it is no longer rejected.
        medial_constraints = {
            MEDIAL_WA: {VOWEL_U, VOWEL_UU},  # Wa + u/uu invalid (phonetically incompatible)
            # MEDIAL_YA, MEDIAL_RA, and MEDIAL_HA are generally permissive
        }

        # Check single medial constraints
        for medial in medials_in_syllable:
            if medial in medial_constraints:
                forbidden = medial_constraints[medial]
                if any(v in forbidden for v in vowels_in_syllable):
                    return False

        # Combined medial constraints
        # If both Ya and Wa present, or both Ra and Wa present
        has_ya = MEDIAL_YA in medials_in_syllable
        has_ra = MEDIAL_RA in medials_in_syllable
        has_wa = MEDIAL_WA in medials_in_syllable

        # Ya+Wa (ျွ) or Ra+Wa (ြွ) cannot combine with u vowels
        if (has_ya and has_wa) or (has_ra and has_wa):
            if VOWEL_U in vowels_in_syllable or VOWEL_UU in vowels_in_syllable:
                return False

        # Additional constraint: Multiple stacked vowels with certain medials
        # E.g., Ya + multiple vowels is rare and often invalid
        if has_ya and len(vowels_in_syllable) > 2:
            return False

        return True

    def _check_corruption(self, syllable: str) -> bool:
        """Check for obvious data corruption."""
        # Excessive repetition (>threshold consecutive identical chars)
        # Default threshold is 3, so checks for 4 consecutive chars
        if len(syllable) > self.corruption_threshold:
            for i in range(len(syllable) - self.corruption_threshold):
                chunk = syllable[i : i + self.corruption_threshold + 1]
                if len(set(chunk)) == 1:
                    return False

        # Diacritic Spam Check
        # A valid syllable should not have >2 of the same diacritic
        # (e.g. 3 'Ya-pins' or 3 'Tone-dots' are impossible)
        counts: dict[str, int] = {}
        all_diacritics = MEDIALS.union(VOWEL_SIGNS).union(TONE_MARKS)

        for char in syllable:
            if char in all_diacritics:
                counts[char] = counts.get(char, 0) + 1
                if counts[char] > 2:
                    return False

        return True

    def _check_independent_vowel(self, syllable: str) -> bool:
        """
        Independent vowels cannot take medials or dependent vowels.
        They CAN take finals (with Asat) and tones (Visarga, Dot Below).

        Note: U+1021 (အ) is handled separately in validate() - it behaves
        like a consonant carrier and goes through normal validation flow.
        """
        # Start with independent vowel
        # Should not have medials
        if any(c in MEDIALS for c in syllable):
            return False

        # Should not have dependent vowels
        # Note: some sources suggest even this might have rare exceptions,
        # but in standard spelling, Indep Vowels replace base+vowel.
        if any(c in VOWEL_SIGNS for c in syllable):
            return False

        return True

    def _check_unexpected_consonant(self, syllable: str) -> bool:
        """
        Check for multiple base consonants that are not properly stacked.
        A syllable should generally have only one initial base consonant (or independent vowel).
        Subsequent consonants must be finals (killed), stacked (under Virama),
        or stackers (followed by Virama).

        Special case: Great Sa (U+103F) is a self-contained conjunct (doubled Sa)
        that can appear after vowels without virama stacking.
        """
        num_active_bases = 0
        syllable_len = len(syllable)

        for i, char in enumerate(syllable):
            if char in CONSONANTS or char in INDEPENDENT_VOWELS:
                # It is a potential base.

                # Special case: Great Sa (U+103F) is a self-contained conjunct
                # It can follow vowels and is treated as a syllable extension
                # rather than a new base consonant
                if char == GREAT_SA:
                    # Great Sa after vowel signs is valid in Pali loanwords
                    # e.g., ဗိဿ (vissa), ပိဿာ (pissa)
                    continue

                # 1. Is it a Final? (Followed by Asat)
                is_final = False
                if i + 1 < syllable_len and syllable[i + 1] == ASAT:
                    is_final = True
                elif char == "\u100a" and i + 1 < syllable_len and syllable[i + 1] == DOT_BELOW:
                    # Exception: Nya (100A) + Dot Below acts as final (creaky tone)
                    is_final = True

                # 2. Is it Stacked? (Preceded by Virama)
                is_stacked = False
                if i > 0 and syllable[i - 1] == VIRAMA:
                    is_stacked = True

                # 3. Is it a Stacker? (Followed by Virama)
                # e.g. the 'Ta' in Met-ta (Ma + E + Ta + Virama + Ta + Aa)
                is_stacker = False
                if i + 1 < syllable_len and syllable[i + 1] == VIRAMA:
                    is_stacker = True

                # If it's none of these, it's an independent base in this block
                if not is_final and not is_stacked and not is_stacker:
                    num_active_bases += 1

        # Allow max 1 active base
        return num_active_bases <= 1

    def _check_virama_usage(self, syllable: str) -> bool:
        """Check correct usage of Virama (stacking)."""
        # Virama must not be at the end of the syllable
        if syllable.endswith(VIRAMA):
            return False

        # Virama must be followed by a consonant (stacking)
        for i, char in enumerate(syllable):
            if char == VIRAMA and i + 1 < len(syllable):
                next_char = syllable[i + 1]
                if next_char not in CONSONANTS:
                    return False

        return True

    def _check_tone_rules(self, syllable: str) -> bool:
        """
        Check tone mark rules.

        Rules:
        1. Dot Below and Visarga cannot appear in the same syllable
        2. Stop finals (certain consonants with Asat) cannot have tone marks
        3. Each tone mark can only appear once per syllable

        Returns:
            True if tone rules are valid, False otherwise.
        """
        has_dot = DOT_BELOW in syllable
        has_visarga = VISARGA in syllable

        # Dot Below and Visarga are mutually exclusive
        if has_dot and has_visarga:
            return False

        # Each tone mark can only appear once per syllable
        # This check applies in both strict and non-strict modes
        if syllable.count(DOT_BELOW) > 1:
            return False
        if syllable.count(VISARGA) > 1:
            return False

        # If no tone marks, no further checks needed
        if not (has_dot or has_visarga):
            return True

        # Check for stop finals - they cannot have tone marks
        for i in range(1, len(syllable)):
            if syllable[i] == ASAT:
                prev_char = syllable[i - 1]
                if prev_char in STOP_FINALS:
                    return False

        return True

    def _check_pat_sint_validity(self, syllable: str) -> bool:
        """
        Check validity of stacked consonants (Pat Sint / Vagga rules).
        Only called if Virama is present.

        Priority order for validation:
        1. Kinzi pattern (Nga + Asat + Virama) - special case
        2. STACKING_EXCEPTIONS - Pali/Sanskrit loanword patterns take priority
        3. Wet (Vagga) Logic - native Myanmar stacking rules
        """
        if not self.strict:
            return True

        # Find Virama index
        try:
            v_idx = syllable.index(VIRAMA)
        except ValueError:
            return True  # Should not happen if called correctly

        # Check bounds
        if v_idx == 0 or v_idx == len(syllable) - 1:
            return False  # Invalid position (handled by _check_virama_usage, but safe to recheck)

        upper = syllable[v_idx - 1]
        lower = syllable[v_idx + 1]

        # Exception: Kinzi (Nga + Asat + Virama)
        # Virama is preceded by Asat in this specific case.
        if upper == ASAT:
            if v_idx >= 2 and syllable[v_idx - 2] == NGA:
                return True
            return False

        # Strict check: Upper and Lower must be Consonants
        # (or Independent Vowels, though stacking on them is rare/invalid,
        # but strictly they function as bases).
        # Medials, Vowels, Tones cannot be stacked upon or underscripted.
        if upper not in CONSONANTS:
            return False
        if lower not in CONSONANTS:
            return False

        # Priority 1: Check stacking pairs first for ALL consonant pairs
        # Pali/Sanskrit loanwords often have cross-vagga stacking that violates
        # native Myanmar Wet Logic but is valid in Indic languages.
        # Uses configurable stacking pairs (loaded from YAML or hardcoded defaults).
        if (upper, lower) in self._stacking_pairs:
            return True

        # Priority 2: Handle special cases for non-Vagga consonants
        # If either is not a standard Vagga consonant, and not in exceptions,
        # apply strict rejection for unknown combinations.
        if upper not in WET_MAPPING or lower not in WET_MAPPING:
            # "Sa" (U+101E) is common. Sa+Sa is 103F.
            # If we see Sa+Virama+Sa, it's likely invalid (should be 103F).
            if upper == "\u101e" and lower == "\u101e":
                return False

            # Non-Vagga stacks that are not in STACKING_EXCEPTIONS are invalid.
            # Unknown combinations should be rejected for stricter validation.
            # This prevents garbage data from passing through.
            return False

        # Priority 3: Both are Vagga consonants - apply Wet Logic for native patterns
        row1, col1 = WET_MAPPING[upper]
        row2, col2 = WET_MAPPING[lower]

        # Rule 1: Must be same Wet (Row)
        if row1 != row2:
            # Cross-row stacking not allowed for native patterns
            # (Pali/Sanskrit cross-row patterns should be in STACKING_EXCEPTIONS)
            return False

        # Rule 2: Column Logic (Upper -> Lower)
        # Col 1 stacks on 1 or 2
        if col1 == 1:
            if col2 not in (1, 2):
                return False

        # Col 3 stacks on 3 or 4
        elif col1 == 3:
            if col2 not in (3, 4):
                return False

        # Col 5 (Nasal) is typically UPPER (Nga-Kinzi). When upper (e.g. Nya),
        # it can stack on self (Nya+Nya = 100A+1039+100A) or Nya+Ca.
        # Permit Col 5 Upper to stack on anything in the same row.
        return True

    # =========================================================================
    # Phase 1 & 2 Improvements (Encoding & Phonotactic Rules)
    # =========================================================================

    def _check_zero_width_chars(self, syllable: str) -> bool:
        """
        Reject syllables containing zero-width characters.

        Zero-width characters (ZWSP, ZWNJ, ZWJ, BOM) can hide malformed
        syllables and cause comparison/matching issues. They should be
        stripped during normalization, but validation should also reject them.
        """
        for char in syllable:
            if char in ZERO_WIDTH_CHARS:
                return False
        return True

    def _check_tall_a_exclusivity(self, syllable: str) -> bool:
        """
        Check that Tall A (102B) and Aa (102C) are mutually exclusive.

        These vowels occupy the same phonetic slot. Having both in a single
        syllable indicates encoding error or data corruption.
        """
        if TALL_A in syllable and AA_VOWEL in syllable:
            return False
        return True

    def _check_tall_aa_after_medial_wa(self, syllable: str) -> bool:
        """
        Check that Tall A (U+102B) does not appear after Medial Wa (U+103D).

        In standard Myanmar orthography, after medial ွ (wa-hswe), the vowel
        is always written as ာ (U+102C, AA), never ါ (U+102B, Tall AA).
        The ါ form after ွ is a Zawgyi-era artifact with zero valid exceptions
        in standard Burmese.

        Examples:
            - ပွား (valid) vs ပွါး (invalid)
            - ဂွာ (valid) vs ဂွါ (invalid)
            - သွား (valid) vs သွါး (invalid)

        See: https://github.com/thettwe/my-spellchecker/issues/1357
        """
        if MEDIAL_WA not in syllable:
            return True
        if TALL_A not in syllable:
            return True
        # Both medial Wa and Tall A present → invalid
        return False

    def _check_dot_below_position(self, syllable: str) -> bool:
        """
        Check that Dot Below (1037) is not followed by vowels, medials, or anusvara.

        Dot below (Auk-myit) is a tone marker that should appear near the end
        of a syllable, after vowels and anusvara.
        """
        if DOT_BELOW not in syllable:
            return True

        dot_idx = syllable.find(DOT_BELOW)

        # Check that nothing invalid follows the dot below
        for i in range(dot_idx + 1, len(syllable)):
            char = syllable[i]
            # Vowels, medials, or anusvara should not follow dot below
            if char in VOWEL_SIGNS or char in MEDIALS or char == ANUSVARA:
                return False
        return True

    def _check_virama_count(self, syllable: str) -> bool:
        """
        Check that virama count is limited.

        A syllable should have at most 1 virama (stacking marker).
        Exception: Kinzi pattern may combine with stacking, allowing 2 viramas.
        """
        virama_count = syllable.count(VIRAMA)
        if virama_count <= 1:
            return True

        # Allow 2 viramas only if Kinzi pattern is present
        kinzi_seq = NGA + ASAT + VIRAMA
        if virama_count == 2 and kinzi_seq in syllable:
            return True

        return False

    def _check_anusvara_asat_conflict(self, syllable: str) -> bool:
        """
        Check that Anusvara (1036) is not immediately followed by Asat (103A).

        Anusvara (nasalization marker) cannot be immediately followed by Asat
        (vowel killer). This combination is phonotactically impossible.
        """
        # Check for the invalid sequence: Anusvara + Asat
        invalid_seq = ANUSVARA + ASAT
        if invalid_seq in syllable:
            return False
        return True

    def _check_asat_before_vowel(self, syllable: str) -> bool:
        """
        Check that Asat (103A) is not immediately followed by a vowel sign.

        Asat (vowel killer) terminates the vowel of a syllable. Having a vowel
        sign immediately after Asat is structurally invalid and typically
        indicates encoding error or data corruption (e.g., တင်ူ should be တင်းူ).

        This pattern was detected in 925 syllables during database analysis.
        """
        for i in range(len(syllable) - 1):
            if syllable[i] == ASAT:
                if syllable[i + 1] in VOWEL_SIGNS:
                    return False
        return True

    def _check_tone_strictness(self, syllable: str) -> bool:
        """
        Strictly enforce maximum one tone mark per syllable.
        Allowed: 0 or 1 mark (Dot Below OR Visarga).
        Forbidden: > 1 marks (e.g., Dot+Dot, Visarga+Visarga, Dot+Visarga).

        NOTE on Visarga (U+1038, း) dual role:
        1. As a tone marker (wit-sa-pauk/ဝစ်ဆပေါက်): Indicates creaky tone,
           attached to syllables like "ကား" (car), "လား" (question particle).
        2. As a standalone sentence-final particle: "း" alone can mark emphasis
           or sentence boundaries, especially in informal/colloquial writing.

        This syllable-level validation treats Visarga strictly as a tone marker
        that must be part of a well-formed syllable. Standalone "း" as a
        sentence-final particle should be handled at the word/sentence level.
        """
        dot_count = syllable.count(DOT_BELOW)
        visarga_count = syllable.count(VISARGA)

        if dot_count + visarga_count > 1:
            return False
        return True

    def _check_tone_position(self, syllable: str) -> bool:
        """
        Strictly enforce tone marks (Dot/Visarga) to be at the end of the syllable.

        A tone mark modifies the entire syllable vowel/final. Nothing should follow it
        except maybe invisible formatting chars (which are already rejected).

        Specifically: No Consonants, Medials, Vowels, Asat, or Anusvara after a Tone.
        """
        # Check Dot Below
        if DOT_BELOW in syllable:
            idx = syllable.find(DOT_BELOW)
            # Nothing allowed after Dot Below in strict mode
            # (Standard spelling puts Dot Below at very end, after Anusvara/Finals)
            if idx != len(syllable) - 1:
                return False

        # Check Visarga
        if VISARGA in syllable:
            idx = syllable.find(VISARGA)
            # Nothing allowed after Visarga
            if idx != len(syllable) - 1:
                return False

        return True

    def _check_character_scope(self, syllable: str) -> bool:
        """
        Enforce Myanmar character scope based on allow_extended_myanmar setting.

        When allow_extended_myanmar is False (default):
            - Accepts only standard Burmese (U+1000-U+104F)
            - Rejects Extended Core (U+1050-U+109F), Extended-A/B, and non-standard chars

        When allow_extended_myanmar is True:
            - Accepts Extended Core (U+1050-U+109F), Extended-A (U+AA60-AA7F),
              Extended-B (U+A9E0-A9FF), and non-standard core chars
        """
        # Get valid character set based on scope
        valid_chars = get_myanmar_char_set(self.allow_extended_myanmar)

        for char in syllable:
            if char not in valid_chars:
                return False

        return True

    def _check_diacritic_uniqueness(self, syllable: str) -> bool:
        """
        Strictly enforce uniqueness of specific diacritics.
        A syllable cannot have two 'Ya-pins' or two 'Aa' vowels.

        Exceptions:
        - Virama (Stacking) handled elsewhere (max 2 with Kinzi).
        - Asat (Killer) handled elsewhere (max 2 with Kinzi).
        """
        # Check counts of all non-base chars
        counts: dict[str, int] = {}
        for char in syllable:
            if char in MEDIALS or char in VOWEL_SIGNS:
                counts[char] = counts.get(char, 0) + 1
                if counts[char] > 1:
                    return False
        return True

    def _check_one_final_rule(self, syllable: str) -> bool:
        """
        Enforce One Final Rule: A syllable can have at most one final.

        Finals include:
        1. Anusvara (1036)
        2. Consonant + Asat (103A) - unless part of Kinzi (Asat+Virama)

        Invalid: Anusvara + Consonant+Asat (e.g. ကံန်း)
        """
        if ANUSVARA not in syllable:
            return True

        # If Anusvara is present, check for non-Kinzi Asat
        for i, char in enumerate(syllable):
            if char == ASAT:
                # Check if it's Kinzi (followed by Virama)
                is_kinzi = False
                if i + 1 < len(syllable) and syllable[i + 1] == VIRAMA:
                    is_kinzi = True

                if not is_kinzi:
                    return False
        return True

    def _check_strict_kinzi(self, syllable: str) -> bool:
        """
        Enforce Strict Kinzi: Nga + Virama without Asat is invalid.

        Proper Kinzi sequence is: Nga (U+1004) + Asat (U+103A) + Virama (U+1039).
        If we see Nga followed directly by Virama (without Asat), it's typically
        a typo for Kinzi (missing Asat) and should be rejected in strict mode.

        Example:
        - Valid Kinzi: င်္က (Nga + Asat + Virama + Ka)
        - Invalid: င္က (Nga + Virama + Ka) - missing Asat

        Returns:
            True if valid (no Nga+Virama without Asat), False otherwise.
        """
        for i in range(len(syllable) - 1):
            if syllable[i] == NGA and syllable[i + 1] == VIRAMA:
                # Found Nga + Virama without Asat
                # In standard Burmese, this is usually a typo for Kinzi
                return False
        return True


# Select implementation (Cython if available, else Python)
try:
    from myspellchecker.core.syllable_rules_c import (
        SyllableRuleValidator as _SyllableRuleValidatorCython,
    )

    SyllableRuleValidator = _SyllableRuleValidatorCython
    _USING_CYTHON = True
except ImportError:
    SyllableRuleValidator = _SyllableRuleValidatorPython
    _USING_CYTHON = False
