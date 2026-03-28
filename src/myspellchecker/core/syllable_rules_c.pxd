# cython: language_level=3
from libc.stdint cimport uint32_t

cdef class SyllableRuleValidator:
    cdef int max_syllable_length
    cdef int corruption_threshold
    cdef bint strict
    cdef public bint allow_extended_myanmar
    cdef object _valid_myanmar_chars
    cdef object _medial_extractor
    cdef object _stacking_pairs

    cpdef bint validate(self, str syllable)
    
    # Internal C methods for performance
    cdef bint _check_zero_width_chars(self, str syllable)
    cdef bint _check_corruption(self, str syllable)
    cdef bint _check_start_char(self, str syllable)
    cdef bint _check_independent_vowel(self, str syllable)
    cdef bint _check_structure_sanity(self, str syllable)
    cdef bint _check_kinzi_pattern(self, str syllable)
    cdef bint _check_asat_predecessor(self, str syllable)
    cdef bint _check_unexpected_consonant(self, str syllable)
    cdef uint32_t _get_medial_base_consonant(self, str syllable)
    cdef bint _check_medial_compatibility(self, str syllable)
    cdef bint _check_medial_vowel_compatibility(self, str syllable)
    cdef bint _check_tone_rules(self, str syllable)
    cdef bint _check_virama_usage(self, str syllable)
    cdef bint _check_vowel_combinations(self, str syllable)
    cdef bint _check_vowel_exclusivity(self, str syllable)
    cdef bint _check_e_vowel_combinations(self, str syllable)
    cdef bint _check_e_vowel_position(self, str syllable)
    cdef bint _check_great_sa_rules(self, str syllable)
    cdef bint _check_anusvara_compatibility(self, str syllable)
    cdef bint _check_asat_count(self, str syllable)
    cdef bint _check_double_diacritics(self, str syllable)
    cdef bint _check_tall_a_exclusivity(self, str syllable)
    cdef bint _check_tall_aa_after_medial_wa(self, str syllable)
    cdef bint _check_dot_below_position(self, str syllable)
    cdef bint _check_virama_count(self, str syllable)
    cdef bint _check_anusvara_asat_conflict(self, str syllable)
    cdef bint _check_asat_before_vowel(self, str syllable)
    cdef bint _check_virama_ordering(self, str syllable)
    cdef bint _check_pat_sint_validity(self, str syllable)
    
    # New strict methods
    cdef bint _check_tone_strictness(self, str syllable)
    cdef bint _check_tone_position(self, str syllable)
    cdef bint _check_character_scope(self, str syllable)
    cdef bint _check_diacritic_uniqueness(self, str syllable)
    cdef bint _check_one_final_rule(self, str syllable)
    cdef bint _check_strict_kinzi(self, str syllable)
