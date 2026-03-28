"""Tests for the Python implementation of SyllableRuleValidator.

These tests specifically target the _SyllableRuleValidatorPython class
to ensure coverage of the Python fallback implementation, regardless
of whether Cython is available.
"""

import pytest

from myspellchecker.core.syllable_rules import _SyllableRuleValidatorPython


@pytest.fixture
def validator():
    """Create a Python-only validator for testing."""
    return _SyllableRuleValidatorPython()


@pytest.fixture
def lenient_validator():
    """Create a lenient (non-strict) validator for testing."""
    return _SyllableRuleValidatorPython(strict=False)


class TestValidatorInitialization:
    """Tests for validator initialization and configuration."""

    def test_default_initialization(self):
        """Test default parameter values."""
        v = _SyllableRuleValidatorPython()
        assert v.max_syllable_length == 15
        assert v.corruption_threshold == 3
        assert v.strict is True

    def test_custom_initialization(self):
        """Test custom parameter values."""
        v = _SyllableRuleValidatorPython(corruption_threshold=5, strict=False)
        assert v.corruption_threshold == 5
        assert v.strict is False


class TestBasicValidationAndCorruption:
    """Tests for basic syllable validation and corruption detection."""

    def test_empty_syllable(self, validator):
        """Empty strings should be invalid."""
        assert not validator.validate("")

    def test_invalid_start_chars(self, validator):
        """Syllables starting with vowel signs or medials should be invalid."""
        assert not validator.validate("\u102c")  # Aa vowel
        assert not validator.validate("\u103b")  # Ya-pin

    def test_max_length_exceeded(self, validator):
        """Syllables exceeding max length should be invalid."""
        assert not validator.validate("က" * 11)

    def test_repeated_characters(self, validator):
        """Excessive character repetition should be detected."""
        assert not validator.validate("ကကကက")

    def test_diacritic_spam(self, validator):
        """Excessive diacritic repetition should be detected."""
        assert not validator.validate("က\u103b\u103b\u103b")


class TestStartCharAndConsonants:
    """Tests for start character validation."""

    def test_consonant_and_independent_vowel_start(self, validator):
        """Consonants and independent vowels are valid start characters."""
        assert validator._check_start_char("က")
        assert validator._check_start_char("မ")
        assert validator._check_start_char("အ")
        assert validator._check_start_char("ဣ")

    def test_diacritic_start_invalid(self, validator):
        """Diacritics should not start a syllable."""
        assert not validator._check_start_char("\u102c")
        assert not validator._check_start_char("\u103b")


class TestAsatAndViramaRules:
    """Tests for asat count, virama ordering, and virama usage."""

    def test_asat_count_with_kinzi(self, validator):
        """Two asats with kinzi should pass, without kinzi should fail."""
        assert validator._check_asat_count("ကန်")
        assert validator._check_asat_count("မင်္ဂ")
        assert not validator._check_asat_count("က်န်")

    def test_virama_ordering_strict_and_lenient(self, validator, lenient_validator):
        """Virama before medial is valid; after medial fails in strict only."""
        assert validator._check_virama_ordering("က္က")
        assert not validator._check_virama_ordering("ကျ\u1039က")
        assert lenient_validator._check_virama_ordering("ကျ\u1039က")

    def test_virama_at_end_fails(self, validator):
        """Virama at end should fail; in middle should pass."""
        assert not validator._check_virama_usage("က္")
        assert validator._check_virama_usage("က္က")

    def test_virama_count(self, validator):
        """Two viramas with kinzi should pass; three should fail."""
        assert validator._check_virama_count("င်္က္ခ")
        assert not validator._check_virama_count("က္က္က္က")

    def test_asat_predecessor(self, validator):
        """Asat after consonant should pass; at start should fail."""
        assert validator._check_asat_predecessor("ကန်")
        assert not validator._check_asat_predecessor("်က")

    def test_asat_before_vowel(self, validator):
        """Asat immediately followed by vowel should fail."""
        assert not validator._check_asat_before_vowel("က်ာ")

    def test_anusvara_asat_conflict(self, validator):
        """Anusvara immediately followed by asat should fail."""
        assert not validator._check_anusvara_asat_conflict("ကံ်")


class TestVowelAndToneRules:
    """Tests for vowel exclusivity, anusvara, tone marks, and E vowel combinations."""

    def test_vowel_exclusivity(self, validator):
        """Multiple vowels in same slot should fail."""
        assert not validator._check_vowel_exclusivity("က\u102d\u102e")  # upper
        assert not validator._check_vowel_exclusivity("က\u102f\u1030")  # lower

    def test_anusvara_compatibility(self, validator):
        """Anusvara with compatible vowel should pass; incompatible should fail."""
        assert validator._check_anusvara_compatibility("ကိံ")
        assert not validator._check_anusvara_compatibility("က\u102b\u1036")

    def test_tall_a_exclusivity(self, validator):
        """Both Tall A and Aa in same syllable should fail."""
        assert not validator._check_tall_a_exclusivity("က\u102b\u102c")

    def test_tone_strictness(self, validator):
        """Two tone marks should fail."""
        assert not validator._check_tone_strictness("က\u1037\u1038")

    def test_tone_position(self, validator):
        """Dot below at end should pass; not at end should fail."""
        assert validator._check_tone_position("က့")
        assert not validator._check_tone_position("က့က")

    def test_dot_below_position(self, validator):
        """Dot below at end should pass; before vowel should fail."""
        assert validator._check_dot_below_position("က့")
        assert not validator._check_dot_below_position("က့ာ")


class TestMedialAndDiacriticRules:
    """Tests for medial compatibility, double diacritics, and zero-width chars."""

    def test_double_diacritics(self, validator):
        """Consecutive identical diacritics should fail."""
        assert not validator._check_double_diacritics("ကျျ")
        assert not validator._check_double_diacritics("က\u102c\u102c")

    def test_medial_vowel_compatibility(self, validator):
        """Valid medial-vowel combos should pass; invalid should fail."""
        assert validator._check_medial_vowel_compatibility("ကျာ")
        assert not validator._check_medial_vowel_compatibility("ကွု")

    def test_zero_width_chars(self, validator):
        """Zero-width characters should fail."""
        assert not validator._check_zero_width_chars("က\u200bာ")
        assert not validator._check_zero_width_chars("က\u200cာ")


class TestSpecialCharacterRules:
    """Tests for Great Sa, independent vowels, character scope, and stacking."""

    def test_great_sa_rules(self, validator):
        """Great Sa alone should pass; with medial or virama should fail."""
        assert validator._check_great_sa_rules("ဿ")
        assert not validator._check_great_sa_rules("ဿျ")
        assert not validator._check_great_sa_rules("ဿ္က")

    def test_independent_vowel(self, validator):
        """Independent vowel alone should pass; with medial should fail."""
        assert validator._check_independent_vowel("ဣ")
        assert not validator._check_independent_vowel("ဣျ")

    def test_character_scope(self, validator):
        """Standard Myanmar should pass; extended should fail in strict mode."""
        assert validator._check_character_scope("ကာ")
        assert not validator._check_character_scope("\u1050")

    def test_pat_sint_validity(self, validator):
        """Same-wet stacking should pass; cross-wet should fail."""
        assert validator._check_pat_sint_validity("က္က")
        assert validator._check_pat_sint_validity("က္ခ")
        assert not validator._check_pat_sint_validity("က္ပ")
        assert validator._check_pat_sint_validity("\u1010\u1039\u1010")  # exception


class TestLenientVsStrict:
    """Tests for differences between strict and lenient modes."""

    def test_lenient_vs_strict(self, validator, lenient_validator):
        """Test difference between strict and lenient modes."""
        assert validator.strict is True
        assert lenient_validator.strict is False
        assert not validator._check_character_scope("\u1050")
        assert lenient_validator._check_pat_sint_validity("\u1010\u1039\u1000")


class TestMedialBaseConsonantResolution:
    """Tests for _get_medial_base_consonant and Kinzi-aware medial compatibility."""

    def test_simple_base_consonant(self, validator):
        """Simple syllable: base is the first consonant."""
        assert validator._get_medial_base_consonant("ကျာ") == "က"

    def test_kinzi_initial_base_consonant(self, validator):
        """Kinzi-initial: base is the consonant after Kinzi sequence."""
        kinzi = "\u1004\u103a\u1039"
        assert validator._get_medial_base_consonant(kinzi + "\u1002\u102b") == "\u1002"

    def test_stacked_initial_base_consonant(self, validator):
        """Stacked-initial: base is the consonant after the last virama."""
        assert validator._get_medial_base_consonant("\u1015\u1039\u1015\u102c") == "\u1015"

    def test_kinzi_with_compatible_medial_passes(self, validator):
        """Kinzi + medial + compatible base should pass."""
        kinzi = "\u1004\u103a\u1039"
        assert validator._check_medial_compatibility(kinzi + "\u1000\u103b\u102c")

    def test_empty_and_no_consonant(self, validator):
        """Empty syllable or vowel-only should return None."""
        assert validator._get_medial_base_consonant("") is None
        assert validator._get_medial_base_consonant("\u102c\u102d") is None


class TestYaRaMedialSequenceValidation:
    """Tests for YA+RA medial sequence validation per UTN #11."""

    def test_ya_ra_sequence_valid(self, validator):
        """YA+RA sequence should be accepted and validate."""
        syllable = "\u1000\u103b\u103c\u102c"
        assert validator._check_structure_sanity(syllable)
        assert validator.validate(syllable)

    def test_valid_medial_sequences_pass(self, validator):
        """Various valid medial sequences should pass."""
        assert validator._check_structure_sanity("\u1000\u103c\u103d\u102c")  # Ra + Wa
        assert validator._check_structure_sanity("\u1000\u103b\u103d\u102c")  # Ya + Wa
        assert validator._check_structure_sanity("\u1000\u103b\u103e\u102c")  # Ya + Ha

    def test_ya_ra_wa_sequence_valid(self, validator):
        """YA+RA+WA sequence should be accepted."""
        syllable = "\u1000\u103b\u103c\u103d\u102c"
        assert validator.validate(syllable)
