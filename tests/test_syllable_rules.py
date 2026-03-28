import pytest

from myspellchecker.core.constants import GREAT_SA, VIRAMA
from myspellchecker.core.syllable_rules import SyllableRuleValidator


class TestSyllableRulesExtended:
    @pytest.fixture
    def validator(self):
        return SyllableRuleValidator()

    def test_thwa_validation(self, validator):
        # Thwa (Go/Tooth) = Tha + Wa + Aa + Visarga
        # \u101e \u103d \u102c \u1038
        thwa = "\u101e\u103d\u102c\u1038"
        assert validator.validate(thwa) is True

    def test_kinzi_validation(self, validator):
        # Valid Kinzi: Nga + Asat + Virama + Ka (e.g. in Mingala)
        valid_kinzi = "\u1004\u103a\u1039\u1000"
        assert validator.validate(valid_kinzi) is True

        # Invalid: Kinzi at end (no consonant after)
        invalid_kinzi_end = "\u1000\u1004\u103a\u1039"
        assert validator.validate(invalid_kinzi_end) is False

        # Invalid: Kinzi followed by vowel
        invalid_kinzi_vowel = "\u1004\u103a\u1039\u102c"
        assert validator.validate(invalid_kinzi_vowel) is False

    @pytest.mark.parametrize(
        "syllable",
        ["\u1000\u103d", "\u101c\u103d", "\u1014\u103d"],
        ids=["ka_wa", "la_wa", "na_wa"],
    )
    def test_wa_medial_compatibility_valid(self, validator, syllable):
        assert validator.validate(syllable) is True

    @pytest.mark.parametrize(
        "syllable,expected",
        [
            ("\u1000\u102d\u102e", False),  # I + II (mutually exclusive)
            ("\u1000\u102f\u1030", False),  # U + UU (mutually exclusive)
            ("\u1000\u102d", True),  # Single vowel I (valid)
        ],
        ids=["i_plus_ii", "u_plus_uu", "single_i"],
    )
    def test_vowel_exclusivity(self, validator, syllable, expected):
        assert validator.validate(syllable) is expected

    def test_great_sa_rules(self, validator):
        # Valid Great Sa
        assert validator.validate(GREAT_SA) is True

        # Invalid: Great Sa + Medial Ya
        assert validator.validate(f"{GREAT_SA}\u103b") is False

        # Invalid: Great Sa + Virama
        assert validator.validate(f"{GREAT_SA}{VIRAMA}") is False

    @pytest.mark.parametrize(
        "syllable,expected",
        [
            ("\u1000\u1031\u102c", True),  # E + Aa (Aw) — valid combination
            ("\u1000\u1031\u102d", False),  # E + I — invalid
            ("\u1000\u1031\u102f", False),  # E + U — invalid
        ],
        ids=["e_aa_valid", "e_i_invalid", "e_u_invalid"],
    )
    def test_e_vowel_combinations(self, validator, syllable, expected):
        assert validator.validate(syllable) is expected

    @pytest.mark.parametrize(
        "syllable,expected",
        [
            ("\u1000\u1036", True),  # Ka + Anusvara (Kam)
            ("\u1000\u102d\u1036", True),  # Ka + I + Anusvara (Kim)
            ("\u1000\u102f\u1036", True),  # Ka + U + Anusvara (Kum)
            ("\u1000\u102c\u1036", False),  # Ka + Aa + Anusvara
            ("\u1000\u102b\u1036", False),  # Ka + Tall A + Anusvara
            ("\u1000\u1031\u1036", False),  # Ka + E + Anusvara
            ("\u1000\u102e\u1036", False),  # Ka + II + Anusvara
        ],
        ids=[
            "base_anusvara",
            "base_i_anusvara",
            "base_u_anusvara",
            "base_aa_anusvara_invalid",
            "base_tall_a_anusvara_invalid",
            "base_e_anusvara_invalid",
            "base_ii_anusvara_invalid",
        ],
    )
    def test_anusvara_compatibility(self, validator, syllable, expected):
        assert validator.validate(syllable) is expected

    def test_virama_ordering_and_operands(self, validator):
        # Valid: Stacked Consonants (Kkha)
        # Ka + Virama + Kha
        assert validator.validate("\u1000\u1039\u1001") is True

        # Valid: Stacked + Medial (Kkhya)
        # Ka + Virama + Kha + Ya
        assert validator.validate("\u1000\u1039\u1001\u103b") is True

        # Invalid: Medial before Virama (Ka + Ya + Virama + Ka)
        # Trying to stack under the medial
        assert validator.validate("\u1000\u103b\u1039\u1000") is False

        # Invalid: Stacking Non-Consonants
        # Ka + Virama + Ya(Medial)
        assert validator.validate("\u1000\u1039\u103b") is False

    def test_asat_count(self, validator):
        # Valid: 1 Asat (Final)
        assert validator.validate("\u1000\u1000\u103a") is True

        # Valid: 2 Asats (Kinzi + Final)
        # Ming (Nga+Asat+Virama) + Ga + La + ...
        # Let's construct: Ma + (Kinzi) + Ga + Ta + Asat (Mingu-gat)
        ming_gat = "\u1019\u1004\u103a\u1039\u1002\u1010\u103a"
        assert validator.validate(ming_gat) is True

        # Invalid: 2 Asats (No Kinzi)
        # Ka + Asat + Ka + Asat
        assert validator.validate("\u1000\u103a\u1000\u103a") is False

        # Invalid: 3 Asats
        assert validator.validate("\u1000\u103a\u1000\u103a\u103a") is False


class TestPhase1Improvements:
    """Tests for Phase 1 improvements: Encoding & Exclusivity Rules."""

    @pytest.fixture
    def validator(self):
        return SyllableRuleValidator()

    @pytest.mark.parametrize(
        "syllable,expected",
        [
            ("\u1000\u102c", True),  # Ka + Aa (normal, valid)
            ("\u1000\u200b\u102c", False),  # Ka + ZWSP + Aa
            ("\u1000\u200c\u102c", False),  # Ka + ZWNJ + Aa
            ("\u1000\u200d\u102c", False),  # Ka + ZWJ + Aa
            ("\ufeff\u1000\u102c", False),  # BOM + Ka + Aa
        ],
        ids=["normal_valid", "zwsp", "zwnj", "zwj", "bom"],
    )
    def test_zero_width_char_rejection(self, validator, syllable, expected):
        """Zero-width characters should be rejected."""
        assert validator.validate(syllable) is expected

    @pytest.mark.parametrize(
        "syllable,expected",
        [
            ("\u1000\u1031\u102b", True),  # Ka + E + Tall A (Kaw) — valid
            ("\u1000\u102c", True),  # Ka + Aa (Kaa) — valid
            ("\u1000\u102b\u102c", False),  # Ka + Tall A + Aa — invalid (both present)
            ("\u1000\u102c\u102b", False),  # Ka + Aa + Tall A — invalid (both, reverse)
        ],
        ids=["tall_a_only", "aa_only", "both_tall_a_aa", "both_aa_tall_a"],
    )
    def test_tall_a_aa_exclusivity(self, validator, syllable, expected):
        """Tall A (102B) and Aa (102C) are mutually exclusive."""
        assert validator.validate(syllable) is expected


class TestPhase2Improvements:
    """Tests for Phase 2 improvements: Phonotactic Rules."""

    @pytest.fixture
    def validator(self):
        return SyllableRuleValidator()

    @pytest.mark.parametrize(
        "syllable,expected",
        [
            ("\u1000\u102c\u1037", True),  # Ka + Aa + Dot Below (at end)
            ("\u1000\u1037", True),  # Ka + Dot Below (at end)
            ("\u1000\u1037\u102c", False),  # Ka + Dot Below + Aa (vowel after)
            ("\u1000\u1037\u103d", False),  # Ka + Dot Below + Wa (medial after)
            ("\u1000\u1037\u1036", False),  # Ka + Dot Below + Anusvara
        ],
        ids=[
            "dot_below_after_vowel",
            "dot_below_at_end",
            "dot_below_before_vowel",
            "dot_below_before_medial",
            "dot_below_before_anusvara",
        ],
    )
    def test_dot_below_position(self, validator, syllable, expected):
        """Dot below should not be followed by vowels, medials, or anusvara."""
        assert validator.validate(syllable) is expected

    @pytest.mark.parametrize(
        "syllable,expected",
        [
            ("\u1000\u1039\u1000", True),  # Single virama (Ka + Virama + Ka)
            ("\u1004\u103a\u1039\u1000\u1039\u1000", True),  # Two viramas with Kinzi
            ("\u1000\u1039\u1000\u1039\u1000", False),  # Two viramas without Kinzi
            ("\u1000\u1039\u1000\u1039\u1000\u1039\u1000", False),  # Three viramas
        ],
        ids=[
            "single_virama",
            "double_virama_with_kinzi",
            "double_virama_no_kinzi",
            "triple_virama",
        ],
    )
    def test_virama_count(self, validator, syllable, expected):
        """Maximum virama count should be enforced."""
        assert validator.validate(syllable) is expected

    @pytest.mark.parametrize(
        "syllable,expected",
        [
            ("\u1000\u1036", True),  # Ka + Anusvara alone
            ("\u1000\u1014\u103a", True),  # Ka + Na + Asat (final)
            ("\u1000\u1036\u103a", False),  # Ka + Anusvara + Asat (conflict)
        ],
        ids=["anusvara_alone", "asat_alone", "anusvara_then_asat"],
    )
    def test_anusvara_asat_conflict(self, validator, syllable, expected):
        """Anusvara cannot be immediately followed by Asat."""
        assert validator.validate(syllable) is expected


class TestBasicValidation:
    """Tests for basic syllable validation (consonants, vowels, diacritics)."""

    @pytest.fixture
    def validator(self):
        return SyllableRuleValidator()

    def test_basic_consonants(self, validator):
        """Bare consonants are valid syllables."""
        assert validator.validate("က") is True
        assert validator.validate("ခ") is True

    def test_independent_vowels(self, validator):
        """Independent vowels are valid syllables."""
        assert validator.validate("ဣ") is True
        assert validator.validate("ဤ") is True

    def test_vowels_and_diacritics(self, validator):
        """Consonant + vowel diacritics are valid."""
        assert validator.validate("ကိ") is True
        assert validator.validate("ကေ") is True

    def test_asat_and_tones(self, validator):
        """Asat and tone marks are valid in correct positions."""
        assert validator.validate("ကင်") is True
        assert validator.validate("ကင်း") is True
        # ဆင့် (ဆ + င + ် + ့) - correctly ordered
        assert validator.validate("\u1006\u1004\u103a\u1037") is True

    def test_invalid_structures(self, validator):
        """Double diacritics and excessive length are rejected."""
        assert validator.validate("ကိိ") is False
        assert validator.validate("က" * 20) is False

    def test_stacking_exceptions(self, validator):
        """Valid stacking exceptions (e.g., န္တ) are accepted."""
        assert validator.validate("န္တ") is True

    def test_ha_htoe_medial_strict_vs_lenient(self):
        """Ha-htoe medial requires sonorant consonant in strict mode."""
        strict_validator = SyllableRuleValidator(strict=True)
        lenient_validator = SyllableRuleValidator(strict=False)

        # Ma + Ha-htoe: valid in strict (Ma is a sonorant)
        assert strict_validator.validate("မှ") is True
        # Ka + Ha-htoe: invalid in strict (Ka is a stop)
        assert strict_validator.validate("ကှ") is False
        # Ka + Ha-htoe: valid in lenient
        assert lenient_validator.validate("ကှ") is True


class TestAsatUsageRules:
    """Tests for invalid asat usage patterns."""

    @pytest.fixture
    def validator(self):
        return SyllableRuleValidator()

    def test_invalid_asat_after_vowel(self, validator):
        """Asat after vowel signs is invalid."""
        assert validator.validate("နူ်း") is False  # U + Asat
        assert validator.validate("ကာ်") is False  # Aa + Asat

    def test_valid_asat_after_consonant(self, validator):
        """Asat after consonant is valid (final consonant)."""
        assert validator.validate("ကန်") is True
        assert validator.validate("မွန်") is True

    def test_invalid_asat_after_medial(self, validator):
        """Asat directly after medial is invalid."""
        assert validator.validate("ကျ်") is False

    def test_invalid_asat_after_tone(self, validator):
        """Asat after tone mark is invalid."""
        assert validator.validate("ကး်") is False

    def test_corruption_case(self, validator):
        """Corrupted syllable with multiple invalid markers is rejected."""
        assert validator.validate("မူန်ူး") is False


class TestStrictnessMode:
    """Tests for strict vs lenient validation modes."""

    def test_kinzi_strictness(self):
        """Strict mode rejects kinzi with invalid follower consonant."""
        # Kinzi pattern: Nga + Asat + Virama
        # Follower: Nya (U+100A) - Valid consonant, but NOT in KINZI_VALID_FOLLOWERS list.
        kinzi_syllable = "\u1004\u103a\u1039" + "\u100a" + "\u102c"  # Nga+Asat+Virama + Nya + Aa

        strict_validator = SyllableRuleValidator(strict=True)
        assert not strict_validator.validate(kinzi_syllable), (
            "Strict validator should reject invalid Kinzi follower"
        )

        lenient_validator = SyllableRuleValidator(strict=False)
        assert lenient_validator.validate(kinzi_syllable), (
            "Lenient validator should accept invalid Kinzi follower"
        )

    def test_pat_sint_strictness(self):
        """Strict mode rejects invalid pat sint (consonant stacking) combinations."""
        # Invalid stack: Ka (1000) stacked on Pa (1015) -> Ka + Virama + Pa
        # Row 1 (Ka) cannot stack on Row 5 (Pa) in standard Pali rules.
        invalid_stack = "\u1000\u1039\u1015"  # Ka + Virama + Pa

        strict_validator = SyllableRuleValidator(strict=True)
        assert not strict_validator.validate(invalid_stack), (
            "Strict validator should reject invalid Pat Sint stack"
        )

        lenient_validator = SyllableRuleValidator(strict=False)
        assert lenient_validator.validate(invalid_stack), (
            "Lenient validator should accept invalid Pat Sint stack"
        )


class TestStrictnessValidation:
    """Tests for strict-mode syllable/word validation (standard Burmese only)."""

    def test_strict_reject_non_standard_chars(self):
        validator = SyllableRuleValidator(strict=True)
        # Mon II (U+1033)
        assert not validator.validate("စုဳ"), "Should reject Mon II (U+1033)"
        # Mon O (U+1034)
        assert not validator.validate("ပဴ"), "Should reject Mon O (U+1034)"
        # Shan E Above (U+1035)
        assert not validator.validate("ဟန်ဵ"), "Should reject E Above (U+1035)"
        # Shan A (U+1022)
        assert not validator.validate("အၒ"), "Should reject Shan A (U+1022)"

    def test_strict_reject_double_tones(self):
        validator = SyllableRuleValidator(strict=True)
        assert not validator.validate("ဦးး"), "Should reject Double Visarga"
        assert not validator.validate("ဧည့့်"), "Should reject Double Dot"

    def test_strict_reject_tone_conflicts(self):
        validator = SyllableRuleValidator(strict=True)
        assert not validator.validate("ဦ့း"), "Should reject Dot + Visarga"

    def test_strict_reject_non_standard_finals(self):
        validator = SyllableRuleValidator(strict=True)
        # Ah + Asat (U+1021 + U+103A)
        assert not validator.validate("နအ်"), "Should reject Ah + Asat"

    def test_strict_accept_valid_standard_burmese(self):
        validator = SyllableRuleValidator(strict=True)
        assert validator.validate("မြန်"), "Should accept Standard Burmese (Myan)"
        assert validator.validate("မာ"), "Should accept Standard Burmese (Mar)"
        assert validator.validate("\u1000\u103c\u100a\u1037"), "Should accept Nya + Dot (Look)"
        assert validator.validate("၏"), "Should accept E + Asat (Genitive)"

    def test_word_validator_rejects_non_standard(self):
        from myspellchecker.text.validator import validate_word

        assert not validate_word("စုဳ"), "Should reject Mon II (U+1033)"
        assert not validate_word("ဟန်ဵ"), "Should reject E Above (U+1035)"

    def test_word_validator_rejects_double_tones(self):
        from myspellchecker.text.validator import validate_word

        assert not validate_word("ဦးး"), "Should reject Double Visarga"
        assert not validate_word("ဧည့့်"), "Should reject Double Dot"
        assert not validate_word("ဦ့း"), "Should reject Dot + Visarga"

    def test_word_validator_rejects_vowel_asat(self):
        from myspellchecker.text.validator import validate_word

        assert not validate_word("မို်း"), "Should reject Vowel + Asat"

    def test_word_validator_accepts_valid(self):
        from myspellchecker.text.validator import validate_word

        assert validate_word("မြန်မာ"), "Should accept Standard Burmese"
        assert validate_word("ကြည့်"), "Should accept Nya + Dot"
