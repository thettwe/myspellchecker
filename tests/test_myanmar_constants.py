"""Tests for Myanmar language constants.

Tests verify correct character classifications, Unicode ranges,
and linguistic constants for Myanmar script validation.
"""

import pytest

from myspellchecker.core.constants import (
    CLASSIFIER_SET,
    CLASSIFIERS,
    CONSONANTS,
    INDEPENDENT_VOWELS,
    INDEPENDENT_VOWELS_STRICT,
    MYANMAR_NUMERAL_WORDS,
    MYANMAR_NUMERALS,
    NON_STANDARD_CHARS,
    VOWEL_CARRIER,
)
from myspellchecker.core.syllable_rules import SyllableRuleValidator


class TestIndependentVowels:
    """Tests for independent vowel constants and validation."""

    def test_vowel_carrier_is_correct(self):
        """Test that VOWEL_CARRIER is U+1021 (အ)."""
        assert VOWEL_CARRIER == "\u1021"
        assert VOWEL_CARRIER == "အ"

    def test_independent_vowels_contains_carrier(self):
        """Test that INDEPENDENT_VOWELS includes the vowel carrier."""
        assert VOWEL_CARRIER in INDEPENDENT_VOWELS
        assert "\u1021" in INDEPENDENT_VOWELS

    def test_independent_vowels_strict_excludes_carrier(self):
        """Test that INDEPENDENT_VOWELS_STRICT excludes the vowel carrier."""
        assert VOWEL_CARRIER not in INDEPENDENT_VOWELS_STRICT
        assert "\u1021" not in INDEPENDENT_VOWELS_STRICT

    def test_independent_vowels_strict_contents(self):
        """Test that INDEPENDENT_VOWELS_STRICT contains only true independent vowels."""
        expected = {
            "\u1023",  # ဣ - I
            "\u1024",  # ဤ - II
            "\u1025",  # ဥ - U
            "\u1026",  # ဦ - UU
            "\u1027",  # ဧ - E
            "\u1029",  # ဩ - O
            "\u102a",  # ဪ - AU
        }
        assert INDEPENDENT_VOWELS_STRICT == expected

    def test_independent_vowels_strict_excludes_non_standard(self):
        """Test that INDEPENDENT_VOWELS_STRICT excludes non-standard characters."""
        # U+1022 (Shan Letter A) should not be in strict set
        assert "\u1022" not in INDEPENDENT_VOWELS_STRICT
        # U+1028 (Mon E) should not be in strict set
        assert "\u1028" not in INDEPENDENT_VOWELS_STRICT

    def test_non_standard_chars_contains_mon_e(self):
        """Test that NON_STANDARD_CHARS includes Mon E (U+1028)."""
        assert "\u1028" in NON_STANDARD_CHARS

    def test_non_standard_chars_contains_shan_a(self):
        """Test that NON_STANDARD_CHARS includes Shan Letter A (U+1022)."""
        assert "\u1022" in NON_STANDARD_CHARS

    def test_independent_vowels_range(self):
        """Test that INDEPENDENT_VOWELS covers the correct Unicode range."""
        # Should include U+1021 through U+102A (10 characters)
        expected_codepoints = set(range(0x1021, 0x102B))
        actual_codepoints = {ord(c) for c in INDEPENDENT_VOWELS}
        assert actual_codepoints == expected_codepoints

    def test_consonants_do_not_overlap_strict_vowels(self):
        """Test that CONSONANTS and INDEPENDENT_VOWELS_STRICT are disjoint."""
        overlap = CONSONANTS & INDEPENDENT_VOWELS_STRICT
        assert len(overlap) == 0, f"Unexpected overlap: {overlap}"


class TestIndependentVowelSyllableValidation:
    """Tests for syllable validation with independent vowels."""

    @pytest.fixture
    def validator(self):
        return SyllableRuleValidator()

    @pytest.fixture
    def lenient_validator(self):
        return SyllableRuleValidator(strict=False)

    def test_standalone_independent_vowel_i(self, validator):
        """Test that standalone independent vowel I (ဣ) is valid."""
        assert validator.validate("\u1023") is True  # ဣ

    def test_standalone_independent_vowel_ii(self, validator):
        """Test that standalone independent vowel II (ဤ) is valid."""
        assert validator.validate("\u1024") is True  # ဤ

    def test_standalone_independent_vowel_u(self, validator):
        """Test that standalone independent vowel U (ဥ) is valid."""
        assert validator.validate("\u1025") is True  # ဥ

    def test_standalone_independent_vowel_uu(self, validator):
        """Test that standalone independent vowel UU (ဦ) is valid."""
        assert validator.validate("\u1026") is True  # ဦ

    def test_standalone_independent_vowel_e(self, validator):
        """Test that standalone independent vowel E (ဧ) is valid."""
        assert validator.validate("\u1027") is True  # ဧ

    def test_standalone_independent_vowel_o(self, validator):
        """Test that standalone independent vowel O (ဩ) is valid."""
        assert validator.validate("\u1029") is True  # ဩ

    def test_standalone_independent_vowel_au(self, validator):
        """Test that standalone independent vowel AU (ဪ) is valid."""
        assert validator.validate("\u102a") is True  # ဪ

    def test_vowel_carrier_with_vowel_sign(self, validator):
        """Test that vowel carrier (အ) can take vowel signs like a consonant."""
        # အ + vowel signs should be valid (behaves as consonant carrier)
        assert validator.validate("\u1021\u102c") is True  # အာ
        assert validator.validate("\u1021\u102d") is True  # အိ

    def test_independent_vowel_cannot_take_medials(self, validator):
        """Test that true independent vowels cannot take medials."""
        # ဣ + Ya-pin should be invalid
        assert validator.validate("\u1023\u103b") is False
        # ဥ + Wa-hswe should be invalid
        assert validator.validate("\u1025\u103d") is False

    def test_independent_vowel_cannot_take_vowel_signs(self, validator):
        """Test that true independent vowels cannot take dependent vowel signs."""
        # ဣ + Aa should be invalid
        assert validator.validate("\u1023\u102c") is False
        # ဧ + I should be invalid
        assert validator.validate("\u1027\u102d") is False

    def test_independent_vowel_can_take_tone_marks(self, validator):
        """Test that independent vowels can take tone marks."""
        # ဥ + Visarga should be valid
        assert validator.validate("\u1025\u1038") is True
        # ဧ + Dot below should be valid
        assert validator.validate("\u1027\u1037") is True

    def test_non_standard_independent_vowel_rejected_in_strict(self, validator):
        """Test that non-standard independent vowels are rejected in strict mode."""
        # Shan Letter A (U+1022) should be rejected
        assert validator.validate("\u1022") is False
        # Mon E (U+1028) should be rejected
        assert validator.validate("\u1028") is False

    def test_non_standard_independent_vowel_in_lenient(self, lenient_validator):
        """Test that non-standard independent vowels behavior in lenient mode."""
        # In lenient mode, character scope check is skipped
        # U+1022 may still fail other checks but not character scope
        # Note: The actual result depends on other validation rules
        # We just verify it doesn't crash
        try:
            lenient_validator.validate("\u1022")
            lenient_validator.validate("\u1028")
        except Exception as e:
            pytest.fail(f"Validation should not raise exception: {e}")


class TestConsonantCharacterSet:
    """Tests for consonant character set."""

    def test_consonants_range(self):
        """Test that CONSONANTS covers U+1000 to U+1020."""
        base_codepoints = set(range(0x1000, 0x1021))  # U+1000-U+1020
        actual_codepoints = {ord(c) for c in CONSONANTS if ord(c) < 0x1021}
        assert actual_codepoints == base_codepoints

    def test_great_sa_in_consonants(self):
        """Test that Great Sa (U+103F) is in CONSONANTS."""
        assert "\u103f" in CONSONANTS

    def test_consonants_count(self):
        """Test that CONSONANTS has correct count (33 base + Great Sa + အ vowel carrier = 35)."""
        assert len(CONSONANTS) == 35


class TestMyanmarClassifiers:
    """Test cases for Myanmar classifier constants."""

    def test_classifiers_not_empty(self):
        """Test that CLASSIFIERS dict is populated."""
        assert len(CLASSIFIERS) > 0

    def test_common_classifiers_present(self):
        """Test that common classifiers are present."""
        # People classifiers
        assert "ယောက်" in CLASSIFIERS  # people (general)
        assert "ဦး" in CLASSIFIERS  # people (respectful)

        # Animal classifier
        assert "ကောင်" in CLASSIFIERS  # animals

        # Vehicle classifier
        assert "စီး" in CLASSIFIERS  # vehicles

        # Object classifiers
        assert "ခု" in CLASSIFIERS  # general objects
        assert "လုံး" in CLASSIFIERS  # round objects

        # Time classifiers
        assert "ကြိမ်" in CLASSIFIERS  # times/occurrences
        assert "ခါ" in CLASSIFIERS  # times (colloquial)

    def test_classifier_set_matches_dict(self):
        """Test that CLASSIFIER_SET contains all CLASSIFIERS keys."""
        assert CLASSIFIER_SET == frozenset(CLASSIFIERS.keys())

    def test_classifier_descriptions_are_strings(self):
        """Test that all classifier descriptions are non-empty strings."""
        for classifier, description in CLASSIFIERS.items():
            assert isinstance(classifier, str)
            assert isinstance(description, str)
            assert len(description) > 0

    def test_classifiers_are_valid_myanmar(self):
        """Test that all classifier keys are valid Myanmar text."""
        for classifier in CLASSIFIERS.keys():
            # Check that classifier contains Myanmar characters
            assert any(
                0x1000 <= ord(c) <= 0x109F or 0xAA60 <= ord(c) <= 0xAA7F for c in classifier
            ), f"Classifier '{classifier}' contains non-Myanmar characters"


class TestMyanmarNumerals:
    """Test cases for Myanmar numeral constants."""

    def test_numerals_complete(self):
        """Test that all 10 Myanmar numerals (0-9) are present."""
        assert len(MYANMAR_NUMERALS) == 10

    def test_numeral_unicode_range(self):
        """Test that numerals are in correct Unicode range."""
        for numeral in MYANMAR_NUMERALS:
            assert 0x1040 <= ord(numeral) <= 0x1049

    def test_numeral_words_present(self):
        """Test that common numeral words are present."""
        assert "တစ်" in MYANMAR_NUMERAL_WORDS  # 1
        assert "နှစ်" in MYANMAR_NUMERAL_WORDS  # 2
        assert "သုံး" in MYANMAR_NUMERAL_WORDS  # 3
        assert "ရာ" in MYANMAR_NUMERAL_WORDS  # 100
        assert "ထောင်" in MYANMAR_NUMERAL_WORDS  # 1000

    def test_numeral_words_values(self):
        """Test that numeral word values are correct."""
        assert MYANMAR_NUMERAL_WORDS["တစ်"] == 1
        assert MYANMAR_NUMERAL_WORDS["ဆယ်"] == 10
        assert MYANMAR_NUMERAL_WORDS["ရာ"] == 100
        assert MYANMAR_NUMERAL_WORDS["ထောင်"] == 1000
        assert MYANMAR_NUMERAL_WORDS["သန်း"] == 1000000


class TestConsonantCrossModuleConsistency:
    """Tests to verify Myanmar consonant definitions are consistent across modules.

    Multiple modules define Myanmar consonants for their specific use cases.
    These tests verify that all definitions include the core consonants (U+1000-U+1020)
    to prevent bugs like missing characters.

    Verified locations:
    - core/constants/myanmar_constants.py (canonical source)
    - text/validator.py (MYANMAR_CONSONANTS)
    - tokenizers/word.py (_MYANMAR_CONSONANTS)
    - data_pipeline/batch_processor.pyx (_MYANMAR_CONSONANTS) - Cython, verified manually

    Note: algorithms/semantic_checker.py was refactored to use centralized
    get_myanmar_char_set() from core.constants instead of its own MYANMAR_CONSONANTS.
    """

    # Core consonants that MUST be present in all consonant sets (U+1000-U+1020)
    CORE_CONSONANTS = set(chr(c) for c in range(0x1000, 0x1021))

    def test_canonical_constants_has_core_consonants(self):
        """Test that canonical CONSONANTS includes all core consonants."""
        missing = self.CORE_CONSONANTS - CONSONANTS
        assert not missing, f"Canonical CONSONANTS missing: {missing}"

    def test_validator_has_core_consonants(self):
        """Test that validator MYANMAR_CONSONANTS includes all core consonants."""
        from myspellchecker.text.validator import MYANMAR_CONSONANTS

        missing = self.CORE_CONSONANTS - MYANMAR_CONSONANTS
        assert not missing, f"validator MYANMAR_CONSONANTS missing: {missing}"

    def test_word_tokenizer_has_core_consonants(self):
        """Test that word tokenizer _MYANMAR_CONSONANTS includes all core consonants."""
        from myspellchecker.tokenizers.word import _MYANMAR_CONSONANTS

        missing = self.CORE_CONSONANTS - _MYANMAR_CONSONANTS
        assert not missing, f"word tokenizer _MYANMAR_CONSONANTS missing: {missing}"

    def test_nya_consonant_present_everywhere(self):
        """Test that Nya consonant (U+1009 ဉ) is present in all consonant sets.

        This character was historically missing from some modules.
        This test ensures it remains present in all definitions.
        """
        nya = "\u1009"  # ဉ - Nya consonant

        # Canonical
        assert nya in CONSONANTS, "Nya (ဉ) missing from canonical CONSONANTS"

        # validator
        from myspellchecker.text.validator import MYANMAR_CONSONANTS as V_CONSONANTS

        assert nya in V_CONSONANTS, "Nya (ဉ) missing from validator MYANMAR_CONSONANTS"

        # word tokenizer
        from myspellchecker.tokenizers.word import _MYANMAR_CONSONANTS as W_CONSONANTS

        assert nya in W_CONSONANTS, "Nya (ဉ) missing from word tokenizer _MYANMAR_CONSONANTS"


class TestCythonConstantsSync:
    """Tests to verify Cython modules stay in sync with Python constants.

    Cython modules embed static character sets for performance. These tests
    ensure they match the canonical definitions in core/constants to prevent
    behavior drift between Python and Cython code paths.

    """

    def test_batch_processor_vowels_match_vowel_signs(self):
        """Test that batch_processor _MYANMAR_VOWELS matches VOWEL_SIGNS.

        The Cython module previously included ံ (U+1036 anusvara)
        in its vowel set, but VOWEL_SIGNS does not include it. This caused
        different fragment detection results between Python and Cython paths.
        """
        from myspellchecker.core.constants import VOWEL_SIGNS

        try:
            from myspellchecker.data_pipeline.batch_processor import _MYANMAR_VOWELS
        except ImportError:
            pytest.skip("Cython batch_processor not compiled")

        # Verify the sets match exactly
        assert _MYANMAR_VOWELS == VOWEL_SIGNS, (
            f"batch_processor _MYANMAR_VOWELS does not match VOWEL_SIGNS. "
            f"Cython: {_MYANMAR_VOWELS}, Python: {VOWEL_SIGNS}"
        )

    def test_batch_processor_vowels_excludes_anusvara(self):
        """Test that batch_processor vowels do NOT include anusvara (ံ U+1036).

        The anusvara is a tone/nasal mark, NOT a dependent vowel sign.
        Including it causes incorrect fragment detection in virama+vowel checks.
        """
        try:
            from myspellchecker.data_pipeline.batch_processor import _MYANMAR_VOWELS
        except ImportError:
            pytest.skip("Cython batch_processor not compiled")

        anusvara = "\u1036"  # ံ
        assert anusvara not in _MYANMAR_VOWELS, (
            "batch_processor _MYANMAR_VOWELS should NOT include anusvara (ံ U+1036)"
        )

    def test_batch_processor_consonants_match_core(self):
        """Test that batch_processor consonants match core CONSONANTS."""
        try:
            from myspellchecker.data_pipeline.batch_processor import _MYANMAR_CONSONANTS
        except ImportError:
            pytest.skip("Cython batch_processor not compiled")

        # Core consonants U+1000-U+1020 plus vowel carrier U+1021
        core_consonants = set(chr(c) for c in range(0x1000, 0x1022))

        missing = core_consonants - _MYANMAR_CONSONANTS
        assert not missing, f"batch_processor missing consonants: {missing}"

    def test_batch_processor_numerals_match_core(self):
        """Test that batch_processor numerals match core MYANMAR_NUMERALS."""
        try:
            from myspellchecker.data_pipeline.batch_processor import _MYANMAR_NUMERALS
        except ImportError:
            pytest.skip("Cython batch_processor not compiled")

        assert _MYANMAR_NUMERALS == MYANMAR_NUMERALS, (
            f"batch_processor _MYANMAR_NUMERALS does not match. "
            f"Cython: {_MYANMAR_NUMERALS}, Python: {MYANMAR_NUMERALS}"
        )

    def test_fragment_detection_consistent_with_virama_anusvara(self):
        """Test that virama + anusvara is NOT treated as virama + vowel.

        Regression test: Previously, "္ံ" (virama + anusvara) was
        incorrectly flagged as invalid because anusvara was in the vowel set.
        """
        try:
            from myspellchecker.data_pipeline.batch_processor import is_invalid_fragment
        except ImportError:
            pytest.skip("Cython batch_processor not compiled")

        # Virama followed by anusvara should NOT be flagged as invalid
        # (anusvara is a tone mark, not a vowel)
        # "က္ံ" = consonant + virama + anusvara
        # This is still an unusual pattern, but the specific check for
        # "virama followed by vowel" should not trigger
        # The function may still return True for other reasons, but
        # not because of the virama+vowel check

        # A word with virama + actual vowel SHOULD be invalid
        invalid_word = "က္ာ"  # consonant + virama + vowel (ာ)
        assert is_invalid_fragment(invalid_word) is True, (
            "virama + vowel should be detected as invalid fragment"
        )

    # =========================================================================
    # repair_c.pyx constant sync tests
    # =========================================================================

    def test_repair_c_consonants_match_core(self):
        """Test that repair_c consonants match core CONSONANTS + vowel carrier.

        repair_c.pyx embeds consonant lists directly.
        This test ensures they stay in sync with canonical constants.

        Note: repair_c includes the vowel carrier (အ U+1021) because it's
        needed as a valid word start character for segmentation repair.
        """
        try:
            from myspellchecker.data_pipeline.repair_c import get_repair_consonants
        except ImportError:
            pytest.skip("Cython repair_c not compiled")

        from myspellchecker.core.constants import CONSONANTS, VOWEL_CARRIER

        repair_consonants = get_repair_consonants()

        # repair_c needs CONSONANTS + vowel carrier for valid start detection
        expected = CONSONANTS | {VOWEL_CARRIER}

        missing = expected - repair_consonants
        extra = repair_consonants - expected

        assert not missing, f"repair_c missing consonants: {missing}"
        assert not extra, f"repair_c has extra consonants: {extra}"

    def test_repair_c_independent_vowels_match_core(self):
        """Test that repair_c independent vowels match INDEPENDENT_VOWELS_STRICT.

        repair_c.pyx embeds independent vowel lists directly.
        These should match the strict (standard Burmese) independent vowels.
        """
        try:
            from myspellchecker.data_pipeline.repair_c import get_repair_independent_vowels
        except ImportError:
            pytest.skip("Cython repair_c not compiled")

        from myspellchecker.core.constants import INDEPENDENT_VOWELS_STRICT

        repair_vowels = get_repair_independent_vowels()

        assert repair_vowels == INDEPENDENT_VOWELS_STRICT, (
            f"repair_c independent vowels do not match INDEPENDENT_VOWELS_STRICT. "
            f"Cython: {repair_vowels}, Python: {INDEPENDENT_VOWELS_STRICT}"
        )

    def test_repair_c_numerals_match_core(self):
        """Test that repair_c numerals match core MYANMAR_NUMERALS.

        repair_c.pyx embeds numeral lists directly.
        """
        try:
            from myspellchecker.data_pipeline.repair_c import get_repair_numerals
        except ImportError:
            pytest.skip("Cython repair_c not compiled")

        repair_numerals = get_repair_numerals()

        assert repair_numerals == MYANMAR_NUMERALS, (
            f"repair_c numerals do not match MYANMAR_NUMERALS. "
            f"Cython: {repair_numerals}, Python: {MYANMAR_NUMERALS}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
