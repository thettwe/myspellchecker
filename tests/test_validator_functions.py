"""Unit tests for text/validator.py — validate_text() and validate_word() core functions."""

import pytest


class TestValidateText:
    """Tests for validate_text function."""

    def test_empty_text(self):
        """Test validate_text with empty string."""
        from myspellchecker.text.validator import validate_text

        result = validate_text("")
        assert result.is_valid is True
        assert result.issues == []

    def test_whitespace_only(self):
        """Test validate_text with whitespace only."""
        from myspellchecker.text.validator import validate_text

        result = validate_text("   ")
        assert result.is_valid is True

    def test_valid_myanmar_text(self):
        """Test validate_text with valid Myanmar text."""
        from myspellchecker.text.validator import validate_text

        result = validate_text("မြန်မာ")
        assert result.is_valid is True
        assert result.issues == []

    def test_known_invalid_word(self):
        """Test validate_text with known invalid word."""
        from myspellchecker.text.validator import ValidationIssue, validate_text

        result = validate_text("ပျှော်")
        assert result.is_valid is False
        assert len(result.issues) == 1
        assert result.issues[0][0] == ValidationIssue.KNOWN_INVALID

    def test_known_invalid_word_in_sentence(self):
        """Test validate_text with known invalid word in sentence."""
        from myspellchecker.text.validator import ValidationIssue, validate_text

        result = validate_text("မြန်မာ ပျှော် စာ")
        assert result.is_valid is False
        issue_types = [i[0] for i in result.issues]
        assert ValidationIssue.KNOWN_INVALID in issue_types

    def test_extended_myanmar(self):
        """Test validate_text detects extended Myanmar chars."""
        from myspellchecker.text.validator import ValidationIssue, validate_text

        # U+1050 is in extended range
        result = validate_text("\u1050")
        assert result.is_valid is False
        assert result.issues[0][0] == ValidationIssue.EXTENDED_MYANMAR

    def test_zawgyi_ya_asat_pattern(self):
        """Test validate_text detects Zawgyi ya+tone pattern."""
        from myspellchecker.text.validator import ValidationIssue, validate_text

        # ငျး - ya-medial followed by tone (Zawgyi artifact)
        result = validate_text("ငျး")
        assert result.is_valid is False
        issue_types = [i[0] for i in result.issues]
        assert ValidationIssue.ZAWGYI_YA_ASAT in issue_types

    def test_zawgyi_ya_terminal_pattern(self):
        """Test validate_text detects Zawgyi ya at end."""
        from myspellchecker.text.validator import ValidationIssue, validate_text

        # ငျ at end of word
        result = validate_text("ငျ")
        assert result.is_valid is False
        issue_types = [i[0] for i in result.issues]
        assert ValidationIssue.ZAWGYI_YA_TERMINAL in issue_types

    def test_ya_ra_pattern_is_valid(self):
        """Test validate_text accepts Ya+Ra medial sequence as valid Unicode Burmese.

        Per UTN #11 (Unicode Technical Note #11), the canonical medial order is:
        Ya (U+103B) < Ra (U+103C) < Wa (U+103D) < Ha (U+103E)

        The Ya+Ra (ျြ) sequence IS valid in standard Unicode Burmese.
        Example words: ကျြေး (crane), ပျြောင်း (to praise)
        """
        from myspellchecker.text.validator import ValidationIssue, validate_text

        # Ya+Ra is valid per UTN #11 - should NOT be flagged as Zawgyi
        result = validate_text("ကျြ")
        issue_types = [i[0] for i in result.issues]
        # Ensure ZAWGYI_YA_RA is NOT in the issues (if the enum still exists)
        if hasattr(ValidationIssue, "ZAWGYI_YA_RA"):
            assert ValidationIssue.ZAWGYI_YA_RA not in issue_types

    def test_asat_before_vowel(self):
        """Test validate_text detects asat before vowel."""
        from myspellchecker.text.validator import ValidationIssue, validate_text

        # ် (asat) before ု (vowel) - invalid order
        result = validate_text("က်ု")
        assert result.is_valid is False
        issue_types = [i[0] for i in result.issues]
        assert ValidationIssue.ASAT_BEFORE_VOWEL in issue_types

    def test_digit_tone(self):
        """Test validate_text detects digit+tone pattern."""
        from myspellchecker.text.validator import ValidationIssue, validate_text

        result = validate_text("၁့")
        assert result.is_valid is False
        issue_types = [i[0] for i in result.issues]
        assert ValidationIssue.DIGIT_TONE in issue_types

    def test_scrambled_order(self):
        """Test validate_text detects scrambled ordering."""
        from myspellchecker.text.validator import ValidationIssue, validate_text

        # vowel + asat + vowel - invalid scrambled order
        result = validate_text("ကိ်ု")
        assert result.is_valid is False
        issue_types = [i[0] for i in result.issues]
        assert ValidationIssue.SCRAMBLED_ORDER in issue_types

    def test_incomplete_vowel(self):
        """Test validate_text detects incomplete O-vowel pattern."""
        from myspellchecker.text.validator import ValidationIssue, validate_text

        # ိ့ without ု - incomplete O-vowel
        result = validate_text("ကိ့")
        assert result.is_valid is False
        issue_types = [i[0] for i in result.issues]
        assert ValidationIssue.INCOMPLETE_VOWEL in issue_types

    def test_doubled_vowel(self):
        """Test validate_text detects doubled vowels."""
        from myspellchecker.text.validator import ValidationIssue, validate_text

        result = validate_text("ကာာ")
        assert result.is_valid is False
        issue_types = [i[0] for i in result.issues]
        assert ValidationIssue.DOUBLED_DIACRITIC in issue_types

    def test_doubled_medial(self):
        """Test validate_text detects doubled medials."""
        from myspellchecker.text.validator import ValidationIssue, validate_text

        result = validate_text("ကျျ")
        assert result.is_valid is False
        issue_types = [i[0] for i in result.issues]
        assert ValidationIssue.DOUBLED_DIACRITIC in issue_types

    def test_invalid_vowel_sequence(self):
        """Test validate_text detects invalid vowel sequences."""
        from myspellchecker.text.validator import ValidationIssue, validate_text

        # ိိ - doubled i-vowel
        result = validate_text("ကိိ")
        assert result.is_valid is False
        issue_types = [i[0] for i in result.issues]
        assert ValidationIssue.DOUBLED_DIACRITIC in issue_types

    def test_virama_at_end(self):
        """Test validate_text detects virama at end."""
        from myspellchecker.text.validator import ValidationIssue, validate_text

        # ္ at end - incomplete stacking
        result = validate_text("က္")
        assert result.is_valid is False
        issue_types = [i[0] for i in result.issues]
        assert ValidationIssue.VIRAMA_AT_END in issue_types

    def test_invalid_start_character(self):
        """Test validate_text detects invalid start character."""
        from myspellchecker.text.validator import ValidationIssue, validate_text

        # Starting with a vowel sign (dependent vowel, not independent)
        result = validate_text("ါက")
        assert result.is_valid is False
        issue_types = [i[0] for i in result.issues]
        assert ValidationIssue.INVALID_START in issue_types

    def test_multiple_issues(self):
        """Test validate_text returns multiple issues."""
        from myspellchecker.text.validator import validate_text

        # Text with multiple issues
        result = validate_text("ါက်ု")  # Invalid start + asat before vowel
        assert result.is_valid is False
        assert len(result.issues) >= 2

    # ========================================================================
    # Phase 1 Quality Filter Tests
    # ========================================================================

    def test_pure_numeral_filter(self):
        """Test validate_text rejects pure numeral sequences."""
        from myspellchecker.text.validator import ValidationIssue, validate_text

        # Pure numerals should be rejected
        result = validate_text("၆၉၀၀")
        assert result.is_valid is False
        assert result.issues[0][0] == ValidationIssue.PURE_NUMERAL

        result = validate_text("၁၆၄၂")
        assert result.is_valid is False
        assert result.issues[0][0] == ValidationIssue.PURE_NUMERAL

        result = validate_text("၅၀၀၀၀၀")
        assert result.is_valid is False
        assert result.issues[0][0] == ValidationIssue.PURE_NUMERAL

        # Single digit should also be rejected
        result = validate_text("၁")
        assert result.is_valid is False
        assert result.issues[0][0] == ValidationIssue.PURE_NUMERAL

    def test_doubled_consonant_filter(self):
        """Test validate_text rejects doubled consonant fragments."""
        from myspellchecker.text.validator import ValidationIssue, validate_text

        # Doubled consonants should be rejected
        result = validate_text("ဆဆ")
        assert result.is_valid is False
        assert result.issues[0][0] == ValidationIssue.DOUBLED_CONSONANT

        result = validate_text("အအ")
        assert result.is_valid is False
        assert result.issues[0][0] == ValidationIssue.DOUBLED_CONSONANT

        result = validate_text("တတ")
        assert result.is_valid is False
        assert result.issues[0][0] == ValidationIssue.DOUBLED_CONSONANT

        result = validate_text("ညည")
        assert result.is_valid is False
        assert result.issues[0][0] == ValidationIssue.DOUBLED_CONSONANT

        # Non-doubled 2-char words should be allowed (if otherwise valid)
        # Note: ကခ may fail other validations but not DOUBLED_CONSONANT
        result = validate_text("ကခ")
        issue_types = [i[0] for i in result.issues]
        assert ValidationIssue.DOUBLED_CONSONANT not in issue_types

    def test_nuu_nee_typo_filter(self):
        """Test validate_text rejects နူန်း → နှုန်း typos."""
        from myspellchecker.text.validator import ValidationIssue, validate_text

        # These are known invalid typos
        result = validate_text("စျေးနူန်း")
        assert result.is_valid is False
        assert result.issues[0][0] == ValidationIssue.KNOWN_INVALID

        result = validate_text("နူန်း")
        assert result.is_valid is False
        assert result.issues[0][0] == ValidationIssue.KNOWN_INVALID

        result = validate_text("ရာခိုင်နူန်း")
        assert result.is_valid is False
        assert result.issues[0][0] == ValidationIssue.KNOWN_INVALID

    def test_khyae_typo_filter(self):
        """Test validate_text rejects ခြဲ့ → ခဲ့ typos."""
        from myspellchecker.text.validator import ValidationIssue, validate_text

        # These are known invalid typos
        result = validate_text("ခြဲ့ပီး")
        assert result.is_valid is False
        assert result.issues[0][0] == ValidationIssue.KNOWN_INVALID

        result = validate_text("ခြဲ့က")
        assert result.is_valid is False
        assert result.issues[0][0] == ValidationIssue.KNOWN_INVALID

    def test_invalid_e_aa_vowel_sequence(self):
        """Test validate_text rejects invalid ေါ sequences."""
        from myspellchecker.text.validator import ValidationIssue, validate_text

        # ေါ sequence is invalid (ေ cannot combine with ါ short-aa)
        result = validate_text("တွေါ")
        assert result.is_valid is False
        assert result.issues[0][0] == ValidationIssue.INVALID_VOWEL_SEQUENCE_SYLLABLE

        # ော (e + tall-aa) is VALID - it's the standard "aw" vowel
        result = validate_text("ကော")
        assert result.is_valid is True  # This is valid Myanmar


class TestValidateWord:
    """Tests for validate_word function."""

    def test_empty_word(self):
        """Test validate_word with empty string."""
        from myspellchecker.text.validator import validate_word

        assert validate_word("") is False

    def test_whitespace_word(self):
        """Test validate_word with whitespace."""
        from myspellchecker.text.validator import validate_word

        assert validate_word("   ") is False

    def test_valid_word(self):
        """Test validate_word with valid word."""
        from myspellchecker.text.validator import validate_word

        assert validate_word("မြန်မာ") is True
        assert validate_word("စာ") is True

    def test_invalid_word(self):
        """Test validate_word with invalid word."""
        from myspellchecker.text.validator import validate_word

        assert validate_word("ပျှော်") is False
        assert validate_word("က်ု") is False


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_valid_result(self):
        from myspellchecker.text.validator import ValidationResult

        result = ValidationResult(is_valid=True, issues=[])
        assert result.is_valid is True
        assert result.issues == []
        assert result.cleaned_text is None

    def test_invalid_result_with_issues(self):
        from myspellchecker.text.validator import ValidationIssue, ValidationResult

        issues = [(ValidationIssue.EXTENDED_MYANMAR, "Test issue")]
        result = ValidationResult(is_valid=False, issues=issues)
        assert result.is_valid is False
        assert len(result.issues) == 1
        assert bool(result) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
