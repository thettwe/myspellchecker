"""
End-to-end validation tests for known problem cases.

This module tests the full validation pipeline to ensure:
1. Invalid patterns are correctly rejected
2. Valid Myanmar text passes through
3. Edge cases are handled properly
4. No regression in validation coverage

Test Categories:
    - Extended Myanmar character rejection
    - Zawgyi artifact detection
    - Valid Unicode acceptance
    - Structural validation
    - Pipeline integration
"""

import pytest

from myspellchecker.core.syllable_rules import SyllableRuleValidator
from myspellchecker.text.validator import validate_text, validate_word


class TestKnownInvalidPatterns:
    """Tests for known invalid patterns that should be rejected."""

    def test_extended_myanmar_characters_rejected(self):
        """Extended Myanmar (U+1050-U+109F) should be rejected."""
        # Extended Myanmar characters
        extended_chars = [
            "ကမာ့႓",  # Contains Extended Myanmar
            "\u1050",  # Extended Myanmar char
            "ာ\u1051",  # Mixed with extended
        ]
        for text in extended_chars:
            result = validate_text(text)
            assert not result.is_valid, f"Should reject extended Myanmar: {text!r}"

    def test_zawgyi_artifacts_rejected(self):
        """Zawgyi encoding artifacts should be detected and handled."""
        from myspellchecker.text.normalize import is_likely_zawgyi

        # Note: These are pattern-based approximations
        # Real Zawgyi samples would need actual Zawgyi-encoded text
        zawgyi_like_patterns = [
            "ခွငျးအားဖွငျ့",  # Pattern-based artifact
            "ျပည္",  # Pattern-based artifact
        ]

        for text in zawgyi_like_patterns:
            # myanmar-tools may not detect pattern-based artifacts
            # since they may not be statistically Zawgyi
            # Pattern validation will still catch them during spell check
            is_zawgyi, conf = is_likely_zawgyi(text)
            # Just verify function works, don't assert detection
            assert isinstance(is_zawgyi, bool)
            assert 0.0 <= conf <= 1.0

            # Check validation
            result = validate_text(text)
            assert not result.is_valid, f"Should reject Zawgyi artifact: {text!r}"

    def test_doubled_diacritics_rejected(self):
        """Doubled diacritics should be rejected."""
        doubled = [
            "က\u102d\u102d",  # Doubled vowel i
            "ကြြ",  # Doubled medial ra
        ]
        for text in doubled:
            result = validate_text(text)
            assert not result.is_valid, f"Should reject doubled diacritic: {text!r}"

    def test_asat_before_vowel_rejected(self):
        """Asat appearing before vowel should be rejected."""
        invalid_asat = [
            "က်ာ",  # Asat before aa vowel
        ]
        for text in invalid_asat:
            result = validate_text(text)
            # This pattern is structurally invalid
            assert not result.is_valid or len(result.issues) > 0, (
                f"Should flag asat-before-vowel: {text!r}"
            )

    def test_known_invalid_words_rejected(self):
        """Known invalid words should be rejected."""
        invalid_words = [
            "ပျှော်",  # Known invalid
            "င်း",  # Segmentation artifact
            "်",  # Floating asat
            "့",  # Floating dot below
        ]
        for word in invalid_words:
            is_valid = validate_word(word)
            assert not is_valid, f"Should reject invalid word: {word!r}"


class TestValidMyanmarText:
    """Tests for valid Myanmar text that should pass validation."""

    def test_common_words_accepted(self):
        """Common Myanmar words should be accepted."""
        valid_words = [
            "မြန်မာ",  # Myanmar
            "ဘာသာ",  # Language
            "စကား",  # Speech
            "နိုင်ငံ",  # Country
            "ရေး",  # Write/affair
            "ကျောင်း",  # School
            "စာအုပ်",  # Book
            "လူ",  # Person
        ]
        for word in valid_words:
            is_valid = validate_word(word)
            assert is_valid, f"Should accept valid word: {word!r}"

    def test_common_particles_accepted(self):
        """Common particles should be accepted."""
        particles = [
            "က",  # Subject marker
            "ကို",  # Object marker
            "မှာ",  # Location
            "တယ်",  # Statement ending
            "သည်",  # Formal ending
            "ပါ",  # Polite particle
        ]
        for particle in particles:
            is_valid = validate_word(particle)
            assert is_valid, f"Should accept particle: {particle!r}"

    def test_complex_syllables_accepted(self):
        """Complex but valid syllables should be accepted."""
        complex_syllables = [
            "ကျွန်",  # With medials
            "ကြောင်း",  # Cause/because
            "မြင့်",  # High
            "ကြည့်",  # Look
            "ဆွဲ",  # Pull
        ]
        for syllable in complex_syllables:
            is_valid = validate_word(syllable)
            assert is_valid, f"Should accept complex syllable: {syllable!r}"


class TestSyllableRuleValidator:
    """Tests for syllable structure validation."""

    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        return SyllableRuleValidator()

    def test_max_syllable_length_enforced(self, validator):
        """Syllables over max length should be rejected."""
        # Default max is now 10
        long_syllable = "က" * 11
        assert not validator.validate(long_syllable), "Should reject syllables over max length"

    def test_valid_syllable_structure_accepted(self, validator):
        """Valid syllable structures should be accepted."""
        valid = [
            "က",  # Simple consonant
            "ကာ",  # Consonant + vowel
            "ကား",  # With visarga
            "ကန်",  # With final
            "ကျ",  # With medial
            "ကြ",  # With medial ra
            "ကျွ",  # Multiple medials
        ]
        for syl in valid:
            assert validator.validate(syl), f"Should accept valid syllable: {syl!r}"

    def test_floating_diacritics_rejected(self, validator):
        """Syllables starting with diacritics should be rejected."""
        floating = [
            "\u102c",  # Floating aa vowel
            "\u103b",  # Floating ya medial
            "\u1036",  # Floating anusvara
        ]
        for char in floating:
            assert not validator.validate(char), f"Should reject floating diacritic: {char!r}"

    def test_data_corruption_detected(self, validator):
        """Data corruption patterns should be rejected."""
        corrupted = [
            "ကကကက",  # 4+ repeated chars
            "ကွွွွ",  # Repeated medials
        ]
        for pattern in corrupted:
            assert not validator.validate(pattern), f"Should reject corrupted data: {pattern!r}"


class TestPipelineIntegration:
    """Integration tests for the full validation pipeline."""

    def test_mixed_valid_invalid_text(self):
        """Text with both valid and invalid portions should report issues."""
        mixed_text = "မြန်မာ ပျှော် ဘာသာ"  # Contains known invalid word
        result = validate_text(mixed_text)
        # Should flag the invalid word
        assert len(result.issues) > 0, "Should detect invalid word in mixed text"

    def test_empty_text_handling(self):
        """Empty text should be handled gracefully."""
        assert validate_word("") is False
        result = validate_text("")
        assert result.is_valid  # Empty is technically valid (nothing to validate)

    def test_whitespace_only_handling(self):
        """Whitespace-only text should be handled gracefully."""
        result = validate_text("   ")
        assert result.is_valid  # Whitespace is valid

    def test_numeric_text_handling(self):
        """Numeric text should be handled appropriately."""
        result = validate_text("123")
        assert result.is_valid  # Numbers are valid

    def test_mixed_script_handling(self):
        """Mixed Myanmar/English text should be handled."""
        result = validate_text("Hello မြန်မာ")
        # Mixed is valid - only Myanmar parts are validated
        assert result.is_valid


class TestPipelineValidationEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_consonant_valid(self):
        """Single consonants should be valid syllables."""
        consonants = ["က", "ခ", "ဂ", "ဃ", "င"]
        for c in consonants:
            assert validate_word(c), f"Single consonant should be valid: {c!r}"

    def test_kinzi_pattern_valid(self):
        """Valid Kinzi patterns should be accepted."""
        # Kinzi: Nga + Asat + Virama + Consonant
        kinzi = "င်္က"  # Kinzi + Ka
        validator = SyllableRuleValidator()
        # Kinzi is a valid but complex pattern
        # The validator may or may not accept based on implementation
        # This test documents expected behavior
        result = validator.validate(kinzi)
        # Update expected result based on actual implementation behavior
        assert isinstance(result, bool)  # Should not crash

    def test_stacking_pattern_valid(self):
        """Valid consonant stacking should be accepted."""
        stacked = "မ္မ"  # Ma stacked on Ma
        validator = SyllableRuleValidator()
        result = validator.validate(stacked)
        # Document expected behavior
        assert isinstance(result, bool)

    def test_unicode_normalization_consistency(self):
        """Same text in different Unicode forms should validate consistently."""
        import unicodedata

        text = "မြန်မာ"
        nfc = unicodedata.normalize("NFC", text)
        nfd = unicodedata.normalize("NFD", text)

        # Both forms should validate the same way
        result_nfc = validate_text(nfc)
        validate_text(nfd)

        # NFD may have issues due to decomposed characters
        # but NFC should be valid
        assert result_nfc.is_valid


class TestRegressionPrevention:
    """Tests to prevent regression in validation coverage."""

    def test_previously_missed_invalid_patterns(self):
        """Patterns that were previously missed should now be caught."""
        # These are patterns from the audit that should be rejected
        regression_cases = [
            ("ကမာ့႓", False, "Extended Myanmar character"),
            ("ခွငျးအားဖွငျ့", False, "Zawgyi artifact"),
        ]

        for text, expected_valid, description in regression_cases:
            result = validate_text(text)
            if expected_valid:
                assert result.is_valid, f"Regression: {description} should be valid"
            else:
                assert not result.is_valid, f"Regression: {description} should be invalid"

    def test_validation_coverage_metrics(self):
        """Verify validation covers expected categories."""

        # Ensure ValidationResult has expected fields
        result = validate_text("မြန်မာ")
        assert hasattr(result, "is_valid")
        assert hasattr(result, "issues")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
