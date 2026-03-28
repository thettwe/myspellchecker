"""Tests for Medial Ya + AI vowel combination.

Tests that the Medial Ya + AI vowel constraint is not too strict.
The combination was previously rejected but does occur in Pali/Sanskrit
loanwords and proper nouns.
"""

import pytest


def _has_cython_extension():
    """Check if Cython extension is available."""
    try:
        from myspellchecker.core.syllable_rules_c import SyllableRuleValidator  # noqa: F401

        return True
    except ImportError:
        return False


class TestMedialYaAIVowel:
    """Test that Medial Ya + AI vowel combination is now allowed."""

    def test_medial_ya_with_ai_vowel_python(self):
        """
        Test that Medial Ya + AI vowel is valid in Python implementation.

        ျ (U+103B) + ဲ (U+1032) combination should be allowed.
        """
        from myspellchecker.core.syllable_rules import _SyllableRuleValidatorPython

        validator = _SyllableRuleValidatorPython(strict=True)

        # Syllables with Medial Ya + AI vowel
        # ကျဲ = Ka + Ya-pin + AI vowel
        syllable = "\u1000\u103b\u1032"  # ကျဲ
        result = validator._check_medial_vowel_compatibility(syllable)
        assert result is True, f"ကျဲ should be valid, got {result}"

        # ချဲ = Kha + Ya-pin + AI vowel
        syllable2 = "\u1001\u103b\u1032"  # ချဲ
        result2 = validator._check_medial_vowel_compatibility(syllable2)
        assert result2 is True, f"ချဲ should be valid, got {result2}"

    def test_medial_wa_with_u_vowel_still_invalid(self):
        """
        Test that Medial Wa + U vowel is still invalid (phonetically incompatible).

        ွ (U+103D) + ု (U+102F) should still be rejected.
        """
        from myspellchecker.core.syllable_rules import _SyllableRuleValidatorPython

        validator = _SyllableRuleValidatorPython(strict=True)

        # ကွု = Ka + Wa-hswe + U vowel - should be invalid
        syllable = "\u1000\u103d\u102f"  # ကွု
        result = validator._check_medial_vowel_compatibility(syllable)
        assert result is False, f"ကွု should be invalid, got {result}"

        # ကွူ = Ka + Wa-hswe + UU vowel - should also be invalid
        syllable2 = "\u1000\u103d\u1030"  # ကွူ
        result2 = validator._check_medial_vowel_compatibility(syllable2)
        assert result2 is False, f"ကွူ should be invalid, got {result2}"

    def test_combined_ya_wa_with_u_vowel_still_invalid(self):
        """
        Test that combined Ya+Wa medials with U vowel is still invalid.

        ျွ + ု should still be rejected (phonetically incompatible).
        """
        from myspellchecker.core.syllable_rules import _SyllableRuleValidatorPython

        validator = _SyllableRuleValidatorPython(strict=True)

        # ကျွု = Ka + Ya-pin + Wa-hswe + U vowel
        syllable = "\u1000\u103b\u103d\u102f"  # ကျွု
        result = validator._check_medial_vowel_compatibility(syllable)
        assert result is False, f"ကျွု should be invalid, got {result}"

    @pytest.mark.skipif(
        not _has_cython_extension(),
        reason="Cython extension not available",
    )
    def test_medial_ya_with_ai_vowel_cython(self):
        """
        Test that Medial Ya + AI vowel is valid in Cython implementation.
        """
        from myspellchecker.core.syllable_rules_c import SyllableRuleValidator

        validator = SyllableRuleValidator(strict=True)

        # ကျဲ = Ka + Ya-pin + AI vowel
        syllable = "\u1000\u103b\u1032"
        result = validator.validate(syllable)
        # Note: validate() checks many things, this syllable should pass
        # all checks including medial-vowel compatibility
        # We're primarily testing that it's not rejected due to Ya+AI
        assert result is True, f"ကျဲ should be valid in Cython, got {result}"
