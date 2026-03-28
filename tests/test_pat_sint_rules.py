"""
Tests for 'Pat Sint' (Stacking) rules based on 'Wet' (Vagga) logic.
"""

import pytest

from myspellchecker.core.syllable_rules import SyllableRuleValidator


@pytest.fixture
def validator():
    return SyllableRuleValidator()


class TestPatSintLogic:
    """Tests for 'Pat Sint' (stacked consonant) validation."""

    def test_valid_homorganic_stacking(self, validator):
        """
        Test valid stacking within the same Wet (Group).
        Rule: Upper (Col 1/3) -> Lower (Col 1/2 or 3/4).
        """
        # Ka-Wet: Ka (1) + Ka (1) - e.g. တက္ကသိုလ် (University)
        assert validator.validate("က္က")
        # Ka-Wet: Ka (1) + Kha (2) - e.g. ဒုက္ခ (Trouble)
        assert validator.validate("က္ခ")
        # Pa-Wet: Pa (1) + Pa (1) - e.g. ပပ္ပ (Pappa)
        assert validator.validate("ပ္ပ")
        # Pa-Wet: Ma (5) + Ma (5) - e.g. ဓမ္မ (Dhamma)
        assert validator.validate("မ္မ")

    def test_invalid_cross_wet_stacking(self, validator):
        """
        Test invalid stacking across different Wets.
        Rule: Consonants from different groups usually cannot stack (with exceptions).
        Note: Some cross-Wet stackings like တ္က are valid exceptions for Pali/Sanskrit.
        """
        # Ka (Ka-Wet) + Pa (Pa-Wet) - Invalid
        assert not validator.validate("က္ပ")
        # Tha (Ta-Wet) + Ka (Ka-Wet) - Invalid (not in STACKING_EXCEPTIONS)
        assert not validator.validate("ထ္က")
        # Da (Ta-Wet) + Ka (Ka-Wet) - Invalid (not in STACKING_EXCEPTIONS)
        assert not validator.validate("ဒ္က")

    def test_valid_exceptions(self, validator):
        """
        Test valid exceptions like 'Great Sa' and 'La' stacking.
        """
        # Great Sa (Thaa Gyi) - Ligature for Sa + Sa
        # Note: 'ဿ' is a single char U+103F, not a stack U+1039.
        # But if someone types Sa + Virama + Sa...
        assert validator.validate("ဿ")  # Native Char

        # Sa (Thaa) + Ma (Ma) - e.g. သမီး (Daughter - abbr spelling သ္မီး)
        assert validator.validate("သ္မ")

        # Na (Na) + Sa (Thaa) - Exception mentioned in research?
        # Actually research said "Na and Ma can stack over Sa".
        # Let's test "န္သ"
        assert validator.validate("န္သ")

    def test_invalid_column_logic(self, validator):
        """
        Test invalid column combinations within same Wet.
        """
        # Ka (1) on Ga (3) - Invalid
        assert not validator.validate("က္ဂ")
        # Ga (3) on Ka (1) - Invalid
        assert not validator.validate("ဂ္က")

    def test_kinzi_stacking(self, validator):
        """
        Test Kinzi (Nga + Asat + Virama) stacking.
        Kinzi can generally stack on any consonant.
        """
        # Mingala (Nga+Asat+Virama on Ga) - Testing the Kinzi block "Ming-Ga"
        assert validator.validate("မင်္ဂ")
        # Sangha-Baw (Nga+Asat+Virama on Ba)
        assert validator.validate("သင်္ဘော")
