"""Parametric consonant-medial compatibility tests.

Verifies that every consonant + medial combination produces the correct
accept/reject result in strict mode based on the compatibility sets
defined in myanmar_constants.py.
"""

import pytest

from myspellchecker.core.constants import (
    COMPATIBLE_HA,
    COMPATIBLE_RA,
    COMPATIBLE_WA,
    COMPATIBLE_YA,
    CONSONANTS,
    GREAT_SA,
    MEDIAL_HA,
    MEDIAL_RA,
    MEDIAL_WA,
    MEDIAL_YA,
)
from myspellchecker.core.syllable_rules import _SyllableRuleValidatorPython

# Standard consonants (exclude Great Sa which has its own rules)
STANDARD_CONSONANTS = sorted(CONSONANTS - {GREAT_SA})


@pytest.fixture
def strict_validator():
    return _SyllableRuleValidatorPython(strict=True)


@pytest.fixture
def lenient_validator():
    return _SyllableRuleValidatorPython(strict=False)


class TestMedialYaCompatibility:
    """Test Medial Ya (ျ U+103B) against all consonants."""

    @pytest.mark.parametrize("consonant", sorted(COMPATIBLE_YA))
    def test_compatible_consonants_accepted(self, consonant, strict_validator):
        syllable = consonant + MEDIAL_YA
        assert strict_validator.validate(syllable), (
            f"{consonant!r} (U+{ord(consonant):04X}) + Medial Ya should be accepted"
        )

    @pytest.mark.parametrize("consonant", sorted(set(STANDARD_CONSONANTS) - COMPATIBLE_YA))
    def test_incompatible_consonants_rejected(self, consonant, strict_validator):
        syllable = consonant + MEDIAL_YA
        assert not strict_validator.validate(syllable), (
            f"{consonant!r} (U+{ord(consonant):04X}) + Medial Ya should be rejected"
        )


class TestMedialRaCompatibility:
    """Test Medial Ra (ြ U+103C) against all consonants."""

    @pytest.mark.parametrize("consonant", sorted(COMPATIBLE_RA))
    def test_compatible_consonants_accepted(self, consonant, strict_validator):
        syllable = consonant + MEDIAL_RA
        assert strict_validator.validate(syllable), (
            f"{consonant!r} (U+{ord(consonant):04X}) + Medial Ra should be accepted"
        )

    @pytest.mark.parametrize("consonant", sorted(set(STANDARD_CONSONANTS) - COMPATIBLE_RA))
    def test_incompatible_consonants_rejected(self, consonant, strict_validator):
        syllable = consonant + MEDIAL_RA
        assert not strict_validator.validate(syllable), (
            f"{consonant!r} (U+{ord(consonant):04X}) + Medial Ra should be rejected"
        )


class TestMedialWaCompatibility:
    """Test Medial Wa (ွ U+103D) against all consonants."""

    @pytest.mark.parametrize("consonant", sorted(COMPATIBLE_WA))
    def test_compatible_consonants_accepted(self, consonant, strict_validator):
        syllable = consonant + MEDIAL_WA
        assert strict_validator.validate(syllable), (
            f"{consonant!r} (U+{ord(consonant):04X}) + Medial Wa should be accepted"
        )

    @pytest.mark.parametrize("consonant", sorted(set(STANDARD_CONSONANTS) - COMPATIBLE_WA))
    def test_incompatible_consonants_rejected(self, consonant, strict_validator):
        syllable = consonant + MEDIAL_WA
        assert not strict_validator.validate(syllable), (
            f"{consonant!r} (U+{ord(consonant):04X}) + Medial Wa should be rejected"
        )


class TestMedialHaCompatibility:
    """Test Medial Ha (ှ U+103E) against all consonants."""

    @pytest.mark.parametrize("consonant", sorted(COMPATIBLE_HA))
    def test_compatible_consonants_accepted(self, consonant, strict_validator):
        syllable = consonant + MEDIAL_HA
        assert strict_validator.validate(syllable), (
            f"{consonant!r} (U+{ord(consonant):04X}) + Medial Ha should be accepted"
        )

    @pytest.mark.parametrize("consonant", sorted(set(STANDARD_CONSONANTS) - COMPATIBLE_HA))
    def test_incompatible_consonants_rejected(self, consonant, strict_validator):
        syllable = consonant + MEDIAL_HA
        assert not strict_validator.validate(syllable), (
            f"{consonant!r} (U+{ord(consonant):04X}) + Medial Ha should be rejected"
        )


class TestLenientModeAcceptsAll:
    """In lenient mode, all consonant+medial combos should be accepted."""

    @pytest.mark.parametrize("consonant", STANDARD_CONSONANTS)
    @pytest.mark.parametrize("medial", [MEDIAL_YA, MEDIAL_RA, MEDIAL_WA, MEDIAL_HA])
    def test_all_accepted_in_lenient(self, consonant, medial, lenient_validator):
        syllable = consonant + medial
        assert lenient_validator.validate(syllable), (
            f"{consonant!r} + {medial!r} should be accepted in lenient mode"
        )


class TestMedialCompatibilityIntegration:
    """Integration tests for medial compatibility using the public SyllableRuleValidator."""

    def test_wa_with_ha_medial_syllable(self):
        """Test that Wa + Ha-htoe syllable is valid in strict mode."""
        from myspellchecker.core.syllable_rules import SyllableRuleValidator

        validator = SyllableRuleValidator(strict=True)

        # ဝှ (Wa + Ha-htoe) - should be valid since Wa is in COMPATIBLE_HA
        wa_ha = "\u101d\u103e"  # Wa + Medial Ha

        result = validator.validate(wa_ha)
        assert result is True, "ဝှ (Wa + Ha-htoe) should be valid"

    def test_lla_with_ha_medial_syllable(self):
        """Test that Lla + Ha-htoe syllable is valid in strict mode."""
        from myspellchecker.core.syllable_rules import SyllableRuleValidator

        validator = SyllableRuleValidator(strict=True)

        # ဠှ (Lla + Ha-htoe) - should be valid as Lla is a retroflex lateral sonorant
        lla_ha = "\u1020\u103e"  # Lla + Medial Ha

        result = validator.validate(lla_ha)
        assert result is True, "ဠှ (Lla + Ha-htoe) should be valid"

    def test_other_sonorants_with_ha_medial_still_valid(self):
        """Test that other sonorants with Ha-htoe are still valid via public API."""
        from myspellchecker.core.syllable_rules import SyllableRuleValidator

        validator = SyllableRuleValidator(strict=True)

        test_cases = [
            ("\u1014\u103e", "နှ"),  # Na + Ha-htoe
            ("\u1019\u103e", "မှ"),  # Ma + Ha-htoe
            ("\u101c\u103e", "လှ"),  # La + Ha-htoe
            ("\u1020\u103e", "ဠှ"),  # Lla + Ha-htoe (retroflex lateral)
            ("\u101a\u103e", "ယှ"),  # Ya + Ha-htoe
            ("\u101b\u103e", "ရှ"),  # Ra + Ha-htoe
        ]

        for syllable, display in test_cases:
            result = validator.validate(syllable)
            assert result is True, f"{display} should be valid"
