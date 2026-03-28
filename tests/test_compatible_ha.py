"""Tests for COMPATIBLE_HA set and medial compatibility validation.

Add COMPATIBLE_HA set for medial Ha consonant compatibility.

Medial Ha (ှ Ha-htoe) creates aspirated sonorant clusters and should only
combine with sonorant consonants: nasals (Nga, Nya, Na, Ma), liquids (La),
and glides (Ya, Ra).
"""

import pytest

from myspellchecker.core.constants import (
    COMPATIBLE_HA,
    COMPATIBLE_RA,
    COMPATIBLE_WA,
    COMPATIBLE_YA,
    CONSONANTS,
    MEDIAL_HA,
    MEDIAL_RA,
    MEDIAL_WA,
    MEDIAL_YA,
)
from myspellchecker.core.syllable_rules import _SyllableRuleValidatorPython


class TestCompatibleHaConstant:
    """Tests for the COMPATIBLE_HA constant definition."""

    def test_compatible_ha_exists(self):
        """Test that COMPATIBLE_HA is defined and non-empty."""
        assert COMPATIBLE_HA is not None
        assert len(COMPATIBLE_HA) > 0

    def test_compatible_ha_is_set(self):
        """Test that COMPATIBLE_HA is a set for O(1) lookup."""
        assert isinstance(COMPATIBLE_HA, (set, frozenset))

    def test_compatible_ha_contains_sonorants(self):
        """Test that COMPATIBLE_HA contains expected sonorant consonants."""
        # Expected sonorants that can take Medial Ha
        expected_sonorants = {
            "\u1004",  # Nga (င) - velar nasal
            "\u100a",  # Nya (ည) - palatal nasal
            "\u1009",  # Nya variant (ည)
            "\u1014",  # Na (န) - alveolar nasal
            "\u1019",  # Ma (မ) - bilabial nasal
            "\u101c",  # La (လ) - lateral
            "\u101a",  # Ya (ယ) - palatal glide
            "\u101b",  # Ra (ရ) - alveolar tap
        }
        for char in expected_sonorants:
            assert char in COMPATIBLE_HA, f"Expected {repr(char)} in COMPATIBLE_HA"

    def test_compatible_ha_excludes_stops(self):
        """Test that COMPATIBLE_HA excludes stop consonants."""
        # Stops cannot be aspirated via medial Ha
        stops = {
            "\u1000",  # Ka
            "\u1001",  # Kha
            "\u1002",  # Ga
            "\u1003",  # Gha
            "\u1005",  # Ca
            "\u1006",  # Cha
            "\u1007",  # Ja
            "\u1008",  # Jha
            "\u1010",  # Ta
            "\u1011",  # Tha
            "\u1012",  # Da
            "\u1013",  # Dha
            "\u1015",  # Pa
            "\u1016",  # Pha
            "\u1017",  # Ba
            "\u1018",  # Bha
        }
        for char in stops:
            assert char not in COMPATIBLE_HA, f"Stop {repr(char)} should not be in COMPATIBLE_HA"

    def test_compatible_ha_all_valid_myanmar_chars(self):
        """Test that all COMPATIBLE_HA entries are valid Myanmar consonants."""
        for char in COMPATIBLE_HA:
            assert char in CONSONANTS, f"{repr(char)} is not a valid consonant"


class TestCompatibleSetsConsistency:
    """Tests for consistency across all COMPATIBLE_* sets."""

    def test_all_compatible_sets_exist(self):
        """Test that all COMPATIBLE_* sets are defined."""
        assert COMPATIBLE_YA is not None
        assert COMPATIBLE_RA is not None
        assert COMPATIBLE_WA is not None
        assert COMPATIBLE_HA is not None

    def test_compatible_sets_contain_only_consonants(self):
        """Test that all entries in COMPATIBLE_* sets are consonants."""
        for name, compat_set in [
            ("COMPATIBLE_YA", COMPATIBLE_YA),
            ("COMPATIBLE_RA", COMPATIBLE_RA),
            ("COMPATIBLE_WA", COMPATIBLE_WA),
            ("COMPATIBLE_HA", COMPATIBLE_HA),
        ]:
            for char in compat_set:
                assert char in CONSONANTS, f"{repr(char)} in {name} is not a consonant"

    def test_compatible_wa_is_broadest(self):
        """Test that COMPATIBLE_WA is the broadest compatibility set."""
        # Wa has the broadest compatibility in Myanmar phonotactics
        assert len(COMPATIBLE_WA) >= len(COMPATIBLE_YA)
        assert len(COMPATIBLE_WA) >= len(COMPATIBLE_RA)
        assert len(COMPATIBLE_WA) >= len(COMPATIBLE_HA)


class TestMedialCompatibilityValidation:
    """Tests for the _check_medial_compatibility method."""

    @pytest.fixture
    def validator_strict(self):
        """Create a strict mode validator."""
        return _SyllableRuleValidatorPython(strict=True)

    @pytest.fixture
    def validator_lenient(self):
        """Create a lenient mode validator."""
        return _SyllableRuleValidatorPython(strict=False)

    # Valid Medial Ha combinations
    def test_valid_medial_ha_with_nga(self, validator_strict):
        """Test Nga + Medial Ha is valid (aspirated velar nasal)."""
        syllable = "\u1004" + MEDIAL_HA  # င + ှ = ငှ
        assert validator_strict._check_medial_compatibility(syllable) is True

    def test_valid_medial_ha_with_na(self, validator_strict):
        """Test Na + Medial Ha is valid (aspirated alveolar nasal)."""
        syllable = "\u1014" + MEDIAL_HA  # န + ှ = နှ
        assert validator_strict._check_medial_compatibility(syllable) is True

    def test_valid_medial_ha_with_ma(self, validator_strict):
        """Test Ma + Medial Ha is valid (aspirated bilabial nasal)."""
        syllable = "\u1019" + MEDIAL_HA  # မ + ှ = မှ
        assert validator_strict._check_medial_compatibility(syllable) is True

    def test_valid_medial_ha_with_la(self, validator_strict):
        """Test La + Medial Ha is valid (aspirated lateral)."""
        syllable = "\u101c" + MEDIAL_HA  # လ + ှ = လှ
        assert validator_strict._check_medial_compatibility(syllable) is True

    def test_valid_medial_ha_with_ya(self, validator_strict):
        """Test Ya + Medial Ha is valid (aspirated palatal glide)."""
        syllable = "\u101a" + MEDIAL_HA  # ယ + ှ = ယှ
        assert validator_strict._check_medial_compatibility(syllable) is True

    # Invalid Medial Ha combinations (stops)
    def test_invalid_medial_ha_with_ka(self, validator_strict):
        """Test Ka + Medial Ha is invalid (stop cannot aspirate via medial Ha)."""
        syllable = "\u1000" + MEDIAL_HA  # က + ှ = ကှ (invalid)
        assert validator_strict._check_medial_compatibility(syllable) is False

    def test_invalid_medial_ha_with_ta(self, validator_strict):
        """Test Ta + Medial Ha is invalid."""
        syllable = "\u1010" + MEDIAL_HA  # တ + ှ = တှ (invalid)
        assert validator_strict._check_medial_compatibility(syllable) is False

    def test_invalid_medial_ha_with_pa(self, validator_strict):
        """Test Pa + Medial Ha is invalid."""
        syllable = "\u1015" + MEDIAL_HA  # ပ + ှ = ပှ (invalid)
        assert validator_strict._check_medial_compatibility(syllable) is False

    def test_invalid_medial_ha_with_ca(self, validator_strict):
        """Test Ca + Medial Ha is invalid."""
        syllable = "\u1005" + MEDIAL_HA  # စ + ှ = စှ (invalid)
        assert validator_strict._check_medial_compatibility(syllable) is False

    # Lenient mode allows invalid combinations
    def test_lenient_mode_allows_invalid_ha(self, validator_lenient):
        """Test that lenient mode allows invalid medial Ha combinations."""
        syllable = "\u1000" + MEDIAL_HA  # က + ှ = ကှ
        # In lenient mode, compatibility is not enforced
        assert validator_lenient._check_medial_compatibility(syllable) is True

    # Other medials
    def test_valid_medial_ya_with_ka(self, validator_strict):
        """Test Ka + Medial Ya is valid."""
        syllable = "\u1000" + MEDIAL_YA  # က + ျ = ကျ
        assert validator_strict._check_medial_compatibility(syllable) is True

    def test_valid_medial_ra_with_pa(self, validator_strict):
        """Test Pa + Medial Ra is valid."""
        syllable = "\u1015" + MEDIAL_RA  # ပ + ြ = ပြ
        assert validator_strict._check_medial_compatibility(syllable) is True

    def test_valid_medial_wa_with_ta(self, validator_strict):
        """Test Ta + Medial Wa is valid (Wa has broad compatibility)."""
        syllable = "\u1010" + MEDIAL_WA  # တ + ွ = တွ
        assert validator_strict._check_medial_compatibility(syllable) is True

    def test_invalid_medial_ya_with_ta(self, validator_strict):
        """Test Ta + Medial Ya is invalid (Ta not in COMPATIBLE_YA)."""
        syllable = "\u1010" + MEDIAL_YA  # တ + ျ = တျ (invalid)
        assert validator_strict._check_medial_compatibility(syllable) is False

    def test_invalid_medial_ra_with_ta(self, validator_strict):
        """Test Ta + Medial Ra is invalid (Ta not in COMPATIBLE_RA)."""
        syllable = "\u1010" + MEDIAL_RA  # တ + ြ = တြ (invalid)
        assert validator_strict._check_medial_compatibility(syllable) is False

    # Empty and no-medial cases
    def test_empty_syllable_returns_true(self, validator_strict):
        """Test empty syllable returns True (no medials to check)."""
        assert validator_strict._check_medial_compatibility("") is True

    def test_no_medial_returns_true(self, validator_strict):
        """Test syllable without medials returns True."""
        syllable = "\u1000\u102c"  # ကာ (Ka + Aa vowel)
        assert validator_strict._check_medial_compatibility(syllable) is True


class TestMedialCompatibilityEdgeCases:
    """Edge case tests for medial compatibility validation."""

    @pytest.fixture
    def validator_strict(self):
        """Create a strict mode validator."""
        return _SyllableRuleValidatorPython(strict=True)

    def test_combined_medials_ya_wa(self, validator_strict):
        """Test syllable with combined Ya and Wa medials."""
        # ကျွ = Ka + Medial Ya + Medial Wa
        syllable = "\u1000\u103b\u103d"
        # Ka is compatible with Ya and Wa
        assert validator_strict._check_medial_compatibility(syllable) is True

    def test_combined_medials_ra_wa(self, validator_strict):
        """Test syllable with combined Ra and Wa medials."""
        # ပြွ = Pa + Medial Ra + Medial Wa
        syllable = "\u1015\u103c\u103d"
        # Pa is compatible with Ra and Wa
        assert validator_strict._check_medial_compatibility(syllable) is True

    def test_combined_medials_ya_ha_invalid(self, validator_strict):
        """Test invalid combination: Ya and Ha medials with stop consonant."""
        # ကျှ = Ka + Medial Ya + Medial Ha (Ha is invalid with Ka)
        syllable = "\u1000\u103b\u103e"
        # Ka is compatible with Ya but NOT with Ha
        assert validator_strict._check_medial_compatibility(syllable) is False

    def test_combined_medials_all_valid(self, validator_strict):
        """Test valid syllable with multiple medials on compatible consonant."""
        # မျွှ = Ma + Medial Ya + Medial Wa + Medial Ha
        # Ma is compatible with all four medials (it's a sonorant)
        syllable = "\u1019\u103b\u103d\u103e"
        assert validator_strict._check_medial_compatibility(syllable) is True

    def test_ra_glide_with_sonorant(self, validator_strict):
        """Test Ra on invalid consonant (Ta not in COMPATIBLE_RA)."""
        syllable = "\u1010\u103c"  # တြ
        assert validator_strict._check_medial_compatibility(syllable) is False

    def test_wa_broad_compatibility(self, validator_strict):
        """Test Wa's broad compatibility with various consonants."""
        # Wa should be compatible with most consonants
        test_consonants = [
            "\u1000",  # Ka
            "\u1005",  # Ca
            "\u1010",  # Ta
            "\u1015",  # Pa
            "\u101a",  # Ya
            "\u101b",  # Ra
            "\u101c",  # La
            "\u101d",  # Wa
            "\u101f",  # Ha
        ]
        for consonant in test_consonants:
            syllable = consonant + MEDIAL_WA
            assert validator_strict._check_medial_compatibility(syllable) is True, (
                f"Wa should be compatible with {repr(consonant)}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
