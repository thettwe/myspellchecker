"""Python/Cython SyllableRuleValidator equivalence tests.

Ensures both implementations produce identical results for all inputs.
This is critical for the dual-implementation pattern to catch drift.
"""

import pytest

from myspellchecker.core.syllable_rules import _SyllableRuleValidatorPython

# Try to import Cython implementation
try:
    from myspellchecker.core.syllable_rules_c import SyllableRuleValidator as CythonValidator

    HAS_CYTHON = True
except ImportError:
    HAS_CYTHON = False


@pytest.fixture
def py_validator():
    return _SyllableRuleValidatorPython(strict=True)


@pytest.fixture
def cy_validator():
    if not HAS_CYTHON:
        pytest.skip("Cython implementation not available")
    return CythonValidator(strict=True)


@pytest.fixture
def py_lenient():
    return _SyllableRuleValidatorPython(strict=False)


@pytest.fixture
def cy_lenient():
    if not HAS_CYTHON:
        pytest.skip("Cython implementation not available")
    return CythonValidator(strict=False)


# Comprehensive test syllables covering all 22+ validation checks
VALID_SYLLABLES = [
    # Simple consonant + vowel
    "က",  # bare consonant
    "ကာ",  # Ka + Aa
    "ကိ",  # Ka + I
    "ကု",  # Ka + U
    "ကူ",  # Ka + UU
    "ကေ",  # Ka + E
    "ကဲ",  # Ka + AI
    "ကော",  # Ka + E + Aa
    "ကော်",  # Ka + E + Aa + Asat
    # Medials
    "ကျ",  # Ka + Medial Ya
    "ကြ",  # Ka + Medial Ra
    "ကွ",  # Ka + Medial Wa
    "ကျွ",  # Ka + Ya + Wa
    "ကြွ",  # Ka + Ra + Wa
    # Tone marks
    "ကး",  # Ka + Visarga
    "ပို့",  # Pa + I + Dot Below + Asat (common word)
    # Asat (killer)
    "ကန်",  # Ka + Na + Asat
    "မင်း",  # Ma + Nga + Asat + Visarga
    # Stacked consonants (Pali)
    "တ္တာ",  # Ta + Virama + Ta + Aa
    "က္က",  # Ka + Virama + Ka
    "ဒ္ဓ",  # Da + Virama + Dha
    "န္ဒာ",  # Na + Virama + Da + Aa
    # Kinzi
    "င်္ကာ",  # Kinzi + Ka + Aa
    # Merged Kinzi
    "သင်္ဘော",  # Thingyay + Kinzi + Bha + Aw
    "အင်္ဂါ",  # A + Kinzi + Ga + Aa
    "အင်္ကျီ",  # A + Kinzi + Ka + Medial Ya + II
    # Great Sa
    "ဿ",  # Great Sa alone
    "ဿာ",  # Great Sa + Aa
    # Independent vowels
    "ဣ",  # Independent I
    "ဤ",  # Independent II
    "ဥ",  # Independent U
    # Particles
    "၌",  # Locative particle
    "၍",  # Yi particle
    # Anusvara combinations
    "ကိံ",  # Ka + I + Anusvara
    "ကံ",  # Ka + Anusvara (bare consonant)
    # Medial Wa + AA
    "ကွာ",  # Ka + Medial Wa + Aa
    # Dot Below final position
    "ပို့",  # Pa + I + Dot Below (already present but verifying)
    # Complex syllables
    "ကျွန်",  # Ka + Ya + Wa + Na + Asat
    "တော်",  # Ta + E + Aa + Asat (common word ending)
    "ကြောင်း",  # Ka + Ra + E + Aa + Nga + Asat + Visarga
]

INVALID_SYLLABLES = [
    # Empty
    "",
    # Floating diacritics
    "\u103b",  # Medial Ya alone
    "\u102c",  # Aa vowel alone
    "\u1031",  # E vowel alone
    # Corruption
    "ကကကက",  # 4 consecutive identical
    # Zero-width chars
    "က\u200bာ",  # Ka + ZWSP + Aa
    # Invalid start
    "\u1037က",  # Dot Below + Ka
    # Tall A + Aa together (mutually exclusive)
    "ကါာ",  # Ka + Tall A + Aa — invalid (custom: this tests exclusivity)
    # Multiple upper vowels
    "ကိီ",  # Ka + I + II
    # Multiple lower vowels
    "ကုူ",  # Ka + U + UU
    # Virama at end
    "က္",  # Ka + Virama at end
    # Great Sa + Asat
    "ဿ်",  # Great Sa + Asat
    # Double diacritics
    "ကျျ",  # Ka + double Ya
    # E + invalid combinations
    "ကေိ",  # Ka + E + I (invalid E combination)
    # Anusvara + invalid vowel
    "ကာံ",  # Ka + Aa + Anusvara (Aa not in allowed vowels)
    # E + Anusvara (invalid combination)
    "ကေံ",  # Ka + E + Anusvara (phonotactically impossible)
]


class TestStrictModeEquivalence:
    """Verify Python and Cython produce identical results in strict mode."""

    @pytest.mark.parametrize("syllable", VALID_SYLLABLES)
    def test_valid_syllables_match(self, syllable, py_validator, cy_validator):
        py_result = py_validator.validate(syllable)
        cy_result = cy_validator.validate(syllable)
        assert py_result == cy_result, (
            f"Divergence for {syllable!r}: Python={py_result}, Cython={cy_result}"
        )

    @pytest.mark.parametrize("syllable", INVALID_SYLLABLES)
    def test_invalid_syllables_match(self, syllable, py_validator, cy_validator):
        py_result = py_validator.validate(syllable)
        cy_result = cy_validator.validate(syllable)
        assert py_result == cy_result, (
            f"Divergence for {syllable!r}: Python={py_result}, Cython={cy_result}"
        )


class TestLenientModeEquivalence:
    """Verify Python and Cython produce identical results in lenient mode."""

    @pytest.mark.parametrize("syllable", VALID_SYLLABLES)
    def test_valid_syllables_match(self, syllable, py_lenient, cy_lenient):
        py_result = py_lenient.validate(syllable)
        cy_result = cy_lenient.validate(syllable)
        assert py_result == cy_result, (
            f"Divergence for {syllable!r}: Python={py_result}, Cython={cy_result}"
        )

    @pytest.mark.parametrize("syllable", INVALID_SYLLABLES)
    def test_invalid_syllables_match(self, syllable, py_lenient, cy_lenient):
        py_result = py_lenient.validate(syllable)
        cy_result = cy_lenient.validate(syllable)
        assert py_result == cy_result, (
            f"Divergence for {syllable!r}: Python={py_result}, Cython={cy_result}"
        )


class TestEdgeCaseEquivalence:
    """Edge cases that specifically test areas of known divergence risk."""

    @pytest.mark.parametrize(
        "syllable",
        [
            # Virama followed by non-consonant (M5 fix)
            "က္ိ",  # Virama + vowel (should be rejected)
            "က္ေ",  # Virama + E vowel (should be rejected)
            # Great Sa medial scope (H2 fix)
            "ဿ",  # Great Sa alone (valid)
            "ဿ်",  # Great Sa + Asat (invalid - M2 fix)
            # Kinzi edge cases
            "င်္ဿ",  # Kinzi + Great Sa
            "သင်္ချိုင်း",  # Long Kinzi segment (11 codepoints)
            # Stacked consonant validity
            "တ္တု",  # Valid stack
            "ဝိဇ္ဇာ",  # Valid Pali stack
        ],
    )
    def test_edge_cases_match(self, syllable, py_validator, cy_validator):
        py_result = py_validator.validate(syllable)
        cy_result = cy_validator.validate(syllable)
        assert py_result == cy_result, (
            f"Divergence for {syllable!r}: Python={py_result}, Cython={cy_result}"
        )
