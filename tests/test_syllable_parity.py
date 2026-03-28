"""Comprehensive parity test between Python and Cython SyllableRuleValidator.

Ensures both implementations produce identical results on a wide range of
inputs: valid syllables, invalid syllables, edge cases (kinzi, stacking,
medials, independent vowels, great sa), and random Myanmar character sequences.
"""

import pytest

from myspellchecker.core.syllable_rules import _SyllableRuleValidatorPython

try:
    from myspellchecker.core.syllable_rules_c import (
        SyllableRuleValidator as CValidator,
    )

    HAS_CYTHON = True
except ImportError:
    HAS_CYTHON = False

pytestmark = pytest.mark.skipif(not HAS_CYTHON, reason="Cython module not built")

# ── Character constants ──
C = {
    "ka": "\u1000",
    "kha": "\u1001",
    "ga": "\u1002",
    "gha": "\u1003",
    "nga": "\u1004",
    "ca": "\u1005",
    "cha": "\u1006",
    "ja": "\u1007",
    "jha": "\u1008",
    "nya_sm": "\u1009",
    "nya": "\u100a",
    "ta_r": "\u100b",
    "tha_r": "\u100c",
    "da_r": "\u100d",
    "dha_r": "\u100e",
    "na_r": "\u100f",
    "ta": "\u1010",
    "tha": "\u1011",
    "da": "\u1012",
    "dha": "\u1013",
    "na": "\u1014",
    "pa": "\u1015",
    "pha": "\u1016",
    "ba": "\u1017",
    "bha": "\u1018",
    "ma": "\u1019",
    "ya": "\u101a",
    "ra": "\u101b",
    "la": "\u101c",
    "wa": "\u101d",
    "sa": "\u101e",
    "ha": "\u101f",
    "lla": "\u1020",
    "ah": "\u1021",
}
M = {"ya": "\u103b", "ra": "\u103c", "wa": "\u103d", "ha": "\u103e"}
V = {
    "aa": "\u102c",
    "tall_aa": "\u102b",
    "i": "\u102d",
    "ii": "\u102e",
    "u": "\u102f",
    "uu": "\u1030",
    "e": "\u1031",
    "ai": "\u1032",
}
T = {"asat": "\u103a", "visarga": "\u1038", "dot": "\u1037", "anusvara": "\u1036"}
VIRAMA = "\u1039"
GREAT_SA = "\u103f"

# Independent vowels
IV = {
    "i": "\u1023",
    "ii": "\u1024",
    "u": "\u1025",
    "uu": "\u1026",
    "e": "\u1027",
    "o": "\u1029",
    "au": "\u102a",
}


# ── Test data ──

# Valid syllables spanning all major patterns
VALID_SYLLABLES = [
    # Basic consonant
    C["ka"],
    # Consonant + asat (closed syllable)
    C["ka"] + T["asat"],
    # Consonant + vowel
    C["ka"] + V["aa"],
    C["ka"] + V["i"],
    C["ka"] + V["u"],
    C["ma"] + V["aa"],
    # Consonant + vowel + tone
    C["ka"] + V["aa"] + T["visarga"],
    C["ka"] + V["i"] + T["dot"],
    # Consonant + medial + vowel
    C["ka"] + M["ya"] + V["aa"],
    C["ka"] + M["ra"] + V["i"],
    C["ka"] + M["wa"] + V["aa"],
    C["ka"] + M["ha"] + V["aa"],
    # Multi-medial
    C["ka"] + M["ya"] + M["wa"],
    C["ka"] + M["ra"] + M["wa"],
    # E-vowel (pre-posed)
    V["e"] + C["ka"] + V["aa"],
    V["e"] + C["ka"] + V["aa"] + T["asat"],  # ကော် pattern
    # Kinzi pattern (Nga + Asat + Virama + Consonant)
    C["nga"] + T["asat"] + VIRAMA + C["ka"],
    C["nga"] + T["asat"] + VIRAMA + C["ga"],
    # Virama stacking (Pali/Sanskrit)
    C["ka"] + VIRAMA + C["ka"],  # က္က gemination
    C["ta"] + VIRAMA + C["ta"],  # တ္တ gemination
    C["na"] + VIRAMA + C["da"],  # န္ဒ cross-row
    # Complex valid syllables from real words
    C["ka"] + M["ya"] + V["e"] + V["aa"] + C["nga"] + T["asat"] + T["visarga"],  # ကျောင်း
    C["ma"] + M["ra"] + C["na"] + T["asat"],  # မြန်
    # Consonant + anusvara
    C["ka"] + T["anusvara"],
    # Great Sa
    GREAT_SA,
    # Independent vowels
    IV["u"],
    IV["e"],
    # Ah (U+1021) acts as consonant carrier
    C["ah"] + V["aa"],
    C["ah"] + M["ya"],
]

# Invalid syllables
INVALID_SYLLABLES = [
    # Empty
    "",
    # Starts with medial
    M["ya"] + C["ka"],
    # Starts with vowel sign
    V["aa"] + C["ka"],
    # Double upper vowels (exclusivity)
    C["ka"] + V["i"] + V["ii"],
    # Double lower vowels (exclusivity)
    C["ka"] + V["u"] + V["uu"],
    # Tall A + AA exclusivity
    C["ka"] + V["tall_aa"] + V["aa"],
    # Virama at end
    C["ka"] + VIRAMA,
    # Virama followed by non-consonant
    C["ka"] + VIRAMA + V["aa"],
    # Dot below + visarga in same syllable
    C["ka"] + T["dot"] + T["visarga"],
    # Too long
    "က" * 16,
    # Corruption (4+ identical)
    "ကကကက",
    # Asat preceded by vowel
    C["ka"] + V["i"] + T["asat"],
    # Zero-width characters
    C["ka"] + "\u200b" + V["aa"],  # ZWSP
    # Invalid medial on consonant (NGA + ya-pin is invalid)
    C["nga"] + M["ra"],
    # Tall A after medial Wa
    C["ka"] + M["wa"] + V["tall_aa"],
    # Invalid e-vowel combinations
    V["e"] + C["ka"] + V["i"],
]

# Edge cases that test boundary conditions
EDGE_CASES = [
    # Single characters
    C["ka"],
    C["ah"],
    T["asat"],
    V["e"],
    M["ya"],
    VIRAMA,
    GREAT_SA,
    # Anusvara combinations
    C["ka"] + V["i"] + T["anusvara"],
    C["ka"] + T["anusvara"] + T["dot"],
    # Great Sa rules
    C["ka"] + GREAT_SA,
    GREAT_SA + V["aa"],
    # Double diacritics
    C["ka"] + M["ya"] + M["ya"],
    # Stacking with non-vagga consonants
    C["ha"] + VIRAMA + C["ma"],  # ဟ္မ valid
    C["sa"] + VIRAMA + C["ta"],  # သ္တ valid
    # Independent vowel + medial (invalid)
    IV["u"] + M["ya"],
    # Independent vowel + dependent vowel (invalid)
    IV["u"] + V["aa"],
    # Independent vowel + tone (valid)
    IV["u"] + T["visarga"],
    # Virama count
    C["ka"] + VIRAMA + C["ta"] + VIRAMA + C["ka"],
]

# All inputs combined
ALL_TEST_INPUTS = VALID_SYLLABLES + INVALID_SYLLABLES + EDGE_CASES


@pytest.fixture(params=["strict", "lenient"])
def validator_pair(request):
    """Create matched Python and Cython validators."""
    strict = request.param == "strict"
    py = _SyllableRuleValidatorPython(strict=strict)
    cy = CValidator(strict=strict)
    return py, cy, request.param


class TestPythonCythonParity:
    """Ensure Python and Cython implementations produce identical results."""

    @pytest.mark.parametrize("syllable", ALL_TEST_INPUTS)
    def test_parity_on_curated_inputs(self, validator_pair, syllable):
        """Both implementations must agree on all curated test inputs."""
        py_val, cy_val, mode = validator_pair
        py_result = py_val.validate(syllable)
        cy_result = cy_val.validate(syllable)
        assert py_result == cy_result, (
            f"Parity mismatch in {mode} mode for {syllable!r} "
            f"(U+{' U+'.join(f'{ord(c):04X}' for c in syllable)}): "
            f"Python={py_result}, Cython={cy_result}"
        )

    def test_parity_on_all_single_consonants(self, validator_pair):
        """Both must agree on every single Myanmar consonant."""
        py_val, cy_val, _ = validator_pair
        for code in range(0x1000, 0x1022):
            char = chr(code)
            assert py_val.validate(char) == cy_val.validate(char), (
                f"Mismatch on consonant U+{code:04X}"
            )

    def test_parity_on_consonant_plus_each_medial(self, validator_pair):
        """Both must agree on consonant + medial for all combinations."""
        py_val, cy_val, _ = validator_pair
        consonants = [chr(c) for c in range(0x1000, 0x1022)]
        medials = list(M.values())
        for con in consonants:
            for med in medials:
                syl = con + med
                assert py_val.validate(syl) == cy_val.validate(syl), (
                    f"Mismatch on {syl!r} (U+{ord(con):04X} + U+{ord(med):04X})"
                )

    def test_parity_on_stacking_pairs_exhaustive(self, validator_pair):
        """Both must agree on all virama stacking combinations C1+virama+C2."""
        py_val, cy_val, _ = validator_pair
        consonants = [chr(c) for c in range(0x1000, 0x1022)]
        mismatches = []
        for c1 in consonants:
            for c2 in consonants:
                syl = c1 + VIRAMA + c2
                py_r = py_val.validate(syl)
                cy_r = cy_val.validate(syl)
                if py_r != cy_r:
                    mismatches.append(
                        f"U+{ord(c1):04X}+virama+U+{ord(c2):04X}: Py={py_r}, Cy={cy_r}"
                    )
        assert not mismatches, f"{len(mismatches)} stacking parity mismatches:\n" + "\n".join(
            mismatches[:10]
        )

    def test_parity_on_configurable_stacking_pairs(self):
        """Both must agree when custom stacking pairs are provided."""
        # Only allow gemination (same consonant)
        custom_pairs = {(chr(c), chr(c)) for c in range(0x1000, 0x1022)}
        py = _SyllableRuleValidatorPython(strict=True, stacking_pairs=custom_pairs)
        cy = CValidator(strict=True, stacking_pairs=custom_pairs)

        # Gemination should be valid
        assert py.validate(C["ka"] + VIRAMA + C["ka"]) is True
        assert cy.validate(C["ka"] + VIRAMA + C["ka"]) is True
        # Cross-row should be invalid with custom pairs
        assert py.validate(C["ka"] + VIRAMA + C["ta"]) == cy.validate(C["ka"] + VIRAMA + C["ta"])

    def test_parity_on_independent_vowels(self, validator_pair):
        """Both must agree on independent vowel combinations."""
        py_val, cy_val, _ = validator_pair
        for iv in IV.values():
            # Bare independent vowel
            assert py_val.validate(iv) == cy_val.validate(iv)
            # With tone marks
            for tone in T.values():
                syl = iv + tone
                assert py_val.validate(syl) == cy_val.validate(syl), f"Mismatch on {syl!r}"
            # With medials (should be invalid)
            for med in M.values():
                syl = iv + med
                assert py_val.validate(syl) == cy_val.validate(syl), f"Mismatch on {syl!r}"
