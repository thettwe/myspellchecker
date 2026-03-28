"""Property-based tests for SyllableRuleValidator using Hypothesis.

Tests universal invariants and structural properties that must hold
for all inputs, using randomly generated Myanmar character sequences
and structured syllable generators.

Marked @pytest.mark.slow — skipped by default in fast test runs.
Run with: pytest tests/test_syllable_rules_property.py -m slow
"""

from __future__ import annotations

import sqlite3
import string
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from myspellchecker.core.constants import (
    ANUSVARA_ALLOWED_VOWELS,
    COMPATIBLE_HA,
    COMPATIBLE_RA,
    COMPATIBLE_WA,
    COMPATIBLE_YA,
    CONSONANTS,
    INDEPENDENT_VOWELS,
    MEDIALS,
    STACKING_EXCEPTIONS,
    TONE_MARKS,
    VALID_PARTICLES,
    VOWEL_SIGNS,
)
from myspellchecker.core.syllable_rules import _SyllableRuleValidatorPython

try:
    from myspellchecker.core.syllable_rules_c import (
        SyllableRuleValidator as CValidator,
    )

    HAS_CYTHON = True
except ImportError:
    HAS_CYTHON = False

# ── Constants ──

# Myanmar character alphabet for random generation (U+1000 to U+109F)
MYANMAR_CHARS = [chr(c) for c in range(0x1000, 0x10A0)]

# Core Myanmar chars most relevant for syllable validation
CORE_MYANMAR = [chr(c) for c in range(0x1000, 0x1050)]

# Sorted lists for deterministic sampling
CONSONANT_LIST = sorted(CONSONANTS)
MEDIAL_LIST = sorted(MEDIALS)
VOWEL_LIST = sorted(VOWEL_SIGNS)
TONE_LIST = sorted(TONE_MARKS)
IV_LIST = sorted(INDEPENDENT_VOWELS - {"\u1021"})  # Exclude Ah (consonant carrier)

VIRAMA = "\u1039"
ASAT = "\u103a"
DOT_BELOW = "\u1037"
VISARGA = "\u1038"
ANUSVARA = "\u1036"
VOWEL_E = "\u1031"
AA = "\u102c"

# Medial compatibility lookup
_MEDIAL_COMPAT = {
    "\u103b": COMPATIBLE_YA,
    "\u103c": COMPATIBLE_RA,
    "\u103d": COMPATIBLE_WA,
    "\u103e": COMPATIBLE_HA,
}

# Production DB path
PROD_DB = Path(__file__).parent.parent / "data" / "mySpellChecker_production.db"


# ── Hypothesis Strategies (Generators) ──


@st.composite
def myanmar_random_sequence(draw):
    """Random sequence of Myanmar characters (0-15 chars)."""
    length = draw(st.integers(min_value=0, max_value=15))
    return "".join(draw(st.sampled_from(CORE_MYANMAR)) for _ in range(length))


@st.composite
def valid_simple_syllable(draw):
    """Generate a structurally valid simple syllable: C + optional(M) + optional(V) + optional(T).

    Respects medial compatibility, vowel exclusivity, and tone mark rules.
    Does NOT generate kinzi, stacking, e-vowel, or great-sa patterns.
    """
    # Consonant (required)
    consonant = draw(st.sampled_from(CONSONANT_LIST))

    # Optional medial (respecting compatibility)
    use_medial = draw(st.booleans())
    medial = ""
    if use_medial:
        compatible = []
        for m, compat_set in _MEDIAL_COMPAT.items():
            if consonant in compat_set:
                compatible.append(m)
        if compatible:
            medial = draw(st.sampled_from(sorted(compatible)))

    # Optional vowel sign (one only, no e-vowel in this generator)
    use_vowel = draw(st.booleans())
    vowel = ""
    if use_vowel:
        excluded_vowels = {VOWEL_E}
        if medial == "\u103d":
            # Medial Wa: Tall AA (U+102B) invalid, U/UU (U+102F/U+1030) phonetically incompatible
            excluded_vowels.update({"\u102b", "\u102f", "\u1030"})
        candidates = sorted(VOWEL_SIGNS - excluded_vowels)
        if candidates:
            vowel = draw(st.sampled_from(candidates))

    # Optional final: consonant + asat OR tone mark
    # Cannot have both stop final + tone mark
    ending = draw(st.sampled_from(["none", "asat", "tone", "anusvara"]))

    suffix = ""
    if ending == "asat":
        # Final consonant + asat
        # Exclude U+1021 (Ah) — strict mode rejects Ah + Asat as non-standard
        # Exclude U+103F (Great Sa) — Great Sa is a self-contained conjunct,
        # cannot take asat as a coda consonant
        final_candidates = sorted(CONSONANTS - {"\u1021", "\u103f"})
        final_c = draw(st.sampled_from(final_candidates))
        suffix = final_c + ASAT
    elif ending == "tone":
        # Tone mark only (no asat)
        suffix = draw(st.sampled_from([DOT_BELOW, VISARGA]))
    elif ending == "anusvara":
        # Anusvara only with compatible vowels
        if vowel in ANUSVARA_ALLOWED_VOWELS or vowel == "":
            suffix = ANUSVARA
        # else: skip anusvara (incompatible vowel)

    return consonant + medial + vowel + suffix


@st.composite
def valid_e_vowel_syllable(draw):
    """Generate a valid e-vowel (ေ) syllable: C + (M) + ေ + ာ + optional(်).

    In Unicode encoding order, E-vowel (U+1031) comes AFTER the consonant
    (and any medials), even though it visually renders before the consonant.
    """
    consonant = draw(st.sampled_from(CONSONANT_LIST))
    use_medial = draw(st.booleans())
    medial = ""
    if use_medial:
        compatible = []
        for m, compat_set in _MEDIAL_COMPAT.items():
            if consonant in compat_set:
                # Exclude medial Wa — Tall AA + Wa conflict, and AA + Wa
                # would need careful handling
                if m != "\u103d":
                    compatible.append(m)
        if compatible:
            medial = draw(st.sampled_from(sorted(compatible)))

    # E-vowel + AA forms the "aw" vowel (ော)
    # Use regular AA (U+102C), not Tall AA (U+102B)
    use_asat = draw(st.booleans())
    asat_suffix = ASAT if use_asat else ""

    return consonant + medial + VOWEL_E + AA + asat_suffix


# Nga (U+1004) + virama pairs are kinzi patterns, NOT regular stacking.
# In proper Unicode, these use kinzi encoding (င်္C), not virama stacking (င္C).
_NGA = "\u1004"
_STACKING_PAIRS_NO_KINZI = sorted(
    (upper, lower) for upper, lower in STACKING_EXCEPTIONS if upper != _NGA
)


@st.composite
def stacking_syllable(draw):
    """Generate a virama stacking sequence: C1 + virama + C2.

    Excludes Nga-initial pairs which are kinzi patterns (handled separately).
    """
    pair = draw(st.sampled_from(_STACKING_PAIRS_NO_KINZI))
    return pair[0] + VIRAMA + pair[1]


# ── Phase 1: Universal Invariants ──


class TestUniversalInvariants:
    """Properties that must hold for ALL inputs."""

    @pytest.mark.slow
    @given(myanmar_random_sequence())
    @settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
    def test_determinism(self, text):
        """validate(s) must return the same result every time."""
        v = _SyllableRuleValidatorPython(strict=True)
        assert v.validate(text) == v.validate(text)

    @pytest.mark.slow
    @given(
        st.text(
            min_size=1,
            max_size=15,
            alphabet=st.sampled_from(CORE_MYANMAR),
        )
    )
    @settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
    def test_strict_subset_of_lenient(self, text):
        """If strict accepts, lenient MUST also accept."""
        strict = _SyllableRuleValidatorPython(strict=True)
        lenient = _SyllableRuleValidatorPython(strict=False)
        if strict.validate(text):
            assert lenient.validate(text), (
                f"Strict accepted but lenient rejected: {text!r} "
                f"(U+{' U+'.join(f'{ord(c):04X}' for c in text)})"
            )

    @given(st.text(min_size=1, max_size=10, alphabet=string.ascii_letters + string.digits))
    @settings(max_examples=200)
    def test_non_myanmar_always_invalid(self, text):
        """Pure ASCII/Latin text is never a valid Myanmar syllable."""
        v = _SyllableRuleValidatorPython(strict=True)
        assert not v.validate(text)

    @given(
        st.text(
            min_size=16,
            max_size=25,
            alphabet=st.sampled_from(CORE_MYANMAR),
        )
    )
    @settings(max_examples=200)
    def test_overlength_always_invalid(self, text):
        """Text longer than max_syllable_length is always invalid."""
        v = _SyllableRuleValidatorPython(strict=True)
        assert not v.validate(text)

    def test_empty_always_invalid(self):
        """Empty string is never valid."""
        v = _SyllableRuleValidatorPython(strict=True)
        assert not v.validate("")

    @pytest.mark.slow
    @given(myanmar_random_sequence())
    @settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
    def test_valid_starts_with_base(self, text):
        """If valid, first char must be consonant, IV, e-vowel, or valid particle."""
        v = _SyllableRuleValidatorPython(strict=True)
        if v.validate(text) and text:
            first = text[0]
            # VALID_PARTICLES (၌ ၍ ၎ ၏) are standalone logographic particles
            is_valid_start = (
                first in CONSONANTS
                or first in INDEPENDENT_VOWELS
                or first == VOWEL_E
                or text in VALID_PARTICLES
            )
            assert is_valid_start, (
                f"Valid syllable starts with U+{ord(first):04X}, "
                f"not a consonant/IV/e-vowel/particle: {text!r}"
            )


# ── Phase 2: Structured Generators ──


class TestStructuredGenerators:
    """Tests using generators that produce structurally valid/invalid syllables."""

    @pytest.mark.slow
    @given(valid_simple_syllable())
    @settings(max_examples=1000, suppress_health_check=[HealthCheck.too_slow])
    def test_valid_simple_syllable_accepted(self, syllable):
        """Every syllable from the valid generator MUST pass validation."""
        v = _SyllableRuleValidatorPython(strict=True)
        assert v.validate(syllable), (
            f"Valid generator produced rejected syllable: {syllable!r} "
            f"(U+{' U+'.join(f'{ord(c):04X}' for c in syllable)})"
        )

    @pytest.mark.slow
    @given(valid_e_vowel_syllable())
    @settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
    def test_valid_e_vowel_accepted(self, syllable):
        """Every e-vowel syllable from the generator MUST pass."""
        v = _SyllableRuleValidatorPython(strict=True)
        assert v.validate(syllable), (
            f"E-vowel generator produced rejected syllable: {syllable!r} "
            f"(U+{' U+'.join(f'{ord(c):04X}' for c in syllable)})"
        )

    @pytest.mark.slow
    @given(stacking_syllable())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_valid_stacking_accepted(self, syllable):
        """Every stacking pair from STACKING_EXCEPTIONS MUST pass."""
        v = _SyllableRuleValidatorPython(strict=True)
        assert v.validate(syllable), (
            f"Valid stacking pair rejected: {syllable!r} "
            f"(U+{' U+'.join(f'{ord(c):04X}' for c in syllable)})"
        )

    # ── Invalid generators ──

    @given(st.sampled_from(MEDIAL_LIST))
    def test_medial_first_always_invalid(self, medial):
        """A syllable starting with a medial is always invalid."""
        v = _SyllableRuleValidatorPython(strict=True)
        for c in CONSONANT_LIST[:5]:
            assert not v.validate(medial + c)

    @given(st.sampled_from(CONSONANT_LIST))
    def test_virama_at_end_always_invalid(self, consonant):
        """Virama at end of syllable is always invalid."""
        v = _SyllableRuleValidatorPython(strict=True)
        assert not v.validate(consonant + VIRAMA)

    @given(
        st.sampled_from(CONSONANT_LIST),
        st.sampled_from([DOT_BELOW, VISARGA]),
    )
    def test_dot_and_visarga_exclusive(self, consonant, _tone):
        """Dot below and visarga cannot co-occur."""
        v = _SyllableRuleValidatorPython(strict=True)
        assert not v.validate(consonant + DOT_BELOW + VISARGA)
        assert not v.validate(consonant + VISARGA + DOT_BELOW)


# ── Phase 3: Production DB Oracle ──


class TestProductionDBOracle:
    """Validate every syllable in the production DB passes structural checks."""

    @pytest.mark.slow
    @pytest.mark.skipif(not PROD_DB.exists(), reason="Production DB not available")
    def test_all_db_syllables_pass_lenient(self):
        """Every syllable in the production DB should pass lenient validation.

        Failures indicate either validator false positives or DB noise.
        """
        conn = sqlite3.connect(str(PROD_DB))
        cursor = conn.execute("SELECT syllable FROM syllables")
        syllables = [row[0] for row in cursor.fetchall()]
        conn.close()

        v = _SyllableRuleValidatorPython(strict=False)
        failures = []
        for syl in syllables:
            if not v.validate(syl):
                failures.append(f"{syl!r} (U+{' U+'.join(f'{ord(c):04X}' for c in syl)})")

        # Report but don't fail on small number of noise entries
        if failures:
            failure_rate = len(failures) / len(syllables) * 100
            # Allow up to 5% noise in DB
            assert failure_rate < 5.0, (
                f"{len(failures)}/{len(syllables)} syllables ({failure_rate:.1f}%) "
                f"failed lenient validation. First 20:\n" + "\n".join(failures[:20])
            )

    @pytest.mark.slow
    @pytest.mark.skipif(not PROD_DB.exists(), reason="Production DB not available")
    def test_high_freq_syllables_pass_strict(self):
        """High-frequency syllables (freq >= 1000) must pass strict validation.

        These are the most common syllables — any rejection is a false positive.
        """
        conn = sqlite3.connect(str(PROD_DB))
        cursor = conn.execute("SELECT syllable, frequency FROM syllables WHERE frequency >= 1000")
        syllables = cursor.fetchall()
        conn.close()

        v = _SyllableRuleValidatorPython(strict=True)
        failures = []
        for syl, freq in syllables:
            if not v.validate(syl):
                failures.append(f"{syl!r} (freq={freq})")

        assert not failures, (
            f"{len(failures)} high-frequency syllables failed strict validation:\n"
            + "\n".join(failures[:20])
        )


# ── Phase 4: Python/Cython Parity on Random Inputs ──


@pytest.mark.skipif(not HAS_CYTHON, reason="Cython module not built")
class TestParityOnRandomInputs:
    """Python and Cython must agree on randomly generated inputs."""

    @pytest.mark.slow
    @given(myanmar_random_sequence())
    @settings(max_examples=1000, suppress_health_check=[HealthCheck.too_slow])
    def test_parity_random_strict(self, text):
        """Python and Cython must agree on random Myanmar sequences (strict)."""
        py = _SyllableRuleValidatorPython(strict=True)
        cy = CValidator(strict=True)
        py_r = py.validate(text)
        cy_r = cy.validate(text)
        assert py_r == cy_r, (
            f"Parity mismatch (strict) on {text!r} "
            f"(U+{' U+'.join(f'{ord(c):04X}' for c in text)}): "
            f"Python={py_r}, Cython={cy_r}"
        )

    @pytest.mark.slow
    @given(myanmar_random_sequence())
    @settings(max_examples=1000, suppress_health_check=[HealthCheck.too_slow])
    def test_parity_random_lenient(self, text):
        """Python and Cython must agree on random Myanmar sequences (lenient)."""
        py = _SyllableRuleValidatorPython(strict=False)
        cy = CValidator(strict=False)
        py_r = py.validate(text)
        cy_r = cy.validate(text)
        assert py_r == cy_r, f"Parity mismatch (lenient) on {text!r}: Python={py_r}, Cython={cy_r}"

    @pytest.mark.slow
    @given(valid_simple_syllable())
    @settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
    def test_parity_valid_syllables(self, syllable):
        """Both implementations must agree on structurally valid syllables."""
        py = _SyllableRuleValidatorPython(strict=True)
        cy = CValidator(strict=True)
        assert py.validate(syllable) == cy.validate(syllable), (
            f"Parity mismatch on valid syllable: {syllable!r}"
        )
