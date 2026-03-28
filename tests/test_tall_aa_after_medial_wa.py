"""
Tests for Tall AA after Medial Wa validation and normalization.

Issue: https://github.com/thettwe/my-spellchecker/issues/1357

In standard Myanmar orthography, after medial ွ (wa-hswe, U+103D),
the vowel is always written as ာ (U+102C), never ါ (U+102B).
"""

import pytest

from myspellchecker.core.syllable_rules import (
    SyllableRuleValidator,
    _SyllableRuleValidatorPython,
)
from myspellchecker.text.normalize import (
    normalize,
    normalize_for_lookup,
    normalize_tall_aa_after_wa,
)

# Unicode constants for readability
MEDIAL_WA = "\u103d"
TALL_AA = "\u102b"
AA = "\u102c"
VISARGA = "\u1038"
DOT_BELOW = "\u1037"
ANUSVARA = "\u1036"
ASAT = "\u103a"
MEDIAL_YA = "\u103b"
MEDIAL_RA = "\u103c"
MEDIAL_HA = "\u103e"

# =============================================================================
# Validation Tests
# =============================================================================


class TestTallAaAfterMedialWaValidation:
    """Test _check_tall_aa_after_medial_wa() validation rule."""

    @pytest.fixture
    def validator(self):
        return SyllableRuleValidator()

    @pytest.fixture
    def strict_validator(self):
        return SyllableRuleValidator(strict=True)

    # --- Invalid: ွ + ါ (should be rejected) ---

    def test_reject_wa_tall_aa(self, validator):
        """ပွါ should be invalid (ပ + ွ + ါ)."""
        assert validator.validate("\u1015\u103d\u102b") is False

    def test_reject_wa_tall_aa_visarga(self, validator):
        """ပွါး should be invalid (ပ + ွ + ါ + း)."""
        assert validator.validate("\u1015\u103d\u102b\u1038") is False

    def test_reject_ga_wa_tall_aa(self, validator):
        """ဂွါ should be invalid (ဂ + ွ + ါ)."""
        assert validator.validate("\u1002\u103d\u102b") is False

    def test_reject_da_wa_tall_aa(self, validator):
        """ဒွါ should be invalid (ဒ + ွ + ါ)."""
        assert validator.validate("\u1012\u103d\u102b") is False

    def test_reject_tha_wa_tall_aa(self, validator):
        """သွါး should be invalid (သ + ွ + ါ + း)."""
        assert validator.validate("\u101e\u103d\u102b\u1038") is False

    def test_reject_ka_wa_tall_aa(self, validator):
        """ကွါ should be invalid (က + ွ + ါ)."""
        assert validator.validate("\u1000\u103d\u102b") is False

    # --- Valid: ွ + ာ (should be accepted) ---

    def test_accept_wa_aa(self, validator):
        """ပွား should be valid (ပ + ွ + ာ + း)."""
        assert validator.validate("\u1015\u103d\u102c\u1038") is True

    def test_accept_ga_wa_aa(self, validator):
        """ဂွာ should be valid (ဂ + ွ + ာ)."""
        assert validator.validate("\u1002\u103d\u102c") is True

    def test_accept_tha_wa_aa_visarga(self, validator):
        """သွား should be valid (သ + ွ + ာ + း)."""
        assert validator.validate("\u101e\u103d\u102c\u1038") is True

    def test_accept_da_wa_aa(self, validator):
        """ဒွာ should be valid (ဒ + ွ + ာ)."""
        assert validator.validate("\u1012\u103d\u102c") is True

    # --- Valid: ါ without ွ (should still be accepted) ---

    def test_accept_tall_aa_without_wa(self, validator):
        """ကါ should be valid (no medial Wa)."""
        assert validator.validate("\u1000\u102b") is True

    def test_accept_tall_aa_with_other_medial(self, validator):
        """ကျါ should be valid (medial Ya, not Wa)."""
        assert validator.validate("\u1000\u103b\u102b") is True

    # --- Valid: ွ without ါ (should still be accepted) ---

    def test_accept_wa_without_tall_aa(self, validator):
        """ကွ should be valid (no Tall AA)."""
        assert validator.validate("\u1000\u103d") is True

    def test_accept_wa_with_other_vowel(self, validator):
        """ကွိ should be valid (Wa + I vowel, not Tall AA)."""
        assert validator.validate("\u1000\u103d\u102d") is True

    # --- Strict mode ---

    def test_reject_wa_tall_aa_strict(self, strict_validator):
        """ပွါ should be invalid in strict mode too."""
        assert strict_validator.validate("\u1015\u103d\u102b") is False

    # --- Real-world error cases from gold standard review ---

    def test_reject_real_error_dhaa_taa(self, validator):
        """ဓွါ variant should be invalid (if it existed as syllable)."""
        # ဓ + ွ + ါ
        assert validator.validate("\u1013\u103d\u102b") is False

    # --- Parametrized: all consonants with medial Wa + Tall AA must be invalid ---

    @pytest.mark.parametrize(
        "consonant,name",
        [
            ("\u1000", "ka"),
            ("\u1001", "kha"),
            ("\u1002", "ga"),
            ("\u1003", "gha"),
            ("\u1004", "nga"),
            ("\u1005", "ca"),
            ("\u1006", "cha"),
            ("\u1007", "ja"),
            ("\u1008", "jha"),
            ("\u1009", "nya"),
            ("\u100a", "nnya"),
            ("\u100b", "tta"),
            ("\u100c", "ttha"),
            ("\u100d", "dda"),
            ("\u100e", "ddha"),
            ("\u100f", "nna"),
            ("\u1010", "ta"),
            ("\u1011", "tha"),
            ("\u1012", "da"),
            ("\u1013", "dha"),
            ("\u1014", "na"),
            ("\u1015", "pa"),
            ("\u1016", "pha"),
            ("\u1017", "ba"),
            ("\u1018", "bha"),
            ("\u1019", "ma"),
            ("\u101a", "ya"),
            ("\u101b", "ra"),
            ("\u101c", "la"),
            ("\u101d", "wa"),
            ("\u101e", "sa"),
            ("\u101f", "ha"),
            ("\u1020", "lla"),
        ],
    )
    def test_reject_all_consonants_with_wa_tall_aa(self, validator, consonant, name):
        """Every consonant + ွ + ါ must be invalid."""
        syllable = consonant + MEDIAL_WA + TALL_AA
        assert validator.validate(syllable) is False, (
            f"{name} ({consonant}) + medial Wa + Tall AA should be rejected"
        )

    @pytest.mark.parametrize(
        "consonant,name",
        [
            ("\u1000", "ka"),
            ("\u1002", "ga"),
            ("\u1012", "da"),
            ("\u1014", "na"),
            ("\u1015", "pa"),
            ("\u1019", "ma"),
            ("\u101c", "la"),
            ("\u101e", "sa"),
        ],
    )
    def test_accept_all_consonants_with_wa_aa(self, validator, consonant, name):
        """Consonant + ွ + ာ should be valid (correct form)."""
        syllable = consonant + MEDIAL_WA + AA
        assert validator.validate(syllable) is True, (
            f"{name} ({consonant}) + medial Wa + AA should be accepted"
        )

    # --- Combined medials: ွ with other medials + ါ must still be invalid ---

    def test_reject_ya_wa_tall_aa(self, validator):
        """ကျွါ should be invalid (ျ + ွ + ါ)."""
        # ka + medial Ya + medial Wa + Tall AA
        assert validator.validate("\u1000\u103b\u103d\u102b") is False

    def test_reject_ha_wa_tall_aa(self, validator):
        """ငွှါ should be invalid (ွ + ှ + ါ)."""
        # nga + medial Wa + medial Ha + Tall AA
        assert validator.validate("\u1004\u103d\u103e\u102b") is False

    # --- Combined medials: ွ with other medials + ာ should be valid ---

    def test_accept_ya_wa_aa(self, validator):
        """ကျွာ should be valid (ျ + ွ + ာ)."""
        assert validator.validate("\u1000\u103b\u103d\u102c") is True

    def test_accept_ha_wa_aa(self, validator):
        """ငွှာ should be valid (ွ + ှ + ာ)."""
        assert validator.validate("\u1004\u103d\u103e\u102c") is True

    # --- Medial Wa + Tall AA with trailing diacritics ---

    def test_reject_wa_tall_aa_dot_below(self, validator):
        """ကွါ့ should be invalid (ွ + ါ + ့)."""
        assert validator.validate("\u1000\u103d\u102b\u1037") is False

    def test_reject_wa_tall_aa_anusvara(self, validator):
        """ကွါံ should be invalid (ွ + ါ + ံ)."""
        assert validator.validate("\u1000\u103d\u102b\u1036") is False

    # --- Tall AA with RA medial (not Wa) should be valid ---

    def test_accept_ra_tall_aa(self, validator):
        """ကြါ should be valid (medial Ra, not Wa)."""
        assert validator.validate("\u1000\u103c\u102b") is True

    def test_accept_ha_tall_aa(self, validator):
        """ငှါ should be valid (medial Ha, not Wa)."""
        assert validator.validate("\u1004\u103e\u102b") is True

    # --- Medial Wa with non-AA vowels (should be valid) ---

    def test_accept_wa_e_vowel(self, validator):
        """ကွေ should be valid (ွ + ေ)."""
        assert validator.validate("\u1000\u103d\u1031") is True

    def test_accept_wa_ii_vowel(self, validator):
        """ကွီ should be valid (ွ + ီ)."""
        assert validator.validate("\u1000\u103d\u102e") is True

    # --- Strict mode parametrized ---

    def test_reject_wa_tall_aa_strict_with_visarga(self, strict_validator):
        """ပွါး should be invalid in strict mode too."""
        assert strict_validator.validate("\u1015\u103d\u102b\u1038") is False

    def test_accept_wa_aa_strict(self, strict_validator):
        """ပွား should be valid in strict mode."""
        assert strict_validator.validate("\u1015\u103d\u102c\u1038") is True


# =============================================================================
# Validation Tests: Python implementation specifically
# =============================================================================


class TestTallAaAfterMedialWaPythonImpl:
    """Test _check_tall_aa_after_medial_wa() on the pure-Python implementation."""

    @pytest.fixture
    def py_validator(self):
        return _SyllableRuleValidatorPython()

    @pytest.fixture
    def py_strict_validator(self):
        return _SyllableRuleValidatorPython(strict=True)

    def test_reject_wa_tall_aa_python(self, py_validator):
        """Python impl: ပွါ should be invalid."""
        assert py_validator.validate("\u1015\u103d\u102b") is False

    def test_accept_wa_aa_python(self, py_validator):
        """Python impl: ပွား should be valid."""
        assert py_validator.validate("\u1015\u103d\u102c\u1038") is True

    def test_accept_tall_aa_without_wa_python(self, py_validator):
        """Python impl: ကါ should be valid (no medial Wa)."""
        assert py_validator.validate("\u1000\u102b") is True

    def test_accept_wa_without_tall_aa_python(self, py_validator):
        """Python impl: ကွ should be valid (no Tall AA)."""
        assert py_validator.validate("\u1000\u103d") is True

    def test_reject_wa_tall_aa_strict_python(self, py_strict_validator):
        """Python impl: ပွါ should be invalid in strict mode."""
        assert py_strict_validator.validate("\u1015\u103d\u102b") is False

    def test_check_method_directly_rejects(self, py_validator):
        """Directly call _check_tall_aa_after_medial_wa for rejection."""
        # ပွါ = Pa + medial Wa + Tall AA
        assert py_validator._check_tall_aa_after_medial_wa("\u1015\u103d\u102b") is False

    def test_check_method_directly_accepts_no_wa(self, py_validator):
        """Directly call _check_tall_aa_after_medial_wa: no medial Wa → True."""
        assert py_validator._check_tall_aa_after_medial_wa("\u1000\u102b") is True

    def test_check_method_directly_accepts_no_tall_aa(self, py_validator):
        """Directly call _check_tall_aa_after_medial_wa: no Tall AA → True."""
        assert py_validator._check_tall_aa_after_medial_wa("\u1000\u103d\u102c") is True

    def test_check_method_directly_accepts_neither(self, py_validator):
        """Directly call: neither medial Wa nor Tall AA → True."""
        assert py_validator._check_tall_aa_after_medial_wa("\u1000") is True


# =============================================================================
# Validation: Cython vs Python parity
# =============================================================================


class TestTallAaCythonPythonParity:
    """Ensure Cython and Python implementations agree on all cases."""

    @pytest.fixture
    def cython_or_default(self):
        """SyllableRuleValidator (may be Cython if compiled)."""
        return SyllableRuleValidator()

    @pytest.fixture
    def python_only(self):
        """Pure Python implementation."""
        return _SyllableRuleValidatorPython()

    @pytest.mark.parametrize(
        "syllable,description",
        [
            # Invalid: medial Wa + Tall AA
            ("\u1015\u103d\u102b", "pa+wa+tall_aa"),
            ("\u1015\u103d\u102b\u1038", "pa+wa+tall_aa+visarga"),
            ("\u1002\u103d\u102b", "ga+wa+tall_aa"),
            ("\u101e\u103d\u102b\u1038", "sa+wa+tall_aa+visarga"),
            ("\u1000\u103b\u103d\u102b", "ka+ya+wa+tall_aa"),
            ("\u1004\u103d\u103e\u102b", "nga+wa+ha+tall_aa"),
            # Valid: medial Wa + AA
            ("\u1015\u103d\u102c\u1038", "pa+wa+aa+visarga"),
            ("\u1002\u103d\u102c", "ga+wa+aa"),
            ("\u101e\u103d\u102c\u1038", "sa+wa+aa+visarga"),
            # Valid: Tall AA without medial Wa
            ("\u1000\u102b", "ka+tall_aa"),
            ("\u1000\u103b\u102b", "ka+ya+tall_aa"),
            # Valid: medial Wa without Tall AA
            ("\u1000\u103d", "ka+wa"),
            ("\u1000\u103d\u102d", "ka+wa+i_vowel"),
        ],
    )
    def test_parity(self, cython_or_default, python_only, syllable, description):
        """Cython and Python must produce identical results."""
        cython_result = cython_or_default.validate(syllable)
        python_result = python_only.validate(syllable)
        assert cython_result == python_result, (
            f"Parity mismatch for {description}: Cython={cython_result}, Python={python_result}"
        )


# =============================================================================
# Validation: Real-world error syllables from gold standard review
# =============================================================================


class TestRealWorldErrorSyllables:
    """Test the 5 real-world errors from the gold standard database review.

    These were the actual errors found in the syllable database that
    motivated the _check_tall_aa_after_medial_wa rule (issue #1357).
    """

    @pytest.fixture
    def validator(self):
        return SyllableRuleValidator()

    @pytest.mark.parametrize(
        "wrong,correct,description",
        [
            # wrong form (ွ+ါ) → correct form (ွ+ာ)
            ("\u1015\u103d\u102b\u1038", "\u1015\u103d\u102c\u1038", "ပွါး→ပွား"),
            ("\u1002\u103d\u102b", "\u1002\u103d\u102c", "ဂွါ→ဂွာ"),
            ("\u1012\u103d\u102b", "\u1012\u103d\u102c", "ဒွါ→ဒွာ"),
            ("\u101e\u103d\u102b\u1038", "\u101e\u103d\u102c\u1038", "သွါး→သွား"),
            ("\u1000\u103d\u102b", "\u1000\u103d\u102c", "ကွါ→ကွာ"),
        ],
    )
    def test_wrong_form_rejected(self, validator, wrong, correct, description):
        """Wrong form (ွ+ါ) should be rejected by validation."""
        assert validator.validate(wrong) is False, (
            f"Wrong form {description.split('→')[0]} should be rejected"
        )

    @pytest.mark.parametrize(
        "wrong,correct,description",
        [
            ("\u1015\u103d\u102b\u1038", "\u1015\u103d\u102c\u1038", "ပွါး→ပွား"),
            ("\u1002\u103d\u102b", "\u1002\u103d\u102c", "ဂွါ→ဂွာ"),
            ("\u1012\u103d\u102b", "\u1012\u103d\u102c", "ဒွါ→ဒွာ"),
            ("\u101e\u103d\u102b\u1038", "\u101e\u103d\u102c\u1038", "သွါး→သွား"),
            ("\u1000\u103d\u102b", "\u1000\u103d\u102c", "ကွါ→ကွာ"),
        ],
    )
    def test_correct_form_accepted(self, validator, wrong, correct, description):
        """Correct form (ွ+ာ) should be accepted by validation."""
        assert validator.validate(correct) is True, (
            f"Correct form {description.split('→')[1]} should be accepted"
        )

    @pytest.mark.parametrize(
        "wrong,correct,description",
        [
            ("\u1015\u103d\u102b\u1038", "\u1015\u103d\u102c\u1038", "ပွါး→ပွား"),
            ("\u1002\u103d\u102b", "\u1002\u103d\u102c", "ဂွါ→ဂွာ"),
            ("\u1012\u103d\u102b", "\u1012\u103d\u102c", "ဒွါ→ဒွာ"),
            ("\u101e\u103d\u102b\u1038", "\u101e\u103d\u102c\u1038", "သွါး→သွား"),
            ("\u1000\u103d\u102b", "\u1000\u103d\u102c", "ကွါ→ကွာ"),
        ],
    )
    def test_normalization_fixes_wrong_to_correct(self, wrong, correct, description):
        """Normalization should convert wrong form to correct form."""
        result = normalize_tall_aa_after_wa(wrong)
        assert result == correct, f"Normalization failed for {description}: got {result!r}"

    @pytest.mark.parametrize(
        "wrong,correct,description",
        [
            ("\u1015\u103d\u102b\u1038", "\u1015\u103d\u102c\u1038", "ပွါး→ပွား"),
            ("\u1002\u103d\u102b", "\u1002\u103d\u102c", "ဂွါ→ဂွာ"),
            ("\u1012\u103d\u102b", "\u1012\u103d\u102c", "ဒွါ→ဒွာ"),
            ("\u101e\u103d\u102b\u1038", "\u101e\u103d\u102c\u1038", "သွါး→သွား"),
            ("\u1000\u103d\u102b", "\u1000\u103d\u102c", "ကွါ→ကွာ"),
        ],
    )
    def test_normalize_then_validate_passes(self, validator, wrong, correct, description):
        """After normalization, the result should pass validation."""
        normalized = normalize_tall_aa_after_wa(wrong)
        assert validator.validate(normalized) is True, (
            f"Normalized form of {description} should pass validation"
        )


# =============================================================================
# Normalization Tests
# =============================================================================


class TestNormalizeTallAaAfterWa:
    """Test normalize_tall_aa_after_wa() normalization function."""

    def test_correct_wa_tall_aa(self):
        """ွ+ါ should be normalized to ွ+ာ."""
        # ပွါး → ပွား
        result = normalize_tall_aa_after_wa("\u1015\u103d\u102b\u1038")
        assert result == "\u1015\u103d\u102c\u1038"

    def test_correct_multiple_occurrences(self):
        """Multiple ွ+ါ in same text should all be corrected."""
        text = "\u1015\u103d\u102b\u1038 \u1002\u103d\u102b"  # ပွါး ဂွါ
        result = normalize_tall_aa_after_wa(text)
        assert result == "\u1015\u103d\u102c\u1038 \u1002\u103d\u102c"  # ပွား ဂွာ

    def test_no_change_correct_text(self):
        """Already correct text should be unchanged."""
        text = "\u1015\u103d\u102c\u1038"  # ပွား (already correct)
        assert normalize_tall_aa_after_wa(text) == text

    def test_no_change_tall_aa_without_wa(self):
        """ါ without preceding ွ should be left alone."""
        text = "\u1000\u102b"  # ကါ
        assert normalize_tall_aa_after_wa(text) == text

    def test_empty_string(self):
        """Empty string should return empty."""
        assert normalize_tall_aa_after_wa("") == ""

    def test_non_myanmar(self):
        """Non-Myanmar text should be unchanged."""
        assert normalize_tall_aa_after_wa("hello") == "hello"

    def test_mixed_text(self):
        """Mixed Myanmar/English text should only affect Myanmar parts."""
        text = "test \u1015\u103d\u102b\u1038 test"
        result = normalize_tall_aa_after_wa(text)
        assert result == "test \u1015\u103d\u102c\u1038 test"

    def test_consecutive_without_space(self):
        """Consecutive ွ+ါ occurrences without spaces should all be corrected."""
        # ပွါးဂွါ (no space between)
        text = "\u1015\u103d\u102b\u1038\u1002\u103d\u102b"
        result = normalize_tall_aa_after_wa(text)
        expected = "\u1015\u103d\u102c\u1038\u1002\u103d\u102c"
        assert result == expected

    def test_combined_medials_wa_tall_aa(self):
        """ွ+ါ in combined medial context should be corrected."""
        # ကျွါ (ka + ya + wa + tall AA) → ကျွာ
        text = "\u1000\u103b\u103d\u102b"
        result = normalize_tall_aa_after_wa(text)
        assert result == "\u1000\u103b\u103d\u102c"

    def test_wa_tall_aa_with_trailing_diacritics(self):
        """ွ+ါ followed by diacritics should still be corrected."""
        # ကွါံ (ka + wa + tall AA + anusvara) → ကွာံ
        text = "\u1000\u103d\u102b\u1036"
        result = normalize_tall_aa_after_wa(text)
        assert result == "\u1000\u103d\u102c\u1036"

    def test_none_input(self):
        """None-like empty input edge case."""
        assert normalize_tall_aa_after_wa("") == ""

    def test_only_wa_tall_aa_pattern(self):
        """Just the bare ွ+ါ pattern should be corrected."""
        text = "\u103d\u102b"
        result = normalize_tall_aa_after_wa(text)
        assert result == "\u103d\u102c"

    def test_preserves_surrounding_text(self):
        """Text before and after the pattern should be untouched."""
        text = "\u1019\u102e\u1038 \u1015\u103d\u102b\u1038 \u1019\u102e\u1038"
        result = normalize_tall_aa_after_wa(text)
        # Only the middle word should change
        assert result == "\u1019\u102e\u1038 \u1015\u103d\u102c\u1038 \u1019\u102e\u1038"

    def test_tall_aa_with_other_medial_untouched(self):
        """RA medial + Tall AA should NOT be changed (only Wa medial triggers)."""
        # ကြါ (ka + medial Ra + Tall AA) — should stay as is
        text = "\u1000\u103c\u102b"
        assert normalize_tall_aa_after_wa(text) == text


# =============================================================================
# Integration Tests: normalize() pipeline
# =============================================================================


class TestNormalizePipelineIntegration:
    """Test that normalize() applies tall-aa correction by default."""

    def test_normalize_corrects_wa_tall_aa(self):
        """normalize() should correct ွ+ါ by default."""
        text = "\u1015\u103d\u102b\u1038"  # ပွါး
        result = normalize(text)
        assert "\u103d\u102b" not in result  # ွ+ါ should be gone
        assert "\u103d\u102c" in result  # ွ+ာ should be present

    def test_normalize_opt_out(self):
        """normalize() with normalize_tall_aa=False should skip correction."""
        text = "\u1015\u103d\u102b\u1038"  # ပွါး
        result = normalize(text, normalize_tall_aa=False)
        assert "\u103d\u102b" in result  # ွ+ါ should remain

    def test_normalize_for_lookup_corrects(self):
        """normalize_for_lookup() should also correct ွ+ါ."""
        text = "\u1015\u103d\u102b\u1038"  # ပွါး
        result = normalize_for_lookup(text, convert_zawgyi=False)
        assert "\u103d\u102b" not in result
        assert "\u103d\u102c" in result

    def test_idempotent(self):
        """Applying normalization twice should give same result."""
        text = "\u1015\u103d\u102b\u1038"  # ပွါး
        once = normalize(text)
        twice = normalize(once)
        assert once == twice

    def test_idempotent_normalize_for_lookup(self):
        """normalize_for_lookup() is also idempotent."""
        text = "\u1015\u103d\u102b\u1038"  # ပွါး
        once = normalize_for_lookup(text, convert_zawgyi=False)
        twice = normalize_for_lookup(once, convert_zawgyi=False)
        assert once == twice

    def test_normalize_with_other_steps_combined(self):
        """Tall-aa correction works alongside other normalization steps."""
        # Text with zero-width space + medial Wa + Tall AA
        text = "\u200b\u1015\u103d\u102b\u1038"  # ZWSP + ပွါး
        result = normalize(text)
        # Zero-width should be removed AND tall-aa should be corrected
        assert "\u200b" not in result
        assert "\u103d\u102b" not in result
        assert "\u103d\u102c" in result

    def test_normalize_opt_out_preserves_others(self):
        """Opting out of tall-aa should not affect other normalization steps."""
        text = "\u200b\u1015\u103d\u102b\u1038"  # ZWSP + ပွါး
        result = normalize(text, normalize_tall_aa=False)
        # Zero-width should still be removed
        assert "\u200b" not in result
        # But tall-aa should remain
        assert "\u103d\u102b" in result

    def test_normalize_multiple_words(self):
        """normalize() handles multiple words with mixed correct/incorrect forms."""
        # သွါး ပွား ဂွါ (wrong, correct, wrong)
        text = "\u101e\u103d\u102b\u1038 \u1015\u103d\u102c\u1038 \u1002\u103d\u102b"
        result = normalize(text)
        # Both wrong forms should be fixed, correct form unchanged
        assert "\u103d\u102b" not in result
        assert result.count("\u103d\u102c") == 3

    def test_normalize_for_lookup_multiple_words(self):
        """normalize_for_lookup() handles multiple words."""
        text = "\u101e\u103d\u102b\u1038 \u1002\u103d\u102b"
        result = normalize_for_lookup(text, convert_zawgyi=False)
        assert "\u103d\u102b" not in result

    def test_normalize_then_validate_pipeline(self):
        """Full pipeline: normalize → validate should pass for corrected text."""
        validator = SyllableRuleValidator()
        wrong = "\u1015\u103d\u102b\u1038"  # ပွါး
        normalized = normalize(wrong)
        # After normalization, the corrected syllable should pass validation
        # Extract the Myanmar part (normalize may add/strip whitespace)
        assert validator.validate(normalized.strip()) is True
