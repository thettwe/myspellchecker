"""
Unit tests for Myanmar medial ordering per UTN #11 specification.

UTN #11 canonical order: Ya (U+103B) < Ra (U+103C) < Wa (U+103D) < Ha (U+103E)
Reference: https://unicode.org/notes/tn11/UTN11_4.pdf (Section: Diacritic storage order)
"""

from myspellchecker.core.constants import ORDER_WEIGHTS, VALID_MEDIAL_SEQUENCES


class TestMedialOrderingConstants:
    """Test that ORDER_WEIGHTS follows UTN #11 canonical order."""

    def test_ya_before_ra(self):
        """Ya (U+103B) should have lower weight than Ra (U+103C)."""
        ya_weight = ORDER_WEIGHTS.get("\u103b")
        ra_weight = ORDER_WEIGHTS.get("\u103c")
        assert ya_weight < ra_weight, "Ya must come before Ra"

    def test_ra_before_wa(self):
        """Ra (U+103C) should have lower weight than Wa (U+103D)."""
        ra_weight = ORDER_WEIGHTS.get("\u103c")
        wa_weight = ORDER_WEIGHTS.get("\u103d")
        assert ra_weight < wa_weight, "Ra must come before Wa"

    def test_wa_before_ha(self):
        """Wa (U+103D) should have lower weight than Ha (U+103E)."""
        wa_weight = ORDER_WEIGHTS.get("\u103d")
        ha_weight = ORDER_WEIGHTS.get("\u103e")
        assert wa_weight < ha_weight, "Wa must come before Ha"

    def test_full_medial_order(self):
        """Test complete medial ordering: Ya < Ra < Wa < Ha."""
        medials = ["\u103b", "\u103c", "\u103d", "\u103e"]  # Ya, Ra, Wa, Ha
        weights = [ORDER_WEIGHTS.get(m) for m in medials]
        assert weights == sorted(weights), "Medials must be in ascending weight order"


class TestValidMedialSequences:
    """Test that VALID_MEDIAL_SEQUENCES uses correct order."""

    def test_ya_ra_sequence_correct(self):
        """ျြ (Ya+Ra) should be in valid sequences."""
        assert "ျြ" in VALID_MEDIAL_SEQUENCES, "Ya+Ra sequence must be valid"

    def test_ra_ya_sequence_not_valid(self):
        """ြျ (Ra+Ya) should NOT be in valid sequences."""
        assert "ြျ" not in VALID_MEDIAL_SEQUENCES, "Ra+Ya sequence must be invalid"

    def test_ya_ra_wa_sequence_correct(self):
        """ျြွ (Ya+Ra+Wa) should be in valid sequences."""
        assert "ျြွ" in VALID_MEDIAL_SEQUENCES

    def test_ya_ra_wa_ha_sequence_correct(self):
        """ျြွှ (Ya+Ra+Wa+Ha) should be in valid sequences."""
        assert "ျြွှ" in VALID_MEDIAL_SEQUENCES


class TestMedialSorting:
    """Test that sorting by ORDER_WEIGHTS produces correct order."""

    def test_sort_ra_ya_to_ya_ra(self):
        """Sorting ြျ (Ra+Ya) should produce ျြ (Ya+Ra)."""
        wrong_order = "ြျ"
        chars = list(wrong_order)
        sorted_chars = sorted(chars, key=lambda c: ORDER_WEIGHTS.get(c, 999))
        result = "".join(sorted_chars)
        assert result == "ျြ", f"Expected ျြ, got {result}"

    def test_sort_wa_ya_to_ya_wa(self):
        """Sorting ွျ (Wa+Ya) should produce ျွ (Ya+Wa)."""
        wrong_order = "ွျ"
        chars = list(wrong_order)
        sorted_chars = sorted(chars, key=lambda c: ORDER_WEIGHTS.get(c, 999))
        result = "".join(sorted_chars)
        assert result == "ျွ", f"Expected ျွ, got {result}"

    def test_sort_ha_wa_ra_ya_to_correct_order(self):
        """Sorting ှွြျ (Ha+Wa+Ra+Ya) should produce ျြွှ (Ya+Ra+Wa+Ha)."""
        wrong_order = "ှွြျ"  # Completely reversed
        chars = list(wrong_order)
        sorted_chars = sorted(chars, key=lambda c: ORDER_WEIGHTS.get(c, 999))
        result = "".join(sorted_chars)
        assert result == "ျြွှ", f"Expected ျြွှ, got {result}"

    def test_correct_order_unchanged(self):
        """Sorting ျြွှ (already correct) should remain unchanged."""
        correct_order = "ျြွှ"
        chars = list(correct_order)
        sorted_chars = sorted(chars, key=lambda c: ORDER_WEIGHTS.get(c, 999))
        result = "".join(sorted_chars)
        assert result == correct_order, "Correct order should remain unchanged"


class TestSyllableWithMedials:
    """Test complete syllables with medials maintain correct order."""

    def test_syllable_ka_ya_ra(self):
        """ကျြ (Ka+Ya+Ra) is valid order."""
        syllable = "ကျြ"
        # Extract medials
        medials = [c for c in syllable if c in ORDER_WEIGHTS and c in "ျြွှ"]
        # Check order
        weights = [ORDER_WEIGHTS.get(m) for m in medials]
        assert weights == sorted(weights), f"Syllable {syllable} has incorrect medial order"

    def test_syllable_ka_ya_ra_wa(self):
        """ကျြွ (Ka+Ya+Ra+Wa) is valid order."""
        syllable = "ကျြွ"
        medials = [c for c in syllable if c in ORDER_WEIGHTS and c in "ျြွှ"]
        weights = [ORDER_WEIGHTS.get(m) for m in medials]
        assert weights == sorted(weights), f"Syllable {syllable} has incorrect medial order"


class TestAsatBeforeVowel:
    """Test the new Asat-before-vowel validation."""

    def test_asat_before_vowel_invalid(self):
        """Asat followed by vowel should be invalid."""
        from myspellchecker.core.syllable_rules import SyllableRuleValidator

        validator = SyllableRuleValidator()
        invalid_syllables = ["တင်ူ", "ရန်ု", "နည်ူး"]
        for syllable in invalid_syllables:
            assert not validator.validate(syllable), f"{syllable} should be invalid"

    def test_normal_asat_valid(self):
        """Normal Asat endings should be valid."""
        from myspellchecker.core.syllable_rules import SyllableRuleValidator

        validator = SyllableRuleValidator()
        valid_syllables = ["တင်", "ကျွန်", "မြန်"]
        for syllable in valid_syllables:
            assert validator.validate(syllable), f"{syllable} should be valid"


class TestNormalizationIdempotence:
    """Test that normalization is idempotent: normalize(normalize(x)) == normalize(x)."""

    def test_idempotence_correct_order(self):
        """Correctly ordered text should remain unchanged after normalization."""
        from myspellchecker.text.normalize import normalize

        test_cases = ["ကျြ", "ကျြွှ", "မြန်မာ", "အေး", "အင်္ဂလိပ်"]
        for text in test_cases:
            n1 = normalize(text)
            n2 = normalize(n1)
            assert n1 == n2, f"Not idempotent: {text} → {n1} → {n2}"

    def test_idempotence_wrong_order(self):
        """Wrongly ordered text should be fixed, then remain stable."""
        from myspellchecker.text.normalize import normalize

        test_cases = [
            ("ကြျ", "ကျြ"),  # Ra + Ya → Ya + Ra
            ("ကွျ", "ကျွ"),  # Wa + Ya → Ya + Wa
            ("ကှွြျ", "ကျြွှ"),  # Reversed → Correct order
        ]
        for wrong, expected in test_cases:
            n1 = normalize(wrong)
            n2 = normalize(n1)
            assert n1 == expected, f"Wrong normalization: {wrong} → {n1}, expected {expected}"
            assert n1 == n2, f"Not idempotent: {wrong} → {n1} → {n2}"

    def test_normalization_medial_order_utn11(self):
        """Test that medial order follows UTN #11: Ya < Ra < Wa < Ha."""
        from myspellchecker.text.normalize import normalize

        # These inputs have wrong medial order and should be fixed
        assert normalize("ကြျ") == "ကျြ"  # Ra+Ya → Ya+Ra
        assert normalize("ကွြ") == "ကြွ"  # Wa+Ra → Ra+Wa
        assert normalize("ကှွ") == "ကွှ"  # Ha+Wa → Wa+Ha
        assert normalize("ကှြ") == "ကြှ"  # Ha+Ra → Ra+Ha

    def test_normalization_preserves_correct_order(self):
        """Test that correct medial order is preserved."""
        from myspellchecker.text.normalize import normalize

        correct_order_cases = [
            "ကျ",  # Ya only
            "ကြ",  # Ra only
            "ကျြ",  # Ya + Ra
            "ကျြွ",  # Ya + Ra + Wa
            "ကျြွှ",  # Ya + Ra + Wa + Ha
        ]
        for text in correct_order_cases:
            assert normalize(text) == text, f"Correct order changed: {text} → {normalize(text)}"


class TestMyanmarExtendedCharsets:
    """Test Myanmar Extended-A and Extended-B character set support (GAP-001)."""

    def test_extended_a_chars_defined(self):
        """MYANMAR_EXTENDED_A_CHARS should be defined and contain U+AA7B."""
        from myspellchecker.core.constants import MYANMAR_EXTENDED_A_CHARS

        assert len(MYANMAR_EXTENDED_A_CHARS) == 32  # U+AA60 to U+AA7F
        assert "\uaa7b" in MYANMAR_EXTENDED_A_CHARS  # Shan/Pao Karen tone mark

    def test_extended_b_chars_defined(self):
        """MYANMAR_EXTENDED_B_CHARS should be defined and contain U+A9E5."""
        from myspellchecker.core.constants import MYANMAR_EXTENDED_B_CHARS

        assert len(MYANMAR_EXTENDED_B_CHARS) == 32  # U+A9E0 to U+A9FF
        assert "\ua9e5" in MYANMAR_EXTENDED_B_CHARS  # Shan saw

    def test_all_myanmar_chars_includes_extended(self):
        """ALL_MYANMAR_CHARS should include main block plus Extended-A and Extended-B."""
        from myspellchecker.core.constants import ALL_MYANMAR_CHARS

        # Main block chars
        assert "\u1000" in ALL_MYANMAR_CHARS  # Ka
        assert "\u1021" in ALL_MYANMAR_CHARS  # A
        assert "\u103a" in ALL_MYANMAR_CHARS  # Asat

        # Extended-A
        assert "\uaa7b" in ALL_MYANMAR_CHARS

        # Extended-B
        assert "\ua9e5" in ALL_MYANMAR_CHARS


class TestDoubleDiacritics:
    """Test double diacritics validation (GAP-003).

    The validator should reject syllables with consecutive identical diacritics
    including those starting with U+1021 (အ) which acts as a consonant carrier.
    """

    def test_double_e_vowel_invalid(self):
        """Double e-vowel (ေ) should be invalid."""
        from myspellchecker.core.syllable_rules import SyllableRuleValidator

        validator = SyllableRuleValidator()
        invalid_syllables = ["အေေး", "ကေေ", "ရေေ"]
        for syllable in invalid_syllables:
            assert not validator.validate(syllable), f"{syllable} should be invalid"

    def test_double_wa_hswe_invalid(self):
        """Double wa-hswe (ွွ) should be invalid."""
        from myspellchecker.core.syllable_rules import SyllableRuleValidator

        validator = SyllableRuleValidator()
        invalid_syllables = ["ကွွ", "ငွွ"]
        for syllable in invalid_syllables:
            assert not validator.validate(syllable), f"{syllable} should be invalid"

    def test_double_medials_invalid(self):
        """Double medials should be invalid."""
        from myspellchecker.core.syllable_rules import SyllableRuleValidator

        validator = SyllableRuleValidator()
        invalid_syllables = ["ကျျ", "ကြြ"]
        for syllable in invalid_syllables:
            assert not validator.validate(syllable), f"{syllable} should be invalid"

    def test_double_vowels_invalid(self):
        """Double vowels should be invalid."""
        from myspellchecker.core.syllable_rules import SyllableRuleValidator

        validator = SyllableRuleValidator()
        invalid_syllables = ["ကိိ", "ကုု", "ကံံ"]
        for syllable in invalid_syllables:
            assert not validator.validate(syllable), f"{syllable} should be invalid"

    def test_u1021_syllables_fully_validated(self):
        """U+1021 (အ) syllables should go through full validation.

        Previously, syllables starting with U+1021 would return True early
        without checking for double diacritics. This test ensures the fix works.
        """
        from myspellchecker.core.syllable_rules import SyllableRuleValidator

        validator = SyllableRuleValidator()

        # Invalid: double diacritics
        assert not validator.validate("အေေး"), "အေေး should be invalid (double e-vowel)"
        assert not validator.validate("အုုံ"), "အုုံ should be invalid (double u-vowel)"

        # Valid: single diacritics
        assert validator.validate("အေး"), "အေး should be valid"
        assert validator.validate("အား"), "အား should be valid"
        assert validator.validate("အင်း"), "အင်း should be valid"
