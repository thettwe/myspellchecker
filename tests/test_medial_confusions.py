"""
Tests for MED-001: Incomplete Medial Confusion Patterns

Verifies that medial confusion patterns are properly detected:
- Ya (ျ) vs Ra (ြ) visual confusions
- Medial order corrections per UTN #11
- Aspiration confusions with medials
"""


class TestMedialOrderCorrections:
    """Test medial order corrections per UTN #11."""

    def test_medial_order_dict_exists(self):
        """MEDIAL_ORDER_CORRECTIONS should exist."""
        from myspellchecker.grammar.patterns import MEDIAL_ORDER_CORRECTIONS

        assert isinstance(MEDIAL_ORDER_CORRECTIONS, dict)
        assert len(MEDIAL_ORDER_CORRECTIONS) >= 6  # 6 basic order errors

    def test_ya_ra_order(self):
        """Ra before Ya should be corrected to Ya before Ra."""
        from myspellchecker.grammar.patterns import MEDIAL_ORDER_CORRECTIONS

        # UTN #11: Ya (U+103B) < Ra (U+103C)
        assert "ြျ" in MEDIAL_ORDER_CORRECTIONS
        assert MEDIAL_ORDER_CORRECTIONS["ြျ"] == "ျြ"

    def test_wa_ya_order(self):
        """Wa before Ya should be corrected to Ya before Wa."""
        from myspellchecker.grammar.patterns import MEDIAL_ORDER_CORRECTIONS

        # UTN #11: Ya (U+103B) < Wa (U+103D)
        assert "ွျ" in MEDIAL_ORDER_CORRECTIONS
        assert MEDIAL_ORDER_CORRECTIONS["ွျ"] == "ျွ"

    def test_ha_ya_order(self):
        """Ha before Ya should be corrected to Ya before Ha."""
        from myspellchecker.grammar.patterns import MEDIAL_ORDER_CORRECTIONS

        # UTN #11: Ya (U+103B) < Ha (U+103E)
        assert "ှျ" in MEDIAL_ORDER_CORRECTIONS
        assert MEDIAL_ORDER_CORRECTIONS["ှျ"] == "ျှ"

    def test_order_correction_function(self):
        """get_medial_order_correction should fix order errors."""
        from myspellchecker.grammar.patterns import get_medial_order_correction

        # Test with Ra+Ya error
        text_with_error = "ကြျ"  # Wrong: Ra before Ya
        correction = get_medial_order_correction(text_with_error)
        assert correction == "ကျြ"  # Correct: Ya before Ra

    def test_no_correction_needed(self):
        """Correct order should return None."""
        from myspellchecker.grammar.patterns import get_medial_order_correction

        correct_text = "ကျြ"  # Correct order: Ya before Ra
        assert get_medial_order_correction(correct_text) is None


class TestAspirationConfusions:
    """Test aspiration confusion patterns."""

    def test_aspiration_dict_exists(self):
        """ASPIRATION_MEDIAL_CONFUSIONS should exist."""
        from myspellchecker.grammar.patterns import ASPIRATION_MEDIAL_CONFUSIONS

        assert isinstance(ASPIRATION_MEDIAL_CONFUSIONS, dict)
        assert len(ASPIRATION_MEDIAL_CONFUSIONS) >= 7  # 7 basic confusions

    def test_ka_series_confusions(self):
        """Ka-series aspiration confusions should be detected."""
        from myspellchecker.grammar.patterns import ASPIRATION_MEDIAL_CONFUSIONS

        # ချ (Kha+Ya) confused with ကျ (Ka+Ya)
        assert "ချ" in ASPIRATION_MEDIAL_CONFUSIONS
        correction, desc = ASPIRATION_MEDIAL_CONFUSIONS["ချ"]
        assert correction == "ကျ"

        # ဂျ (Ga+Ya) confused with ကျ (Ka+Ya)
        assert "ဂျ" in ASPIRATION_MEDIAL_CONFUSIONS
        correction, desc = ASPIRATION_MEDIAL_CONFUSIONS["ဂျ"]
        assert correction == "ကျ"

    def test_pa_series_confusions(self):
        """Pa-series aspiration confusions should be detected."""
        from myspellchecker.grammar.patterns import ASPIRATION_MEDIAL_CONFUSIONS

        # ဖျ (Pha+Ya) confused with ပျ (Pa+Ya)
        assert "ဖျ" in ASPIRATION_MEDIAL_CONFUSIONS
        correction, desc = ASPIRATION_MEDIAL_CONFUSIONS["ဖျ"]
        assert correction == "ပျ"

    def test_aspiration_function(self):
        """get_aspiration_confusion should return correction info."""
        from myspellchecker.grammar.patterns import get_aspiration_confusion

        result = get_aspiration_confusion("ချ")
        assert result is not None
        correction, description = result
        assert correction == "ကျ"


class TestMedialConfusionPatterns:
    """Test word-level medial confusion patterns."""

    def test_pattern_dict_exists(self):
        """MEDIAL_CONFUSION_PATTERNS should exist."""
        from myspellchecker.grammar.patterns import MEDIAL_CONFUSION_PATTERNS

        assert isinstance(MEDIAL_CONFUSION_PATTERNS, dict)
        assert len(MEDIAL_CONFUSION_PATTERNS) >= 5

    def test_kyeizoo_correction(self):
        """ကြေးဇူး should be corrected to ကျေးဇူး (thanks)."""
        from myspellchecker.grammar.patterns import MEDIAL_CONFUSION_PATTERNS

        # This is a common real-world typo
        if "ကြေးဇူး" in MEDIAL_CONFUSION_PATTERNS:
            correction, _desc, _ctx = MEDIAL_CONFUSION_PATTERNS["ကြေးဇူး"]
            assert correction == "ကျေးဇူး"

    def test_kyaung_correction(self):
        """ကြောင် should suggest ကျောင်း (school)."""
        from myspellchecker.grammar.patterns import MEDIAL_CONFUSION_PATTERNS

        if "ကြောင်" in MEDIAL_CONFUSION_PATTERNS:
            correction, _desc, _ctx = MEDIAL_CONFUSION_PATTERNS["ကြောင်"]
            assert correction == "ကျောင်း"

    def test_get_medial_confusion_function(self):
        """get_medial_confusion_correction should work."""
        from myspellchecker.grammar.patterns import get_medial_confusion_correction

        result = get_medial_confusion_correction("ကြေးဇူး")
        if result:
            correction, desc, ctx = result
            assert correction == "ကျေးဇူး"


class TestMedialPatternCoverage:
    """Test coverage of medial patterns."""

    def test_has_order_corrections(self):
        """Should have medial order corrections."""
        from myspellchecker.grammar.patterns import MEDIAL_ORDER_CORRECTIONS

        # All 6 basic order errors
        expected_errors = ["ြျ", "ွျ", "ှျ", "ွြ", "ှြ", "ှွ"]
        for error in expected_errors:
            assert error in MEDIAL_ORDER_CORRECTIONS, f"Missing order correction: {error}"

    def test_has_aspiration_confusions(self):
        """Should have aspiration confusion patterns."""
        from myspellchecker.grammar.patterns import ASPIRATION_MEDIAL_CONFUSIONS

        # At least some Ka and Pa series
        assert "ချ" in ASPIRATION_MEDIAL_CONFUSIONS
        assert "ဖျ" in ASPIRATION_MEDIAL_CONFUSIONS

    def test_functions_exported(self):
        """Helper functions should be importable."""
        from myspellchecker.grammar.patterns import (
            get_aspiration_confusion,
            get_medial_confusion_correction,
            get_medial_order_correction,
        )

        # All should be callable/accessible
        assert callable(get_medial_confusion_correction)
        assert callable(get_medial_order_correction)
        assert callable(get_aspiration_confusion)
