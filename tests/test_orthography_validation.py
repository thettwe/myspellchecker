"""
Unit tests for OrthographyValidationStrategy.

Tests medial order validation and medial-consonant compatibility checking.
"""

from myspellchecker.core.validation_strategies import (
    OrthographyValidationStrategy,
    ValidationContext,
)
from myspellchecker.core.validation_strategies.orthography_strategy import (
    _check_medial_compatibility,
    _has_medial_order_violation,
)
from myspellchecker.grammar.patterns import get_medial_order_correction


class TestMedialOrderCorrection:
    """Tests for medial order correction per UTN #11."""

    def test_ya_ra_order_violation(self):
        """Test Ra+Ya → Ya+Ra correction."""
        # "ြျ" (Ra+Ya) should be corrected to "ျြ" (Ya+Ra)
        # Note: these are structural test syllables to validate medial reordering,
        # not necessarily common dictionary words.
        assert _has_medial_order_violation("ကြျင်း")
        correction = get_medial_order_correction("ကြျင်း")
        assert correction == "ကျြင်း"

    def test_wa_ya_order_violation(self):
        """Test Wa+Ya → Ya+Wa correction."""
        # "ွျ" (Wa+Ya) should be corrected to "ျွ" (Ya+Wa)
        assert _has_medial_order_violation("လွျင်")
        correction = get_medial_order_correction("လွျင်")
        assert correction == "လျွင်"

    def test_no_violation(self):
        """Test word with correct medial order."""
        # "ကျွန်" (Ya+Wa) is correct order
        assert not _has_medial_order_violation("ကျွန်")
        correction = get_medial_order_correction("ကျွန်")
        assert correction is None

    def test_no_medials(self):
        """Test word without medials."""
        assert not _has_medial_order_violation("ကောင်း")
        correction = get_medial_order_correction("ကောင်း")
        assert correction is None


class TestMedialCompatibility:
    """Tests for medial-consonant compatibility checking."""

    def test_ha_compatible_with_sonorant(self):
        """Test Ha-htoe is valid with sonorant consonants."""
        # "နှ" (Na + Ha-htoe) - valid, Na is sonorant
        result = _check_medial_compatibility("နှစ်")
        assert result is None

        # "မှ" (Ma + Ha-htoe) - valid, Ma is sonorant
        result = _check_medial_compatibility("မှန်")
        assert result is None

        # "လှ" (La + Ha-htoe) - valid, La is sonorant
        result = _check_medial_compatibility("လှပ")
        assert result is None

    def test_ha_incompatible_with_stop(self):
        """Test Ha-htoe is invalid with stop consonants."""
        # "ကှ" (Ka + Ha-htoe) - invalid, Ka is a stop
        result = _check_medial_compatibility("ကှား")
        assert result is not None
        assert "Ha-htoe" in result[2]

    def test_ya_compatible_with_ka_group(self):
        """Test Ya-pin is valid with Ka-group consonants."""
        # "ကျ" (Ka + Ya-pin) - valid
        result = _check_medial_compatibility("ကျွန်")
        assert result is None

    def test_ra_incompatible_with_tha(self):
        """Test Ya-yit (ြ) is invalid with Tha (သ)."""
        # "သြ" (Tha + Ya-yit) - invalid
        result = _check_medial_compatibility("သြား")
        assert result is not None
        assert "Ya-yit" in result[2]  # Ya-yit (ြ U+103C) per standard naming


class TestOrthographyValidationStrategy:
    """Tests for OrthographyValidationStrategy."""

    def test_strategy_priority(self):
        """Test strategy has correct priority."""
        strategy = OrthographyValidationStrategy()
        assert strategy.priority() == 15

    def test_strategy_repr(self):
        """Test strategy string representation."""
        strategy = OrthographyValidationStrategy()
        assert "OrthographyValidationStrategy" in repr(strategy)
        assert "priority=15" in repr(strategy)

    def test_validate_medial_order_violation(self):
        """Test detection of medial order violation."""
        strategy = OrthographyValidationStrategy()

        context = ValidationContext(
            sentence="ကြျင်း",
            words=["ကြျင်း"],
            word_positions=[0],
            is_name_mask=[False],
        )

        errors = strategy.validate(context)

        assert len(errors) == 1
        assert errors[0].text == "ကြျင်း"
        assert errors[0].error_type == "medial_order_error"
        assert "ကျြင်း" in errors[0].suggestions

    def test_validate_medial_compatibility_violation(self):
        """Test detection of medial compatibility violation."""
        strategy = OrthographyValidationStrategy()

        # "ကှား" - Ka + Ha-htoe is invalid
        context = ValidationContext(
            sentence="ကှား",
            words=["ကှား"],
            word_positions=[0],
            is_name_mask=[False],
        )

        errors = strategy.validate(context)

        assert len(errors) == 1
        assert errors[0].text == "ကှား"
        assert errors[0].error_type == "medial_compatibility_error"

    def test_validate_no_errors(self):
        """Test valid words produce no errors."""
        strategy = OrthographyValidationStrategy()

        context = ValidationContext(
            sentence="ကျွန်တော် သွားပါမယ်",
            words=["ကျွန်တော်", "သွားပါမယ်"],
            word_positions=[0, 15],
            is_name_mask=[False, False],
        )

        errors = strategy.validate(context)
        assert len(errors) == 0

    def test_validate_skips_names(self):
        """Test that proper names are skipped."""
        strategy = OrthographyValidationStrategy()

        context = ValidationContext(
            sentence="ကြျင်း",  # Would normally be error
            words=["ကြျင်း"],
            word_positions=[0],
            is_name_mask=[True],  # Marked as name
        )

        errors = strategy.validate(context)
        assert len(errors) == 0

    def test_validate_skips_existing_errors(self):
        """Test that words with existing errors are skipped."""
        strategy = OrthographyValidationStrategy()

        context = ValidationContext(
            sentence="ကြျင်း",
            words=["ကြျင်း"],
            word_positions=[0],
            is_name_mask=[False],
        )
        context.existing_errors[0] = "test"  # Mark position as having error

        errors = strategy.validate(context)
        assert len(errors) == 0

    def test_validate_empty_words(self):
        """Test validation with empty word list."""
        strategy = OrthographyValidationStrategy()

        context = ValidationContext(
            sentence="",
            words=[],
            word_positions=[],
        )

        errors = strategy.validate(context)
        assert len(errors) == 0

    def test_validate_multiple_errors(self):
        """Test detection of multiple orthographic errors."""
        strategy = OrthographyValidationStrategy()

        context = ValidationContext(
            sentence="ကြျင်း ကှား",
            words=["ကြျင်း", "ကှား"],
            word_positions=[0, 10],
            is_name_mask=[False, False],
        )

        errors = strategy.validate(context)

        assert len(errors) == 2
        # First error: medial order
        assert errors[0].error_type == "medial_order_error"
        # Second error: medial compatibility
        assert errors[1].error_type == "medial_compatibility_error"
