"""
Comprehensive edge case tests for empty/None inputs.

Tests covering:
- SpellChecker.check() with empty, whitespace, None inputs
- SpellChecker.check_batch() with empty lists, None, mixed inputs
- Config with disabled validators, empty/None values
- Individual validators with edge case inputs
"""

import pytest

from myspellchecker import SpellChecker
from myspellchecker.core.config import SpellCheckerConfig
from myspellchecker.core.exceptions import ProcessingError, TokenizationError
from myspellchecker.core.response import Response
from myspellchecker.providers import MemoryProvider


class TestSpellCheckerEmptyInputs:
    """Test SpellChecker.check() with empty and edge case inputs."""

    @pytest.fixture
    def checker(self):
        """Create a SpellChecker with MemoryProvider for testing."""
        provider = MemoryProvider()
        config = SpellCheckerConfig(
            fallback_to_empty_provider=True,
            use_context_checker=False,
        )
        return SpellChecker(config=config, provider=provider)

    def test_check_empty_string(self, checker):
        """Test check() with empty string."""
        result = checker.check("")
        assert isinstance(result, Response)
        assert result.text == ""
        assert result.has_errors is False
        assert result.errors == []

    def test_check_whitespace_only(self, checker):
        """Test check() with whitespace-only string."""
        result = checker.check("   ")
        assert isinstance(result, Response)
        assert result.has_errors is False

    def test_check_tabs_and_newlines(self, checker):
        """Test check() with tabs and newlines."""
        result = checker.check("\n\t")
        assert isinstance(result, Response)
        assert result.has_errors is False

    def test_check_mixed_whitespace(self, checker):
        """Test check() with mixed whitespace characters."""
        result = checker.check("  \t\n  \r\n  ")
        assert isinstance(result, Response)
        assert result.has_errors is False

    def test_check_none_input(self, checker):
        """Test check() with None input raises TypeError."""
        with pytest.raises(TypeError, match="text must be a string"):
            checker.check(None)

    def test_check_single_space(self, checker):
        """Test check() with single space."""
        result = checker.check(" ")
        assert isinstance(result, Response)
        assert result.has_errors is False

    def test_check_single_myanmar_character(self, checker):
        """Test check() with single Myanmar character."""
        result = checker.check("က")
        assert isinstance(result, Response)
        # Single consonant may or may not be valid depending on rules

    def test_check_single_non_myanmar_character(self, checker):
        """Test check() with single non-Myanmar character."""
        result = checker.check("A")
        assert isinstance(result, Response)

    def test_check_zero_width_characters(self, checker):
        """Test check() with zero-width characters only."""
        # Zero-width space, zero-width non-joiner
        result = checker.check("\u200b\u200c\u200d")
        assert isinstance(result, Response)


class TestSpellCheckerBatchEmptyInputs:
    """Test SpellChecker.check_batch() with empty and edge case inputs."""

    @pytest.fixture
    def checker(self):
        """Create a SpellChecker with MemoryProvider for testing."""
        provider = MemoryProvider()
        config = SpellCheckerConfig(
            fallback_to_empty_provider=True,
            use_context_checker=False,
        )
        return SpellChecker(config=config, provider=provider)

    def test_check_batch_empty_list(self, checker):
        """Test check_batch() with empty list raises ValueError."""
        with pytest.raises(ProcessingError, match="texts list cannot be empty"):
            checker.check_batch([])

    def test_check_batch_list_of_empty_strings(self, checker):
        """Test check_batch() with list of empty strings."""
        results = checker.check_batch(["", "", ""])
        assert isinstance(results, list)
        assert len(results) == 3
        for result in results:
            assert isinstance(result, Response)
            assert result.has_errors is False

    def test_check_batch_none_input(self, checker):
        """Test check_batch() with None input raises TypeError."""
        with pytest.raises((TypeError, AttributeError)):
            checker.check_batch(None)

    def test_check_batch_mixed_empty_and_valid(self, checker):
        """Test check_batch() with mixed empty and valid inputs."""
        results = checker.check_batch(["", "ကောင်း", "   ", "မြန်မာ", "\n\t"])
        assert isinstance(results, list)
        assert len(results) == 5

    def test_check_batch_single_empty_item(self, checker):
        """Test check_batch() with single empty item."""
        results = checker.check_batch([""])
        assert isinstance(results, list)
        assert len(results) == 1
        assert results[0].has_errors is False

    def test_check_batch_whitespace_items(self, checker):
        """Test check_batch() with various whitespace items."""
        results = checker.check_batch([" ", "  ", "\t", "\n", "\r\n"])
        assert isinstance(results, list)
        assert len(results) == 5

    def test_check_batch_with_none_in_list(self, checker):
        """Test check_batch() with None items in list raises ValueError."""
        with pytest.raises(ProcessingError, match="must be a string"):
            checker.check_batch(["valid", None, "text"])


class TestConfigEdgeCases:
    """Test SpellCheckerConfig with edge case values."""

    def test_config_all_validators_disabled(self):
        """Test config with all validators disabled."""
        config = SpellCheckerConfig(
            use_phonetic=False,
            use_context_checker=False,
            use_ner=False,
            use_rule_based_validation=False,
            fallback_to_empty_provider=True,
        )
        provider = MemoryProvider()
        checker = SpellChecker(config=config, provider=provider)

        # Should still work with basic validation
        result = checker.check("ကောင်း")
        assert isinstance(result, Response)

    def test_config_minimal_settings(self):
        """Test config with minimal settings."""
        config = SpellCheckerConfig(
            max_edit_distance=1,
            max_suggestions=1,
            fallback_to_empty_provider=True,
        )
        provider = MemoryProvider()
        checker = SpellChecker(config=config, provider=provider)

        result = checker.check("ကောင်း")
        assert isinstance(result, Response)

    def test_config_empty_provider(self):
        """Test with empty MemoryProvider (no words/syllables)."""
        provider = MemoryProvider()
        config = SpellCheckerConfig(fallback_to_empty_provider=True)
        checker = SpellChecker(config=config, provider=provider)

        # Should handle empty dictionary gracefully
        result = checker.check("ကောင်း")
        assert isinstance(result, Response)

    def test_config_max_values(self):
        """Test config with maximum allowed values."""
        config = SpellCheckerConfig(
            max_edit_distance=3,  # Maximum allowed
            max_suggestions=100,
            fallback_to_empty_provider=True,
        )
        provider = MemoryProvider()
        checker = SpellChecker(config=config, provider=provider)

        result = checker.check("ကောင်း")
        assert isinstance(result, Response)

    def test_config_zero_suggestions(self):
        """Test that zero max_suggestions raises ValueError."""
        with pytest.raises(ValueError, match="greater than or equal to 1"):
            SpellCheckerConfig(
                max_suggestions=0,
                fallback_to_empty_provider=True,
            )


class TestValidatorEmptyInputs:
    """Test individual validators with empty/None inputs."""

    def test_syllable_validator_requires_args(self):
        """Test SyllableValidator requires constructor arguments."""
        from myspellchecker.core.validators import SyllableValidator

        with pytest.raises(TypeError, match="missing .* required positional argument"):
            SyllableValidator()

    def test_word_validator_requires_args(self):
        """Test WordValidator requires constructor arguments."""
        from myspellchecker.core.validators import WordValidator

        with pytest.raises(TypeError, match="missing .* required positional argument"):
            WordValidator()

    def test_context_validator_behavior(self):
        """Test context validation behavior through SpellChecker."""
        # ContextValidator is an internal implementation detail
        # Test context validation through SpellChecker interface
        from myspellchecker import SpellChecker
        from myspellchecker.core.config import SpellCheckerConfig
        from myspellchecker.providers import MemoryProvider

        provider = MemoryProvider()
        config = SpellCheckerConfig(
            use_context_checker=True,
            fallback_to_empty_provider=True,
        )
        checker = SpellChecker(config=config, provider=provider)

        result = checker.check("")
        assert result is not None


class TestNormalizationEmptyInputs:
    """Test text normalization with empty/None inputs."""

    def test_normalize_empty_string(self):
        """Test normalize_myanmar_text with empty string."""
        from myspellchecker.text.normalize import normalize_myanmar_text

        result = normalize_myanmar_text("")
        assert result == ""

    def test_normalize_whitespace_only(self):
        """Test normalize_myanmar_text with whitespace only."""
        from myspellchecker.text.normalize import normalize_myanmar_text

        result = normalize_myanmar_text("   ")
        assert isinstance(result, str)

    def test_normalize_none_input(self):
        """Test normalize_myanmar_text with None input returns None."""
        from myspellchecker.text.normalize import normalize_myanmar_text

        result = normalize_myanmar_text(None)
        assert result is None


class TestSegmenterEmptyInputs:
    """Test segmenters with empty/None inputs."""

    def test_default_segmenter_empty_input(self):
        """Test DefaultSegmenter with empty input raises ValueError."""
        from myspellchecker.segmenters import DefaultSegmenter

        segmenter = DefaultSegmenter()
        with pytest.raises(TokenizationError, match="cannot be empty"):
            segmenter.segment_syllables("")

    def test_default_segmenter_whitespace_input(self):
        """Test DefaultSegmenter with whitespace input raises ValueError."""
        from myspellchecker.segmenters import DefaultSegmenter

        segmenter = DefaultSegmenter()
        with pytest.raises(TokenizationError, match="cannot be empty"):
            segmenter.segment_syllables("   ")

    def test_regex_segmenter_empty_input(self):
        """Test RegexSegmenter with empty input raises ValueError."""
        from myspellchecker.segmenters.regex import RegexSegmenter

        segmenter = RegexSegmenter()
        with pytest.raises(TokenizationError, match="cannot be empty"):
            segmenter.segment_syllables("")

    def test_regex_segmenter_whitespace_input(self):
        """Test RegexSegmenter with whitespace input raises ValueError."""
        from myspellchecker.segmenters.regex import RegexSegmenter

        segmenter = RegexSegmenter()
        with pytest.raises(TokenizationError, match="cannot be empty"):
            segmenter.segment_syllables("   ")


class TestProviderEmptyInputs:
    """Test providers with empty/None inputs."""

    def test_memory_provider_empty_syllable(self):
        """Test MemoryProvider with empty syllable lookup."""
        from myspellchecker.providers import MemoryProvider

        provider = MemoryProvider()

        assert provider.is_valid_syllable("") is False
        assert provider.get_syllable_frequency("") == 0

    def test_memory_provider_empty_word(self):
        """Test MemoryProvider with empty word lookup."""
        from myspellchecker.providers import MemoryProvider

        provider = MemoryProvider()

        assert provider.is_valid_word("") is False
        assert provider.get_word_frequency("") == 0

    def test_memory_provider_whitespace_input(self):
        """Test MemoryProvider with whitespace inputs."""
        from myspellchecker.providers import MemoryProvider

        provider = MemoryProvider()

        assert provider.is_valid_syllable("   ") is False
        assert provider.is_valid_word("   ") is False

    def test_memory_provider_none_input(self):
        """Test MemoryProvider with None inputs returns False."""
        from myspellchecker.providers import MemoryProvider

        provider = MemoryProvider()
        assert provider.is_valid_syllable(None) is False


class TestResponseEmptyInputs:
    """Test Response object creation with edge cases."""

    def test_response_empty_text(self):
        """Test Response with empty text."""
        from myspellchecker.core.response import Response

        response = Response(
            text="",
            corrected_text="",
            has_errors=False,
            level="syllable",
            errors=[],
            metadata={},
        )
        assert response.text == ""
        assert response.has_errors is False
        assert len(response.errors) == 0

    def test_response_whitespace_text(self):
        """Test Response with whitespace text."""
        from myspellchecker.core.response import Response

        response = Response(
            text="   ",
            corrected_text="   ",
            has_errors=False,
            level="syllable",
            errors=[],
            metadata={},
        )
        assert response.text == "   "
        assert response.has_errors is False

    def test_response_to_dict_empty(self):
        """Test Response.to_dict() with empty text."""
        from myspellchecker.core.response import Response

        response = Response(
            text="",
            corrected_text="",
            has_errors=False,
            level="syllable",
            errors=[],
            metadata={},
        )
        result = response.to_dict()

        assert isinstance(result, dict)
        assert result["text"] == ""
        assert result["has_errors"] is False


class TestSyllableRuleValidatorEmptyInputs:
    """Test SyllableRuleValidator with empty inputs."""

    def test_rule_validator_empty_string(self):
        """Test SyllableRuleValidator with empty string."""
        from myspellchecker.core.syllable_rules import SyllableRuleValidator

        validator = SyllableRuleValidator()
        result = validator.validate("")
        assert result is False  # Empty string is not a valid syllable

    def test_rule_validator_whitespace(self):
        """Test SyllableRuleValidator with whitespace."""
        from myspellchecker.core.syllable_rules import SyllableRuleValidator

        validator = SyllableRuleValidator()
        result = validator.validate("   ")
        assert result is False  # Whitespace is not a valid syllable

    def test_rule_validator_single_consonant(self):
        """Test SyllableRuleValidator with single consonant."""
        from myspellchecker.core.syllable_rules import SyllableRuleValidator

        validator = SyllableRuleValidator()
        # Single Myanmar consonant may or may not be valid
        result = validator.validate("က")
        assert isinstance(result, bool)


class TestPhoneticEmptyInputs:
    """Test phonetic hashing with empty inputs."""

    def test_phonetic_hasher_empty_input(self):
        """Test PhoneticHasher with empty input."""
        from myspellchecker.text.phonetic import PhoneticHasher

        hasher = PhoneticHasher()

        if hasattr(hasher, "hash"):
            result = hasher.hash("")
            # Empty string should return empty or specific marker
            assert result is not None

    def test_phonetic_hasher_whitespace(self):
        """Test PhoneticHasher with whitespace input."""
        from myspellchecker.text.phonetic import PhoneticHasher

        hasher = PhoneticHasher()

        if hasattr(hasher, "hash"):
            result = hasher.hash("   ")
            assert result is not None


class TestBuilderEmptyInputs:
    """Test SpellCheckerBuilder with edge case inputs."""

    def test_builder_minimal_build(self):
        """Test builder with minimal configuration."""
        from myspellchecker.core.builder import SpellCheckerBuilder
        from myspellchecker.providers import MemoryProvider

        builder = SpellCheckerBuilder()
        builder.with_provider(MemoryProvider())
        checker = builder.build()

        assert checker is not None
        result = checker.check("")
        assert isinstance(result, Response)

    def test_builder_all_features_disabled(self):
        """Test builder with all features disabled."""
        from myspellchecker.core.builder import SpellCheckerBuilder
        from myspellchecker.providers import MemoryProvider

        builder = SpellCheckerBuilder()
        builder.with_provider(MemoryProvider())
        builder.with_phonetic(False)
        builder.with_context_checking(False)
        builder.with_ner(False)
        builder.with_rule_based_validation(False)

        checker = builder.build()

        result = checker.check("ကောင်း")
        assert isinstance(result, Response)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
