"""Unit tests for colloquial spelling variant recognition.

Tests the colloquial variant detection feature implemented
including:
- Colloquial detection functions in phonetic_data.py
- SyllableValidator colloquial checking
- WordValidator colloquial checking
- Different strictness levels (strict, lenient, off)
- ErrorType constants for colloquial variants
"""

import pytest

from myspellchecker.core.constants import ErrorType
from myspellchecker.text.phonetic_data import (
    COLLOQUIAL_SUBSTITUTIONS,
    STANDARD_TO_COLLOQUIAL,
    get_standard_forms,
    is_colloquial_variant,
)


class TestColloquialDataStructures:
    """Tests for colloquial data structures in phonetic_data.py."""

    def test_colloquial_substitutions_is_dict(self):
        """Test COLLOQUIAL_SUBSTITUTIONS is a dictionary."""
        assert isinstance(COLLOQUIAL_SUBSTITUTIONS, dict)
        assert len(COLLOQUIAL_SUBSTITUTIONS) > 0

    def test_colloquial_substitutions_values_are_sets(self):
        """Test all COLLOQUIAL_SUBSTITUTIONS values are sets."""
        for colloquial, standards in COLLOQUIAL_SUBSTITUTIONS.items():
            assert isinstance(standards, set), f"{colloquial} value is not a set"
            assert len(standards) > 0, f"{colloquial} has empty standards set"

    def test_standard_to_colloquial_is_reverse_mapping(self):
        """Test STANDARD_TO_COLLOQUIAL is the reverse mapping."""
        assert isinstance(STANDARD_TO_COLLOQUIAL, dict)
        # Verify reverse mapping correctness
        for colloquial, standards in COLLOQUIAL_SUBSTITUTIONS.items():
            for standard in standards:
                assert standard in STANDARD_TO_COLLOQUIAL
                assert colloquial in STANDARD_TO_COLLOQUIAL[standard]

    def test_key_colloquial_mappings_exist(self):
        """Test key colloquial -> standard mappings exist."""
        # Pronoun colloquialisms
        assert "ကျနော်" in COLLOQUIAL_SUBSTITUTIONS
        assert "ကျွန်တော်" in COLLOQUIAL_SUBSTITUTIONS["ကျနော်"]

        assert "ကျမ" in COLLOQUIAL_SUBSTITUTIONS
        assert "ကျွန်မ" in COLLOQUIAL_SUBSTITUTIONS["ကျမ"]

        # Common word colloquialisms
        assert "အဲ" in COLLOQUIAL_SUBSTITUTIONS
        assert "ထို" in COLLOQUIAL_SUBSTITUTIONS["အဲ"]

        assert "ဘယ်လို" in COLLOQUIAL_SUBSTITUTIONS
        assert "မည်သို့" in COLLOQUIAL_SUBSTITUTIONS["ဘယ်လို"]


class TestIsColloquialVariant:
    """Tests for is_colloquial_variant function."""

    def test_colloquial_words_return_true(self):
        """Test known colloquial words return True."""
        assert is_colloquial_variant("ကျနော်") is True
        assert is_colloquial_variant("ကျမ") is True
        assert is_colloquial_variant("မင်း") is True
        assert is_colloquial_variant("အဲ") is True
        assert is_colloquial_variant("ဘယ်လို") is True

    def test_standard_words_return_false(self):
        """Test standard (non-colloquial) words return False."""
        assert is_colloquial_variant("ကျွန်တော်") is False
        assert is_colloquial_variant("ကျွန်မ") is False
        assert is_colloquial_variant("ဤ") is False
        assert is_colloquial_variant("အဘယ်") is False

    def test_unknown_words_return_false(self):
        """Test unknown words return False."""
        assert is_colloquial_variant("မြန်မာ") is False
        assert is_colloquial_variant("စာ") is False
        assert is_colloquial_variant("ကျောင်း") is False

    def test_empty_string_returns_false(self):
        """Test empty string returns False."""
        assert is_colloquial_variant("") is False


class TestGetStandardForms:
    """Tests for get_standard_forms function."""

    def test_colloquial_words_return_standards(self):
        """Test colloquial words return their standard forms."""
        standards = get_standard_forms("ကျနော်")
        assert "ကျွန်တော်" in standards

        standards = get_standard_forms("ကျမ")
        assert "ကျွန်မ" in standards

        standards = get_standard_forms("အဲ")
        assert "ထို" in standards

    def test_informal_pronouns_map_to_multiple_standards(self):
        """Test very informal pronouns can map to multiple standards."""
        standards = get_standard_forms("ငါ")
        # ငါ can map to either male or female formal form
        assert "ကျွန်တော်" in standards or "ကျွန်မ" in standards

    def test_unknown_words_return_empty_set(self):
        """Test unknown words return empty set."""
        assert get_standard_forms("မြန်မာ") == set()
        assert get_standard_forms("unknown") == set()
        assert get_standard_forms("") == set()


class TestErrorTypeConstants:
    """Tests for ErrorType constants related to colloquial variants."""

    def test_colloquial_variant_error_type_exists(self):
        """Test COLLOQUIAL_VARIANT error type exists."""
        assert hasattr(ErrorType, "COLLOQUIAL_VARIANT")
        assert ErrorType.COLLOQUIAL_VARIANT.value == "colloquial_variant"

    def test_colloquial_info_error_type_exists(self):
        """Test COLLOQUIAL_INFO error type exists."""
        assert hasattr(ErrorType, "COLLOQUIAL_INFO")
        assert ErrorType.COLLOQUIAL_INFO.value == "colloquial_info"


class TestColloquialStrictnessConfig:
    """Tests for colloquial strictness configuration."""

    def test_default_strictness_is_lenient(self):
        """Test default colloquial_strictness is 'lenient'."""
        from myspellchecker.core.config import ValidationConfig

        config = ValidationConfig()
        assert config.colloquial_strictness == "lenient"

    def test_strictness_accepts_valid_values(self):
        """Test colloquial_strictness accepts valid values."""
        from myspellchecker.core.config import ValidationConfig

        # These should not raise
        config_strict = ValidationConfig(colloquial_strictness="strict")
        assert config_strict.colloquial_strictness == "strict"

        config_lenient = ValidationConfig(colloquial_strictness="lenient")
        assert config_lenient.colloquial_strictness == "lenient"

        config_off = ValidationConfig(colloquial_strictness="off")
        assert config_off.colloquial_strictness == "off"

    def test_strictness_rejects_invalid_values(self):
        """Test colloquial_strictness rejects invalid values."""
        from pydantic import ValidationError

        from myspellchecker.core.config import ValidationConfig

        with pytest.raises(ValidationError):
            ValidationConfig(colloquial_strictness="invalid")

    def test_colloquial_info_confidence_default(self):
        """Test colloquial_info_confidence has correct default."""
        from myspellchecker.core.config import ValidationConfig

        config = ValidationConfig()
        assert config.colloquial_info_confidence == 0.3


class TestSyllableValidatorColloquialCheck:
    """Tests for SyllableValidator colloquial checking."""

    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies for SyllableValidator."""
        from unittest.mock import MagicMock

        segmenter = MagicMock()
        repository = MagicMock()
        symspell = MagicMock()
        return segmenter, repository, symspell

    @pytest.fixture
    def validator_strict(self, mock_dependencies):
        """Create SyllableValidator with strict colloquial mode."""
        from myspellchecker.core.config import SpellCheckerConfig, ValidationConfig
        from myspellchecker.core.validators import SyllableValidator

        segmenter, repository, symspell = mock_dependencies
        config = SpellCheckerConfig(validation=ValidationConfig(colloquial_strictness="strict"))
        return SyllableValidator(
            config=config,
            segmenter=segmenter,
            repository=repository,
            symspell=symspell,
        )

    @pytest.fixture
    def validator_lenient(self, mock_dependencies):
        """Create SyllableValidator with lenient colloquial mode."""
        from myspellchecker.core.config import SpellCheckerConfig, ValidationConfig
        from myspellchecker.core.validators import SyllableValidator

        segmenter, repository, symspell = mock_dependencies
        config = SpellCheckerConfig(validation=ValidationConfig(colloquial_strictness="lenient"))
        return SyllableValidator(
            config=config,
            segmenter=segmenter,
            repository=repository,
            symspell=symspell,
        )

    @pytest.fixture
    def validator_off(self, mock_dependencies):
        """Create SyllableValidator with colloquial checking off."""
        from myspellchecker.core.config import SpellCheckerConfig, ValidationConfig
        from myspellchecker.core.validators import SyllableValidator

        segmenter, repository, symspell = mock_dependencies
        config = SpellCheckerConfig(validation=ValidationConfig(colloquial_strictness="off"))
        return SyllableValidator(
            config=config,
            segmenter=segmenter,
            repository=repository,
            symspell=symspell,
        )

    def test_check_colloquial_variant_method_exists(self, validator_lenient):
        """Test _check_colloquial_variant method exists."""
        assert hasattr(validator_lenient, "_check_colloquial_variant")
        assert callable(validator_lenient._check_colloquial_variant)

    def test_strict_mode_returns_error(self, validator_strict):
        """Test strict mode returns error for colloquial words."""
        result = validator_strict._check_colloquial_variant("ကျနော်", 0)

        assert result is not None
        assert result.error_type == ErrorType.COLLOQUIAL_VARIANT.value
        assert "ကျွန်တော်" in result.suggestions

    def test_lenient_mode_returns_info(self, validator_lenient):
        """Test lenient mode returns info note for colloquial words."""
        result = validator_lenient._check_colloquial_variant("ကျနော်", 0)

        assert result is not None
        assert result.error_type == ErrorType.COLLOQUIAL_INFO.value
        assert "ကျွန်တော်" in result.suggestions

    def test_off_mode_returns_none(self, validator_off):
        """Test off mode returns None for colloquial words."""
        result = validator_off._check_colloquial_variant("ကျနော်", 0)
        assert result is None

    def test_non_colloquial_returns_none(self, validator_strict):
        """Test non-colloquial words return None."""
        result = validator_strict._check_colloquial_variant("မြန်မာ", 0)
        assert result is None


class TestWordValidatorColloquialCheck:
    """Tests for WordValidator colloquial checking."""

    @pytest.fixture
    def mock_word_dependencies(self):
        """Create mock dependencies for WordValidator."""
        from unittest.mock import MagicMock

        segmenter = MagicMock()
        word_repository = MagicMock()
        syllable_repository = MagicMock()
        symspell = MagicMock()
        return segmenter, word_repository, syllable_repository, symspell

    @pytest.fixture
    def word_validator_strict(self, mock_word_dependencies):
        """Create WordValidator with strict colloquial mode."""
        from myspellchecker.core.config import SpellCheckerConfig, ValidationConfig
        from myspellchecker.core.validators import WordValidator

        segmenter, word_repo, syllable_repo, symspell = mock_word_dependencies
        config = SpellCheckerConfig(validation=ValidationConfig(colloquial_strictness="strict"))
        return WordValidator(
            config=config,
            segmenter=segmenter,
            word_repository=word_repo,
            syllable_repository=syllable_repo,
            symspell=symspell,
        )

    @pytest.fixture
    def word_validator_lenient(self, mock_word_dependencies):
        """Create WordValidator with lenient colloquial mode."""
        from myspellchecker.core.config import SpellCheckerConfig, ValidationConfig
        from myspellchecker.core.validators import WordValidator

        segmenter, word_repo, syllable_repo, symspell = mock_word_dependencies
        config = SpellCheckerConfig(validation=ValidationConfig(colloquial_strictness="lenient"))
        return WordValidator(
            config=config,
            segmenter=segmenter,
            word_repository=word_repo,
            syllable_repository=syllable_repo,
            symspell=symspell,
        )

    def test_check_colloquial_variant_method_exists(self, word_validator_lenient):
        """Test _check_colloquial_variant method exists in WordValidator."""
        assert hasattr(word_validator_lenient, "_check_colloquial_variant")
        assert callable(word_validator_lenient._check_colloquial_variant)

    def test_word_validator_strict_returns_error(self, word_validator_strict):
        """Test WordValidator strict mode returns error for colloquial words."""
        result = word_validator_strict._check_colloquial_variant("အဲ", 0)

        assert result is not None
        assert result.error_type == ErrorType.COLLOQUIAL_VARIANT.value
        assert "ထို" in result.suggestions

    def test_word_validator_lenient_returns_info(self, word_validator_lenient):
        """Test WordValidator lenient mode returns info for colloquial words."""
        result = word_validator_lenient._check_colloquial_variant("အဲ", 0)

        assert result is not None
        assert result.error_type == ErrorType.COLLOQUIAL_INFO.value


class TestColloquialVariantExamples:
    """Integration tests with common colloquial usage examples."""

    def test_pronoun_colloquialisms(self):
        """Test pronoun colloquialisms are detected correctly."""
        # Male first person
        assert is_colloquial_variant("ကျနော်") is True
        assert "ကျွန်တော်" in get_standard_forms("ကျနော်")

        # Female first person
        assert is_colloquial_variant("ကျမ") is True
        assert "ကျွန်မ" in get_standard_forms("ကျမ")

        # Second person informal
        assert is_colloquial_variant("မင်း") is True
        assert "သင်" in get_standard_forms("မင်း")

    def test_demonstrative_colloquialisms(self):
        """Test demonstrative colloquialisms are detected correctly."""
        # That (colloquial -> formal)
        assert is_colloquial_variant("အဲ") is True
        assert "ထို" in get_standard_forms("အဲ")

        # That thing (colloquial -> formal)
        assert is_colloquial_variant("အဲဒါ") is True
        assert "ထိုအရာ" in get_standard_forms("အဲဒါ")

    def test_question_word_colloquialisms(self):
        """Test question word colloquialisms are detected correctly."""
        # How (colloquial -> formal)
        assert is_colloquial_variant("ဘယ်လို") is True
        assert "မည်သို့" in get_standard_forms("ဘယ်လို")

        # Why (colloquial -> formal)
        assert is_colloquial_variant("ဘာကြောင့်") is True
        assert "အဘယ်ကြောင့်" in get_standard_forms("ဘာကြောင့်")

    def test_adverb_colloquialisms(self):
        """Test adverb colloquialisms are detected correctly."""
        # Very (multiple colloquial forms)
        assert is_colloquial_variant("တော်တော်") is True
        assert is_colloquial_variant("သိပ်") is True

        standards = get_standard_forms("တော်တော်")
        assert "အလွန်" in standards

    def test_reduplication_variants(self):
        """Test reduplication variants are detected correctly."""
        assert is_colloquial_variant("ကောင်းကောင်း") is True
        assert "ကောင်းမွန်စွာ" in get_standard_forms("ကောင်းကောင်း")

        assert is_colloquial_variant("မြန်မြန်") is True
        assert "မြန်ဆန်စွာ" in get_standard_forms("မြန်မြန်")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
