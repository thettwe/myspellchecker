"""
Tests for Extended Myanmar validation configuration and propagation.

Tests cover:
1. ValidationConfig.allow_extended_myanmar field
2. Validator functions with allow_extended_myanmar parameter
3. Environment variable mapping
4. Propagation to ContextValidator, SyllableValidator, factories
"""

from myspellchecker.core.config.validation_configs import ValidationConfig


class TestValidationConfigAllowExtendedMyanmar:
    """Tests for ValidationConfig.allow_extended_myanmar field."""

    def test_default_is_false(self) -> None:
        """allow_extended_myanmar should default to False."""
        config = ValidationConfig()
        assert config.allow_extended_myanmar is False

    def test_can_set_to_true(self) -> None:
        """allow_extended_myanmar can be set to True."""
        config = ValidationConfig(allow_extended_myanmar=True)
        assert config.allow_extended_myanmar is True


class TestValidateTextExtendedMyanmar:
    """Tests for validate_text() with Extended Myanmar characters."""

    def test_flags_extended_a_and_b_by_default(self) -> None:
        """validate_text() should flag both Extended-A and Extended-B chars by default."""
        from myspellchecker.text.validator import ValidationIssue, validate_text

        for char in ["\uaa60", "\ua9e0"]:
            result = validate_text(f"test{char}text")
            issues = [issue for issue, _ in result.issues]
            assert ValidationIssue.EXTENDED_MYANMAR in issues

    def test_allows_extended_when_true(self) -> None:
        """validate_text() should NOT flag Extended chars when allow_extended_myanmar=True."""
        from myspellchecker.text.validator import ValidationIssue, validate_text

        for char in ["\uaa60", "\ua9e0"]:
            result = validate_text(f"test{char}text", allow_extended_myanmar=True)
            issues = [issue for issue, _ in result.issues]
            assert ValidationIssue.EXTENDED_MYANMAR not in issues


class TestEnvMappingAllowExtendedMyanmar:
    """Tests for MYSPELL_ALLOW_EXTENDED_MYANMAR environment variable mapping."""

    def test_env_var_sets_config(self, monkeypatch) -> None:
        """MYSPELL_ALLOW_EXTENDED_MYANMAR env var should set config."""
        from myspellchecker.core.config.loader import ConfigLoader

        monkeypatch.setenv("MYSPELL_ALLOW_EXTENDED_MYANMAR", "true")
        config = ConfigLoader().load(use_env=True)
        assert config.validation.allow_extended_myanmar is True


class TestValidatorIsMyanmarWithConfig:
    """Tests for Validator._is_myanmar_with_config() method."""

    def test_is_myanmar_with_config_default_strict(self) -> None:
        """_is_myanmar_with_config should use strict scope by default."""
        from unittest.mock import MagicMock

        from myspellchecker.core.config import SpellCheckerConfig
        from myspellchecker.core.validators import SyllableValidator

        config = SpellCheckerConfig()
        validator = SyllableValidator(
            config=config,
            segmenter=MagicMock(),
            repository=MagicMock(),
            symspell=MagicMock(),
            syllable_rule_validator=MagicMock(),
        )

        assert validator._is_myanmar_with_config("\uaa60") is False
        assert validator._is_myanmar_with_config("မြန်မာ") is True
        assert validator._is_myanmar_with_config("") is False

    def test_is_myanmar_with_config_extended_enabled(self) -> None:
        """_is_myanmar_with_config should accept extended when config allows."""
        from unittest.mock import MagicMock

        from myspellchecker.core.config import SpellCheckerConfig
        from myspellchecker.core.validators import SyllableValidator

        validation = ValidationConfig(allow_extended_myanmar=True)
        config = SpellCheckerConfig(validation=validation)
        validator = SyllableValidator(
            config=config,
            segmenter=MagicMock(),
            repository=MagicMock(),
            symspell=MagicMock(),
            syllable_rule_validator=MagicMock(),
        )

        assert validator._is_myanmar_with_config("\uaa60") is True


class TestValidatorFilterSuggestions:
    """Tests for Validator._filter_suggestions() respecting config."""

    def test_filter_suggestions_uses_config_flag(self) -> None:
        """_filter_suggestions should pass allow_extended_myanmar to validate_word."""
        from unittest.mock import MagicMock, patch

        from myspellchecker.core.config import SpellCheckerConfig
        from myspellchecker.core.validators import SyllableValidator

        validation = ValidationConfig(allow_extended_myanmar=True)
        config = SpellCheckerConfig(validation=validation)
        validator = SyllableValidator(
            config=config,
            segmenter=MagicMock(),
            repository=MagicMock(),
            symspell=MagicMock(),
            syllable_rule_validator=MagicMock(),
        )

        with patch("myspellchecker.core.validators.base.validate_word") as mock_validate_word:
            mock_validate_word.return_value = True
            validator._filter_suggestions(["မြန်မာ"])

            for call in mock_validate_word.call_args_list:
                _, kwargs = call
                assert kwargs.get("allow_extended_myanmar") is True


class TestFactoryFlagPropagation:
    """Tests for factory methods passing allow_extended_myanmar."""

    def test_build_suggestion_strategy_passes_flag(self) -> None:
        """build_suggestion_strategy should pass allow_extended_myanmar from config."""
        from unittest.mock import MagicMock, patch

        from myspellchecker.core.config import SpellCheckerConfig

        validation = ValidationConfig(allow_extended_myanmar=True)
        config = SpellCheckerConfig(validation=validation)

        mock_symspell = MagicMock()
        mock_provider = MagicMock()
        mock_provider.is_valid_word.return_value = True
        mock_provider.is_valid_syllable.return_value = True

        with patch(
            "myspellchecker.algorithms.suggestion_strategy.MorphologySuggestionStrategy"
        ) as MockMorphStrategy:
            MockMorphStrategy.return_value = MagicMock()

            from myspellchecker.core.factories.builders import build_suggestion_strategy

            build_suggestion_strategy(symspell=mock_symspell, provider=mock_provider, config=config)

            _, kwargs = MockMorphStrategy.call_args
            assert kwargs.get("allow_extended_myanmar") is True

    def test_create_semantic_checker_passes_flag(self) -> None:
        """create_semantic_checker should pass allow_extended_myanmar from config."""
        from unittest.mock import MagicMock, patch

        from myspellchecker.core.config import SpellCheckerConfig
        from myspellchecker.core.config.algorithm_configs import SemanticConfig

        validation = ValidationConfig(allow_extended_myanmar=True)
        semantic = SemanticConfig(model_path="/fake/model.onnx", tokenizer_path="/fake/tokenizer")
        config = SpellCheckerConfig(validation=validation, semantic=semantic)

        mock_container = MagicMock()
        mock_container.get_config.return_value = config

        with patch(
            "myspellchecker.algorithms.semantic_checker.SemanticChecker"
        ) as MockSemanticChecker:
            MockSemanticChecker.return_value = MagicMock()

            from myspellchecker.core.factories.optional_services_factory import (
                create_semantic_checker,
            )

            create_semantic_checker(mock_container)

            _, kwargs = MockSemanticChecker.call_args
            assert kwargs.get("allow_extended_myanmar") is True


class TestContextValidatorExtendedMyanmar:
    """Tests for ContextValidator respecting allow_extended_myanmar config."""

    def test_context_validator_extended_a_config_behavior(self) -> None:
        """ContextValidator should accept/reject Extended-A based on config."""
        from unittest.mock import MagicMock

        from myspellchecker.core.config import SpellCheckerConfig
        from myspellchecker.core.context_validator import ContextValidator

        extended_a_word = "\uaa60"

        for allow, expected in [(True, True), (False, False)]:
            validation = ValidationConfig(allow_extended_myanmar=allow)
            config = SpellCheckerConfig(validation=validation)
            mock_segmenter = MagicMock()
            validator = ContextValidator(config=config, segmenter=mock_segmenter, strategies=[])
            assert validator._is_myanmar_with_config(extended_a_word) is expected


class TestValidatorIsMyanmarRemoved:
    """Tests verifying Validator.is_myanmar() was removed and replacement works."""

    def test_is_myanmar_static_method_removed(self) -> None:
        """Validator.is_myanmar() static method should not exist."""
        from myspellchecker.core.validators import Validator

        assert not hasattr(Validator, "is_myanmar")

    def test_contains_myanmar_helper_works(self) -> None:
        """contains_myanmar() from constants is the replacement."""
        from myspellchecker.core.constants import contains_myanmar

        assert contains_myanmar("မြန်မာ") is True
        assert contains_myanmar("english") is False
        assert contains_myanmar("") is False
