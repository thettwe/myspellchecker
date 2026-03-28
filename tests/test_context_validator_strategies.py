"""Unit tests for ContextValidator orchestrator."""

from unittest.mock import Mock, patch

import pytest


class TestContextValidatorInit:
    """Tests for ContextValidator initialization."""

    def test_init_with_strategies(self):
        """Test initialization with strategies."""
        from myspellchecker.core.context_validator import ContextValidator

        mock_config = Mock()
        mock_segmenter = Mock()

        mock_strategy1 = Mock()
        mock_strategy1.priority.return_value = 20
        mock_strategy2 = Mock()
        mock_strategy2.priority.return_value = 10

        validator = ContextValidator(
            config=mock_config,
            segmenter=mock_segmenter,
            strategies=[mock_strategy1, mock_strategy2],
        )

        assert validator.config == mock_config
        assert validator.segmenter == mock_segmenter
        # Strategies should be sorted by priority
        assert validator.strategies[0].priority() == 10
        assert validator.strategies[1].priority() == 20

    def test_init_without_strategies(self):
        """Test initialization without strategies."""
        from myspellchecker.core.context_validator import ContextValidator

        mock_config = Mock()
        mock_segmenter = Mock()

        validator = ContextValidator(
            config=mock_config,
            segmenter=mock_segmenter,
        )

        assert validator.strategies == []

    def test_init_with_name_heuristic(self):
        """Test initialization with name heuristic."""
        from myspellchecker.core.context_validator import ContextValidator

        mock_config = Mock()
        mock_segmenter = Mock()
        mock_name_heuristic = Mock()

        validator = ContextValidator(
            config=mock_config,
            segmenter=mock_segmenter,
            name_heuristic=mock_name_heuristic,
        )

        assert validator.name_heuristic == mock_name_heuristic


class TestContextValidatorValidate:
    """Tests for ContextValidator.validate method."""

    def test_validate_empty_strategies(self):
        """Test validate returns empty when no strategies."""
        from myspellchecker.core.context_validator import ContextValidator

        mock_config = Mock()
        mock_segmenter = Mock()

        validator = ContextValidator(
            config=mock_config,
            segmenter=mock_segmenter,
        )

        errors = validator.validate("test text")
        assert errors == []

    def test_validate_with_strategies(self):
        """Test validate calls strategies and returns errors."""
        from myspellchecker.core.config import SpellCheckerConfig
        from myspellchecker.core.config.validation_configs import ValidationConfig
        from myspellchecker.core.context_validator import ContextValidator
        from myspellchecker.core.response import ContextError

        # Use real config instead of Mock to ensure proper attribute access
        config = SpellCheckerConfig(
            use_ner=False,
            validation=ValidationConfig(raise_on_strategy_error=False),
        )

        mock_segmenter = Mock()
        mock_segmenter.segment_sentences.return_value = ["test sentence"]
        mock_segmenter.segment_words.return_value = ["test", "sentence"]

        mock_error = ContextError(
            text="test",
            position=0,
            error_type="test_error",
            suggestions=["suggestion"],
            confidence=0.8,
            probability=0.5,
            prev_word="",
        )

        mock_strategy = Mock()
        mock_strategy.priority.return_value = 10
        mock_strategy.validate.return_value = [mock_error]

        validator = ContextValidator(
            config=config,
            segmenter=mock_segmenter,
            strategies=[mock_strategy],
        )

        # Need to mock _is_myanmar_with_config and is_punctuation
        with patch.object(validator, "_is_myanmar_with_config", return_value=True):
            with patch.object(validator, "is_punctuation", return_value=False):
                errors = validator.validate("test sentence")

        assert len(errors) == 1
        assert errors[0].text == "test"

    def test_validate_empty_sentences(self):
        """Test validate handles empty sentences."""
        from myspellchecker.core.context_validator import ContextValidator

        mock_config = Mock()
        mock_config.use_ner = False

        mock_segmenter = Mock()
        mock_segmenter.segment_sentences.return_value = ["", ""]

        mock_strategy = Mock()
        mock_strategy.priority.return_value = 10

        validator = ContextValidator(
            config=mock_config,
            segmenter=mock_segmenter,
            strategies=[mock_strategy],
        )

        errors = validator.validate("test")
        assert errors == []

    def test_validate_sentence_not_found(self):
        """Test validate handles sentence not found in text."""
        from myspellchecker.core.context_validator import ContextValidator

        mock_config = Mock()
        mock_segmenter = Mock()
        mock_segmenter.segment_sentences.return_value = ["not in text"]

        mock_strategy = Mock()
        mock_strategy.priority.return_value = 10

        validator = ContextValidator(
            config=mock_config,
            segmenter=mock_segmenter,
            strategies=[mock_strategy],
        )

        # Original text doesn't contain the sentence
        errors = validator.validate("different text")
        assert errors == []


class TestContextValidatorValidateSentence:
    """Tests for ContextValidator._validate_sentence method."""

    def test_validate_sentence_empty_words(self):
        """Test _validate_sentence handles empty words."""
        from myspellchecker.core.context_validator import ContextValidator

        mock_config = Mock()
        mock_config.use_ner = False

        mock_segmenter = Mock()
        mock_segmenter.segment_words.return_value = []

        mock_strategy = Mock()
        mock_strategy.priority.return_value = 10

        validator = ContextValidator(
            config=mock_config,
            segmenter=mock_segmenter,
            strategies=[mock_strategy],
        )

        errors = validator._validate_sentence("test", 0)
        assert errors == []

    def test_validate_sentence_with_ner(self):
        """Test _validate_sentence uses name heuristic when enabled."""
        from myspellchecker.core.config import SpellCheckerConfig
        from myspellchecker.core.config.validation_configs import ValidationConfig
        from myspellchecker.core.context_validator import ContextValidator

        # Use real config instead of Mock to ensure proper attribute access
        config = SpellCheckerConfig(
            use_ner=True,
            validation=ValidationConfig(raise_on_strategy_error=False),
        )

        mock_segmenter = Mock()
        mock_segmenter.segment_words.return_value = ["John", "walks"]

        mock_name_heuristic = Mock()
        mock_name_heuristic.analyze_sentence.return_value = [True, False]

        mock_strategy = Mock()
        mock_strategy.priority.return_value = 10
        mock_strategy.validate.return_value = []

        validator = ContextValidator(
            config=config,
            segmenter=mock_segmenter,
            strategies=[mock_strategy],
            name_heuristic=mock_name_heuristic,
        )

        with patch.object(validator, "_is_myanmar_with_config", return_value=True):
            with patch.object(validator, "is_punctuation", return_value=False):
                validator._validate_sentence("John walks", 0)

        mock_name_heuristic.analyze_sentence.assert_called_once()

    def test_validate_sentence_filters_punctuation(self):
        """Test _validate_sentence filters punctuation."""
        from myspellchecker.core.config import SpellCheckerConfig
        from myspellchecker.core.config.validation_configs import ValidationConfig
        from myspellchecker.core.context_validator import ContextValidator

        # Use real config instead of Mock to ensure proper attribute access
        config = SpellCheckerConfig(
            use_ner=False,
            validation=ValidationConfig(raise_on_strategy_error=False),
        )

        mock_segmenter = Mock()
        mock_segmenter.segment_words.return_value = ["word", ".", "another"]

        mock_strategy = Mock()
        mock_strategy.priority.return_value = 10
        mock_strategy.validate.return_value = []

        validator = ContextValidator(
            config=config,
            segmenter=mock_segmenter,
            strategies=[mock_strategy],
        )

        with patch.object(validator, "_is_myanmar_with_config", return_value=True):
            with patch.object(validator, "is_punctuation", side_effect=[False, True, False]):
                validator._validate_sentence("word . another", 0)

        # Strategy should be called with filtered words
        mock_strategy.validate.assert_called()

    def test_validate_sentence_filters_non_myanmar(self):
        """Test _validate_sentence filters non-Myanmar text."""
        from myspellchecker.core.config import SpellCheckerConfig
        from myspellchecker.core.config.validation_configs import ValidationConfig
        from myspellchecker.core.context_validator import ContextValidator

        # Use real config instead of Mock to ensure proper attribute access
        config = SpellCheckerConfig(
            use_ner=False,
            validation=ValidationConfig(raise_on_strategy_error=False),
        )

        mock_segmenter = Mock()
        mock_segmenter.segment_words.return_value = ["word", "english", "another"]

        mock_strategy = Mock()
        mock_strategy.priority.return_value = 10
        mock_strategy.validate.return_value = []

        validator = ContextValidator(
            config=config,
            segmenter=mock_segmenter,
            strategies=[mock_strategy],
        )

        with patch.object(validator, "_is_myanmar_with_config", side_effect=[True, False, True]):
            with patch.object(validator, "is_punctuation", return_value=False):
                validator._validate_sentence("word english another", 0)

        mock_strategy.validate.assert_called()

    def test_validate_sentence_strategy_exception(self):
        """Test _validate_sentence handles strategy exceptions."""
        from myspellchecker.core.config import SpellCheckerConfig
        from myspellchecker.core.config.validation_configs import ValidationConfig
        from myspellchecker.core.context_validator import ContextValidator
        from myspellchecker.core.response import ContextError

        # Use real config instead of Mock to ensure proper attribute access
        config = SpellCheckerConfig(
            use_ner=False,
            validation=ValidationConfig(raise_on_strategy_error=False),
        )

        mock_segmenter = Mock()
        mock_segmenter.segment_words.return_value = ["word"]

        # First strategy throws, second returns error
        mock_strategy1 = Mock()
        mock_strategy1.priority.return_value = 10
        # Use RuntimeError which is one of the caught exception types
        mock_strategy1.validate.side_effect = RuntimeError("Test error")
        mock_strategy1.__class__.__name__ = "FailingStrategy"

        mock_error = ContextError(
            text="word",
            position=0,
            error_type="test_error",
            suggestions=[],
            confidence=0.8,
            probability=0.0,
            prev_word="",
        )
        mock_strategy2 = Mock()
        mock_strategy2.priority.return_value = 20
        mock_strategy2.validate.return_value = [mock_error]

        validator = ContextValidator(
            config=config,
            segmenter=mock_segmenter,
            strategies=[mock_strategy1, mock_strategy2],
        )

        with patch.object(validator, "_is_myanmar_with_config", return_value=True):
            with patch.object(validator, "is_punctuation", return_value=False):
                errors = validator._validate_sentence("word", 0)

        # Second strategy should still run
        assert len(errors) == 1

    def test_validate_sentence_word_not_found(self):
        """Test _validate_sentence handles word not found in sentence."""
        from myspellchecker.core.config import SpellCheckerConfig
        from myspellchecker.core.config.validation_configs import ValidationConfig
        from myspellchecker.core.context_validator import ContextValidator

        # Use real config instead of Mock to ensure proper attribute access
        config = SpellCheckerConfig(
            use_ner=False,
            validation=ValidationConfig(raise_on_strategy_error=False),
        )

        mock_segmenter = Mock()
        mock_segmenter.segment_words.return_value = ["notfound"]

        mock_strategy = Mock()
        mock_strategy.priority.return_value = 10
        mock_strategy.validate.return_value = []

        validator = ContextValidator(
            config=config,
            segmenter=mock_segmenter,
            strategies=[mock_strategy],
        )

        # Word "notfound" is not in "different text"
        errors = validator._validate_sentence("different text", 0)
        # Should handle gracefully
        assert errors == []


class TestContextValidatorRepr:
    """Tests for ContextValidator.__repr__ method."""

    def test_repr(self):
        """Test __repr__ returns expected format."""
        from myspellchecker.core.context_validator import ContextValidator

        mock_config = Mock()
        mock_segmenter = Mock()

        mock_strategy1 = Mock()
        mock_strategy1.priority.return_value = 10
        mock_strategy2 = Mock()
        mock_strategy2.priority.return_value = 20

        validator = ContextValidator(
            config=mock_config,
            segmenter=mock_segmenter,
            strategies=[mock_strategy1, mock_strategy2],
        )

        repr_str = repr(validator)
        assert "ContextValidator" in repr_str
        assert "strategies=2" in repr_str
        assert "priorities=[10, 20]" in repr_str


class TestMergeAppendedSuggestions:
    """Tests for _merge_appended_suggestions."""

    def test_merge_appends_new_suggestions(self):
        """Test that appended suggestions are merged into Error objects."""
        from myspellchecker.core.context_validator import ContextValidator
        from myspellchecker.core.response import ContextError
        from myspellchecker.core.validation_strategies.base import ValidationContext

        error = ContextError(
            text="သား",
            position=4,
            error_type="tone_ambiguity",
            suggestions=["သာ"],
            confidence=0.9,
        )

        context = ValidationContext(
            sentence="သူ သား တယ်",
            words=["သူ", "သား", "တယ်"],
            word_positions=[0, 4, 10],
        )
        # Homophone strategy appended "စား" via existing_suggestions
        context.existing_suggestions[4] = ["သာ", "စား"]

        ContextValidator._merge_appended_suggestions([error], context)

        assert error.suggestions == ["သာ", "စား"]

    def test_merge_no_duplicate_suggestions(self):
        """Test that merge does not create duplicate suggestions."""
        from myspellchecker.core.context_validator import ContextValidator
        from myspellchecker.core.response import ContextError
        from myspellchecker.core.validation_strategies.base import ValidationContext

        error = ContextError(
            text="သား",
            position=4,
            error_type="tone_ambiguity",
            suggestions=["သာ", "စား"],
            confidence=0.9,
        )

        context = ValidationContext(
            sentence="သူ သား တယ်",
            words=["သူ", "သား", "တယ်"],
            word_positions=[0, 4, 10],
        )
        context.existing_suggestions[4] = ["သာ", "စား"]

        ContextValidator._merge_appended_suggestions([error], context)

        # No duplicates
        assert error.suggestions == ["သာ", "စား"]

    def test_merge_no_suggestions_no_change(self):
        """Test that merge is a no-op when no suggestions were appended."""
        from myspellchecker.core.context_validator import ContextValidator
        from myspellchecker.core.response import ContextError
        from myspellchecker.core.validation_strategies.base import ValidationContext

        error = ContextError(
            text="သား",
            position=4,
            error_type="tone_ambiguity",
            suggestions=["သာ"],
            confidence=0.9,
        )

        context = ValidationContext(
            sentence="test",
            words=["test"],
            word_positions=[0],
        )

        ContextValidator._merge_appended_suggestions([error], context)

        assert error.suggestions == ["သာ"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
