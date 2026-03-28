"""
Unit tests for SemanticChecker proactive scanning integration.

Tests the use_proactive_scanning config option and the integration
of scan_sentence() in the validation pipeline via SemanticValidationStrategy.
"""

from unittest.mock import Mock

import pytest

from myspellchecker.core.config import SemanticConfig, SpellCheckerConfig
from myspellchecker.core.context_validator import ContextValidator
from myspellchecker.core.validation_strategies import (
    SemanticValidationStrategy,
    ValidationContext,
)


class TestSemanticConfigProactiveScan:
    """Tests for SemanticConfig proactive scanning options."""

    def test_default_proactive_scanning_disabled(self):
        """Test that proactive scanning is disabled by default."""
        config = SemanticConfig()
        assert config.use_proactive_scanning is False

    def test_default_confidence_threshold(self):
        """Test that default confidence threshold is 0.85."""
        config = SemanticConfig()
        assert config.proactive_confidence_threshold == 0.85

    def test_use_proactive_scanning(self):
        """Test enabling proactive scanning."""
        config = SemanticConfig(use_proactive_scanning=True)
        assert config.use_proactive_scanning is True

    def test_custom_confidence_threshold(self):
        """Test setting custom confidence threshold."""
        config = SemanticConfig(proactive_confidence_threshold=0.7)
        assert config.proactive_confidence_threshold == 0.7

    def test_confidence_threshold_validation_min(self):
        """Test that confidence threshold rejects values below 0.0."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            SemanticConfig(proactive_confidence_threshold=-0.1)

    def test_confidence_threshold_validation_max(self):
        """Test that confidence threshold rejects values above 1.0."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            SemanticConfig(proactive_confidence_threshold=1.5)

    def test_confidence_threshold_boundary_values(self):
        """Test that boundary values 0.0 and 1.0 are accepted."""
        config_min = SemanticConfig(proactive_confidence_threshold=0.0)
        assert config_min.proactive_confidence_threshold == 0.0

        config_max = SemanticConfig(proactive_confidence_threshold=1.0)
        assert config_max.proactive_confidence_threshold == 1.0


class TestSemanticValidationStrategyProactiveScan:
    """Tests for SemanticValidationStrategy proactive scanning."""

    def test_validate_returns_empty_without_checker(self):
        """Test that validate returns empty without semantic checker."""
        strategy = SemanticValidationStrategy(
            semantic_checker=None,
            use_proactive_scanning=True,
        )

        context = ValidationContext(
            words=["test", "sentence"],
            word_positions=[0, 5],
            sentence="test sentence",
            is_name_mask=[False, False],
        )

        errors = strategy.validate(context)
        assert errors == []

    def test_validate_returns_empty_when_disabled(self):
        """Test that validate returns empty when proactive scanning disabled."""
        mock_semantic = Mock()
        strategy = SemanticValidationStrategy(
            semantic_checker=mock_semantic,
            use_proactive_scanning=False,
        )

        context = ValidationContext(
            words=["test", "sentence"],
            word_positions=[0, 5],
            sentence="test sentence",
            is_name_mask=[False, False],
        )

        errors = strategy.validate(context)
        assert errors == []

    def test_validate_calls_semantic_checker(self):
        """Test that validate calls scan_sentence for proactive scanning."""
        mock_semantic = Mock()
        mock_semantic.scan_sentence.return_value = []

        strategy = SemanticValidationStrategy(
            semantic_checker=mock_semantic,
            use_proactive_scanning=True,
        )

        # Sentence with subject particle က — triggers subject-position check
        context = ValidationContext(
            words=["စားပွဲ", "က", "ကျိုး", "တယ်"],
            word_positions=[0, 12, 15, 24],
            sentence="စားပွဲ က ကျိုး တယ်",
            is_name_mask=[False, False, False, False],
        )

        strategy.validate(context)

        # Verify scan_sentence was called
        assert mock_semantic.scan_sentence.called

    def test_validate_skips_existing_errors(self):
        """Test that validate skips positions with existing errors."""
        mock_semantic = Mock()
        mock_semantic.scan_sentence.return_value = []

        strategy = SemanticValidationStrategy(
            semantic_checker=mock_semantic,
            use_proactive_scanning=True,
        )

        context = ValidationContext(
            words=["test", "error", "word"],
            word_positions=[0, 5, 11],
            sentence="test error word",
            is_name_mask=[False, False, False],
            existing_errors={5: "test"},  # Position 5 already has an error
        )

        errors = strategy.validate(context)
        # Should not return errors for position 5
        assert all(e.position != 5 for e in errors)

    def test_validate_skips_names(self):
        """Test that validate skips words marked as names."""
        mock_semantic = Mock()
        # Return a result for word at index 1 (which is marked as a name)
        mock_semantic.scan_sentence.return_value = [(1, "Aung", ["Aye"], 0.8)]

        strategy = SemanticValidationStrategy(
            semantic_checker=mock_semantic,
            use_proactive_scanning=True,
        )

        context = ValidationContext(
            words=["U", "Aung"],
            word_positions=[0, 2],
            sentence="U Aung",
            is_name_mask=[False, True],  # "Aung" is a name
        )

        errors = strategy.validate(context)

        # Should skip "Aung" because it's a name
        assert all(e.text != "Aung" for e in errors)

    def test_validate_handles_exception(self):
        """Test that validate handles exceptions gracefully."""
        mock_semantic = Mock()
        mock_semantic.scan_sentence.side_effect = RuntimeError("Model error")

        strategy = SemanticValidationStrategy(
            semantic_checker=mock_semantic,
            use_proactive_scanning=True,
        )

        context = ValidationContext(
            words=["test", "sentence"],
            word_positions=[0, 5],
            sentence="test sentence",
            is_name_mask=[False, False],
        )

        # Should not raise exception
        errors = strategy.validate(context)
        assert errors == []


class TestProactiveScanIntegration:
    """Integration tests for proactive scanning via ContextValidator."""

    def test_context_validator_with_semantic_strategy(self):
        """Test ContextValidator with SemanticValidationStrategy."""
        config = SpellCheckerConfig()

        mock_semantic = Mock()
        mock_semantic.scan_sentence.return_value = []

        strategy = SemanticValidationStrategy(
            semantic_checker=mock_semantic,
            use_proactive_scanning=True,
        )

        mock_segmenter = Mock()
        mock_segmenter.segment_sentences.return_value = ["test sentence"]
        mock_segmenter.segment_words.return_value = ["test", "sentence"]

        validator = ContextValidator(
            config=config,
            segmenter=mock_segmenter,
            strategies=[strategy],
        )

        assert hasattr(validator, "strategies")
        assert isinstance(validator.strategies, list)

    def test_semantic_strategy_in_validator(self):
        """Test that SemanticValidationStrategy is properly added."""
        config = SpellCheckerConfig()

        mock_semantic = Mock()
        mock_semantic.scan_sentence.return_value = []
        strategy = SemanticValidationStrategy(
            semantic_checker=mock_semantic,
            use_proactive_scanning=True,
        )

        mock_segmenter = Mock()
        mock_segmenter.segment_sentences.return_value = ["test"]
        mock_segmenter.segment_words.return_value = ["test"]

        validator = ContextValidator(
            config=config,
            segmenter=mock_segmenter,
            strategies=[strategy],
        )

        strategy_types = [type(s).__name__ for s in validator.strategies]
        assert "SemanticValidationStrategy" in strategy_types
