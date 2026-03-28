"""
Unit tests for POS sequence validation in the spell checker.

Tests the POSSequenceValidationStrategy that detects grammatically
incorrect POS tag sequences.
"""

from unittest.mock import Mock

import pytest

from myspellchecker.core.config import SpellCheckerConfig
from myspellchecker.core.validation_strategies import (
    POSSequenceValidationStrategy,
    ValidationContext,
)
from myspellchecker.grammar.patterns import INVALID_POS_SEQUENCES


class TestInvalidPOSSequences:
    """Tests for the INVALID_POS_SEQUENCES dictionary."""

    def test_invalid_pos_sequences_exists(self):
        """Test that INVALID_POS_SEQUENCES is defined."""
        assert INVALID_POS_SEQUENCES is not None
        assert isinstance(INVALID_POS_SEQUENCES, dict)

    def test_invalid_pos_sequences_contains_key_patterns(self):
        """Test that key invalid patterns are defined."""
        # V-V: Two consecutive verbs without particle
        assert ("V", "V") in INVALID_POS_SEQUENCES
        # P-P: Two consecutive particles
        assert ("P", "P") in INVALID_POS_SEQUENCES
        # N-N: May need particle between nouns
        assert ("N", "N") in INVALID_POS_SEQUENCES

    def test_invalid_pos_sequences_format(self):
        """Test that sequence entries have correct format."""
        for (prev_tag, curr_tag), (severity, description) in INVALID_POS_SEQUENCES.items():
            assert isinstance(prev_tag, str)
            assert isinstance(curr_tag, str)
            assert severity in ("error", "warning", "info")
            assert isinstance(description, str)
            assert len(description) > 0


class TestPOSSequenceValidationStrategy:
    """Tests for POSSequenceValidationStrategy."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration."""
        config = SpellCheckerConfig()
        config.use_context_checker = True
        return config

    @pytest.fixture
    def mock_viterbi_tagger(self):
        """Create a mock ViterbiTagger."""
        tagger = Mock()
        return tagger

    def test_strategy_priority(self, mock_viterbi_tagger):
        """Test that strategy has correct priority."""
        strategy = POSSequenceValidationStrategy(mock_viterbi_tagger)
        assert strategy.priority() == 30

    def test_strategy_repr(self, mock_viterbi_tagger):
        """Test strategy string representation."""
        strategy = POSSequenceValidationStrategy(mock_viterbi_tagger)
        assert "POSSequenceValidationStrategy" in repr(strategy)

    def test_validate_no_tagger(self):
        """Test that validation returns empty list when no tagger is provided."""
        strategy = POSSequenceValidationStrategy(viterbi_tagger=None)

        context = ValidationContext(
            words=["စား", "သွား"],
            word_positions=[0, 4],
            sentence="စားသွား",
            is_name_mask=[False, False],
        )

        errors = strategy.validate(context)
        assert errors == []

    def test_validate_single_word(self, mock_viterbi_tagger):
        """Test that single word returns no errors."""
        strategy = POSSequenceValidationStrategy(mock_viterbi_tagger)

        context = ValidationContext(
            words=["စား"],
            word_positions=[0],
            sentence="စား",
            is_name_mask=[False],
        )

        errors = strategy.validate(context)
        assert errors == []

    def test_validate_valid_sequence(self, mock_viterbi_tagger):
        """Test that valid POS sequence returns no errors."""
        mock_viterbi_tagger.tag_sequence.return_value = ["N", "V", "P"]

        strategy = POSSequenceValidationStrategy(mock_viterbi_tagger)

        context = ValidationContext(
            words=["သူ", "စား", "သည်"],
            word_positions=[0, 3, 7],
            sentence="သူစားသည်",
            is_name_mask=[False, False, False],
        )

        errors = strategy.validate(context)
        assert errors == []

    def test_validate_consecutive_verbs(self, mock_viterbi_tagger):
        """Test that consecutive main verbs (V-V) are info-level, not flagged as errors."""
        mock_viterbi_tagger.tag_sequence.return_value = ["V", "V"]

        strategy = POSSequenceValidationStrategy(mock_viterbi_tagger)

        # V-V is severity "info" (serial verb constructions are common in Myanmar),
        # so the strategy should NOT flag it as an error.
        context = ValidationContext(
            words=["စား", "ရေး"],
            word_positions=[0, 4],
            sentence="စားရေး",
            is_name_mask=[False, False],
        )

        errors = strategy.validate(context)

        # V-V severity is "info", strategy only reports "error" severity
        assert len(errors) == 0

    def test_validate_valid_serial_verb_not_flagged(self, mock_viterbi_tagger):
        """Test that valid serial verb constructions (V + auxiliary V) are NOT flagged."""
        mock_viterbi_tagger.tag_sequence.return_value = ["V", "V"]

        strategy = POSSequenceValidationStrategy(mock_viterbi_tagger)

        # Use main verb + auxiliary verb (eat + go-away/direction marker)
        # This is valid in Myanmar serial verb constructions
        context = ValidationContext(
            words=["စား", "သွား"],
            word_positions=[0, 4],
            sentence="စားသွား",
            is_name_mask=[False, False],
        )

        errors = strategy.validate(context)

        # Should NOT be flagged as error - valid auxiliary verb pattern
        assert errors == []

    def test_validate_consecutive_particles(self, mock_viterbi_tagger):
        """Test that consecutive particles (P-P) are flagged as errors."""
        mock_viterbi_tagger.tag_sequence.return_value = ["V", "P", "P"]

        strategy = POSSequenceValidationStrategy(mock_viterbi_tagger)

        context = ValidationContext(
            words=["စား", "ပါ", "တယ်"],
            word_positions=[0, 4, 7],
            sentence="စားပါတယ်",
            is_name_mask=[False, False, False],
        )

        errors = strategy.validate(context)

        # P-P severity is "error", consecutive particles are flagged
        assert len(errors) == 1
        assert errors[0].error_type == "pos_sequence_error"
        assert errors[0].position == 7

    def test_validate_skip_names(self, mock_viterbi_tagger):
        """Test that names are skipped in POS validation."""
        mock_viterbi_tagger.tag_sequence.return_value = ["V", "V"]

        strategy = POSSequenceValidationStrategy(mock_viterbi_tagger)

        context = ValidationContext(
            words=["စား", "အောင်အောင်"],
            word_positions=[0, 4],
            sentence="စားအောင်အောင်",
            is_name_mask=[False, True],  # Second word is a name
        )

        errors = strategy.validate(context)

        # Should not flag as error since it's a name
        assert errors == []

    def test_validate_warning_not_reported(self, mock_viterbi_tagger):
        """Test that N-N (warning) is not reported as error."""
        mock_viterbi_tagger.tag_sequence.return_value = ["N", "N"]

        strategy = POSSequenceValidationStrategy(mock_viterbi_tagger)

        context = ValidationContext(
            words=["လူ", "အိမ်"],
            word_positions=[0, 3],
            sentence="လူအိမ်",
            is_name_mask=[False, False],
        )

        errors = strategy.validate(context)

        # N-N is a warning, not an error, so should not be reported
        assert errors == []

    def test_validate_tagger_exception(self, mock_viterbi_tagger):
        """Test graceful handling of tagger exceptions."""
        mock_viterbi_tagger.tag_sequence.side_effect = RuntimeError("Tagger error")

        strategy = POSSequenceValidationStrategy(mock_viterbi_tagger)

        context = ValidationContext(
            words=["စား", "သွား"],
            word_positions=[0, 4],
            sentence="စားသွား",
            is_name_mask=[False, False],
        )

        errors = strategy.validate(context)

        # Should return empty list on exception
        assert errors == []

    def test_validate_multiple_errors(self, mock_viterbi_tagger):
        """Test V-V (info, no error) and P-P (error, flagged)."""
        mock_viterbi_tagger.tag_sequence.return_value = ["V", "V", "P", "P"]

        strategy = POSSequenceValidationStrategy(mock_viterbi_tagger)

        # V-V is "info" severity (serial verbs are common) — not flagged.
        # P-P is "error" severity (consecutive particles) — flagged.
        context = ValidationContext(
            words=["စား", "ရေး", "ပါ", "တယ်"],
            word_positions=[0, 4, 9, 12],
            sentence="စားရေးပါတယ်",
            is_name_mask=[False, False, False, False],
        )

        errors = strategy.validate(context)

        # Only P-P produces an error; V-V is info-level
        assert len(errors) == 1
        assert errors[0].error_type == "pos_sequence_error"
        assert errors[0].position == 12


class TestContextValidatorPOSIntegration:
    """Integration tests for POS validation via ContextValidator with strategies."""

    @pytest.fixture
    def spellchecker(self):
        """Create a SpellChecker instance for integration testing."""
        try:
            from myspellchecker import SpellChecker

            return SpellChecker.create_default()
        except Exception as e:
            pytest.skip(f"SpellChecker not available: {e}")

    def test_context_validator_has_strategies(self, spellchecker):
        """Test that ContextValidator has validation strategies."""
        context_validator = spellchecker.context_validator
        assert context_validator is not None
        assert hasattr(context_validator, "strategies")
        assert isinstance(context_validator.strategies, list)

    def test_pos_sequence_strategy_in_validator(self, spellchecker):
        """Test that POSSequenceValidationStrategy is included in strategies."""
        context_validator = spellchecker.context_validator
        strategy_types = [type(s).__name__ for s in context_validator.strategies]
        assert "POSSequenceValidationStrategy" in strategy_types


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
