from unittest.mock import MagicMock

from myspellchecker.core.validation_strategies.base import ValidationContext
from myspellchecker.core.validation_strategies.syntactic_strategy import SyntacticValidationStrategy


def test_syntactic_strategy_disabled():
    """Test when disabled."""
    strategy = SyntacticValidationStrategy(None)
    assert strategy.validate(MagicMock()) == []
    assert "disabled" in repr(strategy)


def test_syntactic_strategy_repr():
    """Test __repr__."""
    checker = MagicMock()
    strategy = SyntacticValidationStrategy(checker, confidence=0.95)
    assert "enabled" in repr(strategy)
    assert "confidence=0.95" in repr(strategy)


def test_syntactic_strategy_validate_success():
    """Test successful syntactic rule check."""
    checker = MagicMock()
    # (idx, word, suggestion, confidence)
    checker.check_sequence.return_value = [(1, "မှာ", "မှ", 0.85)]

    strategy = SyntacticValidationStrategy(checker)
    context = ValidationContext(
        sentence="သူ မှာ သွား",
        words=["သူ", "မှာ", "သွား"],
        word_positions=[0, 4, 10],
        is_name_mask=[False, False, False],
    )

    errors = strategy.validate(context)
    assert len(errors) == 1
    assert errors[0].text == "မှာ"
    assert errors[0].suggestions == ["မှ"]
    assert 4 in context.existing_errors


def test_syntactic_strategy_skipping():
    """Test skipping logic."""
    checker = MagicMock()
    checker.check_sequence.return_value = [
        (0, "Name", "Sug", 0.85),  # Skip: name
        (1, "Error", "Sug", 0.85),  # Skip: existing error
        (99, "Bound", "Sug", 0.85),  # Skip: out of bounds
        (2, "Valid", "Sug", 0.85),  # Pass
    ]

    strategy = SyntacticValidationStrategy(checker)
    context = ValidationContext(
        sentence="Name Error Valid",
        words=["Name", "Error", "Valid"],
        word_positions=[0, 5, 12],
        is_name_mask=[True, False, False],
    )
    context.existing_errors[5] = "test"

    errors = strategy.validate(context)
    assert len(errors) == 1
    assert errors[0].text == "Valid"


def test_syntactic_strategy_exception_handling():
    """Test exception handling."""
    checker = MagicMock()
    checker.check_sequence.side_effect = RuntimeError("Rule engine crashed")
    strategy = SyntacticValidationStrategy(checker)
    context = ValidationContext("test", ["test"], [0], [False])

    # Should catch and return empty
    errors = strategy.validate(context)
    assert errors == []


def test_syntactic_strategy_detects_duplicated_sentence_ending_token():
    checker = MagicMock()
    checker.check_sequence.return_value = []

    strategy = SyntacticValidationStrategy(checker)
    context = ValidationContext(
        sentence="သူ လာ သည်သည်",
        words=["သူ", "လာ", "သည်သည်"],
        word_positions=[0, 3, 7],
        is_name_mask=[False, False, False],
    )

    errors = strategy.validate(context)

    assert len(errors) == 1
    assert errors[0].error_type == "syntax_error"
    assert errors[0].text == "သည်သည်"
    assert errors[0].suggestions == ["သည်"]


def test_syntactic_strategy_detects_duplicated_sentence_ending_with_punctuation():
    checker = MagicMock()
    checker.check_sequence.return_value = []

    strategy = SyntacticValidationStrategy(checker)
    context = ValidationContext(
        sentence="သူ လာ ပါသည်ပါသည်။",
        words=["သူ", "လာ", "ပါသည်ပါသည်။"],
        word_positions=[0, 3, 7],
        is_name_mask=[False, False, False],
    )

    errors = strategy.validate(context)

    assert len(errors) == 1
    assert errors[0].text == "ပါသည်ပါသည်။"
    assert errors[0].suggestions == ["ပါသည်။"]
