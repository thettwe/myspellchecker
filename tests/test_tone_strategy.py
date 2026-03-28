from unittest.mock import MagicMock

from myspellchecker.core.validation_strategies.base import ValidationContext
from myspellchecker.core.validation_strategies.tone_strategy import ToneValidationStrategy


def test_tone_strategy_disabled():
    """Test when disabled."""
    strategy = ToneValidationStrategy(None)
    assert strategy.validate(MagicMock()) == []
    assert "disabled" in repr(strategy)


def test_tone_strategy_repr():
    """Test __repr__."""
    checker = MagicMock()
    strategy = ToneValidationStrategy(checker)
    assert "enabled" in repr(strategy)


def test_tone_strategy_validate_success():
    """Test successful tone correction."""
    checker = MagicMock()
    # (idx, original, correction, confidence)
    checker.check_sentence.return_value = [(1, "ပြီ", "ပြီး", 0.9)]

    strategy = ToneValidationStrategy(checker)
    context = ValidationContext(
        sentence="သူ ပြီ တယ်",
        words=["သူ", "ပြီ", "တယ်"],
        word_positions=[0, 4, 10],
        is_name_mask=[False, False, False],
    )

    errors = strategy.validate(context)
    assert len(errors) == 1
    assert errors[0].text == "ပြီ"
    assert errors[0].suggestions == ["ပြီး"]
    assert 4 in context.existing_errors


def test_tone_strategy_skipping():
    """Test skipping logic."""
    checker = MagicMock()
    checker.check_sentence.return_value = [
        (0, "LowConf", "Sug", 0.1),  # Skip: low confidence
        (1, "Name", "Sug", 0.9),  # Skip: name
        (2, "Exists", "Sug", 0.9),  # Skip: existing error
        (99, "Bound", "Sug", 0.9),  # Skip: out of bounds
        (3, "Valid", "Sug", 0.9),  # Pass
    ]

    strategy = ToneValidationStrategy(checker, confidence_threshold=0.5)
    context = ValidationContext(
        sentence="LowConf Name Exists Valid",
        words=["LowConf", "Name", "Exists", "Valid"],
        word_positions=[0, 8, 13, 20],
        is_name_mask=[False, True, False, False],
    )
    context.existing_errors[13] = "test"

    errors = strategy.validate(context)
    assert len(errors) == 1
    assert errors[0].text == "Valid"


def test_tone_strategy_exception_handling():
    """Test exception handling."""
    checker = MagicMock()
    checker.check_sentence.side_effect = RuntimeError("Tone engine crashed")
    strategy = ToneValidationStrategy(checker)
    context = ValidationContext("test", ["test"], [0], [False])

    # Should catch and return empty
    errors = strategy.validate(context)
    assert errors == []
