from unittest.mock import MagicMock

from myspellchecker.core.validation_strategies.base import ValidationContext
from myspellchecker.core.validation_strategies.semantic_strategy import SemanticValidationStrategy


def test_semantic_strategy_disabled():
    """Test when disabled via flag or missing checker."""
    # Case 1: missing checker
    strategy1 = SemanticValidationStrategy(None, True)
    assert strategy1.validate(MagicMock()) == []
    assert "no model" in repr(strategy1)

    # Case 2: flag False
    strategy2 = SemanticValidationStrategy(MagicMock(), False)
    ctx2 = MagicMock()
    ctx2.global_error_count = 0
    assert strategy2.validate(ctx2) == []
    assert "proactive scanning off" in repr(strategy2)


def test_semantic_strategy_repr():
    """Test __repr__ for enabled state."""
    checker = MagicMock()
    strategy = SemanticValidationStrategy(checker, True, proactive_confidence_threshold=0.8)
    assert "enabled" in repr(strategy)
    assert "threshold=0.8" in repr(strategy)


def test_semantic_strategy_validate_success():
    """Test successful proactive scan via scan_sentence()."""
    checker = MagicMock()
    # scan_sentence returns: [(word_idx, error_word, suggestions, confidence)]
    checker.scan_sentence.return_value = [
        (0, "စားပွဲ", ["သူ", "ကျွန်တော်"], 0.85),
    ]

    strategy = SemanticValidationStrategy(checker, use_proactive_scanning=True)
    context = ValidationContext(
        sentence="စားပွဲ က ချက် တယ်",
        words=["စားပွဲ", "က", "ချက်", "တယ်"],
        word_positions=[0, 12, 15, 24],
        is_name_mask=[False, False, False, False],
    )

    errors = strategy.validate(context)
    assert len(errors) == 1
    assert errors[0].text == "စားပွဲ"
    assert errors[0].position == 0
    assert 0 in context.existing_errors


def test_semantic_strategy_skipping():
    """Test various skipping logic (names, existing errors)."""
    checker = MagicMock()
    # scan_sentence returns errors for indices 0 and 2
    checker.scan_sentence.return_value = [
        (0, "NameWord", ["သူ"], 0.9),  # will be skipped (name)
        (2, "စားပွဲ", ["သူ"], 0.85),  # valid detection
    ]

    strategy = SemanticValidationStrategy(checker, True)
    context = ValidationContext(
        sentence="NameWord က စားပွဲ က ချက် တယ်",
        words=["NameWord", "က", "စားပွဲ", "က", "ချက်", "တယ်"],
        word_positions=[0, 9, 12, 24, 27, 36],
        is_name_mask=[True, False, False, False, False, False],
    )

    errors = strategy.validate(context)
    # NameWord is skipped (is_name_mask=True), only စားပွဲ flagged
    assert len(errors) == 1
    assert errors[0].text == "စားပွဲ"


def test_semantic_strategy_skips_existing_errors():
    """Test that positions with existing errors are skipped."""
    checker = MagicMock()
    checker.scan_sentence.return_value = [
        (0, "စားပွဲ", ["သူ"], 0.9),
    ]

    strategy = SemanticValidationStrategy(checker, True)
    context = ValidationContext(
        sentence="စားပွဲ က ချက် တယ်",
        words=["စားပွဲ", "က", "ချက်", "တယ်"],
        word_positions=[0, 12, 15, 24],
        is_name_mask=[False, False, False, False],
    )
    # Pre-existing error at position 0
    context.existing_errors[0] = "some_error"

    errors = strategy.validate(context)
    assert len(errors) == 0


def test_semantic_strategy_skip_particles():
    """Test B2: particles/function words are skipped."""
    checker = MagicMock()
    # scan_sentence returns a particle word as error
    checker.scan_sentence.return_value = [
        (1, "ကို", ["က"], 0.9),  # particle — should be skipped
    ]

    strategy = SemanticValidationStrategy(checker, True)
    context = ValidationContext(
        sentence="စာ ကို ပေး တယ်",
        words=["စာ", "ကို", "ပေး", "တယ်"],
        word_positions=[0, 6, 12, 18],
        is_name_mask=[False, False, False, False],
    )

    errors = strategy.validate(context)
    assert len(errors) == 0


def test_semantic_strategy_skip_high_freq():
    """Test B1: high-frequency words are skipped."""
    checker = MagicMock()
    checker.scan_sentence.return_value = [
        (0, "ကောင်း", ["ခေါင်း"], 0.8),
    ]

    mock_provider = MagicMock()
    mock_provider.get_word_frequency.return_value = 100000  # > 50K threshold

    strategy = SemanticValidationStrategy(checker, True, provider=mock_provider)
    context = ValidationContext(
        sentence="ကောင်း ပါ တယ်",
        words=["ကောင်း", "ပါ", "တယ်"],
        word_positions=[0, 15, 21],
        is_name_mask=[False, False, False],
    )

    errors = strategy.validate(context)
    assert len(errors) == 0


def test_semantic_strategy_exception_graceful():
    """Test that it doesn't crash on model failure."""
    checker = MagicMock()
    checker.scan_sentence.side_effect = Exception("Model exploded")

    strategy = SemanticValidationStrategy(checker, True)
    context = ValidationContext("test", ["test"], [0], [False])

    # Should catch and return empty
    errors = strategy.validate(context)
    assert errors == []


def test_semantic_strategy_main_exception():
    """Test exception in validate high level."""
    strategy = SemanticValidationStrategy(MagicMock(), True)
    # Trigger exception by passing None
    errors = strategy.validate(None)
    assert errors == []
