"""Tests for HomophoneValidationStrategy.

The strategy now delegates n-gram comparison to
NgramContextChecker.check_word_in_context(), so these tests mock that
method to verify the strategy's wiring and error creation logic.
"""

from unittest.mock import MagicMock

from myspellchecker.algorithms.ngram_context_checker import NgramVerdict
from myspellchecker.core.validation_strategies.base import ValidationContext
from myspellchecker.core.validation_strategies.homophone_strategy import HomophoneValidationStrategy


def _make_verdict(
    is_error: bool = False,
    best_alt: str | None = None,
    probability: float = 0.0,
) -> NgramVerdict:
    """Build a mock NgramVerdict."""
    return NgramVerdict(
        is_error=is_error,
        best_alternative=best_alt,
        confidence=0.8,
        error_type="confusable_error" if best_alt else "context_probability",
        probability=probability,
        suggestions=[best_alt] if best_alt else [],
    )


def test_homophone_strategy_disabled():
    """Test when homophone checker is None."""
    strategy = HomophoneValidationStrategy(None, MagicMock())
    context = ValidationContext("test", ["w1", "w2"], [0, 5], [False, False])
    assert strategy.validate(context) == []


def test_homophone_strategy_disabled_no_context_checker():
    """Test when context_checker is None."""
    checker = MagicMock()
    strategy = HomophoneValidationStrategy(checker, MagicMock(), context_checker=None)
    context = ValidationContext("test", ["w1", "w2"], [0, 5], [False, False])
    assert strategy.validate(context) == []


def test_homophone_strategy_repr():
    """Test __repr__."""
    checker = MagicMock()
    strategy = HomophoneValidationStrategy(checker, MagicMock())
    r = repr(strategy)
    assert "HomophoneValidationStrategy" in r
    assert "enabled" in r

    strategy_disabled = HomophoneValidationStrategy(None, MagicMock())
    assert "disabled" in repr(strategy_disabled)


def test_homophone_strategy_validate_full():
    """Test the full validate loop with a mocked context_checker."""
    checker = MagicMock()
    checker.get_homophones.side_effect = lambda w: {"စား"} if w == "သား" else set()

    provider = MagicMock()
    provider.get_word_frequency.return_value = 100

    context_checker = MagicMock()
    context_checker.check_word_in_context.return_value = _make_verdict(
        is_error=True, best_alt="စား", probability=0.5
    )

    strategy = HomophoneValidationStrategy(checker, provider, context_checker=context_checker)
    context = ValidationContext(
        sentence="သူ သား တယ်",
        words=["သူ", "သား", "တယ်"],
        word_positions=[0, 4, 10],
        is_name_mask=[False, False, False],
    )

    errors = strategy.validate(context)
    assert len(errors) == 1
    assert errors[0].text == "သား"
    assert errors[0].suggestions == ["စား"]
    assert 4 in context.existing_errors

    # Verify check_word_in_context was called with correct args
    call_kwargs = context_checker.check_word_in_context.call_args
    assert call_kwargs.kwargs["word"] == "သား"
    assert call_kwargs.kwargs["prev_words"] == ["သူ"]
    assert call_kwargs.kwargs["next_words"] == ["တယ်"]
    assert call_kwargs.kwargs["candidates"] == [("စား", "homophone")]


def test_homophone_no_error_when_verdict_is_not_error():
    """When check_word_in_context returns is_error=False, no errors."""
    checker = MagicMock()
    checker.get_homophones.return_value = {"စား"}

    provider = MagicMock()
    provider.get_word_frequency.return_value = 100

    context_checker = MagicMock()
    context_checker.check_word_in_context.return_value = _make_verdict(is_error=False)

    strategy = HomophoneValidationStrategy(checker, provider, context_checker=context_checker)
    context = ValidationContext(
        sentence="သူ သား တယ်",
        words=["သူ", "သား", "တယ်"],
        word_positions=[0, 4, 10],
        is_name_mask=[False, False, False],
    )

    errors = strategy.validate(context)
    assert len(errors) == 0


def test_homophone_strategy_exception_handling():
    """Test exception handling in validate."""
    checker = MagicMock()
    checker.get_homophones.side_effect = RuntimeError("error")

    context_checker = MagicMock()
    strategy = HomophoneValidationStrategy(checker, MagicMock(), context_checker=context_checker)
    context = ValidationContext("test", ["w1", "w2"], [0, 5], [False, False])

    # Should catch exception and return empty list
    errors = strategy.validate(context)
    assert errors == []


def test_homophone_appends_suggestion_to_existing_error():
    """Test that homophone strategy appends suggestions to already-flagged positions."""
    checker = MagicMock()
    checker.get_homophones.side_effect = lambda w: {"စား"} if w == "သား" else set()

    provider = MagicMock()
    provider.get_word_frequency.return_value = 100

    context_checker = MagicMock()
    context_checker.check_word_in_context.return_value = _make_verdict(
        is_error=True, best_alt="စား", probability=0.5
    )

    strategy = HomophoneValidationStrategy(checker, provider, context_checker=context_checker)
    context = ValidationContext(
        sentence="သူ သား တယ်",
        words=["သူ", "သား", "တယ်"],
        word_positions=[0, 4, 10],
        is_name_mask=[False, False, False],
    )
    # Position 4 already flagged by a higher-priority strategy
    context.existing_errors[4] = "tone_ambiguity"
    context.existing_suggestions[4] = ["သာ"]

    errors = strategy.validate(context)
    # Should NOT create a new error (position already flagged)
    assert len(errors) == 0
    # BUT should append homophone suggestion to existing suggestions
    assert "စား" in context.existing_suggestions[4]
    # Original suggestion preserved
    assert "သာ" in context.existing_suggestions[4]


def test_homophone_no_duplicate_append():
    """Test that homophone strategy doesn't append duplicate suggestions."""
    checker = MagicMock()
    checker.get_homophones.side_effect = lambda w: {"စား"} if w == "သား" else set()

    provider = MagicMock()
    provider.get_word_frequency.return_value = 100

    context_checker = MagicMock()
    context_checker.check_word_in_context.return_value = _make_verdict(
        is_error=True, best_alt="စား", probability=0.5
    )

    strategy = HomophoneValidationStrategy(checker, provider, context_checker=context_checker)
    context = ValidationContext(
        sentence="သူ သား တယ်",
        words=["သူ", "သား", "တယ်"],
        word_positions=[0, 4, 10],
        is_name_mask=[False, False, False],
    )
    # Position 4 already has the SAME suggestion
    context.existing_errors[4] = "tone_ambiguity"
    context.existing_suggestions[4] = ["စား"]

    errors = strategy.validate(context)
    assert len(errors) == 0
    # Should NOT duplicate the suggestion
    assert context.existing_suggestions[4].count("စား") == 1


def test_homophone_skips_names():
    """Test that words marked as names are skipped."""
    checker = MagicMock()
    checker.get_homophones.return_value = {"alt"}

    context_checker = MagicMock()
    context_checker.check_word_in_context.return_value = _make_verdict(
        is_error=True, best_alt="alt"
    )

    strategy = HomophoneValidationStrategy(checker, MagicMock(), context_checker=context_checker)
    context = ValidationContext(
        sentence="name word",
        words=["name", "word"],
        word_positions=[0, 5],
        is_name_mask=[True, True],
    )
    errors = strategy.validate(context)
    assert errors == []


def test_homophone_passes_prev_prev_context():
    """Test that prev_prev_word is included in prev_words."""
    checker = MagicMock()
    checker.get_homophones.side_effect = lambda w: {"alt"} if w == "w2" else set()

    provider = MagicMock()
    provider.get_word_frequency.return_value = 100

    context_checker = MagicMock()
    context_checker.check_word_in_context.return_value = _make_verdict(
        is_error=True, best_alt="alt"
    )

    strategy = HomophoneValidationStrategy(checker, provider, context_checker=context_checker)
    context = ValidationContext(
        sentence="w0 w1 w2 w3",
        words=["w0", "w1", "w2", "w3"],
        word_positions=[0, 3, 6, 9],
        is_name_mask=[False, False, False, False],
    )
    strategy.validate(context)

    # Find the call for w2 (index 2)
    for call in context_checker.check_word_in_context.call_args_list:
        if call.kwargs["word"] == "w2":
            assert call.kwargs["prev_words"] == ["w0", "w1"]
            assert call.kwargs["next_words"] == ["w3"]
            break
    else:
        raise AssertionError("check_word_in_context not called for w2")  # noqa: TRY003


def test_config_tunables_passthrough():
    """Verify the surviving homophone tunable is correctly defined on the config."""
    from myspellchecker.core.config.validation_configs import ValidationConfig

    config = ValidationConfig(homophone_confidence=0.75)
    assert config.homophone_confidence == 0.75


def test_legacy_kwargs_accepted():
    """Legacy kwargs like improvement_ratio are silently accepted."""
    checker = MagicMock()
    # Should not raise
    strategy = HomophoneValidationStrategy(
        checker,
        MagicMock(),
        confidence=0.85,
        improvement_ratio=15.0,
        min_probability=0.001,
        high_freq_threshold=1000,
        high_freq_improvement_ratio=50.0,
    )
    assert strategy.confidence == 0.85
