"""Tests for NgramContextValidationStrategy.

The strategy now delegates detection to
NgramContextChecker.check_word_in_context(), so these tests mock that
method to verify the strategy's wiring and error creation logic.
"""

from unittest.mock import MagicMock

import pytest

from myspellchecker.algorithms.ngram_context_checker import NgramVerdict
from myspellchecker.core.validation_strategies.base import ValidationContext
from myspellchecker.core.validation_strategies.ngram_strategy import NgramContextValidationStrategy


def _ok_verdict(prob: float = 0.0) -> NgramVerdict:
    """Build a verdict for a non-error."""
    return NgramVerdict(is_error=False, probability=prob)


def _error_verdict(prob: float = 0.01) -> NgramVerdict:
    """Build a verdict for an error."""
    return NgramVerdict(
        is_error=True,
        confidence=0.9 if prob == 0.0 else 0.6,
        error_type="context_probability",
        probability=prob,
    )


def test_ngram_strategy_disabled():
    """Test when context checker is None."""
    strategy = NgramContextValidationStrategy(None, MagicMock())
    context = ValidationContext("test", ["w1", "w2"], [0, 5], [False, False])
    assert strategy.validate(context) == []


def test_ngram_strategy_repr():
    """Test __repr__."""
    checker = MagicMock()
    strategy = NgramContextValidationStrategy(checker, MagicMock())
    r = repr(strategy)
    assert "NgramContextValidationStrategy" in r
    assert "enabled" in r

    strategy_disabled = NgramContextValidationStrategy(None, MagicMock())
    assert "disabled" in repr(strategy_disabled)


def test_ngram_strategy_validate_full():
    """Test the full validate loop with an error."""
    checker = MagicMock()

    # check_word_in_context: flag "သား" as error, others OK
    def _check(word, prev_words, next_words, candidates=None, word_freq=0):
        if word == "သား":
            return _error_verdict(prob=0.01)
        return _ok_verdict()

    checker.check_word_in_context.side_effect = _check

    # Mock suggestions
    sug = MagicMock()
    sug.term = "စား"
    checker.suggest.return_value = [sug]

    provider = MagicMock()
    provider.get_trigram_probability.return_value = 0.001
    provider.get_bigram_probability.return_value = 0.01
    provider.get_word_frequency.return_value = 100

    strategy = NgramContextValidationStrategy(checker, provider)
    # Context: "သူ" (0), "သား" (4), "တယ်" (10)
    context = ValidationContext("သူ သား တယ်", ["သူ", "သား", "တယ်"], [0, 4, 10], [False, False, False])

    errors = strategy.validate(context)
    assert len(errors) == 1
    assert errors[0].text == "သား"
    assert errors[0].suggestions == ["စား"]
    assert 4 in context.existing_errors


def test_ngram_strategy_skipped_words():
    """Test that certain words are skipped for context validation."""
    from myspellchecker.core.constants import SKIPPED_CONTEXT_WORDS

    if not SKIPPED_CONTEXT_WORDS:
        pytest.skip("SKIPPED_CONTEXT_WORDS is empty")

    skipped_word = list(SKIPPED_CONTEXT_WORDS)[0]

    checker = MagicMock()
    checker.check_word_in_context.return_value = _error_verdict()

    strategy = NgramContextValidationStrategy(checker, MagicMock())
    context = ValidationContext(f"သူ {skipped_word}", ["သူ", skipped_word], [0, 4], [False, False])

    errors = strategy.validate(context)
    # Should be empty because skipped_word is in SKIPPED_CONTEXT_WORDS
    assert len(errors) == 0


def test_ngram_strategy_generate_suggestions_exception():
    """Test exception handling in suggestion generation."""
    checker = MagicMock()
    # Use RuntimeError which is one of the caught exception types
    checker.suggest.side_effect = RuntimeError("boom")
    strategy = NgramContextValidationStrategy(checker, MagicMock())

    sugs = strategy._generate_suggestions("w1", "w2", "w3")
    assert sugs == []


def test_ngram_strategy_validate_exception():
    """Test exception handling in main validate loop."""
    checker = MagicMock()
    checker.check_word_in_context.side_effect = RuntimeError("fail")
    strategy = NgramContextValidationStrategy(checker, MagicMock())
    context = ValidationContext("test", ["w1", "w2"], [0, 5], [False, False])

    # Should catch and return empty
    errors = strategy.validate(context)
    assert errors == []


def test_ngram_appends_suggestions_to_existing_error():
    """Test that ngram strategy appends suggestions to already-flagged positions."""
    checker = MagicMock()
    # Suggestion generation for the already-flagged word
    sug = MagicMock()
    sug.term = "စား"
    checker.suggest.return_value = [sug]

    strategy = NgramContextValidationStrategy(checker, MagicMock())
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
    # Should NOT create a new error
    assert len(errors) == 0
    # BUT should append n-gram suggestion
    assert "စား" in context.existing_suggestions[4]
    assert "သာ" in context.existing_suggestions[4]


def test_ngram_uses_original_when_preceding_word_has_error_no_suggestion():
    """Test that ngram uses original word when preceding word has error but no suggestion."""
    checker = MagicMock()
    sug = MagicMock()
    sug.term = "စား"
    checker.suggest.return_value = [sug]

    strategy = NgramContextValidationStrategy(checker, MagicMock())
    context = ValidationContext(
        sentence="သူ သား တယ်",
        words=["သူ", "သား", "တယ်"],
        word_positions=[0, 4, 10],
        is_name_mask=[False, False, False],
    )
    # Preceding word has error but NO suggestion — uses original word for context
    context.existing_errors[0] = "invalid_syllable"
    context.existing_errors[4] = "tone_ambiguity"
    context.existing_suggestions[4] = ["သာ"]

    errors = strategy.validate(context)
    assert len(errors) == 0
    # Should append n-gram suggestion since word[1] (pos 4) is still checked
    assert "သာ" in context.existing_suggestions[4]
    assert "စား" in context.existing_suggestions[4]


def test_ngram_uses_corrected_form_for_preceding_error():
    """Test that ngram uses corrected form when preceding word has error with suggestion."""
    checker = MagicMock()

    def _check(word, prev_words, next_words, candidates=None, word_freq=0):
        # Flag "သား" only when preceded by corrected "သူ"
        if word == "သား" and prev_words and prev_words[-1] == "သူ":
            return _error_verdict(prob=0.01)
        return _ok_verdict()

    checker.check_word_in_context.side_effect = _check

    sug = MagicMock()
    sug.term = "စား"
    checker.suggest.return_value = [sug]

    provider = MagicMock()
    provider.get_trigram_probability.return_value = 0.0
    provider.get_bigram_probability.return_value = 0.01
    provider.get_word_frequency.return_value = 100

    strategy = NgramContextValidationStrategy(checker, provider)
    context = ValidationContext(
        sentence="သူူ သား တယ်",
        words=["သူူ", "သား", "တယ်"],
        word_positions=[0, 6, 12],
        is_name_mask=[False, False, False],
    )
    # Preceding word "သူူ" has error with correction "သူ"
    context.existing_errors[0] = "invalid_syllable"
    context.existing_suggestions[0] = ["သူ"]

    errors = strategy.validate(context)
    # Should detect error using corrected "သူ" as preceding word
    assert len(errors) == 1
    assert errors[0].text == "သား"

    # Verify check_word_in_context was called with corrected form in prev_words
    for call in checker.check_word_in_context.call_args_list:
        if call.kwargs["word"] == "သား":
            assert "သူ" in call.kwargs["prev_words"]
            break
    else:
        raise AssertionError("check_word_in_context not called for 'သား'")  # noqa: TRY003
