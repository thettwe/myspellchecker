from unittest.mock import patch

import pytest

from myspellchecker.core.validation_strategies.base import ValidationContext
from myspellchecker.core.validation_strategies.question_strategy import (
    QuestionStructureValidationStrategy,
)


def test_question_strategy_repr():
    """Test __repr__."""
    strategy = QuestionStructureValidationStrategy(confidence=0.8)
    assert "priority=40" in repr(strategy)
    assert "confidence=0.8" in repr(strategy)


def test_question_strategy_validate_success():
    """Test detection of question missing particle."""
    # Mock helpers
    with (
        patch(
            "myspellchecker.core.validation_strategies.question_strategy.detect_sentence_type",
            return_value="question",
        ),
        patch(
            "myspellchecker.core.validation_strategies.question_strategy.has_question_particle_context",
            return_value=False,
        ),
        patch(
            "myspellchecker.core.validation_strategies.question_strategy.has_question_word_context",
            return_value=True,
        ),
    ):
        strategy = QuestionStructureValidationStrategy()
        context = ValidationContext(
            sentence="ဘယ် သွား တယ်",
            words=["ဘယ်", "သွား", "တယ်"],
            word_positions=[0, 4, 10],
            is_name_mask=[False, False, False],
        )

        errors = strategy.validate(context)
        assert len(errors) == 1
        assert errors[0].text == "တယ်"
        assert errors[0].error_type == "question_structure"
        assert 10 in context.existing_errors


def test_question_strategy_no_question():
    """Test when sentence is not a question."""
    with patch(
        "myspellchecker.core.validation_strategies.question_strategy.detect_sentence_type",
        return_value="statement",
    ):
        strategy = QuestionStructureValidationStrategy()
        context = ValidationContext("test", ["w1", "w2"], [0, 5], [False, False])
        assert strategy.validate(context) == []


def test_question_strategy_has_particle():
    """Test when question already has particle."""
    with (
        patch(
            "myspellchecker.core.validation_strategies.question_strategy.detect_sentence_type",
            return_value="question",
        ),
        patch(
            "myspellchecker.core.validation_strategies.question_strategy.has_question_particle_context",
            return_value=True,
        ),
    ):
        strategy = QuestionStructureValidationStrategy()
        context = ValidationContext(
            sentence="ဘယ် သွား လား",
            words=["ဘယ်", "သွား", "လား"],
            word_positions=[0, 4, 10],
            is_name_mask=[False, False, False],
        )

        assert strategy.validate(context) == []


def test_question_strategy_exception():
    """Test exception handling."""
    with patch(
        "myspellchecker.core.validation_strategies.question_strategy.detect_sentence_type",
        side_effect=RuntimeError("boom"),
    ):
        strategy = QuestionStructureValidationStrategy()
        context = ValidationContext("test", ["w1", "w2"], [0, 5], [False, False])
        assert strategy.validate(context) == []


@pytest.mark.parametrize(
    ("words", "positions", "expected_top"),
    [
        (["ဒါရိုက်တာ", "ဘယ်အချိန်", "ရောက်", "မယ်"], [0, 10, 20, 26], "မလဲ"),
        (["ဘယ်သူ", "ဒီအလုပ်ကို", "တာဝန်ယူ", "သည်"], [0, 6, 15, 23], "သလဲ"),
        (["နင်", "မနက်ဖြန်", "လာနိုင်", "တယ်"], [0, 3, 10, 17], "မလား"),
        (["ခင်ဗျား", "အဆင်ပြေရဲ့လဲ"], [0, 8], "ရဲ့လား"),
        (["ခင်ဗျား", "အဆင်ပြေ", "ရဲ့", "လဲ"], [0, 8, 13, 17], "ရဲ့လား"),
        (["မင်း", "ဘယ်သွားရဲ့လဲ"], [0, 4], "လဲ"),
        (["ဒါကို", "ဘာလုပ်ရဲ့လဲ"], [0, 4], "လဲ"),
        (["ခင်ဗျား", "ဘယ်သွား"], [0, 8], "ဘယ်သွားလဲ"),
        (["ဒါ", "ဘာလုပ်"], [0, 2], "ဘာလုပ်လဲ"),
    ],
)
def test_question_strategy_regression_cases(words, positions, expected_top):
    """Top suggestion should match BM regression gold correction forms."""
    strategy = QuestionStructureValidationStrategy()
    context = ValidationContext(
        sentence=" ".join(words),
        words=words,
        word_positions=positions,
        is_name_mask=[False] * len(words),
    )

    errors = strategy.validate(context)
    assert errors
    assert errors[0].suggestions
    assert errors[0].suggestions[0] == expected_top


def test_question_strategy_accepts_split_question_particle_before_request_particle():
    """Question particles before polite request tails should not trigger fallback errors."""
    strategy = QuestionStructureValidationStrategy()
    words = ["ဒါ", "က", "ဘာ", "လဲ", "ပြော", "ပါ", "ဦး"]
    context = ValidationContext(
        sentence="ဒါက ဘာလဲ ပြောပါဦး",
        words=words,
        word_positions=[0, 2, 4, 7, 10, 14, 15],
        is_name_mask=[False] * len(words),
    )

    assert strategy.validate(context) == []
