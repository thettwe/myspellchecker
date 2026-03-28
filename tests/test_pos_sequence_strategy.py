from unittest.mock import MagicMock, patch

from myspellchecker.core.validation_strategies.base import ValidationContext
from myspellchecker.core.validation_strategies.pos_sequence_strategy import (
    POSSequenceValidationStrategy,
)


def test_pos_sequence_strategy_disabled():
    """Test when disabled."""
    strategy = POSSequenceValidationStrategy(None)
    assert strategy.validate(MagicMock()) == []
    assert "disabled" in repr(strategy)


def test_pos_sequence_strategy_repr():
    """Test __repr__."""
    checker = MagicMock()
    strategy = POSSequenceValidationStrategy(checker)
    assert "enabled" in repr(strategy)


def test_pos_sequence_strategy_validate_success():
    """Test successful detection of invalid POS sequence."""
    tagger = MagicMock()
    # "V", "V" is an error in INVALID_POS_SEQUENCES
    tagger.tag_sequence.return_value = ["V", "V"]

    with patch(
        "myspellchecker.core.validation_strategies.pos_sequence_strategy.INVALID_POS_SEQUENCES",
        {("V", "V"): ("error", "Consecutive verbs")},
    ):
        strategy = POSSequenceValidationStrategy(tagger)
        context = ValidationContext(
            sentence="word1 word2",
            words=["word1", "word2"],
            word_positions=[0, 6],
            is_name_mask=[False, False],
        )

        errors = strategy.validate(context)
        assert len(errors) == 1
        assert errors[0].error_type == "pos_sequence_error"
        assert 6 in context.existing_errors
        assert context.pos_tags == ["V", "V"]


def test_pos_sequence_strategy_skipping():
    """Test skipping logic."""
    tagger = MagicMock()
    # V V V V -> all are invalid sequences
    tagger.tag_sequence.return_value = ["V", "V", "V", "V"]

    with patch(
        "myspellchecker.core.validation_strategies.pos_sequence_strategy.INVALID_POS_SEQUENCES",
        {("V", "V"): ("error", "err")},
    ):
        strategy = POSSequenceValidationStrategy(tagger)
        context = ValidationContext(
            sentence="W1 W2 W3 W4",
            words=["W1", "W2", "W3", "W4"],
            word_positions=[0, 3, 6, 9],
            is_name_mask=[False, True, False, False],  # W2 is name
        )
        context.existing_errors[6] = "test"  # W3 already has error

        errors = strategy.validate(context)
        # Sequence checks:
        # (0,1): W1-W2. W2 is name -> skip
        # (1,2): W2-W3. W3 has existing error -> skip
        # (2,3): W3-W4. W4 is valid error
        assert len(errors) == 1
        assert errors[0].position == 9


def test_pos_sequence_strategy_mismatch():
    """Test length mismatch handling."""
    tagger = MagicMock()
    tagger.tag_sequence.return_value = ["V"]  # Only 1 tag for 2 words

    strategy = POSSequenceValidationStrategy(tagger)
    # Mock the logger instance directly
    strategy.logger = MagicMock()

    context = ValidationContext("w1 w2", ["w1", "w2"], [0, 3], [False, False])

    errors = strategy.validate(context)
    assert errors == []
    strategy.logger.warning.assert_called()


def test_pos_sequence_strategy_exception():
    """Test exception handling."""
    tagger = MagicMock()
    tagger.tag_sequence.side_effect = RuntimeError("Viterbi failed")
    strategy = POSSequenceValidationStrategy(tagger)
    context = ValidationContext("test", ["test", "test"], [0, 5], [False, False])

    errors = strategy.validate(context)
    assert errors == []


def test_pos_sequence_pp_produces_error():
    """P-P consecutive particles should produce an error (severity='error')."""
    tagger = MagicMock()
    tagger.tag_sequence.return_value = ["P", "P"]

    strategy = POSSequenceValidationStrategy(tagger)
    context = ValidationContext(
        sentence="ကို မှာ",
        words=["ကို", "မှာ"],
        word_positions=[0, 9],
        is_name_mask=[False, False],
    )

    errors = strategy.validate(context)
    assert len(errors) == 1
    assert errors[0].error_type == "pos_sequence_error"
    assert errors[0].position == 9


def test_pos_sequence_vv_no_error():
    """V-V consecutive verbs should NOT produce an error (severity='info')."""
    tagger = MagicMock()
    tagger.tag_sequence.return_value = ["V", "V"]

    strategy = POSSequenceValidationStrategy(tagger)
    context = ValidationContext(
        sentence="သွား လာ",
        words=["သွား", "လာ"],
        word_positions=[0, 9],
        is_name_mask=[False, False],
    )

    errors = strategy.validate(context)
    assert len(errors) == 0
