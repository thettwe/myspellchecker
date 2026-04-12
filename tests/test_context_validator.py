from unittest.mock import MagicMock, patch

from myspellchecker.core.config import SpellCheckerConfig
from myspellchecker.core.context_validator import ContextValidator
from myspellchecker.core.response import Error
from myspellchecker.core.validation_strategies import ValidationContext, ValidationStrategy


class MockStrategy(ValidationStrategy):
    def __init__(self, priority=10, fail=False):
        self._priority = priority
        self.fail = fail
        self.called_with = None

    def validate(self, context: ValidationContext):
        self.called_with = context
        if self.fail:
            raise RuntimeError("Strategy failed")
        return [
            Error(
                text="mock", position=context.word_positions[0], error_type="mock", suggestions=[]
            )
        ]

    def priority(self):
        return self._priority


def test_context_validator_sentence_not_found():
    """Test case where sentence is not found in original text (should be skipped)."""
    from myspellchecker.core.config.validation_configs import ValidationConfig

    # Use real config instead of MagicMock to ensure proper attribute access
    config = SpellCheckerConfig(
        use_ner=False,
        validation=ValidationConfig(raise_on_strategy_error=False),
    )
    segmenter = MagicMock()
    # Mock segmenter to return a sentence that text.find won't find
    # (though in reality find should usually work if segmenter is correct)
    segmenter.segment_sentences.return_value = ["ghost"]

    validator = ContextValidator(config, segmenter, strategies=[MockStrategy()])

    # "ghost" not in "real text"
    errors = validator.validate("real text")
    assert len(errors) == 0


def test_context_validator_strategy_failure():
    """Test that validator continues if one strategy fails."""
    from myspellchecker.core.config.validation_configs import ValidationConfig

    # Use real config instead of MagicMock to ensure proper attribute access
    config = SpellCheckerConfig(
        use_ner=False,
        validation=ValidationConfig(raise_on_strategy_error=False),
    )
    segmenter = MagicMock()
    segmenter.segment_sentences.return_value = ["မြန်မာစာ"]
    segmenter.segment_words.return_value = ["မြန်မာစာ"]

    s1 = MockStrategy(priority=10, fail=True)
    s2 = MockStrategy(priority=20, fail=False)

    validator = ContextValidator(config, segmenter, strategies=[s1, s2])

    # Should not raise exception
    errors = validator.validate("မြန်မာစာ")
    assert len(errors) == 1  # Only from s2
    assert s1.called_with is not None
    assert s2.called_with is not None


def test_context_validator_ner_masking():
    """Test NER name masking logic."""
    from myspellchecker.core.config.validation_configs import ValidationConfig

    # Use real config instead of MagicMock to ensure proper attribute access
    config = SpellCheckerConfig(
        use_ner=True,
        validation=ValidationConfig(raise_on_strategy_error=False),
    )
    segmenter = MagicMock()
    segmenter.segment_sentences.return_value = ["မောင်မောင် သွားသည်"]
    segmenter.segment_words.return_value = ["မောင်မောင်", " ", "သွားသည်"]

    name_heuristic = MagicMock()
    # "မောင်မောင်" is a name, " " is not, "သွားသည်" is not
    name_heuristic.analyze_sentence.return_value = [True, False, False]

    strategy = MockStrategy()
    validator = ContextValidator(
        config, segmenter, strategies=[strategy], name_heuristic=name_heuristic
    )

    validator.validate("မောင်မောင် သွားသည်")

    context = strategy.called_with
    assert context.words == ["မောင်မောင်", "သွားသည်"]
    assert context.is_name_mask == [True, False]


def test_context_validator_word_not_found():
    """Test case where a word is not found in the sentence (logging coverage)."""
    from myspellchecker.core.config.validation_configs import ValidationConfig

    # Use real config instead of MagicMock to ensure proper attribute access
    config = SpellCheckerConfig(
        use_ner=False,
        validation=ValidationConfig(raise_on_strategy_error=False),
    )
    segmenter = MagicMock()
    segmenter.segment_sentences.return_value = ["sent"]
    # Word "missing" is not in "sent"
    segmenter.segment_words.return_value = ["missing"]

    validator = ContextValidator(config, segmenter, strategies=[MockStrategy()])
    with patch("myspellchecker.core.context_validator.logger") as mock_logger:
        validator.validate("sent")
        mock_logger.debug.assert_any_call(
            "Word not found in sentence from cursor %d: %s", 0, "missing"
        )


def test_context_validator_repr():
    """Test __repr__."""
    from myspellchecker.core.config.validation_configs import ValidationConfig

    # Use real config instead of MagicMock to ensure proper attribute access
    config = SpellCheckerConfig(
        use_ner=False,
        validation=ValidationConfig(raise_on_strategy_error=False),
    )
    segmenter = MagicMock()
    validator = ContextValidator(config, segmenter, strategies=[MockStrategy(10), MockStrategy(20)])

    r = repr(validator)
    assert "ContextValidator" in r
    assert "priorities=[10, 20]" in r


def test_context_validator_raise_on_strategy_error():
    """Test that raise_on_strategy_error=True causes exceptions to propagate."""
    import pytest

    from myspellchecker.core.config.validation_configs import ValidationConfig

    # Create real config with raise_on_strategy_error=True
    config = SpellCheckerConfig(validation=ValidationConfig(raise_on_strategy_error=True))

    segmenter = MagicMock()
    segmenter.segment_sentences.return_value = ["မြန်မာစာ"]
    segmenter.segment_words.return_value = ["မြန်မာစာ"]

    failing_strategy = MockStrategy(priority=10, fail=True)
    validator = ContextValidator(config, segmenter, strategies=[failing_strategy])

    # Should raise the exception when raise_on_strategy_error=True
    with pytest.raises(RuntimeError, match="Strategy failed"):
        validator.validate("မြန်မာစာ")


def test_context_validator_raise_on_strategy_error_default_false():
    """Test that raise_on_strategy_error defaults to False (graceful degradation)."""
    from myspellchecker.core.config.validation_configs import ValidationConfig

    # Create real config with default settings (raise_on_strategy_error=False)
    config = SpellCheckerConfig(validation=ValidationConfig())

    segmenter = MagicMock()
    segmenter.segment_sentences.return_value = ["မြန်မာစာ"]
    segmenter.segment_words.return_value = ["မြန်မာစာ"]

    failing_strategy = MockStrategy(priority=10, fail=True)
    ok_strategy = MockStrategy(priority=20, fail=False)
    validator = ContextValidator(config, segmenter, strategies=[failing_strategy, ok_strategy])

    # Should NOT raise exception - graceful degradation
    errors = validator.validate("မြန်မာစာ")
    # The ok_strategy should still produce its error
    assert len(errors) == 1
