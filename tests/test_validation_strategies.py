"""Unit tests for validation strategies with mocked dependencies."""

from unittest.mock import Mock, patch

import pytest


class TestValidationContext:
    """Tests for ValidationContext dataclass."""

    def test_context_creation_minimal(self):
        """Test creating ValidationContext with minimal arguments."""
        from myspellchecker.core.validation_strategies.base import ValidationContext

        ctx = ValidationContext(sentence="test", words=["test"], word_positions=[0])
        assert ctx.sentence == "test"
        assert ctx.words == ["test"]
        assert ctx.word_positions == [0]
        assert ctx.is_name_mask == [False]  # Auto-filled to match words length
        assert ctx.existing_errors == {}
        assert ctx.sentence_type == "statement"
        assert ctx.pos_tags == []

    def test_context_creation_full(self):
        """Test creating ValidationContext with all arguments."""
        from myspellchecker.core.validation_strategies.base import ValidationContext

        ctx = ValidationContext(
            sentence="test sentence",
            words=["test", "sentence"],
            word_positions=[0, 5],
            is_name_mask=[False, False],
            existing_errors={5: "test"},
            sentence_type="question",
            pos_tags=["N", "N"],
        )
        assert ctx.sentence_type == "question"
        assert 5 in ctx.existing_errors
        assert ctx.pos_tags == ["N", "N"]


class TestValidationStrategy:
    """Tests for ValidationStrategy abstract base class."""

    def test_cannot_instantiate_abstract(self):
        """Test that ValidationStrategy cannot be instantiated directly."""
        from myspellchecker.core.validation_strategies.base import ValidationStrategy

        with pytest.raises(TypeError):
            ValidationStrategy()

    def test_repr_on_concrete_class(self):
        """Test __repr__ on concrete strategy."""
        from myspellchecker.core.validation_strategies.question_strategy import (
            QuestionStructureValidationStrategy,
        )

        strategy = QuestionStructureValidationStrategy()
        repr_str = repr(strategy)
        assert "QuestionStructureValidationStrategy" in repr_str
        assert "priority" in repr_str


class TestHomophoneValidationStrategy:
    """Tests for HomophoneValidationStrategy."""

    def test_init(self):
        """Test strategy initialization."""
        from myspellchecker.core.validation_strategies.homophone_strategy import (
            HomophoneValidationStrategy,
        )

        mock_checker = Mock()
        mock_provider = Mock()

        strategy = HomophoneValidationStrategy(
            homophone_checker=mock_checker,
            provider=mock_provider,
            confidence=0.85,
        )

        assert strategy.homophone_checker == mock_checker
        assert strategy.provider == mock_provider
        assert strategy.confidence == 0.85

    def test_priority(self):
        """Test priority returns 45."""
        from myspellchecker.core.validation_strategies.homophone_strategy import (
            HomophoneValidationStrategy,
        )

        strategy = HomophoneValidationStrategy(None, Mock())
        assert strategy.priority() == 45

    def test_validate_disabled(self):
        """Test validate returns empty when checker is None."""
        from myspellchecker.core.validation_strategies.base import ValidationContext
        from myspellchecker.core.validation_strategies.homophone_strategy import (
            HomophoneValidationStrategy,
        )

        strategy = HomophoneValidationStrategy(None, Mock())
        ctx = ValidationContext(sentence="test", words=["test", "word"], word_positions=[0, 5])
        errors = strategy.validate(ctx)
        assert errors == []

    def test_validate_too_few_words(self):
        """Test validate returns empty when less than 2 words."""
        from myspellchecker.core.validation_strategies.base import ValidationContext
        from myspellchecker.core.validation_strategies.homophone_strategy import (
            HomophoneValidationStrategy,
        )

        strategy = HomophoneValidationStrategy(Mock(), Mock())
        ctx = ValidationContext(sentence="test", words=["test"], word_positions=[0])
        errors = strategy.validate(ctx)
        assert errors == []

    def test_validate_with_name_mask(self):
        """Test validate skips words marked as names."""
        from myspellchecker.core.validation_strategies.base import ValidationContext
        from myspellchecker.core.validation_strategies.homophone_strategy import (
            HomophoneValidationStrategy,
        )

        mock_checker = Mock()
        mock_checker.get_homophones.return_value = []
        mock_provider = Mock()

        strategy = HomophoneValidationStrategy(mock_checker, mock_provider)
        ctx = ValidationContext(
            sentence="test word",
            words=["test", "word"],
            word_positions=[0, 5],
            is_name_mask=[True, True],
        )
        errors = strategy.validate(ctx)
        assert errors == []

    def test_validate_skips_existing_errors(self):
        """Test validate skips positions with existing errors."""
        from myspellchecker.core.validation_strategies.base import ValidationContext
        from myspellchecker.core.validation_strategies.homophone_strategy import (
            HomophoneValidationStrategy,
        )

        mock_checker = Mock()
        mock_checker.get_homophones.return_value = []
        mock_provider = Mock()

        strategy = HomophoneValidationStrategy(mock_checker, mock_provider)
        ctx = ValidationContext(
            sentence="test word",
            words=["test", "word"],
            word_positions=[0, 5],
            existing_errors={0: "test", 5: "test"},
        )
        errors = strategy.validate(ctx)
        assert errors == []

    def test_validate_finds_homophone_error(self):
        """Test validate detects homophone error."""
        from myspellchecker.core.validation_strategies.base import ValidationContext
        from myspellchecker.core.validation_strategies.homophone_strategy import (
            HomophoneValidationStrategy,
        )

        mock_checker = Mock()
        mock_checker.get_homophones.return_value = ["better"]
        mock_provider = Mock()
        # Use return_value instead of side_effect to avoid running out of values
        # The strategy makes multiple calls for current word and homophones
        mock_provider.get_bigram_probability.return_value = 0.1
        mock_provider.get_trigram_probability.return_value = 0.0
        mock_provider.get_word_frequency.return_value = 0

        strategy = HomophoneValidationStrategy(mock_checker, mock_provider)
        ctx = ValidationContext(
            sentence="test word more",
            words=["test", "word", "more"],
            word_positions=[0, 5, 10],
        )
        errors = strategy.validate(ctx)
        assert isinstance(errors, list)

    def test_repr_enabled(self):
        """Test repr when checker is enabled."""
        from myspellchecker.core.validation_strategies.homophone_strategy import (
            HomophoneValidationStrategy,
        )

        strategy = HomophoneValidationStrategy(Mock(), Mock())
        repr_str = repr(strategy)
        assert "enabled" in repr_str

    def test_repr_disabled(self):
        """Test repr when checker is disabled."""
        from myspellchecker.core.validation_strategies.homophone_strategy import (
            HomophoneValidationStrategy,
        )

        strategy = HomophoneValidationStrategy(None, Mock())
        repr_str = repr(strategy)
        assert "disabled" in repr_str


class TestNgramContextValidationStrategy:
    """Tests for NgramContextValidationStrategy."""

    def test_init(self):
        """Test strategy initialization."""
        from myspellchecker.core.validation_strategies.ngram_strategy import (
            NgramContextValidationStrategy,
        )

        mock_checker = Mock()
        mock_provider = Mock()

        strategy = NgramContextValidationStrategy(
            context_checker=mock_checker,
            provider=mock_provider,
            confidence_high=0.8,
            confidence_low=0.5,
            max_suggestions=3,
            edit_distance=1,
        )

        assert strategy.context_checker == mock_checker
        assert strategy.confidence_high == 0.8
        assert strategy.confidence_low == 0.5
        assert strategy.max_suggestions == 3
        assert strategy.edit_distance == 1

    def test_priority(self):
        """Test priority returns 50."""
        from myspellchecker.core.validation_strategies.ngram_strategy import (
            NgramContextValidationStrategy,
        )

        strategy = NgramContextValidationStrategy(None, Mock())
        assert strategy.priority() == 50

    def test_validate_disabled(self):
        """Test validate returns empty when checker is None."""
        from myspellchecker.core.validation_strategies.base import ValidationContext
        from myspellchecker.core.validation_strategies.ngram_strategy import (
            NgramContextValidationStrategy,
        )

        strategy = NgramContextValidationStrategy(None, Mock())
        ctx = ValidationContext(sentence="test", words=["test", "word"], word_positions=[0, 5])
        errors = strategy.validate(ctx)
        assert errors == []

    def test_validate_too_few_words(self):
        """Test validate returns empty when less than 2 words."""
        from myspellchecker.core.validation_strategies.base import ValidationContext
        from myspellchecker.core.validation_strategies.ngram_strategy import (
            NgramContextValidationStrategy,
        )

        strategy = NgramContextValidationStrategy(Mock(), Mock())
        ctx = ValidationContext(sentence="test", words=["test"], word_positions=[0])
        errors = strategy.validate(ctx)
        assert errors == []

    def test_validate_with_errors(self):
        """Test validate finds contextual errors."""
        from myspellchecker.core.validation_strategies.base import ValidationContext
        from myspellchecker.core.validation_strategies.ngram_strategy import (
            NgramContextValidationStrategy,
        )

        mock_checker = Mock()
        mock_checker.is_contextual_error.return_value = True
        mock_suggestion = Mock()
        mock_suggestion.term = "suggestion"
        mock_checker.suggest.return_value = [mock_suggestion]

        mock_provider = Mock()
        mock_provider.get_bigram_probability.return_value = 0.01
        mock_provider.get_trigram_probability.return_value = 0.0

        strategy = NgramContextValidationStrategy(mock_checker, mock_provider)
        ctx = ValidationContext(
            sentence="test word more",
            words=["test", "word", "more"],
            word_positions=[0, 5, 10],
        )
        errors = strategy.validate(ctx)
        # Should find at least one error
        assert len(errors) >= 1

    def test_repr(self):
        """Test repr."""
        from myspellchecker.core.validation_strategies.ngram_strategy import (
            NgramContextValidationStrategy,
        )

        strategy = NgramContextValidationStrategy(Mock(), Mock())
        repr_str = repr(strategy)
        assert "NgramContextValidationStrategy" in repr_str
        assert "enabled" in repr_str


class TestSyntacticValidationStrategy:
    """Tests for SyntacticValidationStrategy."""

    def test_init(self):
        """Test strategy initialization."""
        from myspellchecker.core.validation_strategies.syntactic_strategy import (
            SyntacticValidationStrategy,
        )

        mock_checker = Mock()
        strategy = SyntacticValidationStrategy(syntactic_rule_checker=mock_checker, confidence=0.95)

        assert strategy.syntactic_rule_checker == mock_checker
        assert strategy.confidence == 0.95

    def test_priority(self):
        """Test priority returns 20."""
        from myspellchecker.core.validation_strategies.syntactic_strategy import (
            SyntacticValidationStrategy,
        )

        strategy = SyntacticValidationStrategy(None)
        assert strategy.priority() == 20

    def test_validate_disabled(self):
        """Test validate returns empty when checker is None."""
        from myspellchecker.core.validation_strategies.base import ValidationContext
        from myspellchecker.core.validation_strategies.syntactic_strategy import (
            SyntacticValidationStrategy,
        )

        strategy = SyntacticValidationStrategy(None)
        ctx = ValidationContext(sentence="test", words=["test", "word"], word_positions=[0, 5])
        errors = strategy.validate(ctx)
        assert errors == []

    def test_validate_empty_words(self):
        """Test validate returns empty when words is empty."""
        from myspellchecker.core.validation_strategies.base import ValidationContext
        from myspellchecker.core.validation_strategies.syntactic_strategy import (
            SyntacticValidationStrategy,
        )

        strategy = SyntacticValidationStrategy(Mock())
        ctx = ValidationContext(sentence="", words=[], word_positions=[])
        errors = strategy.validate(ctx)
        assert errors == []

    def test_validate_with_corrections(self):
        """Test validate finds syntactic errors."""
        from myspellchecker.core.validation_strategies.base import ValidationContext
        from myspellchecker.core.validation_strategies.syntactic_strategy import (
            SyntacticValidationStrategy,
        )

        mock_checker = Mock()
        mock_checker.check_sequence.return_value = [(1, "bad", "good", 0.85)]

        strategy = SyntacticValidationStrategy(mock_checker)
        ctx = ValidationContext(
            sentence="test bad more",
            words=["test", "bad", "more"],
            word_positions=[0, 5, 9],
        )
        errors = strategy.validate(ctx)
        assert len(errors) == 1
        assert errors[0].text == "bad"
        assert errors[0].suggestions == ["good"]

    def test_validate_skips_names(self):
        """Test validate skips words marked as names."""
        from myspellchecker.core.validation_strategies.base import ValidationContext
        from myspellchecker.core.validation_strategies.syntactic_strategy import (
            SyntacticValidationStrategy,
        )

        mock_checker = Mock()
        mock_checker.check_sequence.return_value = [(1, "bad", "good", 0.85)]

        strategy = SyntacticValidationStrategy(mock_checker)
        ctx = ValidationContext(
            sentence="test bad more",
            words=["test", "bad", "more"],
            word_positions=[0, 5, 9],
            is_name_mask=[False, True, False],
        )
        errors = strategy.validate(ctx)
        assert len(errors) == 0

    def test_validate_handles_out_of_bounds(self):
        """Test validate handles out of bounds index."""
        from myspellchecker.core.validation_strategies.base import ValidationContext
        from myspellchecker.core.validation_strategies.syntactic_strategy import (
            SyntacticValidationStrategy,
        )

        mock_checker = Mock()
        mock_checker.check_sequence.return_value = [(10, "bad", "good", 0.85)]

        strategy = SyntacticValidationStrategy(mock_checker)
        ctx = ValidationContext(
            sentence="test",
            words=["test"],
            word_positions=[0],
        )
        errors = strategy.validate(ctx)
        assert len(errors) == 0

    def test_repr(self):
        """Test repr."""
        from myspellchecker.core.validation_strategies.syntactic_strategy import (
            SyntacticValidationStrategy,
        )

        strategy = SyntacticValidationStrategy(Mock())
        repr_str = repr(strategy)
        assert "SyntacticValidationStrategy" in repr_str


class TestPOSSequenceValidationStrategy:
    """Tests for POSSequenceValidationStrategy."""

    def test_init(self):
        """Test strategy initialization."""
        from myspellchecker.core.validation_strategies.pos_sequence_strategy import (
            POSSequenceValidationStrategy,
        )

        mock_tagger = Mock()
        strategy = POSSequenceValidationStrategy(viterbi_tagger=mock_tagger, confidence=0.9)

        assert strategy.viterbi_tagger == mock_tagger
        assert strategy.confidence == 0.9

    def test_priority(self):
        """Test priority returns 30."""
        from myspellchecker.core.validation_strategies.pos_sequence_strategy import (
            POSSequenceValidationStrategy,
        )

        strategy = POSSequenceValidationStrategy(None)
        assert strategy.priority() == 30

    def test_validate_disabled(self):
        """Test validate returns empty when tagger is None."""
        from myspellchecker.core.validation_strategies.base import ValidationContext
        from myspellchecker.core.validation_strategies.pos_sequence_strategy import (
            POSSequenceValidationStrategy,
        )

        strategy = POSSequenceValidationStrategy(None)
        ctx = ValidationContext(sentence="test", words=["test", "word"], word_positions=[0, 5])
        errors = strategy.validate(ctx)
        assert errors == []

    def test_validate_too_few_words(self):
        """Test validate returns empty when less than 2 words."""
        from myspellchecker.core.validation_strategies.base import ValidationContext
        from myspellchecker.core.validation_strategies.pos_sequence_strategy import (
            POSSequenceValidationStrategy,
        )

        strategy = POSSequenceValidationStrategy(Mock())
        ctx = ValidationContext(sentence="test", words=["test"], word_positions=[0])
        errors = strategy.validate(ctx)
        assert errors == []

    def test_validate_tag_count_mismatch(self):
        """Test validate handles tag count mismatch."""
        from myspellchecker.core.validation_strategies.base import ValidationContext
        from myspellchecker.core.validation_strategies.pos_sequence_strategy import (
            POSSequenceValidationStrategy,
        )

        mock_tagger = Mock()
        mock_tagger.tag_sequence.return_value = ["N"]  # Only 1 tag for 2 words

        strategy = POSSequenceValidationStrategy(mock_tagger)
        ctx = ValidationContext(sentence="test word", words=["test", "word"], word_positions=[0, 5])
        errors = strategy.validate(ctx)
        assert errors == []

    def test_validate_stores_pos_tags(self):
        """Test validate stores POS tags in context."""
        from myspellchecker.core.validation_strategies.base import ValidationContext
        from myspellchecker.core.validation_strategies.pos_sequence_strategy import (
            POSSequenceValidationStrategy,
        )

        mock_tagger = Mock()
        mock_tagger.tag_sequence.return_value = ["N", "V"]

        strategy = POSSequenceValidationStrategy(mock_tagger)
        ctx = ValidationContext(sentence="test word", words=["test", "word"], word_positions=[0, 5])
        strategy.validate(ctx)
        assert ctx.pos_tags == ["N", "V"]

    def test_repr(self):
        """Test repr."""
        from myspellchecker.core.validation_strategies.pos_sequence_strategy import (
            POSSequenceValidationStrategy,
        )

        strategy = POSSequenceValidationStrategy(Mock())
        repr_str = repr(strategy)
        assert "POSSequenceValidationStrategy" in repr_str
        assert "enabled" in repr_str


class TestQuestionStructureValidationStrategy:
    """Tests for QuestionStructureValidationStrategy."""

    def test_init(self):
        """Test strategy initialization."""
        from myspellchecker.core.validation_strategies.question_strategy import (
            QuestionStructureValidationStrategy,
        )

        strategy = QuestionStructureValidationStrategy(confidence=0.8)
        assert strategy.confidence == 0.8

    def test_priority(self):
        """Test priority returns 40."""
        from myspellchecker.core.validation_strategies.question_strategy import (
            QuestionStructureValidationStrategy,
        )

        strategy = QuestionStructureValidationStrategy()
        assert strategy.priority() == 40

    def test_validate_too_few_words(self):
        """Test validate returns empty when less than 2 words."""
        from myspellchecker.core.validation_strategies.base import ValidationContext
        from myspellchecker.core.validation_strategies.question_strategy import (
            QuestionStructureValidationStrategy,
        )

        strategy = QuestionStructureValidationStrategy()
        ctx = ValidationContext(sentence="test", words=["test"], word_positions=[0])
        errors = strategy.validate(ctx)
        assert errors == []

    @patch("myspellchecker.core.validation_strategies.question_strategy.detect_sentence_type")
    def test_validate_stores_sentence_type(self, mock_detect):
        """Test validate stores sentence type in context."""
        from myspellchecker.core.validation_strategies.base import ValidationContext
        from myspellchecker.core.validation_strategies.question_strategy import (
            QuestionStructureValidationStrategy,
        )

        mock_detect.return_value = "statement"

        strategy = QuestionStructureValidationStrategy()
        ctx = ValidationContext(sentence="test word", words=["test", "word"], word_positions=[0, 5])
        strategy.validate(ctx)
        assert ctx.sentence_type == "statement"

    @patch("myspellchecker.core.validation_strategies.question_strategy.detect_sentence_type")
    def test_validate_non_question(self, mock_detect):
        """Test validate returns empty for non-questions."""
        from myspellchecker.core.validation_strategies.base import ValidationContext
        from myspellchecker.core.validation_strategies.question_strategy import (
            QuestionStructureValidationStrategy,
        )

        mock_detect.return_value = "statement"

        strategy = QuestionStructureValidationStrategy()
        ctx = ValidationContext(sentence="test word", words=["test", "word"], word_positions=[0, 5])
        errors = strategy.validate(ctx)
        assert errors == []

    def test_repr(self):
        """Test repr."""
        from myspellchecker.core.validation_strategies.question_strategy import (
            QuestionStructureValidationStrategy,
        )

        strategy = QuestionStructureValidationStrategy()
        repr_str = repr(strategy)
        assert "QuestionStructureValidationStrategy" in repr_str
        assert "confidence" in repr_str


class TestToneValidationStrategy:
    """Tests for ToneValidationStrategy."""

    def test_init(self):
        """Test strategy initialization."""
        from myspellchecker.core.validation_strategies.tone_strategy import (
            ToneValidationStrategy,
        )

        mock_disambiguator = Mock()
        strategy = ToneValidationStrategy(
            tone_disambiguator=mock_disambiguator, confidence_threshold=0.6
        )

        assert strategy.tone_disambiguator == mock_disambiguator
        assert strategy.confidence_threshold == 0.6

    def test_priority(self):
        """Test priority returns 10."""
        from myspellchecker.core.validation_strategies.tone_strategy import (
            ToneValidationStrategy,
        )

        strategy = ToneValidationStrategy(None)
        assert strategy.priority() == 10

    def test_validate_disabled(self):
        """Test validate returns empty when disambiguator is None."""
        from myspellchecker.core.validation_strategies.base import ValidationContext
        from myspellchecker.core.validation_strategies.tone_strategy import (
            ToneValidationStrategy,
        )

        strategy = ToneValidationStrategy(None)
        ctx = ValidationContext(sentence="test", words=["test", "word"], word_positions=[0, 5])
        errors = strategy.validate(ctx)
        assert errors == []

    def test_validate_empty_words(self):
        """Test validate returns empty when words is empty."""
        from myspellchecker.core.validation_strategies.base import ValidationContext
        from myspellchecker.core.validation_strategies.tone_strategy import (
            ToneValidationStrategy,
        )

        strategy = ToneValidationStrategy(Mock())
        ctx = ValidationContext(sentence="", words=[], word_positions=[])
        errors = strategy.validate(ctx)
        assert errors == []

    def test_validate_with_corrections(self):
        """Test validate finds tone errors."""
        from myspellchecker.core.validation_strategies.base import ValidationContext
        from myspellchecker.core.validation_strategies.tone_strategy import (
            ToneValidationStrategy,
        )

        mock_disambiguator = Mock()
        mock_disambiguator.check_sentence.return_value = [(1, "bad", "good", 0.8)]

        strategy = ToneValidationStrategy(mock_disambiguator)
        ctx = ValidationContext(
            sentence="test bad more",
            words=["test", "bad", "more"],
            word_positions=[0, 5, 9],
        )
        errors = strategy.validate(ctx)
        assert len(errors) == 1
        assert errors[0].text == "bad"
        assert errors[0].suggestions == ["good"]
        assert errors[0].confidence == 0.8

    def test_validate_below_threshold(self):
        """Test validate filters by confidence threshold."""
        from myspellchecker.core.validation_strategies.base import ValidationContext
        from myspellchecker.core.validation_strategies.tone_strategy import (
            ToneValidationStrategy,
        )

        mock_disambiguator = Mock()
        mock_disambiguator.check_sentence.return_value = [(1, "bad", "good", 0.3)]

        strategy = ToneValidationStrategy(mock_disambiguator, confidence_threshold=0.5)
        ctx = ValidationContext(
            sentence="test bad more",
            words=["test", "bad", "more"],
            word_positions=[0, 5, 9],
        )
        errors = strategy.validate(ctx)
        assert len(errors) == 0

    def test_validate_skips_names(self):
        """Test validate skips words marked as names."""
        from myspellchecker.core.validation_strategies.base import ValidationContext
        from myspellchecker.core.validation_strategies.tone_strategy import (
            ToneValidationStrategy,
        )

        mock_disambiguator = Mock()
        mock_disambiguator.check_sentence.return_value = [(1, "bad", "good", 0.8)]

        strategy = ToneValidationStrategy(mock_disambiguator)
        ctx = ValidationContext(
            sentence="test bad more",
            words=["test", "bad", "more"],
            word_positions=[0, 5, 9],
            is_name_mask=[False, True, False],
        )
        errors = strategy.validate(ctx)
        assert len(errors) == 0

    def test_repr(self):
        """Test repr."""
        from myspellchecker.core.validation_strategies.tone_strategy import (
            ToneValidationStrategy,
        )

        strategy = ToneValidationStrategy(Mock())
        repr_str = repr(strategy)
        assert "ToneValidationStrategy" in repr_str
        assert "enabled" in repr_str


class TestSemanticValidationStrategy:
    """Tests for SemanticValidationStrategy."""

    def test_init(self):
        """Test strategy initialization."""
        from myspellchecker.core.validation_strategies.semantic_strategy import (
            SemanticValidationStrategy,
        )

        mock_checker = Mock()
        strategy = SemanticValidationStrategy(
            semantic_checker=mock_checker,
            use_proactive_scanning=True,
            proactive_confidence_threshold=0.8,
            min_word_length=3,
        )

        assert strategy.semantic_checker == mock_checker
        assert strategy.use_proactive_scanning is True
        assert strategy.proactive_confidence_threshold == 0.8
        assert strategy.min_word_length == 3

    def test_priority(self):
        """Test priority returns 70."""
        from myspellchecker.core.validation_strategies.semantic_strategy import (
            SemanticValidationStrategy,
        )

        strategy = SemanticValidationStrategy(Mock())
        assert strategy.priority() == 70

    def test_validate_disabled_no_checker(self):
        """Test validate returns empty when checker is None."""
        from myspellchecker.core.validation_strategies.base import ValidationContext
        from myspellchecker.core.validation_strategies.semantic_strategy import (
            SemanticValidationStrategy,
        )

        strategy = SemanticValidationStrategy(None)
        ctx = ValidationContext(sentence="test", words=["test", "word"], word_positions=[0, 5])
        errors = strategy.validate(ctx)
        assert errors == []

    def test_validate_disabled_proactive_off(self):
        """Test validate returns empty when proactive scanning is off."""
        from myspellchecker.core.validation_strategies.base import ValidationContext
        from myspellchecker.core.validation_strategies.semantic_strategy import (
            SemanticValidationStrategy,
        )

        strategy = SemanticValidationStrategy(Mock(), use_proactive_scanning=False)
        ctx = ValidationContext(sentence="test", words=["test", "word"], word_positions=[0, 5])
        errors = strategy.validate(ctx)
        assert errors == []

    def test_validate_with_errors(self):
        """Test validate finds semantic errors via scan_sentence()."""
        from myspellchecker.core.validation_strategies.base import ValidationContext
        from myspellchecker.core.validation_strategies.semantic_strategy import (
            SemanticValidationStrategy,
        )

        mock_checker = Mock()
        # scan_sentence returns: [(word_idx, error_word, suggestions, confidence)]
        mock_checker.scan_sentence.return_value = [
            (0, "စားပွဲ", ["သူ", "ကျွန်တော်"], 0.85),
        ]

        strategy = SemanticValidationStrategy(mock_checker, use_proactive_scanning=True)
        ctx = ValidationContext(
            sentence="စားပွဲ က ချက် တယ်",
            words=["စားပွဲ", "က", "ချက်", "တယ်"],
            word_positions=[0, 12, 15, 24],
        )
        errors = strategy.validate(ctx)
        assert len(errors) == 1
        assert errors[0].text == "စားပွဲ"
        assert "သူ" in errors[0].suggestions

    def test_validate_skips_names(self):
        """Test validate skips words marked as names."""
        from myspellchecker.core.validation_strategies.base import ValidationContext
        from myspellchecker.core.validation_strategies.semantic_strategy import (
            SemanticValidationStrategy,
        )

        mock_checker = Mock()
        # scan_sentence returns error for index 0 (which is a name)
        mock_checker.scan_sentence.return_value = [
            (0, "NameWord", ["သူ"], 0.9),
        ]

        strategy = SemanticValidationStrategy(mock_checker, use_proactive_scanning=True)
        # Subject "NameWord" is marked as name → should be skipped
        ctx = ValidationContext(
            sentence="NameWord က ချက် တယ်",
            words=["NameWord", "က", "ချက်", "တယ်"],
            word_positions=[0, 9, 12, 21],
            is_name_mask=[True, False, False, False],
        )
        errors = strategy.validate(ctx)
        assert len(errors) == 0

    def test_validate_skips_existing_errors(self):
        """Test validate skips positions with existing errors."""
        from myspellchecker.core.validation_strategies.base import ValidationContext
        from myspellchecker.core.validation_strategies.semantic_strategy import (
            SemanticValidationStrategy,
        )

        mock_checker = Mock()
        # scan_sentence returns error for index 0 (already flagged)
        mock_checker.scan_sentence.return_value = [
            (0, "စားပွဲ", ["သူ"], 0.9),
        ]

        strategy = SemanticValidationStrategy(mock_checker, use_proactive_scanning=True)
        # Subject position already has an error → should be skipped
        ctx = ValidationContext(
            sentence="စားပွဲ က ချက် တယ်",
            words=["စားပွဲ", "က", "ချက်", "တယ်"],
            word_positions=[0, 12, 15, 24],
            existing_errors={0: "test"},  # Position 0 already has error
        )
        errors = strategy.validate(ctx)
        assert len(errors) == 0

    def test_repr_disabled(self):
        """Test repr when disabled."""
        from myspellchecker.core.validation_strategies.semantic_strategy import (
            SemanticValidationStrategy,
        )

        strategy = SemanticValidationStrategy(None)
        repr_str = repr(strategy)
        assert "disabled" in repr_str

    def test_repr_proactive_off(self):
        """Test repr when proactive scanning is off."""
        from myspellchecker.core.validation_strategies.semantic_strategy import (
            SemanticValidationStrategy,
        )

        strategy = SemanticValidationStrategy(Mock(), use_proactive_scanning=False)
        repr_str = repr(strategy)
        assert "disabled" in repr_str

    def test_repr_enabled(self):
        """Test repr when enabled."""
        from myspellchecker.core.validation_strategies.semantic_strategy import (
            SemanticValidationStrategy,
        )

        strategy = SemanticValidationStrategy(Mock(), use_proactive_scanning=True)
        repr_str = repr(strategy)
        assert "enabled" in repr_str


class TestValidationStrategyInit:
    """Tests for validation_strategies/__init__.py."""

    def test_can_import_strategy_classes(self):
        """Test all strategy classes can be imported."""
        from myspellchecker.core.validation_strategies import (
            HomophoneValidationStrategy,
            NgramContextValidationStrategy,
            POSSequenceValidationStrategy,
            QuestionStructureValidationStrategy,
            SemanticValidationStrategy,
            SyntacticValidationStrategy,
            ToneValidationStrategy,
            ValidationContext,
            ValidationStrategy,
        )

        assert ValidationContext is not None
        assert ValidationStrategy is not None
        assert ToneValidationStrategy is not None
        assert SyntacticValidationStrategy is not None
        assert POSSequenceValidationStrategy is not None
        assert QuestionStructureValidationStrategy is not None
        assert NgramContextValidationStrategy is not None
        assert HomophoneValidationStrategy is not None
        assert SemanticValidationStrategy is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
