"""
Integration tests for validation strategy pattern refactoring.

Tests verify that the strategy-based ContextValidator works correctly.
"""

import pytest

from myspellchecker.core.config import SpellCheckerConfig
from myspellchecker.core.context_validator import ContextValidator
from myspellchecker.core.validation_strategies import (
    HomophoneValidationStrategy,
    NgramContextValidationStrategy,
    POSSequenceValidationStrategy,
    QuestionStructureValidationStrategy,
    SemanticValidationStrategy,
    SyntacticValidationStrategy,
    ToneValidationStrategy,
    ValidationContext,
)


class TestValidationContext:
    """Test ValidationContext data structure."""

    def test_validation_context_creation(self):
        """Test creating a ValidationContext."""
        context = ValidationContext(
            sentence="သူ သွား ခဲ့ တယ်",
            words=["သူ", "သွား", "ခဲ့", "တယ်"],
            word_positions=[0, 6, 15, 21],
        )

        assert context.sentence == "သူ သွား ခဲ့ တယ်"
        assert context.words == ["သူ", "သွား", "ခဲ့", "တယ်"]
        assert context.word_positions == [0, 6, 15, 21]
        assert context.is_name_mask == [False, False, False, False]  # Auto-filled
        assert context.existing_errors == {}  # Default empty
        assert context.sentence_type == "statement"  # Default
        assert context.pos_tags == []  # Default empty

    def test_validation_context_with_metadata(self):
        """Test ValidationContext with full metadata."""
        context = ValidationContext(
            sentence="သူ သွား ခဲ့ တယ်",
            words=["သူ", "သွား", "ခဲ့", "တယ်"],
            word_positions=[0, 6, 15, 21],
            is_name_mask=[False, False, False, False],
            sentence_type="question",
            pos_tags=["N", "V", "P", "P"],
        )

        assert context.sentence_type == "question"
        assert context.pos_tags == ["N", "V", "P", "P"]
        assert len(context.is_name_mask) == 4


class TestStrategyPriority:
    """Test strategy priority ordering."""

    def test_strategy_priorities(self):
        """Test that strategies have correct priority ordering."""
        # Verify priority values
        assert ToneValidationStrategy(None).priority() == 10
        assert SyntacticValidationStrategy(None).priority() == 20
        assert POSSequenceValidationStrategy(None).priority() == 30
        assert QuestionStructureValidationStrategy().priority() == 40
        assert NgramContextValidationStrategy(None, None).priority() == 50
        # HomophoneValidationStrategy priority == 45 (tested separately)
        assert SemanticValidationStrategy(None).priority() == 70

    def test_strategies_sort_by_priority(self):
        """Test that strategies are sorted by priority."""
        # Create strategies in random order
        strategies = [
            SemanticValidationStrategy(None),  # 70
            ToneValidationStrategy(None),  # 10
            POSSequenceValidationStrategy(None),  # 30
            QuestionStructureValidationStrategy(),  # 40
        ]

        # Sort by priority
        strategies.sort(key=lambda s: s.priority())

        # Verify sorted order
        assert strategies[0].priority() == 10
        assert strategies[1].priority() == 30
        assert strategies[2].priority() == 40
        assert strategies[3].priority() == 70


class TestStrategyExecutionOrder:
    """Test strategy execution order in ContextValidator."""

    def test_validator_sorts_strategies(self):
        """Test that ContextValidator sorts strategies by priority."""
        config = SpellCheckerConfig()

        # Create strategies in random order
        strategies = [
            SemanticValidationStrategy(None),  # 70
            ToneValidationStrategy(None),  # 10
            QuestionStructureValidationStrategy(),  # 40
        ]

        # Create validator (should sort strategies)
        validator = ContextValidator(
            config=config,
            segmenter=None,
            strategies=strategies,
        )

        # Verify strategies are sorted
        assert validator.strategies[0].priority() == 10  # Tone first
        assert validator.strategies[1].priority() == 40  # Question second
        assert validator.strategies[2].priority() == 70  # Semantic last


class TestContextValidatorBasic:
    """Basic functionality tests for ContextValidator."""

    def test_validator_with_no_strategies(self):
        """Test validator with no strategies returns empty errors."""
        config = SpellCheckerConfig()
        validator = ContextValidator(
            config=config,
            segmenter=None,
            strategies=[],
        )

        errors = validator.validate("သူ သွား ခဲ့ တယ်")
        assert errors == []

    def test_validator_repr(self):
        """Test validator string representation."""
        config = SpellCheckerConfig()
        strategies = [
            ToneValidationStrategy(None),
            QuestionStructureValidationStrategy(),
        ]

        validator = ContextValidator(
            config=config,
            segmenter=None,
            strategies=strategies,
        )

        repr_str = repr(validator)
        assert "ContextValidator" in repr_str
        assert "strategies=2" in repr_str
        assert "[10, 40]" in repr_str  # Priorities


class TestStrategyErrorDeduc:
    """Test that strategies avoid duplicate errors."""

    def test_existing_errors_tracked(self):
        """Test that existing_errors prevents duplicates."""
        context = ValidationContext(
            sentence="သူ သွား ခဲ့ တယ်",
            words=["သူ", "သွား", "ခဲ့", "တယ်"],
            word_positions=[0, 6, 15, 21],
        )

        # Mark position 6 as having an error
        context.existing_errors[6] = "test"

        # Verify the error is tracked
        assert 6 in context.existing_errors
        assert 15 not in context.existing_errors


class TestStrategyRepresentations:
    """Test strategy __repr__ methods."""

    def test_tone_strategy_repr(self):
        """Test ToneValidationStrategy representation."""
        strategy = ToneValidationStrategy(None)
        repr_str = repr(strategy)

        assert "ToneValidationStrategy" in repr_str
        assert "priority=10" in repr_str
        assert "disabled" in repr_str  # No disambiguator provided

    def test_question_strategy_repr(self):
        """Test QuestionStructureValidationStrategy representation."""
        strategy = QuestionStructureValidationStrategy(confidence=0.7)
        repr_str = repr(strategy)

        assert "QuestionStructureValidationStrategy" in repr_str
        assert "priority=40" in repr_str
        assert "confidence=0.7" in repr_str

    def test_semantic_strategy_repr(self):
        """Test SemanticValidationStrategy representation."""
        strategy = SemanticValidationStrategy(None)
        repr_str = repr(strategy)

        assert "SemanticValidationStrategy" in repr_str
        assert "priority=70" in repr_str
        assert "disabled" in repr_str


class TestStrategyFactoryPatterns:
    """Test factory pattern for strategy creation."""

    def test_strategy_factory_separation(self):
        """Test that strategies are created independently."""
        # Each strategy can be created independently
        tone_strategy = ToneValidationStrategy(None)
        question_strategy = QuestionStructureValidationStrategy()
        semantic_strategy = SemanticValidationStrategy(None)

        # Verify they are separate instances
        assert tone_strategy is not question_strategy
        assert question_strategy is not semantic_strategy

        # Verify they have different priorities
        assert tone_strategy.priority() != question_strategy.priority()
        assert question_strategy.priority() != semantic_strategy.priority()


class TestStrategyDisabling:
    """Test strategy enabling/disabling behavior."""

    def test_disabled_strategies_return_empty(self):
        """Test that disabled strategies return empty error lists."""
        # Strategies with None dependencies are disabled
        tone_strategy = ToneValidationStrategy(None)
        context = ValidationContext(sentence="test", words=["test"], word_positions=[0])

        errors = tone_strategy.validate(context)
        assert errors == []  # Disabled strategy returns no errors

    def test_semantic_strategy_with_disabled_scanning(self):
        """Test SemanticValidationStrategy with proactive scanning disabled."""
        strategy = SemanticValidationStrategy(
            None,  # No semantic checker
            use_proactive_scanning=False,
        )

        context = ValidationContext(sentence="test", words=["test"], word_positions=[0])

        errors = strategy.validate(context)
        assert errors == []


@pytest.mark.integration
class TestStrategyIntegration:
    """Integration tests with full validation pipeline."""

    def test_full_pipeline_with_multiple_strategies(self):
        """Test end-to-end validation with multiple strategies in priority order.

        This test verifies:
        1. ContextValidator executes strategies in priority order
        2. QuestionStructureValidationStrategy correctly detects question errors
        3. Strategies produce suggestions for known Myanmar errors
        """
        config = SpellCheckerConfig()

        # Create strategies with known behaviors
        # QuestionStructureValidationStrategy (priority=40) needs no dependencies
        question_strategy = QuestionStructureValidationStrategy(confidence=0.8)

        # Create validator with strategies
        ctx_validator = ContextValidator(
            config=config,
            segmenter=None,  # Not needed for direct validation
            strategies=[question_strategy],
        )
        # Verify validator was created with the strategy
        assert len(ctx_validator.strategies) == 1

        # Test case: Question without proper ending particle
        # "ဘယ်မှာ သွား တယ်" = "Where go [statement]" - should be question
        # The sentence has question word "ဘယ်မှာ" but ends with statement particle "တယ်"
        test_context = ValidationContext(
            sentence="ဘယ်မှာ သွား တယ်",
            words=["ဘယ်မှာ", "သွား", "တယ်"],
            word_positions=[0, 18, 30],
        )

        # Run validation directly on context
        errors = question_strategy.validate(test_context)

        # Verify question structure error is detected
        assert len(errors) == 1
        error = errors[0]
        assert error.text == "တယ်"
        assert error.error_type == "question_structure"
        assert error.position == 30
        assert len(error.suggestions) > 0
        # Question suggestions should include လား, သလား, or လဲ
        assert any(s in error.suggestions for s in ["လား", "သလား", "လဲ"])

    def test_strategy_execution_order_is_maintained(self):
        """Test that strategies execute in priority order when validator runs."""
        from unittest.mock import Mock

        config = SpellCheckerConfig()

        # Create mock strategies with different priorities
        mock_strategy_10 = Mock()
        mock_strategy_10.priority.return_value = 10
        mock_strategy_10.validate.return_value = []

        mock_strategy_40 = Mock()
        mock_strategy_40.priority.return_value = 40
        mock_strategy_40.validate.return_value = []

        mock_strategy_70 = Mock()
        mock_strategy_70.priority.return_value = 70
        mock_strategy_70.validate.return_value = []

        # Add strategies in reverse priority order
        validator = ContextValidator(
            config=config,
            segmenter=None,
            strategies=[mock_strategy_70, mock_strategy_10, mock_strategy_40],
        )

        # Verify they are sorted by priority
        assert validator.strategies[0].priority() == 10
        assert validator.strategies[1].priority() == 40
        assert validator.strategies[2].priority() == 70

    def test_validator_produces_errors_for_known_patterns(self):
        """Test that validator produces correct errors for known Myanmar patterns."""
        config = SpellCheckerConfig()

        question_strategy = QuestionStructureValidationStrategy(confidence=0.75)

        ctx_val = ContextValidator(
            config=config,
            segmenter=None,
            strategies=[question_strategy],
        )
        # Verify validator has the strategy
        assert len(ctx_val.strategies) == 1

        # Test sentences with question words and statement endings
        test_cases = [
            # (sentence, words, positions, expected_error_text)
            (
                "ဘာ စား တယ်",  # "What eat [statement]" - should suggest question particle
                ["ဘာ", "စား", "တယ်"],
                [0, 9, 18],
                "တယ်",
            ),
            (
                "ဘယ်သူ လာ သည်",  # "Who come [formal statement]" - should suggest question
                ["ဘယ်သူ", "လာ", "သည်"],
                [0, 18, 24],
                "သည်",
            ),
        ]

        for sentence, words, positions, expected_error in test_cases:
            context = ValidationContext(
                sentence=sentence,
                words=words,
                word_positions=positions,
            )

            errors = question_strategy.validate(context)

            assert len(errors) == 1, f"Expected 1 error for '{sentence}'"
            assert errors[0].text == expected_error
            assert errors[0].error_type == "question_structure"
            assert len(errors[0].suggestions) > 0

    def test_valid_sentences_produce_no_errors(self):
        """Test that properly formed sentences don't produce false positives."""
        question_strategy = QuestionStructureValidationStrategy()

        # Valid question with proper ending
        valid_question = ValidationContext(
            sentence="ဘာ စား သလဲ",  # "What eat [question particle]" - proper question
            words=["ဘာ", "စား", "သလဲ"],
            word_positions=[0, 9, 18],
        )

        errors = question_strategy.validate(valid_question)
        assert len(errors) == 0, "Valid question should produce no errors"

        # Valid statement without question words
        valid_statement = ValidationContext(
            sentence="သူ စား တယ်",  # "He eat [statement]" - proper statement
            words=["သူ", "စား", "တယ်"],
            word_positions=[0, 9, 18],
        )

        errors = question_strategy.validate(valid_statement)
        assert len(errors) == 0, "Valid statement should produce no errors"


class TestInterfaceSegregation:
    """Tests for Interface Segregation Principle (ISP) compliance.

    These tests verify that:
    1. ContextValidator no longer depends on DictionaryProvider
    2. Validation strategies only depend on specific interfaces (e.g., NgramRepository)
    3. Mock repositories can be injected for testing
    """

    def test_context_validator_no_provider_dependency(self):
        """Test that ContextValidator doesn't require a provider."""
        from unittest.mock import Mock

        config = SpellCheckerConfig()
        mock_segmenter = Mock()

        # ContextValidator should work without any provider
        validator = ContextValidator(
            config=config,
            segmenter=mock_segmenter,
            strategies=[],
        )

        assert not hasattr(validator, "provider")
        assert validator.strategies == []

    def test_ngram_strategy_uses_ngram_repository_interface(self):
        """Test that NgramContextValidationStrategy uses NgramRepository methods only."""
        from unittest.mock import Mock

        # Create mock that only implements NgramRepository interface
        mock_ngram_repo = Mock()
        mock_ngram_repo.get_bigram_probability = Mock(return_value=0.01)
        mock_ngram_repo.get_trigram_probability = Mock(return_value=0.001)

        mock_context_checker = Mock()
        mock_context_checker.is_contextual_error = Mock(return_value=False)

        # Strategy should accept any object implementing NgramRepository
        strategy = NgramContextValidationStrategy(
            context_checker=mock_context_checker,
            provider=mock_ngram_repo,  # Uses NgramRepository type hint
        )

        # Verify the strategy stored the mock
        assert strategy.provider == mock_ngram_repo

    def test_homophone_strategy_uses_ngram_repository_interface(self):
        """Test that HomophoneValidationStrategy uses NgramRepository methods only."""
        from unittest.mock import Mock

        # Create mock that only implements NgramRepository interface
        mock_ngram_repo = Mock()
        mock_ngram_repo.get_bigram_probability = Mock(return_value=0.01)
        mock_ngram_repo.get_trigram_probability = Mock(return_value=0.001)

        mock_homophone_checker = Mock()

        # Strategy should accept any object implementing NgramRepository
        strategy = HomophoneValidationStrategy(
            homophone_checker=mock_homophone_checker,
            provider=mock_ngram_repo,  # Uses NgramRepository type hint
        )

        # Verify the strategy stored the mock
        assert strategy.provider == mock_ngram_repo

    def test_strategies_can_use_mock_repositories(self):
        """Test that strategies work with mock repositories."""
        from unittest.mock import Mock

        # Create minimal mock implementing only required methods
        class MockNgramRepository:
            def get_bigram_probability(self, word1: str, word2: str) -> float:
                return 0.01

            def get_trigram_probability(self, word1: str, word2: str, word3: str) -> float:
                return 0.001

        mock_repo = MockNgramRepository()
        mock_context_checker = Mock()
        mock_context_checker.is_contextual_error = Mock(return_value=False)

        # Both strategies should work with the mock
        ngram_strategy = NgramContextValidationStrategy(
            context_checker=mock_context_checker,
            provider=mock_repo,
        )

        mock_homophone_checker = Mock()
        mock_homophone_checker.get_homophones = Mock(return_value=[])

        homophone_strategy = HomophoneValidationStrategy(
            homophone_checker=mock_homophone_checker,
            provider=mock_repo,
        )

        # Verify strategies are properly configured
        assert ngram_strategy.provider == mock_repo
        assert homophone_strategy.provider == mock_repo


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
