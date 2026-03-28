"""Comprehensive tests for algorithms/suggestion_strategy.py.

Tests cover:
- SuggestionContext dataclass
- BaseSuggestionStrategy base class
- CompositeSuggestionStrategy implementation
- Edge cases and protocol compliance
"""

from unittest.mock import MagicMock, PropertyMock

import pytest

from myspellchecker.algorithms.ranker import SuggestionData
from myspellchecker.algorithms.suggestion_strategy import (
    BaseSuggestionStrategy,
    CompositeSuggestionStrategy,
    SuggestionContext,
    SuggestionResult,
    SuggestionStrategy,
)


class TestBaseSuggestionStrategy:
    """Tests for BaseSuggestionStrategy base class."""

    def test_suggest_not_implemented(self):
        """Test suggest raises NotImplementedError."""
        strategy = BaseSuggestionStrategy()
        with pytest.raises(NotImplementedError, match="Subclasses must implement"):
            strategy.suggest("test")

    def test_suggest_batch_default(self):
        """Test suggest_batch calls suggest for each term."""

        class TestStrategy(BaseSuggestionStrategy):
            def suggest(self, term, context=None):
                return SuggestionResult(
                    suggestions=[
                        SuggestionData(
                            term=f"corrected_{term}",
                            edit_distance=1,
                            frequency=100,
                            confidence=0.9,
                        )
                    ],
                    strategy_name="test",
                )

        strategy = TestStrategy()
        results = strategy.suggest_batch(["word1", "word2", "word3"])
        assert len(results) == 3
        assert results[0].terms == ["corrected_word1"]
        assert results[1].terms == ["corrected_word2"]
        assert results[2].terms == ["corrected_word3"]

    def test_suggest_batch_with_contexts(self):
        """Test suggest_batch with matching contexts."""

        class TestStrategy(BaseSuggestionStrategy):
            def suggest(self, term, context=None):
                ctx_info = context.max_suggestions if context else 0
                return SuggestionResult(
                    suggestions=[
                        SuggestionData(
                            term=f"{term}_ctx{ctx_info}",
                            edit_distance=1,
                            frequency=100,
                            confidence=0.9,
                        )
                    ],
                    strategy_name="test",
                )

        strategy = TestStrategy()
        contexts = [
            SuggestionContext(max_suggestions=1),
            SuggestionContext(max_suggestions=2),
        ]
        results = strategy.suggest_batch(["word1", "word2"], contexts)
        assert results[0].terms == ["word1_ctx1"]
        assert results[1].terms == ["word2_ctx2"]

    def test_create_result_truncates(self):
        """Test _create_result truncates to max_suggestions and sets flag."""
        strategy = BaseSuggestionStrategy(max_suggestions=3)
        suggestions = [
            SuggestionData(term=f"term{i}", edit_distance=i, frequency=100 - i, confidence=0.9)
            for i in range(5)
        ]
        result = strategy._create_result(suggestions)
        assert len(result.suggestions) == 3
        assert result.is_truncated is True
        assert result.strategy_name == "base"

    def test_create_result_not_truncated(self):
        """Test _create_result when not truncated."""
        strategy = BaseSuggestionStrategy(max_suggestions=10)
        suggestions = [SuggestionData(term="term1", edit_distance=1, frequency=100, confidence=0.9)]
        result = strategy._create_result(suggestions)
        assert len(result.suggestions) == 1
        assert result.is_truncated is False


class TestCompositeSuggestionStrategy:
    """Tests for CompositeSuggestionStrategy."""

    @pytest.fixture
    def mock_strategy1(self):
        """Create a mock strategy."""
        strategy = MagicMock(spec=SuggestionStrategy)
        type(strategy).name = PropertyMock(return_value="strategy1")
        strategy.supports_context.return_value = False
        strategy.suggest.return_value = SuggestionResult(
            suggestions=[
                SuggestionData(term="hello", edit_distance=0, frequency=100, confidence=1.0),
                SuggestionData(term="help", edit_distance=2, frequency=80, confidence=0.6),
            ],
            strategy_name="strategy1",
        )
        return strategy

    @pytest.fixture
    def mock_strategy2(self):
        """Create another mock strategy."""
        strategy = MagicMock(spec=SuggestionStrategy)
        type(strategy).name = PropertyMock(return_value="strategy2")
        strategy.supports_context.return_value = True
        strategy.suggest.return_value = SuggestionResult(
            suggestions=[
                SuggestionData(term="hallo", edit_distance=1, frequency=50, confidence=0.8),
                SuggestionData(
                    term="hello", edit_distance=0, frequency=100, confidence=0.9
                ),  # duplicate
            ],
            strategy_name="strategy2",
        )
        return strategy

    def test_suggest_single_strategy(self, mock_strategy1):
        """Test suggest with single strategy."""
        composite = CompositeSuggestionStrategy(strategies=[mock_strategy1])
        result = composite.suggest("test")

        assert mock_strategy1.suggest.called
        assert len(result.suggestions) > 0
        assert result.strategy_name == "composite"
        assert "strategies" in result.metadata
        assert "strategy1" in result.metadata["strategies"]

    def test_suggest_multiple_strategies(self, mock_strategy1, mock_strategy2):
        """Test suggest with multiple strategies merges results."""
        composite = CompositeSuggestionStrategy(
            strategies=[mock_strategy1, mock_strategy2],
            max_suggestions=10,
        )
        result = composite.suggest("test")

        assert mock_strategy1.suggest.called
        assert mock_strategy2.suggest.called
        assert result.strategy_name == "composite"
        assert len(result.metadata["strategies"]) == 2

    def test_suggest_with_context(self, mock_strategy1):
        """Test suggest passes context to strategies."""
        composite = CompositeSuggestionStrategy(strategies=[mock_strategy1])
        context = SuggestionContext(prev_words=["the"])
        composite.suggest("test", context)

        mock_strategy1.suggest.assert_called_with("test", context)

    def test_suggest_deduplication(self, mock_strategy1, mock_strategy2):
        """Test that duplicates are removed."""
        composite = CompositeSuggestionStrategy(
            strategies=[mock_strategy1, mock_strategy2],
            deduplicate=True,
        )
        result = composite.suggest("test")

        # Both strategies return "hello", should be deduplicated
        terms = result.terms
        assert terms.count("hello") == 1

    def test_suggest_truncation(self, mock_strategy1, mock_strategy2):
        """Test that results are truncated to max_suggestions."""
        composite = CompositeSuggestionStrategy(
            strategies=[mock_strategy1, mock_strategy2],
            max_suggestions=2,
        )
        result = composite.suggest("test")

        assert len(result.suggestions) <= 2

    def test_deduplicate_suggestions_keeps_highest_confidence(self, mock_strategy1):
        """Test _deduplicate_suggestions keeps highest confidence."""
        composite = CompositeSuggestionStrategy(strategies=[mock_strategy1])

        suggestions = [
            SuggestionData(term="hello", edit_distance=1, frequency=100, confidence=0.5),
            SuggestionData(
                term="hello", edit_distance=0, frequency=100, confidence=0.9
            ),  # higher confidence
            SuggestionData(term="world", edit_distance=1, frequency=50, confidence=0.7),
        ]

        deduped = composite._deduplicate_suggestions(suggestions)

        hello_suggestions = [s for s in deduped if s.term == "hello"]
        assert len(hello_suggestions) == 1
        assert hello_suggestions[0].confidence == 0.9

        world_suggestions = [s for s in deduped if s.term == "world"]
        assert len(world_suggestions) == 1

    def test_empty_strategies_list(self):
        """Test composite with no strategies."""
        composite = CompositeSuggestionStrategy(strategies=[])
        result = composite.suggest("test")
        assert len(result.suggestions) == 0
        assert result.metadata["strategies"] == []

    def test_supports_context_delegates(self, mock_strategy1, mock_strategy2):
        """Test supports_context returns True when any sub-strategy supports it."""
        mock_strategy1.supports_context.return_value = False
        mock_strategy2.supports_context.return_value = True
        composite = CompositeSuggestionStrategy(strategies=[mock_strategy1, mock_strategy2])
        assert composite.supports_context() is True

    def test_supports_context_false_when_none_support(self, mock_strategy1):
        """Test supports_context returns False when no strategy supports it."""
        mock_strategy1.supports_context.return_value = False
        composite = CompositeSuggestionStrategy(strategies=[mock_strategy1])
        assert composite.supports_context() is False


class TestSuggestionContextImmutability:
    """Tests for SuggestionContext default factory isolation."""

    def test_default_factories_dont_share_state(self):
        """Test that context default factories don't share state."""
        ctx1 = SuggestionContext()
        ctx1.prev_words.append("test")

        ctx2 = SuggestionContext()
        assert ctx2.prev_words == []  # Should not be affected
