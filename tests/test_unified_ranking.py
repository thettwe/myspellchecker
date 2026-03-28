"""
Unit tests for the UnifiedRanker suggestion ranking system.

Tests the unified ranking that consolidates suggestions from multiple sources
(symspell, particle_typo, medial_confusion, morphology, semantic) with
source-specific weighting and deduplication.
"""

import pytest

from myspellchecker.algorithms.ranker import (
    SuggestionData,
    UnifiedRanker,
)


class TestSuggestionDataSource:
    """Tests for source and confidence fields in SuggestionData."""

    def test_default_source_is_symspell(self):
        """Test that default source is 'symspell'."""
        data = SuggestionData(term="test", edit_distance=1, frequency=100)
        assert data.source == "symspell"

    def test_default_confidence_is_one(self):
        """Test that default confidence is 1.0."""
        data = SuggestionData(term="test", edit_distance=1, frequency=100)
        assert data.confidence == 1.0

    def test_custom_source(self):
        """Test setting custom source."""
        data = SuggestionData(
            term="test",
            edit_distance=1,
            frequency=100,
            source="particle_typo",
            confidence=0.95,
        )
        assert data.source == "particle_typo"
        assert data.confidence == 0.95


class TestUnifiedRankerWeights:
    """Tests for UnifiedRanker source weights."""

    def test_default_weights_defined(self):
        """Test that default weights are defined for all sources in RankerConfig."""
        from myspellchecker.core.config import RankerConfig

        config = RankerConfig()
        weights = config.get_source_weights()
        assert "particle_typo" in weights
        assert "medial_confusion" in weights
        assert "semantic" in weights
        assert "symspell" in weights
        assert "morphology" in weights

    def test_particle_typo_highest_weight(self):
        """Test that particle_typo has highest weight."""
        from myspellchecker.core.config import RankerConfig

        config = RankerConfig()
        weights = config.get_source_weights()
        assert weights["particle_typo"] >= weights["symspell"]
        assert weights["particle_typo"] >= weights["morphology"]

    def test_get_source_weight(self):
        """Test get_source_weight method."""
        ranker = UnifiedRanker()
        assert ranker.get_source_weight("particle_typo") == 1.2
        assert ranker.get_source_weight("symspell") == 1.0
        assert ranker.get_source_weight("unknown") == 1.0  # Default fallback

    def test_custom_weights(self):
        """Test custom source weights."""
        custom_weights = {"particle_typo": 1.5, "symspell": 0.8}
        ranker = UnifiedRanker(source_weights=custom_weights)
        assert ranker.get_source_weight("particle_typo") == 1.5
        assert ranker.get_source_weight("symspell") == 0.8


class TestUnifiedRankerScoring:
    """Tests for UnifiedRanker scoring logic."""

    def test_ranker_name(self):
        """Test that ranker name is 'unified'."""
        ranker = UnifiedRanker()
        assert ranker.name == "unified"

    def test_higher_source_weight_reduces_score(self):
        """Test that higher source weight results in lower (better) score."""
        ranker = UnifiedRanker()

        # Same base data, different sources
        particle_data = SuggestionData(
            term="တယ်",
            edit_distance=1,
            frequency=1000,
            source="particle_typo",
            confidence=0.95,
        )
        symspell_data = SuggestionData(
            term="တယ်",
            edit_distance=1,
            frequency=1000,
            source="symspell",
            confidence=0.95,
        )

        particle_score = ranker.score(particle_data)
        symspell_score = ranker.score(symspell_data)

        # particle_typo has higher weight (1.2 vs 1.0), so score should be lower
        assert particle_score < symspell_score

    def test_higher_confidence_reduces_score(self):
        """Test that higher confidence results in lower (better) score."""
        ranker = UnifiedRanker()

        high_conf = SuggestionData(
            term="test",
            edit_distance=1,
            frequency=1000,
            source="symspell",
            confidence=0.95,
        )
        low_conf = SuggestionData(
            term="test",
            edit_distance=1,
            frequency=1000,
            source="symspell",
            confidence=0.5,
        )

        high_score = ranker.score(high_conf)
        low_score = ranker.score(low_conf)

        assert high_score < low_score

    def test_minimum_confidence_floor(self):
        """Test that very low confidence is floored to avoid division by zero."""
        ranker = UnifiedRanker()

        zero_conf = SuggestionData(
            term="test",
            edit_distance=1,
            frequency=1000,
            source="symspell",
            confidence=0.0,
        )

        # Should not raise ZeroDivisionError
        score = ranker.score(zero_conf)
        assert score > 0

    def test_context_source_uses_context_strategy_weight(self):
        """Context source should use context_strategy_score_weight override."""
        from myspellchecker.core.config import RankerConfig

        config = RankerConfig(
            strategy_score_weight=0.1,
            context_strategy_score_weight=0.9,
            strategy_score_cap=10.0,
        )
        ranker = UnifiedRanker(
            ranker_config=config,
            source_weights={"context": 1.0, "symspell": 1.0},
        )

        context_suggestion = SuggestionData(
            term="context_option",
            edit_distance=1,
            frequency=1000,
            source="context",
            confidence=0.8,
            strategy_score=8.0,
        )
        non_context_suggestion = SuggestionData(
            term="non_context_option",
            edit_distance=1,
            frequency=1000,
            source="symspell",
            confidence=0.8,
            strategy_score=8.0,
        )

        # Higher context-specific strategy weight should move score toward
        # normalized strategy score (higher/worse for this case).
        assert ranker.score(context_suggestion) > ranker.score(non_context_suggestion)

    def test_strategy_score_normalization_cap(self):
        """Strategy scores above cap should saturate to the same normalized value."""
        from myspellchecker.core.config import RankerConfig

        config = RankerConfig(
            strategy_score_weight=1.0,
            context_strategy_score_weight=1.0,
            strategy_score_cap=10.0,
        )
        ranker = UnifiedRanker(
            ranker_config=config,
            source_weights={"symspell": 1.0},
        )

        score_at_cap = ranker.score(
            SuggestionData(
                term="cap",
                edit_distance=1,
                frequency=100,
                source="symspell",
                strategy_score=10.0,
            )
        )
        score_above_cap = ranker.score(
            SuggestionData(
                term="above_cap",
                edit_distance=1,
                frequency=100,
                source="symspell",
                strategy_score=25.0,
            )
        )
        score_below_cap = ranker.score(
            SuggestionData(
                term="below_cap",
                edit_distance=1,
                frequency=100,
                source="symspell",
                strategy_score=5.0,
            )
        )

        assert score_above_cap == pytest.approx(score_at_cap)
        assert score_below_cap < score_at_cap


class TestUnifiedRankerDeduplication:
    """Tests for UnifiedRanker deduplication logic."""

    def test_deduplicate_keeps_higher_weight_source(self):
        """Test that deduplication keeps suggestion from higher-weight source."""
        ranker = UnifiedRanker()

        suggestions = [
            SuggestionData(
                term="တယ်",
                edit_distance=1,
                frequency=1000,
                source="symspell",
                confidence=0.9,
            ),
            SuggestionData(
                term="တယ်",
                edit_distance=1,
                frequency=1000,
                source="particle_typo",
                confidence=0.9,
            ),
        ]

        result = ranker._deduplicate(suggestions)

        assert len(result) == 1
        assert result[0].source == "particle_typo"

    def test_deduplicate_considers_confidence(self):
        """Test that deduplication considers confidence in tie-breaking."""
        ranker = UnifiedRanker()

        suggestions = [
            SuggestionData(
                term="တယ်",
                edit_distance=1,
                frequency=1000,
                source="symspell",
                confidence=0.95,  # Higher confidence
            ),
            SuggestionData(
                term="တယ်",
                edit_distance=1,
                frequency=1000,
                source="morphology",  # Lower weight (0.9)
                confidence=0.5,
            ),
        ]

        result = ranker._deduplicate(suggestions)

        assert len(result) == 1
        # symspell (1.0 * 0.95 = 0.95) > morphology (0.9 * 0.5 = 0.45)
        assert result[0].source == "symspell"

    def test_deduplicate_preserves_unique_terms(self):
        """Test that deduplication preserves all unique terms."""
        ranker = UnifiedRanker()

        suggestions = [
            SuggestionData(term="တယ်", edit_distance=1, frequency=1000),
            SuggestionData(term="သည်", edit_distance=1, frequency=800),
            SuggestionData(term="ပါတယ်", edit_distance=2, frequency=600),
        ]

        result = ranker._deduplicate(suggestions)

        assert len(result) == 3
        terms = {s.term for s in result}
        assert terms == {"တယ်", "သည်", "ပါတယ်"}


class TestUnifiedRankerRankSuggestions:
    """Tests for the full rank_suggestions workflow."""

    def test_rank_suggestions_empty_list(self):
        """Test that empty input returns empty list."""
        ranker = UnifiedRanker()
        result = ranker.rank_suggestions([])
        assert result == []

    def test_rank_suggestions_sorts_by_score(self):
        """Test that suggestions are sorted by score (lower first)."""
        ranker = UnifiedRanker()

        suggestions = [
            SuggestionData(
                term="high_edit",
                edit_distance=3,
                frequency=100,
            ),
            SuggestionData(
                term="low_edit",
                edit_distance=1,
                frequency=100,
            ),
        ]

        result = ranker.rank_suggestions(suggestions, deduplicate=False)

        assert len(result) == 2
        assert result[0].term == "low_edit"  # Lower edit distance = better score

    def test_rank_suggestions_with_deduplication(self):
        """Test rank_suggestions with deduplication enabled."""
        ranker = UnifiedRanker()

        suggestions = [
            SuggestionData(
                term="တယ်",
                edit_distance=1,
                frequency=1000,
                source="symspell",
            ),
            SuggestionData(
                term="တယ်",
                edit_distance=1,
                frequency=1000,
                source="particle_typo",
                confidence=0.95,
            ),
            SuggestionData(
                term="သည်",
                edit_distance=2,
                frequency=500,
            ),
        ]

        result = ranker.rank_suggestions(suggestions, deduplicate=True)

        assert len(result) == 2
        # particle_typo version of "တယ်" should be kept
        terms = [s.term for s in result]
        assert "တယ်" in terms
        assert "သည်" in terms

    def test_rank_suggestions_prioritizes_particle_typo(self):
        """Test that particle_typo suggestions rank higher."""
        ranker = UnifiedRanker()

        suggestions = [
            SuggestionData(
                term="symspell_term",
                edit_distance=1,
                frequency=1000,
                source="symspell",
            ),
            SuggestionData(
                term="particle_term",
                edit_distance=1,
                frequency=1000,
                source="particle_typo",
                confidence=0.95,
            ),
        ]

        result = ranker.rank_suggestions(suggestions, deduplicate=False)

        assert len(result) == 2
        assert result[0].term == "particle_term"  # Higher source weight


class TestUnifiedRankerIntegration:
    """Integration tests for UnifiedRanker with various scenarios."""

    def test_mixed_source_ranking(self):
        """Test ranking with suggestions from multiple sources."""
        ranker = UnifiedRanker()

        suggestions = [
            # SymSpell suggestion
            SuggestionData(
                term="option1",
                edit_distance=1,
                frequency=1000,
                source="symspell",
                confidence=0.8,
            ),
            # Particle typo (high weight, high confidence)
            SuggestionData(
                term="option2",
                edit_distance=1,
                frequency=500,
                source="particle_typo",
                confidence=0.95,
            ),
            # Morphology (lower weight)
            SuggestionData(
                term="option3",
                edit_distance=1,
                frequency=2000,
                source="morphology",
                confidence=0.7,
            ),
        ]

        result = ranker.rank_suggestions(suggestions, deduplicate=False)

        assert len(result) == 3
        # particle_typo should rank first due to high weight and confidence
        assert result[0].source == "particle_typo"

    def test_semantic_source_ranking(self):
        """Test that semantic source is properly weighted."""
        ranker = UnifiedRanker()

        suggestions = [
            SuggestionData(
                term="symspell_term",
                edit_distance=1,
                frequency=1000,
                source="symspell",
                confidence=0.9,
            ),
            SuggestionData(
                term="semantic_term",
                edit_distance=1,
                frequency=1000,
                source="semantic",
                confidence=0.9,
            ),
        ]

        result = ranker.rank_suggestions(suggestions, deduplicate=False)

        # semantic (weight=1.15) > symspell (weight=1.0)
        assert result[0].source == "semantic"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
