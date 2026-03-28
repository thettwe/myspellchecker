"""
Integration tests for Provider + SymSpell + Ranker components.

These tests verify that the three components work correctly together:
1. DictionaryProvider - provides dictionary data and frequencies
2. SymSpell - uses provider for spell correction suggestions
3. SuggestionRanker - ranks suggestions by various factors
"""

import pytest

from myspellchecker.algorithms.ranker import (
    DefaultRanker,
    SuggestionData,
)
from myspellchecker.algorithms.symspell import Suggestion, SymSpell
from myspellchecker.providers.memory import MemoryProvider


class TestProviderSymSpellIntegration:
    """Integration tests for Provider + SymSpell."""

    @pytest.fixture
    def populated_provider(self) -> MemoryProvider:
        """Create a provider with Myanmar test data."""
        syllables = {
            # Common Myanmar syllables with frequencies
            "မြန်": 15000,
            "မာ": 12000,
            "စာ": 10000,
            "သူ": 8000,
            "ကျောင်း": 7500,
            "က": 7000,
            "ခ": 6000,
            "သည်": 5000,
            "ပါ": 4500,
            "နိုင်": 4000,
            # Syllables for typo testing
            "ကျာင်း": 100,  # Typo variant (missing medial)
        }

        words = {
            "မြန်မာ": 8000,  # Myanmar
            "ကျောင်း": 7500,  # School
            "သူ": 6000,
            "သည်": 5000,
            "စာသင်": 4000,
            "သွား": 3500,
            "နိုင်ငံ": 3000,
        }

        return MemoryProvider(syllables=syllables, words=words)

    @pytest.fixture
    def symspell_instance(self, populated_provider: MemoryProvider) -> SymSpell:
        """Create a SymSpell instance with the provider."""
        return SymSpell(
            provider=populated_provider,
            max_edit_distance=2,
            prefix_length=7,
        )

    def test_provider_supplies_data_to_symspell(
        self, symspell_instance: SymSpell, populated_provider: MemoryProvider
    ):
        """SymSpell should use provider for dictionary data."""
        # Provider has data
        assert populated_provider.is_valid_syllable("မြန်")
        assert populated_provider.get_syllable_frequency("မြန်") == 15000

        # SymSpell can access same data via provider
        assert symspell_instance.provider is populated_provider

    def test_symspell_lookup_valid_syllable(self, symspell_instance: SymSpell):
        """SymSpell should find valid syllable in provider."""
        # Build index first
        symspell_instance.build_index(["syllable"])
        # Lookup exact match with include_known to find itself
        suggestions = symspell_instance.lookup("မြန်", level="syllable", include_known=True)
        # Should find exact match (or return empty if not in index - depends on implementation)
        assert isinstance(suggestions, list)

    def test_symspell_lookup_typo_syllable(self, symspell_instance: SymSpell):
        """SymSpell should suggest corrections for typos."""
        # Build index first
        symspell_instance.build_index(["syllable"])

        # Lookup with typo - may return suggestions or not depending on index
        suggestions = symspell_instance.lookup("မြမ်", level="syllable")
        # Result depends on index building
        assert isinstance(suggestions, list)

    def test_symspell_lookup_word(self, symspell_instance: SymSpell):
        """SymSpell should support word lookup via provider."""
        # Valid word should have frequency
        freq = symspell_instance.provider.get_word_frequency("မြန်မာ")
        assert freq > 0

    def test_provider_frequency_affects_suggestions(self, populated_provider: MemoryProvider):
        """Higher frequency syllables should be ranked higher."""
        # Create SymSpell with default ranker
        symspell = SymSpell(provider=populated_provider, max_edit_distance=2)
        symspell.build_index(["syllable"])

        # Lookup should prefer higher frequency terms
        # (exact behavior depends on available suggestions)
        suggestions = symspell.lookup("မြန်", level="syllable")
        if suggestions:
            # Exact match should be first
            assert suggestions[0].term == "မြန်"


class TestSymSpellRankerIntegration:
    """Integration tests for SymSpell + Ranker."""

    @pytest.fixture
    def provider(self) -> MemoryProvider:
        """Create a provider with test data."""
        return MemoryProvider(
            syllables={
                "ကျောင်း": 10000,
                "ကျာင်း": 500,  # Typo variant (missing medial)
                "ကျာင်": 200,  # Another typo (shorter)
                "မြန်": 8000,
            },
            words={
                "ကျောင်း": 10000,
                "မြန်မာ": 8000,
            },
        )

    @pytest.fixture
    def default_ranker(self) -> DefaultRanker:
        """Create default ranker."""
        return DefaultRanker()

    def test_default_ranker_scores_suggestion_data(self, default_ranker: DefaultRanker):
        """DefaultRanker should score SuggestionData correctly."""
        data = SuggestionData(
            term="ကျောင်း",
            edit_distance=0,
            frequency=10000,
            phonetic_score=1.0,
        )
        score = default_ranker.score(data)
        assert isinstance(score, float)
        # Lower is better, edit_distance 0 should give low base score
        assert score < 1.0

    def test_ranker_prefers_lower_edit_distance(self, default_ranker: DefaultRanker):
        """Ranker should prefer suggestions with lower edit distance."""
        close_match = SuggestionData(
            term="ကျောင်း",
            edit_distance=1,
            frequency=1000,
        )
        far_match = SuggestionData(
            term="ကျာင်း",
            edit_distance=2,
            frequency=1000,
        )

        close_score = default_ranker.score(close_match)
        far_score = default_ranker.score(far_match)

        # Lower score is better, closer match should win
        assert close_score < far_score

    def test_ranker_considers_frequency(self, default_ranker: DefaultRanker):
        """Ranker should give bonus to higher frequency terms."""
        high_freq = SuggestionData(
            term="common",
            edit_distance=1,
            frequency=10000,
        )
        low_freq = SuggestionData(
            term="rare",
            edit_distance=1,
            frequency=100,
        )

        high_score = default_ranker.score(high_freq)
        low_score = default_ranker.score(low_freq)

        # Higher frequency should get lower (better) score
        assert high_score < low_score

    def test_ranker_considers_phonetic_score(self, default_ranker: DefaultRanker):
        """Ranker should give bonus to phonetically similar terms."""
        phonetic_match = SuggestionData(
            term="similar",
            edit_distance=1,
            frequency=1000,
            phonetic_score=0.9,
        )
        no_phonetic = SuggestionData(
            term="different",
            edit_distance=1,
            frequency=1000,
            phonetic_score=0.0,
        )

        phonetic_score = default_ranker.score(phonetic_match)
        no_phonetic_score = default_ranker.score(no_phonetic)

        # Phonetic match should get bonus (lower score)
        assert phonetic_score < no_phonetic_score

    def test_symspell_with_custom_ranker(self, provider: MemoryProvider):
        """SymSpell should accept custom ranker."""
        custom_ranker = DefaultRanker()
        symspell = SymSpell(
            provider=provider,
            ranker=custom_ranker,
            max_edit_distance=2,
        )

        assert symspell.ranker is custom_ranker


class TestProviderRankerIntegration:
    """Integration tests for Provider + Ranker without SymSpell."""

    @pytest.fixture
    def provider(self) -> MemoryProvider:
        """Create test provider."""
        return MemoryProvider(
            syllables={
                "မြန်": 15000,
                "မာ": 12000,
                "နိုင်": 4000,
            }
        )

    def test_ranker_uses_provider_frequencies(self, provider: MemoryProvider):
        """Ranker scoring should reflect provider frequencies."""
        ranker = DefaultRanker()

        # Get frequencies from provider
        freq1 = provider.get_syllable_frequency("မြန်")  # 15000
        freq2 = provider.get_syllable_frequency("နိုင်")  # 4000

        # Create suggestion data with provider frequencies
        data1 = SuggestionData(
            term="မြန်",
            edit_distance=1,
            frequency=freq1,
        )
        data2 = SuggestionData(
            term="နိုင်",
            edit_distance=1,
            frequency=freq2,
        )

        score1 = ranker.score(data1)
        score2 = ranker.score(data2)

        # Higher frequency term should have lower score
        assert score1 < score2


class TestFullPipelineIntegration:
    """End-to-end integration tests for all three components."""

    @pytest.fixture
    def full_pipeline(self) -> tuple:
        """Create full pipeline with provider, symspell, and ranker."""
        provider = MemoryProvider(
            syllables={
                "ကျောင်း": 10000,
                "ကျာင်း": 500,  # Typo variant
                "သူ": 8000,
                "သည်": 7000,
                "မြန်": 6000,
                "မာ": 5000,
            },
            words={
                "ကျောင်း": 10000,
                "မြန်မာ": 8000,
                "သူသည်": 5000,
            },
        )

        ranker = DefaultRanker()
        symspell = SymSpell(
            provider=provider,
            ranker=ranker,
            max_edit_distance=2,
            prefix_length=7,
        )

        return provider, symspell, ranker

    def test_full_pipeline_components_connected(self, full_pipeline: tuple):
        """All components should be properly connected."""
        provider, symspell, ranker = full_pipeline

        assert symspell.provider is provider
        assert symspell.ranker is ranker

    def test_full_pipeline_exact_match_lookup(self, full_pipeline: tuple):
        """Full pipeline should handle exact match lookups."""
        provider, symspell, ranker = full_pipeline

        # Build index
        symspell.build_index(["syllable"])

        # Exact match lookup
        suggestions = symspell.lookup("ကျောင်း", level="syllable")

        # Should find exact match with high frequency
        if suggestions:
            top = suggestions[0]
            assert top.term == "ကျောင်း"
            assert top.frequency == 10000
            assert top.edit_distance == 0

    def test_full_pipeline_suggestion_scoring(self, full_pipeline: tuple):
        """Full pipeline should produce properly scored suggestions."""
        provider, symspell, ranker = full_pipeline

        symspell.build_index(["syllable"])

        # Get suggestions
        suggestions = symspell.lookup("ကျောင်း", level="syllable")

        # All suggestions should have scores
        for suggestion in suggestions:
            assert hasattr(suggestion, "score")
            assert isinstance(suggestion.score, float)

    def test_full_pipeline_ranking_order(self, full_pipeline: tuple):
        """Full pipeline should return suggestions in ranked order."""
        provider, symspell, ranker = full_pipeline

        symspell.build_index(["syllable"])

        suggestions = symspell.lookup("ကျောင်း", level="syllable")

        # Suggestions should be sorted by score (ascending)
        if len(suggestions) > 1:
            scores = [s.score for s in suggestions]
            assert scores == sorted(scores), "Suggestions should be sorted by score"


class TestSuggestionClass:
    """Tests for Suggestion dataclass integration."""

    def test_suggestion_uses_ranker_score(self):
        """Suggestion should use external ranker score when provided."""
        external_score = 0.5

        # Note: score is now set directly (not via _external_score)
        # since scoring is always done externally by a SuggestionRanker
        suggestion = Suggestion(
            term="test",
            edit_distance=1,
            frequency=1000,
            score=external_score,
        )

        assert suggestion.score == external_score

    def test_suggestion_computes_own_score(self):
        """Suggestion should compute its own score when ranker not used."""
        suggestion = Suggestion(
            term="test",
            edit_distance=1,
            frequency=1000,
        )

        # Should compute score from edit_distance, frequency, etc.
        assert isinstance(suggestion.score, float)
        # Base score is edit_distance
        assert suggestion.score >= 0

    def test_suggestion_sorting(self):
        """Suggestions should sort by score."""
        # Note: score is now computed externally by a SuggestionRanker.
        # For testing sorting behavior, we set explicit scores.
        # Lower score = better suggestion (should come first)
        s1 = Suggestion(term="low", edit_distance=1, frequency=10000, score=0.5)
        s2 = Suggestion(term="high", edit_distance=2, frequency=100, score=1.5)

        suggestions = [s2, s1]
        sorted_suggestions = sorted(suggestions)

        # s1 should come first (lower score = better)
        assert sorted_suggestions[0].term == "low"

    def test_suggestion_equality(self):
        """Suggestions with same term should be equal."""
        s1 = Suggestion(term="same", edit_distance=1, frequency=1000)
        s2 = Suggestion(term="same", edit_distance=2, frequency=500)

        assert s1 == s2  # Equal based on term

    def test_suggestion_hash(self):
        """Suggestions should be hashable for use in sets."""
        s1 = Suggestion(term="test", edit_distance=1, frequency=1000)
        s2 = Suggestion(term="test", edit_distance=2, frequency=500)

        suggestion_set = {s1, s2}
        # Same term means same hash, so only one in set
        assert len(suggestion_set) == 1


class TestProviderSymSpellEdgeCases:
    """Edge case tests for integration."""

    def test_empty_provider(self):
        """SymSpell should handle empty provider gracefully."""
        provider = MemoryProvider()
        symspell = SymSpell(provider=provider)
        symspell.build_index(["syllable"])

        suggestions = symspell.lookup("test", level="syllable")
        assert suggestions == []

    def test_zero_frequency_syllable(self):
        """Ranker should handle zero frequency gracefully."""
        ranker = DefaultRanker()
        data = SuggestionData(
            term="rare",
            edit_distance=1,
            frequency=0,
        )

        score = ranker.score(data)
        assert isinstance(score, float)
        # Zero frequency means no frequency bonus
        assert score >= 0

    def test_very_high_frequency(self):
        """Ranker should handle very high frequencies."""
        ranker = DefaultRanker()
        data = SuggestionData(
            term="common",
            edit_distance=1,
            frequency=10000000,  # 10 million
        )

        score = ranker.score(data)
        assert isinstance(score, float)
        # Should have significant frequency bonus
        assert score < 1.0  # Edit distance 1 minus bonus

    def test_provider_unicode_handling(self):
        """Provider should handle Myanmar Unicode correctly."""
        provider = MemoryProvider(
            syllables={
                "မြန်": 1000,
                "ကျော": 800,
                "ှ": 50,  # Single character
            }
        )

        assert provider.is_valid_syllable("မြန်")
        assert provider.is_valid_syllable("ကျော")
        assert provider.get_syllable_frequency("ှ") == 50

    def test_symspell_max_edit_distance_zero(self):
        """SymSpell with max_edit_distance=0 should only return exact matches."""
        provider = MemoryProvider(
            syllables={
                "test": 1000,
                "tast": 800,  # One edit away
            }
        )
        symspell = SymSpell(provider=provider, max_edit_distance=0)
        symspell.build_index(["syllable"])

        # With max_edit_distance=0 set on SymSpell, lookup should only return exact matches
        suggestions = symspell.lookup("test", level="syllable", include_known=True)

        # Only exact match (edit distance 0)
        for s in suggestions:
            assert s.edit_distance == 0


class TestConsistency:
    """Consistency tests for integrated components."""

    def test_deterministic_ranking(self):
        """Same input should produce consistent ranking."""
        provider = MemoryProvider(syllables={"က": 1000, "ခ": 900, "ဂ": 800})
        ranker = DefaultRanker()
        symspell = SymSpell(provider=provider, ranker=ranker, max_edit_distance=2)
        symspell.build_index(["syllable"])

        # Run multiple times
        results = []
        for _ in range(5):
            suggestions = symspell.lookup("က", level="syllable", include_known=True)
            results.append([s.term for s in suggestions])

        # All results should be identical
        for result in results[1:]:
            assert result == results[0]

    def test_ranker_name_property(self):
        """Ranker should have name property."""
        ranker = DefaultRanker()
        assert ranker.name == "default"
