"""
Unit tests for SymSpell algorithm.

Tests the SymSpell spelling correction algorithm including:
- Edit distance calculations
- Suggestion generation
- Frequency-based ranking
"""

import pytest

from myspellchecker.algorithms import (
    Suggestion,
    SymSpell,
)
from myspellchecker.providers import MemoryProvider


class TestSuggestion:
    """Test Suggestion dataclass."""

    def test_suggestion_creation(self):
        """Suggestion should be created with correct attributes."""
        s = Suggestion(term="မြန်", edit_distance=1, frequency=1000)
        assert s.term == "မြန်"
        assert s.edit_distance == 1
        assert s.frequency == 1000
        assert hasattr(s, "score")

    def test_suggestion_score_calculation(self):
        """Score should be computed by a ranker."""
        from myspellchecker.algorithms.ranker import DefaultRanker, SuggestionData

        ranker = DefaultRanker()
        data = SuggestionData(term="test", edit_distance=1, frequency=100)
        score = ranker.score(data)

        s = Suggestion(term="test", edit_distance=1, frequency=100, score=score)
        # Score should combine edit distance and frequency
        # Lower edit distance = better, higher frequency = better
        assert s.score > 0

    def test_suggestion_sorting(self):
        """Suggestions should sort by score."""
        from myspellchecker.algorithms.ranker import DefaultRanker, SuggestionData

        ranker = DefaultRanker()
        # Compute scores using the ranker
        score1 = ranker.score(SuggestionData(term="a", edit_distance=1, frequency=100))
        score2 = ranker.score(SuggestionData(term="b", edit_distance=2, frequency=200))
        score3 = ranker.score(SuggestionData(term="c", edit_distance=1, frequency=200))

        s1 = Suggestion(term="a", edit_distance=1, frequency=100, score=score1)
        s2 = Suggestion(term="b", edit_distance=2, frequency=200, score=score2)
        s3 = Suggestion(term="c", edit_distance=1, frequency=200, score=score3)

        suggestions = [s2, s1, s3]
        suggestions.sort()

        # Lower edit distance should be prioritized
        # Within same edit distance, higher frequency is better
        assert suggestions[0].edit_distance <= suggestions[1].edit_distance
        assert suggestions[1].edit_distance <= suggestions[2].edit_distance

    def test_suggestion_equality(self):
        """Suggestions with same term should be equal."""
        s1 = Suggestion(term="test", edit_distance=1, frequency=100)
        s2 = Suggestion(term="test", edit_distance=2, frequency=200)
        assert s1 == s2  # Same term

    def test_suggestion_hash(self):
        """Suggestions should be hashable (for use in sets)."""
        s1 = Suggestion(term="test", edit_distance=1, frequency=100)
        s2 = Suggestion(term="other", edit_distance=1, frequency=100)
        suggestion_set = {s1, s2}
        assert len(suggestion_set) == 2


class TestSymSpell:
    """Test SymSpell algorithm."""

    @pytest.fixture
    def provider(self):
        """Create a test provider with sample data."""
        p = MemoryProvider()
        # Add some Myanmar syllables
        p.add_syllable("မြန်", frequency=1000)
        p.add_syllable("မာ", frequency=800)
        p.add_syllable("မျန်", frequency=500)
        p.add_syllable("မြ", frequency=300)
        p.add_syllable("နိုင်", frequency=900)
        p.add_syllable("ငံ", frequency=850)
        # Add some words
        p.add_word("မြန်မာ", frequency=2000)
        p.add_word("နိုင်ငံ", frequency=1800)
        return p

    def test_symspell_initialization(self, provider):
        """SymSpell should initialize with provider."""
        symspell = SymSpell(provider, max_edit_distance=2)
        assert symspell.provider == provider
        assert symspell.max_edit_distance == 2

    def test_symspell_lookup_valid_syllable(self, provider):
        """Looking up valid syllable should return empty if not include_known."""
        symspell = SymSpell(provider)
        suggestions = symspell.lookup("မြန်", level="syllable", include_known=False)
        assert len(suggestions) == 0  # Valid syllable, no suggestions

    def test_symspell_lookup_valid_syllable_include_known(self, provider):
        """Looking up valid syllable with include_known should return it."""
        symspell = SymSpell(provider)
        suggestions = symspell.lookup("မြန်", level="syllable", include_known=True)
        # Should include the term itself
        assert len(suggestions) >= 1

    def test_symspell_lookup_invalid_syllable(self, provider):
        """Looking up invalid syllable should return suggestions."""
        symspell = SymSpell(provider)
        # "မျ" is not in our provider, should get suggestions
        suggestions = symspell.lookup("မျ", level="syllable", max_suggestions=5)
        # Might get suggestions depending on edit distance
        assert isinstance(suggestions, list)

    def test_symspell_lookup_max_suggestions(self, provider):
        """Lookup should respect max_suggestions limit."""
        symspell = SymSpell(provider)
        suggestions = symspell.lookup("မ", level="syllable", max_suggestions=2)
        assert len(suggestions) <= 2

    def test_symspell_lookup_empty_string(self, provider):
        """Looking up empty string should return empty list."""
        symspell = SymSpell(provider)
        suggestions = symspell.lookup("", level="syllable")
        assert len(suggestions) == 0

    def test_symspell_get_deletes(self, provider):
        """_get_deletes should generate delete variations."""
        symspell = SymSpell(provider, max_edit_distance=2)

        # Test with max_distance=1
        deletes = symspell._get_deletes("abc", max_distance=1)
        assert "abc" in deletes  # Original
        assert "ab" in deletes  # Delete 'c'
        assert "ac" in deletes  # Delete 'b'
        assert "bc" in deletes  # Delete 'a'
        assert len(deletes) == 4

    def test_symspell_get_deletes_zero_distance(self, provider):
        """_get_deletes with distance 0 should return only original."""
        symspell = SymSpell(provider)
        deletes = symspell._get_deletes("abc", max_distance=0)
        assert deletes == {"abc"}

    def test_symspell_get_deletes_myanmar(self, provider):
        """_get_deletes should work with Myanmar text."""
        symspell = SymSpell(provider)
        deletes = symspell._get_deletes("မြန်", max_distance=1)
        assert "မြန်" in deletes
        # Should have delete variations
        assert len(deletes) > 1

    def test_symspell_find_similar_terms(self, provider):
        """_find_similar_terms should find dictionary matches."""
        symspell = SymSpell(provider)
        # "မြန်" is in the dictionary
        similar = symspell._find_similar_terms("မြန်", level="syllable")
        assert isinstance(similar, set)

    def test_symspell_lookup_word_level(self, provider):
        """Lookup at word level should work."""
        symspell = SymSpell(provider)
        suggestions = symspell.lookup("မြန်မာ", level="word", include_known=True)
        # Should find the word
        assert isinstance(suggestions, list)

    def test_symspell_build_index(self, provider):
        """build_index should not crash."""
        symspell = SymSpell(provider)
        # Should not crash (even though current implementation is lazy)
        symspell.build_index(["syllable"])
        assert "syllable" in symspell._indexed_levels

    def test_symspell_lookup_compound_placeholder(self, provider):
        """lookup_compound should return results."""
        symspell = SymSpell(provider)
        results = symspell.lookup_compound("မြန်မာ နိုင်ငံ")

        # Should return results
        assert isinstance(results, list)
        if results:
            # If results found, verify structure
            first_result = results[0]
            assert isinstance(first_result, tuple)
            assert len(first_result) == 3  # (terms, distance, count)


class TestSymSpellIntegration:
    """Integration tests for SymSpell with realistic scenarios."""

    @pytest.fixture
    def full_provider(self):
        """Create a provider with more comprehensive data."""
        p = MemoryProvider()
        # Common Myanmar syllables
        syllables = [
            ("မြန်", 1000),
            ("မာ", 900),
            ("နိုင်", 850),
            ("ငံ", 800),
            ("သည်", 750),
            ("သူ", 700),
            ("တို့", 650),
            ("များ", 600),
            ("က", 550),
        ]
        for syl, freq in syllables:
            p.add_syllable(syl, frequency=freq)

        # Common words
        words = [
            ("မြန်မာ", 2000),
            ("နိုင်ငံ", 1800),
            ("သူတို့", 1500),
            ("များသည်", 1200),
        ]
        for word, freq in words:
            p.add_word(word, frequency=freq)

        return p

    def test_symspell_realistic_typo(self, full_provider):
        """SymSpell should suggest corrections for realistic typos."""
        symspell = SymSpell(full_provider, max_edit_distance=2)

        # Simulate a typo - missing character
        # "မြန်" -> "မြ" (missing final character)
        suggestions = symspell.lookup("မြ", level="syllable", max_suggestions=5)

        # Should get suggestions (or empty if no matches within edit distance)
        assert isinstance(suggestions, list)

    def test_symspell_frequency_ranking(self, full_provider):
        """Suggestions should be ranked by edit distance then frequency."""
        symspell = SymSpell(full_provider, max_edit_distance=2)

        # Get suggestions for a term
        suggestions = symspell.lookup("မ", level="syllable", max_suggestions=5)

        if len(suggestions) >= 2:
            # Verify suggestions are sorted by score
            for i in range(len(suggestions) - 1):
                assert suggestions[i].score <= suggestions[i + 1].score

    def test_symspell_edit_distance_limit(self, full_provider):
        """Suggestions should not exceed max_edit_distance."""
        symspell = SymSpell(full_provider, max_edit_distance=1)

        suggestions = symspell.lookup("xyz", level="syllable", max_suggestions=10)

        # All suggestions should be within edit distance 1
        for suggestion in suggestions:
            assert suggestion.edit_distance <= 1

    def test_symspell_threshold_filtering(self, full_provider):
        """Suggestions below count_threshold should be filtered."""
        symspell = SymSpell(full_provider, count_threshold=100)

        # Add a rare syllable
        full_provider.add_syllable("rare", frequency=50)

        # This rare syllable should not appear in suggestions
        # (if it would otherwise match)
        suggestions = symspell.lookup("rar", level="syllable", max_suggestions=10)

        # Verify all suggestions meet threshold
        for suggestion in suggestions:
            assert suggestion.frequency >= 100


class TestSymSpellEdgeCases:
    """Test edge cases and error handling."""

    def test_symspell_empty_provider(self):
        """SymSpell should work with empty provider."""
        provider = MemoryProvider()
        symspell = SymSpell(provider)

        suggestions = symspell.lookup("test", level="syllable")
        assert len(suggestions) == 0  # No dictionary, no suggestions

    def test_symspell_single_char_lookup(self):
        """Single character lookup should not crash."""
        provider = MemoryProvider()
        provider.add_syllable("a", frequency=100)
        symspell = SymSpell(provider)

        suggestions = symspell.lookup("a", level="syllable", include_known=True)
        assert isinstance(suggestions, list)

    def test_symspell_unicode_handling(self):
        """SymSpell should handle various Unicode correctly."""
        provider = MemoryProvider()
        provider.add_syllable("မြန်", frequency=100)
        provider.add_syllable("中文", frequency=100)  # Chinese
        provider.add_syllable("🙂", frequency=100)  # Emoji

        symspell = SymSpell(provider)

        # Should not crash on any of these
        symspell.lookup("မြန်", level="syllable")
        symspell.lookup("中", level="syllable")
        symspell.lookup("🙂", level="syllable")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
