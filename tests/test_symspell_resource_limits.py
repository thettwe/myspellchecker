"""Tests for SymSpell resource limits.

These tests verify that the SymSpell algorithm has proper safeguards
against resource exhaustion from exponential delete generation.
"""

import pytest

from myspellchecker.algorithms.symspell import SymSpell
from myspellchecker.providers.memory import MemoryProvider


class TestMaxEditDistanceValidation:
    """Tests for max_edit_distance parameter validation."""

    @pytest.fixture
    def provider(self):
        """Create a basic memory provider for testing."""
        return MemoryProvider()

    def test_negative_max_edit_distance_raises_error(self, provider):
        """Test that negative max_edit_distance raises ValueError."""
        with pytest.raises(ValueError, match="must be non-negative"):
            SymSpell(provider, max_edit_distance=-1)

    def test_zero_max_edit_distance_allowed(self, provider):
        """Test that zero max_edit_distance is allowed."""
        symspell = SymSpell(provider, max_edit_distance=0)
        assert symspell.max_edit_distance == 0

    def test_valid_max_edit_distance(self, provider):
        """Test that valid max_edit_distance values (1-3) work without warnings."""
        for distance in [1, 2, 3]:
            symspell = SymSpell(provider, max_edit_distance=distance)
            assert symspell.max_edit_distance == distance

    def test_high_max_edit_distance_warns(self, provider):
        """Test that max_edit_distance 4-5 triggers a warning."""
        with pytest.warns(RuntimeWarning, match="may cause high memory usage"):
            symspell = SymSpell(provider, max_edit_distance=4)
            assert symspell.max_edit_distance == 4

        with pytest.warns(RuntimeWarning, match="may cause high memory usage"):
            symspell = SymSpell(provider, max_edit_distance=5)
            assert symspell.max_edit_distance == 5

    def test_excessive_max_edit_distance_raises_error(self, provider):
        """Test that max_edit_distance > 5 raises ValueError."""
        with pytest.raises(ValueError, match="is too high"):
            SymSpell(provider, max_edit_distance=6)

        with pytest.raises(ValueError, match="is too high"):
            SymSpell(provider, max_edit_distance=10)


class TestMaxDeletesPerTermValidation:
    """Tests for max_deletes_per_term parameter validation."""

    @pytest.fixture
    def provider(self):
        """Create a basic memory provider for testing."""
        return MemoryProvider()

    def test_low_max_deletes_per_term_raises_error(self, provider):
        """Test that max_deletes_per_term < 100 raises ValueError."""
        with pytest.raises(ValueError, match="must be at least 100"):
            SymSpell(provider, max_deletes_per_term=50)

        with pytest.raises(ValueError, match="must be at least 100"):
            SymSpell(provider, max_deletes_per_term=99)

    def test_valid_max_deletes_per_term(self, provider):
        """Test that valid max_deletes_per_term values work."""
        symspell = SymSpell(provider, max_deletes_per_term=100)
        assert symspell._max_deletes_per_term == 100

        symspell = SymSpell(provider, max_deletes_per_term=5000)
        assert symspell._max_deletes_per_term == 5000

    def test_high_max_deletes_per_term_warns(self, provider):
        """Test that max_deletes_per_term > 100000 triggers a warning."""
        with pytest.warns(RuntimeWarning, match="is very high"):
            symspell = SymSpell(provider, max_deletes_per_term=100001)
            assert symspell._max_deletes_per_term == 100001

    def test_default_max_deletes_per_term(self, provider):
        """Test that default max_deletes_per_term is 5000."""
        symspell = SymSpell(provider)
        assert symspell._max_deletes_per_term == 5000


class TestDeleteGenerationLimits:
    """Tests for delete generation resource limits."""

    @pytest.fixture
    def provider(self):
        """Create a memory provider with test data."""
        provider = MemoryProvider()
        provider.add_syllable("test", frequency=100)
        provider.add_syllable("testing", frequency=80)
        return provider

    def test_delete_generation_respects_limit(self, provider):
        """Test that delete generation stops at the configured limit."""
        # Use a very low limit to force early termination
        symspell = SymSpell(provider, max_edit_distance=2, max_deletes_per_term=100)

        # A short term should generate deletes within the limit
        deletes = symspell._get_deletes("test", 2)
        assert len(deletes) <= 100

    def test_delete_generation_logs_warning_on_limit(self, provider, caplog):
        """Test that a warning is logged when the limit is exceeded."""
        import logging

        # Use a very low limit with a longer term to force limit
        symspell = SymSpell(provider, max_edit_distance=3, max_deletes_per_term=100)

        # A longer term with high edit distance will exceed limit
        long_term = "abcdefghij"  # 10 characters

        # Capture at the myspellchecker logger level since it doesn't propagate to root
        with caplog.at_level(logging.WARNING, logger="myspellchecker"):
            deletes = symspell._get_deletes(long_term, 3)

        # Should have logged a warning
        assert "Delete generation limit reached" in caplog.text
        assert len(deletes) == 100  # Should stop at exactly the limit

    def test_delete_generation_short_term_within_limit(self, provider):
        """Test that short terms don't trigger the limit."""
        symspell = SymSpell(provider, max_edit_distance=2, max_deletes_per_term=5000)

        # Short term should generate all deletes without hitting limit
        deletes = symspell._get_deletes("abc", 2)

        # For "abc" with distance 2:
        # Original: "abc"
        # Distance 1: "bc", "ac", "ab" (3)
        # Distance 2: "b", "c", "a" (3)
        # Total: 7
        assert len(deletes) == 7

    def test_delete_generation_empty_term(self, provider):
        """Test delete generation with empty term."""
        symspell = SymSpell(provider, max_deletes_per_term=100)
        deletes = symspell._get_deletes("", 2)
        assert deletes == {""}

    def test_delete_generation_zero_distance(self, provider):
        """Test delete generation with zero max distance."""
        symspell = SymSpell(provider, max_deletes_per_term=100)
        deletes = symspell._get_deletes("test", 0)
        assert deletes == {"test"}


class TestResourceLimitsIntegration:
    """Integration tests for resource limits with actual spell checking."""

    @pytest.fixture
    def provider_with_data(self):
        """Create a memory provider with Myanmar syllables."""
        provider = MemoryProvider()
        # Add some Myanmar syllables
        provider.add_syllable("မြန်", frequency=1000)
        provider.add_syllable("မာ", frequency=800)
        provider.add_syllable("စာ", frequency=600)
        provider.add_syllable("လုံး", frequency=500)
        return provider

    def test_build_index_with_limits(self, provider_with_data):
        """Test that index building respects resource limits."""
        symspell = SymSpell(
            provider_with_data,
            max_edit_distance=2,
            max_deletes_per_term=5000,
        )

        # Should build index without issues
        symspell.build_index(["syllable"])

        # Verify index was built
        assert symspell._indexed_levels == {"syllable"}

    def test_lookup_with_limits(self, provider_with_data):
        """Test that lookup works correctly with resource limits."""
        symspell = SymSpell(
            provider_with_data,
            max_edit_distance=2,
            max_deletes_per_term=5000,
        )
        symspell.build_index(["syllable"])

        # Should find suggestions
        suggestions = symspell.lookup("မြ", level="syllable")
        assert len(suggestions) > 0

    def test_low_limit_still_provides_suggestions(self, provider_with_data):
        """Test that even with low limits, basic functionality works."""
        symspell = SymSpell(
            provider_with_data,
            max_edit_distance=1,  # Lower distance helps with low limits
            max_deletes_per_term=200,
        )
        symspell.build_index(["syllable"])

        # Should still work for simple lookups
        suggestions = symspell.lookup("မာ", level="syllable", include_known=True)
        # At minimum, should find the exact match
        assert any(s.term == "မာ" for s in suggestions)


class TestSymSpellEdgeCases:
    """Tests for edge cases in resource limit handling."""

    @pytest.fixture
    def provider(self):
        """Create a basic memory provider."""
        return MemoryProvider()

    def test_unicode_term_with_limits(self, provider):
        """Test that Unicode terms work correctly with limits."""
        symspell = SymSpell(provider, max_deletes_per_term=5000)

        # Myanmar text
        myanmar_term = "မြန်မာ"  # 6 characters
        deletes = symspell._get_deletes(myanmar_term, 2)

        # Should generate deletes without errors
        assert myanmar_term in deletes
        assert len(deletes) > 1

    def test_prefix_length_interaction_with_limits(self, provider):
        """Test that prefix_length and limits work together."""
        symspell = SymSpell(
            provider,
            prefix_length=5,
            max_deletes_per_term=1000,
            max_edit_distance=2,
        )

        # Long term should be truncated by prefix_length
        long_term = "abcdefghijklmnop"  # 16 characters
        # With prefix_length=5, only "abcde" should be used for deletes
        deletes = symspell._get_deletes(long_term[:5], 2)

        # Should be within reasonable bounds
        assert len(deletes) <= 1000

    def test_combined_validation_errors(self, provider):
        """Test that multiple validation errors are reported appropriately."""
        # Should raise for max_edit_distance first (it's checked first)
        with pytest.raises(ValueError, match="is too high"):
            SymSpell(
                provider,
                max_edit_distance=10,
                max_deletes_per_term=50,  # Also invalid
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
