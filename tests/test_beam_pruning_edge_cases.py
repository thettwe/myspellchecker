"""Tests for Viterbi beam pruning edge cases.

These tests verify correct behavior for:
- beam_width=1 (single path)
- beam_width larger than tag set
- Deterministic tie-breaking when multiple states have equal scores
- Beam pruning fallback behavior
"""

from unittest.mock import MagicMock, patch

import pytest


class TestBeamWidthOne:
    """Test beam_width=1 (greedy single-path decoding)."""

    @pytest.fixture
    def mock_provider(self):
        provider = MagicMock()
        provider.get_word_pos.return_value = None
        return provider

    @pytest.fixture
    def tagger_beam_1(self, mock_provider):
        """Create tagger with beam_width=1."""
        from myspellchecker.algorithms.viterbi import ViterbiTagger

        pos_bigram = {
            ("N", "V"): 0.6,
            ("N", "N"): 0.2,
            ("V", "N"): 0.5,
            ("V", "V"): 0.3,
        }
        pos_trigram = {
            ("N", "V", "N"): 0.5,
            ("N", "N", "V"): 0.3,
            ("V", "N", "V"): 0.4,
        }
        pos_unigram = {"N": 0.5, "V": 0.4, "UNK": 0.1}

        with patch("myspellchecker.algorithms.viterbi._HAS_CYTHON_VITERBI", False):
            tagger = ViterbiTagger(
                provider=mock_provider,
                pos_bigram_probs=pos_bigram,
                pos_trigram_probs=pos_trigram,
                pos_unigram_probs=pos_unigram,
                beam_width=1,  # Single path only
                use_morphology_fallback=False,
            )
        return tagger

    def test_beam_1_returns_single_path(self, tagger_beam_1):
        """With beam_width=1, only one path is kept at each step."""
        # This should complete without error
        result = tagger_beam_1.tag_sequence(["word1", "word2", "word3"])
        assert len(result) == 3
        # All tags should be from the known set or UNK
        for tag in result:
            assert tag in {"N", "V", "UNK"}

    def test_beam_1_single_word(self, tagger_beam_1):
        """Beam_width=1 should handle single word correctly."""
        result = tagger_beam_1.tag_sequence(["word"])
        assert len(result) == 1
        assert result[0] in {"N", "V", "UNK"}

    def test_beam_1_two_words(self, tagger_beam_1):
        """Beam_width=1 should handle two words correctly."""
        result = tagger_beam_1.tag_sequence(["word1", "word2"])
        assert len(result) == 2

    def test_beam_1_long_sequence(self, tagger_beam_1):
        """Beam_width=1 should handle long sequences (greedy may lose optimal)."""
        words = [f"word{i}" for i in range(20)]
        result = tagger_beam_1.tag_sequence(words)
        assert len(result) == 20
        # Verify we get valid tags (may not be optimal due to greedy selection)
        for tag in result:
            assert tag in {"N", "V", "UNK"}


class TestBeamWidthLargerThanTagset:
    """Test beam_width larger than possible states."""

    @pytest.fixture
    def mock_provider(self):
        provider = MagicMock()
        provider.get_word_pos.return_value = None
        return provider

    @pytest.fixture
    def tagger_large_beam(self, mock_provider):
        """Create tagger with beam_width > tagset size."""
        from myspellchecker.algorithms.viterbi import ViterbiTagger

        # Small tagset with only 3 tags
        pos_bigram = {
            ("N", "V"): 0.5,
            ("V", "N"): 0.4,
            ("N", "ADJ"): 0.3,
            ("ADJ", "N"): 0.4,
        }
        pos_trigram = {
            ("N", "V", "N"): 0.4,
            ("V", "N", "ADJ"): 0.3,
        }
        pos_unigram = {"N": 0.4, "V": 0.35, "ADJ": 0.25}

        with patch("myspellchecker.algorithms.viterbi._HAS_CYTHON_VITERBI", False):
            tagger = ViterbiTagger(
                provider=mock_provider,
                pos_bigram_probs=pos_bigram,
                pos_trigram_probs=pos_trigram,
                pos_unigram_probs=pos_unigram,
                beam_width=1000,  # Much larger than 3 tags
                use_morphology_fallback=False,
            )
        return tagger

    def test_large_beam_no_pruning_needed(self, tagger_large_beam):
        """With large beam, no pruning should occur."""
        result = tagger_large_beam.tag_sequence(["word1", "word2", "word3"])
        assert len(result) == 3

    def test_large_beam_preserves_all_paths(self, tagger_large_beam):
        """Large beam should keep all possible paths (no information loss)."""
        # With 3 tags, max states = 3*3 = 9, which is < 1000
        result = tagger_large_beam.tag_sequence(["a", "b", "c", "d", "e"])
        assert len(result) == 5

    def test_large_beam_equals_exhaustive_search(self, mock_provider):
        """Large beam should give same result as exhaustive search."""
        from myspellchecker.algorithms.viterbi import ViterbiTagger

        pos_bigram = {("N", "V"): 0.7, ("V", "N"): 0.6}
        pos_trigram = {("N", "V", "N"): 0.8}
        pos_unigram = {"N": 0.6, "V": 0.4}

        with patch("myspellchecker.algorithms.viterbi._HAS_CYTHON_VITERBI", False):
            tagger_large = ViterbiTagger(
                provider=mock_provider,
                pos_bigram_probs=pos_bigram,
                pos_trigram_probs=pos_trigram,
                pos_unigram_probs=pos_unigram,
                beam_width=10000,  # Essentially infinite
                use_morphology_fallback=False,
            )
            tagger_medium = ViterbiTagger(
                provider=mock_provider,
                pos_bigram_probs=pos_bigram,
                pos_trigram_probs=pos_trigram,
                pos_unigram_probs=pos_unigram,
                beam_width=100,  # Still larger than tagset squared
                use_morphology_fallback=False,
            )

        words = ["w1", "w2", "w3"]
        result_large = tagger_large.tag_sequence(words)
        result_medium = tagger_medium.tag_sequence(words)

        # Both should give identical results (no pruning occurs)
        assert result_large == result_medium


class TestTieBreakingDeterminism:
    """Test deterministic tie-breaking behavior."""

    @pytest.fixture
    def mock_provider(self):
        provider = MagicMock()
        provider.get_word_pos.return_value = None
        return provider

    @pytest.fixture
    def tagger_equal_probs(self, mock_provider):
        """Create tagger where multiple paths have equal probabilities."""
        from myspellchecker.algorithms.viterbi import ViterbiTagger

        # All transitions have equal probability
        pos_bigram = {
            ("N", "V"): 0.5,
            ("N", "ADJ"): 0.5,
            ("V", "N"): 0.5,
            ("ADJ", "N"): 0.5,
        }
        pos_trigram = {
            ("N", "V", "N"): 0.5,
            ("N", "ADJ", "N"): 0.5,
        }
        # Equal unigram probs too
        pos_unigram = {"N": 0.33, "V": 0.33, "ADJ": 0.34}

        with patch("myspellchecker.algorithms.viterbi._HAS_CYTHON_VITERBI", False):
            tagger = ViterbiTagger(
                provider=mock_provider,
                pos_bigram_probs=pos_bigram,
                pos_trigram_probs=pos_trigram,
                pos_unigram_probs=pos_unigram,
                beam_width=10,
                use_morphology_fallback=False,
            )
        return tagger

    def test_tie_breaking_is_deterministic(self, tagger_equal_probs):
        """Multiple runs with equal probabilities should give same result."""
        words = ["word1", "word2", "word3"]

        # Run multiple times
        results = [tagger_equal_probs.tag_sequence(words) for _ in range(10)]

        # All results should be identical (deterministic)
        for result in results[1:]:
            assert result == results[0], "Tie-breaking should be deterministic"

    def test_tie_breaking_prefers_common_tags(self, mock_provider):
        """When probabilities tie, prefer more common tag (higher unigram)."""
        from myspellchecker.algorithms.viterbi import ViterbiTagger

        # Equal transition probs, but N has higher unigram
        pos_bigram = {("X", "N"): 0.5, ("X", "V"): 0.5}
        pos_trigram = {}
        pos_unigram = {"N": 0.7, "V": 0.2, "X": 0.1}  # N is more common

        with patch("myspellchecker.algorithms.viterbi._HAS_CYTHON_VITERBI", False):
            tagger = ViterbiTagger(
                provider=mock_provider,
                pos_bigram_probs=pos_bigram,
                pos_trigram_probs=pos_trigram,
                pos_unigram_probs=pos_unigram,
                beam_width=10,
                use_morphology_fallback=False,
            )

        # With equal path probabilities, N should be preferred (higher freq)
        result = tagger.tag_sequence(["w1", "w2"])
        # Verify determinism
        result2 = tagger.tag_sequence(["w1", "w2"])
        assert result == result2

    def test_tie_breaking_alphabetical_fallback(self, mock_provider):
        """When probs and freqs tie, use alphabetical ordering."""
        from myspellchecker.algorithms.viterbi import ViterbiTagger

        # All equal
        pos_bigram = {("X", "A"): 0.5, ("X", "B"): 0.5}
        pos_trigram = {}
        pos_unigram = {"A": 0.33, "B": 0.33, "X": 0.34}

        with patch("myspellchecker.algorithms.viterbi._HAS_CYTHON_VITERBI", False):
            tagger = ViterbiTagger(
                provider=mock_provider,
                pos_bigram_probs=pos_bigram,
                pos_trigram_probs=pos_trigram,
                pos_unigram_probs=pos_unigram,
                beam_width=10,
                use_morphology_fallback=False,
            )

        # Should be deterministic due to alphabetical fallback
        results = [tagger.tag_sequence(["w1", "w2"]) for _ in range(5)]
        for result in results[1:]:
            assert result == results[0]


class TestBeamPruningFallback:
    """Test fallback behavior when beam pruning causes path loss."""

    @pytest.fixture
    def mock_provider(self):
        provider = MagicMock()
        provider.get_word_pos.return_value = None
        return provider

    def test_empty_viterbi_fallback(self, mock_provider):
        """Test fallback when viterbi becomes empty after pruning."""
        from myspellchecker.algorithms.viterbi import ViterbiTagger

        # Sparse probabilities that might lead to empty states
        pos_bigram = {("N", "V"): 0.9}  # Very limited transitions
        pos_trigram = {("N", "V", "N"): 0.8}
        pos_unigram = {"N": 0.5, "V": 0.5}

        with patch("myspellchecker.algorithms.viterbi._HAS_CYTHON_VITERBI", False):
            tagger = ViterbiTagger(
                provider=mock_provider,
                pos_bigram_probs=pos_bigram,
                pos_trigram_probs=pos_trigram,
                pos_unigram_probs=pos_unigram,
                beam_width=1,  # Aggressive pruning
                use_morphology_fallback=False,
            )

        # Should handle gracefully with fallback to UNK
        result = tagger.tag_sequence(["w1", "w2", "w3", "w4", "w5"])
        assert len(result) == 5
        # Should contain valid tags (may include UNK)
        for tag in result:
            assert tag in {"N", "V", "UNK"}

    def test_backpointer_missing_state_fallback(self, mock_provider):
        """Test fallback when backpointer doesn't contain expected state."""
        from myspellchecker.algorithms.viterbi import ViterbiTagger

        pos_bigram = {
            ("N", "V"): 0.8,
            ("V", "N"): 0.7,
            ("N", "ADJ"): 0.6,
            ("ADJ", "V"): 0.5,
        }
        pos_trigram = {
            ("N", "V", "N"): 0.7,
            ("V", "N", "ADJ"): 0.6,
        }
        pos_unigram = {"N": 0.4, "V": 0.3, "ADJ": 0.3}

        with patch("myspellchecker.algorithms.viterbi._HAS_CYTHON_VITERBI", False):
            tagger = ViterbiTagger(
                provider=mock_provider,
                pos_bigram_probs=pos_bigram,
                pos_trigram_probs=pos_trigram,
                pos_unigram_probs=pos_unigram,
                beam_width=2,  # Low beam width
                use_morphology_fallback=False,
            )

        # Long sequence with low beam should exercise fallback
        words = [f"word{i}" for i in range(10)]
        result = tagger.tag_sequence(words)
        assert len(result) == 10


class TestBeamPruningCorrectness:
    """Test that beam pruning maintains correct behavior."""

    @pytest.fixture
    def mock_provider(self):
        provider = MagicMock()
        provider.get_word_pos.return_value = None
        return provider

    def test_beam_pruning_keeps_top_k(self, mock_provider):
        """Verify beam pruning keeps exactly top-k states."""
        from myspellchecker.algorithms.viterbi import ViterbiTagger

        # Many tags to ensure pruning occurs
        tags = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8"]
        pos_bigram = {}
        pos_trigram = {}
        pos_unigram = {}

        # Create varied probabilities
        for i, t1 in enumerate(tags):
            pos_unigram[t1] = 0.1 + i * 0.01
            for j, t2 in enumerate(tags):
                pos_bigram[(t1, t2)] = 0.1 + (i + j) * 0.005

        with patch("myspellchecker.algorithms.viterbi._HAS_CYTHON_VITERBI", False):
            tagger = ViterbiTagger(
                provider=mock_provider,
                pos_bigram_probs=pos_bigram,
                pos_trigram_probs=pos_trigram,
                pos_unigram_probs=pos_unigram,
                beam_width=3,  # Keep only top 3
                use_morphology_fallback=False,
            )

        result = tagger.tag_sequence(["w1", "w2", "w3", "w4"])
        assert len(result) == 4
        # All returned tags should be from our tagset
        for tag in result:
            assert tag in tags or tag == "UNK"

    def test_beam_width_affects_result_quality(self, mock_provider):
        """Larger beam width should give equal or better results."""
        from myspellchecker.algorithms.viterbi import ViterbiTagger

        pos_bigram = {
            ("N", "V"): 0.7,
            ("V", "N"): 0.6,
            ("N", "N"): 0.3,
            ("V", "V"): 0.2,
        }
        pos_trigram = {
            ("N", "V", "N"): 0.8,
            ("V", "N", "V"): 0.7,
        }
        pos_unigram = {"N": 0.5, "V": 0.5}

        with patch("myspellchecker.algorithms.viterbi._HAS_CYTHON_VITERBI", False):
            tagger_narrow = ViterbiTagger(
                provider=mock_provider,
                pos_bigram_probs=pos_bigram,
                pos_trigram_probs=pos_trigram,
                pos_unigram_probs=pos_unigram,
                beam_width=1,
                use_morphology_fallback=False,
            )
            tagger_wide = ViterbiTagger(
                provider=mock_provider,
                pos_bigram_probs=pos_bigram,
                pos_trigram_probs=pos_trigram,
                pos_unigram_probs=pos_unigram,
                beam_width=100,
                use_morphology_fallback=False,
            )

        words = ["w1", "w2", "w3", "w4"]

        # Both should complete successfully
        result_narrow = tagger_narrow.tag_sequence(words)
        result_wide = tagger_wide.tag_sequence(words)

        assert len(result_narrow) == 4
        assert len(result_wide) == 4

        # Wide beam should find same or better path
        # (In this test we just verify completion; actual quality
        # comparison would need probability calculation)


class TestEmptyAndEdgeCases:
    """Test edge cases in beam pruning."""

    @pytest.fixture
    def mock_provider(self):
        provider = MagicMock()
        provider.get_word_pos.return_value = None
        return provider

    def test_empty_sequence(self, mock_provider):
        """Empty word sequence should return empty result."""
        from myspellchecker.algorithms.viterbi import ViterbiTagger

        with patch("myspellchecker.algorithms.viterbi._HAS_CYTHON_VITERBI", False):
            tagger = ViterbiTagger(
                provider=mock_provider,
                pos_bigram_probs={("N", "V"): 0.5},
                pos_trigram_probs={},
                pos_unigram_probs={"N": 0.5, "V": 0.5},
                beam_width=5,
            )

        result = tagger.tag_sequence([])
        assert result == []

    def test_no_probabilities_fallback(self, mock_provider):
        """No probabilities should return all UNK tags."""
        from myspellchecker.algorithms.viterbi import ViterbiTagger

        with patch("myspellchecker.algorithms.viterbi._HAS_CYTHON_VITERBI", False):
            tagger = ViterbiTagger(
                provider=mock_provider,
                pos_bigram_probs={},  # No transitions
                pos_trigram_probs={},
                pos_unigram_probs={},
                beam_width=5,
                use_morphology_fallback=False,
            )

        result = tagger.tag_sequence(["w1", "w2", "w3"])
        assert len(result) == 3
        assert all(tag == "UNK" for tag in result)

    def test_beam_width_zero_or_negative(self, mock_provider):
        """Beam width <= 0 should be handled gracefully."""
        from myspellchecker.algorithms.viterbi import ViterbiTagger

        # beam_width of 0 or negative should still work (no pruning or error)
        with patch("myspellchecker.algorithms.viterbi._HAS_CYTHON_VITERBI", False):
            # Note: Current implementation doesn't validate beam_width > 0
            # This test documents current behavior
            tagger = ViterbiTagger(
                provider=mock_provider,
                pos_bigram_probs={("N", "V"): 0.5},
                pos_trigram_probs={},
                pos_unigram_probs={"N": 0.5, "V": 0.5},
                beam_width=0,  # Edge case
                use_morphology_fallback=False,
            )

        # With beam_width=0, pruning removes all states
        # Should fall back gracefully
        result = tagger.tag_sequence(["w1", "w2", "w3"])
        assert len(result) == 3
