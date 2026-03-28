"""
Tests for Joint Segmentation and POS Tagging.

Tests the unified Viterbi decoder that simultaneously optimizes
word boundaries and POS tags in a single pass.
"""

from unittest.mock import MagicMock

import pytest

from myspellchecker.algorithms.joint_segment_tagger import JointSegmentTagger
from myspellchecker.core.config import JointConfig, SpellCheckerConfig
from myspellchecker.core.spellchecker import SpellChecker
from myspellchecker.providers import MemoryProvider


def _create_mock_provider():
    """Create a mock provider with proper return types for POS methods."""
    provider = MagicMock()
    # Ensure POS probability methods return empty dicts, not MagicMock
    provider.get_pos_unigram_probabilities.return_value = {}
    provider.get_pos_bigram_probabilities.return_value = {}
    provider.get_pos_trigram_probabilities.return_value = {}
    # Ensure dictionary-related methods return proper types
    provider.get_all_syllables.return_value = []
    provider.get_all_words.return_value = []
    # Word lookups
    provider.get_word_pos.return_value = None
    provider.get_word_frequency.return_value = 0
    provider.get_bigram_probability.return_value = 0.0
    return provider


class TestJointSegmentTaggerInit:
    """Test JointSegmentTagger initialization."""

    def test_init_with_minimal_params(self):
        """Test initialization with minimal parameters."""
        provider = _create_mock_provider()
        tagger = JointSegmentTagger(
            provider=provider,
            pos_bigram_probs={},
            pos_trigram_probs={},
        )
        assert tagger is not None
        assert tagger.beam_width == 15  # Default
        assert tagger.max_word_length == 20  # Default

    def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        provider = _create_mock_provider()
        tagger = JointSegmentTagger(
            provider=provider,
            pos_bigram_probs={("N", "V"): 0.3},
            pos_trigram_probs={("N", "V", "P"): 0.1},
            pos_unigram_probs={"N": 0.4, "V": 0.3, "P": 0.3},
            beam_width=20,
            max_word_length=15,
            emission_weight=1.5,
            word_score_weight=0.8,
        )
        assert tagger.beam_width == 20
        assert tagger.max_word_length == 15
        assert tagger.emission_weight == 1.5
        assert tagger.word_score_weight == 0.8


class TestJointSegmentTaggerSegmentAndTag:
    """Test the segment_and_tag method."""

    def test_empty_text(self):
        """Test with empty text."""
        provider = _create_mock_provider()
        tagger = JointSegmentTagger(
            provider=provider,
            pos_bigram_probs={},
            pos_trigram_probs={},
        )
        words, tags = tagger.segment_and_tag("")
        assert words == []
        assert tags == []

    def test_whitespace_only(self):
        """Test with whitespace-only text."""
        provider = _create_mock_provider()
        tagger = JointSegmentTagger(
            provider=provider,
            pos_bigram_probs={},
            pos_trigram_probs={},
        )
        words, tags = tagger.segment_and_tag("   ")
        assert words == []
        assert tags == []

    def test_single_character(self):
        """Test with single character."""
        provider = _create_mock_provider()
        provider.get_word_pos.return_value = "N"
        provider.get_word_frequency.return_value = 100

        tagger = JointSegmentTagger(
            provider=provider,
            pos_bigram_probs={("<S>", "N"): 0.5},
            pos_trigram_probs={},
            pos_unigram_probs={"N": 0.5},
        )
        words, tags = tagger.segment_and_tag("က")
        assert len(words) == 1
        assert len(tags) == 1
        assert words[0] == "က"

    def test_known_word(self):
        """Test with a known word in the provider."""
        provider = _create_mock_provider()
        provider.get_word_pos.return_value = "N"
        provider.get_word_frequency.return_value = 1000

        tagger = JointSegmentTagger(
            provider=provider,
            pos_bigram_probs={("<S>", "N"): 0.5},
            pos_trigram_probs={},
            pos_unigram_probs={"N": 0.5},
        )
        # Single Myanmar word
        words, tags = tagger.segment_and_tag("မြန်မာ")
        assert len(words) >= 1
        assert len(tags) == len(words)


class TestJointSegmentTaggerScoring:
    """Test scoring functions."""

    def test_word_score_with_known_word(self):
        """Test word score for known word."""
        provider = _create_mock_provider()
        provider.get_bigram_probability.return_value = 0.1

        tagger = JointSegmentTagger(
            provider=provider,
            pos_bigram_probs={},
            pos_trigram_probs={},
        )
        # Should use bigram probability
        score = tagger._get_word_score("word", "prev")
        assert score < 0  # Log probability is negative

    def test_tag_transition_score_with_trigram(self):
        """Test tag transition with trigram available."""
        provider = _create_mock_provider()
        tagger = JointSegmentTagger(
            provider=provider,
            pos_bigram_probs={("N", "V"): 0.3},
            pos_trigram_probs={("<S>", "N", "V"): 0.2},
            pos_unigram_probs={"V": 0.3},
        )
        # Should prefer trigram
        score = tagger._get_tag_transition_score("V", "N", "<S>")
        assert score < 0  # Log probability is negative

    def test_emission_score_with_word_tag_probs(self):
        """Test emission score with word-tag probabilities."""
        provider = _create_mock_provider()
        tagger = JointSegmentTagger(
            provider=provider,
            pos_bigram_probs={},
            pos_trigram_probs={},
            word_tag_probs={"test": {"N": 0.8, "V": 0.2}},
        )
        # Known word-tag pair
        score_n = tagger._get_emission_score("test", "N")
        score_v = tagger._get_emission_score("test", "V")
        # N should have higher score (less negative) than V
        assert score_n > score_v


class TestJointSegmentTaggerTagResolution:
    """Test tag resolution for words."""

    def test_tags_from_provider(self):
        """Test getting tags from provider."""
        provider = _create_mock_provider()
        provider.get_word_pos.return_value = "N|V"

        tagger = JointSegmentTagger(
            provider=provider,
            pos_bigram_probs={},
            pos_trigram_probs={},
        )
        tags = tagger._get_valid_tags_for_word("test")
        assert "N" in tags
        assert "V" in tags

    def test_tags_from_word_tag_probs(self):
        """Test getting tags from word-tag probabilities."""
        provider = _create_mock_provider()

        tagger = JointSegmentTagger(
            provider=provider,
            pos_bigram_probs={},
            pos_trigram_probs={},
            word_tag_probs={"test": {"ADJ": 0.5, "ADV": 0.5}},
        )
        tags = tagger._get_valid_tags_for_word("test")
        assert "ADJ" in tags
        assert "ADV" in tags

    def test_fallback_to_unknown_tag(self):
        """Test fallback to unknown tag when no tags found."""
        provider = _create_mock_provider()

        tagger = JointSegmentTagger(
            provider=provider,
            pos_bigram_probs={},
            pos_trigram_probs={},
            use_morphology_fallback=False,
        )
        tags = tagger._get_valid_tags_for_word("unknown_word")
        assert tags == {"UNK"}


class TestJointConfigValidation:
    """Test JointConfig validation."""

    def test_default_config(self):
        """Test default JointConfig values."""
        config = JointConfig()
        assert config.enabled is False
        assert config.beam_width == 15
        assert config.max_word_length == 20

    def test_enabled_config(self):
        """Test enabled JointConfig."""
        config = JointConfig(enabled=True)
        assert config.enabled is True

    def test_invalid_beam_width(self):
        """Test invalid beam_width raises error."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            JointConfig(beam_width=0)

    def test_invalid_max_word_length(self):
        """Test invalid max_word_length raises error."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            JointConfig(max_word_length=0)


class TestSpellCheckerJointIntegration:
    """Test SpellChecker integration with joint mode."""

    def test_spellchecker_stores_joint_tagger(self):
        """Test that SpellChecker stores joint_segment_tagger."""
        provider = MemoryProvider()
        provider.add_syllable("သူ", 100)
        provider.add_word("သူ", 100)

        config = SpellCheckerConfig(
            use_context_checker=False,
            use_phonetic=False,
            joint=JointConfig(enabled=True),
        )
        checker = SpellChecker(config=config, provider=provider)

        assert hasattr(checker, "joint_segment_tagger")
        assert checker.joint_segment_tagger is not None

    def test_spellchecker_joint_disabled_by_default(self):
        """Test that joint mode is disabled by default."""
        provider = MemoryProvider()
        provider.add_syllable("သူ", 100)
        provider.add_word("သူ", 100)

        config = SpellCheckerConfig(
            use_context_checker=False,
            use_phonetic=False,
        )
        checker = SpellChecker(config=config, provider=provider)

        assert checker.joint_segment_tagger is None

    def test_segment_and_tag_with_joint_enabled(self):
        """Test segment_and_tag uses joint model when enabled."""
        provider = MemoryProvider()
        provider.add_syllable("သူ", 100)
        provider.add_word("သူ", 100)

        config = SpellCheckerConfig(
            use_context_checker=False,
            use_phonetic=False,
            joint=JointConfig(enabled=True),
        )
        checker = SpellChecker(config=config, provider=provider)

        words, tags = checker.segment_and_tag("သူ")
        assert len(words) == len(tags)

    def test_segment_and_tag_with_joint_disabled(self):
        """Test segment_and_tag falls back to sequential when disabled."""
        provider = MemoryProvider()
        provider.add_syllable("သူ", 100)
        provider.add_word("သူ", 100)

        config = SpellCheckerConfig(
            use_context_checker=False,
            use_phonetic=False,
            joint=JointConfig(enabled=False),
        )
        checker = SpellChecker(config=config, provider=provider)

        words, tags = checker.segment_and_tag("သူ")
        assert len(words) == len(tags)

    def test_segment_and_tag_empty_text(self):
        """Test segment_and_tag with empty text."""
        provider = MemoryProvider()
        config = SpellCheckerConfig(
            use_context_checker=False,
            use_phonetic=False,
        )
        checker = SpellChecker(config=config, provider=provider)

        words, tags = checker.segment_and_tag("")
        assert words == []
        assert tags == []

    def test_segment_and_tag_whitespace(self):
        """Test segment_and_tag with whitespace."""
        provider = MemoryProvider()
        config = SpellCheckerConfig(
            use_context_checker=False,
            use_phonetic=False,
        )
        checker = SpellChecker(config=config, provider=provider)

        words, tags = checker.segment_and_tag("   ")
        assert words == []
        assert tags == []


class TestCacheManagement:
    """Test cache management."""

    def test_clear_cache(self):
        """Test cache clearing."""
        provider = _create_mock_provider()
        tagger = JointSegmentTagger(
            provider=provider,
            pos_bigram_probs={},
            pos_trigram_probs={},
        )
        # Add something to cache
        tagger._get_word_score("word1", "word2")
        assert len(tagger._word_score_cache) > 0

        # Clear cache
        tagger.clear_cache()
        assert len(tagger._word_score_cache) == 0
