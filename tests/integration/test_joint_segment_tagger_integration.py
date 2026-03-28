"""
Unit tests for JointSegmentTagger integration with SpellChecker.

Tests the joint.enabled config option and the integration of
segment_and_tag() using the unified Viterbi decoder.
"""

from unittest.mock import Mock

import pytest

from myspellchecker.core.config import JointConfig, SpellCheckerConfig


class TestJointConfigOptions:
    """Tests for JointConfig options."""

    def test_default_disabled(self):
        """Test that joint mode is disabled by default."""
        config = JointConfig()
        assert config.enabled is False

    def test_default_beam_width(self):
        """Test that default beam width is 15."""
        config = JointConfig()
        assert config.beam_width == 15

    def test_default_max_word_length(self):
        """Test that default max word length is 20."""
        config = JointConfig()
        assert config.max_word_length == 20

    def test_enable_joint_mode(self):
        """Test enabling joint mode."""
        config = JointConfig(enabled=True)
        assert config.enabled is True

    def test_custom_beam_width(self):
        """Test setting custom beam width."""
        config = JointConfig(beam_width=20)
        assert config.beam_width == 20

    def test_beam_width_validation(self):
        """Test that beam width must be >= 1."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            JointConfig(beam_width=0)

    def test_max_word_length_validation(self):
        """Test that max word length must be >= 1."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            JointConfig(max_word_length=0)

    def test_emission_weight_validation(self):
        """Test that emission weight must be > 0."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            JointConfig(emission_weight=0)

    def test_word_score_weight_validation(self):
        """Test that word score weight must be > 0."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            JointConfig(word_score_weight=0)

    def test_min_prob_validation(self):
        """Test that min prob must be > 0."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            JointConfig(min_prob=0)


class TestSpellCheckerJointConfig:
    """Tests for SpellChecker joint configuration."""

    def test_spellchecker_config_has_joint(self):
        """Test that SpellCheckerConfig has joint config."""
        config = SpellCheckerConfig()
        assert hasattr(config, "joint")
        assert isinstance(config.joint, JointConfig)

    def test_joint_config_defaults_from_spellchecker(self):
        """Test that joint config in SpellCheckerConfig has correct defaults."""
        config = SpellCheckerConfig()
        assert config.joint.enabled is False
        assert config.joint.beam_width == 15
        assert config.joint.max_word_length == 20


class TestJointSegmentTaggerClass:
    """Tests for JointSegmentTagger class."""

    def test_class_exists(self):
        """Test that JointSegmentTagger class exists."""
        from myspellchecker.algorithms import JointSegmentTagger

        assert JointSegmentTagger is not None

    def test_segment_and_tag_method_exists(self):
        """Test that segment_and_tag method exists."""
        from myspellchecker.algorithms import JointSegmentTagger

        assert hasattr(JointSegmentTagger, "segment_and_tag")

    def test_initialization_requires_provider(self):
        """Test that JointSegmentTagger requires a provider."""
        from myspellchecker.algorithms import JointSegmentTagger

        mock_provider = Mock()
        mock_provider.lookup.return_value = []

        # Should not raise with minimal args
        tagger = JointSegmentTagger(
            provider=mock_provider,
            pos_bigram_probs={},
            pos_trigram_probs={},
        )

        assert tagger is not None


class TestComponentFactoryJointTagger:
    """Tests for ComponentFactory joint tagger creation."""

    def test_create_joint_segment_tagger_when_disabled(self):
        """Test that None is returned when joint mode is disabled."""
        from myspellchecker.core.component_factory import ComponentFactory

        config = SpellCheckerConfig()
        config.joint.enabled = False

        factory = ComponentFactory(config)
        tagger = factory.create_joint_segment_tagger(
            provider=Mock(),
            bigram_probs={},
            trigram_probs={},
        )

        assert tagger is None

    def test_create_joint_segment_tagger_when_enabled(self):
        """Test that tagger is created when joint mode is enabled."""
        from myspellchecker.core.component_factory import ComponentFactory

        config = SpellCheckerConfig()
        config.joint.enabled = True

        mock_provider = Mock()
        mock_provider.lookup.return_value = []

        factory = ComponentFactory(config)
        tagger = factory.create_joint_segment_tagger(
            provider=mock_provider,
            bigram_probs={},
            trigram_probs={},
        )

        assert tagger is not None

    def test_joint_tagger_uses_config_settings(self):
        """Test that created tagger uses config settings."""
        from myspellchecker.core.component_factory import ComponentFactory

        config = SpellCheckerConfig()
        config.joint.enabled = True
        config.joint.beam_width = 20
        config.joint.max_word_length = 25
        config.joint.emission_weight = 1.5

        mock_provider = Mock()
        mock_provider.lookup.return_value = []

        factory = ComponentFactory(config)
        tagger = factory.create_joint_segment_tagger(
            provider=mock_provider,
            bigram_probs={},
            trigram_probs={},
        )

        assert tagger.beam_width == 20
        assert tagger.max_word_length == 25
        assert tagger.emission_weight == 1.5


class TestSpellCheckerJointTaggerIntegration:
    """Integration tests for JointSegmentTagger with SpellChecker."""

    @pytest.fixture
    def spellchecker(self):
        """Create a SpellChecker instance for integration testing."""
        try:
            from myspellchecker import SpellChecker

            return SpellChecker.create_default()
        except Exception as e:
            pytest.skip(f"SpellChecker not available: {e}")

    def test_spellchecker_has_joint_segment_tagger_attr(self, spellchecker):
        """Test that SpellChecker has joint_segment_tagger attribute."""
        assert hasattr(spellchecker, "joint_segment_tagger")

    def test_joint_segment_tagger_is_none_when_disabled(self, spellchecker):
        """Test that joint_segment_tagger is None when disabled."""
        # Default config has joint.enabled = False
        assert spellchecker.joint_segment_tagger is None

    def test_segment_and_tag_method_exists(self, spellchecker):
        """Test that segment_and_tag method exists in SpellChecker."""
        assert hasattr(spellchecker, "segment_and_tag")
        assert callable(spellchecker.segment_and_tag)

    def test_segment_and_tag_returns_tuple(self, spellchecker):
        """Test that segment_and_tag returns tuple of (words, tags)."""
        result = spellchecker.segment_and_tag("သူ စား တယ်")
        assert isinstance(result, tuple)
        assert len(result) == 2
        words, tags = result
        assert isinstance(words, list)
        assert isinstance(tags, list)

    def test_segment_and_tag_empty_input(self, spellchecker):
        """Test that segment_and_tag handles empty input."""
        words, tags = spellchecker.segment_and_tag("")
        assert words == []
        assert tags == []

    def test_segment_and_tag_fallback_to_sequential(self, spellchecker):
        """Test that segment_and_tag uses sequential pipeline when joint disabled."""
        # Since joint.enabled = False, should use segmenter + viterbi
        words, tags = spellchecker.segment_and_tag("သူ စား တယ်")

        # Should have processed the text
        assert len(words) > 0
        assert len(tags) == len(words)


class TestJointSegmentTaggerMethods:
    """Tests for JointSegmentTagger internal methods."""

    @pytest.fixture
    def mock_tagger(self):
        """Create a JointSegmentTagger with mocked dependencies."""
        from myspellchecker.algorithms import JointSegmentTagger

        mock_provider = Mock()
        mock_provider.lookup.return_value = []
        mock_provider.get_frequency.return_value = 0

        return JointSegmentTagger(
            provider=mock_provider,
            pos_bigram_probs={},
            pos_trigram_probs={},
        )

    def test_start_and_end_tags(self, mock_tagger):
        """Test that START_TAG and END_TAG are defined."""
        assert mock_tagger.START_TAG == "<S>"
        assert mock_tagger.END_TAG == "</S>"
        assert mock_tagger.UNKNOWN_TAG == "UNK"

    def test_default_parameters(self, mock_tagger):
        """Test default parameter values."""
        assert mock_tagger.beam_width == 15
        assert mock_tagger.max_word_length == 20
        assert mock_tagger.min_prob == 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
