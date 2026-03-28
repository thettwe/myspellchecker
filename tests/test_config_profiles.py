"""Extended tests for config/profiles.py to boost coverage."""

import pytest

from myspellchecker.core.exceptions import InvalidConfigError


class TestIndividualProfiles:
    """Test individual profile functions directly."""

    def test_get_development_profile(self):
        """Test development profile function directly."""
        from myspellchecker.core.config.profiles import get_development_profile

        config = get_development_profile()
        assert config is not None
        assert config.max_edit_distance == 2
        assert config.use_context_checker is False
        assert config.pos_tagger.tagger_type == "rule_based"

    def test_get_production_profile(self):
        """Test production profile function directly."""
        from myspellchecker.core.config.profiles import get_production_profile

        config = get_production_profile()
        assert config is not None
        assert config.use_context_checker is True
        assert config.use_ner is True
        assert config.pos_tagger.tagger_type == "viterbi"

    def test_get_testing_profile(self):
        """Test testing profile function directly."""
        from myspellchecker.core.config.profiles import get_testing_profile

        config = get_testing_profile()
        assert config is not None
        assert config.validation.strict_validation is True
        assert config.pos_tagger.tagger_type == "rule_based"

    def test_get_fast_profile(self):
        """Test fast profile function directly."""
        from myspellchecker.core.config.profiles import get_fast_profile

        config = get_fast_profile()
        assert config is not None
        assert config.max_edit_distance == 1
        assert config.use_phonetic is False
        assert config.validation.use_zawgyi_detection is False

    def test_get_accurate_profile(self):
        """Test accurate profile function directly."""
        from myspellchecker.core.config.profiles import get_accurate_profile

        config = get_accurate_profile()
        assert config is not None
        assert config.max_edit_distance == 3
        assert config.max_suggestions == 10
        assert config.semantic.use_proactive_scanning is True


class TestGetProfileWithNames:
    """Test get_profile with various names."""

    def test_get_profile_unknown_raises_error(self):
        """Test get_profile raises ValueError for unknown profile."""
        from myspellchecker.core.config.profiles import get_profile

        with pytest.raises(InvalidConfigError, match="Unknown profile"):
            get_profile("nonexistent_profile")

    def test_get_profile_default(self):
        """Test get_profile with default (production)."""
        from myspellchecker.core.config.profiles import get_profile

        config = get_profile()
        assert config is not None
        # Production defaults
        assert config.use_context_checker is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
