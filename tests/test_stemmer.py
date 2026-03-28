"""Tests for text/stemmer.py."""

import pytest


class TestStemmer:
    """Test Stemmer class."""

    def test_stem_empty(self):
        """Test stemming empty string."""
        from myspellchecker.text.stemmer import Stemmer

        stemmer = Stemmer()
        root, suffixes = stemmer.stem("")
        assert root == ""
        assert suffixes == []

    def test_stem_no_suffix(self):
        """Test stemming word with no recognizable suffix."""
        from myspellchecker.text.stemmer import Stemmer

        stemmer = Stemmer()
        root, suffixes = stemmer.stem("မြန်မာ")
        assert root == "မြန်မာ"
        assert suffixes == []

    def test_stem_simple_word(self):
        """Test stemming a simple word with suffix."""
        from myspellchecker.text.stemmer import Stemmer

        stemmer = Stemmer()
        root, suffixes = stemmer.stem("စားသည်")
        assert isinstance(root, str)
        assert isinstance(suffixes, list)

    def test_stem_multiple_suffixes(self):
        """Test stemming with multiple suffixes."""
        from myspellchecker.text.stemmer import Stemmer

        stemmer = Stemmer()
        root, suffixes = stemmer.stem("စားခဲ့သည်")
        assert isinstance(root, str)
        assert isinstance(suffixes, list)

    def test_stem_cached(self):
        """Test that stemming is cached and cache works correctly."""
        from myspellchecker.text.stemmer import Stemmer

        stemmer = Stemmer()
        stemmer.clear_cache()
        result1 = stemmer.stem("စားသည်")

        info = stemmer.cache_info()
        assert info["misses"] >= 1

        result2 = stemmer.stem("စားသည်")
        assert result1 == result2

        info = stemmer.cache_info()
        assert info["hits"] >= 1

    def test_clear_cache(self):
        """Test cache clearing."""
        from myspellchecker.text.stemmer import Stemmer

        stemmer = Stemmer()
        stemmer.stem("စားသည်")
        stemmer.stem("လာမည်")

        info = stemmer.cache_info()
        assert info["currsize"] > 0

        stemmer.clear_cache()

        info = stemmer.cache_info()
        assert info["currsize"] == 0

    def test_stem_with_custom_config(self):
        """Test Stemmer creation with custom StemmerConfig."""
        from myspellchecker.core.config.text_configs import StemmerConfig
        from myspellchecker.text.stemmer import Stemmer

        config = StemmerConfig(cache_size=500)
        stemmer = Stemmer(config=config)
        assert stemmer._cache_size == 500

    def test_stem_returns_list_not_tuple(self):
        """Test that stem() returns a list for suffixes, not tuple."""
        from myspellchecker.text.stemmer import Stemmer

        stemmer = Stemmer()
        root, suffixes = stemmer.stem("စားသည်")
        assert isinstance(suffixes, list), "suffixes should be a list for API compatibility"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
