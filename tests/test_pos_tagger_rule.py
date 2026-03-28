"""Tests for RuleBasedPOSTagger behavior."""

import pytest


class TestRuleBasedPOSTaggerWithConfidence:
    """Test tag_word_with_confidence method."""

    def test_tag_word_with_confidence_empty_word(self):
        """Test tagging empty word returns unknown with 0 confidence."""
        from myspellchecker.algorithms.pos_tagger_rule import RuleBasedPOSTagger

        tagger = RuleBasedPOSTagger()
        pred = tagger.tag_word_with_confidence("")

        assert pred.tag == tagger.unknown_tag
        assert pred.confidence == 0.0
        assert pred.metadata["source"] == "unknown"

    def test_tag_word_with_confidence_from_pos_map(self):
        """Test tagging word from POS map returns high confidence."""
        from myspellchecker.algorithms.pos_tagger_rule import RuleBasedPOSTagger

        pos_map = {"test": {"N", "V"}}
        tagger = RuleBasedPOSTagger(pos_map=pos_map)
        pred = tagger.tag_word_with_confidence("test")

        assert pred.tag in ["N", "V"]
        assert pred.confidence == 1.0
        assert pred.metadata["source"] == "pos_map"
        assert "all_tags" in pred.metadata

    def test_tag_word_with_confidence_unknown(self):
        """Test tagging completely unknown word."""
        from myspellchecker.algorithms.pos_tagger_rule import RuleBasedPOSTagger

        tagger = RuleBasedPOSTagger(use_morphology_fallback=False)
        pred = tagger.tag_word_with_confidence("xyz_unknown_word")

        assert pred.tag == tagger.unknown_tag
        assert pred.confidence == 0.0
        assert pred.metadata["source"] == "fallback"


class TestRuleBasedPOSTaggerCacheMethods:
    """Test cache-related methods."""

    def test_cache_populate_and_clear(self):
        """Test that cache populates and clears correctly."""
        from myspellchecker.algorithms.pos_tagger_rule import RuleBasedPOSTagger

        tagger = RuleBasedPOSTagger()
        tagger.tag_word("test")
        tagger.tag_word("test")  # Second call should be cached

        info = tagger.cache_info()
        assert len(info) == 4

        tagger.clear_cache()


class TestRuleBasedPOSTaggerProperties:
    """Test tagger properties."""

    def test_tagger_type(self):
        """Test tagger_type property."""
        from myspellchecker.algorithms.pos_tagger_base import TaggerType
        from myspellchecker.algorithms.pos_tagger_rule import RuleBasedPOSTagger

        tagger = RuleBasedPOSTagger()
        assert tagger.tagger_type == TaggerType.RULE_BASED


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
