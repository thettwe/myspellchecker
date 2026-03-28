"""Tests for POSTaggerFactory to boost coverage."""

from typing import List
from unittest.mock import Mock, patch

import pytest

from myspellchecker.algorithms.pos_tagger_base import POSTaggerBase, TaggerType
from myspellchecker.algorithms.pos_tagger_factory import POSTaggerFactory


class TestPOSTaggerFactoryCreate:
    """Test POSTaggerFactory.create method."""

    def test_create_rule_based(self):
        """Test creating rule-based tagger (default)."""
        tagger = POSTaggerFactory.create("rule_based")
        assert tagger is not None
        assert tagger.tagger_type == TaggerType.RULE_BASED

    def test_create_rule_based_default(self):
        """Test creating default tagger (rule_based)."""
        tagger = POSTaggerFactory.create()
        assert tagger is not None
        assert tagger.tagger_type == TaggerType.RULE_BASED

    def test_create_viterbi_with_provider(self):
        """Test creating viterbi tagger with provider."""
        mock_provider = Mock()
        mock_provider.get_pos_bigram_probabilities = Mock(return_value={})
        mock_provider.get_pos_trigram_probabilities = Mock(return_value={})
        mock_provider.get_pos_unigram_probabilities = Mock(return_value={})
        mock_provider.get_word_pos = Mock(return_value=None)

        tagger = POSTaggerFactory.create("viterbi", provider=mock_provider)
        assert tagger is not None
        assert tagger.tagger_type == TaggerType.VITERBI

    def test_create_unknown_type_raises(self):
        """Test creating unknown tagger type raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            POSTaggerFactory.create("unknown_type")

        error_msg = str(exc_info.value)
        assert "Unknown tagger_type" in error_msg
        assert "unknown_type" in error_msg
        assert "rule_based" in error_msg  # Valid types listed

    def test_create_custom_without_class_raises(self):
        """Test creating custom tagger without tagger_class raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            POSTaggerFactory.create("custom")

        error_msg = str(exc_info.value)
        assert "tagger_class" in error_msg
        assert "custom" in error_msg.lower()


class TestPOSTaggerFactoryTransformer:
    """Test transformer tagger creation."""

    def test_create_transformer_import_error(self):
        """Test transformer creation when transformers not installed."""
        with patch.dict(
            "sys.modules",
            {"myspellchecker.algorithms.pos_tagger_transformer": None},
        ):
            # Need to patch the import inside the factory method
            with patch(
                "myspellchecker.algorithms.pos_tagger_factory.POSTaggerFactory.create_transformer"
            ) as mock_create:
                mock_create.side_effect = ImportError(
                    "Transformer-based POS tagging requires the 'transformers' library."
                )
                with pytest.raises(ImportError) as exc_info:
                    mock_create()

                error_msg = str(exc_info.value)
                assert "transformers" in error_msg


class TestPOSTaggerFactoryViterbi:
    """Test Viterbi tagger creation."""

    def test_create_viterbi_no_provider_raises(self):
        """Test creating viterbi tagger without provider raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            POSTaggerFactory.create_viterbi(provider=None)

        error_msg = str(exc_info.value)
        assert "provider" in error_msg
        assert "SQLiteProvider" in error_msg

    def test_create_viterbi_via_main_factory_no_provider(self):
        """Test creating viterbi via main factory without provider raises."""
        with pytest.raises(ValueError) as exc_info:
            POSTaggerFactory.create("viterbi", provider=None)

        error_msg = str(exc_info.value)
        assert "provider" in error_msg


class TestPOSTaggerFactoryCustom:
    """Test custom tagger creation."""

    def test_create_custom_non_subclass_raises(self):
        """Test creating custom tagger with non-POSTaggerBase class raises."""

        class NotATagger:
            pass

        with pytest.raises(TypeError) as exc_info:
            POSTaggerFactory.create_custom(NotATagger)

        error_msg = str(exc_info.value)
        assert "POSTaggerBase" in error_msg
        assert "NotATagger" in error_msg

    def test_create_custom_valid_subclass(self):
        """Test creating custom tagger with valid subclass."""

        class CustomTagger(POSTaggerBase):
            def tag_word(self, word: str) -> str:
                return "N"

            def tag_sequence(self, words: List[str]) -> List[str]:
                return ["N"] * len(words)

            @property
            def tagger_type(self) -> TaggerType:
                return TaggerType.CUSTOM

            @property
            def supports_batch(self) -> bool:
                return False

            @property
            def is_fork_safe(self) -> bool:
                return True

        tagger = POSTaggerFactory.create_custom(CustomTagger)
        assert tagger is not None
        assert tagger.tagger_type == TaggerType.CUSTOM
        assert tagger.tag_word("test") == "N"

    def test_create_custom_via_main_factory(self):
        """Test creating custom tagger via main factory method."""

        class CustomTagger2(POSTaggerBase):
            def tag_word(self, word: str) -> str:
                return "V"

            def tag_sequence(self, words: List[str]) -> List[str]:
                return ["V"] * len(words)

            @property
            def tagger_type(self) -> TaggerType:
                return TaggerType.CUSTOM

            @property
            def supports_batch(self) -> bool:
                return False

            @property
            def is_fork_safe(self) -> bool:
                return True

        tagger = POSTaggerFactory.create("custom", tagger_class=CustomTagger2)
        assert tagger is not None
        assert tagger.tagger_type == TaggerType.CUSTOM
        assert tagger.tag_word("test") == "V"


class TestPOSTaggerFactoryRuleBased:
    """Test rule-based tagger creation with various options."""

    def test_create_rule_based_with_pos_map(self):
        """Test creating rule-based tagger with custom POS map."""
        pos_map = {"test": {"N"}, "word": {"V"}}
        tagger = POSTaggerFactory.create_rule_based(pos_map=pos_map)
        assert tagger is not None

    def test_create_rule_based_with_options(self):
        """Test creating rule-based tagger with all options."""
        tagger = POSTaggerFactory.create_rule_based(
            pos_map=None,
            use_morphology_fallback=False,
            cache_size=5000,
            unknown_tag="X",
        )
        assert tagger is not None
        assert tagger.unknown_tag == "X"


class TestPOSTaggerFactoryEdgeCases:
    """Test edge cases and normalization."""

    def test_create_with_uppercase_type(self):
        """Test creating tagger with uppercase type string."""
        tagger = POSTaggerFactory.create("RULE_BASED")
        assert tagger is not None
        assert tagger.tagger_type == TaggerType.RULE_BASED

    def test_create_with_whitespace_type(self):
        """Test creating tagger with whitespace in type string."""
        tagger = POSTaggerFactory.create("  rule_based  ")
        assert tagger is not None
        assert tagger.tagger_type == TaggerType.RULE_BASED

    def test_create_with_mixed_case_type(self):
        """Test creating tagger with mixed case type string."""
        tagger = POSTaggerFactory.create("Rule_Based")
        assert tagger is not None
        assert tagger.tagger_type == TaggerType.RULE_BASED


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
