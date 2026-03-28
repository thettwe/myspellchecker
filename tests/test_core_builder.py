from unittest.mock import MagicMock, patch

import pytest

from myspellchecker.core.builder import ConfigPresets, SpellCheckerBuilder
from myspellchecker.core.config import SpellCheckerConfig
from myspellchecker.core.exceptions import MissingDatabaseError


def test_builder_with_methods():
    """Test all with_* methods set config correctly."""
    builder = SpellCheckerBuilder()

    builder.with_phonetic(False)
    assert builder._config.use_phonetic is False

    builder.with_context_checking(False)
    assert builder._config.use_context_checker is False

    builder.with_ner(False)
    assert builder._config.use_ner is False

    builder.with_rule_based_validation(False)
    assert builder._config.use_rule_based_validation is False

    builder.with_max_edit_distance(3)
    assert builder._config.max_edit_distance == 3

    with pytest.raises(ValueError):
        builder.with_max_edit_distance(4)

    builder.with_max_suggestions(5)
    assert builder._config.max_suggestions == 5

    with pytest.raises(ValueError):
        builder.with_max_suggestions(0)

    builder.with_symspell_prefix_length(10)
    assert builder._config.symspell.prefix_length == 10

    builder.with_cache_size(500)
    assert builder._config.provider_config.cache_size == 500

    builder.with_bigram_threshold(0.5)
    assert builder._config.ngram_context.bigram_threshold == 0.5

    builder.with_trigram_threshold(0.6)
    assert builder._config.ngram_context.trigram_threshold == 0.6

    builder.with_word_engine("crf")
    assert builder._config.word_engine == "crf"

    # This should now work with atomic update
    builder.with_semantic_model(model="model", tokenizer="tok")
    assert builder._config.semantic.model == "model"
    assert builder._config.semantic.tokenizer == "tok"


def test_builder_with_components():
    """Test setting custom components."""
    builder = SpellCheckerBuilder()

    prov = MagicMock()
    builder.with_provider(prov)
    assert builder._provider == prov

    seg = MagicMock()
    builder.with_segmenter(seg)
    assert builder._segmenter == seg

    conf = SpellCheckerConfig(use_phonetic=False)
    builder.with_config(conf)
    assert builder._config == conf


def test_build_defaults():
    """Test building with defaults."""
    builder = SpellCheckerBuilder()

    # Patch where the class is defined/imported from
    with (
        patch("myspellchecker.providers.SQLiteProvider") as MockProvider,
        patch("myspellchecker.segmenters.DefaultSegmenter") as MockSegmenter,
        patch("myspellchecker.core.spellchecker.SpellChecker") as MockChecker,
    ):
        builder.build()

        MockProvider.assert_called()
        MockSegmenter.assert_called()
        MockChecker.assert_called()


def test_build_no_db_raises_error_by_default():
    """Test building raises MissingDatabaseError when DB not found (default behavior)."""
    builder = SpellCheckerBuilder()

    with (
        patch(
            "myspellchecker.providers.SQLiteProvider",
            side_effect=MissingDatabaseError(message="test db not found"),
        ),
        patch("myspellchecker.segmenters.DefaultSegmenter"),
    ):
        with pytest.raises(MissingDatabaseError, match="No database available"):
            builder.build()


def test_build_no_db_fallback_when_enabled():
    """Test building falls back to MemoryProvider when fallback is enabled."""
    config = SpellCheckerConfig(fallback_to_empty_provider=True)
    builder = SpellCheckerBuilder().with_config(config)

    with (
        patch(
            "myspellchecker.providers.SQLiteProvider",
            side_effect=MissingDatabaseError(message="test db not found"),
        ),
        patch("myspellchecker.providers.MemoryProvider") as MockMemProvider,
        patch("myspellchecker.segmenters.DefaultSegmenter"),
        patch("myspellchecker.core.spellchecker.SpellChecker"),
    ):
        with pytest.warns(RuntimeWarning, match="Default database not found"):
            builder.build()

        MockMemProvider.assert_called()


def test_build_custom_components():
    """Test building with custom components."""
    prov = MagicMock()
    seg = MagicMock()
    builder = SpellCheckerBuilder().with_provider(prov).with_segmenter(seg)

    with patch("myspellchecker.core.spellchecker.SpellChecker") as MockChecker:
        builder.build()
        args = MockChecker.call_args[1]
        assert args["provider"] == prov
        assert args["segmenter"] == seg


def test_config_presets():
    """Test preset configurations."""

    # Access properties to trigger lazy loading
    def check_preset(preset, **kwargs):
        for k, v in kwargs.items():
            if "." in k:
                # Handle nested config (e.g. symspell.prefix_length)
                parts = k.split(".")
                obj = preset
                for p in parts[:-1]:
                    obj = getattr(obj, p)
                assert getattr(obj, parts[-1]) == v
            else:
                assert getattr(preset, k) == v

    check_preset(ConfigPresets.default(), use_phonetic=True)

    check_preset(
        ConfigPresets.fast(), use_phonetic=False, use_context_checker=False, max_edit_distance=1
    )

    check_preset(ConfigPresets.accurate(), max_edit_distance=3, use_ner=True)

    check_preset(ConfigPresets.minimal(), use_rule_based_validation=False, max_suggestions=3)

    check_preset(ConfigPresets.strict(), max_edit_distance=2, max_suggestions=10)

    # Test property access via class attributes (metaclass magic)
    assert ConfigPresets.FAST.use_phonetic is False
    assert ConfigPresets.ACCURATE.max_edit_distance == 3


def test_config_presets_invalid():
    """Test accessing invalid preset."""
    with pytest.raises(AttributeError):
        _ = ConfigPresets.INVALID_PRESET
