"""
Tests for POS tagger system functionality.

Tests ensure the new POS tagger configuration and FrequencyBuilder work correctly.
Note: Legacy pos_map parameter and ViterbiConfig have been removed in v1.0.0.
"""

from unittest.mock import MagicMock

import pytest

from myspellchecker.algorithms.pos_tagger_rule import RuleBasedPOSTagger
from myspellchecker.data_pipeline.frequency_builder import FrequencyBuilder
from myspellchecker.data_pipeline.pipeline import Pipeline


@pytest.fixture
def mock_console():
    """Create a mock PipelineConsole to avoid Rich rendering issues."""
    mock = MagicMock()
    mock.console = MagicMock()
    return mock


class TestFrequencyBuilderWithPosTagger:
    """Test FrequencyBuilder with POS tagger parameter."""

    def test_frequency_builder_with_pos_tagger(self, tmp_path):
        """Test that FrequencyBuilder accepts pos_tagger parameter."""
        pos_map = {
            "မြန်မာ": {"N"},
            "ကောင်း": {"ADJ"},
        }
        tagger = RuleBasedPOSTagger(pos_map=pos_map)

        builder = FrequencyBuilder(
            input_dir=tmp_path,
            output_dir=tmp_path,
            pos_tagger=tagger,
        )

        assert builder.pos_tagger is tagger

    def test_frequency_builder_without_tagger_is_ok(self, tmp_path):
        """Test that FrequencyBuilder works without pos_tagger."""
        builder = FrequencyBuilder(
            input_dir=tmp_path,
            output_dir=tmp_path,
        )

        # Should be None (no POS tagging)
        assert builder.pos_tagger is None


class TestPipelineWithPosTagger:
    """Test Pipeline with POS tagger configuration."""

    @pytest.mark.slow
    def test_pipeline_with_pos_tagger_config(self, tmp_path, mock_console):
        """Test that pipeline works with POSTaggerConfig."""
        from myspellchecker.core.config import POSTaggerConfig
        from myspellchecker.data_pipeline.config import PipelineConfig

        corpus_file = tmp_path / "corpus.txt"
        corpus_file.write_text("မြန်မာ\n", encoding="utf-8")

        output_db = tmp_path / "test.db"

        # New way: POSTaggerConfig with mock console to avoid Rich rendering issues
        config = PipelineConfig(
            pos_tagger=POSTaggerConfig(tagger_type="rule_based"),
            console=mock_console,
        )

        pipeline = Pipeline(config=config, work_dir=str(tmp_path / "work"))

        result = pipeline.build_database(
            input_files=[corpus_file],
            database_path=output_db,
            pos_tagger_config=config.pos_tagger,
            sample=True,  # Use sample mode for faster tests
        )

        assert result.exists()


class TestExistingCodePatterns:
    """Test that existing code patterns continue to work."""

    @pytest.mark.slow
    def test_basic_pipeline_without_pos_features(self, tmp_path, mock_console):
        """Test that pipeline works without any POS-related parameters."""
        from myspellchecker.data_pipeline.config import PipelineConfig

        corpus_file = tmp_path / "corpus.txt"
        corpus_file.write_text("မြန်မာ\n", encoding="utf-8")

        output_db = tmp_path / "test.db"

        # Standard usage: no POS configuration at all, but with mock console
        config = PipelineConfig(console=mock_console)
        pipeline = Pipeline(config=config, work_dir=str(tmp_path / "work"))

        result = pipeline.build_database(
            input_files=[corpus_file],
            database_path=output_db,
            sample=True,  # Use sample mode for faster tests
        )

        assert result.exists()

    @pytest.mark.slow
    def test_spellchecker_without_pos_config(self, tmp_path, mock_console):
        """Test that SpellChecker works without POS configuration."""
        from myspellchecker import SpellChecker
        from myspellchecker.data_pipeline.config import PipelineConfig
        from myspellchecker.data_pipeline.pipeline import Pipeline

        # Build database without POS config
        corpus_file = tmp_path / "corpus.txt"
        corpus_file.write_text("မြန်မာ ကောင်း\n" * 5, encoding="utf-8")

        output_db = tmp_path / "test.db"

        config = PipelineConfig(console=mock_console)
        pipeline = Pipeline(config=config, work_dir=str(tmp_path / "work"))
        pipeline.build_database(
            input_files=[corpus_file],
            database_path=output_db,
            sample=True,  # Use sample mode for faster tests
        )

        # Standard SpellChecker usage: no POS config
        from myspellchecker.core.config import SpellCheckerConfig
        from myspellchecker.providers.sqlite import SQLiteProvider

        # Create provider without explicit POS tagger
        provider = SQLiteProvider(database_path=str(output_db))
        config = SpellCheckerConfig(provider=provider)
        checker = SpellChecker(config=config)

        # Should work without POS configuration
        result = checker.check("မြန်မာ")
        assert result is not None


class TestMigrationPath:
    """Test migration path from old to new API."""

    def test_use_pos_tagger_with_pos_map(self, tmp_path):
        """Test using RuleBasedPOSTagger with pos_map."""
        # Create RuleBasedPOSTagger with pos_map
        pos_map = {"test": {"N"}, "word": {"V"}}
        tagger = RuleBasedPOSTagger(pos_map=pos_map)

        builder = FrequencyBuilder(
            input_dir=tmp_path,
            output_dir=tmp_path,
            pos_tagger=tagger,
        )

        assert builder.pos_tagger is tagger
        assert builder.pos_tagger.pos_map == pos_map
