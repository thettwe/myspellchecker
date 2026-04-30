"""
Tests for POS tagger system functionality.

Tests ensure the new POS tagger configuration and FrequencyBuilder work correctly.
Note: Legacy pos_map parameter and ViterbiConfig have been removed in v1.0.0.
"""

import pytest

duckdb = pytest.importorskip("duckdb", reason="DuckDB required for POS backward compat tests")

from myspellchecker.algorithms.pos_tagger_rule import RuleBasedPOSTagger  # noqa: E402
from myspellchecker.data_pipeline.frequency_builder import FrequencyBuilder  # noqa: E402


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
