"""Tests for POS tagger integration with the pipeline.

Tests ensure the pipeline works with POSTaggerConfig for POS tagging.
"""

import sqlite3
import tempfile
from pathlib import Path

import pytest

from myspellchecker.core.config import POSTaggerConfig
from myspellchecker.data_pipeline.config import PipelineConfig
from myspellchecker.data_pipeline.pipeline import Pipeline

# Mark all tests in this module as slow (they build databases)
pytestmark = pytest.mark.slow


class TestPosTaggerIntegration:
    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            yield Path(tmp)

    def test_pipeline_with_pos_tagger_config(self, temp_dir, mock_console):
        """Test pipeline works with POSTaggerConfig."""
        # 1. Setup Input Files
        corpus_file = temp_dir / "corpus.txt"
        corpus_content = "ကျွန်ုပ်သည် မြန်မာပြည် သို့သွားသည်"
        corpus_file.write_text(corpus_content, encoding="utf-8")

        output_db = temp_dir / "output.db"

        # 2. Run Pipeline with POSTaggerConfig and mock console
        config = PipelineConfig(console=mock_console)
        pipeline = Pipeline(config=config, work_dir=temp_dir / "work", keep_intermediate=True)

        pos_config = POSTaggerConfig(
            tagger_type="rule_based",
        )

        pipeline.build_database(
            input_files=[corpus_file],
            database_path=output_db,
            pos_tagger_config=pos_config,
            min_frequency=1,
            word_engine="myword",
            sample=True,  # Use sample mode for faster tests
        )

        # 3. Verify Database was created
        assert output_db.exists()
        conn = sqlite3.connect(str(output_db))
        cursor = conn.cursor()

        # Check that words table exists and has content
        cursor.execute("SELECT COUNT(*) FROM words")
        count = cursor.fetchone()[0]
        conn.close()

        assert count > 0, "No words found in database"
