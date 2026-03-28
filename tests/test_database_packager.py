"""
Tests for DatabasePackager.
"""

import sqlite3
import tempfile
from pathlib import Path

import pytest

from myspellchecker.core.constants import (
    DEFAULT_PIPELINE_BIGRAM_PROBS_FILE,
    DEFAULT_PIPELINE_SYLLABLE_FREQS_FILE,
    DEFAULT_PIPELINE_TRIGRAM_PROBS_FILE,
    DEFAULT_PIPELINE_WORD_FREQS_FILE,
)
from myspellchecker.data_pipeline.database_packager import DatabasePackager


class TestDatabasePackager:
    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            yield Path(tmp)

    def test_init_creates_db(self, temp_dir):
        database_path = temp_dir / "test.db"
        packager = DatabasePackager(input_dir=temp_dir, database_path=database_path)
        packager.connect()
        packager.create_schema()
        packager.close()

        assert database_path.exists()

        conn = sqlite3.connect(str(database_path))
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()

        assert "syllables" in tables
        assert "words" in tables
        assert "bigrams" in tables
        assert "trigrams" in tables

    def test_load_data(self, temp_dir):
        database_path = temp_dir / "test.db"

        # Create dummy input files
        (temp_dir / DEFAULT_PIPELINE_SYLLABLE_FREQS_FILE).write_text(
            "syllable\tfrequency\nမြန်\t10\n", encoding="utf-8"
        )
        # Words file needs 3 columns: word, syllable_count, frequency
        # Add "မြန်" and "မာ" as words so bigrams/trigrams are valid
        (temp_dir / DEFAULT_PIPELINE_WORD_FREQS_FILE).write_text(
            "word\tsyllable_count\tfrequency\nမြန်မာ\t2\t5\nမြန်\t1\t10\nမာ\t1\t10\nနိုင်ငံ\t2\t5\n",
            encoding="utf-8",
        )
        (temp_dir / DEFAULT_PIPELINE_BIGRAM_PROBS_FILE).write_text(
            "word1\tword2\tprobability\nမြန်\tမာ\t0.5\n", encoding="utf-8"
        )
        (temp_dir / DEFAULT_PIPELINE_TRIGRAM_PROBS_FILE).write_text(
            "word1\tword2\tword3\tprobability\nမြန်\tမာ\tနိုင်ငံ\t0.1\n", encoding="utf-8"
        )

        packager = DatabasePackager(input_dir=temp_dir, database_path=database_path)
        packager.connect()
        packager.create_schema()

        packager.load_syllables()
        packager.load_words()
        packager.load_bigrams()
        packager.load_trigrams()

        packager.optimize_database()
        packager.verify_database()
        packager.print_stats()

        packager.close()

        # Verify data
        conn = sqlite3.connect(str(database_path))
        cursor = conn.cursor()

        cursor.execute("SELECT frequency FROM syllables WHERE syllable='မြန်'")
        assert cursor.fetchone()[0] == 10

        cursor.execute("SELECT frequency FROM words WHERE word='မြန်မာ'")
        assert cursor.fetchone()[0] == 5

        # Verify bigram (checking word IDs might be complex, just check count)
        cursor.execute("SELECT count(*) FROM bigrams")
        assert cursor.fetchone()[0] == 1

        cursor.execute("SELECT count(*) FROM trigrams")
        assert cursor.fetchone()[0] == 1

        conn.close()

    def test_load_missing_files(self, temp_dir):
        database_path = temp_dir / "test.db"
        packager = DatabasePackager(input_dir=temp_dir, database_path=database_path)
        packager.connect()

        # Should handle missing files gracefully (print error but maybe not crash?)
        # Actually the code uses `open()` which raises FileNotFoundError
        # The `load_*` methods have try-except blocks? No.
        # Let's check source.
        # It calls `_load_csv_to_table` which opens file.

        with pytest.raises(FileNotFoundError):
            packager.load_syllables()

        packager.close()
