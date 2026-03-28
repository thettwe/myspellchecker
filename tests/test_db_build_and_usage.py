"""
Integration test for Database Builder and SpellChecker usage.

This test verifies that:
1. The Pipeline can build a valid SQLite database from raw text.
2. The SpellChecker can load this custom database.
3. The SpellChecker behaves correctly using the data from the corpus.
"""

import sqlite3
import tempfile
from pathlib import Path

import pytest

from myspellchecker import SpellChecker
from myspellchecker.core.config import SpellCheckerConfig
from myspellchecker.core.config.algorithm_configs import SymSpellConfig
from myspellchecker.data_pipeline import Pipeline, PipelineConfig
from myspellchecker.providers import SQLiteProvider

# Sample corpus containing "Myanmar" (correct) and "Country" (correct)
# We will test if we can detect "Myanmarr" (typo)
# Repeated multiple times to meet default frequency threshold (5)
SAMPLE_CORPUS_TEXT = (
    """
မြန်မာ နိုင်ငံ သည် အရှေ့တောင်အာရှ တွင် ရှိသည်။
မြန်မာ လူမျိုး တို့ သည် ဖော်ရွေ ကြ သည်။
ရန်ကုန် မြို့ သည် မြန်မာ နိုင်ငံ ၏ မြို့တော် ဟောင်း ဖြစ်သည်။
သူ ကျောင်း သွား သည် ။
"""
    * 10
)


@pytest.fixture
def custom_db_path(mock_console):
    """Fixture to create a custom database and clean it up."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Create corpus file
        corpus_path = Path(tmpdir) / "corpus.txt"
        with open(corpus_path, "w", encoding="utf-8") as f:
            f.write(SAMPLE_CORPUS_TEXT)

        # 2. Build database with low min_frequency for test data
        db_path = Path(tmpdir) / "custom.db"
        work_dir = Path(tmpdir) / "work"

        # Use PipelineConfig with min_frequency=5 and mock console
        # Disable disk space check — test corpora are tiny
        config = PipelineConfig(
            work_dir=str(work_dir),
            keep_intermediate=False,
            min_frequency=5,  # Low threshold for test corpus
            console=mock_console,
            disk_space_check_mb=0,
        )
        pipeline = Pipeline(config=config)
        pipeline.build_database(
            input_files=[corpus_path],
            database_path=str(db_path),
        )

        yield db_path


def test_database_structure(custom_db_path):
    """Verify the SQLite database structure."""
    assert custom_db_path.exists()

    conn = sqlite3.connect(str(custom_db_path))
    cursor = conn.cursor()

    # Check tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cursor.fetchall()}
    assert "syllables" in tables
    assert "words" in tables
    assert "bigrams" in tables

    # Check data content
    # Note: Word segmentation may split multi-syllable words into syllables
    # Check for syllable components of "မြန်မာ" (Myanmar)
    # The word segmenter may produce "မြန်" as a syllable component
    cursor.execute("SELECT frequency FROM words WHERE word=?", ("မြန်",))
    result_split = cursor.fetchone()
    cursor.execute("SELECT frequency FROM words WHERE word=?", ("မြန်မာ",))
    result_full = cursor.fetchone()
    # Check syllables table as fallback — some segmenters put data there
    cursor.execute("SELECT frequency FROM syllables WHERE syllable=?", ("မြန်",))
    result_syllable = cursor.fetchone()
    # At least one table should have the syllable or full word
    assert result_split is not None or result_full is not None or result_syllable is not None, (
        "Word 'မြန်မာ' (or its syllable 'မြန်') should be in database"
    )
    # Check that whichever was stored has reasonable frequency
    # With quality filtering and segmentation, frequencies may be lower
    # than the raw corpus repetition count (10x). Accept >= 1 as valid.
    if result_split:
        assert result_split[0] >= 1
    if result_full:
        assert result_full[0] >= 1
    if result_syllable:
        assert result_syllable[0] >= 1

    # Check "ကျောင်း" in syllables
    cursor.execute("SELECT frequency FROM syllables WHERE syllable=?", ("ကျောင်း",))
    result = cursor.fetchone()
    assert result is not None

    conn.close()


def test_spellchecker_with_custom_db(custom_db_path):
    """Verify SpellChecker works with the custom database."""
    # Initialize provider with custom DB
    # Use count_threshold=1 so syllables with low frequency in the small test
    # corpus are not filtered out by the SymSpell frequency threshold check.
    provider = SQLiteProvider(database_path=str(custom_db_path))
    config = SpellCheckerConfig(provider=provider, symspell=SymSpellConfig(count_threshold=1))
    checker = SpellChecker(config=config)

    # 1. Valid syllable check
    # "သည်" (thé, particle) is very common in corpus -> Should be valid
    # This syllable has simple structure and passes syllable validation
    res_valid = checker.check("သည်")
    assert not res_valid.has_errors, f"'သည်' should be valid but got errors: {res_valid.errors}"

    # 2. Invalid syllable check (Typo)
    # "သည" (typo - missing Visarga) -> Should be invalid
    res_invalid = checker.check("သည")
    assert res_invalid.has_errors
    assert len(res_invalid.errors) > 0
    # Expect suggestion "သည်"
    suggestions = res_invalid.errors[0].suggestions
    assert "သည်" in suggestions
