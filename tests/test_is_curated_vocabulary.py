"""
Tests for the is_curated vocabulary feature.

This feature allows distinguishing between curated vocabulary words (from --curated-input)
and corpus-derived words that may include segmentation artifacts.
"""

import sqlite3

import pytest


class TestIsCuratedSchema:
    """Test the is_curated column in the words table schema."""

    def test_schema_includes_is_curated_column(self, tmp_path):
        """Verify the words table schema includes is_curated column."""
        from myspellchecker.data_pipeline.database_packager import DatabasePackager

        db_path = tmp_path / "test.db"
        packager = DatabasePackager(input_dir=tmp_path, database_path=db_path)
        packager.connect()
        packager.create_schema()

        # Check schema
        packager.cursor.execute("PRAGMA table_info(words)")
        columns = {row[1]: row[2] for row in packager.cursor.fetchall()}

        assert "is_curated" in columns, "is_curated column should exist"
        assert columns["is_curated"] == "INTEGER", "is_curated should be INTEGER type"

        packager.close()

    def test_is_curated_default_value(self, tmp_path):
        """Verify is_curated defaults to 0."""
        from myspellchecker.data_pipeline.database_packager import DatabasePackager

        db_path = tmp_path / "test.db"
        packager = DatabasePackager(input_dir=tmp_path, database_path=db_path)
        packager.connect()
        packager.create_schema()

        # Insert a word without specifying is_curated
        packager.cursor.execute(
            "INSERT INTO words (word, syllable_count, frequency) VALUES (?, ?, ?)",
            ("test_word", 1, 100),
        )
        packager.conn.commit()

        # Verify default is 0
        packager.cursor.execute("SELECT is_curated FROM words WHERE word = ?", ("test_word",))
        result = packager.cursor.fetchone()
        assert result[0] == 0, "is_curated should default to 0"

        packager.close()


class TestLoadWordsWithCuration:
    """Test load_words() sets is_curated correctly."""

    def test_corpus_words_without_curated_set_are_not_curated(self, tmp_path):
        """Words from corpus without curated_words set should have is_curated=0."""
        from myspellchecker.data_pipeline.database_packager import DatabasePackager

        # Create word frequencies file
        word_freqs_path = tmp_path / "word_freqs.tsv"
        word_freqs_path.write_text(
            "word\tsyllable_count\tfrequency\nမြန်မာ\t2\t1000\nကျောင်း\t1\t500\n"
        )

        db_path = tmp_path / "test.db"
        packager = DatabasePackager(input_dir=tmp_path, database_path=db_path)
        packager.connect()
        packager.create_schema()
        packager.load_words(filename="word_freqs.tsv", curated_words=None)

        # Verify all words have is_curated=0
        packager.cursor.execute("SELECT word, is_curated FROM words")
        results = {row[0]: row[1] for row in packager.cursor.fetchall()}

        assert results.get("မြန်မာ") == 0, "Word without curated set should not be curated"
        assert results.get("ကျောင်း") == 0, "Word without curated set should not be curated"

        packager.close()

    def test_corpus_words_in_curated_set_are_curated(self, tmp_path):
        """Words from corpus that are in curated_words set should have is_curated=1."""
        from myspellchecker.data_pipeline.database_packager import DatabasePackager

        # Create word frequencies file
        # Using valid Myanmar words - some in curated set, some not
        word_freqs_path = tmp_path / "word_freqs.tsv"
        word_freqs_path.write_text(
            "word\tsyllable_count\tfrequency\n"
            "မြန်မာ\t2\t1000\n"
            "ကျောင်း\t1\t500\n"
            "စမ်းသပ်\t2\t10\n"  # Valid Myanmar word NOT in curated_words
        )

        # Create curated_words set (from --curated-input)
        curated_words = {"မြန်မာ", "ကျောင်း"}

        db_path = tmp_path / "test.db"
        packager = DatabasePackager(input_dir=tmp_path, database_path=db_path)
        packager.connect()
        packager.create_schema()
        packager.load_words(filename="word_freqs.tsv", curated_words=curated_words)

        # Verify curated status
        packager.cursor.execute("SELECT word, is_curated FROM words")
        results = {row[0]: row[1] for row in packager.cursor.fetchall()}

        assert results.get("မြန်မာ") == 1, "Word in curated set should be curated"
        assert results.get("ကျောင်း") == 1, "Word in curated set should be curated"
        assert results.get("စမ်းသပ်") == 0, "Word NOT in curated set should NOT be curated"

        packager.close()


class TestLoadCuratedWordsDirect:
    """Test load_curated_words() inserts curated words directly into database."""

    def test_load_curated_words_inserts_directly(self, tmp_path):
        """Curated words should be inserted directly with is_curated=1 and frequency=0."""
        from myspellchecker.data_pipeline.database_packager import DatabasePackager

        db_path = tmp_path / "test.db"
        packager = DatabasePackager(input_dir=tmp_path, database_path=db_path)
        packager.connect()
        packager.create_schema()

        # Load curated words directly (no corpus needed)
        curated_words = {"မြန်မာ", "ကောင်း", "စာ"}
        count = packager.load_curated_words(curated_words)

        assert count == 3, "Should insert 3 curated words"

        # Verify all words are in database with correct values
        packager.cursor.execute("SELECT word, frequency, is_curated FROM words")
        results = {row[0]: (row[1], row[2]) for row in packager.cursor.fetchall()}

        assert "မြန်မာ" in results, "Curated word should be in database"
        assert results["မြန်မာ"][0] == 0, "Initial frequency should be 0"
        assert results["မြန်မာ"][1] == 1, "is_curated should be 1"

        packager.close()

    def test_load_curated_words_computes_syllable_count(self, tmp_path):
        """Curated words should have syllable_count computed via segmentation."""
        from myspellchecker.data_pipeline.database_packager import DatabasePackager

        db_path = tmp_path / "test.db"
        packager = DatabasePackager(input_dir=tmp_path, database_path=db_path)
        packager.connect()
        packager.create_schema()

        curated_words = {"မြန်မာ"}  # Should have 2 syllables
        packager.load_curated_words(curated_words)

        packager.cursor.execute("SELECT syllable_count FROM words WHERE word = ?", ("မြန်မာ",))
        syllable_count = packager.cursor.fetchone()[0]

        assert syllable_count == 2, "မြန်မာ should have 2 syllables"

        packager.close()

    def test_corpus_updates_frequency_preserves_curated(self, tmp_path):
        """When corpus word overlaps with curated, frequency updates but is_curated stays 1."""
        from myspellchecker.data_pipeline.database_packager import DatabasePackager

        # Create word frequencies file
        word_freqs_path = tmp_path / "word_freqs.tsv"
        word_freqs_path.write_text(
            "word\tsyllable_count\tfrequency\tpos_tag\n"
            "မြန်မာ\t2\t500\tN\n"  # Overlaps with curated
            "အသစ်\t2\t100\t\n"  # Not in curated
        )

        db_path = tmp_path / "test.db"
        packager = DatabasePackager(input_dir=tmp_path, database_path=db_path)
        packager.connect()
        packager.create_schema()

        # Step 1: Load curated words FIRST
        curated_words = {"မြန်မာ", "ကောင်း"}
        packager.load_curated_words(curated_words)

        # Step 2: Load corpus words
        packager.load_words(filename="word_freqs.tsv", curated_words=curated_words)

        # Verify results
        packager.cursor.execute("SELECT word, frequency, is_curated FROM words")
        results = {row[0]: (row[1], row[2]) for row in packager.cursor.fetchall()}

        # မြန်မာ: freq=500 (updated from corpus), is_curated=1 (preserved)
        assert results["မြန်မာ"][0] == 500, "Frequency should be updated from corpus"
        assert results["မြန်မာ"][1] == 1, "is_curated should remain 1"

        # ကောင်း: freq=0 (curated only, not in corpus), is_curated=1
        assert results["ကောင်း"][0] == 0, "Curated-only word should have freq=0"
        assert results["ကောင်း"][1] == 1, "Curated-only word should be curated"

        # အသစ်: freq=100 (corpus only), is_curated=0
        assert results["အသစ်"][0] == 100, "Corpus-only word should have corpus freq"
        assert results["အသစ်"][1] == 0, "Corpus-only word should not be curated"

        packager.close()

    def test_load_curated_words_empty_set(self, tmp_path):
        """Empty curated_words set should return 0 and not fail."""
        from myspellchecker.data_pipeline.database_packager import DatabasePackager

        db_path = tmp_path / "test.db"
        packager = DatabasePackager(input_dir=tmp_path, database_path=db_path)
        packager.connect()
        packager.create_schema()

        count = packager.load_curated_words(set())

        assert count == 0, "Empty set should return 0"

        packager.cursor.execute("SELECT COUNT(*) FROM words")
        word_count = packager.cursor.fetchone()[0]
        assert word_count == 0, "No words should be inserted"

        packager.close()


class TestSQLiteProviderVocabularyMethods:
    """Test SQLiteProvider vocabulary validation methods."""

    @pytest.fixture
    def provider_with_curated_data(self, tmp_path):
        """Create a provider with test data including curated words."""
        from myspellchecker.providers.sqlite import SQLiteProvider

        db_path = tmp_path / "test.db"

        # Create database with curated and non-curated words
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE syllables (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                syllable TEXT UNIQUE NOT NULL,
                frequency INTEGER DEFAULT 0
            )
        """)

        cursor.execute("""
            CREATE TABLE words (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                word TEXT UNIQUE NOT NULL,
                syllable_count INTEGER,
                frequency INTEGER DEFAULT 0,
                pos_tag TEXT,
                is_curated INTEGER DEFAULT 0
            )
        """)

        cursor.execute("""
            CREATE TABLE bigrams (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                word1_id INTEGER,
                word2_id INTEGER,
                probability REAL DEFAULT 0.0
            )
        """)

        cursor.execute("""
            CREATE TABLE trigrams (
                id INTEGER PRIMARY KEY,
                word1_id INTEGER,
                word2_id INTEGER,
                word3_id INTEGER,
                probability REAL DEFAULT 0.0
            )
        """)

        # Insert test words
        test_words = [
            ("မြန်မာ", 2, 1000, "N", 1),  # Curated
            ("ကျောင်း", 1, 500, "N", 1),  # Curated
            ("artifact1", 1, 10, None, 0),  # Not curated (corpus artifact)
            ("artifact2", 1, 5, None, 0),  # Not curated (corpus artifact)
        ]

        cursor.executemany(
            """INSERT INTO words (word, syllable_count, frequency, pos_tag, is_curated)
            VALUES (?, ?, ?, ?, ?)""",
            test_words,
        )

        conn.commit()
        conn.close()

        # Create provider
        provider = SQLiteProvider(database_path=str(db_path))
        yield provider
        provider.close()

    def test_is_valid_vocabulary_returns_true_for_curated(self, provider_with_curated_data):
        """is_valid_vocabulary should return True for curated words."""
        provider = provider_with_curated_data

        assert provider.is_valid_vocabulary("မြန်မာ") is True
        assert provider.is_valid_vocabulary("ကျောင်း") is True

    def test_is_valid_vocabulary_returns_false_for_non_curated(self, provider_with_curated_data):
        """is_valid_vocabulary should return False for non-curated words."""
        provider = provider_with_curated_data

        assert provider.is_valid_vocabulary("artifact1") is False
        assert provider.is_valid_vocabulary("artifact2") is False

    def test_is_valid_vocabulary_returns_false_for_unknown(self, provider_with_curated_data):
        """is_valid_vocabulary should return False for unknown words."""
        provider = provider_with_curated_data

        assert provider.is_valid_vocabulary("unknown_word") is False
        assert provider.is_valid_vocabulary("") is False

    def test_is_valid_word_still_works_for_all_words(self, provider_with_curated_data):
        """is_valid_word should return True for ANY word in database (curated or not)."""
        provider = provider_with_curated_data

        # Both curated and non-curated words should be valid
        assert provider.is_valid_word("မြန်မာ") is True
        assert provider.is_valid_word("artifact1") is True
        assert provider.is_valid_word("unknown_word") is False

    def test_is_valid_vocabulary_bulk(self, provider_with_curated_data):
        """is_valid_vocabulary_bulk should correctly identify curated words."""
        provider = provider_with_curated_data

        words = ["မြန်မာ", "ကျောင်း", "artifact1", "unknown"]
        result = provider.is_valid_vocabulary_bulk(words)

        assert result["မြန်မာ"] is True
        assert result["ကျောင်း"] is True
        assert result["artifact1"] is False
        assert result["unknown"] is False

    def test_get_statistics_includes_curated_count(self, provider_with_curated_data):
        """get_statistics should include curated_word_count."""
        provider = provider_with_curated_data

        stats = provider.get_statistics()

        assert "curated_word_count" in stats
        assert stats["curated_word_count"] == 2  # Only 2 curated words
        assert stats["word_count"] == 4  # Total 4 words


class TestBackwardsCompatibility:
    """Test backwards compatibility with databases without is_curated column."""

    @pytest.fixture
    def provider_without_curated_column(self, tmp_path):
        """Create a provider with old schema (no is_curated column)."""
        from myspellchecker.providers.sqlite import SQLiteProvider

        db_path = tmp_path / "old_test.db"

        # Create database WITHOUT is_curated column (old schema)
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE syllables (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                syllable TEXT UNIQUE NOT NULL,
                frequency INTEGER DEFAULT 0
            )
        """)

        cursor.execute("""
            CREATE TABLE words (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                word TEXT UNIQUE NOT NULL,
                syllable_count INTEGER,
                frequency INTEGER DEFAULT 0,
                pos_tag TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE bigrams (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                word1_id INTEGER,
                word2_id INTEGER,
                probability REAL DEFAULT 0.0
            )
        """)

        cursor.execute("""
            CREATE TABLE trigrams (
                id INTEGER PRIMARY KEY,
                word1_id INTEGER,
                word2_id INTEGER,
                word3_id INTEGER,
                probability REAL DEFAULT 0.0
            )
        """)

        # Insert test words
        cursor.execute(
            "INSERT INTO words (word, syllable_count, frequency) VALUES (?, ?, ?)",
            ("မြန်မာ", 2, 1000),
        )

        conn.commit()
        conn.close()

        provider = SQLiteProvider(database_path=str(db_path))
        yield provider
        provider.close()

    def test_is_valid_vocabulary_falls_back_to_is_valid_word(self, provider_without_curated_column):
        """is_valid_vocabulary should fall back to is_valid_word for old databases."""
        provider = provider_without_curated_column

        # Should not raise an error, should fall back
        result = provider.is_valid_vocabulary("မြန်မာ")
        # Falls back to is_valid_word, which returns True
        assert result is True

    def test_is_valid_vocabulary_bulk_falls_back(self, provider_without_curated_column):
        """is_valid_vocabulary_bulk should fall back for old databases."""
        provider = provider_without_curated_column

        words = ["မြန်မာ", "unknown"]
        result = provider.is_valid_vocabulary_bulk(words)

        # Falls back to is_valid_words_bulk
        assert result["မြန်မာ"] is True
        assert result["unknown"] is False

    def test_get_statistics_returns_zero_curated_for_old_db(self, provider_without_curated_column):
        """get_statistics should return 0 curated_word_count for old databases."""
        provider = provider_without_curated_column

        stats = provider.get_statistics()

        assert stats["curated_word_count"] == 0
