"""Tests for SQLiteProvider exception handling.

These tests verify that the SQLiteProvider properly distinguishes between
missing column errors (safe to catch) and critical database errors
(must be re-raised).
"""

import sqlite3
from unittest.mock import MagicMock, patch

import pytest

from myspellchecker.providers.sqlite import (
    SQLiteProvider,
    _is_missing_column_error,
    _is_missing_table_error,
)


class TestMissingColumnErrorDetection:
    """Tests for _is_missing_column_error helper function."""

    def test_detects_no_such_column_error(self):
        """Test detection of 'no such column' error."""
        error = sqlite3.OperationalError("no such column: is_curated")
        assert _is_missing_column_error(error) is True

    def test_detects_no_column_named_error(self):
        """Test detection of 'no column named' error."""
        error = sqlite3.OperationalError("no column named pos_tag")
        assert _is_missing_column_error(error) is True

    def test_detects_has_no_column_named_error(self):
        """Test detection of 'has no column named' error."""
        error = sqlite3.OperationalError("table words has no column named curated")
        assert _is_missing_column_error(error) is True

    def test_detects_case_insensitive(self):
        """Test that detection is case-insensitive."""
        error = sqlite3.OperationalError("NO SUCH COLUMN: is_curated")
        assert _is_missing_column_error(error) is True

        error = sqlite3.OperationalError("No Column Named pos_tag")
        assert _is_missing_column_error(error) is True

    def test_does_not_match_disk_io_error(self):
        """Test that disk I/O errors are not matched."""
        error = sqlite3.OperationalError("disk I/O error")
        assert _is_missing_column_error(error) is False

    def test_does_not_match_database_locked(self):
        """Test that database locked errors are not matched."""
        error = sqlite3.OperationalError("database is locked")
        assert _is_missing_column_error(error) is False

    def test_does_not_match_database_corruption(self):
        """Test that database corruption errors are not matched."""
        error = sqlite3.OperationalError("database disk image is malformed")
        assert _is_missing_column_error(error) is False

    def test_does_not_match_unable_to_open(self):
        """Test that unable to open errors are not matched."""
        error = sqlite3.OperationalError("unable to open database file")
        assert _is_missing_column_error(error) is False

    def test_does_not_match_generic_error(self):
        """Test that generic operational errors are not matched."""
        error = sqlite3.OperationalError("some unknown error occurred")
        assert _is_missing_column_error(error) is False


class TestMissingTableErrorDetection:
    """Tests for _is_missing_table_error helper function."""

    def test_detects_no_such_table_error(self):
        """Test detection of 'no such table' error."""
        error = sqlite3.OperationalError("no such table: pos_unigrams")
        assert _is_missing_table_error(error) is True

    def test_detects_table_not_found_error(self):
        """Test detection of 'table not found' error."""
        error = sqlite3.OperationalError("table not found: pos_bigrams")
        assert _is_missing_table_error(error) is True

    def test_detects_case_insensitive(self):
        """Test that detection is case-insensitive."""
        error = sqlite3.OperationalError("NO SUCH TABLE: pos_trigrams")
        assert _is_missing_table_error(error) is True

    def test_does_not_match_disk_io_error(self):
        """Test that disk I/O errors are not matched."""
        error = sqlite3.OperationalError("disk I/O error")
        assert _is_missing_table_error(error) is False

    def test_does_not_match_database_locked(self):
        """Test that database locked errors are not matched."""
        error = sqlite3.OperationalError("database is locked")
        assert _is_missing_table_error(error) is False

    def test_does_not_match_database_corruption(self):
        """Test that database corruption errors are not matched."""
        error = sqlite3.OperationalError("database disk image is malformed")
        assert _is_missing_table_error(error) is False

    def test_does_not_match_generic_error(self):
        """Test that generic operational errors are not matched."""
        error = sqlite3.OperationalError("some unknown error occurred")
        assert _is_missing_table_error(error) is False


class TestIsValidVocabularyExceptionHandling:
    """Tests for is_valid_vocabulary exception handling."""

    @pytest.fixture
    def mock_provider(self, tmp_path):
        """Create a provider with a mock database."""
        db_path = tmp_path / "test.db"
        # Create minimal database
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE words (word TEXT PRIMARY KEY)")
        cursor.execute("INSERT INTO words VALUES ('test')")
        conn.commit()
        conn.close()

        provider = SQLiteProvider(database_path=str(db_path))
        return provider

    def test_missing_column_falls_back_gracefully(self, mock_provider):
        """Test that missing is_curated column triggers fallback."""
        # Should fall back to is_valid_word without crashing
        result = mock_provider.is_valid_vocabulary("test")
        # Since our mock DB doesn't have is_curated, should use fallback
        # and return is_valid_word result
        assert isinstance(result, bool)

    def test_critical_error_is_reraised(self, mock_provider):
        """Test that critical errors are re-raised, not swallowed."""
        # Mock the connection to raise a critical error
        critical_error = sqlite3.OperationalError("disk I/O error")

        with patch.object(
            mock_provider,
            "_execute_query",
        ) as mock_execute:
            # Create a mock context manager that raises on cursor.execute
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_cursor.execute.side_effect = critical_error
            mock_conn.cursor.return_value = mock_cursor
            mock_conn.__enter__ = MagicMock(return_value=mock_conn)
            mock_conn.__exit__ = MagicMock(return_value=False)
            mock_execute.return_value = mock_conn

            with pytest.raises(sqlite3.OperationalError, match="disk I/O error"):
                mock_provider.is_valid_vocabulary("test")


class TestGetWordPosExceptionHandling:
    """Tests for get_word_pos exception handling."""

    @pytest.fixture
    def mock_provider(self, tmp_path):
        """Create a provider with a mock database without pos_tag column."""
        db_path = tmp_path / "test.db"
        # Create database without pos_tag column
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE words (word TEXT PRIMARY KEY)")
        cursor.execute("INSERT INTO words VALUES ('test')")
        cursor.execute("CREATE TABLE syllables (syllable TEXT PRIMARY KEY, frequency INT)")
        conn.commit()
        conn.close()

        provider = SQLiteProvider(database_path=str(db_path))
        return provider

    def test_missing_pos_column_returns_none(self, mock_provider):
        """Test that missing pos_tag column returns None without crashing."""
        result = mock_provider.get_word_pos("test")
        # Should return None gracefully, not crash
        assert result is None

    def test_critical_error_is_reraised(self, mock_provider):
        """Test that critical errors in get_word_pos are re-raised."""
        critical_error = sqlite3.OperationalError("database disk image is malformed")

        with patch.object(
            mock_provider,
            "_execute_query",
        ) as mock_execute:
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_cursor.execute.side_effect = critical_error
            mock_conn.cursor.return_value = mock_cursor
            mock_conn.__enter__ = MagicMock(return_value=mock_conn)
            mock_conn.__exit__ = MagicMock(return_value=False)
            mock_execute.return_value = mock_conn

            with pytest.raises(sqlite3.OperationalError, match="malformed"):
                mock_provider.get_word_pos("test")


class TestGetStatisticsExceptionHandling:
    """Tests for get_statistics exception handling."""

    @pytest.fixture
    def mock_provider(self, tmp_path):
        """Create a provider with a mock database."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        # Create tables without is_curated column
        cursor.execute("CREATE TABLE words (word TEXT PRIMARY KEY)")
        cursor.execute("CREATE TABLE syllables (syllable TEXT PRIMARY KEY, frequency INT)")
        cursor.execute("CREATE TABLE bigrams (bigram TEXT PRIMARY KEY, frequency INT)")
        cursor.execute("CREATE TABLE trigrams (trigram TEXT PRIMARY KEY, frequency INT)")
        cursor.execute("CREATE TABLE statistics (key TEXT PRIMARY KEY, value TEXT)")
        cursor.execute("INSERT INTO statistics VALUES ('syllable_count', '100')")
        cursor.execute("INSERT INTO statistics VALUES ('word_count', '50')")
        cursor.execute("INSERT INTO statistics VALUES ('bigram_count', '200')")
        cursor.execute("INSERT INTO statistics VALUES ('trigram_count', '150')")
        conn.commit()
        conn.close()

        provider = SQLiteProvider(database_path=str(db_path))
        return provider

    def test_missing_curated_column_sets_zero(self, mock_provider):
        """Test that missing is_curated column results in curated_count=0."""
        stats = mock_provider.get_statistics()
        # Should have curated_word_count as 0 due to missing column
        assert "curated_word_count" in stats
        assert stats["curated_word_count"] == 0


class TestIsValidVocabularyBulkExceptionHandling:
    """Tests for is_valid_vocabularys_bulk exception handling."""

    @pytest.fixture
    def mock_provider(self, tmp_path):
        """Create a provider with a mock database without is_curated."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE words (word TEXT PRIMARY KEY)")
        cursor.execute("INSERT INTO words VALUES ('hello')")
        cursor.execute("INSERT INTO words VALUES ('world')")
        cursor.execute("CREATE TABLE syllables (syllable TEXT PRIMARY KEY, frequency INT)")
        conn.commit()
        conn.close()

        provider = SQLiteProvider(database_path=str(db_path))
        return provider

    def test_missing_column_falls_back_to_valid_words_bulk(self, mock_provider):
        """Test that missing is_curated triggers fallback to is_valid_words_bulk."""
        result = mock_provider.is_valid_vocabulary_bulk(["hello", "world", "unknown"])
        # Should work without crashing, using fallback
        assert isinstance(result, dict)
        assert "hello" in result
        assert "world" in result
        assert "unknown" in result


class TestCriticalErrorsAreNotSwallowed:
    """Integration tests ensuring critical errors propagate correctly."""

    def test_locked_database_error_propagates(self, tmp_path):
        """Test that database locked errors are not swallowed."""
        db_path = tmp_path / "test.db"

        # Create a valid database
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE words (word TEXT PRIMARY KEY)")
        cursor.execute("CREATE TABLE syllables (syllable TEXT PRIMARY KEY, frequency INT)")
        conn.commit()
        conn.close()

        provider = SQLiteProvider(database_path=str(db_path))

        # Simulate a locked database error
        locked_error = sqlite3.OperationalError("database is locked")

        with patch.object(
            provider,
            "_execute_query",
        ) as mock_execute:
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_cursor.execute.side_effect = locked_error
            mock_conn.cursor.return_value = mock_cursor
            mock_conn.__enter__ = MagicMock(return_value=mock_conn)
            mock_conn.__exit__ = MagicMock(return_value=False)
            mock_execute.return_value = mock_conn

            # All these methods should re-raise the error
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                provider.is_valid_vocabulary("test")

            with pytest.raises(sqlite3.OperationalError, match="locked"):
                provider.get_word_pos("test")

        # Bulk methods use _bulk_ops._execute_query (a separate reference)
        with patch.object(
            provider._bulk_ops,
            "_execute_query",
        ) as mock_bulk_execute:
            mock_bulk_execute.return_value = mock_conn

            with pytest.raises(sqlite3.OperationalError, match="locked"):
                provider.is_valid_vocabulary_bulk(["test"])


class TestPOSProbabilityExceptionHandling:
    """Tests for POS probability methods exception handling."""

    @pytest.fixture
    def mock_provider(self, tmp_path):
        """Create a provider with a mock database without POS tables."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        # Create base tables but NOT pos_unigrams/bigrams/trigrams
        cursor.execute("CREATE TABLE words (word TEXT PRIMARY KEY)")
        cursor.execute("CREATE TABLE syllables (syllable TEXT PRIMARY KEY, frequency INT)")
        cursor.execute("CREATE TABLE bigrams (word1_id INT, word2_id INT, probability REAL)")
        cursor.execute(
            "CREATE TABLE trigrams (word1_id INT, word2_id INT, word3_id INT, probability REAL)"
        )
        conn.commit()
        conn.close()

        provider = SQLiteProvider(database_path=str(db_path))
        return provider

    def test_missing_pos_unigrams_returns_empty_dict(self, mock_provider):
        """Test that missing pos_unigrams table returns empty dict."""
        result = mock_provider.get_pos_unigram_probabilities()
        assert result == {}

    def test_missing_pos_bigrams_returns_empty_dict(self, mock_provider):
        """Test that missing pos_bigrams table returns empty dict."""
        result = mock_provider.get_pos_bigram_probabilities()
        assert result == {}

    def test_missing_pos_trigrams_returns_empty_dict(self, mock_provider):
        """Test that missing pos_trigrams table returns empty dict."""
        result = mock_provider.get_pos_trigram_probabilities()
        assert result == {}

    def test_pos_unigram_critical_error_propagates(self, mock_provider):
        """Test that critical errors in get_pos_unigram_probabilities propagate."""
        critical_error = sqlite3.OperationalError("disk I/O error")

        with patch.object(
            mock_provider,
            "_execute_query",
        ) as mock_execute:
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_cursor.execute.side_effect = critical_error
            mock_conn.cursor.return_value = mock_cursor
            mock_conn.__enter__ = MagicMock(return_value=mock_conn)
            mock_conn.__exit__ = MagicMock(return_value=False)
            mock_execute.return_value = mock_conn

            with pytest.raises(sqlite3.OperationalError, match="disk I/O error"):
                mock_provider.get_pos_unigram_probabilities()

    def test_pos_bigram_critical_error_propagates(self, mock_provider):
        """Test that critical errors in get_pos_bigram_probabilities propagate."""
        critical_error = sqlite3.OperationalError("database is locked")

        with patch.object(
            mock_provider,
            "_execute_query",
        ) as mock_execute:
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_cursor.execute.side_effect = critical_error
            mock_conn.cursor.return_value = mock_cursor
            mock_conn.__enter__ = MagicMock(return_value=mock_conn)
            mock_conn.__exit__ = MagicMock(return_value=False)
            mock_execute.return_value = mock_conn

            with pytest.raises(sqlite3.OperationalError, match="database is locked"):
                mock_provider.get_pos_bigram_probabilities()

    def test_pos_trigram_critical_error_propagates(self, mock_provider):
        """Test that critical errors in get_pos_trigram_probabilities propagate."""
        critical_error = sqlite3.OperationalError("database disk image is malformed")

        with patch.object(
            mock_provider,
            "_execute_query",
        ) as mock_execute:
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_cursor.execute.side_effect = critical_error
            mock_conn.cursor.return_value = mock_cursor
            mock_conn.__enter__ = MagicMock(return_value=mock_conn)
            mock_conn.__exit__ = MagicMock(return_value=False)
            mock_execute.return_value = mock_conn

            with pytest.raises(sqlite3.OperationalError, match="malformed"):
                mock_provider.get_pos_trigram_probabilities()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
