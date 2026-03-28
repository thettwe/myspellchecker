"""
Security tests for SQLite provider.

Tests for path traversal prevention and batch validation.
"""

from pathlib import Path

import pytest

from myspellchecker.core.exceptions import DataLoadingError
from myspellchecker.providers.sqlite import (
    SQLITE_MAX_BATCH_SIZE,
    VALID_DB_EXTENSIONS,
    _validate_batch_items,
    _validate_database_path,
)


class TestPathTraversalPrevention:
    """Test path traversal attack prevention."""

    def test_valid_db_extension(self, tmp_path):
        """Test valid database extensions are accepted."""
        for ext in VALID_DB_EXTENSIONS:
            db_path = tmp_path / f"test{ext}"
            result = _validate_database_path(db_path)
            assert result.suffix.lower() == ext

    def test_invalid_extension_rejected(self, tmp_path):
        """Test invalid extensions are rejected."""
        invalid_extensions = [".txt", ".json", ".py", ".exe", ""]
        for ext in invalid_extensions:
            db_path = tmp_path / f"test{ext}"
            with pytest.raises(DataLoadingError, match="Invalid database file extension"):
                _validate_database_path(db_path)

    def test_path_traversal_rejected(self, tmp_path):
        """Test path traversal attempts are rejected."""
        traversal_paths = [
            "../../../etc/passwd.db",
            "..\\..\\..\\windows\\system32\\config.db",
            "foo/../../../bar.db",
            "./test/../../../etc.db",
        ]
        for path_str in traversal_paths:
            with pytest.raises(DataLoadingError, match="Path traversal detected"):
                _validate_database_path(Path(path_str))

    def test_absolute_path_accepted(self, tmp_path):
        """Test absolute paths are accepted."""
        db_path = tmp_path / "test.db"
        result = _validate_database_path(db_path)
        assert result.is_absolute()

    def test_relative_path_resolved(self):
        """Test relative paths without traversal are resolved to absolute."""
        result = _validate_database_path(Path("test.db"))
        assert result.is_absolute()

    def test_symlink_resolved(self, tmp_path):
        """Test symlinks are resolved to real path."""
        real_db = tmp_path / "real.db"
        real_db.touch()
        link_db = tmp_path / "link.db"
        link_db.symlink_to(real_db)

        result = _validate_database_path(link_db)
        # Result should be the resolved path
        assert result == real_db.resolve()


class TestBatchValidation:
    """Test batch validation for SQL queries."""

    def test_valid_batch_accepted(self):
        """Test valid batch is accepted."""
        batch = ["word1", "word2", "word3"]
        # Should not raise
        _validate_batch_items(batch, "word")

    def test_empty_batch_accepted(self):
        """Test empty batch is accepted."""
        batch = []
        # Should not raise
        _validate_batch_items(batch, "word")

    def test_max_size_batch_accepted(self):
        """Test batch at max size is accepted."""
        batch = [f"word{i}" for i in range(SQLITE_MAX_BATCH_SIZE)]
        # Should not raise
        _validate_batch_items(batch, "word")

    def test_oversized_batch_rejected(self):
        """Test batch exceeding max size is rejected."""
        batch = [f"word{i}" for i in range(SQLITE_MAX_BATCH_SIZE + 1)]
        with pytest.raises(ValueError, match="Batch size .* exceeds maximum"):
            _validate_batch_items(batch, "word")

    def test_non_string_items_rejected(self):
        """Test non-string items are rejected."""
        invalid_batches = [
            [1, 2, 3],  # integers
            [None, "word"],  # None mixed
            [{"key": "value"}],  # dict
            [["nested"]],  # list
            [b"bytes"],  # bytes
        ]
        for batch in invalid_batches:
            with pytest.raises(ValueError, match="Invalid .* type at index"):
                _validate_batch_items(batch, "item")

    def test_mixed_types_rejected(self):
        """Test mixed types are rejected."""
        batch = ["valid", 123, "also_valid"]
        with pytest.raises(ValueError, match="Invalid .* type at index 1"):
            _validate_batch_items(batch, "word")

    def test_error_message_includes_type_name(self):
        """Test error message includes actual type name."""
        batch = [123]
        with pytest.raises(ValueError, match="expected str, got int"):
            _validate_batch_items(batch, "word")
