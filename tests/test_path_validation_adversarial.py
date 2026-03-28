"""
Adversarial tests for path validation in SQLiteProvider.

These tests verify that the path validation properly prevents:
- Path traversal attacks (../)
- Null byte injection
- Unicode normalization bypasses (NFC/NFD)
- Control character injection
- Symlink-based attacks
"""

import os
import tempfile
from pathlib import Path

import pytest

from myspellchecker.core.exceptions import DataLoadingError
from myspellchecker.providers.sqlite import _validate_database_path


class TestPathTraversalPrevention:
    """Test prevention of path traversal attacks."""

    def test_simple_path_traversal_rejected(self):
        """Test that simple ../ path traversal is rejected."""
        with pytest.raises(DataLoadingError, match="Path traversal detected"):
            _validate_database_path(Path("../../../etc/passwd.db"))

    def test_path_traversal_in_middle_rejected(self):
        """Test that path traversal in middle of path is rejected."""
        with pytest.raises(DataLoadingError, match="Path traversal detected"):
            _validate_database_path(Path("/valid/path/../../../etc/passwd.db"))

    def test_windows_style_traversal_rejected(self):
        """Test that Windows-style path traversal is rejected."""
        with pytest.raises(DataLoadingError, match="Path traversal detected"):
            _validate_database_path(Path("..\\..\\windows\\system32\\config.db"))

    def test_mixed_separator_traversal_rejected(self):
        """Test mixed separator path traversal is rejected."""
        with pytest.raises(DataLoadingError, match="Path traversal detected"):
            _validate_database_path(Path("valid/..\\..\\etc/passwd.db"))

    def test_encoded_traversal_with_dots(self):
        """Test that literal dots in path are detected."""
        # These are literal dots, not URL-encoded, so should be rejected
        with pytest.raises(DataLoadingError, match="Path traversal detected"):
            _validate_database_path(Path("folder/..%2f../secret.db"))


class TestNullByteInjection:
    """Test prevention of null byte injection attacks."""

    def test_null_byte_in_path_rejected(self):
        """Test that null bytes in path are rejected."""
        with pytest.raises(DataLoadingError, match="Null byte detected"):
            _validate_database_path(Path("valid.db\x00.txt"))

    def test_null_byte_at_end_rejected(self):
        """Test that null byte at end of path is rejected."""
        with pytest.raises(DataLoadingError, match="Null byte detected"):
            _validate_database_path(Path("valid.db\x00"))

    def test_null_byte_at_start_rejected(self):
        """Test that null byte at start of path is rejected."""
        with pytest.raises(DataLoadingError, match="Null byte detected"):
            _validate_database_path(Path("\x00valid.db"))


class TestUnicodeNormalization:
    """Test Unicode normalization attack prevention."""

    def test_nfd_normalized_to_nfc(self):
        """Test that NFD paths are normalized to NFC."""
        # NFD: a + combining umlaut (U+0308)
        nfd_path = "te\u0308st.db"  # "tëst.db" in NFD
        # NFC: precomposed ë (U+00EB)
        nfc_path = "t\u00ebst.db"  # "tëst.db" in NFC

        # Both should resolve to the same normalized form
        # (This test verifies normalization happens, not that files exist)
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create file with NFC name
            nfc_file = Path(tmpdir) / nfc_path
            nfc_file.touch()

            # Validate NFD path should normalize to NFC
            result = _validate_database_path(Path(tmpdir) / nfd_path)
            # Result should be a valid path (normalization applied)
            assert result is not None

    def test_unicode_homoglyph_detection(self):
        """Test that Unicode homoglyphs don't bypass validation."""
        # Cyrillic 'а' (U+0430) looks like Latin 'a' (U+0061)
        # Using Cyrillic in path should still work (no bypass)
        cyrillic_path = Path("/tmp/dаtаbase.db")  # 'а' is Cyrillic here

        # This should not raise - Cyrillic characters are valid
        # The key is that they shouldn't bypass extension checking
        result = _validate_database_path(cyrillic_path)
        assert result.suffix.lower() == ".db"


class TestControlCharacterPrevention:
    """Test prevention of control character injection."""

    def test_bell_character_rejected(self):
        """Test that bell character is rejected."""
        with pytest.raises(DataLoadingError, match="Control character detected"):
            _validate_database_path(Path("test\x07.db"))

    def test_backspace_rejected(self):
        """Test that backspace character is rejected."""
        with pytest.raises(DataLoadingError, match="Control character detected"):
            _validate_database_path(Path("test\x08.db"))

    def test_escape_rejected(self):
        """Test that escape character is rejected."""
        with pytest.raises(DataLoadingError, match="Control character detected"):
            _validate_database_path(Path("test\x1b.db"))

    def test_tab_in_filename_rejected(self):
        """Test that tab character in filename is rejected."""
        with pytest.raises(DataLoadingError, match="Control character detected"):
            _validate_database_path(Path("test\tfile.db"))

    def test_newline_in_path_rejected(self):
        """Test that newline in path is rejected."""
        with pytest.raises(DataLoadingError, match="Control character detected"):
            _validate_database_path(Path("test\nfile.db"))

    def test_carriage_return_rejected(self):
        """Test that carriage return in path is rejected."""
        with pytest.raises(DataLoadingError, match="Control character detected"):
            _validate_database_path(Path("test\rfile.db"))


class TestExtensionValidation:
    """Test file extension validation."""

    def test_valid_db_extension(self):
        """Test that .db extension is accepted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_file = Path(tmpdir) / "test.db"
            db_file.touch()
            result = _validate_database_path(db_file)
            assert result.suffix == ".db"

    def test_valid_sqlite_extension(self):
        """Test that .sqlite extension is accepted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_file = Path(tmpdir) / "test.sqlite"
            db_file.touch()
            result = _validate_database_path(db_file)
            assert result.suffix == ".sqlite"

    def test_valid_sqlite3_extension(self):
        """Test that .sqlite3 extension is accepted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_file = Path(tmpdir) / "test.sqlite3"
            db_file.touch()
            result = _validate_database_path(db_file)
            assert result.suffix == ".sqlite3"

    def test_invalid_extension_rejected(self):
        """Test that invalid extensions are rejected."""
        with pytest.raises(DataLoadingError, match="Invalid database file extension"):
            _validate_database_path(Path("/tmp/test.txt"))

    def test_double_extension_attack(self):
        """Test that double extension doesn't bypass validation."""
        with pytest.raises(DataLoadingError, match="Invalid database file extension"):
            _validate_database_path(Path("/tmp/malicious.db.exe"))

    def test_case_insensitive_extension(self):
        """Test that extension check is case-insensitive."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_file = Path(tmpdir) / "test.DB"
            db_file.touch()
            result = _validate_database_path(db_file)
            assert result.suffix.lower() == ".db"


class TestSymlinkHandling:
    """Test symlink resolution and validation."""

    def test_symlink_to_valid_db_allowed(self):
        """Test that symlinks to valid .db files are allowed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create actual db file
            real_db = Path(tmpdir) / "real.db"
            real_db.touch()

            # Create symlink to it
            symlink = Path(tmpdir) / "link.db"
            symlink.symlink_to(real_db)

            result = _validate_database_path(symlink)
            assert result == real_db.resolve()

    def test_symlink_loop_handled(self):
        """Test that symlink loops are handled gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            link1 = Path(tmpdir) / "link1.db"
            link2 = Path(tmpdir) / "link2.db"

            # Create symlink loop (if OS allows)
            try:
                link1.symlink_to(link2)
                link2.symlink_to(link1)

                # Should raise an error due to loop
                with pytest.raises(DataLoadingError):
                    _validate_database_path(link1)
            except OSError:
                # Some OS configurations prevent symlink loops
                pytest.skip("OS prevents symlink loops")


class TestValidPaths:
    """Test that valid paths are accepted."""

    def test_absolute_path_accepted(self):
        """Test that valid absolute paths are accepted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_file = Path(tmpdir) / "valid.db"
            db_file.touch()
            result = _validate_database_path(db_file)
            assert result == db_file.resolve()

    def test_relative_path_accepted(self):
        """Test that valid relative paths (without ..) are accepted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = Path(tmpdir) / "subdir"
            subdir.mkdir()
            db_file = subdir / "valid.db"
            db_file.touch()

            # Change to tmpdir and use relative path
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                result = _validate_database_path(Path("subdir/valid.db"))
                assert result.name == "valid.db"
            finally:
                os.chdir(original_cwd)

    def test_path_with_spaces_accepted(self):
        """Test that paths with spaces are accepted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_file = Path(tmpdir) / "valid file.db"
            db_file.touch()
            result = _validate_database_path(db_file)
            assert result.name == "valid file.db"

    def test_path_with_unicode_chars_accepted(self):
        """Test that paths with valid Unicode characters are accepted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_file = Path(tmpdir) / "မြန်မာ.db"  # Myanmar script
            db_file.touch()
            result = _validate_database_path(db_file)
            assert "မြန်မာ" in result.name
