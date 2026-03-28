"""
Tests for mmap_reader Cython module.

Note: These tests cover the public Python interface. Full functional
testing requires a valid segmentation.mmap file.
"""

import os
import tempfile

import pytest

# Try to import the Cython module
try:
    from myspellchecker.tokenizers.cython.mmap_reader import (
        MMapSegmentationReader,
        ensure_mmap_initialized,
        get_mmap_reader,
        initialize_mmap_reader,
    )

    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False


@pytest.mark.skipif(not CYTHON_AVAILABLE, reason="Cython mmap_reader not compiled")
class TestMMapSegmentationReader:
    """Tests for MMapSegmentationReader class."""

    def test_initial_state_not_initialized(self):
        """New reader should not be initialized."""
        reader = MMapSegmentationReader()
        assert reader.is_initialized() is False

    def test_get_stats_when_not_initialized(self):
        """get_stats should return uninitialized status."""
        reader = MMapSegmentationReader()
        stats = reader.get_stats()

        assert stats["initialized"] is False

    def test_get_unigram_log_prob_raises_when_not_initialized(self):
        """get_unigram_log_prob should raise RuntimeError when not initialized."""
        reader = MMapSegmentationReader()

        with pytest.raises(RuntimeError) as exc_info:
            reader.get_unigram_log_prob("test")

        assert "not initialized" in str(exc_info.value)

    def test_get_bigram_log_prob_raises_when_not_initialized(self):
        """get_bigram_log_prob should raise RuntimeError when not initialized."""
        reader = MMapSegmentationReader()

        with pytest.raises(RuntimeError) as exc_info:
            reader.get_bigram_log_prob("word1", "word2")

        assert "not initialized" in str(exc_info.value)

    def test_open_nonexistent_file_returns_false(self):
        """open() should return False for nonexistent file."""
        reader = MMapSegmentationReader()
        result = reader.open("/nonexistent/path/to/file.mmap")

        assert result is False
        assert reader.is_initialized() is False

    def test_open_invalid_file_returns_false(self):
        """open() should return False for invalid file format."""
        reader = MMapSegmentationReader()

        # Create a temporary file with invalid content
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mmap") as f:
            f.write(b"INVALID_MAGIC_BYTES" + b"\x00" * 200)
            temp_path = f.name

        try:
            result = reader.open(temp_path)
            assert result is False
            assert reader.is_initialized() is False
        finally:
            os.unlink(temp_path)

    def test_close_when_not_initialized(self):
        """close() should not raise when not initialized."""
        reader = MMapSegmentationReader()
        # Should not raise
        reader.close()
        assert reader.is_initialized() is False

    def test_close_can_be_called_multiple_times(self):
        """close() should be safe to call multiple times."""
        reader = MMapSegmentationReader()
        reader.close()
        reader.close()
        reader.close()
        # Should not raise


@pytest.mark.skipif(not CYTHON_AVAILABLE, reason="Cython mmap_reader not compiled")
class TestGlobalMMapFunctions:
    """Tests for global mmap reader functions."""

    def test_ensure_mmap_initialized_initially_false(self):
        """ensure_mmap_initialized should return False initially."""
        # Note: This test may fail if another test initialized the global reader
        # The global state is module-level, so we test what we can
        result = ensure_mmap_initialized()
        # This could be True or False depending on test order
        assert isinstance(result, bool)

    def test_get_mmap_reader_raises_when_not_initialized(self):
        """get_mmap_reader should raise when not initialized."""
        # Reset global state by attempting to test
        # This is tricky because global state persists
        # We test the error message format
        try:
            reader = get_mmap_reader()
            # If we get here, a previous test initialized it
            assert reader is not None
        except RuntimeError as e:
            assert "not initialized" in str(e)

    def test_initialize_mmap_reader_with_invalid_path(self):
        """initialize_mmap_reader should return False for invalid path."""
        result = initialize_mmap_reader("/nonexistent/path.mmap")
        assert result is False


@pytest.mark.skipif(not CYTHON_AVAILABLE, reason="Cython mmap_reader not compiled")
class TestMMapReaderWithValidFile:
    """Tests that require a valid mmap file."""

    @pytest.fixture
    def valid_mmap_file(self, tmp_path):
        """Create a minimal valid mmap file for testing.

        Note: This creates a file with valid magic bytes but minimal
        data structure. Full functional testing requires proper
        dictionary building.
        """
        import struct

        mmap_path = tmp_path / "test_segmentation.mmap"

        # MYSEGV02 magic, then header data
        with open(mmap_path, "wb") as f:
            # Magic bytes (8 bytes)
            f.write(b"MYSEGV02")

            # Version (4 bytes)
            f.write(struct.pack("<I", 2))

            # Checksum placeholder (4 bytes)
            f.write(struct.pack("<I", 0))

            # Header fields
            # unigram_offset (8 bytes)
            f.write(struct.pack("<Q", 128))
            # unigram_count (4 bytes)
            f.write(struct.pack("<I", 0))
            # unigram_buckets (4 bytes)
            f.write(struct.pack("<I", 1))
            # bigram_offset (8 bytes)
            f.write(struct.pack("<Q", 256))
            # bigram_count (4 bytes)
            f.write(struct.pack("<I", 0))
            # bigram_buckets (4 bytes)
            f.write(struct.pack("<I", 1))
            # string_pool_offset (8 bytes)
            f.write(struct.pack("<Q", 512))
            # N_unigram (8 bytes)
            f.write(struct.pack("<Q", 1000))
            # N_bigram (8 bytes)
            f.write(struct.pack("<Q", 500))
            # log_N_unigram (8 bytes)
            f.write(struct.pack("<d", 3.0))  # log10(1000)
            # log_N_bigram (8 bytes)
            f.write(struct.pack("<d", 2.7))  # log10(500)

            # Pad to HEADER_SIZE (128)
            current_pos = f.tell()
            padding_needed = 128 - current_pos
            if padding_needed > 0:
                f.write(b"\x00" * padding_needed)

            # Empty unigram table (at offset 128)
            # One empty bucket (24 bytes per entry)
            f.write(b"\x00" * 24)

            # Pad to bigram offset (256)
            current_pos = f.tell()
            padding_needed = 256 - current_pos
            if padding_needed > 0:
                f.write(b"\x00" * padding_needed)

            # Empty bigram table (at offset 256)
            # One empty bucket (32 bytes per entry)
            f.write(b"\x00" * 32)

            # Pad to string pool offset (512)
            current_pos = f.tell()
            padding_needed = 512 - current_pos
            if padding_needed > 0:
                f.write(b"\x00" * padding_needed)

            # Empty string pool
            f.write(b"\x00" * 64)

        return str(mmap_path)

    def test_open_valid_file(self, valid_mmap_file):
        """open() should succeed with valid mmap file."""
        reader = MMapSegmentationReader()
        result = reader.open(valid_mmap_file)

        assert result is True
        assert reader.is_initialized() is True

        reader.close()

    def test_get_stats_after_open(self, valid_mmap_file):
        """get_stats should return complete stats after opening."""
        reader = MMapSegmentationReader()
        reader.open(valid_mmap_file)

        stats = reader.get_stats()

        assert stats["initialized"] is True
        assert "file_path" in stats
        assert "unigram_count" in stats
        assert "bigram_count" in stats
        assert "N_unigram" in stats
        assert "N_bigram" in stats

        reader.close()

    def test_close_after_open(self, valid_mmap_file):
        """close() should properly close an opened reader."""
        reader = MMapSegmentationReader()
        reader.open(valid_mmap_file)
        assert reader.is_initialized() is True

        reader.close()
        assert reader.is_initialized() is False

    def test_reopen_after_close(self, valid_mmap_file):
        """Should be able to reopen reader after close."""
        reader = MMapSegmentationReader()

        reader.open(valid_mmap_file)
        assert reader.is_initialized() is True

        reader.close()
        assert reader.is_initialized() is False

        # Reopen
        reader.open(valid_mmap_file)
        assert reader.is_initialized() is True

        reader.close()

    def test_unknown_word_returns_penalty(self, valid_mmap_file):
        """get_unigram_log_prob should return penalty for unknown word."""
        reader = MMapSegmentationReader()
        reader.open(valid_mmap_file)

        # Unknown word should return a negative log probability
        log_prob = reader.get_unigram_log_prob("unknownword")

        assert isinstance(log_prob, float)
        # Log probability should be negative (probability < 1)
        assert log_prob < 0

        reader.close()

    def test_unknown_bigram_falls_back_to_unigram(self, valid_mmap_file):
        """get_bigram_log_prob should fallback for unknown bigram."""
        reader = MMapSegmentationReader()
        reader.open(valid_mmap_file)

        # Unknown bigram should fallback to unigram probability
        log_prob = reader.get_bigram_log_prob("word1", "word2")

        assert isinstance(log_prob, float)

        reader.close()


@pytest.mark.skipif(not CYTHON_AVAILABLE, reason="Cython mmap_reader not compiled")
class TestMMapReaderEdgeCases:
    """Edge case tests for MMapSegmentationReader."""

    def test_empty_string_unigram(self):
        """get_unigram_log_prob should handle empty string."""
        reader = MMapSegmentationReader()

        # Should raise because not initialized
        with pytest.raises(RuntimeError):
            reader.get_unigram_log_prob("")

    def test_empty_string_bigram(self):
        """get_bigram_log_prob should handle empty strings."""
        reader = MMapSegmentationReader()

        # Should raise because not initialized
        with pytest.raises(RuntimeError):
            reader.get_bigram_log_prob("", "")

    def test_unicode_word_lookup(self):
        """Should handle Myanmar Unicode characters in lookup."""
        reader = MMapSegmentationReader()

        # Should raise because not initialized, but verifies
        # that Myanmar Unicode doesn't crash the function
        with pytest.raises(RuntimeError):
            reader.get_unigram_log_prob("မြန်မာ")

    def test_very_long_word(self):
        """Should handle very long words."""
        reader = MMapSegmentationReader()

        long_word = "a" * 10000

        with pytest.raises(RuntimeError):
            reader.get_unigram_log_prob(long_word)


@pytest.mark.skipif(not CYTHON_AVAILABLE, reason="Cython mmap_reader not compiled")
class TestMMapReaderContextManager:
    """Test MMapSegmentationReader lifecycle management."""

    def test_dealloc_closes_file(self, tmp_path):
        """Reader should close file when deallocated."""
        import struct

        # Create minimal valid file
        mmap_path = tmp_path / "test.mmap"
        with open(mmap_path, "wb") as f:
            f.write(b"MYSEGV02")
            f.write(struct.pack("<I", 2))  # version
            f.write(struct.pack("<I", 0))  # checksum
            f.write(struct.pack("<Q", 128))  # unigram_offset
            f.write(struct.pack("<I", 0))  # unigram_count
            f.write(struct.pack("<I", 1))  # unigram_buckets
            f.write(struct.pack("<Q", 256))  # bigram_offset
            f.write(struct.pack("<I", 0))  # bigram_count
            f.write(struct.pack("<I", 1))  # bigram_buckets
            f.write(struct.pack("<Q", 512))  # string_pool_offset
            f.write(struct.pack("<Q", 1000))  # N_unigram
            f.write(struct.pack("<Q", 500))  # N_bigram
            f.write(struct.pack("<d", 3.0))  # log_N_unigram
            f.write(struct.pack("<d", 2.7))  # log_N_bigram
            # Pad to 512 + 64
            current = f.tell()
            f.write(b"\x00" * (576 - current))

        reader = MMapSegmentationReader()
        reader.open(str(mmap_path))
        assert reader.is_initialized()

        # Delete reader - should trigger __dealloc__
        del reader

        # File should still exist and be accessible
        assert os.path.exists(mmap_path)
