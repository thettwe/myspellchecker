"""Tests for CSVProvider — a testing-only provider.

Tests cover: basic CRUD, error handling for missing/invalid files,
POS and syllable count loading, and statistics.
"""

import csv
import tempfile
from pathlib import Path

import pytest

from myspellchecker.core.exceptions import ProviderError


class TestCSVProviderBasicOperations:
    """Test CSVProvider core operations."""

    def test_csv_provider_empty_file(self):
        """Test CSVProvider with empty CSV file (header only)."""
        from myspellchecker.providers.csv_provider import CSVProvider

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(["syllable", "frequency"])
            path = f.name

        try:
            provider = CSVProvider(path)
            assert provider.is_valid_syllable("anything") is False
            assert provider.get_syllable_frequency("anything") == 0
        finally:
            Path(path).unlink()

    def test_csv_provider_syllable_and_word_methods(self):
        """Test CSVProvider syllable/word validation and frequency."""
        from myspellchecker.providers.csv_provider import CSVProvider

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(["syllable", "frequency"])
            writer.writerow(["a", "10"])
            writer.writerow(["b", "20"])
            writer.writerow(["c", "30"])
            path = f.name

        try:
            provider = CSVProvider(path)
            syllables = list(provider.get_all_syllables())
            assert len(syllables) == 3
            assert provider.get_word_pos("test") is None
            assert provider.get_bigram_probability("a", "b") == 0.0
            assert provider.get_top_continuations("a") == []
        finally:
            Path(path).unlink()

    def test_csv_provider_no_files(self):
        """Test CSVProvider with no CSV files."""
        from myspellchecker.providers.csv_provider import CSVProvider

        provider = CSVProvider()
        assert repr(provider) == "CSVProvider()"


class TestCSVProviderFileErrors:
    """Test CSVProvider file not found and invalid format errors."""

    def test_syllables_csv_not_found(self):
        """Test FileNotFoundError for missing syllables CSV."""
        from myspellchecker.providers.csv_provider import CSVProvider

        with pytest.raises(FileNotFoundError, match="Syllables CSV not found"):
            CSVProvider(syllables_csv="/nonexistent/path/syllables.csv")

    def test_words_csv_not_found(self):
        """Test FileNotFoundError for missing words CSV."""
        from myspellchecker.providers.csv_provider import CSVProvider

        with pytest.raises(FileNotFoundError, match="Words CSV not found"):
            CSVProvider(words_csv="/nonexistent/path/words.csv")

    def test_bigrams_csv_not_found(self):
        """Test FileNotFoundError for missing bigrams CSV."""
        from myspellchecker.providers.csv_provider import CSVProvider

        with pytest.raises(FileNotFoundError, match="Bigrams CSV not found"):
            CSVProvider(bigrams_csv="/nonexistent/path/bigrams.csv")

    def test_invalid_syllables_csv_format(self):
        """Test error with invalid syllables CSV format."""
        from myspellchecker.providers.csv_provider import CSVProvider

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(["wrong_column", "another_wrong"])
            writer.writerow(["test", "100"])
            path = f.name

        try:
            with pytest.raises(ProviderError, match="Invalid Syllables CSV format"):
                CSVProvider(syllables_csv=path)
        finally:
            Path(path).unlink()

    def test_invalid_words_csv_format(self):
        """Test error with invalid words CSV format."""
        from myspellchecker.providers.csv_provider import CSVProvider

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(["wrong_column", "another_wrong"])
            writer.writerow(["test", "100"])
            path = f.name

        try:
            with pytest.raises(ProviderError, match="Invalid Words CSV format"):
                CSVProvider(words_csv=path)
        finally:
            Path(path).unlink()

    def test_invalid_bigrams_csv_format(self):
        """Test error with invalid bigrams CSV format."""
        from myspellchecker.providers.csv_provider import CSVProvider

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(["wrong_column", "another_wrong"])
            writer.writerow(["test", "100"])
            path = f.name

        try:
            with pytest.raises(ProviderError, match="Invalid Bigrams CSV format"):
                CSVProvider(bigrams_csv=path)
        finally:
            Path(path).unlink()


class TestCSVProviderWordsWithPOS:
    """Test CSVProvider with words CSV containing POS and syllable count columns."""

    def test_words_csv_with_pos_column(self):
        """Test loading words CSV with POS tags."""
        from myspellchecker.providers.csv_provider import CSVProvider

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(["word", "frequency", "pos"])
            writer.writerow(["မြန်မာ", "100", "N"])
            writer.writerow(["သွား", "50", "V"])
            writer.writerow(["word2", "50", ""])  # Empty POS
            path = f.name

        try:
            provider = CSVProvider(words_csv=path)
            assert provider.get_word_pos("မြန်မာ") == "N"
            assert provider.get_word_pos("သွား") == "V"
            assert provider.get_word_pos("word2") is None
        finally:
            Path(path).unlink()

    def test_words_csv_with_syllable_count(self):
        """Test loading words CSV with syllable count column."""
        from myspellchecker.providers.csv_provider import CSVProvider

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(["word", "frequency", "syllable_count"])
            writer.writerow(["မြန်မာ", "100", "2"])
            writer.writerow(["ကား", "50", "1"])
            path = f.name

        try:
            provider = CSVProvider(words_csv=path)
            assert provider.get_word_syllable_count("မြန်မာ") == 2
            assert provider.get_word_syllable_count("ကား") == 1
            assert provider.get_word_syllable_count("nonexistent") is None
            assert provider.supports_syllable_count is True
        finally:
            Path(path).unlink()


class TestCSVProviderStatistics:
    """Test CSVProvider statistics method."""

    def test_get_statistics_with_all_files(self):
        """Test get_statistics with all CSV files."""
        from myspellchecker.providers.csv_provider import CSVProvider

        with tempfile.NamedTemporaryFile(mode="w", suffix="_syl.csv", delete=False) as f:
            csv.writer(f).writerow(["syllable", "frequency"])
            csv.writer(f).writerow(["a", "10"])
            syl_path = f.name

        with tempfile.NamedTemporaryFile(mode="w", suffix="_words.csv", delete=False) as f:
            csv.writer(f).writerow(["word", "frequency"])
            csv.writer(f).writerow(["word1", "100"])
            words_path = f.name

        with tempfile.NamedTemporaryFile(mode="w", suffix="_bigrams.csv", delete=False) as f:
            csv.writer(f).writerow(["word1", "word2", "probability"])
            csv.writer(f).writerow(["a", "b", "0.5"])
            bigrams_path = f.name

        try:
            provider = CSVProvider(
                syllables_csv=syl_path, words_csv=words_path, bigrams_csv=bigrams_path
            )
            stats = provider.get_statistics()
            assert "syllable_count" in stats
            assert "word_count" in stats
            assert "bigram_count" in stats
            assert stats["syllables_csv"] is not None
        finally:
            Path(syl_path).unlink()
            Path(words_path).unlink()
            Path(bigrams_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
