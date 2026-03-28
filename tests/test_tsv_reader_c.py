"""Tests for tsv_reader_c Cython module."""

import os

import pytest

# Try to import the Cython module
try:
    from myspellchecker.data_pipeline.tsv_reader_c import (
        read_bigrams_tsv,
        read_syllables_tsv,
        read_trigrams_tsv,
        read_words_tsv,
    )

    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False


@pytest.mark.skipif(not CYTHON_AVAILABLE, reason="Cython tsv_reader_c not compiled")
class TestReadSyllablesTsv:
    """Tests for read_syllables_tsv function."""

    def test_read_valid_syllables_file(self, tmp_path):
        """Should read valid syllables TSV file."""
        tsv_file = tmp_path / "syllables.tsv"
        content = "syllable\tfrequency\n"
        content += "test\t1000\n"
        content += "word\t500\n"
        content += "three\t250\n"
        tsv_file.write_text(content, encoding="utf-8")

        result = list(read_syllables_tsv(str(tsv_file)))

        assert len(result) == 3
        assert result[0] == ("test", 1000)
        assert result[1] == ("word", 500)
        assert result[2] == ("three", 250)

    def test_read_file_not_found(self):
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            list(read_syllables_tsv("/nonexistent/path/file.tsv"))

    def test_read_empty_file(self, tmp_path):
        """Should handle empty file (header only)."""
        tsv_file = tmp_path / "empty.tsv"
        tsv_file.write_text("syllable\tfrequency\n", encoding="utf-8")

        result = list(read_syllables_tsv(str(tsv_file)))

        assert result == []

    def test_skip_malformed_lines(self, tmp_path):
        """Should skip lines without tab separator."""
        tsv_file = tmp_path / "syllables.tsv"
        content = "syllable\tfrequency\n"
        content += "valid\t1000\n"
        content += "malformed_no_tab\n"
        content += "also_valid\t500\n"
        tsv_file.write_text(content, encoding="utf-8")

        result = list(read_syllables_tsv(str(tsv_file)))

        assert len(result) == 2

    def test_skip_invalid_frequency(self, tmp_path):
        """Should skip lines with non-integer frequency."""
        tsv_file = tmp_path / "syllables.tsv"
        content = "syllable\tfrequency\n"
        content += "valid\t1000\n"
        content += "bad\tnot_a_number\n"
        content += "ok\t500\n"
        tsv_file.write_text(content, encoding="utf-8")

        result = list(read_syllables_tsv(str(tsv_file)))

        assert len(result) == 2

    def test_skip_empty_lines(self, tmp_path):
        """Should skip empty lines."""
        tsv_file = tmp_path / "syllables.tsv"
        content = "syllable\tfrequency\n"
        content += "one\t1000\n"
        content += "\n"
        content += "two\t500\n"
        tsv_file.write_text(content, encoding="utf-8")

        result = list(read_syllables_tsv(str(tsv_file)))

        assert len(result) == 2

    def test_handle_large_frequency(self, tmp_path):
        """Should handle large frequency values."""
        tsv_file = tmp_path / "syllables.tsv"
        content = "syllable\tfrequency\n"
        content += "big\t1000000000\n"
        tsv_file.write_text(content, encoding="utf-8")

        result = list(read_syllables_tsv(str(tsv_file)))

        assert result[0] == ("big", 1000000000)


@pytest.mark.skipif(not CYTHON_AVAILABLE, reason="Cython tsv_reader_c not compiled")
class TestReadWordsTsv:
    """Tests for read_words_tsv function."""

    def test_read_valid_words_file_3_columns(self, tmp_path):
        """Should read valid 3-column words TSV file."""
        tsv_file = tmp_path / "words.tsv"
        content = "word\tsyllable_count\tfrequency\n"
        content += "hello\t2\t1000\n"
        content += "world\t1\t500\n"
        tsv_file.write_text(content, encoding="utf-8")

        result = list(read_words_tsv(str(tsv_file)))

        assert len(result) == 2
        assert result[0] == ("hello", 2, 1000, "")
        assert result[1] == ("world", 1, 500, "")

    def test_read_valid_words_file_4_columns(self, tmp_path):
        """Should read valid 4-column words TSV file with POS tags."""
        tsv_file = tmp_path / "words.tsv"
        content = "word\tsyllable_count\tfrequency\tpos_tag\n"
        content += "hello\t2\t1000\tNOUN\n"
        content += "run\t1\t500\tVERB\n"
        tsv_file.write_text(content, encoding="utf-8")

        result = list(read_words_tsv(str(tsv_file)))

        assert len(result) == 2
        assert result[0] == ("hello", 2, 1000, "NOUN")
        assert result[1] == ("run", 1, 500, "VERB")

    def test_read_file_not_found(self):
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            list(read_words_tsv("/nonexistent/path/file.tsv"))

    def test_skip_malformed_lines(self, tmp_path):
        """Should skip lines with insufficient columns."""
        tsv_file = tmp_path / "words.tsv"
        content = "word\tsyllable_count\tfrequency\n"
        content += "valid\t2\t1000\n"
        content += "only_one_tab\t100\n"
        content += "also_valid\t2\t500\n"
        tsv_file.write_text(content, encoding="utf-8")

        result = list(read_words_tsv(str(tsv_file)))

        assert len(result) == 2

    def test_skip_invalid_values(self, tmp_path):
        """Should skip lines with invalid numeric values."""
        tsv_file = tmp_path / "words.tsv"
        content = "word\tsyllable_count\tfrequency\n"
        content += "valid\t2\t1000\n"
        content += "bad\tnot_int\t500\n"
        content += "also_bad\t2\tnot_int\n"
        content += "ok\t2\t500\n"
        tsv_file.write_text(content, encoding="utf-8")

        result = list(read_words_tsv(str(tsv_file)))

        assert len(result) == 2


@pytest.mark.skipif(not CYTHON_AVAILABLE, reason="Cython tsv_reader_c not compiled")
class TestReadBigramsTsv:
    """Tests for read_bigrams_tsv function."""

    def test_read_valid_bigrams_file(self, tmp_path):
        """Should read valid bigrams TSV file."""
        tsv_file = tmp_path / "bigrams.tsv"
        content = "w1\tw2\tprob\tcount\n"
        content += "word1\tword2\t0.5\t100\n"
        content += "word2\tword3\t0.3\t50\n"
        tsv_file.write_text(content, encoding="utf-8")

        word_to_id = {"word1": 1, "word2": 2, "word3": 3}
        result = list(read_bigrams_tsv(str(tsv_file), word_to_id))

        assert len(result) == 2
        assert result[0] == (1, 2, 0.5, 100)
        assert result[1] == (2, 3, 0.3, 50)

    def test_read_file_not_found(self):
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            list(read_bigrams_tsv("/nonexistent/path/file.tsv", {}))

    def test_skip_unknown_words(self, tmp_path):
        """Should skip bigrams with words not in word_to_id."""
        tsv_file = tmp_path / "bigrams.tsv"
        content = "w1\tw2\tprob\tcount\n"
        content += "known\tunknown\t0.5\t100\n"
        content += "word1\tword2\t0.3\t50\n"
        tsv_file.write_text(content, encoding="utf-8")

        word_to_id = {"word1": 1, "word2": 2, "known": 3}
        result = list(read_bigrams_tsv(str(tsv_file), word_to_id))

        assert len(result) == 1
        assert result[0] == (1, 2, 0.3, 50)

    def test_read_without_count_column(self, tmp_path):
        """Should handle 3-column format (without count)."""
        tsv_file = tmp_path / "bigrams.tsv"
        content = "w1\tw2\tprob\n"
        content += "word1\tword2\t0.5\n"
        tsv_file.write_text(content, encoding="utf-8")

        word_to_id = {"word1": 1, "word2": 2}
        result = list(read_bigrams_tsv(str(tsv_file), word_to_id))

        assert len(result) == 1
        assert result[0][0] == 1
        assert result[0][1] == 2
        assert result[0][2] == 0.5

    def test_empty_word_to_id(self, tmp_path):
        """Should skip all entries when word_to_id is empty."""
        tsv_file = tmp_path / "bigrams.tsv"
        content = "w1\tw2\tprob\tcount\n"
        content += "word1\tword2\t0.5\t100\n"
        tsv_file.write_text(content, encoding="utf-8")

        result = list(read_bigrams_tsv(str(tsv_file), {}))

        assert result == []


@pytest.mark.skipif(not CYTHON_AVAILABLE, reason="Cython tsv_reader_c not compiled")
class TestReadTrigramsTsv:
    """Tests for read_trigrams_tsv function."""

    def test_read_valid_trigrams_file(self, tmp_path):
        """Should read valid trigrams TSV file."""
        tsv_file = tmp_path / "trigrams.tsv"
        content = "w1\tw2\tw3\tprob\tcount\n"
        content += "word1\tword2\tword3\t0.5\t100\n"
        content += "word2\tword3\tword4\t0.3\t50\n"
        tsv_file.write_text(content, encoding="utf-8")

        word_to_id = {"word1": 1, "word2": 2, "word3": 3, "word4": 4}
        result = list(read_trigrams_tsv(str(tsv_file), word_to_id))

        assert len(result) == 2
        assert result[0] == (1, 2, 3, 0.5, 100)
        assert result[1] == (2, 3, 4, 0.3, 50)

    def test_read_file_not_found(self):
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            list(read_trigrams_tsv("/nonexistent/path/file.tsv", {}))

    def test_skip_unknown_words(self, tmp_path):
        """Should skip trigrams with words not in word_to_id."""
        tsv_file = tmp_path / "trigrams.tsv"
        content = "w1\tw2\tw3\tprob\tcount\n"
        content += "known\tunknown\tword3\t0.5\t100\n"
        content += "word1\tword2\tword3\t0.3\t50\n"
        tsv_file.write_text(content, encoding="utf-8")

        word_to_id = {"word1": 1, "word2": 2, "word3": 3, "known": 4}
        result = list(read_trigrams_tsv(str(tsv_file), word_to_id))

        assert len(result) == 1
        assert result[0] == (1, 2, 3, 0.3, 50)

    def test_read_without_count_column(self, tmp_path):
        """Should handle 4-column format (without count)."""
        tsv_file = tmp_path / "trigrams.tsv"
        content = "w1\tw2\tw3\tprob\n"
        content += "word1\tword2\tword3\t0.5\n"
        tsv_file.write_text(content, encoding="utf-8")

        word_to_id = {"word1": 1, "word2": 2, "word3": 3}
        result = list(read_trigrams_tsv(str(tsv_file), word_to_id))

        assert len(result) == 1
        assert result[0][0] == 1
        assert result[0][1] == 2
        assert result[0][2] == 3
        assert result[0][3] == 0.5


@pytest.mark.skipif(not CYTHON_AVAILABLE, reason="Cython tsv_reader_c not compiled")
class TestTsvReaderEdgeCases:
    """Edge case tests for TSV reader functions."""

    def test_syllables_with_unicode(self, tmp_path):
        """Should handle various Myanmar Unicode characters."""
        tsv_file = tmp_path / "syllables.tsv"
        content = "syllable\tfrequency\n"
        content += "\u1000\u103a\u103d\u1014\u103a\t100\n"
        content += "\u1000\u102d\u102f\t200\n"
        tsv_file.write_text(content, encoding="utf-8")

        result = list(read_syllables_tsv(str(tsv_file)))

        assert len(result) == 2

    def test_large_file(self, tmp_path):
        """Should handle large TSV files efficiently."""
        tsv_file = tmp_path / "large.tsv"
        lines = ["syllable\tfrequency"]
        for i in range(10000):
            lines.append(f"syl{i}\t{i}")
        tsv_file.write_text("\n".join(lines), encoding="utf-8")

        result = list(read_syllables_tsv(str(tsv_file)))

        assert len(result) == 10000

    def test_words_with_empty_pos_tag(self, tmp_path):
        """Should handle 4-column format with empty POS tag."""
        tsv_file = tmp_path / "words.tsv"
        content = "word\tsyllable_count\tfrequency\tpos_tag\n"
        content += "test\t2\t1000\t\n"
        tsv_file.write_text(content, encoding="utf-8")

        result = list(read_words_tsv(str(tsv_file)))

        assert len(result) == 1
        assert result[0][3] == ""

    def test_bigrams_with_special_characters_in_words(self, tmp_path):
        """Should handle words with special characters."""
        tsv_file = tmp_path / "bigrams.tsv"
        content = "w1\tw2\tprob\tcount\n"
        content += "word-1\tword_2\t0.5\t100\n"
        tsv_file.write_text(content, encoding="utf-8")

        word_to_id = {"word-1": 1, "word_2": 2}
        result = list(read_bigrams_tsv(str(tsv_file), word_to_id))

        assert len(result) == 1
        assert result[0] == (1, 2, 0.5, 100)

    def test_reader_closes_file_properly(self, tmp_path):
        """Reader should close file after reading."""
        tsv_file = tmp_path / "syllables.tsv"
        content = "syllable\tfrequency\n"
        content += "test\t100\n"
        tsv_file.write_text(content, encoding="utf-8")

        list(read_syllables_tsv(str(tsv_file)))

        os.unlink(tsv_file)
        assert not os.path.exists(tsv_file)

    def test_zero_frequency(self, tmp_path):
        """Should handle zero frequency values."""
        tsv_file = tmp_path / "syllables.tsv"
        content = "syllable\tfrequency\n"
        content += "zero\t0\n"
        tsv_file.write_text(content, encoding="utf-8")

        result = list(read_syllables_tsv(str(tsv_file)))

        assert len(result) == 1
        assert result[0] == ("zero", 0)

    def test_float_probability(self, tmp_path):
        """Should handle float probability values correctly."""
        tsv_file = tmp_path / "bigrams.tsv"
        content = "w1\tw2\tprob\tcount\n"
        content += "a\tb\t0.123456789\t100\n"
        content += "c\td\t1e-10\t50\n"
        tsv_file.write_text(content, encoding="utf-8")

        word_to_id = {"a": 1, "b": 2, "c": 3, "d": 4}
        result = list(read_bigrams_tsv(str(tsv_file), word_to_id))

        assert len(result) == 2
        assert abs(result[0][2] - 0.123456789) < 1e-9
        assert abs(result[1][2] - 1e-10) < 1e-15
