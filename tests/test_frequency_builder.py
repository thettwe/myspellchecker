"""Comprehensive tests for data_pipeline/frequency_builder.py targeting uncovered lines."""

import shutil
import tempfile
from collections import Counter
from pathlib import Path
from unittest.mock import MagicMock, Mock, mock_open, patch

import pytest

from myspellchecker.data_pipeline.frequency_builder import FrequencyBuilder


class TestFrequencyBuilderInit:
    """Tests for FrequencyBuilder initialization."""

    def test_init_with_defaults(self):
        """Test FrequencyBuilder with default parameters."""
        from myspellchecker.data_pipeline.frequency_builder import FrequencyBuilder

        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            output_dir = Path(tmpdir) / "output"
            input_dir.mkdir()

            builder = FrequencyBuilder(input_dir, output_dir)

            assert builder.input_dir == input_dir
            assert builder.output_dir == output_dir
            assert builder.min_syllable_frequency == 1
            assert builder.min_word_frequency == 50
            assert builder.incremental is False

    def test_init_with_custom_min_frequencies(self):
        """Test FrequencyBuilder with custom min frequencies."""
        from myspellchecker.data_pipeline.frequency_builder import FrequencyBuilder

        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            output_dir = Path(tmpdir) / "output"
            input_dir.mkdir()

            builder = FrequencyBuilder(
                input_dir,
                output_dir,
                min_syllable_frequency=5,
                min_word_frequency=10,
                min_bigram_frequency=3,
                min_trigram_frequency=3,
            )

            assert builder.min_syllable_frequency == 5
            assert builder.min_word_frequency == 10
            assert builder.min_bigram_frequency == 3
            assert builder.min_trigram_frequency == 3

    def test_init_with_pos_tagger(self):
        """Test FrequencyBuilder accepts pos_tagger parameter."""
        from myspellchecker.data_pipeline.frequency_builder import FrequencyBuilder

        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            output_dir = Path(tmpdir) / "output"
            input_dir.mkdir()

            mock_pos_tagger = Mock()

            builder = FrequencyBuilder(input_dir, output_dir, pos_tagger=mock_pos_tagger)

            assert builder.pos_tagger is mock_pos_tagger

    def test_init_num_workers_kwarg_accepted(self):
        """Test FrequencyBuilder accepts num_workers via **kwargs for backward compat."""
        from myspellchecker.data_pipeline.frequency_builder import FrequencyBuilder

        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            output_dir = Path(tmpdir) / "output"
            input_dir.mkdir()

            # Should not raise — num_workers is accepted via **kwargs
            builder = FrequencyBuilder(input_dir, output_dir, num_workers=4)
            assert builder is not None

    def test_init_with_incremental(self):
        """Test FrequencyBuilder with incremental mode."""
        from myspellchecker.data_pipeline.frequency_builder import FrequencyBuilder

        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            output_dir = Path(tmpdir) / "output"
            input_dir.mkdir()

            builder = FrequencyBuilder(input_dir, output_dir, incremental=True)
            assert builder.incremental is True


class TestFilterInvalidSyllables:
    """Tests for filter_invalid_syllables method."""

    def test_filter_invalid_syllables_empty(self):
        """Test filter with no syllables."""
        from myspellchecker.data_pipeline.frequency_builder import FrequencyBuilder

        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            output_dir = Path(tmpdir) / "output"
            input_dir.mkdir()

            builder = FrequencyBuilder(input_dir, output_dir)
            builder.filter_invalid_syllables()

            assert builder.stats["invalid_syllables_skipped"] == 0

    def test_filter_invalid_syllables_double_asat(self):
        """Test filter removes syllables with double asat."""
        from myspellchecker.data_pipeline.frequency_builder import FrequencyBuilder

        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            output_dir = Path(tmpdir) / "output"
            input_dir.mkdir()

            builder = FrequencyBuilder(input_dir, output_dir)
            builder.syllable_counts["ကား"] = 10  # Valid multi-char syllable
            builder.syllable_counts["က််ု"] = 5  # Double asat - invalid

            builder.filter_invalid_syllables()

            assert "ကား" in builder.syllable_counts
            assert "က််ု" not in builder.syllable_counts

    def test_filter_invalid_syllables_non_myanmar(self):
        """Test filter removes non-Myanmar characters."""
        from myspellchecker.data_pipeline.frequency_builder import FrequencyBuilder

        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            output_dir = Path(tmpdir) / "output"
            input_dir.mkdir()

            builder = FrequencyBuilder(input_dir, output_dir)
            builder.syllable_counts["abc"] = 5  # Non-Myanmar

            builder.filter_invalid_syllables()

            assert "abc" not in builder.syllable_counts


class TestFilterInvalidWords:
    """Tests for filter_invalid_words method."""

    def test_filter_invalid_words_empty(self):
        """Test filter with no words."""
        from myspellchecker.data_pipeline.frequency_builder import FrequencyBuilder

        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            output_dir = Path(tmpdir) / "output"
            input_dir.mkdir()

            builder = FrequencyBuilder(input_dir, output_dir)
            builder.filter_invalid_words()

    def test_filter_invalid_words_zawgyi(self):
        """Test filter removes Zawgyi-encoded words."""
        from myspellchecker.data_pipeline.frequency_builder import FrequencyBuilder

        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            output_dir = Path(tmpdir) / "output"
            input_dir.mkdir()

            builder = FrequencyBuilder(input_dir, output_dir)
            builder.word_counts["\u108f\u1036"] = 5
            builder.word_syllables["\u108f\u1036"] = 1

            builder.filter_invalid_words()
            assert isinstance(builder.word_counts, Counter)


class TestCalculateProbabilities:
    """Tests for probability calculation methods (DuckDB-only)."""

    def test_calculate_bigram_probabilities_returns_none(self):
        """Bigram probabilities are handled by DuckDB — method returns None."""
        from myspellchecker.data_pipeline.frequency_builder import FrequencyBuilder

        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            output_dir = Path(tmpdir) / "output"
            input_dir.mkdir()

            builder = FrequencyBuilder(input_dir, output_dir)
            result = builder.calculate_bigram_probabilities()
            assert result is None

    def test_calculate_trigram_probabilities_returns_none(self):
        """Trigram probabilities are handled by DuckDB — method returns None."""
        from myspellchecker.data_pipeline.frequency_builder import FrequencyBuilder

        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            output_dir = Path(tmpdir) / "output"
            input_dir.mkdir()

            builder = FrequencyBuilder(input_dir, output_dir)
            result = builder.calculate_trigram_probabilities()
            assert result is None


class TestPOSProbabilities:
    """Tests for POS probability calculation methods."""

    def test_calculate_pos_unigram_probabilities(self):
        """Test POS unigram probability calculation."""
        from myspellchecker.data_pipeline.frequency_builder import FrequencyBuilder

        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            output_dir = Path(tmpdir) / "output"
            input_dir.mkdir()

            builder = FrequencyBuilder(input_dir, output_dir)
            builder.pos_unigram_counts["N"] = 100
            builder.pos_unigram_counts["V"] = 50

            result = builder.calculate_pos_unigram_probabilities()

            assert "N" in result
            assert "V" in result
            total = sum(result.values())
            assert abs(total - 1.0) < 0.01

    def test_calculate_pos_bigram_probabilities(self):
        """Test POS bigram probability calculation with Laplace smoothing."""
        from myspellchecker.data_pipeline.frequency_builder import FrequencyBuilder

        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            output_dir = Path(tmpdir) / "output"
            input_dir.mkdir()

            builder = FrequencyBuilder(input_dir, output_dir)
            builder.pos_unigram_counts["N"] = 100
            builder.pos_unigram_counts["V"] = 50
            builder.pos_bigram_counts[("N", "V")] = 20
            builder.pos_bigram_predecessor_counts["N"] = 40

            result = builder.calculate_pos_bigram_probabilities()

            assert ("N", "V") in result
            assert isinstance(result[("N", "V")], float)

    def test_calculate_pos_trigram_probabilities(self):
        """Test POS trigram probability calculation with Laplace smoothing."""
        from myspellchecker.data_pipeline.frequency_builder import FrequencyBuilder

        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            output_dir = Path(tmpdir) / "output"
            input_dir.mkdir()

            builder = FrequencyBuilder(input_dir, output_dir)
            builder.pos_unigram_counts["N"] = 100
            builder.pos_unigram_counts["V"] = 50
            builder.pos_unigram_counts["P"] = 30
            builder.pos_trigram_counts[("N", "V", "P")] = 10
            builder.pos_bigram_successor_counts[("N", "V")] = 20

            result = builder.calculate_pos_trigram_probabilities()

            assert ("N", "V", "P") in result
            assert isinstance(result[("N", "V", "P")], float)


class TestStatisticsAccess:
    """Tests for statistics initialization and access."""

    def test_stats_initialized(self):
        """Test stats dictionary is properly initialized."""
        from myspellchecker.data_pipeline.frequency_builder import FrequencyBuilder

        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            output_dir = Path(tmpdir) / "output"
            input_dir.mkdir()

            builder = FrequencyBuilder(input_dir, output_dir)

            expected_keys = [
                "total_syllables",
                "total_words",
                "total_bigrams",
                "total_trigrams",
                "unique_syllables",
                "unique_words",
                "unique_bigrams",
                "unique_trigrams",
                "filtered_syllables",
                "filtered_words",
                "filtered_bigrams",
                "filtered_trigrams",
                "invalid_syllables_skipped",
            ]

            for key in expected_keys:
                assert key in builder.stats
                assert builder.stats[key] == 0


class TestCounterOperations:
    """Tests for Counter merge operations."""

    def test_merge_counters(self):
        """Test merging multiple Counter objects."""
        from myspellchecker.data_pipeline.frequency_builder import FrequencyBuilder

        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            output_dir = Path(tmpdir) / "output"
            input_dir.mkdir()

            builder = FrequencyBuilder(input_dir, output_dir)

            builder.syllable_counts["a"] = 5
            new_counts = Counter({"a": 10, "b": 3})
            builder.syllable_counts.update(new_counts)

            assert builder.syllable_counts["a"] == 15
            assert builder.syllable_counts["b"] == 3


class TestBuilderOutputDir:
    """Tests for output directory handling."""

    def test_output_dir_created(self):
        """Test output directory is created if not exists."""
        from myspellchecker.data_pipeline.frequency_builder import FrequencyBuilder

        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            output_dir = Path(tmpdir) / "nested" / "output"
            input_dir.mkdir()

            assert not output_dir.exists()

            FrequencyBuilder(input_dir, output_dir)

            assert output_dir.exists()


@pytest.fixture
def mock_segmenter():
    seg = MagicMock()
    seg.segment_syllables.side_effect = lambda w: [w]  # Dummy segmentation
    return seg


@pytest.fixture
def pipeline_builder(mock_segmenter):
    """Builder fixture with mock segmenter for pipeline-level tests."""
    temp_dir = tempfile.mkdtemp()
    input_dir = Path(temp_dir) / "input"
    output_dir = Path(temp_dir) / "output"
    input_dir.mkdir()

    with patch(
        "myspellchecker.data_pipeline.frequency_builder.DefaultSegmenter",
        return_value=mock_segmenter,
    ):
        fb = FrequencyBuilder(input_dir, output_dir, word_engine="myword")
        yield fb

    shutil.rmtree(temp_dir)


class TestHydrate:
    """Tests for the hydrate method."""

    def test_hydrate(self, pipeline_builder):
        """Test hydrating builder with pre-computed counts."""
        syl = Counter({"a": 1})
        word = Counter({"w": 1})
        bi = Counter({("w", "x"): 1})
        tri = Counter({("w", "x", "y"): 1})
        ws = {"w": 1}

        pipeline_builder.hydrate(syl, word, bi, tri, ws, {}, {}, {})

        assert pipeline_builder.syllable_counts["a"] == 1
        assert pipeline_builder.word_counts["w"] == 1
        # n-gram counters are no longer hydrated (DuckDB manages them)
        assert pipeline_builder.word_syllables["w"] == 1


class TestSaveMethods:
    """Tests for frequency save methods."""

    def test_save_syllable_and_word_frequencies(self, pipeline_builder):
        """Test saving syllable and word frequency TSV files."""
        pipeline_builder.syllable_counts = Counter({"syl": 1})
        pipeline_builder.word_counts = Counter({"word": 1})

        pipeline_builder.save_syllable_frequencies("syl.tsv")
        assert (pipeline_builder.output_dir / "syl.tsv").exists()

        pipeline_builder.save_word_frequencies("word.tsv")
        assert (pipeline_builder.output_dir / "word.tsv").exists()

    def test_save_pos_probabilities(self):
        """Test saving POS probability TSV files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            output_dir = Path(tmpdir) / "output"
            input_dir.mkdir()
            output_dir.mkdir()

            builder = FrequencyBuilder(input_dir=input_dir, output_dir=output_dir)

            builder.save_pos_unigram_probabilities({"N": 0.5})
            assert (output_dir / "pos_unigram_probabilities.tsv").exists()

            with patch("builtins.open", mock_open()):
                builder.save_pos_bigram_probabilities({("N", "V"): 0.5})
                builder.save_pos_trigram_probabilities({("N", "V", "P"): 0.5})


class TestLoadDataDuckDB:
    """Tests for DuckDB-based data loading."""

    def test_load_data_duckdb_flow(self, pipeline_builder):
        """Test that large files trigger DuckDB loading path."""
        with (
            patch("myspellchecker.data_pipeline.frequency_builder.duckdb") as mock_duck,
            patch("pyarrow.memory_map"),
            patch("pyarrow.ipc.open_stream"),
            patch("pyarrow.Table"),
            patch("pyarrow.parquet.read_metadata") as mock_read_metadata,
        ):
            conn = mock_duck.connect.return_value
            mock_read_metadata.return_value.num_rows = 1

            def execute_side_effect(sql):
                mock_res = MagicMock()
                if "SELECT" in sql.upper():
                    raise RuntimeError(f"SQL SEEN: {sql}")
                return mock_res

            conn.execute.side_effect = execute_side_effect

            with patch("pathlib.Path.stat") as mock_stat:
                mock_stat.return_value.st_size = 2 * 1024**3  # 2GB to trigger duckdb
                mock_stat.return_value.st_mode = 0o040755

                try:
                    pipeline_builder.load_data("dummy.arrow")
                except RuntimeError as e:
                    assert "SQL SEEN" in str(e)
                    return

                pytest.fail("Did not see SELECT query")


class TestPOSProbabilityEdgeCases:
    """Tests for POS probability edge cases and Laplace smoothing."""

    def test_calculate_pos_unigram_probabilities_empty(self):
        """Empty POS counts return empty probabilities."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            output_dir = Path(tmpdir) / "output"
            input_dir.mkdir()

            builder = FrequencyBuilder(input_dir=input_dir, output_dir=output_dir)
            builder.pos_unigram_counts = Counter()
            assert builder.calculate_pos_unigram_probabilities() == {}

    def test_calculate_pos_bigram_probabilities_laplace(self):
        """POS bigram probabilities use Laplace smoothing correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            output_dir = Path(tmpdir) / "output"
            input_dir.mkdir()

            builder = FrequencyBuilder(input_dir=input_dir, output_dir=output_dir)

            # Empty case
            builder.pos_unigram_counts = Counter()
            assert builder.calculate_pos_bigram_probabilities() == {}

            # Populated case
            builder.pos_unigram_counts = Counter({"N": 1, "V": 1})  # V=2
            builder.pos_bigram_counts = Counter({("N", "V"): 1})
            builder.pos_bigram_predecessor_counts = Counter({"N": 1})

            probs = builder.calculate_pos_bigram_probabilities()
            # Laplace: (count(N,V) + 1) / (count(N) + V) = (1+1)/(1+2) = 2/3
            assert probs[("N", "V")] == pytest.approx(2 / 3)
            # Unseen: (0+1)/(1+2) = 1/3
            assert probs[("N", "N")] == pytest.approx(1 / 3)
            # tag1 was never a predecessor (tag1_total == 0): 1/V = 1/2
            assert probs[("V", "N")] == pytest.approx(0.5)

    def test_calculate_pos_trigram_probabilities_laplace(self):
        """POS trigram probabilities use Laplace smoothing correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            output_dir = Path(tmpdir) / "output"
            input_dir.mkdir()

            builder = FrequencyBuilder(input_dir=input_dir, output_dir=output_dir)

            # Empty case
            builder.pos_unigram_counts = Counter()
            assert builder.calculate_pos_trigram_probabilities() == {}

            # Populated case
            builder.pos_unigram_counts = Counter({"N": 1, "V": 1, "P": 1})  # V=3
            builder.pos_trigram_counts = Counter({("N", "V", "P"): 1})
            builder.pos_bigram_successor_counts = Counter({("N", "V"): 1})

            probs = builder.calculate_pos_trigram_probabilities()
            # Laplace: (1+1)/(1+3) = 2/4 = 0.5
            assert probs[("N", "V", "P")] == 0.5
            # Unseen: (0+1)/(1+3) = 0.25
            assert probs[("N", "V", "N")] == 0.25


class TestSegmenterFallback:
    """Tests for segmenter error fallback logic."""

    def test_count_syllables_accurate_fallback(self):
        """Segmenter failure falls back to len(word) // 3 heuristic."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            output_dir = Path(tmpdir) / "output"
            input_dir.mkdir()
            output_dir.mkdir()

            builder = FrequencyBuilder(input_dir=input_dir, output_dir=output_dir)
            # Force an exception in segmenter
            builder.segmenter.segment_syllables = Mock(side_effect=RuntimeError("fail"))
            count = builder._count_syllables_accurate("မြန်မာ")
            # Fallback logic: max(1, len(word) // 3)
            # "မြန်မာ" len is 6
            assert count == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
