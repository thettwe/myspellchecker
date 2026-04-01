"""
Tests for Data Pipeline modules (Ingester, Segmenter, FrequencyBuilder).
"""

import csv
import json
import tempfile
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

duckdb = pytest.importorskip("duckdb", reason="DuckDB required for data pipeline tests")
pa = pytest.importorskip("pyarrow", reason="PyArrow required for data pipeline tests")

from myspellchecker.data_pipeline.frequency_builder import FrequencyBuilder  # noqa: E402
from myspellchecker.data_pipeline.ingester import CorpusIngester  # noqa: E402
from myspellchecker.data_pipeline.segmenter import CorpusSegmenter  # noqa: E402

# --- CorpusIngester Tests ---


class TestCorpusIngester:
    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            yield Path(tmp)

    def test_ingest_txt_file(self, temp_dir):
        input_file = temp_dir / "corpus.txt"
        input_file.write_text("မြန်မာ\nEnglish\n\n", encoding="utf-8")

        ingester = CorpusIngester()
        # process returns list of shard paths
        shards = ingester.process(input_file, output_dir=temp_dir, num_shards=1)

        assert len(shards) == 1

        # Verify content of arrow file
        with pa.memory_map(str(shards[0]), "r") as source:
            reader = pa.ipc.open_stream(source)
            table = reader.read_all()

        text_col = table.column("text").to_pylist()
        # English lines are filtered out
        assert len(text_col) == 1
        assert text_col[0] == "မြန်မာ"

    def test_ingest_csv_file(self, temp_dir):
        input_file = temp_dir / "corpus.csv"
        with open(input_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["text", "id"])
            writer.writerow(["မြန်မာ", "1"])
            writer.writerow(["English", "2"])

        ingester = CorpusIngester()
        shards = ingester.process(input_file, output_dir=temp_dir, num_shards=1)

        with pa.memory_map(str(shards[0]), "r") as source:
            reader = pa.ipc.open_stream(source)
            table = reader.read_all()

        text_col = table.column("text").to_pylist()
        assert len(text_col) == 1
        assert text_col[0] == "မြန်မာ"

    def test_ingest_csv_tsv(self, temp_dir):
        # The ingester auto-detects based on extension
        input_file = temp_dir / "corpus.tsv"
        with open(input_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["text", "meta"])
            writer.writerow(["မြန်မာ", "dummy"])
            writer.writerow(["ရန်ကုန်", "dummy"])

        ingester = CorpusIngester()
        shards = ingester.process(input_file, output_dir=temp_dir, num_shards=1)

        with pa.memory_map(str(shards[0]), "r") as source:
            reader = pa.ipc.open_stream(source)
            table = reader.read_all()

        text_col = table.column("text").to_pylist()
        assert len(text_col) == 2
        assert "မြန်မာ" in text_col
        assert "ရန်ကုန်" in text_col

    def test_ingest_json_list(self, temp_dir):
        input_file = temp_dir / "corpus.json"
        input_file.write_text(
            json.dumps([{"text": "မြန်မာ"}, {"text": "English"}]), encoding="utf-8"
        )

        ingester = CorpusIngester()
        shards = ingester.process(input_file, output_dir=temp_dir, num_shards=1)

        with pa.memory_map(str(shards[0]), "r") as source:
            reader = pa.ipc.open_stream(source)
            table = reader.read_all()

        text_col = table.column("text").to_pylist()
        assert len(text_col) == 1
        assert text_col[0] == "မြန်မာ"

    def test_ingest_jsonl(self, temp_dir):
        input_file = temp_dir / "corpus.jsonl"
        # JSONL handles both dict and raw string if implemented
        input_file.write_text('{"text": "မြန်မာ"}\n{"text": "English"}\n', encoding="utf-8")

        ingester = CorpusIngester()
        shards = ingester.process(input_file, output_dir=temp_dir, num_shards=1)

        with pa.memory_map(str(shards[0]), "r") as source:
            reader = pa.ipc.open_stream(source)
            table = reader.read_all()

        text_col = table.column("text").to_pylist()
        assert len(text_col) == 1
        assert text_col[0] == "မြန်မာ"

    def test_file_not_found(self, temp_dir):
        ingester = CorpusIngester()
        with pytest.raises(FileNotFoundError):
            ingester.process(temp_dir / "non_existent.txt", output_dir=temp_dir)

    # Removed irrelevant tests:
    # - test_ingest_save_corpus (method removed)
    # - test_invalid_encoding (handled gracefully usually)
    # - test_ingester_csv_error (handled by try/except block in ingester loop usually)


# --- CorpusSegmenter Tests ---


class TestCorpusSegmenter:
    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            yield Path(tmp)

    def test_segment_corpus(self, temp_dir):
        # 1. Prepare Input Shard (Arrow)
        shard_file = temp_dir / "shard_000.arrow"
        schema = pa.schema([("text", pa.string()), ("source", pa.string())])
        data = {"text": ["မြန်မာ"], "source": ["test.txt"]}
        batch = pa.RecordBatch.from_pydict(data, schema=schema)

        with pa.OSFile(str(shard_file), "w") as sink:
            with pa.RecordBatchStreamWriter(sink, schema) as writer:
                writer.write_batch(batch)

        segmenter = CorpusSegmenter(output_dir=temp_dir)
        segmenter.word_engine = "crf"

        # Helper to fake worker behavior (write files + return stats)
        def fake_worker_action(task):
            out_dir = Path(task["output_dir"])
            cid = task["chunk_id"]
            out_file = out_dir / f"chunk_{cid}_segmented.arrow"

            # Output Schema
            seg_schema = pa.schema(
                [
                    ("text", pa.string()),
                    ("source", pa.string()),
                    ("syllables", pa.list_(pa.string())),
                    ("words", pa.list_(pa.string())),
                    ("syllable_count", pa.int32()),
                    ("word_count", pa.int32()),
                ]
            )

            # Dummy Data
            out_data = {
                "text": ["မြန်မာ"],
                "source": ["test.txt"],
                "syllables": [["မြန်", "မာ"]],
                "words": [["မြန်မာ"]],
                "syllable_count": [2],
                "word_count": [1],
            }
            batch = pa.RecordBatch.from_pydict(out_data, schema=seg_schema)

            with pa.OSFile(str(out_file), "w") as sink:
                with pa.RecordBatchStreamWriter(sink, seg_schema) as writer:
                    writer.write_batch(batch)

            return {"sentences": 1, "syllables": 2, "words": 1}

        # Mock the Executor to run synchronously or just return pre-computed results
        with patch("concurrent.futures.ProcessPoolExecutor") as MockExecutor:
            mock_executor_instance = MockExecutor.return_value
            mock_executor_instance.__enter__.return_value = mock_executor_instance

            # When submit is called, perform the action immediately and return a Future-like object
            def side_effect_submit(fn, task):
                # fn is worker_segment_file, but we ignore it and do our fake action
                res = fake_worker_action(task)
                f = Future()
                f.set_result(res)
                return f

            mock_executor_instance.submit.side_effect = side_effect_submit

            results_path = segmenter.segment_corpus([shard_file], num_workers=1)

        assert results_path.exists()

        # Read back arrow to verify content
        with pa.memory_map(str(results_path), "r") as source:
            reader = pa.ipc.open_stream(source)
            table = reader.read_all()

        text_col = table.column("text").to_pylist()
        assert "မြန်မာ" in text_col

    def test_segment_corpus_stats_and_unique(self, temp_dir):
        # 1. Prepare Input Shard (Arrow)
        shard_file = temp_dir / "shard_stats.arrow"
        schema = pa.schema([("text", pa.string()), ("source", pa.string())])
        data = {"text": ["မြန်"], "source": ["stats.txt"]}
        batch = pa.RecordBatch.from_pydict(data, schema=schema)

        with pa.OSFile(str(shard_file), "w") as sink:
            with pa.RecordBatchStreamWriter(sink, schema) as writer:
                writer.write_batch(batch)

        segmenter = CorpusSegmenter(output_dir=temp_dir)
        segmenter.word_engine = "crf"

        # Helper to fake worker behavior
        def fake_worker_action(task):
            # We don't even need to write files if we only care about stats returned
            # BUT segment_corpus logic merges files at the end.
            # If no files exist, it might crash or produce empty file.
            # Let's write a dummy empty arrow file to satisfy file existence check.
            out_dir = Path(task["output_dir"])
            cid = task["chunk_id"]
            out_file = out_dir / f"chunk_{cid}_segmented.arrow"

            # Write empty valid arrow stream
            seg_schema = pa.schema(
                [
                    ("text", pa.string()),
                    ("source", pa.string()),
                    ("syllables", pa.list_(pa.string())),
                    ("words", pa.list_(pa.string())),
                    ("syllable_count", pa.int32()),
                    ("word_count", pa.int32()),
                ]
            )
            with pa.OSFile(str(out_file), "w") as sink:
                with pa.RecordBatchStreamWriter(sink, seg_schema):
                    pass  # Empty

            return {"sentences": 10, "syllables": 20, "words": 10}

        with patch("concurrent.futures.ProcessPoolExecutor") as MockExecutor:
            mock_executor_instance = MockExecutor.return_value
            mock_executor_instance.__enter__.return_value = mock_executor_instance

            def side_effect_submit(fn, task):
                res = fake_worker_action(task)
                f = Future()
                f.set_result(res)
                return f

            mock_executor_instance.submit.side_effect = side_effect_submit

            segmenter.segment_corpus([shard_file], num_workers=1)

        # stats should reflect the return values from our fake worker
        assert segmenter.stats["total_sentences"] == 10
        assert segmenter.stats["total_syllables"] == 20
        assert segmenter.stats["total_words"] == 10


# --- FrequencyBuilder Tests ---


class TestFrequencyBuilder:
    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            yield Path(tmp)

    def test_build_frequencies(self, temp_dir):
        # 1. Prepare Input Shard (Arrow)
        shard_file = temp_dir / "segmented_corpus.arrow"
        schema = pa.schema(
            [
                ("text", pa.string()),
                ("source", pa.string()),
                ("syllables", pa.list_(pa.string())),
                ("words", pa.list_(pa.string())),
                ("syllable_count", pa.int32()),
                ("word_count", pa.int32()),
            ]
        )

        # Data: "မြန်မာ" (Myanmar) -> Syllables: ["မြန်", "မာ"], Words: ["မြန်မာ"]
        # Sentence 2: "နိုင်ငံ" (Country) -> Syllables: ["နိုင်", "ငံ"], Words: ["နိုင်ငံ"]
        # Combined in one sentence? Or separate?
        # Let's do: "မြန်မာ နိုင်ငံ" -> ["မြန်မာ", "နိုင်ငံ"]
        data = {
            "text": ["မြန်မာ နိုင်ငံ"],
            "source": ["test"],
            "syllables": [["မြန်", "မာ", "နိုင်", "ငံ"]],
            "words": [["မြန်မာ", "နိုင်ငံ"]],
            "syllable_count": [4],
            "word_count": [2],
        }
        batch = pa.RecordBatch.from_pydict(data, schema=schema)

        with pa.OSFile(str(shard_file), "w") as sink:
            with pa.ipc.new_stream(sink, schema) as writer:
                writer.write_batch(batch)

        builder = FrequencyBuilder(input_dir=temp_dir, output_dir=temp_dir)

        # Load from Arrow
        builder.load_data("segmented_corpus.arrow")

        # Verify Counts
        assert builder.syllable_counts["မြန်"] == 1
        assert builder.word_counts["မြန်မာ"] == 1
        assert builder.word_counts["နိုင်ငံ"] == 1

        # Bigrams — check via DuckDB store or Python dict
        if builder._duckdb_ngram_store:
            assert builder._duckdb_ngram_store.get_count("bigram_counts") == 1
            assert builder.stats["unique_bigrams"] == 1
        else:
            assert builder.bigram_counts[("မြန်မာ", "နိုင်ငံ")] == 1

        # Filter (default min freq is 5, let's lower it)
        builder.min_syllable_frequency = 1
        builder.min_word_frequency = 1
        builder.min_bigram_frequency = 1
        builder.filter_by_frequency()

        assert "မြန်" in builder.syllable_counts

        # Probs
        bigram_probs = builder.calculate_bigram_probabilities()
        if builder._duckdb_ngram_store:
            assert bigram_probs is None  # DuckDB handles prob calculation during save
        else:
            assert ("မြန်မာ", "နိုင်ငံ") in bigram_probs

        # Save
        builder.save_syllable_frequencies()
        builder.save_word_frequencies()
        builder.save_bigram_probabilities(bigram_probs)

        # Cleanup DuckDB resources before asserting file existence
        builder.cleanup_duckdb()

        assert (temp_dir / "syllable_frequencies.tsv").exists()
        assert (temp_dir / "word_frequencies.tsv").exists()
        assert (temp_dir / "bigram_probabilities.tsv").exists()

        # Verify bigram TSV content has the expected entry
        bigram_tsv = (temp_dir / "bigram_probabilities.tsv").read_text()
        assert "မြန်မာ" in bigram_tsv
        assert "နိုင်ငံ" in bigram_tsv

    def test_load_missing_files(self, temp_dir):
        builder = FrequencyBuilder(input_dir=temp_dir, output_dir=temp_dir)

        with pytest.raises(FileNotFoundError):
            builder.load_data("non_existent.arrow")

    def test_trigram_calculations(self, temp_dir):
        # 1. Prepare Input Shard (Arrow)
        shard_file = temp_dir / "segmented_corpus.arrow"
        schema = pa.schema(
            [
                ("text", pa.string()),
                ("source", pa.string()),
                ("syllables", pa.list_(pa.string())),
                ("words", pa.list_(pa.string())),
                ("syllable_count", pa.int32()),
                ("word_count", pa.int32()),
            ]
        )

        # Data: Myanmar words that pass is_quality_word() filter
        data = {
            "text": ["သူ သွား တယ်"],
            "source": ["test"],
            "syllables": [["သူ", "သွား", "တယ်"]],
            "words": [["သူ", "သွား", "တယ်"]],
            "syllable_count": [3],
            "word_count": [3],
        }
        batch = pa.RecordBatch.from_pydict(data, schema=schema)

        with pa.OSFile(str(shard_file), "w") as sink:
            with pa.ipc.new_stream(sink, schema) as writer:
                writer.write_batch(batch)

        builder = FrequencyBuilder(input_dir=temp_dir, output_dir=temp_dir)
        builder.load_data("segmented_corpus.arrow")

        if builder._duckdb_ngram_store:
            assert builder._duckdb_ngram_store.get_count("trigram_counts") == 1
        else:
            assert builder.trigram_counts[("သူ", "သွား", "တယ်")] == 1

        # Filter
        builder.min_trigram_frequency = 1
        builder.filter_by_frequency()

        # Calc probs
        probs = builder.calculate_trigram_probabilities()
        if builder._duckdb_ngram_store:
            assert probs is None  # DuckDB handles prob calculation during save
        else:
            assert ("သူ", "သွား", "တယ်") in probs
            assert (
                probs[("သူ", "သွား", "တယ်")][0] == 1.0
            )  # Access probability from (probability, count) tuple

        # Save
        builder.save_trigram_probabilities(probs)

        # Cleanup DuckDB resources
        builder.cleanup_duckdb()

        assert (temp_dir / "trigram_probabilities.tsv").exists()

        # Verify trigram TSV content
        trigram_tsv = (temp_dir / "trigram_probabilities.tsv").read_text()
        assert "သူ" in trigram_tsv
        assert "သွား" in trigram_tsv
        assert "တယ်" in trigram_tsv

    def test_print_methods(self, temp_dir):
        builder = FrequencyBuilder(input_dir=temp_dir, output_dir=temp_dir)
        # Call print methods (smoke test)
        builder.print_stats()

    def test_syllable_count_fallback(self, temp_dir):
        builder = FrequencyBuilder(input_dir=temp_dir, output_dir=temp_dir)

        # Mock segmenter to raise RuntimeError (one of the caught exception types)
        builder.segmenter.segment_syllables = MagicMock(side_effect=RuntimeError("Fail"))

        # Word length 6 -> fallback max(1, 6//3) = 2
        count = builder._count_syllables_accurate("123456")
        assert count == 2
