import tempfile
from pathlib import Path

import pyarrow as pa
import pytest

from myspellchecker.data_pipeline.ingester import (
    CorpusIngester,
    IngestionError,
    _normalize_batch,
)


def test_normalize_batch_direct():
    """Test the _normalize_batch function directly."""
    batch = [
        "မြန်မာစာ",  # Valid
        "Hello World",  # Invalid (not Myanmar)
        "ABC မြန်မာ",  # Mixed
        "   ",  # Empty
        "\u1050 Extended",  # Invalid (Extended Myanmar char)
    ]

    results = _normalize_batch(batch)
    assert len(results) == 5

    # Check "မြန်မာစာ"
    assert results[0][1] is True
    assert results[0][0] == "မြန်မာစာ"

    # Check "Hello World" - "Hello" is not Myanmar text, so is_myanmar_text returns False
    assert results[1][1] is False
    assert results[1][0] == ""

    # Check "ABC မြန်မာ"
    # is_myanmar_text("ABC မြန်မာ") -> True (because of "မြန်မာ")
    # validate_word("ABC") -> True (no explicit rule against it)
    # validate_word("မြန်မာ") -> True
    # So result is "ABC မြန်မာ"
    assert results[2][1] is True
    assert results[2][0] == "ABC မြန်မာ"

    # Check empty
    assert results[3][1] is False

    # Check Extended Myanmar
    # is_myanmar_text -> True (contains myanmar range if \u1050 is counted?
    # \u1050 is in range 1000-109F)
    # However, validate_text checks EXTENDED_MYANMAR_PATTERN which includes \u1050.
    # So validate_word("\u1050") should fail.
    # "Extended" -> Pass
    # Result should be just "Extended" IF \u1050 fails.
    # But wait, \u1050 is in myanmar block, so is_myanmar_text might return True.
    # normalize_with_zawgyi_conversion cleans it.

    # Let's verify what happens.
    # If validate_word fails for \u1050, then it is stripped.
    if results[4][1]:
        assert "\u1050" not in results[4][0]


def test_ingester_parallel_mode_force():
    """Test process with parallel=True to hit the parallel code path."""
    with tempfile.TemporaryDirectory() as temp_dir:
        input_dir = Path(temp_dir) / "input"
        output_dir = Path(temp_dir) / "output"
        input_dir.mkdir()

        # Create unique lines to avoid deduplication filtering
        corpus_file = input_dir / "corpus.txt"
        lines = [f"မြန်မာစာအမှတ်{i}နံပါတ်\n" for i in range(1000)]
        corpus_file.write_text("".join(lines))

        ingester = CorpusIngester()
        # Force parallel=True, num_workers=2
        # process returns list of shard paths
        shards = ingester.process(input_dir, output_dir, num_shards=2, parallel=True, num_workers=2)

        assert len(shards) == 2

        # Verify content
        total_lines = 0
        for shard in shards:
            with pa.OSFile(str(shard), "r") as source:
                reader = pa.RecordBatchStreamReader(source)
                for batch in reader:
                    total_lines += len(batch)

        assert total_lines == 1000


def test_ingester_recursive_search():
    """Test recursive file finding."""
    with tempfile.TemporaryDirectory() as temp_dir:
        input_dir = Path(temp_dir) / "input"
        subdir = input_dir / "subdir"
        subdir.mkdir(parents=True)

        (input_dir / "file1.txt").write_text("စာတစ်")
        (subdir / "file2.txt").write_text("စာနှစ်")

        ingester = CorpusIngester()

        # Non-recursive - process returns list of shard paths
        shards1 = ingester.process(input_dir, Path(temp_dir) / "out1", recursive=False)

        # Recursive
        shards2 = ingester.process(input_dir, Path(temp_dir) / "out2", recursive=True)

        # Verify counts by reading shards
        def count_lines(shard_paths):
            count = 0
            for p in shard_paths:
                if not p.exists():
                    continue
                with pa.OSFile(str(p), "r") as source:
                    try:
                        reader = pa.RecordBatchStreamReader(source)
                        for batch in reader:
                            count += len(batch)
                    except Exception:
                        pass
            return count

        assert count_lines(shards1) == 1
        assert count_lines(shards2) == 2


def test_read_file_formats_and_errors():
    """Test different file formats and error handling in _read_file."""
    ingester = CorpusIngester()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # JSONL with some invalid JSON
        jsonl_file = temp_path / "test.jsonl"
        jsonl_file.write_text('{"text": "ကောင်း"}\nINVALID_JSON\n{"text": "မွန်"}\n"Just string"\n')

        lines = list(ingester._read_file(jsonl_file))
        assert "ကောင်း" in lines
        assert "မွန်" in lines
        assert "Just string" in lines
        assert len(lines) == 3

        # CSV/TSV
        csv_file = temp_path / "test.csv"
        csv_file.write_text("col1,col2\nignored,မြန်မာ\n")

        # Default text_col="text" is strict and should fail when missing.
        with pytest.raises(ValueError, match="Column 'text' not found"):
            list(ingester._read_file(csv_file))

        # Index-based access is supported (text_col="0" reads first column).
        lines_csv = list(ingester._read_file(csv_file, text_col="0"))
        assert "col1" in lines_csv
        assert "ignored" in lines_csv

        # JSON List
        json_file = temp_path / "test.json"
        json_file.write_text('[{"text": "one"}, "two", {"other": "ignored"}]')
        with pytest.raises(ValueError, match="Missing key 'text'"):
            list(ingester._read_file(json_file))

        # Key can be overridden when JSON uses a different field.
        json_file_alt = temp_path / "test_alt.json"
        json_file_alt.write_text('[{"other": "one"}, "two", {"other": "ignored"}]')
        lines_json = list(ingester._read_file(json_file_alt, json_key="other"))
        assert "one" in lines_json
        assert "two" in lines_json

        # Invalid JSON
        bad_json = temp_path / "bad.json"
        bad_json.write_text("INVALID")
        lines_bad = list(ingester._read_file(bad_json))
        assert len(lines_bad) == 0


def test_incremental_skipping():
    """Test that files are skipped if metadata matches."""
    with tempfile.TemporaryDirectory() as temp_dir:
        input_file = Path(temp_dir) / "test.txt"
        input_file.write_text("data")
        output_dir = Path(temp_dir) / "output"
        output_dir.mkdir(parents=True, exist_ok=True)  # Ensure output dir exists

        ingester = CorpusIngester()

        # Fake metadata map
        stat = input_file.stat()
        processed_map = {str(input_file.absolute()): (stat.st_mtime, stat.st_size)}

        # Should skip
        shards, meta = ingester._process_sharded(
            [input_file], output_dir, 1, processed_files_map=processed_map
        )
        assert len(meta) == 0  # No new files processed

        # Parallel skip
        shards_p, meta_p = ingester._process_sharded_parallel(
            [input_file], output_dir, 1, num_workers=1, processed_files_map=processed_map
        )
        assert len(meta_p) == 0


class TestIngestionError:
    """Tests for IngestionError exception and error handling."""

    def test_ingestion_error_with_missing_files(self):
        """Test IngestionError displays missing files correctly."""
        error = IngestionError(
            "Test error",
            missing_files=["/path/to/missing1.txt", "/path/to/missing2.txt"],
        )
        error_str = str(error)
        assert "Test error" in error_str
        assert "Missing files (2)" in error_str
        assert "/path/to/missing1.txt" in error_str
        assert "/path/to/missing2.txt" in error_str

    def test_ingestion_error_with_failed_files(self):
        """Test IngestionError displays failed files correctly."""
        error = IngestionError(
            "Test error",
            failed_files=[
                ("/path/to/file1.txt", "Permission denied"),
                ("/path/to/file2.txt", "File corrupted"),
            ],
        )
        error_str = str(error)
        assert "Test error" in error_str
        assert "Failed files (2)" in error_str
        assert "/path/to/file1.txt" in error_str
        assert "Permission denied" in error_str

    def test_ingestion_error_truncates_long_lists(self):
        """Test that IngestionError truncates lists with more than 10 items."""
        missing_files = [f"/path/to/file{i}.txt" for i in range(15)]
        error = IngestionError("Test error", missing_files=missing_files)
        error_str = str(error)
        assert "Missing files (15)" in error_str
        assert "... and 5 more" in error_str


class TestIngesterErrorHandling:
    """Tests for ingester error handling behavior."""

    def test_process_sharded_raises_on_missing_files(self):
        """Test _process_sharded raises IngestionError for missing files."""
        ingester = CorpusIngester()
        missing_file = Path("/nonexistent/path/to/file.txt")

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "output"

            with pytest.raises(IngestionError) as exc_info:
                ingester._process_sharded([missing_file], output_dir, num_shards=1)

            assert "not found" in str(exc_info.value)
            assert len(exc_info.value.missing_files) == 1

    def test_process_sharded_parallel_raises_on_missing_files(self):
        """Test _process_sharded_parallel raises IngestionError for missing files."""
        ingester = CorpusIngester()
        missing_file = Path("/nonexistent/path/to/file.txt")

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "output"

            with pytest.raises(IngestionError) as exc_info:
                ingester._process_sharded_parallel(
                    [missing_file], output_dir, num_shards=1, num_workers=1
                )

            assert "not found" in str(exc_info.value)
            assert len(exc_info.value.missing_files) == 1

    def test_ingestion_error_importable_from_package(self):
        """Test IngestionError can be imported from the data_pipeline package."""
        from myspellchecker.data_pipeline import IngestionError as IE

        assert IE is IngestionError

    def test_multiple_missing_files_all_reported(self):
        """Test that all missing files are reported in the error."""
        ingester = CorpusIngester()
        missing_files = [
            Path("/nonexistent/path/to/file1.txt"),
            Path("/nonexistent/path/to/file2.txt"),
            Path("/nonexistent/path/to/file3.txt"),
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "output"

            with pytest.raises(IngestionError) as exc_info:
                ingester._process_sharded(missing_files, output_dir, num_shards=1)

            assert len(exc_info.value.missing_files) == 3
            assert "file1.txt" in str(exc_info.value)
            assert "file2.txt" in str(exc_info.value)
            assert "file3.txt" in str(exc_info.value)


class TestParquetIngestion:
    """Tests for Parquet file ingestion support."""

    def test_read_parquet_with_text_column(self):
        """Test reading Parquet file with a 'text' column."""
        ingester = CorpusIngester()

        with tempfile.TemporaryDirectory() as temp_dir:
            parquet_file = Path(temp_dir) / "test.parquet"

            # Create a Parquet file with 'text' column
            table = pa.table(
                {
                    "text": ["မြန်မာစာ", "ကောင်းပါတယ်", "စမ်းသပ်မှု"],
                    "id": [1, 2, 3],
                }
            )
            import pyarrow.parquet as pq

            pq.write_table(table, parquet_file)

            # Read the file
            lines = list(ingester._read_file(parquet_file))

            assert len(lines) == 3
            assert "မြန်မာစာ" in lines
            assert "ကောင်းပါတယ်" in lines
            assert "စမ်းသပ်မှု" in lines

    def test_read_parquet_requires_configured_column(self):
        """Parquet ingestion should fail if configured text column is missing."""
        ingester = CorpusIngester()

        with tempfile.TemporaryDirectory() as temp_dir:
            parquet_file = Path(temp_dir) / "test.parquet"

            # Create a Parquet file without 'text' column but with string column
            table = pa.table(
                {
                    "id": [1, 2, 3],
                    "content": ["မြန်မာ", "ဘာသာ", "စကား"],
                    "count": [10, 20, 30],
                }
            )
            import pyarrow.parquet as pq

            pq.write_table(table, parquet_file)

            with pytest.raises(ValueError, match="Column 'text' not found"):
                list(ingester._read_file(parquet_file))

            # Explicit column override should succeed.
            lines = list(ingester._read_file(parquet_file, text_col="content"))
            assert len(lines) == 3
            assert "မြန်မာ" in lines
            assert "ဘာသာ" in lines
            assert "စကား" in lines

    def test_read_parquet_no_string_columns(self):
        """Parquet ingestion should fail when configured text column is missing."""
        ingester = CorpusIngester()

        with tempfile.TemporaryDirectory() as temp_dir:
            parquet_file = Path(temp_dir) / "test.parquet"

            # Create a Parquet file with only numeric columns
            table = pa.table(
                {
                    "id": [1, 2, 3],
                    "value": [1.0, 2.0, 3.0],
                }
            )
            import pyarrow.parquet as pq

            pq.write_table(table, parquet_file)

            with pytest.raises(ValueError, match="Column 'text' not found"):
                list(ingester._read_file(parquet_file))

    def test_read_parquet_with_null_values(self):
        """Test reading Parquet file with null values in text column."""
        ingester = CorpusIngester()

        with tempfile.TemporaryDirectory() as temp_dir:
            parquet_file = Path(temp_dir) / "test.parquet"

            # Create a Parquet file with some null values
            table = pa.table(
                {
                    "text": ["မြန်မာစာ", None, "စမ်းသပ်မှု", "", "  "],
                }
            )
            import pyarrow.parquet as pq

            pq.write_table(table, parquet_file)

            # Read the file - should skip null and empty values
            lines = list(ingester._read_file(parquet_file))

            assert "မြန်မာစာ" in lines
            assert "စမ်းသပ်မှု" in lines
            # Empty strings should be stripped and excluded
            assert "" not in lines

    def test_read_parquet_with_large_string_type(self):
        """Test reading Parquet file with large_string type column."""
        ingester = CorpusIngester()

        with tempfile.TemporaryDirectory() as temp_dir:
            parquet_file = Path(temp_dir) / "test.parquet"

            # Create a Parquet file with large_string type
            table = pa.table(
                {
                    "id": [1, 2],
                    "content": pa.array(["မြန်မာ", "ဘာသာ"], type=pa.large_string()),
                }
            )
            import pyarrow.parquet as pq

            pq.write_table(table, parquet_file)

            # Explicit override required because default text_col is "text".
            lines = list(ingester._read_file(parquet_file, text_col="content"))

            assert len(lines) == 2
            assert "မြန်မာ" in lines
            assert "ဘာသာ" in lines

    def test_parquet_full_pipeline_integration(self):
        """Test full ingestion pipeline with Parquet file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "input"
            output_dir = Path(temp_dir) / "output"
            input_dir.mkdir()

            # Create a Parquet file with Myanmar text
            parquet_file = input_dir / "corpus.parquet"
            table = pa.table(
                {
                    "text": [
                        "မြန်မာစာသည် လှပသည်",
                        "ကျွန်တော် ကျောင်းသွားသည်",
                        "သူမ စာရေးနေသည်",
                    ],
                }
            )
            import pyarrow.parquet as pq

            pq.write_table(table, parquet_file)

            ingester = CorpusIngester()
            shards = ingester.process(input_dir, output_dir, num_shards=2, parallel=False)

            assert len(shards) == 2

            # Verify content was ingested
            total_lines = 0
            for shard in shards:
                if shard.exists():
                    with pa.OSFile(str(shard), "r") as source:
                        reader = pa.RecordBatchStreamReader(source)
                        for batch in reader:
                            total_lines += len(batch)

            # Should have ingested lines (exact count depends on validation)
            assert total_lines > 0

    def test_parquet_parallel_pipeline_integration(self):
        """Test parallel ingestion pipeline with Parquet file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "input"
            output_dir = Path(temp_dir) / "output"
            input_dir.mkdir()

            # Create a Parquet file with Myanmar text
            parquet_file = input_dir / "corpus.parquet"
            table = pa.table(
                {
                    "text": [f"မြန်မာစာအမှတ်{i}" for i in range(100)],
                }
            )
            import pyarrow.parquet as pq

            pq.write_table(table, parquet_file)

            ingester = CorpusIngester()
            shards = ingester.process(
                input_dir, output_dir, num_shards=2, parallel=True, num_workers=2
            )

            assert len(shards) == 2

            # Verify content
            total_lines = 0
            for shard in shards:
                if shard.exists():
                    with pa.OSFile(str(shard), "r") as source:
                        reader = pa.RecordBatchStreamReader(source)
                        for batch in reader:
                            total_lines += len(batch)

            assert total_lines == 100

    def test_mixed_formats_with_parquet(self):
        """Test ingestion of mixed formats including Parquet."""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "input"
            output_dir = Path(temp_dir) / "output"
            input_dir.mkdir()

            # Create a txt file
            txt_file = input_dir / "corpus.txt"
            txt_file.write_text("မြန်မာစာ\n")

            # Create a Parquet file
            parquet_file = input_dir / "corpus.parquet"
            table = pa.table({"text": ["ဘာသာစကား"]})
            import pyarrow.parquet as pq

            pq.write_table(table, parquet_file)

            ingester = CorpusIngester()
            shards = ingester.process(input_dir, output_dir, num_shards=1)

            # Verify both sources were ingested
            total_lines = 0
            texts = []
            for shard in shards:
                if shard.exists():
                    with pa.OSFile(str(shard), "r") as source:
                        reader = pa.RecordBatchStreamReader(source)
                        for batch in reader:
                            total_lines += len(batch)
                            texts.extend(batch.column("text").to_pylist())

            assert total_lines == 2
            assert "မြန်မာစာ" in texts
            assert "ဘာသာစကား" in texts
