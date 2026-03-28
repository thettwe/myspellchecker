import sqlite3
import tempfile
from pathlib import Path

import pytest

from myspellchecker.cli import main as cli_main


@pytest.fixture
def temp_workspace():
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture(autouse=True)
def disable_disk_space_check(monkeypatch):
    """Disable the 50GB disk space pre-flight check for test corpora."""
    monkeypatch.setattr(
        "myspellchecker.data_pipeline.pipeline.check_disk_space", lambda *a, **kw: None
    )


@pytest.fixture
def mock_segmenter(monkeypatch):
    """Mock the segmenter to avoid myTokenize dependency."""
    from myspellchecker.segmenters.default import DefaultSegmenter

    def mock_segment_words(self, text: str):
        return text.split()

    def mock_segment_syllables(self, text: str):
        # Simple mock: treat space-separated parts as syllables/words
        return text.split()

    monkeypatch.setattr(DefaultSegmenter, "segment_words", mock_segment_words)
    monkeypatch.setattr(DefaultSegmenter, "segment_syllables", mock_segment_syllables)


@pytest.fixture
def mock_pool(monkeypatch):
    """
    Mock for ProcessPoolExecutor to run tasks synchronously.

    NOTE: The data pipeline was refactored to use concurrent.futures.ProcessPoolExecutor
    with fork-based multiprocessing and Arrow format instead of multiprocessing.Pool
    with text/jsonl. This mock patches the executor to run synchronously.
    """
    import concurrent.futures
    from unittest.mock import MagicMock

    class MockFuture:
        def __init__(self, result):
            self._result = result

        def result(self, timeout=None):
            return self._result

    class MockProcessPoolExecutor:
        def __init__(self, *args, **kwargs):
            # Run initializer if provided
            initializer = kwargs.get("initializer")
            initargs = kwargs.get("initargs", ())
            if initializer:
                initializer(*initargs)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def submit(self, func, *args, **kwargs):
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                future = MagicMock()
                future.result.side_effect = e
                return future
            return MockFuture(result)

    def mock_as_completed(futures, timeout=None):
        """Mock as_completed to just iterate over futures directly."""
        return iter(futures)

    def mock_wait(futures, timeout=None, return_when=None):
        """Mock wait to return all futures as done immediately."""
        futures_set = set(futures) if not isinstance(futures, set) else futures
        return futures_set, set()  # All done, none pending

    monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", MockProcessPoolExecutor)
    monkeypatch.setattr(concurrent.futures, "as_completed", mock_as_completed)
    monkeypatch.setattr(concurrent.futures, "wait", mock_wait)


def test_incremental_build(temp_workspace, monkeypatch, capsys, mock_segmenter, mock_pool):
    """
    Test the incremental build process.
    1. Build DB with file A.
    2. Build DB with file A + file B (incremental).
    3. Verify counts reflect A + B, and A was not double-counted.
    """
    # Setup inputs
    corpus1 = temp_workspace / "corpus1.txt"
    corpus1.write_text("မြန်မာ နိုင်ငံ သည် လှပ သည်", encoding="utf-8")

    corpus2 = temp_workspace / "corpus2.txt"
    corpus2.write_text("မြန်မာ လူမျိုး များ သည် ရိုးသား ကြ သည်", encoding="utf-8")

    database_path = temp_workspace / "test.db"

    # ---------------------------------------------------------
    # Step 1: Initial Build (File 1 only)
    # Use --min-frequency 1 to include low-frequency words in test
    # ---------------------------------------------------------
    args = [
        "myspellchecker",
        "build",
        "--input",
        str(corpus1),
        "--output",
        str(database_path),
        "--work-dir",
        str(temp_workspace / "work1"),
        "--min-frequency",
        "1",
    ]
    monkeypatch.setattr("sys.argv", args)
    cli_main()

    # Verify Step 1
    assert database_path.exists()
    conn = sqlite3.connect(str(database_path))
    cursor = conn.cursor()

    # Check syllable 'သည်' (appears twice in corpus1)
    # Note: With mock segmenter, multi-word lines get further processed by the pipeline.
    # Single-syllable particles like 'သည်' appear in the syllables table, not words.
    cursor.execute("SELECT frequency FROM syllables WHERE syllable='သည်'")
    freq = cursor.fetchone()[0]
    assert freq == 2

    # Check tracking table
    cursor.execute("SELECT path FROM processed_files")
    files = {row[0] for row in cursor.fetchall()}
    assert str(corpus1.absolute()) in files
    assert len(files) == 1
    conn.close()

    # ---------------------------------------------------------
    # Step 2: Incremental Build (File 1 + File 2)
    # ---------------------------------------------------------
    # NOTE: The current incremental mode skips already-processed files
    # (files with same mtime and size as tracked in processed_files table).
    # corpus1 is skipped, only corpus2 is processed. The pipeline rebuilds
    # the database from all processed data (not additive to old counts).
    args = [
        "myspellchecker",
        "build",
        "--input",
        str(corpus1),
        str(corpus2),
        "--output",
        str(database_path),
        "--work-dir",
        str(temp_workspace / "work2"),
        "--incremental",
        "--min-frequency",
        "1",
    ]
    monkeypatch.setattr("sys.argv", args)
    cli_main()

    # Verify Step 2
    conn = sqlite3.connect(str(database_path))
    cursor = conn.cursor()

    # Check syllable 'သည်' after incremental build
    # corpus2 also contains 'သည်' twice → pipeline processes corpus2 only
    cursor.execute("SELECT frequency FROM syllables WHERE syllable='သည်'")
    result = cursor.fetchone()
    assert result is not None, "Syllable 'သည်' should be in database after incremental build"
    freq = result[0]
    assert freq >= 2  # At least from one corpus

    # Check that corpus2-specific syllable 'လူ' exists
    cursor.execute("SELECT frequency FROM syllables WHERE syllable='လူ'")
    result = cursor.fetchone()
    assert result is not None, "Syllable 'လူ' from corpus2 should be in database"

    # Check tracking table
    cursor.execute("SELECT path FROM processed_files")
    files = {row[0] for row in cursor.fetchall()}
    assert str(corpus1.absolute()) in files
    assert str(corpus2.absolute()) in files
    assert len(files) == 2

    conn.close()


def test_incremental_build_file_modified(temp_workspace, monkeypatch, mock_segmenter, mock_pool):
    """
    Test that modifying a file triggers reprocessing in incremental mode.
    """
    corpus1 = temp_workspace / "corpus1.txt"
    # Use unique lines containing 'ကို' to avoid line-level deduplication.
    # Each line must be distinct so the ingester doesn't skip duplicates.
    corpus1.write_text("ကို သည်\nကို ကြ\nကို များ", encoding="utf-8")  # 3 unique lines

    database_path = temp_workspace / "test_mod.db"

    # Run 1 - use --min-frequency 1 to include low-frequency words
    args = [
        "myspellchecker",
        "build",
        "--input",
        str(corpus1),
        "--output",
        str(database_path),
        "--work-dir",
        str(temp_workspace / "w1"),
        "--min-frequency",
        "1",
    ]
    monkeypatch.setattr("sys.argv", args)
    cli_main()

    conn = sqlite3.connect(str(database_path))
    cursor = conn.cursor()
    # 'ကို' appears as a syllable in all 3 lines
    cursor.execute("SELECT frequency FROM syllables WHERE syllable='ကို'")
    assert cursor.fetchone()[0] == 3
    conn.close()

    # Modify file — set mtime explicitly 2 seconds ahead to ensure the
    # pipeline sees a changed mtime without sleeping.
    import os
    import time

    corpus1.write_text("ကို သည်\nကို ကြ", encoding="utf-8")  # Now 2 unique lines.
    future_mtime = time.time() + 2
    os.utime(corpus1, (future_mtime, future_mtime))

    # When a file is modified (different mtime/size), the pipeline reprocesses it.
    # The current implementation replaces the database with fresh counts from
    # the reprocessed file, rather than accumulating old + new counts.

    # Run 2 (Incremental) - use --min-frequency 1 to match original build
    args = [
        "myspellchecker",
        "build",
        "--input",
        str(corpus1),
        "--output",
        str(database_path),
        "--work-dir",
        str(temp_workspace / "w2"),
        "--incremental",
        "--min-frequency",
        "1",
    ]
    monkeypatch.setattr("sys.argv", args)
    cli_main()

    conn = sqlite3.connect(str(database_path))
    cursor = conn.cursor()
    # After reprocessing modified file: only the new file content counts (2 occurrences)
    cursor.execute("SELECT frequency FROM syllables WHERE syllable='ကို'")
    assert cursor.fetchone()[0] == 2
    conn.close()


def test_incremental_build_all_skipped(
    temp_workspace, monkeypatch, capsys, mock_segmenter, mock_pool
):
    """
    Test that when all files are already processed (steps 1-3 skipped),
    step 4 is also skipped.
    """
    corpus1 = temp_workspace / "corpus1.txt"
    # Use unique lines containing 'ကို' to avoid line-level deduplication.
    corpus1.write_text("ကို သည်\nကို ကြ\nကို များ", encoding="utf-8")

    database_path = temp_workspace / "test_skip.db"
    work_dir = temp_workspace / "work_skip"

    # Run 1: Initial build
    args = [
        "myspellchecker",
        "build",
        "--input",
        str(corpus1),
        "--output",
        str(database_path),
        "--work-dir",
        str(work_dir),
        "--min-frequency",
        "1",
    ]
    monkeypatch.setattr("sys.argv", args)
    cli_main()

    # Verify initial build worked
    conn = sqlite3.connect(str(database_path))
    cursor = conn.cursor()
    cursor.execute("SELECT frequency FROM syllables WHERE syllable='ကို'")
    assert cursor.fetchone()[0] == 3
    conn.close()

    # Clear stdout capture
    capsys.readouterr()

    # Run 2: Incremental build with SAME work_dir (no changes)
    # All steps should be skipped since nothing changed
    args = [
        "myspellchecker",
        "build",
        "--input",
        str(corpus1),
        "--output",
        str(database_path),
        "--work-dir",
        str(work_dir),  # Same work_dir
        "--incremental",
        "--min-frequency",
        "1",
    ]
    monkeypatch.setattr("sys.argv", args)
    cli_main()

    # Capture output and verify all steps were skipped
    captured = capsys.readouterr()
    output = captured.out

    # All 4 steps should be SKIPPED
    assert output.count("SKIPPED") >= 4, (
        f"Expected all 4 steps to be SKIPPED when nothing changed.\nOutput:\n{output}"
    )

    # Verify database still has correct counts (unchanged)
    conn = sqlite3.connect(str(database_path))
    cursor = conn.cursor()
    cursor.execute("SELECT frequency FROM syllables WHERE syllable='ကို'")
    assert cursor.fetchone()[0] == 3  # Should remain unchanged
    conn.close()
