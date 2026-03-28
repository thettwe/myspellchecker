import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pyarrow as pa

from myspellchecker.data_pipeline import _segmenter_config as config_module
from myspellchecker.data_pipeline import _segmenter_workers as workers_module

# We need to ensure psutil is mocked before importing segmenter if it was imported at top level
# But segmenter imports it inside functions.
# We will use patch.dict(sys.modules) for psutil.
from myspellchecker.data_pipeline.segmenter import (
    SEGMENT_SCHEMA,
    CorpusSegmenter,
    _SegmenterCapabilities,
    get_optimal_batch_size,
    get_optimal_worker_count,
    init_worker,
    init_worker_fork,
    is_myanmar_token,
    preload_models,
    worker_segment_file,
)


def test_is_myanmar_token():
    assert is_myanmar_token("မြန်မာ")
    assert not is_myanmar_token("abc")

    # Test Python fallback path by replacing _CAPABILITIES with has_cython_check=False
    mock_caps = _SegmenterCapabilities(
        has_cython_check=False,
        has_batch_processor=False,
        has_openmp=False,
        has_cython_repair=False,
        use_fork_optimization=False,
        openmp_threads_per_worker=4,
    )
    with patch.object(config_module, "_CAPABILITIES", mock_caps):
        assert is_myanmar_token("မြန်မာ")
        assert not is_myanmar_token("abc")


def test_get_optimal_worker_count():
    # Mock psutil module
    mock_psutil = MagicMock()
    mock_psutil.cpu_count.return_value = 8
    mock_psutil.virtual_memory.return_value.available = 16 * 1024**3

    with patch.dict(sys.modules, {"psutil": mock_psutil}):
        # We need to reload or re-import? No, local import will pick up sys.modules

        # Plenty of resources
        assert get_optimal_worker_count(file_count=100) == 8

        # Limited files
        assert get_optimal_worker_count(file_count=2) == 2

        # Limited memory
        mock_psutil.virtual_memory.return_value.available = 1 * 1024**3  # 1GB
        # 1GB / 0.3 = 3 workers
        assert get_optimal_worker_count(file_count=100) <= 3


def test_get_optimal_batch_size():
    # Mock psutil
    mock_psutil = MagicMock()
    mock_psutil.virtual_memory.return_value.available = 16 * 1024**3

    with patch.dict(sys.modules, {"psutil": mock_psutil}):
        # Large file, plenty memory
        # Passing available_memory_gb overrides psutil
        # base_batch=20000 (file>500MB). Mem>4GB -> min(20000*2, 50000) = 40000
        assert get_optimal_batch_size(file_size_mb=600, available_memory_gb=16) == 40000

        # Low memory
        # 600MB file -> base 20000. Mem < 2GB -> max(5000, 20000 // 2) = 10000
        assert get_optimal_batch_size(file_size_mb=600, available_memory_gb=1) == 10000


def test_preload_models():
    with (
        patch("myspellchecker.data_pipeline._segmenter_workers.DefaultSegmenter") as MockSeg,
        patch("myspellchecker.data_pipeline._segmenter_workers.SegmentationRepair"),
    ):
        # Reset state using the _STATE object
        config_module._STATE.models_preloaded = False
        config_module._STATE.preloaded_segmenter = None

        # Reset allow_extended_myanmar to default for isolation
        config_module._allow_extended_myanmar = False

        preload_models("myword", ["custom"])

        # DefaultSegmenter called with word_engine, allow_extended_myanmar,
        # and transformer-related params (seg_model, seg_device)
        MockSeg.assert_called_with(
            word_engine="myword",
            allow_extended_myanmar=False,
            seg_model=None,
            seg_device=-1,
        )
        assert config_module._STATE.models_preloaded is True


def test_init_worker_fork():
    config_module._STATE.preloaded_segmenter = MagicMock()
    config_module._STATE.preloaded_repair = MagicMock()

    init_worker_fork()

    assert config_module._STATE.worker_segmenter == config_module._STATE.preloaded_segmenter
    assert config_module._STATE.is_forked_worker is True


def test_init_worker_spawn():
    config_module._STATE.worker_segmenter = None

    # Use mock capabilities with use_fork_optimization=False
    mock_caps = _SegmenterCapabilities(
        has_cython_check=False,
        has_batch_processor=False,
        has_openmp=False,
        has_cython_repair=False,
        use_fork_optimization=False,
        openmp_threads_per_worker=4,
    )
    with (
        patch.object(config_module, "_CAPABILITIES", mock_caps),
        patch.object(workers_module, "_CAPABILITIES", mock_caps),
        patch("myspellchecker.data_pipeline._segmenter_workers.DefaultSegmenter") as MockSeg,
        patch("myspellchecker.data_pipeline._segmenter_workers.SegmentationRepair"),
    ):
        init_worker("myword")
        MockSeg.assert_called()
        assert config_module._STATE.worker_segmenter is not None


def test_worker_segment_file_success():
    # Reset globals using _STATE
    mock_seg = MagicMock()
    mock_seg.segment_syllables.return_value = ["မြန်", "မာ"]
    mock_seg.segment_words.return_value = ["မြန်မာ"]

    mock_repair = MagicMock()
    mock_repair.repair.side_effect = lambda w: w

    config_module._STATE.worker_segmenter = mock_seg
    config_module._STATE.worker_repair = mock_repair

    # Mock PyArrow IO
    # We need to return a proper RecordBatchReader from open_stream

    # Create a real batch to return
    batch = pa.RecordBatch.from_pydict(
        {
            "text": ["မြန်မာ"],
            "source": ["src"],
            "syllables": [[""]],  # dummy, will be overwritten
            "words": [[""]],
            "syllable_count": [0],
            "word_count": [0],
        },
        schema=SEGMENT_SCHEMA,
    )

    # Mocking memory_map is hard because it returns a NativeFile.
    # Better to create a temp file and let it read it?
    # But then we need to write a valid IPC stream to it.

    with tempfile.TemporaryDirectory() as temp_dir:
        input_file = Path(temp_dir) / "in.arrow"
        Path(temp_dir) / "out.arrow"

        # Write valid input
        with pa.OSFile(str(input_file), "w") as sink:
            with pa.ipc.new_stream(sink, SEGMENT_SCHEMA) as writer:
                writer.write_batch(batch)

        args = {"input_file": str(input_file), "output_dir": temp_dir, "chunk_id": 0}

        # We need to NOT mock everything, just let it run with the real file
        # but with our mocked segmenter/repairer.
        # Force has_batch_processor=False to test Python path.
        mock_caps = _SegmenterCapabilities(
            has_cython_check=False,
            has_batch_processor=False,
            has_openmp=False,
            has_cython_repair=False,
            use_fork_optimization=False,
            openmp_threads_per_worker=4,
        )
        with (
            patch.object(config_module, "_CAPABILITIES", mock_caps),
            patch.object(workers_module, "_CAPABILITIES", mock_caps),
        ):
            stats = worker_segment_file(args)

            assert stats["sentences"] == 1
            assert stats["syllables"] == 2
            assert stats["words"] == 1

            # Verify output
            assert (Path(temp_dir) / "chunk_0_segmented.arrow").exists()


def test_worker_segment_file_retry():
    # Mock globals using _STATE
    config_module._STATE.worker_segmenter = MagicMock()
    config_module._STATE.worker_repair = MagicMock()

    # Mock pa.OSFile to raise exception then succeed?
    # Or mock memory_map to raise exception

    with patch("pyarrow.OSFile") as mock_osfile:
        # First attempt fails with OSError
        # Second attempt succeeds (return a mock context manager)

        # We need a mock that works as context manager
        good_sink = MagicMock()
        good_sink.__enter__.return_value = good_sink

        mock_osfile.side_effect = [OSError("Fail once"), good_sink]

        # Also need to mock other calls to not crash on second attempt
        with (
            patch("pyarrow.RecordBatchStreamWriter"),
            patch("pyarrow.memory_map"),
            patch("pyarrow.ipc.open_stream") as mock_reader,
        ):
            mock_reader.return_value = []  # Empty stream

            args = {
                "input_file": "dummy.arrow",
                "output_dir": "dummy",
                "chunk_id": 0,
                "retry_base_delay": 0.01,
            }

            stats = worker_segment_file(args)
            assert stats["sentences"] == 0
            assert mock_osfile.call_count == 2


def test_corpus_segmenter_init():
    with tempfile.TemporaryDirectory() as temp_dir:
        with patch("myspellchecker.data_pipeline.segmenter.DefaultSegmenter"):
            seg = CorpusSegmenter(output_dir=temp_dir)
            assert seg.output_dir == Path(temp_dir)
            assert seg.word_engine == "myword"  # default


def test_segment_corpus_workflow():
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir) / "out"
        input_dir = Path(temp_dir) / "in"
        input_dir.mkdir()
        (input_dir / "shard_0.arrow").touch()
        output_dir.mkdir()

        # Create dummy chunk output to simulate worker success
        (output_dir / "chunks").mkdir()
        chunk_file = output_dir / "chunks" / "chunk_0_segmented.arrow"
        # Write valid empty stream to chunk file so merge doesn't crash
        with pa.OSFile(str(chunk_file), "w") as sink:
            with pa.ipc.new_stream(sink, SEGMENT_SCHEMA):
                pass

        with (
            patch("myspellchecker.data_pipeline.segmenter.DefaultSegmenter"),
            patch("myspellchecker.data_pipeline.segmenter.preload_models"),
            patch("concurrent.futures.ProcessPoolExecutor") as MockExecutor,
        ):
            mock_future = MagicMock()
            mock_future.result.return_value = {"sentences": 0, "syllables": 0, "words": 0}
            MockExecutor.return_value.__enter__.return_value.submit.return_value = mock_future

            seg = CorpusSegmenter(output_dir=output_dir)
            seg.segment_corpus(input_dir, num_workers=1)

            assert (output_dir / "segmented_corpus.arrow").exists()
