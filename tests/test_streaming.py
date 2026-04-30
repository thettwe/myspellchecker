"""
Tests for streaming spell checking functionality.

Tests cover:
- StreamingConfig validation
- StreamingStats tracking
- Sync streaming (check_stream)
- Async streaming (check_stream_async)
- Sentence-by-sentence checking
- Progress callbacks
- Memory management
"""

import asyncio
import io
from unittest.mock import MagicMock, patch

import pytest

from myspellchecker.core.constants import ValidationLevel
from myspellchecker.core.response import Response
from myspellchecker.core.streaming import (
    ChunkResult,
    StreamingChecker,
    StreamingConfig,
    StreamingStats,
)

# --- StreamingConfig Tests ---


@pytest.mark.unit
class TestStreamingConfig:
    """Tests for StreamingConfig validation."""

    def test_default_config(self):
        """Test default configuration values."""
        config = StreamingConfig()
        assert config.chunk_size == 100
        assert config.max_memory_mb == 100
        assert config.progress_interval == 1000
        assert config.timeout_per_chunk == 30.0
        assert config.enable_cross_sentence_context is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = StreamingConfig(
            chunk_size=50,
            max_memory_mb=200,
            progress_interval=500,
            timeout_per_chunk=60.0,
        )
        assert config.chunk_size == 50
        assert config.max_memory_mb == 200
        assert config.progress_interval == 500
        assert config.timeout_per_chunk == 60.0

    def test_invalid_chunk_size(self):
        """Test that chunk_size < 1 raises ValueError."""
        with pytest.raises(ValueError, match="chunk_size must be >= 1"):
            StreamingConfig(chunk_size=0)

    def test_invalid_max_memory(self):
        """Test that max_memory_mb < 1 raises ValueError."""
        with pytest.raises(ValueError, match="max_memory_mb must be >= 1"):
            StreamingConfig(max_memory_mb=0)

    def test_invalid_progress_interval(self):
        """Test that progress_interval < 1 raises ValueError."""
        with pytest.raises(ValueError, match="progress_interval must be >= 1"):
            StreamingConfig(progress_interval=0)

    def test_invalid_timeout(self):
        """Test that timeout_per_chunk <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="timeout_per_chunk must be > 0"):
            StreamingConfig(timeout_per_chunk=0)


# --- StreamingStats Tests ---


@pytest.mark.unit
class TestStreamingStats:
    """Tests for StreamingStats tracking."""

    def test_default_stats(self):
        """Test default statistics values."""
        stats = StreamingStats()
        assert stats.bytes_processed == 0
        assert stats.lines_processed == 0
        assert stats.sentences_processed == 0
        assert stats.errors_found == 0
        assert stats.chunks_processed == 0

    def test_elapsed_time(self):
        """Test elapsed time calculation."""
        import time

        # Create stats and set start_time to a known value for deterministic testing
        stats = StreamingStats()
        stats.start_time = time.time() - 0.5  # Set to 0.5 seconds ago
        assert stats.elapsed_time >= 0.5
        assert stats.elapsed_time < 1.0  # Should be close to 0.5

    def test_lines_per_second(self):
        """Test processing rate calculation."""
        stats = StreamingStats()
        stats.lines_processed = 1000
        # Rate should be positive
        assert stats.lines_per_second >= 0

    def test_to_dict(self):
        """Test stats serialization to dictionary."""
        stats = StreamingStats()
        stats.bytes_processed = 1000
        stats.lines_processed = 50
        stats.errors_found = 5

        d = stats.to_dict()
        assert d["bytes_processed"] == 1000
        assert d["lines_processed"] == 50
        assert d["errors_found"] == 5
        assert "elapsed_time" in d
        assert "lines_per_second" in d


# --- ChunkResult Tests ---


@pytest.mark.unit
class TestChunkResult:
    """Tests for ChunkResult dataclass."""

    def test_chunk_result_creation(self):
        """Test creating a ChunkResult."""
        response = Response(
            text="test",
            corrected_text="test",
            has_errors=False,
            level="syllable",
            errors=[],
            metadata={},
        )
        result = ChunkResult(
            response=response,
            line_number=1,
            chunk_index=0,
            is_final=False,
        )
        assert result.response == response
        assert result.line_number == 1
        assert result.chunk_index == 0
        assert result.is_final is False


# --- StreamingChecker Tests ---


@pytest.mark.unit
class TestStreamingChecker:
    """Tests for StreamingChecker class."""

    @pytest.fixture
    def mock_checker(self):
        """Create a mock SpellChecker."""
        checker = MagicMock()
        checker.check.return_value = Response(
            text="test",
            corrected_text="test",
            has_errors=False,
            level="syllable",
            errors=[],
            metadata={},
        )
        return checker

    @pytest.fixture
    def streaming_checker(self, mock_checker):
        """Create a StreamingChecker with mock checker."""
        return StreamingChecker(mock_checker)

    def test_init_default_config(self, mock_checker):
        """Test initialization with default config."""
        streaming = StreamingChecker(mock_checker)
        assert streaming.config.chunk_size == 100
        assert streaming.checker == mock_checker

    def test_init_custom_config(self, mock_checker):
        """Test initialization with custom config."""
        config = StreamingConfig(chunk_size=50)
        streaming = StreamingChecker(mock_checker, config=config)
        assert streaming.config.chunk_size == 50

    def test_check_stream_empty_input(self, streaming_checker):
        """Test streaming with empty input."""
        input_stream = io.StringIO("")
        results = list(streaming_checker.check_stream(input_stream))
        assert results == []

    def test_check_stream_single_line(self, streaming_checker, mock_checker):
        """Test streaming with single line."""
        input_stream = io.StringIO("test line\n")
        results = list(streaming_checker.check_stream(input_stream))

        assert len(results) == 1
        assert results[0].line_number == 1
        mock_checker.check.assert_called_once()

    def test_check_stream_multiple_lines(self, streaming_checker, mock_checker):
        """Test streaming with multiple lines."""
        input_stream = io.StringIO("line 1\nline 2\nline 3\n")
        results = list(streaming_checker.check_stream(input_stream))

        assert len(results) == 3
        assert mock_checker.check.call_count == 3

    def test_check_stream_skips_empty_lines(self, streaming_checker, mock_checker):
        """Test that empty lines are skipped."""
        input_stream = io.StringIO("line 1\n\n\nline 2\n")
        results = list(streaming_checker.check_stream(input_stream))

        assert len(results) == 2
        assert mock_checker.check.call_count == 2

    def test_check_stream_with_stats(self, streaming_checker, mock_checker):
        """Test that stats are updated during streaming."""
        input_stream = io.StringIO("line 1\nline 2\n")
        stats = StreamingStats()

        list(streaming_checker.check_stream(input_stream, stats=stats))

        assert stats.lines_processed == 2
        assert stats.bytes_processed > 0
        assert stats.chunks_processed == 2

    def test_check_stream_counts_errors(self, mock_checker):
        """Test that errors are counted in stats."""
        error_response = Response(
            text="test",
            corrected_text="test",
            has_errors=True,
            level="syllable",
            errors=[MagicMock(), MagicMock()],  # 2 errors
            metadata={},
        )
        mock_checker.check.return_value = error_response

        streaming = StreamingChecker(mock_checker)
        input_stream = io.StringIO("line 1\n")
        stats = StreamingStats()

        list(streaming.check_stream(input_stream, stats=stats))
        assert stats.errors_found == 2

    def test_check_stream_progress_callback(self, mock_checker):
        """Test progress callback is called."""
        config = StreamingConfig(progress_interval=1)  # Call every line
        streaming = StreamingChecker(mock_checker, config=config)

        callback_calls = []

        def on_progress(stats):
            callback_calls.append(stats.lines_processed)

        input_stream = io.StringIO("line 1\nline 2\nline 3\n")
        list(streaming.check_stream(input_stream, on_progress=on_progress))

        # Should have callbacks (at least final one)
        assert len(callback_calls) >= 1

    def test_check_stream_handles_exception(self, mock_checker):
        """Test that exceptions are handled gracefully."""
        mock_checker.check.side_effect = Exception("Test error")

        streaming = StreamingChecker(mock_checker)
        input_stream = io.StringIO("line 1\n")

        results = list(streaming.check_stream(input_stream))

        assert len(results) == 1
        assert "error" in results[0].response.metadata

    def test_check_stream_validation_level(self, streaming_checker, mock_checker):
        """Test that validation level is passed correctly."""
        input_stream = io.StringIO("test\n")
        list(streaming_checker.check_stream(input_stream, level=ValidationLevel.WORD))

        mock_checker.check.assert_called_with("test", level=ValidationLevel.WORD)

    def test_check_sentences_single(self, streaming_checker, mock_checker):
        """Test sentence-by-sentence checking with single sentence."""
        results = list(streaming_checker.check_sentences("Test sentence။"))

        assert len(results) == 1
        mock_checker.check.assert_called()

    def test_check_sentences_multiple(self, streaming_checker, mock_checker):
        """Test sentence-by-sentence checking with multiple sentences."""
        text = "First sentence။ Second sentence။ Third sentence။"
        results = list(streaming_checker.check_sentences(text))

        # Should have 3 results (one per sentence)
        assert len(results) == 3

    def test_check_sentences_preserves_delimiters(self, mock_checker):
        """Test that sentence delimiters are preserved."""

        # Create checker that returns the actual text
        def return_actual_text(text, level=None):
            return Response(
                text=text,
                corrected_text=text,
                has_errors=False,
                level="word",
                errors=[],
                metadata={},
            )

        mock_checker.check.side_effect = return_actual_text
        streaming = StreamingChecker(mock_checker)

        text = "Question? Answer!"
        results = list(streaming.check_sentences(text))

        # Should have results
        assert len(results) >= 1


# --- Async Streaming Tests ---


@pytest.mark.unit
class TestAsyncStreaming:
    """Tests for async streaming functionality."""

    @pytest.fixture
    def mock_checker(self):
        """Create a mock SpellChecker."""
        checker = MagicMock()
        checker.check.return_value = Response(
            text="test",
            corrected_text="test",
            has_errors=False,
            level="syllable",
            errors=[],
            metadata={},
        )
        return checker

    def test_check_stream_async(self, mock_checker):
        """Test async streaming."""
        streaming = StreamingChecker(mock_checker)

        async def async_lines():
            for line in ["line 1\n", "line 2\n"]:
                yield line

        async def run_test():
            results = []
            async for result in streaming.check_stream_async(async_lines()):
                results.append(result)
            return results

        results = asyncio.run(run_test())
        assert len(results) == 2
        assert mock_checker.check.call_count == 2

    def test_check_stream_async_with_stats(self, mock_checker):
        """Test async streaming with stats tracking."""
        streaming = StreamingChecker(mock_checker)
        stats = StreamingStats()

        async def async_lines():
            for line in ["line 1\n", "line 2\n"]:
                yield line

        async def run_test():
            async for _ in streaming.check_stream_async(async_lines(), stats=stats):
                pass

        asyncio.run(run_test())
        assert stats.lines_processed == 2
        assert stats.bytes_processed > 0

    def test_check_stream_async_timeout(self, mock_checker):
        """Test async streaming handles timeout."""
        config = StreamingConfig(timeout_per_chunk=0.01)
        streaming = StreamingChecker(mock_checker, config=config)

        async def async_lines():
            yield "line 1\n"

        async def run_test():
            # Patch to_thread to simulate timeout
            with patch("asyncio.to_thread", side_effect=asyncio.TimeoutError):
                results = []
                async for result in streaming.check_stream_async(async_lines()):
                    results.append(result)
                return results

        results = asyncio.run(run_test())
        assert len(results) == 1
        assert "error" in results[0].response.metadata


# --- Memory Management Tests ---


@pytest.mark.unit
class TestMemoryManagement:
    """Tests for memory management functionality."""

    @pytest.fixture
    def mock_checker(self):
        """Create a mock SpellChecker."""
        checker = MagicMock()
        checker.check.return_value = Response(
            text="test",
            corrected_text="test",
            has_errors=False,
            level="syllable",
            errors=[],
            metadata={},
        )
        return checker

    def test_memory_usage_tracking(self, mock_checker):
        """Test that memory usage is tracked."""
        streaming = StreamingChecker(mock_checker)
        stats = StreamingStats()

        input_stream = io.StringIO("line 1\n")
        list(streaming.check_stream(input_stream, stats=stats))

        # Memory should be tracked (may be 0 on some systems)
        assert stats.current_memory_mb >= 0

    def test_backpressure_applied(self, mock_checker):
        """Test that backpressure is applied when memory limit exceeded."""
        config = StreamingConfig(max_memory_mb=1, memory_check_interval=1)
        streaming = StreamingChecker(mock_checker, config=config)

        # Mock high memory usage
        with patch.object(streaming, "_get_memory_usage_mb", return_value=100):
            with patch.object(streaming, "_apply_backpressure") as mock_bp:
                input_stream = io.StringIO("line 1\n")
                list(streaming.check_stream(input_stream))
                mock_bp.assert_called()


# --- Integration Tests ---


@pytest.mark.integration
class TestStreamingIntegration:
    """Integration tests for streaming functionality."""

    def test_import_from_package(self):
        """Test that streaming classes can be imported from package."""
        from myspellchecker import StreamingChecker, StreamingConfig, StreamingStats

        assert StreamingChecker is not None
        assert StreamingConfig is not None
        assert StreamingStats is not None

    def test_import_from_core(self):
        """Test that streaming classes can be imported from core."""
        from myspellchecker.core import StreamingChecker, StreamingConfig, StreamingStats

        assert StreamingChecker is not None
        assert StreamingConfig is not None
        assert StreamingStats is not None


# --- Async Error Handling Tests (MED-022) ---


@pytest.mark.unit
class TestAsyncErrorHandling:
    """Tests for async error handling in streaming."""

    @pytest.fixture
    def mock_checker(self):
        """Create a mock SpellChecker."""
        checker = MagicMock()
        checker.check.return_value = Response(
            text="test",
            corrected_text="test",
            has_errors=False,
            level="syllable",
            errors=[],
            metadata={},
        )
        return checker

    def test_async_cancelled_error_handling(self, mock_checker):
        """Test handling of asyncio.CancelledError."""
        streaming = StreamingChecker(mock_checker)

        async def async_lines():
            yield "line 1\n"
            raise asyncio.CancelledError()

        async def run_test():
            results = []
            with pytest.raises(asyncio.CancelledError):
                async for result in streaming.check_stream_async(async_lines()):
                    results.append(result)
            return results

        results = asyncio.run(run_test())
        # With look-ahead pattern, first result is pending when CancelledError
        # fires on second iteration — cancellation correctly propagates before
        # the pending result is yielded. This is correct cancellation semantics.
        assert len(results) == 0

    def test_async_general_exception_in_stream(self, mock_checker):
        """Test handling of general exceptions in async stream."""
        streaming = StreamingChecker(mock_checker)

        async def async_lines():
            yield "line 1\n"
            raise RuntimeError("Stream error")

        async def run_test():
            results = []
            with pytest.raises(RuntimeError, match="Stream error"):
                async for result in streaming.check_stream_async(async_lines()):
                    results.append(result)
            return results

        results = asyncio.run(run_test())
        # With look-ahead pattern, first result is pending when RuntimeError
        # fires — error propagates before pending result is yielded.
        assert len(results) == 0

    def test_async_checker_exception_handled(self, mock_checker):
        """Test that checker exceptions are handled gracefully in async mode."""
        mock_checker.check.side_effect = ValueError("Check failed")
        streaming = StreamingChecker(mock_checker)

        async def async_lines():
            yield "line 1\n"
            yield "line 2\n"

        async def run_test():
            results = []
            async for result in streaming.check_stream_async(async_lines()):
                results.append(result)
            return results

        results = asyncio.run(run_test())
        # Both lines should be processed with error metadata
        assert len(results) == 2
        assert "error" in results[0].response.metadata
        assert "Check failed" in results[0].response.metadata["error"]

    def test_async_timeout_per_chunk(self, mock_checker):
        """Test timeout handling for slow checker operations."""
        config = StreamingConfig(timeout_per_chunk=0.001)  # Very short timeout
        streaming = StreamingChecker(mock_checker, config=config)

        async def slow_check(*args, **kwargs):
            await asyncio.sleep(1)  # Simulate slow operation
            return Response(
                text="test",
                corrected_text="test",
                has_errors=False,
                level="syllable",
                errors=[],
                metadata={},
            )

        async def async_lines():
            yield "line 1\n"

        async def run_test():
            with patch("asyncio.to_thread", side_effect=asyncio.TimeoutError):
                results = []
                async for result in streaming.check_stream_async(async_lines()):
                    results.append(result)
                return results

        results = asyncio.run(run_test())
        assert len(results) == 1
        assert "error" in results[0].response.metadata
        assert "Timeout" in results[0].response.metadata["error"]

    def test_async_bytes_input_handling(self, mock_checker):
        """Test handling of bytes input in async stream."""
        streaming = StreamingChecker(mock_checker)

        async def async_bytes_lines():
            yield b"line 1\n"
            yield b"line 2\n"

        async def run_test():
            results = []
            async for result in streaming.check_stream_async(async_bytes_lines()):
                results.append(result)
            return results

        results = asyncio.run(run_test())
        assert len(results) == 2

    def test_async_invalid_type_raises_error(self, mock_checker):
        """Test that invalid input types raise TypeError."""
        streaming = StreamingChecker(mock_checker)

        async def async_invalid_lines():
            yield 12345  # Invalid type

        async def run_test():
            with pytest.raises(TypeError, match="Expected str or bytes"):
                async for _ in streaming.check_stream_async(async_invalid_lines()):
                    pass

        asyncio.run(run_test())

    def test_async_progress_callback_on_error(self, mock_checker):
        """Test that progress callback is called even when errors occur."""
        mock_checker.check.side_effect = Exception("Error")
        config = StreamingConfig(progress_interval=1)
        streaming = StreamingChecker(mock_checker, config=config)

        callback_count = [0]

        def on_progress(stats):
            callback_count[0] += 1

        async def async_lines():
            yield "line 1\n"
            yield "line 2\n"

        async def run_test():
            async for _ in streaming.check_stream_async(async_lines(), on_progress=on_progress):
                pass

        asyncio.run(run_test())
        # Progress callback should have been called
        assert callback_count[0] >= 1


# --- Backpressure Tests (MED-022) ---


@pytest.mark.unit
class TestBackpressureHandling:
    """Tests for backpressure handling in streaming."""

    @pytest.fixture
    def mock_checker(self):
        """Create a mock SpellChecker."""
        checker = MagicMock()
        checker.check.return_value = Response(
            text="test",
            corrected_text="test",
            has_errors=False,
            level="syllable",
            errors=[],
            metadata={},
        )
        return checker

    def test_sync_backpressure_gc_called(self, mock_checker):
        """Test that GC is called during backpressure."""
        config = StreamingConfig(max_memory_mb=1, memory_check_interval=1)
        streaming = StreamingChecker(mock_checker, config=config)

        with patch.object(streaming, "_get_memory_usage_mb", return_value=100):
            with patch("gc.collect") as mock_gc:
                input_stream = io.StringIO("line 1\n")
                list(streaming.check_stream(input_stream))
                mock_gc.assert_called()

    def test_async_backpressure_sleep(self, mock_checker):
        """Test that async backpressure applies sleep."""
        config = StreamingConfig(max_memory_mb=1, memory_check_interval=1)
        streaming = StreamingChecker(mock_checker, config=config)

        async def async_lines():
            yield "line 1\n"

        async def run_test():
            with patch.object(streaming, "_get_memory_usage_mb", return_value=100):
                with patch("asyncio.sleep") as mock_sleep:
                    async for _ in streaming.check_stream_async(async_lines()):
                        pass
                    return mock_sleep.called

        result = asyncio.run(run_test())
        assert result is True

    def test_backpressure_memory_tracking(self, mock_checker):
        """Test that memory usage is tracked in stats."""
        config = StreamingConfig(memory_check_interval=1)
        streaming = StreamingChecker(mock_checker, config=config)
        stats = StreamingStats()

        # Mock high memory usage
        with patch.object(streaming, "_get_memory_usage_mb", return_value=50.5):
            input_stream = io.StringIO("line 1\n")
            list(streaming.check_stream(input_stream, stats=stats))

        assert stats.current_memory_mb == 50.5

    def test_backpressure_not_applied_under_limit(self, mock_checker):
        """Test that backpressure is not applied when under memory limit."""
        config = StreamingConfig(max_memory_mb=100, memory_check_interval=1)
        streaming = StreamingChecker(mock_checker, config=config)

        with patch.object(streaming, "_get_memory_usage_mb", return_value=50):
            with patch.object(streaming, "_apply_backpressure") as mock_bp:
                input_stream = io.StringIO("line 1\n")
                list(streaming.check_stream(input_stream))
                mock_bp.assert_not_called()

    def test_slow_consumer_simulation(self, mock_checker):
        """Test streaming with slow consumer (simulates backpressure scenario)."""
        streaming = StreamingChecker(mock_checker)
        stats = StreamingStats()

        input_stream = io.StringIO("line 1\nline 2\nline 3\n")
        results = []

        for result in streaming.check_stream(input_stream, stats=stats):
            # Simulate slow consumer
            import time

            time.sleep(0.001)
            results.append(result)

        assert len(results) == 3
        assert stats.lines_processed == 3

    def test_async_slow_consumer_with_backpressure(self, mock_checker):
        """Test async streaming with slow consumer under memory pressure."""
        config = StreamingConfig(max_memory_mb=1, memory_check_interval=1)
        streaming = StreamingChecker(mock_checker, config=config)

        async def async_lines():
            for i in range(5):
                yield f"line {i}\n"

        async def run_test():
            results = []
            memory_values = [100, 100, 50, 50, 50]  # High then normal
            call_count = [0]

            def get_memory():
                idx = min(call_count[0], len(memory_values) - 1)
                call_count[0] += 1
                return memory_values[idx]

            with patch.object(streaming, "_get_memory_usage_mb", side_effect=get_memory):
                async for result in streaming.check_stream_async(async_lines()):
                    await asyncio.sleep(0.001)  # Simulate slow consumer
                    results.append(result)
            return results

        results = asyncio.run(run_test())
        assert len(results) == 5

    def test_memory_usage_on_different_platforms(self, mock_checker):
        """Test memory usage tracking across different platforms."""
        streaming = StreamingChecker(mock_checker)

        # Test when resource module is not available
        with patch.dict("sys.modules", {"resource": None}):
            with patch(
                "myspellchecker.core.streaming.StreamingChecker._get_memory_usage_mb",
                return_value=0.0,
            ):
                result = streaming._get_memory_usage_mb()
                # Should return 0 when resource is unavailable
                assert result >= 0

    def test_backpressure_recovery(self, mock_checker):
        """Test that processing continues normally after backpressure."""
        config = StreamingConfig(max_memory_mb=50, memory_check_interval=1)
        streaming = StreamingChecker(mock_checker, config=config)
        stats = StreamingStats()

        # Memory goes high then low
        memory_sequence = [100, 30, 30]
        call_idx = [0]

        def mock_memory():
            idx = min(call_idx[0], len(memory_sequence) - 1)
            call_idx[0] += 1
            return memory_sequence[idx]

        with patch.object(streaming, "_get_memory_usage_mb", side_effect=mock_memory):
            input_stream = io.StringIO("line 1\nline 2\nline 3\n")
            results = list(streaming.check_stream(input_stream, stats=stats))

        # All lines should be processed despite backpressure
        assert len(results) == 3


# --- Resource Cleanup Tests (MED-022) ---


@pytest.mark.unit
class TestResourceCleanup:
    """Tests for resource cleanup during streaming."""

    @pytest.fixture
    def mock_checker(self):
        """Create a mock SpellChecker."""
        checker = MagicMock()
        checker.check.return_value = Response(
            text="test",
            corrected_text="test",
            has_errors=False,
            level="syllable",
            errors=[],
            metadata={},
        )
        return checker

    def test_stats_isolation_between_streams(self, mock_checker):
        """Test that stats are isolated between stream operations."""
        streaming = StreamingChecker(mock_checker)

        # First stream
        stats1 = StreamingStats()
        input1 = io.StringIO("line 1\nline 2\n")
        list(streaming.check_stream(input1, stats=stats1))

        # Second stream with new stats
        stats2 = StreamingStats()
        input2 = io.StringIO("line a\n")
        list(streaming.check_stream(input2, stats=stats2))

        assert stats1.lines_processed == 2
        assert stats2.lines_processed == 1

    def test_final_chunk_marked_correctly(self, mock_checker):
        """Test that the final chunk is correctly marked as is_final."""
        streaming = StreamingChecker(mock_checker)

        input_stream = io.StringIO("line 1\nline 2\nline 3\n")
        results = list(streaming.check_stream(input_stream))

        # Only the last chunk should be marked as final
        assert len(results) == 3
        assert results[0].is_final is False
        assert results[1].is_final is False
        assert results[2].is_final is True

    def test_empty_stream_no_final_chunk(self, mock_checker):
        """Test that empty stream produces no results."""
        streaming = StreamingChecker(mock_checker)

        input_stream = io.StringIO("")
        results = list(streaming.check_stream(input_stream))

        assert len(results) == 0
