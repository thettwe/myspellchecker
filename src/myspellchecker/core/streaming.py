"""
Streaming Support Module for SpellChecker.

This module provides streaming interfaces for processing large text files
without loading them entirely into memory. It supports:

- Generator-based streaming (check_stream)
- Async streaming (check_stream_async)
- Progress callbacks for long operations
- Memory usage limits and backpressure
- Sentence-by-sentence context validation

Example:
    >>> from myspellchecker import SpellChecker
    >>> from myspellchecker.core.streaming import StreamingChecker
    >>>
    >>> checker = SpellChecker()
    >>> streaming = StreamingChecker(checker)
    >>>
    >>> # Process large file with progress callback
    >>> def on_progress(stats):
    ...     print(f"Processed {stats.lines_processed} lines")
    >>>
    >>> with open("large_file.txt") as f:
    ...     for result in streaming.check_stream(f, on_progress=on_progress):
    ...         if result.has_errors:
    ...             print(result.errors)

    >>> # Async usage
    >>> async for result in streaming.check_stream_async(async_reader):
    ...     process(result)
"""

from __future__ import annotations

import asyncio
import logging
import re
import sys
import time
from copy import copy
from dataclasses import dataclass, field
from typing import (
    IO,
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Callable,
    Iterator,
    Protocol,
    TextIO,
    runtime_checkable,
)

from myspellchecker.core.constants import ValidationLevel
from myspellchecker.core.correction_utils import generate_corrected_text
from myspellchecker.core.response import Response

if TYPE_CHECKING:
    from myspellchecker.core.spellchecker import SpellChecker

logger = logging.getLogger(__name__)

__all__ = [
    "AsyncTextReader",
    "ChunkResult",
    "ProgressCallback",
    "StreamingChecker",
    "StreamingConfig",
    "StreamingStats",
]


@dataclass
class StreamingConfig:
    """
    Configuration for streaming spell checking.

    Attributes:
        chunk_size: Number of lines to process in each chunk (default: 100).
        max_memory_mb: Maximum memory usage in MB before applying backpressure (default: 100).
        sentence_boundary_pattern: Regex pattern for sentence boundaries.
        enable_cross_sentence_context: Enable context validation across sentence boundaries.
        progress_interval: Lines between progress callbacks (default: 1000).
        timeout_per_chunk: Maximum seconds per chunk processing (default: 30).
    """

    chunk_size: int = 100
    max_memory_mb: int = 100
    sentence_boundary_pattern: str = r"[။!?]+"
    enable_cross_sentence_context: bool = True
    progress_interval: int = 1000
    timeout_per_chunk: float = 30.0

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        if self.max_memory_mb < 1:
            raise ValueError("max_memory_mb must be >= 1")
        if self.progress_interval < 1:
            raise ValueError("progress_interval must be >= 1")
        if self.timeout_per_chunk <= 0:
            raise ValueError("timeout_per_chunk must be > 0")
        try:
            re.compile(self.sentence_boundary_pattern)
        except re.error as e:
            raise ValueError(f"Invalid sentence_boundary_pattern: {e}") from e


@dataclass
class StreamingStats:
    """
    Statistics for streaming processing.

    Updated incrementally as processing progresses.
    """

    bytes_processed: int = 0
    lines_processed: int = 0
    sentences_processed: int = 0
    errors_found: int = 0
    chunks_processed: int = 0
    start_time: float = field(default_factory=time.time)
    current_memory_mb: float = 0.0

    @property
    def elapsed_time(self) -> float:
        """Elapsed time in seconds since processing started."""
        return time.time() - self.start_time

    @property
    def lines_per_second(self) -> float:
        """Processing rate in lines per second."""
        elapsed = self.elapsed_time
        return self.lines_processed / elapsed if elapsed > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert stats to dictionary for serialization."""
        return {
            "bytes_processed": self.bytes_processed,
            "lines_processed": self.lines_processed,
            "sentences_processed": self.sentences_processed,
            "errors_found": self.errors_found,
            "chunks_processed": self.chunks_processed,
            "elapsed_time": self.elapsed_time,
            "lines_per_second": self.lines_per_second,
            "current_memory_mb": self.current_memory_mb,
        }


@dataclass
class ChunkResult:
    """
    Result for a processed chunk.

    Contains the Response object and metadata about the chunk.
    """

    response: Response
    line_number: int
    chunk_index: int
    is_final: bool = False


# Protocol for progress callbacks
ProgressCallback = Callable[[StreamingStats], None]


@runtime_checkable
class AsyncTextReader(Protocol):
    """Protocol for async text readers."""

    async def readline(self) -> str:
        """Read a single line asynchronously."""
        ...

    def __aiter__(self) -> AsyncIterator[str]:
        """Async iterator over lines."""
        ...

    async def __anext__(self) -> str:
        """Get next line."""
        ...


class StreamingChecker:
    """
    Streaming interface for SpellChecker.

    Provides memory-efficient processing of large text files with:
    - Generator-based streaming for sync contexts
    - Async streaming for async contexts
    - Progress callbacks for monitoring
    - Memory limits and backpressure

    The streaming checker processes text line-by-line or sentence-by-sentence,
    yielding results incrementally instead of loading everything into memory.

    Example:
        >>> checker = SpellChecker()
        >>> streaming = StreamingChecker(checker)
        >>>
        >>> # Simple streaming
        >>> with open("large.txt") as f:
        ...     for result in streaming.check_stream(f):
        ...         process(result)
        >>>
        >>> # With progress tracking
        >>> stats = StreamingStats()
        >>> for result in streaming.check_stream(f, stats=stats):
        ...     if stats.lines_processed % 1000 == 0:
        ...         print(f"Progress: {stats.lines_per_second:.1f} lines/sec")
    """

    def __init__(
        self,
        checker: SpellChecker,
        config: StreamingConfig | None = None,
    ) -> None:
        """
        Initialize StreamingChecker.

        Args:
            checker: SpellChecker instance to use for validation.
            config: StreamingConfig for tuning behavior.
        """
        self.checker = checker
        self.config = config or StreamingConfig()
        self._sentence_pattern = re.compile(self.config.sentence_boundary_pattern)
        self._previous_context: str | None = None

    def check_stream(
        self,
        input_stream: TextIO | IO[str] | Iterator[str],
        level: ValidationLevel = ValidationLevel.SYLLABLE,
        on_progress: ProgressCallback | None = None,
        stats: StreamingStats | None = None,
    ) -> Iterator[ChunkResult]:
        """
        Stream spell check results from an input stream.

        Processes the input line-by-line, yielding ChunkResult objects
        for each line. Memory usage stays bounded regardless of file size.

        Args:
            input_stream: Text stream to read from (file, StringIO, or iterator).
            level: Validation level (syllable, word).
            on_progress: Optional callback for progress updates.
            stats: Optional StreamingStats object to update (created if None).

        Yields:
            ChunkResult objects for each processed line.

        Example:
            >>> with open("file.txt") as f:
            ...     for result in streaming.check_stream(f):
            ...         if result.response.has_errors:
            ...             print(f"Line {result.line_number}: {result.response.errors}")
        """
        if stats is None:
            stats = StreamingStats()

        chunk_index = 0
        line_number = 0

        lines: Iterator[str] = iter(input_stream)

        # Use look-ahead pattern to properly set is_final on the last chunk
        pending_result: ChunkResult | None = None

        for line in lines:
            line_number += 1
            line_text = line.rstrip("\n\r")

            # Track bytes and lines
            stats.bytes_processed += len(line.encode("utf-8", errors="replace"))
            stats.lines_processed = line_number

            # Skip empty lines
            if not line_text.strip():
                continue

            # Check memory usage
            stats.current_memory_mb = self._get_memory_usage_mb()
            if stats.current_memory_mb > self.config.max_memory_mb:
                self._apply_backpressure()

            # Process the line
            try:
                response = self.checker.check(line_text, level=level)
            # Catch all to prevent single-sentence failures from crashing the stream
            except Exception as e:
                logger.warning("Sentence check failed, continuing stream: %s", e, exc_info=True)
                response = Response(
                    text=line_text,
                    corrected_text=line_text,
                    has_errors=False,
                    level=level.value,
                    errors=[],
                    metadata={"error": str(e)},
                )

            if response.has_errors:
                stats.errors_found += len(response.errors)

            # Count sentences
            sentences = self._sentence_pattern.split(line_text)
            stats.sentences_processed += len([s for s in sentences if s.strip()])

            # Yield previous result (not final) before creating new one
            if pending_result is not None:
                yield pending_result

            # Create new pending result
            chunk_index += 1
            stats.chunks_processed = chunk_index
            pending_result = ChunkResult(
                response=response,
                line_number=line_number,
                chunk_index=chunk_index,
                is_final=False,  # Will be updated for the last chunk
            )

            # Progress callback
            if on_progress and stats.lines_processed % self.config.progress_interval == 0:
                on_progress(stats)

        # Yield the final result with is_final=True
        if pending_result is not None:
            pending_result.is_final = True
            yield pending_result

        # Final progress callback
        if on_progress:
            on_progress(stats)

    async def check_stream_async(
        self,
        input_stream: AsyncTextReader | AsyncIterator[str],
        level: ValidationLevel = ValidationLevel.SYLLABLE,
        on_progress: ProgressCallback | None = None,
        stats: StreamingStats | None = None,
    ) -> AsyncIterator[ChunkResult]:
        """
        Asynchronously stream spell check results.

        Non-blocking version of check_stream for async contexts.
        Uses asyncio.to_thread for CPU-bound spell checking.

        Args:
            input_stream: Async text reader or async iterator of lines.
            level: Validation level.
            on_progress: Optional callback for progress updates.
            stats: Optional StreamingStats object to update.

        Yields:
            ChunkResult objects for each processed line.

        Example:
            >>> async with aiofiles.open("file.txt") as f:
            ...     async for result in streaming.check_stream_async(f):
            ...         await process_async(result)
        """
        if stats is None:
            stats = StreamingStats()

        chunk_index = 0
        line_number = 0

        # Use look-ahead pattern to properly set is_final on the last chunk
        pending_result: ChunkResult | None = None

        async for line in input_stream:
            line_number += 1
            # Handle different line types consistently
            if isinstance(line, str):
                line_text = line.rstrip("\n\r")
            elif isinstance(line, bytes):
                line_text = line.decode("utf-8").rstrip("\n\r")
            else:
                raise TypeError(f"Expected str or bytes, got {type(line).__name__}")

            stats.bytes_processed += len(
                line.encode("utf-8", errors="replace") if isinstance(line, str) else line
            )
            stats.lines_processed = line_number

            if not line_text.strip():
                continue

            # Check memory
            stats.current_memory_mb = self._get_memory_usage_mb()
            if stats.current_memory_mb > self.config.max_memory_mb:
                await asyncio.sleep(0.01)  # Async backpressure

            # Run CPU-bound check in thread pool
            try:
                response = await asyncio.wait_for(
                    asyncio.to_thread(self.checker.check, line_text, level),
                    timeout=self.config.timeout_per_chunk,
                )
            except asyncio.TimeoutError:
                response = Response(
                    text=line_text,
                    corrected_text=line_text,
                    has_errors=False,
                    level=level.value,
                    errors=[],
                    metadata={"error": "Timeout exceeded"},
                )
            # Catch all to prevent single-sentence failures from crashing the stream
            except Exception as e:
                logger.warning("Sentence check failed, continuing stream: %s", e, exc_info=True)
                response = Response(
                    text=line_text,
                    corrected_text=line_text,
                    has_errors=False,
                    level=level.value,
                    errors=[],
                    metadata={"error": str(e)},
                )

            if response.has_errors:
                stats.errors_found += len(response.errors)

            sentences = self._sentence_pattern.split(line_text)
            stats.sentences_processed += len([s for s in sentences if s.strip()])

            # Yield previous result (not final) before creating new one
            if pending_result is not None:
                yield pending_result

            chunk_index += 1
            stats.chunks_processed = chunk_index
            pending_result = ChunkResult(
                response=response,
                line_number=line_number,
                chunk_index=chunk_index,
                is_final=False,
            )

            if on_progress and stats.lines_processed % self.config.progress_interval == 0:
                on_progress(stats)

        # Yield the final result with is_final=True
        if pending_result is not None:
            pending_result.is_final = True
            yield pending_result

        if on_progress:
            on_progress(stats)

    def check_sentences(
        self,
        text: str,
        level: ValidationLevel = ValidationLevel.WORD,
        on_progress: ProgressCallback | None = None,
    ) -> Iterator[ChunkResult]:
        """
        Check text sentence-by-sentence with context preservation.

        Splits text on sentence boundaries and validates each sentence
        while maintaining cross-sentence context for better accuracy.

        Args:
            text: Full text to check.
            level: Validation level (defaults to WORD for context).
            on_progress: Optional callback for progress updates.

        Yields:
            ChunkResult for each sentence.

        Example:
            >>> text = "ဤစာကြောင်း။ ဒုတိယစာကြောင်း။"
            >>> for result in streaming.check_sentences(text):
            ...     print(f"Sentence {result.chunk_index}: {result.response.text}")
        """
        stats = StreamingStats()

        # Reset context from any previous document to avoid contamination
        self._previous_context = None

        # Split into sentences while preserving delimiters
        sentences = self._split_sentences(text)

        # Use look-ahead pattern to properly set is_final on the last chunk
        pending_result: ChunkResult | None = None
        chunk_index = 0

        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue

            stats.bytes_processed += len(sentence.encode("utf-8", errors="replace"))
            stats.sentences_processed += 1

            # Check with context from previous sentence if enabled
            text_to_check = sentence
            if self.config.enable_cross_sentence_context and self._previous_context:
                # Add previous context but only check new sentence
                context_text = self._previous_context + " " + sentence
                text_to_check = context_text

            try:
                response = self.checker.check(text_to_check, level=level)
            except Exception as e:
                logger.warning("Sentence check failed, continuing stream: %s", e, exc_info=True)
                response = Response(
                    text=text_to_check,
                    corrected_text=text_to_check,
                    has_errors=False,
                    level=level.value,
                    errors=[],
                    metadata={"error": str(e)},
                )

            # Adjust positions if we used context
            if self.config.enable_cross_sentence_context and self._previous_context:
                context_offset = len(self._previous_context) + 1
                adjusted_errors = []
                for error in response.errors:
                    if error.position >= context_offset:
                        # Clone error with adjusted position
                        adjusted_error = copy(error)
                        adjusted_error.position -= context_offset
                        adjusted_errors.append(adjusted_error)
                corrected = (
                    generate_corrected_text(sentence, adjusted_errors)
                    if adjusted_errors
                    else sentence
                )
                response = Response(
                    text=sentence,
                    corrected_text=corrected,
                    has_errors=len(adjusted_errors) > 0,
                    level=response.level,
                    errors=adjusted_errors,
                    metadata=response.metadata,
                )

            if response.has_errors:
                stats.errors_found += len(response.errors)

            # Store context for next sentence (last N characters)
            if self.config.enable_cross_sentence_context:
                self._previous_context = sentence[-100:] if len(sentence) > 100 else sentence

            # Yield previous result (not final) before creating new one
            if pending_result is not None:
                yield pending_result

            chunk_index += 1
            stats.chunks_processed = chunk_index
            pending_result = ChunkResult(
                response=response,
                line_number=i + 1,
                chunk_index=chunk_index,
                is_final=False,  # Will be updated for the last chunk
            )

            if on_progress and stats.sentences_processed % self.config.progress_interval == 0:
                on_progress(stats)

        # Yield the final result with is_final=True
        if pending_result is not None:
            pending_result.is_final = True
            yield pending_result

        if on_progress:
            on_progress(stats)

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences while preserving delimiters."""
        parts = self._sentence_pattern.split(text)
        delimiters = self._sentence_pattern.findall(text)

        sentences = []
        for i, part in enumerate(parts):
            if part.strip():
                sentence = part
                if i < len(delimiters):
                    sentence += delimiters[i]
                sentences.append(sentence.strip())
        return sentences

    def _get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB (RSS, not peak)."""
        try:
            import psutil

            return psutil.Process().memory_info().rss / (1024 * 1024)
        except ImportError:
            pass
        try:
            import resource

            usage = resource.getrusage(resource.RUSAGE_SELF)
            # ru_maxrss is peak RSS, not current — fallback only
            if sys.platform.startswith("darwin"):
                return usage.ru_maxrss / (1024 * 1024)  # bytes to MB
            else:
                return usage.ru_maxrss / 1024  # KB to MB
        except (ImportError, AttributeError):
            # Windows or resource not available
            return 0.0

    def _apply_backpressure(self) -> None:
        """Apply backpressure when memory limit is reached."""
        import gc

        gc.collect()
        time.sleep(0.01)  # Small delay to allow cleanup
