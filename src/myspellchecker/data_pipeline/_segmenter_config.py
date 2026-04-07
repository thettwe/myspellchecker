"""
Segmenter configuration state, capability detection, and utility functions.

This module contains module-level state required for fork-based multiprocessing,
capability flags, and helper functions used by the segmenter pipeline.

These MUST remain at module level for fork() copy-on-write memory sharing to work.
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from typing import Any

import pyarrow as pa  # type: ignore

from ..core.constants import get_myanmar_char_set
from ..segmenters import DefaultSegmenter
from ..utils.logging_utils import get_logger
from .config import runtime_flags as _flags
from .pipeline_config_unified import WorkerTuningConfig
from .repair import SegmentationRepair

# =============================================================================
# CAPABILITY DETECTION
# =============================================================================
# Feature flags detected at module import time. These are immutable after
# initialization and encapsulated in a frozen dataclass for type safety.


@dataclass(frozen=True)
class _SegmenterCapabilities:
    """
    Immutable container for segmenter capability flags.

    Detected at module import time based on available Cython extensions
    and platform features. Frozen to prevent accidental modification.
    """

    has_cython_check: bool
    has_batch_processor: bool
    has_openmp: bool
    has_cython_repair: bool
    use_fork_optimization: bool
    openmp_threads_per_worker: int


def _detect_capabilities() -> _SegmenterCapabilities:
    """Detect available capabilities at module import time."""
    # Check for Cython Myanmar string checker
    has_cython_check = False
    try:
        from ..text.normalize_c import is_myanmar_string as _  # noqa: F401

        has_cython_check = True
    except ImportError:
        pass

    # Check for Cython batch processor
    has_batch_processor = False
    has_openmp = False
    try:
        from myspellchecker.data_pipeline.batch_processor import (
            has_openmp as check_openmp,
        )

        has_batch_processor = True
        has_openmp = check_openmp()
    except ImportError:
        pass

    # Check for Cython repair module
    has_cython_repair = False
    try:
        from myspellchecker.data_pipeline.repair_c import (
            init_repair_module as _,  # noqa: F401
        )

        has_cython_repair = True
    except ImportError:
        pass

    # Platform-based fork optimization (macOS/Linux support fork with COW)
    use_fork_optimization = platform.system() in ("Darwin", "Linux")

    # OpenMP thread count per worker (configurable via environment)
    # Uses MYSPELL_ prefix for consistency with other environment variables
    openmp_raw = os.environ.get("MYSPELL_OPENMP_THREADS", "4")
    try:
        openmp_threads = int(openmp_raw)
    except ValueError:
        get_logger(__name__).warning(
            "Invalid MYSPELL_OPENMP_THREADS=%r; falling back to 4", openmp_raw
        )
        openmp_threads = 4

    if openmp_threads < 1:
        get_logger(__name__).warning(
            "MYSPELL_OPENMP_THREADS must be >= 1, got %d; using 1", openmp_threads
        )
        openmp_threads = 1

    return _SegmenterCapabilities(
        has_cython_check=has_cython_check,
        has_batch_processor=has_batch_processor,
        has_openmp=has_openmp,
        has_cython_repair=has_cython_repair,
        use_fork_optimization=use_fork_optimization,
        openmp_threads_per_worker=openmp_threads,
    )


# Module-level capabilities instance (immutable after import)
_CAPABILITIES = _detect_capabilities()


# =============================================================================
# PROCESS STATE MANAGEMENT
# =============================================================================
# Mutable state required for fork-based multiprocessing. This state MUST be
# module-level global for fork() copy-on-write memory sharing to work.
# Encapsulated in a class for better organization and type safety.


class _ProcessState:
    """
    Mutable process state for fork-based multiprocessing.

    This class encapsulates the global mutable state required for fork-based
    multiprocessing optimization. The state MUST remain at module level for
    fork() copy-on-write memory sharing to work correctly.

    State Categories:
        - Preloaded models: Loaded in parent process, inherited by workers via COW
        - Worker state: Initialized in each worker process after fork
        - Progress tracking: Shared counter for real-time progress display
    """

    def __init__(self) -> None:
        """Initialize empty process state."""
        # Pre-loaded models for fork-based multiprocessing (parent process)
        self.preloaded_segmenter: DefaultSegmenter | None = None
        self.preloaded_repair: SegmentationRepair | None = None
        self.models_preloaded: bool = False

        # Worker process state (initialized after fork)
        self.worker_segmenter: DefaultSegmenter | None = None
        self.worker_repair: SegmentationRepair | None = None
        self.is_forked_worker: bool = False

        # Shared progress counter for real-time progress display
        # This is a multiprocessing.Value that workers increment after each batch
        self.shared_progress_counter: Any | None = None

        # Transformer segmenter config (stored for passing to DefaultSegmenter)
        self.seg_model: str | None = None
        self.seg_device: int = -1


# Module-level process state instance
_STATE = _ProcessState()


# =============================================================================
# SHARED RUNTIME FLAGS (replaces per-module globals)
# =============================================================================

# Legacy module-level aliases — kept for fork()-based COW worker compatibility.
# Workers (e.g. _segmenter_workers.py) still read these after fork().
_allow_extended_myanmar = _flags.allow_extended_myanmar
_auto_detect_pre_segmented = _flags.auto_detect_pre_segmented


# =============================================================================
# LAZY IMPORTS FOR CYTHON MODULES
# =============================================================================
# Import Cython functions only when capabilities are available.
# This avoids import errors on systems without Cython extensions.

if _CAPABILITIES.has_cython_check:
    from ..text.normalize_c import is_myanmar_string_scoped as c_is_myanmar_string_scoped

if _CAPABILITIES.has_batch_processor:
    from myspellchecker.data_pipeline.batch_processor import (  # noqa: F401
        process_batch,
        process_batch_parallel,
    )
    from myspellchecker.data_pipeline.batch_processor import (  # noqa: F401
        set_crf_engine as bp_set_crf_engine,
    )
    from myspellchecker.data_pipeline.batch_processor import (  # noqa: F401
        set_crf_tagger as bp_set_crf_tagger,
    )

if _CAPABILITIES.has_cython_repair:
    from myspellchecker.data_pipeline.repair_c import (  # noqa: F401
        init_repair_module as c_init_repair,
    )
    from myspellchecker.data_pipeline.repair_c import (  # noqa: F401
        repair_batch as c_repair_batch,
    )


# =============================================================================
# MODULE CONSTANTS
# =============================================================================

# Zero-width characters to skip when finding first significant char
ZERO_WIDTH_CHARS: frozenset[str] = frozenset(
    [
        "\u200b",  # Zero Width Space
        "\u200c",  # Zero Width Non-Joiner
        "\u200d",  # Zero Width Joiner
        "\ufeff",  # Byte Order Mark / Zero Width No-Break Space
    ]
)

# Leading punctuation to skip (ASCII brackets/quotes + Myanmar punctuation)
LEADING_PUNCT_CHARS: frozenset[str] = frozenset(
    [
        # ASCII punctuation commonly attached to tokens
        "(",
        ")",
        "[",
        "]",
        "{",
        "}",
        '"',
        "'",
        "`",
        "<",
        ">",
        # Myanmar punctuation (U+104A-U+104F)
        "\u104a",  # ၊ (comma)
        "\u104b",  # ။ (full stop)
        "\u104c",  # ၌ (locative)
        "\u104d",  # ၍ (completed)
        "\u104e",  # ၎ (aforementioned)
        "\u104f",  # ၏ (genitive)
    ]
)


def _format_count(n: int) -> str:
    """Format large numbers for display (e.g., 285439 -> '285K')."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.0f}K"
    return str(n)


def _first_significant_char(text: str) -> str:
    """Extract first significant character, skipping whitespace/zero-width/punctuation.

    Args:
        text: Input token text.

    Returns:
        First significant character, or empty string if none found.
    """
    for ch in text:
        if ch.isspace():
            continue
        if ch in ZERO_WIDTH_CHARS:
            continue
        if ch in LEADING_PUNCT_CHARS:
            continue
        return ch
    return ""


def is_myanmar_token(text: str, *, allow_extended: bool = False) -> bool:
    """Check if token is Myanmar using first-significant-char semantics.

    Skips leading whitespace, zero-width characters, and punctuation
    to find the first significant character, then checks if it's Myanmar.

    Args:
        text: Token to check.
        allow_extended: If True, accept Extended Myanmar characters including:
            - Extended Core Block (U+1050-U+109F)
            - Extended-A (U+AA60-U+AA7F)
            - Extended-B (U+A9E0-U+A9FF)
            - Non-standard core chars (U+1022, U+1028, U+1033-U+1035)

    Returns:
        True if first significant character is Myanmar within configured scope.
    """
    if _CAPABILITIES.has_cython_check:
        return bool(c_is_myanmar_string_scoped(text, allow_extended))

    # Python fallback: first-significant-char semantics
    first = _first_significant_char(text)
    if not first:
        return False
    valid_chars = get_myanmar_char_set(allow_extended)
    return first in valid_chars


def _is_pre_segmented(text: str, threshold: float = 0.7) -> bool:
    """
    Detect if Myanmar text is already space-delimited (pre-segmented).

    Heuristic: If >threshold of space-delimited tokens are valid Myanmar
    words/syllables, the text is likely pre-segmented.

    This uses a lightweight check (is_myanmar_token) rather than full word
    validation to avoid circular imports and keep detection fast.

    Args:
        text: Input text to check.
        threshold: Minimum ratio of valid Myanmar tokens to total tokens.

    Returns:
        True if text appears to be pre-segmented.
    """
    tokens = text.split()
    if len(tokens) < 2:
        return False  # Single token -- can't determine

    myanmar_token_count = 0
    for token in tokens:
        stripped = token.strip()
        if stripped and is_myanmar_token(stripped, allow_extended=_flags.allow_extended_myanmar):
            myanmar_token_count += 1

    ratio = myanmar_token_count / len(tokens)
    return ratio >= threshold


# Define schema for segmentation output
SEGMENT_SCHEMA = pa.schema(
    [
        ("text", pa.string()),
        ("source", pa.string()),
        ("syllables", pa.list_(pa.string())),
        ("words", pa.list_(pa.string())),
        ("syllable_count", pa.int32()),
        ("word_count", pa.int32()),
    ]
)


def get_optimal_worker_count(
    file_count: int = 0,
    file_size_mb: float = 0,
    total_sentences: int = 0,
    tuning: WorkerTuningConfig | None = None,
) -> int:
    """
    Determine optimal number of workers based on system resources.

    Considers:
    - CPU cores available
    - Memory constraints (models + data processing)
    - Workload size (no point having more workers than files)
    - Hyperthreading (typically use physical cores for CPU-bound work)
    - Total sentences to process (large corpora need more memory)

    Args:
        file_count: Number of files to process
        file_size_mb: Total size of files in MB
        total_sentences: Total sentences to process (memory estimation)
        tuning: Worker tuning configuration (uses defaults if None)

    Returns:
        Recommended number of workers
    """
    if tuning is None:
        tuning = WorkerTuningConfig()

    try:
        import psutil

        cpu_count = psutil.cpu_count(logical=False) or os.cpu_count() or 4
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
    except ImportError:
        cpu_count = os.cpu_count() or 4
        available_memory_gb = 8.0  # Conservative default

    # Base calculation: physical cores
    # For CPU-bound work, logical cores (hyperthreading) often hurts
    base_workers = cpu_count

    # Memory estimation per worker:
    # - Base model memory (mmap shared via COW, still needs overhead)
    # - Processing memory: depends on chunk size
    #   - Each sentence ~100 chars, ~500 bytes with overhead
    #   - With file_chunk_divisor chunks, each worker processes a
    #     fraction of total_sentences
    #   - Add 2x safety factor for intermediate data structures

    base_mem = tuning.base_memory_per_worker_gb
    if total_sentences > 0 and file_count > 0:
        sentences_per_chunk = total_sentences / max(file_count, tuning.file_chunk_divisor)
        # Estimate: 1KB per sentence (including intermediates)
        data_mem = (sentences_per_chunk * 1024) / (1024**3)
        memory_per_worker_gb = base_mem + data_mem
    else:
        memory_per_worker_gb = base_mem

    # Use only usable_memory_ratio of available memory for OS room
    usable_memory_gb = available_memory_gb * tuning.usable_memory_ratio
    memory_limited_workers = max(1, int(usable_memory_gb / memory_per_worker_gb))

    # Don't exceed file count (no point having idle workers)
    multiplier = tuning.file_limited_multiplier
    if file_count > 0:
        file_limited_workers = min(file_count, base_workers * multiplier)
    else:
        file_limited_workers = base_workers * multiplier

    # Take minimum of all constraints
    optimal = min(base_workers, memory_limited_workers, file_limited_workers)

    # Ensure at least 1 worker, at most max_workers_cap
    return max(1, min(optimal, tuning.max_workers_cap))


def get_optimal_batch_size(
    file_size_mb: float = 0,
    available_memory_gb: float = 0,
    tuning: WorkerTuningConfig | None = None,
) -> int:
    """
    Determine optimal batch size based on file size and memory.

    Larger batches reduce overhead but increase memory pressure.

    Args:
        file_size_mb: Size of input file in MB
        available_memory_gb: Available system memory in GB
        tuning: Worker tuning configuration (uses defaults if None)

    Returns:
        Recommended batch size
    """
    if tuning is None:
        tuning = WorkerTuningConfig()

    if available_memory_gb == 0:
        try:
            import psutil

            available_memory_gb = psutil.virtual_memory().available / (1024**3)
        except ImportError:
            available_memory_gb = 4.0  # Conservative default

    # Base batch size determined by file size thresholds
    if file_size_mb > tuning.large_file_threshold_mb:
        base_batch = tuning.large_file_batch
    elif file_size_mb > tuning.medium_file_threshold_mb:
        base_batch = tuning.medium_file_batch
    else:
        base_batch = tuning.small_file_batch

    # Memory adjustment: reduce if memory is tight
    if available_memory_gb < tuning.low_memory_threshold_gb:
        return max(5000, base_batch // 2)
    elif available_memory_gb < tuning.medium_memory_threshold_gb:
        return base_batch
    else:
        # Plenty of memory: can use larger batches
        return min(
            base_batch * tuning.high_memory_batch_multiplier,
            tuning.max_batch_cap,
        )


# Default retry settings for worker error recovery
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BASE_DELAY = 1.0  # seconds
