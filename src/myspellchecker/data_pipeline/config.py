"""
Pipeline configuration dataclass.

This module defines configuration for the data pipeline, allowing customization
of batch sizes, parallelization, and processing parameters.

Also provides ``PipelineRuntimeFlags``, a thread-aware singleton that replaces
the per-module ``_allow_extended_myanmar`` / ``_remove_segmentation_markers`` /
``_deduplicate_lines`` / ``_auto_detect_pre_segmented`` globals.  Pipeline.py
populates it once before fork; every module reads from the same object.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from ..core.constants import DEFAULT_BATCH_SIZE
from ..utils.logging_utils import get_logger

if TYPE_CHECKING:
    from ..utils.console import PipelineConsole
    from .reporter import PipelineReporter

_logger = get_logger(__name__)

__all__ = [
    "FrequencyBuilderConfig",
    "IngesterConfig",
    "PackagerConfig",
    "PipelineConfig",
    "PipelineRuntimeFlags",
    "SegmenterConfig",
]


@dataclass
class PipelineConfig:
    """
    Configuration for the data pipeline.

    This configuration controls how the pipeline processes corpus data,
    including batch sizes, parallelization, and output formatting.

    Attributes:
        batch_size: Number of records to process per batch (default: 10000).
            Larger values use more memory but may be faster.
        num_shards: Number of shards to split ingested data into (default: 20).
            More shards enable better parallelization but create more files.
        num_workers: Number of parallel workers for segmentation (default: None).
            If None, auto-detects based on CPU cores and available memory.
        min_frequency: Minimum frequency for words to be included (default: 50).
            Higher values reduce dictionary size and filter training data noise.
            Default of 50 matches SpellCheckerConfig.symspell.count_threshold.
        deduplicate_lines: Deduplicate lines during ingestion (default: True).
            Uses hash-based tracking to skip duplicate lines both within files and
            across files. Disable with ``--no-dedup`` CLI flag.
        remove_segmentation_markers: Strip artificial word segmentation markers
            (spaces/underscores between Myanmar characters) from corpus text (default: True).
            Disable with ``--no-desegment`` CLI flag for pre-segmented text that should
            preserve its segmentation.
        allow_extended_myanmar: Include extended Myanmar blocks (default: False).
            When False (default): Only strict Burmese (U+1000-U+104F minus non-standard).
            When True: Includes Extended Core (U+1050-U+109F), Extended-A (U+AA60-U+AA7F),
            Extended-B (U+A9E0-U+A9FF), and non-standard core chars.
        keep_intermediate: Whether to keep intermediate files after build (default: False).
            Useful for debugging but uses more disk space.
        text_col: Column name for CSV/TSV text ingestion (default: "text").
        json_key: Key name for JSON text ingestion (default: "text").
        word_engine: Word segmentation engine (default: "myword").
            Options: "crf", "myword", "transformer".
        enable_resume: Enable resume capability for interrupted builds (default: True).
        allow_partial_ingestion: Continue build when some input files fail ingestion
            (default: False). When False, any failed file raises IngestionError.
        validate_inputs: Validate input files before processing (default: True).
        disk_space_check_mb: Minimum disk space required in MB (default: 51200).
            Set to 0 to disable disk space checking.
        worker_timeout: Timeout in seconds for worker processes (default: 1800).
            Controls how long to wait for parallel processing operations.
        pos_tagger: POS tagger configuration (default: None).
            Optional POSTaggerConfig for configuring part-of-speech tagging
            during pipeline processing.
            Supports rule-based, transformer, and custom taggers.
        console: Console instance for pipeline output (default: None).
            If None, creates a new PipelineConsole instance.
            Useful for dependency injection in tests or custom output handling.
        reporter: Reporter instance for progress reporting (default: None).
            If None, creates a new PipelineReporter using the console.
            Useful for testing with MockReporter or custom reporting.

    Example:
        >>> from myspellchecker.data_pipeline import PipelineConfig, Pipeline
        >>>
        >>> # Use defaults
        >>> config = PipelineConfig()
        >>> pipeline = Pipeline(config=config)
        >>>
        >>> # Custom configuration for large corpus
        >>> config = PipelineConfig(
        ...     batch_size=50000,
        ...     num_shards=50,
        ...     num_workers=8,
        ...     min_frequency=100,  # Higher threshold for cleaner data
        ... )
        >>> pipeline = Pipeline(config=config)
    """

    # Processing parameters
    batch_size: int = DEFAULT_BATCH_SIZE
    num_shards: int = 20
    num_workers: int | None = None  # None = auto-detect

    # Filtering parameters
    min_frequency: int = 50
    min_syllable_frequency: int = 1
    min_bigram_count: int = 10
    min_trigram_count: int = 20
    min_fourgram_count: int = 3
    min_fivegram_count: int = 2

    # Corpus cleaning
    deduplicate_lines: bool = True  # Deduplicate lines during ingestion
    remove_segmentation_markers: bool = True  # Strip spaces/underscores between Myanmar chars

    # Extended Myanmar scope
    # When False (default): Only strict Burmese (U+1000-U+104F minus non-standard).
    # When True: Includes Extended Core (U+1050-U+109F), Extended-A (U+AA60-U+AA7F),
    # Extended-B (U+A9E0-U+A9FF), and non-standard core chars (U+1022, U+1028, U+1033-U+1035).
    allow_extended_myanmar: bool = False

    # File handling
    keep_intermediate: bool = False
    text_col: str = "text"
    json_key: str = "text"

    # Segmentation
    word_engine: str = "myword"

    # Transformer word segmentation options (only used when word_engine="transformer")
    seg_model: str | None = None
    seg_device: int = -1

    # POS Tagger configuration (new pluggable system)
    pos_tagger: POSTaggerConfig | None = None  # type: ignore  # noqa: F821

    # Pipeline behavior
    enable_resume: bool = True
    allow_partial_ingestion: bool = False
    validate_inputs: bool = True
    disk_space_check_mb: int = 51200  # 50 GB minimum for large corpus builds

    # Worker timeout (in seconds) for parallel processing operations
    worker_timeout: int = 1800  # 30 minutes default (for large corpus files)

    # Work directory (None = use temp directory)
    work_dir: str | Path | None = None

    # Console for output (dependency injection for testing)
    console: PipelineConsole | None = field(default=None)  # type: ignore

    # Reporter for progress reporting (dependency injection for testing)
    reporter: PipelineReporter | None = field(default=None)  # type: ignore

    # Enrichment (Step 5): mine confusable pairs, compounds, collocations, register tags
    enrich: bool = True  # Master toggle (--enrich / --no-enrich)
    enrich_confusables: bool = True
    enrich_compounds: bool = True
    enrich_collocations: bool = True
    enrich_register: bool = True

    # Batch processor parameters
    viterbi_max_chunk_size: int = 100  # Max chars per Viterbi chunk (O(n^2) trade-off)
    streaming_chunk_size: int = 1000  # Sentences per streaming batch chunk
    max_memory_batch_size: int = 10000  # Max sentences to hold in memory before yielding
    pre_segmented_detection_ratio: float = 0.7  # Ratio of valid Myanmar tokens for pre-seg

    # Build-mode SQLite PRAGMA tuning
    build_pragma_cache_size: int = -1048576  # 1 GB page cache during build (negative = KiB)
    build_pragma_mmap_size: int = 2147483648  # 2 GB memory-mapped I/O during build

    # Quality gate thresholds (warn if corpus statistics fall outside these bounds)
    quality_max_empty_pct: int = 20  # Max percentage of empty sentences before warning
    quality_min_avg_words: int = 3  # Min average words per sentence before warning
    quality_max_avg_words: int = 30  # Max average words per sentence before warning

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        from ..core.config import POSTaggerConfig
        from ..core.exceptions import ConfigurationError

        if self.batch_size < 100:
            raise ConfigurationError(f"batch_size must be >= 100, got {self.batch_size}")

        if self.num_shards < 1:
            raise ConfigurationError(f"num_shards must be >= 1, got {self.num_shards}")

        if self.num_workers is not None and self.num_workers < 1:
            raise ConfigurationError(f"num_workers must be >= 1 or None, got {self.num_workers}")

        if self.min_frequency < 1:
            raise ConfigurationError(f"min_frequency must be >= 1, got {self.min_frequency}")

        if self.min_syllable_frequency < 1:
            raise ConfigurationError(
                f"min_syllable_frequency must be >= 1, got {self.min_syllable_frequency}"
            )

        if self.min_bigram_count < 1:
            raise ConfigurationError(f"min_bigram_count must be >= 1, got {self.min_bigram_count}")

        if self.min_trigram_count < 1:
            raise ConfigurationError(
                f"min_trigram_count must be >= 1, got {self.min_trigram_count}"
            )

        if self.min_fourgram_count < 1:
            raise ConfigurationError(
                f"min_fourgram_count must be >= 1, got {self.min_fourgram_count}"
            )

        if self.min_fivegram_count < 1:
            raise ConfigurationError(
                f"min_fivegram_count must be >= 1, got {self.min_fivegram_count}"
            )

        if self.word_engine not in ("crf", "myword", "transformer"):
            raise ConfigurationError(
                f"word_engine must be 'crf', 'myword', or 'transformer', got {self.word_engine}"
            )

        if self.disk_space_check_mb < 0:
            raise ConfigurationError(
                f"disk_space_check_mb must be >= 0, got {self.disk_space_check_mb}"
            )

        if self.worker_timeout < 1:
            raise ConfigurationError(f"worker_timeout must be >= 1, got {self.worker_timeout}")

        # Batch processor parameters
        if self.viterbi_max_chunk_size < 10:
            raise ConfigurationError(
                f"viterbi_max_chunk_size must be >= 10, got {self.viterbi_max_chunk_size}"
            )

        if self.streaming_chunk_size < 1:
            raise ConfigurationError(
                f"streaming_chunk_size must be >= 1, got {self.streaming_chunk_size}"
            )

        if self.max_memory_batch_size < 1:
            raise ConfigurationError(
                f"max_memory_batch_size must be >= 1, got {self.max_memory_batch_size}"
            )

        if not (0.0 < self.pre_segmented_detection_ratio <= 1.0):
            raise ConfigurationError(
                f"pre_segmented_detection_ratio must be in (0.0, 1.0], "
                f"got {self.pre_segmented_detection_ratio}"
            )

        # Build pragma: mmap_size must be non-negative
        if self.build_pragma_mmap_size < 0:
            raise ConfigurationError(
                f"build_pragma_mmap_size must be >= 0, got {self.build_pragma_mmap_size}"
            )

        # Quality gate thresholds
        if self.quality_max_empty_pct < 0 or self.quality_max_empty_pct > 100:
            raise ConfigurationError(
                f"quality_max_empty_pct must be in [0, 100], got {self.quality_max_empty_pct}"
            )

        if self.quality_min_avg_words < 1:
            raise ConfigurationError(
                f"quality_min_avg_words must be >= 1, got {self.quality_min_avg_words}"
            )

        if self.quality_max_avg_words < self.quality_min_avg_words:
            raise ConfigurationError(
                f"quality_max_avg_words ({self.quality_max_avg_words}) must be >= "
                f"quality_min_avg_words ({self.quality_min_avg_words})"
            )

        # Validate pos_tagger is correct type
        if self.pos_tagger is not None and not isinstance(self.pos_tagger, POSTaggerConfig):
            raise ConfigurationError(
                f"pos_tagger must be POSTaggerConfig or None, got {type(self.pos_tagger).__name__}"
            )

    def with_overrides(self, **kwargs) -> "PipelineConfig":
        """
        Create a new config with specified overrides.

        This is useful for making small adjustments to a base configuration
        without modifying the original.

        Args:
            **kwargs: Parameters to override.

        Returns:
            New PipelineConfig with overrides applied.

        Example:
            >>> base_config = PipelineConfig(batch_size=10000)
            >>> large_batch_config = base_config.with_overrides(batch_size=50000)
        """
        import dataclasses

        current_values = dataclasses.asdict(self)
        current_values.update(kwargs)
        return PipelineConfig(**current_values)


@dataclass
class IngesterConfig:
    """
    Configuration specifically for the corpus ingestion step.

    Attributes:
        batch_size: Records per batch for Arrow writing (default: 10000).
        encoding: File encoding for text files (default: "utf-8").
        skip_empty_lines: Skip empty lines during ingestion (default: True).
        normalize_unicode: Apply Unicode normalization (default: True).
    """

    batch_size: int = DEFAULT_BATCH_SIZE
    encoding: str = "utf-8"
    skip_empty_lines: bool = True
    normalize_unicode: bool = True


@dataclass
class SegmenterConfig:
    """
    Configuration for the corpus segmentation step.

    Attributes:
        batch_size: Records per batch for processing (default: 10000).
        word_engine: Word segmentation engine (default: "myword").
        num_workers: Number of parallel workers (default: None for auto).
        enable_pos_tagging: Enable POS tagging during segmentation (default: True).
        chunk_size: Lines per chunk for parallel processing (default: 50000).
    """

    batch_size: int = DEFAULT_BATCH_SIZE
    word_engine: str = "myword"
    num_workers: int | None = None
    enable_pos_tagging: bool = True
    chunk_size: int = 50000


@dataclass
class FrequencyBuilderConfig:
    """
    Configuration for the frequency building step.

    Attributes:
        min_syllable_frequency: Minimum frequency for syllables (default: 1).
        min_word_frequency: Minimum frequency for words (default: 1).
        min_bigram_count: Minimum count for bigrams (default: 1).
        min_trigram_count: Minimum count for trigrams (default: 1).
        smoothing_factor: Laplace smoothing factor (default: 0.0).
    """

    min_syllable_frequency: int = 1
    min_word_frequency: int = 1
    min_bigram_count: int = 1
    min_trigram_count: int = 1
    smoothing_factor: float = 0.0


@dataclass
class PackagerConfig:
    """
    Configuration for the database packaging step.

    Attributes:
        batch_size: Records per batch for database insertion (default: 10000).
        create_indexes: Create database indexes after loading (default: True).
        vacuum_after_build: Run VACUUM after build to optimize DB (default: True).
        enable_fts: Enable full-text search (default: False).
    """

    batch_size: int = DEFAULT_BATCH_SIZE
    create_indexes: bool = True
    vacuum_after_build: bool = True
    enable_fts: bool = False


# ---------------------------------------------------------------------------
# PipelineRuntimeFlags — replaces per-module global setters
# ---------------------------------------------------------------------------


class PipelineRuntimeFlags:
    """Shared mutable flags for all data-pipeline modules.

    This replaces the scattered ``_allow_extended_myanmar`` (and similar)
    module-level globals + ``set_*()`` setters that were thread-unsafe and
    hard to test.

    Usage from any pipeline module::

        from .config import runtime_flags
        if runtime_flags.allow_extended_myanmar:
            ...

    Pipeline.py sets the values once before spawning workers::

        runtime_flags.apply(pipeline_config)

    The object uses a lock so concurrent threads that read/write flags do
    not see torn values.  For fork-based multiprocessing the parent sets
    the flags before ``fork()`` and children inherit them via COW — the
    lock is not needed post-fork (each process has its own copy).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # Corpus cleaning
        self._allow_extended_myanmar: bool = False
        self._remove_segmentation_markers: bool = True
        self._deduplicate_lines: bool = True
        # Segmentation
        self._auto_detect_pre_segmented: bool = True

    # -- properties with lock-protected access --

    @property
    def allow_extended_myanmar(self) -> bool:
        return self._allow_extended_myanmar

    @allow_extended_myanmar.setter
    def allow_extended_myanmar(self, value: bool) -> None:
        with self._lock:
            self._allow_extended_myanmar = value

    @property
    def remove_segmentation_markers(self) -> bool:
        return self._remove_segmentation_markers

    @remove_segmentation_markers.setter
    def remove_segmentation_markers(self, value: bool) -> None:
        with self._lock:
            self._remove_segmentation_markers = value

    @property
    def deduplicate_lines(self) -> bool:
        return self._deduplicate_lines

    @deduplicate_lines.setter
    def deduplicate_lines(self, value: bool) -> None:
        with self._lock:
            self._deduplicate_lines = value

    @property
    def auto_detect_pre_segmented(self) -> bool:
        return self._auto_detect_pre_segmented

    @auto_detect_pre_segmented.setter
    def auto_detect_pre_segmented(self, value: bool) -> None:
        with self._lock:
            self._auto_detect_pre_segmented = value

    def apply(self, config: PipelineConfig) -> None:
        """Populate all flags from a ``PipelineConfig`` in one shot."""
        with self._lock:
            self._allow_extended_myanmar = config.allow_extended_myanmar
            self._remove_segmentation_markers = config.remove_segmentation_markers
            self._deduplicate_lines = config.deduplicate_lines
            # auto_detect_pre_segmented is not on PipelineConfig today;
            # keep its current value unless explicitly set elsewhere.

    def reset(self) -> None:
        """Reset all flags to defaults (useful in tests)."""
        with self._lock:
            self._allow_extended_myanmar = False
            self._remove_segmentation_markers = True
            self._deduplicate_lines = True
            self._auto_detect_pre_segmented = True


# Module-level singleton — every pipeline module imports this.
runtime_flags = PipelineRuntimeFlags()
