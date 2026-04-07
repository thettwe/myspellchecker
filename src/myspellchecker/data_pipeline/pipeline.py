"""
Data pipeline orchestrator.

This module provides the high-level API for building the spell checker database
from raw corpus text.

Resume Capability:
    The pipeline supports resuming from interrupted builds by checking
    file modification times. Each step is skipped if its outputs are
    newer than its inputs.
"""

from __future__ import annotations

import errno
import shutil
import signal
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from myspellchecker.core.config import POSTaggerConfig

    from .config import PipelineConfig

import pyarrow as pa  # type: ignore

from ..core.constants import (
    DEFAULT_DB_NAME,
    QUALITY_MAX_AVG_WORDS,
    QUALITY_MAX_EMPTY_PCT,
    QUALITY_MIN_AVG_WORDS,
)
from ..core.exceptions import InsufficientStorageError, PackagingError, PipelineError
from ..utils.console import PipelineConsole, create_build_complete_panel
from ..utils.io_utils import check_disk_space
from ..utils.logging_utils import get_logger
from .config import runtime_flags as _flags
from .database_packager import DatabasePackager
from .frequency_builder import FrequencyBuilder
from .ingester import CorpusIngester, IngestionError
from .reporter import PipelineReporter
from .segmenter import CorpusSegmenter

__all__ = [
    "Pipeline",
    "run_pipeline",
]


class Pipeline:
    """
    Orchestrates the data pipeline steps with resume capability.

    The pipeline can be configured using PipelineConfig for fine-grained control
    over processing parameters, or using simple constructor arguments for
    backward compatibility.

    Example:
        >>> from myspellchecker.data_pipeline import Pipeline, PipelineConfig
        >>>
        >>> # Simple usage with defaults
        >>> pipeline = Pipeline()
        >>> pipeline.build_database(["corpus.txt"], "output.db")
        >>>
        >>> # Using PipelineConfig for custom settings
        >>> config = PipelineConfig(
        ...     batch_size=50000,
        ...     num_workers=8,
        ...     min_frequency=100,  # Higher threshold for cleaner data
        ... )
        >>> pipeline = Pipeline(config=config)
    """

    # Expected frequency output files from Step 3
    FREQUENCY_FILES = [
        "syllable_frequencies.tsv",
        "word_frequencies.tsv",
        "bigram_probabilities.tsv",
        "trigram_probabilities.tsv",
        "pos_unigram_probabilities.tsv",
        "pos_bigram_probabilities.tsv",
        "pos_trigram_probabilities.tsv",
    ]

    # Subdirectory names for work directory organization
    INTERMEDIATE_DIR = "intermediate"
    FREQUENCIES_DIR = "frequencies"

    def __init__(
        self,
        work_dir: str | Path | None = None,
        keep_intermediate: bool = False,
        config: PipelineConfig | None = None,
    ):
        """
        Initialize the pipeline.

        Args:
            work_dir: Directory for intermediate files (default: temp dir).
            keep_intermediate: Whether to keep intermediate files after completion.
            config: PipelineConfig object for fine-grained control.
                If provided, work_dir and keep_intermediate arguments are ignored.

        Note:
            If config is provided, it takes precedence over work_dir and
            keep_intermediate arguments.
        """
        from .config import PipelineConfig

        self.logger = get_logger(__name__)
        self._temp_dir_context: tempfile.TemporaryDirectory | None = None
        self._owns_temp_dir = False
        self._shutdown_requested = False
        self._original_sigint: Any = None
        self._original_sigterm: Any = None

        # Use config if provided, otherwise create from arguments
        if config is not None:
            self.config = config
            self.keep_intermediate = config.keep_intermediate
            work_dir = config.work_dir
        else:
            self.config = PipelineConfig(
                keep_intermediate=keep_intermediate,
                work_dir=work_dir,
            )
            self.keep_intermediate = keep_intermediate

        # Use console from config or create default (dependency injection)
        self.console = self.config.console or PipelineConsole()

        # Use reporter from config or create default (dependency injection)
        self.reporter = self.config.reporter or PipelineReporter(self.console)

        if work_dir:
            self.work_dir = Path(work_dir)
            self.work_dir.mkdir(parents=True, exist_ok=True)
        else:
            # Use TemporaryDirectory context manager for proper cleanup
            # This ensures cleanup even if the process crashes
            self._temp_dir_context = tempfile.TemporaryDirectory(prefix="myspell_pipeline_")
            self._owns_temp_dir = True
            self.work_dir = Path(self._temp_dir_context.name)

    def cleanup(self) -> None:
        """
        Explicitly cleanup temporary resources.

        Call this method when done with the pipeline to ensure temporary
        directories are cleaned up. This is called automatically in
        build_database() but can be called manually if needed.
        """
        if self._temp_dir_context is not None:
            try:
                self._temp_dir_context.cleanup()
            except OSError as e:
                self.logger.debug(f"Error during temp dir cleanup: {e}")
            finally:
                self._temp_dir_context = None

    def _cleanup_intermediates(self) -> None:
        """Remove known intermediate directories when keep_intermediate=False."""
        if self.keep_intermediate:
            return
        intermediate_dir = self.work_dir / self.INTERMEDIATE_DIR
        for subdir in ["raw_shards", "chunks", ".duckdb_temp"]:
            path = intermediate_dir / subdir
            if path.exists():
                try:
                    shutil.rmtree(path)
                except OSError as e:
                    self.logger.warning("Failed to clean up %s: %s", path, e)
                else:
                    self.logger.debug("Cleaned up %s", path)

    def _setup_signal_handlers(self) -> None:
        """Install signal handlers for graceful shutdown on Ctrl+C / SIGTERM."""
        self._original_sigint = signal.getsignal(signal.SIGINT)
        self._original_sigterm = signal.getsignal(signal.SIGTERM)

        def _handler(signum: int, frame: object) -> None:
            self._shutdown_requested = True
            sig_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
            self.logger.warning(
                "%s received — finishing current step. Resume with --resume.", sig_name
            )

        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)

    def _restore_signal_handlers(self) -> None:
        """Restore original signal handlers."""
        if self._original_sigint is not None:
            signal.signal(signal.SIGINT, self._original_sigint)
        if self._original_sigterm is not None:
            signal.signal(signal.SIGTERM, self._original_sigterm)

    def _check_shutdown(self, completed_step: int) -> bool:
        """Check if shutdown was requested and log if so. Returns True if shutdown."""
        if self._shutdown_requested:
            self.reporter.report_info(
                f"Build interrupted after Step {completed_step}/5. "
                "Intermediate files preserved for resume."
            )
            return True
        return False

    def _wire_extended_myanmar_scope(self) -> None:
        """
        Wire pipeline config flags to the shared ``runtime_flags`` singleton
        and sync per-module globals for fork()-based COW worker inheritance.

        This ensures consistent scope handling across ingestion, segmentation,
        frequency building, and database packaging.
        """
        # 1. Populate the shared runtime flags from PipelineConfig
        _flags.apply(self.config)

        if self.config.allow_extended_myanmar:
            self.logger.debug("Extended Myanmar scope enabled for pipeline")

        if not self.config.remove_segmentation_markers:
            self.logger.debug("Segmentation marker removal disabled")

        if not self.config.deduplicate_lines:
            self.logger.debug("Line deduplication disabled")

        # 2. Sync with Cython ingester module if available (fork()-ed workers
        #    inherit the shared _flags object via COW, but the Cython module
        #    has its own copies that need explicit syncing).
        allow_ext = self.config.allow_extended_myanmar

        # Also sync with Cython ingester module if available
        try:
            from myspellchecker.data_pipeline.ingester_c import (
                set_allow_extended_myanmar_c,
                set_remove_segmentation_markers_c,
            )

            set_allow_extended_myanmar_c(allow_ext)
            set_remove_segmentation_markers_c(self.config.remove_segmentation_markers)
        except ImportError:
            pass

        # _segmenter_config.py module-level globals
        from . import _segmenter_config as _seg_cfg_mod

        _seg_cfg_mod._allow_extended_myanmar = allow_ext
        _seg_cfg_mod._auto_detect_pre_segmented = _flags.auto_detect_pre_segmented

        # Propagate to Cython batch_processor / repair_c (unchanged .pyx modules)
        if _seg_cfg_mod._CAPABILITIES.has_batch_processor:
            from myspellchecker.data_pipeline.batch_processor import (
                set_allow_extended_myanmar as bp_set_ext,
            )

            bp_set_ext(allow_ext)

        if _seg_cfg_mod._CAPABILITIES.has_cython_repair:
            from myspellchecker.data_pipeline.repair_c import (
                set_allow_extended_myanmar as rc_set_ext,
            )

            rc_set_ext(allow_ext)

    def __del__(self) -> None:
        """Ensure cleanup on garbage collection (fallback)."""
        if hasattr(self, "_temp_dir_context") and self._temp_dir_context is not None:
            try:
                self._temp_dir_context.cleanup()
            except Exception:
                pass  # Cannot reliably log from __del__

    def _get_newest_mtime(self, paths: list[Path]) -> float:
        """Get the newest modification time from a list of files."""
        if not paths:
            return 0.0
        mtimes = [p.stat().st_mtime for p in paths if p.exists()]
        return max(mtimes) if mtimes else 0.0

    def _get_oldest_mtime(self, paths: list[Path]) -> float:
        """Get the oldest modification time from a list of files."""
        if not paths:
            return float("inf")
        mtimes = [p.stat().st_mtime for p in paths if p.exists()]
        return min(mtimes) if mtimes else float("inf")

    def _check_step1_complete(
        self, input_files: list[Path], raw_shards_dir: Path
    ) -> tuple[bool, list[Path]]:
        """
        Check if Step 1 (Ingestion) can be skipped.

        Returns:
            Tuple of (can_skip, shard_paths)
        """
        if not raw_shards_dir.exists():
            return False, []

        shard_paths = sorted(raw_shards_dir.glob("raw_shard_*.arrow"))
        if not shard_paths:
            return False, []

        # Check all shards have content
        valid_shards = [p for p in shard_paths if p.stat().st_size > 0]
        if not valid_shards:
            return False, []

        # Get newest input file mtime
        input_mtime = self._get_newest_mtime(input_files)

        # Get oldest shard mtime (all shards must be newer than inputs)
        shard_mtime = self._get_oldest_mtime(valid_shards)

        can_skip = shard_mtime > input_mtime
        return can_skip, valid_shards

    def _check_step2_complete(self, raw_shards_dir: Path) -> tuple[bool, Path | None]:
        """
        Check if Step 2 (Segmentation) can be skipped.

        Returns:
            Tuple of (can_skip, segmented_corpus_path)
        """
        segmented_path = self.work_dir / self.INTERMEDIATE_DIR / "segmented_corpus.arrow"
        sentinel_path = self.work_dir / self.INTERMEDIATE_DIR / "segmented_corpus.arrow.complete"
        if not segmented_path.exists() or segmented_path.stat().st_size == 0:
            return False, None

        # Validate sentinel file exists (written only after successful completion)
        if not sentinel_path.exists():
            self.logger.warning(
                "Segmented corpus file exists but completion sentinel is missing. "
                "File may be corrupted from an interrupted build. Re-running Step 2."
            )
            return False, None

        # Validate Arrow file is readable
        try:
            import pyarrow as pa

            with pa.OSFile(str(segmented_path), "r") as f:
                reader = pa.RecordBatchStreamReader(f)
                _ = reader.schema  # Validates file header is intact
        except (OSError, pa.ArrowInvalid, pa.ArrowSerializationError) as e:
            self.logger.warning(f"Segmented corpus file is corrupted: {e}. Re-running Step 2.")
            return False, None

        # Get newest shard mtime
        shard_paths = sorted(raw_shards_dir.glob("raw_shard_*.arrow"))
        if not shard_paths:
            # Shards were cleaned up but segmented output exists — Step 2 is done
            return True, segmented_path

        shard_mtime = self._get_newest_mtime(shard_paths)
        segmented_mtime = segmented_path.stat().st_mtime

        can_skip = segmented_mtime > shard_mtime
        return can_skip, segmented_path

    def _check_step3_complete(self) -> bool:
        """
        Check if Step 3 (Frequency Building) can be skipped.

        Returns:
            True if all frequency files exist and are newer than segmented_corpus.arrow
        """
        segmented_path = self.work_dir / self.INTERMEDIATE_DIR / "segmented_corpus.arrow"
        if not segmented_path.exists():
            return False

        segmented_mtime = segmented_path.stat().st_mtime

        # Check all frequency files exist and are newer than segmented corpus
        for filename in self.FREQUENCY_FILES:
            freq_path = self.work_dir / self.FREQUENCIES_DIR / filename
            if not freq_path.exists() or freq_path.stat().st_size == 0:
                return False
            if freq_path.stat().st_mtime < segmented_mtime:
                return False

        return True

    def _check_step4_complete(self, database_path: Path) -> bool:
        """
        Check if Step 4 (Packaging) can be skipped.

        Returns:
            True if database exists and is newer than all frequency files
        """
        if not database_path.exists() or database_path.stat().st_size == 0:
            return False

        db_mtime = database_path.stat().st_mtime

        # Check DB is newer than all frequency files
        for filename in self.FREQUENCY_FILES:
            freq_path = self.work_dir / self.FREQUENCIES_DIR / filename
            if freq_path.exists() and freq_path.stat().st_mtime > db_mtime:
                return False

        return True

    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable form."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes}m {secs:.0f}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"

    def _show_pipeline_header(
        self,
        min_frequency: int,
        incremental: bool,
        word_engine: str,
    ) -> None:
        """Display the pipeline header panel."""
        from rich.panel import Panel

        header_content = (
            f"[info]Min Frequency:[/] {min_frequency}\n"
            f"[info]Incremental Mode:[/] {incremental}\n"
            f"[info]Word Engine:[/] {word_engine}\n"
            f"[info]Work Directory:[/] {self.work_dir}"
        )
        self.reporter.print_newline()
        self.reporter.print_raw(
            Panel(
                header_content,
                title="\U0001f680 Starting Data Pipeline",
                title_align="left",
                border_style="info",
                padding=(0, 2),
            )
        )

    def _preflight_checks(
        self,
        database_path: Path,
        incremental: bool,
        frequencies_dir: Path,
    ) -> tuple[dict[str, tuple[float, int]], tuple | None, "DatabasePackager"]:
        """
        Run pre-flight checks and prepare for incremental updates.

        Returns:
            Tuple of (processed_files_map, existing_counts, packager)
        """
        # Check for disk space (configurable via config.disk_space_check_mb)
        if self.config.disk_space_check_mb > 0:
            try:
                check_disk_space(self.work_dir, required_mb=self.config.disk_space_check_mb)
                check_disk_space(
                    database_path.parent,
                    required_mb=self.config.disk_space_check_mb // 2,
                )
            except InsufficientStorageError as e:
                self.reporter.report_error(f"Pre-flight check failed: {e}")
                raise

        processed_files_map: dict[str, tuple[float, int]] = {}
        existing_counts = None
        packager = DatabasePackager(input_dir=frequencies_dir, database_path=database_path)

        if incremental and database_path.exists():
            self.reporter.report_info("Preparing for incremental update...")
            try:
                packager.connect(incremental=True)
                # Create schema if needed to ensure tables exist
                packager.create_schema()

                processed_files_map = packager.get_processed_files()
                existing_counts = packager.get_current_counts()
            except (sqlite3.Error, OSError) as e:
                self.reporter.report_warning(f"Could not prepare for incremental update: {e}")
                self.reporter.report_warning("Falling back to full build.")
                processed_files_map = {}
                existing_counts = None
            finally:
                packager.close()

        return processed_files_map, existing_counts, packager

    def _step1_ingest(
        self,
        input_files: list[Path],
        raw_shards_dir: Path,
        sample: bool,
        text_col: str,
        json_key: str,
        processed_files_map: dict[str, tuple[float, int]],
        incremental: bool,
        step_durations: dict[str, tuple[str, float, list[tuple[str, float]]]],
    ) -> tuple[list[Path], list[tuple[str, float, int]]]:
        """
        Step 1: Ingest corpus files into Arrow shards.

        Returns:
            Tuple of (shard_paths, new_files_meta)
        """
        import time

        step1_start = time.time()
        num_shards = self.config.num_shards

        # Check if Step 1 can be skipped (resume capability)
        step1_complete, existing_shards = self._check_step1_complete(input_files, raw_shards_dir)
        new_files_meta: list[tuple[str, float, int]] = []

        if step1_complete and not sample:
            self.reporter.report_step_skipped(
                1, 5, "Ingesting Corpus", f"Found {len(existing_shards)} existing shards"
            )
            step_durations["1. Ingestion"] = ("skipped", 0.0, [])
            return existing_shards, new_files_meta

        self.reporter.report_step_start(1, 5, "Ingesting Corpus")
        ingester = CorpusIngester(
            worker_timeout=self.config.worker_timeout,
            text_col=text_col,
            json_key=json_key,
            allow_partial_ingestion=self.config.allow_partial_ingestion,
        )

        if sample:
            shard_paths = self._ingest_sample(ingester, raw_shards_dir)
        else:
            if not input_files:
                raise PipelineError("No input files provided and sample mode is False.")

            shard_paths, new_files_meta = self._ingest_files(
                ingester,
                input_files,
                raw_shards_dir,
                num_shards,
                processed_files_map if incremental else None,
            )

        step1_duration = time.time() - step1_start
        self.reporter.report_step_complete(
            1, 5, "Ingestion complete", self._format_duration(step1_duration)
        )
        step_durations["1. Ingestion"] = ("complete", step1_duration, [])

        return shard_paths, new_files_meta

    def _ingest_sample(
        self,
        ingester: "CorpusIngester",
        raw_shards_dir: Path,
    ) -> list[Path]:
        """Ingest sample corpus for testing."""
        sample_lines = ingester.generate_sample_corpus()
        raw_shards_dir.mkdir(parents=True, exist_ok=True)
        sample_shard = raw_shards_dir / "raw_shard_000.arrow"

        from .ingester import _get_ingest_schema

        ingest_schema = _get_ingest_schema()
        data = [{"text": line, "source": "sample"} for line in sample_lines]
        batch = pa.RecordBatch.from_pylist(data, schema=ingest_schema)

        with pa.OSFile(str(sample_shard), "w") as f:
            with pa.RecordBatchStreamWriter(f, ingest_schema) as writer:
                writer.write_batch(batch)

        return [sample_shard]

    def _ingest_files(
        self,
        ingester: "CorpusIngester",
        path_list: list[Path],
        raw_shards_dir: Path,
        num_shards: int,
        processed_files_map: dict[str, tuple[float, int]] | None,
    ) -> tuple[list[Path], list[tuple[str, float, int]]]:
        """Ingest corpus files with parallel or sequential processing.

        Raises:
            IngestionError: If any input files are missing or fail to process.
        """
        from .ingester import MIN_PARALLEL_FILE_SIZE

        # Validate all input files exist before processing
        missing_files = [str(f) for f in path_list if not f.exists()]
        if missing_files:
            raise IngestionError(
                f"Cannot start ingestion: {len(missing_files)} input file(s) not found",
                missing_files=missing_files,
            )

        raw_shards_dir.mkdir(parents=True, exist_ok=True)

        # Auto-detect parallel mode based on total file size (all files exist at this point)
        total_size = sum(f.stat().st_size for f in path_list)
        use_parallel = total_size >= MIN_PARALLEL_FILE_SIZE

        if use_parallel:
            self.reporter.report_info(
                f"Using parallel ingestion ({total_size / (1024**3):.2f} GB corpus)"
            )
            return ingester._process_sharded_parallel(
                path_list,
                raw_shards_dir,
                num_shards,
                num_workers=self.config.num_workers or 8,
                processed_files_map=processed_files_map,
            )
        else:
            return ingester._process_sharded(
                path_list,
                raw_shards_dir,
                num_shards,
                processed_files_map=processed_files_map,
            )

    def _step2_segment(
        self,
        raw_shards_dir: Path,
        intermediate_dir: Path,
        word_engine: str,
        step_durations: dict[str, tuple[str, float, list[tuple[str, float]]]],
        seg_model: str | None = None,
        seg_device: int = -1,
    ) -> Path:
        """
        Step 2: Segment corpus into syllables and words.

        Returns:
            Path to segmented corpus Arrow file.
        """
        import time

        step2_start = time.time()
        step2_complete, segmented_path = self._check_step2_complete(raw_shards_dir)

        if step2_complete:
            self.reporter.report_step_skipped(
                2, 5, "Segmenting Corpus", "Found existing segmented_corpus.arrow"
            )
            step_durations["2. Segmentation"] = ("skipped", 0.0, [])
            return segmented_path  # type: ignore[return-value]

        self.reporter.report_step_start(2, 5, "Segmenting Corpus")
        segmenter = CorpusSegmenter(
            output_dir=intermediate_dir,
            word_engine=word_engine,
            seg_model=seg_model,
            seg_device=seg_device,
            worker_timeout=self.config.worker_timeout,
        )

        sentences_arrow_path = segmenter.segment_corpus(
            raw_shards_dir, num_workers=self.config.num_workers
        )

        # Clean up raw shards — no longer needed after segmentation
        if raw_shards_dir.exists():
            try:
                shutil.rmtree(raw_shards_dir)
            except OSError as e:
                self.logger.warning("Failed to clean up raw shards %s: %s", raw_shards_dir, e)
            else:
                self.logger.debug("Cleaned up raw shards: %s", raw_shards_dir)

        step2_duration = time.time() - step2_start
        self.reporter.report_step_complete(
            2, 5, "Segmentation complete", self._format_duration(step2_duration)
        )
        step_durations["2. Segmentation"] = ("complete", step2_duration, [])

        # Write sentinel file to mark successful Step 2 completion
        sentinel_path = intermediate_dir / "segmented_corpus.arrow.complete"
        sentinel_path.touch()
        self.logger.debug(f"Wrote completion sentinel: {sentinel_path}")

        # Post-segmentation quality report
        self._step2_quality_report(sentences_arrow_path)

        return sentences_arrow_path

    def _step2_quality_report(self, segmented_path: Path) -> None:
        """Log quality metrics from segmented corpus for early issue detection."""
        try:
            total_sentences = 0
            empty_sentences = 0
            total_words = 0

            with pa.OSFile(str(segmented_path), "r") as f:
                reader = pa.RecordBatchStreamReader(f)
                for batch in reader:
                    words_col = batch.column("words")
                    for word_list in words_col.to_pylist():
                        total_sentences += 1
                        if not word_list:
                            empty_sentences += 1
                        else:
                            total_words += len(word_list)

            avg_words = total_words / max(total_sentences, 1)
            empty_pct = (empty_sentences / max(total_sentences, 1)) * 100

            self.reporter.report_info("Segmentation Quality Report:")
            self.reporter.report_info(f"  Total sentences: {total_sentences:,}")
            self.reporter.report_info(f"  Empty sentences: {empty_sentences:,} ({empty_pct:.1f}%)")
            self.reporter.report_info(f"  Average words/sentence: {avg_words:.1f}")
            self.reporter.report_info(f"  Total words: {total_words:,}")

            if empty_pct > QUALITY_MAX_EMPTY_PCT:
                self.logger.warning(
                    "High empty sentence rate (%.1f%%) — check input data quality", empty_pct
                )
            if avg_words < QUALITY_MIN_AVG_WORDS or avg_words > QUALITY_MAX_AVG_WORDS:
                self.logger.warning(
                    "Unusual avg words/sentence (%.1f) — check segmenter", avg_words
                )
        except (ValueError, ZeroDivisionError, AttributeError) as e:
            self.logger.warning(f"Could not generate quality report: {e}")

    def _init_pos_tagger(
        self,
        pos_tagger_config: POSTaggerConfig | None,
    ) -> object | None:
        """
        Step 2.5: Initialize POS Tagger (pluggable system).

        Returns:
            Initialized POS tagger or None.
        """
        # Priority: CLI pos_tagger_config > self.config.pos_tagger
        active_pos_config = pos_tagger_config or (
            self.config.pos_tagger if hasattr(self.config, "pos_tagger") else None
        )

        if not active_pos_config:
            return None

        from myspellchecker.algorithms.pos_tagger_factory import POSTaggerFactory

        try:
            self.reporter.report_info(f"Initializing {active_pos_config.tagger_type} POS tagger")

            # Build kwargs based on tagger type to avoid passing irrelevant parameters
            tagger_type = active_pos_config.tagger_type
            kwargs = {"tagger_type": tagger_type}

            if tagger_type == "rule_based":
                # Rule-based only accepts: pos_map, use_morphology_fallback, cache_size, unknown_tag
                kwargs["use_morphology_fallback"] = active_pos_config.use_morphology_fallback
                kwargs["cache_size"] = active_pos_config.cache_size
                kwargs["unknown_tag"] = active_pos_config.unknown_tag
            elif tagger_type == "transformer":
                # Transformer accepts: model_name, device, batch_size, max_length, cache_dir,
                # use_fp16, use_torch_compile
                kwargs["model_name"] = active_pos_config.model_name
                kwargs["device"] = active_pos_config.device
                kwargs["batch_size"] = active_pos_config.batch_size
                kwargs["cache_dir"] = active_pos_config.cache_dir
                kwargs["use_fp16"] = active_pos_config.use_fp16
                kwargs["use_torch_compile"] = active_pos_config.use_torch_compile
            elif tagger_type == "viterbi":
                # Load precomputed HMM params for bootstrap (from myPOS corpus)
                hmm_provider, hmm_probs = self._load_hmm_bootstrap_data(active_pos_config)
                kwargs["provider"] = hmm_provider
                kwargs["pos_unigram_probs"] = hmm_probs["unigram"]
                kwargs["pos_bigram_probs"] = hmm_probs["bigram"]
                kwargs["pos_trigram_probs"] = hmm_probs["trigram"]
                # Disable morphological fallback during build: it produces granular
                # P_* particle subtags (P_REL, P_NOM, etc.) that pollute the HMM
                # tag set. OOV words get UNK, which is cleaner for transition tables.
                kwargs["use_morphology_fallback"] = False
                kwargs["beam_width"] = active_pos_config.beam_width
                kwargs["emission_weight"] = active_pos_config.emission_weight
                kwargs["min_prob"] = active_pos_config.min_prob
                kwargs["unknown_tag"] = active_pos_config.unknown_tag

            pos_tagger = POSTaggerFactory.create(**kwargs)
            self.reporter.report_success(f"Initialized {active_pos_config.tagger_type} POS tagger")
            return pos_tagger
        except (RuntimeError, ValueError, ImportError, OSError, TypeError) as e:
            self.reporter.report_warning(f"Failed to initialize POS tagger: {e}")
            self.reporter.report_info("Falling back to rule-based tagger")
            from myspellchecker.algorithms.pos_tagger_rule import RuleBasedPOSTagger

            return RuleBasedPOSTagger()

    def _load_hmm_bootstrap_data(self, config: "POSTaggerConfig") -> tuple:
        """Load precomputed HMM params from myPOS JSON for Viterbi bootstrap.

        Returns:
            Tuple of (MemoryProvider, probs_dict) where probs_dict has
            keys 'unigram', 'bigram', 'trigram'.
        """
        import json

        from myspellchecker.providers.memory import MemoryProvider

        # Determine JSON path: config override or bundled default
        if config.hmm_params_path:
            json_path = Path(config.hmm_params_path)
        else:
            json_path = Path(__file__).parent.parent / "data" / "mypos_hmm_params.json"

        if not json_path.exists():
            raise FileNotFoundError(
                f"HMM params not found at {json_path}. "
                "Run: python scripts/build_hmm_from_mypos.py to generate."
            )

        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        # Create MemoryProvider with word→POS mappings
        provider = MemoryProvider(word_pos=data["word_pos"])

        # Convert JSON arrays back to tuple-keyed dicts
        bigram_probs = {(item[0], item[1]): item[2] for item in data["pos_bigram_probs"]}
        trigram_probs = {(item[0], item[1], item[2]): item[3] for item in data["pos_trigram_probs"]}

        probs = {
            "unigram": data["pos_unigram_probs"],
            "bigram": bigram_probs,
            "trigram": trigram_probs,
        }

        self.reporter.report_info(
            f"Loaded HMM bootstrap: {len(data['word_pos'])} words, "
            f"{len(data['pos_unigram_probs'])} tags from {json_path.name}"
        )
        return provider, probs

    def _step3_build_frequencies(
        self,
        intermediate_dir: Path,
        frequencies_dir: Path,
        incremental: bool,
        existing_counts: tuple | None,
        pos_tagger: object | None,
        step_durations: dict[str, tuple[str, float, list[tuple[str, float]]]],
        min_frequency: int = 5,
        curated_words: dict[str, str] | None = None,
    ) -> None:
        """Step 3: Build frequency tables from segmented corpus."""
        import time

        step3_complete = self._check_step3_complete()
        step3_start = time.time()
        sub_steps: list[tuple[str, float]] = []

        if step3_complete:
            self.reporter.report_step_skipped(
                3,
                5,
                "Building Frequencies",
                f"Found {len(self.FREQUENCY_FILES)} existing frequency files",
            )
            step_durations["3. Frequency Building"] = ("skipped", 0.0, [])
            return

        self.reporter.report_step_start(3, 5, "Building Frequencies")
        # Get POS checkpoint/batch config from pipeline config
        pos_ckpt_interval = 250_000
        pos_batch_buf = 2000
        if hasattr(self.config, "pos_tagger") and self.config.pos_tagger:
            pos_ckpt_interval = self.config.pos_tagger.pos_checkpoint_interval
            pos_batch_buf = self.config.pos_tagger.pos_batch_buffer_size

        builder = FrequencyBuilder(
            input_dir=intermediate_dir,
            output_dir=frequencies_dir,
            word_engine="myword",  # Accurate syllable counting for frequency stats
            incremental=incremental,
            pos_tagger=pos_tagger,  # Pass POS tagger instance
            min_word_frequency=min_frequency,  # Apply user-specified threshold
            min_syllable_frequency=self.config.min_syllable_frequency,
            min_bigram_frequency=self.config.min_bigram_count,
            min_trigram_frequency=self.config.min_trigram_count,
            pos_checkpoint_interval=pos_ckpt_interval,
            pos_batch_buffer_size=pos_batch_buf,
        )

        # Hydrate with existing counts if incremental
        if existing_counts:
            builder.hydrate(*existing_counts)

        # Unified data loading from Arrow (sub-step timings recorded by builder)
        builder.load_data(filename="segmented_corpus.arrow")

        # Pull fine-grained sub-step timings from builder
        # (e.g., Arrow loading, DuckDB counting, Post-processing, POS tagging)
        sub_steps.extend(builder.sub_step_timings)

        # Sub-step: Frequency filtering
        t0 = time.time()
        builder.filter_by_frequency(curated_words=curated_words)
        sub_steps.append(("Frequency filtering", time.time() - t0))

        # Sub-step: Calculate & save frequencies
        t0 = time.time()
        builder.calculate_bigram_probabilities()
        builder.calculate_trigram_probabilities()
        builder.save_syllable_frequencies()
        builder.save_word_frequencies()
        builder.save_bigram_probabilities()
        builder.save_trigram_probabilities()

        pos_unigram_probs = builder.calculate_pos_unigram_probabilities()
        pos_bigram_probs = builder.calculate_pos_bigram_probabilities()
        pos_trigram_probs = builder.calculate_pos_trigram_probabilities()
        builder.save_pos_unigram_probabilities(pos_unigram_probs)
        builder.save_pos_bigram_probabilities(pos_bigram_probs)
        builder.save_pos_trigram_probabilities(pos_trigram_probs)

        # Cleanup DuckDB resources (connection + temp files) after all saves
        builder.cleanup_duckdb()

        sub_steps.append(("Save all frequencies", time.time() - t0))

        step3_duration = time.time() - step3_start
        self.reporter.report_step_complete(
            3, 5, "Frequency building complete", self._format_duration(step3_duration)
        )
        step_durations["3. Frequency Building"] = ("complete", step3_duration, sub_steps)

    def _step4_package(
        self,
        packager: "DatabasePackager",
        database_path: Path,
        new_files_meta: list[tuple[str, float, int]],
        incremental: bool,
        step_durations: dict[str, tuple[str, float, list[tuple[str, float]]]],
        curated_words: dict[str, str] | None = None,
    ) -> None:
        """Step 4: Package frequency data into SQLite database."""
        import time

        # Check if all previous steps were skipped (no work done)
        all_previous_skipped = all(
            status == "skipped"
            for step, (status, *_rest) in step_durations.items()
            if step.startswith(("1.", "2.", "3."))
        )

        # Skip step 4 if:
        # 1. DB is up-to-date with frequency files AND no new files to track, OR
        # 2. All previous steps were skipped AND no new files to track AND DB exists
        step4_complete = (self._check_step4_complete(database_path) and not new_files_meta) or (
            all_previous_skipped and not new_files_meta and database_path.exists()
        )
        step4_start = time.time()
        sub_steps: list[tuple[str, float]] = []

        if step4_complete:
            self.reporter.report_step_skipped(
                4, 5, "Packaging Database", f"Database is up-to-date: {database_path}"
            )
            step_durations["4. Database Packaging"] = ("skipped", 0.0, [])
            return

        self.reporter.report_step_start(4, 5, "Packaging Database")
        packager.connect(incremental=incremental)

        try:
            # Begin transaction for atomic database updates
            packager.begin_transaction()

            # Defer indexes for full builds (bulk load is faster without indexes).
            # Incremental builds need indexes for ON CONFLICT lookups.
            defer_indexes = not incremental
            packager.create_schema(defer_indexes=defer_indexes)

            # Sub-step: Load data into database
            t0 = time.time()
            packager.load_syllables()

            # Load curated words FIRST (before corpus words)
            # This ensures all curated vocabulary is in the database
            # Corpus words will update frequency via ON CONFLICT
            if curated_words:
                packager.load_curated_words(curated_words)

            packager.load_words(curated_words=curated_words)
            sub_steps.append(("Load syllables & words", time.time() - t0))

            t0 = time.time()
            packager.load_bigrams()
            packager.load_trigrams()
            packager.load_pos_unigrams()
            packager.load_pos_bigrams()
            packager.load_pos_trigrams()
            sub_steps.append(("Load n-grams & POS", time.time() - t0))

            # Create indexes after bulk data load (deferred from create_schema)
            if defer_indexes:
                t0 = time.time()
                packager.create_indexes()
                sub_steps.append(("Create indexes", time.time() - t0))

            # Update tracking for successfully processed files
            if new_files_meta:
                msg = f"Updating tracking for {len(new_files_meta)} files..."
                self.reporter.report_progress(msg)
                for path, mtime, size in new_files_meta:
                    packager.update_processed_file(path, mtime, size)

            # Commit transaction on success
            packager.commit_transaction()

            t0 = time.time()
            packager.optimize_database()
            packager.verify_database()
            sub_steps.append(("Optimize & verify", time.time() - t0))

            # Apply rule-based POS inference to untagged words
            t0 = time.time()
            self.reporter.report_progress("Applying POS inference to untagged words...")
            inference_stats = packager.apply_inferred_pos(
                min_frequency=0,  # Infer for all words
                min_confidence=0.0,  # Accept all confidence levels
            )
            if inference_stats["inferred"] > 0:
                self.reporter.report_success(
                    f"Inferred POS for {inference_stats['inferred']:,} words "
                    f"({inference_stats['ambiguous']:,} ambiguous)"
                )
            sub_steps.append(("POS inference", time.time() - t0))

            packager.print_stats()

            step4_duration = time.time() - step4_start
            self.reporter.report_step_complete(
                4, 5, "Database packaging complete", self._format_duration(step4_duration)
            )
            step_durations["4. Database Packaging"] = ("complete", step4_duration, sub_steps)

        except (sqlite3.Error, OSError, ValueError, PackagingError) as e:
            # Rollback transaction on any database/IO/data failure
            self.reporter.report_error(f"Database packaging failed: {e}")
            if packager._in_transaction:
                packager.rollback_transaction()
            raise

        finally:
            packager.close()

    def _step5_enrich(
        self,
        database_path: Path,
        step_durations: dict[str, tuple[str, float, list[tuple[str, float]]]],
    ) -> None:
        """Step 5: Enrich database with confusable pairs, compounds, collocations, register tags."""
        import time

        if not self.config.enrich:
            self.reporter.report_step_skipped(5, 5, "Enrichment", "Disabled (--no-enrich)")
            step_durations["5. Enrichment"] = ("skipped", 0.0, [])
            return

        if not database_path.exists():
            self.reporter.report_step_skipped(5, 5, "Enrichment", "No database to enrich")
            step_durations["5. Enrichment"] = ("skipped", 0.0, [])
            return

        self.reporter.report_step_start(5, 5, "Enriching Database")
        step5_start = time.time()
        sub_steps: list[tuple[str, float]] = []

        try:
            from .enrichment import EnrichmentConfig, run_enrichment

            enrich_config = EnrichmentConfig(
                enrich_confusables=self.config.enrich_confusables,
                enrich_compounds=self.config.enrich_compounds,
                enrich_collocations=self.config.enrich_collocations,
                enrich_register=self.config.enrich_register,
            )

            report = run_enrichment(str(database_path), enrich_config)

            if report.confusable_pairs > 0:
                sub_steps.append((f"Confusable pairs: {report.confusable_pairs:,}", 0.0))
            if report.compound_confusions > 0:
                sub_steps.append((f"Compound confusions: {report.compound_confusions:,}", 0.0))
            if report.collocations > 0:
                sub_steps.append((f"Collocations: {report.collocations:,}", 0.0))
            if report.register_tags > 0:
                sub_steps.append((f"Register tags: {report.register_tags:,}", 0.0))

            if report.errors:
                for err in report.errors:
                    self.reporter.report_warning(err)

            step5_duration = time.time() - step5_start
            self.reporter.report_step_complete(
                5, 5, "Enrichment complete", self._format_duration(step5_duration)
            )
            step_durations["5. Enrichment"] = ("complete", step5_duration, sub_steps)

        except Exception as e:
            self.reporter.report_warning(f"Enrichment failed (non-fatal): {e}")
            step5_duration = time.time() - step5_start
            step_durations["5. Enrichment"] = ("failed", step5_duration, [])

    def _show_pipeline_summary(
        self,
        step_durations: dict[str, tuple[str, float, list[tuple[str, float]]]],
        total_duration: float,
        database_path: Path,
    ) -> None:
        """Display the pipeline completion summary with sub-step breakdowns."""
        from rich.table import Table

        summary_table = Table(
            title="\U0001f4ca Pipeline Summary",
            title_style="header",
            show_header=True,
            header_style="bold",
            border_style="dim",
        )
        summary_table.add_column("Step", style="info")
        summary_table.add_column("Duration", justify="right", style="highlight")
        summary_table.add_column("Status", justify="center")

        for step_name, (status, duration, sub_steps) in step_durations.items():
            if status == "skipped":
                summary_table.add_row(step_name, "--", "[muted]SKIPPED[/]")
            else:
                summary_table.add_row(
                    step_name,
                    self._format_duration(duration),
                    "[success]\u2713[/]",
                )
                # Show sub-steps with indented style
                for i, (sub_name, sub_dur) in enumerate(sub_steps):
                    is_last = i == len(sub_steps) - 1
                    prefix = "└─" if is_last else "├─"
                    summary_table.add_row(
                        f"[dim]   {prefix} {sub_name}[/]",
                        f"[dim]{self._format_duration(sub_dur)}[/]",
                        "",
                    )

        # Add total row
        summary_table.add_section()
        summary_table.add_row(
            "[bold]Total Duration[/]",
            f"[bold]{self._format_duration(total_duration)}[/]",
            "",
        )

        self.reporter.print_newline()
        self.reporter.print_raw(summary_table)

        # Show completion panel
        self.reporter.print_newline()
        self.reporter.print_raw(create_build_complete_panel(str(database_path)))

    def build_database(
        self,
        input_files: list[str | Path],
        database_path: str | Path,
        sample: bool = False,
        text_col: str | None = None,
        json_key: str | None = None,
        min_frequency: int | None = None,
        pos_tagger_config: POSTaggerConfig | None = None,
        incremental: bool = False,
        word_engine: str | None = None,
        curated_words: dict[str, str] | None = None,
        seg_model: str | None = None,
        seg_device: int = -1,
    ) -> Path:
        """
        Run the full pipeline to build a database.

        Args:
            input_files: List of raw corpus text files
            database_path: Path to save the final SQLite database
            sample: If True, ignore inputs and generate a sample corpus
            text_col: Column name/index for CSV/TSV ingestion (defaults to PipelineConfig.text_col)
            json_key: Key name for JSON ingestion (defaults to PipelineConfig.json_key)
            min_frequency: Minimum frequency for words to be included.
                          If None, uses config.min_frequency (default: 50)
            pos_tagger_config: POS tagger configuration for dynamic tagging
            incremental: If True, perform incremental update
            word_engine: Word segmentation engine ('crf', 'myword', 'transformer').
                        Defaults to PipelineConfig.word_engine when omitted.
            curated_words: Optional dict of word→pos_tag for curated vocabulary
            seg_model: Custom model name/path for transformer word segmentation
            seg_device: Device for transformer inference (-1=CPU, 0+=CUDA GPU)

        Returns:
            Path to the generated database
        """
        # Use config.min_frequency if min_frequency not explicitly provided
        if min_frequency is None:
            min_frequency = self.config.min_frequency
        effective_text_col = text_col if text_col is not None else self.config.text_col
        effective_json_key = json_key if json_key is not None else self.config.json_key
        effective_word_engine = word_engine if word_engine is not None else self.config.word_engine

        # Wire extended Myanmar scope to all pipeline modules
        self._wire_extended_myanmar_scope()

        # Show pipeline header
        self._show_pipeline_header(min_frequency, incremental, effective_word_engine)

        # Setup directories
        intermediate_dir = self.work_dir / self.INTERMEDIATE_DIR
        frequencies_dir = self.work_dir / self.FREQUENCIES_DIR
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        frequencies_dir.mkdir(parents=True, exist_ok=True)

        output_db = Path(database_path)

        self._shutdown_requested = False
        self._setup_signal_handlers()

        try:
            # Start tracking total pipeline duration
            pipeline_start = time.time()

            # Pre-flight checks and incremental preparation
            processed_files_map, existing_counts, packager = self._preflight_checks(
                output_db, incremental, frequencies_dir
            )

            # Track step durations for summary
            # Value: (status, duration, sub_steps) where sub_steps is [(name, duration), ...]
            step_durations: dict[str, tuple[str, float, list[tuple[str, float]]]] = {}

            # Convert input_files to Path objects
            path_list = sorted([Path(p) for p in input_files]) if input_files else []
            raw_shards_dir = intermediate_dir / "raw_shards"

            # Detect corpus files removed since last incremental build
            if incremental and processed_files_map and path_list:
                current_paths = {str(p) for p in path_list}
                previously_processed = set(processed_files_map.keys())
                removed_files = previously_processed - current_paths
                if removed_files:
                    self.reporter.report_warning(
                        f"{len(removed_files)} previously processed file(s) no longer "
                        f"present in input list. Data from removed files persists in "
                        f"the database. Consider a full rebuild (without --incremental) "
                        f"to remove stale data."
                    )
                    for rf in sorted(removed_files):
                        self.logger.warning("Removed corpus file: %s", rf)

            # Check if we can skip Steps 1-2 entirely (segmented output exists,
            # shards may have been cleaned up)
            step2_complete, _ = self._check_step2_complete(raw_shards_dir)

            if step2_complete:
                # Skip Step 1 — segmented corpus already exists
                self.reporter.report_step_skipped(
                    1, 5, "Ingesting Corpus", "Segmented corpus already exists"
                )
                step_durations["1. Ingestion"] = ("skipped", 0.0, [])
                new_files_meta: list[tuple[str, float, int]] = []

                # Skip Step 2
                self._step2_segment(
                    raw_shards_dir,
                    intermediate_dir,
                    effective_word_engine,
                    step_durations,
                    seg_model=seg_model,
                    seg_device=seg_device,
                )
            else:
                # Step 1: Ingest
                shard_paths, new_files_meta = self._step1_ingest(
                    path_list,
                    raw_shards_dir,
                    sample,
                    effective_text_col,
                    effective_json_key,
                    processed_files_map,
                    incremental,
                    step_durations,
                )

                if self._check_shutdown(1):
                    return output_db

                if not any(p.exists() and p.stat().st_size > 0 for p in shard_paths):
                    raise PipelineError("No valid Myanmar text found in input files.")

                # Step 2: Segment
                self._step2_segment(
                    raw_shards_dir,
                    intermediate_dir,
                    effective_word_engine,
                    step_durations,
                    seg_model=seg_model,
                    seg_device=seg_device,
                )

            if self._check_shutdown(2):
                return output_db

            # Step 2.5: Initialize POS Tagger
            pos_tagger = self._init_pos_tagger(pos_tagger_config)

            # Step 3: Build Frequencies
            self._step3_build_frequencies(
                intermediate_dir,
                frequencies_dir,
                incremental,
                existing_counts,
                pos_tagger,
                step_durations,
                min_frequency,
                curated_words=curated_words,
            )

            if self._check_shutdown(3):
                return output_db

            # Step 4: Package Database
            self._step4_package(
                packager,
                output_db,
                new_files_meta,
                incremental,
                step_durations,
                curated_words=curated_words,
            )

            if self._check_shutdown(4):
                return output_db

            # Step 5: Enrich Database (confusable pairs, compounds, collocations, register)
            self._step5_enrich(output_db, step_durations)

            # Calculate total duration and show summary
            total_duration = time.time() - pipeline_start
            self._show_pipeline_summary(step_durations, total_duration, output_db)

            return output_db

        except OSError as e:
            if e.errno == errno.ENOSPC:
                self.reporter.report_error("CRITICAL: Disk full")
                self.reporter.report_error("Not enough space to complete the database build.")
                self.reporter.report_error("Please free up space or use a different --work-dir.")
                raise InsufficientStorageError(f"Disk full: {e}") from e
            raise

        finally:
            self._restore_signal_handlers()
            # Clean up intermediate subdirectories (chunks, raw_shards, duckdb_temp)
            self._cleanup_intermediates()
            # Cleanup using TemporaryDirectory context manager
            if self._owns_temp_dir and not self.keep_intermediate and self._temp_dir_context:
                self.reporter.report_progress(f"Cleaning up temporary directory: {self.work_dir}")
                self._temp_dir_context.cleanup()
                self._temp_dir_context = None


def run_pipeline(
    input_files: list[str] | None = None,
    database_path: str = DEFAULT_DB_NAME,
    work_dir: str | None = None,
    keep_intermediate: bool = False,
    sample: bool = False,
    text_col: str = "text",
    json_key: str = "text",
    pos_tagger_config: POSTaggerConfig | None = None,
    incremental: bool = False,
    word_engine: str = "myword",
    seg_model: str | None = None,
    seg_device: int = -1,
    min_frequency: int | None = None,
    worker_timeout: int = 300,
    num_workers: int | None = None,
    batch_size: int | None = None,
    allow_extended_myanmar: bool = False,
    curated_words: dict[str, str] | None = None,
    remove_segmentation_markers: bool = True,
    deduplicate_lines: bool = True,
    enrich: bool = True,
) -> None:
    """
    Convenience function to run the pipeline.

    Args:
        input_files: List of raw corpus text files
        database_path: Path to save the final SQLite database
        work_dir: Working directory for intermediate files
        keep_intermediate: If True, keep intermediate files after build
        sample: If True, ignore inputs and generate a sample corpus
        text_col: Column name/index for CSV/TSV ingestion
        json_key: Key name for JSON ingestion
        pos_tagger_config: POS tagger configuration for dynamic tagging
        incremental: If True, perform incremental update
        word_engine: Word segmentation engine ('crf', 'myword', 'transformer')
        seg_model: Custom model name for transformer engine (optional).
        seg_device: Device for transformer inference (-1=CPU, 0+=GPU).
        min_frequency: Minimum word frequency threshold. If None, uses config default.
        worker_timeout: Timeout in seconds for worker processes (default: 300).
        num_workers: Number of parallel workers. If None, auto-detects based on CPU cores.
        batch_size: Batch size for processing. If None, uses default.
            Reduce if running out of memory.
        allow_extended_myanmar: Include extended Myanmar blocks (default: False).
            When True, includes Extended Core, Extended-A, Extended-B blocks.
        curated_words: Optional dict of word→pos_tag for curated vocabulary.
            These words are marked 'is_curated=1' with POS tags preserved.
        remove_segmentation_markers: Strip spaces/underscores between Myanmar chars
            in pre-segmented corpus text (default: True).
        deduplicate_lines: Deduplicate lines during ingestion using hash-based
            tracking, both within and across files (default: True).
        enrich: Run Step 5 enrichment (confusable pairs, compounds, collocations,
            register tags) after database packaging (default: True).
    """
    from .config import PipelineConfig

    # Create config with all parameters
    config_kwargs = {
        "work_dir": work_dir,
        "keep_intermediate": keep_intermediate,
        "text_col": text_col,
        "json_key": json_key,
        "word_engine": word_engine,
        "seg_model": seg_model,
        "seg_device": seg_device,
        "pos_tagger": pos_tagger_config,
        "worker_timeout": worker_timeout,
        "allow_extended_myanmar": allow_extended_myanmar,
        "remove_segmentation_markers": remove_segmentation_markers,
        "deduplicate_lines": deduplicate_lines,
        "enrich": enrich,
    }
    # Add optional parameters if specified
    if num_workers is not None:
        config_kwargs["num_workers"] = num_workers
    if batch_size is not None:
        config_kwargs["batch_size"] = batch_size

    config = PipelineConfig(**config_kwargs)  # type: ignore[arg-type]
    if min_frequency is not None:
        config = config.with_overrides(min_frequency=min_frequency)

    pipeline = Pipeline(config=config)
    pipeline.build_database(
        input_files=[Path(p) for p in input_files] if input_files else [],
        database_path=database_path,
        sample=sample,
        text_col=text_col,
        json_key=json_key,
        min_frequency=min_frequency,
        pos_tagger_config=pos_tagger_config,
        incremental=incremental,
        word_engine=word_engine,
        curated_words=curated_words,
        seg_model=seg_model,
        seg_device=seg_device,
    )
