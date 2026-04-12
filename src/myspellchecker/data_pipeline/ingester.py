"""
Corpus ingestion module for the data pipeline.

Supports parallel ingestion via multiprocessing for improved performance
on large corpora. Uses fork-based workers with copy-on-write memory sharing.
"""

from __future__ import annotations

import multiprocessing as mp
import platform
import time
from collections.abc import Generator
from functools import lru_cache
from pathlib import Path

import xxhash

from ..core.config.text_configs import ZawgyiConfig
from ..core.constants import DEFAULT_BATCH_SIZE, DEFAULT_FILE_ENCODING

# Import IngestionError from central exceptions module (lightweight, no heavy deps)
from ..core.exceptions import IngestionError
from ..text.normalize import (
    is_myanmar_text,
    is_space_segmented_myanmar,
    normalize_with_zawgyi_conversion,
    remove_word_segmentation_markers,
)
from ..text.validator import validate_word
from ..utils.logging_utils import get_logger
from .config import runtime_flags as _flags

# Import from unified config to avoid duplicate constants
from .pipeline_config_unified import (
    get_default_num_workers,
    get_ingestion_parallel_threshold,
)

__all__ = [
    "CorpusIngester",
]

# Lazy imports for heavy dependencies (pyarrow, rich)
# These are only loaded when actually needed, reducing import time and memory
# for code that imports this module but doesn't use ingestion functionality.

# Lazy import functions using lru_cache for thread-safe caching.
# lru_cache is inherently thread-safe and handles concurrent access properly,
# avoiding the race condition that would occur with manual global variable caching.


@lru_cache(maxsize=1)
def _get_pyarrow():
    """Lazy load pyarrow module (thread-safe via lru_cache)."""
    import pyarrow as pa

    return pa


@lru_cache(maxsize=1)
def _get_rich_live():
    """Lazy load rich.live.Live class (thread-safe via lru_cache)."""
    from rich.live import Live

    return Live


@lru_cache(maxsize=1)
def _get_rich_table():
    """Lazy load rich.table.Table class (thread-safe via lru_cache)."""
    from rich.table import Table

    return Table


# Global flag for fork-based optimization (same pattern as frequency_builder.py)
_USE_FORK_OPTIMIZATION = platform.system() in ("Darwin", "Linux")

# Use unified config for constants
DEFAULT_NUM_WORKERS = get_default_num_workers()
MIN_PARALLEL_FILE_SIZE = get_ingestion_parallel_threshold()

# Use DEFAULT_BATCH_SIZE for parallel normalization batches


@lru_cache(maxsize=1)
def _get_ingest_schema():
    """Lazy load the Arrow schema for ingestion output (thread-safe via lru_cache)."""
    pa = _get_pyarrow()
    return pa.schema([("text", pa.string()), ("source", pa.string())])


# Myanmar text threshold for ingestion (more lenient than runtime to include more corpus data)
# Runtime threshold is 0.5 (see config.py), but ingestion uses 0.1 to capture
# mixed-language text that still contains significant Myanmar content
INGESTER_MYANMAR_TEXT_THRESHOLD = 0.1

# Module-level config for ingester's Myanmar text detection
_ingester_zawgyi_config = ZawgyiConfig(myanmar_text_threshold=INGESTER_MYANMAR_TEXT_THRESHOLD)


def _normalize_batch_py(batch: list[str]) -> list[tuple[str, bool]]:
    """
    Worker function: normalize a batch of lines (Python implementation).
    """
    results = []
    for line in batch:
        # Step 1-2: Zawgyi conversion + normalization
        cleaned = normalize_with_zawgyi_conversion(line)
        if cleaned:
            # Step 2.5: Remove word segmentation markers (spaces/underscores)
            if _flags.remove_segmentation_markers:
                cleaned = remove_word_segmentation_markers(cleaned)
            # Step 3: Check Myanmar text ratio (scope-aware)
            is_myanmar = is_myanmar_text(
                cleaned,
                config=_ingester_zawgyi_config,
                allow_extended=_flags.allow_extended_myanmar,
            )
            if is_myanmar:
                # Step 4: Validate text structure (reject contaminated patterns)
                # For line-level processing, we check each word
                words = cleaned.split()
                valid_words = [
                    w
                    for w in words
                    if validate_word(w, allow_extended_myanmar=_flags.allow_extended_myanmar)
                ]
                if valid_words:
                    # Reconstruct line with only valid words
                    cleaned = " ".join(valid_words)
                    results.append((cleaned, True))
                else:
                    results.append(("", False))
            else:
                results.append(("", False))
        else:
            results.append(("", False))
    return results


# Try to use Cython implementation for performance
try:
    from myspellchecker.data_pipeline.ingester_c import normalize_batch_c

    _normalize_batch = normalize_batch_c
    get_logger(__name__).debug("Using Cython optimized ingestion normalization")
except ImportError:
    _normalize_batch = _normalize_batch_py
    get_logger(__name__).debug("Using Python ingestion normalization (Cython extension not found)")


class CorpusIngester:
    """
    Ingests raw text from various formats and normalizes it.

    Supports: .txt, .csv, .tsv, .json, .jsonl, .parquet
    Output: Sharded Arrow IPC files (.arrow) suitable for parallel processing.

    For Parquet files, the ingester looks for a 'text' column first. If not found,
    it falls back to the first string column in the schema.
    """

    def __init__(
        self,
        worker_timeout: int = 1800,
        text_col: str = "text",
        json_key: str = "text",
        allow_partial_ingestion: bool = False,
    ):
        """
        Initialize the corpus ingester.

        Args:
            worker_timeout: Timeout in seconds for worker processes (default: 1800).
                Controls how long to wait for each parallel normalization batch
                to return a result. Set higher for very large corpora.
            text_col: CSV/TSV text column name or index (as string). Defaults to "text".
            json_key: JSON/JSONL key to read text from. Defaults to "text".
            allow_partial_ingestion: If True, log failed files and continue.
                If False (default), any failed file raises IngestionError.
        """
        self.logger = get_logger(__name__)
        self.worker_timeout = worker_timeout
        self.text_col = text_col
        self.json_key = json_key
        self.allow_partial_ingestion = allow_partial_ingestion

    def generate_sample_corpus(self) -> list[str]:
        """
        Generate a sample corpus for testing and demonstration.

        This method delegates to the sample_corpus module which contains
        ~270 diverse Myanmar sentences covering all supported grammar rules.

        See myspellchecker.data_pipeline.sample_corpus for full documentation.

        Returns:
            List of Myanmar sentences suitable for building a test database.

        Note:
            For production use, build with a real corpus of 10,000+ sentences.
        """
        from .sample_corpus import get_sample_corpus

        return get_sample_corpus()

    def process(
        self,
        input_path: str | Path,
        output_dir: str | Path,
        num_shards: int = 10,
        recursive: bool = False,
        parallel: bool | None = None,
        num_workers: int = DEFAULT_NUM_WORKERS,
    ) -> list[Path]:
        """
        Process input file or directory and write clean text to sharded Arrow files.

        Args:
            input_path: Path to input file or directory.
            output_dir: Directory to save sharded output files.
            num_shards: Number of output shards to distribute data into.
            recursive: If input is directory, read recursively (default: False).
            parallel: Force parallel (True) or sequential (False) mode.
                If None (default), auto-detect based on total file size.
            num_workers: Number of worker processes for parallel mode (default: min(8, CPU count)).

        Returns:
            list[Path]: List of generated shard files.
        """
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if input_path.is_file():
            files = [input_path]
        elif input_path.is_dir():
            pattern = "**/*" if recursive else "*"
            files = [
                f
                for f in input_path.glob(pattern)
                if f.is_file()
                and f.suffix.lower() in [".txt", ".csv", ".tsv", ".json", ".jsonl", ".parquet"]
            ]
        else:
            raise FileNotFoundError(f"Input path not found: {input_path}")

        self.logger.info(f"Found {len(files)} files to ingest.")

        # Auto-detect parallel mode based on total file size
        total_size = sum(f.stat().st_size for f in files if f.exists())
        use_parallel = parallel if parallel is not None else (total_size >= MIN_PARALLEL_FILE_SIZE)

        if use_parallel:
            self.logger.info(
                f"Using parallel mode (total: {total_size / (1024**2):.1f} MB, "
                f"threshold: {MIN_PARALLEL_FILE_SIZE / (1024**2):.0f} MB)"
            )
            shard_paths, _ = self._process_sharded_parallel(
                files, output_dir, num_shards, num_workers
            )
            return shard_paths
        else:
            self.logger.info(
                f"Using sequential mode (total: {total_size / (1024**2):.1f} MB, "
                f"threshold: {MIN_PARALLEL_FILE_SIZE / (1024**2):.0f} MB)"
            )
            shard_paths, _ = self._process_sharded(files, output_dir, num_shards)
            return shard_paths

    def _process_sharded(
        self,
        files: list[Path],
        output_dir: Path,
        num_shards: int,
        processed_files_map: dict[str, tuple[float, int]] | None = None,
    ) -> tuple[list[Path], list[tuple[str, float, int]]]:
        """
        Read files and distribute lines round-robin into sharded Arrow IPC files.

        Args:
            files: List of input file paths to process.
            output_dir: Directory to write output shards.
            num_shards: Number of output shards.
            processed_files_map: Optional dict mapping file path -> (mtime, size)
                for previously processed files. Files with matching mtime and size
                will be skipped (incremental mode).

        Returns:
            Tuple of (shard_paths, new_files_meta) where new_files_meta contains
            metadata for files that were actually processed (not skipped).

        Raises:
            IngestionError: If any input files are missing or fail to process.
        """
        # Lazy load heavy dependencies only when method is called
        pa = _get_pyarrow()
        Table = _get_rich_table()
        Live = _get_rich_live()
        INGEST_SCHEMA = _get_ingest_schema()

        # Validate all input files exist before processing
        missing_files = [str(f) for f in files if not f.exists()]
        if missing_files:
            raise IngestionError(
                f"Cannot start ingestion: {len(missing_files)} input file(s) not found",
                missing_files=missing_files,
            )

        shard_writers = []
        shard_handles = []
        shard_paths = []

        # Buffer structure: List of dicts of lists (columns)
        # shard_buffers[i] = {'text': [], 'source': []}
        shard_buffers: list[dict[str, list[str]]] = [
            {"text": [], "source": []} for _ in range(num_shards)
        ]

        # Track metadata for successfully processed input files
        processed_input_files_meta: list[tuple[str, float, int]] = []
        # Track files that failed during processing
        failed_files: list[tuple[str, str]] = []

        # Deduplication tracking (shared across all files for cross-file dedup)
        seen_hashes: set | None = set() if _flags.deduplicate_lines else None
        dedup_count = 0

        try:
            # Initialize writers
            for i in range(num_shards):
                p = output_dir / f"raw_shard_{i:03d}.arrow"
                shard_paths.append(p)
                f = pa.OSFile(str(p), "w")
                shard_handles.append(f)
                writer = pa.RecordBatchStreamWriter(f, INGEST_SCHEMA)
                shard_writers.append(writer)

            line_count = 0
            total_bytes = sum(f.stat().st_size for f in files if f.exists())
            bytes_processed: float = 0
            total_files = len(files)
            files_completed = 0
            skipped_files = 0
            start_time = time.perf_counter()
            last_update_lines = 0
            update_interval = 10000  # Update display every 10k lines

            def create_progress_display():
                """Create a Rich table showing real-time ingestion progress."""
                elapsed = time.perf_counter() - start_time
                rate = line_count / elapsed if elapsed > 0 else 0
                bytes_pct = (bytes_processed / total_bytes * 100) if total_bytes > 0 else 0

                # Format bytes for display
                if bytes_processed >= 1024**3:
                    bytes_str = f"{bytes_processed / 1024**3:.2f}GB"
                    total_str = f"{total_bytes / 1024**3:.2f}GB"
                elif bytes_processed >= 1024**2:
                    bytes_str = f"{bytes_processed / 1024**2:.1f}MB"
                    total_str = f"{total_bytes / 1024**2:.1f}MB"
                else:
                    bytes_str = f"{bytes_processed / 1024:.0f}KB"
                    total_str = f"{total_bytes / 1024:.0f}KB"

                # Estimate remaining time based on bytes
                if bytes_processed > 0 and total_bytes > bytes_processed:
                    bytes_rate = bytes_processed / elapsed if elapsed > 0 else 0
                    eta_seconds = (
                        (total_bytes - bytes_processed) / bytes_rate if bytes_rate > 0 else 0
                    )
                    eta_min = int(eta_seconds // 60)
                    eta_sec = int(eta_seconds % 60)
                    eta_str = f"{eta_min}m {eta_sec}s"
                else:
                    eta_str = "--"

                table = Table.grid(padding=(0, 2))
                table.add_column(justify="left")
                table.add_row(
                    f"[cyan]⠋ Ingesting...[/cyan] "
                    f"[bold]{files_completed}/{total_files}[/bold] files | "
                    f"[green]{bytes_str}[/green] / {total_str} "
                    f"([yellow]{bytes_pct:.1f}%[/yellow]) | "
                    f"[blue]{line_count:,}[/blue] lines | "
                    f"[magenta]{rate:,.0f}[/magenta] lines/sec | "
                    f"ETA: [cyan]{eta_str}[/cyan]"
                )
                return table

            live_display = Live(create_progress_display(), refresh_per_second=4, transient=True)
            with live_display as live:
                for file_path in files:
                    try:
                        # Get file metadata before processing
                        file_stat = file_path.stat()
                        file_mtime = file_stat.st_mtime
                        file_size = file_stat.st_size

                        # Incremental mode: Skip unchanged files
                        path_str = str(file_path.absolute())
                        if processed_files_map and path_str in processed_files_map:
                            old_mtime, old_size = processed_files_map[path_str]
                            if file_mtime == old_mtime and file_size == old_size:
                                self.logger.debug(f"Skipping unchanged file: {file_path.name}")
                                skipped_files += 1
                                bytes_processed += file_size
                                files_completed += 1
                                live.update(create_progress_display())
                                continue

                        # Auto-detect space-segmented files by sampling first 50 lines
                        file_remove_spaces = False
                        if _flags.remove_segmentation_markers:
                            sample_lines = []
                            for sample_line in self._read_file(
                                file_path, text_col=self.text_col, json_key=self.json_key
                            ):
                                if sample_line:
                                    sample_lines.append(sample_line)
                                if len(sample_lines) >= 50:
                                    break
                            if is_space_segmented_myanmar(sample_lines):
                                file_remove_spaces = True
                                self.logger.info(
                                    f"Auto-detected space-segmented Myanmar: {file_path.name}"
                                )

                        for line in self._read_file(
                            file_path, text_col=self.text_col, json_key=self.json_key
                        ):
                            # Cleaning Logic with Zawgyi conversion and validation
                            # Step 1-2: Zawgyi conversion + normalization
                            cleaned = normalize_with_zawgyi_conversion(line)

                            if not cleaned:
                                continue

                            # Step 2.5: Remove word segmentation markers
                            if _flags.remove_segmentation_markers:
                                cleaned = remove_word_segmentation_markers(
                                    cleaned, remove_spaces=file_remove_spaces
                                )

                            # Step 3: Check Myanmar text ratio (scope-aware)
                            if not is_myanmar_text(
                                cleaned,
                                config=_ingester_zawgyi_config,
                                allow_extended=_flags.allow_extended_myanmar,
                            ):
                                continue

                            # Step 4: Validate text structure (reject contaminated patterns)
                            words = cleaned.split()
                            valid_words = [
                                w
                                for w in words
                                if validate_word(
                                    w, allow_extended_myanmar=_flags.allow_extended_myanmar
                                )
                            ]
                            if not valid_words:
                                continue

                            # Reconstruct line with only valid words
                            cleaned = " ".join(valid_words)

                            # Step 5: Deduplication (hash-based, cross-file)
                            if seen_hashes is not None:
                                line_hash = xxhash.xxh3_64_intdigest(cleaned.encode("utf-8"))
                                if line_hash in seen_hashes:
                                    dedup_count += 1
                                    continue
                                seen_hashes.add(line_hash)

                            shard_idx = line_count % num_shards
                            shard_buffers[shard_idx]["text"].append(cleaned)
                            shard_buffers[shard_idx]["source"].append(file_path.name)

                            line_count += 1

                            # Flush buffer if full
                            if len(shard_buffers[shard_idx]["text"]) >= DEFAULT_BATCH_SIZE:
                                batch = pa.RecordBatch.from_pydict(
                                    shard_buffers[shard_idx], schema=INGEST_SCHEMA
                                )
                                shard_writers[shard_idx].write_batch(batch)
                                # Clear buffer
                                shard_buffers[shard_idx]["text"].clear()
                                shard_buffers[shard_idx]["source"].clear()

                            # Update display periodically
                            if line_count - last_update_lines >= update_interval:
                                last_update_lines = line_count
                                live.update(create_progress_display())

                        # Update progress after each file is processed
                        bytes_processed += file_size
                        files_completed += 1
                        live.update(create_progress_display())

                        # If file processed successfully, add its metadata
                        processed_input_files_meta.append((str(file_path), file_mtime, file_size))

                    except (OSError, pa.ArrowIOError, pa.ArrowInvalid, ValueError) as e:
                        self.logger.error(f"Error processing file {file_path}: {e}")
                        failed_files.append((str(file_path), str(e)))

            if skipped_files > 0:
                self.logger.info(f"Incremental mode: Skipped {skipped_files} unchanged file(s)")
            if dedup_count > 0:
                total_processed = line_count + dedup_count
                pct = dedup_count / total_processed * 100 if total_processed > 0 else 0
                self.logger.info(
                    f"Deduplicated {dedup_count:,} duplicate lines ({pct:.1f}% of total)"
                )
        finally:
            # Flush remaining buffers and close writers
            # Use separate loops to handle partial initialization
            for i in range(len(shard_writers)):
                try:
                    if i < len(shard_buffers) and shard_buffers[i]["text"]:
                        batch = pa.RecordBatch.from_pydict(shard_buffers[i], schema=INGEST_SCHEMA)
                        shard_writers[i].write_batch(batch)
                except (pa.ArrowIOError, pa.ArrowInvalid, OSError) as e:
                    self.logger.error(f"Error flushing buffer for shard {i}: {e}")

            # Close writers in separate loop for robustness
            for i in range(len(shard_writers)):
                try:
                    shard_writers[i].close()
                except (pa.ArrowIOError, OSError) as e:
                    self.logger.error(f"Error closing writer for shard {i}: {e}")

            # Close file handles separately to ensure cleanup even if writer close fails
            for i in range(len(shard_handles)):
                try:
                    shard_handles[i].close()
                except OSError as e:
                    self.logger.error(f"Error closing file handle for shard {i}: {e}")

        if failed_files:
            if self.allow_partial_ingestion:
                self.logger.warning(
                    "Ingestion completed with %d failed file(s) out of %d: %s",
                    len(failed_files),
                    len(files),
                    [(name, err) for name, err in failed_files],
                )
            else:
                raise IngestionError(
                    "Ingestion failed for one or more files.",
                    failed_files=failed_files,
                )

        return shard_paths, processed_input_files_meta

    def _process_sharded_parallel(
        self,
        files: list[Path],
        output_dir: Path,
        num_shards: int,
        num_workers: int = DEFAULT_NUM_WORKERS,
        processed_files_map: dict[str, tuple[float, int]] | None = None,
    ) -> tuple[list[Path], list[tuple[str, float, int]]]:
        """
        Parallel version of _process_sharded using multiprocessing for normalization.

        Uses fork-based workers for copy-on-write memory sharing on Unix systems.
        Falls back to spawn on Windows.

        Args:
            files: List of input file paths to process.
            output_dir: Directory to write output shards.
            num_shards: Number of output shards.
            num_workers: Number of worker processes (default: CPU count, max 8).
            processed_files_map: Optional dict for incremental mode.

        Returns:
            Tuple of (shard_paths, new_files_meta).

        Raises:
            IngestionError: If any input files are missing or fail to process.
        """
        # Lazy load heavy dependencies only when method is called
        pa = _get_pyarrow()
        Table = _get_rich_table()
        Live = _get_rich_live()
        INGEST_SCHEMA = _get_ingest_schema()

        # Validate all input files exist before processing
        missing_files = [str(f) for f in files if not f.exists()]
        if missing_files:
            raise IngestionError(
                f"Cannot start ingestion: {len(missing_files)} input file(s) not found",
                missing_files=missing_files,
            )

        shard_writers = []
        shard_handles = []
        shard_paths = []

        shard_buffers: list[dict[str, list[str]]] = [
            {"text": [], "source": []} for _ in range(num_shards)
        ]

        processed_input_files_meta: list[tuple[str, float, int]] = []
        # Track files that failed during processing
        failed_files: list[tuple[str, str]] = []

        # Deduplication tracking (shared across all files for cross-file dedup)
        seen_hashes: set | None = set() if _flags.deduplicate_lines else None
        dedup_count = 0

        # Select multiprocessing context
        ctx: mp.context.BaseContext
        if _USE_FORK_OPTIMIZATION:
            ctx = mp.get_context("fork")
            self.logger.info(f"Using fork-based parallelism with {num_workers} workers")
        else:
            ctx = mp.get_context("spawn")
            self.logger.info(f"Using spawn-based parallelism with {num_workers} workers")

        try:
            # Initialize shard writers
            for i in range(num_shards):
                p = output_dir / f"raw_shard_{i:03d}.arrow"
                shard_paths.append(p)
                f = pa.OSFile(str(p), "w")
                shard_handles.append(f)
                writer = pa.RecordBatchStreamWriter(f, INGEST_SCHEMA)
                shard_writers.append(writer)

            line_count = 0
            total_bytes = sum(f.stat().st_size for f in files if f.exists())
            bytes_processed: float = 0
            cumulative_file_bytes = 0  # Track actual file sizes for progress correction
            total_files = len(files)
            files_completed = 0
            skipped_files = 0
            start_time = time.perf_counter()

            def create_progress_display():
                """Create a Rich table showing real-time ingestion progress."""
                elapsed = time.perf_counter() - start_time
                rate = line_count / elapsed if elapsed > 0 else 0
                bytes_pct = (bytes_processed / total_bytes * 100) if total_bytes > 0 else 0

                # Format bytes for display
                if bytes_processed >= 1024**3:
                    bytes_str = f"{bytes_processed / 1024**3:.2f}GB"
                    total_str = f"{total_bytes / 1024**3:.2f}GB"
                elif bytes_processed >= 1024**2:
                    bytes_str = f"{bytes_processed / 1024**2:.1f}MB"
                    total_str = f"{total_bytes / 1024**2:.1f}MB"
                else:
                    bytes_str = f"{bytes_processed / 1024:.0f}KB"
                    total_str = f"{total_bytes / 1024:.0f}KB"

                # Estimate remaining time based on bytes
                if bytes_processed > 0 and total_bytes > bytes_processed:
                    bytes_rate = bytes_processed / elapsed if elapsed > 0 else 0
                    remaining_bytes = total_bytes - bytes_processed
                    eta_seconds = remaining_bytes / bytes_rate if bytes_rate > 0 else 0
                    eta_min = int(eta_seconds // 60)
                    eta_sec = int(eta_seconds % 60)
                    eta_str = f"{eta_min}m {eta_sec}s"
                else:
                    eta_str = "--"

                table = Table.grid(padding=(0, 2))
                table.add_column(justify="left")
                table.add_row(
                    f"[cyan]⠋ Ingesting...[/cyan] "
                    f"[bold]{files_completed}/{total_files}[/bold] files | "
                    f"[green]{bytes_str}[/green] / {total_str} "
                    f"([yellow]{bytes_pct:.1f}%[/yellow]) | "
                    f"[blue]{line_count:,}[/blue] lines | "
                    f"[magenta]{rate:,.0f}[/magenta] lines/sec | "
                    f"ETA: [cyan]{eta_str}[/cyan]"
                )
                return table

            # Create worker pool
            with ctx.Pool(processes=num_workers) as pool:
                live_display = Live(create_progress_display(), refresh_per_second=4, transient=True)
                with live_display as live:
                    for file_path in files:
                        try:
                            file_stat = file_path.stat()
                            file_mtime = file_stat.st_mtime
                            file_size = file_stat.st_size

                            # Incremental mode: skip unchanged files
                            path_str = str(file_path.absolute())
                            if processed_files_map and path_str in processed_files_map:
                                old_mtime, old_size = processed_files_map[path_str]
                                if file_mtime == old_mtime and file_size == old_size:
                                    self.logger.debug(f"Skipping unchanged: {file_path.name}")
                                    skipped_files += 1
                                    cumulative_file_bytes += file_size
                                    bytes_processed = cumulative_file_bytes
                                    files_completed += 1
                                    live.update(create_progress_display())
                                    continue

                            # Stream lines into batches and submit to pool incrementally
                            # (never materializes entire file — only one batch in memory at a time)
                            def _iter_batches(file_reader):
                                current_batch: list = []
                                for raw_line in file_reader:
                                    current_batch.append(raw_line)
                                    if len(current_batch) >= DEFAULT_BATCH_SIZE:
                                        yield current_batch
                                        current_batch = []
                                if current_batch:
                                    yield current_batch

                            batch_gen = _iter_batches(
                                self._read_file(
                                    file_path, text_col=self.text_col, json_key=self.json_key
                                )
                            )
                            # Estimate bytes per batch for progress
                            est_lines_per_file = max(file_size // 80, 1)  # ~80 bytes/line
                            est_batches = max(est_lines_per_file // DEFAULT_BATCH_SIZE, 1)
                            bytes_per_batch = file_size / est_batches

                            # Process batches in parallel (pool.imap consumes lazily)
                            # Apply per-batch timeout to detect hung workers
                            source_name = file_path.name
                            imap_iter = pool.imap(_normalize_batch, batch_gen)
                            while True:
                                try:
                                    batch_results = imap_iter.next(timeout=self.worker_timeout)
                                except StopIteration:
                                    break
                                except mp.TimeoutError:
                                    self.logger.error(
                                        "Worker timed out after %ds processing "
                                        "batch from %s — worker may be hung",
                                        self.worker_timeout,
                                        file_path.name,
                                    )
                                    raise
                                for cleaned, is_valid in batch_results:
                                    if is_valid and cleaned:
                                        # Deduplication (hash-based, cross-file)
                                        if seen_hashes is not None:
                                            line_hash = xxhash.xxh3_64_intdigest(
                                                cleaned.encode("utf-8")
                                            )
                                            if line_hash in seen_hashes:
                                                dedup_count += 1
                                                continue
                                            seen_hashes.add(line_hash)

                                        shard_idx = line_count % num_shards
                                        shard_buffers[shard_idx]["text"].append(cleaned)
                                        shard_buffers[shard_idx]["source"].append(source_name)
                                        line_count += 1

                                        # Flush buffer if full
                                        if (
                                            len(shard_buffers[shard_idx]["text"])
                                            >= DEFAULT_BATCH_SIZE
                                        ):
                                            batch = pa.RecordBatch.from_pydict(
                                                shard_buffers[shard_idx], schema=INGEST_SCHEMA
                                            )
                                            shard_writers[shard_idx].write_batch(batch)
                                            shard_buffers[shard_idx]["text"].clear()
                                            shard_buffers[shard_idx]["source"].clear()

                                # Update progress after each batch
                                bytes_processed += bytes_per_batch
                                live.update(create_progress_display())

                            # Correct bytes_processed for batch estimation drift
                            # Track cumulative actual file sizes instead of O(n²) re-stat
                            cumulative_file_bytes += file_size
                            bytes_processed = cumulative_file_bytes
                            files_completed += 1
                            processed_input_files_meta.append(
                                (str(file_path), file_mtime, file_size)
                            )

                        except (OSError, pa.ArrowIOError, pa.ArrowInvalid, ValueError) as e:
                            self.logger.error(f"Error processing file {file_path}: {e}")
                            failed_files.append((str(file_path), str(e)))

            if skipped_files > 0:
                self.logger.info(f"Incremental mode: Skipped {skipped_files} unchanged file(s)")
            if dedup_count > 0:
                total_processed = line_count + dedup_count
                pct = dedup_count / total_processed * 100 if total_processed > 0 else 0
                self.logger.info(
                    f"Deduplicated {dedup_count:,} duplicate lines ({pct:.1f}% of total)"
                )
            self.logger.info(f"Parallel ingestion complete. Total lines: {line_count:,}")

        finally:
            # Flush remaining buffers and close writers
            # Use separate loops to handle partial initialization
            for i in range(len(shard_writers)):
                try:
                    if i < len(shard_buffers) and shard_buffers[i]["text"]:
                        batch = pa.RecordBatch.from_pydict(shard_buffers[i], schema=INGEST_SCHEMA)
                        shard_writers[i].write_batch(batch)
                except (pa.ArrowIOError, pa.ArrowInvalid, OSError) as e:
                    self.logger.error(f"Error flushing buffer for shard {i}: {e}")

            # Close writers in separate loop for robustness
            for i in range(len(shard_writers)):
                try:
                    shard_writers[i].close()
                except (pa.ArrowIOError, OSError) as e:
                    self.logger.error(f"Error closing writer for shard {i}: {e}")

            # Close file handles separately to ensure cleanup even if writer close fails
            for i in range(len(shard_handles)):
                try:
                    shard_handles[i].close()
                except OSError as e:
                    self.logger.error(f"Error closing file handle for shard {i}: {e}")

        if failed_files:
            if self.allow_partial_ingestion:
                self.logger.warning(
                    "Ingestion completed with %d failed file(s) out of %d: %s",
                    len(failed_files),
                    len(files),
                    [(name, err) for name, err in failed_files],
                )
            else:
                raise IngestionError(
                    "Ingestion failed for one or more files.",
                    failed_files=failed_files,
                )

        return shard_paths, processed_input_files_meta

    def _read_file(
        self,
        file_path: Path,
        text_col: str | None = None,
        json_key: str | None = None,
    ) -> Generator[str, None, None]:
        """
        Yield lines from a file, handling different formats.
        """
        # We need to import json here since we removed global orjson import
        import json

        suffix = file_path.suffix.lower()
        resolved_text_col = self.text_col if text_col is None else text_col
        resolved_json_key = self.json_key if json_key is None else json_key

        # Support index form for CSV/TSV/Parquet (e.g. "0", "2")
        col_index: int | None = None
        if resolved_text_col.isdigit():
            col_index = int(resolved_text_col)
            if col_index < 0:
                raise ValueError(f"text_col index must be >= 0, got {resolved_text_col}")

        try:
            with open(file_path, encoding=DEFAULT_FILE_ENCODING) as f:
                if suffix == ".txt":
                    for line in f:
                        yield line.strip()
                elif suffix == ".jsonl":
                    for line_num, line in enumerate(f, start=1):
                        try:
                            data = json.loads(line)
                            if isinstance(data, dict):
                                if resolved_json_key not in data:
                                    raise ValueError(
                                        f"Missing key '{resolved_json_key}' in JSONL {file_path} "
                                        f"(line {line_num})"
                                    )
                                yield str(data.get(resolved_json_key, "")).strip()
                            else:
                                yield str(data).strip()
                        except json.JSONDecodeError:
                            continue
                elif suffix == ".csv" or suffix == ".tsv":
                    import csv

                    delimiter = "\t" if suffix == ".tsv" else ","
                    if col_index is not None:
                        csv_reader = csv.reader(f, delimiter=delimiter)
                        for row_num, row in enumerate(csv_reader, start=1):
                            if col_index >= len(row):
                                raise ValueError(
                                    f"text_col index {col_index} out of range for {file_path} "
                                    f"(line {row_num}, columns={len(row)})"
                                )
                            value = row[col_index].strip()
                            if value:
                                yield value
                    else:
                        dict_reader = csv.DictReader(f, delimiter=delimiter)
                        fieldnames = dict_reader.fieldnames or []
                        if resolved_text_col not in fieldnames:
                            raise ValueError(
                                f"Column '{resolved_text_col}' not found in {file_path}. "
                                f"Available columns: {fieldnames}"
                            )
                        for dict_row in dict_reader:
                            value = dict_row.get(resolved_text_col, "")
                            if value:
                                stripped = value.strip()
                                if stripped:
                                    yield stripped
                elif suffix == ".json":
                    file_size = file_path.stat().st_size
                    if file_size > 500 * 1024 * 1024:  # 500MB
                        raise ValueError(
                            f"JSON file too large ({file_size / (1024**3):.1f} GB) "
                            f"for in-memory parsing: {file_path}. "
                            f"Convert to .jsonl: jq -c '.[]' {file_path} > output.jsonl"
                        )
                    try:
                        data = json.load(f)
                        if isinstance(data, list):
                            for idx, item in enumerate(data):
                                if isinstance(item, dict):
                                    if resolved_json_key not in item:
                                        raise ValueError(
                                            "Missing key "
                                            f"'{resolved_json_key}' in JSON {file_path} "
                                            f"(item index {idx})"
                                        )
                                    yield str(item.get(resolved_json_key, "")).strip()
                                else:
                                    yield str(item).strip()
                        elif isinstance(data, dict):
                            if resolved_json_key not in data:
                                raise ValueError(
                                    f"Missing key '{resolved_json_key}' in JSON {file_path}"
                                )
                            yield str(data.get(resolved_json_key, "")).strip()
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Invalid JSON in {file_path}: {e}")

        except (OSError, UnicodeDecodeError) as e:
            self.logger.error(f"Failed to read {file_path}: {e}")
            raise  # Re-raise to propagate to caller

        # Handle Parquet separately (binary format, not opened with text mode)
        if suffix == ".parquet":
            try:
                import pyarrow.parquet as pq

                parquet_file = pq.ParquetFile(file_path)
                schema = parquet_file.schema_arrow

                text_column: str | None = None
                if col_index is not None:
                    if col_index >= len(schema.names):
                        raise ValueError(
                            f"text_col index {col_index} out of range for {file_path}. "
                            f"Columns: {schema.names}"
                        )
                    text_column = schema.names[col_index]
                elif resolved_text_col in schema.names:
                    text_column = resolved_text_col

                if text_column is None:
                    raise ValueError(
                        f"Column '{resolved_text_col}' not found in Parquet file {file_path}. "
                        f"Available columns: {schema.names}"
                    )

                # Stream batches — never loads entire file into memory
                for batch in parquet_file.iter_batches(batch_size=10000, columns=[text_column]):
                    for value in batch.column(text_column):
                        text = value.as_py()
                        if text and isinstance(text, str):
                            stripped = text.strip()
                            if stripped:
                                yield stripped

            except (OSError, ValueError, KeyError) as e:
                self.logger.error(f"Failed to read Parquet file {file_path}: {e}")
                raise
