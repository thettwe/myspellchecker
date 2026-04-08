"""
Corpus segmentation module for the data pipeline.

Optimized for high-throughput processing of large Myanmar text corpora.
Implements fork-based multiprocessing with pre-loaded models for maximum efficiency.

This module re-exports all public symbols from the split submodules for
backward compatibility:
- ``_segmenter_config``: Capability detection, state management, utility functions
- ``_segmenter_workers``: Worker functions for parallel processing
"""

from __future__ import annotations

import errno
import multiprocessing
import time
from pathlib import Path
from typing import Any, Callable, cast

import pyarrow as pa  # type: ignore
from rich.console import Console

from ..core.constants import SEGMENTER_ENGINE_MYWORD
from ..core.exceptions import ConfigurationError, PipelineError
from ..segmenters import DefaultSegmenter
from ..utils.logging_utils import get_logger

# Re-export everything from _segmenter_config for backward compatibility
from ._segmenter_config import (  # noqa: F401
    _CAPABILITIES,
    _STATE,
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_BASE_DELAY,
    LEADING_PUNCT_CHARS,
    SEGMENT_SCHEMA,
    ZERO_WIDTH_CHARS,
    _first_significant_char,
    _format_count,
    _is_pre_segmented,
    _ProcessState,
    _SegmenterCapabilities,
    get_optimal_batch_size,
    get_optimal_worker_count,
    is_myanmar_token,
)

# Re-export everything from _segmenter_workers for backward compatibility
from ._segmenter_workers import (  # noqa: F401
    init_worker,
    init_worker_fork,
    preload_models,
    worker_segment_file,
)

__all__ = [
    "CorpusSegmenter",
]


class CorpusSegmenter:
    """Handles segmentation and cleaning of Myanmar corpus data."""

    def __init__(
        self,
        output_dir: str | Path | None = None,
        word_engine: str = SEGMENTER_ENGINE_MYWORD,
        seg_model: str | None = None,
        seg_device: int = -1,
        worker_timeout: int = 1800,
    ):
        """
        Initialize the corpus segmenter.

        Args:
            output_dir: Directory to save segmented files (optional)
            word_engine: Word segmentation engine ('crf', 'myword', 'transformer')
            seg_model: Custom model name for transformer engine (optional)
            seg_device: Device for transformer inference (-1=CPU, 0+=GPU)
            worker_timeout: Timeout in seconds for worker processes (default: 1800).
                Controls how long to wait for each worker future to return a result.
        """
        self.logger = get_logger(__name__)
        self.output_dir: Path | None
        if output_dir:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.output_dir = None

        # Store engine but don't load models yet - preload_models() will do it
        # This avoids duplicate model loading when using parallel processing
        self._segmenter: DefaultSegmenter | None = None
        self.word_engine = word_engine
        self.seg_model = seg_model
        self.seg_device = seg_device
        self.worker_timeout = worker_timeout

        # Statistics
        self.stats = {
            "total_sentences": 0,
            "total_syllables": 0,
            "total_words": 0,
            "unique_syllables": 0,
            "unique_words": 0,
            "avg_syllables_per_sentence": 0.0,
            "avg_words_per_sentence": 0.0,
            "avg_syllables_per_word": 0.0,
        }

    def segment_corpus(
        self,
        input_files: list[Path] | Path,
        num_workers: int | None = None,
        batch_size: int | None = None,
    ) -> Path:
        """
        Segment all sentences using file-based parallel processing.
        Accepts a list of sharded Arrow files from Ingester.

        Optimizations applied:
        - Fork-based multiprocessing (Unix): Workers inherit pre-loaded models
        - Auto-tuned worker count based on system resources
        - Auto-tuned batch size based on file size

        Args:
            input_files: List of paths to input Arrow shards OR path to directory containing shards.
            num_workers: Number of parallel processes (auto-detected if None)
            batch_size: Batch size for processing (auto-detected if None)

        Returns:
            Path to the output Arrow file
        """
        if not self.output_dir:
            raise PipelineError("Output directory not set.")

        # Resolve input files
        files_to_process = []
        if isinstance(input_files, list):
            files_to_process = input_files
        else:
            p = Path(input_files)
            if p.is_dir():
                files_to_process = sorted(p.glob("*.arrow"))
            elif p.is_file():
                files_to_process = [p]

        if not files_to_process:
            raise FileNotFoundError(f"No input files found in {input_files}")

        # Calculate total file size for optimization decisions
        total_size_mb = sum(f.stat().st_size for f in files_to_process) / (1024 * 1024)

        # Create Rich console for nice output
        console = Console()

        # Count total sentences FIRST for accurate memory estimation
        # This is critical for auto-detecting optimal worker count
        total_sentences = 0
        from rich.status import Status

        with Status("[cyan]Counting sentences...[/cyan]", console=console, spinner="dots"):
            for chunk_file in files_to_process:
                try:
                    with pa.memory_map(str(chunk_file), "r") as source:
                        reader = pa.ipc.open_stream(source)
                        for batch in reader:
                            total_sentences += batch.num_rows
                except (pa.ArrowIOError, pa.ArrowInvalid, OSError) as e:
                    self.logger.warning(f"Could not count rows in {chunk_file.name}: {e}")
        console.print(f"[green]✓[/green] Total sentences: {total_sentences:,}")

        # Auto-tune worker count if not specified
        # Now uses total_sentences for better memory estimation
        if num_workers is None:
            num_workers = get_optimal_worker_count(
                file_count=len(files_to_process),
                file_size_mb=total_size_mb,
                total_sentences=total_sentences,
            )
            self.logger.debug(f"Auto-detected optimal worker count: {num_workers}")

        # Auto-tune batch size if not specified
        if batch_size is None:
            batch_size = get_optimal_batch_size(file_size_mb=total_size_mb)
            self.logger.debug(f"Auto-detected optimal batch size: {batch_size}")

        # Validate output directory is configured
        if self.output_dir is None:
            raise ConfigurationError(
                "Output directory not configured. "
                "Pass output_dir parameter to CorpusSegmenter constructor."
            )

        # Output file is now an Arrow file
        sentences_path = self.output_dir / "segmented_corpus.arrow"

        # 1. Check for existing progress (Resume logic)
        if sentences_path.exists():
            self.logger.debug("Found existing output. Resuming/Merging...")

        # 2. Pre-load models in parent process (for fork optimization)
        if _CAPABILITIES.use_fork_optimization:
            preload_models(
                self.word_engine,
                console=console,
                seg_model=self.seg_model,
                seg_device=self.seg_device,
            )
            # Show worker config after model loading
            mode_str = "fork" if _CAPABILITIES.use_fork_optimization else "spawn"
            openmp_str = "OpenMP enabled" if _CAPABILITIES.has_openmp else "OpenMP disabled"
            console.print(
                f"[green]✓[/green] Workers: {num_workers} ({mode_str} mode, {openmp_str})"
            )
            console.print(f"[green]✓[/green] Batch size: {batch_size:,} sentences")
        else:
            self.logger.info("Fork optimization not available (Windows?), using spawn mode...")

        import concurrent.futures

        from rich.live import Live
        from rich.table import Table

        tasks = []
        skipped_chunks = 0

        # Ensure temporary chunk directory exists
        chunk_out_dir = self.output_dir / "chunks"
        chunk_out_dir.mkdir(parents=True, exist_ok=True)

        # Map input files to output chunk files
        chunk_files = []

        for i, chunk_file in enumerate(files_to_process):
            c_out = chunk_out_dir / f"chunk_{i}_segmented.arrow"
            chunk_files.append(c_out)

            # Resume Check: File must exist AND have valid content (non-zero size)
            if c_out.exists():
                file_size = c_out.stat().st_size
                if file_size > 0:
                    skipped_chunks += 1
                    continue
                else:
                    # Remove empty/corrupted file from previous failed run
                    self.logger.debug(f"Removing empty chunk file: {c_out.name}")
                    c_out.unlink()

            tasks.append(
                {
                    "input_file": str(chunk_file),
                    "output_dir": str(chunk_out_dir),
                    "chunk_id": i,
                }
            )

        if skipped_chunks > 0:
            self.logger.info(f"Resumed: Skipped {skipped_chunks} already processed chunks.")

        # Create shared progress counter for real-time display
        # 'q' = unsigned long long (8 bytes), supports up to 18 quintillion
        shared_counter = multiprocessing.Value("q", 0)

        # Configure multiprocessing context
        initializer: Callable[..., Any]
        initargs: tuple[Any, ...]
        if _CAPABILITIES.use_fork_optimization and _STATE.models_preloaded:
            # Use fork context for copy-on-write memory sharing
            # Workers will inherit preloaded models automatically
            mp_context = multiprocessing.get_context("fork")
            initializer = init_worker_fork
            initargs = (shared_counter,)
            self.logger.debug("Workers will inherit preloaded models via fork (COW).")
        else:
            # Fallback to default context (spawn on Windows)
            mp_context = None
            initializer = init_worker
            initargs = (self.word_engine, None, shared_counter)

        # Process tasks
        pipeline_start_time = time.perf_counter()

        # Aggregated performance stats from all workers
        perf_aggregate: dict[str, float | int | list[float]] = {
            "total_read_time": 0.0,
            "total_process_time": 0.0,
            "total_repair_time": 0.0,
            "total_write_time": 0.0,
            "total_batches": 0,
            "max_memory_mb": 0.0,
            "worker_throughputs": [],
            "worker_times": [],
        }

        def _accumulate_worker_result(result: dict[str, Any]) -> None:
            """Accumulate one worker result into global stats and perf aggregates."""
            self.stats["total_sentences"] += result["sentences"]
            self.stats["total_syllables"] += result["syllables"]
            self.stats["total_words"] += result["words"]

            if "perf" in result:
                perf = result["perf"]
                perf_aggregate["total_read_time"] += perf.get("time_read", 0)
                perf_aggregate["total_process_time"] += perf.get("time_process", 0)
                perf_aggregate["total_repair_time"] += perf.get("time_repair", 0)
                perf_aggregate["total_write_time"] += perf.get("time_write", 0)
                perf_aggregate["total_batches"] += perf.get("batches_processed", 0)
                perf_aggregate["max_memory_mb"] = max(
                    perf_aggregate["max_memory_mb"],
                    perf.get("memory_peak_mb", 0),
                )

        def _process_tasks_sequentially(
            tasks_to_process: list[dict[str, Any]],
            reason: str,
        ) -> None:
            """Process pending tasks sequentially in main process as a safe fallback."""
            if not tasks_to_process:
                return

            # Ensure worker state is initialized for direct worker_segment_file() calls.
            if _STATE.models_preloaded and _CAPABILITIES.use_fork_optimization:
                _STATE.worker_segmenter = _STATE.preloaded_segmenter
                _STATE.worker_repair = _STATE.preloaded_repair
            else:
                init_worker(self.word_engine, None, shared_counter)

            total_tasks = len(tasks_to_process)
            console.print(
                f"[yellow]⚠️ {reason}. Falling back to sequential processing "
                f"for {total_tasks} chunks.[/yellow]"
            )
            for i, task in enumerate(tasks_to_process):
                t_chunk_start = time.perf_counter()
                result = worker_segment_file(task)
                t_chunk_elapsed = time.perf_counter() - t_chunk_start
                _accumulate_worker_result(result)

                rate = result["sentences"] / t_chunk_elapsed if t_chunk_elapsed > 0 else 0
                console.print(
                    f"  [green]✓[/green] Chunk {i + 1}/{total_tasks}: "
                    f"{result['sentences']:,} sentences, {result['words']:,} words "
                    f"({rate:,.0f} sent/sec, {t_chunk_elapsed:.1f}s)"
                )

        def _is_process_restriction_error(err: BaseException) -> bool:
            """
            Detect sandbox/OS process limits (permission/semaphore restrictions).
            """
            cur: BaseException | None = err
            for _ in range(6):
                if cur is None:
                    break
                msg = str(cur).lower()
                if isinstance(cur, PermissionError):
                    return True
                if isinstance(cur, OSError) and cur.errno in {errno.EPERM, errno.EACCES}:
                    return True
                if any(
                    token in msg
                    for token in (
                        "operation not permitted",
                        "permission denied",
                        "semaphore",
                        "sem_open",
                        "resource_tracker",
                    )
                ):
                    return True
                cur = cur.__cause__ or cur.__context__
            return False

        # =====================================================================
        # TRANSFORMER ENGINE: Sequential processing in main process.
        # PyTorch/HuggingFace models don't survive fork() (C++ state corruption),
        # and loading a separate ~1.1GB model per spawn worker is impractical.
        # Process chunks sequentially using the already-loaded model instead.
        # After processing, tasks is cleared so the multiprocessing section
        # below is naturally skipped via its "if not tasks: break" check.
        # =====================================================================
        if self.word_engine == "transformer" and tasks:
            # Set up main process as the "worker"
            _STATE.worker_segmenter = _STATE.preloaded_segmenter
            _STATE.worker_repair = _STATE.preloaded_repair

            total_tasks = len(tasks)
            console.print(
                f"[cyan]Processing {total_tasks} chunks sequentially "
                f"(transformer model not multiprocessing-safe)[/cyan]"
            )

            for i, task in enumerate(tasks):
                t_chunk_start = time.perf_counter()
                result = worker_segment_file(task)
                t_chunk_elapsed = time.perf_counter() - t_chunk_start

                _accumulate_worker_result(result)

                rate = result["sentences"] / t_chunk_elapsed if t_chunk_elapsed > 0 else 0
                console.print(
                    f"  [green]✓[/green] Chunk {i + 1}/{total_tasks}: "
                    f"{result['sentences']:,} sentences, "
                    f"{result['words']:,} words "
                    f"({rate:,.0f} sent/sec, {t_chunk_elapsed:.1f}s)"
                )

            # Clear tasks so the multiprocessing section below is skipped
            tasks = []

        # Auto-retry configuration for crash recovery
        max_worker_retries = 4  # More retries to allow fork->spawn fallback
        current_workers = num_workers
        last_error = None
        use_fork_mode = _CAPABILITIES.use_fork_optimization and _STATE.models_preloaded

        for retry_attempt in range(max_worker_retries):
            # Rebuild tasks list based on what's already completed (for retry scenarios)
            if retry_attempt > 0:
                tasks = []
                for i, chunk_file in enumerate(files_to_process):
                    c_out = chunk_out_dir / f"chunk_{i}_segmented.arrow"
                    if c_out.exists() and c_out.stat().st_size > 0:
                        continue  # Already completed
                    tasks.append(
                        {
                            "input_file": str(chunk_file),
                            "output_dir": str(chunk_out_dir),
                            "chunk_id": i,
                        }
                    )
                # Reset shared counter for retry
                shared_counter = multiprocessing.Value("q", 0)
                # Reconfigure initargs with new counter
                if _CAPABILITIES.use_fork_optimization and _STATE.models_preloaded:
                    initargs = (shared_counter,)
                else:
                    initargs = (self.word_engine, None, shared_counter)

            if not tasks:
                self.logger.info("All chunks were already processed.")
                break

            try:
                # Create process pool with appropriate context
                # Reconfigure based on current mode (may have switched to spawn)
                if use_fork_mode:
                    current_mp_context = mp_context
                    current_initializer = initializer
                    current_initargs = initargs
                else:
                    # Spawn mode
                    current_mp_context = None
                    current_initializer = init_worker
                    current_initargs = (self.word_engine, None, shared_counter)

                if current_mp_context is not None:
                    executor = concurrent.futures.ProcessPoolExecutor(
                        max_workers=current_workers,
                        mp_context=current_mp_context,
                        initializer=current_initializer,  # type: ignore[arg-type]
                        initargs=current_initargs,  # type: ignore[arg-type]
                    )
                else:
                    executor = concurrent.futures.ProcessPoolExecutor(
                        max_workers=current_workers,
                        initializer=current_initializer,  # type: ignore[arg-type]
                        initargs=current_initargs,  # type: ignore[arg-type]
                    )

                with executor:
                    futures = [executor.submit(worker_segment_file, task) for task in tasks]

                    if futures:
                        # Track completion state
                        completed_chunks = 0
                        total_chunks = len(futures)
                        futures_done: set[concurrent.futures.Future[dict[str, Any]]] = set()

                        def create_progress_display():  # noqa: B023
                            """Create a Rich table showing real-time progress."""
                            current_count = shared_counter.value  # noqa: B023
                            elapsed = time.perf_counter() - pipeline_start_time
                            rate = current_count / elapsed if elapsed > 0 else 0
                            if total_sentences > 0:
                                pct = current_count / total_sentences * 100
                            else:
                                pct = 0

                            # Estimate remaining time
                            if rate > 0 and total_sentences > current_count:
                                eta_seconds = (total_sentences - current_count) / rate
                                eta_min = int(eta_seconds // 60)
                                eta_sec = int(eta_seconds % 60)
                                eta_str = f"{eta_min}m {eta_sec}s"
                            else:
                                eta_str = "--"

                            table = Table.grid(padding=(0, 2))
                            table.add_column(justify="left")
                            table.add_row(
                                f"[cyan]⠋ Segmenting...[/cyan] "
                                f"[bold]{completed_chunks}/{total_chunks}[/bold] chunks | "  # noqa: B023
                                f"[green]{current_count:,}[/green] / {total_sentences:,} sentences "
                                f"([yellow]{pct:.1f}%[/yellow]) | "
                                f"[blue]{rate:,.0f}[/blue] sent/sec | "
                                f"ETA: [magenta]{eta_str}[/magenta]"
                            )
                            return table

                        # Use Rich Live display for real-time progress
                        live_display = Live(
                            create_progress_display(), refresh_per_second=2, transient=True
                        )
                        with live_display as live:
                            last_progress_time = time.time()
                            last_completed = 0
                            stall_timeout = 1800  # 30 min stall timeout
                            while futures_done != set(futures):
                                # Check for completed futures (non-blocking)
                                done, _ = concurrent.futures.wait(
                                    [f for f in futures if f not in futures_done],
                                    timeout=0.5,
                                    return_when=concurrent.futures.FIRST_COMPLETED,
                                )
                                # Stall detection: if no progress for stall_timeout, warn
                                if completed_chunks > last_completed:
                                    last_progress_time = time.time()
                                    last_completed = completed_chunks
                                elif time.time() - last_progress_time > stall_timeout:
                                    remaining = len(futures) - len(futures_done)
                                    self.logger.warning(
                                        f"No progress for {stall_timeout}s. "
                                        f"{remaining} chunks still pending. "
                                        "Workers may be stuck."
                                    )
                                    last_progress_time = time.time()  # Reset to warn again later

                                for future in done:
                                    if future not in futures_done:
                                        futures_done.add(future)
                                        completed_chunks += 1

                                        try:
                                            res = future.result(timeout=self.worker_timeout)
                                        except concurrent.futures.TimeoutError:
                                            self.logger.error(
                                                "Worker timed out after %ds — chunk may be "
                                                "too large or worker hung",
                                                self.worker_timeout,
                                            )
                                            raise
                                        except (RuntimeError, MemoryError, OSError) as worker_err:
                                            # Log the actual worker error for debugging
                                            self.logger.error(
                                                f"Worker error (chunk {completed_chunks}): "
                                                f"{type(worker_err).__name__}: {worker_err}"
                                            )
                                            raise

                                        _accumulate_worker_result(res)

                                        # Aggregate per-worker derived metrics
                                        if "perf" in res:
                                            perf = res["perf"]
                                            if "throughput" in perf:
                                                cast(
                                                    list[float],
                                                    perf_aggregate["worker_throughputs"],
                                                ).append(perf["throughput"])
                                            if "total_time" in perf:
                                                cast(
                                                    list[float], perf_aggregate["worker_times"]
                                                ).append(perf["total_time"])

                                # Update display
                                live.update(create_progress_display())

                # Success - break out of retry loop
                break

            except (
                concurrent.futures.process.BrokenProcessPool,
                concurrent.futures.BrokenExecutor,
                RuntimeError,
                OSError,
            ) as e:
                last_error = e
                # Log detailed error info for debugging
                self.logger.error(f"Pool error: {type(e).__name__}: {e}")
                # Check how many workers we have left
                if current_workers <= 1:
                    if _is_process_restriction_error(e):
                        self.logger.warning(
                            "Process pool unavailable due to permission/resource limits. "
                            "Switching to sequential fallback."
                        )
                        _process_tasks_sequentially(
                            tasks,
                            reason="Process pool restricted by environment",
                        )
                        tasks = []
                        break

                    self.logger.error(
                        "Worker crash with only 1 worker - cannot reduce further. "
                        "Try reducing --batch-size or processing smaller corpus."
                    )
                    raise PipelineError(
                        "Segmentation failed even with 1 worker. "
                        "This indicates severe memory pressure. "
                        "Try: (1) reducing --batch-size, "
                        "(2) processing smaller corpus chunks, or "
                        "(3) running on a machine with more memory."
                    ) from e

                # Reduce workers by 50% and retry
                old_workers = current_workers
                current_workers = max(1, current_workers // 2)

                # After 2 failed attempts with fork, switch to spawn mode
                # (slower but more stable)
                if retry_attempt >= 2 and use_fork_mode:
                    use_fork_mode = False
                    # Reconfigure to spawn mode
                    mp_context = None
                    initializer = init_worker
                    console.print(
                        "[yellow]⚠️ Fork mode unstable. "
                        "Switching to spawn mode (slower but more stable)[/yellow]"
                    )
                    self.logger.warning(
                        "Fork mode crashed multiple times. Switching to spawn mode."
                    )

                attempt_str = f"{retry_attempt + 1}/{max_worker_retries}"
                console.print(
                    f"[yellow]⚠️ Worker crash detected (attempt {attempt_str}). "
                    f"Reducing workers: {old_workers} → {current_workers}[/yellow]"
                )
                self.logger.warning(
                    f"Crash recovery: Reducing workers from {old_workers} to {current_workers} "
                    f"(attempt {retry_attempt + 1}/{max_worker_retries})"
                )

                # Small delay before retry
                time.sleep(2.0)
        else:
            # All retries exhausted
            raise PipelineError(
                f"Segmentation failed after {max_worker_retries} attempts with worker reduction. "
                f"Final worker count was {current_workers}. "
                f"Try: (1) reducing --batch-size, or (2) processing smaller corpus."
            ) from last_error

        pipeline_end_time = time.perf_counter()
        pipeline_total_time = pipeline_end_time - pipeline_start_time

        # Log aggregated performance summary
        worker_throughputs = cast(list[float], perf_aggregate["worker_throughputs"])
        if worker_throughputs:
            avg_throughput = sum(worker_throughputs) / len(worker_throughputs)
            effective_throughput = (
                self.stats["total_sentences"] / pipeline_total_time
                if pipeline_total_time > 0
                else 0
            )
            theoretical_throughput = avg_throughput * current_workers

            self.logger.info("=" * 70)
            self.logger.info("SEGMENTATION PERFORMANCE SUMMARY")
            self.logger.info("=" * 70)
            self.logger.info(f"Total sentences:        {self.stats['total_sentences']:,}")
            self.logger.info(f"Pipeline time:          {pipeline_total_time:.2f}s")
            self.logger.info(f"Effective throughput:   {effective_throughput:,.0f} sent/sec")
            self.logger.info(f"Avg worker throughput:  {avg_throughput:,.0f} sent/sec")
            self.logger.info(f"Theoretical (parallel): {theoretical_throughput:,.0f} sent/sec")
            self.logger.info(
                f"Parallelization efficiency: "
                f"{(effective_throughput / theoretical_throughput * 100):.1f}%"
                if theoretical_throughput > 0
                else "N/A"
            )
            self.logger.info("-" * 70)
            self.logger.info("Cumulative worker time breakdown:")
            self.logger.info(f"  Read:    {perf_aggregate['total_read_time']:.2f}s")
            self.logger.info(f"  Process: {perf_aggregate['total_process_time']:.2f}s")
            self.logger.info(f"  Repair:  {perf_aggregate['total_repair_time']:.2f}s")
            self.logger.info(f"  Write:   {perf_aggregate['total_write_time']:.2f}s")
            self.logger.info(f"  Batches: {perf_aggregate['total_batches']}")
            self.logger.info(f"Peak memory (max worker): {perf_aggregate['max_memory_mb']:.0f}MB")
            self.logger.info("=" * 70)

        # 4. Merge Results
        self.logger.info("Merging chunk files...")

        # Merge all arrow chunks into one file
        # We can use a RecordBatchStreamWriter to merge streams
        try:
            merged_count = 0
            skipped_count = 0

            with pa.OSFile(str(sentences_path), "w") as sink:
                with pa.RecordBatchStreamWriter(sink, SEGMENT_SCHEMA) as writer:
                    for chunk_path in chunk_files:
                        if not chunk_path.exists():
                            continue

                        # Skip empty files (0 bytes = no valid Arrow data)
                        file_size = chunk_path.stat().st_size
                        if file_size == 0:
                            self.logger.warning(f"Skipping empty chunk: {chunk_path.name}")
                            chunk_path.unlink()
                            skipped_count += 1
                            continue

                        try:
                            with pa.memory_map(str(chunk_path), "r") as source:
                                reader = pa.ipc.open_stream(source)
                                for batch in reader:
                                    writer.write_batch(batch)
                            merged_count += 1
                        except (pa.ArrowIOError, pa.ArrowInvalid, OSError) as chunk_err:
                            self.logger.warning(
                                f"Skipping corrupted chunk {chunk_path.name}: {chunk_err}"
                            )
                            skipped_count += 1

                        # Clean up chunk after merge (success or skip)
                        if chunk_path.exists():
                            chunk_path.unlink()

            self.logger.info(f"Merged {merged_count} chunks ({skipped_count} skipped)")
        except (pa.ArrowIOError, pa.ArrowInvalid, OSError) as e:
            self.logger.error(f"Error merging files: {e}")
            raise

        # Clean chunk dir (non-critical, may fail if not empty)
        try:
            chunk_out_dir.rmdir()
        except OSError as e:
            self.logger.debug(f"Could not remove chunk directory {chunk_out_dir}: {e}")

        self.logger.info("Completed segmentation.")
        return sentences_path
