"""
Worker functions for parallel corpus segmentation.

Contains:
- preload_models(): Pre-load models in parent process before forking
- init_worker_fork(): Initialize worker when using fork-based multiprocessing
- init_worker(): Initialize worker (legacy/spawn mode)
- worker_segment_file(): Process a single chunk file
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pyarrow as pa  # type: ignore
from rich.console import Console
from rich.status import Status

from ..core.constants import DEFAULT_BATCH_SIZE, ENG_TOKEN
from ..core.exceptions import PipelineError, TokenizationError
from ..segmenters import DefaultSegmenter
from ..text.normalize import normalize
from ..text.validator import validate_word
from ..utils.logging_utils import get_logger
from . import _segmenter_config
from ._segmenter_config import (
    _CAPABILITIES,
    _STATE,
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_BASE_DELAY,
    SEGMENT_SCHEMA,
    _format_count,
    _is_pre_segmented,
    is_myanmar_token,
)
from .repair import SegmentationRepair

# Conditional imports from _segmenter_config (only available if capabilities exist)
if _CAPABILITIES.has_batch_processor:
    from ._segmenter_config import (
        bp_set_crf_engine,
        bp_set_crf_tagger,
        process_batch,
        process_batch_parallel,
    )

if _CAPABILITIES.has_cython_repair:
    from ._segmenter_config import c_init_repair, c_repair_batch


def preload_models(
    word_engine: str,
    custom_words: list[str] | None = None,
    console: Console | None = None,
    seg_model: str | None = None,
    seg_device: int = -1,
) -> None:
    """
    Pre-load models in the parent process before forking workers.

    On Unix systems (macOS/Linux), fork() creates workers that inherit
    the parent's memory space via copy-on-write. This means:
    - Models are loaded ONCE in the parent
    - Workers share the same memory pages for read-only data
    - Massive reduction in initialization time and memory usage

    Args:
        word_engine: Word segmentation engine ('crf', 'myword', 'transformer')
        custom_words: Optional list of custom words to add to dictionary
        console: Optional Rich console for status output
        seg_model: Custom model name for transformer engine (optional)
        seg_device: Device for transformer inference (-1=CPU, 0+=GPU)
    """
    logger = get_logger(__name__)

    if _STATE.models_preloaded:
        logger.debug("Models already preloaded, skipping.")
        return

    # Use provided console or create one
    _console = console or Console()

    with Status(
        f"[cyan]Loading segmentation model ({word_engine})...[/cyan]",
        console=_console,
        spinner="dots",
    ) as status:
        # Store transformer config for workers
        _STATE.seg_model = seg_model
        _STATE.seg_device = seg_device

        # Load segmenter (includes MyWord/CRF/Transformer model loading)
        _STATE.preloaded_segmenter = DefaultSegmenter(
            word_engine=word_engine,
            allow_extended_myanmar=_segmenter_config._allow_extended_myanmar,
            seg_model=seg_model,
            seg_device=seg_device,
        )

        # Load custom dictionary if provided
        if custom_words:
            status.update(f"[cyan]Loading {len(custom_words):,} custom words...[/cyan]")
            _STATE.preloaded_segmenter.load_custom_dictionary(custom_words)

        # CRITICAL: batch_processor.pyx uses Viterbi DIRECTLY for myword engine.
        # We MUST initialize the mmap reader for Viterbi to work correctly.
        # Without this, Viterbi returns individual characters (score 0.0).
        # SKIP for transformer engine: it uses its own model and bypasses Cython batch
        # processor entirely (falls back to Python loop).
        # SKIP for CRF engine: it uses pycrfsuite tagger, not Viterbi/mmap.
        if _CAPABILITIES.has_batch_processor and word_engine not in ("transformer", "crf"):
            status.update("[cyan]Initializing Viterbi mmap dictionary...[/cyan]")
            try:
                from myspellchecker.tokenizers.cython.word_segment import initialize_mmap
                from myspellchecker.tokenizers.resource_loader import get_segmentation_mmap_path

                mmap_path = get_segmentation_mmap_path()
                if not initialize_mmap(str(mmap_path)):
                    logger.warning("Failed to initialize mmap for Viterbi")
                else:
                    logger.debug(f"Mmap initialized from: {mmap_path}")
            except ImportError as e:
                logger.warning(f"Could not import mmap initialization: {e}")
            except (OSError, RuntimeError, TokenizationError) as e:
                logger.warning(f"Mmap initialization failed: {e}")

        if _CAPABILITIES.has_batch_processor and word_engine == "crf":
            status.update("[cyan]Initializing CRF engine in batch processor...[/cyan]")
            _STATE.preloaded_segmenter._ensure_word_segmenter_initialized()
            assert _STATE.preloaded_segmenter.word_tokenizer is not None
            crf_tagger = _STATE.preloaded_segmenter.word_tokenizer.tagger
            bp_set_crf_tagger(crf_tagger)
            bp_set_crf_engine(True)
            logger.debug("CRF tagger passed to batch_processor")

        # Warm up the segmentation engine to ensure model state is initialized
        engine_label = {"transformer": "transformer", "crf": "CRF"}.get(word_engine, "Viterbi")
        status.update(f"[cyan]Warming up {engine_label} segmentation...[/cyan]")
        try:
            warmup_text = "မြန်မာ"  # Simple Myanmar word
            _STATE.preloaded_segmenter.segment_words(warmup_text)
        except (RuntimeError, ValueError) as e:
            logger.warning(f"{engine_label} warmup failed: {e}")

        # Load repair module
        status.update("[cyan]Initializing repair module...[/cyan]")
        _STATE.preloaded_repair = SegmentationRepair()

        # Also initialize Cython repair if available
        if _CAPABILITIES.has_cython_repair:
            c_init_repair()

    _STATE.models_preloaded = True

    try:
        from myspellchecker.tokenizers.cython.mmap_reader import get_mmap_reader

        mmap_reader = get_mmap_reader()
        if mmap_reader:
            stats = mmap_reader.get_stats()
            unigram_str = _format_count(stats.get("unigram_count", 0))
            bigram_str = _format_count(stats.get("bigram_count", 0))
            _console.print(
                f"[green]✓[/green] Model loaded: {unigram_str} unigrams, "
                f"{bigram_str} bigrams (mmap, fork-safe)"
            )
    except (ImportError, AttributeError, RuntimeError):
        _console.print(f"[green]✓[/green] Model loaded ({word_engine})")

    # Log OpenMP status (debug level - not user-facing)
    if _CAPABILITIES.has_openmp:
        if _CAPABILITIES.use_fork_optimization:
            logger.debug(
                "OpenMP available but DISABLED in fork mode. "
                "OpenMP + fork causes libomp conflicts on macOS."
            )
        else:
            logger.debug(
                f"OpenMP parallelization ENABLED "
                f"({_CAPABILITIES.openmp_threads_per_worker} threads/worker)."
            )
    else:
        logger.debug("OpenMP parallelization NOT available.")


def init_worker_fork(shared_counter=None):
    """
    Initialize worker when using fork-based multiprocessing.

    Workers inherit preloaded models from parent process via copy-on-write.

    For mmap-based word segmentation:
    - The memory-mapped file survives fork automatically via COW
    - No reinitialization needed - just verify it's ready

    For C++ map-based word segmentation (legacy):
    - C++ unordered_map objects may become invalid after fork
    - Need to reinitialize from Python ProbDist objects

    NOTE: OpenMP is disabled in forked workers to avoid "libomp already initialized"
    error on macOS. The fork-based parallelism (12 workers) provides sufficient
    parallelization without needing intra-worker OpenMP threads.

    Args:
        shared_counter: Optional multiprocessing.Value for progress tracking.
    """
    # Store shared counter for progress updates
    if shared_counter is not None:
        _STATE.shared_progress_counter = shared_counter

    # Mark this as a forked worker - disables OpenMP to avoid libomp conflict
    _STATE.is_forked_worker = True

    logger = get_logger(__name__)

    # Simply reference the preloaded models
    # On fork, these are already in memory via copy-on-write
    _STATE.worker_segmenter = _STATE.preloaded_segmenter
    _STATE.worker_repair = _STATE.preloaded_repair

    # Verify models are available
    if _STATE.worker_segmenter is None:
        raise PipelineError(
            "Fork worker initialization failed: models not preloaded. "
            "Call preload_models() before creating worker pool."
        )

    # For transformer engine, skip all Viterbi/mmap verification.
    # Transformer uses HuggingFace pipeline (Python-level) which survives fork via COW.
    # It does NOT use Viterbi C++ maps, so verifying/reinitializing them would fail.
    if _STATE.worker_segmenter.word_engine == "transformer":
        logger.debug("Worker: Transformer engine - skipping Viterbi/mmap verification.")
        return

    # For CRF engine, the tagger in batch_processor module survives fork via COW.
    # Just verify with a quick sanity test.
    if _STATE.worker_segmenter.word_engine == "crf":
        from myspellchecker.data_pipeline.batch_processor import get_crf_tagger

        tagger = get_crf_tagger()
        if tagger is not None:
            test_feats = [{"BOS": True, "EOS": True, "number": False}]
            tagger.tag(test_feats)  # Quick verify
        logger.debug("Worker: CRF engine - tagger intact (fork-safe via COW).")
        return

    # Check if using mmap (fast path - no reinitialization needed)
    try:
        from myspellchecker.tokenizers.cython.mmap_reader import (
            ensure_mmap_initialized,
        )
        from myspellchecker.tokenizers.cython.word_segment import is_using_mmap

        if is_using_mmap():
            # Mmap survives fork via COW - just verify it's ready
            if ensure_mmap_initialized():
                logger.debug("Worker: Mmap reader intact (fork-safe via COW).")
                return
            else:
                logger.warning("Worker: Mmap reader lost after fork, falling back to C++ maps...")
    except ImportError:
        pass  # Mmap module not available

    # Fallback: Verify and reinitialize C++ maps if needed
    # C++ unordered_map objects don't survive fork properly (internal pointers become invalid)
    # But the Python ProbDist objects DO survive via COW, so we can repopulate the C++ maps
    try:
        from myspellchecker.tokenizers.cython.word_segment import (
            ensure_models_initialized,
            get_model_stats,
        )

        # Check current state
        stats = get_model_stats()

        if stats["unigram_map_size"] == 0 or stats["bigram_map_size"] == 0:
            logger.info(
                f"Worker: C++ maps empty after fork (unigram={stats['unigram_map_size']}, "
                f"bigram={stats['bigram_map_size']}). Reinitializing from Python objects..."
            )

            # Reinitialize C++ maps from Python ProbDist objects
            if ensure_models_initialized():
                # Verify success
                new_stats = get_model_stats()
                logger.info(
                    f"Worker: C++ maps reinitialized successfully "
                    f"(unigram={new_stats['unigram_map_size']}, "
                    f"bigram={new_stats['bigram_map_size']})"
                )
            else:
                logger.error(
                    "Worker: Failed to reinitialize C++ maps! Viterbi will be slow/broken."
                )
        else:
            logger.debug(
                f"Worker: C++ maps intact (unigram={stats['unigram_map_size']}, "
                f"bigram={stats['bigram_map_size']})"
            )

    except ImportError:
        pass  # Pure Python fallback - no Cython module


def init_worker(word_engine: str, custom_words: list[str] | None = None, shared_counter=None):
    """
    Initialize the worker process (legacy/spawn mode).

    This runs ONCE per worker process, not per task.
    Used when fork optimization is not available (Windows) or disabled.

    Args:
        word_engine: The word segmentation engine to use.
        custom_words: Optional list of custom words to add to dictionary.
        shared_counter: Optional multiprocessing.Value for progress tracking.
    """
    # Store shared counter for progress updates
    if shared_counter is not None:
        _STATE.shared_progress_counter = shared_counter

    # Check if we can use preloaded models (fork mode)
    if _STATE.models_preloaded and _CAPABILITIES.use_fork_optimization:
        init_worker_fork(shared_counter)
        return

    _STATE.worker_segmenter = DefaultSegmenter(
        word_engine=word_engine,
        allow_extended_myanmar=_segmenter_config._allow_extended_myanmar,
        seg_model=_STATE.seg_model,
        seg_device=_STATE.seg_device,
    )
    _STATE.worker_repair = SegmentationRepair()

    # CRITICAL: batch_processor.pyx uses Viterbi DIRECTLY for myword engine.
    # We MUST initialize the mmap reader for Viterbi to work correctly.
    # SKIP for transformer engine: it uses its own model and bypasses Cython batch processor.
    # SKIP for CRF engine: it uses pycrfsuite tagger, not Viterbi/mmap.
    if _CAPABILITIES.has_batch_processor and word_engine not in ("transformer", "crf"):
        try:
            from myspellchecker.tokenizers.cython.word_segment import initialize_mmap
            from myspellchecker.tokenizers.resource_loader import get_segmentation_mmap_path

            mmap_path = get_segmentation_mmap_path()
            if not initialize_mmap(str(mmap_path)):
                get_logger(__name__).warning("Worker: Failed to initialize mmap for Viterbi")
            else:
                get_logger(__name__).debug(f"Worker: Mmap initialized from: {mmap_path}")
        except ImportError as e:
            get_logger(__name__).warning(f"Worker: Could not import mmap initialization: {e}")
        except (OSError, RuntimeError) as e:
            get_logger(__name__).warning(f"Worker: Mmap initialization failed: {e}")

    if _CAPABILITIES.has_batch_processor and word_engine == "crf":
        _STATE.worker_segmenter._ensure_word_segmenter_initialized()
        assert _STATE.worker_segmenter.word_tokenizer is not None
        crf_tagger = _STATE.worker_segmenter.word_tokenizer.tagger
        bp_set_crf_tagger(crf_tagger)
        bp_set_crf_engine(True)
        get_logger(__name__).debug("Worker: CRF tagger initialized in batch_processor")

    # Load custom dictionary if provided
    if custom_words:
        get_logger(__name__).info(
            f"Worker: Loading {len(custom_words)} custom words into dictionary."
        )
        _STATE.worker_segmenter.load_custom_dictionary(custom_words)


# Worker function for parallel processing
def worker_segment_file(args: dict) -> dict:
    """
    Process a single chunk file (Worker function).
    Reads chunk file (Arrow), segments, and writes outputs to chunk-specific Arrow files.

    Implements automatic retry with exponential backoff for transient failures.
    """
    import os

    input_file = Path(args["input_file"])
    output_dir = Path(args["output_dir"])
    chunk_id = args["chunk_id"]
    max_retries = args.get("max_retries", DEFAULT_MAX_RETRIES)
    retry_base_delay = args.get("retry_base_delay", DEFAULT_RETRY_BASE_DELAY)

    # Use the pre-initialized process state
    if _STATE.worker_segmenter is None or _STATE.worker_repair is None:
        raise PipelineError("Worker segmenter/repair not initialized!")

    segmenter = _STATE.worker_segmenter
    repairer = _STATE.worker_repair

    # Define final output paths
    out_file = output_dir / f"chunk_{chunk_id}_segmented.arrow"

    stats: dict[str, Any] = {"sentences": 0, "syllables": 0, "words": 0}

    # Performance tracking
    perf_stats = {
        "time_read": 0.0,
        "time_process": 0.0,
        "time_repair": 0.0,
        "time_write": 0.0,
        "batches_processed": 0,
        "memory_start_mb": 0.0,
        "memory_peak_mb": 0.0,
    }

    try:
        import psutil

        process = psutil.Process(os.getpid())
        perf_stats["memory_start_mb"] = process.memory_info().rss / (1024 * 1024)
    except ImportError:
        pass  # psutil not available

    worker_start_time = time.perf_counter()

    # Buffers for columnar writing
    # structure: {col: [values]}
    batch_buffer: dict[str, list] = {
        "text": [],
        "source": [],
        "syllables": [],
        "words": [],
        "syllable_count": [],
        "word_count": [],
    }

    logger = get_logger(__name__)
    last_error = None

    for attempt in range(max_retries):
        try:
            # Open output stream
            with pa.OSFile(str(out_file), "w") as sink:
                with pa.RecordBatchStreamWriter(sink, SEGMENT_SCHEMA) as writer:
                    # Open input stream - MUST keep mmap open while iterating reader
                    t_read_start = time.perf_counter()
                    with pa.memory_map(str(input_file), "r") as source:
                        reader = pa.ipc.open_stream(source)
                        perf_stats["time_read"] += time.perf_counter() - t_read_start

                        for batch in reader:
                            # OPTIMIZATION: Use Arrow arrays directly instead of to_pydict()
                            # This avoids massive python object creation overhead
                            text_col = batch.column("text")
                            source_col = batch.column("source")

                            # Use simple iteration which is fast enough for string arrays in Arrow
                            # or convert to python list which is faster than to_pydict for single
                            # cols
                            text_list = text_col.to_pylist()
                            source_list = source_col.to_pylist()

                            use_cython_batch = (
                                _CAPABILITIES.has_batch_processor
                                and segmenter.word_engine != "transformer"
                            )
                            if use_cython_batch:
                                # Use optimized Cython batch processor
                                # This loop runs entirely in C++, avoiding Python interpreter
                                # overhead per sentence
                                # NOTE: Cython batch processor dispatches to CRF or Viterbi
                                # based on engine config. Transformer is excluded (own path).
                                t_process_start = time.perf_counter()
                                if _CAPABILITIES.has_openmp and not _STATE.is_forked_worker:
                                    # Use OpenMP-parallelized batch processor
                                    # Each worker spawns openmp_threads_per_worker threads
                                    # NOTE: Disabled in forked workers to avoid libomp conflict on
                                    # macOS
                                    batch_results = process_batch_parallel(
                                        text_list,
                                        source_list,
                                        _CAPABILITIES.openmp_threads_per_worker,
                                    )
                                else:
                                    # Use sequential batch processor
                                    # This is used in forked workers (fork provides parallelism)
                                    # or when OpenMP is not available
                                    batch_results = process_batch(text_list, source_list)
                                perf_stats["time_process"] += time.perf_counter() - t_process_start

                                # REPAIR STEP: Fix broken syllables in batch results
                                # Use Cython repair if available for additional speedup
                                t_repair_start = time.perf_counter()
                                if _CAPABILITIES.has_cython_repair:
                                    # Batch repair: process all word lists in one call
                                    repaired_words_batch = c_repair_batch(batch_results["words"])
                                else:
                                    # Fallback to Python repair
                                    repaired_words_batch = []
                                    for words in batch_results["words"]:
                                        repaired_words_batch.append(repairer.repair(words))
                                perf_stats["time_repair"] += time.perf_counter() - t_repair_start

                                batch_results["words"] = repaired_words_batch

                                # Re-calculate word counts based on repaired words
                                batch_results["word_count"] = [len(w) for w in repaired_words_batch]

                                # Update stats
                                batch_sentence_count = len(batch_results["text"])
                                stats["sentences"] += batch_sentence_count
                                stats["syllables"] += sum(batch_results["syllable_count"])
                                stats["words"] += sum(batch_results["word_count"])

                                # Update shared progress counter for real-time display
                                if _STATE.shared_progress_counter is not None:
                                    with _STATE.shared_progress_counter.get_lock():
                                        _STATE.shared_progress_counter.value += batch_sentence_count

                                # Write batch
                                t_write_start = time.perf_counter()
                                out_batch = pa.RecordBatch.from_pydict(
                                    batch_results, schema=SEGMENT_SCHEMA
                                )
                                writer.write_batch(out_batch)
                                perf_stats["time_write"] += time.perf_counter() - t_write_start

                                # Track batch count and memory
                                perf_stats["batches_processed"] += 1
                                try:
                                    import psutil

                                    process = psutil.Process(os.getpid())
                                    mem_current = process.memory_info().rss / (1024 * 1024)
                                    perf_stats["memory_peak_mb"] = max(
                                        perf_stats["memory_peak_mb"], mem_current
                                    )
                                except ImportError:
                                    pass

                            elif (
                                segmenter.word_engine == "transformer"
                                and hasattr(segmenter, "_transformer_segmenter")
                                and segmenter._transformer_segmenter is not None
                            ):
                                # =============================================
                                # TRANSFORMER BATCH PATH
                                # Uses segment_batch() for 10-20x speedup over
                                # per-sentence processing on CPU.
                                # =============================================
                                t_process_start = time.perf_counter()
                                ext_mm = _segmenter_config._allow_extended_myanmar
                                transformer_seg = segmenter._transformer_segmenter

                                # Step 1: Normalize and pre-process all sentences
                                norm_sentences: list[str] = []
                                norm_sources = []
                                sentence_parts = []  # list of ENG_TOKEN-split parts
                                all_parts = []  # flat list of Myanmar parts
                                part_map = []  # (sent_idx, part_idx) for mapping back
                                # Track pre-segmented parts to skip transformer
                                pre_seg_result_map: dict[int, dict[int, list]] = {}

                                for sentence, src in zip(text_list, source_list, strict=False):
                                    if not sentence:
                                        continue
                                    sentence = normalize(sentence)
                                    parts = sentence.split(ENG_TOKEN)
                                    sent_idx = len(norm_sentences)
                                    norm_sentences.append(sentence)
                                    norm_sources.append(src)
                                    sentence_parts.append(parts)
                                    for p_idx, part in enumerate(parts):
                                        stripped = part.strip()
                                        if stripped:
                                            # Check for pre-segmented input before
                                            # sending to transformer
                                            if (
                                                _segmenter_config._auto_detect_pre_segmented
                                                and _is_pre_segmented(stripped)
                                            ):
                                                # Pre-segmented: store tokens directly,
                                                # skip transformer
                                                pre_seg_result_map.setdefault(sent_idx, {})[
                                                    p_idx
                                                ] = stripped.split()
                                            else:
                                                all_parts.append(stripped)
                                                part_map.append((sent_idx, p_idx))

                                # Step 2: Batch word segmentation (single call)
                                if all_parts:
                                    all_word_results = transformer_seg.segment_batch(all_parts)
                                else:
                                    all_word_results = []

                                # Step 3: Map batch results back to sentences
                                word_result_map: dict[int, dict[int, list]] = {}
                                for (s_idx, p_idx), w_res in zip(
                                    part_map, all_word_results, strict=False
                                ):
                                    word_result_map.setdefault(s_idx, {})[p_idx] = w_res
                                # Merge pre-segmented results into the word result map
                                for s_idx, parts_map in pre_seg_result_map.items():
                                    for p_idx, tokens in parts_map.items():
                                        word_result_map.setdefault(s_idx, {})[p_idx] = tokens

                                perf_stats["time_process"] += time.perf_counter() - t_process_start

                                # Step 4: Assemble results with syllables and filtering
                                batch_results_t: dict[str, list] = {
                                    "text": [],
                                    "source": [],
                                    "syllables": [],
                                    "words": [],
                                    "syllable_count": [],
                                    "word_count": [],
                                }

                                for s_idx, (sentence, src) in enumerate(
                                    zip(norm_sentences, norm_sources, strict=False)
                                ):
                                    parts = sentence_parts[s_idx]
                                    syllables = []
                                    words = []

                                    for p_idx, part in enumerate(parts):
                                        stripped = part.strip()
                                        if stripped:
                                            # Syllables (fast, regex-based)
                                            s_part = segmenter.segment_syllables(stripped)
                                            s_part = [
                                                s
                                                for s in s_part
                                                if is_myanmar_token(s, allow_extended=ext_mm)
                                                and validate_word(s, allow_extended_myanmar=ext_mm)
                                            ]
                                            syllables.extend(s_part)

                                            # Words from batch results
                                            w_part = word_result_map.get(s_idx, {}).get(p_idx, [])
                                            w_part = [
                                                w
                                                for w in w_part
                                                if is_myanmar_token(w, allow_extended=ext_mm)
                                                and validate_word(w, allow_extended_myanmar=ext_mm)
                                            ]
                                            w_part = repairer.repair(w_part)
                                            words.extend(w_part)

                                        # Restore ENG_TOKEN
                                        if p_idx < len(parts) - 1:
                                            words.append(ENG_TOKEN)

                                    batch_results_t["text"].append(sentence)
                                    batch_results_t["source"].append(src)
                                    batch_results_t["syllables"].append(syllables)
                                    batch_results_t["words"].append(words)
                                    batch_results_t["syllable_count"].append(len(syllables))
                                    batch_results_t["word_count"].append(len(words))

                                # Update stats
                                batch_sentence_count = len(batch_results_t["text"])
                                stats["sentences"] += batch_sentence_count
                                stats["syllables"] += sum(batch_results_t["syllable_count"])
                                stats["words"] += sum(batch_results_t["word_count"])

                                # Update shared progress counter
                                if _STATE.shared_progress_counter is not None:
                                    with _STATE.shared_progress_counter.get_lock():
                                        _STATE.shared_progress_counter.value += batch_sentence_count

                                # Write batch
                                t_write_start = time.perf_counter()
                                out_batch = pa.RecordBatch.from_pydict(
                                    batch_results_t, schema=SEGMENT_SCHEMA
                                )
                                writer.write_batch(out_batch)
                                perf_stats["time_write"] += time.perf_counter() - t_write_start

                                perf_stats["batches_processed"] += 1
                                try:
                                    import psutil

                                    process = psutil.Process(os.getpid())
                                    mem_current = process.memory_info().rss / (1024 * 1024)
                                    perf_stats["memory_peak_mb"] = max(
                                        perf_stats["memory_peak_mb"], mem_current
                                    )
                                except ImportError:
                                    pass

                            else:
                                # Fallback to Python loop (CRF/myword without Cython)
                                for sentence, src in zip(text_list, source_list, strict=False):
                                    if not sentence:
                                        continue

                                    # Normalize ONCE here.
                                    # RegexSegmenter no longer normalizes internally.
                                    sentence = normalize(sentence)

                                    # Split by ENG_TOKEN logic
                                    parts = sentence.split(ENG_TOKEN)
                                    syllables = []
                                    words = []

                                    for i, part in enumerate(parts):
                                        if part.strip():
                                            # OPTIMIZATION: Removed redundant normalize calls
                                            # We trust the segmenter returns valid tokens if input
                                            # is valid

                                            # Syllables
                                            s_part = segmenter.segment_syllables(part)
                                            # Filter: Myanmar check + text validation
                                            # Uses module-level config for extended chars
                                            ext_mm = _segmenter_config._allow_extended_myanmar
                                            s_part = [
                                                s
                                                for s in s_part
                                                if is_myanmar_token(s, allow_extended=ext_mm)
                                                and validate_word(s, allow_extended_myanmar=ext_mm)
                                            ]
                                            syllables.extend(s_part)

                                            # Words — check for pre-segmented input first
                                            if (
                                                _segmenter_config._auto_detect_pre_segmented
                                                and _is_pre_segmented(part)
                                            ):
                                                # Pre-segmented: use space-split tokens directly
                                                w_part = [
                                                    w
                                                    for w in part.split()
                                                    if w.strip()
                                                    and is_myanmar_token(w, allow_extended=ext_mm)
                                                    and validate_word(
                                                        w, allow_extended_myanmar=ext_mm
                                                    )
                                                ]
                                            else:
                                                w_part = segmenter.segment_words(part)
                                                # Filter: Myanmar check + text validation
                                                w_part = [
                                                    w
                                                    for w in w_part
                                                    if is_myanmar_token(w, allow_extended=ext_mm)
                                                    and validate_word(
                                                        w, allow_extended_myanmar=ext_mm
                                                    )
                                                ]

                                            # REPAIR STEP: Fix broken syllables
                                            w_part = repairer.repair(w_part)

                                            words.extend(w_part)

                                        # Restore ENG_TOKEN
                                        if i < len(parts) - 1:
                                            words.append(ENG_TOKEN)

                                    batch_buffer["text"].append(sentence)
                                    batch_buffer["source"].append(src)
                                    batch_buffer["syllables"].append(syllables)
                                    batch_buffer["words"].append(words)
                                    batch_buffer["syllable_count"].append(len(syllables))
                                    batch_buffer["word_count"].append(len(words))

                                    stats["sentences"] += 1
                                    stats["syllables"] += len(syllables)
                                    stats["words"] += len(words)

                                    # Flush buffer if full
                                    if len(batch_buffer["text"]) >= DEFAULT_BATCH_SIZE:
                                        flush_count = len(batch_buffer["text"])
                                        out_batch = pa.RecordBatch.from_pydict(
                                            batch_buffer, schema=SEGMENT_SCHEMA
                                        )
                                        writer.write_batch(out_batch)
                                        # Update shared progress counter
                                        if _STATE.shared_progress_counter is not None:
                                            with _STATE.shared_progress_counter.get_lock():
                                                _STATE.shared_progress_counter.value += flush_count
                                        for k in batch_buffer:
                                            batch_buffer[k].clear()

                                # Flush remaining items (fallback)
                                if batch_buffer["text"]:
                                    flush_count = len(batch_buffer["text"])
                                    out_batch = pa.RecordBatch.from_pydict(
                                        batch_buffer, schema=SEGMENT_SCHEMA
                                    )
                                    writer.write_batch(out_batch)
                                    # Update shared progress counter
                                    if _STATE.shared_progress_counter is not None:
                                        with _STATE.shared_progress_counter.get_lock():
                                            _STATE.shared_progress_counter.value += flush_count
                                    for k in batch_buffer:
                                        batch_buffer[k].clear()

            # Success - calculate final performance metrics and log
            worker_total_time = time.perf_counter() - worker_start_time
            throughput = stats["sentences"] / worker_total_time if worker_total_time > 0 else 0

            # Log performance statistics (DEBUG to avoid interfering with Rich progress)
            logger.debug(
                f"Worker chunk {chunk_id} completed: "
                f"{stats['sentences']:,} sentences in {worker_total_time:.2f}s "
                f"({throughput:,.0f} sent/sec)"
            )
            logger.debug(
                f"Worker chunk {chunk_id} timing breakdown: "
                f"read={perf_stats['time_read']:.2f}s, "
                f"process={perf_stats['time_process']:.2f}s, "
                f"repair={perf_stats['time_repair']:.2f}s, "
                f"write={perf_stats['time_write']:.2f}s, "
                f"batches={perf_stats['batches_processed']}"
            )
            logger.debug(
                f"Worker chunk {chunk_id} memory: "
                f"start={perf_stats['memory_start_mb']:.0f}MB, "
                f"peak={perf_stats['memory_peak_mb']:.0f}MB, "
                f"delta={perf_stats['memory_peak_mb'] - perf_stats['memory_start_mb']:.0f}MB"
            )

            # Add perf_stats to return value for aggregation
            stats["perf"] = perf_stats
            stats["perf"]["total_time"] = worker_total_time
            stats["perf"]["throughput"] = throughput

            return stats

        except (pa.ArrowIOError, pa.ArrowInvalid, OSError, MemoryError, RuntimeError) as e:
            last_error = e
            # Clean up partial output file
            if out_file.exists():
                try:
                    out_file.unlink()
                except OSError:
                    pass

            # Check if we should retry
            if attempt < max_retries - 1:
                delay = retry_base_delay * (2**attempt)  # Exponential backoff
                logger.warning(
                    f"Worker chunk {chunk_id} failed (attempt {attempt + 1}/{max_retries}): {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
                # Reset stats for retry
                stats = {"sentences": 0, "syllables": 0, "words": 0}
            else:
                logger.error(f"Worker chunk {chunk_id} failed after {max_retries} attempts: {e}")

    # All retries exhausted
    raise PipelineError(
        f"Worker failed to process chunk {chunk_id} after {max_retries} attempts"
    ) from last_error
