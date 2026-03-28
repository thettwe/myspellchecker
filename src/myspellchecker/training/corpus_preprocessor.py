"""
Corpus preprocessor for cleaning raw text before pipeline ingestion.

This module provides streaming line-by-line preprocessing for large corpora
(50GB+), handling Zawgyi detection/conversion, Unicode normalization,
quality filtering, deduplication, and register detection.

Supports parallel processing for large files via ``process_file_parallel()``,
which splits files into byte-range chunks and processes them concurrently
with ``multiprocessing.Pool``.

Reuses existing utilities:
- text/zawgyi_support.py: Zawgyi detection and conversion
- text/normalize.py: Myanmar text normalization
- text/normalize_c.pyx: get_myanmar_ratio() via Cython

Example:
    >>> from myspellchecker.training.corpus_preprocessor import (
    ...     CorpusPreprocessor,
    ...     PreprocessingConfig,
    ... )
    >>> config = PreprocessingConfig(min_myanmar_ratio=0.6)
    >>> preprocessor = CorpusPreprocessor(config)
    >>> report = preprocessor.process_file("raw_corpus.txt", "clean_corpus.txt")
    >>> print(f"Kept {report.kept}/{report.total} lines")

    Parallel processing for large files:
    >>> report = preprocessor.process_file_parallel(
    ...     "large_corpus.txt", "clean_corpus.txt", num_workers=4
    ... )
"""

from __future__ import annotations

import multiprocessing
import os
import tempfile
from dataclasses import dataclass, fields
from pathlib import Path

from myspellchecker.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Default threshold: files >= 100 MiB get chunk-parallel processing
_PARALLEL_SIZE_THRESHOLD = 100 * 1024 * 1024  # 100 MiB

# Formal register particles
_FORMAL_PARTICLES = {"သည်", "ပါသည်", "၏", "ခဲ့သည်", "မည်", "လိမ့်မည်"}
# Colloquial register particles
_COLLOQUIAL_PARTICLES = {"တယ်", "ပါတယ်", "ဘူး", "တာ", "မယ်", "လိမ့်မယ်"}


@dataclass
class PreprocessingConfig:
    """Configuration for corpus preprocessing.

    Attributes:
        zawgyi_threshold: Probability above which text is treated as Zawgyi
            and converted to Unicode (default: 0.7).
        zawgyi_ambiguous_low: Probability below which text is treated as
            Unicode (default: 0.3). Lines between ambiguous_low and
            zawgyi_threshold are discarded as ambiguous.
        min_myanmar_ratio: Minimum ratio of Myanmar characters in a line
            for it to be kept (default: 0.6).
        min_line_length: Minimum line length in characters (default: 10).
        max_line_length: Maximum line length in characters (default: 1000).
        enable_dedup: Enable line-level deduplication via hash set (default: True).
            Uses ~8 bytes per line; ~800MB for 100M unique lines.
    """

    zawgyi_threshold: float = 0.7
    zawgyi_ambiguous_low: float = 0.3
    min_myanmar_ratio: float = 0.6
    min_line_length: int = 10
    max_line_length: int = 1000
    enable_dedup: bool = True


@dataclass
class PreprocessingReport:
    """Counters for corpus preprocessing results.

    Attributes:
        total: Total lines read from input.
        kept: Lines written to output.
        zawgyi_converted: Lines converted from Zawgyi to Unicode.
        zawgyi_discarded: Lines discarded as ambiguous Zawgyi.
        filtered_empty: Lines discarded because empty after stripping.
        filtered_length: Lines discarded for length (too short or too long).
        filtered_myanmar_ratio: Lines discarded for low Myanmar character ratio.
        duplicates: Lines discarded as duplicates.
        formal_count: Lines containing formal register particles.
        colloquial_count: Lines containing colloquial register particles.
    """

    total: int = 0
    kept: int = 0
    zawgyi_converted: int = 0
    zawgyi_discarded: int = 0
    filtered_empty: int = 0
    filtered_length: int = 0
    filtered_myanmar_ratio: int = 0
    duplicates: int = 0
    formal_count: int = 0
    colloquial_count: int = 0

    def summary(self) -> str:
        """Return a human-readable summary of the report."""
        discarded = self.total - self.kept
        pct = (self.kept / self.total * 100) if self.total > 0 else 0
        return (
            f"Preprocessed {self.total:,} lines: kept {self.kept:,} ({pct:.1f}%), "
            f"discarded {discarded:,} "
            f"(empty={self.filtered_empty:,}, length={self.filtered_length:,}, "
            f"myanmar_ratio={self.filtered_myanmar_ratio:,}, "
            f"zawgyi_ambiguous={self.zawgyi_discarded:,}, "
            f"duplicates={self.duplicates:,}), "
            f"zawgyi_converted={self.zawgyi_converted:,}, "
            f"register: formal={self.formal_count:,} colloquial={self.colloquial_count:,}"
        )

    def merge(self, other: "PreprocessingReport") -> None:
        """Add all counters from *other* into this report (in-place)."""
        for f in fields(self):
            setattr(self, f.name, getattr(self, f.name) + getattr(other, f.name))


def _get_chunk_boundaries(file_path: str | Path, num_chunks: int) -> list[int]:
    """Split a file into *num_chunks* byte ranges aligned to line boundaries.

    Returns a list of ``num_chunks + 1`` byte offsets.  Chunk *i* covers
    bytes ``[boundaries[i], boundaries[i+1])``.
    """
    file_size = os.path.getsize(file_path)
    if num_chunks <= 1 or file_size == 0:
        return [0, file_size]

    target_chunk = file_size // num_chunks
    boundaries = [0]

    with open(file_path, "rb") as f:
        for i in range(1, num_chunks):
            f.seek(i * target_chunk)
            f.readline()  # advance past partial line to next \n boundary
            pos = f.tell()
            # Avoid duplicate boundaries when file is small relative to chunks
            if pos >= file_size:
                break
            if pos != boundaries[-1]:
                boundaries.append(pos)

    boundaries.append(file_size)
    return boundaries


def _process_chunk(args: tuple) -> tuple[str, "PreprocessingReport"]:
    """Worker function: process a byte range of a file and write to temp file.

    Designed as a top-level function so it is compatible with multiprocessing
    (multiprocessing requires top-level callables for its process pool).

    Args:
        args: Tuple of (file_path, start_byte, end_byte, config_dict, temp_path).

    Returns:
        Tuple of (temp_path, PreprocessingReport).
    """
    file_path, start_byte, end_byte, config_dict, temp_path = args

    config = PreprocessingConfig(**config_dict)
    preprocessor = CorpusPreprocessor(config)
    preprocessor._init_zawgyi()

    report = PreprocessingReport()
    log_interval = 1_000_000

    with (
        open(file_path, "rb") as fin,
        open(temp_path, "w", encoding="utf-8") as fout,
    ):
        fin.seek(start_byte)
        while fin.tell() < end_byte:
            raw = fin.readline()
            if not raw:
                break
            line = raw.decode("utf-8", errors="replace")
            report.total += 1
            result = preprocessor.process_line(line, report)
            if result is not None:
                fout.write(result + "\n")
                report.kept += 1

            if report.total % log_interval == 0:
                logger.info(
                    "Chunk %s: processed %d lines, kept %d",
                    temp_path,
                    report.total,
                    report.kept,
                )

    return temp_path, report


class CorpusPreprocessor:
    """Streaming line-by-line corpus preprocessor.

    Processes raw corpus text through:
    1. Empty line filtering
    2. Zawgyi detection and conversion (or discard if ambiguous)
    3. Unicode normalization (NFC + zero-width removal + diacritic reordering)
    4. Quality filtering (Myanmar ratio, line length)
    5. Deduplication (hash-based)
    6. Register detection (informational)

    The preprocessor streams line-by-line, never accumulating full file content
    in memory (except the dedup hash set).

    Args:
        config: Preprocessing configuration. Uses defaults if not provided.

    Example:
        >>> preprocessor = CorpusPreprocessor()
        >>> report = preprocessor.process_file("input.txt", "output.txt")
        >>> print(report.summary())
    """

    def __init__(self, config: PreprocessingConfig | None = None):
        self.config = config or PreprocessingConfig()
        self._seen_hashes: set[int] = set()
        self._zawgyi_detector = None
        self._zawgyi_converter_available = False

    def _init_zawgyi(self) -> None:
        """Lazily initialize Zawgyi detection/conversion utilities."""
        from myspellchecker.text.zawgyi_support import (
            get_zawgyi_detector,
            is_zawgyi_converter_available,
        )

        self._zawgyi_detector = get_zawgyi_detector()
        self._zawgyi_converter_available = is_zawgyi_converter_available()

    def process_line(self, line: str, report: PreprocessingReport) -> str | None:
        """Process a single line through the preprocessing pipeline.

        Args:
            line: Raw input line (may include trailing newline).
            report: Report object to update counters on.

        Returns:
            Cleaned line if it passes all filters, None otherwise.
        """
        # 1. Strip whitespace and skip empty
        line = line.strip()
        if not line:
            report.filtered_empty += 1
            return None

        # 2. Zawgyi detection and handling
        if self._zawgyi_detector is not None:
            prob = self._zawgyi_detector.get_zawgyi_probability(line)
            if prob > self.config.zawgyi_threshold:
                # High confidence Zawgyi → convert
                if self._zawgyi_converter_available:
                    from myspellchecker.text.zawgyi_support import convert_zawgyi_to_unicode

                    line = convert_zawgyi_to_unicode(line)
                    report.zawgyi_converted += 1
                else:
                    # Can't convert without converter, discard
                    report.zawgyi_discarded += 1
                    return None
            elif prob > self.config.zawgyi_ambiguous_low:
                # Ambiguous range → discard
                report.zawgyi_discarded += 1
                return None
            # else: prob <= ambiguous_low → treat as Unicode, keep

        # 3. Normalize
        from myspellchecker.text.normalize import normalize

        line = normalize(
            line,
            remove_zero_width=True,
            reorder_diacritics=True,
            normalize_variants=True,
        )

        # 4. Quality checks
        if len(line) < self.config.min_line_length or len(line) > self.config.max_line_length:
            report.filtered_length += 1
            return None

        # Myanmar ratio check
        from myspellchecker.text.normalize_c import get_myanmar_ratio as c_get_myanmar_ratio

        ratio = c_get_myanmar_ratio(line)
        if ratio < self.config.min_myanmar_ratio:
            report.filtered_myanmar_ratio += 1
            return None

        # 5. Deduplication
        if self.config.enable_dedup:
            line_hash = hash(line)
            if line_hash in self._seen_hashes:
                report.duplicates += 1
                return None
            self._seen_hashes.add(line_hash)

        # 6. Register detection (informational only)
        if any(p in line for p in _FORMAL_PARTICLES):
            report.formal_count += 1
        if any(p in line for p in _COLLOQUIAL_PARTICLES):
            report.colloquial_count += 1

        return line

    def process_file(
        self,
        input_path: str | Path,
        output_path: str | Path,
    ) -> PreprocessingReport:
        """Process a corpus file line-by-line, writing cleaned output.

        Streams input and output simultaneously to minimize memory usage.
        Only the dedup hash set accumulates in memory.

        Args:
            input_path: Path to raw input corpus file.
            output_path: Path to write cleaned output.

        Returns:
            PreprocessingReport with processing statistics.
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize Zawgyi support lazily
        self._init_zawgyi()

        # Reset dedup set for fresh processing
        self._seen_hashes = set()

        report = PreprocessingReport()
        log_interval = 1_000_000

        with (
            open(input_path, encoding="utf-8") as fin,
            open(output_path, "w", encoding="utf-8") as fout,
        ):
            for line in fin:
                report.total += 1
                result = self.process_line(line, report)
                if result is not None:
                    fout.write(result + "\n")
                    report.kept += 1

                if report.total % log_interval == 0:
                    logger.info(
                        "Processed %d lines: kept %d, discarded %d",
                        report.total,
                        report.kept,
                        report.total - report.kept,
                    )

        logger.info(report.summary())
        return report

    def process_file_parallel(
        self,
        input_path: str | Path,
        output_path: str | Path,
        num_workers: int | None = None,
    ) -> PreprocessingReport:
        """Process a corpus file using multiple workers for large files.

        For files smaller than 100 MiB, delegates to the sequential
        ``process_file()`` method.  For larger files the input is split into
        byte-range chunks (one per worker), each chunk is processed in a
        separate process, and the results are concatenated.

        Deduplication is **per-chunk** (each worker maintains its own hash
        set).  A post-merge dedup pass over the concatenated output removes
        any cross-chunk duplicates.

        Args:
            input_path: Path to raw input corpus file.
            output_path: Path to write cleaned output.
            num_workers: Number of parallel workers.
                ``None`` (default) uses ``os.cpu_count()``.
                ``0`` also auto-detects.
                ``1`` forces sequential processing.

        Returns:
            PreprocessingReport with processing statistics.
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if num_workers is None or num_workers == 0:
            num_workers = os.cpu_count() or 1

        file_size = os.path.getsize(input_path)

        # Fall back to sequential for small files or single worker
        if num_workers == 1 or file_size < _PARALLEL_SIZE_THRESHOLD:
            return self.process_file(input_path, output_path)

        logger.info(
            "Parallel processing %s (%d MiB) with %d workers",
            input_path.name,
            file_size // (1024 * 1024),
            num_workers,
        )

        boundaries = _get_chunk_boundaries(input_path, num_workers)
        actual_chunks = len(boundaries) - 1

        # Serializable config dict (dataclass → dict for multiprocessing)
        config_dict = {
            "zawgyi_threshold": self.config.zawgyi_threshold,
            "zawgyi_ambiguous_low": self.config.zawgyi_ambiguous_low,
            "min_myanmar_ratio": self.config.min_myanmar_ratio,
            "min_line_length": self.config.min_line_length,
            "max_line_length": self.config.max_line_length,
            "enable_dedup": self.config.enable_dedup,
        }

        # Create temp files for each chunk output
        tmp_dir = tempfile.mkdtemp(prefix="preprocess_")
        chunk_args = []
        for i in range(actual_chunks):
            tmp_path = os.path.join(tmp_dir, f"chunk_{i:04d}.txt")
            chunk_args.append(
                (
                    str(input_path),
                    boundaries[i],
                    boundaries[i + 1],
                    config_dict,
                    tmp_path,
                )
            )

        # Process chunks in parallel
        merged_report = PreprocessingReport()
        temp_paths = []

        try:
            with multiprocessing.Pool(processes=actual_chunks) as pool:
                results = pool.map(_process_chunk, chunk_args)

            for tmp_path, chunk_report in results:
                temp_paths.append(tmp_path)
                merged_report.merge(chunk_report)

            # Concatenate chunk outputs → final file
            # If dedup is enabled, do a single-pass cross-chunk dedup during concat
            seen_hashes: set[int] = set()
            dedup_removed = 0
            with open(output_path, "w", encoding="utf-8") as fout:
                for tmp_path in temp_paths:
                    with open(tmp_path, encoding="utf-8") as fin:
                        for line in fin:
                            if self.config.enable_dedup:
                                h = hash(line.rstrip("\n"))
                                if h in seen_hashes:
                                    dedup_removed += 1
                                    continue
                                seen_hashes.add(h)
                            fout.write(line)

            # Adjust report for cross-chunk dedup
            if dedup_removed > 0:
                merged_report.duplicates += dedup_removed
                merged_report.kept -= dedup_removed

            logger.info(
                "Parallel complete: %d chunks, cross-chunk dedup removed %d lines",
                actual_chunks,
                dedup_removed,
            )

        finally:
            # Clean up temp files
            import shutil

            try:
                shutil.rmtree(tmp_dir)
            except OSError as e:
                logger.warning("Failed to clean up temp dir %s: %s", tmp_dir, e)

        logger.info(merged_report.summary())
        return merged_report
