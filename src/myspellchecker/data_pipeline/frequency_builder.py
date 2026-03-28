"""
Frequency builder module for the data pipeline.

Uses DuckDB for fast SQL-based frequency counting, supporting large corpora
without loading n-gram data into Python memory.
"""

from __future__ import annotations

import re
import shutil
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rich.console import Console

import pyarrow as pa  # type: ignore

from ..core.constants import (
    INVALID_WORDS,
    get_myanmar_char_set,  # For scope-aware char validation
)

try:
    import duckdb
except ImportError:
    duckdb = None  # type: ignore[assignment]

from ..core.syllable_rules import SyllableRuleValidator
from ..segmenters import DefaultSegmenter
from ..text.normalize import normalize
from ..text.validator import is_quality_word, validate_word
from ..utils.logging_utils import get_logger
from .config import runtime_flags as _flags
from .frequency_checkpoint import _NGRAM_BUCKET_COUNT, CheckpointMixin
from .frequency_io import FrequencyIOMixin
from .frequency_pos import POSProcessorMixin
from .pipeline_config_unified import DuckDBResourceConfig

__all__ = [
    "DuckDBNgramStore",
    "FrequencyBuilder",
    "set_allow_extended_myanmar",
]

# Module-level logger for standalone functions
logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Shared runtime flags (replaces per-module globals)
# ---------------------------------------------------------------------------

# Pre-compiled rejection patterns for filter_invalid_syllables
_REJECT_DIGITS_ONLY = re.compile(r"^[\u1040-\u1049]+$")
_REJECT_PUNCT_ONLY = re.compile(r"^[\u104A\u104B]+$")
_REJECT_ASCII = re.compile(r"[\u0020-\u007F]")
_REJECT_MODIFIERS_ONLY = re.compile(r"^[\u1039\u103A\u1037\u103B-\u103E]+$")
_REJECT_ZERO_WIDTH = re.compile(r"[\u200B-\u200D\uFEFF]")
_REJECT_REPEATED = re.compile(r"^(.)\1{2,}$")
_REJECT_SINGLE_CONSONANT = re.compile(r"^[\u1000-\u1021]$")

_SYLLABLE_REJECT_PATTERNS = [
    _REJECT_DIGITS_ONLY,
    _REJECT_PUNCT_ONLY,
    _REJECT_ASCII,
    _REJECT_MODIFIERS_ONLY,
    _REJECT_ZERO_WIDTH,
    _REJECT_REPEATED,
    _REJECT_SINGLE_CONSONANT,
]


def set_allow_extended_myanmar(allow: bool) -> None:
    """Configure Extended Myanmar character handling for frequency builder.

    .. deprecated::
        Set ``PipelineConfig.allow_extended_myanmar`` instead and pass the
        config to ``Pipeline``.  This function will be removed in a future
        release.
    """
    import warnings

    warnings.warn(
        "set_allow_extended_myanmar() is deprecated. "
        "Use PipelineConfig.allow_extended_myanmar instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    _flags.allow_extended_myanmar = allow


# SQL injection prevention for DuckDB queries
# Note: '/' is excluded because it's a valid path separator
_SQL_DANGEROUS_CHARS = frozenset("'\"\\;")
_SQL_DANGEROUS_PATTERNS = ("--", "/*", "*/")


def _validate_path_for_sql(path: Path) -> str:
    """
    Validate a path is safe to use in SQL SET commands.

    Args:
        path: Path to validate

    Returns:
        String representation of the path

    Raises:
        ValueError: If path contains SQL metacharacters or dangerous patterns
    """
    path_str = str(path)
    # Check for dangerous individual characters
    if any(c in path_str for c in _SQL_DANGEROUS_CHARS):
        raise ValueError(f"Path contains unsafe characters for SQL: {path_str}")
    # Check for dangerous patterns (SQL comments)
    if any(pattern in path_str for pattern in _SQL_DANGEROUS_PATTERNS):
        raise ValueError(f"Path contains unsafe patterns for SQL: {path_str}")
    return path_str


def _escape_sql_string(value: str) -> str:
    """
    Escape a string for safe use in SQL queries.

    Args:
        value: String to escape

    Returns:
        Escaped string with single quotes doubled
    """
    return value.replace("'", "''")


class DuckDBNgramStore:
    """Disk-backed n-gram storage using DuckDB persistent tables.

    Keeps bigram and trigram counts on disk. All filtering,
    probability calculation, and TSV export happen in SQL.
    This avoids loading hundreds of millions of n-gram entries
    into Python memory, preventing OOM on large corpora.
    """

    _VALID_TABLES = frozenset(
        {
            "bigram_counts",
            "trigram_counts",
            "ngram_windows",
            "quality_words",
        }
    )

    def __init__(self, conn: "duckdb.DuckDBPyConnection", logger: Any) -> None:
        self.conn = conn
        self.logger = logger
        # Stored for cleanup by FrequencyBuilder
        self.duckdb_temp_dir: Path | None = None

    def register_quality_words(self, words: set[str]) -> None:
        """Insert quality word set into a DuckDB table for SQL JOINs.

        Uses PyArrow table registration for fast bulk loading.
        """
        word_list = list(words)
        arrow_table = pa.table({"word": pa.array(word_list, type=pa.utf8())})
        self.conn.register("_quality_words_arrow", arrow_table)
        self.conn.execute(
            "CREATE OR REPLACE TABLE quality_words AS SELECT word FROM _quality_words_arrow"
        )
        self.conn.unregister("_quality_words_arrow")
        self.logger.info("Registered %d quality words in DuckDB", len(word_list))

    def build_bigram_table(self) -> None:
        """Create bigram_counts table from ngram_windows with quality word filter."""
        self.conn.execute("""
            CREATE TABLE bigram_counts AS
            SELECT n.w1, n.w2, COUNT(*) AS cnt
            FROM ngram_windows n
            INNER JOIN quality_words q1 ON n.w1 = q1.word
            INNER JOIN quality_words q2 ON n.w2 = q2.word
            WHERE n.w2 IS NOT NULL
            GROUP BY n.w1, n.w2
        """)

    def build_trigram_table(self, trigram_mem_limit_gb: int) -> None:
        """Create trigram_counts table from ngram_windows with quality word filter."""
        import os

        trigram_threads = max(2, int(os.cpu_count() or 8) // 2)
        self.conn.execute(f"SET memory_limit = '{int(trigram_mem_limit_gb)}GB'")
        self.conn.execute(f"SET threads TO {int(trigram_threads)}")

        self.conn.execute("""
            CREATE TABLE trigram_counts AS
            SELECT n.w1, n.w2, n.w3, COUNT(*) AS cnt
            FROM ngram_windows n
            INNER JOIN quality_words q1 ON n.w1 = q1.word
            INNER JOIN quality_words q2 ON n.w2 = q2.word
            INNER JOIN quality_words q3 ON n.w3 = q3.word
            WHERE n.w2 IS NOT NULL AND n.w3 IS NOT NULL
            GROUP BY n.w1, n.w2, n.w3
        """)

        self.conn.execute(f"SET threads TO {int(os.cpu_count() or 8)}")

    def filter_bigrams(self, min_count: int) -> tuple[int, int]:
        """Delete bigrams below min_count. Returns (original, remaining)."""
        row = self.conn.execute("SELECT COUNT(*) FROM bigram_counts").fetchone()
        original = int(row[0]) if row else 0
        self.conn.execute("DELETE FROM bigram_counts WHERE cnt < ?", [min_count])
        row = self.conn.execute("SELECT COUNT(*) FROM bigram_counts").fetchone()
        remaining = int(row[0]) if row else 0
        return original, remaining

    def filter_trigrams(self, min_count: int) -> tuple[int, int]:
        """Delete trigrams below min_count. Returns (original, remaining)."""
        row = self.conn.execute("SELECT COUNT(*) FROM trigram_counts").fetchone()
        original = int(row[0]) if row else 0
        self.conn.execute("DELETE FROM trigram_counts WHERE cnt < ?", [min_count])
        row = self.conn.execute("SELECT COUNT(*) FROM trigram_counts").fetchone()
        remaining = int(row[0]) if row else 0
        return original, remaining

    def save_bigram_probabilities(self, output_path: Path) -> int:
        """Calculate probabilities and export bigram TSV via DuckDB COPY TO.

        Returns the number of entries written.
        """
        path_safe = _validate_path_for_sql(output_path)
        row = self.conn.execute("SELECT COUNT(*) FROM bigram_counts").fetchone()
        count = int(row[0]) if row else 0
        self.conn.execute(f"""
            COPY (
                SELECT
                    b.w1 AS word1,
                    b.w2 AS word2,
                    ROUND(CAST(b.cnt AS DOUBLE) / w1t.total, 6) AS probability,
                    b.cnt AS count
                FROM bigram_counts b
                JOIN (
                    SELECT w1, SUM(cnt) AS total
                    FROM bigram_counts
                    GROUP BY w1
                ) w1t ON b.w1 = w1t.w1
            ) TO '{path_safe}' (DELIMITER '\t', HEADER)
        """)
        return count

    def save_trigram_probabilities(self, output_path: Path) -> int:
        """Calculate probabilities and export trigram TSV via DuckDB COPY TO.

        Returns the number of entries written.
        """
        path_safe = _validate_path_for_sql(output_path)
        row = self.conn.execute("SELECT COUNT(*) FROM trigram_counts").fetchone()
        count = int(row[0]) if row else 0
        self.conn.execute(f"""
            COPY (
                SELECT
                    t.w1 AS word1,
                    t.w2 AS word2,
                    t.w3 AS word3,
                    ROUND(CAST(t.cnt AS DOUBLE) / bt.total, 6) AS probability,
                    t.cnt AS count
                FROM trigram_counts t
                JOIN (
                    SELECT w1, w2, SUM(cnt) AS total
                    FROM trigram_counts
                    GROUP BY w1, w2
                ) bt ON t.w1 = bt.w1 AND t.w2 = bt.w2
            ) TO '{path_safe}' (DELIMITER '\t', HEADER)
        """)
        return count

    def get_stats(self) -> dict:
        """Get aggregate stats from DuckDB tables."""
        stats = {}
        if self.has_table("bigram_counts"):
            row = self.conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(cnt), 0) FROM bigram_counts"
            ).fetchone()
            if row is not None:
                stats["unique_bigrams"] = row[0]
                stats["total_bigrams"] = row[1]
            else:
                stats["unique_bigrams"] = 0
                stats["total_bigrams"] = 0
        else:
            stats["unique_bigrams"] = 0
            stats["total_bigrams"] = 0

        if self.has_table("trigram_counts"):
            row = self.conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(cnt), 0) FROM trigram_counts"
            ).fetchone()
            if row is not None:
                stats["unique_trigrams"] = row[0]
                stats["total_trigrams"] = row[1]
            else:
                stats["unique_trigrams"] = 0
                stats["total_trigrams"] = 0
        else:
            stats["unique_trigrams"] = 0
            stats["total_trigrams"] = 0

        return stats

    def has_table(self, table_name: str) -> bool:
        """Check if a table exists in the DuckDB database."""
        result = self.conn.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
            [table_name],
        ).fetchone()
        if result is None:
            return False
        return bool(result[0] > 0)

    def get_count(self, table_name: str) -> int:
        """Get row count for progress reporting."""
        if table_name not in self._VALID_TABLES:
            raise ValueError(f"Invalid table name: {table_name}")
        row = self.conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
        return int(row[0]) if row else 0

    def close(self) -> None:
        """Close the DuckDB connection."""
        try:
            self.conn.close()
        except OSError:
            pass


class FrequencyBuilder(CheckpointMixin, POSProcessorMixin, FrequencyIOMixin):
    """Builds frequency tables from segmented corpus data."""

    # Frequency threshold defaults matching PipelineConfig values.
    # Defined independently to avoid fragile cross-module coupling —
    # if PipelineConfig defaults change, update these to match.
    _DEFAULT_MIN_SYLLABLE_FREQ: int = 1
    _DEFAULT_MIN_WORD_FREQ: int = 50
    _DEFAULT_MIN_BIGRAM_FREQ: int = 10
    _DEFAULT_MIN_TRIGRAM_FREQ: int = 20

    def __init__(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        min_syllable_frequency: int | None = None,
        min_word_frequency: int | None = None,
        min_bigram_frequency: int | None = None,
        min_trigram_frequency: int | None = None,
        pos_tagger: POSTaggerBase | None = None,  # type: ignore  # noqa: F821
        incremental: bool = False,
        word_engine: str = "myword",
        pos_checkpoint_interval: int = 250_000,
        pos_batch_buffer_size: int = 2000,
        duckdb_resources: DuckDBResourceConfig | None = None,
        **kwargs: Any,
    ):
        """
        Initialize the frequency builder.

        Args:
            input_dir: Directory containing segmented corpus files
            output_dir: Directory to save frequency tables
            min_syllable_frequency: Min frequency for syllables
                (default from PipelineConfig: 1)
            min_word_frequency: Min frequency for words
                (default from PipelineConfig: 50)
            min_bigram_frequency: Min frequency for bigrams
                (default from PipelineConfig: 10)
            min_trigram_frequency: Min frequency for trigrams
                (default from PipelineConfig: 20)
            pos_tagger: POS tagger instance for dynamic tagging.
            incremental: Whether running in incremental mode
            word_engine: Word segmentation engine (default: "myword")
            pos_checkpoint_interval: Save POS checkpoint every N
                sentences (default: 250000)
            pos_batch_buffer_size: Sentences buffered before POS batch
                (default: 2000)
            duckdb_resources: DuckDB resource limits (uses defaults
                if None)
            **kwargs: Accepted for backward compat (num_workers
                is ignored).
        """
        if duckdb is None:
            raise ImportError(
                "DuckDB is required for the data pipeline. Install with: pip install duckdb"
            )

        self.logger = get_logger(__name__)
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Frequency thresholds — fall back to PipelineConfig defaults
        self.min_syllable_frequency = (
            min_syllable_frequency
            if min_syllable_frequency is not None
            else self._DEFAULT_MIN_SYLLABLE_FREQ
        )
        self.min_word_frequency = (
            min_word_frequency if min_word_frequency is not None else self._DEFAULT_MIN_WORD_FREQ
        )
        self.min_bigram_frequency = (
            min_bigram_frequency
            if min_bigram_frequency is not None
            else self._DEFAULT_MIN_BIGRAM_FREQ
        )
        self.min_trigram_frequency = (
            min_trigram_frequency
            if min_trigram_frequency is not None
            else self._DEFAULT_MIN_TRIGRAM_FREQ
        )
        self.incremental = incremental

        # DuckDB resource configuration
        self._duckdb_resources = (
            duckdb_resources if duckdb_resources is not None else DuckDBResourceConfig()
        )

        # POS checkpoint/batch settings
        self._pos_checkpoint_interval = pos_checkpoint_interval
        self._pos_batch_buffer_size = pos_batch_buffer_size

        # POS Tagger configuration
        self.pos_tagger = pos_tagger

        # Initialize segmenter for accurate syllable counting
        self.segmenter = DefaultSegmenter(
            word_engine=word_engine,
            allow_extended_myanmar=_flags.allow_extended_myanmar,
        )

        # Initialize validator for hygiene check
        self.validator = SyllableRuleValidator(allow_extended_myanmar=_flags.allow_extended_myanmar)

        # Disk-backed n-gram store (set during DuckDB load_data)
        self._duckdb_ngram_store: DuckDBNgramStore | None = None

        # Frequency counters (unigrams — bigrams/trigrams are in DuckDB tables)
        self.syllable_counts: Counter = Counter()
        self.word_counts: Counter = Counter()

        # Word syllable counts
        self.word_syllables: dict[str, int] = {}

        # POS-related counts for Phase 2
        self.pos_bigram_counts: Counter = Counter()  # (pos1, pos2) -> count
        self.pos_unigram_counts: Counter = Counter()  # pos -> count
        self.pos_bigram_predecessor_counts: Counter = (
            Counter()
        )  # pos1 -> count (sum of all bigrams starting with pos1)
        self.pos_trigram_counts: Counter = Counter()  # (pos1, pos2, pos3) -> count
        self.pos_bigram_successor_counts: Counter = (
            Counter()
        )  # (pos1, pos2) -> count (sum of all trigrams starting with pos1, pos2)

        # Word → POS tag mapping (populated when pos_tagger is used)
        # Stores the most frequent POS tag for each word from transformer tagging
        self.word_pos_tags: dict[str, Counter] = {}  # word -> Counter of POS tags

        # Sub-step timings for pipeline summary (populated by load_data)
        self.sub_step_timings: list[tuple[str, float]] = []

        # Statistics
        self.stats = {
            "total_syllables": 0,
            "total_words": 0,
            "total_bigrams": 0,
            "total_trigrams": 0,
            "unique_syllables": 0,
            "unique_words": 0,
            "unique_bigrams": 0,
            "unique_trigrams": 0,
            "filtered_syllables": 0,
            "filtered_words": 0,
            "filtered_bigrams": 0,
            "filtered_trigrams": 0,
            "invalid_syllables_skipped": 0,
            "total_pos_unigrams": 0,
            "unique_pos_unigrams": 0,
            "total_pos_bigrams": 0,
            "unique_pos_bigrams": 0,
            "total_pos_trigrams": 0,
            "unique_pos_trigrams": 0,
        }

    def reset(self) -> None:
        """
        Reset all counters to initial state.

        This method should be called before each load_data() invocation if the same
        FrequencyBuilder instance is reused for multiple corpus files. Without reset,
        counts from previous runs will accumulate, leading to incorrect statistics.

        Example:
            >>> builder = FrequencyBuilder(input_dir="data", output_dir="output")
            >>> builder.load_data(Path("corpus1.arrow"))
            >>> builder.reset()  # Clear previous counts
            >>> builder.load_data(Path("corpus2.arrow"))  # Start fresh
        """
        self._duckdb_ngram_store = None
        self.syllable_counts = Counter()
        self.word_counts = Counter()
        self.word_syllables = {}
        self.word_pos_tags = {}
        self.pos_bigram_counts = Counter()
        self.pos_unigram_counts = Counter()
        self.pos_bigram_predecessor_counts = Counter()
        self.pos_trigram_counts = Counter()
        self.pos_bigram_successor_counts = Counter()
        self.stats = {
            "total_syllables": 0,
            "total_words": 0,
            "total_bigrams": 0,
            "total_trigrams": 0,
            "unique_syllables": 0,
            "unique_words": 0,
            "unique_bigrams": 0,
            "unique_trigrams": 0,
            "filtered_syllables": 0,
            "filtered_words": 0,
            "filtered_bigrams": 0,
            "filtered_trigrams": 0,
            "invalid_syllables_skipped": 0,
            "total_pos_unigrams": 0,
            "unique_pos_unigrams": 0,
            "total_pos_bigrams": 0,
            "unique_pos_bigrams": 0,
            "total_pos_trigrams": 0,
            "unique_pos_trigrams": 0,
        }

        self.logger.debug("FrequencyBuilder counters reset to initial state")

    def cleanup_duckdb(self) -> None:
        """Close DuckDB connection and clean up temp directory.

        Must be called after all save methods complete when using the
        disk-backed n-gram store. Safe to call multiple times or when
        no store is active.
        """
        if self._duckdb_ngram_store is not None:
            temp_dir = self._duckdb_ngram_store.duckdb_temp_dir
            self._duckdb_ngram_store.close()
            self._duckdb_ngram_store = None
            if temp_dir and temp_dir.exists():
                try:
                    shutil.rmtree(temp_dir)
                except OSError as e:
                    self.logger.warning(
                        "Failed to clean up DuckDB temp directory %s: %s", temp_dir, e
                    )
                else:
                    self.logger.info("Cleaned up DuckDB temp directory: %s", temp_dir)

    def filter_invalid_syllables(self) -> None:
        """
        Filter out invalid syllables from the counts.
        This is crucial when using C++ counters which might skip validation for speed.

        Uses both:
        - SyllableRuleValidator for structural validation
        - validate_word for pattern-based validation (Zawgyi, extended Myanmar, etc.)
        - Additional pattern checks for structurally invalid sequences
        """
        self.logger.info("Filtering invalid syllables...")
        invalid_syllables = []

        # Use scope-aware character set for non-Myanmar character detection
        myanmar_range = get_myanmar_char_set(_flags.allow_extended_myanmar)

        for syllable in list(self.syllable_counts.keys()):
            is_invalid = False

            # Check 1: Use existing validators
            if not self.validator.validate(syllable) or not validate_word(
                syllable, allow_extended_myanmar=_flags.allow_extended_myanmar
            ):
                is_invalid = True

            # Check 2: Pattern-based detection for structurally invalid sequences
            if not is_invalid:
                for pattern in _SYLLABLE_REJECT_PATTERNS:
                    if pattern.search(syllable):
                        is_invalid = True
                        break

            # Check 3: Non-Myanmar characters
            if not is_invalid:
                for char in syllable:
                    if char not in myanmar_range and not char.isspace():
                        is_invalid = True
                        break

            if is_invalid:
                invalid_syllables.append(syllable)

        for syl in invalid_syllables:
            count = self.syllable_counts.pop(syl)
            self.stats["invalid_syllables_skipped"] += count

        self.logger.info(f"  Removed {len(invalid_syllables):,} invalid unique syllables")

    def filter_invalid_words(self) -> None:
        """
        Filter out invalid words from the counts.

        Uses validate_word for pattern-based validation to catch:
        - Zawgyi encoding artifacts
        - Extended Myanmar characters (U+1050-U+109F)
        - Invalid character ordering
        - Other contamination patterns
        """
        self.logger.info("Filtering invalid words...")
        invalid_words = []
        invalid_word_count = 0

        for word in list(self.word_counts.keys()):
            if not validate_word(word, allow_extended_myanmar=_flags.allow_extended_myanmar):
                invalid_words.append(word)

        for word in invalid_words:
            count = self.word_counts.pop(word)
            invalid_word_count += count
            # Also remove from word_syllables if present
            if word in self.word_syllables:
                del self.word_syllables[word]

        self.logger.info(
            f"  Removed {len(invalid_words):,} invalid unique words "
            f"(total count: {invalid_word_count:,})"
        )

    def load_data(
        self,
        filename: str = "segmented_corpus.arrow",
        **kwargs: Any,
    ) -> None:
        """
        Load sentence data from Arrow file and build all frequency tables.

        Uses DuckDB for fast SQL-based aggregation.

        Args:
            filename: Arrow IPC stream file name (default: segmented_corpus.arrow)
            **kwargs: Accepted for backward compatibility (parallel, use_duckdb ignored).
        """
        # Reset all counters to ensure clean state
        self.reset()

        filepath = self.input_dir / filename
        file_size = filepath.stat().st_size
        file_size_gb = file_size / (1024**3)

        self.logger.info(f"Streaming data from: {filepath} (DUCKDB mode - {file_size_gb:.1f} GB)")
        self._load_data_duckdb(filepath)

    def _load_data_duckdb(self, filepath: Path) -> None:
        """
        Ultra-fast frequency counting using DuckDB with Arrow table registration.

        OPTIMIZED VERSION (Phase 1): Reads Arrow file with PyArrow memory-mapping,
        registers with DuckDB, and runs single-pass SQL queries. Eliminates Python
        loop overhead and per-batch SQL parsing.

        Performance: ~10-12x faster than per-batch processing.
        - Old: 33GB in ~3 hours (per-batch, 5.6M SQL queries)
        - New: 33GB in ~15-20 minutes (4 SQL queries total)

        Args:
            filepath: Path to the Arrow IPC stream file
        """
        import os
        import time

        from rich.console import Console

        console = Console()
        file_size = filepath.stat().st_size
        file_size_gb = file_size / (1024**3)

        console.print(f"[bold blue]DuckDB Direct Arrow Query[/bold blue] ({file_size_gb:.1f} GB)")
        start_time = time.time()

        # Build invalid words filter for SQL (with proper escaping)
        invalid_words_sql = ", ".join(f"'{_escape_sql_string(w)}'" for w in INVALID_WORDS)

        # ================================================================
        # STEP 0: Setup DuckDB with disk-based temp storage
        # ================================================================
        # Create temp directory in work dir (not /tmp which has limited space on macOS)
        duckdb_temp_dir = filepath.parent / ".duckdb_temp"
        duckdb_temp_dir.mkdir(exist_ok=True)
        database_path = str(duckdb_temp_dir / "freq_build.duckdb")
        conn = duckdb.connect(database_path)

        # Optimize DuckDB settings - use disk for temp storage
        # Dynamically size memory based on system RAM using
        # DuckDBResourceConfig thresholds.
        db_cfg = self._duckdb_resources
        try:
            total_ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024**3)
        except (ValueError, OSError):
            total_ram_gb = 16.0  # Conservative fallback
        mem_limit_gb = int(
            max(
                db_cfg.memory_min_gb,
                min(db_cfg.memory_max_gb, total_ram_gb * db_cfg.memory_ratio),
            )
        )
        trigram_mem_limit_gb = int(
            max(
                db_cfg.trigram_memory_min_gb,
                min(
                    db_cfg.trigram_memory_max_gb,
                    total_ram_gb * db_cfg.trigram_memory_ratio,
                ),
            )
        )
        logger.info(
            "System RAM: %.0f GB -> DuckDB memory_limit=%dGB (trigram=%dGB)",
            total_ram_gb,
            mem_limit_gb,
            trigram_mem_limit_gb,
        )

        num_threads = int(os.cpu_count() or 8)  # Ensure integer
        # Isolate DuckDB spill files in a subdirectory so cleanup on
        # reconnect doesn't wipe checkpoint files (corpus.parquet, etc.)
        spill_dir = duckdb_temp_dir / "spill"
        spill_dir.mkdir(exist_ok=True)
        spill_dir_safe = _validate_path_for_sql(spill_dir)
        conn.execute(f"SET threads TO {int(num_threads)}")
        conn.execute(f"SET memory_limit = '{int(mem_limit_gb)}GB'")
        flush_mb = db_cfg.allocator_flush_threshold_mb
        conn.execute(f"SET allocator_flush_threshold = '{int(flush_mb)}MB'")
        conn.execute(f"SET temp_directory = '{spill_dir_safe}'")
        disk_usage = shutil.disk_usage(spill_dir)
        max_temp_gb = max(
            100,
            int(disk_usage.free * db_cfg.disk_usage_cap_ratio / (1024**3)),
        )
        logger.info(
            "DuckDB spill dir: %s (%.0f GB free, limit set to %d GB)",
            spill_dir,
            disk_usage.free / (1024**3),
            max_temp_gb,
        )
        conn.execute(f"SET max_temp_directory_size = '{int(max_temp_gb)}GB'")
        conn.execute("SET preserve_insertion_order = false")

        # Log existing checkpoints for resume visibility
        existing_ckpts = sorted(duckdb_temp_dir.glob("checkpoint_*.parquet"))
        corpus_exists = (duckdb_temp_dir / "corpus.parquet").exists()
        if existing_ckpts or corpus_exists:
            names = ["corpus.parquet"] if corpus_exists else []
            names.extend(p.name for p in existing_ckpts)
            console.print(f"[green]Resuming with checkpoints:[/green] {', '.join(names)}")

        try:
            self._run_duckdb_frequency_queries(
                conn,
                filepath,
                duckdb_temp_dir,
                console=console,
                file_size=file_size,
                start_time=start_time,
                invalid_words_sql=invalid_words_sql,
                mem_limit_gb=mem_limit_gb,
                trigram_mem_limit_gb=trigram_mem_limit_gb,
            )
        except Exception:
            conn.close()
            logger.info("Checkpoints preserved in %s for resume on next run", duckdb_temp_dir)
            raise
        else:
            if self._duckdb_ngram_store:
                # Keep connection open — DuckDB tables needed for filter/save steps.
                # Store temp dir path for cleanup after all saves complete.
                self._duckdb_ngram_store.duckdb_temp_dir = duckdb_temp_dir
            else:
                conn.close()
                if duckdb_temp_dir.exists():
                    try:
                        shutil.rmtree(duckdb_temp_dir)
                    except OSError as e:
                        logger.warning(
                            "Failed to clean up DuckDB temp dir %s: %s", duckdb_temp_dir, e
                        )

    def _run_duckdb_frequency_queries(
        self,
        conn: "duckdb.DuckDBPyConnection",
        filepath: Path,
        duckdb_temp_dir: Path,
        *,
        console: "Console",
        file_size: int,
        start_time: float,
        invalid_words_sql: str,
        mem_limit_gb: int = 6,
        trigram_mem_limit_gb: int = 2,
    ) -> None:
        """Execute all DuckDB frequency counting queries.

        Orchestrates 6 steps: Parquet conversion, syllable/word counting,
        n-gram base table + bigram/trigram counting, and POS tagging.
        Each step supports checkpoint-based resume.
        """
        import time

        from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

        # Step 0.5: Convert Arrow → Parquet (or resume from cached Parquet)
        temp_parquet = duckdb_temp_dir / "corpus.parquet"
        total_rows, load_time = self._duckdb_convert_to_parquet(
            conn, filepath, duckdb_temp_dir, console, file_size
        )

        # Estimate duration based on file size
        file_size_gb = file_size / (1024**3)
        est_min = int(file_size_gb * 2)
        est_max = int(file_size_gb * 5)
        if est_min > 0:
            console.print(
                f"[cyan]Starting frequency queries ({file_size_gb:.1f} GB, "
                f"est. {est_min}-{est_max} min)[/cyan]"
            )

        # Steps 1-5: Counting queries
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            self._duckdb_step_syllables(conn, duckdb_temp_dir, progress, console)
            self._duckdb_step_words(conn, duckdb_temp_dir, progress, console, invalid_words_sql)
            quality_words = set(self.word_counts.keys())
            store = self._duckdb_step_bigrams(
                conn, duckdb_temp_dir, progress, console, invalid_words_sql, quality_words
            )
            self._duckdb_step_trigrams(
                conn, progress, console, store, trigram_mem_limit_gb, mem_limit_gb
            )
            self._duckdb_ngram_store = store

        # Cleanup corpus view
        conn.execute("DROP VIEW IF EXISTS corpus")

        total_query_time = time.time() - start_time
        sql_time = total_query_time - load_time
        speed_mb = file_size / (1024**2) / total_query_time
        console.print(
            f"[bold green]DuckDB queries complete:[/bold green] {total_query_time:.1f}s "
            f"({speed_mb:.0f} MB/s)"
        )
        self.sub_step_timings.append(("DuckDB counting", sql_time))

        # Post-processing
        self._duckdb_post_process(console, time)

        # POS tagging (if tagger available)
        if self.pos_tagger:
            self._duckdb_step_pos(console, temp_parquet, total_rows, duckdb_temp_dir)

        # Update statistics
        self._duckdb_update_stats()

        total_time = time.time() - start_time
        final_speed = file_size / (1024**2) / total_time
        console.print(
            f"[bold green]✓ Frequency building complete:[/bold green] {total_time:.1f}s "
            f"({final_speed:.0f} MB/s)"
        )
        console.print(f"  Total syllables: {self.stats['total_syllables']:,}")
        console.print(f"  Total words: {self.stats['total_words']:,}")
        console.print(f"  Total bigrams: {self.stats['total_bigrams']:,}")
        console.print(f"  Total trigrams: {self.stats['total_trigrams']:,}")
        if self.pos_tagger:
            console.print(f"  POS unigrams: {self.stats['total_pos_unigrams']:,}")
            console.print(f"  POS bigrams: {self.stats['total_pos_bigrams']:,}")

    # ------------------------------------------------------------------
    # Step methods called by _run_duckdb_frequency_queries
    # ------------------------------------------------------------------

    def _duckdb_convert_to_parquet(
        self,
        conn: "duckdb.DuckDBPyConnection",
        filepath: Path,
        duckdb_temp_dir: Path,
        console: "Console",
        file_size: int,
    ) -> tuple[int, float]:
        """Convert Arrow IPC file to Parquet for DuckDB streaming queries.

        Returns (total_rows, load_time).  Skips conversion if Parquet already
        exists (checkpoint resume).
        """
        import time

        import pyarrow.parquet as pq
        from rich.progress import (
            BarColumn,
            DownloadColumn,
            Progress,
            SpinnerColumn,
            TextColumn,
            TimeElapsedColumn,
            TimeRemainingColumn,
            TransferSpeedColumn,
        )

        temp_parquet = duckdb_temp_dir / "corpus.parquet"
        total_rows = 0

        if temp_parquet.exists():
            load_start = time.time()
            total_rows = pq.read_metadata(str(temp_parquet)).num_rows
            load_time = time.time() - load_start
            console.print(
                f"  [green]✓ Parquet: resumed from checkpoint[/green] ({total_rows:,} rows)"
            )
            self.sub_step_timings.append(("Arrow→Parquet conversion (cached)", load_time))
        else:
            console.print("[cyan]Converting Arrow to Parquet for streaming query...[/cyan]")
            load_start = time.time()

            with Progress(
                SpinnerColumn(),
                TextColumn("[cyan]Converting...[/cyan]"),
                BarColumn(),
                DownloadColumn(),
                TransferSpeedColumn(),
                TimeElapsedColumn(),
                TextColumn("•"),
                TextColumn("ETA"),
                TimeRemainingColumn(),
                console=console,
            ) as load_progress:
                task = load_progress.add_task("Converting", total=file_size)
                bytes_read = 0
                writer = None

                with pa.memory_map(str(filepath), "r") as source:
                    reader = pa.ipc.open_stream(source)
                    schema = reader.schema

                    for batch in reader:
                        if writer is None:
                            writer = pq.ParquetWriter(
                                str(temp_parquet), schema, compression="snappy"
                            )
                        writer.write_batch(batch)
                        total_rows += batch.num_rows
                        bytes_read += batch.nbytes
                        load_progress.update(task, completed=bytes_read)

                if writer is not None:
                    writer.close()

            load_time = time.time() - load_start
            console.print(f"  ✓ Converted {total_rows:,} rows to Parquet ({load_time:.1f}s)")
            self.sub_step_timings.append(("Arrow→Parquet conversion", load_time))

        parquet_path_safe = _validate_path_for_sql(temp_parquet)
        conn.execute(
            f"CREATE OR REPLACE VIEW corpus AS SELECT * FROM read_parquet('{parquet_path_safe}')"
        )
        return total_rows, load_time

    def _duckdb_step_syllables(
        self,
        conn: "duckdb.DuckDBPyConnection",
        duckdb_temp_dir: Path,
        progress: Any,
        console: "Console",
    ) -> None:
        """Step 1/5: Count syllables from DuckDB with checkpoint support."""
        import time

        task1 = progress.add_task("[cyan]Step 1/5: Counting syllables...", total=None)
        syl_start = time.time()

        ckpt = self._load_checkpoint(duckdb_temp_dir, "syllables")
        if ckpt is not None:
            for syl, cnt in zip(
                ckpt.column("syllable").to_pylist(),
                ckpt.column("count").to_pylist(),
                strict=True,
            ):
                self.syllable_counts[syl] = cnt
            del ckpt
            syl_time = time.time() - syl_start
            progress.update(task1, completed=True)
            console.print(
                f"  [green]✓ Syllables: resumed from checkpoint[/green] "
                f"({len(self.syllable_counts):,} unique, {syl_time:.1f}s)"
            )
        else:
            syl_cursor = conn.execute("""
                SELECT syllable, COUNT(*) as cnt
                FROM (
                    SELECT UNNEST(syllables) as syllable
                    FROM corpus
                )
                WHERE syllable IS NOT NULL AND syllable != ''
                GROUP BY syllable
            """)

            raw_syl_counts: Counter = Counter()
            while True:
                chunk = syl_cursor.fetchmany(100_000)
                if not chunk:
                    break
                for syl, cnt in chunk:
                    raw_syl_counts[syl] = cnt
            del syl_cursor
            self.syllable_counts = Counter()
            for syl, cnt in raw_syl_counts.items():
                self.syllable_counts[normalize(syl)] += cnt
            del raw_syl_counts
            syl_time = time.time() - syl_start
            progress.update(task1, completed=True)
            console.print(f"  ✓ Syllables: {len(self.syllable_counts):,} unique ({syl_time:.1f}s)")
            self._save_checkpoint(
                duckdb_temp_dir,
                "syllables",
                {
                    "syllable": list(self.syllable_counts.keys()),
                    "count": list(self.syllable_counts.values()),
                },
            )

    def _duckdb_step_words(
        self,
        conn: "duckdb.DuckDBPyConnection",
        duckdb_temp_dir: Path,
        progress: Any,
        console: "Console",
        invalid_words_sql: str,
    ) -> None:
        """Step 2/5: Count words from DuckDB with checkpoint support."""
        import time

        task2 = progress.add_task("[cyan]Step 2/5: Counting words...", total=None)
        word_start = time.time()

        ckpt = self._load_checkpoint(duckdb_temp_dir, "words")
        if ckpt is not None:
            for word, cnt in zip(
                ckpt.column("word").to_pylist(),
                ckpt.column("count").to_pylist(),
                strict=True,
            ):
                self.word_counts[word] = cnt
            del ckpt
            word_time = time.time() - word_start
            progress.update(task2, completed=True)
            console.print(
                f"  [green]✓ Words: resumed from checkpoint[/green] "
                f"({len(self.word_counts):,} unique, {word_time:.1f}s)"
            )
        else:
            word_cursor = conn.execute(f"""
                SELECT word, COUNT(*) as cnt
                FROM (
                    SELECT UNNEST(words) as word
                    FROM corpus
                )
                WHERE word IS NOT NULL AND word != ''
                  AND word NOT IN ({invalid_words_sql})
                GROUP BY word
            """)

            raw_word_counts: Counter = Counter()
            while True:
                chunk = word_cursor.fetchmany(100_000)
                if not chunk:
                    break
                for word, cnt in chunk:
                    raw_word_counts[word] = cnt
            del word_cursor
            self.word_counts = Counter(
                {
                    w: c
                    for w, c in raw_word_counts.items()
                    if is_quality_word(w, allow_extended_myanmar=_flags.allow_extended_myanmar)
                }
            )
            quality_filtered = len(raw_word_counts) - len(self.word_counts)
            del raw_word_counts
            word_time = time.time() - word_start
            progress.update(task2, completed=True)
            console.print(
                f"  ✓ Words: {len(self.word_counts):,} unique ({word_time:.1f}s)"
                + (
                    f" [dim](filtered {quality_filtered:,} low-quality)[/dim]"
                    if quality_filtered
                    else ""
                )
            )
            self._save_checkpoint(
                duckdb_temp_dir,
                "words",
                {
                    "word": list(self.word_counts.keys()),
                    "count": list(self.word_counts.values()),
                },
            )

    def _duckdb_step_bigrams(
        self,
        conn: "duckdb.DuckDBPyConnection",
        duckdb_temp_dir: Path,
        progress: Any,
        console: "Console",
        invalid_words_sql: str,
        quality_words: set,
    ) -> "DuckDBNgramStore":
        """Steps 3-4/5: Build n-gram base table and bigram counts.

        Returns the DuckDBNgramStore with bigram_counts populated.
        """
        import time

        # Step 3: Build n-gram base table (persistent for resume)
        task3 = progress.add_task("[cyan]Step 3/5: Building n-gram base table...", total=None)
        ngram_start = time.time()

        existing_tables = [t[0] for t in conn.execute("SHOW TABLES").fetchall()]
        needs_rebuild = False
        if "ngram_windows" in existing_tables:
            _bucket_row = conn.execute("""
                SELECT COUNT(*) FROM information_schema.columns
                WHERE table_name = 'ngram_windows' AND column_name = 'bucket'
            """).fetchone()
            has_bucket = _bucket_row is not None and _bucket_row[0] > 0
            if has_bucket:
                base_time = time.time() - ngram_start
                progress.update(task3, completed=True)
                console.print(
                    f"  [green]✓ N-gram base table: resumed from checkpoint[/green] "
                    f"({base_time:.1f}s)"
                )
            else:
                console.print(
                    "  [yellow]Rebuilding n-gram table with bucket partitions "
                    "for incremental checkpoints...[/yellow]"
                )
                conn.execute("DROP TABLE ngram_windows")
                needs_rebuild = True
        else:
            needs_rebuild = True

        if needs_rebuild:
            conn.execute(f"""
                CREATE TABLE ngram_windows AS
                WITH numbered_sentences AS (
                    SELECT
                        ROW_NUMBER() OVER () as sentence_id,
                        words
                    FROM corpus
                ),
                words_flat AS (
                    SELECT
                        sentence_id,
                        UNNEST(words) as word,
                        GENERATE_SUBSCRIPTS(words, 1) as word_idx
                    FROM numbered_sentences
                ),
                all_ngrams AS (
                    SELECT
                        word as w1,
                        LEAD(word, 1) OVER (
                            PARTITION BY sentence_id ORDER BY word_idx
                        ) as w2,
                        LEAD(word, 2) OVER (
                            PARTITION BY sentence_id ORDER BY word_idx
                        ) as w3
                    FROM words_flat
                    WHERE word IS NOT NULL AND word != ''
                )
                SELECT w1, w2, w3, hash(w1) % {_NGRAM_BUCKET_COUNT} as bucket
                FROM all_ngrams
                WHERE w1 NOT IN ({invalid_words_sql})
                  AND (w2 IS NULL OR w2 NOT IN ({invalid_words_sql}))
                  AND (w3 IS NULL OR w3 NOT IN ({invalid_words_sql}))
                ORDER BY bucket
            """)

            base_time = time.time() - ngram_start
            progress.update(task3, completed=True)
            console.print(
                f"  ✓ N-gram base table created with {_NGRAM_BUCKET_COUNT} "
                f"bucket partitions ({base_time:.1f}s)"
            )

        # Step 4: Count bigrams (disk-backed via DuckDB table)
        task4 = progress.add_task("[cyan]Step 4/5: Counting bigrams...", total=None)
        bigram_start = time.time()

        store = DuckDBNgramStore(conn, self.logger)
        store.duckdb_temp_dir = duckdb_temp_dir
        store.register_quality_words(quality_words)

        if store.has_table("bigram_counts"):
            bigram_time = time.time() - bigram_start
            progress.update(task4, completed=True)
            console.print(
                f"  [green]✓ Bigrams: resumed from DuckDB table[/green] "
                f"({store.get_count('bigram_counts'):,} unique, {bigram_time:.1f}s)"
            )
        else:
            store.build_bigram_table()
            bigram_time = time.time() - bigram_start
            progress.update(task4, completed=True)
            console.print(
                f"  ✓ Bigrams: {store.get_count('bigram_counts'):,} unique ({bigram_time:.1f}s)"
            )

        return store

    def _duckdb_step_trigrams(
        self,
        conn: "duckdb.DuckDBPyConnection",
        progress: Any,
        console: "Console",
        store: "DuckDBNgramStore",
        trigram_mem_limit_gb: int,
        mem_limit_gb: int,
    ) -> None:
        """Step 5/5: Build trigram counts (disk-backed via DuckDB table)."""
        import time

        task5 = progress.add_task("[cyan]Step 5/5: Counting trigrams...", total=None)
        trigram_start = time.time()

        if store.has_table("trigram_counts"):
            trigram_time = time.time() - trigram_start
            progress.update(task5, completed=True)
            console.print(
                f"  [green]✓ Trigrams: resumed from DuckDB table[/green] "
                f"({store.get_count('trigram_counts'):,} unique, {trigram_time:.1f}s)"
            )
        else:
            store.build_trigram_table(trigram_mem_limit_gb)
            conn.execute(f"SET memory_limit = '{int(mem_limit_gb)}GB'")
            trigram_time = time.time() - trigram_start
            progress.update(task5, completed=True)
            console.print(
                f"  ✓ Trigrams: {store.get_count('trigram_counts'):,} unique ({trigram_time:.1f}s)"
            )

    def _duckdb_post_process(self, console: "Console", time_module: Any) -> None:
        """Post-processing: syllable counts, invalid filtering."""
        from tqdm import tqdm

        post_start = time_module.time()
        console.print("[cyan]Post-processing...[/cyan]")

        for word in tqdm(self.word_counts, desc="  • Syllable counts", unit="words"):
            self.word_syllables[word] = self._count_syllables_accurate(word)

        self.filter_invalid_syllables()
        self.filter_invalid_words()
        self.sub_step_timings.append(("Post-processing", time_module.time() - post_start))

    def _duckdb_update_stats(self) -> None:
        """Update statistics counters after all DuckDB queries complete."""
        self.stats["total_syllables"] = sum(self.syllable_counts.values())
        self.stats["unique_syllables"] = len(self.syllable_counts)
        self.stats["total_words"] = sum(self.word_counts.values())
        self.stats["unique_words"] = len(self.word_counts)
        if self._duckdb_ngram_store:
            ngram_stats = self._duckdb_ngram_store.get_stats()
            self.stats["total_bigrams"] = ngram_stats["total_bigrams"]
            self.stats["unique_bigrams"] = ngram_stats["unique_bigrams"]
            self.stats["total_trigrams"] = ngram_stats["total_trigrams"]
            self.stats["unique_trigrams"] = ngram_stats["unique_trigrams"]
        self.stats["total_pos_unigrams"] = sum(self.pos_unigram_counts.values())
        self.stats["unique_pos_unigrams"] = len(self.pos_unigram_counts)
        self.stats["total_pos_bigrams"] = sum(self.pos_bigram_counts.values())
        self.stats["unique_pos_bigrams"] = len(self.pos_bigram_counts)
        self.stats["total_pos_trigrams"] = sum(self.pos_trigram_counts.values())
        self.stats["unique_pos_trigrams"] = len(self.pos_trigram_counts)

    def _count_syllables_accurate(self, word: str) -> int:
        """
        Accurately count syllables in a word using DefaultSegmenter.

        Args:
            word: Word to count syllables for

        Returns:
            Exact syllable count
        """
        try:
            syllables = self.segmenter.segment_syllables(word)
            return len(syllables)
        except (RuntimeError, ValueError) as e:
            # Fallback if segmentation fails for any reason
            self.logger.warning(f"Syllable segmentation failed for '{word}': {e}")
            return max(1, len(word) // 3)

    def calculate_bigram_probabilities(self) -> None:
        """Bigram probabilities are calculated and saved by DuckDB COPY TO.

        Returns None. The DuckDB store handles calculation + export in one
        SQL step inside ``save_bigram_probabilities()``.
        """
        return None

    def calculate_trigram_probabilities(self) -> None:
        """Trigram probabilities are calculated and saved by DuckDB COPY TO.

        Returns None. The DuckDB store handles calculation + export in one
        SQL step inside ``save_trigram_probabilities()``.
        """
        return None

    def filter_by_frequency(self, curated_words: dict[str, str] | None = None) -> None:
        """Filter items below minimum frequency threshold.

        Args:
            curated_words: Optional dict of word→pos_tag for curated vocabulary.
                          These words are never filtered out, ensuring they retain their
                          corpus frequency even if below the threshold.
        """
        self.logger.info("\nFiltering items with frequency < thresholds...")
        self.logger.info(f"  Syllable threshold: {self.min_syllable_frequency}")
        self.logger.info(f"  Word threshold: {self.min_word_frequency}")
        self.logger.info(f"  Bigram threshold: {self.min_bigram_frequency}")
        if curated_words:
            self.logger.info(
                f"  Preserving {len(curated_words):,} curated words (bypass freq filter)"
            )

        # Filter syllables
        if self.min_syllable_frequency > 1:
            original_syllable_count = len(self.syllable_counts)
            self.syllable_counts = Counter(
                {
                    syllable: count
                    for syllable, count in self.syllable_counts.items()
                    if count >= self.min_syllable_frequency
                }
            )
            self.stats["filtered_syllables"] = original_syllable_count - len(self.syllable_counts)

        # Filter words (but preserve curated words regardless of frequency)
        if self.min_word_frequency > 1:
            original_word_count = len(self.word_counts)
            curated_set: set[str] = set(curated_words) if curated_words else set()
            self.word_counts = Counter(
                {
                    word: count
                    for word, count in self.word_counts.items()
                    if count >= self.min_word_frequency or word in curated_set
                }
            )
            self.stats["filtered_words"] = original_word_count - len(self.word_counts)

        # Filter bigrams and trigrams via DuckDB SQL DELETE
        if self._duckdb_ngram_store:
            if self.min_bigram_frequency > 1:
                orig, remaining = self._duckdb_ngram_store.filter_bigrams(self.min_bigram_frequency)
                self.stats["filtered_bigrams"] = orig - remaining
            if self.min_trigram_frequency > 1:
                orig, remaining = self._duckdb_ngram_store.filter_trigrams(
                    self.min_trigram_frequency
                )
                self.stats["filtered_trigrams"] = orig - remaining

        self.logger.info(f"  Filtered {self.stats['filtered_syllables']:,} syllables")
        self.logger.info(f"  Filtered {self.stats['filtered_words']:,} words")
        self.logger.info(f"  Filtered {self.stats['filtered_bigrams']:,} bigrams")
        self.logger.info(f"  Filtered {self.stats['filtered_trigrams']:,} trigrams")
