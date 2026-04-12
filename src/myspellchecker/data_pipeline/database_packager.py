"""
Database packager module for the data pipeline.

This module coordinates the packaging of frequency data into SQLite database.
It delegates specialized responsibilities to:
- SchemaManager: Database schema creation and migrations
- POSInferenceManager: POS inference application and statistics
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any

from myspellchecker.core.exceptions import PackagingError

from ..core.constants import (
    BUILD_PRAGMA_CACHE_SIZE,
    BUILD_PRAGMA_MMAP_SIZE,
    DEFAULT_FILE_ENCODING,
    DEFAULT_PIPELINE_BIGRAM_PROBS_FILE,
    DEFAULT_PIPELINE_POS_BIGRAM_PROBS_FILE,
    DEFAULT_PIPELINE_POS_TRIGRAM_PROBS_FILE,
    DEFAULT_PIPELINE_POS_UNIGRAM_PROBS_FILE,
    DEFAULT_PIPELINE_SYLLABLE_FREQS_FILE,
    DEFAULT_PIPELINE_TRIGRAM_PROBS_FILE,
    DEFAULT_PIPELINE_WORD_FREQS_FILE,
)
from ..text.validator import is_quality_word, validate_word
from ..utils.console import PipelineConsole
from ..utils.logging_utils import get_logger
from .config import runtime_flags as _flags
from .pos_inference_manager import POSInferenceManager
from .schema_manager import SchemaManager

__all__ = [
    "DatabasePackager",
]

# Try import Cython TSV reader
try:
    from myspellchecker.data_pipeline import tsv_reader_c  # type: ignore[attr-defined]

    _HAS_TSV_READER = True
except ImportError:
    _HAS_TSV_READER = False

# ---------------------------------------------------------------------------
# Shared runtime flags (replaces per-module globals)
# ---------------------------------------------------------------------------


def _parse_multi_pos(raw_pos: str) -> str:
    """Parse multi-POS tag string from TSV into pipe-separated tags for storage.

    Handles both new format ``N:85|V:15`` (with counts) and legacy format ``N``
    (single tag without count). Returns pipe-separated tags ordered by count,
    e.g. ``N|V``.
    """
    if not raw_pos:
        return ""
    parts = raw_pos.split("|")
    tags = []
    for part in parts:
        if ":" in part:
            tag = part.split(":")[0]
        else:
            tag = part
        tag = tag.strip()
        if tag:
            tags.append(tag)
    return "|".join(tags)


class DatabasePackager:
    """Packages frequency data into SQLite database."""

    def __init__(self, input_dir: str | Path, database_path: str | Path):
        """
        Initialize the database packager.

        Args:
            input_dir: Directory containing frequency TSV files
            database_path: Output database path
        """
        self.input_dir = Path(input_dir)
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_common_attrs()

    def _init_common_attrs(self) -> None:
        """Initialize attributes shared between __init__ and from_existing."""
        self.logger = get_logger(__name__)
        self.console = PipelineConsole()

        # Database connection
        self.conn: sqlite3.Connection | None = None
        self.cursor: sqlite3.Cursor | None = None

        # Transaction state flag - when True, load_ methods skip individual commits
        self._in_transaction: bool = False
        self._transaction_lock: threading.Lock = threading.Lock()

        # Delegated managers (initialized on connect)
        self._schema_manager: SchemaManager | None = None
        self._pos_inference_manager: POSInferenceManager | None = None

        # Word ID mapping for bigrams
        self.word_to_id: dict[str, int] = {}

        # Statistics
        self.stats = {
            "syllables_inserted": 0,
            "words_inserted": 0,
            "curated_words_inserted": 0,
            "bigrams_inserted": 0,
            "trigrams_inserted": 0,
            "pos_unigrams_inserted": 0,
            "pos_bigrams_inserted": 0,
            "pos_trigrams_inserted": 0,
        }

        # Lite mode limits (None = unlimited)
        self.limits: dict[str, int | None] = {
            "syllables": None,
            "words": None,
            "bigrams": None,
            "trigrams": None,
        }

        # Track verified tables for rich output
        self._verified_tables: list[str] = []

    @classmethod
    def from_existing(cls, database_path: str | Path) -> DatabasePackager:
        """
        Create a DatabasePackager instance for an existing database.

        This is useful for operations like POS inference that only need
        to update an existing database without requiring input files.

        Args:
            database_path: Path to existing database file.

        Returns:
            DatabasePackager instance with connection to existing database.

        Raises:
            FileNotFoundError: If database file doesn't exist.

        Example:
            >>> packager = DatabasePackager.from_existing("dictionary.db")
            >>> stats = packager.apply_inferred_pos()
            >>> packager.close()
        """
        database_path = Path(database_path)
        if not database_path.exists():
            raise FileNotFoundError(f"Database not found: {database_path}")

        instance = cls.__new__(cls)
        instance.input_dir = database_path.parent  # Use db parent as dummy input dir
        instance.database_path = database_path
        instance._init_common_attrs()

        # Connect to existing database
        instance.conn = sqlite3.connect(str(database_path), check_same_thread=False)
        instance.cursor = instance.conn.cursor()

        try:
            # Build-mode optimization PRAGMAs (safe to lose data on crash -- rebuild)
            instance.conn.execute("PRAGMA journal_mode = WAL")
            instance.conn.execute("PRAGMA synchronous = OFF")
            instance.conn.execute(f"PRAGMA cache_size = {BUILD_PRAGMA_CACHE_SIZE}")
            instance.conn.execute("PRAGMA temp_store = MEMORY")
            instance.conn.execute(f"PRAGMA mmap_size = {BUILD_PRAGMA_MMAP_SIZE}")

            # Initialize managers
            instance._schema_manager = SchemaManager(
                instance.conn, instance.cursor, instance.console
            )
            instance._pos_inference_manager = POSInferenceManager(
                instance.conn, instance.cursor, instance.console
            )

            # Check and add missing columns for POS inference
            instance._ensure_inferred_pos_columns()
        except Exception:
            instance.conn.close()
            instance.conn = None
            instance.cursor = None
            raise

        return instance

    def _ensure_inferred_pos_columns(self) -> None:
        """
        Ensure the inferred_pos columns exist in the words table.

        Adds the columns if they don't exist (for database migration).
        Delegates to SchemaManager.
        """
        if self._schema_manager:
            self._schema_manager.ensure_inferred_pos_columns()

    def connect(self, incremental: bool = False) -> None:
        """
        Connect to SQLite database.

        Args:
            incremental: If True, keep existing database for updates.
        """
        self.console.info(f"Connecting to database: {self.database_path}")

        if not incremental:
            # Remove existing database
            if self.database_path.exists():
                self.console.step("Removing existing database...")
                self.database_path.unlink()
        else:
            if self.database_path.exists():
                self.console.step("Incremental mode: Opening existing database...")

        # Create connection (check_same_thread=False for multi-threaded POS inference)
        self.conn = sqlite3.connect(str(self.database_path), check_same_thread=False)
        self.cursor = self.conn.cursor()

        # Build-mode optimization PRAGMAs (safe to lose data on crash — rebuild)
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA synchronous = OFF")
        self.conn.execute(f"PRAGMA cache_size = {BUILD_PRAGMA_CACHE_SIZE}")
        self.conn.execute("PRAGMA temp_store = MEMORY")
        self.conn.execute(f"PRAGMA mmap_size = {BUILD_PRAGMA_MMAP_SIZE}")

        # Initialize delegated managers
        self._schema_manager = SchemaManager(self.conn, self.cursor, self.console)
        self._pos_inference_manager = POSInferenceManager(self.conn, self.cursor, self.console)

        self.console.success("Database connected")

    def begin_transaction(self) -> None:
        """
        Begin a database transaction.

        Use this to group multiple database operations into a single atomic unit.
        Call commit_transaction() on success or rollback_transaction() on failure.

        Thread-safe: Uses a lock to prevent concurrent transaction state changes.
        """
        if not self.conn:
            raise PackagingError("Database not connected. Call connect() first.")

        with self._transaction_lock:
            if self._in_transaction:
                raise PackagingError("Transaction already in progress")
            self.console.step("Beginning database transaction...")
            self._in_transaction = True
            self.conn.execute("BEGIN TRANSACTION")

    def commit_transaction(self) -> None:
        """
        Commit the current transaction.

        All changes since begin_transaction() will be permanently saved.

        Thread-safe: Uses a lock to prevent concurrent transaction state changes.
        """
        if not self.conn:
            raise PackagingError("Database not connected.")

        with self._transaction_lock:
            if not self._in_transaction:
                raise PackagingError("No transaction in progress")
            self.console.step("Committing database transaction...")
            self.conn.commit()
            self._in_transaction = False

    def rollback_transaction(self) -> None:
        """
        Rollback the current transaction.

        All changes since begin_transaction() will be discarded.

        Thread-safe: Uses a lock to prevent concurrent transaction state changes.
        """
        if not self.conn:
            raise PackagingError("Database not connected.")

        with self._transaction_lock:
            if not self._in_transaction:
                raise PackagingError("No transaction in progress")
            self.console.warning("Rolling back database transaction...")
            self.conn.rollback()
            self._in_transaction = False

    def create_schema(self, defer_indexes: bool = False) -> None:
        """Create database schema with tables and indexes.

        Creates all required tables in a transaction to ensure atomicity.
        If any table creation fails, all changes are rolled back.

        Delegates to SchemaManager for schema creation.

        Args:
            defer_indexes: If True, skip index creation during schema setup.
                Call create_indexes() explicitly after bulk data loading
                for significantly faster builds.

        Raises:
            RuntimeError: If database is not connected.
            sqlite3.Error: If schema creation fails (after rollback).
        """
        if not self._schema_manager:
            raise PackagingError("Database not connected. Call connect() first.")

        self._verified_tables = self._schema_manager.create_schema(
            in_transaction=self._in_transaction,
            defer_indexes=defer_indexes,
        )

    def create_indexes(self) -> None:
        """Create indexes for fast lookups.

        Delegates to SchemaManager for index creation.
        """
        if not self._schema_manager:
            return

        self._schema_manager.create_indexes()

    def get_processed_files(self) -> dict[str, tuple[float, int]]:
        """
        Get list of processed files.

        Returns:
            Dictionary mapping file path to (mtime, size)
        """
        if not self.cursor:
            return {}

        try:
            self.cursor.execute("SELECT path, mtime, size FROM processed_files")
            return {row[0]: (row[1], row[2]) for row in self.cursor.fetchall()}
        except sqlite3.OperationalError:
            # Table might not exist yet if upgrading
            return {}

    def update_processed_file(self, path: str, mtime: float, size: int) -> None:
        """
        Record a processed file.

        Args:
            path: File path (string)
            mtime: Modification time
            size: File size in bytes
        """
        if not self.cursor or not self.conn:
            return

        self.cursor.execute(
            "INSERT OR REPLACE INTO processed_files (path, mtime, size) VALUES (?, ?, ?)",
            (str(path), mtime, size),
        )
        if not self._in_transaction:
            self.conn.commit()

    def get_current_counts(self):
        """
        Retrieve current counts from the database for incremental update hydration.

        Returns:
            Tuple of (syllable_counts, word_counts, bigram_counts, trigram_counts,
            word_syllables, pos_unigram_probs, pos_bigram_probs, pos_trigram_probs)
            where counts are Counters and word_syllables is a Dict.
        """
        if not self.cursor:
            raise PackagingError("Database not connected")

        from collections import Counter

        self.console.step("Hydrating counts from existing database...")

        syllable_counts: Counter[str] = Counter()
        word_counts: Counter[str] = Counter()
        bigram_counts: Counter[tuple[str, str]] = Counter()
        trigram_counts: Counter[tuple[str, str, str]] = Counter()
        word_syllables = {}
        pos_unigram_probs = {}
        pos_bigram_probs = {}
        pos_trigram_probs = {}

        # Load syllables
        self.cursor.execute("SELECT syllable, frequency FROM syllables")
        for row in self.cursor.fetchall():
            syllable_counts[row[0]] = row[1]

        # Load words
        self.cursor.execute("SELECT word, syllable_count, frequency FROM words")
        for row in self.cursor.fetchall():
            word = row[0]
            word_counts[word] = row[2]
            word_syllables[word] = row[1]

        # Rebuild word_to_id for n-gram decoding
        if not self.word_to_id:
            self.build_word_id_mapping()
        id_to_word = {v: k for k, v in self.word_to_id.items()}

        # Load bigrams
        try:
            self.cursor.execute("SELECT word1_id, word2_id, count FROM bigrams")
            for row in self.cursor.fetchall():
                w1 = id_to_word.get(row[0])
                w2 = id_to_word.get(row[1])
                if w1 and w2:
                    bigram_counts[(w1, w2)] = row[2]
        except sqlite3.OperationalError:
            # 'count' column might not exist if loading from old DB
            self.logger.warning(
                "Could not load bigram counts (column missing). "
                "Incremental update may be inaccurate."
            )

        # Load trigrams
        try:
            self.cursor.execute("SELECT word1_id, word2_id, word3_id, count FROM trigrams")
            for row in self.cursor.fetchall():
                w1 = id_to_word.get(row[0])
                w2 = id_to_word.get(row[1])
                w3 = id_to_word.get(row[2])
                if w1 and w2 and w3:
                    trigram_counts[(w1, w2, w3)] = row[3]
        except sqlite3.OperationalError:
            # 'count' column might not exist
            pass

        # Load POS unigrams
        try:
            self.cursor.execute("SELECT pos, probability FROM pos_unigrams")
            for row in self.cursor.fetchall():
                pos_unigram_probs[row[0]] = row[1]
        except sqlite3.OperationalError:
            self.console.warning("POS unigrams table not found. Skipping hydration.")

        # Load POS bigrams
        try:
            self.cursor.execute("SELECT pos1, pos2, probability FROM pos_bigrams")
            for row in self.cursor.fetchall():
                pos_bigram_probs[(row[0], row[1])] = row[2]
        except sqlite3.OperationalError:
            self.console.warning("POS bigrams table not found. Skipping hydration.")

        # Load POS trigrams
        try:
            self.cursor.execute("SELECT pos1, pos2, pos3, probability FROM pos_trigrams")
            for row in self.cursor.fetchall():
                pos_trigram_probs[(row[0], row[1], row[2])] = row[3]
        except sqlite3.OperationalError:
            self.console.warning("POS trigrams table not found. Skipping hydration.")

        self.console.show_hydration(len(syllable_counts), len(word_counts))
        return (
            syllable_counts,
            word_counts,
            bigram_counts,
            trigram_counts,
            word_syllables,
            pos_unigram_probs,
            pos_bigram_probs,
            pos_trigram_probs,
        )

    def load_syllables(self, filename: str = DEFAULT_PIPELINE_SYLLABLE_FREQS_FILE) -> None:
        """
        Load and insert syllables from TSV file.

        Args:
            filename: Syllable frequencies filename
        """
        if not self.cursor or not self.conn:
            raise PackagingError("Database not connected.")

        filepath = self.input_dir / filename
        self.console.info(f"Loading syllables from: {filepath.name}")

        try:
            syllables_data = []
            skipped_invalid = 0

            if _HAS_TSV_READER:
                # Optimized Cython path
                for syllable, frequency in tsv_reader_c.read_syllables_tsv(str(filepath)):
                    if validate_word(
                        syllable, allow_extended_myanmar=_flags.allow_extended_myanmar
                    ):
                        syllables_data.append((syllable, frequency))
                    else:
                        skipped_invalid += 1
            else:
                # Python fallback
                with open(filepath, encoding=DEFAULT_FILE_ENCODING) as f:
                    next(f)  # Skip header
                    for line in f:
                        parts = line.strip().split("\t")
                        if len(parts) == 2:
                            syllable, frequency = parts
                            if validate_word(
                                syllable,
                                allow_extended_myanmar=_flags.allow_extended_myanmar,
                            ):
                                syllables_data.append((syllable, int(frequency)))
                            else:
                                skipped_invalid += 1

            if skipped_invalid > 0:
                self.console.step(f"Skipped {skipped_invalid:,} invalid syllables")

            # Batch insert
            limit = self.limits["syllables"]
            if limit:
                syllables_data = syllables_data[:limit]
                self.console.step(f"Limiting to top {limit} syllables")

            self.cursor.executemany(
                """
                INSERT INTO syllables (syllable, frequency) VALUES (?, ?)
                ON CONFLICT(syllable) DO UPDATE SET frequency=excluded.frequency
                """,
                syllables_data,
            )
            if not self._in_transaction:
                self.conn.commit()

            self.stats["syllables_inserted"] = len(syllables_data)
            self.console.success(f"Inserted/Updated {len(syllables_data):,} syllables")

        except FileNotFoundError:
            self.console.error(f"File not found: {filepath}")
            raise

    def load_words(
        self,
        filename: str = DEFAULT_PIPELINE_WORD_FREQS_FILE,
        curated_words: dict[str, str] | None = None,
    ) -> None:
        """
        Load and insert words from TSV file.

        Args:
            filename: Word frequencies filename
            curated_words: Dict of word→pos_tag to mark as 'is_curated=1' (from curated lexicon)
        """
        if not self.cursor or not self.conn:
            raise PackagingError("Database not connected.")

        filepath = self.input_dir / filename
        self.console.info(f"Loading words from: {filepath.name}")

        try:
            words_data = []
            skipped_invalid = 0
            curated_count = 0
            transformer_tagged_count = 0

            multi_pos_count = 0

            if _HAS_TSV_READER:
                # Optimized Cython path (now returns 4 values including pos_tag)
                tsv_data = tsv_reader_c.read_words_tsv(str(filepath))
                for word, syllable_count, frequency, tsv_pos_tag in tsv_data:
                    # Use comprehensive quality filter
                    if is_quality_word(word):
                        # Parse multi-POS format (N:85|V:15 → N|V)
                        parsed_pos = _parse_multi_pos(tsv_pos_tag) if tsv_pos_tag else ""
                        # Priority: curated_words > tsv_pos_tag
                        is_in_curated_words = curated_words and word in curated_words
                        if is_in_curated_words:
                            pos = parsed_pos if parsed_pos else None
                            is_curated = 1
                            curated_count += 1
                        elif parsed_pos:
                            pos = parsed_pos
                            is_curated = 0
                            transformer_tagged_count += 1
                        else:
                            pos = None
                            is_curated = 0
                        if pos and "|" in pos:
                            multi_pos_count += 1
                        words_data.append((word, syllable_count, frequency, pos, is_curated))
                    else:
                        skipped_invalid += 1
            else:
                # Python fallback
                with open(filepath, encoding=DEFAULT_FILE_ENCODING) as f:
                    next(f)  # Skip header
                    for line in f:
                        parts = line.strip().split("\t")
                        if len(parts) >= 3:
                            word = parts[0]
                            syllable_count = parts[1]
                            frequency = parts[2]
                            tsv_pos_tag = parts[3] if len(parts) >= 4 else ""
                            # Use comprehensive quality filter
                            if is_quality_word(word):
                                # Parse multi-POS format (N:85|V:15 → N|V)
                                parsed_pos = _parse_multi_pos(tsv_pos_tag) if tsv_pos_tag else ""
                                # Priority: curated_words > tsv_pos_tag
                                is_in_curated_words = curated_words and word in curated_words
                                if is_in_curated_words:
                                    pos = parsed_pos if parsed_pos else None
                                    is_curated = 1
                                    curated_count += 1
                                elif parsed_pos:
                                    pos = parsed_pos
                                    is_curated = 0
                                    transformer_tagged_count += 1
                                else:
                                    pos = None
                                    is_curated = 0
                                if pos and "|" in pos:
                                    multi_pos_count += 1
                                words_data.append(
                                    (word, int(syllable_count), int(frequency), pos, is_curated)
                                )
                            else:
                                skipped_invalid += 1

            if skipped_invalid > 0:
                self.console.step(f"Skipped {skipped_invalid:,} low-quality/invalid words")

            # Batch insert
            limit = self.limits["words"]
            if limit:
                words_data = words_data[:limit]
                self.console.step(f"Limiting to top {limit} words")

            self.cursor.executemany(
                """
                INSERT INTO words (word, syllable_count, frequency, pos_tag, is_curated)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(word) DO UPDATE SET
                    frequency=excluded.frequency,
                    syllable_count=excluded.syllable_count,
                    pos_tag=COALESCE(excluded.pos_tag, words.pos_tag),
                    is_curated=MAX(words.is_curated, excluded.is_curated)
                """,
                words_data,
            )
            if not self._in_transaction:
                self.conn.commit()

            self.stats["words_inserted"] = len(words_data)
            self.console.success(f"Inserted/Updated {len(words_data):,} words")
            if transformer_tagged_count > 0:
                self.console.info(f"  ├─ {transformer_tagged_count:,} with transformer POS tags")
            if multi_pos_count > 0:
                self.console.info(f"  ├─ {multi_pos_count:,} with multiple POS tags")
            if curated_count > 0:
                self.console.info(f"  ├─ {curated_count:,} marked as curated vocabulary")

            # Store total word count (sum of all frequencies) for dynamic unigram denominator
            total_word_count = sum(freq for _, _, freq, _, _ in words_data)
            self.store_metadata("total_word_count", str(total_word_count))
            self.console.info(f"  └─ Total word count: {total_word_count:,}")

            # Build word-to-ID mapping for bigrams
            self.build_word_id_mapping()

        except FileNotFoundError:
            self.console.error(f"File not found: {filepath}")
            raise

    def load_curated_words(self, curated_words: set[str] | dict[str, str]) -> int:
        """
        Load curated vocabulary words directly into the database.

        This method inserts curated words BEFORE corpus processing, ensuring
        all curated vocabulary is included in the database regardless of
        whether they appear in the corpus. When corpus words are loaded later,
        the ON CONFLICT clause will update frequency while preserving is_curated=1.

        Args:
            curated_words: Curated vocabulary to insert. Can be either:
                - dict[str, str]: word → pos_tag mapping (preserves POS from CSV)
                - set[str]: word set (no POS tags, for backward compatibility)

        Returns:
            Number of curated words successfully inserted.

        Notes:
            - Curated words bypass is_quality_word() validation (they're pre-verified)
            - Only empty/whitespace entries are skipped
            - Syllable count is computed via RegexSegmenter
            - Words are inserted with frequency=0, is_curated=1
            - POS tags from curated CSV are preserved via the dict mapping
            - Should be called BEFORE load_words() in the pipeline
        """
        if not self.cursor or not self.conn:
            raise PackagingError("Database not connected.")

        if not curated_words:
            return 0

        self.console.info(f"Loading {len(curated_words):,} curated vocabulary words...")

        # Import segmenter for syllable counting
        from ..segmenters.regex import RegexSegmenter

        segmenter = RegexSegmenter(allow_extended_myanmar=_flags.allow_extended_myanmar)

        words_data = []
        curated_syllables: set[str] = set()
        skipped_empty = 0
        pos_preserved = 0

        # Support both dict (word→pos_tag) and set (word only) inputs
        for word in curated_words:
            # Curated words bypass is_quality_word() since they're verified vocabulary
            # Only skip empty/whitespace-only strings
            if not word or not word.strip():
                skipped_empty += 1
                continue

            # Get syllable count via segmentation
            syllables = segmenter.segment_syllables(word)
            syllable_count = len(syllables)

            # Collect unique syllables for syllables table insertion
            curated_syllables.update(syllables)

            # Get POS tag from dict mapping, or None for set input
            pos_tag = None
            if isinstance(curated_words, dict):
                pos_tag = curated_words[word] or None  # "" → None
                if pos_tag:
                    pos_preserved += 1

            # Insert with frequency=0 (will be updated by corpus), is_curated=1
            words_data.append((word, syllable_count, 0, pos_tag, 1))

        if skipped_empty > 0:
            self.console.step(f"Skipped {skipped_empty:,} empty curated entries")

        if not words_data:
            self.console.warning("No valid curated words to insert")
            return 0

        # Batch insert curated syllables into syllables table
        # Uses ON CONFLICT to avoid duplicates with corpus-derived syllables;
        # frequency is kept as-is if the syllable already exists from corpus
        syllables_data = [(syl, 0) for syl in curated_syllables if syl and syl.strip()]
        if syllables_data:
            _syl_changes_before = self.conn.total_changes
            self.cursor.executemany(
                """
                INSERT INTO syllables (syllable, frequency) VALUES (?, ?)
                ON CONFLICT(syllable) DO NOTHING
                """,
                syllables_data,
            )
            new_syllables = self.conn.total_changes - _syl_changes_before
            self.console.step(
                f"Inserted {new_syllables:,} new syllables from curated words "
                f"({len(syllables_data):,} total unique)"
            )

        # Batch insert curated words
        # ON CONFLICT: preserve curated POS tag (takes priority over corpus-derived),
        # keep existing frequency if word was already loaded from corpus
        self.cursor.executemany(
            """
            INSERT INTO words (word, syllable_count, frequency, pos_tag, is_curated)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(word) DO UPDATE SET
                syllable_count=COALESCE(excluded.syllable_count, words.syllable_count),
                pos_tag=COALESCE(excluded.pos_tag, words.pos_tag),
                is_curated=1
            """,
            words_data,
        )
        if not self._in_transaction:
            self.conn.commit()

        inserted_count = len(words_data)
        self.stats["curated_words_inserted"] = inserted_count
        self.console.success(f"Inserted {inserted_count:,} curated vocabulary words")
        if pos_preserved > 0:
            self.console.info(f"  └─ {pos_preserved:,} with POS tags from curated lexicon")

        return inserted_count

    def apply_inferred_pos(
        self,
        min_frequency: int = 0,
        skip_tagged: bool = True,
        min_confidence: float = 0.0,
    ) -> dict[str, Any]:
        """
        Apply rule-based POS inference to words without POS tags.

        Uses the POSInferenceEngine to infer POS tags for untagged words
        based on morphological patterns (prefixes, suffixes), numeral detection,
        proper noun patterns, and the ambiguous words registry.

        Delegates to POSInferenceManager for inference logic.

        Args:
            min_frequency: Minimum word frequency threshold for inference.
                          Words below this frequency are skipped. Default 0.
            skip_tagged: If True, skip words that already have pos_tag.
                        If False, also infer for tagged words (overwrites inferred_pos).
            min_confidence: Minimum confidence threshold. Only apply inference
                           if confidence >= this value. Default 0.0.

        Returns:
            Dictionary with inference statistics:
            - total_words: Total words processed
            - inferred: Number of words with successful inference
            - skipped_tagged: Number of words skipped (already had pos_tag)
            - skipped_low_conf: Number skipped due to low confidence
            - ambiguous: Number of ambiguous words (multi-POS)
            - by_source: Dict of counts by inference source

        Example:
            >>> packager = DatabasePackager(input_dir, database_path)
            >>> packager.connect()
            >>> packager.create_schema()
            >>> packager.load_words("word_freqs.tsv")
            >>> stats = packager.apply_inferred_pos(min_frequency=5)
            >>> print(f"Inferred POS for {stats['inferred']} words")
        """
        if not self._pos_inference_manager:
            raise PackagingError("Database not connected.")

        return self._pos_inference_manager.apply_inferred_pos(
            min_frequency=min_frequency,
            skip_tagged=skip_tagged,
            min_confidence=min_confidence,
            in_transaction=self._in_transaction,
        )

    def get_pos_coverage_stats(self) -> dict[str, int]:
        """
        Get statistics on POS tag coverage in the database.

        Delegates to POSInferenceManager for coverage statistics.

        Returns:
            Dictionary with:
            - total_words: Total words in database
            - with_pos_tag: Words with pos_tag from seed
            - with_inferred_pos: Words with inferred POS
            - combined_coverage: Words with either pos_tag or inferred_pos
            - no_pos: Words without any POS information
            - ambiguous: Words with multi-POS (contains |)
        """
        if not self._pos_inference_manager:
            raise PackagingError("Database not connected.")

        return self._pos_inference_manager.get_pos_coverage_stats()

    def build_word_id_mapping(self) -> None:
        """Build mapping from word text to database ID."""
        if not self.cursor:
            return

        self.console.step("Building word ID mapping...")

        self.cursor.execute("SELECT id, word FROM words")
        rows = self.cursor.fetchall()

        self.word_to_id = {word: word_id for word_id, word in rows}
        self.console.show_word_mapping(len(self.word_to_id))

    def load_pos_unigrams(self, filename: str = DEFAULT_PIPELINE_POS_UNIGRAM_PROBS_FILE) -> None:
        """
        Load and insert POS unigrams from TSV file.

        Args:
            filename: POS unigram probabilities filename
        """
        if not self.cursor or not self.conn:
            raise PackagingError("Database not connected.")

        filepath = self.input_dir / filename
        self.console.info(f"Loading POS unigrams from: {filepath.name}")

        try:
            pos_data = []
            with open(filepath, encoding=DEFAULT_FILE_ENCODING) as f:
                next(f)  # Skip header
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) == 2:
                        pos, prob = parts
                        pos_data.append((pos, float(prob)))

            self.cursor.executemany(
                """
                INSERT INTO pos_unigrams (pos, probability) VALUES (?, ?)
                ON CONFLICT(pos) DO UPDATE SET probability=excluded.probability
                """,
                pos_data,
            )
            if not self._in_transaction:
                self.conn.commit()
            self.stats["pos_unigrams_inserted"] = len(pos_data)
            self.console.success(f"Inserted/Updated {len(pos_data):,} POS unigrams")
        except FileNotFoundError:
            self.console.warning(f"File not found: {filepath.name}. Skipping POS unigrams.")

    def load_pos_bigrams(self, filename: str = DEFAULT_PIPELINE_POS_BIGRAM_PROBS_FILE) -> None:
        """
        Load and insert POS bigrams from TSV file.

        Args:
            filename: POS bigram probabilities filename
        """
        if not self.cursor or not self.conn:
            raise PackagingError("Database not connected.")

        filepath = self.input_dir / filename
        self.console.info(f"Loading POS bigrams from: {filepath.name}")

        try:
            pos_data = []
            with open(filepath, encoding=DEFAULT_FILE_ENCODING) as f:
                next(f)  # Skip header
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) == 3:
                        pos1, pos2, prob = parts
                        pos_data.append((pos1, pos2, float(prob)))

            self.cursor.executemany(
                """
                INSERT INTO pos_bigrams (pos1, pos2, probability) VALUES (?, ?, ?)
                ON CONFLICT(pos1, pos2) DO UPDATE SET probability=excluded.probability
                """,
                pos_data,
            )
            if not self._in_transaction:
                self.conn.commit()
            self.stats["pos_bigrams_inserted"] = len(pos_data)
            self.console.success(f"Inserted/Updated {len(pos_data):,} POS bigrams")
        except FileNotFoundError:
            self.console.warning(f"File not found: {filepath.name}. Skipping POS bigrams.")

    def load_pos_trigrams(self, filename: str = DEFAULT_PIPELINE_POS_TRIGRAM_PROBS_FILE) -> None:
        """
        Load and insert POS trigrams from TSV file.

        Args:
            filename: POS trigram probabilities filename
        """
        if not self.cursor or not self.conn:
            raise PackagingError("Database not connected.")

        filepath = self.input_dir / filename
        self.console.info(f"Loading POS trigrams from: {filepath.name}")

        try:
            pos_data = []
            with open(filepath, encoding=DEFAULT_FILE_ENCODING) as f:
                next(f)  # Skip header
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) == 4:
                        pos1, pos2, pos3, prob = parts
                        pos_data.append((pos1, pos2, pos3, float(prob)))

            self.cursor.executemany(
                """
                INSERT INTO pos_trigrams (pos1, pos2, pos3, probability) VALUES (?, ?, ?, ?)
                ON CONFLICT(pos1, pos2, pos3) DO UPDATE SET probability=excluded.probability
                """,
                pos_data,
            )
            if not self._in_transaction:
                self.conn.commit()
            self.stats["pos_trigrams_inserted"] = len(pos_data)
            self.console.success(f"Inserted/Updated {len(pos_data):,} POS trigrams")
        except FileNotFoundError:
            self.console.warning(f"File not found: {filepath.name}. Skipping POS trigrams.")

    def load_bigrams(self, filename: str = DEFAULT_PIPELINE_BIGRAM_PROBS_FILE) -> None:
        """
        Load and insert bigrams from TSV file.

        Args:
            filename: Bigram probabilities filename
        """
        from rich.progress import (
            BarColumn,
            Progress,
            SpinnerColumn,
            TaskProgressColumn,
            TextColumn,
            TimeElapsedColumn,
            TimeRemainingColumn,
        )

        if not self.cursor or not self.conn:
            raise PackagingError("Database not connected.")

        filepath = self.input_dir / filename
        self.console.info(f"Loading bigrams from: {filepath.name}")

        try:
            # Get file size for progress bar
            file_size = filepath.stat().st_size
            bigrams_data = []
            skipped = 0

            if _HAS_TSV_READER:
                self.console.step("Loading bigrams (Cython optimized)...")
                # Optimized Cython path
                # Note: Progress bar skipped for performance
                for id1, id2, probability, count in tsv_reader_c.read_bigrams_tsv(
                    str(filepath), self.word_to_id
                ):
                    bigrams_data.append((id1, id2, probability, count))
            else:
                with open(filepath, encoding=DEFAULT_FILE_ENCODING) as f:
                    # Skip header
                    next(f)

                    # Prepare data with progress bar
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        BarColumn(),
                        TaskProgressColumn(),
                        TimeElapsedColumn(),
                        TextColumn("•"),
                        TextColumn("ETA"),
                        TimeRemainingColumn(),
                        console=self.console.console,
                        transient=True,
                    ) as progress:
                        task = progress.add_task("Loading bigrams...", total=file_size)

                        for line in f:
                            progress.update(task, advance=len(line.encode("utf-8")))
                            parts = line.strip().split("\t")
                            count = 0
                            if len(parts) >= 3:
                                word1, word2, probability = parts[:3]
                                if len(parts) >= 4:
                                    count = int(parts[3])

                                # Get word IDs
                                word1_id = self.word_to_id.get(word1)
                                word2_id = self.word_to_id.get(word2)

                                # Skip if either word not in database
                                if word1_id is None or word2_id is None:
                                    skipped += 1
                                    continue

                                bigrams_data.append((word1_id, word2_id, float(probability), count))

            # Batch insert
            limit = self.limits["bigrams"]
            if limit and len(bigrams_data) > limit:
                bigrams_data.sort(key=lambda x: x[2], reverse=True)
                bigrams_data = bigrams_data[:limit]
                self.console.step(f"Limiting to top {limit} bigrams")

            self.cursor.executemany(
                """
                INSERT INTO bigrams (word1_id, word2_id, probability, count) VALUES (?, ?, ?, ?)
                ON CONFLICT(word1_id, word2_id) DO UPDATE SET
                    probability=excluded.probability,
                    count=excluded.count
                """,
                bigrams_data,
            )
            if not self._in_transaction:
                self.conn.commit()

            self.stats["bigrams_inserted"] = len(bigrams_data)
            self.console.success(f"Inserted/Updated {len(bigrams_data):,} bigrams")
            if skipped > 0:
                self.console.step(f"Skipped {skipped:,} bigrams (words not in database)")

        except FileNotFoundError:
            self.console.error(f"File not found: {filepath}")
            raise

    def load_trigrams(self, filename: str = DEFAULT_PIPELINE_TRIGRAM_PROBS_FILE) -> None:
        """
        Load and insert trigrams from TSV file.

        Args:
            filename: Trigram probabilities filename
        """
        from rich.progress import (
            BarColumn,
            Progress,
            SpinnerColumn,
            TaskProgressColumn,
            TextColumn,
            TimeElapsedColumn,
            TimeRemainingColumn,
        )

        if not self.cursor or not self.conn:
            raise PackagingError("Database not connected.")

        filepath = self.input_dir / filename
        self.console.info(f"Loading trigrams from: {filepath.name}")

        try:
            # Get file size for progress bar
            file_size = filepath.stat().st_size
            trigrams_data = []
            skipped = 0

            if _HAS_TSV_READER:
                self.console.step("Loading trigrams (Cython optimized)...")
                # Optimized Cython path
                for id1, id2, id3, probability, count in tsv_reader_c.read_trigrams_tsv(
                    str(filepath), self.word_to_id
                ):
                    trigrams_data.append((id1, id2, id3, probability, count))
            else:
                with open(filepath, encoding=DEFAULT_FILE_ENCODING) as f:
                    # Skip header
                    next(f)

                    # Prepare data with progress bar
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        BarColumn(),
                        TaskProgressColumn(),
                        TimeElapsedColumn(),
                        TextColumn("•"),
                        TextColumn("ETA"),
                        TimeRemainingColumn(),
                        console=self.console.console,
                        transient=True,
                    ) as progress:
                        task = progress.add_task("Loading trigrams...", total=file_size)

                        for line in f:
                            progress.update(task, advance=len(line.encode("utf-8")))
                            parts = line.strip().split("\t")
                            count = 0
                            if len(parts) >= 4:
                                word1, word2, word3, probability = parts[:4]
                                if len(parts) >= 5:
                                    count = int(parts[4])

                                # Get word IDs
                                word1_id = self.word_to_id.get(word1)
                                word2_id = self.word_to_id.get(word2)
                                word3_id = self.word_to_id.get(word3)

                                # Skip if any word not in database
                                if word1_id is None or word2_id is None or word3_id is None:
                                    skipped += 1
                                    continue

                                trigrams_data.append(
                                    (word1_id, word2_id, word3_id, float(probability), count)
                                )

            # Batch insert
            limit = self.limits["trigrams"]
            if limit and len(trigrams_data) > limit:
                trigrams_data.sort(key=lambda x: x[3], reverse=True)
                trigrams_data = trigrams_data[:limit]
                self.console.step(f"Limiting to top {limit} trigrams")

            self.cursor.executemany(
                """
                INSERT INTO trigrams (word1_id, word2_id, word3_id, probability, count)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(word1_id, word2_id, word3_id) DO UPDATE SET
                    probability=excluded.probability,
                    count=excluded.count
                """,
                trigrams_data,
            )
            if not self._in_transaction:
                self.conn.commit()

            self.stats["trigrams_inserted"] = len(trigrams_data)
            self.console.success(f"Inserted/Updated {len(trigrams_data):,} trigrams")
            if skipped > 0:
                self.console.step(f"Skipped {skipped:,} trigrams (words not in database)")

        except FileNotFoundError:
            self.console.warning(f"File not found: {filepath.name}. Skipping trigrams load.")

    def store_metadata(self, key: str, value: str) -> None:
        """Store a key-value pair in the metadata table.

        Args:
            key: Metadata key (e.g., 'total_word_count')
            value: Metadata value as string
        """
        if not self.cursor or not self.conn:
            raise PackagingError("Database not connected.")

        self.cursor.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            (key, value),
        )
        if not self._in_transaction:
            self.conn.commit()

    def optimize_database(self) -> None:
        """Optimize database for read performance."""
        if not self.cursor or not self.conn:
            return

        self.console.info("Optimizing database...")

        # Analyze tables for query optimization
        self.cursor.execute("ANALYZE")
        self.conn.commit()

        # VACUUM requires autocommit (no implicit transaction).
        # Temporarily switch to autocommit, run VACUUM, then restore.
        prev_isolation = self.conn.isolation_level
        self.conn.isolation_level = None
        try:
            self.cursor.execute("VACUUM")
        finally:
            self.conn.isolation_level = prev_isolation

        # Restore WAL journal mode (VACUUM silently resets it)
        self.cursor.execute("PRAGMA journal_mode = WAL")
        self.conn.commit()
        self.console.success("Database optimized")

    def verify_database(self) -> None:
        """Verify database integrity and print sample data."""
        if not self.cursor:
            return

        self.console.step("Verifying database...")

        try:
            # Count records
            self.cursor.execute("SELECT COUNT(*) FROM syllables")
            syllable_count = self.cursor.fetchone()[0]

            self.cursor.execute("SELECT COUNT(*) FROM words")
            word_count = self.cursor.fetchone()[0]

            self.cursor.execute("SELECT COUNT(*) FROM bigrams")
            bigram_count = self.cursor.fetchone()[0]

            self.cursor.execute("SELECT COUNT(*) FROM trigrams")
            trigram_count = self.cursor.fetchone()[0]

            # Show counts as a summary table
            from ..utils.console import create_stats_table

            counts = {
                "Syllables": syllable_count,
                "Words": word_count,
                "Bigrams": bigram_count,
                "Trigrams": trigram_count,
            }
            self.console.console.print(create_stats_table(counts, "Record Counts"))

            # Sample syllables
            self.cursor.execute(
                """
                SELECT syllable, frequency
                FROM syllables
                ORDER BY frequency DESC
                LIMIT 5
            """
            )
            syllable_samples = [
                {"Syllable": row[0], "Frequency": f"{row[1]:,}"} for row in self.cursor.fetchall()
            ]
            self.console.show_sample_data("Sample Syllables (Top 5)", syllable_samples)

            # Sample words
            self.cursor.execute(
                """
                SELECT word, syllable_count, frequency, pos_tag
                FROM words
                ORDER BY frequency DESC
                LIMIT 5
            """
            )
            word_samples = [
                {
                    "Word": row[0],
                    "Syllables": str(row[1]),
                    "Frequency": f"{row[2]:,}",
                    "POS": row[3] or "-",
                }
                for row in self.cursor.fetchall()
            ]
            self.console.show_sample_data("Sample Words (Top 5)", word_samples)

            # Sample bigrams
            self.cursor.execute(
                """
                SELECT w1.word, w2.word, b.probability
                FROM bigrams b
                JOIN words w1 ON b.word1_id = w1.id
                JOIN words w2 ON b.word2_id = w2.id
                ORDER BY b.probability DESC
                LIMIT 5
            """
            )
            bigram_samples = [
                {"Word 1": row[0], "Word 2": row[1], "Probability": f"{row[2]:.4f}"}
                for row in self.cursor.fetchall()
            ]
            self.console.show_sample_data("Sample Bigrams (Top 5)", bigram_samples)

        except sqlite3.OperationalError as e:
            self.console.error(f"Database verification failed: {e}")
            self.logger.error(f"Database verification failed: {e}")
            return

    def close(self) -> None:
        """Close database connection and reset connection attributes."""
        if self.conn:
            self.conn.close()
            self.conn = None
            self.cursor = None
            self._in_transaction = False
            self._schema_manager = None
            self._pos_inference_manager = None
            self.console.success(f"Database saved to: {self.database_path}")

    def print_stats(self) -> None:
        """Print packaging statistics."""
        total_words = self.stats["words_inserted"] + self.stats["curated_words_inserted"]
        stats_dict: dict[str, int | str] = {
            "Syllables inserted": self.stats["syllables_inserted"],
            "Words inserted": total_words,
            "  - Corpus words": self.stats["words_inserted"],
            "  - Curated words": self.stats["curated_words_inserted"],
            "Bigrams inserted": self.stats["bigrams_inserted"],
            "Trigrams inserted": self.stats["trigrams_inserted"],
            "POS Unigrams inserted": self.stats["pos_unigrams_inserted"],
            "POS Bigrams inserted": self.stats["pos_bigrams_inserted"],
            "POS Trigrams inserted": self.stats["pos_trigrams_inserted"],
        }
        if self.database_path.exists():
            db_size = self.database_path.stat().st_size
            if db_size >= 1024 * 1024:
                stats_dict["Database size"] = f"{db_size / (1024 * 1024):.2f} MB"
            else:
                stats_dict["Database size"] = f"{db_size / 1024:.2f} KB"

        self.console.show_stats(stats_dict, "Database Packaging Statistics")
