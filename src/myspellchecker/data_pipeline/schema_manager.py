"""
Schema management for the database packager.

This module handles database schema creation, indexes, and migrations.
"""

from __future__ import annotations

import sqlite3

from ..core.exceptions import PipelineError
from ..utils.console import PipelineConsole
from ..utils.logging_utils import get_logger

__all__ = [
    "SchemaManager",
]


class SchemaManager:
    """
    Manages database schema creation and migrations.

    This class is responsible for:
    - Creating database tables
    - Creating indexes for fast lookups
    - Ensuring migration columns exist
    - Verifying schema integrity
    """

    # Table definitions
    TABLES = {
        "syllables": """
            CREATE TABLE IF NOT EXISTS syllables (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                syllable TEXT UNIQUE NOT NULL,
                frequency INTEGER DEFAULT 0
            )
        """,
        "words": """
            CREATE TABLE IF NOT EXISTS words (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                word TEXT UNIQUE NOT NULL,
                syllable_count INTEGER,
                frequency INTEGER DEFAULT 0,
                pos_tag TEXT,
                is_curated INTEGER DEFAULT 0,
                inferred_pos TEXT,
                inferred_confidence REAL,
                inferred_source TEXT
            )
        """,
        "bigrams": """
            CREATE TABLE IF NOT EXISTS bigrams (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                word1_id INTEGER,
                word2_id INTEGER,
                probability REAL DEFAULT 0.0,
                count INTEGER DEFAULT 0,
                FOREIGN KEY(word1_id) REFERENCES words(id),
                FOREIGN KEY(word2_id) REFERENCES words(id),
                UNIQUE(word1_id, word2_id)
            )
        """,
        "trigrams": """
            CREATE TABLE IF NOT EXISTS trigrams (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                word1_id INTEGER,
                word2_id INTEGER,
                word3_id INTEGER,
                probability REAL DEFAULT 0.0,
                count INTEGER DEFAULT 0,
                FOREIGN KEY(word1_id) REFERENCES words(id),
                FOREIGN KEY(word2_id) REFERENCES words(id),
                FOREIGN KEY(word3_id) REFERENCES words(id),
                UNIQUE(word1_id, word2_id, word3_id)
            )
        """,
        "fourgrams": """
            CREATE TABLE IF NOT EXISTS fourgrams (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                word1_id INTEGER,
                word2_id INTEGER,
                word3_id INTEGER,
                word4_id INTEGER,
                probability REAL DEFAULT 0.0,
                count INTEGER DEFAULT 0,
                FOREIGN KEY(word1_id) REFERENCES words(id),
                FOREIGN KEY(word2_id) REFERENCES words(id),
                FOREIGN KEY(word3_id) REFERENCES words(id),
                FOREIGN KEY(word4_id) REFERENCES words(id),
                UNIQUE(word1_id, word2_id, word3_id, word4_id)
            )
        """,
        "fivegrams": """
            CREATE TABLE IF NOT EXISTS fivegrams (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                word1_id INTEGER,
                word2_id INTEGER,
                word3_id INTEGER,
                word4_id INTEGER,
                word5_id INTEGER,
                probability REAL DEFAULT 0.0,
                count INTEGER DEFAULT 0,
                FOREIGN KEY(word1_id) REFERENCES words(id),
                FOREIGN KEY(word2_id) REFERENCES words(id),
                FOREIGN KEY(word3_id) REFERENCES words(id),
                FOREIGN KEY(word4_id) REFERENCES words(id),
                FOREIGN KEY(word5_id) REFERENCES words(id),
                UNIQUE(word1_id, word2_id, word3_id, word4_id, word5_id)
            )
        """,
        "pos_unigrams": """
            CREATE TABLE IF NOT EXISTS pos_unigrams (
                pos TEXT UNIQUE NOT NULL,
                probability REAL DEFAULT 0.0
            )
        """,
        "pos_bigrams": """
            CREATE TABLE IF NOT EXISTS pos_bigrams (
                pos1 TEXT NOT NULL,
                pos2 TEXT NOT NULL,
                probability REAL DEFAULT 0.0,
                UNIQUE(pos1, pos2)
            )
        """,
        "pos_trigrams": """
            CREATE TABLE IF NOT EXISTS pos_trigrams (
                pos1 TEXT NOT NULL,
                pos2 TEXT NOT NULL,
                pos3 TEXT NOT NULL,
                probability REAL DEFAULT 0.0,
                UNIQUE(pos1, pos2, pos3)
            )
        """,
        "processed_files": """
            CREATE TABLE IF NOT EXISTS processed_files (
                path TEXT PRIMARY KEY,
                mtime REAL,
                size INTEGER
            )
        """,
        "metadata": """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """,
        "confusable_pairs": """
            CREATE TABLE IF NOT EXISTS confusable_pairs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                word1 TEXT NOT NULL,
                word2 TEXT NOT NULL,
                confusion_type TEXT NOT NULL,
                context_overlap REAL DEFAULT 0.0,
                freq_ratio REAL,
                suppress INTEGER DEFAULT 0,
                source TEXT DEFAULT 'mined',
                UNIQUE(word1, word2, confusion_type)
            )
        """,
        "compound_confusions": """
            CREATE TABLE IF NOT EXISTS compound_confusions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                compound TEXT NOT NULL,
                part1 TEXT NOT NULL,
                part2 TEXT NOT NULL,
                compound_freq INTEGER DEFAULT 0,
                split_freq INTEGER DEFAULT 0,
                pmi REAL DEFAULT 0.0,
                UNIQUE(compound, part1, part2)
            )
        """,
        "collocations": """
            CREATE TABLE IF NOT EXISTS collocations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                word1 TEXT NOT NULL,
                word2 TEXT NOT NULL,
                pmi REAL NOT NULL,
                npmi REAL,
                count INTEGER NOT NULL,
                UNIQUE(word1, word2)
            )
        """,
        "register_tags": """
            CREATE TABLE IF NOT EXISTS register_tags (
                word TEXT PRIMARY KEY,
                register TEXT NOT NULL,
                confidence REAL DEFAULT 0.0,
                formal_count INTEGER DEFAULT 0,
                informal_count INTEGER DEFAULT 0
            )
        """,
    }

    # Index definitions
    INDEXES = {
        "idx_syllables_text": (
            "CREATE INDEX IF NOT EXISTS idx_syllables_text ON syllables (syllable)"
        ),
        "idx_words_text": ("CREATE INDEX IF NOT EXISTS idx_words_text ON words (word)"),
        "idx_bigrams_w1_w2": (
            "CREATE INDEX IF NOT EXISTS idx_bigrams_w1_w2 ON bigrams (word1_id, word2_id)"
        ),
        "idx_trigrams_w1_w2_w3": (
            "CREATE INDEX IF NOT EXISTS idx_trigrams_w1_w2_w3 "
            "ON trigrams (word1_id, word2_id, word3_id)"
        ),
        "idx_fourgrams_w1_w2_w3_w4": (
            "CREATE INDEX IF NOT EXISTS idx_fourgrams_w1_w2_w3_w4 "
            "ON fourgrams (word1_id, word2_id, word3_id, word4_id)"
        ),
        "idx_fivegrams_w1_w2_w3_w4_w5": (
            "CREATE INDEX IF NOT EXISTS idx_fivegrams_w1_w2_w3_w4_w5 "
            "ON fivegrams (word1_id, word2_id, word3_id, word4_id, word5_id)"
        ),
        "idx_words_frequency": (
            "CREATE INDEX IF NOT EXISTS idx_words_frequency ON words (frequency)"
        ),
        "idx_confusable_word1": (
            "CREATE INDEX IF NOT EXISTS idx_confusable_word1 ON confusable_pairs (word1)"
        ),
        "idx_confusable_word2": (
            "CREATE INDEX IF NOT EXISTS idx_confusable_word2 ON confusable_pairs (word2)"
        ),
        "idx_compound_word": (
            "CREATE INDEX IF NOT EXISTS idx_compound_word ON compound_confusions (compound)"
        ),
        "idx_compound_parts": (
            "CREATE INDEX IF NOT EXISTS idx_compound_parts ON compound_confusions (part1, part2)"
        ),
        "idx_colloc_word1": ("CREATE INDEX IF NOT EXISTS idx_colloc_word1 ON collocations (word1)"),
        "idx_colloc_word2": ("CREATE INDEX IF NOT EXISTS idx_colloc_word2 ON collocations (word2)"),
    }

    # POS inference columns for migration
    POS_INFERENCE_COLUMNS = ["inferred_pos", "inferred_confidence", "inferred_source"]

    def __init__(
        self,
        conn: sqlite3.Connection,
        cursor: sqlite3.Cursor,
        console: PipelineConsole | None = None,
    ):
        """
        Initialize the schema manager.

        Args:
            conn: SQLite database connection
            cursor: SQLite database cursor
            console: Optional console for output (creates default if not provided)
        """
        self.logger = get_logger(__name__)
        self.conn = conn
        self.cursor = cursor
        self.console = console or PipelineConsole()
        self._verified_tables: list[str] = []

    def create_schema(self, in_transaction: bool = False, defer_indexes: bool = False) -> list[str]:
        """
        Create database schema with tables.

        Creates all required tables in a transaction to ensure atomicity.
        If any table creation fails, all changes are rolled back.

        Args:
            in_transaction: If True, don't start/commit own transaction
            defer_indexes: If True, skip index creation (for bulk loading
                performance — call create_indexes() after data load)

        Returns:
            List of verified table names

        Raises:
            RuntimeError: If schema creation fails (after rollback).
        """
        self._verified_tables = []
        started_transaction = False

        try:
            # Begin transaction if not already in one
            if not in_transaction:
                self.conn.execute("BEGIN")
                started_transaction = True

            # Create each table
            for table_name, create_sql in self.TABLES.items():
                self.cursor.execute(create_sql)
                self._verified_tables.append(table_name)

            # Show schema table with all verified tables
            self.console.console.print()
            for table in self._verified_tables:
                self.console.table_verified(table)
            self.console.show_schema_summary()

            # Create indexes (unless deferred for bulk loading)
            if not defer_indexes:
                self.create_indexes()

            # Commit schema (only if we started our own transaction)
            if started_transaction:
                self.conn.commit()
            self.console.success("Schema verified successfully")

            return self._verified_tables

        except Exception as e:
            # Rollback on any error to ensure database consistency
            if started_transaction:
                self.console.warning(f"Schema creation failed, rolling back: {e}")
                try:
                    self.conn.rollback()
                except Exception as re:
                    self.logger.debug("Rollback also failed: %s", re)
            # Re-raise with context
            raise PipelineError(
                f"Failed to create database schema after creating "
                f"{len(self._verified_tables)} tables. Error: {e}"
            ) from e

    def create_indexes(self):
        """Create indexes for fast lookups."""
        self.console.info("Creating indexes...")

        for _index_name, create_sql in self.INDEXES.items():
            self.cursor.execute(create_sql)

        self.console.success("Indexes verified")

    def ensure_inferred_pos_columns(self):
        """
        Ensure the inferred_pos columns exist in the words table.

        Adds the columns if they don't exist (for database migration).
        """
        # Check if columns exist
        self.cursor.execute("PRAGMA table_info(words)")
        columns = {row[1] for row in self.cursor.fetchall()}

        # Add missing columns
        _ALLOWED_COLUMNS: dict[str, str] = {
            "inferred_pos": "TEXT",
            "inferred_confidence": "REAL",
            "inferred_source": "TEXT",
        }

        for col_name in self.POS_INFERENCE_COLUMNS:
            if col_name not in columns:
                if col_name not in _ALLOWED_COLUMNS:
                    raise ValueError(f"Unknown column name: {col_name!r}")
                col_type = _ALLOWED_COLUMNS[col_name]
                # DDL cannot use parameter placeholders; col_name and col_type
                # are validated against the hardcoded _ALLOWED_COLUMNS whitelist.
                self.cursor.execute(f"ALTER TABLE words ADD COLUMN {col_name} {col_type}")

        self.conn.commit()
