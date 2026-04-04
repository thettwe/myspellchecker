"""
Bulk / batch query operations for SQLiteProvider.

Extracts all ``*_bulk`` methods and the shared ``_bulk_query`` helper so that
``sqlite.py`` stays focused on single-item lookups and connection management.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Callable
from typing import Any

from myspellchecker.providers._sqlite_schema import (
    SQLITE_MAX_BATCH_SIZE,
    _is_missing_column_error,
    _validate_batch_items,
)


class BulkQueryExecutor:
    """
    Executes batched SQL queries for the SQLiteProvider bulk API.

    The executor handles chunking, input validation, cache integration,
    and placeholder generation so that each public ``*_bulk`` method is
    a thin wrapper.

    Args:
        execute_query_fn: Context-manager callable that yields a
            ``sqlite3.Connection`` (typically ``SQLiteProvider._execute_query``).
        syllable_freq_cache: LRU cache for syllable frequencies.
        word_freq_cache: LRU cache for word frequencies.
        curated_min_frequency: Minimum effective frequency floor for curated
            words.  ``0`` means disabled.
        logger: Logger instance for warnings.
    """

    def __init__(
        self,
        execute_query_fn: Callable,
        syllable_freq_cache: Any,
        word_freq_cache: Any,
        curated_min_frequency: int,
        logger: logging.Logger,
    ) -> None:
        """Initialize with query function and cache references from SQLiteProvider."""
        self._execute_query = execute_query_fn
        self._syllable_freq_cache = syllable_freq_cache
        self._word_freq_cache = word_freq_cache
        self._curated_min_frequency = curated_min_frequency
        self._logger = logger
        # Lazy cache for PRAGMA table_info(words) — schema doesn't change at runtime
        self._words_columns: set[str] | None = None

    # ------------------------------------------------------------------
    # Core helper
    # ------------------------------------------------------------------

    def _bulk_query(
        self,
        items: list[str],
        sql_template: str,
        result_extractor: Callable[[sqlite3.Row], tuple[str, Any]],
        item_type: str = "item",
    ) -> dict[str, Any]:
        """
        Execute a batched SQL query over a list of string items.

        Handles chunking to respect SQLite's parameter limit, input
        validation, and placeholder generation.  Callers provide the SQL
        template and a function to extract key-value pairs from each
        result row.

        Args:
            items: Pre-filtered list of non-empty strings to query.
            sql_template: SQL with a ``{placeholders}`` slot for the IN clause.
            result_extractor: Callable ``(sqlite3.Row) -> (key, value)``.
            item_type: Label used in validation error messages.

        Returns:
            Dictionary of extracted results (only items found in DB).
        """
        found: dict[str, Any] = {}
        with self._execute_query() as conn:
            cursor = conn.cursor()
            for i in range(0, len(items), SQLITE_MAX_BATCH_SIZE):
                batch = items[i : i + SQLITE_MAX_BATCH_SIZE]
                _validate_batch_items(batch, item_type)
                placeholders = ",".join("?" for _ in batch)
                cursor.execute(sql_template.format(placeholders=placeholders), batch)
                for row in cursor.fetchall():
                    key, value = result_extractor(row)
                    found[key] = value
        return found

    # ------------------------------------------------------------------
    # Public bulk methods
    # ------------------------------------------------------------------

    def is_valid_syllables_bulk(self, syllables: list[str]) -> dict[str, bool]:
        """Check validity of multiple syllables using optimized batch query."""
        if not syllables:
            return {}

        valid_syllables = [s for s in syllables if s]
        if not valid_syllables:
            return {s: False for s in syllables}

        found = self._bulk_query(
            valid_syllables,
            "SELECT syllable FROM syllables WHERE syllable IN ({placeholders})",
            lambda row: (row["syllable"], True),
            item_type="syllable",
        )
        return {s: s in found for s in syllables}

    def is_valid_words_bulk(self, words: list[str]) -> dict[str, bool]:
        """Check validity of multiple words using optimized batch query."""
        if not words:
            return {}

        valid_words = [w for w in words if w]
        if not valid_words:
            return {w: False for w in words}

        found = self._bulk_query(
            valid_words,
            "SELECT word FROM words WHERE word IN ({placeholders})",
            lambda row: (row["word"], True),
            item_type="word",
        )
        return {w: w in found for w in words}

    def is_valid_vocabulary_bulk(self, words: list[str]) -> dict[str, bool]:
        """Check validity of multiple words as curated vocabulary."""
        if not words:
            return {}

        valid_words = [w for w in words if w]
        if not valid_words:
            return {w: False for w in words}

        try:
            found = self._bulk_query(
                valid_words,
                "SELECT word FROM words WHERE word IN ({placeholders}) AND is_curated = 1",
                lambda row: (row["word"], True),
                item_type="word",
            )
            return {w: w in found for w in words}
        except sqlite3.OperationalError as e:
            if not _is_missing_column_error(e):
                raise
            self._logger.warning("is_curated column not found. Using is_valid_words_bulk fallback.")
            return self.is_valid_words_bulk(words)

    def get_syllable_frequencies_bulk(self, syllables: list[str]) -> dict[str, int]:
        """Get corpus frequencies for multiple syllables using batch query."""
        if not syllables:
            return {}

        result: dict[str, int] = {}
        uncached: list[str] = []

        for s in syllables:
            cached = self._syllable_freq_cache.get(s)
            if cached is not None:
                result[s] = cached
            else:
                uncached.append(s)

        if uncached:
            found = self._bulk_query(
                uncached,
                "SELECT syllable, frequency FROM syllables WHERE syllable IN ({placeholders})",
                lambda row: (row["syllable"], row["frequency"]),
                item_type="syllable",
            )
            for s in uncached:
                freq = found.get(s, 0)
                result[s] = freq
                self._syllable_freq_cache.set(s, freq)

        return result

    def get_word_frequencies_bulk(self, words: list[str]) -> dict[str, int]:
        """Get corpus frequencies for multiple words using batch query."""
        if not words:
            return {}

        result: dict[str, int] = {}
        uncached: list[str] = []

        for w in words:
            cached = self._word_freq_cache.get(w)
            if cached is not None:
                result[w] = cached
            else:
                uncached.append(w)

        if uncached:
            found = self._bulk_query(
                uncached,
                "SELECT word, frequency FROM words WHERE word IN ({placeholders})",
                lambda row: (row["word"], row["frequency"]),
                item_type="word",
            )

            # Apply curated frequency floor (same logic as SQLiteProvider.get_word_frequency)
            if self._curated_min_frequency > 0:
                below_floor = [w for w in uncached if found.get(w, 0) < self._curated_min_frequency]
                if below_floor:
                    curated = self._bulk_query(
                        below_floor,
                        "SELECT word FROM words WHERE word IN ({placeholders}) AND is_curated = 1",
                        lambda row: (row["word"], True),
                        item_type="word",
                    )
                    for w in below_floor:
                        if w in curated:
                            found[w] = self._curated_min_frequency

            for w in uncached:
                freq = found.get(w, 0)
                result[w] = freq
                self._word_freq_cache.set(w, freq)

        return result

    def get_word_pos_bulk(self, words: list[str]) -> dict[str, str | None]:
        """Get POS tags for multiple words using batch query."""
        if not words:
            return {}

        if self._words_columns is None:
            with self._execute_query() as conn:
                cursor = conn.cursor()
                cursor.execute("PRAGMA table_info(words)")
                self._words_columns = {row["name"] for row in cursor.fetchall()}

        has_pos_tag = "pos_tag" in self._words_columns
        has_inferred = "inferred_pos" in self._words_columns
        if not has_pos_tag and not has_inferred:
            return {w: None for w in words}

        valid_words = [w for w in words if w]
        if not valid_words:
            return {w: None for w in words}

        # Build SQL based on available columns
        def _extract_pos_both(row: Any) -> tuple[str, str | None]:
            """Extract POS from pos_tag with inferred_pos fallback."""
            return (row["word"], row["pos_tag"] or row["inferred_pos"])

        def _extract_pos_tag(row: Any) -> tuple[str, str | None]:
            """Extract POS from pos_tag column only."""
            return (row["word"], row["pos_tag"])

        def _extract_inferred(row: Any) -> tuple[str, str | None]:
            """Extract POS from inferred_pos column only."""
            return (row["word"], row["inferred_pos"])

        if has_pos_tag and has_inferred:
            sql = "SELECT word, pos_tag, inferred_pos FROM words WHERE word IN ({placeholders})"
            extractor = _extract_pos_both
        elif has_pos_tag:
            sql = "SELECT word, pos_tag FROM words WHERE word IN ({placeholders})"
            extractor = _extract_pos_tag
        else:
            sql = "SELECT word, inferred_pos FROM words WHERE word IN ({placeholders})"
            extractor = _extract_inferred

        found = self._bulk_query(
            valid_words,
            sql,
            extractor,
            item_type="word",
        )
        return {w: found.get(w) for w in words}
