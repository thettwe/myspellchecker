"""
Enrichment table query methods for SQLiteProvider.

Extracts all enrichment/confusable/compound/collocation/register query
methods so that ``sqlite.py`` stays focused on core lookup operations.

Uses a mixin class that is composed into ``SQLiteProvider``.
"""

from __future__ import annotations

import sqlite3
import threading
from typing import TYPE_CHECKING

from myspellchecker.providers._sqlite_schema import _is_missing_table_error

if TYPE_CHECKING:
    import logging
    from collections.abc import Generator


class EnrichmentMixin:
    """Mixin providing enrichment table queries for SQLiteProvider.

    Expects the host class to provide:
    - ``_execute_query() -> ContextManager[sqlite3.Connection]``
    - ``logger: logging.Logger``
    - ``_confusable_map``, ``_compound_map``, ``_collocation_map``,
      ``_register_map`` attributes (initialized to ``None``).
    - ``_enrichment_lock: threading.Lock``
    """

    # These attributes are declared here for type-checking purposes only.
    # At runtime they are set by SQLiteProvider.__init__.
    _confusable_map: dict[str, list[tuple[str, str, float, float, int]]] | None
    _compound_map: dict[str, tuple[str, str, int, int, float]] | None
    _collocation_map: dict[tuple[str, str], float] | None
    _register_map: dict[str, str] | None
    _enrichment_lock: threading.Lock
    logger: logging.Logger

    def _execute_query(self) -> Generator[sqlite3.Connection, None, None]: ...  # type: ignore[override]

    # ------------------------------------------------------------------
    # Confusable pairs
    # ------------------------------------------------------------------

    def _ensure_confusable_map(self) -> None:
        """Lazily load confusable pairs into memory on first access.

        Uses double-checked locking to prevent concurrent threads from
        redundantly loading the entire table and exhausting the connection pool.
        """
        if self._confusable_map is not None:
            return
        with self._enrichment_lock:
            if self._confusable_map is not None:
                return
            new_map: dict[str, list[tuple[str, str, float, float, int]]] = {}
            try:
                with self._execute_query() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT word1, word2, confusion_type, context_overlap, "
                        "freq_ratio, suppress FROM confusable_pairs"
                    )
                    for row in cursor:
                        w1, w2, ctype, overlap, freq_ratio, suppress = (
                            row[0],
                            row[1],
                            row[2],
                            row[3],
                            row[4],
                            row[5],
                        )
                        new_map.setdefault(w1, []).append(
                            (w2, ctype, overlap, freq_ratio or 0.0, suppress)
                        )
                        new_map.setdefault(w2, []).append(
                            (w1, ctype, overlap, freq_ratio or 0.0, suppress)
                        )
                    self.logger.info("Pre-loaded confusable pairs for %d words", len(new_map))
            except sqlite3.OperationalError as e:
                if not _is_missing_table_error(e):
                    raise
                self.logger.debug("confusable_pairs table not found; skipping")
            self._confusable_map = new_map

    def get_confusable_pairs(self, word: str) -> list[tuple[str, str, float, float, int]]:
        """Get confusable pairs for a word.

        Returns list of (variant, confusion_type, context_overlap, freq_ratio, suppress) tuples.
        """
        self._ensure_confusable_map()
        if self._confusable_map is None:
            return []
        return self._confusable_map.get(word, [])

    def is_confusable_suppressed(self, word1: str, word2: str) -> bool:
        """Check if a confusable pair is suppressed (exempt from detection)."""
        self._ensure_confusable_map()
        if self._confusable_map is None:
            return False
        for variant, _ctype, _overlap, _fratio, suppress in self._confusable_map.get(word1, []):
            if variant == word2 and suppress:
                return True
        return False

    def get_confusable_context_overlap(self, word1: str, word2: str) -> float | None:
        """Get context overlap score for a confusable pair. None if not found."""
        self._ensure_confusable_map()
        if self._confusable_map is None:
            return None
        for variant, _ctype, overlap, _fratio, _suppress in self._confusable_map.get(word1, []):
            if variant == word2:
                return overlap
        return None

    # ------------------------------------------------------------------
    # Compound confusions
    # ------------------------------------------------------------------

    def _ensure_compound_map(self) -> None:
        """Lazily load compound confusions into memory."""
        if self._compound_map is not None:
            return
        with self._enrichment_lock:
            if self._compound_map is not None:
                return
            new_map: dict[str, tuple] = {}
            try:
                with self._execute_query() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT compound, part1, part2, compound_freq, split_freq, pmi "
                        "FROM compound_confusions"
                    )
                    for row in cursor:
                        new_map[row[0]] = (row[1], row[2], row[3], row[4], row[5])
                    self.logger.info("Pre-loaded %d compound confusions", len(new_map))
            except sqlite3.OperationalError as e:
                if not _is_missing_table_error(e):
                    raise
                self.logger.debug("compound_confusions table not found; skipping")
            self._compound_map = new_map

    def get_compound_confusion(self, compound: str) -> tuple[str, str, int, int, float] | None:
        """Get compound confusion data: (part1, part2, compound_freq, split_freq, pmi)."""
        self._ensure_compound_map()
        if self._compound_map is None:
            return None
        return self._compound_map.get(compound)

    # ------------------------------------------------------------------
    # Collocations
    # ------------------------------------------------------------------

    def _ensure_collocation_map(self) -> None:
        """Lazily load collocations into memory."""
        if self._collocation_map is not None:
            return
        with self._enrichment_lock:
            if self._collocation_map is not None:
                return
            new_map: dict[tuple[str, str], float] = {}
            try:
                with self._execute_query() as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT word1, word2, pmi FROM collocations")
                    for row in cursor:
                        new_map[(row[0], row[1])] = float(row[2])
                    self.logger.info("Pre-loaded %d collocations", len(new_map))
            except sqlite3.OperationalError as e:
                if not _is_missing_table_error(e):
                    raise
                self.logger.debug("collocations table not found; skipping")
            self._collocation_map = new_map

    def get_collocation_pmi(self, word1: str, word2: str) -> float | None:
        """Get PMI score for a word pair. None if not a known collocation."""
        self._ensure_collocation_map()
        if self._collocation_map is None:
            return None
        return self._collocation_map.get((word1, word2))

    # ------------------------------------------------------------------
    # Register tags
    # ------------------------------------------------------------------

    def _ensure_register_map(self) -> None:
        """Lazily load register tags into memory."""
        if self._register_map is not None:
            return
        with self._enrichment_lock:
            if self._register_map is not None:
                return
            new_map: dict[str, str] = {}
            try:
                with self._execute_query() as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT word, register FROM register_tags")
                    for row in cursor:
                        new_map[row[0]] = row[1]
                    self.logger.info("Pre-loaded %d register tags", len(new_map))
            except sqlite3.OperationalError as e:
                if not _is_missing_table_error(e):
                    raise
                self.logger.debug("register_tags table not found; skipping")
            self._register_map = new_map

    def get_register_tag(self, word: str) -> str | None:
        """Get register tag for a word: 'formal', 'informal', 'neutral', or None."""
        self._ensure_register_map()
        if self._register_map is None:
            return None
        return self._register_map.get(word)
