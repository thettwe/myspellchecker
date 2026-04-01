"""
NER entity table query methods for SQLiteProvider.

Provides access to the ``ner_entities`` table which stores named entities
(persons, locations, organizations, etc.) for false-positive suppression
during spell checking.

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


class NEREntityMixin:
    """Mixin providing NER entity table queries for SQLiteProvider.

    Expects the host class to provide:
    - ``_execute_query() -> ContextManager[sqlite3.Connection]``
    - ``logger: logging.Logger``
    - ``_ner_entity_map`` attribute (initialized to ``None``).
    - ``_ner_lock: threading.Lock``
    """

    _ner_entity_map: dict[str, set[str]] | None
    _ner_lock: threading.Lock
    logger: logging.Logger

    def _execute_query(self) -> Generator[sqlite3.Connection, None, None]: ...  # type: ignore[override]

    def _ensure_ner_map(self) -> None:
        """Lazily load NER entities into memory on first access.

        Uses double-checked locking to prevent concurrent threads from
        redundantly loading the table.
        """
        if self._ner_entity_map is not None:
            return
        with self._ner_lock:
            if self._ner_entity_map is not None:
                return
            new_map: dict[str, set[str]] = {}
            try:
                with self._execute_query() as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT entity, entity_type FROM ner_entities")
                    for row in cursor:
                        entity, entity_type = row[0], row[1]
                        new_map.setdefault(entity, set()).add(entity_type)
                    self.logger.info("Pre-loaded NER entities: %d entries", len(new_map))
            except sqlite3.OperationalError as e:
                if not _is_missing_table_error(e):
                    raise
                self.logger.debug("ner_entities table not found; skipping")
            self._ner_entity_map = new_map

    def is_corpus_entity(self, word: str) -> bool:
        """Check if *word* is a known NER entity in the database."""
        self._ensure_ner_map()
        return word in (self._ner_entity_map or {})

    def get_entity_types(self, word: str) -> set[str]:
        """Return the set of entity types for *word*, or empty set."""
        self._ensure_ner_map()
        return (self._ner_entity_map or {}).get(word, set())
