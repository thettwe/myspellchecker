"""
POS inference management for the database packager.

This module handles applying rule-based POS inference to words
and tracking POS coverage statistics.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from ..core.constants import DEFAULT_BATCH_SIZE
from ..utils.console import PipelineConsole
from ..utils.logging_utils import get_logger


class POSInferenceManager:
    """
    Manages POS inference for the database.

    This class is responsible for:
    - Applying rule-based POS inference to words
    - Tracking POS coverage statistics
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        cursor: sqlite3.Cursor,
        console: PipelineConsole | None = None,
    ):
        """
        Initialize the POS inference manager.

        Args:
            conn: SQLite database connection
            cursor: SQLite database cursor
            console: Optional console for output (creates default if not provided)
        """
        self.logger = get_logger(__name__)
        self.conn = conn
        self.cursor = cursor
        self.console = console or PipelineConsole()

    def apply_inferred_pos(
        self,
        min_frequency: int = 0,
        skip_tagged: bool = True,
        min_confidence: float = 0.0,
        in_transaction: bool = False,
    ) -> dict[str, Any]:
        """
        Apply rule-based POS inference to words without POS tags.

        Uses the POSInferenceEngine to infer POS tags for untagged words
        based on morphological patterns (prefixes, suffixes), numeral detection,
        proper noun patterns, and the ambiguous words registry.

        Args:
            min_frequency: Minimum word frequency threshold for inference.
                          Words below this frequency are skipped. Default 0.
            skip_tagged: If True, skip words that already have pos_tag.
                        If False, also infer for tagged words (overwrites inferred_pos).
            min_confidence: Minimum confidence threshold. Only apply inference
                           if confidence >= this value. Default 0.0.
            in_transaction: If True, don't commit (caller manages transaction).

        Returns:
            Dictionary with inference statistics:
            - total_words: Total words processed
            - inferred: Number of words with successful inference
            - skipped_tagged: Number of words skipped (already had pos_tag)
            - skipped_low_conf: Number skipped due to low confidence
            - ambiguous: Number of ambiguous words (multi-POS)
            - by_source: Dict of counts by inference source

        Example:
            >>> manager = POSInferenceManager(conn, cursor, console)
            >>> stats = manager.apply_inferred_pos(min_frequency=5)
            >>> print(f"Inferred POS for {stats['inferred']} words")
        """
        # Import here to avoid circular dependency
        from ..algorithms.pos_inference import POSInferenceEngine

        self.console.info("Applying rule-based POS inference...")

        # Initialize engine
        engine = POSInferenceEngine()

        # Build query based on options
        params = []
        if skip_tagged:
            query = """
                SELECT id, word, frequency, pos_tag
                FROM words
                WHERE (pos_tag IS NULL OR pos_tag = '')
            """
        else:
            query = "SELECT id, word, frequency, pos_tag FROM words WHERE 1=1"

        if min_frequency > 0:
            query += " AND frequency >= ?"
            params.append(min_frequency)

        self.cursor.execute(query, params)
        rows = self.cursor.fetchall()

        # Use typed local counters for clean type inference
        total_words = len(rows)
        inferred_count = 0
        skipped_tagged_count = 0
        skipped_low_conf_count = 0
        ambiguous_count = 0
        by_source: dict[str, int] = {}

        if not rows:
            self.console.info("No words need POS inference")
            return {
                "total_words": total_words,
                "inferred": inferred_count,
                "skipped_tagged": skipped_tagged_count,
                "skipped_low_conf": skipped_low_conf_count,
                "ambiguous": ambiguous_count,
                "by_source": by_source,
            }

        self.console.step(f"Processing {total_words:,} words for POS inference...")

        # Process in batches for efficiency
        updates = []
        for word_id, word, _frequency, existing_pos in rows:
            # Skip if already tagged and skip_tagged is True
            if skip_tagged and existing_pos:
                skipped_tagged_count += 1
                continue

            # Infer POS
            result = engine.infer_pos(word)

            # Skip if confidence too low
            if result.confidence < min_confidence:
                skipped_low_conf_count += 1
                continue

            # Skip if no inference possible
            if not result.inferred_pos:
                continue

            # Track statistics
            inferred_count += 1
            if result.is_ambiguous:
                ambiguous_count += 1

            source_name = result.source.value
            by_source[source_name] = by_source.get(source_name, 0) + 1

            # Prepare update
            # Store multi-POS as pipe-separated string
            multi_pos = result.to_multi_pos_string()
            updates.append(
                (
                    multi_pos,
                    result.confidence,
                    source_name,
                    word_id,
                )
            )

        # Apply updates in batches
        if updates:
            self.console.step(f"Updating {len(updates):,} words with inferred POS...")
            for i in range(0, len(updates), DEFAULT_BATCH_SIZE):
                chunk = updates[i : i + DEFAULT_BATCH_SIZE]
                self.cursor.executemany(
                    """
                    UPDATE words
                    SET inferred_pos = ?,
                        inferred_confidence = ?,
                        inferred_source = ?
                    WHERE id = ?
                    """,
                    chunk,
                )

            if not in_transaction:
                self.conn.commit()

        # Log summary
        self.console.success(f"POS inference complete: {inferred_count:,} words inferred")
        if ambiguous_count:
            self.console.info(f"  - {ambiguous_count:,} ambiguous words (multi-POS)")
        if by_source:
            for source, count in sorted(by_source.items()):
                self.console.info(f"  - {source}: {count:,}")

        return {
            "total_words": total_words,
            "inferred": inferred_count,
            "skipped_tagged": skipped_tagged_count,
            "skipped_low_conf": skipped_low_conf_count,
            "ambiguous": ambiguous_count,
            "by_source": by_source,
        }

    def get_pos_coverage_stats(self) -> dict[str, int]:
        """
        Get statistics on POS tag coverage in the database.

        Returns:
            Dictionary with:
            - total_words: Total words in database
            - with_pos_tag: Words with pos_tag from seed
            - with_inferred_pos: Words with inferred POS
            - combined_coverage: Words with either pos_tag or inferred_pos
            - no_pos: Words without any POS information
            - ambiguous: Words with multi-POS (contains |)
        """
        stats = {}

        # Total words
        self.cursor.execute("SELECT COUNT(*) FROM words")
        stats["total_words"] = self.cursor.fetchone()[0]

        # With pos_tag (from seed)
        self.cursor.execute(
            "SELECT COUNT(*) FROM words WHERE pos_tag IS NOT NULL AND pos_tag != ''"
        )
        stats["with_pos_tag"] = self.cursor.fetchone()[0]

        # With inferred_pos
        self.cursor.execute(
            "SELECT COUNT(*) FROM words WHERE inferred_pos IS NOT NULL AND inferred_pos != ''"
        )
        stats["with_inferred_pos"] = self.cursor.fetchone()[0]

        # Combined coverage (either has pos_tag or inferred_pos)
        self.cursor.execute(
            """
            SELECT COUNT(*) FROM words
            WHERE (pos_tag IS NOT NULL AND pos_tag != '')
               OR (inferred_pos IS NOT NULL AND inferred_pos != '')
            """
        )
        stats["combined_coverage"] = self.cursor.fetchone()[0]

        # No POS at all
        stats["no_pos"] = stats["total_words"] - stats["combined_coverage"]

        # Ambiguous words (multi-POS)
        self.cursor.execute(
            """
            SELECT COUNT(*) FROM words
            WHERE (pos_tag LIKE '%|%')
               OR (inferred_pos LIKE '%|%')
            """
        )
        stats["ambiguous"] = self.cursor.fetchone()[0]

        return stats
