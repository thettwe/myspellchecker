"""Checkpoint management mixin for FrequencyBuilder.

Handles Parquet checkpoint save/load/recovery for crash resilience
during long-running frequency counting operations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

# Number of hash-partitioned buckets for incremental n-gram checkpoints.
# Each bucket processes ~2% of data, allowing crash recovery without losing
# hours of work. Bucket count must stay consistent within a build run.
_NGRAM_BUCKET_COUNT = 50


class CheckpointMixin:
    """Checkpoint save/load/recovery methods for FrequencyBuilder.

    Provides Parquet-based checkpointing with per-step and per-bucket
    granularity.  All methods assume ``self.logger`` is available
    (set by ``FrequencyBuilder.__init__``).
    """

    logger: Any

    def _save_checkpoint(self, duckdb_temp_dir: Path, name: str, columns: dict[str, list]) -> None:
        """Save frequency data to a Parquet checkpoint file."""
        import pyarrow.parquet as pq

        path = duckdb_temp_dir / f"checkpoint_{name}.parquet"
        tbl = pa.table(columns)
        pq.write_table(tbl, str(path), compression="snappy")
        self.logger.debug("Saved checkpoint: %s (%d rows)", path.name, tbl.num_rows)

    def _load_checkpoint(self, duckdb_temp_dir: Path, name: str) -> pa.Table | None:
        """Load a Parquet checkpoint. Returns Arrow table or None."""
        import pyarrow.parquet as pq

        path = duckdb_temp_dir / f"checkpoint_{name}.parquet"
        if path.exists():
            return pq.read_table(str(path))
        return None
