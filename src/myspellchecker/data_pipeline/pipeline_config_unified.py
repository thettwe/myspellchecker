"""
Unified Pipeline Configuration.

This module consolidates all pipeline configuration constants into a single
source of truth, eliminating duplication across pipeline modules.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ParallelizationConfig:
    """
    Configuration for parallel processing thresholds.

    Attributes:
        ingestion_threshold_mb: File size threshold for parallel ingestion (MB)
        segmentation_threshold_mb: File size threshold for parallel segmentation (MB)
        frequency_threshold_mb: File size threshold for parallel frequency building (MB)
        default_num_workers: Default number of worker processes (0 = auto-detect)
        max_workers: Maximum number of worker processes
    """

    # Minimum file sizes to trigger parallel processing (in MB)
    ingestion_threshold_mb: int = 100
    segmentation_threshold_mb: int = 50
    frequency_threshold_mb: int = 50

    # Worker configuration
    default_num_workers: int = 0  # 0 = auto-detect (min(8, cpu_count))
    max_workers: int = 8  # Maximum number of workers


@dataclass
class BatchSizeConfig:
    """
    Configuration for batch sizes across pipeline stages.

    Attributes:
        ingestion_batch: Batch size for ingestion (rows per flush)
        segmentation_batch: Batch size for segmentation (rows per flush)
        frequency_batch: Batch size for frequency calculations
        database_batch: Batch size for database inserts
        parallel_normalization_batch: Batch size for parallel normalization
    """

    # Batch sizes for different stages
    ingestion_batch: int = 10000
    segmentation_batch: int = 10000
    frequency_batch: int = 10000
    database_batch: int = 10000

    # Parallel processing batch sizes
    parallel_normalization_batch: int = 10000


@dataclass
class WorkerSettings:
    """
    Configuration for worker processes.

    Attributes:
        openmp_threads: Number of OpenMP threads per worker
        max_retries: Maximum number of retries for failed operations
        timeout_seconds: Timeout for worker operations
        enable_fork_optimization: Whether to use fork-based optimization on Unix
    """

    openmp_threads: int = 4  # Number of OpenMP threads per worker
    max_retries: int = 3  # Maximum number of retries
    timeout_seconds: int = 300  # 5 minutes timeout
    enable_fork_optimization: bool = True  # Use fork on Unix for faster startup


@dataclass
class WorkerTuningConfig:
    """
    Fine-grained worker resource tuning.

    Controls memory estimation, worker count limits, and batch size
    auto-tuning for the segmentation pipeline.

    Attributes:
        base_memory_per_worker_gb: Base memory overhead per worker (GB)
        usable_memory_ratio: Fraction of available memory to use
        max_workers_cap: Hard upper limit on worker count
        file_chunk_divisor: Divisor for sentences-per-chunk estimate
        file_limited_multiplier: Multiplier on base_workers for
            file-limited constraint
        large_file_threshold_mb: File size threshold for large batches
        large_file_batch: Batch size for large files
        medium_file_threshold_mb: File size threshold for medium batches
        medium_file_batch: Batch size for medium files
        small_file_batch: Batch size for small files
        low_memory_threshold_gb: Memory threshold for halving batch
        medium_memory_threshold_gb: Memory threshold for default batch
        high_memory_batch_multiplier: Batch multiplier when memory is
            plentiful
        max_batch_cap: Hard upper limit on batch size
    """

    base_memory_per_worker_gb: float = 0.3
    usable_memory_ratio: float = 0.7
    max_workers_cap: int = 16
    file_chunk_divisor: int = 20
    file_limited_multiplier: int = 2
    # Batch size auto-tuning
    large_file_threshold_mb: int = 500
    large_file_batch: int = 20000
    medium_file_threshold_mb: int = 100
    medium_file_batch: int = 15000
    small_file_batch: int = 10000
    low_memory_threshold_gb: int = 2
    medium_memory_threshold_gb: int = 4
    high_memory_batch_multiplier: int = 2
    max_batch_cap: int = 50000


@dataclass
class DuckDBResourceConfig:
    """
    DuckDB memory and disk resource limits.

    Controls how DuckDB allocates memory and disk for frequency
    building operations (general queries and trigram GROUP BY).

    Attributes:
        memory_ratio: Fraction of total RAM for general DuckDB memory
        memory_min_gb: Minimum memory limit (GB)
        memory_max_gb: Maximum memory limit (GB)
        trigram_memory_ratio: Fraction of total RAM for trigram queries
        trigram_memory_min_gb: Minimum trigram memory limit (GB)
        trigram_memory_max_gb: Maximum trigram memory limit (GB)
        allocator_flush_threshold_mb: DuckDB allocator flush threshold
        disk_usage_cap_ratio: Fraction of free disk to allow for temp
    """

    memory_ratio: float = 0.25
    memory_min_gb: int = 2
    memory_max_gb: int = 8
    trigram_memory_ratio: float = 0.20
    trigram_memory_min_gb: int = 2
    trigram_memory_max_gb: int = 5
    allocator_flush_threshold_mb: int = 512
    disk_usage_cap_ratio: float = 0.80


@dataclass
class UnifiedPipelineConfig:
    """
    Unified configuration for the entire data pipeline.

    This consolidates all configuration constants from:
    - ingester.py (MIN_PARALLEL_FILE_SIZE, DEFAULT_NUM_WORKERS, PARALLEL_BATCH_SIZE)
    - frequency_builder.py (MIN_PARALLEL_FILE_SIZE, DEFAULT_NUM_WORKERS)
    - segmenter.py (batch sizes, worker settings)
    - database_packager.py (batch sizes)

    Attributes:
        parallelization: Parallel processing configuration
        batch_sizes: Batch sizes for different stages
        worker_settings: Worker process configuration
        worker_tuning: Fine-grained worker resource tuning
        duckdb_resources: DuckDB memory and disk limits
        custom_settings: Custom configuration for extension

    Example:
        >>> config = UnifiedPipelineConfig()
        >>> batch_size = config.batch_sizes.ingestion_batch
    """

    parallelization: ParallelizationConfig = field(default_factory=ParallelizationConfig)
    batch_sizes: BatchSizeConfig = field(default_factory=BatchSizeConfig)
    worker_settings: WorkerSettings = field(default_factory=WorkerSettings)
    worker_tuning: WorkerTuningConfig = field(default_factory=WorkerTuningConfig)
    duckdb_resources: DuckDBResourceConfig = field(default_factory=DuckDBResourceConfig)
    custom_settings: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns:
            Dictionary representation of configuration
        """
        return {
            "parallelization": {
                "ingestion_threshold_mb": self.parallelization.ingestion_threshold_mb,
                "segmentation_threshold_mb": self.parallelization.segmentation_threshold_mb,
                "frequency_threshold_mb": self.parallelization.frequency_threshold_mb,
                "default_num_workers": self.parallelization.default_num_workers,
                "max_workers": self.parallelization.max_workers,
            },
            "batch_sizes": {
                "ingestion_batch": self.batch_sizes.ingestion_batch,
                "segmentation_batch": self.batch_sizes.segmentation_batch,
                "frequency_batch": self.batch_sizes.frequency_batch,
                "database_batch": self.batch_sizes.database_batch,
                "parallel_normalization_batch": self.batch_sizes.parallel_normalization_batch,
            },
            "worker_settings": {
                "openmp_threads": self.worker_settings.openmp_threads,
                "max_retries": self.worker_settings.max_retries,
                "timeout_seconds": self.worker_settings.timeout_seconds,
                "enable_fork_optimization": (self.worker_settings.enable_fork_optimization),
            },
            "worker_tuning": {
                "base_memory_per_worker_gb": (self.worker_tuning.base_memory_per_worker_gb),
                "usable_memory_ratio": (self.worker_tuning.usable_memory_ratio),
                "max_workers_cap": self.worker_tuning.max_workers_cap,
                "file_chunk_divisor": (self.worker_tuning.file_chunk_divisor),
                "file_limited_multiplier": (self.worker_tuning.file_limited_multiplier),
                "large_file_threshold_mb": (self.worker_tuning.large_file_threshold_mb),
                "large_file_batch": self.worker_tuning.large_file_batch,
                "medium_file_threshold_mb": (self.worker_tuning.medium_file_threshold_mb),
                "medium_file_batch": self.worker_tuning.medium_file_batch,
                "small_file_batch": self.worker_tuning.small_file_batch,
                "low_memory_threshold_gb": (self.worker_tuning.low_memory_threshold_gb),
                "medium_memory_threshold_gb": (self.worker_tuning.medium_memory_threshold_gb),
                "high_memory_batch_multiplier": (self.worker_tuning.high_memory_batch_multiplier),
                "max_batch_cap": self.worker_tuning.max_batch_cap,
            },
            "duckdb_resources": {
                "memory_ratio": (self.duckdb_resources.memory_ratio),
                "memory_min_gb": self.duckdb_resources.memory_min_gb,
                "memory_max_gb": self.duckdb_resources.memory_max_gb,
                "trigram_memory_ratio": (self.duckdb_resources.trigram_memory_ratio),
                "trigram_memory_min_gb": (self.duckdb_resources.trigram_memory_min_gb),
                "trigram_memory_max_gb": (self.duckdb_resources.trigram_memory_max_gb),
                "allocator_flush_threshold_mb": (
                    self.duckdb_resources.allocator_flush_threshold_mb
                ),
                "disk_usage_cap_ratio": (self.duckdb_resources.disk_usage_cap_ratio),
            },
            "custom_settings": self.custom_settings,
        }


# Convenience functions for centralized configuration
def get_default_num_workers() -> int:
    """
    Get default number of workers (replaces module-level DEFAULT_NUM_WORKERS).

    Returns:
        Number of workers (min of 8 and CPU count)
    """
    return min(8, os.cpu_count() or 4)


def get_ingestion_parallel_threshold() -> int:
    """
    Get ingestion parallel threshold in bytes (replaces ingester.MIN_PARALLEL_FILE_SIZE).

    Returns:
        Threshold in bytes (100 MB)
    """
    return 100 * 1024 * 1024


__all__ = [
    "ParallelizationConfig",
    "BatchSizeConfig",
    "WorkerSettings",
    "WorkerTuningConfig",
    "DuckDBResourceConfig",
    "UnifiedPipelineConfig",
    # Convenience functions
    "get_default_num_workers",
    "get_ingestion_parallel_threshold",
]
