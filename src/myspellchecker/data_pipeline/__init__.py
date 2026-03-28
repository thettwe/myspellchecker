"""
Data pipeline module for building the spell checker database from corpus text.

This module provides tools to ingest raw text, segment it, calculate frequencies,
and package it into a SQLite database compatible with the SpellChecker.

This module requires the 'build' optional dependencies:
    pip install myspellchecker[build]

Configuration:
    Pipeline behavior can be customized using PipelineConfig and related
    configuration dataclasses for fine-grained control over processing.

Example:
    >>> from myspellchecker.data_pipeline import Pipeline, PipelineConfig
    >>>
    >>> # Use defaults
    >>> pipeline = Pipeline()
    >>> pipeline.build_database(input_files, database_path)
    >>>
    >>> # Custom configuration
    >>> config = PipelineConfig(
    ...     batch_size=50000,
    ...     num_workers=8,
    ...     min_frequency=100,  # Higher threshold for cleaner data
    ... )
    >>> pipeline = Pipeline(config=config)
"""

__all__ = [
    # Pipeline and runner
    "Pipeline",
    "run_pipeline",
    # Configuration classes
    "PipelineConfig",
    "PipelineRuntimeFlags",
    "runtime_flags",
    "IngesterConfig",
    "SegmenterConfig",
    "FrequencyBuilderConfig",
    "PackagerConfig",
    # Reporter classes
    "PipelineReporter",
    "MockReporter",
    "ReporterInterface",
    # Component classes
    "CorpusIngester",
    "CorpusSegmenter",
    "FrequencyBuilder",
    "DatabasePackager",
    # Exceptions
    "IngestionError",
]

# Config classes have no heavy dependencies — import eagerly
from .config import (
    FrequencyBuilderConfig,
    IngesterConfig,
    PackagerConfig,
    PipelineConfig,
    PipelineRuntimeFlags,
    SegmenterConfig,
    runtime_flags,
)


# Lazy imports for classes that pull in pyarrow/xxhash/duckdb
def __getattr__(name):
    if name == "Pipeline":
        from .pipeline import Pipeline

        return Pipeline
    if name == "run_pipeline":
        from .pipeline import run_pipeline

        return run_pipeline
    if name == "DatabasePackager":
        from .database_packager import DatabasePackager

        return DatabasePackager
    if name == "FrequencyBuilder":
        from .frequency_builder import FrequencyBuilder

        return FrequencyBuilder
    if name in ("CorpusIngester", "IngestionError"):
        from . import ingester as _ingester

        return getattr(_ingester, name)
    if name == "CorpusSegmenter":
        from .segmenter import CorpusSegmenter

        return CorpusSegmenter
    if name in ("PipelineReporter", "MockReporter", "ReporterInterface"):
        from . import reporter as _reporter

        return getattr(_reporter, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
