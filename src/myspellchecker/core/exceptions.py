"""
Custom exceptions for the myspellchecker library.

This module defines a hierarchy of exceptions to allow library consumers
to catch specific errors (e.g., configuration issues, data loading failures)
rather than relying on generic exceptions.

Exception Hierarchy:
    MyanmarSpellcheckError (base)
    ├── ConfigurationError
    │   └── InvalidConfigError
    ├── DataLoadingError
    │   └── MissingDatabaseError
    ├── ProcessingError
    │   ├── ValidationError
    │   ├── TokenizationError
    │   └── NormalizationError
    ├── ProviderError
    │   └── ConnectionPoolError
    ├── PipelineError
    │   ├── IngestionError
    │   └── PackagingError
    ├── ModelError
    │   ├── ModelLoadError
    │   └── InferenceError
    ├── MissingDependencyError
    ├── InsufficientStorageError
    └── CacheError

Example:
    >>> from myspellchecker.core.exceptions import (
    ...     MyanmarSpellcheckError,
    ...     ValidationError,
    ...     MissingDatabaseError,
    ... )
    >>> try:
    ...     checker.check(text)
    ... except ValidationError as e:
    ...     print(f"Invalid input: {e}")
    ... except MissingDatabaseError as e:
    ...     print(f"Database not found: {e}")
    ... except MyanmarSpellcheckError as e:
    ...     print(f"Spell check error: {e}")
"""

from __future__ import annotations

__all__ = [
    "CacheError",
    "ConfigurationError",
    "ConnectionPoolError",
    "DataLoadingError",
    "InferenceError",
    "IngestionError",
    "InsufficientStorageError",
    "InvalidConfigError",
    "MissingDatabaseError",
    "MissingDependencyError",
    "ModelError",
    "ModelLoadError",
    "MyanmarSpellcheckError",
    "NormalizationError",
    "PackagingError",
    "PipelineError",
    "ProcessingError",
    "ProviderError",
    "TokenizationError",
    "ValidationError",
]


class MyanmarSpellcheckError(Exception):
    """
    Base exception class for all errors in the myspellchecker library.

    All library-specific exceptions inherit from this class, allowing
    users to catch all myspellchecker errors with a single handler.

    Example:
        >>> try:
        ...     checker.check(text)
        ... except MyanmarSpellcheckError as e:
        ...     print(f"Spell check error: {e}")
    """



# =============================================================================
# Configuration Errors
# =============================================================================


class ConfigurationError(MyanmarSpellcheckError):
    """
    Raised when the library configuration is invalid.

    This includes invalid parameter values, incompatible settings,
    or missing required configuration.
    """



class InvalidConfigError(ConfigurationError):
    """
    Raised when a specific configuration value is invalid.

    Example:
        >>> raise InvalidConfigError("max_edit_distance must be >= 0, got -1")
    """



# =============================================================================
# Data Loading Errors
# =============================================================================


class DataLoadingError(MyanmarSpellcheckError):
    """
    Raised when dictionary data, models, or resources fail to load.

    This includes database files, model weights, configuration files,
    and other external resources required for operation.
    """



# =============================================================================
# Processing Errors
# =============================================================================


class ProcessingError(MyanmarSpellcheckError):
    """
    Raised when text processing or spell checking fails unexpectedly.

    This is a general error for processing failures. More specific
    subclasses should be used when possible.
    """



class ValidationError(ProcessingError):
    """
    Raised when text validation fails.

    This includes invalid Myanmar text, encoding issues,
    or text that violates expected patterns.

    Example:
        >>> raise ValidationError("Text contains invalid Myanmar characters")
    """



class TokenizationError(ProcessingError):
    """
    Raised when text tokenization/segmentation fails.

    This includes failures in syllable breaking, word segmentation,
    or sentence splitting.

    Example:
        >>> raise TokenizationError("Failed to segment text into syllables")
    """



class NormalizationError(ProcessingError):
    """
    Raised when text normalization fails.

    This includes failures in Unicode normalization, encoding conversion
    (e.g., Zawgyi to Unicode), or character standardization.

    Example:
        >>> raise NormalizationError("Failed to convert Zawgyi to Unicode")
    """



# =============================================================================
# Provider Errors
# =============================================================================


class ProviderError(MyanmarSpellcheckError):
    """
    Raised when a dictionary provider encounters an error.

    This includes database connection issues, query failures,
    and provider-specific errors.
    """



class ConnectionPoolError(ProviderError):
    """
    Raised when the connection pool encounters an error.

    This includes pool exhaustion, connection creation failures,
    and health check failures.

    Example:
        >>> raise ConnectionPoolError("Connection pool exhausted")
    """



# =============================================================================
# Pipeline Errors
# =============================================================================


class PipelineError(MyanmarSpellcheckError):
    """
    Raised when the data pipeline encounters an error.

    This includes corpus ingestion, frequency calculation,
    and database packaging errors.
    """



class IngestionError(PipelineError):
    """
    Raised when corpus ingestion fails.

    This includes file reading errors, parsing failures,
    and data validation issues during ingestion.

    Attributes:
        message: Human-readable error description.
        failed_files: List of (filename, error) tuples for files that failed to process.
        missing_files: List of filenames that were not found.

    Example:
        >>> raise IngestionError("Failed to parse corpus file")
        >>> raise IngestionError(
        ...     "Ingestion failed",
        ...     failed_files=[("file.txt", "Parse error")],
        ...     missing_files=["missing.txt"]
        ... )
    """

    def __init__(
        self,
        message: str,
        failed_files: list[tuple] | None = None,
        missing_files: list[str] | None = None,
    ):
        self.message = message
        self.failed_files = failed_files or []
        self.missing_files = missing_files or []
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format the error message with details about failed/missing files."""
        parts = [self.message]
        if self.missing_files:
            parts.append(f"\nMissing files ({len(self.missing_files)}):")
            for f in self.missing_files[:10]:  # Limit to first 10
                parts.append(f"  - {f}")
            if len(self.missing_files) > 10:
                parts.append(f"  ... and {len(self.missing_files) - 10} more")
        if self.failed_files:
            parts.append(f"\nFailed files ({len(self.failed_files)}):")
            for f, err in self.failed_files[:10]:  # Limit to first 10
                parts.append(f"  - {f}: {err}")
            if len(self.failed_files) > 10:
                parts.append(f"  ... and {len(self.failed_files) - 10} more")
        return "\n".join(parts)


class PackagingError(PipelineError):
    """
    Raised when database packaging fails.

    This includes SQLite creation errors, index building failures,
    and data serialization issues.

    Example:
        >>> raise PackagingError("Failed to create SQLite database")
    """



# =============================================================================
# Model Errors
# =============================================================================


class ModelError(MyanmarSpellcheckError):
    """
    Raised when a machine learning model encounters an error.

    This includes model loading, training, and inference errors.
    """



class ModelLoadError(ModelError):
    """
    Raised when a model fails to load.

    This includes missing model files, incompatible formats,
    and memory issues during loading.

    Example:
        >>> raise ModelLoadError("Failed to load POS tagger model")
    """



class InferenceError(ModelError):
    """
    Raised when model inference fails.

    This includes prediction errors, tensor shape mismatches,
    and runtime errors during inference.

    Example:
        >>> raise InferenceError("Failed to run semantic similarity model")
    """



# =============================================================================
# Resource Errors
# =============================================================================


class MissingDependencyError(MyanmarSpellcheckError):
    """
    Raised when a required external dependency is missing.

    This is raised when optional dependencies like transformers,
    onnxruntime, or torch are not installed but required.

    Example:
        >>> raise MissingDependencyError(
        ...     "transformers package required for transformer POS tagger. "
        ...     "Install with: pip install myspellchecker[transformers]"
        ... )
    """



class InsufficientStorageError(MyanmarSpellcheckError):
    """
    Raised when there is not enough disk space to perform an operation.

    Example:
        >>> raise InsufficientStorageError(
        ...     "Need 500MB free space, only 100MB available"
        ... )
    """



class CacheError(MyanmarSpellcheckError):
    """
    Raised when caching operations fail.

    This includes cache initialization, storage, and retrieval errors.

    Example:
        >>> raise CacheError("Failed to initialize LRU cache")
    """



class MissingDatabaseError(DataLoadingError):
    """
    Raised when the spell checker database is not found.

    This error is raised instead of silently falling back to an empty provider,
    ensuring users are aware that no dictionary data is available.

    To resolve this error:
    1. Build a database: `myspellchecker build --sample`
    2. Or provide a custom database path via SpellCheckerConfig

    Example:
        >>> from myspellchecker import SpellChecker
        >>> from myspellchecker.core.exceptions import MissingDatabaseError
        >>> try:
        ...     checker = SpellChecker()
        ... except MissingDatabaseError as e:
        ...     print(f"Database not found: {e}")
        ...     # Build or download the database

    Attributes:
        searched_paths: List of paths that were searched for the database
        suggestion: Suggested action to resolve the error
    """

    def __init__(
        self,
        message: str = "Spell checker database not found.",
        searched_paths: list[str] | None = None,
        suggestion: str | None = None,
    ):
        self.searched_paths = searched_paths or []
        self.suggestion = suggestion or (
            "No bundled database is included. Build one first:\n"
            "  myspellchecker build --sample\n"
            "Then pass the path explicitly:\n"
            "  SpellChecker(provider=SQLiteProvider(database_path='mySpellChecker-default.db'))"
        )

        # Build full message
        full_message = message
        if self.searched_paths:
            paths_str = ", ".join(str(p) for p in self.searched_paths)
            full_message += f" Searched: {paths_str}."
        full_message += f" {self.suggestion}"

        super().__init__(full_message)
