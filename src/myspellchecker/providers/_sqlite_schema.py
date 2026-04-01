"""
Schema validation helpers and constants for SQLiteProvider.

Extracts module-level validation functions and constants so that
``sqlite.py`` stays focused on the provider class itself, and other
helper modules (e.g. ``_sqlite_bulk_ops.py``) can import these
utilities without a circular dependency on ``sqlite.py``.
"""

from __future__ import annotations

import sqlite3
import unicodedata
from pathlib import Path
from typing import Any

from myspellchecker.core.config.validation_configs import ProviderConfig
from myspellchecker.core.constants import SCHEMA_CHECK_TIMEOUT
from myspellchecker.core.exceptions import DataLoadingError

# Default provider config instance — single source of truth for defaults.
# Module-level constants below are derived from it for backward compatibility.
_DEFAULT_PROVIDER_CONFIG = ProviderConfig()

# SQLite has a limit of 999 parameters per query (SQLITE_MAX_VARIABLE_NUMBER)
# We use a slightly lower value to be safe with different SQLite builds
SQLITE_MAX_BATCH_SIZE: int = _DEFAULT_PROVIDER_CONFIG.sqlite_max_batch_size

# Batch size for iterator methods to prevent memory exhaustion
# This limits how many rows are loaded into memory at once
ITERATOR_FETCH_SIZE: int = _DEFAULT_PROVIDER_CONFIG.iterator_fetch_size

# Valid table names for SQL queries (defense-in-depth against SQL injection)
# These are the only tables that can be queried via _cached_frequency_query
VALID_TABLES: frozenset[str] = frozenset(
    {"syllables", "words", "bigrams", "trigrams", "fourgrams", "fivegrams", "ner_entities"}
)

# Valid column names for SQL queries
VALID_COLUMNS: frozenset[str] = frozenset({"syllable", "word", "frequency", "probability"})

# Default cache size — derived from ProviderConfig.default_cache_size.
DEFAULT_PROVIDER_CACHE_SIZE: int = _DEFAULT_PROVIDER_CONFIG.default_cache_size

# Valid database file extensions (security: prevent arbitrary file access)
VALID_DB_EXTENSIONS: frozenset[str] = frozenset({".db", ".sqlite", ".sqlite3"})

# SQLite error message patterns for missing column errors
# These patterns identify OperationalErrors that are safe to catch and handle
# as schema compatibility issues (e.g., old database without new columns)
_MISSING_COLUMN_PATTERNS = (
    "no such column:",
    "no column named",
    "has no column named",
)

# SQLite error message patterns for missing table errors
# These patterns identify OperationalErrors for missing tables (schema compatibility)
_MISSING_TABLE_PATTERNS = (
    "no such table:",
    "table not found",
)

# Schema version tracking
# Increment this when making breaking schema changes
# Format: MAJOR.MINOR where MAJOR = breaking changes, MINOR = backwards-compatible
CURRENT_SCHEMA_VERSION = "1.1"

# Minimum compatible schema version for reading
# Older databases may lack some features but can still be read
MIN_COMPATIBLE_SCHEMA_VERSION = "1.0"


def _is_missing_column_error(error: sqlite3.OperationalError) -> bool:
    """
    Check if an OperationalError is specifically about a missing column.

    This allows safe handling of schema compatibility issues while re-raising
    critical errors like disk I/O failures, database corruption, or locks.

    Args:
        error: The sqlite3.OperationalError to check

    Returns:
        True if the error is about a missing column, False otherwise

    Note:
        Critical errors that should NOT be caught and suppressed:
        - "disk I/O error" - Storage failure
        - "database disk image is malformed" - Database corruption
        - "database is locked" - Concurrency issue
        - "unable to open database" - File access issue
    """
    error_message = str(error).lower()
    return any(pattern in error_message for pattern in _MISSING_COLUMN_PATTERNS)


def _is_missing_table_error(error: sqlite3.OperationalError) -> bool:
    """
    Check if an OperationalError is specifically about a missing table.

    This allows safe handling of schema compatibility issues (e.g., old database
    without POS probability tables) while re-raising critical errors.

    Args:
        error: The sqlite3.OperationalError to check

    Returns:
        True if the error is about a missing table, False otherwise

    Note:
        Critical errors that should NOT be caught and suppressed:
        - "disk I/O error" - Storage failure
        - "database disk image is malformed" - Database corruption
        - "database is locked" - Concurrency issue
        - "unable to open database" - File access issue
    """
    error_message = str(error).lower()
    return any(pattern in error_message for pattern in _MISSING_TABLE_PATTERNS)


def _check_schema_version(db_path: Path, logger: Any) -> str | None:
    """
    Check the schema version of a database file.

    This function connects to the database and checks if it has a schema_info
    table with version information. It provides helpful error messages for
    version mismatches.

    Args:
        db_path: Path to the database file
        logger: Logger instance for warnings

    Returns:
        The schema version string if found, None if no version table exists

    Raises:
        DataLoadingError: If schema version is incompatible
    """
    try:
        conn = sqlite3.connect(str(db_path), timeout=SCHEMA_CHECK_TIMEOUT)
        try:
            cursor = conn.cursor()

            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_info'"
            )
            if cursor.fetchone() is None:
                # No schema_info table - this is an older database
                logger.info(
                    "Database has no schema_info table. "
                    "This is normal for databases created before schema versioning."
                )
                return None

            cursor.execute("SELECT value FROM schema_info WHERE key='version'")
            row = cursor.fetchone()
        finally:
            conn.close()

        if row is None:
            logger.warning(
                "schema_info table exists but has no version key. Assuming compatible schema."
            )
            return None

        db_version: str = row[0]

        # Check compatibility
        try:
            db_major = int(db_version.split(".")[0])
            current_major = int(CURRENT_SCHEMA_VERSION.split(".")[0])
            min_major = int(MIN_COMPATIBLE_SCHEMA_VERSION.split(".")[0])

            if db_major < min_major:
                raise DataLoadingError(
                    f"Database schema version {db_version} is too old. "
                    f"Minimum required version is {MIN_COMPATIBLE_SCHEMA_VERSION}. "
                    f"Please rebuild the database:\n"
                    f"  myspellchecker build --sample                    # Quick sample\n"
                    f"  myspellchecker build --input <corpus> --output <db>  # From corpus"
                )

            if db_major > current_major:
                logger.warning(
                    f"Database schema version {db_version} is newer than "
                    f"the current code version {CURRENT_SCHEMA_VERSION}. "
                    f"Some features may not work correctly. "
                    f"Consider upgrading the myspellchecker package."
                )

            if db_version != CURRENT_SCHEMA_VERSION:
                logger.info(
                    f"Database schema version: {db_version} "
                    f"(code expects: {CURRENT_SCHEMA_VERSION})"
                )

        except (ValueError, IndexError) as e:
            logger.warning(f"Could not parse schema version '{db_version}': {e}")

        return db_version

    except sqlite3.OperationalError as e:
        if _is_missing_table_error(e):
            # No schema_info table - this is an older database
            logger.debug("No schema_info table found (older database format)")
            return None
        # Re-raise other operational errors
        raise
    except sqlite3.Error as e:
        logger.warning(f"Could not check schema version: {e}")
        return None


def _validate_database_path(path: Path) -> Path:
    """
    Validate and normalize database path to prevent path traversal attacks.

    Security measures:
    - Resolves symlinks to get the real path
    - Validates file extension to prevent arbitrary file access
    - Normalizes path to prevent directory traversal (../)
    - Unicode normalization to prevent NFC/NFD bypass attacks
    - Null byte injection prevention
    - Control character detection

    Args:
        path: Path to validate

    Returns:
        Validated and normalized Path

    Raises:
        DataLoadingError: If path is invalid or potentially malicious
    """
    path_str = str(path)

    # Check for null bytes (injection attack prevention)
    if "\x00" in path_str:
        raise DataLoadingError("Null byte detected in database path. This may indicate an attack.")

    # Check for control characters (except path separators)
    for char in path_str:
        if unicodedata.category(char).startswith("C") and char not in ("/", "\\"):
            raise DataLoadingError(
                f"Control character detected in database path: {repr(char)}. "
                "Path contains potentially malicious characters."
            )

    # Normalize Unicode to NFC to prevent NFC/NFD bypass attacks
    # e.g., "a\u0308" (a + combining umlaut) vs "\u00e4" (precomposed ä)
    normalized_path_str = unicodedata.normalize("NFC", path_str)
    if normalized_path_str != path_str:
        # Path contains non-canonical Unicode - use normalized form
        path = Path(normalized_path_str)

    # Detect path traversal attempts in the original path
    # This catches cases like "../../../etc/passwd.db"
    if ".." in normalized_path_str:
        raise DataLoadingError(
            f"Path traversal detected in database path: {path}. "
            "Use absolute paths or paths without '..' components."
        )

    # Resolve to absolute path and follow symlinks
    # Use strict=True to detect symlink loops and missing files
    try:
        # First try strict resolution to catch symlink loops
        resolved_path = path.resolve(strict=True)
    except OSError as e:
        # Check if it's a symlink loop (Too many levels of symbolic links)
        error_msg = str(e)
        is_symlink_loop = (
            "Too many levels of symbolic links" in error_msg
            or getattr(e, "errno", None) == 40  # ELOOP
        )
        if is_symlink_loop:
            raise DataLoadingError(
                f"Symlink loop detected in database path: {path}. "
                "The path contains circular symbolic links."
            ) from e
        # For other errors (file not found, etc.), fall back to non-strict
        try:
            resolved_path = path.resolve(strict=False)
        except (OSError, RuntimeError) as e2:
            raise DataLoadingError(f"Invalid database path: {path} ({e2})") from e2
    except RuntimeError as e:
        raise DataLoadingError(f"Invalid database path: {path} ({e})") from e

    # After resolution, verify no path traversal occurred
    # (symlink could point to parent directory)
    resolved_str = str(resolved_path)
    if ".." in resolved_str:
        raise DataLoadingError(
            f"Path traversal detected after symlink resolution: {resolved_path}. "
            "Symlink may point to an unsafe location."
        )

    # Check file extension (defense against arbitrary file access)
    suffix = resolved_path.suffix.lower()
    if suffix not in VALID_DB_EXTENSIONS:
        raise DataLoadingError(
            f"Invalid database file extension: '{suffix}'. "
            f"Expected one of: {', '.join(sorted(VALID_DB_EXTENSIONS))}"
        )

    return resolved_path


def _validate_batch_items(batch: list, item_type: str = "item") -> None:
    """
    Validate batch items for SQL query parameters.

    Security measures:
    - Ensures batch size is within SQLite limits
    - Validates all items are strings (prevents type confusion attacks)

    Args:
        batch: List of items to validate
        item_type: Description of item type for error messages

    Raises:
        ValueError: If validation fails
    """
    # Enforce batch size limit
    if len(batch) > SQLITE_MAX_BATCH_SIZE:
        raise ValueError(
            f"Batch size {len(batch)} exceeds maximum {SQLITE_MAX_BATCH_SIZE}. "
            "This is a bug - batches should be chunked before this point."
        )

    # Validate all items are strings (defense against type confusion)
    for i, item in enumerate(batch):
        if not isinstance(item, str):
            raise ValueError(
                f"Invalid {item_type} type at index {i}: expected str, got {type(item).__name__}"
            )
