"""
Connection pooling for SQLite database connections.

This module provides a thread-safe connection pool to improve performance
by reusing database connections across threads instead of creating new
connections for each thread.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Full, Queue

from myspellchecker.core.config.validation_configs import (
    ConnectionPoolConfig,
    ProviderConfig,
)
from myspellchecker.core.exceptions import ConnectionPoolError
from myspellchecker.utils.logging_utils import get_logger

# Default provider config — single source of truth for PRAGMA defaults.
_DEFAULT_PROVIDER_CONFIG = ProviderConfig()

__all__ = [
    "ConnectionPool",
    "PooledConnection",
]

# Default Connection Pool configuration (module-level singleton)
_default_pool_config = ConnectionPoolConfig()


@dataclass
class PooledConnection:
    """Wrapper for a pooled database connection with metadata."""

    connection: sqlite3.Connection
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    use_count: int = 0

    def mark_used(self) -> None:
        """Update metadata when connection is used."""
        self.last_used = time.time()
        self.use_count += 1


class ConnectionPool:
    """
    Thread-safe connection pool for SQLite databases.

    Features:
        - Configurable pool size (min/max connections)
        - Automatic connection creation and reuse
        - Health checks and connection validation
        - Pool statistics and monitoring
        - Graceful degradation on errors

    Example:
        >>> pool = ConnectionPool("/path/to/db.sqlite", min_size=2, max_size=10)
        >>> with pool.checkout() as conn:
        ...     cursor = conn.cursor()
        ...     cursor.execute("SELECT * FROM table")
        ...     results = cursor.fetchall()
    """

    def __init__(
        self,
        database_path: str | Path,
        pool_config: ConnectionPoolConfig | None = None,
    ):
        """
        Initialize connection pool.

        Args:
            database_path: Path to SQLite database file.
            pool_config: ConnectionPoolConfig with pool settings (uses defaults if None).

        Raises:
            ValueError: If min_size > max_size or sizes are invalid.
            FileNotFoundError: If database file doesn't exist.
        """
        self.logger = get_logger(__name__)
        self.pool_config = pool_config or _default_pool_config

        # Extract config values
        min_size = self.pool_config.min_size
        max_size = self.pool_config.max_size

        if min_size < 0 or max_size < 1:
            raise ConnectionPoolError("min_size must be >= 0 and max_size must be >= 1")
        if min_size > max_size:
            raise ConnectionPoolError(f"min_size ({min_size}) cannot exceed max_size ({max_size})")

        self.database_path = Path(database_path)
        if not self.database_path.exists():
            raise FileNotFoundError(f"Database file not found: {self.database_path}")

        self.min_size = min_size
        self.max_size = max_size
        self.timeout = self.pool_config.timeout
        self.max_connection_age = self.pool_config.max_connection_age
        self.check_same_thread = self.pool_config.check_same_thread
        self.sqlite_timeout = self.pool_config.sqlite_timeout
        self.skip_health_check = self.pool_config.skip_health_check

        # Connection pool (FIFO queue)
        self._pool: Queue[PooledConnection] = Queue(maxsize=max_size)

        # Tracking
        self._lock = threading.Lock()
        self._active_connections = 0
        self._total_checkouts = 0
        self._total_wait_time = 0.0
        self._peak_active = 0
        self._closed = False

        # Initialize pool with minimum connections
        self._initialize_pool(min_size)

        self.logger.info(
            f"Connection pool initialized: min={min_size}, max={max_size}, "
            f"timeout={self.timeout}s, db={self.database_path.name}"
        )

    def _initialize_pool(self, count: int) -> None:
        """Create initial connections to fill pool (thread-safe)."""
        with self._lock:
            for _ in range(count):
                try:
                    conn = self._create_connection()
                    pooled = PooledConnection(connection=conn)
                    self._pool.put(pooled, block=False)
                    self._active_connections += 1
                except sqlite3.Error as e:
                    self.logger.error(f"Failed to create initial connection: {e}")
                    break

    def _create_connection(self) -> sqlite3.Connection:
        """
        Create a new SQLite connection with configured timeout.

        The timeout parameter specifies how long SQLite should wait when the
        database is locked before raising an OperationalError. This prevents
        hangs in concurrent access scenarios.

        Returns:
            Configured SQLite connection.

        Raises:
            sqlite3.DatabaseError: If connection fails.
            sqlite3.OperationalError: If database is locked and timeout expires.
        """
        conn = sqlite3.connect(
            str(self.database_path),
            timeout=self.sqlite_timeout,
            check_same_thread=self.check_same_thread,
        )
        conn.row_factory = sqlite3.Row

        # Runtime optimization PRAGMAs (values from ProviderConfig)
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute(f"PRAGMA mmap_size = {_DEFAULT_PROVIDER_CONFIG.pragma_mmap_size}")
        conn.execute(f"PRAGMA cache_size = {_DEFAULT_PROVIDER_CONFIG.pragma_cache_size}")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA temp_store = MEMORY")
        if self.skip_health_check:
            # Read-only mode: spell check databases are never written to
            conn.execute("PRAGMA query_only = ON")

        return conn

    def _is_connection_healthy(self, conn: sqlite3.Connection) -> bool:
        """
        Check if connection is healthy and usable.

        Args:
            conn: Connection to check.

        Returns:
            True if connection is healthy, False otherwise.
        """
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            return True
        except sqlite3.Error as e:
            # Log health check failures for debugging
            self.logger.debug(f"Connection health check failed: {e}")
            return False

    def _should_recreate_connection(self, pooled: PooledConnection) -> bool:
        """
        Check if connection should be recreated due to age.

        Args:
            pooled: Pooled connection to check.

        Returns:
            True if connection should be recreated, False otherwise.
        """
        age = time.time() - pooled.created_at
        return age > self.max_connection_age

    @contextmanager
    def checkout(self) -> Generator[sqlite3.Connection, None, None]:
        """
        Check out a connection from the pool.

        This is the main interface for using pooled connections. It ensures
        the connection is returned to the pool even if an exception occurs.

        Yields:
            SQLite connection from the pool.

        Raises:
            TimeoutError: If no connection available within timeout period.

        Example:
            >>> pool = ConnectionPool("/path/to/db.sqlite")
            >>> with pool.checkout() as conn:
            ...     cursor = conn.cursor()
            ...     cursor.execute("SELECT * FROM table")
        """
        with self._lock:
            if self._closed:
                raise ConnectionPoolError("Connection pool is closed. No new checkouts allowed.")

        start_time = time.time()
        pooled_conn: PooledConnection | None = None

        try:
            try:
                pooled_conn = self._pool.get(timeout=self.timeout)
            except Empty:
                # Pool exhausted, try to create new connection if under max
                with self._lock:
                    if self._closed:
                        raise ConnectionPoolError(
                            "Connection pool is closed. No new checkouts allowed."
                        ) from None
                    if self._active_connections < self.max_size:
                        self._active_connections += 1
                        create_new = True
                    else:
                        create_new = False

                if create_new:
                    try:
                        conn = self._create_connection()
                        pooled_conn = PooledConnection(connection=conn)
                        self.logger.debug(
                            f"Created new connection "
                            f"(active: {self._active_connections}/{self.max_size})"
                        )
                    except sqlite3.Error as e:
                        # Decrement counter if connection creation fails
                        # to prevent resource leak in the counter
                        with self._lock:
                            self._active_connections -= 1
                        self.logger.error(f"Failed to create connection: {e}")
                        raise
                else:
                    # Get metrics for actionable error message
                    with self._lock:
                        active = self._active_connections
                        avg_wait = (
                            self._total_wait_time / self._total_checkouts
                            if self._total_checkouts > 0
                            else 0.0
                        )
                    raise TimeoutError(
                        f"Connection pool exhausted: "
                        f"active={active}/{self.max_size}, "
                        f"waited={self.timeout}s, avg_wait={avg_wait:.3f}s. "
                        f"Consider increasing max_size or reducing connection hold time."
                    ) from None

            # Check connection health and age (skip if configured for read-only DBs)
            if pooled_conn is not None:
                needs_recreate = self._should_recreate_connection(pooled_conn) or (
                    not self.skip_health_check
                    and not self._is_connection_healthy(pooled_conn.connection)
                )
                if needs_recreate:
                    self.logger.debug("Recreating unhealthy or aged connection")
                    try:
                        pooled_conn.connection.close()
                    except sqlite3.Error as e:
                        # Log close errors for debugging but don't propagate
                        self.logger.debug(f"Ignoring close error on unhealthy connection: {e}")
                    try:
                        pooled_conn.connection = self._create_connection()
                        pooled_conn.created_at = time.time()
                    except sqlite3.Error as e:
                        # Recreation failed - decrement counter and re-raise
                        # to prevent connection leak in counter tracking
                        with self._lock:
                            self._active_connections -= 1
                        self.logger.error(f"Failed to recreate connection: {e}")
                        pooled_conn = None  # Prevent return to pool
                        raise

                # Mark as used
                if pooled_conn is not None:
                    pooled_conn.mark_used()

            # Track statistics
            wait_time = time.time() - start_time
            with self._lock:
                self._total_checkouts += 1
                self._total_wait_time += wait_time
                # Use _active_connections counter instead of max_size - qsize
                # to handle cases where connections are discarded (unhealthy)
                current_active = self._active_connections - self._pool.qsize()
                self._peak_active = max(self._peak_active, current_active)

            # Yield connection - pooled_conn should always be valid here
            # (either from pool or newly created), otherwise an exception
            # would have been raised above. This assertion documents the invariant.
            if pooled_conn is None:
                raise ConnectionPoolError(
                    "Internal error: pooled_conn is None after successful checkout. "
                    "This indicates a bug in the connection pool logic."
                )
            yield pooled_conn.connection

        finally:
            # Return connection to pool (health is checked on next checkout)
            if pooled_conn is not None:
                # If pool is closed, close the connection instead of returning it
                if self._closed:
                    try:
                        pooled_conn.connection.close()
                    except sqlite3.Error as close_err:
                        self.logger.debug(
                            f"Ignoring close error on post-shutdown return: {close_err}"
                        )
                    with self._lock:
                        self._active_connections -= 1
                else:
                    try:
                        self._pool.put(pooled_conn, block=False)
                    except Full as e:
                        self.logger.error(f"Failed to return connection to pool: {e}")
                        # Close the connection to prevent resource leak
                        try:
                            pooled_conn.connection.close()
                        except sqlite3.Error as close_err:
                            self.logger.debug(f"Ignoring close error during cleanup: {close_err}")
                        with self._lock:
                            self._active_connections -= 1

    def close_all(self) -> None:
        """
        Close all connections in the pool.

        This should be called during cleanup/shutdown.
        Thread-safe and idempotent - safe to call multiple times.
        """
        # Check if already closed (thread-safe)
        with self._lock:
            if self._closed:
                return
            self._closed = True

        closed_count = 0
        while not self._pool.empty():
            try:
                pooled = self._pool.get(block=False)
                pooled.connection.close()
                closed_count += 1
                with self._lock:
                    self._active_connections -= 1
            except Empty:
                break
            except sqlite3.Error as e:
                self.logger.error(f"Error closing connection: {e}")

        self.logger.info(f"Closed {closed_count} connections")

    def get_stats(self) -> dict:
        """
        Get pool statistics.

        Returns:
            Dictionary with pool statistics including:
            - pool_size: Total connections managed by the pool
            - active_connections: Total connections created
            - available_connections: Connections ready for checkout
            - total_checkouts: Total number of checkout operations
            - average_wait_time_ms: Average wait time for checkout in ms
            - peak_active: Maximum concurrent active connections
        """
        with self._lock:
            available = self._pool.qsize()
            avg_wait_ms = (
                (self._total_wait_time / self._total_checkouts) * 1000
                if self._total_checkouts > 0
                else 0.0
            )

            return {
                "pool_size": self.max_size,
                "active_connections": self._active_connections,
                "available_connections": available,
                "total_checkouts": self._total_checkouts,
                "average_wait_time_ms": round(avg_wait_ms, 2),
                "peak_active": self._peak_active,
                "min_size": self.min_size,
                "max_size": self.max_size,
            }

    def __enter__(self) -> ConnectionPool:
        """Context manager entry."""
        return self

    def __exit__(
        self,
        exc_type: type | None,
        exc_val: BaseException | None,
        exc_tb: object | None,
    ) -> None:
        """Context manager exit - close all connections."""
        self.close_all()

    def __del__(self) -> None:
        """Destructor - ensure connections are closed.

        Note: __del__ is not guaranteed to be called and should not be relied
        upon for cleanup. Always use the context manager or explicitly call
        close_all() when possible.
        """
        # Guard against __del__ being called on partially initialized objects
        # (e.g., when __init__ raises an exception before _lock is created)
        if not hasattr(self, "_lock"):
            return
        try:
            self.close_all()
        except (RuntimeError, sqlite3.Error, OSError) as e:
            # Destructor errors cannot be reliably logged as logger may be torn down.
            # Best effort: try to log, but fail silently if logging unavailable.
            try:
                self.logger.debug(f"Error during connection pool cleanup: {e}")
            except (RuntimeError, AttributeError):
                pass  # Logger may be unavailable during interpreter shutdown
