"""
Comprehensive tests for ConnectionPool.

This test suite covers:
- Pool initialization and configuration
- Connection checkout and return
- Thread safety
- Health checks and connection aging
- Pool size limits and timeout handling
- Statistics tracking
- Error handling and cleanup
"""

import sqlite3
import threading
import time
from unittest.mock import patch

import pytest

from myspellchecker.core.config.validation_configs import ConnectionPoolConfig
from myspellchecker.providers.connection_pool import ConnectionPool, PooledConnection


@pytest.fixture
def test_db(tmp_path):
    """Create a temporary test database."""
    database_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(database_path))
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)")
    cursor.execute("INSERT INTO test (value) VALUES ('test')")
    conn.commit()
    conn.close()
    return database_path


class TestPooledConnection:
    """Tests for PooledConnection dataclass."""

    def test_initialization(self):
        """Test PooledConnection initialization."""
        conn = sqlite3.connect(":memory:")
        pooled = PooledConnection(connection=conn)

        assert pooled.connection == conn
        assert pooled.use_count == 0
        assert isinstance(pooled.created_at, float)
        assert isinstance(pooled.last_used, float)
        # Timestamps should be very close (within 1ms)
        assert abs(pooled.created_at - pooled.last_used) < 0.001

    def test_mark_used(self):
        """Test marking connection as used updates metadata."""
        conn = sqlite3.connect(":memory:")
        pooled = PooledConnection(connection=conn)

        initial_last_used = pooled.last_used
        initial_use_count = pooled.use_count

        # Patch time.time where it's used (in connection_pool module) to advance the clock
        with patch("myspellchecker.providers.connection_pool.time") as mock_time:
            mock_time.time.return_value = initial_last_used + 1.0
            pooled.mark_used()

        assert pooled.use_count == initial_use_count + 1
        assert pooled.last_used > initial_last_used

    def test_multiple_uses(self):
        """Test connection can be marked as used multiple times."""
        conn = sqlite3.connect(":memory:")
        pooled = PooledConnection(connection=conn)

        base_time = pooled.last_used
        for i in range(5):
            # Advance the clock deterministically instead of sleeping
            with patch("myspellchecker.providers.connection_pool.time") as mock_time:
                mock_time.time.return_value = base_time + (i + 1) * 1.0
                pooled.mark_used()
            assert pooled.use_count == i + 1


class TestConnectionPoolInitialization:
    """Tests for ConnectionPool initialization."""

    def test_basic_initialization(self, test_db):
        """Test basic pool initialization."""
        config = ConnectionPoolConfig(min_size=2, max_size=5)
        pool = ConnectionPool(test_db, pool_config=config)

        assert pool.database_path == test_db
        assert pool.min_size == 2
        assert pool.max_size == 5
        assert pool.timeout == 5.0  # default
        assert pool.max_connection_age == 3600.0  # default
        assert not pool.check_same_thread

        pool.close_all()

    def test_custom_configuration(self, test_db):
        """Test pool with custom configuration."""
        config = ConnectionPoolConfig(
            min_size=3,
            max_size=10,
            timeout=10.0,
            max_connection_age=7200.0,
            check_same_thread=True,
        )
        pool = ConnectionPool(test_db, pool_config=config)

        assert pool.min_size == 3
        assert pool.max_size == 10
        assert pool.timeout == 10.0
        assert pool.max_connection_age == 7200.0
        assert pool.check_same_thread

        pool.close_all()

    def test_invalid_min_size(self, test_db):
        """Test initialization fails with invalid min_size (via Pydantic)."""
        with pytest.raises(ValueError):
            ConnectionPoolConfig(min_size=-1, max_size=5)

    def test_invalid_max_size(self, test_db):
        """Test initialization fails with invalid max_size (via Pydantic)."""
        with pytest.raises(ValueError):
            ConnectionPoolConfig(min_size=0, max_size=0)

    def test_min_size_exceeds_max_size(self, test_db):
        """Test initialization fails when min_size > max_size (via Pydantic)."""
        with pytest.raises(ValueError, match="max_size.*must be >= min_size"):
            ConnectionPoolConfig(min_size=10, max_size=5)

    def test_database_not_found(self, tmp_path):
        """Test initialization fails with non-existent database."""
        non_existent = tmp_path / "nonexistent.db"
        config = ConnectionPoolConfig(min_size=2, max_size=5)
        with pytest.raises(FileNotFoundError, match="Database file not found"):
            ConnectionPool(non_existent, pool_config=config)

    def test_initial_connections_created(self, test_db):
        """Test min_size connections are created on initialization."""
        config = ConnectionPoolConfig(min_size=3, max_size=5)
        pool = ConnectionPool(test_db, pool_config=config)

        stats = pool.get_stats()
        assert stats["pool_size"] == 5  # max_size, not active_connections
        assert stats["active_connections"] == 3  # min_size connections created
        assert stats["available_connections"] == 3

        pool.close_all()


class TestConnectionCheckout:
    """Tests for connection checkout and return."""

    def test_basic_checkout(self, test_db):
        """Test basic connection checkout and return."""
        config = ConnectionPoolConfig(min_size=2, max_size=5)
        pool = ConnectionPool(test_db, pool_config=config)

        with pool.checkout() as conn:
            assert conn is not None
            assert isinstance(conn, sqlite3.Connection)

            # Connection should work
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            assert result[0] == 1

        pool.close_all()

    def test_connection_returned_to_pool(self, test_db):
        """Test connection is returned to pool after use."""
        config = ConnectionPoolConfig(min_size=2, max_size=5)
        pool = ConnectionPool(test_db, pool_config=config)

        initial_available = pool.get_stats()["available_connections"]

        with pool.checkout():
            # During checkout, available should decrease
            stats_during = pool.get_stats()
            assert stats_during["available_connections"] == initial_available - 1

        # After return, available should be back to initial
        stats_after = pool.get_stats()
        assert stats_after["available_connections"] == initial_available

        pool.close_all()

    def test_multiple_checkouts(self, test_db):
        """Test multiple sequential checkouts."""
        config = ConnectionPoolConfig(min_size=2, max_size=5)
        pool = ConnectionPool(test_db, pool_config=config)

        for i in range(10):
            with pool.checkout() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT ?", (i,))
                result = cursor.fetchone()
                assert result[0] == i

        pool.close_all()

    def test_checkout_exception_returns_connection(self, test_db):
        """Test connection is returned even if exception occurs."""
        config = ConnectionPoolConfig(min_size=2, max_size=5)
        pool = ConnectionPool(test_db, pool_config=config)

        initial_available = pool.get_stats()["available_connections"]

        try:
            with pool.checkout():
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Connection should still be returned
        stats = pool.get_stats()
        assert stats["available_connections"] == initial_available

        pool.close_all()

    def test_checkout_updates_statistics(self, test_db):
        """Test checkout updates pool statistics."""
        config = ConnectionPoolConfig(min_size=2, max_size=5)
        pool = ConnectionPool(test_db, pool_config=config)

        initial_checkouts = pool.get_stats()["total_checkouts"]

        with pool.checkout():
            pass

        stats = pool.get_stats()
        assert stats["total_checkouts"] == initial_checkouts + 1
        assert stats["average_wait_time_ms"] >= 0

        pool.close_all()


class TestPoolSizeLimits:
    """Tests for pool size limits and expansion."""

    def test_pool_expands_when_needed(self, test_db):
        """Test pool creates new connections up to max_size."""
        config = ConnectionPoolConfig(min_size=1, max_size=3)
        pool = ConnectionPool(test_db, pool_config=config)

        connections = []
        try:
            # Checkout all connections
            for _ in range(3):
                connections.append(pool.checkout())
                connections[-1].__enter__()

            stats = pool.get_stats()
            assert stats["pool_size"] == 3  # max_size
            assert stats["active_connections"] == 3
            assert stats["available_connections"] == 0

        finally:
            # Return all connections
            for ctx in connections:
                ctx.__exit__(None, None, None)
            pool.close_all()

    def test_timeout_when_pool_exhausted(self, test_db):
        """Test timeout error when pool is exhausted."""
        config = ConnectionPoolConfig(min_size=1, max_size=2, timeout=0.5)
        pool = ConnectionPool(test_db, pool_config=config)

        connections = []
        try:
            # Exhaust the pool
            for _ in range(2):
                connections.append(pool.checkout())
                connections[-1].__enter__()

            # Try to get another connection - should timeout
            with pytest.raises(TimeoutError, match="Connection pool exhausted"):
                with pool.checkout():
                    pass

        finally:
            for ctx in connections:
                ctx.__exit__(None, None, None)
            pool.close_all()

    def test_peak_active_tracking(self, test_db):
        """Test peak active connections tracking."""
        config = ConnectionPoolConfig(min_size=1, max_size=5)
        pool = ConnectionPool(test_db, pool_config=config)

        # Checkout 3 connections
        connections = []
        try:
            for _ in range(3):
                connections.append(pool.checkout())
                connections[-1].__enter__()

            stats = pool.get_stats()
            assert stats["peak_active"] >= 3

        finally:
            for ctx in connections:
                ctx.__exit__(None, None, None)
            pool.close_all()


class TestHealthChecks:
    """Tests for connection health checks."""

    def test_healthy_connection_reused(self, test_db):
        """Test healthy connections are reused."""
        config = ConnectionPoolConfig(min_size=1, max_size=2)
        pool = ConnectionPool(test_db, pool_config=config)

        # First checkout
        with pool.checkout() as conn1:
            cursor = conn1.cursor()
            cursor.execute("SELECT 1")

        # Second checkout should get the same connection
        with pool.checkout() as conn2:
            cursor = conn2.cursor()
            cursor.execute("SELECT 2")

        pool.close_all()

    def test_unhealthy_connection_recreated(self, test_db):
        """Test unhealthy connections are recreated."""
        config = ConnectionPoolConfig(min_size=1, max_size=2)
        pool = ConnectionPool(test_db, pool_config=config)

        # Checkout and corrupt connection
        with pool.checkout() as conn:
            pass

        # Manually close a connection to make it unhealthy
        # Then check it's recreated on next checkout
        pooled_conn = pool._pool.get()
        pooled_conn.connection.close()
        pool._pool.put(pooled_conn)

        # Next checkout should recreate the connection
        with pool.checkout() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            assert result[0] == 1

        pool.close_all()

    def test_aged_connection_recreated(self, test_db):
        """Test connections exceeding max age are recreated."""
        config = ConnectionPoolConfig(min_size=1, max_size=2, max_connection_age=0.1)
        pool = ConnectionPool(test_db, pool_config=config)

        # Checkout connection to exercise it
        with pool.checkout() as conn:
            pass

        # Instead of sleeping, artificially age the connection by backdating created_at.
        # This makes _should_recreate_connection() see the connection as expired
        # without relying on wall-clock time.
        pooled_conn = pool._pool.get()
        pooled_conn.created_at = pooled_conn.created_at - 1.0  # 1s ago, well past 0.1s max age
        pool._pool.put(pooled_conn)

        # Next checkout should recreate aged connection
        with pool.checkout() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            assert result[0] == 1

        pool.close_all()


class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_checkouts(self, test_db):
        """Test pool handles concurrent checkouts from multiple threads."""
        config = ConnectionPoolConfig(min_size=2, max_size=10)
        pool = ConnectionPool(test_db, pool_config=config)
        results = []
        errors = []

        def worker(worker_id):
            try:
                for i in range(5):
                    with pool.checkout() as conn:
                        cursor = conn.cursor()
                        cursor.execute("SELECT ?", (worker_id * 100 + i,))
                        result = cursor.fetchone()
                        results.append(result[0])
                        time.sleep(0.001)  # Simulate work
            except Exception as e:
                errors.append(e)

        # Create 5 threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        # Wait for all threads
        for t in threads:
            t.join()

        # Verify no errors
        assert len(errors) == 0
        assert len(results) == 25  # 5 threads * 5 iterations

        pool.close_all()

    def test_concurrent_statistics_tracking(self, test_db):
        """Test statistics are correctly tracked with concurrent access."""
        config = ConnectionPoolConfig(min_size=2, max_size=10)
        pool = ConnectionPool(test_db, pool_config=config)

        def worker():
            for _ in range(10):
                with pool.checkout() as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT 1")
                    time.sleep(0.001)

        threads = []
        for _ in range(5):
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        stats = pool.get_stats()
        assert stats["total_checkouts"] == 50  # 5 threads * 10 iterations
        assert stats["peak_active"] >= 2

        pool.close_all()


class TestStatistics:
    """Tests for pool statistics."""

    def test_get_stats_structure(self, test_db):
        """Test get_stats returns correct structure."""
        config = ConnectionPoolConfig(min_size=2, max_size=5)
        pool = ConnectionPool(test_db, pool_config=config)

        stats = pool.get_stats()

        assert "pool_size" in stats
        assert "active_connections" in stats
        assert "available_connections" in stats
        assert "total_checkouts" in stats
        assert "average_wait_time_ms" in stats
        assert "peak_active" in stats
        assert "min_size" in stats
        assert "max_size" in stats

        pool.close_all()

    def test_statistics_accuracy(self, test_db):
        """Test statistics are accurate."""
        config = ConnectionPoolConfig(min_size=2, max_size=5)
        pool = ConnectionPool(test_db, pool_config=config)

        # Initial stats
        stats = pool.get_stats()
        assert stats["pool_size"] == 5  # max_size
        assert stats["active_connections"] == 2
        assert stats["available_connections"] == 2
        assert stats["total_checkouts"] == 0
        assert stats["min_size"] == 2
        assert stats["max_size"] == 5

        # After one checkout
        with pool.checkout():
            pass

        stats = pool.get_stats()
        assert stats["total_checkouts"] == 1
        assert stats["average_wait_time_ms"] >= 0

        pool.close_all()


class TestCleanup:
    """Tests for pool cleanup and resource management."""

    def test_close_all(self, test_db):
        """Test close_all closes all connections."""
        config = ConnectionPoolConfig(min_size=3, max_size=5)
        pool = ConnectionPool(test_db, pool_config=config)

        initial_stats = pool.get_stats()
        assert initial_stats["pool_size"] == 5  # max_size
        assert initial_stats["active_connections"] == 3

        pool.close_all()

        final_stats = pool.get_stats()
        assert final_stats["pool_size"] == 5  # max_size unchanged after close
        assert final_stats["active_connections"] == 0
        assert final_stats["available_connections"] == 0

    def test_context_manager(self, test_db):
        """Test pool works as context manager."""
        config = ConnectionPoolConfig(min_size=2, max_size=5)
        with ConnectionPool(test_db, pool_config=config) as pool:
            stats = pool.get_stats()
            assert stats["pool_size"] == 5  # max_size

            with pool.checkout() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")

        # After exiting context, pool should be closed
        # (we can't easily test this without accessing internals)

    def test_destructor_cleanup(self, test_db):
        """Test destructor closes connections."""
        config = ConnectionPoolConfig(min_size=2, max_size=5)
        pool = ConnectionPool(test_db, pool_config=config)
        id(pool)

        # Delete pool
        del pool

        # Pool should be garbage collected and connections closed
        # (difficult to test directly without accessing internals)


class TestErrorHandling:
    """Tests for error handling."""

    def test_connection_creation_failure_during_init(self, tmp_path):
        """Test graceful handling of connection failures during initialization."""
        # Create a directory instead of a file to cause connection failure
        database_path = tmp_path / "test_dir"
        database_path.mkdir()

        # Pool should handle this gracefully - no exception raised, but 0 connections created
        config = ConnectionPoolConfig(min_size=2, max_size=5)
        pool = ConnectionPool(database_path, pool_config=config)

        # Pool created, but with 0 active connections due to initialization failures
        stats = pool.get_stats()
        assert stats["pool_size"] == 5  # max_size unchanged
        assert stats["active_connections"] == 0
        assert stats["available_connections"] == 0

    def test_pool_put_failure(self, test_db):
        """Test handling of errors when returning connection to pool."""
        config = ConnectionPoolConfig(min_size=1, max_size=2)
        pool = ConnectionPool(test_db, pool_config=config)

        with pool.checkout() as conn:
            # Connection should work normally
            cursor = conn.cursor()
            cursor.execute("SELECT 1")

        # Pool should handle put errors gracefully
        # (hard to test without mocking)

        pool.close_all()


class TestIntegration:
    """Integration tests with actual SQLite operations."""

    def test_real_world_usage_pattern(self, test_db):
        """Test realistic usage pattern with queries."""
        config = ConnectionPoolConfig(min_size=2, max_size=5)
        pool = ConnectionPool(test_db, pool_config=config)

        # Insert data
        with pool.checkout() as conn:
            cursor = conn.cursor()
            for i in range(10):
                cursor.execute("INSERT INTO test (value) VALUES (?)", (f"value_{i}",))
            conn.commit()

        # Read data
        with pool.checkout() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM test")
            count = cursor.fetchone()[0]
            assert count == 11  # 1 initial + 10 inserted

        # Update data
        with pool.checkout() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE test SET value = 'updated' WHERE id = 1")
            conn.commit()

        # Verify update
        with pool.checkout() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM test WHERE id = 1")
            value = cursor.fetchone()[0]
            assert value == "updated"

        pool.close_all()

    def test_transaction_isolation(self, test_db):
        """Test each connection maintains proper transaction isolation."""
        config = ConnectionPoolConfig(min_size=2, max_size=5)
        pool = ConnectionPool(test_db, pool_config=config)

        # Start transaction in first connection
        with pool.checkout() as conn1:
            cursor1 = conn1.cursor()
            cursor1.execute("INSERT INTO test (value) VALUES ('tx1')")
            # Don't commit yet

            # Second connection should not see uncommitted data
            with pool.checkout() as conn2:
                cursor2 = conn2.cursor()
                cursor2.execute("SELECT COUNT(*) FROM test WHERE value = 'tx1'")
                count = cursor2.fetchone()[0]
                assert count == 0  # Transaction not committed

            # Commit in first connection
            conn1.commit()

        # Now second connection should see the data
        with pool.checkout() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM test WHERE value = 'tx1'")
            count = cursor.fetchone()[0]
            assert count == 1

        pool.close_all()
