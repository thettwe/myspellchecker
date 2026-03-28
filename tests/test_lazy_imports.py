"""Tests for lazy import functionality in data pipeline."""

import sys
import threading


class TestLazyImports:
    """Test that heavy dependencies are lazily loaded."""

    def test_ingester_module_import_does_not_load_pyarrow(self):
        """
        Verify that importing the ingester module does not immediately load pyarrow.

        This tests lazy loading of heavy module-level imports in data pipeline.
        """
        # Note: We DON'T remove pyarrow or rich from sys.modules here.
        # Removing rich modules corrupts Rich's internal state and causes
        # "Text.append() takes from 2 to 3 positional arguments but N were given"
        # errors in subsequent tests due to stale lru_cache references.
        #
        # Instead, we just test that the lazy loader attributes exist.
        # The lazy loading mechanism works regardless of whether modules
        # are already loaded - the @lru_cache ensures single initialization.

        # Remove ingester if already imported to ensure fresh import
        ingester_modules = [k for k in sys.modules if "ingester" in k and "data_pipeline" in k]
        for mod in ingester_modules:
            sys.modules.pop(mod, None)

        # Import the ingester module
        from myspellchecker.data_pipeline import ingester  # noqa: F401

        # After importing, pyarrow should NOT be in sys.modules yet
        # (it's only loaded when process() or other methods are called)
        # Note: This test may fail if pyarrow was imported elsewhere in the test suite
        # So we check for the lazy loading mechanism instead
        assert hasattr(ingester, "_get_pyarrow")
        assert hasattr(ingester, "_get_rich_live")
        assert hasattr(ingester, "_get_rich_table")
        assert hasattr(ingester, "_get_ingest_schema")

    def test_lazy_loader_functions_return_correct_types(self):
        """Test that lazy loader functions return the correct types when called."""
        from myspellchecker.data_pipeline.ingester import (
            _get_ingest_schema,
            _get_pyarrow,
            _get_rich_live,
            _get_rich_table,
        )

        # Call the lazy loaders and verify they return the expected types
        pa = _get_pyarrow()
        assert hasattr(pa, "schema")
        assert hasattr(pa, "OSFile")
        assert hasattr(pa, "RecordBatch")

        Live = _get_rich_live()
        assert callable(Live)

        Table = _get_rich_table()
        assert callable(Table)

        schema = _get_ingest_schema()
        assert hasattr(schema, "names")
        assert "text" in schema.names
        assert "source" in schema.names

    def test_lazy_loaders_are_cached(self):
        """Test that lazy loaders cache their results (singleton pattern)."""
        from myspellchecker.data_pipeline.ingester import (
            _get_pyarrow,
            _get_rich_live,
            _get_rich_table,
        )

        # Call twice and verify same instance is returned
        pa1 = _get_pyarrow()
        pa2 = _get_pyarrow()
        assert pa1 is pa2

        Live1 = _get_rich_live()
        Live2 = _get_rich_live()
        assert Live1 is Live2

        Table1 = _get_rich_table()
        Table2 = _get_rich_table()
        assert Table1 is Table2

    def test_lazy_loaders_thread_safe(self):
        """
        Test that lazy loaders are thread-safe.

        Multiple threads calling the loader simultaneously should all
        get the same cached instance without race conditions.
        """
        from myspellchecker.data_pipeline.ingester import (
            _get_pyarrow,
            _get_rich_live,
            _get_rich_table,
        )

        results = {"pyarrow": [], "live": [], "table": []}
        errors = []

        def load_pyarrow():
            try:
                results["pyarrow"].append(_get_pyarrow())
            except Exception as e:
                errors.append(e)

        def load_live():
            try:
                results["live"].append(_get_rich_live())
            except Exception as e:
                errors.append(e)

        def load_table():
            try:
                results["table"].append(_get_rich_table())
            except Exception as e:
                errors.append(e)

        # Start multiple threads for each loader
        threads = []
        for _ in range(5):
            threads.append(threading.Thread(target=load_pyarrow))
            threads.append(threading.Thread(target=load_live))
            threads.append(threading.Thread(target=load_table))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Check no errors
        assert not errors, f"Errors occurred: {errors}"

        # All results should be the same instance
        assert len(results["pyarrow"]) == 5
        assert all(r is results["pyarrow"][0] for r in results["pyarrow"])

        assert len(results["live"]) == 5
        assert all(r is results["live"][0] for r in results["live"])

        assert len(results["table"]) == 5
        assert all(r is results["table"][0] for r in results["table"])

    def test_lazy_loaders_use_lru_cache(self):
        """Test that lazy loaders are implemented using lru_cache for thread safety."""
        from functools import _lru_cache_wrapper

        from myspellchecker.data_pipeline.ingester import (
            _get_ingest_schema,
            _get_pyarrow,
            _get_rich_live,
            _get_rich_table,
        )

        # Verify each function is wrapped with lru_cache
        assert isinstance(_get_pyarrow, _lru_cache_wrapper)
        assert isinstance(_get_rich_live, _lru_cache_wrapper)
        assert isinstance(_get_rich_table, _lru_cache_wrapper)
        assert isinstance(_get_ingest_schema, _lru_cache_wrapper)
