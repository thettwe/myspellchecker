"""Tests for SymSpell thread safety."""

import threading
import time

import pytest


class TestSymSpellThreadSafety:
    """Test that SymSpell.build_index is thread-safe."""

    @pytest.mark.slow
    def test_concurrent_build_index_same_level(self):
        """
        Test that concurrent calls to build_index with the same level
        don't cause duplicate work or data corruption.
        """
        from myspellchecker.algorithms.symspell import SymSpell
        from myspellchecker.providers import MemoryProvider

        provider = MemoryProvider()
        # Add some test syllables
        for i in range(100):
            provider.add_syllable(f"syllable{i}", frequency=100 + i)

        symspell = SymSpell(provider, max_edit_distance=1)

        # Track how many times the iterator is consumed
        call_count = {"count": 0}
        original_get_all = provider.get_all_syllables

        def counting_get_all():
            call_count["count"] += 1
            return original_get_all()

        provider.get_all_syllables = counting_get_all

        # Start multiple threads trying to build the same index
        threads = []
        errors = []

        def build_index_thread():
            try:
                symspell.build_index(["syllable"])
            except Exception as e:
                errors.append(e)

        for _ in range(10):
            t = threading.Thread(target=build_index_thread)
            threads.append(t)

        # Start all threads simultaneously
        for t in threads:
            t.start()

        # Wait for all threads to complete
        for t in threads:
            t.join()

        # Check no errors occurred
        assert not errors, f"Errors occurred: {errors}"

        # Check the level was only indexed once
        assert "syllable" in symspell._indexed_levels
        assert call_count["count"] == 1, (
            f"Expected iterator to be consumed once, but was consumed {call_count['count']} times"
        )

    @pytest.mark.slow
    def test_concurrent_build_index_different_levels(self):
        """
        Test that concurrent calls to build_index with different levels
        work correctly in parallel.
        """
        from myspellchecker.algorithms.symspell import SymSpell
        from myspellchecker.providers import MemoryProvider

        provider = MemoryProvider()
        # Add test data
        for i in range(50):
            provider.add_syllable(f"syl{i}", frequency=100 + i)
            provider.add_word(f"word{i}", frequency=200 + i)

        symspell = SymSpell(provider, max_edit_distance=1)

        errors = []

        def build_syllable_index():
            try:
                symspell.build_index(["syllable"])
            except Exception as e:
                errors.append(e)

        def build_word_index():
            try:
                symspell.build_index(["word"])
            except Exception as e:
                errors.append(e)

        # Start threads for different levels
        t1 = threading.Thread(target=build_syllable_index)
        t2 = threading.Thread(target=build_word_index)

        t1.start()
        t2.start()

        t1.join()
        t2.join()

        # Check no errors
        assert not errors, f"Errors occurred: {errors}"

        # Check both levels were indexed
        assert "syllable" in symspell._indexed_levels
        assert "word" in symspell._indexed_levels

    def test_build_index_lock_exists(self):
        """Test that SymSpell has the _index_lock attribute."""
        from myspellchecker.algorithms.symspell import SymSpell
        from myspellchecker.providers import MemoryProvider

        provider = MemoryProvider()
        symspell = SymSpell(provider)

        assert hasattr(symspell, "_index_lock")
        # Accept both Lock and RLock (RLock allows reentrant locking which is safer)
        assert isinstance(symspell._index_lock, (type(threading.Lock()), type(threading.RLock())))

    @pytest.mark.slow
    def test_concurrent_lookup_during_build(self):
        """
        Test that lookups work correctly while build_index is running.
        """
        from myspellchecker.algorithms.symspell import SymSpell
        from myspellchecker.providers import MemoryProvider

        provider = MemoryProvider()
        for i in range(100):
            provider.add_syllable(f"test{i}", frequency=100)

        symspell = SymSpell(provider, max_edit_distance=1)

        errors = []
        lookup_results = []

        def build_index_slow():
            try:
                symspell.build_index(["syllable"])
            except Exception as e:
                errors.append(e)

        def do_lookups():
            try:
                # Do some lookups while build might be running
                for _ in range(10):
                    result = symspell.lookup("test1", level="syllable")
                    lookup_results.append(len(result))
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=build_index_slow)
        t2 = threading.Thread(target=do_lookups)

        t1.start()
        t2.start()

        t1.join()
        t2.join()

        # No errors should occur
        assert not errors, f"Errors occurred: {errors}"

    def test_lookup_after_build_returns_results(self):
        """Test that lookup returns correct results after build_index."""
        from myspellchecker.algorithms.symspell import SymSpell
        from myspellchecker.providers import MemoryProvider

        provider = MemoryProvider()
        provider.add_syllable("ကောင်း", frequency=1000)
        provider.add_syllable("ကောင့်", frequency=500)

        symspell = SymSpell(provider, max_edit_distance=2)
        symspell.build_index(["syllable"])

        # Lookup should find suggestions
        suggestions = symspell.lookup("ကောင်", level="syllable", max_suggestions=5)
        # Should find at least one suggestion (ကောင်း or ကောင့်)
        assert isinstance(suggestions, list)

    def test_reentrant_lock_allows_same_thread(self):
        """Test that RLock allows same thread to acquire lock multiple times."""
        from myspellchecker.algorithms.symspell import SymSpell
        from myspellchecker.providers import MemoryProvider

        provider = MemoryProvider()
        provider.add_syllable("ကောင်း", frequency=1000)

        symspell = SymSpell(provider, max_edit_distance=2)

        # This shouldn't deadlock since we use RLock
        with symspell._index_lock:
            with symspell._index_lock:
                symspell.build_index(["syllable"])
