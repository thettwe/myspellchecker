"""
Real provider integration tests.

Tests covering:
- Provider integration with SpellChecker
- Provider switching at runtime
- Multiple provider types working together
- Provider caching behavior
- Error handling across provider boundaries
"""

import sqlite3
import tempfile
from pathlib import Path

import pytest

from myspellchecker import SpellChecker
from myspellchecker.core.config import SpellCheckerConfig
from myspellchecker.core.response import Response
from myspellchecker.providers import DictionaryProvider, MemoryProvider


class TestProviderSpellCheckerIntegration:
    """Test providers integrated with SpellChecker."""

    def test_spellchecker_with_memory_provider(self):
        """Test SpellChecker works with MemoryProvider."""
        provider = MemoryProvider()
        provider._syllables = {"ကောင်း": 100, "ပါ": 80, "တယ်": 60}
        provider._words = {"ကောင်းပါတယ်": 50}

        config = SpellCheckerConfig(
            fallback_to_empty_provider=True,
            use_context_checker=False,
        )
        checker = SpellChecker(config=config, provider=provider)

        result = checker.check("ကောင်းပါတယ်")
        assert isinstance(result, Response)

    def test_spellchecker_with_empty_memory_provider(self):
        """Test SpellChecker handles empty MemoryProvider."""
        provider = MemoryProvider()
        config = SpellCheckerConfig(
            fallback_to_empty_provider=True,
            use_context_checker=False,
        )
        checker = SpellChecker(config=config, provider=provider)

        # Should not crash even with empty dictionary
        result = checker.check("ကောင်းပါတယ်")
        assert isinstance(result, Response)

    def test_spellchecker_with_sqlite_provider(self):
        """Test SpellChecker works with SQLiteProvider."""
        from myspellchecker.providers import SQLiteProvider

        with tempfile.TemporaryDirectory() as tmpdir:
            database_path = Path(tmpdir) / "test.db"

            # Create minimal database
            conn = sqlite3.connect(str(database_path))
            conn.execute("""
                CREATE TABLE syllables (
                    syllable TEXT PRIMARY KEY,
                    frequency INTEGER
                )
            """)
            conn.execute("""
                CREATE TABLE words (
                    word TEXT PRIMARY KEY,
                    frequency INTEGER
                )
            """)
            conn.execute("INSERT INTO syllables VALUES ('ကောင်း', 100)")
            conn.execute("INSERT INTO words VALUES ('ကောင်းပါတယ်', 50)")
            conn.commit()
            conn.close()

            provider = SQLiteProvider(str(database_path))
            config = SpellCheckerConfig(
                fallback_to_empty_provider=True,
                use_context_checker=False,
            )
            checker = SpellChecker(config=config, provider=provider)

            result = checker.check("ကောင်းပါတယ်")
            assert isinstance(result, Response)

            provider.close()


class TestProviderDataConsistency:
    """Test data consistency across provider operations."""

    def test_memory_provider_syllable_word_consistency(self):
        """Test syllable and word data are consistent."""
        provider = MemoryProvider()

        # Add syllables
        provider._syllables = {
            "က": 100,
            "ကောင်း": 200,
            "ပါ": 150,
            "တယ်": 120,
        }

        # Add word composed of those syllables
        provider._words = {"ကောင်းပါတယ်": 50}

        # Both should work independently
        assert provider.is_valid_syllable("ကောင်း") is True
        assert provider.is_valid_word("ကောင်းပါတယ်") is True

    def test_provider_frequency_ordering(self):
        """Test frequency values maintain proper ordering."""
        provider = MemoryProvider()

        provider._syllables = {
            "က": 1000,  # High frequency
            "ခ": 500,  # Medium frequency
            "ဂ": 100,  # Low frequency
        }

        assert provider.get_syllable_frequency("က") > provider.get_syllable_frequency("ခ")
        assert provider.get_syllable_frequency("ခ") > provider.get_syllable_frequency("ဂ")

    def test_sqlite_provider_data_persistence(self):
        """Test SQLiteProvider data persists correctly."""
        from myspellchecker.providers import SQLiteProvider

        with tempfile.TemporaryDirectory() as tmpdir:
            database_path = Path(tmpdir) / "persist.db"

            # Create and populate database
            conn = sqlite3.connect(str(database_path))
            conn.execute("""
                CREATE TABLE syllables (
                    syllable TEXT PRIMARY KEY,
                    frequency INTEGER
                )
            """)
            conn.execute("""
                CREATE TABLE words (
                    word TEXT PRIMARY KEY,
                    frequency INTEGER
                )
            """)
            conn.execute("INSERT INTO syllables VALUES ('ပါ', 200)")
            conn.commit()
            conn.close()

            # First provider instance
            provider1 = SQLiteProvider(str(database_path))
            assert provider1.is_valid_syllable("ပါ") is True
            freq1 = provider1.get_syllable_frequency("ပါ")
            provider1.close()

            # Second provider instance - data should persist
            provider2 = SQLiteProvider(str(database_path))
            assert provider2.is_valid_syllable("ပါ") is True
            freq2 = provider2.get_syllable_frequency("ပါ")
            provider2.close()

            assert freq1 == freq2 == 200


class TestProviderErrorHandling:
    """Test error handling across provider operations."""

    def test_memory_provider_nonexistent_syllable(self):
        """Test MemoryProvider handles nonexistent syllables."""
        provider = MemoryProvider()

        assert provider.is_valid_syllable("nonexistent") is False
        assert provider.get_syllable_frequency("nonexistent") == 0

    def test_memory_provider_nonexistent_word(self):
        """Test MemoryProvider handles nonexistent words."""
        provider = MemoryProvider()

        assert provider.is_valid_word("nonexistent") is False
        assert provider.get_word_frequency("nonexistent") == 0

    def test_sqlite_provider_missing_database(self):
        """Test SQLiteProvider handles missing database."""
        from myspellchecker.core.exceptions import DataLoadingError
        from myspellchecker.providers import SQLiteProvider

        with pytest.raises(DataLoadingError, match="Database not found"):
            SQLiteProvider("/nonexistent/path/db.sqlite")

    def test_sqlite_provider_invalid_database(self):
        """Test SQLiteProvider raises error for invalid database file."""
        from myspellchecker.providers import SQLiteProvider

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            # Write invalid data
            f.write(b"not a sqlite database")
            path = f.name

        try:
            with pytest.raises(sqlite3.DatabaseError, match="file is not a database"):
                provider = SQLiteProvider(path)
                provider.is_valid_syllable("test")
        finally:
            Path(path).unlink()


class TestProviderCaching:
    """Test provider caching behavior."""

    def test_memory_provider_cache_coherence(self):
        """Test MemoryProvider maintains cache coherence."""
        provider = MemoryProvider()

        # Add data
        provider._syllables = {"က": 100}

        # First lookup
        result1 = provider.is_valid_syllable("က")

        # Modify data
        provider._syllables["က"] = 200

        # Second lookup should reflect update (no stale cache)
        freq = provider.get_syllable_frequency("က")
        assert result1 is True
        assert freq == 200

    def test_spellchecker_multiple_checks_same_provider(self):
        """Test multiple SpellChecker checks with same provider."""
        provider = MemoryProvider()
        provider._syllables = {"က": 100}

        config = SpellCheckerConfig(
            fallback_to_empty_provider=True,
            use_context_checker=False,
        )
        checker = SpellChecker(config=config, provider=provider)

        # Multiple checks should work
        for _ in range(10):
            result = checker.check("က")
            assert isinstance(result, Response)


class TestProviderBatchOperations:
    """Test provider behavior with batch operations."""

    def test_memory_provider_batch_check(self):
        """Test MemoryProvider with batch spell checking."""
        provider = MemoryProvider()
        provider._syllables = {
            "ကောင်း": 100,
            "ပါ": 80,
            "တယ်": 60,
        }

        config = SpellCheckerConfig(
            fallback_to_empty_provider=True,
            use_context_checker=False,
        )
        checker = SpellChecker(config=config, provider=provider)

        texts = ["ကောင်း", "ပါ", "တယ်"] * 10
        results = checker.check_batch(texts)

        assert len(results) == 30
        for result in results:
            assert isinstance(result, Response)

    def test_sqlite_provider_batch_check(self):
        """Test SQLiteProvider with batch spell checking."""
        from myspellchecker.providers import SQLiteProvider

        with tempfile.TemporaryDirectory() as tmpdir:
            database_path = Path(tmpdir) / "batch.db"

            conn = sqlite3.connect(str(database_path))
            conn.execute("""
                CREATE TABLE syllables (
                    syllable TEXT PRIMARY KEY,
                    frequency INTEGER
                )
            """)
            conn.execute("""
                CREATE TABLE words (
                    word TEXT PRIMARY KEY,
                    frequency INTEGER
                )
            """)
            for syl in ["က", "ခ", "ဂ"]:
                conn.execute(f"INSERT INTO syllables VALUES ('{syl}', 100)")
            conn.commit()
            conn.close()

            provider = SQLiteProvider(str(database_path))
            config = SpellCheckerConfig(
                fallback_to_empty_provider=True,
                use_context_checker=False,
            )
            checker = SpellChecker(config=config, provider=provider)

            texts = ["က", "ခ", "ဂ"] * 5
            results = checker.check_batch(texts)

            assert len(results) == 15
            provider.close()


class TestProviderTypeSpecific:
    """Test type-specific provider behaviors."""

    def test_json_provider_integration(self):
        """Test JSONProvider integration."""
        import json

        from myspellchecker.providers.json_provider import JSONProvider

        data = {
            "syllables": {"က": 100, "ကောင်း": 200},
            "words": {"ကောင်းပါတယ်": 50},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = f.name

        try:
            provider = JSONProvider(path)

            assert provider.is_valid_syllable("က") is True
            assert provider.is_valid_syllable("nonexistent") is False
            assert provider.get_syllable_frequency("ကောင်း") == 200
        finally:
            Path(path).unlink()

    def test_csv_provider_integration(self):
        """Test CSVProvider integration."""
        import csv

        from myspellchecker.providers.csv_provider import CSVProvider

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["syllable", "frequency"])
            writer.writerow(["က", "100"])
            writer.writerow(["ကောင်း", "200"])
            path = f.name

        try:
            provider = CSVProvider(path)

            assert provider.is_valid_syllable("က") is True
            assert provider.get_syllable_frequency("ကောင်း") == 200
        finally:
            Path(path).unlink()


class TestProviderInteroperability:
    """Test interoperability between different providers."""

    def test_provider_interface_consistency(self):
        """Test all providers implement consistent interface."""
        providers = [
            MemoryProvider(),
        ]

        for provider in providers:
            # All providers should have these methods
            assert hasattr(provider, "is_valid_syllable")
            assert hasattr(provider, "is_valid_word")
            assert hasattr(provider, "get_syllable_frequency")
            assert hasattr(provider, "get_word_frequency")

            # Methods should return correct types
            assert isinstance(provider.is_valid_syllable("test"), bool)
            assert isinstance(provider.is_valid_word("test"), bool)
            assert isinstance(provider.get_syllable_frequency("test"), int)
            assert isinstance(provider.get_word_frequency("test"), int)

    def test_spellchecker_accepts_any_provider(self):
        """Test SpellChecker accepts any DictionaryProvider implementation."""

        # Custom provider implementation
        class CustomProvider(DictionaryProvider):
            def is_valid_syllable(self, syllable):
                return True

            def is_valid_word(self, word):
                return True

            def get_syllable_frequency(self, syllable):
                return 100

            def get_word_frequency(self, word):
                return 50

            def get_word_pos(self, word):
                return None

            def get_bigram_probability(self, prev_word, current_word):
                return 0.0

            def get_trigram_probability(self, w1, w2, w3):
                return 0.0

            def get_fourgram_probability(self, w1, w2, w3, w4):
                return 0.0

            def get_fivegram_probability(self, w1, w2, w3, w4, w5):
                return 0.0

            def get_top_continuations(self, prev_word, limit=20):
                return []

            def get_all_syllables(self):
                return iter([])

            def get_all_words(self):
                return iter([])

            def get_pos_unigram_probabilities(self):
                return {}

            def get_pos_bigram_probabilities(self):
                return {}

            def get_pos_trigram_probabilities(self):
                return {}

        provider = CustomProvider()
        config = SpellCheckerConfig(
            fallback_to_empty_provider=True,
            use_context_checker=False,
        )
        checker = SpellChecker(config=config, provider=provider)

        result = checker.check("ကောင်းပါတယ်")
        assert isinstance(result, Response)


class TestProviderResourceManagement:
    """Test provider resource management."""

    def test_sqlite_provider_close(self):
        """Test SQLiteProvider closes properly."""
        from myspellchecker.providers import SQLiteProvider

        with tempfile.TemporaryDirectory() as tmpdir:
            database_path = Path(tmpdir) / "close.db"

            conn = sqlite3.connect(str(database_path))
            conn.execute("""
                CREATE TABLE syllables (
                    syllable TEXT PRIMARY KEY,
                    frequency INTEGER
                )
            """)
            conn.execute("""
                CREATE TABLE words (
                    word TEXT PRIMARY KEY,
                    frequency INTEGER
                )
            """)
            conn.commit()
            conn.close()

            provider = SQLiteProvider(str(database_path))
            provider.close()

            # After close, provider should still exist but may have limited functionality
            # (implementation-dependent)

    def test_memory_provider_no_close_needed(self):
        """Test MemoryProvider doesn't need explicit close."""
        provider = MemoryProvider()
        provider._syllables = {"က": 100}

        # Use provider
        assert provider.is_valid_syllable("က") is True

        # No close method needed for MemoryProvider
        # (but should handle if called)
        if hasattr(provider, "close"):
            provider.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
