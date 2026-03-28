import sqlite3
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from myspellchecker.core.exceptions import DataLoadingError
from myspellchecker.providers.sqlite import LRUCache, SQLiteProvider


def test_lru_cache():
    """Test LRUCache functionality."""
    cache = LRUCache(maxsize=2)
    cache.set("a", 1)
    cache.set("b", 2)
    assert len(cache) == 2
    assert "a" in cache

    # Access 'a' to make it most recently used
    assert cache.get("a") == 1

    # Add 'c', should evict 'b' (LRU) because 'a' was just used
    cache.set("c", 3)
    assert len(cache) == 2
    assert "a" in cache
    assert "c" in cache
    assert "b" not in cache

    # Test update
    cache.set("a", 10)
    assert cache.get("a") == 10

    # Test clear
    cache.clear()
    assert len(cache) == 0

    # Test stats
    stats = cache.stats()
    assert stats["size"] == 0
    assert stats["maxsize"] == 2


def test_sqlite_provider_init_error():
    """Test initialization error when DB missing."""
    with pytest.raises(DataLoadingError):
        SQLiteProvider(database_path="nonexistent.db")


def test_sqlite_provider_pool_stats():
    """Test get_pool_stats."""
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        # Create dummy DB
        conn = sqlite3.connect(tmp.name)
        conn.execute(
            "CREATE TABLE syllables (id INTEGER PRIMARY KEY, syllable TEXT, frequency INTEGER)"
        )
        conn.close()

        provider = SQLiteProvider(database_path=tmp.name)
        stats = provider.get_pool_stats()
        assert stats is not None
        assert "pool_size" in stats
        provider.close()


def test_sqlite_provider_cache_stats():
    """Test get_cache_stats."""
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        # Create dummy DB
        conn = sqlite3.connect(tmp.name)
        conn.close()

        provider = SQLiteProvider(database_path=tmp.name)
        stats = provider.get_cache_stats()
        assert "word_id_cache" in stats
        provider.close()


def test_sqlite_provider_clear_caches():
    """Test clear_caches."""
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        conn = sqlite3.connect(tmp.name)
        conn.close()

        provider = SQLiteProvider(database_path=tmp.name)
        provider._word_id_cache.set("test", 1)
        provider.clear_caches()
        assert len(provider._word_id_cache) == 0
        provider.close()


def test_sqlite_provider_bulk_operations():
    """Test bulk validation and retrieval."""
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        conn = sqlite3.connect(tmp.name)
        conn.execute(
            "CREATE TABLE syllables (id INTEGER PRIMARY KEY, syllable TEXT, frequency INTEGER)"
        )
        conn.execute(
            "CREATE TABLE words (id INTEGER PRIMARY KEY, word TEXT, "
            "frequency INTEGER, is_curated INTEGER DEFAULT 0)"
        )
        conn.execute("CREATE INDEX idx_syllables_syllable ON syllables(syllable)")
        conn.execute("CREATE INDEX idx_words_word ON words(word)")

        conn.execute("INSERT INTO syllables (syllable, frequency) VALUES (?, ?)", ("s1", 10))
        conn.execute(
            "INSERT INTO words (word, frequency, is_curated) VALUES (?, ?, ?)", ("w1", 100, 1)
        )
        conn.commit()
        conn.close()

        provider = SQLiteProvider(database_path=tmp.name)

        # Syllables bulk
        res = provider.is_valid_syllables_bulk(["s1", "s2"])
        assert res["s1"] is True
        assert res["s2"] is False

        freqs = provider.get_syllable_frequencies_bulk(["s1", "s2"])
        assert freqs["s1"] == 10
        assert freqs.get("s2", 0) == 0

        # Words bulk
        res = provider.is_valid_words_bulk(["w1", "w2"])
        assert res["w1"] is True
        assert res["w2"] is False

        freqs = provider.get_word_frequencies_bulk(["w1", "w2"])
        assert freqs["w1"] == 100

        # Vocabulary bulk
        res = provider.is_valid_vocabulary_bulk(["w1", "w2"])
        assert res["w1"] is True

        provider.close()


def test_sqlite_provider_get_word_pos_bulk_missing_table():
    """Test get_word_pos_bulk handles missing columns gracefully."""
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        conn = sqlite3.connect(tmp.name)
        conn.execute("CREATE TABLE words (id INTEGER PRIMARY KEY, word TEXT)")
        conn.commit()
        conn.close()

        provider = SQLiteProvider(database_path=tmp.name)
        res = provider.get_word_pos_bulk(["w1"])
        assert res["w1"] is None
        provider.close()


def test_sqlite_provider_get_statistics_missing_tables():
    """Test get_statistics with minimal tables."""
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        conn = sqlite3.connect(tmp.name)
        conn.execute("CREATE TABLE syllables (id INTEGER PRIMARY KEY)")
        conn.execute("CREATE TABLE words (id INTEGER PRIMARY KEY)")
        conn.execute("CREATE TABLE bigrams (id INTEGER PRIMARY KEY)")
        conn.commit()
        conn.close()

        provider = SQLiteProvider(database_path=tmp.name)
        stats = provider.get_statistics()
        assert stats["syllable_count"] == 0
        assert stats["trigram_count"] == 0  # Missing table
        provider.close()


def test_sqlite_provider_get_word_pos_fallback():
    """Test get_word_pos fallback chain."""
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        conn = sqlite3.connect(tmp.name)
        conn.execute(
            "CREATE TABLE words (id INTEGER PRIMARY KEY, word TEXT, "
            "pos_tag TEXT, inferred_pos TEXT)"
        )
        conn.commit()
        conn.close()

        mock_tagger = MagicMock()
        mock_tagger.tag_word.return_value = "NOUN"

        provider = SQLiteProvider(database_path=tmp.name, pos_tagger=mock_tagger)

        # Word not in DB, should use tagger
        pos = provider.get_word_pos("unknown")
        assert pos == "NOUN"
        mock_tagger.tag_word.assert_called_with("unknown")

        provider.close()


def test_sqlite_provider_get_word_id_caching():
    """Test get_word_id caching."""
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        conn = sqlite3.connect(tmp.name)
        conn.execute("CREATE TABLE words (id INTEGER PRIMARY KEY, word TEXT)")
        conn.execute("INSERT INTO words (word) VALUES (?)", ("w1",))
        conn.commit()
        conn.close()

        provider = SQLiteProvider(database_path=tmp.name)

        # First call hits DB
        id1 = provider.get_word_id("w1")
        assert id1 is not None

        # Second call hits cache (mock DB to verify)
        with patch.object(provider, "_execute_query", side_effect=Exception("Should not hit DB")):
            id2 = provider.get_word_id("w1")
            assert id2 == id1

        provider.close()


def test_is_valid_vocabulary_fallback():
    """Test is_valid_vocabulary fallback when column missing."""
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        conn = sqlite3.connect(tmp.name)
        conn.execute("CREATE TABLE words (id INTEGER PRIMARY KEY, word TEXT)")
        conn.execute("INSERT INTO words (word) VALUES (?)", ("w1",))
        conn.commit()
        conn.close()

        provider = SQLiteProvider(database_path=tmp.name)
        # Should fallback to is_valid_word -> True
        assert provider.is_valid_vocabulary("w1") is True
        provider.close()


def test_get_word_pos_suffix_rules():
    """Test POS suffix transformation rules."""
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        conn = sqlite3.connect(tmp.name)
        conn.execute(
            "CREATE TABLE words (id INTEGER PRIMARY KEY, word TEXT, "
            "pos_tag TEXT, inferred_pos TEXT)"
        )
        conn.execute("INSERT INTO words (word, pos_tag) VALUES (?, ?)", ("root", "V"))
        conn.commit()
        conn.close()

        provider = SQLiteProvider(database_path=tmp.name)

        # Mock stemmer (update both provider and resolver references)
        mock_stemmer = MagicMock()
        noun_suffix = "suffix"
        mock_stemmer.stem.return_value = ("root", [noun_suffix])
        provider.stemmer = mock_stemmer
        provider._pos_resolver._stemmer = mock_stemmer

        # Patch NOUN_SUFFIXES to include our test suffix
        with patch("myspellchecker.providers._sqlite_pos_resolver.NOUN_SUFFIXES", {noun_suffix}):
            # Should transform V -> N
            pos = provider.get_word_pos(f"root{noun_suffix}")
            assert pos == "N"

        provider.close()


def test_ngram_probabilities_missing_ids():
    """Test ngram probability methods return 0 when IDs missing."""
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        conn = sqlite3.connect(tmp.name)
        conn.execute("CREATE TABLE words (id INTEGER PRIMARY KEY, word TEXT)")
        conn.close()

        provider = SQLiteProvider(database_path=tmp.name)

        # w1 not in DB -> ID None -> returns 0.0
        assert provider.get_bigram_probability("w1", "w2") == 0.0
        assert provider.get_trigram_probability("w1", "w2", "w3") == 0.0

        provider.close()


def test_pos_probabilities_tables():
    """Test POS probability methods with and without tables."""
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        conn = sqlite3.connect(tmp.name)
        conn.execute("CREATE TABLE pos_unigrams (pos TEXT, probability REAL)")
        conn.execute("INSERT INTO pos_unigrams VALUES ('N', 0.5)")
        conn.commit()
        conn.close()

        provider = SQLiteProvider(database_path=tmp.name)

        # Unigrams exist
        unis = provider.get_pos_unigram_probabilities()
        assert unis["N"] == 0.5

        # Bigrams/Trigrams missing -> empty dict, warning logged
        assert provider.get_pos_bigram_probabilities() == {}
        assert provider.get_pos_trigram_probabilities() == {}

        provider.close()


def test_top_continuations():
    """Test get_top_continuations."""
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        conn = sqlite3.connect(tmp.name)
        conn.execute("CREATE TABLE words (id INTEGER PRIMARY KEY, word TEXT)")
        conn.execute("CREATE TABLE bigrams (word1_id INTEGER, word2_id INTEGER, probability REAL)")

        conn.execute("INSERT INTO words (id, word) VALUES (1, 'w1'), (2, 'w2'), (3, 'w3')")
        conn.execute("INSERT INTO bigrams VALUES (1, 2, 0.8), (1, 3, 0.2)")
        conn.commit()
        conn.close()

        provider = SQLiteProvider(database_path=tmp.name)

        # w1 -> w2 (0.8), w3 (0.2)
        conts = provider.get_top_continuations("w1")
        assert len(conts) == 2
        assert conts[0] == ("w2", 0.8)
        assert conts[1] == ("w3", 0.2)

        # Unknown word
        assert provider.get_top_continuations("unknown") == []

        provider.close()


def test_iterators():
    """Test get_all_* iterators."""
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        conn = sqlite3.connect(tmp.name)
        conn.execute("CREATE TABLE syllables (syllable TEXT, frequency INTEGER)")
        conn.execute("INSERT INTO syllables VALUES ('s1', 10)")

        conn.execute("CREATE TABLE words (word TEXT, frequency INTEGER)")
        conn.execute("INSERT INTO words VALUES ('w1', 100)")
        conn.commit()
        conn.close()

        provider = SQLiteProvider(database_path=tmp.name)

        assert list(provider.get_all_syllables()) == [("s1", 10)]
        assert list(provider.get_all_words()) == [("w1", 100)]

        provider.close()


def test_close_idempotent():
    """Test that close can be called multiple times."""
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        conn = sqlite3.connect(tmp.name)
        conn.close()

        provider = SQLiteProvider(database_path=tmp.name)
        provider.close()
        provider.close()  # Should not error
