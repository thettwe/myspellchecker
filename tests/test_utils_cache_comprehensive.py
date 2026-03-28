"""Comprehensive tests for utils/cache.py.

Tests cover:
- CacheConfig validation and factory methods
- LRUCache operations and eviction
- CacheManager multi-cache management
"""

import threading

import pytest

from myspellchecker.core.exceptions import CacheError
from myspellchecker.utils.cache import (
    Cache,
    CacheConfig,
    CacheManager,
    LRUCache,
)


class TestCacheConfig:
    """Tests for CacheConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = CacheConfig()
        assert config.maxsize == 1024
        assert config.ttl_seconds == 0.0
        assert config.enable_stats is True
        assert config.name == ""

    def test_custom_values(self):
        """Test custom configuration values."""
        config = CacheConfig(
            maxsize=512,
            ttl_seconds=60.0,
            enable_stats=False,
            name="test_cache",
        )
        assert config.maxsize == 512
        assert config.ttl_seconds == 60.0
        assert config.enable_stats is False
        assert config.name == "test_cache"

    def test_frozen_dataclass(self):
        """Test that CacheConfig is immutable."""
        config = CacheConfig()
        with pytest.raises(AttributeError):
            config.maxsize = 2048  # type: ignore

    def test_invalid_maxsize(self):
        """Test validation for invalid maxsize."""
        with pytest.raises(CacheError, match="maxsize must be >= 1"):
            CacheConfig(maxsize=0)

        with pytest.raises(CacheError, match="maxsize must be >= 1"):
            CacheConfig(maxsize=-1)

    def test_invalid_ttl(self):
        """Test validation for invalid ttl_seconds."""
        with pytest.raises(CacheError, match="ttl_seconds must be >= 0"):
            CacheConfig(ttl_seconds=-1.0)


class TestLRUCache:
    """Tests for LRUCache implementation."""

    def test_initialization(self):
        """Test cache initialization."""
        cache: LRUCache[int] = LRUCache(maxsize=100)
        assert cache.maxsize == 100
        assert len(cache) == 0

    def test_invalid_maxsize(self):
        """Test initialization with invalid maxsize."""
        with pytest.raises(CacheError, match="maxsize must be >= 1"):
            LRUCache(maxsize=0)

    def test_set_and_get(self):
        """Test basic set and get operations."""
        cache: LRUCache[str] = LRUCache()
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_get_missing_key(self):
        """Test get with missing key returns default."""
        cache: LRUCache[str] = LRUCache()
        assert cache.get("missing") is None
        assert cache.get("missing", "default") == "default"

    def test_contains(self):
        """Test __contains__ method."""
        cache: LRUCache[int] = LRUCache()
        cache.set("key1", 42)
        assert "key1" in cache
        assert "key2" not in cache

    def test_len(self):
        """Test __len__ method."""
        cache: LRUCache[int] = LRUCache()
        assert len(cache) == 0
        cache.set("key1", 1)
        assert len(cache) == 1
        cache.set("key2", 2)
        assert len(cache) == 2

    def test_clear(self):
        """Test clear method."""
        cache: LRUCache[int] = LRUCache()
        cache.set("key1", 1)
        cache.set("key2", 2)
        cache.clear()
        assert len(cache) == 0
        assert "key1" not in cache

    def test_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        cache: LRUCache[int] = LRUCache(maxsize=2)
        cache.set("key1", 1)
        cache.set("key2", 2)
        # Cache is full, adding key3 should evict key1
        cache.set("key3", 3)
        assert "key1" not in cache
        assert "key2" in cache
        assert "key3" in cache

    def test_access_updates_lru_order(self):
        """Test that accessing an item moves it to end."""
        cache: LRUCache[int] = LRUCache(maxsize=2)
        cache.set("key1", 1)
        cache.set("key2", 2)
        # Access key1 to make it most recently used
        cache.get("key1")
        # Adding key3 should evict key2 (least recently used)
        cache.set("key3", 3)
        assert "key1" in cache
        assert "key2" not in cache
        assert "key3" in cache

    def test_update_existing_key(self):
        """Test updating an existing key."""
        cache: LRUCache[int] = LRUCache()
        cache.set("key1", 1)
        cache.set("key1", 100)
        assert cache.get("key1") == 100
        assert len(cache) == 1

    def test_stats(self):
        """Test statistics tracking."""
        cache: LRUCache[int] = LRUCache(maxsize=10)
        cache.set("key1", 1)
        cache.get("key1")  # Hit
        cache.get("key2")  # Miss

        stats = cache.stats()
        assert stats["size"] == 1
        assert stats["maxsize"] == 10
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5

    def test_stats_empty_cache(self):
        """Test statistics for empty cache."""
        cache: LRUCache[int] = LRUCache()
        stats = cache.stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["hit_rate"] == 0.0

    def test_thread_safety(self):
        """Test thread-safe operations."""
        cache: LRUCache[int] = LRUCache(maxsize=1000)
        errors = []

        def writer(start: int):
            try:
                for i in range(100):
                    cache.set(f"key{start + i}", i)
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(100):
                    cache.get("key50")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer, args=(0,)),
            threading.Thread(target=writer, args=(100,)),
            threading.Thread(target=reader),
            threading.Thread(target=reader),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


class TestCacheManager:
    """Tests for CacheManager."""

    def test_initialization(self):
        """Test manager initialization."""
        manager = CacheManager(default_maxsize=512)
        assert manager.default_maxsize == 512

    def test_invalid_default_maxsize(self):
        """Test initialization with invalid default_maxsize."""
        with pytest.raises(CacheError, match="default_maxsize must be >= 1"):
            CacheManager(default_maxsize=0)

    def test_get_cache_creates_new(self):
        """Test get_cache creates new cache if not exists."""
        manager = CacheManager()
        cache = manager.get_cache("test")
        assert cache is not None

    def test_get_cache_returns_existing(self):
        """Test get_cache returns existing cache."""
        manager = CacheManager()
        cache1 = manager.get_cache("test")
        cache2 = manager.get_cache("test")
        assert cache1 is cache2

    def test_get_cache_with_custom_maxsize(self):
        """Test get_cache with custom maxsize."""
        manager = CacheManager(default_maxsize=100)
        cache = manager.get_cache("custom", maxsize=500)
        assert cache.maxsize == 500


class TestCacheProtocol:
    """Tests for Cache protocol compliance."""

    def test_lru_cache_implements_protocol(self):
        """Test that LRUCache implements Cache protocol."""
        cache: Cache[int] = LRUCache()

        # Test all protocol methods
        cache.set("key", 42)
        assert cache.get("key") == 42
        assert "key" in cache
        assert len(cache) == 1
        assert "size" in cache.stats()
        cache.clear()
        assert len(cache) == 0

    def test_isinstance_check(self):
        """Test runtime_checkable protocol with isinstance."""
        lru = LRUCache[int]()
        assert isinstance(lru, Cache)
