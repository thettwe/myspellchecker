"""
Generic LRU cache and cache management utilities.

This module provides reusable caching infrastructure:
- Cache: Protocol defining the cache interface
- CacheConfig: Configuration dataclass for cache settings
- LRUCache: Thread-safe LRU cache with bounded size
- CacheManager: Manages multiple named caches with unified API

These are general-purpose utilities that can be used across the codebase
for any caching needs.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import (
    Any,
    Generic,
    Protocol,
    TypeVar,
    runtime_checkable,
)

from myspellchecker.core.exceptions import CacheError

from .logging_utils import get_logger

__all__ = [
    "Cache",
    "CacheConfig",
    "LRUCache",
    "CacheManager",
]

# Type variable for cache values
T = TypeVar("T")
K = TypeVar("K")  # Key type


@dataclass(frozen=True)
class CacheConfig:
    """Configuration for cache instances.

    Attributes:
        maxsize: Maximum number of items in cache.
        ttl_seconds: Time-to-live in seconds (0 = no expiration).
        enable_stats: Whether to track hit/miss statistics.
        name: Optional name for the cache (for logging).
    """

    maxsize: int = 1024
    ttl_seconds: float = 0.0
    enable_stats: bool = True
    name: str = ""

    def __post_init__(self) -> None:
        if self.maxsize < 1:
            raise CacheError(f"maxsize must be >= 1, got {self.maxsize}")
        if self.ttl_seconds < 0:
            raise CacheError(f"ttl_seconds must be >= 0, got {self.ttl_seconds}")


@runtime_checkable
class Cache(Protocol[T]):
    """Protocol defining the cache interface.

    All cache implementations must provide these methods.
    This allows for different backends (memory, disk, Redis)
    to be swapped transparently.

    Example:
        >>> def process_with_cache(cache: Cache[int], key: str) -> int:
        ...     result = cache.get(key)
        ...     if result is None:
        ...         result = expensive_computation()
        ...         cache.set(key, result)
        ...     return result
    """

    def get(self, key: Any, default: T | None = None) -> T | None:
        """Get value from cache."""
        ...

    def set(self, key: Any, value: T) -> None:
        """Set value in cache."""
        ...

    def __contains__(self, key: Any) -> bool:
        """Check if key exists in cache."""
        ...

    def __len__(self) -> int:
        """Return current cache size."""
        ...

    def clear(self) -> None:
        """Clear all cached entries."""
        ...

    def stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        ...


class LRUCache(Generic[T]):
    """
    Thread-safe LRU cache with bounded size.

    Provides a simple key-value cache with Least Recently Used eviction policy.
    All operations are thread-safe via a reentrant lock.

    Attributes:
        maxsize: Maximum number of items to store in cache.

    Example:
        >>> cache = LRUCache[int](maxsize=100)
        >>> cache.set("key1", 42)
        >>> cache.get("key1")
        42
        >>> "key1" in cache
        True
    """

    def __init__(self, maxsize: int = 1024):
        """
        Initialize the LRU cache.

        Args:
            maxsize: Maximum number of items to store (default: 1024).
                    When capacity is reached, oldest items are evicted.
        """
        if maxsize < 1:
            raise CacheError(f"maxsize must be >= 1, got {maxsize}")
        self._cache: OrderedDict[Any, T] = OrderedDict()
        self._maxsize = maxsize
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
        self.logger = get_logger(__name__)

    @property
    def maxsize(self) -> int:
        """Maximum cache size."""
        return self._maxsize

    def get(self, key: Any, default: T | None = None) -> T | None:
        """
        Get value from cache, moving to end if found.

        Args:
            key: The cache key to look up (any hashable type).
            default: Value to return if key not found (default: None).

        Returns:
            Cached value if found, otherwise default.
        """
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key]
            self._misses += 1
            return default

    def set(self, key: Any, value: T) -> None:
        """
        Set value in cache, evicting oldest if at capacity.

        Args:
            key: The cache key (any hashable type).
            value: The value to cache.
        """
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._cache[key] = value
            else:
                if len(self._cache) >= self._maxsize:
                    self._cache.popitem(last=False)
                self._cache[key] = value

    def __contains__(self, key: Any) -> bool:
        """Check if key exists in cache."""
        with self._lock:
            return key in self._cache

    def __len__(self) -> int:
        """Return current cache size."""
        with self._lock:
            return len(self._cache)

    def clear(self) -> None:
        """Clear all cached entries and reset statistics."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def stats(self) -> dict[str, Any]:
        """
        Return cache statistics.

        Returns:
            Dictionary with size, maxsize, hits, misses, and hit_rate.
        """
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._cache),
                "maxsize": self._maxsize,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / total if total > 0 else 0.0,
            }


class CacheManager:
    """
    Manages multiple named caches with unified API.

    Provides a central point for managing multiple LRU caches,
    with support for global operations like clearing all caches
    and collecting statistics.

    Example:
        >>> manager = CacheManager(default_maxsize=512)
        >>> word_cache = manager.get_cache("words")
        >>> syllable_cache = manager.get_cache("syllables", maxsize=256)
        >>>
        >>> word_cache.set("hello", 100)
        >>> manager.get_all_stats()
        {'words': {...}, 'syllables': {...}}
        >>> manager.clear_all()
    """

    def __init__(self, default_maxsize: int = 1024):
        """
        Initialize the cache manager.

        Args:
            default_maxsize: Default max size for new caches (default: 1024).
        """
        if default_maxsize < 1:
            raise CacheError(f"default_maxsize must be >= 1, got {default_maxsize}")
        self._caches: dict[str, LRUCache] = {}
        self._default_maxsize = default_maxsize
        self._lock = threading.RLock()
        self.logger = get_logger(__name__)

    @property
    def default_maxsize(self) -> int:
        """Default maximum size for new caches."""
        return self._default_maxsize

    def clear_all(self) -> None:
        """Clear all managed caches."""
        with self._lock:
            for cache in self._caches.values():
                cache.clear()

    def get_all_stats(self) -> dict[str, dict[str, Any]]:
        """Return statistics for all managed caches."""
        with self._lock:
            return {name: cache.stats() for name, cache in self._caches.items()}

    def get_cache(self, name: str, maxsize: int | None = None) -> LRUCache:
        """
        Get or create a named cache.

        If a cache with the given name doesn't exist, one is created
        with the specified maxsize (or default_maxsize if not specified).

        Args:
            name: The cache name.
            maxsize: Maximum cache size (optional, uses default if not specified).

        Returns:
            The named LRUCache instance.
        """
        with self._lock:
            if name not in self._caches:
                size = maxsize or self._default_maxsize
                self._caches[name] = LRUCache(maxsize=size)
                self.logger.debug(f"Created cache '{name}' with maxsize={size}")
            return self._caches[name]
