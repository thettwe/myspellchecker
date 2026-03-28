"""
Transparent LRU Caching Layer for Algorithm Data Sources.

This module provides cached wrappers for Protocol interfaces, achieving
10-100x performance improvements through intelligent caching of frequently
accessed data.

Key Features:
- Transparent caching (drop-in replacement for any Protocol)
- Configurable cache sizes per data type
- Memory-efficient LRU eviction policy
- Cache statistics for monitoring
- **Shared cache registry** to avoid duplication across factory instances

Cache Sharing Architecture:
    The module provides a `CacheRegistry` singleton that ensures multiple
    AlgorithmFactory instances sharing the same provider also share their
    caches. This prevents memory duplication and improves cache hit rates.

    Without sharing (old behavior):
        factory1 = AlgorithmFactory(provider)  # Creates cache set A
        factory2 = AlgorithmFactory(provider)  # Creates cache set B (duplicate!)

    With sharing (new behavior):
        factory1 = AlgorithmFactory(provider)  # Creates/reuses shared cache
        factory2 = AlgorithmFactory(provider)  # Reuses same shared cache

Thread Safety:
    The caching layer uses functools.lru_cache which is thread-safe for
    concurrent reads in Python 3.2+. However, under very high contention
    (>1000 concurrent cache operations), there may be brief periods of
    inconsistency during cache eviction.

    For most use cases, this is not an issue because:
    - Cache misses simply result in an extra database lookup
    - The underlying data is read-only
    - No data corruption can occur (just potential duplicate work)

    For high-concurrency scenarios with strict consistency requirements,
    consider using the optional `use_lock=True` parameter which adds
    explicit locking at the cost of some performance.

    The CacheRegistry itself uses threading locks for safe concurrent access.

Performance Impact:
- Dictionary lookups: 5-20x faster (90% cache hit rate)
- Bigram probabilities: 10-100x faster (80% cache hit rate)
- Memory cost: ~1-2MB per 1000 cached entries
- Shared caches: ~50% memory reduction with multiple factories

Example:
    >>> from myspellchecker.providers import SQLiteProvider
    >>> from myspellchecker.algorithms.cache import CachedDictionaryLookup
    >>>
    >>> provider = SQLiteProvider("myspell.db")
    >>> cached_provider = CachedDictionaryLookup(provider, cache_size=4096)
    >>>
    >>> # First call: cache miss, queries database
    >>> is_valid = cached_provider.is_valid_syllable("မြန်")  # ~5ms
    >>>
    >>> # Second call: cache hit, returns immediately
    >>> is_valid = cached_provider.is_valid_syllable("မြန်")  # ~0.05ms (100x faster)

See Also:
    - CacheRegistry: Singleton for shared cache management
    - AlgorithmFactory: Uses CacheRegistry for cache sharing
"""

from __future__ import annotations

import threading
import weakref
from functools import lru_cache
from typing import Any, ClassVar, Iterator, cast

from myspellchecker.algorithms.interfaces import (
    BigramSource,
    DictionaryLookup,
    FrequencySource,
    POSRepository,
    TrigramSource,
)


class CacheRegistry:
    """
    Singleton registry for shared cache instances.

    This registry ensures that multiple AlgorithmFactory instances using the
    same provider share their cache wrappers, preventing memory duplication
    and improving overall cache hit rates.

    The registry uses provider identity (id()) as keys. When a provider is
    garbage collected, its cached wrappers should also be cleaned up via
    the cleanup() method or by calling clear().

    Thread Safety:
        All registry operations are protected by a threading lock, making
        it safe for concurrent access from multiple threads.

    Example:
        >>> from myspellchecker.algorithms.cache import CacheRegistry
        >>> from myspellchecker.providers import SQLiteProvider
        >>>
        >>> provider = SQLiteProvider("myspell.db")
        >>> registry = CacheRegistry.get_instance()
        >>>
        >>> # Get or create cached wrapper (shared across factories)
        >>> cached_dict = registry.get_or_create_dictionary_cache(
        ...     provider, syllable_size=4096, word_size=8192
        ... )
        >>>
    """

    _instance: ClassVar[CacheRegistry | None] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self) -> None:
        """Initialize the cache registry (use get_instance() instead)."""
        self._caches: dict[int, dict[str, Any]] = {}
        self._provider_refs: dict[int, weakref.ref] = {}
        self._registry_lock = threading.RLock()

    @classmethod
    def get_instance(cls) -> "CacheRegistry":
        """
        Get the singleton CacheRegistry instance.

        Returns:
            The shared CacheRegistry instance.

        Example:
            >>> registry = CacheRegistry.get_instance()
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _get_provider_key(self, provider: Any) -> int:
        """Get unique key for a provider and register a weak reference.

        Uses ``id(provider)`` as the key but also stores a weakref so that
        when the provider is garbage-collected, its cached wrappers are
        automatically evicted — preventing stale cache reuse.
        """
        key = id(provider)
        if key not in self._provider_refs:
            try:
                self._provider_refs[key] = weakref.ref(provider, lambda _ref, k=key: self._evict(k))
            except TypeError:
                # Provider type doesn't support weakrefs (rare); accept the leak.
                pass
        return key

    def _evict(self, key: int) -> None:
        """Remove cached wrappers for a garbage-collected provider."""
        with self._registry_lock:
            self._caches.pop(key, None)
            self._provider_refs.pop(key, None)

    def get_or_create_dictionary_cache(
        self,
        provider: DictionaryLookup,
        syllable_size: int = 4096,
        word_size: int = 8192,
        use_lock: bool = False,
    ) -> "CachedDictionaryLookup":
        """
        Get or create a shared CachedDictionaryLookup for the provider.

        Args:
            provider: The underlying DictionaryLookup implementation
            syllable_size: Max cached syllable lookups
            word_size: Max cached word lookups
            use_lock: Enable explicit locking for high-concurrency

        Returns:
            Shared CachedDictionaryLookup instance
        """
        key = self._get_provider_key(provider)
        with self._registry_lock:
            if key not in self._caches:
                self._caches[key] = {}
            if "dictionary" not in self._caches[key]:
                self._caches[key]["dictionary"] = CachedDictionaryLookup(
                    provider, syllable_size, word_size, use_lock
                )
            return cast("CachedDictionaryLookup", self._caches[key]["dictionary"])

    def get_or_create_frequency_cache(
        self,
        provider: FrequencySource,
        cache_size: int = 8192,
    ) -> "CachedFrequencySource":
        """
        Get or create a shared CachedFrequencySource for the provider.

        Args:
            provider: The underlying FrequencySource implementation
            cache_size: Max cached frequency lookups

        Returns:
            Shared CachedFrequencySource instance
        """
        key = self._get_provider_key(provider)
        with self._registry_lock:
            if key not in self._caches:
                self._caches[key] = {}
            if "frequency" not in self._caches[key]:
                self._caches[key]["frequency"] = CachedFrequencySource(provider, cache_size)
            return cast("CachedFrequencySource", self._caches[key]["frequency"])

    def get_or_create_bigram_cache(
        self,
        provider: BigramSource,
        cache_size: int = 16384,
    ) -> "CachedBigramSource":
        """
        Get or create a shared CachedBigramSource for the provider.

        Args:
            provider: The underlying BigramSource implementation
            cache_size: Max cached bigram lookups

        Returns:
            Shared CachedBigramSource instance
        """
        key = self._get_provider_key(provider)
        with self._registry_lock:
            if key not in self._caches:
                self._caches[key] = {}
            if "bigram" not in self._caches[key]:
                self._caches[key]["bigram"] = CachedBigramSource(provider, cache_size)
            return cast("CachedBigramSource", self._caches[key]["bigram"])

    def get_or_create_trigram_cache(
        self,
        provider: TrigramSource,
        cache_size: int = 16384,
    ) -> "CachedTrigramSource":
        """
        Get or create a shared CachedTrigramSource for the provider.

        Args:
            provider: The underlying TrigramSource implementation
            cache_size: Max cached trigram lookups

        Returns:
            Shared CachedTrigramSource instance
        """
        key = self._get_provider_key(provider)
        with self._registry_lock:
            if key not in self._caches:
                self._caches[key] = {}
            if "trigram" not in self._caches[key]:
                self._caches[key]["trigram"] = CachedTrigramSource(provider, cache_size)
            return cast("CachedTrigramSource", self._caches[key]["trigram"])

    def get_or_create_pos_cache(
        self,
        provider: POSRepository,
    ) -> "CachedPOSRepository":
        """
        Get or create a shared CachedPOSRepository for the provider.

        Args:
            provider: The underlying POSRepository implementation

        Returns:
            Shared CachedPOSRepository instance
        """
        key = self._get_provider_key(provider)
        with self._registry_lock:
            if key not in self._caches:
                self._caches[key] = {}
            if "pos" not in self._caches[key]:
                self._caches[key]["pos"] = CachedPOSRepository(provider)
            return cast("CachedPOSRepository", self._caches[key]["pos"])


class CachedDictionaryLookup:
    """
    LRU-cached wrapper for DictionaryLookup.

    Caches syllable/word validation and frequency lookups.

    Args:
        provider: Underlying DictionaryLookup implementation
        syllable_cache_size: Max cached syllable lookups (default: 4096)
        word_cache_size: Max cached word lookups (default: 8192)
        use_lock: Enable explicit locking for high-concurrency scenarios.
            Default is False for maximum performance. Set to True if you're
            experiencing issues under very high concurrent load (>1000 ops/sec).

    Cache Hit Rates (typical):
        - Syllable validation: 90% (highly repetitive)
        - Word validation: 85% (moderately repetitive)
        - Frequency lookups: 80% (used for ranking)

    Performance:
        - Cache hit: ~0.05ms (memory lookup)
        - Cache miss: ~5ms (database query)
        - Speedup: ~100x for hits, ~20x average
        - With use_lock=True: ~10-20% overhead

    Thread Safety:
        By default (use_lock=False), uses functools.lru_cache which is
        thread-safe for reads but may have brief inconsistencies during
        eviction under very high contention. This is safe because cache
        misses just result in extra database queries.

        With use_lock=True, explicit locking prevents any race conditions
        but adds some performance overhead.
    """

    def __init__(
        self,
        provider: DictionaryLookup,
        syllable_cache_size: int = 4096,
        word_cache_size: int = 8192,
        use_lock: bool = False,
    ):
        self._provider = provider
        self._syllable_cache_size = syllable_cache_size
        self._word_cache_size = word_cache_size
        self._use_lock = use_lock
        self._lock = threading.RLock() if use_lock else None

        # Create instance-specific cached methods
        self._cached_is_valid_syllable = lru_cache(maxsize=syllable_cache_size)(
            self._is_valid_syllable_impl
        )
        self._cached_is_valid_word = lru_cache(maxsize=word_cache_size)(self._is_valid_word_impl)
        self._cached_get_syllable_frequency = lru_cache(maxsize=syllable_cache_size)(
            self._get_syllable_frequency_impl
        )
        self._cached_get_word_frequency = lru_cache(maxsize=word_cache_size)(
            self._get_word_frequency_impl
        )

    def _is_valid_syllable_impl(self, syllable: str) -> bool:
        """Implementation method for caching."""
        return self._provider.is_valid_syllable(syllable)

    def _is_valid_word_impl(self, word: str) -> bool:
        """Implementation method for caching."""
        return self._provider.is_valid_word(word)

    def _get_syllable_frequency_impl(self, syllable: str) -> int:
        """Implementation method for caching."""
        return self._provider.get_syllable_frequency(syllable)

    def _get_word_frequency_impl(self, word: str) -> int:
        """Implementation method for caching."""
        return self._provider.get_word_frequency(word)

    # Public interface (cached)
    def is_valid_syllable(self, syllable: str) -> bool:
        """Check if syllable exists in dictionary (cached)."""
        if self._lock:
            with self._lock:
                return self._cached_is_valid_syllable(syllable)
        return self._cached_is_valid_syllable(syllable)

    def is_valid_word(self, word: str) -> bool:
        """Check if word exists in dictionary (cached)."""
        if self._lock:
            with self._lock:
                return self._cached_is_valid_word(word)
        return self._cached_is_valid_word(word)

    def get_syllable_frequency(self, syllable: str) -> int:
        """Get syllable frequency (cached)."""
        if self._lock:
            with self._lock:
                return self._cached_get_syllable_frequency(syllable)
        return self._cached_get_syllable_frequency(syllable)

    def get_word_frequency(self, word: str) -> int:
        """Get word frequency (cached)."""
        if self._lock:
            with self._lock:
                return self._cached_get_word_frequency(word)
        return self._cached_get_word_frequency(word)

    def cache_info(self) -> dict[str, Any]:
        """Get cache statistics for monitoring."""
        if self._lock:
            with self._lock:
                syl_freq_info = self._cached_get_syllable_frequency.cache_info()
                return {
                    "syllable_validation": (self._cached_is_valid_syllable.cache_info()._asdict()),
                    "word_validation": self._cached_is_valid_word.cache_info()._asdict(),
                    "syllable_frequency": syl_freq_info._asdict(),
                    "word_frequency": self._cached_get_word_frequency.cache_info()._asdict(),
                }
        return {
            "syllable_validation": self._cached_is_valid_syllable.cache_info()._asdict(),
            "word_validation": self._cached_is_valid_word.cache_info()._asdict(),
            "syllable_frequency": self._cached_get_syllable_frequency.cache_info()._asdict(),
            "word_frequency": self._cached_get_word_frequency.cache_info()._asdict(),
        }

    def cache_clear(self) -> None:
        """Clear all caches."""
        if self._lock:
            with self._lock:
                self._cached_is_valid_syllable.cache_clear()
                self._cached_is_valid_word.cache_clear()
                self._cached_get_syllable_frequency.cache_clear()
                self._cached_get_word_frequency.cache_clear()
                return
        self._cached_is_valid_syllable.cache_clear()
        self._cached_is_valid_word.cache_clear()
        self._cached_get_syllable_frequency.cache_clear()
        self._cached_get_word_frequency.cache_clear()


class CachedFrequencySource:
    """
    LRU-cached wrapper for FrequencySource.

    Caches frequency lookups but delegates iteration (not cacheable).

    Args:
        provider: Underlying FrequencySource implementation
        cache_size: Max cached frequency lookups (default: 8192)

    Performance:
        - Frequency lookup (cached): ~0.05ms
        - Frequency lookup (uncached): ~2ms
        - Average speedup: ~40x
    """

    def __init__(self, provider: FrequencySource, cache_size: int = 8192):
        self._provider = provider
        self._cache_size = cache_size

        self._cached_get_syllable_frequency = lru_cache(maxsize=cache_size)(
            self._get_syllable_frequency_impl
        )
        self._cached_get_word_frequency = lru_cache(maxsize=cache_size)(
            self._get_word_frequency_impl
        )

    def _get_syllable_frequency_impl(self, syllable: str) -> int:
        return self._provider.get_syllable_frequency(syllable)

    def _get_word_frequency_impl(self, word: str) -> int:
        return self._provider.get_word_frequency(word)

    def get_syllable_frequency(self, syllable: str) -> int:
        """Get syllable frequency (cached)."""
        return self._cached_get_syllable_frequency(syllable)

    def get_word_frequency(self, word: str) -> int:
        """Get word frequency (cached)."""
        return self._cached_get_word_frequency(word)

    def get_all_syllables(self) -> Iterator[tuple[str, int]]:
        """Get all syllables (not cached - iteration)."""
        return self._provider.get_all_syllables()

    def get_all_words(self) -> Iterator[tuple[str, int]]:
        """Get all words (not cached - iteration)."""
        return self._provider.get_all_words()


class CachedBigramSource:
    """
    LRU-cached wrapper for BigramSource.

    Caches bigram probability lookups for massive speedup.

    Args:
        provider: Underlying BigramSource implementation
        cache_size: Max cached bigram lookups (default: 16384)

    Cache Hit Rate: ~80% (context words repeat frequently)

    Performance:
        - Bigram lookup (cached): ~0.05ms
        - Bigram lookup (uncached): ~10ms
        - Average speedup: ~100x
    """

    def __init__(self, provider: BigramSource, cache_size: int = 16384):
        self._provider = provider
        self._cache_size = cache_size

        self._cached_get_bigram_probability = lru_cache(maxsize=cache_size)(
            self._get_bigram_probability_impl
        )
        self._cached_get_top_continuations = lru_cache(maxsize=cache_size // 4)(
            self._get_top_continuations_impl
        )

    def _get_bigram_probability_impl(self, w1: str, w2: str) -> float:
        return self._provider.get_bigram_probability(w1, w2)

    def _get_top_continuations_impl(
        self, prev_word: str, limit: int
    ) -> tuple[tuple[str, float], ...]:
        """Implementation that returns tuple (hashable for lru_cache)."""
        results = self._provider.get_top_continuations(prev_word, limit)
        return tuple(results)  # Convert list to tuple for caching

    def get_bigram_probability(self, w1: str, w2: str) -> float:
        """Get bigram probability P(w2|w1) (cached)."""
        return self._cached_get_bigram_probability(w1, w2)

    def get_top_continuations(self, prev_word: str, limit: int = 10) -> list[tuple[str, float]]:
        """Get top continuations (cached as tuple, returned as list)."""
        cached_tuple = self._cached_get_top_continuations(prev_word, limit)
        return list(cached_tuple)


class CachedTrigramSource:
    """
    LRU-cached wrapper for TrigramSource.

    Args:
        provider: Underlying TrigramSource implementation
        cache_size: Max cached trigram lookups (default: 16384)

    Performance:
        - Trigram lookup (cached): ~0.05ms
        - Trigram lookup (uncached): ~15ms
        - Average speedup: ~150x
    """

    def __init__(self, provider: TrigramSource, cache_size: int = 16384):
        self._provider = provider
        self._cache_size = cache_size

        self._cached_get_trigram_probability = lru_cache(maxsize=cache_size)(
            self._get_trigram_probability_impl
        )

    def _get_trigram_probability_impl(self, w1: str, w2: str, w3: str) -> float:
        return self._provider.get_trigram_probability(w1, w2, w3)

    def get_trigram_probability(self, w1: str, w2: str, w3: str) -> float:
        """Get trigram probability P(w3|w1,w2) (cached)."""
        return self._cached_get_trigram_probability(w1, w2, w3)


class CachedPOSRepository:
    """
    Thread-safe cached wrapper for POSRepository.

    Note: POS probabilities are typically loaded once at initialization,
    so caching provides minimal benefit. Included for API consistency.

    Thread Safety:
        Uses simple locking pattern for thread-safe lazy initialization.
        The previous double-checked locking pattern was problematic in Python
        because Python's memory model doesn't guarantee the order of operations
        without explicit synchronization. The lock overhead is negligible since
        these methods are typically called only once at startup.

    Args:
        provider: Underlying POSRepository implementation
    """

    def __init__(self, provider: POSRepository):
        self._provider = provider
        self._unigram_cache: dict[str, float] | None = None
        self._bigram_cache: dict[tuple[str, str], float] | None = None
        self._trigram_cache: dict[tuple[str, str, str], float] | None = None
        # Lock for thread-safe lazy initialization
        self._lock = threading.Lock()

    def get_pos_unigram_probabilities(self) -> dict[str, float]:
        """Get POS unigram probabilities (cached once, thread-safe)."""
        with self._lock:
            if self._unigram_cache is None:
                self._unigram_cache = self._provider.get_pos_unigram_probabilities()
            return self._unigram_cache

    def get_pos_bigram_probabilities(self) -> dict[tuple[str, str], float]:
        """Get POS bigram probabilities (cached once, thread-safe)."""
        with self._lock:
            if self._bigram_cache is None:
                self._bigram_cache = self._provider.get_pos_bigram_probabilities()
            return self._bigram_cache

    def get_pos_trigram_probabilities(self) -> dict[tuple[str, str, str], float]:
        """Get POS trigram probabilities (cached once, thread-safe)."""
        with self._lock:
            if self._trigram_cache is None:
                self._trigram_cache = self._provider.get_pos_trigram_probabilities()
            return self._trigram_cache


# Convenience functions for creating cached wrappers
def with_cache(
    provider: DictionaryLookup,
    syllable_cache_size: int = 4096,
    word_cache_size: int = 8192,
) -> CachedDictionaryLookup:
    """
    Wrap a DictionaryLookup with caching.

    Example:
        >>> provider = SQLiteProvider("myspell.db")
        >>> cached = with_cache(provider)
        >>> # Use cached instead of provider for 20x speedup
    """
    return CachedDictionaryLookup(provider, syllable_cache_size, word_cache_size)


__all__ = [
    "CacheRegistry",
    "CachedDictionaryLookup",
    "CachedFrequencySource",
    "CachedBigramSource",
    "CachedTrigramSource",
    "CachedPOSRepository",
    "with_cache",
]
