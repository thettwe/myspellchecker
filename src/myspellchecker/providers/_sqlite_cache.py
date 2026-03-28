"""
Cache management helpers for SQLiteProvider.

Extracts cache initialization, statistics, and clearing logic so that
``sqlite.py`` stays focused on core lookup operations.

Uses a mixin class that is composed into ``SQLiteProvider``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from myspellchecker.utils.cache import LRUCache

if TYPE_CHECKING:
    from myspellchecker.providers.connection_pool import ConnectionPool
    from myspellchecker.utils.cache import CacheManager


def create_caches(
    cache_size: int,
    cache_manager: CacheManager | None,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Create all LRU caches used by SQLiteProvider.

    Returns a dict keyed by cache attribute name (without leading underscore)
    so the caller can assign them to ``self``.

    Args:
        cache_size: Maximum entries for word/syllable caches.
        cache_manager: Optional ``CacheManager`` for centralized cache management.
            When provided, caches are created through the manager.
        logger: Logger instance for debug messages.

    Returns:
        Dictionary mapping cache name to cache instance::

            {
                "word_id_cache": ...,
                "syllable_freq_cache": ...,
                "word_freq_cache": ...,
                "valid_word_cache": ...,
                "valid_syllable_cache": ...,
                "bigram_prob_cache": ...,
                "trigram_prob_cache": ...,
                "fourgram_prob_cache": ...,
                "fivegram_prob_cache": ...,
            }
    """
    # N-gram cache size: smaller than word caches since n-gram keys are larger
    ngram_cache_size = max(cache_size // 4, 4096)
    higher_order_cache_size = max(cache_size // 8, 2048)

    if cache_manager is not None:
        caches = {
            "word_id_cache": cache_manager.get_cache("word_id", maxsize=cache_size),
            "syllable_freq_cache": cache_manager.get_cache("syllable_freq", maxsize=cache_size),
            "word_freq_cache": cache_manager.get_cache("word_freq", maxsize=cache_size),
            "valid_word_cache": cache_manager.get_cache("valid_word", maxsize=cache_size),
            "valid_syllable_cache": cache_manager.get_cache("valid_syllable", maxsize=cache_size),
            "bigram_prob_cache": cache_manager.get_cache("bigram_prob", maxsize=ngram_cache_size),
            "trigram_prob_cache": cache_manager.get_cache("trigram_prob", maxsize=ngram_cache_size),
            "fourgram_prob_cache": cache_manager.get_cache(
                "fourgram_prob", maxsize=higher_order_cache_size
            ),
            "fivegram_prob_cache": cache_manager.get_cache(
                "fivegram_prob", maxsize=higher_order_cache_size
            ),
        }
        logger.debug("Using CacheManager for cache management")
    else:
        caches = {
            "word_id_cache": LRUCache(maxsize=cache_size),
            "syllable_freq_cache": LRUCache(maxsize=cache_size),
            "word_freq_cache": LRUCache(maxsize=cache_size),
            "valid_word_cache": LRUCache(maxsize=cache_size),
            "valid_syllable_cache": LRUCache(maxsize=cache_size),
            "bigram_prob_cache": LRUCache(maxsize=ngram_cache_size),
            "trigram_prob_cache": LRUCache(maxsize=ngram_cache_size),
            "fourgram_prob_cache": LRUCache(maxsize=higher_order_cache_size),
            "fivegram_prob_cache": LRUCache(maxsize=higher_order_cache_size),
        }

    return caches


class CacheMixin:
    """Mixin providing cache management methods for SQLiteProvider.

    Expects the host class to provide the cache attributes created by
    :func:`create_caches` (prefixed with ``_``), plus ``_pool``.
    """

    # Declared for type-checking only; set by SQLiteProvider.__init__
    _word_id_cache: Any
    _syllable_freq_cache: Any
    _word_freq_cache: Any
    _valid_word_cache: Any
    _valid_syllable_cache: Any
    _bigram_prob_cache: Any
    _trigram_prob_cache: Any
    _fourgram_prob_cache: Any
    _fivegram_prob_cache: Any
    _pool: ConnectionPool | None

    def get_cache_stats(self) -> dict[str, dict[str, int]]:
        """
        Get statistics for all LRU caches.

        This is a utility method for debugging and performance monitoring.
        Returns information about each cache including current size and max size.

        Returns:
            Dictionary with cache names as keys and statistics as values:
                - word_id_cache: Cache for word ID lookups
                - syllable_freq_cache: Cache for syllable frequency lookups
                - word_freq_cache: Cache for word frequency lookups

        Example:
            >>> provider = SQLiteProvider()  # Uses default cache_size=8192
            >>> # ... perform some lookups ...
            >>> stats = provider.get_cache_stats()
            >>> cache = stats['word_freq_cache']
            >>> print(f"Word freq cache: {cache['size']}/{cache['maxsize']}")

        Notes:
            - Default cache_size (8192) matches AlgorithmCacheConfig.frequency_cache_size
            - Useful for tuning cache_size parameter
            - Thread-safe
        """
        return {
            "word_id_cache": self._word_id_cache.stats(),
            "syllable_freq_cache": self._syllable_freq_cache.stats(),
            "word_freq_cache": self._word_freq_cache.stats(),
            "valid_word_cache": self._valid_word_cache.stats(),
            "valid_syllable_cache": self._valid_syllable_cache.stats(),
            "bigram_prob_cache": self._bigram_prob_cache.stats(),
            "trigram_prob_cache": self._trigram_prob_cache.stats(),
            "fourgram_prob_cache": self._fourgram_prob_cache.stats(),
            "fivegram_prob_cache": self._fivegram_prob_cache.stats(),
        }

    def clear_caches(self) -> None:
        """
        Clear all LRU caches.

        This method is useful for testing or when you want to force
        fresh database queries.

        Example:
            >>> provider = SQLiteProvider()
            >>> provider.clear_caches()  # Reset all caches

        Notes:
            - Thread-safe
            - After clearing, next lookups will query the database
        """
        self._word_id_cache.clear()
        self._syllable_freq_cache.clear()
        self._word_freq_cache.clear()
        self._valid_word_cache.clear()
        self._valid_syllable_cache.clear()
        self._bigram_prob_cache.clear()
        self._trigram_prob_cache.clear()
        self._fourgram_prob_cache.clear()
        self._fivegram_prob_cache.clear()

    def get_pool_stats(self) -> dict[str, Any] | None:
        """
        Get connection pool statistics.

        Returns pool metrics from the connection pool.

        Returns:
            Dictionary with pool statistics:
                - pool_size: Current number of connections in pool
                - active_connections: Total connections created
                - available_connections: Connections ready for checkout
                - total_checkouts: Total number of checkout operations
                - average_wait_time_ms: Average wait time for checkout
                - peak_active: Maximum concurrent active connections
                - min_size: Minimum pool size (configuration)
                - max_size: Maximum pool size (configuration)

        Example:
            >>> provider = SQLiteProvider()
            >>> # ... perform some queries ...
            >>> stats = provider.get_pool_stats()
            >>> if stats:
            ...     print(f"Pool: {stats['available_connections']}/{stats['pool_size']} available")
            ...     print(f"Avg wait: {stats['average_wait_time_ms']:.2f}ms")

        Notes:
            - Useful for performance monitoring and tuning pool size
        """
        if self._pool is not None:
            return self._pool.get_stats()
        return None
