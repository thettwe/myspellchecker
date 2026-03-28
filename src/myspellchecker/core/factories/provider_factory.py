"""Provider factory for DI container."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from myspellchecker.core.di.container import ServiceContainer
    from myspellchecker.providers import DictionaryProvider

# Type alias for provider factory function
ProviderFactory = Callable[["ServiceContainer"], "DictionaryProvider"]


def create_provider_factory() -> ProviderFactory:
    """
    Create factory function for DictionaryProvider.

    The factory creates a DictionaryProvider for dictionary data access.
    It supports multiple backends (SQLite, Memory). Requires an explicit
    database path — no bundled database is included.

    Returns:
        Callable that accepts ServiceContainer and returns a
        DictionaryProvider instance (SQLiteProvider or MemoryProvider).

    Example:
        >>> from myspellchecker.core.di.container import ServiceContainer
        >>> from myspellchecker.core.config import SpellCheckerConfig
        >>> config = SpellCheckerConfig()
        >>> container = ServiceContainer(config)
        >>> container.register_factory('provider', create_provider_factory())
        >>> provider = container.get('provider')
        >>> is_valid = provider.is_valid_syllable("မြန်")

    Note:
        - If config.provider is set, that provider is used directly
        - If database file not found, falls back to MemoryProvider
        - Provider configuration (cache_size, pool settings) comes from
          config.provider_config
    """

    def factory(container: "ServiceContainer") -> "DictionaryProvider":
        from myspellchecker.core.exceptions import MissingDatabaseError
        from myspellchecker.providers import MemoryProvider, SQLiteProvider

        config = container.get_config()

        # If provider explicitly configured, use it
        if config.provider is not None:
            return config.provider

        # Try to create SQLiteProvider
        try:
            return SQLiteProvider(
                database_path=config.provider_config.database_path,
                cache_size=config.provider_config.cache_size,
                pool_min_size=config.provider_config.pool_min_size,
                pool_max_size=config.provider_config.pool_max_size,
                pool_timeout=config.provider_config.pool_timeout,
                pool_max_connection_age=config.provider_config.pool_max_connection_age,
                curated_min_frequency=config.provider_config.curated_min_frequency,
            )
        except MissingDatabaseError:
            # Check if fallback is allowed
            if config.fallback_to_empty_provider:
                return MemoryProvider()
            else:
                raise

    return factory
