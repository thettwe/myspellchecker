"""
Service Container for Dependency Injection.

This module implements a lightweight service locator pattern with:
- Lazy initialization of services
- Singleton and transient service support
- Factory-based component creation
- Type-safe service resolution

Factory Pattern Overview:
    The codebase has three factory patterns for different use cases:

    1. **ComponentFactory** (core/component_factory.py) - RECOMMENDED
       - Used internally by SpellChecker.__init__()
       - Creates all components in proper dependency order
       - For most users, use SpellChecker directly

    2. **ServiceContainer** (this module)
       - Lightweight service locator for standalone service access
       - For advanced users who need DI container benefits:
         >>> container = create_default_container(config)
         >>> symspell = container.get("symspell")

    3. **AlgorithmFactory** (algorithms/factory.py)
       - Standalone algorithm creation with caching
       - For users who need algorithms without SpellChecker

See Also:
    - core/component_factory.py: ComponentFactory (recommended)
    - core/factories/: DI-compatible factory functions
    - algorithms/factory.py: Standalone AlgorithmFactory
"""

from __future__ import annotations

import threading
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    TypeVar,
    overload,
)

from myspellchecker.utils.logging_utils import get_logger

if TYPE_CHECKING:
    from myspellchecker.core.config import SpellCheckerConfig

# TypeVar for generic service resolution
T = TypeVar("T")

logger = get_logger(__name__)

# Type alias for factory functions
ServiceFactory = Callable[["ServiceContainer"], Any]


class ServiceContainer:
    """
    Lightweight DI container with lazy initialization.

    The container manages component lifecycle and dependencies through
    factory functions. Services can be registered as singletons (created once)
    or transient (created on each request).

    Example:
        >>> container = ServiceContainer(config)
        >>> container.register_factory("provider", create_provider_factory())
        >>> provider = container.get("provider")  # Created on first access
        >>> provider2 = container.get("provider")  # Returns same instance (singleton)
    """

    def __init__(self, config: SpellCheckerConfig) -> None:
        """
        Initialize service container.

        Args:
            config: Spell checker configuration
        """
        from myspellchecker.core.config import SpellCheckerConfig as ConfigType

        self._config: ConfigType = config
        self._services: dict[str, Any] = {}
        self._factories: dict[str, ServiceFactory] = {}
        self._singletons: set[str] = set()
        # Use RLock because factories may recursively call container.get(...) for dependencies.
        self._lock: threading.RLock = threading.RLock()

        logger.debug("ServiceContainer initialized")

    def register_factory(
        self,
        service_name: str,
        factory: ServiceFactory,
        *,
        singleton: bool = True,
    ) -> None:
        """
        Register factory for lazy component creation.

        Args:
            service_name: Unique identifier for the service
            factory: Factory function that accepts ServiceContainer and returns service instance
            singleton: If True, service is created once and cached; if False,
                created on each request

        Raises:
            ValueError: If service_name is already registered

        Example:
            >>> def create_provider(container: ServiceContainer):
            ...     return SQLiteProvider(container.get_config())
            >>> container.register_factory("provider", create_provider, singleton=True)
        """
        if service_name in self._factories:
            raise ValueError(f"Service '{service_name}' is already registered")

        self._factories[service_name] = factory
        if singleton:
            self._singletons.add(service_name)

        logger.debug("Registered factory for %r (singleton=%s)", service_name, singleton)

    # Type-safe overloads for common service types
    @overload
    def get(self, service_name: str, service_type: type[T]) -> T: ...

    @overload
    def get(self, service_name: str) -> Any: ...

    def get(self, service_name: str, service_type: type[T] | None = None) -> Any:
        """
        Retrieve service, constructing if needed.

        For singleton services, the instance is created on first access and cached.
        For transient services, a new instance is created on each call.
        This method is thread-safe for singleton creation.

        Args:
            service_name: Name of the service to retrieve

        Returns:
            Service instance. Common return types by service_name:

            - ``'provider'``: DictionaryProvider (typically SQLiteProvider)
            - ``'segmenter'``: Segmenter (typically DefaultSegmenter)
            - ``'phonetic_hasher'``: PhoneticHasher | None (None if disabled)
            - ``'ranker'``: SuggestionRanker (typically DefaultRanker)
            - ``'symspell'``: SymSpell algorithm instance
            - ``'context_checker'``: NgramContextChecker | None (None if disabled)
            - ``'syllable_validator'``: SyllableValidator
            - ``'word_validator'``: WordValidator
            - ``'context_validator'``: ContextValidator

        Raises:
            ValueError: If service is not registered
            RuntimeError: If factory function fails to create service
            TypeError: If factory returns incompatible type
            KeyError: If factory has missing dependencies
            OSError: If factory encounters I/O errors
            ImportError: If factory has missing optional dependencies

        Example:
            >>> provider = container.get("provider")
            >>> symspell = container.get("symspell")
        """
        # Fast path: return cached instance without lock (dict read is atomic
        # under CPython's GIL, but another thread could modify the cache between
        # this check and the lock acquisition below — the double-check inside
        # the lock handles that race).
        if service_name in self._services:
            instance = self._services[service_name]
            if service_type is not None and not isinstance(instance, service_type):
                raise TypeError(
                    f"Service '{service_name}' expected type {service_type.__name__}, "
                    f"got {type(instance).__name__}"
                )
            logger.debug("Retrieved cached service %r", service_name)
            return instance

        # Thread-safe singleton creation
        with self._lock:
            # Check if factory exists (inside lock to avoid TOCTOU with cache)
            if service_name not in self._factories and service_name not in self._services:
                raise ValueError(
                    f"Service '{service_name}' not registered. "
                    f"Available services: {sorted(self._factories.keys())}"
                )

            # Double-check after acquiring lock (another thread may have created it)
            if service_name in self._services:
                instance = self._services[service_name]
                if service_type is not None and not isinstance(instance, service_type):
                    raise TypeError(
                        f"Service '{service_name}' expected type {service_type.__name__}, "
                        f"got {type(instance).__name__}"
                    )
                logger.debug("Retrieved cached service %r (after lock)", service_name)
                return instance

            # Create new instance
            logger.debug("Creating service %r", service_name)
            try:
                instance = self._factories[service_name](self)
            except Exception:
                logger.exception("Failed to create service %r", service_name)
                raise

            if service_type is not None and not isinstance(instance, service_type):
                raise TypeError(
                    f"Service '{service_name}' expected type {service_type.__name__}, "
                    f"got {type(instance).__name__}"
                )

            # Cache if singleton
            if service_name in self._singletons:
                self._services[service_name] = instance
                logger.debug("Cached singleton service %r", service_name)

        return instance

    def get_config(self) -> "SpellCheckerConfig":
        """
        Get spell checker configuration.

        Returns:
            Spell checker configuration object

        Example:
            >>> config = container.get_config()
            >>> max_distance = config.max_edit_distance
        """
        return self._config

    def has_service(self, service_name: str) -> bool:
        """
        Check if service is registered.

        Args:
            service_name: Name of the service

        Returns:
            True if service is registered, False otherwise

        Example:
            >>> if container.has_service("provider"):
            ...     provider = container.get("provider")
        """
        return service_name in self._factories

    def list_services(self) -> list[str]:
        """
        Get list of all registered services.

        Returns:
            Sorted list of service names

        Example:
            >>> services = container.list_services()
            >>> print(services)
            ['provider', 'segmenter', 'symspell', 'viterbi_tagger']
        """
        return sorted(self._factories.keys())

    def clear_cache(self) -> None:
        """
        Clear all cached singleton instances.

        This forces services to be recreated on next access.
        Useful for testing or when configuration changes.

        Example:
            >>> container.clear_cache()
            >>> provider = container.get("provider")  # Will create new instance
        """
        with self._lock:
            count = len(self._services)
            self._services.clear()
            logger.debug(f"Cleared {count} cached services")

    def __repr__(self) -> str:
        """String representation showing registered services."""
        services = self.list_services()
        cached = len(self._services)
        return f"ServiceContainer(services={len(services)}, cached={cached}, config={self._config})"
