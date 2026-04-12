"""
Generic singleton utilities for module-level instance management.

This module provides thread-safe singleton patterns that can be used
across the codebase to reduce boilerplate code.

Usage:
    # Simple singleton (not thread-safe, for single-threaded use)
    from myspellchecker.utils.singleton import Singleton

    _checker = Singleton[AspectChecker]()

    def get_aspect_checker() -> AspectChecker:
        return _checker.get(AspectChecker)

    # Thread-safe singleton
    from myspellchecker.utils.singleton import ThreadSafeSingleton

    _service = ThreadSafeSingleton[NormalizationService]()

    def get_normalization_service() -> NormalizationService:
        return _service.get(NormalizationService)

    # With custom factory
    def get_custom_checker() -> CustomChecker:
        return _checker.get(CustomChecker, factory=lambda: CustomChecker(config))
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")

__all__ = [
    "Singleton",
    "ThreadSafeSingleton",
]


class Singleton(Generic[T]):
    """
    Simple singleton container for module-level instances.

    Not thread-safe. Use ThreadSafeSingleton for concurrent access.

    Example:
        >>> _instance = Singleton[MyClass]()
        >>> obj = _instance.get(MyClass)
        >>> obj2 = _instance.get(MyClass)
        >>> assert obj is obj2  # Same instance
    """

    def __init__(self) -> None:
        self._instances: dict[type, object] = {}

    def get(
        self,
        cls: type[T],
        factory: Callable[[], T] | None = None,
    ) -> T:
        """
        Get or create singleton instance for the given class.

        Args:
            cls: The class type to get/create instance for.
            factory: Optional factory function to create the instance.
                    If not provided, cls() is called with no arguments.

        Returns:
            The singleton instance.
        """
        if cls not in self._instances:
            self._instances[cls] = factory() if factory else cls()
        return self._instances[cls]  # type: ignore[return-value]

    def clear(self, cls: type[T] | None = None) -> None:
        """
        Clear singleton instance(s).

        Args:
            cls: Specific class to clear. If None, clears all instances.
        """
        if cls is None:
            self._instances.clear()
        else:
            self._instances.pop(cls, None)


class ThreadSafeSingleton(Generic[T]):
    """
    Thread-safe singleton container using double-checked locking.

    Use this when instances may be accessed from multiple threads.

    Example:
        >>> _instance = ThreadSafeSingleton[MyService]()
        >>> service = _instance.get(MyService)
        >>> # Safe to call from multiple threads
    """

    def __init__(self) -> None:
        self._instances: dict[type, object] = {}
        self._lock = threading.Lock()

    def get(
        self,
        cls: type[T],
        factory: Callable[[], T] | None = None,
    ) -> T:
        """
        Get or create singleton instance for the given class (thread-safe).

        Uses double-checked locking pattern for efficient thread-safe access.

        Args:
            cls: The class type to get/create instance for.
            factory: Optional factory function to create the instance.
                    If not provided, cls() is called with no arguments.

        Returns:
            The singleton instance.
        """
        # Fast path: dict read is atomic under CPython's GIL; the slow path
        # below re-checks under the lock to guard against races.
        if cls in self._instances:
            return self._instances[cls]  # type: ignore[return-value]

        # Slow path: acquire lock and double-check
        with self._lock:
            # Double-check after acquiring lock
            if cls not in self._instances:
                self._instances[cls] = factory() if factory else cls()

        return self._instances[cls]  # type: ignore[return-value]

    def clear(self, cls: type[T] | None = None) -> None:
        """
        Clear singleton instance(s) (thread-safe).

        Args:
            cls: Specific class to clear. If None, clears all instances.
        """
        with self._lock:
            if cls is None:
                self._instances.clear()
            else:
                self._instances.pop(cls, None)
