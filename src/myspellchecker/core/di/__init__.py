"""
Dependency Injection Infrastructure.

This module provides a lightweight dependency injection container
for managing component creation and lifecycle in the spell checker.

Usage:
    >>> from myspellchecker.core.di import ServiceContainer, create_default_container
    >>> container = create_default_container(config)
    >>> symspell = container.get("symspell")

See Also:
    - core/component_factory.py: Default factory (recommended for most users)
    - core/di/registry.py: Service registration utilities
"""

from myspellchecker.core.di.container import ServiceContainer
from myspellchecker.core.di.registry import create_default_container, register_core_services

__all__ = [
    "ServiceContainer",
    "create_default_container",
    "register_core_services",
]
