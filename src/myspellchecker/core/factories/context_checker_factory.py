"""
Context checker factory for DI container.

Dependency Graph:
    provider (required)
        └── symspell (required, depends on provider)
                └── context_checker (this service)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from myspellchecker.core.di.service_names import SERVICE_PROVIDER, SERVICE_SYMSPELL

if TYPE_CHECKING:
    from myspellchecker.algorithms import NgramContextChecker
    from myspellchecker.core.di.container import ServiceContainer

    # Type alias for context checker factory function
    ContextCheckerFactory = Callable[[ServiceContainer], NgramContextChecker | None]


def create_context_checker_factory() -> ContextCheckerFactory:
    """
    Create factory function for NgramContextChecker.

    The factory creates a context checker that uses N-gram probabilities
    and POS tagging to validate word sequences in Myanmar text. This
    enables Layer 3 (context-aware) spell checking.

    Dependencies (in resolution order):
        1. 'provider': DictionaryProvider - Base data source (no dependencies)
        2. 'symspell': SymSpell algorithm - Depends on provider

    Returns:
        Callable that accepts ServiceContainer and returns NgramContextChecker
        instance, or None if context checking is disabled.

    Example:
        >>> from myspellchecker.core.di.container import ServiceContainer
        >>> from myspellchecker.core.config import SpellCheckerConfig
        >>> config = SpellCheckerConfig(use_context_checker=True)
        >>> container = ServiceContainer(config)
        >>> container.register_factory('provider', create_provider_factory())
        >>> container.register_factory('symspell', create_symspell_factory())
        >>> container.register_factory('context_checker', create_context_checker_factory())
        >>> checker = container.get('context_checker')
        >>> errors = checker.check_context(["သူ", "သွား", "တယ်"])

    Note:
        Returns None when config.use_context_checker is False.
        Context checking adds latency but improves accuracy for
        detecting contextually inappropriate words.
    """

    def factory(container: "ServiceContainer") -> "NgramContextChecker" | None:
        from myspellchecker.core.factories.builders import build_ngram_context_checker

        config = container.get_config()

        if not config.use_context_checker:
            return None

        # Resolve in dependency order: provider → symspell
        provider = container.get(SERVICE_PROVIDER)
        symspell = container.get(SERVICE_SYMSPELL)

        return build_ngram_context_checker(
            provider=provider,
            config=config,
            symspell=symspell,
        )

    return factory
