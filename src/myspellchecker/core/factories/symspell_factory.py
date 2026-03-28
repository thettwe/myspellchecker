"""
SymSpell factory for DI container.

Dependency Graph:
    provider (required, no dependencies)
    phonetic_hasher (optional, no dependencies)
        └── symspell (this service)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from myspellchecker.core.di.service_names import SERVICE_PHONETIC_HASHER, SERVICE_PROVIDER

if TYPE_CHECKING:
    from myspellchecker.algorithms import SymSpell
    from myspellchecker.core.di.container import ServiceContainer

# Type alias for SymSpell factory function
SymSpellFactory = Callable[["ServiceContainer"], "SymSpell"]


def create_symspell_factory() -> SymSpellFactory:
    """
    Create factory function for SymSpell algorithm.

    The factory creates a SymSpell instance configured for Myanmar spell
    checking with O(1) lookup complexity. It resolves dependencies from
    the container (provider, phonetic_hasher) and applies configuration
    from SpellCheckerConfig.symspell.

    Note: SymSpell uses a base ranker (never UnifiedRanker) to avoid
    double-normalizing scores. UnifiedRanker is only used at the
    composite pipeline level via CompositeSuggestionStrategy.

    Dependencies (in resolution order):
        1. 'provider': DictionaryProvider - Required, no dependencies
        2. 'phonetic_hasher': PhoneticHasher - Optional, no dependencies

    Returns:
        Callable that accepts ServiceContainer and returns a configured
        SymSpell instance with index built (unless skip_init is True).

    Example:
        >>> from myspellchecker.core.di.container import ServiceContainer
        >>> from myspellchecker.core.config import SpellCheckerConfig
        >>> config = SpellCheckerConfig(max_edit_distance=2)
        >>> container = ServiceContainer(config)
        >>> # Register required dependencies first
        >>> container.register_factory('provider', create_provider_factory())
        >>> container.register_factory('symspell', create_symspell_factory())
        >>> symspell = container.get('symspell')
        >>> suggestions = symspell.lookup("မြနမာ", max_edit_distance=2)

    Note:
        Index is built automatically unless config.symspell.skip_init is True.
        Building index can take a few seconds for large dictionaries.
    """

    def factory(container: "ServiceContainer") -> "SymSpell":
        from myspellchecker.core.factories.builders import build_symspell

        config = container.get_config()

        # Resolve dependencies (no circular dependencies possible here)
        provider = container.get(SERVICE_PROVIDER)
        phonetic_hasher = container.get(SERVICE_PHONETIC_HASHER)

        return build_symspell(
            provider=provider,
            config=config,
            phonetic_hasher=phonetic_hasher,
            build_index=not config.symspell.skip_init,
        )

    return factory
