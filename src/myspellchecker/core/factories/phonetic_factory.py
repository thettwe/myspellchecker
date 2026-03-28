"""Phonetic hasher factory for DI container."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from myspellchecker.core.di.container import ServiceContainer
    from myspellchecker.text.phonetic import PhoneticHasher

    # Type alias for phonetic hasher factory function
    PhoneticHasherFactory = Callable[[ServiceContainer], PhoneticHasher | None]


def create_phonetic_hasher_factory() -> PhoneticHasherFactory:
    """
    Create factory function for PhoneticHasher.

    The factory integrates with the DI container system to create
    a phonetic hasher with proper configuration. PhoneticHasher provides
    phonetic encoding for Myanmar text to help find phonetically similar
    suggestions during spell checking.

    Returns:
        Callable that accepts ServiceContainer and returns PhoneticHasher
        instance, or None if phonetic hashing is disabled in config.

    Example:
        >>> from myspellchecker.core.di.container import ServiceContainer
        >>> from myspellchecker.core.config import SpellCheckerConfig
        >>> config = SpellCheckerConfig(use_phonetic=True)
        >>> container = ServiceContainer(config)
        >>> container.register_factory('phonetic', create_phonetic_hasher_factory())
        >>> hasher = container.get('phonetic')  # Returns PhoneticHasher
        >>> hasher.hash("မြန်မာ")  # Get phonetic hash

    Note:
        Returns None when config.use_phonetic is False, which is useful
        for environments where phonetic suggestions are not needed.
    """

    def factory(container: "ServiceContainer") -> "PhoneticHasher" | None:
        from myspellchecker.text.phonetic import PhoneticHasher

        config = container.get_config()

        if not config.use_phonetic:
            return None

        return PhoneticHasher(config=config.phonetic)

    return factory
