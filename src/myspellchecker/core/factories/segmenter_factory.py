"""Segmenter factory for DI container."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from myspellchecker.core.di.container import ServiceContainer
    from myspellchecker.segmenters import Segmenter

# Type alias for segmenter factory function
SegmenterFactory = Callable[["ServiceContainer"], "Segmenter"]


def create_segmenter_factory() -> SegmenterFactory:
    """
    Create factory function for text Segmenter.

    The factory creates a text segmenter that breaks Myanmar text into
    words and syllables. It respects configuration for custom segmenters
    or creates a DefaultSegmenter with the configured word engine.

    Returns:
        Callable that accepts ServiceContainer and returns a Segmenter
        instance for text tokenization.

    Example:
        >>> from myspellchecker.core.di.container import ServiceContainer
        >>> from myspellchecker.core.config import SpellCheckerConfig
        >>> config = SpellCheckerConfig()
        >>> container = ServiceContainer(config)
        >>> container.register_factory('segmenter', create_segmenter_factory())
        >>> segmenter = container.get('segmenter')
        >>> words = segmenter.segment("မြန်မာစာ")  # Segment into words

    Note:
        If config.segmenter is explicitly set, that segmenter is used.
        Otherwise, DefaultSegmenter is created with config.word_engine.
    """

    def factory(container: "ServiceContainer") -> "Segmenter":
        from myspellchecker.segmenters import DefaultSegmenter

        config = container.get_config()

        # If segmenter explicitly configured, use it
        if config.segmenter is not None:
            return config.segmenter

        # Create default segmenter
        return DefaultSegmenter(
            word_engine=config.word_engine,
            allow_extended_myanmar=config.validation.allow_extended_myanmar,
            seg_model=config.seg_model,
            seg_device=config.seg_device,
        )

    return factory
