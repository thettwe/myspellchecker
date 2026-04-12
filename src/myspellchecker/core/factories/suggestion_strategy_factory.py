"""
Suggestion strategy factory for DI container.

This factory creates CompositeSuggestionStrategy which aggregates suggestions
from multiple sources and applies unified ranking.

Dependency Graph:
    provider (required)
        └── symspell (required, depends on provider)
                └── context_checker (optional, depends on symspell)
                        └── suggestion_strategy (this service)

The dependencies MUST be resolved in order: provider → symspell → context_checker.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from myspellchecker.core.di.service_names import (
    SERVICE_CONTEXT_CHECKER,
    SERVICE_PROVIDER,
    SERVICE_SEGMENTER,
    SERVICE_SYMSPELL,
)
from myspellchecker.utils.logging_utils import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from myspellchecker.algorithms.suggestion_strategy import SuggestionStrategy
    from myspellchecker.core.di.container import ServiceContainer

    # Type alias for suggestion strategy factory function
    SuggestionStrategyFactory = Callable[[ServiceContainer], SuggestionStrategy | None]


def create_suggestion_strategy_factory() -> SuggestionStrategyFactory:
    """
    Create factory function for CompositeSuggestionStrategy.

    The factory creates a composite strategy that aggregates suggestions
    from multiple sources (SymSpell, morphology, compound) and applies
    unified ranking. If context checking is enabled, the composite is
    wrapped in ContextSuggestionStrategy for context-aware re-ranking.

    Dependencies (in resolution order):
        1. 'provider': DictionaryProvider - Base data source (no dependencies)
        2. 'symspell': SymSpell algorithm - Depends on provider
        3. 'context_checker': NgramContextChecker - Optional, depends on symspell

    Note: The unified pipeline always uses UnifiedRanker. The base ranker style
    is controlled by config.ranker.unified_base_ranker_type.

    Returns:
        Callable that accepts ServiceContainer and returns a
        SuggestionStrategy instance, or None if required dependencies unavailable.

    Example:
        >>> from myspellchecker.core.di.container import ServiceContainer
        >>> from myspellchecker.core.config import SpellCheckerConfig
        >>> config = SpellCheckerConfig()
        >>> container = ServiceContainer(config)
        >>> container.register_factory('suggestion_strategy', create_suggestion_strategy_factory())
        >>> strategy = container.get('suggestion_strategy')
        >>> if strategy:
        ...     result = strategy.suggest("typo", context)
    """

    def factory(container: "ServiceContainer") -> "SuggestionStrategy" | None:
        from myspellchecker.core.factories.builders import build_suggestion_strategy

        config = container.get_config()

        # =================================================================
        # Dependency Resolution (ORDER MATTERS)
        # Resolve in dependency order: provider → symspell → context_checker
        # =================================================================

        # 1. Get provider first (no dependencies, required by symspell)
        try:
            provider = container.get(SERVICE_PROVIDER)
        except ValueError as e:
            logger.debug(f"Provider service not registered: {e}")
            return None

        # 2. Get SymSpell (depends on provider, required for suggestions)
        try:
            symspell = container.get(SERVICE_SYMSPELL)
            if symspell is None:
                logger.debug("SymSpell not available, cannot create suggestion strategy")
                return None
        except ValueError as e:
            logger.debug(f"SymSpell service not registered: {e}")
            return None

        # 3. Get optional context checker (depends on symspell)
        context_checker = None
        if config.use_context_checker:
            try:
                context_checker = container.get(SERVICE_CONTEXT_CHECKER)
            except ValueError as e:
                logger.debug(f"Context checker not available (optional): {e}")
                # Continue without context checker - it's optional

        # 4. Create morphological synthesis engines if enabled
        reduplication_engine = None
        compound_resolver = None

        if config.validation.use_reduplication_validation:
            from myspellchecker.text.reduplication import ReduplicationEngine

            segmenter = container.get(SERVICE_SEGMENTER)
            reduplication_engine = ReduplicationEngine(
                segmenter=segmenter,
                min_base_frequency=config.validation.reduplication_min_base_frequency,
                cache_size=config.validation.reduplication_cache_size,
                config=config.reduplication,
            )

        if config.validation.use_compound_synthesis:
            from myspellchecker.text.compound_resolver import CompoundResolver

            segmenter = container.get(SERVICE_SEGMENTER)
            compound_resolver = CompoundResolver(
                segmenter=segmenter,
                min_morpheme_frequency=config.validation.compound_min_morpheme_frequency,
                max_parts=config.validation.compound_max_parts,
                cache_size=config.validation.compound_cache_size,
                config=config.compound_resolver,
            )

        return build_suggestion_strategy(
            symspell=symspell,
            provider=provider,
            config=config,
            context_checker=context_checker,
            compound_resolver=compound_resolver,
            reduplication_engine=reduplication_engine,
        )

    return factory
