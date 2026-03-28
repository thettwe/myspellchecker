"""
Default Service Registry.

This module registers all built-in services with the DI container.
Each service is registered with a factory function that knows how
to construct the service with its dependencies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from myspellchecker.core.di.service_names import (
    SERVICE_CONTEXT_CHECKER,
    SERVICE_CONTEXT_VALIDATOR,
    SERVICE_CONTEXT_VALIDATOR_MINIMAL,
    SERVICE_HOMOPHONE_CHECKER,
    SERVICE_NAME_HEURISTIC,
    SERVICE_PHONETIC_HASHER,
    SERVICE_PROVIDER,
    SERVICE_SEGMENTER,
    SERVICE_SEMANTIC_CHECKER,
    SERVICE_SUGGESTION_STRATEGY,
    SERVICE_SYLLABLE_VALIDATOR,
    SERVICE_SYMSPELL,
    SERVICE_SYNTACTIC_RULE_CHECKER,
    SERVICE_TONE_DISAMBIGUATOR,
    SERVICE_VITERBI_TAGGER,
    SERVICE_WORD_VALIDATOR,
)
from myspellchecker.utils.logging_utils import get_logger

if TYPE_CHECKING:
    from myspellchecker.core.config import SpellCheckerConfig
    from myspellchecker.core.di.container import ServiceContainer

logger = get_logger(__name__)


def register_core_services(container: "ServiceContainer") -> None:
    """
    Register all core services with the container.

    This function registers factories for:
    - Provider (dictionary/database access)
    - Segmenter (text segmentation)
    - Phonetic hasher
    - SymSpell algorithm
    - Context checker
    - Suggestion strategy (unified suggestion pipeline)
    - Validators (syllable, word, context)
    - POS taggers
    - Grammar checkers

    Args:
        container: Service container to register services with

    Example:
        >>> container = ServiceContainer(config)
        >>> register_core_services(container)
        >>> provider = container.get("provider")
    """
    from myspellchecker.core.factories import (
        create_context_checker_factory,
        create_phonetic_hasher_factory,
        create_provider_factory,
        create_segmenter_factory,
        create_suggestion_strategy_factory,
        create_symspell_factory,
        create_validators_factory,
    )
    from myspellchecker.core.factories.context_validator_factory import (
        create_context_validator_factory,
        create_context_validator_minimal_factory,
    )
    from myspellchecker.core.factories.optional_services_factory import (
        create_homophone_checker_factory,
        create_name_heuristic_factory,
        create_semantic_checker_factory,
        create_syntactic_rule_checker_factory,
        create_tone_disambiguator_factory,
        create_viterbi_tagger_factory,
    )

    # Register factories in dependency order
    container.register_factory(SERVICE_PROVIDER, create_provider_factory(), singleton=True)
    container.register_factory(SERVICE_SEGMENTER, create_segmenter_factory(), singleton=True)
    container.register_factory(
        SERVICE_PHONETIC_HASHER, create_phonetic_hasher_factory(), singleton=True
    )
    container.register_factory(SERVICE_SYMSPELL, create_symspell_factory(), singleton=True)
    container.register_factory(
        SERVICE_CONTEXT_CHECKER, create_context_checker_factory(), singleton=True
    )
    container.register_factory(
        SERVICE_SUGGESTION_STRATEGY, create_suggestion_strategy_factory(), singleton=True
    )

    # Register validators
    validators = create_validators_factory()
    container.register_factory(SERVICE_SYLLABLE_VALIDATOR, validators["syllable"], singleton=True)
    container.register_factory(SERVICE_WORD_VALIDATOR, validators["word"], singleton=True)

    # Register optional services (for advanced validation strategies)
    # These services may return None if dependencies are not available
    container.register_factory(
        SERVICE_TONE_DISAMBIGUATOR, create_tone_disambiguator_factory(), singleton=True
    )
    container.register_factory(
        SERVICE_SYNTACTIC_RULE_CHECKER,
        create_syntactic_rule_checker_factory(),
        singleton=True,
    )
    container.register_factory(
        SERVICE_VITERBI_TAGGER, create_viterbi_tagger_factory(), singleton=True
    )
    container.register_factory(
        SERVICE_HOMOPHONE_CHECKER, create_homophone_checker_factory(), singleton=True
    )
    container.register_factory(
        SERVICE_SEMANTIC_CHECKER, create_semantic_checker_factory(), singleton=True
    )
    container.register_factory(
        SERVICE_NAME_HEURISTIC, create_name_heuristic_factory(), singleton=True
    )

    # Register context validators
    # Note: These must be registered AFTER optional services since they depend on them
    container.register_factory(
        SERVICE_CONTEXT_VALIDATOR, create_context_validator_factory(), singleton=True
    )
    container.register_factory(
        SERVICE_CONTEXT_VALIDATOR_MINIMAL,
        create_context_validator_minimal_factory(),
        singleton=True,
    )

    logger.info(f"Registered {len(container.list_services())} core services")


def create_default_container(config: SpellCheckerConfig) -> ServiceContainer:
    """
    Create a fully configured service container with all core services.

    This is a convenience function that creates a container and registers
    all built-in services in one step.

    Args:
        config: SpellChecker configuration

    Returns:
        Configured service container

    Example:
        >>> from myspellchecker.core.config import SpellCheckerConfig
        >>> from myspellchecker.core.di import create_default_container
        >>>
        >>> config = SpellCheckerConfig()
        >>> container = create_default_container(config)
        >>>
        >>> # Access services directly from container
        >>> symspell = container.get("symspell")
        >>> provider = container.get("provider")
    """
    from myspellchecker.core.di.container import ServiceContainer

    container = ServiceContainer(config)
    register_core_services(container)
    return container
