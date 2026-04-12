"""
Factory for creating ContextValidator with all validation strategies.

This factory creates and wires all validation strategies, demonstrating
the power of the Strategy pattern and Dependency Injection.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from myspellchecker.core.context_validator import ContextValidator
from myspellchecker.core.di.service_names import (
    SERVICE_CONTEXT_CHECKER,
    SERVICE_HOMOPHONE_CHECKER,
    SERVICE_NAME_HEURISTIC,
    SERVICE_PROVIDER,
    SERVICE_SEGMENTER,
    SERVICE_SEMANTIC_CHECKER,
    SERVICE_SYMSPELL,
    SERVICE_SYNTACTIC_RULE_CHECKER,
    SERVICE_TONE_DISAMBIGUATOR,
    SERVICE_VITERBI_TAGGER,
)
from myspellchecker.core.validation_strategies import (
    QuestionStructureValidationStrategy,
    SyntacticValidationStrategy,
    ToneValidationStrategy,
    ValidationStrategy,
)
from myspellchecker.utils.logging_utils import get_logger

if TYPE_CHECKING:
    from myspellchecker.core.di.container import ServiceContainer

logger = get_logger(__name__)

# Type alias for context validator factory function
ContextValidatorFactory = Callable[["ServiceContainer"], ContextValidator]


def create_context_validator_factory() -> ContextValidatorFactory:
    """
    Create factory function for ContextValidator with all validation strategies.

    Returns:
        Callable that accepts ServiceContainer and returns a ContextValidator instance.

    Example:
        >>> container.register_factory('context_validator', create_context_validator_factory())
    """
    return create_context_validator


def create_context_validator_minimal_factory() -> ContextValidatorFactory:
    """
    Create factory function for minimal ContextValidator (fast mode).

    Returns:
        Callable that accepts ServiceContainer and returns a minimal ContextValidator.

    Example:
        >>> factory = create_context_validator_minimal_factory()
        >>> container.register_factory('context_validator_minimal', factory)
    """
    return create_context_validator_minimal


def create_context_validator(container: "ServiceContainer") -> ContextValidator:
    """
    Create ContextValidator with all validation strategies.

    This factory demonstrates the Strategy pattern benefits:
    - Each strategy is created independently
    - Strategies are wired based on configuration
    - Easy to add/remove strategies
    - Clear dependency management

    Strategies created (in priority order):
    1. ToneValidationStrategy (10) - Tone mark disambiguation
    2. SyntacticValidationStrategy (20) - Grammar rules
    3. POSSequenceValidationStrategy (30) - POS patterns
    4. QuestionStructureValidationStrategy (40) - Question particles
    5. HomophoneValidationStrategy (45) - Homophone detection
    6. NgramContextValidationStrategy (50) - Statistical context
    7. SemanticValidationStrategy (70) - AI-powered semantic checking

    Args:
        container: Service container for dependency resolution.

    Returns:
        ContextValidator instance with all enabled strategies.
    """
    from myspellchecker.core.factories.builders import build_context_validation_strategies

    # Get core dependencies
    config = container.get_config()
    segmenter = container.get(SERVICE_SEGMENTER)
    provider = container.get(SERVICE_PROVIDER)

    # Get optional dependencies for strategies (using public API)
    tone_disambiguator = (
        container.get(SERVICE_TONE_DISAMBIGUATOR)
        if container.has_service(SERVICE_TONE_DISAMBIGUATOR)
        else None
    )
    syntactic_rule_checker = (
        container.get(SERVICE_SYNTACTIC_RULE_CHECKER)
        if container.has_service(SERVICE_SYNTACTIC_RULE_CHECKER)
        else None
    )
    viterbi_tagger = (
        container.get(SERVICE_VITERBI_TAGGER)
        if container.has_service(SERVICE_VITERBI_TAGGER)
        else None
    )
    context_checker = (
        container.get(SERVICE_CONTEXT_CHECKER)
        if container.has_service(SERVICE_CONTEXT_CHECKER)
        else None
    )
    homophone_checker = (
        container.get(SERVICE_HOMOPHONE_CHECKER)
        if container.has_service(SERVICE_HOMOPHONE_CHECKER)
        else None
    )
    semantic_checker = (
        container.get(SERVICE_SEMANTIC_CHECKER)
        if container.has_service(SERVICE_SEMANTIC_CHECKER)
        else None
    )
    name_heuristic = (
        container.get(SERVICE_NAME_HEURISTIC)
        if container.has_service(SERVICE_NAME_HEURISTIC)
        else None
    )
    symspell = (
        container.get(SERVICE_SYMSPELL)
        if container.has_service(SERVICE_SYMSPELL)
        else None
    )

    # Build strategies using shared builder
    strategies = build_context_validation_strategies(
        config=config,
        provider=provider,
        tone_disambiguator=tone_disambiguator,
        syntactic_rule_checker=syntactic_rule_checker,
        viterbi_tagger=viterbi_tagger,
        context_checker=context_checker,
        homophone_checker=homophone_checker,
        semantic_checker=semantic_checker,
        symspell=symspell,
    )

    # Create validator with all strategies + shared POS tagger for pre-computation
    validator = ContextValidator(
        config=config,
        segmenter=segmenter,
        strategies=strategies,
        name_heuristic=name_heuristic,
        viterbi_tagger=viterbi_tagger,
    )

    return validator


def create_context_validator_minimal(container: "ServiceContainer") -> ContextValidator:
    """
    Create ContextValidator with minimal strategies (fast mode).

    This factory creates a lightweight validator with only essential strategies:
    - ToneValidationStrategy
    - SyntacticValidationStrategy
    - QuestionStructureValidationStrategy

    Use this for:
    - Quick validation where latency matters
    - Resource-constrained environments
    - Testing/development

    Note:
        This minimal validator doesn't include n-gram strategies, so no
        NgramRepository is needed. This follows Interface Segregation Principle.

    Args:
        container: Service container for dependency resolution.

    Returns:
        ContextValidator instance with minimal strategies.
    """
    config = container.get_config()
    segmenter = container.get(SERVICE_SEGMENTER)
    provider = container.get(SERVICE_PROVIDER) if container.has_service(SERVICE_PROVIDER) else None

    strategies: list[ValidationStrategy] = []
    validation_config = config.validation

    # Only include lightweight, rule-based strategies (using public API)
    tone_disambiguator = (
        container.get(SERVICE_TONE_DISAMBIGUATOR)
        if container.has_service(SERVICE_TONE_DISAMBIGUATOR)
        else None
    )
    if tone_disambiguator:
        strategies.append(
            ToneValidationStrategy(
                tone_disambiguator=tone_disambiguator,
                confidence_threshold=validation_config.tone_validation_confidence,
                provider=provider,
            )
        )

    syntactic_rule_checker = (
        container.get(SERVICE_SYNTACTIC_RULE_CHECKER)
        if container.has_service(SERVICE_SYNTACTIC_RULE_CHECKER)
        else None
    )
    if syntactic_rule_checker:
        strategies.append(
            SyntacticValidationStrategy(
                syntactic_rule_checker=syntactic_rule_checker,
                confidence=validation_config.syntactic_validation_confidence,
            )
        )

    strategies.append(
        QuestionStructureValidationStrategy(
            confidence=validation_config.question_structure_confidence,
        )
    )

    logger.info(f"Created minimal ContextValidator with {len(strategies)} strategies (fast mode)")

    return ContextValidator(
        config=config,
        segmenter=segmenter,
        strategies=strategies,
    )
