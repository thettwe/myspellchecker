"""
Validators factory for DI container.

Dependency Graph for all validators:
    provider (required)
        ├── segmenter (required, no dependencies)
        └── symspell (required, depends on provider)
                └── context_checker (optional, depends on symspell)
                        └── suggestion_strategy (optional, depends on context_checker)

Resolution order: segmenter, provider → symspell → context_checker → suggestion_strategy
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from myspellchecker.core.di.service_names import (
    SERVICE_CONTEXT_CHECKER,
    SERVICE_PROVIDER,
    SERVICE_SEGMENTER,
    SERVICE_SUGGESTION_STRATEGY,
    SERVICE_SYMSPELL,
)

if TYPE_CHECKING:
    from myspellchecker.core.di.container import ServiceContainer
    from myspellchecker.core.validators import Validator

# Type alias for validator factory function
ValidatorFactory = Callable[["ServiceContainer"], "Validator"]


def create_validators_factory() -> dict[str, ValidatorFactory]:
    """
    Create factory functions for syllable and word validators.

    This function returns a dictionary of factory functions that create
    the first two layers of the validation pipeline:
    - SyllableValidator: Layer 1 - Fast syllable-level validation
    - WordValidator: Layer 2 - Word-level validation with SymSpell

    Note: ContextValidator (Layer 3) is registered separately via
    ``create_context_validator_factory()`` in ``context_validator_factory.py``.

    Returns:
        dict[str, Callable] mapping validator names to factory functions:
        - 'syllable': Creates SyllableValidator
        - 'word': Creates WordValidator

    Dependencies (in resolution order):
        1. 'segmenter': Text segmenter (no dependencies)
        2. 'provider': DictionaryProvider (no dependencies)
        3. 'symspell': SymSpell algorithm (depends on provider)
        4. 'context_checker': NgramContextChecker (depends on symspell)
        5. 'suggestion_strategy': SuggestionStrategy (depends on context_checker)

    Example:
        >>> from myspellchecker.core.di.container import ServiceContainer
        >>> from myspellchecker.core.config import SpellCheckerConfig
        >>> config = SpellCheckerConfig()
        >>> container = ServiceContainer(config)
        >>> # Register dependencies first...
        >>> validators = create_validators_factory()
        >>> container.register_factory('syllable_validator', validators['syllable'])
        >>> container.register_factory('word_validator', validators['word'])
        >>> validator = container.get('syllable_validator')
        >>> errors = validator.validate("မြန်မာစာ")
    """

    def create_syllable_validator(container: "ServiceContainer") -> "Validator":
        from myspellchecker.core.syllable_rules import SyllableRuleValidator
        from myspellchecker.core.validators import SyllableValidator

        config = container.get_config()

        # Resolve in dependency order: segmenter, provider → symspell
        segmenter = container.get(SERVICE_SEGMENTER)
        provider = container.get(SERVICE_PROVIDER)
        symspell = container.get(SERVICE_SYMSPELL)

        # Create syllable rule validator if enabled
        syllable_rule_validator = None
        if config.use_rule_based_validation:
            from pathlib import Path

            from myspellchecker.core.detection_rules import load_stacking_pairs

            raw_path = config.validation.stacking_pairs_path
            stacking_path = Path(raw_path) if isinstance(raw_path, str) else None
            stacking_pairs = load_stacking_pairs(stacking_path)

            syllable_rule_validator = SyllableRuleValidator(
                max_syllable_length=config.validation.max_syllable_length,
                corruption_threshold=config.validation.syllable_corruption_threshold,
                strict=config.validation.strict_validation,
                allow_extended_myanmar=config.validation.allow_extended_myanmar,
                stacking_pairs=stacking_pairs,
            )

        return SyllableValidator(
            config=config,
            segmenter=segmenter,
            repository=provider,  # Pass as SyllableRepository interface
            symspell=symspell,
            syllable_rule_validator=syllable_rule_validator,
        )

    def create_word_validator(container: "ServiceContainer") -> "Validator":
        from myspellchecker.core.validators import WordValidator
        from myspellchecker.segmenters import DefaultSegmenter

        config = container.get_config()

        # Resolve in dependency order:
        # segmenter, provider → symspell → context_checker → suggestion_strategy
        segmenter = container.get(SERVICE_SEGMENTER)
        provider = container.get(SERVICE_PROVIDER)

        # Inject word repository into segmenter for syllable-reassembly fallback
        if isinstance(segmenter, DefaultSegmenter) and hasattr(segmenter, "set_word_repository"):
            segmenter.set_word_repository(provider)
        symspell = container.get(SERVICE_SYMSPELL)
        context_checker = container.get(SERVICE_CONTEXT_CHECKER)
        suggestion_strategy = container.get(SERVICE_SUGGESTION_STRATEGY)

        # Create morphological synthesis engines if enabled
        reduplication_engine = None
        compound_resolver = None

        if config.validation.use_reduplication_validation:
            from myspellchecker.text.reduplication import ReduplicationEngine

            reduplication_engine = ReduplicationEngine(
                segmenter=segmenter,
                min_base_frequency=config.validation.reduplication_min_base_frequency,
                cache_size=config.validation.reduplication_cache_size,
                config=config.reduplication,
            )

        if config.validation.use_compound_synthesis:
            from myspellchecker.text.compound_resolver import CompoundResolver

            compound_resolver = CompoundResolver(
                segmenter=segmenter,
                min_morpheme_frequency=config.validation.compound_min_morpheme_frequency,
                max_parts=config.validation.compound_max_parts,
                cache_size=config.validation.compound_cache_size,
                config=config.compound_resolver,
            )

        return WordValidator(
            config=config,
            segmenter=segmenter,
            word_repository=provider,  # Pass as WordRepository interface
            syllable_repository=provider,  # Pass as SyllableRepository interface
            symspell=symspell,
            context_checker=context_checker,
            suggestion_strategy=suggestion_strategy,
            reduplication_engine=reduplication_engine,
            compound_resolver=compound_resolver,
        )

    return {
        "syllable": create_syllable_validator,
        "word": create_word_validator,
    }
