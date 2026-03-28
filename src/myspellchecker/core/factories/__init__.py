"""
Component Factories.

This module contains factory functions for creating spell checker components.
Each factory is a function that accepts a ServiceContainer and returns
a configured component instance.
"""

from myspellchecker.core.factories.context_checker_factory import (
    create_context_checker_factory,
)
from myspellchecker.core.factories.context_validator_factory import (
    create_context_validator_factory,
    create_context_validator_minimal_factory,
)
from myspellchecker.core.factories.phonetic_factory import create_phonetic_hasher_factory
from myspellchecker.core.factories.provider_factory import create_provider_factory
from myspellchecker.core.factories.segmenter_factory import create_segmenter_factory
from myspellchecker.core.factories.suggestion_strategy_factory import (
    create_suggestion_strategy_factory,
)
from myspellchecker.core.factories.symspell_factory import create_symspell_factory
from myspellchecker.core.factories.validators_factory import create_validators_factory

__all__ = [
    "create_provider_factory",
    "create_segmenter_factory",
    "create_phonetic_hasher_factory",
    "create_symspell_factory",
    "create_context_checker_factory",
    "create_context_validator_factory",
    "create_context_validator_minimal_factory",
    "create_suggestion_strategy_factory",
    "create_validators_factory",
]
