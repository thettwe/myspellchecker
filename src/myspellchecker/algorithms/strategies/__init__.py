"""Suggestion strategy implementations.

Re-exports all concrete strategy classes for backwards compatibility.
Import from here or from myspellchecker.algorithms.suggestion_strategy.
"""

from myspellchecker.algorithms.strategies.compound_strategy import CompoundSuggestionStrategy
from myspellchecker.algorithms.strategies.context_strategy import ContextSuggestionStrategy
from myspellchecker.algorithms.strategies.morphology_strategy import MorphologySuggestionStrategy
from myspellchecker.algorithms.strategies.symspell_strategy import SymSpellSuggestionStrategy

__all__ = [
    "CompoundSuggestionStrategy",
    "ContextSuggestionStrategy",
    "MorphologySuggestionStrategy",
    "SymSpellSuggestionStrategy",
]
