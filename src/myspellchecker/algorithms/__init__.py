"""
Spell checking algorithms.

This module contains implementations of spelling correction algorithms,
including SymSpell, N-gram context checking, suggestion ranking strategies,
and pluggable POS tagging systems.

Note:
    TransformerPOSTagger is an optional component that requires the 'transformers'
    package. Install with: pip install myspellchecker[transformers]
    If not installed, TransformerPOSTagger will be None and transformer-based
    POS tagging will not be available.
"""

from myspellchecker.algorithms.joint_segment_tagger import JointSegmentTagger
from myspellchecker.algorithms.ngram_context_checker import (
    ContextSuggestion,
    NgramContextChecker,
)

# POS Tagging components
from myspellchecker.algorithms.pos_tagger_base import (
    POSPrediction,
    POSTaggerBase,
    TaggerType,
)
from myspellchecker.algorithms.pos_tagger_factory import POSTaggerFactory
from myspellchecker.algorithms.pos_tagger_rule import RuleBasedPOSTagger
from myspellchecker.algorithms.pos_tagger_viterbi import ViterbiPOSTaggerAdapter

# Transformer tagger is optional (requires transformers package)
try:
    from myspellchecker.algorithms.pos_tagger_transformer import (
        TransformerPOSTagger,
    )

    _HAS_TRANSFORMER_TAGGER = True
except ImportError:
    _HAS_TRANSFORMER_TAGGER = False
    TransformerPOSTagger = None  # type: ignore

from myspellchecker.algorithms.distance.edit_distance import (
    damerau_levenshtein_distance,
    levenshtein_distance,
)
from myspellchecker.algorithms.ranker import (
    DefaultRanker,
    EditDistanceOnlyRanker,
    FrequencyFirstRanker,
    PhoneticFirstRanker,
    SuggestionData,
    SuggestionRanker,
    UnifiedRanker,
)

# Suggestion Strategy
from myspellchecker.algorithms.suggestion_strategy import (
    BaseSuggestionStrategy,
    CompositeSuggestionStrategy,
    SuggestionContext,
    SuggestionResult,
    SuggestionStrategy,
)
from myspellchecker.algorithms.symspell import Suggestion, SymSpell

__all__ = [
    # SymSpell algorithm
    "SymSpell",
    "Suggestion",
    # Ranking strategies
    "SuggestionRanker",
    "SuggestionData",
    "DefaultRanker",
    "FrequencyFirstRanker",
    "EditDistanceOnlyRanker",
    "PhoneticFirstRanker",
    "UnifiedRanker",
    # Suggestion Strategy
    "SuggestionStrategy",
    "SuggestionContext",
    "SuggestionResult",
    "BaseSuggestionStrategy",
    "CompositeSuggestionStrategy",
    # Edit distance utilities
    "damerau_levenshtein_distance",
    "levenshtein_distance",
    # Context checking
    "NgramContextChecker",
    "ContextSuggestion",
    # Joint segmentation-tagging
    "JointSegmentTagger",
    # POS Tagging
    "POSTaggerBase",
    "POSPrediction",
    "TaggerType",
    "POSTaggerFactory",
    "RuleBasedPOSTagger",
    "TransformerPOSTagger",  # None if transformers not installed
    "ViterbiPOSTaggerAdapter",
]
