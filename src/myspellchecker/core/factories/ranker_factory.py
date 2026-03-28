"""Suggestion ranker factory for creating base rankers."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from myspellchecker.algorithms.ranker import SuggestionRanker
    from myspellchecker.core.config.algorithm_configs import RankerConfig


def create_base_ranker(ranker_config: RankerConfig) -> SuggestionRanker:
    """
    Create a base ranker (never UnifiedRanker) from a RankerConfig.

    This is used by SymSpell factories to avoid double-normalizing scores.
    UnifiedRanker should only be used at the composite pipeline level,
    not within individual components like SymSpell.

    The base ranker type is controlled by `unified_base_ranker_type`:
    - "default": Balances edit distance and frequency (DefaultRanker)
    - "frequency_first": Prioritizes high-frequency words (FrequencyFirstRanker)
    - "phonetic_first": Prioritizes phonetic similarity (PhoneticFirstRanker)
    - "edit_distance_only": Pure edit distance ranking (EditDistanceOnlyRanker)

    Args:
        ranker_config: RankerConfig with unified_base_ranker_type and parameters.

    Returns:
        SuggestionRanker instance (never UnifiedRanker).

    Example:
        >>> from myspellchecker.core.config import RankerConfig
        >>> config = RankerConfig(unified_base_ranker_type="frequency_first")
        >>> ranker = create_base_ranker(config)
    """
    from myspellchecker.algorithms.ranker import (
        DefaultRanker,
        EditDistanceOnlyRanker,
        FrequencyFirstRanker,
        PhoneticFirstRanker,
    )

    # Use unified_base_ranker_type for base ranker selection
    base_type = ranker_config.unified_base_ranker_type

    if base_type == "frequency_first":
        return FrequencyFirstRanker(ranker_config=ranker_config)
    elif base_type == "phonetic_first":
        return PhoneticFirstRanker(ranker_config=ranker_config)
    elif base_type == "edit_distance_only":
        return EditDistanceOnlyRanker()
    else:
        # Default ranker
        return DefaultRanker(ranker_config=ranker_config)
