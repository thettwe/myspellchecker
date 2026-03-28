"""
Suggestion ranking strategies for spell correction.

This module defines the SuggestionRanker interface and its implementations
for ranking spelling correction suggestions based on various factors like
edit distance, frequency, phonetic similarity, and syllable structure.

Design Pattern:
    - **Strategy Pattern**: SuggestionRanker ABC defines the interface
    - **Template Method**: Each ranker implements score() differently
    - **Composite Pattern**: UnifiedRanker combines multiple scoring strategies

Available Rankers:
    - DefaultRanker: Balanced edit distance + frequency (score: lower = better)
    - FrequencyFirstRanker: Prioritizes corpus frequency (score: lower = better)
    - EditDistanceOnlyRanker: Pure edit distance ranking (score: lower = better)
    - PhoneticFirstRanker: Prioritizes phonetic similarity (score: lower = better)
    - UnifiedRanker: Consolidates multi-source suggestions (score: lower = better)

Result Types:
    - score() returns: float (lower values indicate better suggestions)
    - rank_suggestions() returns: list[SuggestionData] sorted best-first

Parameter Validation:
    - All rankers accept optional RankerConfig for configuration
    - Weights are read from RankerConfig, defaults provided if None
    - UnifiedRanker validates source_weights at initialization

The ranking strategy can be customized by implementing the SuggestionRanker
abstract base class and passing it to SymSpell.

Example:
    >>> from myspellchecker.algorithms.ranker import DefaultRanker, FrequencyFirstRanker
    >>>
    >>> # Use default ranking (edit distance first, then frequency)
    >>> symspell = SymSpell(provider, ranker=DefaultRanker())
    >>>
    >>> # Use frequency-first ranking for common word suggestions
    >>> symspell = SymSpell(provider, ranker=FrequencyFirstRanker())
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

from myspellchecker.core.config import RankerConfig
from myspellchecker.core.constants import CONFIDENCE_FLOOR

__all__ = [
    "DefaultRanker",
    "EditDistanceOnlyRanker",
    "FrequencyFirstRanker",
    "PhoneticFirstRanker",
    "SuggestionData",
    "SuggestionRanker",
    "UnifiedRanker",
]

# Default Ranker configuration (module-level singleton)
_default_ranker_config = RankerConfig()


@dataclass
class SuggestionData:
    """
    Raw data for a spelling suggestion before scoring.

    Attributes:
        term: The suggested correction term.
        edit_distance: Damerau-Levenshtein distance from query.
        frequency: Corpus frequency of the suggestion.
        phonetic_score: Phonetic similarity score (0.0-1.0, higher is better).
        syllable_distance: Myanmar syllable-aware weighted distance (optional).
        weighted_distance: Myanmar-weighted edit distance using substitution costs (optional).
            When enabled, uses MYANMAR_SUBSTITUTION_COSTS to score phonetically similar
            character substitutions (e.g., aspirated pairs, medial confusions) with
            lower costs than unrelated character substitutions.
        is_nasal_variant: True if suggestion differs only in nasal endings (န်/မ်/င်/ံ).
        has_same_nasal_ending: True if suggestion has the same nasal consonant ending.
        source: Origin of the suggestion for unified ranking.
        confidence: Source-specific confidence score (0.0-1.0).
        strategy_score: Strategy-level score (lower is better). Used by strategies
            to pass their internal scoring to the ranker for optional blending.
        score_breakdown: Optional debug info with component scores (e.g., log_prob, penalty).
        pos_fit_score: POS bigram fit score (0.0-1.0, higher = better grammatical fit).
            Computed by ContextSuggestionStrategy using POS bigram probabilities.
        error_length: Character length of the original error span (for span-length bonus).
    """

    term: str
    edit_distance: int
    frequency: int
    phonetic_score: float = 0.0
    syllable_distance: float | None = None
    weighted_distance: float | None = None
    is_nasal_variant: bool = False
    has_same_nasal_ending: bool = False
    # Source types: symspell, particle_typo, medial_confusion, morphology, context, compound
    source: str = "symspell"
    confidence: float = 1.0
    strategy_score: float | None = None
    score_breakdown: dict[str, float] | None = None
    pos_fit_score: float | None = None
    error_length: int | None = None


class SuggestionRanker(ABC):
    """
    Abstract base class for suggestion ranking strategies.

    Implementations define how spelling suggestions are scored and ranked.
    Lower scores indicate better suggestions.

    The default scoring combines:
    - Edit distance (primary factor)
    - Corpus frequency (secondary factor)
    - Phonetic similarity (tertiary factor)
    - Syllable structure awareness (quaternary factor)

    Custom rankers can emphasize different factors or introduce new ones.
    """

    @abstractmethod
    def score(self, data: SuggestionData) -> float:
        """
        Calculate a score for a suggestion.

        Lower scores indicate better suggestions. The score is used to
        sort and rank suggestions before returning them to the user.

        Args:
            data: SuggestionData containing term, distances, and frequencies.

        Returns:
            Float score where lower values indicate better suggestions.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this ranking strategy."""
        raise NotImplementedError


class DefaultRanker(SuggestionRanker):
    """
    Default suggestion ranker balancing edit distance and frequency.

    Scoring formula:
        score = edit_distance - freq_bonus - phonetic_bonus - syllable_bonus
                - nasal_bonus - same_nasal_bonus - span_bonus

    Where:
        - freq_bonus: 0.0 to 0.8 based on normalized frequency
        - phonetic_bonus: phonetic_score * phonetic_bonus_weight
        - syllable_bonus: bonus for syllable-aware distance being lower
        - nasal_bonus: bonus for true nasal variant matches (န် ↔ ံ only)
        - same_nasal_bonus: bonus when suggestion has same nasal ending as input
        - span_bonus: length-scaled bonus/penalty for matching error span

    Attributes:
        ranker_config: RankerConfig with ranking parameters.
    """

    def __init__(
        self,
        ranker_config: RankerConfig | None = None,
    ):
        """
        Initialize the default ranker.

        Args:
            ranker_config: RankerConfig with ranking parameters (uses defaults if None).
        """
        self.ranker_config = ranker_config or _default_ranker_config
        self.frequency_denominator = max(self.ranker_config.frequency_denominator, 1.0)
        self.phonetic_bonus_weight = self.ranker_config.phonetic_bonus_weight
        self.nasal_bonus_weight = self.ranker_config.nasal_bonus_weight
        self.same_nasal_bonus_weight = self.ranker_config.same_nasal_bonus_weight
        self.pos_bonus_weight = self.ranker_config.pos_bonus_weight
        self.plausibility_floor = self.ranker_config.plausibility_floor
        self.plausibility_threshold = self.ranker_config.plausibility_threshold
        # Span-length bonus parameters
        self.freq_bonus_ceiling = self.ranker_config.freq_bonus_ceiling
        self.long_error_threshold = self.ranker_config.long_error_threshold
        self.span_cap = self.ranker_config.span_cap
        self.long_exact_base = self.ranker_config.long_exact_base
        self.long_exact_scale = self.ranker_config.long_exact_scale
        self.long_close_base = self.ranker_config.long_close_base
        self.long_close_scale = self.ranker_config.long_close_scale
        self.long_medium_bonus = self.ranker_config.long_medium_bonus
        self.long_far_penalty = self.ranker_config.long_far_penalty
        self.short_exact_bonus = self.ranker_config.short_exact_bonus
        self.short_close_bonus = self.ranker_config.short_close_bonus
        self.short_medium_bonus = self.ranker_config.short_medium_bonus

    @property
    def name(self) -> str:
        """Return ranker identifier string."""
        return "default"

    def score(self, data: SuggestionData) -> float:
        """
        Calculate score balancing edit distance, frequency, and POS fit.

        Scoring formula:
            score = edit_distance - freq_bonus - phonetic_bonus - syllable_bonus
                    - weighted_bonus - nasal_bonus - same_nasal_bonus - pos_bonus

        Args:
            data: SuggestionData with term information.

        Returns:
            Float score (lower is better).
        """
        # Compute plausibility multiplier from weighted_distance or syllable_distance.
        # Myanmar-specific error patterns (medial swap=0.3, aspiration=0.4) get
        # multiplicatively lower base scores, creating a gap that frequency
        # differences cannot overcome.
        # Only activate multiplicative plausibility for strong Myanmar-specific
        # patterns (ratio < 0.7): medial swap=0.3, aspiration=0.4, nasal=0.5.
        # Borderline ratios (0.7+) stay at 1.0 to avoid over-promoting candidates
        # with only slightly reduced substitution costs.
        plausibility = 1.0
        if data.edit_distance > 0:
            ratio = None
            if data.weighted_distance is not None and data.weighted_distance < data.edit_distance:
                ratio = data.weighted_distance / data.edit_distance
            elif data.syllable_distance is not None and data.syllable_distance < data.edit_distance:
                ratio = data.syllable_distance / data.edit_distance
            if ratio is not None and ratio < self.plausibility_threshold:
                plausibility = max(ratio, self.plausibility_floor)

        base_score = float(data.edit_distance) * plausibility

        # Frequency bonus: 0.0 to freq_bonus_ceiling (asymptotic)
        freq_bonus = 0.0
        if data.frequency > 0:
            freq_bonus = self.freq_bonus_ceiling * (
                1.0 - (1.0 / (1.0 + (data.frequency / self.frequency_denominator)))
            )

        # Phonetic bonus
        phonetic_bonus = 0.0
        if data.phonetic_score > 0:
            phonetic_bonus = self.phonetic_bonus_weight * data.phonetic_score

        # Nasal variant bonuses
        nasal_bonus = self.nasal_bonus_weight if data.is_nasal_variant else 0.0
        same_nasal_bonus = self.same_nasal_bonus_weight if data.has_same_nasal_ending else 0.0

        # POS fit bonus
        pos_bonus = 0.0
        if data.pos_fit_score is not None and data.pos_fit_score > 0:
            pos_bonus = self.pos_bonus_weight * data.pos_fit_score

        # Span-length bonus/penalty: prefer suggestions whose length matches
        # the error span.  For longer errors (>=long_error_threshold chars)
        # the bonus scales up so exact-length matches strongly outrank
        # compound extensions or morpheme fragments.
        #
        # Short errors: fixed tiers
        # Long errors: scaled bonus + penalty for far misses
        span_bonus = 0.0
        if data.error_length is not None and data.error_length > 0:
            len_diff = abs(len(data.term) - data.error_length)
            capped_len = min(data.error_length, self.span_cap)
            if data.error_length >= self.long_error_threshold:
                # Long errors: scaled bonus, penalize far misses
                if len_diff == 0:
                    span_bonus = self.long_exact_base + self.long_exact_scale * capped_len
                elif len_diff <= 2:
                    span_bonus = self.long_close_base + self.long_close_scale * capped_len
                elif len_diff <= 4:
                    span_bonus = self.long_medium_bonus
                else:
                    len_ratio = len_diff / data.error_length
                    span_bonus = self.long_far_penalty * len_ratio
            else:
                # Short errors: conservative fixed tiers, no penalty
                if len_diff == 0:
                    span_bonus = self.short_exact_bonus
                elif len_diff <= 1:
                    span_bonus = self.short_close_bonus
                elif len_diff <= 2:
                    span_bonus = self.short_medium_bonus

        total_bonus = (
            freq_bonus + phonetic_bonus + nasal_bonus + same_nasal_bonus + pos_bonus + span_bonus
        )
        return base_score - total_bonus


class FrequencyFirstRanker(SuggestionRanker):
    """
    Ranker that prioritizes corpus frequency over edit distance.

    Useful for suggesting common words even if they have slightly
    higher edit distance. Good for autocomplete-style suggestions.

    Scoring formula:
        score = edit_distance * edit_weight - log_freq_bonus

    Where:
        - edit_weight: Reduces impact of edit distance
        - log_freq_bonus: Logarithmic frequency bonus

    Attributes:
        ranker_config: RankerConfig with ranking parameters.
    """

    def __init__(
        self,
        ranker_config: RankerConfig | None = None,
    ):
        """
        Initialize the frequency-first ranker.

        Args:
            ranker_config: RankerConfig with ranking parameters (uses defaults if None).
        """
        self.ranker_config = ranker_config or _default_ranker_config
        self.edit_weight = self.ranker_config.frequency_first_edit_weight
        self.frequency_scale = self.ranker_config.frequency_first_scale

    @property
    def name(self) -> str:
        """Return ranker identifier string."""
        return "frequency_first"

    def score(self, data: SuggestionData) -> float:
        """
        Calculate score prioritizing frequency over edit distance.

        Args:
            data: SuggestionData with term information.

        Returns:
            Float score (lower is better).
        """
        base_score = float(data.edit_distance) * self.edit_weight

        # Logarithmic frequency bonus
        freq_bonus = 0.0
        if data.frequency > 0:
            freq_bonus = self.frequency_scale * math.log1p(data.frequency)

        return base_score - freq_bonus


class EditDistanceOnlyRanker(SuggestionRanker):
    """
    Ranker that uses only edit distance for scoring.

    Simple ranker that ignores frequency and phonetic information.
    Useful for testing or when frequency data is unreliable.
    """

    @property
    def name(self) -> str:
        """Return ranker identifier string."""
        return "edit_distance_only"

    def score(self, data: SuggestionData) -> float:
        """
        Calculate score based only on edit distance.

        Args:
            data: SuggestionData with term information.

        Returns:
            Float score equal to edit distance.
        """
        return float(data.edit_distance)


class PhoneticFirstRanker(SuggestionRanker):
    """
    Ranker that prioritizes phonetic similarity for Myanmar text.

    Useful for catching phonetic confusions common in Myanmar spelling
    errors (e.g., similar sounding syllables with different medials).

    Scoring formula:
        score = edit_distance - phonetic_score * phonetic_weight

    Attributes:
        ranker_config: RankerConfig with ranking parameters.
    """

    def __init__(
        self,
        ranker_config: RankerConfig | None = None,
    ):
        """
        Initialize the phonetic-first ranker.

        Args:
            ranker_config: RankerConfig with ranking parameters (uses defaults if None).
        """
        self.ranker_config = ranker_config or _default_ranker_config
        self.phonetic_weight = self.ranker_config.phonetic_first_weight
        self.edit_weight = self.ranker_config.phonetic_first_edit_weight

    @property
    def name(self) -> str:
        """Return ranker identifier string."""
        return "phonetic_first"

    def score(self, data: SuggestionData) -> float:
        """
        Calculate score prioritizing phonetic similarity.

        Args:
            data: SuggestionData with term information.

        Returns:
            Float score (lower is better).
        """
        base_score = float(data.edit_distance) * self.edit_weight
        phonetic_bonus = self.phonetic_weight * data.phonetic_score
        return base_score - phonetic_bonus


class UnifiedRanker(SuggestionRanker):
    """
    Unified ranker that consolidates suggestions from multiple sources.

    This ranker applies source-specific weights to prioritize high-confidence
    suggestions from rule-based sources (particle_typo, medial_confusion) over
    statistical sources (symspell).

    Source weights are configurable via RankerConfig.

    Scoring formula (normalized):
        normalized_score = normalize(base_score, max_edit_distance)
        bonus = source_weight * confidence * weight_scale
        final_score = normalized_score - bonus

    Where:
        - base_score comes from the DefaultRanker formula
        - normalize() maps score to [0, 1] range
        - source_weight and confidence provide bonuses (higher reduces score)

    Attributes:
        ranker_config: RankerConfig with ranking parameters.
        source_weights: Dictionary mapping source names to weight multipliers.
        base_ranker: DefaultRanker used for base score calculation.
        max_edit_distance: Maximum expected edit distance for normalization.
        weight_scale: Scale factor for source weight bonus.
    """

    def __init__(
        self,
        ranker_config: RankerConfig | None = None,
        source_weights: dict[str, float] | None = None,
    ):
        """
        Initialize the unified ranker.

        Args:
            ranker_config: RankerConfig with ranking parameters (uses defaults if None).
            source_weights: Optional custom source weights. Uses config defaults if None.
        """
        self.ranker_config = ranker_config or _default_ranker_config
        self.source_weights = source_weights or self.ranker_config.get_source_weights()
        self.MAX_EDIT_DISTANCE = self.ranker_config.unified_max_edit_distance
        self.WEIGHT_SCALE = self.ranker_config.unified_weight_scale
        # Score tie precision from config (previously hardcoded as 6)
        self.SCORE_TIE_PRECISION = self.ranker_config.score_tie_precision
        self.base_ranker = self._create_base_ranker()

    def _create_base_ranker(self) -> SuggestionRanker:
        """
        Create the base ranker based on unified_base_ranker_type config.

        Returns:
            SuggestionRanker instance (never UnifiedRanker to avoid recursion).
        """
        base_type = self.ranker_config.unified_base_ranker_type

        if base_type == "frequency_first":
            return FrequencyFirstRanker(ranker_config=self.ranker_config)
        elif base_type == "phonetic_first":
            return PhoneticFirstRanker(ranker_config=self.ranker_config)
        elif base_type == "edit_distance_only":
            return EditDistanceOnlyRanker()
        else:
            # Default case
            return DefaultRanker(ranker_config=self.ranker_config)

    @property
    def name(self) -> str:
        """Return ranker identifier string."""
        return "unified"

    def get_source_weight(self, source: str) -> float:
        """Get the weight for a given source, defaulting to 1.0."""
        return self.source_weights.get(source, 1.0)

    def _normalize_score(self, raw_score: float) -> float:
        """
        Normalize raw score to [0, 1] range.

        Uses sigmoid-like transformation to handle negative and extreme values:
        - Negative scores (many bonuses) map to values near 0
        - Zero maps to 0.5 (neutral)
        - Positive scores map to values near 1
        - Large positive scores asymptotically approach 1

        Args:
            raw_score: Raw score from base ranker (can be negative).

        Returns:
            Normalized score in [0, 1] range (lower is better).
        """
        # Shift so edit_distance 0 maps to ~0.2, edit_distance 2 to ~0.5
        # Scale factor controls steepness of sigmoid
        scale = 2.0 / self.MAX_EDIT_DISTANCE
        shifted = raw_score * scale

        # Sigmoid: 1 / (1 + exp(-x)) maps (-inf, inf) to (0, 1)
        # We want lower scores for better suggestions, so use 1 / (1 + exp(x))
        # This maps negative raw_score to low normalized score (good)
        # and positive raw_score to high normalized score (bad)
        try:
            normalized = 1.0 / (1.0 + math.exp(-shifted))
        except OverflowError:
            # Handle extreme values
            normalized = 0.0 if shifted < 0 else 1.0

        return normalized

    def _normalize_strategy_score(self, raw_score: float) -> float:
        """
        Normalize strategy score with capped linear scaling.

        This avoids sigmoid saturation for large context-derived strategy scores.
        """
        cap = self.ranker_config.strategy_score_cap
        clamped = min(max(raw_score, 0.0), cap)
        return clamped / cap

    def score(self, data: SuggestionData) -> float:
        """
        Calculate unified score incorporating source weight, confidence, and strategy_score.

        The score combines feature-based ranking with strategy-level scores:
        1. Feature score: edit_distance, frequency, phonetic similarity, source weight
        2. Strategy score: per-strategy scoring (SymSpell, context, compound)
        3. Final score: weighted blend of feature and strategy scores

        When strategy_score is present, it is blended with the feature score
        using strategy_score_weight (default 0.5 = equal weight).

        Args:
            data: SuggestionData with term and source information.

        Returns:
            Float score in approximately [0, 1] range (lower is better).
        """
        # Get base score from DefaultRanker
        base_score = self.base_ranker.score(data)

        # Normalize to [0, 1] range using sigmoid
        normalized_score = self._normalize_score(base_score)

        # Apply source weight and confidence as independent bonus components
        # Higher source_weight (rule-based) and higher confidence reduce score
        source_weight = self.get_source_weight(data.source)
        confidence = max(data.confidence, CONFIDENCE_FLOOR)

        # Bonus calculation with two independent components:
        # 1. Source weight bonus: rule-based sources (weight > 1.0) get bonus
        # 2. Confidence bonus: higher confidence (> 0.5 baseline) gets bonus
        source_bonus = (source_weight - 1.0) * self.WEIGHT_SCALE
        confidence_bonus = (confidence - 0.5) * self.WEIGHT_SCALE
        bonus = source_bonus + confidence_bonus

        # Subtract bonus from normalized score (lower is better)
        feature_score = normalized_score - bonus

        # Blend with strategy_score when present
        if data.strategy_score is not None:
            # Normalize strategy score with capped linear scaling to preserve
            # relative differences in high-range context scores.
            normalized_strategy = self._normalize_strategy_score(data.strategy_score)
            # Blend: (1 - w) * feature_score + w * normalized_strategy
            strategy_weight = (
                self.ranker_config.context_strategy_score_weight
                if data.source == "context"
                else self.ranker_config.strategy_score_weight
            )
            final_score = (
                1 - strategy_weight
            ) * feature_score + strategy_weight * normalized_strategy
        else:
            final_score = feature_score

        # Clamp to reasonable range (can go slightly below 0 for excellent matches)
        return max(final_score, -0.5)

    def rank_suggestions(
        self,
        suggestions: list[SuggestionData],
        deduplicate: bool = True,
        enforce_diversity: bool = True,
        similarity_threshold: float | None = None,
        error_length: int | None = None,
    ) -> list[SuggestionData]:
        """
        Rank and optionally deduplicate a list of suggestions.

        This method scores all suggestions, optionally removes duplicates
        (keeping the highest-confidence version), filters near-duplicates
        for diversity, and sorts by score.

        Args:
            suggestions: List of SuggestionData to rank.
            deduplicate: If True, remove duplicate terms keeping best source.
            enforce_diversity: If True, filter near-duplicate suggestions.
            similarity_threshold: Threshold for near-duplicate detection (0.0-1.0).
            error_length: Character length of the original error span.
                When provided, injected into each SuggestionData for
                span-length bonus scoring.

        Returns:
            Sorted list of SuggestionData (best first, diverse options).
        """
        if not suggestions:
            return []

        # Inject error_length into each suggestion for span-length bonus
        if error_length is not None:
            for s in suggestions:
                s.error_length = error_length

        if similarity_threshold is None:
            similarity_threshold = self.ranker_config.similarity_threshold

        if deduplicate:
            suggestions = self._deduplicate(suggestions)

        # Score and sort (lower score = better). Use deterministic tie-breaks
        # for near-equal scores: edit_distance, weighted_distance, frequency.
        scored = [(self.score(s), s) for s in suggestions]
        scored.sort(
            key=lambda x: (
                round(x[0], self.SCORE_TIE_PRECISION),
                x[1].edit_distance,
                x[1].weighted_distance if x[1].weighted_distance is not None else float("inf"),
                -(x[1].pos_fit_score or 0.0),
                -x[1].frequency,
                x[1].term,
            )
        )

        ranked = [s for _, s in scored]

        # Apply diversity filter after ranking
        if enforce_diversity:
            ranked = self.filter_near_duplicates(ranked, similarity_threshold)

        return ranked

    def _deduplicate(self, suggestions: list[SuggestionData]) -> list[SuggestionData]:
        """
        Remove duplicate terms, keeping the best version.

        Uses source-weight-aware deduplication: candidates are compared by
        ``get_source_weight(source) * confidence``, with tie-breaks on
        strategy_score, edit_distance, and frequency.

        Args:
            suggestions: List of SuggestionData that may contain duplicates.

        Returns:
            List with duplicates removed.
        """
        from myspellchecker.algorithms.dedup import deduplicate_suggestions

        return deduplicate_suggestions(suggestions, weight_fn=self.get_source_weight)

    def filter_near_duplicates(
        self,
        suggestions: list[SuggestionData],
        similarity_threshold: float = 0.8,
    ) -> list[SuggestionData]:
        """
        Filter near-duplicate suggestions to improve diversity.

        Near-duplicates are suggestions that differ only in tone marks,
        medials, or minor character variations. This method keeps diverse
        suggestions that represent different correction strategies.

        Args:
            suggestions: Sorted list of SuggestionData (best first).
            similarity_threshold: Similarity threshold (0.0-1.0) above which
                suggestions are considered near-duplicates. Default 0.8.

        Returns:
            Filtered list with near-duplicates removed.

        Example:
            >>> ranker = UnifiedRanker()
            >>> # If suggestions are ["မြန်", "မြန်း", "မျန်", "မြန်မာ"]
            >>> # "မြန်း" is near-duplicate of "မြန်" (differs only in tone)
            >>> # Result: ["မြန်", "မျန်", "မြန်မာ"] - diverse options
        """
        if len(suggestions) <= 1:
            return suggestions

        filtered: list[SuggestionData] = []

        for suggestion in suggestions:
            is_near_duplicate = False

            for kept in filtered:
                similarity = self._calculate_similarity(kept.term, suggestion.term)
                if similarity >= similarity_threshold:
                    # Check if they represent different error types
                    if not self._are_diverse_suggestions(kept, suggestion):
                        is_near_duplicate = True
                        break

            if not is_near_duplicate:
                filtered.append(suggestion)

        return filtered

    def _calculate_similarity(self, term1: str, term2: str) -> float:
        """
        Calculate similarity ratio between two terms.

        Uses character-level Jaccard similarity with position weighting.

        Args:
            term1: First term.
            term2: Second term.

        Returns:
            Similarity score between 0.0 (different) and 1.0 (identical).
        """
        if term1 == term2:
            return 1.0
        if not term1 or not term2:
            return 0.0

        # Character-level comparison
        set1 = set(term1)
        set2 = set(term2)

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        if union == 0:
            return 0.0

        jaccard = intersection / union

        # Boost similarity for terms with same length and similar structure
        length_similarity = 1.0 - abs(len(term1) - len(term2)) / max(len(term1), len(term2))

        # Weighted combination
        return 0.6 * jaccard + 0.4 * length_similarity

    def _are_diverse_suggestions(
        self, suggestion1: SuggestionData, suggestion2: SuggestionData
    ) -> bool:
        """
        Check if two suggestions represent diverse correction types.

        Suggestions are considered diverse if they come from different
        sources or represent different error correction strategies.

        Args:
            suggestion1: First suggestion.
            suggestion2: Second suggestion.

        Returns:
            True if suggestions are diverse (should both be kept).
        """
        # Different sources = diverse
        if suggestion1.source != suggestion2.source:
            return True

        # Different nasal variants = diverse (represent different pronunciations)
        if suggestion1.is_nasal_variant != suggestion2.is_nasal_variant:
            return True

        # Significant edit distance difference = diverse
        if abs(suggestion1.edit_distance - suggestion2.edit_distance) >= 2:
            return True

        # Same source and similar characteristics = near-duplicate
        return False
