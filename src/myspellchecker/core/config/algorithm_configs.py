"""
Algorithm Configuration Classes.

This module contains configuration classes for spell checking algorithms:
- SymSpellConfig: Symmetric delete spelling correction algorithm
- NgramContextConfig: Bigram/trigram context validation
- PhoneticConfig: Phonetic matching and similarity
- SemanticConfig: Deep learning semantic validation (BERT/RoBERTa)
- CompoundResolverConfig: Compound word resolution parameters
- ReduplicationConfig: Reduplication detection parameters
- FrequencyGuardConfig: Centralized frequency thresholds for validators/strategies
- ConfusableSemanticConfig: Confusable semantic validation strategy thresholds
- HomophoneStrategyConfig: Homophone validation strategy thresholds
- NgramStrategyConfig: N-gram context validation strategy thresholds
- SemanticStrategyConfig: Semantic validation strategy thresholds
- ToneStrategyConfig: Tone validation strategy thresholds
- BrokenCompoundStrategyConfig: Broken compound detection strategy thresholds
- TokenRefinementConfig: Token boundary refinement scoring parameters
"""

from __future__ import annotations

import warnings
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from myspellchecker.core.constants.detector_thresholds import DEFAULT_ALGORITHM_THRESHOLDS


class SymSpellConfig(BaseModel):
    """
    Configuration for SymSpell algorithm.

    Controls the symmetric delete spelling correction algorithm including
    indexing, lookup, and compound word handling.

    Attributes:
        prefix_length: Prefix length for optimization (default: 10).
            Longer prefixes use more memory but reduce false positives.
        count_threshold: Minimum frequency threshold for dictionary terms (default: 50).
            Higher values filter out noise/typos from training data.
        max_word_length: Max word length for compound segmentation (default: 15).
        compound_lookup_count: Suggestions per word in compound check (default: 3).
        beam_width: Beam width for compound segmentation DP (default: 50).
        compound_max_suggestions: Max suggestions for compound queries (default: 5).
        damerau_cache_size: LRU cache size for edit distance calculations (default: 4096).
        frequency_denominator: Denominator for frequency bonus calculation (default: 10000.0).
        phonetic_bonus_weight: Weight for phonetic similarity bonus (default: 0.4).
        skip_init: Skip SymSpell initialization for POS-only use (default: False).
        known_word_frequency_threshold: Min frequency for "known word" bypass (default: 100).
        use_weighted_distance: Enable Myanmar-weighted edit distance (default: True).
        syllable_bonus_weight: Weight for syllable-aware scoring bonus (default: 0.3).
        weighted_distance_bonus_weight: Weight for weighted distance bonus (default: 0.35).
        max_deletes_per_term: Max delete variations per term (default: 5000).
        use_syllable_distance: Enable syllable-aware edit distance (default: True).
        use_myanmar_variants: Enable Myanmar variant candidate generation (default: True).
        myanmar_variant_max_candidates: Max Myanmar variant candidates per lookup (default: 20).
    """

    model_config = ConfigDict(
        validate_assignment=True,  # Validate on attribute assignment
        frozen=False,  # Allow modification after creation
        extra="forbid",  # Reject unknown fields
    )

    prefix_length: int = Field(
        default=DEFAULT_ALGORITHM_THRESHOLDS.prefix_length,
        ge=1,
        description="Prefix length for optimization (longer = more memory, fewer false positives)",
    )
    count_threshold: int = Field(
        default=50,
        ge=0,
        description="Minimum frequency threshold for dictionary terms",
    )
    max_word_length: int = Field(
        default=15,
        ge=1,
        description="Max word length for compound segmentation",
    )
    compound_lookup_count: int = Field(
        default=3,
        ge=1,
        description="Suggestions per word in compound check",
    )
    beam_width: int = Field(
        default=DEFAULT_ALGORITHM_THRESHOLDS.beam_width_default,
        ge=1,
        description="Beam width for compound segmentation dynamic programming",
    )
    compound_max_suggestions: int = Field(
        default=5,
        ge=1,
        description="Max suggestions for compound queries",
    )
    damerau_cache_size: int = Field(
        default=DEFAULT_ALGORITHM_THRESHOLDS.lru_cache_size,
        ge=0,
        description="LRU cache size for edit distance calculations",
    )
    frequency_denominator: float = Field(
        default=10000.0,
        gt=0,
        description="Denominator for frequency bonus calculation",
    )
    phonetic_bonus_weight: float = Field(
        default=0.4,
        ge=0,
        le=1.0,
        description="Weight for phonetic similarity bonus (0.0-1.0)",
    )
    skip_init: bool = Field(
        default=False,
        description="Skip SymSpell initialization for POS-only use",
    )
    known_word_frequency_threshold: int = Field(
        default=100,
        ge=0,
        description=(
            "Minimum corpus frequency for a word to be considered 'known'. "
            "Words with frequency >= this threshold bypass certain error checks."
        ),
    )
    use_weighted_distance: bool = Field(
        default=True,
        description=(
            "Use Myanmar-weighted edit distance for candidate scoring. "
            "When enabled, uses MYANMAR_SUBSTITUTION_COSTS to give lower "
            "cost to phonetically similar character substitutions (e.g., "
            "aspirated consonant pairs, medial confusions, vowel length)."
        ),
    )
    syllable_bonus_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description=(
            "Weight bonus for syllable-aware scoring (0.0-1.0). "
            "Used to configure the DefaultRanker when no custom ranker "
            "is provided. Treats medial confusions as single edits."
        ),
    )
    weighted_distance_bonus_weight: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description=(
            "Weight for weighted distance bonus (0.0-1.0). "
            "Applied when weighted_distance < edit_distance, indicating "
            "phonetically related character substitutions. "
            "Used to configure the DefaultRanker when no custom ranker "
            "is provided."
        ),
    )
    max_deletes_per_term: int = Field(
        default=5000,
        ge=100,
        description=(
            "Maximum delete variations to generate per term during "
            "indexing. Prevents O(n^d) memory growth for long terms "
            "or high max_edit_distance values."
        ),
    )
    use_syllable_distance: bool = Field(
        default=True,
        description=(
            "Enable Myanmar syllable-aware edit distance. "
            "Treats medial confusions (ျ vs ြ) as single edits."
        ),
    )
    use_myanmar_variants: bool = Field(
        default=True,
        description=(
            "Enable Myanmar-specific variant candidate generation for OOV terms. "
            "Generates candidates via medial swaps, aspiration swaps, nasal "
            "confusion, stop-coda mergers, tone mark and vowel length swaps. "
            "Complements standard delete-based lookup."
        ),
    )
    myanmar_variant_max_candidates: int = Field(
        default=20,
        ge=1,
        le=100,
        description=(
            "Maximum number of Myanmar variant candidates to add per lookup. "
            "Prevents candidate explosion for highly ambiguous terms."
        ),
    )

    # --- Edit distance weights ---
    diacritic_indel_cost: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Insertion/deletion cost for Myanmar diacritics (vs 1.0 for consonants).",
    )
    keyboard_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Weight for keyboard adjacency in edit distance.",
    )
    visual_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Weight for visual similarity in edit distance.",
    )
    transposition_weight: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Weight for transposition (Damerau) operations.",
    )

    # --- SymSpell internal cache sizes ---
    deletes_cache_max: int = Field(
        default=4096,
        ge=0,
        description="Max entries in delete-variation cache (reduces ~70%% regen work).",
    )
    compound_seg_cache_max: int = Field(
        default=512,
        ge=0,
        description="Max entries in compound segmentation result cache.",
    )
    lookup_cache_max: int = Field(
        default=1024,
        ge=0,
        description="Max entries in session-level lookup result cache.",
    )
    max_compound_seg_len: int = Field(
        default=24,
        ge=1,
        description="Max character length for compound DP segmentation.",
    )
    max_syllables_per_word: int = Field(
        default=5,
        ge=1,
        description="Max syllables per word candidate in compound DP.",
    )

    # --- Suggestion pipeline tuning (used by SuggestionPipelineMixin) ---
    morpheme_promote_max_err_len: int = Field(
        default=8,
        ge=1,
        description="Max error span length for morpheme suggestion promotion.",
    )
    morpheme_compound_ctx: int = Field(
        default=15,
        ge=1,
        description="Context window chars for morpheme compound reconstruction.",
    )
    morpheme_compound_min_len: int = Field(
        default=4,
        ge=1,
        description="Min error length for morpheme compound reconstruction.",
    )
    morpheme_compound_max_sugg: int = Field(
        default=8,
        ge=1,
        description="Max suggestions for morpheme compound reconstruction.",
    )
    asat_context_window: int = Field(
        default=10,
        ge=1,
        description="Context window chars for asat insertion detection.",
    )
    distance_rerank_min_gap: float = Field(
        default=1.2,
        ge=0.0,
        description="Min distance gap for distance-based reranking.",
    )
    distance_rerank_max_base_distance: float = Field(
        default=1.0,
        ge=0.0,
        description="Max base distance for rerank eligibility.",
    )
    distance_rerank_max_promote_distance: float = Field(
        default=1.0,
        ge=0.0,
        description="Max promoted distance for reranking.",
    )
    span_length_min_error_len: int = Field(
        default=5,
        ge=1,
        description="Min error char length for span-length penalty.",
    )
    span_length_penalty_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight for span-length mismatch penalty.",
    )


class NgramContextConfig(BaseModel):
    """
    Configuration for N-gram context checker.

    Controls bigram/trigram probability checking for context-aware
    spell checking (Layer 3).

    Attributes:
        threshold: General threshold (used by AlgorithmFactory standalone API;
            the main pipeline uses bigram_threshold instead).
        edit_distance_weight: Weight for edit distance in scoring (default: 0.6).
        probability_weight: Weight for probability in scoring (default: 0.4).
        pos_score_weight: Weight for POS influence in scoring (default: 0.2).
        pos_distance_reduction_factor: Multiplier for POS distance reduction (default: 5.0).
        score_scaling_factor: Scaling factor for N-gram scores (default: 10.0).
        bigram_threshold: Probability threshold for flagging bigram errors (default: 0.0001).
        trigram_threshold: Probability threshold for flagging trigram errors (default: 0.0001).
        right_context_threshold: Threshold for right context rescue (default: 0.001).
        edit_distance: Max edit distance for context corrections (default: 2).
        candidate_limit: Maximum candidates to consider (default: 50).
        min_prob_denominator: Minimum probability denominator (default: 0.001).
        heuristic_multiplier: Heuristic multiplier for context (default: 10.0).
        backoff_floor_multiplier: Multiplier for trigram backoff floor (default: 0.1).
        use_smoothing: Enable probability smoothing (default: True).
        smoothing_strategy: Smoothing strategy string (default: 'stupid_backoff').
        backoff_weight: Stupid Backoff weight (default: 0.4).
        add_k_smoothing: Add-k constant (default: 0.0).
        unigram_denominator: Corpus size estimate for unigram P (default: 500000000.0).
        unigram_prob_cap: Max unigram probability cap (default: 0.1).
        min_unigram_threshold: Min frequency for valid-in-unseen-context (default: 5).
        max_suggestions: Max context-aware suggestions to return (default: 5).
        case1_freq_guard: Frequency guard for Case 1 detection (default: 10000).
        rerank_left_weight: Left context weight for re-ranking (default: 0.6).
        rerank_right_weight: Right context weight for re-ranking (default: 0.35).
    """

    model_config = ConfigDict(
        validate_assignment=True,
        frozen=False,
        extra="forbid",
    )

    threshold: float = Field(
        default=DEFAULT_ALGORITHM_THRESHOLDS.ngram_threshold,
        gt=0.0,
        description=(
            "General probability threshold for N-gram context checking. "
            "Used by AlgorithmFactory standalone API. The main SpellChecker "
            "pipeline uses bigram_threshold instead."
        ),
    )
    edit_distance_weight: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Weight for edit distance in scoring (0.0-1.0)",
    )
    probability_weight: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Weight for probability in scoring (0.0-1.0)",
    )
    pos_score_weight: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Weight for POS influence in scoring",
    )
    pos_distance_reduction_factor: float = Field(
        default=5.0,
        gt=0.0,
        description="Multiplier for reducing edit distance based on POS similarity",
    )
    score_scaling_factor: float = Field(
        default=10.0,
        gt=0.0,
        description="Scaling factor for N-gram probability scores",
    )
    bigram_threshold: float = Field(
        default=0.0001,
        gt=0.0,
        description="Minimum probability threshold for flagging bigram errors",
    )
    trigram_threshold: float = Field(
        default=0.0001,
        gt=0.0,
        description="Minimum probability threshold for flagging trigram errors",
    )
    fourgram_threshold: float = Field(
        default=0.0001,
        gt=0.0,
        description="Minimum probability threshold for flagging 4-gram context errors",
    )
    fivegram_threshold: float = Field(
        default=0.0001,
        gt=0.0,
        description="Minimum probability threshold for flagging 5-gram context errors",
    )
    right_context_threshold: float = Field(
        default=0.001,
        gt=0.0,
        description=(
            "Probability threshold for right context to rescue a word. "
            "When bidirectional context is used, a word is rescued (not flagged) "
            "if the right context probability P(next|current) exceeds this threshold. "
            "Default is 10x higher than bigram_threshold since right context rescue "
            "should require stronger evidence."
        ),
    )
    edit_distance: int = Field(
        default=DEFAULT_ALGORITHM_THRESHOLDS.max_edit_distance,
        ge=0,
        description="Maximum edit distance for context-based corrections",
    )
    candidate_limit: int = Field(
        default=DEFAULT_ALGORITHM_THRESHOLDS.candidate_limit,
        ge=1,
        description="Maximum number of correction candidates to consider",
    )
    min_prob_denominator: float = Field(
        default=DEFAULT_ALGORITHM_THRESHOLDS.min_probability_denom,
        gt=0.0,
        description="Minimum denominator for probability calculations (prevents division by zero)",
    )
    heuristic_multiplier: float = Field(
        default=10.0,
        gt=0.0,
        description="Multiplier for heuristic scoring in context validation",
    )
    backoff_floor_multiplier: float = Field(
        default=0.1,
        gt=0.0,
        le=1.0,
        description=(
            "Multiplier for trigram backoff floor calculation. "
            "When backing off from trigram to unigram, the floor probability is "
            "calculated as: trigram_threshold * backoff_floor_multiplier. "
            "Higher values (e.g., 0.2) are more lenient on rare but valid words. "
            "Lower values (e.g., 0.05) are stricter but may cause false positives."
        ),
    )
    use_smoothing: bool = Field(
        default=True,
        description="Enable probability smoothing for sparse N-gram data.",
    )
    smoothing_strategy: str = Field(
        default="stupid_backoff",
        description=(
            "Smoothing strategy for unseen N-grams. "
            "Options: 'none', 'stupid_backoff' (default), 'add_k'."
        ),
    )
    backoff_weight: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description=(
            "Weight for Stupid Backoff smoothing. "
            "Unseen bigram probability = backoff_weight * P(unigram)."
        ),
    )
    add_k_smoothing: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Add-k smoothing constant. Added to all probabilities when "
            "smoothing_strategy is 'add_k'. 0.0 = disabled."
        ),
    )
    unigram_denominator: float = Field(
        default=500_000_000.0,
        gt=0.0,
        description=(
            "Denominator for unigram probability estimation. "
            "Should approximate total corpus word count. "
            "Default 500M is a reasonable estimate for production corpora. "
            "Overridden at runtime if DB metadata contains total_word_count."
        ),
    )
    unigram_prob_cap: float = Field(
        default=0.1,
        gt=0.0,
        le=1.0,
        description=(
            "Maximum unigram probability cap. Prevents high-frequency words "
            "from dominating backoff scores."
        ),
    )
    min_unigram_threshold: int = Field(
        default=5,
        ge=0,
        description=(
            "Minimum unigram frequency to consider a word valid in unseen contexts. "
            "Words with frequency >= this threshold are not flagged when their "
            "bigram probability is zero (assumed valid in a new context)."
        ),
    )
    max_suggestions: int = Field(
        default=5,
        ge=1,
        description=(
            "Maximum number of context-aware suggestions to return "
            "from NgramContextChecker.suggest()."
        ),
    )
    case1_freq_guard: int = Field(
        default=10_000,
        ge=0,
        description=(
            "Frequency guard for Case 1 (low-but-nonzero left "
            "bigram, absent right context). Words with frequency "
            ">= this threshold are treated as correct in a "
            "rare-but-observed context."
        ),
    )
    rerank_left_weight: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description=(
            "Weight for left context (prev_words) in re-ranking suggestions. "
            "Used by ContextSuggestionStrategy to blend left-context log-probabilities "
            "into the final score. Higher values prioritize left context more. "
            "Default: 0.6."
        ),
    )
    rerank_right_weight: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description=(
            "Weight for right context (next_words) in re-ranking suggestions. "
            "Used by ContextSuggestionStrategy to blend right-context log-probabilities "
            "into the final score. Typically lower than left weight since right context "
            "is less predictive for Myanmar word order. "
            "Default: 0.35."
        ),
    )
    min_meaningful_prob: float = Field(
        default=1e-7,
        ge=0.0,
        description=(
            "Minimum combined probability threshold; skip comparison when "
            "both probs are below this."
        ),
    )
    collocation_pmi_threshold: float = Field(
        default=5.0,
        ge=0.0,
        description="PMI threshold for strong collocations that block error flagging.",
    )
    confidence_zero_prob: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Confidence for errors where word probability is zero.",
    )
    confidence_with_prob: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Base confidence for errors where word has some probability.",
    )
    confusable_confidence_cap: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Maximum confidence for confusable errors.",
    )
    confusable_confidence_base: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Base confidence in confusable confidence formula.",
    )
    confusable_confidence_scale: float = Field(
        default=0.1,
        ge=0.0,
        description="Per-unit scaling in confusable confidence formula.",
    )
    confusable_confidence_ratio_divisor: float = Field(
        default=10.0,
        gt=0.0,
        description="Ratio divisor in confusable confidence formula.",
    )
    confusable_confidence_ratio_cap: float = Field(
        default=3.5,
        ge=0.0,
        description="Max ratio contribution to confusable confidence.",
    )

    # --- Confusable detection ratio parameters ---
    # Control how aggressively the n-gram checker flags confusable word pairs.
    confusable_base_ratio: float = Field(
        default=8.0,
        ge=1.0,
        description=(
            "Base probability ratio required for low-frequency words "
            "(freq < confusable_high_freq_threshold) to flag a confusable "
            "error. Higher = more conservative."
        ),
    )
    confusable_high_freq_ratio: float = Field(
        default=50.0,
        ge=1.0,
        description=(
            "Stricter probability ratio required for high-frequency words "
            "(freq >= confusable_high_freq_threshold). Protects common words from FPs."
        ),
    )
    confusable_high_freq_threshold: int = Field(
        default=5000,
        ge=0,
        description="Word frequency above which confusable_high_freq_ratio applies.",
    )
    overlap_ratio_base: float = Field(
        default=5.0,
        ge=0.0,
        description=(
            "Base value for overlap-based confusable ratio interpolation. "
            "At overlap=0.0, required ratio = overlap_ratio_base."
        ),
    )
    overlap_ratio_range: float = Field(
        default=45.0,
        ge=0.0,
        description=(
            "Range for overlap-based confusable ratio interpolation. "
            "At overlap=1.0, required ratio = overlap_ratio_base + overlap_ratio_range."
        ),
    )
    aspirated_min_ratio: float = Field(
        default=20.0,
        ge=1.0,
        description="Floor ratio for aspirated confusable pairs regardless of base_ratio.",
    )
    bidir_cache_max_size: int = Field(
        default=256,
        ge=0,
        description=(
            "Maximum entries in the bidirectional probability cache. "
            "Prevents unbounded memory growth for long documents."
        ),
    )

    @model_validator(mode="after")
    def validate_weights_sum(self) -> "NgramContextConfig":
        """Warn if edit_distance_weight and probability_weight don't sum to 1.0."""
        if abs(self.edit_distance_weight + self.probability_weight - 1.0) > 1e-6:
            warnings.warn(
                "edit_distance_weight and probability_weight should ideally sum to 1.0",
                UserWarning,
                stacklevel=2,
            )
        return self


class PhoneticConfig(BaseModel):
    """
    Configuration for phonetic matching.

    Controls phonetic encoding, similarity scoring, and suggestion matching
    for Myanmar text.

    Attributes:
        max_code_length: Maximum length for phonetic codes (default: 10).
        chars_per_code_unit: Characters per phonetic code unit for adaptive
            length calculation (default: 6).
        cache_size: LRU cache size for phonetic encoding (default: 4096).
            Set to 0 to disable caching.
        adaptive_code_length_cap: Upper cap for adaptive code length
            (default: 50).
        adaptive_scale_factor: Scale factor for adaptive code length bonus
            (default: 5).
        adaptive_min_input_length: Minimum input length before adaptive
            bonus applies (default: 4).
        max_distance: Maximum edit distance for phonetic code similarity
            comparison (default: 1).
        matching_code_similarity: Similarity score when phonetic codes match
            exactly (default: 0.98).
        length_penalty_weight: Weight for length-difference penalty in
            similarity scoring (default: 0.2).
        code_weight_cap: Maximum weight for code-level similarity blending
            (default: 0.4).
        code_weight_divisor: Divisor for input-length-based code weight
            (default: 20.0).
        cost_to_similarity_multiplier: Multiplier to convert substitution
            cost to similarity (default: 0.75).
        confusable_char_similarity: Similarity for visually confusable
            characters (default: 0.85).
        same_phonetic_group_similarity: Similarity for characters in the
            same phonetic group (default: 0.8).
        same_category_diff_group_similarity: Similarity for same-category
            but different-group characters (default: 0.5).
        medial_confusion_similarity: Similarity for medial character
            confusion (default: 0.6).
        cross_category_similarity: Similarity for cross-category character
            pairs (default: 0.2).
        fallback_similarity: Fallback similarity when no phonetic info is
            available (default: 0.1).
        suggestion_threshold_unseen: Threshold for unseen phonetic
            suggestions (default: 0.001).
        suggestion_threshold_improvement: Threshold for phonetic
            improvement (default: 0.01).
        suggestion_improvement_ratio: Improvement ratio for phonetic
            suggestions (default: 100.0).
        phonetic_bypass_threshold: Minimum similarity to bypass
            edit-distance cap (default: 0.85).
        phonetic_extra_distance: Extra distance allowed for phonetic
            bypass (default: 1).
    """

    model_config = ConfigDict(
        validate_assignment=True,
        frozen=False,
        extra="forbid",
    )

    # --- Encoding parameters ---
    max_code_length: int = Field(
        default=10,
        ge=1,
        description=("Maximum length for phonetic codes (longer codes = more precise matching)"),
    )
    chars_per_code_unit: int = Field(
        default=6,
        ge=1,
        description=("Characters per phonetic code unit for adaptive length calculation"),
    )
    cache_size: int = Field(
        default=4096,
        ge=0,
        description=("LRU cache size for phonetic encoding (0 to disable caching)"),
    )
    adaptive_code_length_cap: int = Field(
        default=50,
        ge=1,
        description="Upper cap for adaptive code length",
    )
    adaptive_scale_factor: int = Field(
        default=5,
        ge=1,
        description="Scale factor for adaptive code length bonus",
    )
    adaptive_min_input_length: int = Field(
        default=4,
        ge=1,
        description=("Minimum input length before adaptive bonus applies"),
    )

    # --- Similarity comparison ---
    max_distance: int = Field(
        default=1,
        ge=0,
        description=("Maximum edit distance for phonetic code similarity comparison"),
    )
    matching_code_similarity: float = Field(
        default=0.98,
        ge=0.0,
        le=1.0,
        description=("Similarity score returned when phonetic codes match exactly"),
    )
    length_penalty_weight: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description=("Weight for length-difference penalty in similarity scoring"),
    )
    code_weight_cap: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description=("Maximum weight for code-level similarity blending"),
    )
    code_weight_divisor: float = Field(
        default=20.0,
        gt=0.0,
        description=("Divisor for input-length-based code weight"),
    )

    # --- Character-level similarity ---
    cost_to_similarity_multiplier: float = Field(
        default=0.75,
        ge=0.0,
        le=2.0,
        description=(
            "Multiplier converting substitution cost to similarity (sim = 1 - cost * multiplier)"
        ),
    )
    confusable_char_similarity: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description=("Similarity score for visually confusable characters"),
    )
    same_phonetic_group_similarity: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description=("Similarity for characters in the same phonetic group"),
    )
    same_category_diff_group_similarity: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=("Similarity for same-category but different-group characters"),
    )
    medial_confusion_similarity: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description=("Similarity for medial character confusion"),
    )
    cross_category_similarity: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description=("Similarity for cross-category character pairs"),
    )
    fallback_similarity: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description=("Fallback similarity when no phonetic info is available"),
    )

    # --- Suggestion matching ---
    suggestion_threshold_unseen: float = Field(
        default=0.001,
        gt=0.0,
        description=("Probability threshold for suggesting unseen phonetic matches"),
    )
    suggestion_threshold_improvement: float = Field(
        default=0.01,
        gt=0.0,
        description=("Minimum improvement threshold for phonetic suggestions"),
    )
    suggestion_improvement_ratio: float = Field(
        default=100.0,
        gt=0.0,
        description=("Ratio by which a suggestion must improve over original"),
    )
    phonetic_bypass_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum graded phonetic similarity required to bypass "
            "the edit-distance cap. Candidates with similarity >= "
            "this threshold can exceed max_edit_distance by up to "
            "phonetic_extra_distance. Conservative default (0.85) "
            "limits false positives."
        ),
    )
    phonetic_extra_distance: int = Field(
        default=1,
        ge=0,
        le=3,
        description=(
            "Maximum additional edit distance allowed for "
            "high-similarity phonetic candidates. Only applies when "
            "phonetic_similarity >= phonetic_bypass_threshold. "
            "Default of 1 allows phonetic variants that are 1 edit "
            "beyond max_edit_distance."
        ),
    )


class SemanticConfig(BaseModel):
    """
    Configuration for semantic model (BERT/RoBERTa/XLM-RoBERTa).

    Controls optional deep learning-based semantic spell checking.

    Attributes:
        model_path: Path to ONNX model file (optional).
        tokenizer_path: Path to tokenizer. Can be either:
            - A file path to tokenizer.json (custom tokenizer format)
            - A directory path containing HuggingFace tokenizer files
              (XLM-RoBERTa, mBERT, etc.)
        model: Pre-loaded model instance (optional).
        tokenizer: Pre-loaded tokenizer instance (optional).
        num_threads: Number of threads for ONNX inference (default: 1).
        predict_top_k: Top-K predictions for semantic suggestions (default: 5).
        check_top_k: Top-K tokens to check for semantic errors (default: 10).
        use_semantic_refinement: Enable semantic refinement layer in
            context validation. When True and a model is configured,
            the SemanticChecker will refine N-gram error detection by:
            - Suppressing false positives (AI says word is correct)
            - Prioritizing AI-suggested corrections (default: True).
        use_proactive_scanning: Enable proactive semantic scanning to detect
            errors independently of N-gram analysis. When True, the model will
            scan each word in the sentence and flag those that the model doesn't
            predict as likely. Requires a well-trained Myanmar language model.
            Default: False (disabled until model quality is verified).
        proactive_confidence_threshold: Minimum confidence for proactive errors
            to be reported. Higher values reduce false positives (default: 0.85).
    """

    model_config = ConfigDict(
        validate_assignment=True,
        frozen=False,
        extra="forbid",
        arbitrary_types_allowed=True,  # Allow model and tokenizer instances
    )

    model_path: str | None = Field(
        default=None,
        description="Path to ONNX model file for semantic checking",
    )
    tokenizer_path: str | None = Field(
        default=None,
        description="Path to tokenizer.json or HuggingFace tokenizer directory",
    )
    model: Any | None = Field(
        default=None,
        description="Pre-loaded ONNX model instance (alternative to model_path)",
    )
    tokenizer: Any | None = Field(
        default=None,
        description="Pre-loaded tokenizer instance (alternative to tokenizer_path)",
    )
    num_threads: int = Field(
        default=0,
        ge=0,
        description=(
            "Number of threads for ONNX inference (CPU only). "
            "0 = auto-detect (use all available cores, recommended)."
        ),
    )
    predict_top_k: int = Field(
        default=5,
        ge=1,
        description="Number of top predictions to generate for suggestions",
    )
    check_top_k: int = Field(
        default=10,
        ge=1,
        description="Number of top tokens to check for semantic errors",
    )
    use_semantic_refinement: bool = Field(
        default=True,
        description="Enable semantic refinement to suppress N-gram false positives",
    )
    use_proactive_scanning: bool = Field(
        default=False,
        description=(
            "Enable proactive scanning to detect unlikely words (requires high-quality model)"
        ),
    )
    proactive_confidence_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for proactive error detection",
    )
    scoring_confidence_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description=(
            "Confidence threshold for semantic scoring in suggestion ranking. "
            "Suggestions with AI confidence below this threshold are penalized. "
            "Lower values are more lenient, higher values require stronger AI agreement."
        ),
    )
    logit_scale: float | None = Field(
        default=None,
        gt=0.0,
        description=(
            "Override automatic logit scale detection for confidence calibration. "
            "If None, auto-detects based on model type (e.g., XLM-RoBERTa: 10.0, BERT: 50.0)."
        ),
    )
    use_pytorch: bool = Field(
        default=False,
        description=(
            "Force PyTorch backend instead of ONNX Runtime. "
            "Useful for models without ONNX export or when debugging."
        ),
    )
    device: str = Field(
        default="cpu",
        description=(
            "Device for model inference (e.g., 'cpu', 'cuda:0', 'cuda:1'). "
            "Only applies when use_pytorch=True. ONNX uses CPU by default."
        ),
    )
    validate_model_architecture: bool = Field(
        default=True,
        description=(
            "Validate that loaded model has Masked Language Model (MLM) architecture. "
            "Disable only for custom models with non-standard output format."
        ),
    )
    myanmar_text_ratio_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum ratio of Myanmar characters in text for semantic checking. "
            "Text with a lower ratio of Myanmar characters is skipped."
        ),
    )
    word_alignment_enabled: bool = Field(
        default=True,
        description=(
            "Enable Myanmar word-aligned masking for BPE tokenizers. "
            "When True, masks are aligned to word boundaries rather than subword tokens."
        ),
    )

    # --- Internal SemanticChecker tuning constants ---
    # These control confidence calibration, beam search, scanning, and scoring
    # inside SemanticChecker. Defaults are empirically calibrated.

    # Confidence calibration: sigmoid-like transformation to bound raw logits [0, 1]
    confidence_high_threshold: float = Field(
        default=1.0,
        ge=0.0,
        description="Normalized score above which saturation applies in confidence calibration.",
    )
    confidence_saturation_numerator: float = Field(
        default=0.2,
        gt=0.0,
        description="Controls approach rate to 1.0 in saturation region.",
    )
    confidence_saturation_offset: float = Field(
        default=1.0,
        gt=0.0,
        description="Denominator offset in saturation formula.",
    )
    confidence_low_base: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Base confidence in linear region (low normalized score).",
    )
    confidence_low_scale: float = Field(
        default=0.5,
        ge=0.0,
        le=2.0,
        description="Linear scaling factor in low-confidence region.",
    )

    # Prefix skip margins: how close prefix prediction must be to top score to skip flagging
    prefix_skip_margin_short: float = Field(
        default=0.25,
        ge=0.0,
        description="Prefix skip margin for single-char words (len <= 1).",
    )
    prefix_skip_margin_medium: float = Field(
        default=0.45,
        ge=0.0,
        description="Prefix skip margin for two-char words (len == 2).",
    )
    prefix_skip_margin_long: float = Field(
        default=1.0,
        ge=0.0,
        description="Prefix skip margin for longer words (len >= 3).",
    )
    prefix_competing_non_prefix_gap: float = Field(
        default=0.25,
        ge=0.0,
        description="Gap below best prefix for non-prefix competitor.",
    )

    # Beam search parameters
    beam_width_multiplier: int = Field(
        default=3,
        ge=1,
        description="Multiplied by top_k for beam width in multi-token decoding.",
    )
    beam_width_cap: int = Field(
        default=20,
        ge=1,
        description="Maximum beam width to prevent combinatorial explosion.",
    )
    beam_dedup_multiplier: int = Field(
        default=2,
        ge=1,
        description="Extra candidates to generate for deduplication in beam search.",
    )

    # Multi-token decoding
    multi_token_candidate_multiplier: int = Field(
        default=2,
        ge=1,
        description="Extra candidates per masked position in multi-token decoding.",
    )

    # Sentence scanning constants
    scan_presence_top_n: int = Field(
        default=5,
        ge=1,
        description="Top-N predictions to check for word presence in scanning.",
    )
    scan_suggestion_pool_size: int = Field(
        default=10,
        ge=1,
        description="How many predictions to consider for suggestions in scanning.",
    )
    scan_max_suggestions: int = Field(
        default=5,
        ge=1,
        description="Maximum suggestions to return per error in scanning.",
    )
    scan_min_suggestion_len: int = Field(
        default=2,
        ge=1,
        description="Minimum character length for a valid suggestion.",
    )
    scan_min_myanmar_char_ratio: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum Myanmar character ratio for valid suggestions (>50%).",
    )
    scan_confidence_base_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Base weight in similarity-boosted confidence formula.",
    )
    scan_confidence_similarity_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Similarity weight in confidence formula.",
    )

    # Score candidates parameters
    score_floor_offset: float = Field(
        default=1.0,
        ge=0.0,
        description="Subtracted from min logit for absent candidates in score_candidates.",
    )
    score_default_top_k: int = Field(
        default=200,
        ge=1,
        description="Default top_k for score_candidates predictions.",
    )

    # Error detection parameters
    error_check_top_n: int = Field(
        default=5,
        ge=1,
        description="Top-N predictions for prefix evidence in is_semantic_error.",
    )
    short_word_max_len: int = Field(
        default=2,
        ge=1,
        description="Words at or below this length are considered 'short' for special handling.",
    )

    @model_validator(mode="after")
    def validate_model_config(self) -> "SemanticConfig":
        """Validate model configuration consistency.

        Ensures that either path-based OR instance-based configuration is used,
        not both. Also validates that model and tokenizer are provided together.
        """
        has_paths = self.model_path is not None or self.tokenizer_path is not None
        has_instances = self.model is not None or self.tokenizer is not None

        # XOR validation: Can't mix path-based and instance-based configuration
        if has_paths and has_instances:
            raise ValueError(
                "Cannot mix path-based (model_path/tokenizer_path) and "
                "instance-based (model/tokenizer) configuration. Use one or the other."
            )

        # If using paths, both must be provided
        if self.model_path is not None and self.tokenizer_path is None:
            raise ValueError("tokenizer_path must be provided when model_path is specified")
        if self.tokenizer_path is not None and self.model_path is None:
            raise ValueError("model_path must be provided when tokenizer_path is specified")

        # If using instances, both must be provided
        if self.model is not None and self.tokenizer is None:
            raise ValueError("tokenizer instance must be provided when model instance is specified")
        if self.tokenizer is not None and self.model is None:
            raise ValueError("model instance must be provided when tokenizer instance is specified")

        return self


class RankerConfig(BaseModel):
    """
    Configuration for suggestion ranking strategies.

    Controls how spelling correction suggestions are scored and ranked based on
    edit distance, frequency, phonetic similarity, syllable structure, and source.

    The unified suggestion pipeline always uses UnifiedRanker at the composite level.
    The base scoring strategy within UnifiedRanker is controlled by `unified_base_ranker_type`.

    Attributes:
        unified_base_ranker_type: Base ranker type for UnifiedRanker (default: "default").
            - "default": Balances edit distance and frequency
            - "frequency_first": Prioritizes high-frequency words
            - "phonetic_first": Prioritizes phonetic similarity
            - "edit_distance_only": Pure edit distance ranking
        frequency_denominator: Denominator for frequency bonus calculation (default: 10000.0).
            Higher values reduce the impact of very high frequencies.
        phonetic_bonus_weight: Weight for phonetic similarity bonus (default: 0.4).
        syllable_bonus_weight: Weight for syllable-aware scoring (default: 0.3).
        nasal_bonus_weight: Weight for nasal variant boosting (default: 0.15).
        same_nasal_bonus_weight: Weight for same nasal ending bonus (default: 0.25).
        frequency_first_edit_weight: Edit distance weight in FrequencyFirstRanker (default: 0.5).
        frequency_first_scale: Frequency bonus scale in FrequencyFirstRanker (default: 0.1).
        phonetic_first_weight: Weight for phonetic similarity in PhoneticFirstRanker (default: 1.0).
        phonetic_first_edit_weight: Weight for edit distance in PhoneticFirstRanker (default: 0.3).
        source_weight_particle_typo: Weight for particle_typo source (default: 1.2).
        source_weight_medial_confusion: Weight for medial_confusion source (default: 1.1).
        source_weight_semantic: Weight for semantic source (default: 1.15).
        source_weight_symspell: Weight for symspell source (default: 1.0).
        source_weight_morphology: Weight for morphology source (default: 0.9).
        source_weight_question_structure: Weight for question_structure source (default: 1.0).
        source_weight_pos_sequence: Weight for pos_sequence source (default: 0.85).
        source_weight_context: Weight for context-enhanced source (default: 1.15).
        source_weight_compound: Weight for compound source (default: 0.95).
        strategy_score_weight: Weight for blending strategy_score in UnifiedRanker (default: 0.5).
            Balances strategy-level scores with feature-based scoring.
        context_strategy_score_weight: Override weight for `context` source strategy scores
            in UnifiedRanker (default: 0.7). Allows stronger context-aware reranking without
            changing non-context sources.
        strategy_score_cap: Cap for linear strategy score normalization in UnifiedRanker
            (default: 10.0). Prevents score saturation from large raw strategy values.
        enable_targeted_rerank_hints: Enable targeted top-1 rerank hint rules in SpellChecker
            (default: True). Disable for ablation/generalization checks.
        enable_targeted_candidate_injections: Enable targeted missing-candidate injections
            in SpellChecker rerank path (default: True). Disable for ablation checks.
        enable_targeted_grammar_completion_templates: Enable targeted grammar completion
            templates in grammar/pattern helpers (default: True). Disable to fall back
            to generic completions.
        score_tie_precision: Decimal precision for score ties (default: 6).
    """

    model_config = ConfigDict(
        validate_assignment=True,
        frozen=False,
        extra="forbid",
    )

    # Base ranker selection for UnifiedRanker
    unified_base_ranker_type: Literal[
        "default", "frequency_first", "phonetic_first", "edit_distance_only"
    ] = Field(
        default="default",
        description=(
            "Base ranker type used within UnifiedRanker for the composite pipeline. "
            "Controls how base scores are computed before source weighting is applied. "
            "Options: 'default' (balanced), 'frequency_first', 'phonetic_first', "
            "'edit_distance_only'. Note: 'unified' is not valid here to avoid recursion."
        ),
    )

    # DefaultRanker settings
    frequency_denominator: float = Field(
        default=10000.0,
        gt=0.0,
        description="Denominator for frequency bonus calculation",
    )
    phonetic_bonus_weight: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Weight for phonetic similarity bonus (0.0-1.0)",
    )
    syllable_bonus_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight for syllable-aware scoring (0.0-1.0)",
    )
    nasal_bonus_weight: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Weight for nasal variant boosting (0.0-1.0)",
    )
    same_nasal_bonus_weight: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Weight for same nasal ending bonus (0.0-1.0)",
    )
    pos_bonus_weight: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Weight for POS bigram fit bonus in ranking (0.0-1.0)",
    )
    plausibility_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum weighted_distance/edit_distance ratio for multiplicative "
            "plausibility gating. Ratios below this activate Myanmar-specific "
            "plausibility multiplier (medial swap=0.3, aspiration=0.4, nasal=0.5). "
            "Ratios at or above stay at 1.0 to avoid over-promoting candidates "
            "with only marginally reduced substitution costs."
        ),
    )
    plausibility_floor: float = Field(
        default=0.2,
        ge=0.1,
        le=1.0,
        description=(
            "Minimum plausibility multiplier for Myanmar error patterns (0.1-1.0). "
            "Prevents over-correction for very low weighted_distance values."
        ),
    )
    # Deprecated: kept for backward compatibility, no longer used by DefaultRanker
    syllable_bonus_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Deprecated: subsumed by multiplicative plausibility",
    )
    weighted_distance_bonus_weight: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Deprecated: subsumed by multiplicative plausibility",
    )

    # FrequencyFirstRanker settings
    frequency_first_edit_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Weight for edit distance in frequency-first ranking",
    )
    frequency_first_scale: float = Field(
        default=0.1,
        gt=0.0,
        description="Scale factor for frequency bonus in frequency-first ranking",
    )

    # PhoneticFirstRanker settings
    phonetic_first_weight: float = Field(
        default=1.0,
        ge=0.0,
        description="Weight for phonetic similarity in phonetic-first ranking",
    )
    phonetic_first_edit_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight for edit distance in phonetic-first ranking",
    )

    # UnifiedRanker source weights
    source_weight_particle_typo: float = Field(
        default=1.2,
        gt=0.0,
        description="Weight for particle_typo source (rule-based patterns)",
    )
    source_weight_medial_confusion: float = Field(
        default=1.1,
        gt=0.0,
        description="Weight for medial_confusion source (context-aware)",
    )
    source_weight_semantic: float = Field(
        default=1.15,
        gt=0.0,
        description="Weight for semantic source (AI-powered validation)",
    )
    source_weight_symspell: float = Field(
        default=1.0,
        gt=0.0,
        description="Weight for symspell source (statistical baseline)",
    )
    source_weight_morphology: float = Field(
        default=0.9,
        gt=0.0,
        description="Weight for morphology source (OOV recovery)",
    )
    source_weight_question_structure: float = Field(
        default=1.0,
        gt=0.0,
        description="Weight for question_structure source",
    )
    source_weight_pos_sequence: float = Field(
        default=0.85,
        gt=0.0,
        description="Weight for pos_sequence source",
    )
    source_weight_context: float = Field(
        default=1.15,
        gt=0.0,
        description="Weight for context-enhanced source (context-aware re-ranking)",
    )
    source_weight_compound: float = Field(
        default=0.95,
        gt=0.0,
        description="Weight for compound source (word splitting/joining)",
    )
    source_weight_morpheme: float = Field(
        default=0.85,
        gt=0.0,
        description="Weight for morpheme source (morpheme-level correction in compounds)",
    )
    source_weight_medial_swap: float = Field(
        default=1.0,
        gt=0.0,
        description="Weight for medial_swap source (ျ↔ြ, ှ insertion, rule-based)",
    )

    # Strategy score blending (for UnifiedRanker)
    strategy_score_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Weight for blending strategy_score with feature-based score (0.0-1.0). "
            "At 0.0, only feature-based scoring is used. "
            "At 1.0, only strategy_score is used. "
            "Default 0.5 provides balanced blend of both scoring methods."
        ),
    )
    context_strategy_score_weight: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description=(
            "Override blend weight for strategy_score when suggestion source is 'context'. "
            "Higher values increase reliance on context strategy scoring for context-reranked "
            "candidates while leaving non-context sources on strategy_score_weight."
        ),
    )
    strategy_score_cap: float = Field(
        default=10.0,
        gt=0.0,
        description=(
            "Upper cap used to linearly normalize strategy_score in UnifiedRanker: "
            "normalized = min(max(strategy_score, 0), strategy_score_cap) / strategy_score_cap. "
            "Avoids sigmoid saturation for large context strategy scores."
        ),
    )
    enable_targeted_rerank_hints: bool = Field(
        default=True,
        description=(
            "Enable targeted top-1 rerank hint rules in SpellChecker suggestion reranking. "
            "Turn off for ablation/generalization experiments."
        ),
    )
    enable_targeted_candidate_injections: bool = Field(
        default=True,
        description=(
            "Enable targeted candidate-injection rules (rank-1 insertions) in SpellChecker "
            "reranking. Turn off for ablation runs."
        ),
    )
    enable_targeted_grammar_completion_templates: bool = Field(
        default=True,
        description=(
            "Enable targeted grammar completion templates in grammar/pattern helpers. "
            "When disabled, grammar completion falls back to generic suggestions."
        ),
    )

    # UnifiedRanker normalization parameters
    unified_max_edit_distance: float = Field(
        default=5.0,
        gt=0.0,
        description=(
            "Maximum expected edit distance for score normalization in UnifiedRanker. "
            "Used to normalize raw scores to [0, 1] range via sigmoid."
        ),
    )
    unified_weight_scale: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description=(
            "Scale factor for source/confidence bonus in UnifiedRanker. "
            "Controls how much source weight and confidence affect the final score."
        ),
    )

    # Score tie-breaking
    score_tie_precision: int = Field(
        default=6,
        ge=0,
        le=15,
        description=(
            "Decimal precision for score tie-breaking in "
            "UnifiedRanker. Scores within this rounding precision "
            "are treated as near-equal and resolved "
            "deterministically using lexical features."
        ),
    )

    # Near-duplicate filtering
    similarity_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description=(
            "Threshold for near-duplicate detection in suggestion ranking. "
            "Suggestions with similarity >= this threshold are filtered for diversity."
        ),
    )

    # DefaultRanker span-length bonus parameters
    freq_bonus_ceiling: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Asymptotic ceiling for frequency bonus.",
    )
    long_error_threshold: int = Field(
        default=6,
        ge=1,
        description="Error length at or above which long-error span bonuses apply.",
    )
    span_cap: int = Field(
        default=10,
        ge=1,
        description="Cap on error length for span bonus scaling.",
    )
    long_exact_base: float = Field(
        default=0.4,
        description="Base span bonus for long exact-length match.",
    )
    long_exact_scale: float = Field(
        default=0.1,
        description="Per-char scaling for long exact-length match.",
    )
    long_close_base: float = Field(
        default=0.2,
        description="Base span bonus for long close-length match (diff<=2).",
    )
    long_close_scale: float = Field(
        default=0.05,
        description="Per-char scaling for long close-length match.",
    )
    long_medium_bonus: float = Field(
        default=0.1,
        description="Span bonus for long medium-length match (diff<=4).",
    )
    long_far_penalty: float = Field(
        default=-0.15,
        description="Penalty multiplier for long far-length match (diff>4).",
    )
    short_exact_bonus: float = Field(
        default=0.4,
        description="Span bonus for short exact-length match.",
    )
    short_close_bonus: float = Field(
        default=0.2,
        description="Span bonus for short ±1-length match.",
    )
    short_medium_bonus: float = Field(
        default=0.1,
        description="Span bonus for short ±2-length match.",
    )
    score_to_confidence_divisor: float = Field(
        default=10.0,
        gt=0.0,
        description="Divisor for converting SymSpell score to confidence.",
    )

    def get_source_weights(self) -> dict[str, float]:
        """Get all source weights as a dictionary for UnifiedRanker."""
        return {
            "particle_typo": self.source_weight_particle_typo,
            "medial_confusion": self.source_weight_medial_confusion,
            "semantic": self.source_weight_semantic,
            "symspell": self.source_weight_symspell,
            "morphology": self.source_weight_morphology,
            "question_structure": self.source_weight_question_structure,
            "pos_sequence": self.source_weight_pos_sequence,
            "context": self.source_weight_context,
            "compound": self.source_weight_compound,
            "morpheme": self.source_weight_morpheme,
            "medial_swap": self.source_weight_medial_swap,
        }


class MorphologyConfig(BaseModel):
    """
    Configuration for morphological analysis and POS guessing.

    Controls confidence values used in OOV (Out-of-Vocabulary) recovery
    when analyzing word morphology (root extraction and suffix stripping),
    as well as suffix-based POS guessing confidence multipliers.

    Attributes:
        particle_confidence_boost: Multiplier for particle suffix confidence
            (default: 1.2). Particles are usually reliable indicators.
        particle_confidence_cap: Maximum confidence after particle boost
            (default: 1.0).
        verb_suffix_weight: Weight for verb suffix confidence scoring
            (default: 0.9). Slightly lower than particles.
        noun_suffix_weight: Weight for noun suffix confidence scoring
            (default: 0.85).
        adverb_suffix_weight: Weight for adverb suffix confidence scoring
            (default: 0.8).
        oov_base_confidence: Base confidence for OOV suffix analysis (default: 0.3).
        oov_scale_factor: Scale factor for suffix ratio contribution (default: 0.7).
        oov_cap: Maximum confidence from suffix analysis alone (default: 0.95).
        dictionary_boost: Confidence boost when dictionary confirms root (default: 0.2).
        dictionary_cap: Maximum confidence after dictionary boost (default: 0.98).
        fallback_with_dict: Confidence when no suffixes found but root in dict (default: 0.5).
        fallback_without_dict: Confidence when no suffixes found and root is unknown
            (default: 0.2).
    """

    model_config = ConfigDict(
        validate_assignment=True,
        frozen=False,
        extra="forbid",
    )

    # -- POS guessing confidence multipliers --

    particle_confidence_boost: float = Field(
        default=1.2,
        ge=0.0,
        le=5.0,
        description="Multiplier for particle suffix confidence",
    )
    particle_confidence_cap: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Maximum confidence after particle boost",
    )
    verb_suffix_weight: float = Field(
        default=0.9,
        ge=0.0,
        le=2.0,
        description="Weight for verb suffix confidence scoring",
    )
    noun_suffix_weight: float = Field(
        default=0.85,
        ge=0.0,
        le=2.0,
        description="Weight for noun suffix confidence scoring",
    )
    adverb_suffix_weight: float = Field(
        default=0.8,
        ge=0.0,
        le=2.0,
        description="Weight for adverb suffix confidence scoring",
    )

    # -- OOV recovery confidence --

    oov_base_confidence: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Base confidence for OOV suffix analysis",
    )
    oov_scale_factor: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Scale factor for suffix ratio contribution to confidence",
    )
    oov_cap: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Maximum confidence from suffix analysis alone",
    )
    dictionary_boost: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Confidence boost when dictionary confirms the extracted root",
    )
    dictionary_cap: float = Field(
        default=0.98,
        ge=0.0,
        le=1.0,
        description="Maximum confidence after dictionary boost",
    )
    fallback_with_dict: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence when no suffixes found but root is in dictionary",
    )
    fallback_without_dict: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Confidence when no suffixes found and root is unknown",
    )
    max_suffix_strip_iterations: int = Field(
        default=5,
        ge=1,
        description=(
            "Maximum iterations for suffix stripping in analyze_word(). "
            "Prevents infinite loops on pathological inputs."
        ),
    )


class AlgorithmCacheConfig(BaseModel):
    """
    Unified cache configuration for algorithm caching layers.

    Consolidates cache sizes from:
    - algorithms/cache.py: CachedDictionaryLookup, CachedBigramSource, etc.
    - algorithms/factory.py: Component factory cache configurations
    - algorithms/semantic_checker.py: Encoding and alignment caches
    - algorithms/joint_segment_tagger.py: Word score, transition, emission caches

    This class provides a single source of truth for cache configuration.

    Attributes:
        syllable_cache_size: LRU cache size for syllable lookups (default: 4096).
        word_cache_size: LRU cache size for word lookups (default: 8192).
        frequency_cache_size: LRU cache size for frequency lookups (default: 8192).
        bigram_cache_size: LRU cache size for bigram probability lookups (default: 16384).
        trigram_cache_size: LRU cache size for trigram probability lookups (default: 16384).
        semantic_encoding_cache_size: Semantic checker encoding cache (default: 512).
        semantic_alignment_cache_size: Semantic checker alignment cache (default: 256).
        joint_word_score_cache_size: Joint tagger word scores (default: 8192).
        joint_transition_cache_size: Joint tagger POS transitions (default: 2048).
        joint_emission_cache_size: Joint tagger emissions (default: 4096).
        joint_valid_tags_cache_size: Joint tagger valid tags (default: 4096).

    Memory Usage (approximate):
        - ~1KB per cached entry (average)
        - Default configuration: ~50MB total cache memory

    Example:
        >>> from myspellchecker.core.config import SpellCheckerConfig, AlgorithmCacheConfig
        >>>
        >>> # Reduce cache sizes for memory-constrained environments
        >>> config = SpellCheckerConfig(
        ...     cache=AlgorithmCacheConfig(
        ...         syllable_cache_size=1024,
        ...         word_cache_size=2048,
        ...         bigram_cache_size=4096,
        ...     )
        ... )
        >>>
        >>> # Increase cache sizes for high-throughput applications
        >>> config = SpellCheckerConfig(
        ...     cache=AlgorithmCacheConfig(
        ...         syllable_cache_size=16384,
        ...         word_cache_size=32768,
        ...         bigram_cache_size=65536,
        ...     )
        ... )
    """

    model_config = ConfigDict(
        validate_assignment=True,
        frozen=False,
        extra="forbid",
    )

    syllable_cache_size: int = Field(
        default=4096,
        ge=0,
        description="LRU cache size for syllable validation and frequency lookups",
    )
    word_cache_size: int = Field(
        default=8192,
        ge=0,
        description="LRU cache size for word validation and frequency lookups",
    )
    frequency_cache_size: int = Field(
        default=8192,
        ge=0,
        description="LRU cache size for frequency source lookups",
    )
    bigram_cache_size: int = Field(
        default=16384,
        ge=0,
        description="LRU cache size for bigram probability lookups",
    )
    trigram_cache_size: int = Field(
        default=16384,
        ge=0,
        description="LRU cache size for trigram probability lookups",
    )
    # SemanticChecker cache sizes
    semantic_encoding_cache_size: int = Field(
        default=512,
        ge=0,
        description="LRU cache size for semantic checker tokenization encoding results",
    )
    semantic_alignment_cache_size: int = Field(
        default=256,
        ge=0,
        description="LRU cache size for semantic checker word-token alignment results",
    )

    # JointSegmentTagger cache sizes
    joint_word_score_cache_size: int = Field(
        default=8192,
        ge=0,
        description="LRU cache size for joint tagger word bigram scores",
    )
    joint_transition_cache_size: int = Field(
        default=2048,
        ge=0,
        description="LRU cache size for joint tagger POS trigram transitions",
    )
    joint_emission_cache_size: int = Field(
        default=4096,
        ge=0,
        description="LRU cache size for joint tagger word-tag emission scores",
    )
    joint_valid_tags_cache_size: int = Field(
        default=4096,
        ge=0,
        description="LRU cache size for joint tagger valid tags per word",
    )


class CompoundResolverConfig(BaseModel):
    """Configuration for compound word resolution.

    Controls the dynamic-programming compound splitter that validates OOV words
    by segmenting them into known dictionary morphemes with valid POS patterns.

    Attributes:
        min_morpheme_frequency: Minimum frequency for each morpheme (default: 10).
        max_parts: Maximum number of parts in a compound (default: 4).
        cache_size: Maximum cache entries (default: 1024).
        parts_penalty_multiplier: Penalty multiplier per extra split beyond
            the first in DP scoring (default: 2.0).
        base_confidence: Base confidence score for compound splits (default: 0.85).
        high_freq_boost: Confidence boost when min morpheme freq >= high_freq_threshold
            (default: 0.05).
        high_freq_threshold: Frequency threshold for high_freq_boost (default: 100).
        medium_freq_boost: Confidence boost when min morpheme freq >= medium_freq_threshold
            (default: 0.03).
        medium_freq_threshold: Frequency threshold for medium_freq_boost (default: 50).
        extra_parts_penalty: Confidence penalty per extra part beyond 2 (default: 0.05).
    """

    model_config = ConfigDict(
        validate_assignment=True,
        frozen=False,
        extra="forbid",
    )

    min_morpheme_frequency: int = Field(
        default=10,
        ge=0,
        description="Minimum frequency for each morpheme to be considered valid",
    )
    max_parts: int = Field(
        default=4,
        ge=2,
        description="Maximum number of parts allowed in compound word splitting",
    )
    cache_size: int = Field(
        default=1024,
        ge=0,
        description="Maximum cache entries for compound resolution results",
    )
    parts_penalty_multiplier: float = Field(
        default=2.0,
        ge=0.0,
        description=(
            "Penalty multiplier per extra split beyond the first in DP scoring. "
            "Higher values prefer fewer parts."
        ),
    )
    base_confidence: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Base confidence score for compound splits",
    )
    high_freq_boost: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Confidence boost when min morpheme freq >= high_freq_threshold",
    )
    high_freq_threshold: int = Field(
        default=100,
        ge=0,
        description="Frequency threshold for high_freq_boost",
    )
    medium_freq_boost: float = Field(
        default=0.03,
        ge=0.0,
        le=1.0,
        description="Confidence boost when min morpheme freq >= medium_freq_threshold",
    )
    medium_freq_threshold: int = Field(
        default=50,
        ge=0,
        description="Frequency threshold for medium_freq_boost",
    )
    extra_parts_penalty: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Confidence penalty per extra part beyond 2",
    )
    confidence_cap: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Maximum confidence after high-frequency boost.",
    )
    confidence_cap_mid: float = Field(
        default=0.93,
        ge=0.0,
        le=1.0,
        description="Maximum confidence after medium-frequency boost.",
    )


class ReduplicationConfig(BaseModel):
    """Configuration for reduplication detection.

    Controls the reduplication engine that validates OOV words formed by
    reduplicating known dictionary words (AA, AABB, ABAB, RHYME patterns).

    Attributes:
        min_base_frequency: Minimum frequency for the base word (default: 5).
        cache_size: Maximum cache entries (default: 1024).
        pattern_confidence_ab: Base confidence for AB (simple doubling) pattern
            (default: 0.90).
        pattern_confidence_aabb: Base confidence for AABB (syllable doubling) pattern
            (default: 0.85).
        pattern_confidence_abab: Base confidence for ABAB (word repeating) pattern
            (default: 0.85).
        pattern_confidence_rhyme: Base confidence for RHYME pattern (default: 0.95).
        pattern_confidence_default: Default base confidence for unknown patterns
            (default: 0.80).
        high_freq_boost: Confidence boost when base freq >= high_freq_threshold
            (default: 0.05).
        high_freq_threshold: Frequency threshold for high_freq_boost (default: 100).
        high_freq_cap: Maximum confidence after high_freq_boost (default: 0.98).
        medium_freq_boost: Confidence boost when base freq >= medium_freq_threshold
            (default: 0.03).
        medium_freq_threshold: Frequency threshold for medium_freq_boost (default: 50).
        medium_freq_cap: Maximum confidence after medium_freq_boost (default: 0.95).
    """

    model_config = ConfigDict(
        validate_assignment=True,
        frozen=False,
        extra="forbid",
    )

    min_base_frequency: int = Field(
        default=5,
        ge=0,
        description="Minimum frequency for the base word to be considered valid",
    )
    cache_size: int = Field(
        default=1024,
        ge=0,
        description="Maximum cache entries for reduplication analysis results",
    )
    pattern_confidence_ab: float = Field(
        default=0.90,
        ge=0.0,
        le=1.0,
        description="Base confidence for AB (simple doubling) pattern",
    )
    pattern_confidence_aabb: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Base confidence for AABB (syllable doubling) pattern",
    )
    pattern_confidence_abab: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Base confidence for ABAB (word repeating) pattern",
    )
    pattern_confidence_rhyme: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Base confidence for RHYME pattern",
    )
    pattern_confidence_default: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        description="Default base confidence for unknown patterns",
    )
    high_freq_boost: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Confidence boost when base freq >= high_freq_threshold",
    )
    high_freq_threshold: int = Field(
        default=100,
        ge=0,
        description="Frequency threshold for high_freq_boost",
    )
    high_freq_cap: float = Field(
        default=0.98,
        ge=0.0,
        le=1.0,
        description="Maximum confidence after high_freq_boost",
    )
    medium_freq_boost: float = Field(
        default=0.03,
        ge=0.0,
        le=1.0,
        description="Confidence boost when base freq >= medium_freq_threshold",
    )
    medium_freq_threshold: int = Field(
        default=50,
        ge=0,
        description="Frequency threshold for medium_freq_boost",
    )
    medium_freq_cap: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Maximum confidence after medium_freq_boost",
    )


class ResourceConfig(BaseModel):
    """
    Configuration for HuggingFace resource loading.

    Controls the download and caching of tokenization resources
    (segmentation model, CRF model, curated lexicon) from HuggingFace.

    Attributes:
        resource_version: Resource version tag on HuggingFace (default: "main").
            Bump with releases for reproducibility.
        hf_repo_base: Base URL for the HuggingFace dataset repository.
        cache_dir: Local cache directory for downloaded resources.
            Defaults to ~/.cache/myspellchecker/resources.
            Can be overridden with MYSPELL_CACHE_DIR environment variable.
    """

    model_config = ConfigDict(
        validate_assignment=True,
        frozen=False,
        extra="forbid",
    )

    resource_version: str = Field(
        default="main",
        description=(
            "Resource version tag on HuggingFace. Use 'main' until a versioned tag is created."
        ),
    )
    hf_repo_base: str = Field(
        default=("https://huggingface.co/datasets/thettwe/myspellchecker-resources/resolve"),
        description=("Base URL for the HuggingFace dataset repository (without version suffix)."),
    )
    cache_dir: str | None = Field(
        default=None,
        description=(
            "Local cache directory for downloaded resources. "
            "Defaults to ~/.cache/myspellchecker/resources. "
            "Can be overridden with MYSPELL_CACHE_DIR env var."
        ),
    )

    @property
    def hf_repo_url(self) -> str:
        """Full HuggingFace repository URL with version."""
        return f"{self.hf_repo_base}/{self.resource_version}"


class TransformerSegmenterConfig(BaseModel):
    """
    Configuration for transformer-based word segmenter.

    Controls the HuggingFace token classification model used for
    Myanmar word boundary detection via B/I labeling.

    Attributes:
        model_name: HuggingFace model ID or local path.
        device: Device for inference (-1=CPU, 0+=GPU index).
        batch_size: Batch size for batch segmentation.
        max_length: Maximum sequence length for the model.
    """

    model_config = ConfigDict(
        validate_assignment=True,
        frozen=False,
        extra="forbid",
    )

    model_name: str = Field(
        default="chuuhtetnaing/myanmar-text-segmentation-model",
        description=("HuggingFace model ID or local path for word segmentation."),
    )
    device: int = Field(
        default=-1,
        ge=-1,
        description=("Device for inference. -1 for CPU, 0+ for GPU index."),
    )
    batch_size: int = Field(
        default=32,
        ge=1,
        description="Batch size for batch segmentation.",
    )
    max_length: int = Field(
        default=512,
        ge=1,
        description="Maximum sequence length for the model.",
    )


class FrequencyGuardConfig(BaseModel):
    """
    Centralized frequency thresholds used across validators and strategies.

    These guards prevent false positives on common words by requiring
    stronger evidence before flagging high-frequency entries as errors.
    Each threshold was previously hardcoded in the consumer file noted
    in its description.

    Attributes:
        colloquial_high_freq_suppression: Frequency above which colloquial
            variant informational notes are suppressed in lenient mode.
            Used by SyllableValidator and WordValidator.
        homophone_high_freq: Frequency above which a stricter improvement
            ratio is required for homophone detection.
        homophone_high_freq_ratio: Improvement ratio required for words
            above ``homophone_high_freq``.
        ngram_high_freq_guard: Frequency above which n-gram false positives
            are suppressed when no better suggestion exists.
        semantic_high_freq_protection: Frequency above which the high-freq
            logit diff threshold applies in confusable semantic detection.
    """

    model_config = ConfigDict(
        validate_assignment=True,
        frozen=False,
        extra="forbid",
    )

    # SyllableValidator + WordValidator: suppress colloquial info for
    # very high-frequency words in lenient mode
    colloquial_high_freq_suppression: int = Field(
        default=100_000,
        ge=0,
        description=(
            "Frequency above which colloquial variant informational "
            "notes are suppressed in lenient mode. "
            "Used by SyllableValidator and WordValidator."
        ),
    )
    # HomophoneValidationStrategy: frequency guard
    homophone_high_freq: int = Field(
        default=1_000,
        ge=0,
        description=(
            "Word frequency above which a stricter improvement ratio "
            "is required for homophone detection."
        ),
    )
    homophone_high_freq_ratio: float = Field(
        default=50.0,
        ge=1.0,
        description=(
            "Improvement ratio required for words above "
            "homophone_high_freq. Much stricter than the default "
            "5x to avoid flagging common words."
        ),
    )
    # NgramContextValidationStrategy: suppress FP on common words
    ngram_high_freq_guard: int = Field(
        default=5_000,
        ge=0,
        description=(
            "Minimum dictionary frequency to suppress n-gram FP on "
            "common words when no better suggestion exists."
        ),
    )
    # ConfusableSemanticStrategy: protect common words from MLM FP
    semantic_high_freq_protection: int = Field(
        default=50_000,
        ge=0,
        description=(
            "Word frequency above which the high-freq logit diff "
            "threshold applies in confusable semantic detection."
        ),
    )


class ConfusableSemanticConfig(BaseModel):
    """
    Configuration for confusable semantic validation strategy (priority 48).

    Controls MLM-enhanced confusable variant detection thresholds.
    Uses masked language modeling to detect valid-word confusables that pass
    all other validation because both the current word and its confusable
    variant are legitimate dictionary words.

    Asymmetric thresholds protect against false positives:
    - Default: logit_diff >= 3.0 (~20x probability ratio)
    - Medial swap: logit_diff >= 2.0 (highest signal)
    - Current in top-K: logit_diff >= 5.0 (model already considers it)
    - High-frequency: logit_diff >= 6.0 (protect common words)
    """

    model_config = ConfigDict(
        validate_assignment=True,
        frozen=False,
        extra="forbid",
    )

    logit_diff_threshold: float = Field(
        default=3.0,
        ge=0.0,
        description=(
            "Default logit diff threshold (~20x probability ratio). "
            "Variant must score this much higher than current word."
        ),
    )
    logit_diff_threshold_medial: float = Field(
        default=2.0,
        ge=0.0,
        description=(
            "Logit diff threshold for medial swap (~7x ratio). "
            "Lower because these are highest-signal confusables."
        ),
    )
    logit_diff_threshold_current_in_topk: float = Field(
        default=5.0,
        ge=0.0,
        description=(
            "Logit diff threshold when current word IS in top-K "
            "(~148x ratio). Need stronger evidence when model "
            "already considers current word."
        ),
    )
    high_freq_threshold: int = Field(
        default=50000,
        ge=0,
        description=("Word frequency above which the high-freq logit diff applies."),
    )
    high_freq_logit_diff: float = Field(
        default=6.0,
        ge=0.0,
        description=(
            "Logit diff threshold for high-frequency words (~403x "
            "ratio). Protects common particles/words against FPs."
        ),
    )
    min_word_length: int = Field(
        default=2,
        ge=1,
        description="Minimum word length for confusable checking.",
    )
    freq_ratio_penalty_high: float = Field(
        default=2.0,
        ge=0.0,
        description=(
            "Extra threshold penalty when variant_freq/word_freq > "
            "5.0. Counters MLM frequency bias. "
            "(Conservative: 3.0)"
        ),
    )
    freq_ratio_penalty_mid: float = Field(
        default=1.0,
        ge=0.0,
        description=(
            "Extra threshold penalty when variant_freq/word_freq > "
            "2.0. Moderate guard against frequency bias. "
            "(Conservative: 1.5)"
        ),
    )
    visarga_penalty: float = Field(
        default=2.0,
        ge=0.0,
        description=(
            "Extra threshold penalty when word and variant differ "
            "only by visarga. Visarga pairs are almost always "
            "different morphemes in Myanmar."
        ),
    )
    sentence_final_penalty: float = Field(
        default=0.5,
        ge=0.0,
        description=(
            "Extra threshold penalty for sentence-final words. "
            "MLM has no right context at sentence boundaries. "
            "(Conservative: 1.0)"
        ),
    )
    logit_diff_threshold_homophone: float = Field(
        default=3.5,
        ge=0.0,
        description=(
            "Logit diff threshold for known homophone pairs from "
            "homophones.yaml. Dedicated threshold that skips "
            "freq_ratio and sentence_final penalties."
        ),
    )
    max_threshold: float = Field(
        default=8.0,
        ge=0.0,
        description=(
            "Maximum allowed stacked threshold. Caps the total "
            "threshold after all penalties are applied. Prevents "
            "impossible thresholds that silence the model. "
            "Set to 0 to disable capping. (Conservative: 0 / no cap)"
        ),
    )
    reverse_ratio_min_freq: int = Field(
        default=50000,
        ge=0,
        description=(
            "Only apply reverse_ratio penalty when word_freq >= "
            "this value. Low-freq misspellings with high reverse "
            "ratio are real errors, not common words needing "
            "protection. (Conservative: 0 / always apply)"
        ),
    )
    visarga_high_freq_hard_block: bool = Field(
        default=True,
        description=(
            "When True, hard-block (inf threshold) visarga pairs "
            "where both word and variant are high-frequency. "
            "When False, use high_freq_logit_diff + visarga_penalty "
            "as a high but reachable threshold. "
            "Keep True — ပြီ/ပြီး FP proves the block is needed."
        ),
    )
    curated_logit_diff_threshold: float = Field(
        default=2.0,
        ge=0.0,
        description=(
            "Logit diff threshold for curated confusable pairs "
            "(from confusable_pairs.yaml). These are known "
            "real-word confusion patterns that bypass the "
            "standard threshold stacking. Lower than default "
            "because the pairs are linguistically verified."
        ),
    )
    near_synonym_logit_diff_threshold: float = Field(
        default=3.0,
        ge=0.0,
        description=(
            "Logit diff threshold for near-synonym confusable "
            "pairs (confusion_type=near_synonym in "
            "confusable_pairs.yaml). Higher than the curated "
            "threshold because near-synonyms share semantic "
            "overlap and both words are often high-frequency, "
            "requiring stronger MLM evidence to avoid FPs. "
            "~20x probability ratio."
        ),
    )
    explicit_non_topk_penalty: float = Field(
        default=5.0,
        ge=0.0,
        description=(
            "Extra threshold penalty when a generated variant is "
            "NOT in the MLM top-K predictions and is NOT a known "
            "homophone. Requires much stronger logit evidence "
            "because the model did not consider the variant likely."
        ),
    )
    explicit_non_topk_homophone_penalty: float = Field(
        default=1.0,
        ge=0.0,
        description=(
            "Extra threshold penalty when a generated variant is "
            "NOT in the MLM top-K predictions but IS a known "
            "homophone (or curated/near-synonym pair). Smaller "
            "than the general non-top-K penalty because homophones "
            "are linguistically plausible confusables."
        ),
    )
    non_boundary_penalty: float = Field(
        default=1.0,
        ge=0.0,
        description=(
            "Extra threshold penalty for confusable errors that "
            "do NOT occur at a morpheme/word boundary (medial or "
            "initial position). Non-boundary confusables are less "
            "likely to be genuine errors."
        ),
    )

    # --- Error budget and adjacency dampening ---

    error_budget_threshold: int = Field(
        default=1,
        ge=0,
        description="Skip proactive scanning when existing errors reach this count.",
    )
    adjacency_window: int = Field(
        default=3,
        ge=0,
        description="Word distance within which adjacency dampening applies.",
    )
    adjacency_penalty_base: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Base confidence penalty for adjacent confusable errors.",
    )
    adjacency_penalty_per_word: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Additional penalty per word of distance within adjacency window.",
    )

    # --- Frequency ratio comparison cutoffs ---

    freq_ratio_high_cutoff: float = Field(
        default=5.0,
        gt=0.0,
        description=(
            "Frequency ratio (variant_freq / word_freq) above which "
            "the high freq_ratio_penalty is applied. Indicates "
            "variant is much more common than current word."
        ),
    )
    freq_ratio_mid_cutoff: float = Field(
        default=2.0,
        gt=0.0,
        description=(
            "Frequency ratio (variant_freq / word_freq) above which "
            "the mid freq_ratio_penalty is applied. Moderate guard "
            "against MLM frequency bias."
        ),
    )
    reverse_ratio_threshold: float = Field(
        default=50.0,
        gt=0.0,
        description=(
            "Reverse frequency ratio (word_freq / variant_freq) "
            "above which the current word is considered much more "
            "common than the variant. Suggests MLM logit advantage "
            "reflects corpus frequency bias, not contextual fit. "
            "E.g., 300x ratio for common words like သူ."
        ),
    )

    # --- Particle confusable thresholds ---

    particle_logit_threshold_default: float = Field(
        default=3.5,
        ge=0.0,
        description=(
            "Default logit diff threshold for particle confusable "
            "pairs (e.g., က/ကို, မှာ/မှ). Particles have strong "
            "MLM signal but also high ambiguity."
        ),
    )
    particle_logit_threshold_one_char: float = Field(
        default=4.5,
        ge=0.0,
        description=(
            "Logit diff threshold for one-char particle confusable "
            "pairs. One-character particles (e.g., က) are highly "
            "ambiguous and produced false positives at lower thresholds."
        ),
    )
    particle_logit_threshold_high_freq_both: float = Field(
        default=5.0,
        ge=0.0,
        description=(
            "Logit diff threshold for one-char particle pairs where "
            "BOTH word and variant are high-frequency. Maximum "
            "protection against false positives on common particles."
        ),
    )
    particle_logit_threshold_current_in_topk: float = Field(
        default=4.5,
        ge=0.0,
        description=(
            "Logit diff threshold for particle pairs when current "
            "word is in the MLM top-K predictions. Model uncertainty "
            "requires stricter evidence."
        ),
    )
    zero_freq_logit_threshold: float = Field(
        default=5.0,
        ge=0.0,
        description=(
            "Logit diff threshold when the confusable variant has "
            "zero frequency in the dictionary. Applies to both "
            "particle confusables and known homophones. High "
            "threshold because zero-freq variants likely reflect "
            "model bias rather than genuine contextual fit."
        ),
    )

    # Rate limiting
    max_semantic_checks_per_sentence: int = Field(
        default=8,
        ge=1,
        description=(
            "Maximum number of predict_mask calls per sentence. "
            "After this limit, remaining words are skipped. "
            "The prewarm batch call does NOT count toward this limit."
        ),
    )


class HomophoneStrategyConfig(BaseModel):
    """
    Configuration for homophone validation strategy (priority 45).

    Controls detection of real-word errors where homophones (words with
    similar pronunciation) are used incorrectly in context.
    """

    model_config = ConfigDict(
        validate_assignment=True,
        frozen=False,
        extra="forbid",
    )

    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Confidence score for homophone errors.",
    )
    improvement_ratio: float = Field(
        default=5.0,
        ge=1.0,
        description=(
            "Minimum probability improvement ratio for homophone "
            "suggestions (e.g., 5.0 means 5x better)."
        ),
    )
    min_probability: float = Field(
        default=0.001,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum n-gram probability threshold for homophone "
            "suggestions. Prevents FPs from infrequent n-grams."
        ),
    )
    high_freq_threshold: int = Field(
        default=1000,
        ge=0,
        description=(
            "Word frequency above which a stricter improvement "
            "ratio is required. Prevents FPs on common words."
        ),
    )
    high_freq_improvement_ratio: float = Field(
        default=50.0,
        ge=1.0,
        description=(
            "Improvement ratio required for high-frequency words. "
            "Much stricter than the default ratio."
        ),
    )


class NgramStrategyConfig(BaseModel):
    """
    Configuration for N-gram context validation strategy (priority 50).

    Controls n-gram probability validation thresholds including literary
    sentence detection and high-frequency word guards.
    """

    model_config = ConfigDict(
        validate_assignment=True,
        frozen=False,
        extra="forbid",
    )

    literary_sentence_markers: tuple[str, ...] = Field(
        default=(
            "\u1024",
            "\u101c\u1031\u101e\u100a\u103a",
            "\u1025",
            "\u1005\u103c",
            "\u1021\u1036\u1037",
        ),
        description=(
            "Unambiguous literary/classical markers. Sentences "
            "containing these are skipped because their rare word "
            "combinations produce n-gram FPs."
        ),
    )
    high_freq_ngram_guard: int = Field(
        default=5000,
        ge=0,
        description=("Minimum dictionary frequency to suppress n-gram FP on common words."),
    )
    confidence_high: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description=("Confidence for high-probability context errors."),
    )
    confidence_low: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description=("Confidence for low-probability context errors."),
    )
    max_suggestions: int = Field(
        default=5,
        ge=1,
        description="Maximum number of suggestions to generate.",
    )
    edit_distance: int = Field(
        default=2,
        ge=0,
        description="Maximum edit distance for suggestions.",
    )


class SemanticStrategyConfig(BaseModel):
    """
    Configuration for semantic validation strategy (priority 70).

    Controls AI-powered semantic scanning thresholds for proactive
    detection of contextual errors using transformer models.
    """

    model_config = ConfigDict(
        validate_assignment=True,
        frozen=False,
        extra="forbid",
    )

    use_proactive_scanning: bool = Field(
        default=False,
        description=(
            "Enable proactive semantic scanning. Warning: "
            "computationally expensive, increases latency."
        ),
    )
    proactive_confidence_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description=("Minimum confidence to report semantic errors from proactive scanning."),
    )
    min_word_length: int = Field(
        default=2,
        ge=1,
        description=("Minimum word length for semantic analysis."),
    )
    scan_freq_threshold: int = Field(
        default=50_000,
        ge=0,
        description=(
            "Frequency threshold for proactive scanning. Words with "
            "corpus frequency above this are skipped as 'common enough'."
        ),
    )
    contrast_top_k: int = Field(
        default=15,
        ge=1,
        description=(
            "Number of top MLM predictions to request in the initial contrastive scan pass."
        ),
    )
    contrast_max_candidates: int = Field(
        default=16,
        ge=1,
        description=(
            "Maximum number of contrastive candidates to keep after "
            "phonetic scoring in the initial scan pass."
        ),
    )
    contrast_base_margin: float = Field(
        default=1.2,
        ge=0.0,
        description=(
            "Base logit margin required for a contrastive candidate to "
            "beat the current word in the initial scan pass."
        ),
    )
    contrast_min_similarity: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum phonetic similarity for a contrastive candidate "
            "to be considered a plausible confusion."
        ),
    )
    escalation_top_k: int = Field(
        default=35,
        ge=1,
        description=(
            "Number of top MLM predictions to request in the escalation (second-pass) scan."
        ),
    )
    escalation_max_candidates: int = Field(
        default=24,
        ge=1,
        description=(
            "Maximum number of contrastive candidates to keep after "
            "phonetic scoring in the escalation pass."
        ),
    )
    escalation_margin_relax: float = Field(
        default=0.35,
        ge=0.0,
        description=(
            "Amount to relax the contrastive margin during escalation. "
            "escalation_margin = max(0.75, base_margin - this value)."
        ),
    )
    escalation_valid_word_freq_cap: int = Field(
        default=1_500,
        ge=0,
        description=(
            "Maximum word frequency for a valid current word to be "
            "eligible for escalation scanning. High-frequency valid "
            "words are unlikely to be errors."
        ),
    )
    person_prediction_threshold: int = Field(
        default=4,
        ge=1,
        le=10,
        description=(
            "Minimum number of person-word predictions (out of top-K) "
            "required to flag an animacy mismatch. 4/5 separates true "
            "implausibility (inanimate subjects) from animate "
            "non-person subjects."
        ),
    )
    margin_boost_short_word: float = Field(
        default=0.6,
        ge=0.0,
        description="Contrast margin boost for short words (len <= 2).",
    )
    margin_boost_high_freq: float = Field(
        default=0.8,
        ge=0.0,
        description="Contrast margin boost for high-frequency words.",
    )
    margin_boost_mid_freq: float = Field(
        default=0.35,
        ge=0.0,
        description="Contrast margin boost for mid-frequency words.",
    )
    margin_boost_escalation: float = Field(
        default=1.0,
        ge=0.0,
        description="Margin boost during escalation pass.",
    )
    escalation_margin_boost: float = Field(
        default=0.4,
        ge=0.0,
        description="Escalation confidence margin boost.",
    )
    escalation_min_similarity: float = Field(
        default=0.45,
        ge=0.0,
        le=1.0,
        description="Minimum similarity threshold during escalation.",
    )
    proactive_confidence_cap: float = Field(
        default=0.93,
        ge=0.0,
        le=1.0,
        description="Maximum confidence for proactive scanning errors.",
    )
    proactive_confidence_base: float = Field(
        default=0.55,
        ge=0.0,
        le=1.0,
        description="Base confidence for proactive scanning errors.",
    )
    proactive_confidence_divisor: float = Field(
        default=8.0,
        gt=0.0,
        description="Divisor for best_diff in proactive confidence formula.",
    )
    error_budget_threshold: int = Field(
        default=1,
        ge=0,
        description="Skip proactive scanning when existing errors reach this count.",
    )

    # Rate limiting
    max_semantic_checks_per_sentence: int = Field(
        default=8,
        ge=1,
        description=(
            "Maximum number of predict_mask calls per sentence in "
            "contrast fallback. After this limit, remaining words are "
            "skipped to bound latency."
        ),
    )


class ToneStrategyConfig(BaseModel):
    """
    Configuration for tone validation strategy (priority 10).

    Controls tone mark disambiguation thresholds including
    high-frequency word suppression.
    """

    model_config = ConfigDict(
        validate_assignment=True,
        frozen=False,
        extra="forbid",
    )

    high_freq_threshold: int = Field(
        default=100_000,
        ge=0,
        description=(
            "Frequency threshold for tone_ambiguity suppression. "
            "When both original and correction exceed this, the "
            "error is suppressed since both forms are valid."
        ),
    )
    confidence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=("Minimum confidence to report a tone error."),
    )


class BrokenCompoundStrategyConfig(BaseModel):
    """Configuration for broken compound detection strategy."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rare_threshold: int = Field(
        default=2000,
        ge=0,
        description="Frequency below which a word is considered rare.",
    )
    compound_min_frequency: int = Field(
        default=5000,
        ge=0,
        description="Minimum compound frequency to flag broken compound.",
    )
    compound_ratio: float = Field(
        default=10.0,
        ge=1.0,
        description="Minimum ratio of compound_freq / rare_word_freq.",
    )
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Default confidence for broken compound errors.",
    )
    both_high_freq: int = Field(
        default=10000,
        ge=0,
        description="Frequency guard for multi-syllable both-high compounds.",
    )
    min_compound_len: int = Field(
        default=4,
        ge=1,
        description="Minimum compound length for both-high-freq guard.",
    )


class NeuralRerankerConfig(BaseModel):
    """
    Configuration for the neural MLP suggestion reranker.

    Controls the ONNX-based MLP that re-scores spell checker suggestions
    using 16 extracted features (edit distance, frequency, phonetic
    similarity, n-gram context, etc.).

    The neural reranker runs AFTER both n-gram and semantic reranking,
    giving it the final say on suggestion ordering.

    Attributes:
        enabled: Enable neural reranking (default: False).
        model_path: Path to ONNX model file.
        stats_path: Path to normalization stats JSON (feature_means/stds).
        confidence_gap_threshold: Skip reranking if score gap > this.
        max_candidates: Max candidates to score per error.
    """

    model_config = ConfigDict(
        validate_assignment=True,
        frozen=False,
        extra="forbid",
    )

    enabled: bool = Field(
        default=False,
        description="Enable neural MLP suggestion reranking.",
    )
    model_path: str | None = Field(
        default=None,
        description="Path to the ONNX model file for neural reranking.",
    )
    stats_path: str | None = Field(
        default=None,
        description=(
            "Path to normalization stats JSON file containing "
            "feature_means and feature_stds arrays."
        ),
    )
    confidence_gap_threshold: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description=(
            "Skip reranking when the gap between top-1 and top-2 "
            "neural scores exceeds this threshold. A large gap "
            "indicates the current ranking is already confident."
        ),
    )
    max_candidates: int = Field(
        default=20,
        ge=1,
        description="Maximum number of candidates to score per error.",
    )


class TokenRefinementConfig(BaseModel):
    """Configuration for token boundary refinement scoring.

    Controls scoring parameters used during validation-time token-lattice
    refinement. The refinement pass operates on already-segmented tokens to
    expose hidden error spans in merged tokens (e.g., particle attachment,
    negation attachment).

    Attributes:
        suffix_score_boost: Score boost when suffix matches a known form.
        known_part_score: Score for known dictionary parts.
        unknown_long_part_penalty: Penalty for unknown long parts.
        split_complexity_penalty: Penalty for complex multi-part splits.
        bigram_scale: Scaling factor for bigram probability contribution.
        min_token_len: Minimum token length for refinement candidates.
        keep_if_freq_at_least: Keep token if frequency is at least this value.
        min_score_gain: Minimum score improvement to accept a split.
        lattice_max_paths: Maximum lattice paths to consider.
        syllable_split_min_token_len: Minimum token length for syllable-level splitting.
        syllable_split_max_syllables: Maximum syllables for syllable-level splitting.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    suffix_score_boost: float = Field(
        default=0.85,
        description="Score boost when suffix matches a known form.",
    )
    known_part_score: float = Field(
        default=1.35,
        description="Score for known dictionary parts.",
    )
    unknown_long_part_penalty: float = Field(
        default=0.45,
        description="Penalty for unknown long parts.",
    )
    split_complexity_penalty: float = Field(
        default=0.30,
        description="Penalty for complex multi-part splits.",
    )
    bigram_scale: float = Field(
        default=120_000.0,
        description="Scaling factor for bigram probability contribution.",
    )
    min_token_len: int = Field(
        default=3,
        ge=1,
        description="Minimum token length for refinement candidates.",
    )
    keep_if_freq_at_least: int = Field(
        default=2_000,
        ge=0,
        description="Keep token if frequency is at least this value.",
    )
    min_score_gain: float = Field(
        default=0.55,
        description="Minimum score improvement to accept a split.",
    )
    lattice_max_paths: int = Field(
        default=2,
        ge=1,
        description="Maximum lattice paths to consider.",
    )
    syllable_split_min_token_len: int = Field(
        default=4,
        ge=1,
        description="Minimum token length for syllable-level splitting.",
    )
    syllable_split_max_syllables: int = Field(
        default=6,
        ge=1,
        description="Maximum syllables for syllable-level splitting.",
    )
