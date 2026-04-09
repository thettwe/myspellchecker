"""Validation strategy configuration classes.

Per-strategy threshold and tuning parameters for the 12-strategy
validation pipeline.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "BrokenCompoundStrategyConfig",
    "ConfusableSemanticConfig",
    "HomophoneStrategyConfig",
    "NgramStrategyConfig",
    "SemanticStrategyConfig",
    "ToneStrategyConfig",
]


class ConfusableSemanticConfig(BaseModel):
    """
    Configuration for confusable semantic validation strategy (priority 48).

    Controls MLM-enhanced confusable variant detection thresholds.
    Uses masked language modeling to detect valid-word confusables that pass
    all other validation because both the current word and its confusable
    variant are legitimate dictionary words.

    Asymmetric thresholds protect against false positives:
    - Default: logit_diff >= 2.5 (~12x probability ratio)
    - Medial swap: logit_diff >= 2.0 (highest signal)
    - Current in top-K: logit_diff >= 5.0 (model already considers it)
    - High-frequency: logit_diff >= 3.5 (~33x probability ratio)
    """

    model_config = ConfigDict(
        validate_assignment=True,
        frozen=False,
        extra="forbid",
    )

    logit_diff_threshold: float = Field(
        default=2.5,
        ge=0.0,
        description=(
            "Default logit diff threshold (~12x probability ratio). "
            "Variant must score this much higher than current word. "
            "Lowered from 3.0 to improve confusable recall."
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
        default=3.5,
        ge=0.0,
        description=(
            "Logit diff threshold for high-frequency words (~33x "
            "ratio). Protects common particles/words against FPs. "
            "Lowered from 4.5 to enable Kinzi and stacking confusable "
            "detection on high-frequency Pali loanwords."
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
        default=10,
        ge=0,
        description="Skip proactive scanning when existing errors reach this count. "
        "Raised from 5 to avoid premature semantic cutoff in error-rich text.",
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
        default=15,
        ge=1,
        description=(
            "Maximum number of predict_mask calls per sentence. "
            "After this limit, remaining words are skipped. "
            "The prewarm batch call does NOT count toward this limit. "
            "Raised from 8 to handle longer sentences."
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
        default=0.45,
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
        default=5,
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
        description="Frequency below which a word is considered rare. "
        "Logic: if min(freq1, freq2) >= threshold, skip detection. "
        "Lower values = fewer candidates = fewer FPs.",
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
