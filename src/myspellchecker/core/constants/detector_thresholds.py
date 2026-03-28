"""Frozen dataclasses for empirically calibrated detector thresholds.

These are internal engineering parameters derived from corpus evaluation.
They are NOT user-facing configuration — use SpellCheckerConfig for that.

Each threshold documents what it controls and why it has that value.
Serialize with ``dataclasses.asdict()`` for reproducibility.

Usage in detector mixins::

    from myspellchecker.core.constants.detector_thresholds import (
        CompoundDetectionThresholds,
        DEFAULT_COMPOUND_THRESHOLDS,
    )

    class CompoundDetectionMixin:
        _compound_thresholds: CompoundDetectionThresholds = DEFAULT_COMPOUND_THRESHOLDS

        def _detect_something(self, ...):
            t = self._compound_thresholds
            if freq < t.rejoin_min_correction_freq:
                continue
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CompoundDetectionThresholds:
    """Thresholds for CompoundDetectionMixin detectors.

    Empirically calibrated. All frequency values reference
    the production database (601K words, 21K syllables).
    """

    # ── Missegmented confusable (aspirated consonant swap) ──

    # Minimum frequency for the corrected (swapped) form.
    # Set to 5000 to avoid suggesting rare dictionary entries.
    rejoin_min_correction_freq: int = 5000

    # Maximum frequency for the incorrectly-joined form.
    # If the joined form is common, it is not a segmentation error.
    rejoin_max_joined_freq: int = 500

    # Minimum ratio of correction_freq / joined_freq to flag as error.
    rejoin_min_freq_ratio: int = 10

    # Maximum Myanmar character count for re-joined tokens.
    # Longer tokens are unlikely single words; caps O(L^d) SymSpell variants.
    rejoin_max_myanmar_chars: int = 8

    # Minimum frequency for asat-form correction (higher than general
    # rejoin threshold to avoid FPs from concatenated standalone words).
    rejoin_min_asat_freq: int = 5000

    # Maximum frequency for first syllable in asat check.
    # Common standalone words (particles, pronouns) are not fragments.
    rejoin_max_first_syllable_freq: int = 10000

    # Maximum frequency for joined form in asat-specific check.
    # Stricter than general rejoin — asat insertion is lower-precision.
    rejoin_asat_max_joined_freq: int = 100

    # ── Broken compound (space-separated) ──

    # Both parts above this threshold = space is intentional, not broken compound.
    rare_standalone_threshold: int = 2000

    # Compound frequency must exceed this to override standalone guard.
    dominant_compound_threshold: int = 50000

    # Minimum compound frequency for full-token joins (left+right→compound).
    broken_compound_full_min_freq: int = 5000

    # Minimum compound frequency for prefix-join cases.
    # Lower than full-join — prefix patterns are higher precision.
    broken_compound_prefix_min_freq: int = 500

    # If right token exceeds this, splitting its prefix is too risky.
    broken_compound_right_token_max_freq: int = 4000

    # Minimum frequency for remaining right tail after prefix-join.
    broken_compound_tail_min_freq: int = 500

    # ── Broken compound (morpheme typo) ──

    # Minimum frequency for the correct compound (ed-1 variant + next word).
    broken_morpheme_compound_freq: int = 3000

    # Maximum frequency for the wrong compound. If common, it is likely valid.
    broken_morpheme_wrong_max_freq: int = 100

    # Skip words above this — ultra-common standalone words are correct.
    broken_morpheme_skip_freq: int = 500_000

    # Maximum Myanmar character count for SymSpell lookup. Caps O(L^d) variants.
    broken_morpheme_max_chars: int = 6

    # Maximum SymSpell lookups per invocation to bound total latency.
    broken_morpheme_max_lookups: int = 3

    # ── Suffix confusion ──

    # Minimum frequency for the corrected suffix form.
    suffix_confusion_min_freq: int = 2000

    # Minimum ratio of correction_freq / original_freq.
    suffix_confusion_min_ratio: int = 5

    # ── Compound validation ──

    # Both compound parts must exceed this for compound to be accepted.
    compound_both_parts_min_freq: int = 50000

    # Leading part must exceed this (lower — leading morpheme is more diagnostic).
    compound_leading_part_min_freq: int = 5000

    # ── Variant detection (phonetic/confusable) ──

    # Minimum frequency for the variant candidate to be worth flagging.
    variant_min_freq: int = 5000

    # Maximum frequency for the base (original) word.
    # High-frequency words are unlikely variants of something else.
    variant_max_base_freq: int = 30000

    # Minimum freq ratio (variant/base) to flag without semantic evidence.
    variant_min_freq_ratio: float = 4.0

    # Minimum semantic delta (MLM score difference) for semantic-assisted detection.
    variant_min_semantic_delta: float = 0.1

    # Freq ratio threshold when semantic evidence IS available.
    # Higher — semantic provides complementary evidence.
    variant_min_freq_ratio_with_semantic: float = 12.0

    # Freq ratio threshold when semantic evidence is NOT available.
    variant_min_freq_ratio_without_semantic: float = 10.0

    # Multiplier for second-best candidate comparison.
    # If 2nd candidate is within 5% of best, signal is ambiguous — skip.
    variant_second_candidate_multiplier: float = 1.05

    # ── Invalid token recovery ──

    # Minimum frequency for general repairs.
    invalid_repair_min_freq: int = 500

    # Minimum frequency for virama-based repairs (lower — high precision).
    invalid_repair_virama_min_freq: int = 10

    # Maximum suggestions to request from SymSpell.
    invalid_token_max_suggestions: int = 8

    # Maximum edit distance for SymSpell lookup.
    invalid_token_max_edit_distance: int = 2

    # Minimum dominance ratio for top suggestion over alternatives.
    invalid_token_min_dominance_ratio: float = 1.1

    # Number of top suggestions to consider.
    invalid_token_top_n: int = 3

    # ── Missing visarga (း) ──

    # Frequency ratio threshold: word+visarga must have
    # freq >= ratio * original_freq to be flagged.
    missing_visarga_freq_ratio: float = 5.0

    # Gate on existing error confidence. Skip visarga check if existing
    # error at this position has confidence above this gate.
    visarga_existing_confidence_gate: float = 0.75

    # Base confidence for visarga errors.
    visarga_confidence_base: float = 0.75

    # Maximum confidence for visarga errors.
    visarga_confidence_max: float = 0.95


@dataclass(frozen=True, slots=True)
class ParticleDetectionThresholds:
    """Thresholds for ParticleDetectionMixin detectors.

    Particle detection is high-precision, low-recall by design
    (FP cost is high for particle errors).
    """

    # ── Missing asat (်) ──

    # Minimum frequency for standalone asat-form correction.
    missing_asat_standalone_min_freq: int = 500

    # Minimum frequency for suffix-based asat detection.
    missing_asat_suffix_min_freq: int = 500

    # Minimum frequency for stem in stem+asat compound patterns.
    missing_asat_stem_min_freq: int = 1000

    # Minimum ratio of (stem+asat freq) / stem_freq.
    missing_asat_stem_min_ratio: float = 5.0

    # ── Missing visarga in compounds ──

    # Minimum frequency for compound with visarga.
    visarga_compound_min_freq: int = 5000

    # Skip compounds above this — lexicalized and correct as-is.
    visarga_compound_skip_freq: int = 5000

    # Minimum ratio of (with_visarga_freq / without_visarga_freq).
    visarga_compound_min_ratio: float = 10.0

    # ── Particle confusion ──

    # Maximum token index for subject pronoun detection.
    # Only check first N tokens (pronouns appear sentence-initially).
    subject_pronoun_max_index: int = 2

    # Lookahead window for finding verb after particle.
    # Myanmar SOV order: verb is within ~5 tokens of particle.
    particle_confusion_lookahead_window: int = 5

    # ── Dangling particles ──

    # Maximum structural errors before suppressing dangling particle reports.
    structural_error_max_count: int = 3


@dataclass(frozen=True, slots=True)
class AlgorithmThresholds:
    """Default values for core algorithm parameters.

    Referenced by SymSpellConfig, NgramContextConfig, RankerConfig etc.
    as their default values. These are infrastructure-level parameters.
    """

    # SymSpell: maximum edit distance for candidate generation.
    max_edit_distance: int = 2

    # SymSpell: prefix length for delete-neighborhood indexing.
    # Myanmar syllables are 2-4 chars; 10 covers 3+ syllable words.
    prefix_length: int = 10

    # Beam search: default width for compound segmentation DP.
    # Practical valid segmentations per position is 1-3; 25 is sufficient.
    beam_width_default: int = 25

    # Beam search: minimal width for fast mode.
    beam_width_minimal: int = 10

    # Beam search: width for POS tagger specifically.
    beam_width_pos_tagger: int = 15

    # N-gram context checker: probability threshold below which
    # a word-in-context is flagged as improbable.
    ngram_threshold: float = 0.01

    # Suggestion ranking: maximum candidates to evaluate.
    candidate_limit: int = 50

    # Suggestion ranking: minimum probability denominator
    # to prevent division-by-zero.
    min_probability_denom: float = 0.001

    # LRU cache size for edit distance, POS lookups, etc.
    lru_cache_size: int = 4096


@dataclass(frozen=True, slots=True)
class SuppressionThresholds:
    """Thresholds for ErrorSuppressionMixin heuristics.

    These control false-positive filtering in the post-detection phase.
    Consolidated here from error_suppression.py for discoverability.
    All values are empirically calibrated; when enrichment DB data is available,
    some heuristics may be bypassed entirely.
    """

    # Confusable suppression
    confusable_short_token_freq: int = 20_000
    confusable_ambiguity_max_confidence: float = 0.85
    confusable_fragment_max_token_len: int = 2
    confusable_self_suggest_max_token_len: int = 3

    # Cascade suppression
    bare_consonant_proximity: int = 3

    # Semantic suppression
    semantic_stable_noun_min_len: int = 5
    semantic_short_suggestion_max_ratio: float = 0.65

    # POS sequence suppression
    pos_seq_long_span_no_suggestion_len: int = 15
    pos_seq_medium_span_tiny_suggestion_len: int = 6
    pos_seq_tiny_suggestion_max_len: int = 2

    # Syntax suppression
    syntax_short_swap_max_token_len: int = 3

    # Syllable suppression
    syllable_tech_compound_min_token_len: int = 3

    # Context probability suppression
    context_prob_min_token_len_for_short_sug: int = 3
    context_prob_short_suggestion_max_len: int = 2

    # NER suppression
    ner_high_confidence_override: float = 0.85
    ner_loc_default_confidence: float = 0.85


# ── Module-level singletons (default instances) ──

DEFAULT_COMPOUND_THRESHOLDS = CompoundDetectionThresholds()
DEFAULT_PARTICLE_THRESHOLDS = ParticleDetectionThresholds()
DEFAULT_ALGORITHM_THRESHOLDS = AlgorithmThresholds()
DEFAULT_SUPPRESSION_THRESHOLDS = SuppressionThresholds()
