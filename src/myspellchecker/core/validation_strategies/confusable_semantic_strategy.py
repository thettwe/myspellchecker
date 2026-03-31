"""
Confusable Semantic Validation Strategy (MLM-Enhanced).

This strategy uses masked language modeling to detect valid-word confusables
that pass all other validation because both the current word and its confusable
variant are legitimate dictionary words.

Example: ကျွန်တော် ကြောင်းကို သွားတယ်။
  - ကြောင်း ("cat") passes all checks -- it's a valid word
  - But MLM strongly predicts ကျောင်း ("school") in "went to [X]" context
  - logit_diff exceeds threshold -> flagged as confusable_error

Priority: 48 (between Homophone 45 and N-gram 50)
  - Uses MLM (ONNX) for deeper semantic analysis than n-gram statistics
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from myspellchecker.core.config.algorithm_configs import ConfusableSemanticConfig
from myspellchecker.core.constants import (
    CONFUSABLE_EXEMPT_PAIRS,
    ET_CONFUSABLE_ERROR,
    PARTICLE_CONFUSABLES,
    VARIANT_BLOCKLIST,
)
from myspellchecker.core.response import ContextError, Error
from myspellchecker.core.validation_strategies.base import ValidationContext, ValidationStrategy
from myspellchecker.core.validation_strategies.confusable_helpers import (
    cap_threshold,
    is_db_suppressed,
    is_exempt_suffix_pair,
    is_medial_confusable,
    is_medial_deletion,
    is_occurrence_token_boundary,
    is_particle_confusable,
    is_tone_marker_only_pair,
    is_visarga_only_pair,
)
from myspellchecker.core.validation_strategies.confusable_strategy import (
    generate_confusable_variants,
)
from myspellchecker.text.phonetic import PhoneticHasher
from myspellchecker.utils.logging_utils import get_logger

if TYPE_CHECKING:
    from myspellchecker.algorithms.semantic_checker import SemanticChecker
    from myspellchecker.providers.interfaces import NgramRepository

logger = get_logger(__name__)


class ConfusableSemanticStrategy(ValidationStrategy):
    """
    MLM-enhanced confusable variant detection.

    For each valid word, generates confusable variants, filters to valid
    dictionary words, then uses a single predict_mask() call to compare
    MLM logits between the current word and the best variant.

    Asymmetric thresholds protect against false positives:
    - Default: logit_diff >= 3.0 (~20x probability ratio)
    - Medial ျ↔ြ swap: logit_diff >= 2.0 (highest signal, different meanings)
    - Current word in top-K: logit_diff >= 5.0 (model already "sees" it)
    - High-frequency word: logit_diff >= 6.0 (protect common words)

    Priority: 48
    """

    # Penalty constants now live in ConfusableSemanticConfig.
    # Accessed at runtime via self._config.<field_name>.
    _SENTENCE_FINAL_PUNCT: frozenset[str] = frozenset({"\u104a", "\u104b"})

    def __init__(
        self,
        semantic_checker: "SemanticChecker",
        provider: "NgramRepository",
        confidence: float = 0.80,
        top_k: int = 50,
        logit_diff_threshold: float | None = None,
        logit_diff_threshold_medial: float | None = None,
        logit_diff_threshold_current_in_topk: float | None = None,
        high_freq_threshold: int | None = None,
        high_freq_logit_diff: float | None = None,
        min_word_length: int | None = None,
        freq_ratio_penalty_high: float | None = None,
        freq_ratio_penalty_mid: float | None = None,
        visarga_penalty: float | None = None,
        sentence_final_penalty: float | None = None,
        homophone_map: dict[str, set[str]] | None = None,
        logit_diff_threshold_homophone: float | None = None,
        max_threshold: float | None = None,
        reverse_ratio_min_freq: int | None = None,
        visarga_high_freq_hard_block: bool | None = None,
        curated_pairs: dict[str, set[str]] | None = None,
        curated_logit_diff_threshold: float | None = None,
        near_synonym_pairs: dict[str, set[str]] | None = None,
        near_synonym_logit_diff_threshold: float | None = None,
        config: ConfusableSemanticConfig | None = None,
    ):
        self._config = config or ConfusableSemanticConfig()
        self.semantic_checker = semantic_checker
        self.provider = provider
        self.confidence = confidence
        self.top_k = top_k
        # Explicit kwargs override config defaults
        self.logit_diff_threshold = (
            logit_diff_threshold
            if logit_diff_threshold is not None
            else self._config.logit_diff_threshold
        )
        self.logit_diff_threshold_medial = (
            logit_diff_threshold_medial
            if logit_diff_threshold_medial is not None
            else self._config.logit_diff_threshold_medial
        )
        self.logit_diff_threshold_current_in_topk = (
            logit_diff_threshold_current_in_topk
            if logit_diff_threshold_current_in_topk is not None
            else self._config.logit_diff_threshold_current_in_topk
        )
        self.high_freq_threshold = (
            high_freq_threshold
            if high_freq_threshold is not None
            else self._config.high_freq_threshold
        )
        self.high_freq_logit_diff = (
            high_freq_logit_diff
            if high_freq_logit_diff is not None
            else self._config.high_freq_logit_diff
        )
        self.min_word_length = (
            min_word_length if min_word_length is not None else self._config.min_word_length
        )
        self.freq_ratio_penalty_high = (
            freq_ratio_penalty_high
            if freq_ratio_penalty_high is not None
            else self._config.freq_ratio_penalty_high
        )
        self.freq_ratio_penalty_mid = (
            freq_ratio_penalty_mid
            if freq_ratio_penalty_mid is not None
            else self._config.freq_ratio_penalty_mid
        )
        self.visarga_penalty = (
            visarga_penalty if visarga_penalty is not None else self._config.visarga_penalty
        )
        self.sentence_final_penalty = (
            sentence_final_penalty
            if sentence_final_penalty is not None
            else self._config.sentence_final_penalty
        )
        self._homophone_map = homophone_map or {}
        self.logit_diff_threshold_homophone = (
            logit_diff_threshold_homophone
            if logit_diff_threshold_homophone is not None
            else self._config.logit_diff_threshold_homophone
        )
        self.max_threshold = (
            max_threshold if max_threshold is not None else self._config.max_threshold
        )
        self.reverse_ratio_min_freq = (
            reverse_ratio_min_freq
            if reverse_ratio_min_freq is not None
            else self._config.reverse_ratio_min_freq
        )
        self.visarga_high_freq_hard_block = (
            visarga_high_freq_hard_block
            if visarga_high_freq_hard_block is not None
            else self._config.visarga_high_freq_hard_block
        )
        self._curated_pairs = curated_pairs or {}
        self.curated_logit_diff_threshold = (
            curated_logit_diff_threshold
            if curated_logit_diff_threshold is not None
            else self._config.curated_logit_diff_threshold
        )
        self._near_synonym_pairs = near_synonym_pairs or {}
        self.near_synonym_logit_diff_threshold = (
            near_synonym_logit_diff_threshold
            if near_synonym_logit_diff_threshold is not None
            else self._config.near_synonym_logit_diff_threshold
        )
        self._hasher = PhoneticHasher(ignore_tones=False)
        self._MAX_SEMANTIC_CHECKS_PER_SENTENCE = self._config.max_semantic_checks_per_sentence
        self.logger = logger
        self._error_budget_threshold = self._config.error_budget_threshold
        self._adjacency_window = self._config.adjacency_window
        self._adjacency_penalty_base = self._config.adjacency_penalty_base
        self._adjacency_penalty_per_word = self._config.adjacency_penalty_per_word

    def _prewarm_logits_cache(self, context: ValidationContext) -> None:
        """Pre-warm the semantic checker's logits cache for all eligible words.

        Runs a lightweight filter pass (no ONNX) to identify which words will
        need semantic checking, then issues a single batched ONNX forward pass
        to pre-populate the cache.  The main validate loop then benefits from
        cache hits for both predict_mask() and score_mask_candidates().
        """
        batch_fn = getattr(self.semantic_checker, "batch_get_mask_logits", None)
        if not callable(batch_fn):
            return

        targets: list[tuple[str, int]] = []
        occurrence_counts: dict[str, int] = {}

        for i in range(len(context.words)):
            word = context.words[i]
            position = context.word_positions[i]

            # Mirror the same skip logic as validate()
            if context.is_name_mask[i]:
                occurrence_counts[word] = occurrence_counts.get(word, 0) + 1
                continue
            if position in context.existing_errors:
                occurrence_counts[word] = occurrence_counts.get(word, 0) + 1
                continue

            is_particle_confusable_flag = word in PARTICLE_CONFUSABLES
            is_curated = word in self._curated_pairs
            is_near_synonym = word in self._near_synonym_pairs
            if (
                len(word) < self.min_word_length
                and not is_particle_confusable_flag
                and not is_curated
                and not is_near_synonym
            ):
                occurrence_counts[word] = occurrence_counts.get(word, 0) + 1
                continue

            if not self.provider.is_valid_word(word):
                occurrence_counts[word] = occurrence_counts.get(word, 0) + 1
                continue

            # Check if this word has any confusable variants
            raw_variants = generate_confusable_variants(word, self._hasher)
            if is_particle_confusable_flag:
                raw_variants.update(PARTICLE_CONFUSABLES[word])
            if word in self._homophone_map:
                raw_variants.update(self._homophone_map[word])
            if is_curated:
                raw_variants.update(self._curated_pairs[word])
            if is_near_synonym:
                raw_variants.update(self._near_synonym_pairs[word])

            # Loan word transliteration variants
            from myspellchecker.core.loan_word_variants import (
                get_loan_word_standard,
                get_loan_word_variants,
            )

            loan_variants = get_loan_word_variants(word)
            if loan_variants:
                raw_variants.update(loan_variants)
            loan_standards = get_loan_word_standard(word)
            if loan_standards:
                raw_variants.update(loan_standards)

            has_valid_variant = any(
                v != word
                and v not in VARIANT_BLOCKLIST
                and (word, v) not in CONFUSABLE_EXEMPT_PAIRS
                and not is_exempt_suffix_pair(word, v)
                and not is_db_suppressed(word, v, self.provider)
                and self.provider.is_valid_word(v)
                for v in raw_variants
            )

            if has_valid_variant:
                occurrence = occurrence_counts.get(word, 0)
                targets.append((word, occurrence))

            occurrence_counts[word] = occurrence_counts.get(word, 0) + 1

        if targets:
            batch_fn(context.sentence, targets)

    def validate(self, context: ValidationContext) -> list[Error]:
        """Validate words for MLM-detected confusable errors."""
        if not self.semantic_checker or len(context.words) < 2:
            return []

        if not hasattr(self.provider, "is_valid_word"):
            return []

        # Guard: Error budget -- skip confusable semantic when the sentence
        # (or a previous sentence/chunk) already has errors from higher-priority
        # strategies.  The MLM context is corrupted by existing errors, causing
        # cascade FPs where valid words are flagged because nearby errors distort
        # the probability distribution (error misattribution).
        if (
            len(context.existing_errors) >= self._error_budget_threshold
            or context.global_error_count >= self._error_budget_threshold
        ):
            return []

        errors: list[Error] = []

        try:
            # Pre-warm logits cache with a single batched ONNX call.
            self._prewarm_logits_cache(context)

            semantic_call_count = 0
            for i in range(len(context.words)):
                word = context.words[i]
                position = context.word_positions[i]

                # Skip names, existing errors, short words
                if context.is_name_mask[i]:
                    continue
                if position in context.existing_errors:
                    continue
                is_particle_confusable_flag = word in PARTICLE_CONFUSABLES
                is_curated = word in self._curated_pairs
                is_near_synonym = word in self._near_synonym_pairs
                if (
                    len(word) < self.min_word_length
                    and not is_particle_confusable_flag
                    and not is_curated
                    and not is_near_synonym
                ):
                    continue

                # Only check valid words (valid but wrong in context)
                if not self.provider.is_valid_word(word):
                    continue

                # Generate confusable variants, filter to valid dictionary words
                raw_variants = generate_confusable_variants(word, self._hasher)

                # Add particle confusable variants (not covered by phonetic generation)
                if is_particle_confusable_flag:
                    raw_variants.update(PARTICLE_CONFUSABLES[word])

                # Add known homophone pairs from homophones.yaml
                if word in self._homophone_map:
                    raw_variants.update(self._homophone_map[word])

                # Add curated confusable pairs (known real-word confusion patterns
                # that cannot be generated by phonetic/visual variant rules)
                curated_variants_for_word: set[str] = set()
                if is_curated:
                    curated_variants_for_word = self._curated_pairs[word]
                    raw_variants.update(curated_variants_for_word)

                # Add near-synonym confusable pairs (semantically similar words
                # that need stronger MLM evidence to distinguish)
                near_synonym_variants_for_word: set[str] = set()
                if is_near_synonym:
                    near_synonym_variants_for_word = self._near_synonym_pairs[word]
                    raw_variants.update(near_synonym_variants_for_word)

                # Loan word transliteration variants
                from myspellchecker.core.loan_word_variants import (
                    get_loan_word_standard,
                    get_loan_word_variants,
                )

                loan_variants = get_loan_word_variants(word)
                if loan_variants:
                    raw_variants.update(loan_variants)
                loan_standards = get_loan_word_standard(word)
                if loan_standards:
                    raw_variants.update(loan_standards)

                valid_variants = {
                    v
                    for v in raw_variants
                    if v != word
                    and v not in VARIANT_BLOCKLIST
                    and (word, v) not in CONFUSABLE_EXEMPT_PAIRS
                    and not is_exempt_suffix_pair(word, v)
                    and not is_db_suppressed(word, v, self.provider)
                    and self.provider.is_valid_word(v)
                }

                if not valid_variants:
                    continue

                # Per-sentence cap: stop checking after reaching the limit.
                if semantic_call_count >= self._MAX_SEMANTIC_CHECKS_PER_SENTENCE:
                    break

                # ONE predict_mask call per candidate word
                # For short words (especially particles), count occurrences before
                # this position to mask the correct one (e.g., "က" appears in
                # "ကျွန်တော်" and as standalone particle -- must mask the right one).
                # Count how many times this word appeared before index i in the
                # words list (using absolute position on context.sentence fails
                # for 2nd+ sentences where position > len(sentence)).
                occurrence = sum(1 for w_idx in range(i) if context.words[w_idx] == word)
                is_boundary = is_occurrence_token_boundary(
                    context.sentence, word, occurrence, self.semantic_checker
                )
                semantic_call_count += 1
                predictions = self.semantic_checker.predict_mask(
                    context.sentence, word, top_k=self.top_k, occurrence=occurrence
                )

                if not predictions:
                    continue

                pred_map: dict[str, float] = {
                    w: float(score) for w, score in predictions if isinstance(w, str)
                }
                if not pred_map:
                    continue

                current_in_topk = word in pred_map

                explicit_scores: dict[str, float] = {}
                score_fn = getattr(self.semantic_checker, "score_mask_candidates", None)
                if callable(score_fn):
                    try:
                        scored = score_fn(
                            context.sentence,
                            word,
                            [word, *valid_variants],
                            occurrence=occurrence,
                        )
                        if isinstance(scored, dict):
                            for candidate, raw_score in scored.items():
                                if not isinstance(candidate, str):
                                    continue
                                try:
                                    explicit_scores[candidate] = float(raw_score)
                                except (TypeError, ValueError):
                                    continue
                    except (RuntimeError, ValueError, TypeError, AttributeError, KeyError) as e:
                        self.logger.debug(f"Explicit candidate scoring failed: {e}")

                current_score = explicit_scores.get(word, pred_map.get(word))

                word_freq = 0
                is_high_freq = False
                if hasattr(self.provider, "get_word_frequency"):
                    word_freq = self.provider.get_word_frequency(word)
                    if word_freq >= self.high_freq_threshold:
                        is_high_freq = True

                # Detect sentence-final position (last word, or before punctuation)
                is_sentence_final = (i == len(context.words) - 1) or (
                    i + 1 < len(context.words)
                    and context.words[i + 1] in self._SENTENCE_FINAL_PUNCT
                )

                # Find best variant that exceeds threshold
                best_variant = self._find_best_variant(
                    word,
                    valid_variants,
                    pred_map,
                    explicit_scores,
                    current_score,
                    current_in_topk,
                    is_high_freq,
                    word_freq,
                    is_sentence_final,
                    is_boundary,
                    curated_variants=curated_variants_for_word,
                    near_synonym_variants=near_synonym_variants_for_word,
                )

                if best_variant:
                    prev_word = context.words[i - 1] if i > 0 else ""

                    # Guard 1: Adjacency dampening -- reduce confidence when
                    # an existing error is within 3 words.  The MLM context
                    # is corrupted by nearby errors, making detections at
                    # adjacent positions less reliable.
                    effective_confidence = self.confidence
                    for ei, epos in enumerate(context.word_positions):
                        if epos in context.existing_errors:
                            word_distance = abs(i - ei)
                            if word_distance <= self._adjacency_window:
                                penalty = (
                                    self._adjacency_penalty_base
                                    + self._adjacency_penalty_per_word * word_distance
                                )
                                effective_confidence = min(
                                    effective_confidence,
                                    self.confidence * penalty,
                                )

                    errors.append(
                        ContextError(
                            text=word,
                            position=position,
                            error_type=ET_CONFUSABLE_ERROR,
                            suggestions=[best_variant],
                            confidence=effective_confidence,
                            probability=0.0,
                            prev_word=prev_word,
                        )
                    )
                    context.existing_errors[position] = ET_CONFUSABLE_ERROR
                    context.existing_suggestions[position] = [best_variant]
                    context.existing_confidences[position] = effective_confidence

        except (RuntimeError, ValueError, KeyError, IndexError, AttributeError, TypeError) as e:
            self.logger.error(f"Error in confusable semantic validation: {e}", exc_info=True)

        return errors

    def _find_best_variant(
        self,
        word: str,
        valid_variants: set[str],
        pred_map: dict[str, float],
        explicit_scores: dict[str, float] | None,
        current_score: float | None,
        current_in_topk: bool,
        is_high_freq: bool,
        word_freq: int = 0,
        is_sentence_final: bool = False,
        is_boundary_occurrence: bool = True,
        curated_variants: set[str] | None = None,
        near_synonym_variants: set[str] | None = None,
    ) -> str | None:
        """
        Find the best variant that exceeds the logit diff threshold.

        Returns the best variant string, or None if no variant qualifies.
        """
        best_variant = None
        best_diff = 0.0
        explicit_scores = explicit_scores or {}
        curated_variants = curated_variants or set()
        near_synonym_variants = near_synonym_variants or set()
        min_score = (
            min(pred_map.values())
            if pred_map
            else min(explicit_scores.values())
            if explicit_scores
            else 0.0
        )

        for variant in valid_variants:
            variant_score = explicit_scores.get(variant, pred_map.get(variant))
            if variant_score is None:
                # Variant has neither explicit nor top-K score.
                continue

            # Compute logit difference
            if current_score is not None:
                logit_diff = variant_score - current_score
            else:
                # Current word not in top-K -- use variant score minus
                # the lowest score in the prediction set as a proxy.
                # Raw variant_score (6-18) far exceeds thresholds (3-6)
                # designed for score *differences*, causing FPs on valid
                # high-freq words like က and ရာ.
                logit_diff = variant_score - min_score

            # Near-synonym pairs use a higher threshold than general curated
            # pairs because both words share semantic overlap and are often
            # high-frequency. Stronger MLM evidence is needed.
            is_near_synonym_variant = variant in near_synonym_variants
            if is_near_synonym_variant:
                threshold = self.near_synonym_logit_diff_threshold
                variant_freq = 0
                if hasattr(self.provider, "get_word_frequency"):
                    variant_freq = self.provider.get_word_frequency(variant)
                variant_in_topk = variant in pred_map
                if not variant_in_topk:
                    threshold += self._config.explicit_non_topk_homophone_penalty
                # Apply sentence-final penalty -- near-synonyms at sentence
                # boundary have less right context for MLM to differentiate.
                if is_sentence_final:
                    threshold += self._config.sentence_final_penalty
                threshold = cap_threshold(threshold, self.max_threshold)

                logger.debug(
                    "confusable_semantic: word=%s variant=%s (NEAR_SYNONYM) word_freq=%d "
                    "variant_freq=%d logit_diff=%.2f threshold=%.2f "
                    "sentence_final=%s decision=%s",
                    word,
                    variant,
                    word_freq,
                    variant_freq,
                    logit_diff,
                    threshold,
                    is_sentence_final,
                    "FLAGGED" if logit_diff >= threshold else "skip",
                )

                if logit_diff >= threshold and logit_diff > best_diff:
                    best_diff = logit_diff
                    best_variant = variant
                continue

            # Curated pairs bypass standard threshold stacking.
            # These are linguistically verified confusable pairs from
            # confusable_pairs.yaml that cannot be generated by phonetic
            # variant rules. Use a flat, low threshold.
            is_curated_variant = variant in curated_variants
            if is_curated_variant:
                threshold = self.curated_logit_diff_threshold
                variant_freq = 0
                if hasattr(self.provider, "get_word_frequency"):
                    variant_freq = self.provider.get_word_frequency(variant)
                # For curated pairs not in top-K, apply a smaller penalty
                # than the default 5.0 -- curated pairs are known to be
                # valid confusables even when the model doesn't predict them.
                variant_in_topk = variant in pred_map
                if not variant_in_topk:
                    threshold += self._config.explicit_non_topk_homophone_penalty
                threshold = cap_threshold(threshold, self.max_threshold)

                logger.debug(
                    "confusable_semantic: word=%s variant=%s (CURATED) word_freq=%d "
                    "variant_freq=%d logit_diff=%.2f threshold=%.2f "
                    "sentence_final=%s decision=%s",
                    word,
                    variant,
                    word_freq,
                    variant_freq,
                    logit_diff,
                    threshold,
                    is_sentence_final,
                    "FLAGGED" if logit_diff >= threshold else "skip",
                )

                if logit_diff >= threshold and logit_diff > best_diff:
                    best_diff = logit_diff
                    best_variant = variant
                continue

            threshold, variant_freq = self._get_threshold(
                word,
                variant,
                current_in_topk,
                is_high_freq,
                word_freq,
                is_sentence_final,
            )
            variant_in_topk = variant in pred_map
            if not variant_in_topk:
                if self._is_known_homophone(word, variant):
                    threshold += self._config.explicit_non_topk_homophone_penalty
                else:
                    threshold += self._config.explicit_non_topk_penalty

            if not is_boundary_occurrence and not self._is_known_homophone(word, variant):
                threshold += self._config.non_boundary_penalty

            logger.debug(
                "confusable_semantic: word=%s variant=%s word_freq=%d "
                "variant_freq=%d logit_diff=%.2f threshold=%.2f "
                "sentence_final=%s decision=%s",
                word,
                variant,
                word_freq,
                variant_freq,
                logit_diff,
                threshold,
                is_sentence_final,
                "FLAGGED" if logit_diff >= threshold else "skip",
            )

            if logit_diff >= threshold and logit_diff > best_diff:
                best_diff = logit_diff
                best_variant = variant

        return best_variant

    def _get_threshold(
        self,
        word: str,
        variant: str,
        current_in_topk: bool,
        is_high_freq: bool,
        word_freq: int = 0,
        is_sentence_final: bool = False,
    ) -> tuple[float, int]:
        """
        Get the logit diff threshold for a word-variant pair.

        Base threshold (highest wins):
        1. High-frequency word: 6.0
        2. Medial confusable (swap or insertion/deletion): 2.0
        3. Current in top-K: 5.0
        4. Default: 3.0

        Additive penalties (stacking):
        - Frequency-ratio penalty: +3.0 if ratio > 5x, +1.5 if > 2x
        - Visarga-pair penalty: +2.0 if difference is only း
        - Sentence-final penalty: +1.0 if word is at sentence end

        Returns:
            Tuple of (threshold, variant_freq).
        """
        # Base threshold -- medial swap (ျ↔ြ) confusables get low threshold
        # even when current word is in top-K, because the MLM signal is
        # highly reliable for these pairs.  Medial deletion (e.g. မြွေ->မြေ
        # via ွ removal) produces two entirely different valid words, so
        # the low swap threshold causes FPs when the variant has higher
        # corpus frequency.  Use stricter threshold for deletion.
        if is_high_freq:
            base = self.high_freq_logit_diff
        elif is_medial_confusable(word, variant):
            if is_medial_deletion(word, variant):
                base = self.logit_diff_threshold_current_in_topk  # 5.0
            else:
                base = self.logit_diff_threshold_medial
        elif current_in_topk:
            base = self.logit_diff_threshold_current_in_topk
        else:
            base = self.logit_diff_threshold

        # Layer 1: Frequency-ratio penalty
        variant_freq = 0
        if hasattr(self.provider, "get_word_frequency"):
            variant_freq = self.provider.get_word_frequency(variant)
        freq_ratio = max(variant_freq, 1) / max(word_freq, 1)

        if freq_ratio > self._config.freq_ratio_high_cutoff:
            base += self.freq_ratio_penalty_high
        elif freq_ratio > self._config.freq_ratio_mid_cutoff:
            base += self.freq_ratio_penalty_mid

        # Reverse frequency guard: if current word is much more common than
        # the variant, the MLM logit advantage likely reflects corpus frequency
        # bias rather than genuine contextual preference. E.g., သူ (1.5M) ->
        # သု (4.9K) has 300x ratio -- the variant is too rare to be meaningful.
        # Only apply when word_freq >= reverse_ratio_min_freq (protect genuinely
        # common words; low-freq misspellings with high reverse ratio are real).
        reverse_ratio = max(word_freq, 1) / max(variant_freq, 1)
        if reverse_ratio > self._config.reverse_ratio_threshold and (
            word_freq >= self.reverse_ratio_min_freq
        ):
            base += self.freq_ratio_penalty_high

        # Layer 2: Tone-marker-only morpheme guard (visarga း or aukmyit ့)
        if is_tone_marker_only_pair(word, variant):
            # Visarga pairs (ပြီ/ပြီး, ငါ/ငါး) occupy similar syntactic slots
            # (both follow verbs), so MLM frequency bias dominates over context.
            # Aukmyit pairs (လို/လို့) diverge more syntactically, so MLM
            # signal is more trustworthy -- use soft penalty only.
            if (
                self.visarga_high_freq_hard_block
                and is_visarga_only_pair(word, variant)
                and word_freq >= self.high_freq_threshold
                and variant_freq >= self.high_freq_threshold
            ):
                return float("inf"), variant_freq
            base += self.visarga_penalty

        # Layer 3: Sentence-final position bonus
        if is_sentence_final:
            base += self.sentence_final_penalty

        # Layer 4: Particle confusable override
        # Particle pairs (က/ကို, မှာ/မှ) have strong MLM signal, but one-char
        # particles are also highly ambiguous and produced false positives when
        # the threshold was too low. Use stricter guard for one-char pairs and
        # when current word is still in top-K (model uncertainty).
        # Guard: if the variant has zero frequency in the DB, the MLM logit
        # advantage likely reflects model bias, not genuine contextual fit
        # (e.g., မှ freq=0 vs မှာ freq=3.2M). Require higher confidence.
        if is_particle_confusable(word, variant):
            if variant_freq == 0:
                return self._config.zero_freq_logit_threshold, variant_freq
            threshold = self._config.particle_logit_threshold_default
            if len(word) == 1 or len(variant) == 1:
                threshold = self._config.particle_logit_threshold_one_char
            if (
                (len(word) == 1 or len(variant) == 1)
                and word_freq >= self.high_freq_threshold
                and variant_freq >= self.high_freq_threshold
            ):
                threshold = max(threshold, self._config.particle_logit_threshold_high_freq_both)
            if current_in_topk:
                threshold = max(threshold, self._config.particle_logit_threshold_current_in_topk)
            return threshold, variant_freq

        # Layer 5: Known homophone override
        # Curated homophone pairs (from homophones.yaml) are linguistically
        # verified confusables. Use a dedicated threshold that skips the
        # freq_ratio and sentence_final penalties that block valid detections.
        # E.g., ဖူး->ဘူး (logit_diff=6.68) was blocked by stacked threshold
        # of 10.0 (6.0 base + 3.0 freq_ratio + 1.0 sentence_final).
        # Guard: reverse_ratio check -- if current word is vastly more common
        # than variant (e.g., သူ 1.5M vs သု 4.9K = 310x), MLM logit
        # advantage reflects corpus frequency bias, not context. Keep the
        # stacked threshold in that case.
        if self._is_known_homophone(word, variant):
            if variant_freq == 0:
                return self._config.zero_freq_logit_threshold, variant_freq
            if reverse_ratio > self._config.reverse_ratio_threshold:
                return cap_threshold(base, self.max_threshold), variant_freq
            return self.logit_diff_threshold_homophone, variant_freq

        return cap_threshold(base, self.max_threshold), variant_freq

    def _is_known_homophone(self, word: str, variant: str) -> bool:
        """Check if word-variant pair is a known homophone from homophones.yaml."""
        return variant in self._homophone_map.get(word, set())

    def priority(self) -> int:
        """Return strategy execution priority (48)."""
        return 48

    def close(self) -> None:
        """Release the underlying SemanticChecker inference session."""
        if self.semantic_checker and hasattr(self.semantic_checker, "close"):
            self.semantic_checker.close()

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ConfusableSemanticStrategy(priority={self.priority()}, "
            f"confidence={self.confidence}, "
            f"logit_diff={self.logit_diff_threshold})"
        )
