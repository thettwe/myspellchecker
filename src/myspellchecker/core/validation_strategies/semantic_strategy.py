"""
Semantic Validation Strategy.

This strategy uses AI-powered semantic models (XLM-RoBERTa) to detect
contextual errors that rule-based and statistical methods might miss.

Priority: 70 (runs last, after all other validation)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from myspellchecker.core.config.algorithm_configs import SemanticStrategyConfig
from myspellchecker.core.constants import ET_SEMANTIC_ERROR
from myspellchecker.core.response import ContextError, Error
from myspellchecker.core.validation_strategies.base import ValidationContext, ValidationStrategy
from myspellchecker.core.validation_strategies.semantic_helpers import (
    build_contrast_candidate_pool,
    build_escalation_candidate_pool,
    char_overlap_similarity,
    contrast_margin,
    should_run_escalation_pass,
)
from myspellchecker.text.phonetic import PhoneticHasher
from myspellchecker.utils.logging_utils import get_logger

if TYPE_CHECKING:
    from myspellchecker.algorithms.semantic_checker import SemanticChecker
    from myspellchecker.providers.interfaces import DictionaryProvider


class SemanticValidationStrategy(ValidationStrategy):
    """
    AI-powered semantic validation strategy.

    This strategy uses pre-trained transformer models (XLM-RoBERTa) to detect
    semantic errors in Myanmar text. It can identify:
    - Contextually inappropriate word choices
    - Subtle semantic errors missed by n-gram models
    - Real-word errors that are grammatically correct but semantically wrong

    The strategy operates in two modes:
    1. **Proactive Scanning**: Scans all words in the sentence for semantic errors
    2. **Refinement Mode**: Validates errors flagged by other strategies (future enhancement)

    Priority: 70 (runs last)
    - AI-powered detection is computationally expensive
    - Should run after all statistical and rule-based methods
    - Can suppress false positives from earlier strategies
    - Confidence: Variable (0.5-0.95, depends on model confidence)

    Example:
        >>> strategy = SemanticValidationStrategy(semantic_checker, config)
        >>> context = ValidationContext(
        ...     sentence="သူ ကား စီး သွား ခဲ့ တယ်",
        ...     words=["သူ", "ကား", "စီး", "သွား", "ခဲ့", "တယ်"],
        ...     word_positions=[0, 6, 12, 18, 27, 36]
        ... )
        >>> errors = strategy.validate(context)
        # AI model may detect semantic inconsistencies
    """

    # B2: Particle/function words to skip in proactive scanning.
    # These are closed-class words where MLM disagreement is noise, not signal.
    _SKIP_WORDS: frozenset[str] = frozenset(
        {
            "\u101e\u100a\u103a",  # သည်
            "\u1019\u103b\u102c\u1038",  # များ
            "\u1000\u102d\u102f",  # ကို
            "\u1010\u103d\u1004\u103a",  # တွင်
            "\u1019\u103e",  # မှ
            "\u1019\u103e\u102c",  # မှာ
            "\u104c",  # ၌
            "\u1000",  # က
            "\u1000\u103c",  # ကြ
            "\u1010\u101a\u103a",  # တယ်
            "\u1019\u101a\u103a",  # မယ်
            "\u1015\u102b",  # ပါ
            "\u1015\u103c\u102e",  # ပြီ
            "\u1001\u1032\u1037",  # ခဲ့
            "\u1014\u1031",  # နေ
            "\u101c\u102d\u102f\u1037",  # လို့
            "\u1014\u1032\u1037",  # နဲ့
            "\u1014\u103e\u1004\u103a\u1037",  # နှင့်
            "\u101e\u102d\u102f\u1037",  # သို့
            "\u1016\u103c\u1004\u103a\u1037",  # ဖြင့်
            "\u1021\u1010\u103d\u1000\u103a",  # အတွက်
            "\u101c\u100a\u103a\u1038",  # လည်း
        }
    )

    def __init__(
        self,
        semantic_checker: "SemanticChecker | None",
        use_proactive_scanning: bool | None = None,
        proactive_confidence_threshold: float | None = None,
        min_word_length: int | None = None,
        config: SemanticStrategyConfig | None = None,
        provider: "DictionaryProvider | None" = None,
    ):
        """
        Initialize semantic validation strategy.

        Args:
            semantic_checker: SemanticChecker instance with loaded AI model.
                            If None, this strategy is disabled.
            use_proactive_scanning: Enable proactive semantic scanning (default: False).
                                     Warning: Computationally expensive, increases latency.
            proactive_confidence_threshold: Minimum confidence to report semantic errors
                                          from proactive scanning (default: 0.85).
            min_word_length: Minimum word length for semantic analysis (default: 2).
            config: SemanticStrategyConfig with all thresholds. Explicit
                   kwargs override config values.
        """
        self._config = config or SemanticStrategyConfig()
        self.semantic_checker = semantic_checker
        self.provider = provider
        self.use_proactive_scanning = (
            use_proactive_scanning
            if use_proactive_scanning is not None
            else self._config.use_proactive_scanning
        )
        self.proactive_confidence_threshold = (
            proactive_confidence_threshold
            if proactive_confidence_threshold is not None
            else self._config.proactive_confidence_threshold
        )
        self.min_word_length = (
            min_word_length if min_word_length is not None else self._config.min_word_length
        )
        self.logger = get_logger(__name__)
        self._hasher = PhoneticHasher(ignore_tones=False)
        self._MAX_SEMANTIC_CHECKS_PER_SENTENCE = self._config.max_semantic_checks_per_sentence

    def validate(self, context: ValidationContext) -> list[Error]:
        """
        Validate words using AI-powered semantic analysis.

        Args:
            context: Validation context with sentence information.

        Returns:
            List of ContextError objects for semantic errors.
        """
        if not self.semantic_checker or not context:
            return []

        # Guard: Error budget -- skip proactive semantic scanning when errors
        # already detected (in this chunk OR in previous chunks).  The MLM
        # context is corrupted by existing errors, causing cascade FPs on
        # valid words.  Keep animacy detection (cheap, independent of MLM
        # context) but skip the heavy proactive scan.
        _skip_proactive = (
            len(context.existing_errors) >= self._config.error_budget_threshold
            or context.global_error_count >= self._config.error_budget_threshold
        )

        errors: list[Error] = []

        try:
            # Run proactive semantic scanning (if enabled and context not corrupted)
            if self.use_proactive_scanning and not _skip_proactive:
                semantic_errors = self._run_proactive_semantic_scan(
                    sentence=context.sentence,
                    words=context.words,
                    word_positions=context.word_positions,
                    is_name_mask=context.is_name_mask,
                    existing_error_positions=context.existing_errors,
                )
                errors.extend(semantic_errors)

            # Always run animacy detection -- it is lightweight and can
            # suppress FPs (animacy constraints are independent of MLM context
            # corruption from nearby errors).
            mlm_text = context.full_text or context.sentence
            animacy_errors = self._run_animacy_detection(
                sentence=mlm_text,
                words=context.words,
                word_positions=context.word_positions,
                is_name_mask=context.is_name_mask,
                existing_error_positions=context.existing_errors,
            )
            errors.extend(animacy_errors)

            # Contrast fallback: for unresolved tokens, compare generated
            # candidates against current word using MLM score margins.
            # Skip when context is corrupted by existing errors (same
            # rationale as proactive scan -- MLM context is unreliable).
            if not _skip_proactive:
                contrast_errors = self._run_semantic_contrast_fallback(
                    sentence=context.sentence,
                    words=context.words,
                    word_positions=context.word_positions,
                    is_name_mask=context.is_name_mask,
                    existing_error_positions=context.existing_errors,
                )
                errors.extend(contrast_errors)

        except Exception as e:
            # Semantic validation is optional enhancement -- AI model failures
            # (ONNX runtime, tensor shape, CUDA, tokenizer) should never break
            # the core validation pipeline.
            self.logger.error(f"Error in semantic validation: {e}", exc_info=True)

        return errors

    def _contrast_margin(self, word: str, frequency: int) -> float:
        """Dynamic margin for candidate-vs-current contrast decisions."""
        return contrast_margin(word, frequency, self._config)

    def _build_contrast_candidate_pool(self, word: str) -> list[str]:
        """Build generalized fallback candidate pool for one token."""
        return build_contrast_candidate_pool(
            word,
            provider=self.provider,
            hasher=self._hasher,
            config=self._config,
        )

    def _filter_and_rank_candidates(
        self,
        *,
        word: str,
        candidates: set[str],
        limit: int,
    ) -> list[str]:
        """Filter to dictionary-valid candidates and rank by frequency."""
        from myspellchecker.core.validation_strategies.semantic_helpers import (
            filter_and_rank_candidates,
        )

        return filter_and_rank_candidates(
            word=word,
            candidates=candidates,
            limit=limit,
            provider=self.provider,
        )

    def _run_semantic_contrast_fallback(
        self,
        sentence: str,
        words: list[str],
        word_positions: list[int],
        is_name_mask: list[bool],
        existing_error_positions: dict,
    ) -> list[ContextError]:
        """
        Semantic contrast fallback for unresolved tokens.

        Compares generated candidates (confusable + de-affix) against the
        original token using MLM score differences. Emits only when the best
        candidate clears a dynamic margin and structural similarity guard.
        """
        if not self.semantic_checker or not self.provider:
            return []

        errors: list[ContextError] = []

        try:
            semantic_call_count = 0
            for i, word in enumerate(words):
                if i >= len(word_positions) or i >= len(is_name_mask):
                    continue

                position = word_positions[i]
                if position in existing_error_positions:
                    continue
                if is_name_mask[i]:
                    continue
                if word in self._SKIP_WORDS:
                    continue
                if len(word) < self.min_word_length:
                    continue

                # Skip words already semantically analyzed by
                # ConfusableSemanticStrategy (logits cache hit means
                # the confusable strategy checked and found it clean).
                occurrence = sum(1 for w_idx in range(i) if words[w_idx] == word)
                if hasattr(self.semantic_checker, "has_cached_logits"):
                    if self.semantic_checker.has_cached_logits(sentence, word, occurrence):
                        continue

                current_freq = 0
                if hasattr(self.provider, "get_word_frequency"):
                    current_value = self.provider.get_word_frequency(word)
                    if isinstance(current_value, (int, float)):
                        current_freq = int(current_value)
                is_valid_current = self.provider.is_valid_word(word)

                # Avoid aggressive fallback on high-frequency stable words.
                if is_valid_current and current_freq >= self._config.scan_freq_threshold:
                    continue
                # Precision guard: for valid dictionary words, only run contrast
                # fallback in low-frequency zones where confusion risk is higher.
                if is_valid_current and len(word) > 2 and current_freq >= 5_000:
                    continue
                # Short words are heavily ambiguous; keep an upper frequency cap
                # but allow semantic contrast for non-extreme-frequency forms.
                if is_valid_current and len(word) <= 2 and current_freq >= 200_000:
                    continue

                candidate_pool = self._build_contrast_candidate_pool(word)
                if not candidate_pool:
                    continue

                # Per-sentence cap: stop checking after reaching the limit.
                if semantic_call_count >= self._MAX_SEMANTIC_CHECKS_PER_SENTENCE:
                    break

                semantic_call_count += 1
                predictions = self.semantic_checker.predict_mask(
                    sentence,
                    word,
                    top_k=self._config.contrast_top_k,
                    occurrence=occurrence,
                )
                if not predictions:
                    continue

                pred_map: dict[str, float] = {pred_word: score for pred_word, score in predictions}
                if not pred_map:
                    continue

                explicit_scores: dict[str, float] = {}
                if hasattr(self.semantic_checker, "score_mask_candidates"):
                    try:
                        explicit_scores = self.semantic_checker.score_mask_candidates(
                            sentence,
                            word,
                            [word, *candidate_pool],
                            occurrence=occurrence,
                        )
                    except Exception as e:
                        self.logger.debug(f"Explicit candidate scoring failed: {e}")

                min_score = min(pred_map.values())
                current_score = explicit_scores.get(word, pred_map.get(word, min_score))
                margin = self._contrast_margin(word, current_freq)
                if is_valid_current:
                    margin += self._config.margin_boost_escalation

                best_candidate = ""
                best_diff = 0.0
                for candidate in candidate_pool:
                    candidate_score = explicit_scores.get(candidate, pred_map.get(candidate))
                    if candidate_score is None:
                        continue
                    diff = float(candidate_score) - float(current_score)
                    if diff < margin:
                        continue
                    similarity = char_overlap_similarity(word, candidate)
                    required_similarity = (
                        0.5 if is_valid_current else self._config.contrast_min_similarity
                    )
                    if similarity < required_similarity:
                        continue
                    if diff > best_diff:
                        best_diff = diff
                        best_candidate = candidate

                active_predictions = predictions
                if not best_candidate and should_run_escalation_pass(
                    word=word,
                    current_freq=current_freq,
                    is_valid_current=is_valid_current,
                    config=self._config,
                ):
                    escalation_pool = build_escalation_candidate_pool(
                        word,
                        base_pool=set(candidate_pool),
                        provider=self.provider,
                        config=self._config,
                    )
                    if escalation_pool:
                        escalation_predictions = self.semantic_checker.predict_mask(
                            sentence,
                            word,
                            top_k=self._config.escalation_top_k,
                            occurrence=occurrence,
                        )
                        if escalation_predictions:
                            escalation_map = {
                                pred_word: score for pred_word, score in escalation_predictions
                            }
                            escalation_scores: dict[str, float] = {}
                            if hasattr(self.semantic_checker, "score_mask_candidates"):
                                try:
                                    escalation_scores = self.semantic_checker.score_mask_candidates(
                                        sentence,
                                        word,
                                        [word, *escalation_pool],
                                        occurrence=occurrence,
                                    )
                                except Exception as e:
                                    self.logger.debug(f"Escalation candidate scoring failed: {e}")

                            escalation_min = min(escalation_map.values())
                            escalation_current = escalation_scores.get(
                                word,
                                escalation_map.get(
                                    word,
                                    current_score if pred_map else escalation_min,
                                ),
                            )
                            escalation_margin = max(
                                0.75, margin - self._config.escalation_margin_relax
                            )
                            if is_valid_current:
                                escalation_margin += self._config.escalation_margin_boost

                            escalation_best = ""
                            escalation_best_diff = 0.0
                            for candidate in escalation_pool:
                                candidate_score = escalation_scores.get(
                                    candidate, escalation_map.get(candidate)
                                )
                                if candidate_score is None:
                                    continue
                                diff = float(candidate_score) - float(escalation_current)
                                if diff < escalation_margin:
                                    continue
                                similarity = char_overlap_similarity(word, candidate)
                                required_similarity = (
                                    self._config.escalation_min_similarity
                                    if is_valid_current
                                    else self._config.contrast_min_similarity
                                )
                                if similarity < required_similarity:
                                    continue
                                if diff > escalation_best_diff:
                                    escalation_best_diff = diff
                                    escalation_best = candidate

                            if escalation_best:
                                best_candidate = escalation_best
                                best_diff = escalation_best_diff
                                active_predictions = escalation_predictions

                if not best_candidate:
                    continue

                confidence = min(
                    self._config.proactive_confidence_cap,
                    self._config.proactive_confidence_base
                    + best_diff / self._config.proactive_confidence_divisor,
                )
                fallback_suggestions = [best_candidate]
                for pred_word, _pred_score in active_predictions:
                    if pred_word in fallback_suggestions or pred_word == word:
                        continue
                    if len(pred_word) < 2:
                        continue
                    fallback_suggestions.append(pred_word)
                    if len(fallback_suggestions) >= 5:
                        break

                errors.append(
                    ContextError(
                        text=word,
                        position=position,
                        error_type=ET_SEMANTIC_ERROR,
                        suggestions=fallback_suggestions,
                        confidence=confidence,
                        probability=0.0,
                        prev_word=words[i - 1] if i > 0 else "",
                    )
                )
                existing_error_positions[position] = ET_SEMANTIC_ERROR

        except Exception as e:
            self.logger.warning(f"Semantic contrast fallback failed: {e}", exc_info=True)

        return errors

    # ── Animacy detection ──────────────────────────────────────────────

    # Subject/topic particles that indicate the preceding word is in subject position.
    _SUBJECT_PARTICLES = frozenset(
        {
            "\u1000",
            "\u1000\u102d\u102f",
            "\u101e\u100a\u103a",
            "\u1019\u103e\u102c",
            "\u1010\u103d\u1004\u103a",
        }
    )

    # NP-internal modifiers that can appear between head noun and topic particle.
    # When a subject particle is found, we look backwards through these to find
    # the head noun. E.g., ငါးများသည် -> [ငါး, များ, သည်] -- skip များ to find ငါး.
    _NP_MODIFIERS = frozenset(
        {
            "\u1019\u103b\u102c\u1038",  # များ (plural marker)
            "\u1010\u102d\u102f\u1037",  # တို့ (plural/collective)
            "\u1000\u103c\u102e\u1038",  # ကြီး (big -- common adjectival modifier)
            "\u1004\u101a\u103a",  # ငယ် (small)
        }
    )

    # Common person/pronoun words in Myanmar. When the model's top predictions
    # are predominantly from this set, the context requires an animate agent --
    # a non-person subject indicates semantic implausibility.
    _PERSON_WORDS = frozenset(
        {
            "\u101e\u1030",  # သူ
            "\u1000\u103b\u103d\u1014\u103a\u1010\u1031\u102c\u103a",  # ကျွန်တော်
            "\u1000\u103b\u103d\u1014\u103a\u1019",  # ကျွန်မ
            "\u1004\u102b",  # ငါ
            "\u1019\u1004\u103a\u1038",  # မင်း
            "\u1021\u1019\u1031",  # အမေ
            "\u1021\u1016\u1031",  # အဖေ
            "\u100a\u102e\u1019",  # ညီမ
            "\u100a\u102e",  # ညီ
            "\u1021\u1018\u103d\u102c\u1038",  # အဘွား
            "\u1021\u1016\u103d\u102c\u1038",  # အဖွား
            "\u1019\u1031\u1019\u1031",  # မေမေ
            "\u1016\u1031\u1016\u1031",  # ဖေဖေ
            "\u101e\u1030\u1019",  # သူမ
            "\u1021\u1019",  # အမ
            "\u1006\u101b\u102c",  # ဆရာ
            "\u1006\u101b\u102c\u1019",  # ဆရာမ
            "\u1000\u101c\u1031\u1038",  # ကလေး
            "\u101c\u1030",  # လူ
            "\u1019\u102d\u1001\u1004\u103a",  # မိခင်
            "\u1016\u1001\u1004\u103a",  # ဖခင်
            "\u101e\u1019\u102e\u1038",  # သမီး
            "\u101e\u102c\u1038",  # သား
            "\u1012\u102b",  # ဒါ
            "\u101f\u102c",  # ဟာ (demonstratives that substitute for persons)
        }
    )

    # Minimum ratio of person-word predictions to flag animacy mismatch.
    # Now configurable via SemanticStrategyConfig.person_prediction_threshold.
    # 4/5 separates true implausibility (inanimate subjects: 5/5) from
    # animate non-person subjects (e.g., ခွေး "dog": 3/5).

    # Expanded person words -- first tokens from multi-token MLM predictions.
    # These are common animate/person words the model predicts in subject position.
    # Used both for: (1) counting person predictions, (2) skipping already-animate subjects.
    _PERSON_WORDS_EXTENDED = _PERSON_WORDS | frozenset(
        {
            # Occupational/role terms
            "\u101c\u1030\u1004\u101a\u103a",  # လူငယ်
            "\u1000\u103b\u1031\u102c\u1004\u103a\u1038\u101e\u102c\u1038",  # ကျောင်းသား
            "\u1010\u1031\u102c\u1004\u103a\u101e\u1030",  # တောင်သူ
            "\u101c\u101a\u103a\u101e\u1019\u102c\u1038",  # လယ်သမား
            "\u101b\u1032\u1018\u1031\u102c\u103a",  # ရဲဘော်
            "\u1021\u101c\u102f\u1015\u103a\u101e\u1019\u102c\u1038",  # အလုပ်သမား
            "\u1005\u1005\u103a\u101e\u102c\u1038",  # စစ်သား
            "\u101b\u103d\u102c\u101e\u102c\u1038",  # ရွာသား
            "\u101e\u102f\u1010\u1031\u101e\u102e",  # သုတေသီ (researcher)
            "\u1005\u102c\u101b\u1031\u1038\u1006\u101b\u102c",  # စာရေးဆရာ (writer)
            # Group/collective person terms
            "\u1015\u101b\u102d\u101e\u1010\u103a",  # ပရိသတ်
            "\u1015\u101b\u102d\u1010\u103a\u101e\u1010\u103a",  # ပရိတ်သတ် (spelling variant)
            "\u1019\u102d\u101e\u102c\u1038\u1005\u102f",  # မိသားစု
            "\u101c\u1030\u1021\u102f\u1015\u103a",  # လူအုပ် (crowd/group of people)
            # Kinship (extended)
            "\u1021\u1018\u102d\u102f\u1038",  # အဘိုး (grandfather)
            # Titles/royalty
            "\u1018\u102f\u101b\u1004\u103a",  # ဘုရင် (king)
            # Residential/locational persons
            "\u1012\u1031\u101e\u1001\u1036",  # ဒေသခံ (local resident)
        }
    )

    # Animate non-person subjects to skip in animacy detection.
    # Animals are animate and naturally occur in subject position with verbs
    # that typically take human subjects. MLM predictions for masked animal
    # subjects are dominated by person words, but flagging them is a false positive.
    _ANIMATE_SKIP_SUBJECTS: frozenset[str] = frozenset(
        {
            "\u1001\u103d\u1031\u1038",  # ခွေး (dog)
            "\u1000\u103c\u1031\u102c\u1004\u103a",  # ကြောင် (cat)
            "\u1019\u103c\u1004\u103a\u1038",  # မြင်း (horse)
            "\u1014\u103d\u102c\u1038",  # နွား (cow)
            "\u1006\u1004\u103a",  # ဆင် (elephant)
            "\u1000\u103b\u103d\u1032",  # ကျွဲ (buffalo)
            "\u1004\u103e\u1000\u103a",  # ငှက် (bird)
            "\u1000\u103c\u1000\u103a",  # ကြက် (chicken)
            "\u101d\u1000\u103a",  # ဝက် (pig)
            "\u1006\u102d\u1010\u103a",  # ဆိတ် (goat)
            "\u101e\u102d\u102f\u1038",  # သိုး (sheep)
            "\u1019\u103b\u1031\u102c\u1000\u103a",  # မျောက် (monkey)
        }
    )

    def _run_proactive_semantic_scan(
        self,
        sentence: str,
        words: list[str],
        word_positions: list[int],
        is_name_mask: list[bool],
        existing_error_positions: dict,
    ) -> list[ContextError]:
        """
        Broad proactive semantic scan using scan_sentence().

        Masks each content word and checks if the model predicts it among
        top-K candidates. Words not predicted by the model are flagged as
        potential real-word errors with the model's top suggestions.

        Skips positions already flagged by earlier strategies and named entities.
        """
        if not self.semantic_checker:
            return []

        errors: list[ContextError] = []

        try:
            # Use scan_sentence() for broad scanning
            scan_results = self.semantic_checker.scan_sentence(
                sentence=sentence,
                words=words,
                min_word_len=self.min_word_length,
                confidence_threshold=self.proactive_confidence_threshold,
            )

            for word_idx, error_word, suggestions, confidence in scan_results:
                # Bounds check
                if word_idx >= len(word_positions) or word_idx >= len(is_name_mask):
                    continue

                abs_pos = word_positions[word_idx]

                # Skip if already flagged by an earlier strategy
                if abs_pos in existing_error_positions:
                    continue

                # Skip named entities
                if is_name_mask[word_idx]:
                    continue

                # B2: Skip closed-class particles/function words
                if error_word in self._SKIP_WORDS:
                    continue

                # B1: Skip high-frequency words (common words are rarely errors)
                if self.provider and hasattr(self.provider, "get_word_frequency"):
                    freq = self.provider.get_word_frequency(error_word)
                    if isinstance(freq, (int, float)) and freq >= self._config.scan_freq_threshold:
                        continue

                errors.append(
                    ContextError(
                        text=error_word,
                        position=abs_pos,
                        error_type=ET_SEMANTIC_ERROR,
                        suggestions=suggestions[:5],
                        confidence=confidence,
                        probability=0.0,
                        prev_word=words[word_idx - 1] if word_idx > 0 else "",
                    )
                )
                existing_error_positions[abs_pos] = ET_SEMANTIC_ERROR

        except Exception as e:
            self.logger.warning(f"Semantic scan failed: {e}", exc_info=True)

        return errors

    def _run_animacy_detection(
        self,
        sentence: str,
        words: list[str],
        word_positions: list[int],
        is_name_mask: list[bool],
        existing_error_positions: dict,
    ) -> list[ContextError]:
        """
        Detect animacy mismatches in subject position.

        Finds words followed by subject/topic particles (သည်, က, etc.),
        masks them, and checks if the model overwhelmingly predicts
        person/animate words. If so, the current (inanimate) subject is
        flagged as semantically implausible.

        Examples caught: "စားပွဲကြီးသည် ဘောလုံးပွဲကို ကြည့်ရှုပြီး..."
        (tables cannot watch football)
        """
        if not self.semantic_checker:
            return []

        errors: list[ContextError] = []

        try:
            for i in range(len(words) - 1):
                # Skip already-flagged positions and named entities
                if word_positions[i] in existing_error_positions:
                    continue
                if is_name_mask[i]:
                    continue

                subject_word = words[i]

                # Skip if subject is already a person/animate word
                if subject_word in self._PERSON_WORDS_EXTENDED:
                    continue
                # Skip animate non-person subjects (animals)
                if subject_word in self._ANIMATE_SKIP_SUBJECTS:
                    continue

                # Skip short words (particles, single syllables)
                if len(subject_word) < self.min_word_length:
                    continue

                # Check if a subject particle appears within the next 1-3 words,
                # with only NP modifiers (များ, တို့, ကြီး, etc.) in between.
                # E.g., [ငါး, များ, သည်] or [စားပွဲ, ကြီး, သည်]
                has_subject_particle = False
                for j in range(i + 1, min(i + 4, len(words))):
                    if words[j] in self._SUBJECT_PARTICLES:
                        has_subject_particle = True
                        break
                    if words[j] not in self._NP_MODIFIERS:
                        break  # non-modifier stops the search

                if not has_subject_particle:
                    continue

                # Mask the subject word and get predictions
                predictions = self.semantic_checker.predict_mask(sentence, subject_word, top_k=10)
                if not predictions:
                    continue

                # Extract first tokens from predictions (for multi-token outputs)
                first_tokens = []
                seen = set()
                for pred_word, _score in predictions:
                    # Get first token/word (before space or combining chars)
                    first = pred_word.split()[0] if " " in pred_word else pred_word
                    if first not in seen:
                        seen.add(first)
                        first_tokens.append(first)
                    if len(first_tokens) >= 5:
                        break

                # Count person-word predictions
                person_count = sum(1 for t in first_tokens if t in self._PERSON_WORDS_EXTENDED)

                if person_count >= self._config.person_prediction_threshold:
                    # Model strongly expects a person/animate subject here
                    person_suggestions = [
                        t for t in first_tokens if t in self._PERSON_WORDS_EXTENDED
                    ][:3]
                    errors.append(
                        ContextError(
                            text=subject_word,
                            position=word_positions[i],
                            error_type=ET_SEMANTIC_ERROR,
                            suggestions=person_suggestions,
                            confidence=min(0.95, person_count / len(first_tokens)),
                            probability=0.0,
                            prev_word=words[i - 1] if i > 0 else "",
                        )
                    )
                    existing_error_positions[word_positions[i]] = ET_SEMANTIC_ERROR

        except Exception as e:
            self.logger.warning(f"Animacy detection failed: {e}", exc_info=True)

        return errors

    def priority(self) -> int:
        """
        Return strategy execution priority.

        Returns:
            70 (runs last - AI validation is computationally expensive)
        """
        return 70

    def close(self) -> None:
        """Release the underlying SemanticChecker inference session."""
        if self.semantic_checker and hasattr(self.semantic_checker, "close"):
            self.semantic_checker.close()

    def __repr__(self) -> str:
        """String representation."""
        if not self.semantic_checker:
            status = "disabled (no model)"
        elif not self.use_proactive_scanning:
            status = "disabled (proactive scanning off)"
        else:
            status = f"enabled (threshold={self.proactive_confidence_threshold})"
        return f"SemanticValidationStrategy(priority={self.priority()}, {status})"
