"""
ContextValidator (Strategy Pattern Orchestrator).

This module implements the strategy-based ContextValidator that
coordinates validation strategies instead of implementing validation
logic directly.
"""

from __future__ import annotations

import copy
import threading
import time
from typing import TYPE_CHECKING, Any

from myspellchecker.core.calibration import StrategyCalibrator
from myspellchecker.core.config import SpellCheckerConfig
from myspellchecker.core.response import Error
from myspellchecker.core.token_refinement import build_validation_token_paths
from myspellchecker.core.validation_strategies import (
    ErrorCandidate,
    ValidationContext,
    ValidationStrategy,
)
from myspellchecker.core.validation_strategies.arbiter import (
    arbitrate_candidates,
    fuse_all_candidates,
)
from myspellchecker.core.validators import Validator
from myspellchecker.segmenters.base import Segmenter
from myspellchecker.text.ner import NameHeuristic
from myspellchecker.utils.logging_utils import get_logger

if TYPE_CHECKING:
    from myspellchecker.text.ner_model import NERModel

logger = get_logger(__name__)

__all__ = [
    "ContextValidator",
]


class ContextValidator(Validator):
    """
    Strategy-based context validator.

    A lightweight orchestrator that coordinates focused validation strategies
    using the Strategy Pattern.

    **Interface Segregation**:
        This orchestrator does not depend on any dictionary provider directly.
        Each validation strategy receives only the interfaces it needs:
        - NgramContextValidationStrategy: NgramRepository (get_bigram/trigram_probability)
        - HomophoneValidationStrategy: NgramRepository
        - POSSequenceValidationStrategy: POSRepository (optional)
        This follows the Interface Segregation Principle (ISP).

    **Validation Flow**:
        1. Segment text into sentences and words
        2. Build ValidationContext with word positions and metadata
        3. Execute strategies in priority order:
           - Priority 10: ToneValidationStrategy (tone disambiguation)
           - Priority 15: OrthographyValidationStrategy (medial order/compat)
           - Priority 20: SyntacticValidationStrategy (grammar rules)
           - Priority 25: BrokenCompoundStrategy (broken compound detection)
           - Priority 30: POSSequenceValidationStrategy (POS patterns)
           - Priority 40: QuestionStructureValidationStrategy (question particles)
           - Priority 45: HomophoneValidationStrategy (homophone detection)
           - Priority 48: ConfusableSemanticStrategy (MLM confusable detection)
           - Priority 50: NgramContextValidationStrategy (statistical context)
           - Priority 70: SemanticValidationStrategy (AI-powered MLM)
        4. Aggregate and deduplicate errors

    **Benefits**:
        - Clear separation of concerns
        - Easy to add/remove validation strategies
        - Each strategy can be tested independently
        - Reduced complexity and improved maintainability
    """

    def __init__(
        self,
        config: SpellCheckerConfig,
        segmenter: Segmenter,
        strategies: list[ValidationStrategy] | None = None,
        name_heuristic: NameHeuristic | None = None,
        ner_model: "NERModel" | None = None,
        viterbi_tagger: object | None = None,
    ):
        """
        Initialize context validator with validation strategies.

        Args:
            config: SpellChecker configuration.
            segmenter: Text segmenter for sentence and word tokenization.
            strategies: List of validation strategies to execute.
                       If None, validator will not perform any validation.
                       Each strategy should be pre-configured with its required
                       interfaces (e.g., NgramRepository for n-gram strategies).
            name_heuristic: Optional name heuristic for proper name detection.
            ner_model: Optional NERModel for more accurate entity detection.
                      When provided, used to build is_name_mask from PER entities
                      instead of relying solely on the simpler NameHeuristic.

        Example:
            >>> from myspellchecker.core.validation_strategies import (
            ...     ToneValidationStrategy,
            ...     SyntacticValidationStrategy,
            ... )
            >>> strategies = [
            ...     ToneValidationStrategy(tone_disambiguator),
            ...     SyntacticValidationStrategy(syntactic_checker),
            ... ]
            >>> validator = ContextValidator(config, segmenter, strategies)
        """
        super().__init__(config)
        self.segmenter = segmenter
        self.name_heuristic = name_heuristic
        self.ner_model: "NERModel" | None = ner_model
        self._viterbi_tagger = viterbi_tagger
        self.strategy_timings: dict[str, float] = {}
        self._timing_lock = threading.Lock()
        self._debug_lock = threading.Lock()
        self._last_strategy_debug_telemetry: dict[str, Any] = {}

        # Sort strategies by priority (lower values run first)
        # Use sorted() to avoid modifying caller's list
        self.strategies = sorted(strategies or [], key=lambda s: s.priority())
        self._fusion_enabled = config.validation.use_candidate_fusion
        self._fusion_threshold = config.validation.fusion_confidence_threshold
        if self._fusion_enabled:
            cal_path = config.validation.calibration_path
            if cal_path and isinstance(cal_path, str):
                self._calibrator = StrategyCalibrator.from_yaml(cal_path)
            else:
                self._calibrator = StrategyCalibrator()
        else:
            self._calibrator = None
        self._refine_is_valid_word = self._resolve_word_validator_callback("is_valid_word")
        self._refine_get_word_frequency = self._resolve_word_validator_callback(
            "get_word_frequency"
        )
        self._refine_bigram_probability = self._resolve_bigram_probability_callback()

        logger.info(
            f"ContextValidator initialized with {len(self.strategies)} strategies: "
            f"{[s.__class__.__name__ for s in self.strategies]}"
        )

    def _resolve_word_validator_callback(self, method_name: str) -> Any | None:
        """Resolve word repository callback from attached strategies, if available."""
        for strategy in self.strategies:
            provider = getattr(strategy, "provider", None)
            callback = getattr(provider, method_name, None)
            if callable(callback):
                return callback
        return None

    def _resolve_bigram_probability_callback(self) -> Any | None:
        """Resolve bigram probability callback for refinement scoring."""
        for strategy in self.strategies:
            context_checker = getattr(strategy, "context_checker", None)
            callback = getattr(context_checker, "get_bigram_probability", None)
            if callable(callback):
                return callback

            provider = getattr(strategy, "provider", None)
            callback = getattr(provider, "get_bigram_probability", None)
            if callable(callback):
                return callback
        return None

    def validate(
        self, text: str, *, exclude_strategy_types: frozenset[type] | None = None
    ) -> list[Error]:
        """
        Validate text using registered validation strategies.

        Args:
            text: Text to validate.
            exclude_strategy_types: Optional set of strategy classes to skip.
                Used to disable semantic checking when ``use_semantic=False``.

        Returns:
            List of Error objects aggregated from all strategies.
        """
        if not self.strategies:
            with self._debug_lock:
                self._last_strategy_debug_telemetry = {}
            logger.info("No validation strategies configured, skipping context validation")
            return []

        errors: list[Error] = []
        strategy_debug_enabled = self.config.validation.enable_strategy_debug
        strategy_debug_telemetry: dict[str, Any] | None = (
            {"enabled": True, "strategies": {}} if strategy_debug_enabled else None
        )

        # Segment text into sentences
        sentences = self.segmenter.segment_sentences(text)
        current_text_offset = 0
        global_error_count = 0

        for sentence in sentences:
            if not sentence:
                continue

            # Find sentence position in original text
            sent_start = text.find(sentence, current_text_offset)
            if sent_start == -1:
                # Sentence not found from current offset. This can happen when the
                # segmenter normalizes whitespace or modifies sentence boundaries.
                # Do NOT fall back to text.find(sentence) from beginning — that can
                # match the wrong instance of a repeated sentence, producing incorrect
                # error positions in the UI.
                logger.warning(
                    f"Sentence not found in text from offset {current_text_offset}, "
                    f"skipping: {sentence[:50]!r}"
                )
                # Advance offset heuristically to avoid cascading misalignment.
                # Use 1 instead of len(sentence) to minimize skip distance.
                current_text_offset += 1
                continue
            current_text_offset = sent_start + len(sentence)

            # Validate this sentence using all strategies
            sentence_errors = self._validate_sentence(
                sentence,
                sent_start,
                full_text=text,
                strategy_debug_telemetry=strategy_debug_telemetry,
                global_error_count=global_error_count,
                exclude_strategy_types=exclude_strategy_types,
            )
            errors.extend(sentence_errors)
            global_error_count += len(sentence_errors)

        with self._debug_lock:
            self._last_strategy_debug_telemetry = (
                strategy_debug_telemetry if strategy_debug_telemetry is not None else {}
            )

        return errors

    def _validate_sentence(
        self,
        sentence: str,
        sentence_offset: int,
        full_text: str = "",
        strategy_debug_telemetry: dict[str, Any] | None = None,
        global_error_count: int = 0,
        exclude_strategy_types: frozenset[type] | None = None,
    ) -> list[Error]:
        """
        Validate a single sentence using all registered strategies.

        Args:
            sentence: Sentence text to validate.
            sentence_offset: Position of sentence in original text.

        Returns:
            List of Error objects from all strategies.
        """
        # Segment sentence into words
        words = self.segmenter.segment_words(sentence)
        token_paths = build_validation_token_paths(
            words,
            is_valid_word=self._refine_is_valid_word,
            get_word_frequency=self._refine_get_word_frequency,
            get_bigram_probability=self._refine_bigram_probability,
            segment_syllables=getattr(self.segmenter, "segment_syllables", None),
        )
        if not token_paths:
            return []

        errors: list[Error] = []
        seen_error_keys: set[tuple[int, str, str]] = set()
        path_existing_errors: dict[int, str] = {}
        path_existing_suggestions: dict[int, list[str]] = {}
        path_existing_confidences: dict[int, float] = {}
        path_error_candidates: dict[int, list[ErrorCandidate]] = {}

        path_iteration = token_paths[1:] + token_paths[:1] if len(token_paths) > 1 else token_paths
        for path_words in path_iteration:
            context = self._build_validation_context(
                sentence=sentence,
                sentence_offset=sentence_offset,
                full_text=full_text,
                words=path_words,
            )
            if context is None:
                continue

            # Propagate global error count from previous sentences
            context.global_error_count = global_error_count

            # Carry errors/suggestions across lattice paths so path-B can still
            # emit new positions while avoiding duplicate emissions on same span.
            context.existing_errors.update(path_existing_errors)
            context.existing_suggestions.update(path_existing_suggestions)
            context.existing_confidences.update(path_existing_confidences)
            for pos, candidates in path_error_candidates.items():
                context.error_candidates.setdefault(pos, []).extend(candidates)

            strategy_errors = self._execute_strategies(
                context=context,
                strategy_debug_telemetry=strategy_debug_telemetry,
                exclude_strategy_types=exclude_strategy_types,
            )
            for error in strategy_errors:
                key = (error.position, error.text, error.error_type)
                if key in seen_error_keys:
                    continue
                seen_error_keys.add(key)
                errors.append(error)

            path_existing_errors.update(context.existing_errors)
            path_existing_suggestions.update(context.existing_suggestions)
            path_existing_confidences.update(context.existing_confidences)
            for pos, candidates in context.error_candidates.items():
                path_error_candidates.setdefault(pos, []).extend(candidates)

        if path_existing_suggestions:
            merged_context = ValidationContext(
                sentence=sentence,
                words=[],
                word_positions=[],
                full_text=full_text,
            )
            merged_context.existing_suggestions = path_existing_suggestions
            self._merge_appended_suggestions(errors, merged_context)

        # Arbiter: for positions with multiple candidates, let the
        # higher-tier strategy win.  Replace the mutex-selected error
        # with the arbiter's choice when they disagree.
        if path_error_candidates:
            if self._fusion_enabled and self._calibrator is not None:
                # Full fusion mode: calibrate + cluster + Noisy-OR merge
                fused = fuse_all_candidates(
                    path_error_candidates,
                    self._calibrator,
                    threshold=self._fusion_threshold,
                )
                self._apply_fusion_winners(errors, fused)
                # Suppress context-strategy errors whose fused confidence fell
                # below the threshold.  Only suppress errors with a source_strategy
                # (context-level); word/syllable errors (empty source_strategy)
                # must survive even if a context candidate at the same position
                # was rejected.
                rejected = {pos for pos in path_error_candidates if pos not in fused}
                if rejected:
                    errors[:] = [
                        e for e in errors if e.position not in rejected or not e.source_strategy
                    ]
            else:
                # Shadow mode: tier-based arbiter, log-only divergence
                winners = arbitrate_candidates(path_error_candidates)
                if winners:
                    self._apply_arbiter_winners(errors, winners)

        return errors

    def _build_validation_context(
        self,
        sentence: str,
        sentence_offset: int,
        full_text: str,
        words: list[str],
    ) -> ValidationContext | None:
        """Build filtered ValidationContext for one tokenization path."""
        if not words:
            return None

        # Detect proper names using NERModel or NameHeuristic
        is_name_mask: list[bool]
        if self.config.use_ner and self.ner_model:
            is_name_mask = self._build_name_mask_from_ner(sentence, words)
        elif self.config.use_ner and self.name_heuristic:
            is_name_mask = self.name_heuristic.analyze_sentence(words)
        else:
            is_name_mask = [False] * len(words)
        if len(is_name_mask) < len(words):
            is_name_mask = list(is_name_mask) + [False] * (len(words) - len(is_name_mask))
        elif len(is_name_mask) > len(words):
            is_name_mask = list(is_name_mask[: len(words)])

        filtered_words: list[str] = []
        filtered_positions: list[int] = []
        filtered_is_name: list[bool] = []

        word_cursor = 0
        for i, word in enumerate(words):
            if not word:
                continue

            w_idx = sentence.find(word, word_cursor)
            if w_idx == -1:
                logger.debug(f"Word not found in sentence from cursor {word_cursor}: {word}")
                continue

            abs_position = sentence_offset + w_idx
            word_cursor = w_idx + len(word)

            is_valid_word = (
                word.strip()
                and not self.is_punctuation(word)
                and self._is_myanmar_with_config(word)
            )
            if is_valid_word:
                filtered_words.append(word)
                filtered_positions.append(abs_position)
                filtered_is_name.append(is_name_mask[i])

        if not filtered_words:
            return None

        context = ValidationContext(
            sentence=sentence,
            words=filtered_words,
            word_positions=filtered_positions,
            is_name_mask=filtered_is_name,
            full_text=full_text,
        )

        if self._viterbi_tagger and len(filtered_words) >= 2:
            try:
                tags = self._viterbi_tagger.tag_sequence(filtered_words)
                if len(tags) == len(filtered_words):
                    context.pos_tags = [t.upper() for t in tags]
            except (RuntimeError, ValueError, TypeError) as e:
                logger.debug(f"POS pre-computation failed, strategies will compute own: {e}")

        return context

    # Strategies at or below this priority are "structural" (Layer 1-2) and
    # always run.  Higher-priority strategies (context, grammar, semantic) are
    # skipped on sentences where Layer 1-2 found zero issues — the "fast path".
    # This dramatically reduces FPR on clean text because most sentences have
    # no structural errors, and the contextual strategies are the main FP source.
    _FAST_PATH_PRIORITY_CUTOFF: int = 25  # Tone(10), Ortho(15), Syntactic(20), Compound(25)

    def _execute_strategies(
        self,
        *,
        context: ValidationContext,
        strategy_debug_telemetry: dict[str, Any] | None,
        exclude_strategy_types: frozenset[type] | None = None,
    ) -> list[Error]:
        """Execute all strategies for one context.

        Fast-path optimization: if structural strategies (priority <= 25) find
        no errors, higher-priority contextual strategies are skipped. This
        reduces FPR on clean sentences without affecting recall on sentences
        that have structural issues.
        """
        errors: list[Error] = []
        timing_enabled = self.config.validation.enable_strategy_timing
        structural_phase_done = False
        structural_phase_ran = False

        for strategy in self.strategies:
            if exclude_strategy_types and type(strategy) in exclude_strategy_types:
                continue

            # Fast-path exit: if structural strategies found nothing, skip
            # contextual strategies (POS, Homophone, Confusable, Ngram, Semantic).
            # Only fires when fast-path is enabled and at least one structural
            # strategy actually ran (not just skipped via exclude_strategy_types).
            if not structural_phase_done and strategy.priority() > self._FAST_PATH_PRIORITY_CUTOFF:
                structural_phase_done = True
                if (
                    self.config.validation.enable_fast_path
                    and not self._fusion_enabled
                    and structural_phase_ran
                    and not errors
                    and not context.existing_errors
                ):
                    logger.debug(
                        "Fast-path: structural strategies found no issues, "
                        "skipping contextual strategies"
                    )
                    break
            if not structural_phase_done:
                structural_phase_ran = True
            strategy_name = strategy.__class__.__name__
            pre_positions: set[int] = set()
            pre_error_types: dict[int, str] = {}
            shadow_semantic_positions: set[int] | None = None
            if strategy_debug_telemetry is not None:
                pre_positions = set(context.existing_errors.keys())
                pre_error_types = dict(context.existing_errors)
                if strategy_name == "SemanticValidationStrategy":
                    shadow_semantic_positions = self._collect_semantic_shadow_positions(
                        strategy, context
                    )

            try:
                if timing_enabled:
                    t0 = time.perf_counter()

                strategy_errors = strategy.validate(context)
                for error in strategy_errors:
                    error.source_strategy = strategy_name
                errors.extend(strategy_errors)

                # Collect error candidates for arbiter (Phase 2 infrastructure).
                # Candidates are emitted alongside the mutex -- both systems
                # active, mutex still determines output.
                for error in strategy_errors:
                    candidate = ErrorCandidate(
                        strategy_name=strategy_name,
                        error_type=error.error_type,
                        confidence=getattr(error, "confidence", 0.0),
                        suggestion=error.suggestions[0] if error.suggestions else None,
                        evidence=strategy_name,
                    )
                    context.error_candidates.setdefault(error.position, []).append(candidate)

                if strategy_debug_telemetry is not None:
                    self._update_strategy_debug_telemetry(
                        strategy_debug_telemetry,
                        strategy_name=strategy_name,
                        strategy_errors=strategy_errors,
                        pre_positions=pre_positions,
                        pre_error_types=pre_error_types,
                        context=context,
                        shadow_semantic_positions=shadow_semantic_positions,
                    )

                if timing_enabled:
                    elapsed = time.perf_counter() - t0
                    with self._timing_lock:
                        self.strategy_timings[strategy_name] = (
                            self.strategy_timings.get(strategy_name, 0.0) + elapsed
                        )
                    logger.debug(f"{strategy_name} took {elapsed:.4f}s")

                logger.debug(
                    f"{strategy_name} found {len(strategy_errors)} errors (total: {len(errors)})"
                )

            except (RuntimeError, ValueError, TypeError, KeyError, IndexError, AttributeError) as e:
                logger.error(f"Strategy {strategy_name} failed: {e}", exc_info=True)
                if self.config.validation.raise_on_strategy_error:
                    raise

        return errors

    @staticmethod
    def _merge_appended_suggestions(
        errors: list[Error],
        context: ValidationContext,
    ) -> None:
        """Merge suggestions appended by lower-priority strategies into Error objects.

        Lower-priority strategies may compute suggestions for positions already
        flagged by higher-priority strategies.  Those suggestions are stored in
        ``context.existing_suggestions`` but are not yet reflected in the Error
        objects returned to the caller.  This method reconciles the two.

        Args:
            errors: Collected Error objects from all strategies.
            context: The shared ValidationContext with ``existing_suggestions``.
        """
        # Build position → Error lookup for O(1) access
        error_by_pos: dict[int, Error] = {}
        for error in errors:
            # First error at each position wins (highest priority)
            if error.position not in error_by_pos:
                error_by_pos[error.position] = error

        for pos, all_suggestions in context.existing_suggestions.items():
            error = error_by_pos.get(pos)
            if error is None:
                continue
            # Append any new suggestions not already on the Error
            for suggestion in all_suggestions:
                if suggestion not in error.suggestions:
                    error.suggestions.append(suggestion)

    @staticmethod
    def _build_error_by_pos(errors: list[Error]) -> dict[int, Error]:
        """Build position -> first Error lookup from an error list."""
        by_pos: dict[int, Error] = {}
        for error in errors:
            if error.position not in by_pos:
                by_pos[error.position] = error
        return by_pos

    @staticmethod
    def _dedup_errors_at_positions(errors: list[Error], positions: set[int]) -> None:
        """Remove duplicate errors at given positions (keep first per position)."""
        if not positions:
            return
        seen: set[int] = set()
        deduped: list[Error] = []
        for e in errors:
            if e.position in positions:
                if e.position in seen:
                    continue
                seen.add(e.position)
            deduped.append(e)
        errors[:] = deduped

    @staticmethod
    def _apply_arbiter_winners(
        errors: list[Error],
        winners: dict[int, ErrorCandidate],
    ) -> None:
        """Log arbiter divergence from mutex-selected errors.

        v1.3.0 shadow mode: the arbiter does NOT mutate live Error
        objects.  It only logs positions where the arbiter disagrees
        with the mutex, to collect divergence data.
        """
        error_by_pos = ContextValidator._build_error_by_pos(errors)

        for position, winner in winners.items():
            error = error_by_pos.get(position)
            if error is None:
                continue

            if error.error_type != winner.error_type:
                logger.debug(
                    "arbiter divergence: pos=%d mutex=%s arbiter=%s (conf=%.2f)",
                    position,
                    error.error_type,
                    winner.error_type,
                    winner.confidence,
                )

    @staticmethod
    def _apply_fusion_winners(
        errors: list[Error],
        fused: dict[int, tuple[float, ErrorCandidate]],
    ) -> None:
        """Apply fusion results: update error details from winning candidates.

        For positions where the fusion winner disagrees with the mutex-selected
        error, the first error at that position is updated in-place and any
        duplicate errors at the same position are removed.
        """
        error_by_pos = ContextValidator._build_error_by_pos(errors)
        fused_positions = set(fused.keys())

        for position, (fused_conf, winner) in fused.items():
            error = error_by_pos.get(position)
            if error is None:
                continue

            if error.error_type != winner.error_type:
                logger.debug(
                    "fusion override: pos=%d mutex=%s winner=%s (fused=%.3f)",
                    position,
                    error.error_type,
                    winner.error_type,
                    fused_conf,
                )
                error.error_type = winner.error_type
                if winner.suggestion:
                    error.suggestions = [winner.suggestion] + [
                        s for s in error.suggestions if s != winner.suggestion
                    ]
            error.confidence = fused_conf

        # Find positions with duplicate errors and dedup.
        dup_positions: set[int] = set()
        seen_count: dict[int, int] = {}
        for e in errors:
            if e.position in fused_positions:
                seen_count[e.position] = seen_count.get(e.position, 0) + 1
                if seen_count[e.position] > 1:
                    dup_positions.add(e.position)
        ContextValidator._dedup_errors_at_positions(errors, dup_positions)

    @staticmethod
    def _init_strategy_debug_entry() -> dict[str, Any]:
        """Create an empty strategy-debug telemetry entry."""
        return {
            "calls": 0,
            "emitted": 0,
            "new_positions": 0,
            "overlap_emitted": 0,
            "shadow_potential_positions": 0,
            "overlap_blocked_positions": 0,
            "overlap_blocked_by_type": {},
            "overlap_blocked_examples": [],
        }

    @staticmethod
    def _word_at_position(context: ValidationContext, position: int) -> str:
        """Return word at absolute position from the current sentence context."""
        for idx, pos in enumerate(context.word_positions):
            if pos == position and idx < len(context.words):
                return context.words[idx]
        return ""

    def _collect_semantic_shadow_positions(
        self,
        strategy: ValidationStrategy,
        context: ValidationContext,
    ) -> set[int] | None:
        """
        Run semantic strategy in shadow mode (no existing errors) to estimate overlap suppression.

        Returns:
            Set of candidate positions from the shadow run, or None when shadow run fails.
        """
        shadow_context = ValidationContext(
            sentence=context.sentence,
            words=list(context.words),
            word_positions=list(context.word_positions),
            is_name_mask=list(context.is_name_mask),
            full_text=context.full_text,
        )
        shadow_context.pos_tags = list(context.pos_tags)

        try:
            shadow_errors = strategy.validate(shadow_context)
        except (RuntimeError, ValueError, TypeError, KeyError, IndexError, AttributeError) as e:
            logger.debug(f"Semantic shadow debug run failed: {e}")
            if self.config.validation.raise_on_strategy_error:
                raise
            return None

        return {error.position for error in shadow_errors}

    def _update_strategy_debug_telemetry(
        self,
        strategy_debug_telemetry: dict[str, Any],
        strategy_name: str,
        strategy_errors: list[Error],
        pre_positions: set[int],
        pre_error_types: dict[int, str],
        context: ValidationContext,
        shadow_semantic_positions: set[int] | None,
    ) -> None:
        """Accumulate per-strategy gate-debug telemetry for the current sentence."""
        strategies = strategy_debug_telemetry.setdefault("strategies", {})
        entry = strategies.setdefault(strategy_name, self._init_strategy_debug_entry())
        if not isinstance(entry, dict):
            return

        emitted_count = len(strategy_errors)
        actual_positions = {error.position for error in strategy_errors}
        post_positions = set(context.existing_errors.keys())

        entry["calls"] = int(entry.get("calls", 0)) + 1
        entry["emitted"] = int(entry.get("emitted", 0)) + emitted_count
        entry["new_positions"] = int(entry.get("new_positions", 0)) + len(
            post_positions - pre_positions
        )
        entry["overlap_emitted"] = int(entry.get("overlap_emitted", 0)) + sum(
            1 for error in strategy_errors if error.position in pre_positions
        )

        if shadow_semantic_positions is None:
            return

        blocked_positions = sorted(shadow_semantic_positions - actual_positions)
        entry["shadow_potential_positions"] = int(entry.get("shadow_potential_positions", 0)) + len(
            shadow_semantic_positions
        )
        entry["overlap_blocked_positions"] = int(entry.get("overlap_blocked_positions", 0)) + len(
            blocked_positions
        )

        blocked_by_type = entry.setdefault("overlap_blocked_by_type", {})
        if isinstance(blocked_by_type, dict):
            for position in blocked_positions:
                blocked_type = pre_error_types.get(position, "unknown")
                blocked_by_type[blocked_type] = int(blocked_by_type.get(blocked_type, 0)) + 1

        blocked_examples = entry.setdefault("overlap_blocked_examples", [])
        if isinstance(blocked_examples, list):
            for position in blocked_positions:
                if len(blocked_examples) >= self.config.validation.debug_blocked_example_limit:
                    break
                blocked_examples.append(
                    {
                        "position": position,
                        "word": self._word_at_position(context, position),
                        "blocked_by": pre_error_types.get(position, "unknown"),
                    }
                )

    def get_last_strategy_debug_telemetry(self) -> dict[str, Any]:
        """Get per-check strategy gate-debug telemetry from the last validate() call."""
        with self._debug_lock:
            return copy.deepcopy(self._last_strategy_debug_telemetry)

    def _build_name_mask_from_ner(self, sentence: str, words: list[str]) -> list[bool]:
        """Build is_name_mask using NERModel entity extraction.

        Extracts entities from the sentence and marks words whose positions
        overlap with PER (person) entities as names.

        Args:
            sentence: The sentence text.
            words: List of words in the sentence.

        Returns:
            Boolean mask where True indicates the word is a person name.
        """
        from myspellchecker.text.ner_model import EntityType

        mask = [False] * len(words)

        try:
            entities = self.ner_model.extract_entities(sentence)  # type: ignore[union-attr]
        except (RuntimeError, ValueError, TypeError) as e:
            logger.debug(f"NER extraction failed for name mask, falling back: {e}")
            # Fall back to NameHeuristic if available
            if self.name_heuristic:
                return self.name_heuristic.analyze_sentence(words)
            return mask

        if not entities:
            # Fall back to NameHeuristic for basic coverage
            if self.name_heuristic:
                return self.name_heuristic.analyze_sentence(words)
            return mask

        # Build span set for PER entities
        per_spans: set[int] = set()
        for entity in entities:
            if entity.label == EntityType.PERSON:
                for pos in range(entity.start, entity.end):
                    per_spans.add(pos)

        if not per_spans:
            # No person entities found, fall back to heuristic
            if self.name_heuristic:
                return self.name_heuristic.analyze_sentence(words)
            return mask

        # Map each word to its position in the sentence and check overlap
        word_cursor = 0
        for i, word in enumerate(words):
            w_idx = sentence.find(word, word_cursor)
            if w_idx == -1:
                continue
            word_cursor = w_idx + len(word)
            # Check if any character of this word overlaps with a PER entity
            if any(pos in per_spans for pos in range(w_idx, w_idx + len(word))):
                mask[i] = True

        return mask

    def close(self) -> None:
        """Release resources held by validation strategies.

        Iterates over registered strategies and calls ``close()`` on any
        that expose it (e.g. strategies wrapping ONNX/PyTorch inference
        sessions).  Idempotent.
        """
        for strategy in self.strategies:
            if hasattr(strategy, "close"):
                try:
                    strategy.close()
                except (RuntimeError, OSError, TypeError) as e:
                    logger.warning(f"Error closing strategy {strategy.__class__.__name__}: {e}")

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ContextValidator("
            f"strategies={len(self.strategies)}, "
            f"priorities={[s.priority() for s in self.strategies]}"
            f")"
        )
