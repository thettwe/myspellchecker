"""
POS Sequence Validation Strategy.

This strategy validates Part-of-Speech (POS) tag sequences to detect
grammatically incorrect patterns like consecutive verbs or particles.

Priority: 30 (runs after rule-based checks)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from myspellchecker.core.constants import ET_POS_SEQUENCE_ERROR
from myspellchecker.core.response import ContextError, Error
from myspellchecker.core.validation_strategies.base import ValidationContext, ValidationStrategy
from myspellchecker.grammar.patterns import (
    ALL_PARTICLES,
    INVALID_POS_SEQUENCES,
    QUESTION_PARTICLES,
    SENTENCE_FINAL_PARTICLES,
    is_valid_verb_sequence,
)
from myspellchecker.utils.logging_utils import get_logger

if TYPE_CHECKING:
    from myspellchecker.algorithms.viterbi import ViterbiTagger

# Combined particle set for V-V guard (avoid flagging mistagged particles)
_ALL_KNOWN_PARTICLES: frozenset[str] = ALL_PARTICLES | QUESTION_PARTICLES

# Light verbs: Myanmar productively borrows loanwords via
# [loanword + light_verb] pattern (like Japanese suru-verbs).
# When one word in a V+V or V+N pair is OOV and the other is a light verb,
# skip flagging — it's likely a valid loanword construction.
_LIGHT_VERBS: frozenset[str] = frozenset(
    {
        "လုပ်",
        "ရိုက်",
        "တင်",
        "ထုတ်",
        "ကူး",
        "ဝင်",
        "ထွက်",
        "ဆွဲ",
        "ဖွင့်",
        "ပိတ်",
        "ဖတ်",
        "စစ်",
        "ကြည့်",
        "ပြ",
        "သုံး",
        "လာ",
        "သွား",
        "ပေး",
        "ယူ",
        "ထည့်",
        "ခေါ်",
    }
)


class POSSequenceValidationStrategy(ValidationStrategy):
    """
    POS tag sequence validation strategy.

    This strategy uses a ViterbiTagger to assign POS tags to words,
    then validates that the tag sequence follows grammatical rules.
    It detects invalid consecutive tag patterns.

    Invalid POS patterns detected:
    - V-V: Two consecutive verbs (unless valid serial verb construction)
    - P-P: Two consecutive particles
    - N-N: Multiple nouns without particle (warning)

    Note: Myanmar is a serial verb language. V-V sequences like auxiliary verbs
    (သွားနေတယ်), modal verbs (လုပ်နိုင်တယ်), and directional verbs (ပြောင်းသွားတယ်)
    are valid and not flagged as errors.

    Priority: 30 (runs after tone and syntactic validation)
    - Requires POS tagging (computationally expensive)
    - Should run before statistical n-gram methods
    - Confidence: 0.85 (high for rule-based POS patterns)

    Example:
        >>> strategy = POSSequenceValidationStrategy(viterbi_tagger)
        >>> context = ValidationContext(
        ...     sentence="သူ သွား သွား ခဲ့ တယ်",
        ...     words=["သူ", "သွား", "သွား", "ခဲ့", "တယ်"],
        ...     word_positions=[0, 6, 15, 24, 33],
        ...     is_name_mask=[False, False, False, False, False]
        ... )
        >>> errors = strategy.validate(context)
        # Detects V-V pattern (သွား သွား)
    """

    _ATTRIBUTIVE_SUFFIXES: tuple[str, ...] = ("သော", "တဲ့")
    _CASE_PARTICLES: frozenset[str] = frozenset({"က", "ကို", "သည်", "မှာ", "တွင်"})
    _PLURAL_MARKERS: frozenset[str] = frozenset({"များ", "တို့"})

    def __init__(
        self,
        viterbi_tagger: "ViterbiTagger" | None,
        confidence: float = 0.85,
        pos_disambiguator: Any | None = None,
        provider: Any | None = None,
    ):
        """
        Initialize POS sequence validation strategy.

        Args:
            viterbi_tagger: ViterbiTagger instance for POS tagging.
                           If None, this strategy is disabled.
            confidence: Confidence score for POS violations (default: 0.85).
            pos_disambiguator: Optional POSDisambiguator for resolving
                              ambiguous multi-POS words using R1-R5 rules.
            provider: Optional DictionaryProvider for dictionary lookups.
        """
        self.viterbi_tagger = viterbi_tagger
        self.confidence = confidence
        self.pos_disambiguator = pos_disambiguator
        self.provider = provider
        self.logger = get_logger(__name__)

    def _should_skip_verb_pair(self, prev_word: str, curr_word: str) -> bool:
        """Return True if a verb pair should be skipped (common guards).

        Checks shared between V+V and V+N/N+V loops:
        - Either word is a known particle (tagger mistag)
        - prev_word is a reduplication pattern (adverb, not verb)
        - prev_word ends with a sentence-final particle (completed predicate)
        - Either word is a light verb (loanword construction)
        """
        if prev_word in _ALL_KNOWN_PARTICLES or curr_word in _ALL_KNOWN_PARTICLES:
            return True
        mid = len(prev_word) // 2
        if mid > 0 and prev_word[:mid] == prev_word[mid:]:
            return True
        if any(prev_word.endswith(sfp) for sfp in SENTENCE_FINAL_PARTICLES):
            return True
        if prev_word in _LIGHT_VERBS or curr_word in _LIGHT_VERBS:
            return True
        return False

    def _check_merged_compound(
        self,
        context: ValidationContext,
        errors: list[Error],
        i: int,
        prev_word: str,
        curr_word: str,
    ) -> bool:
        """Check if a merged compound is invalid and emit an error if so.

        Returns True if an error was emitted (caller should skip further checks).
        Returns False if the merged compound is valid or provider is unavailable.
        """
        if not self.provider:
            return False
        merged = prev_word + curr_word
        try:
            is_valid = self.provider.is_valid_word(merged)
        except (RuntimeError, ValueError, TypeError, KeyError, AttributeError) as e:
            self.logger.debug("Provider lookup failed for '%s': %s", merged, e)
            is_valid = False
        if is_valid:
            return False
        errors.append(
            ContextError(
                text=merged,
                position=context.word_positions[i - 1],
                error_type=ET_POS_SEQUENCE_ERROR,
                suggestions=[],
                confidence=self.confidence,
                probability=0.0,
                prev_word=(context.words[i - 2] if i >= 2 else ""),
            )
        )
        pos_prev = context.word_positions[i - 1]
        context.existing_errors[pos_prev] = ET_POS_SEQUENCE_ERROR
        context.existing_suggestions[pos_prev] = []
        context.existing_confidences[pos_prev] = self.confidence
        # Also claim second word to prevent duplicate detection
        pos_curr = context.word_positions[i]
        context.existing_errors[pos_curr] = ET_POS_SEQUENCE_ERROR
        return True

    def validate(self, context: ValidationContext) -> list[Error]:
        """
        Validate POS tag sequences for grammatical correctness.

        Args:
            context: Validation context with sentence information.

        Returns:
            List of ContextError objects for POS sequence violations.
        """
        if len(context.words) < 2:
            return []

        errors: list[Error] = []

        try:
            # Use pre-computed POS tags from ContextValidator if available,
            # otherwise fall back to computing our own via Viterbi tagger
            if context.pos_tags and len(context.pos_tags) == len(context.words):
                tags = list(context.pos_tags)  # Copy to avoid mutating shared state
            elif self.viterbi_tagger:
                tags = self.viterbi_tagger.tag_sequence(context.words)
                if len(tags) != len(context.words):
                    self.logger.warning(
                        f"POS tag count ({len(tags)}) != word count ({len(context.words)})"
                    )
                    return []
                tags = [t.upper() for t in tags]
            else:
                return []

            # Disambiguate multi-POS tags using R1-R5 rules if available
            # (runs on both pre-computed and freshly-computed tags)
            if self.pos_disambiguator:
                for i, tag in enumerate(tags):
                    if "|" in tag:
                        pos_set = frozenset(tag.split("|"))
                        prev_tag = tags[i - 1] if i > 0 else None
                        next_tag = tags[i + 1] if i < len(tags) - 1 else None
                        if prev_tag and "|" in prev_tag:
                            prev_tag = prev_tag.split("|")[0]
                        if next_tag and "|" in next_tag:
                            next_tag = next_tag.split("|")[0]
                        result = self.pos_disambiguator.disambiguate_in_context(
                            word=context.words[i],
                            word_pos_tags=pos_set,
                            prev_word=context.words[i - 1] if i > 0 else None,
                            prev_word_pos=prev_tag,
                            next_word=(
                                context.words[i + 1] if i < len(context.words) - 1 else None
                            ),
                            next_word_pos=next_tag,
                        )
                        tags[i] = result.resolved_pos

            # Update context with disambiguated tags for downstream strategies
            context.pos_tags = tags

            # Check for invalid consecutive POS tag patterns
            for i in range(1, len(tags)):
                prev_tag = tags[i - 1]
                curr_tag = tags[i]

                # Skip if current word is a proper name
                # Names can have unusual POS patterns that violate normal rules
                if context.is_name_mask[i]:
                    continue

                # Skip if this word already has an error from previous strategy
                if context.word_positions[i] in context.existing_errors:
                    continue

                # Check if this tag pair is defined as invalid
                sequence_key = (prev_tag, curr_tag)
                if sequence_key in INVALID_POS_SEQUENCES:
                    severity, description = INVALID_POS_SEQUENCES[sequence_key]

                    # Special handling for V-V sequences: Myanmar is a serial verb language
                    # Many V-V sequences are valid (auxiliary verbs, modal verbs, etc.)
                    if sequence_key == ("V", "V"):
                        prev_word = context.words[i - 1]
                        curr_word = context.words[i]
                        if is_valid_verb_sequence(prev_word, curr_word):
                            continue

                        if self._should_skip_verb_pair(prev_word, curr_word):
                            continue

                        # Guard: skip if verb1's primary dictionary POS is not V
                        # The tagger sometimes mistags nouns/adverbs as V
                        # (e.g., အတူ=PPM, လုပ်ငန်း=N, ငြိမ်ငြိမ်=ADV)
                        if self.provider and hasattr(self.provider, "get_word_pos"):
                            dict_pos = self.provider.get_word_pos(prev_word)
                            if dict_pos:
                                primary_pos = dict_pos.split("|")[0]
                                if primary_pos != "V":
                                    continue

                        if self._check_merged_compound(context, errors, i, prev_word, curr_word):
                            severity = "error"
                            continue

                    # Only report errors, not warnings
                    # Warnings are lower confidence and may have false positives
                    if severity == "error":
                        errors.append(
                            ContextError(
                                text=context.words[i],
                                position=context.word_positions[i],
                                error_type=ET_POS_SEQUENCE_ERROR,
                                suggestions=[],  # No automatic suggestions for POS errors
                                confidence=self.confidence,
                                probability=0.0,  # Rule-based doesn't use n-gram probability
                                prev_word=context.words[i - 1],
                            )
                        )

                        # Mark this position as having an error
                        pos_cur = context.word_positions[i]
                        context.existing_errors[pos_cur] = ET_POS_SEQUENCE_ERROR
                        context.existing_suggestions[pos_cur] = []
                        context.existing_confidences[pos_cur] = self.confidence

            # ------------------------------------------------------------------
            # Invalid serial verb check for V+N / N+V where the N is also V
            # in the dictionary (multi-POS). Catches ပြော(V)+စား(N but also V)
            # when the merged compound is not a valid word.
            # ------------------------------------------------------------------
            if self.provider and len(tags) >= 2:
                for i in range(1, len(tags)):
                    prev_tag = tags[i - 1]
                    curr_tag = tags[i]

                    # Only V+N or N+V pairs where one is V
                    if not (
                        (prev_tag == "V" and curr_tag == "N")
                        or (prev_tag == "N" and curr_tag == "V")
                    ):
                        continue

                    # Skip names and already-errored positions
                    if context.is_name_mask[i] or context.is_name_mask[i - 1]:
                        continue
                    if (
                        context.word_positions[i] in context.existing_errors
                        or context.word_positions[i - 1] in context.existing_errors
                    ):
                        continue

                    prev_word = context.words[i - 1]
                    curr_word = context.words[i]

                    # Only flag character-adjacent pairs (no space gap)
                    prev_end = context.word_positions[i - 1] + len(prev_word)
                    if prev_end != context.word_positions[i]:
                        continue

                    if self._should_skip_verb_pair(prev_word, curr_word):
                        continue

                    # Check if the N word also has V in its dictionary POS
                    n_word = curr_word if curr_tag == "N" else prev_word
                    try:
                        pos_str = self.provider.get_word_pos(n_word) or ""
                        if "V" not in pos_str.split("|"):
                            continue
                    except (RuntimeError, ValueError, TypeError, KeyError, AttributeError) as e:
                        self.logger.debug("POS lookup failed for '%s': %s", n_word, e)
                        continue

                    # Both are verbs in reality; check auxiliary
                    if is_valid_verb_sequence(prev_word, curr_word):
                        continue

                    self._check_merged_compound(context, errors, i, prev_word, curr_word)

            # ------------------------------------------------------------------
            # Sentence-final predicate check
            # Myanmar is SOV — declarative sentences need a verb/particle ending.
            # Catches sentences like "လှပသော ပန်းပွင့်များ ချစ်စရာ" (missing ဖြစ်သည်).
            # ------------------------------------------------------------------
            if len(context.words) >= 3:
                last_idx = len(context.words) - 1
                last_word = context.words[last_idx]
                last_tag = tags[last_idx]
                last_pos = context.word_positions[last_idx]

                # Guard: skip if already errored or is a name
                if last_pos not in context.existing_errors and not context.is_name_mask[last_idx]:
                    has_structural = any(
                        w.endswith(self._ATTRIBUTIVE_SUFFIXES)
                        or w in self._CASE_PARTICLES
                        or w in self._PLURAL_MARKERS
                        for w in context.words
                    )

                    # Only flag when the entire sentence has no verb.
                    # Sentences with a verb somewhere already have a predicate
                    # (e.g., "ပြောပါဦး" — ပြော is V, ဦး is a trailing particle).
                    has_verb = any("V" in t.split("|") if "|" in t else t == "V" for t in tags)

                    if has_structural and not has_verb:
                        # Resolve multi-POS for last word
                        resolved_last = last_tag.split("|")[0] if "|" in last_tag else last_tag
                        is_verb_like = resolved_last in ("V", "PPM", "PART")

                        # Special -စရာ suffix: always needs copula ဖြစ် when predicative
                        ends_with_sara = last_word.endswith("စရာ")

                        if ends_with_sara or not is_verb_like:
                            # Reconstruct the full unsegmented token from the
                            # original sentence when the segmenter split a compound
                            # (e.g., ချစ်စရာ → ချစ် + စရာ). Walk backward through
                            # adjacent words (no space gap) to find the compound start.
                            full_token = last_word
                            full_pos = last_pos
                            if ends_with_sara:
                                k = last_idx - 1
                                while k >= 0:
                                    prev_end = context.word_positions[k] + len(context.words[k])
                                    if prev_end == context.word_positions[k + 1]:
                                        full_token = context.words[k] + full_token
                                        full_pos = context.word_positions[k]
                                        k -= 1
                                    else:
                                        break

                            suggestions = [
                                full_token + " ဖြစ်သည်",
                                full_token + " ဖြစ်ပါသည်",
                            ]
                            errors.append(
                                ContextError(
                                    text=full_token,
                                    position=full_pos,
                                    error_type=ET_POS_SEQUENCE_ERROR,
                                    suggestions=suggestions,
                                    confidence=self.confidence,
                                    probability=0.0,
                                    prev_word=context.words[last_idx - 1],
                                )
                            )
                            context.existing_errors[full_pos] = ET_POS_SEQUENCE_ERROR
                            context.existing_suggestions[full_pos] = suggestions
                            context.existing_confidences[full_pos] = self.confidence
                            # Also claim last_pos when compound reconstruction changed full_pos
                            if full_pos != last_pos:
                                context.existing_errors[last_pos] = ET_POS_SEQUENCE_ERROR

        except (RuntimeError, ValueError, KeyError, IndexError, AttributeError) as e:
            # Graceful fallback if POS tagging fails
            self.logger.debug(f"POS tagging failed: {e}", exc_info=True)

        return errors

    def priority(self) -> int:
        """
        Return strategy execution priority.

        Returns:
            30 (runs after tone and syntactic validation, before statistical methods)
        """
        return 30

    def __repr__(self) -> str:
        """String representation."""
        enabled = "enabled" if self.viterbi_tagger else "disabled"
        return (
            f"POSSequenceValidationStrategy(priority={self.priority()}, {enabled}, "
            f"confidence={self.confidence})"
        )
