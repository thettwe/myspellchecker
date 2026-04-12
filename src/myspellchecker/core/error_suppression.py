"""Error suppression mixin for SpellChecker.

Provides methods that suppress, filter, or remove errors that are
false positives, cascade artifacts, or low-value alerts.  SpellChecker
inherits from this mixin, keeping the same ``self.method()`` call sites.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from myspellchecker.core.constants import (
    ET_ASPECT_ADVERB_CONFLICT,
    ET_BROKEN_COMPOUND,
    ET_BROKEN_STACKING,
    ET_BROKEN_VIRAMA,
    ET_COLLOCATION_ERROR,
    ET_COLLOQUIAL_CONTRACTION,
    ET_CONFUSABLE_ERROR,
    ET_CONTEXT_PROBABILITY,
    ET_DANGLING_WORD,
    ET_DUPLICATE_PUNCTUATION,
    ET_HA_HTOE_CONFUSION,
    ET_HIDDEN_COMPOUND_TYPO,
    ET_HOMOPHONE_ERROR,
    ET_INCOMPLETE_STACKING,
    ET_LEADING_VOWEL_E,
    ET_MEDIAL_CONFUSION,
    ET_MEDIAL_ORDER_ERROR,
    ET_MERGED_SFP_CONJUNCTION,
    ET_MISSING_ASAT,
    ET_MISSING_PUNCTUATION,
    ET_NEGATION_SFP_MISMATCH,
    ET_PARTICLE_CONFUSION,
    ET_PARTICLE_MISUSE,
    ET_POS_SEQUENCE_ERROR,
    ET_REGISTER_MIXING,
    ET_SEMANTIC_ERROR,
    ET_SYLLABLE,
    ET_SYNTAX_ERROR,
    ET_TENSE_MISMATCH,
    ET_VOWEL_AFTER_ASAT,
    ET_WORD,
    ET_WRONG_PUNCTUATION,
    LEXICALIZED_COMPOUND_MIN_FREQ,
)
from myspellchecker.core.constants.detector_thresholds import (
    DEFAULT_SUPPRESSION_THRESHOLDS as _ST,
)
from myspellchecker.core.correction_utils import (
    _PRESERVE_ERROR_TYPES,
    filter_syllable_errors_in_valid_words,
)
from myspellchecker.core.response import Error, SyllableError
from myspellchecker.text.normalize import normalize

if TYPE_CHECKING:
    from myspellchecker.core.config import SpellCheckerConfig
    from myspellchecker.providers import DictionaryProvider
    from myspellchecker.segmenters import Segmenter


# --- Constants for dedup/NER filtering -------------------------------------------

_TEXT_DETECTOR_TYPES = frozenset(
    {
        ET_HA_HTOE_CONFUSION,
        ET_MEDIAL_ORDER_ERROR,
        ET_BROKEN_VIRAMA,
        ET_BROKEN_STACKING,
        ET_BROKEN_COMPOUND,
        ET_PARTICLE_CONFUSION,
        ET_LEADING_VOWEL_E,
        ET_INCOMPLETE_STACKING,
        ET_NEGATION_SFP_MISMATCH,
        ET_MERGED_SFP_CONJUNCTION,
        ET_ASPECT_ADVERB_CONFLICT,
        ET_DUPLICATE_PUNCTUATION,
        ET_WRONG_PUNCTUATION,
        ET_MISSING_PUNCTUATION,
    }
)

_NER_IMMUNE = frozenset(
    {
        ET_HA_HTOE_CONFUSION,
        ET_CONFUSABLE_ERROR,
        ET_BROKEN_STACKING,
        ET_BROKEN_VIRAMA,
        ET_BROKEN_COMPOUND,
        ET_VOWEL_AFTER_ASAT,
        ET_MEDIAL_CONFUSION,
        ET_MEDIAL_ORDER_ERROR,
        ET_COLLOQUIAL_CONTRACTION,
        ET_PARTICLE_CONFUSION,
        ET_PARTICLE_MISUSE,
        ET_REGISTER_MIXING,
        ET_SEMANTIC_ERROR,
        ET_LEADING_VOWEL_E,
        ET_INCOMPLETE_STACKING,
        ET_NEGATION_SFP_MISMATCH,
        ET_MERGED_SFP_CONJUNCTION,
        ET_ASPECT_ADVERB_CONFLICT,
        ET_COLLOCATION_ERROR,
    }
)

_LOC_SUPPRESSIBLE = frozenset(
    {
        ET_SYNTAX_ERROR,
        ET_SYLLABLE,
        ET_WORD,
        ET_CONTEXT_PROBABILITY,
    }
)

# Narrow root-cause error types that should survive inside wider generic spans
# when they have actionable suggestions.
_ROOT_CAUSE_NARROW_TYPES = frozenset(
    {
        ET_CONFUSABLE_ERROR,
        ET_HOMOPHONE_ERROR,
        ET_HIDDEN_COMPOUND_TYPO,
        ET_MEDIAL_CONFUSION,
        ET_COLLOCATION_ERROR,
    }
)

# Generic/statistical error types that should yield to narrow root-cause errors.
_GENERIC_WIDE_TYPES = frozenset(
    {
        ET_POS_SEQUENCE_ERROR,
        ET_CONTEXT_PROBABILITY,
        ET_WORD,
    }
)

# --- Tuned suppression thresholds ------------------------------------------------
#
# All numeric thresholds are now consolidated in SuppressionThresholds
# (detector_thresholds.py) alongside other empirically calibrated parameters.
# The aliases below provide backward compatibility for existing code.

_CONFUSABLE_SHORT_TOKEN_SUPPRESS_FREQ = _ST.confusable_short_token_freq
_CONFUSABLE_AMBIGUITY_MAX_CONFIDENCE = _ST.confusable_ambiguity_max_confidence
_CONFUSABLE_FRAGMENT_MAX_TOKEN_LEN = _ST.confusable_fragment_max_token_len
_CONFUSABLE_SELF_SUGGEST_MAX_TOKEN_LEN = _ST.confusable_self_suggest_max_token_len
_BARE_CONSONANT_PROXIMITY = _ST.bare_consonant_proximity
_SEMANTIC_STABLE_NOUN_MIN_LEN = _ST.semantic_stable_noun_min_len
_SEMANTIC_SHORT_SUGGESTION_MAX_RATIO = _ST.semantic_short_suggestion_max_ratio
_POS_SEQ_LONG_SPAN_NO_SUGGESTION_LEN = _ST.pos_seq_long_span_no_suggestion_len
_POS_SEQ_MEDIUM_SPAN_TINY_SUGGESTION_LEN = _ST.pos_seq_medium_span_tiny_suggestion_len
_POS_SEQ_TINY_SUGGESTION_MAX_LEN = _ST.pos_seq_tiny_suggestion_max_len
_SYNTAX_SHORT_SWAP_MAX_TOKEN_LEN = _ST.syntax_short_swap_max_token_len
_SYLLABLE_TECH_COMPOUND_MIN_TOKEN_LEN = _ST.syllable_tech_compound_min_token_len
_CONTEXT_PROB_MIN_TOKEN_LEN_FOR_SHORT_SUG = _ST.context_prob_min_token_len_for_short_sug
_CONTEXT_PROB_SHORT_SUGGESTION_MAX_LEN = _ST.context_prob_short_suggestion_max_len
_NER_HIGH_CONFIDENCE_OVERRIDE = _ST.ner_high_confidence_override
_NER_LOC_DEFAULT_CONFIDENCE = _ST.ner_loc_default_confidence

_BOUNDARY_PUNCT = frozenset("၊။,.!?;:\"'()[]{}")
_LOW_VALUE_QUOTATIVE_SYNTAX_TOKENS = frozenset(normalize(token) for token in ("လို့", "ဆိုတာ", "ဆိုပြီး"))


class ErrorSuppressionMixin:
    """Mixin providing error suppression and deduplication methods for SpellChecker."""

    # --- Type stubs for attributes provided by SpellChecker ----------------------

    provider: "DictionaryProvider"
    segmenter: "Segmenter"
    config: "SpellCheckerConfig"
    logger: Any
    _ner_model: Any
    _semantic_checker: Any  # SemanticChecker | None
    _KEEP_ATTACHED_SUFFIXES: Any  # defined in PostNormalizationDetectorsMixin
    _MISSING_ASAT_PARTICLES: Any  # defined in PostNormalizationDetectorsMixin
    _MISSING_VISARGA_SUFFIXES: Any  # defined in PostNormalizationDetectorsMixin

    # --- Suppression methods -----------------------------------------------------

    @staticmethod
    def _suppress_cascade_syllable_errors(errors: list[Error], text: str = "") -> None:
        """Remove secondary syllable errors caused by a prior broken syllable.

        When a syllable error (e.g. broken virama ``က္ေ``) breaks the
        segmenter's syllable boundaries, the residual text (e.g. ``ယျ``)
        is also flagged as invalid.  This method detects such cascades
        within the same word (no space between the errors) and removes
        the secondary error.
        """
        if len(errors) < 2:
            return
        syl_errors = [
            e
            for e in errors
            if isinstance(e, SyllableError)
            and getattr(e, "error_type", "") == ET_SYLLABLE
            and e.text
        ]
        if len(syl_errors) < 2:
            return
        syl_errors.sort(key=lambda e: e.position)
        to_remove: set[int] = set()
        for i in range(len(syl_errors) - 1):
            e1 = syl_errors[i]
            e2 = syl_errors[i + 1]
            e1_end = e1.position + len(e1.text)
            gap = text[e1_end : e2.position] if text else ""
            if " " not in gap:
                to_remove.add(id(e2))
        if to_remove:
            errors[:] = [e for e in errors if id(e) not in to_remove]

    @staticmethod
    def _suppress_pali_stacking_errors(errors: list[Error], text: str) -> None:
        """Suppress errors on fragments from valid Pali/Sanskrit virama stacking.

        Words containing valid consonant stacking via virama (U+1039), e.g.
        ဗုဒ္ဓ (Buddha), မဂ္ဂဇင်း (magazine), ပစ္စည်း (things), break the
        syllable segmenter.  The resulting fragments (e.g. ``ဗု``, ``ဒ္ဓ``)
        are flagged as invalid syllables or unknown words.

        This method:
        1. Validates stacking pairs against the YAML-loaded whitelist
           (``rules/stacking_pairs.yaml``) via ``load_stacking_pairs()``.
        2. Expands each valid stacking site to word boundaries.
        3. Merges overlapping/adjacent ranges so that a word with multiple
           stacking sites (e.g. ပစ္စည်း) produces one continuous range.
        4. Suppresses both ``invalid_syllable`` and ``invalid_word`` errors
           whose span falls entirely within a stacking word range.
        """
        _VIRAMA = "\u1039"
        if _VIRAMA not in text:
            return

        # Load comprehensive stacking pairs from YAML (with hardcoded fallback)
        from myspellchecker.core.detection_rules import load_stacking_pairs

        valid_pairs = load_stacking_pairs()

        # --- Phase 1: find word ranges containing valid stacking sites ---
        raw_ranges: list[tuple[int, int]] = []
        for i, ch in enumerate(text):
            if ch != _VIRAMA:
                continue
            if i < 1 or i + 1 >= len(text):
                continue
            upper = text[i - 1]
            lower = text[i + 1]
            if (upper, lower) not in valid_pairs:
                continue
            # Expand to word boundaries (space-delimited)
            start = i
            while start > 0 and text[start - 1] != " ":
                start -= 1
            end = i + 1
            while end < len(text) and text[end] != " ":
                end += 1
            raw_ranges.append((start, end))

        if not raw_ranges:
            return

        # --- Phase 2: merge overlapping / adjacent ranges ---
        raw_ranges.sort()
        merged: list[tuple[int, int]] = [raw_ranges[0]]
        for rs, re in raw_ranges[1:]:
            prev_s, prev_e = merged[-1]
            if rs <= prev_e:
                # Overlapping or contiguous — extend
                merged[-1] = (prev_s, max(prev_e, re))
            else:
                merged.append((rs, re))

        # --- Phase 3: suppress syllable AND word errors within ranges ---
        _SUPPRESSIBLE = frozenset({ET_SYLLABLE, ET_WORD})
        to_remove: set[int] = set()
        for idx, e in enumerate(errors):
            if getattr(e, "error_type", "") not in _SUPPRESSIBLE:
                continue
            e_start = e.position
            e_end = e_start + (len(e.text) if e.text else 0)
            for ws, we in merged:
                if e_start >= ws and e_end <= we:
                    to_remove.add(idx)
                    break

        if to_remove:
            errors[:] = [e for idx, e in enumerate(errors) if idx not in to_remove]

    @staticmethod
    def _suppress_bare_consonant_near_text_errors(errors: list[Error]) -> None:
        """Suppress bare-consonant invalid_syllable errors near text-level errors.

        When a text-level detector (broken_stacking, broken_virama, etc.)
        flags an error, the syllable segmenter may also produce a single
        bare-consonant fragment at an adjacent position. That fragment is
        a byproduct of the same issue, not an independent error.

        Example: ``ဘဏ္ာ`` (broken stacking) → broken_stacking at pos 1,
        but ``ဘ`` at pos 0 is also flagged as invalid_syllable.
        """
        if len(errors) < 2:
            return

        text_error_positions: set[int] = set()
        for e in errors:
            if getattr(e, "error_type", "") != ET_SYLLABLE:
                text_error_positions.add(e.position)
                if e.text:
                    for offset in range(len(e.text)):
                        text_error_positions.add(e.position + offset)

        if not text_error_positions:
            return

        to_remove: set[int] = set()
        for idx, e in enumerate(errors):
            if getattr(e, "error_type", "") != ET_SYLLABLE:
                continue
            if not e.text or len(e.text) != 1:
                continue
            cp = ord(e.text[0])
            if not (0x1000 <= cp <= 0x1021):
                continue
            for delta in range(-_BARE_CONSONANT_PROXIMITY, _BARE_CONSONANT_PROXIMITY + 1):
                if delta == 0:
                    continue
                if e.position + delta in text_error_positions:
                    to_remove.add(idx)
                    break

        if to_remove:
            errors[:] = [e for idx, e in enumerate(errors) if idx not in to_remove]

    @staticmethod
    def _suppress_generic_pos_sequence_errors(errors: list[Error]) -> None:
        """Drop broad POS-sequence placeholders when specific root-cause errors exist.

        Context strategy can emit generic sentence templates such as
        ``<token> ဖြစ်သည်`` for a wide span. When a narrower detector already
        provides a concrete fix inside the same span (medial, missing-asat, etc.),
        keep the specific error and suppress the generic placeholder.
        """
        if len(errors) < 2:
            return

        specific_types = {
            ET_SYLLABLE,
            ET_MEDIAL_CONFUSION,
            ET_HA_HTOE_CONFUSION,
            ET_PARTICLE_CONFUSION,
            ET_CONFUSABLE_ERROR,
        }
        generic_template_cues = (normalize("ဖြစ်သည်"), normalize("ဖြစ်ပါသည်"))

        to_remove: set[int] = set()
        for i, err in enumerate(errors):
            if err.error_type != ET_POS_SEQUENCE_ERROR:
                continue
            if not err.text or not err.suggestions:
                continue
            if not all(
                any(cue in sug for cue in generic_template_cues) for sug in err.suggestions[:2]
            ):
                continue

            err_start = err.position
            err_end = err_start + len(err.text)
            has_specific_inside = False
            for j, other in enumerate(errors):
                if i == j or not other.text or not other.suggestions:
                    continue
                if other.error_type not in specific_types:
                    continue
                # Guard against two-stage removal: only suppress POS when
                # the inner error will survive filter_syllable_errors_in_valid_words().
                # Non-preserved types (e.g. invalid_syllable) may be removed later,
                # leaving no error at all.
                if other.error_type not in _PRESERVE_ERROR_TYPES:
                    continue
                other_start = other.position
                other_end = other_start + len(other.text)
                if other_start >= err_start and other_end <= err_end:
                    has_specific_inside = True
                    break

            if has_specific_inside:
                to_remove.add(i)

        if to_remove:
            errors[:] = [e for idx, e in enumerate(errors) if idx not in to_remove]

    def _suppress_low_value_context_probability(
        self,
        errors: list[Error],
        text: str | None = None,
    ) -> None:
        """Suppress context_probability errors on frequent valid words with no suggestions."""
        if not errors or not self.provider:
            return

        if text is None:
            text = ""

        def _is_simple_reduplicated(token: str) -> bool:
            if len(token) < 4 or len(token) % 2 != 0:
                return False
            half = len(token) // 2
            return token[:half] == token[half:]

        filtered: list[Error] = []
        for e in errors:
            if e.error_type != ET_CONTEXT_PROBABILITY:
                filtered.append(e)
                continue

            token = normalize(e.text or "")
            if not token or not self.provider.is_valid_word(token):
                filtered.append(e)
                continue
            if not hasattr(self.provider, "get_word_frequency"):
                filtered.append(e)
                continue

            freq = self.provider.get_word_frequency(token)
            if not isinstance(freq, (int, float)):
                filtered.append(e)
                continue

            if _is_simple_reduplicated(token) and freq >= LEXICALIZED_COMPOUND_MIN_FREQ:
                continue

            if text is not None and 0 <= e.position < len(text):
                end = e.position + len(e.text or "")
                if end < len(text) and text[end] != " ":
                    tail = text[end:]
                    if (
                        freq >= LEXICALIZED_COMPOUND_MIN_FREQ
                        and not self._starts_with_attached_suffix(tail)
                    ):
                        continue

            if e.suggestions:
                norm_sugs = [normalize(s) for s in e.suggestions[:5] if isinstance(s, str) and s]
                if (
                    norm_sugs
                    and len(token) >= _CONTEXT_PROB_MIN_TOKEN_LEN_FOR_SHORT_SUG
                    and all(len(s) <= _CONTEXT_PROB_SHORT_SUGGESTION_MAX_LEN for s in norm_sugs)
                    and freq >= LEXICALIZED_COMPOUND_MIN_FREQ
                ):
                    # Low-signal function-word suggestions on frequent words are
                    # typically noise in formal prose.
                    continue

                # Generic low-information pattern:
                # frequent in-vocabulary token, non-boundary span, and all
                # alternatives are shorter than the token itself.
                if text is not None and 0 <= e.position < len(text):
                    end = e.position + len(e.text or "")
                    if (
                        end <= len(text)
                        and not self._is_token_boundary_like(text, e.position, end)
                        and freq >= LEXICALIZED_COMPOUND_MIN_FREQ
                        and norm_sugs
                        and all(len(s) < len(token) for s in norm_sugs)
                    ):
                        continue

                filtered.append(e)
                continue
            if not e.suggestions and freq < LEXICALIZED_COMPOUND_MIN_FREQ:
                filtered.append(e)

        errors[:] = filtered

    @staticmethod
    def _is_token_boundary_like(text: str, start: int, end: int) -> bool:
        """Return True if span is loosely bounded by whitespace/punctuation."""
        if start < 0 or end > len(text) or start >= end:
            return False
        if start == 0:
            left_ok = True
        else:
            left = text[start - 1]
            left_ok = left.isspace() or left in _BOUNDARY_PUNCT

        if end == len(text):
            right_ok = True
        else:
            right = text[end]
            right_ok = right.isspace() or right in _BOUNDARY_PUNCT
        return left_ok and right_ok

    def _starts_with_attached_suffix(self, tail: str) -> bool:
        """Return True when tail begins with a grammatical suffix chain.

        Reject lexical continuations that only share a short prefix with
        suffix markers (for example ``မှ`` in ``မှတ်တမ်း``).
        """
        if not tail:
            return False

        normalized_suffixes = [
            normalize(suf)
            for suf in getattr(self, "_KEEP_ATTACHED_SUFFIXES", ())
            if isinstance(suf, str) and suf
        ]
        for suffix in normalized_suffixes:
            if not tail.startswith(suffix):
                continue

            remainder = tail[len(suffix) :]
            if not remainder:
                return True
            if remainder[0].isspace() or remainder[0] in _BOUNDARY_PUNCT:
                return True
            if any(remainder.startswith(next_suffix) for next_suffix in normalized_suffixes):
                return True

        return False

    def _suppress_low_value_confusable_errors(
        self,
        errors: list[Error],
        text: str | None = None,
    ) -> None:
        """Suppress low-value confusable_error cases with high FP risk."""
        if not errors or not self.provider:
            return

        filtered: list[Error] = []
        for e in errors:
            if e.error_type != ET_CONFUSABLE_ERROR:
                filtered.append(e)
                continue

            token = normalize(e.text or "")
            if not token:
                # Zero-width char errors normalize to empty — keep them.
                filtered.append(e)
                continue

            # Fragment-like short spans inside a longer token are usually
            # segmentation artifacts in natural corpora.
            if (
                text is not None
                and len(token) <= _CONFUSABLE_FRAGMENT_MAX_TOKEN_LEN
                and 0 <= e.position < len(text)
            ):
                end = e.position + len(e.text or "")
                if end <= len(text) and not self._is_token_boundary_like(text, e.position, end):
                    continue

            # Non-boundary confusable spans that include the original token in
            # suggestions are usually segment artifacts and not actionable.
            if text is not None and 0 <= e.position < len(text):
                end = e.position + len(e.text or "")
                if end <= len(text) and not self._is_token_boundary_like(text, e.position, end):
                    norm_sugs = [normalize(s) for s in (e.suggestions or []) if isinstance(s, str)]
                    tail = text[end:] if end < len(text) else ""
                    if token in norm_sugs and not self._starts_with_attached_suffix(tail):
                        continue

            # If the original token is already among suggestions for a short,
            # high-frequency valid token, treat it as low-confidence ambiguity.
            norm_sugs = [normalize(s) for s in (e.suggestions or []) if isinstance(s, str) and s]
            if (
                token in norm_sugs
                and len(token) <= _CONFUSABLE_SELF_SUGGEST_MAX_TOKEN_LEN
                and hasattr(self.provider, "is_valid_word")
                and hasattr(self.provider, "get_word_frequency")
                and self.provider.is_valid_word(token)
                and e.confidence <= _CONFUSABLE_AMBIGUITY_MAX_CONFIDENCE
            ):
                token_freq = self.provider.get_word_frequency(token)
                alt_sugs = [s for s in norm_sugs if s != token]
                if (
                    isinstance(token_freq, (int, float))
                    and token_freq >= _CONFUSABLE_SHORT_TOKEN_SUPPRESS_FREQ
                    and alt_sugs
                    and max(len(s) for s in alt_sugs) <= len(token) + 1
                ):
                    continue

            # Suppress confusable errors on high-frequency valid words with
            # attached Myanmar punctuation (၊ or ။).  The detector flags
            # "ရှိ၊" but "ရှိ" is correct — the comma is tokenization noise.
            _MY_PUNCT = "\u104a\u104b"  # ၊ and ။
            stripped_punct = token.rstrip(_MY_PUNCT)
            if (
                stripped_punct
                and stripped_punct != token
                and hasattr(self.provider, "is_valid_word")
                and self.provider.is_valid_word(stripped_punct)
                and hasattr(self.provider, "get_word_frequency")
            ):
                freq = self.provider.get_word_frequency(stripped_punct)
                if isinstance(freq, (int, float)) and freq >= 5000:
                    continue

            # Suppress confusable errors where word and top suggestion differ
            # only by a dot-below (့, U+1037).  This catches syntactic pairs
            # like သည် ↔ သည့် (declarative vs attributive) that text-level
            # detectors flag but are not actionable spelling errors.
            if e.suggestions:
                sug_str = str(e.suggestions[0])
                _DOT_BELOW = "\u1037"
                if (
                    sug_str == token + _DOT_BELOW
                    or token == sug_str + _DOT_BELOW
                    or (
                        len(sug_str) == len(token) + 1
                        and sug_str.replace(_DOT_BELOW, "", 1) == token
                    )
                    or (
                        len(token) == len(sug_str) + 1
                        and token.replace(_DOT_BELOW, "", 1) == sug_str
                    )
                ):
                    continue

            # One-character high-frequency particle swaps (e.g., က↔မ) are
            # highly ambiguous and generate many FPs in real-world prose.
            if len(token) == 1 and e.suggestions and hasattr(self.provider, "get_word_frequency"):
                suggestion = normalize(e.suggestions[0])
                if len(suggestion) == 1:
                    token_freq = self.provider.get_word_frequency(token)
                    suggestion_freq = self.provider.get_word_frequency(suggestion)
                    if (
                        isinstance(token_freq, (int, float))
                        and isinstance(suggestion_freq, (int, float))
                        and token_freq >= _CONFUSABLE_SHORT_TOKEN_SUPPRESS_FREQ
                        and suggestion_freq >= _CONFUSABLE_SHORT_TOKEN_SUPPRESS_FREQ
                    ):
                        continue

            filtered.append(e)

        errors[:] = filtered

    def _suppress_low_value_word_errors(
        self,
        errors: list[Error],
        text: str | None = None,
    ) -> None:
        """Suppress low-value invalid_word errors with high FP risk.

        Removes word errors where:
        - The word is actually valid in the dictionary (fallback check)
        - The word has high corpus frequency and no useful suggestions
        - The top suggestion is the word itself (ambiguous)
        """
        if not errors or not self.provider:
            return

        filtered: list[Error] = []
        for e in errors:
            if e.error_type != ET_WORD:
                filtered.append(e)
                continue

            token = normalize(e.text or "")
            if not token:
                filtered.append(e)
                continue

            # Suppress if word is actually valid (segmenter/SymSpell disagreement)
            if hasattr(self.provider, "is_valid_word") and self.provider.is_valid_word(token):
                continue

            # Suppress if word has attached boundary punctuation (quotes, brackets)
            # and the stripped form is a valid word.  Corpus text often has
            # "word" or (word) where the punctuation causes invalid_word FPs.
            stripped = token.strip('"\'\u201c\u201d\u2018\u2019()[]{}')
            if (
                stripped
                and stripped != token
                and hasattr(self.provider, "is_valid_word")
                and self.provider.is_valid_word(stripped)
            ):
                continue

            # Suppress high-frequency words with no suggestions
            if not e.suggestions and hasattr(self.provider, "get_word_frequency"):
                freq = self.provider.get_word_frequency(token)
                if isinstance(freq, (int, float)) and freq >= 1000:
                    continue

            # Suppress if top suggestion IS the word itself
            if e.suggestions:
                norm_sugs = [normalize(s) for s in e.suggestions if isinstance(s, str)]
                if norm_sugs and norm_sugs[0] == token:
                    continue

            filtered.append(e)

        errors[:] = filtered

    def _suppress_invalid_word_via_mlm(
        self,
        errors: list[Error],
        text: str,
    ) -> None:
        """Suppress invalid_word/dangling_word errors when MLM confirms context.

        For each error of type ``invalid_word`` or ``dangling_word``, mask the
        flagged word in the sentence and run the semantic model's
        ``predict_mask``.  If the original word appears among the top-K
        predictions (or as a prefix of a top-K prediction) with a probability
        above the configured threshold, the error is suppressed as a likely
        false positive.

        The threshold and top-K are controlled by
        ``ValidationConfig.mlm_plausibility_threshold`` and
        ``ValidationConfig.mlm_plausibility_top_k``.
        """
        semantic = getattr(self, "_semantic_checker", None)
        if semantic is None:
            return

        _MLM_TARGET_TYPES = frozenset({ET_WORD, ET_DANGLING_WORD})
        candidates = [
            (i, e) for i, e in enumerate(errors) if e.error_type in _MLM_TARGET_TYPES and e.text
        ]
        if not candidates:
            return

        threshold = self.config.validation.mlm_plausibility_threshold
        top_k = self.config.validation.mlm_plausibility_top_k

        to_remove: set[int] = set()
        for idx, err in candidates:
            word = normalize(err.text)
            if not word:
                continue

            try:
                predictions = semantic.predict_mask(
                    text,
                    word,
                    top_k=top_k,
                )
            except Exception:
                # Graceful degradation: if inference fails, keep the error.
                continue

            if not predictions:
                continue

            # Check if the original word (or a compound starting with it)
            # appears in the predictions above the threshold.
            for pred_word, score in predictions:
                if score < threshold:
                    # Predictions are sorted best-first; once we drop
                    # below the threshold we can stop.
                    break
                if pred_word == word or pred_word.startswith(word):
                    to_remove.add(idx)
                    break

        if to_remove:
            errors[:] = [e for i, e in enumerate(errors) if i not in to_remove]

    def _suppress_compound_split_valid_words(
        self,
        errors: list[Error],
    ) -> None:
        """Suppress invalid_word errors that split into all-valid dictionary words.

        For each ``invalid_word`` error, segment the token into syllables and
        attempt greedy dictionary-guided reassembly (longest valid word first,
        left-to-right with up to 4-syllable lookahead).  If every resulting
        segment is a valid dictionary word, the original token is a segmenter
        merge of valid parts — not a real spelling error — so suppress it.
        """
        provider = getattr(self, "provider", None)
        segmenter = getattr(self, "segmenter", None)
        if provider is None or segmenter is None:
            return

        to_remove: set[int] = set()
        for idx, err in enumerate(errors):
            if err.error_type != ET_WORD:
                continue
            word = normalize(err.text)
            if not word or len(word) < 4:
                continue

            # Segment into syllables
            try:
                syllables = segmenter.segment_syllables(word)
            except Exception:
                continue
            if len(syllables) < 2:
                continue

            # Greedy reassembly: longest valid dictionary word first
            parts: list[str] = []
            i = 0
            n = len(syllables)
            all_valid = True
            while i < n:
                best_len = 0
                upper = min(4, n - i)
                for k in range(upper, 0, -1):
                    candidate = "".join(syllables[i : i + k])
                    if provider.is_valid_word(candidate):
                        best_len = k
                        break
                if best_len > 0:
                    parts.append("".join(syllables[i : i + best_len]))
                    i += best_len
                else:
                    all_valid = False
                    break

            # If ALL parts are valid words AND the token has 3+ syllables,
            # this is very likely a segmenter merge of valid words — not a
            # real spelling error.  Require 3+ syllables because 2-syllable
            # tokens have higher overlap with genuine compound typos where
            # both syllables happen to be valid words individually.
            if all_valid and len(parts) >= 2 and len(syllables) >= 4:
                to_remove.add(idx)

        if to_remove:
            errors[:] = [e for i, e in enumerate(errors) if i not in to_remove]

    def _suppress_low_value_semantic_errors(
        self,
        errors: list[Error],
        text: str | None = None,
    ) -> None:
        """Suppress low-information semantic_error suggestions on stable nouns.

        Semantic proposals that replace a long, valid noun with very short
        pronoun-like alternatives are frequently non-actionable language-model
        drift rather than concrete spelling issues.
        """
        if not errors or not self.provider:
            return

        if not hasattr(self.provider, "is_valid_word") or not hasattr(
            self.provider, "get_word_frequency"
        ):
            return

        _AGENTIVE_PREFIXES = tuple(normalize(s) for s in ("က", "ကို", "ကတော့", "ကပဲ", "ကသာ", "သည်"))

        filtered: list[Error] = []
        for e in errors:
            if e.error_type != ET_SEMANTIC_ERROR:
                filtered.append(e)
                continue

            token = normalize(e.text or "")
            norm_sugs = [normalize(s) for s in (e.suggestions or []) if isinstance(s, str) and s]
            if not token or not norm_sugs:
                filtered.append(e)
                continue

            if len(token) < _SEMANTIC_STABLE_NOUN_MIN_LEN or not self.provider.is_valid_word(token):
                filtered.append(e)
                continue

            token_freq = self.provider.get_word_frequency(token)
            has_high_freq_signal = (
                isinstance(token_freq, (int, float)) and token_freq >= LEXICALIZED_COMPOUND_MIN_FREQ
            )

            top_sugs = norm_sugs[:3]
            if not top_sugs:
                filtered.append(e)
                continue

            short_suggestion_ratio = max(len(s) for s in top_sugs) / max(1, len(token))
            if short_suggestion_ratio > _SEMANTIC_SHORT_SUGGESTION_MAX_RATIO:
                filtered.append(e)
                continue

            has_agentive_context = False
            if text is not None and 0 <= e.position < len(text):
                end = e.position + len(e.text or "")
                if end <= len(text):
                    tail = normalize(text[end:])
                    has_agentive_context = any(
                        tail.startswith(prefix) for prefix in _AGENTIVE_PREFIXES
                    )

            if has_agentive_context or has_high_freq_signal:
                continue

            filtered.append(e)

        errors[:] = filtered

    def _suppress_low_value_syntax_errors(
        self,
        errors: list[Error],
        text: str | None = None,
    ) -> None:
        """Suppress low-value syntax_error artifacts on quotative connectors.

        Some colloquial quotative connectors (e.g. ``လို့``) are occasionally
        mis-tagged as ``syntax_error`` with generic hints like ``Invalid start``.
        Those alerts are noisy and do not provide actionable corrections.
        """
        if not errors:
            return

        filtered: list[Error] = []
        for e in errors:
            if e.error_type != ET_SYNTAX_ERROR:
                filtered.append(e)
                continue

            token = normalize(e.text or "")
            if token not in _LOW_VALUE_QUOTATIVE_SYNTAX_TOKENS:
                # Low-value short register/modal swap on valid particles.
                norm_sugs = [
                    normalize(s) for s in (e.suggestions or []) if isinstance(s, str) and s
                ]
                if (
                    text is not None
                    and len(norm_sugs) == 1
                    and len(token) <= _SYNTAX_SHORT_SWAP_MAX_TOKEN_LEN
                    and len(norm_sugs[0]) <= _SYNTAX_SHORT_SWAP_MAX_TOKEN_LEN
                    and token != norm_sugs[0]
                    and 0 <= e.position < len(text)
                    and hasattr(self.provider, "is_valid_word")
                    and hasattr(self.provider, "get_word_frequency")
                    and self.provider.is_valid_word(token)
                    and self.provider.is_valid_word(norm_sugs[0])
                ):
                    end = e.position + len(e.text or "")
                    right_boundary = (
                        end >= len(text) or text[end].isspace() or text[end] in _BOUNDARY_PUNCT
                    )
                    if not right_boundary:
                        filtered.append(e)
                        continue

                    token_freq = self.provider.get_word_frequency(token)
                    sug_freq = self.provider.get_word_frequency(norm_sugs[0])
                    if (
                        isinstance(token_freq, (int, float))
                        and isinstance(sug_freq, (int, float))
                        and token_freq >= LEXICALIZED_COMPOUND_MIN_FREQ
                        and sug_freq >= LEXICALIZED_COMPOUND_MIN_FREQ
                    ):
                        continue
                filtered.append(e)
                continue

            is_generic_invalid_start = any(
                isinstance(s, str) and "invalid start" in s.lower() for s in (e.suggestions or [])
            )
            if not is_generic_invalid_start:
                filtered.append(e)
                continue

            if text is not None:
                end = e.position + len(e.text or "")
                if not self._is_token_boundary_like(text, e.position, end):
                    filtered.append(e)
                    continue

            # Drop low-value quotative syntax artifact.
            continue

        errors[:] = filtered

    @staticmethod
    def _suppress_low_value_pos_sequence_errors(errors: list[Error]) -> None:
        """Suppress low-information POS-sequence artifacts.

        POS-sequence alerts with no actionable alternatives or only tiny
        function-word-like candidates are usually placeholders rather than
        concrete, user-fixable errors.
        """
        if not errors:
            return

        filtered: list[Error] = []
        for e in errors:
            if e.error_type != ET_POS_SEQUENCE_ERROR:
                filtered.append(e)
                continue

            token = normalize(e.text or "")
            if not token:
                continue

            norm_sugs = [normalize(s) for s in (e.suggestions or []) if isinstance(s, str) and s]

            # Very long spans with no actionable candidates are usually
            # template artifacts rather than concrete fixable errors.
            if len(token) >= _POS_SEQ_LONG_SPAN_NO_SUGGESTION_LEN and not norm_sugs:
                continue

            # Very long span + only tiny alternatives is also typically noise.
            if (
                len(token) >= _POS_SEQ_MEDIUM_SPAN_TINY_SUGGESTION_LEN
                and norm_sugs
                and all(len(s) <= _POS_SEQ_TINY_SUGGESTION_MAX_LEN for s in norm_sugs[:3])
            ):
                continue

            filtered.append(e)

        errors[:] = filtered

    def _suppress_low_value_syllable_errors(
        self,
        errors: list[Error],
        text: str | None = None,
    ) -> None:
        """Suppress low-value invalid_syllable fragments inside technical compounds."""
        if not errors or text is None:
            return

        filtered: list[Error] = []
        for e in errors:
            if e.error_type != ET_SYLLABLE:
                filtered.append(e)
                continue

            token = normalize(e.text or "")
            if not token or len(token) < _SYLLABLE_TECH_COMPOUND_MIN_TOKEN_LEN:
                filtered.append(e)
                continue

            end = e.position + len(e.text or "")
            if end > len(text):
                filtered.append(e)
                continue

            if token.endswith(normalize("ဂ်")) and not self._is_token_boundary_like(
                text, e.position, end
            ):
                continue

            filtered.append(e)

        errors[:] = filtered

    def _filter_syllable_errors_in_valid_words(
        self,
        text: str,
        errors: list[Error],
        words: list[str] | None = None,
    ) -> None:
        """Filter out syllable errors that occur within valid words.

        Performance: O(n log m) where n = errors, m = valid words.
        Uses binary search instead of O(n*m) linear scan. Accepts pre-segmented
        words to avoid redundant segmentation and uses batch validity check
        to avoid N+1 queries.
        """
        if not errors:
            return

        if words is None:
            words = self.segmenter.segment_words(text)

        # Include adjacent-word compounds in validity check so that
        # errors crossing word boundaries (e.g., ဟစ inside ဂေဟ+စနစ်
        # → ဂေဟစနစ်) can be filtered when the compound is valid.
        word_set = set(words)
        word_positions_for_compound: list[tuple[int, int]] = []
        cursor_c = 0
        for w in words:
            w_pos = text.find(w, cursor_c)
            if w_pos != -1:
                word_positions_for_compound.append((w_pos, w_pos + len(w)))
                cursor_c = w_pos + len(w)
        for i in range(len(word_positions_for_compound) - 1):
            _, end_i = word_positions_for_compound[i]
            start_j, _ = word_positions_for_compound[i + 1]
            if end_i == start_j:  # character-adjacent (no space)
                compound = text[
                    word_positions_for_compound[i][0] : word_positions_for_compound[i + 1][1]
                ]
                word_set.add(compound)
        unique_words = list(word_set)
        validity_map = self.provider.is_valid_words_bulk(unique_words)

        filtered_errors = filter_syllable_errors_in_valid_words(text, errors, words, validity_map)

        errors.clear()
        errors.extend(filtered_errors)

    # --- Deduplication methods ---------------------------------------------------

    def _dedup_errors_by_position(self, errors: list[Error]) -> None:
        """Deduplicate errors at the same position, keeping the best one.

        Priority rules:
        1. Text-level detector errors (with suggestions) beat context_probability
        2. Longer span with suggestions beats shorter span
        3. Higher confidence beats lower confidence (equal length)
        4. Equal confidence: longer span wins
        5. Carry over suggestions from displaced errors
        """
        if not errors:
            return

        best_by_pos: dict[int, Error] = {}
        for e in errors:
            pos = e.position
            if pos not in best_by_pos:
                best_by_pos[pos] = e
                continue
            prev = best_by_pos[pos]
            e_len = len(e.text or "")
            prev_len = len(prev.text or "")

            e_is_detector = e.error_type in _TEXT_DETECTOR_TYPES
            prev_is_detector = prev.error_type in _TEXT_DETECTOR_TYPES
            if e_is_detector and not prev_is_detector and e.suggestions and e_len == prev_len:
                best_by_pos[pos] = e
                continue
            if prev_is_detector and not e_is_detector and prev.suggestions and e_len == prev_len:
                continue

            # Root-cause context errors (homophone/confusable) with suggestions
            # beat wider invalid_word at same position.  The wider span would be
            # removed by span dedup anyway (it contains sub-errors), but by then
            # the precise narrow detection is already lost.
            # Guard: confusable_error fragments must not displace invalid_word.
            if (
                e.error_type in _ROOT_CAUSE_NARROW_TYPES
                and prev.error_type in _GENERIC_WIDE_TYPES
                and e.suggestions
                and e_len < prev_len
                and not (e.error_type == ET_CONFUSABLE_ERROR and prev.error_type == ET_WORD)
            ):
                best_by_pos[pos] = e
                continue
            if (
                prev.error_type in _ROOT_CAUSE_NARROW_TYPES
                and e.error_type in _GENERIC_WIDE_TYPES
                and prev.suggestions
                and prev_len < e_len
                and not (prev.error_type == ET_CONFUSABLE_ERROR and e.error_type == ET_WORD)
            ):
                continue

            if e_len > prev_len:
                best_by_pos[pos] = e
                if e.suggestions and prev.suggestions:
                    for s in prev.suggestions:
                        if s not in e.suggestions:
                            e.suggestions.append(s)
                elif not e.suggestions and prev.suggestions:
                    e.suggestions = prev.suggestions
            elif e.confidence > prev.confidence and e_len >= prev_len:
                best_by_pos[pos] = e
                if e.suggestions and prev.suggestions:
                    for s in prev.suggestions:
                        if s not in e.suggestions:
                            e.suggestions.append(s)
                elif not e.suggestions and prev.suggestions:
                    e.suggestions = prev.suggestions
            elif not prev.suggestions and e.suggestions:
                prev.suggestions = e.suggestions
            elif prev.suggestions and e.suggestions:
                for s in e.suggestions:
                    if s not in prev.suggestions:
                        prev.suggestions.append(s)
        errors.clear()
        errors.extend(best_by_pos.values())

    def _dedup_errors_by_span(self, errors: list[Error]) -> None:
        """Remove errors fully contained within a larger error's span.

        Exceptions:
        - Narrow suffix root-cause fixes (missing asat/visarga) co-exist with
          wider POS-sequence errors.
        - Narrow invalid_syllable with suggestions replaces wider context_probability.
        - Narrow root-cause types (confusable_error, homophone_error,
          medial_confusion) with suggestions replace wider generic spans
          (pos_sequence_error, context_probability, invalid_word).
        """
        if len(errors) < 2:
            return

        errors.sort(key=lambda e: (e.position, -(len(e.text) if e.text else 0)))
        kept: list[Error] = []
        for e in errors:
            e_start = e.position
            e_end = e_start + (len(e.text) if e.text else 0)
            subsumed = False
            for k in kept:
                k_start = k.position
                k_end = k_start + (len(k.text) if k.text else 0)
                if e_start >= k_start and e_end <= k_end and e is not k:
                    _is_suffix_root_cause_type = e.error_type in (
                        ET_SYLLABLE,
                        ET_MISSING_ASAT,
                    )
                    _is_suffix_particle = (
                        e.text in self._MISSING_ASAT_PARTICLES
                        or e.text in self._MISSING_VISARGA_SUFFIXES
                        or (
                            bool(e.text)
                            and bool(e.suggestions)
                            and e.text.endswith("\u103a")
                            and e.suggestions[0] == e.text[:-1]
                        )
                    )
                    keep_nested_suffix_root_cause = (
                        _is_suffix_root_cause_type
                        and bool(e.suggestions)
                        and e_start > k_start
                        and k.error_type == ET_POS_SEQUENCE_ERROR
                        and _is_suffix_particle
                        # SYLLABLE type co-exists, MISSING_ASAT replaces
                        and e.error_type == ET_SYLLABLE
                    )
                    if keep_nested_suffix_root_cause:
                        continue
                    # Narrow suffix root-cause replaces wider errors
                    if (
                        _is_suffix_root_cause_type
                        and k.error_type
                        in (
                            ET_CONTEXT_PROBABILITY,
                            ET_WORD,
                        )
                        and e.suggestions
                    ):
                        kept[:] = [x for x in kept if x is not k]
                        break
                    # MISSING_ASAT replaces wider pos_sequence_error too
                    if (
                        e.error_type == ET_MISSING_ASAT
                        and k.error_type == ET_POS_SEQUENCE_ERROR
                        and bool(e.suggestions)
                        and e_start > k_start
                        and _is_suffix_particle
                    ):
                        kept[:] = [x for x in kept if x is not k]
                        break
                    # Narrow root-cause types (confusable, homophone,
                    # medial_confusion) with suggestions replace wider
                    # generic spans (pos_sequence, context_prob, invalid_word).
                    # Guard: confusable_error on a fragment must not displace
                    # a definite invalid_word that covers the whole token.
                    _confusable_displaces_oov = (
                        e.error_type == ET_CONFUSABLE_ERROR and k.error_type == ET_WORD
                    )
                    if (
                        e.error_type in _ROOT_CAUSE_NARROW_TYPES
                        and bool(e.suggestions)
                        and k.error_type in _GENERIC_WIDE_TYPES
                        and not _confusable_displaces_oov
                    ):
                        kept[:] = [x for x in kept if x is not k]
                        break
                    subsumed = True
                    break
            if not subsumed:
                kept.append(e)
        errors.clear()
        errors.extend(kept)

    @staticmethod
    def _suppress_tense_adjacent_syntax(errors: list[Error]) -> None:
        """Suppress syntax_error on SFPs immediately following tense_mismatch errors."""
        if len(errors) < 2:
            return
        tense_ends: set[int] = set()
        for e in errors:
            if e.error_type == ET_TENSE_MISMATCH and e.text:
                tense_ends.add(e.position + len(e.text))
        if tense_ends:
            errors[:] = [
                e
                for e in errors
                if not (e.error_type == ET_SYNTAX_ERROR and e.position in tense_ends)
            ]

    @staticmethod
    def _suppress_known_entity_errors(errors: list[Error], text: str) -> None:
        """Suppress errors on tokens that match the named entity gazetteer.

        Lightweight supplement to :meth:`_filter_ner_entities` -- works without
        the transformer NER model by checking against a curated whitelist
        of known Myanmar names, places, organizations, and religious terms
        loaded from ``rules/named_entities.yaml``.
        """
        if not errors:
            return

        from myspellchecker.text.ner import is_known_entity

        filtered: list[Error] = []
        for e in errors:
            # Text-level detector errors are immune (same policy as NER filtering)
            err_type = getattr(e, "error_type", "")
            if err_type in _NER_IMMUNE:
                filtered.append(e)
                continue

            token = e.text or ""
            # Check if the error token is a known entity
            if token and is_known_entity(token):
                # Still keep high-confidence errors with suggestions
                if e.confidence >= _NER_HIGH_CONFIDENCE_OVERRIDE and e.suggestions:
                    filtered.append(e)
                # Otherwise suppress -- it's a known name/place/entity
                continue

            filtered.append(e)

        errors[:] = filtered

    def _filter_ner_entities(self, errors: list[Error], text: str) -> None:
        """Remove errors that overlap with recognized named entities.

        Runs after all dedup to filter across ALL layers (syllable, word, context).
        Text-level detector errors are immune to NER suppression.
        """
        if not self._ner_model or not errors:
            return

        from myspellchecker.text.ner_model import EntityType as _ET

        ner_cfg = self.config.ner
        active_types = set(ner_cfg.ner_entity_types) if ner_cfg else {"PER"}
        loc_conf = ner_cfg.loc_confidence_threshold if ner_cfg else _NER_LOC_DEFAULT_CONFIDENCE

        try:
            entities = self._ner_model.extract_entities(text)
            if not entities:
                return

            general_spans: set[int] = set()
            loc_spans: set[int] = set()
            type_map = {
                "PER": _ET.PERSON,
                "LOC": _ET.LOCATION,
                "ORG": _ET.ORGANIZATION,
                "DATE": _ET.DATE,
                "NUM": _ET.NUMBER,
                "TIME": _ET.TIME,
            }
            for entity in entities:
                matched_type = None
                for code, et in type_map.items():
                    if entity.label == et and code in active_types:
                        matched_type = code
                        break
                if matched_type is None:
                    continue
                if matched_type == "LOC":
                    if entity.confidence < loc_conf:
                        continue
                    for pos in range(entity.start, entity.end):
                        loc_spans.add(pos)
                else:
                    for pos in range(entity.start, entity.end):
                        general_spans.add(pos)

            if not general_spans and not loc_spans:
                return

            filtered: list[Error] = []
            for e in errors:
                err_type = getattr(e, "error_type", "")
                if err_type in _NER_IMMUNE:
                    filtered.append(e)
                    continue
                e_range = range(e.position, e.position + len(e.text or ""))
                if general_spans and any(pos in general_spans for pos in e_range):
                    if e.confidence >= _NER_HIGH_CONFIDENCE_OVERRIDE and e.suggestions:
                        filtered.append(e)
                    continue
                if loc_spans and any(pos in loc_spans for pos in e_range):
                    if err_type in _LOC_SUPPRESSIBLE:
                        continue
                filtered.append(e)
            errors[:] = filtered
        except (RuntimeError, ValueError, TypeError) as exc:
            self.logger.debug(f"NER entity filtering skipped: {exc}")

    # --- Pre-normalization error merge helpers ------------------------------------

    @staticmethod
    def _merge_pre_norm_errors(
        errors: list[Error],
        pre_errors: Sequence[Error],
        *,
        replace_mode: str = "if_longer_or_equal",
    ) -> None:
        """Merge pre-normalization errors into the main error list.

        Args:
            errors: Main error list (modified in place).
            pre_errors: Pre-normalization errors to merge.
            replace_mode: How to handle position conflicts:
                - "if_longer_or_equal": Replace if pre-norm span >= existing (default)
                - "if_strictly_longer": Replace only if pre-norm span > existing
                - "always": Always replace at same position
                - "add_only": Only add if position not occupied
        """
        if not pre_errors:
            return

        existing_by_pos = {e.position: i for i, e in enumerate(errors)}
        for pe in pre_errors:
            if pe.position in existing_by_pos:
                idx = existing_by_pos[pe.position]
                pe_len = len(pe.text) if pe.text else 0
                ex_len = len(errors[idx].text) if errors[idx].text else 0

                if replace_mode == "add_only":
                    continue
                elif replace_mode == "always":
                    errors[idx] = pe
                elif replace_mode == "if_strictly_longer":
                    if pe_len > ex_len:
                        errors[idx] = pe
                elif pe_len >= ex_len:
                    errors[idx] = pe
                elif pe.suggestions:
                    for s in reversed(pe.suggestions):
                        if s not in errors[idx].suggestions:
                            errors[idx].suggestions.insert(0, s)
            else:
                errors.append(pe)

    def _dedup_pre_norm_overlaps(self, errors: list[Error]) -> list[Error]:
        """Dedup pre-normalization errors with partial overlap handling.

        Unlike ``_dedup_errors_by_span`` (which handles fully-contained errors),
        this also handles partial overlaps where a text-level pre-normalization
        detector error takes priority.
        """
        if len(errors) < 2:
            return errors

        _TEXT_LEVEL = frozenset(
            {
                ET_BROKEN_VIRAMA,
                ET_BROKEN_STACKING,
                ET_INCOMPLETE_STACKING,
                ET_LEADING_VOWEL_E,
                ET_MEDIAL_ORDER_ERROR,
            }
        )
        errors.sort(key=lambda e: (e.position, -(len(e.text) if e.text else 0)))
        kept: list[Error] = []
        for e in errors:
            e_start = e.position
            e_end = e_start + (len(e.text) if e.text else 0)
            skip = False
            for k in kept:
                k_start = k.position
                k_end = k_start + (len(k.text) if k.text else 0)
                if e_start >= k_start and e_end <= k_end and e is not k:
                    _is_suffix_rc = e.error_type in (
                        ET_SYLLABLE,
                        ET_MISSING_ASAT,
                    )
                    _is_suffix_particle = (
                        e.text in self._MISSING_ASAT_PARTICLES
                        or e.text in self._MISSING_VISARGA_SUFFIXES
                        or (
                            bool(e.text)
                            and bool(e.suggestions)
                            and e.text.endswith("\u103a")
                            and e.suggestions[0] == e.text[:-1]
                        )
                    )
                    keep_nested_suffix_root_cause = (
                        _is_suffix_rc
                        and bool(e.suggestions)
                        and e_start > k_start
                        and k.error_type == ET_POS_SEQUENCE_ERROR
                        and _is_suffix_particle
                        and e.error_type == ET_SYLLABLE
                    )
                    if keep_nested_suffix_root_cause:
                        continue
                    # MISSING_ASAT replaces wider pos_sequence_error
                    if (
                        e.error_type == ET_MISSING_ASAT
                        and k.error_type == ET_POS_SEQUENCE_ERROR
                        and bool(e.suggestions)
                        and e_start > k_start
                        and _is_suffix_particle
                    ):
                        kept[:] = [x for x in kept if x is not k]
                        break
                    # Narrow root-cause types with suggestions replace
                    # wider generic spans (same logic as _dedup_errors_by_span).
                    _confusable_displaces_oov = (
                        e.error_type == ET_CONFUSABLE_ERROR and k.error_type == ET_WORD
                    )
                    if (
                        e.error_type in _ROOT_CAUSE_NARROW_TYPES
                        and bool(e.suggestions)
                        and k.error_type in _GENERIC_WIDE_TYPES
                        and not _confusable_displaces_oov
                    ):
                        kept[:] = [x for x in kept if x is not k]
                        break
                    skip = True
                    break
                if e_start < k_end and e_end > k_start:
                    if k.error_type in _TEXT_LEVEL:
                        skip = True
                        break
                    if e.error_type == k.error_type:
                        k_len = k_end - k_start
                        e_len = e_end - e_start
                        if k_len > e_len:
                            skip = True
                            break
                        if e_len > k_len:
                            kept[:] = [x for x in kept if x is not k]
                            break
            if not skip:
                kept.append(e)
        return kept

    @staticmethod
    def _merge_duplicate_diacritic_errors(errors: list[Error], dde_errors: Sequence[Error]) -> None:
        """Merge duplicate diacritic errors with span-coverage check."""
        if not dde_errors:
            return

        existing_by_pos = {e.position: i for i, e in enumerate(errors)}
        for dde in dde_errors:
            dde_start = dde.position
            dde_end = dde_start + (len(dde.text) if dde.text else 0)

            span_covered = False
            for existing in errors:
                e_start = existing.position
                e_end = e_start + (len(existing.text) if existing.text else 0)
                if dde_start >= e_start and dde_end <= e_end:
                    if dde_start == e_start and dde_end == e_end:
                        continue
                    span_covered = True
                    break
            if span_covered:
                continue

            if dde_start in existing_by_pos:
                idx = existing_by_pos[dde_start]
                errors[idx] = dde
            else:
                errors.append(dde)
