"""Sentence structure detection mixin for SentenceDetectorsMixin.

Provides ``_detect_sentence_structure_issues`` and its associated
module-level and class-level data constants.

Data is loaded from ``rules/grammar_rules.yaml`` (``structure_conjunctions``,
``suffix_safe``, and ``word_order_verb_endings`` sections) at module level,
with fallback to hardcoded defaults if the YAML file is missing or invalid.

Extracted from ``sentence_detectors.py`` to reduce file size while
preserving the exact same method signatures and behaviour.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from myspellchecker.core.constants import (
    ET_DANGLING_WORD,
    ET_MISSING_CONJUNCTION,
    ET_POS_SEQUENCE_ERROR,
)

if TYPE_CHECKING:
    from myspellchecker.providers.base import DictionaryProvider
from myspellchecker.core.detector_data import (
    TEXT_DETECTOR_CONFIDENCES,
)
from myspellchecker.core.detector_data import (
    norm_set as _norm_set,
)
from myspellchecker.core.detectors.utils import get_existing_positions
from myspellchecker.core.response import Error, SyllableError
from myspellchecker.text.normalize import normalize
from myspellchecker.utils.logging_utils import get_logger

logger = get_logger(__name__)

# --- Module-level frozensets (hoisted from method bodies) ---

# ── Hardcoded defaults (fallback when YAML is unavailable) ──

# Known conjunctions/linkers that connect clauses (sentence structure detection).
_DEFAULT_CONJUNCTIONS: frozenset[str] = _norm_set(
    {
        "ပြီး",
        "ပြီးတော့",
        "လို့",
        "ဆိုတာ",
        "ဆိုပြီး",
        "ဆိုတော့",
        "ဆိုရင်",
        "ကြောင့်",
        "တော့",
        "ရင်",
        "လျှင်",
        "သော်လည်း",
        "ပေမယ့်",
        "ဒါပေမယ့်",
        "လည်း",
        "ဟု",
        "၍",
    }
)

# Unambiguous sentence-final suffixes for structure detection.
_DEFAULT_SUFFIX_SAFE: frozenset[str] = _norm_set(
    {
        "ပါသည်",
        "ပါတယ်",
        "ပါမည်",
        "ပါမယ်",
        "ပါသည",
        "ပါတယ",
        "ပါမည",
        "ပါမယ",  # asat-stripped
        "တယ်",
        "မယ်",
        "တယ",
        "မယ",  # asat-stripped
        "ပြီ",  # completive aspect (always sentence-final as suffix)
        "ပါပြီ",  # polite completive
    }
)

# Verb endings for word-order inversion detection (G04).
_DEFAULT_WORD_ORDER_VERB_ENDINGS: tuple[str, ...] = tuple(
    normalize(s)
    for s in (
        "ခဲ့သည်",
        "သည်",
        "တယ်",
        "မယ်",
        "ခဲ့တယ်",
        "ခဲ့မယ်",
        "ပါသည်",
        "ပါတယ်",
    )
)

# ── YAML loading ──

_YAML_PATH = Path(__file__).resolve().parent.parent.parent.parent / "rules" / "grammar_rules.yaml"


def _load_structure_data() -> tuple[frozenset[str], frozenset[str], tuple[str, ...]]:
    """Load structure detection data from grammar_rules.yaml with fallback.

    Reads the ``structure_conjunctions``, ``suffix_safe``, and
    ``word_order_verb_endings`` lists from grammar_rules.yaml and returns
    normalized values. Falls back to hardcoded defaults if the YAML file
    is missing, invalid, or individual sections are absent.

    Returns:
        Tuple of (conjunctions, suffix_safe, word_order_verb_endings).
    """
    if not _YAML_PATH.exists():
        logger.debug(
            "Grammar rules YAML not found at %s, using default structure data",
            _YAML_PATH,
        )
        return _DEFAULT_CONJUNCTIONS, _DEFAULT_SUFFIX_SAFE, _DEFAULT_WORD_ORDER_VERB_ENDINGS

    try:
        import yaml  # type: ignore[import-untyped]

        with open(_YAML_PATH, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not data or not isinstance(data, dict):
            logger.warning("Grammar rules YAML empty or invalid, using default structure data")
            return _DEFAULT_CONJUNCTIONS, _DEFAULT_SUFFIX_SAFE, _DEFAULT_WORD_ORDER_VERB_ENDINGS

        # Parse structure_conjunctions
        raw_conjunctions = data.get("structure_conjunctions", [])
        if isinstance(raw_conjunctions, list) and raw_conjunctions:
            conjunctions = _norm_set(raw_conjunctions)
        else:
            conjunctions = _DEFAULT_CONJUNCTIONS

        # Parse suffix_safe
        raw_suffix_safe = data.get("suffix_safe", [])
        if isinstance(raw_suffix_safe, list) and raw_suffix_safe:
            suffix_safe = _norm_set(raw_suffix_safe)
        else:
            suffix_safe = _DEFAULT_SUFFIX_SAFE

        # Parse word_order_verb_endings
        raw_verb_endings = data.get("word_order_verb_endings", [])
        if isinstance(raw_verb_endings, list) and raw_verb_endings:
            verb_endings = tuple(normalize(s) for s in raw_verb_endings)
        else:
            verb_endings = _DEFAULT_WORD_ORDER_VERB_ENDINGS

        logger.debug(
            "Loaded structure data from YAML: %d conjunctions, %d suffix_safe, "
            "%d word_order_verb_endings",
            len(conjunctions),
            len(suffix_safe),
            len(verb_endings),
        )
        return conjunctions, suffix_safe, verb_endings

    except Exception:
        logger.warning(
            "Failed to load structure data from YAML, using defaults",
            exc_info=True,
        )
        return _DEFAULT_CONJUNCTIONS, _DEFAULT_SUFFIX_SAFE, _DEFAULT_WORD_ORDER_VERB_ENDINGS


# Load at module level (once, at import time)
_CONJUNCTIONS: frozenset[str]
_SUFFIX_SAFE: frozenset[str]
_WORD_ORDER_VERB_ENDINGS: tuple[str, ...]
_CONJUNCTIONS, _SUFFIX_SAFE, _WORD_ORDER_VERB_ENDINGS = _load_structure_data()

# Temporal/transitional expressions that validly begin a new clause
# after a sentence-final particle without conjunction.
_TEMPORAL_ADVERBS: frozenset[str] = _norm_set(
    {
        "နောက်မှ",
        "ပြီးတော့",
        "ပြီးရင်",
        "ထို့နောက်",
        "နောက်",
        "ပြီး",
        "ဒါပေမယ့်",
        "သို့သော်",
        "သို့သော်လည်း",
        "ဒါပေမဲ့",
        "ဒါကြောင့်",
        "ထို့ကြောင့်",
        "ထို့အပြင်",
        "ပြီးတော့မှ",
        "ထို့ပြီးတော့",
        # Short temporal modifiers that validly follow verb endings
        "စောစော",  # early/soon
        "ခဏ",  # briefly/a moment
    }
)

# Short temporal modifiers that validly follow verb endings but are NOT
# clause starters (they modify the preceding clause, not start a new one).
_TEMPORAL_ONLY_ADVERBS: frozenset[str] = _norm_set({"စောစော", "ခဏ"})

# Additional clause starters not in _TEMPORAL_ADVERBS.
_ADDITIONAL_STARTERS: frozenset[str] = _norm_set({"အခုတော့", "အခု"})

# Continuation starters: words that commonly begin a new clause after
# a sentence-final particle without needing an explicit conjunction.
# Computed from _TEMPORAL_ADVERBS (minus temporal-only modifiers) plus
# additional starters.
_CONTINUATION_STARTERS: frozenset[str] = (
    _TEMPORAL_ADVERBS - _TEMPORAL_ONLY_ADVERBS
) | _ADDITIONAL_STARTERS

_COLLOQUIAL_SFP_SUFFIXES: tuple[str, ...] = tuple(
    normalize(s) for s in ("တယ်", "မယ်", "ပါတယ်", "ပါမယ်", "တယ", "မယ")
)
_BOUNDARY_PUNCT_CHARS = "၊။,.!?;:\"'()[]{}"
_WORD_ORDER_CASE_SUFFIXES: tuple[str, ...] = tuple(
    normalize(s) for s in ("ကို", "အား", "မှာ", "တွင်", "သို့", "ဖြင့်", "အတွက်")
)
_WORD_ORDER_BOUNDARIES: frozenset[str] = _norm_set(
    {
        "ဟု",
        "လို့",
        "ဆိုပြီး",
        "ဆိုတာ",
        "ဆိုသည်",
        "ဆိုတယ်",
        "ကြောင်း",
    }
)


class StructureDetectionMixin:
    """Mixin providing sentence structure detection.

    Detects G02 (dangling word) and G03 (missing conjunction) patterns
    across the full text.
    """

    # --- Type stubs for attributes provided by SpellChecker or sibling mixins ---
    provider: "DictionaryProvider"
    _ALL_ENDINGS_WITH_STRIPPED: frozenset[str]
    _DISCOURSE_ENDINGS: frozenset[str]
    _HONORIFIC_TERMS: frozenset[str]
    _INFORMAL_PARTICLES: frozenset[str]

    # --- Class-level constants (extracted from inline magic numbers) ---

    # Word order detection (G04): maximum tail tokens to scan for displaced arguments
    _WORD_ORDER_MAX_TAIL_TOKENS: int = 4

    # Colloquial parataxis suppression (G03): minimum words between SFPs to suppress
    _COLLOQUIAL_SFP_MIN_WORDS_BETWEEN: int = 2

    # Colloquial parataxis suppression (G03): minimum existing errors to suppress
    _COLLOQUIAL_SFP_MIN_EXISTING_ERRORS: int = 2

    # G02 dangling word: suppress for high-frequency function words.
    # Words above this corpus frequency are common particles/markers that
    # naturally appear at clause boundaries and should not be flagged.
    _DANGLING_HIGH_FREQ_THRESHOLD: int = 5_000

    def _detect_sentence_structure_issues(self, text: str, errors: list[Error]) -> None:
        """Detect sentence structure issues across the full text.

        Detects two patterns:
        - G02 (dangling word): A content word after a sentence-final particle
          that has no syntactic connection to the sentence.
        - G03 (missing conjunction): Two sentence-final particles without
          a conjunction/linker between them.

        Runs on full text because the sentence segmenter splits at
        sentence-final particles, preventing per-sentence detection.
        """
        from myspellchecker.core.detectors.utils import get_tokenized

        tokenized = get_tokenized(self, text)
        if len(tokenized) < 2:
            return

        tokens = tokenized.tokens
        token_positions = tokenized.positions

        is_sent_final: list[bool] = []
        for token in tokens:
            matched = False
            # Strip trailing boundary punctuation (။, ၊, etc.) so that
            # tokens like "သည်။" and "တယ်။" still match the SFP sets.
            token_stripped = token.rstrip(_BOUNDARY_PUNCT_CHARS)
            # Exact match against all endings
            if token_stripped in self._ALL_ENDINGS_WITH_STRIPPED:
                matched = True
            else:
                # Suffix match for unambiguous sentence-final endings
                for ending in _SUFFIX_SAFE:
                    if len(ending) < len(token_stripped) and token_stripped.endswith(ending):
                        matched = True
                        break
            is_sent_final.append(matched)

        existing_positions = get_existing_positions(errors)
        has_honorific = any(h in text for h in self._HONORIFIC_TERMS)

        # Temporal/transitional expressions that validly begin a new clause
        # after a sentence-final particle without conjunction.
        # G02: Dangling word — content word after last sentence-final particle
        # Find the last sentence-final token that is NOT the last overall token
        for i in range(len(tokens) - 2, -1, -1):
            if is_sent_final[i]:
                for j in range(i + 1, len(tokens)):
                    next_token = tokens[j]
                    next_pos = token_positions[j]
                    # Strip trailing punctuation for frozenset lookups so
                    # tokens like "စောစော။" match "စောစော" in the sets.
                    # Keep raw next_token for error text reporting.
                    next_stripped = next_token.rstrip(_BOUNDARY_PUNCT_CHARS)
                    if next_stripped in _CONJUNCTIONS:
                        break  # conjunction found, valid structure
                    if next_stripped in _TEMPORAL_ADVERBS:
                        break  # temporal/transitional adverb, valid clause start
                    if is_sent_final[j]:
                        break  # another sentence-final, handled by G03
                    # Pure punctuation tokens (e.g., "၊", "။") are boundary
                    # markers, not dangling content words.
                    if not next_stripped:
                        continue
                    # Multi-clause parataxis: if there's a subsequent SFP,
                    # this word starts a new clause, not dangling.
                    # e.g., "ခေါင်းကိုက်တယ် ဆေးရုံမှာ သွားခဲ့တယ်"
                    has_later_sfp = any(is_sent_final[k] for k in range(j + 1, len(tokens)))
                    if has_later_sfp:
                        continue
                    # High-frequency words are common particles/markers (e.g.,
                    # ခဲ့, ဖြစ်, များ, က, ပါ) that naturally appear at clause
                    # boundaries.  Suppress dangling-word FPs for these.
                    provider = getattr(self, "provider", None)
                    if provider is not None:
                        try:
                            freq = provider.get_word_frequency(next_stripped)
                            if (
                                isinstance(freq, (int, float))
                                and freq >= self._DANGLING_HIGH_FREQ_THRESHOLD
                            ):
                                continue
                        except Exception:
                            pass
                    # This is a dangling content word
                    if next_pos not in existing_positions:
                        dangling_suggestions: list[str] = [""]
                        if has_honorific and next_token in self._INFORMAL_PARTICLES:
                            dangling_suggestions = ["ပါ", "ရှင့်"]
                        errors.append(
                            SyllableError(
                                text=next_token,
                                position=next_pos,
                                suggestions=dangling_suggestions,
                                confidence=TEXT_DETECTOR_CONFIDENCES["dangling_word"],
                                error_type=ET_DANGLING_WORD,
                            )
                        )
                break  # only check the last sentence-final

        # G03: Missing conjunction -- two sentence-final particles without linker
        sent_final_indices = [i for i, sf in enumerate(is_sent_final) if sf]
        if len(sent_final_indices) >= 2:
            for k in range(len(sent_final_indices) - 1):
                idx1 = sent_final_indices[k]
                idx2 = sent_final_indices[k + 1]
                # Discourse particles (နော်, ပေါ့, ပဲ, etc.) naturally follow
                # verb-final particles — don't flag as missing conjunction.
                if tokens[idx2] in self._DISCOURSE_ENDINGS:
                    continue
                # Colloquial parataxis: in spoken Myanmar, clauses are commonly
                # juxtaposed without explicit conjunction (e.g., "ခေါင်းကိုက်တယ်
                # ဆေးရုံမှာ သွားခဲ့တယ်"). When both SFPs end with colloquial
                # particles, suppress under these conditions:
                # 1. Substantial text between SFPs (words_between >= 2), OR
                # 2. The sentence already has other detected errors — adding a
                #    conjunction FP on error-laden text is noise.
                # When SFPs are adjacent in an otherwise clean sentence, the
                # conjunction is genuinely missing.
                t1, t2 = tokens[idx1], tokens[idx2]
                words_between = idx2 - idx1 - 1
                both_colloquial = any(t1.endswith(s) for s in _COLLOQUIAL_SFP_SUFFIXES) and any(
                    t2.endswith(s) for s in _COLLOQUIAL_SFP_SUFFIXES
                )
                if both_colloquial and (
                    words_between >= self._COLLOQUIAL_SFP_MIN_WORDS_BETWEEN
                    or len(existing_positions) >= self._COLLOQUIAL_SFP_MIN_EXISTING_ERRORS
                ):
                    continue
                has_conjunction = any(
                    tokens[m] in _CONJUNCTIONS or tokens[m] in _CONTINUATION_STARTERS
                    for m in range(idx1 + 1, idx2)
                )
                if not has_conjunction:
                    # When SFPs are close (0-1 words between), the first SFP
                    # should be a conjunction (e.g., တယ် → ပြီး).
                    # When far apart, flag the boundary (insert period/conjunction).
                    _SFP_TO_CONJ = {
                        normalize(k): normalize(v)
                        for k, v in {
                            "တယ်": "ပြီး",
                            "ပါတယ်": "ပြီး",
                            "သည်": "ပြီး",
                            "ပါသည်": "ပြီး",
                        }.items()
                    }
                    sfp_flagged = False
                    if words_between <= 1:
                        # Try to flag the first SFP suffix
                        for sfp, conj in sorted(
                            _SFP_TO_CONJ.items(), key=lambda x: len(x[0]), reverse=True
                        ):
                            if t1 == sfp or (len(sfp) < len(t1) and t1.endswith(sfp)):
                                err_pos = token_positions[idx1] + len(t1) - len(sfp)
                                if err_pos not in existing_positions:
                                    errors.append(
                                        SyllableError(
                                            text=sfp,
                                            position=err_pos,
                                            suggestions=[conj],
                                            confidence=TEXT_DETECTOR_CONFIDENCES[
                                                "missing_conjunction"
                                            ],
                                            error_type=ET_MISSING_CONJUNCTION,
                                        )
                                    )
                                    sfp_flagged = True
                                break
                    if not sfp_flagged:
                        # Flag the second SFP (redundant) rather than just the
                        # boundary space, so the span covers enough for overlap.
                        second_sfp_pos = token_positions[idx2]
                        second_sfp_text = tokens[idx2]
                        if second_sfp_pos not in existing_positions:
                            errors.append(
                                SyllableError(
                                    text=second_sfp_text,
                                    position=second_sfp_pos,
                                    suggestions=["\u104b"],  # ။
                                    confidence=TEXT_DETECTOR_CONFIDENCES["missing_conjunction"],
                                    error_type=ET_MISSING_CONJUNCTION,
                                )
                            )
                    break  # only report first occurrence

        # G04: Verb-fronted object/complement phrase (word-order inversion).
        # Canonical Myanmar order keeps the finite verb after its object phrase.
        max_tail_tokens = self._WORD_ORDER_MAX_TAIL_TOKENS
        for i in range(1, len(tokens) - 2):
            verb_token = tokens[i]
            verb_norm = normalize(verb_token).rstrip(_BOUNDARY_PUNCT_CHARS)
            if not verb_norm:
                continue
            if not any(
                len(verb_norm) > len(ending) + 1 and verb_norm.endswith(ending)
                for ending in _WORD_ORDER_VERB_ENDINGS
            ):
                continue

            prev_norm = normalize(tokens[i - 1]).rstrip(_BOUNDARY_PUNCT_CHARS)
            if not prev_norm.endswith(normalize("က")):
                continue

            tail_indices: list[int] = []
            has_case_marker = False
            hit_boundary = False
            for j in range(i + 1, min(len(tokens), i + 1 + max_tail_tokens)):
                tok_norm = normalize(tokens[j]).rstrip(_BOUNDARY_PUNCT_CHARS)
                if not tok_norm:
                    continue
                if tok_norm in _WORD_ORDER_BOUNDARIES:
                    hit_boundary = True
                    break
                tail_indices.append(j)
                if any(
                    len(tok_norm) > len(sfx) and tok_norm.endswith(sfx)
                    for sfx in _WORD_ORDER_CASE_SUFFIXES
                ):
                    has_case_marker = True

            if not tail_indices or not has_case_marker:
                continue
            # Quotative structure: when a boundary marker (e.g., ဟု) follows
            # with only 1 tail token, the token is likely part of the quoted
            # clause — not a genuine word-order inversion. With 2+ tail tokens,
            # the verb's arguments are more likely displaced (real inversion).
            if hit_boundary and len(tail_indices) <= 1:
                continue

            err_pos = token_positions[i]
            if err_pos < 0 or err_pos in existing_positions:
                continue

            suggestion_tokens = [
                tokens[idx].rstrip(_BOUNDARY_PUNCT_CHARS)
                for idx in tail_indices
                if tokens[idx].rstrip(_BOUNDARY_PUNCT_CHARS)
            ]
            suggestion = " ".join(suggestion_tokens + [verb_norm])
            if not suggestion:
                continue

            error_text = verb_norm
            first_idx = tail_indices[0]
            first_token = tokens[first_idx].rstrip(_BOUNDARY_PUNCT_CHARS)
            span_end = token_positions[first_idx] + len(first_token) if first_token else err_pos

            # When the first two tail tokens are also a likely broken compound
            # (e.g., လူနာတင် + ယာဉ်ကို), keep the span on the verb token so
            # this word-order error does not consume the companion boundary error.
            keep_verb_only = False
            if len(tail_indices) >= 2 and first_token:
                second_token = tokens[tail_indices[1]].rstrip(_BOUNDARY_PUNCT_CHARS)
                second_base = second_token
                for suffix in sorted(_WORD_ORDER_CASE_SUFFIXES, key=len, reverse=True):
                    if len(second_base) > len(suffix) + 1 and second_base.endswith(suffix):
                        second_base = second_base[: -len(suffix)]
                        break
                provider = getattr(self, "provider", None)
                if provider is not None and second_base:
                    try:
                        keep_verb_only = bool(provider.is_valid_word(first_token + second_base))
                    except Exception:
                        keep_verb_only = False

            # When the first two tail tokens form a valid compound word
            # (e.g., လူနာတင်+ယာဉ် = ambulance), the "case marker" is on the
            # compound — not a displaced argument of the verb.  Skip the error.
            if keep_verb_only:
                continue

            if span_end > err_pos:
                error_text = text[err_pos:span_end]

            errors.append(
                SyllableError(
                    text=error_text,
                    position=err_pos,
                    suggestions=[suggestion],
                    confidence=TEXT_DETECTOR_CONFIDENCES["missing_conjunction"],
                    error_type=ET_POS_SEQUENCE_ERROR,
                )
            )
            break
