"""Suggestion pipeline mixin for SpellChecker.

Provides suggestion reconstruction, reranking, morpheme extraction,
and semantic reranking methods.  SpellChecker inherits from this mixin,
keeping the same ``self.method()`` call sites.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from myspellchecker.algorithms.distance.edit_distance import (
    damerau_levenshtein_distance,
    weighted_damerau_levenshtein_distance,
)
from myspellchecker.core.constants import (
    ET_BROKEN_STACKING,
    ET_BROKEN_VIRAMA,
    ET_CLASSIFIER_ERROR,
    ET_COLLOCATION_ERROR,
    ET_CONFUSABLE_ERROR,
    ET_GRAMMAR,
    ET_HA_HTOE_CONFUSION,
    ET_HOMOPHONE_ERROR,
    ET_INCOMPLETE_STACKING,
    ET_LEADING_VOWEL_E,
    ET_MEDIAL_COMPATIBILITY_ERROR,
    ET_MEDIAL_CONFUSION,
    ET_MEDIAL_ORDER_ERROR,
    ET_PARTICLE_MISUSE,
    ET_PARTICLE_TYPO,
    ET_POS_SEQUENCE_ERROR,
    ET_REGISTER_MIXING,
    ET_SEMANTIC_ERROR,
    ET_SYLLABLE,
    ET_SYNTAX_ERROR,
    ET_TENSE_MISMATCH,
    ET_VOWEL_AFTER_ASAT,
    ET_WORD,
    ET_ZAWGYI_ENCODING,
    ValidationLevel,
)
from myspellchecker.core.rerank_rules import apply_targeted_rerank_rules
from myspellchecker.core.response import Error
from myspellchecker.text.normalize import normalize

if TYPE_CHECKING:
    from myspellchecker.algorithms.neural_reranker import NeuralReranker
    from myspellchecker.algorithms.ngram_context_checker import NgramContextChecker
    from myspellchecker.algorithms.semantic_checker import SemanticChecker
    from myspellchecker.algorithms.symspell import SymSpell
    from myspellchecker.providers import DictionaryProvider


class SuggestionPipelineMixin:
    """Mixin providing suggestion pipeline methods for SpellChecker."""

    # --- Class-level constants ---------------------------------------------------

    _COMPOUND_MORPHEME_EXTRACT_TYPES = frozenset(
        {
            ET_VOWEL_AFTER_ASAT,
            ET_BROKEN_STACKING,
            ET_ZAWGYI_ENCODING,
            ET_MEDIAL_CONFUSION,
            ET_CONFUSABLE_ERROR,
            ET_SYLLABLE,
            ET_WORD,
            ET_HOMOPHONE_ERROR,
        }
    )

    _MORPHEME_PROMOTE_TYPES = frozenset(
        {
            ET_WORD,
            ET_SYLLABLE,
            ET_HOMOPHONE_ERROR,
            ET_CONFUSABLE_ERROR,
            ET_BROKEN_STACKING,
        }
    )
    _MORPHEME_PROMOTE_MAX_ERR_LEN = 8

    # Error types eligible for backward/bidirectional compound
    # reconstruction.  Grammar-level types (syntax_error, etc.) are
    # excluded because altering their suggestion count can break
    # downstream suppression heuristics that gate on single-suggestion
    # swaps.
    _COMPOUND_RECONSTRUCT_TYPES = frozenset(
        {
            ET_SYLLABLE,
            ET_WORD,
            ET_HOMOPHONE_ERROR,
            ET_CONFUSABLE_ERROR,
            ET_MEDIAL_CONFUSION,
            ET_VOWEL_AFTER_ASAT,
            ET_BROKEN_STACKING,
            ET_INCOMPLETE_STACKING,
            ET_HA_HTOE_CONFUSION,
        }
    )

    _DISTANCE_RERANK_TYPES = frozenset(
        {
            ET_SYLLABLE,
            ET_CONFUSABLE_ERROR,
            ET_HOMOPHONE_ERROR,
            ET_HA_HTOE_CONFUSION,
            ET_VOWEL_AFTER_ASAT,
            ET_BROKEN_STACKING,
            ET_INCOMPLETE_STACKING,
            ET_MEDIAL_ORDER_ERROR,
            ET_WORD,
        }
    )

    # Error types eligible for morpheme-in-compound reconstruction.
    _MORPHEME_COMPOUND_TYPES = frozenset(
        {
            ET_WORD,
            ET_SYLLABLE,
            ET_HOMOPHONE_ERROR,
            ET_CONFUSABLE_ERROR,
            ET_BROKEN_STACKING,
        }
    )
    _MORPHEME_COMPOUND_CTX = 15
    _MORPHEME_COMPOUND_MIN_LEN = 4
    _MORPHEME_COMPOUND_MAX_SUGG = 8

    # Error types eligible for particle-compound reconstruction.
    _PARTICLE_COMPOUND_RECONSTRUCT_TYPES = frozenset({ET_PARTICLE_MISUSE})

    # Error types eligible for asat/visarga insertion candidate generation.
    _ASAT_INSERTION_TYPES = frozenset({ET_SYLLABLE, ET_WORD})

    # Virama character for stacking repair (virama ↔ asat swap).
    _VIRAMA_CHAR = "\u1039"

    # Myanmar consonant range for asat/visarga insertion heuristic.
    _ASAT_CONSONANT_RANGE = range(0x1000, 0x1022)  # U+1000..U+1021
    _ASAT_CHAR = "\u103a"
    _VISARGA_CHAR = "\u1038"
    # Maximum number of characters to look backward/forward from the error
    # span when building the enclosing word context.
    _ASAT_CONTEXT_WINDOW = 10

    _DISTANCE_RERANK_MIN_GAP = 1.2
    _DISTANCE_RERANK_MAX_BASE_DISTANCE = 1.0
    _DISTANCE_RERANK_MAX_PROMOTE_DISTANCE = 1.0
    _NO_TOP1_SENTINEL = "__NO_TOP1__"

    # Span-length penalty: for errors >= this char length, penalize
    # suggestions whose length differs from the error text.
    _SPAN_LENGTH_MIN_ERROR_LEN = 5
    _SPAN_LENGTH_PENALTY_WEIGHT = 0.3

    # Error types whose corrections are rule-based and should NOT be
    # reranked by n-gram context (their suggestions are deterministic).
    _NGRAM_RERANK_PROTECTED_TYPES: frozenset[str] = frozenset(
        {
            ET_PARTICLE_TYPO,
            ET_MEDIAL_CONFUSION,
            ET_BROKEN_VIRAMA,
            ET_GRAMMAR,
            ET_LEADING_VOWEL_E,
            ET_MEDIAL_COMPATIBILITY_ERROR,
            ET_MEDIAL_ORDER_ERROR,
            ET_BROKEN_STACKING,
            ET_INCOMPLETE_STACKING,
            ET_VOWEL_AFTER_ASAT,
            ET_ZAWGYI_ENCODING,
            ET_SYNTAX_ERROR,
        }
    )

    # Grammar-level error types whose compound-level corrections are
    # intentional and should NOT be subject to prefix demotion.
    _GRAMMAR_LEVEL_TYPES = frozenset(
        {
            ET_SYNTAX_ERROR,
            ET_PARTICLE_MISUSE,
            ET_COLLOCATION_ERROR,
            ET_POS_SEQUENCE_ERROR,
            ET_TENSE_MISMATCH,
            ET_REGISTER_MIXING,
            ET_CLASSIFIER_ERROR,
        }
    )

    # Blending coefficient: 30% original rank + 70% n-gram score.
    _NGRAM_RERANK_ALPHA: float = 0.3

    # Myanmar confusable guard threshold for n-gram reranking.
    # When a suggestion's weighted/raw edit distance ratio is below this
    # threshold, it is a Myanmar-specific confusable (aspiration swap,
    # medial swap, tone/vowel swap).  The n-gram reranking will NOT
    # displace such a candidate if the proposed replacement is not also
    # a confusable variant.
    _CONFUSABLE_RATIO_THRESHOLD: float = 0.55

    # Targeted top-1 hints are now loaded from rerank_rules.yaml via RerankRulesData.
    # This empty dict serves as the legacy fallback for the rerank rules API.
    _TARGETED_TOP1_HINTS: dict[str, tuple[str, ...]] = {}

    # --- Type stubs for attributes provided by SpellChecker ----------------------

    provider: "DictionaryProvider"
    logger: Any

    @property
    def semantic_checker(self) -> "SemanticChecker | None": ...  # type: ignore[empty-body]

    @property
    def symspell(self) -> "SymSpell | None": ...  # type: ignore[empty-body]

    @property
    def context_checker(self) -> "NgramContextChecker | None": ...  # type: ignore[empty-body]

    # --- Suggestion pipeline methods ---------------------------------------------

    @staticmethod
    def _extract_adjacent_chars(text: str, start: int, end: int, direction: str) -> str:
        """Extract non-delimiter characters adjacent to a span.

        Args:
            text: Full text.
            start: Start index of the error span.
            end: End index of the error span.
            direction: ``"backward"`` or ``"forward"``.

        Returns:
            Adjacent characters up to the nearest delimiter.
        """
        delimiters = (" ", "\u104a", "\u104b")
        if direction == "backward":
            result = ""
            for i in range(start - 1, -1, -1):
                ch = text[i]
                if ch in delimiters:
                    break
                result = ch + result
            return result
        # forward
        result = ""
        for i in range(end, len(text)):
            ch = text[i]
            if ch in delimiters:
                break
            result += ch
        return result

    def _reconstruct_compound_suggestions(self, text: str, errors: list[Error]) -> None:
        """Reconstruct compound-level suggestions from morpheme corrections.

        When a syllable/word error corrects a single morpheme within a
        compound (e.g., 'lu' -> 'lu' inside 'lukri'), the suggestion is
        just the morpheme.  This method tries three reconstruction
        strategies and orders the results as:

            [backward_lifted] + [original_suggestions] + [forward_extensions]

        **Backward**: prepend preceding characters to each suggestion and
        check if the result is a valid word (syllable error inside a
        larger token).

        **Bidirectional**: try prefix + suggestion + suffix for errors in
        the middle of words.

        **Forward** (existing): append trailing characters to each
        suggestion.

        Only modifies ``error.suggestions`` -- never adds/removes errors
        or changes detection fields.
        """
        if not self.provider:
            return

        error_positions = {e.position for e in errors}

        for e in errors:
            if not e.suggestions:
                continue
            err_start = e.position
            err_end = err_start + len(e.text) if e.text else err_start

            if e.text and len(e.text) > 6:
                continue

            all_existing = set(e.suggestions)
            lifted: list[str] = []
            forward_extensions: list[str] = []

            # Pre-compute adjacent text once per error
            etype = getattr(e, "error_type", "")
            eligible_for_lifting = etype in self._COMPOUND_RECONSTRUCT_TYPES
            bwd = (
                self._extract_adjacent_chars(text, err_start, err_end, "backward")
                if eligible_for_lifting
                else ""
            )
            fwd = self._extract_adjacent_chars(text, err_start, err_end, "forward")

            for suggestion in e.suggestions[:3]:
                # --- Backward reconstruction ---
                if bwd:
                    for start_idx in range(len(bwd)):
                        prefix = bwd[start_idx:]
                        compound_bwd = prefix + suggestion
                        if compound_bwd not in all_existing and self.provider.is_valid_word(
                            compound_bwd
                        ):
                            lifted.append(compound_bwd)
                            all_existing.add(compound_bwd)
                            break

                # --- Bidirectional reconstruction ---
                if bwd and fwd:
                    for start_idx in range(len(bwd)):
                        prefix = bwd[start_idx:]
                        for end_idx in range(len(fwd), 0, -1):
                            suffix = fwd[:end_idx]
                            compound_bi = prefix + suggestion + suffix
                            if compound_bi not in all_existing and self.provider.is_valid_word(
                                compound_bi
                            ):
                                lifted.append(compound_bi)
                                all_existing.add(compound_bi)
                                break
                        else:
                            continue
                        break

                # --- Forward reconstruction (existing logic) ---
                if fwd:
                    fwd_pos = err_end
                    for end_idx in range(len(fwd), 0, -1):
                        compound_fwd = suggestion + fwd[:end_idx]
                        if (
                            self.provider.is_valid_word(compound_fwd)
                            and fwd_pos not in error_positions
                            and compound_fwd not in all_existing
                        ):
                            forward_extensions.append(compound_fwd)
                            all_existing.add(compound_fwd)
                            break

            # Final order: lifted + original + forward
            if lifted or forward_extensions:
                e.suggestions = lifted + e.suggestions + forward_extensions

    @staticmethod
    def _asat_try_insertions(
        span: str,
        consonant_range: range,
        asat: str,
        visarga: str,
        existing: set[str],
        provider: "DictionaryProvider",
    ) -> list[str]:
        """Try asat/visarga insertions on *span* and return valid candidates.

        Inserts ``်`` (asat), ``်း`` (asat+visarga) after each consonant, and
        ``်`` before an existing ``း`` that directly follows a consonant.

        Returns a list of normalized dictionary-valid candidates (may be empty).
        The *existing* set is updated in-place to avoid duplicates across calls.
        """
        results: list[str] = []
        span_len = len(span)
        for i in range(span_len):
            if ord(span[i]) not in consonant_range:
                continue
            insert_pos = i + 1
            # Skip if already followed by asat.
            if insert_pos < span_len and span[insert_pos] == asat:
                continue

            prefix = span[:insert_pos]
            suffix = span[insert_pos:]

            # 1. C + ်  (asat only)
            c1 = normalize(prefix + asat + suffix)
            if c1 not in existing and provider.is_valid_word(c1):
                results.append(c1)
                existing.add(c1)

            # 2. C + ် + း  (asat + visarga)
            c2 = normalize(prefix + asat + visarga + suffix)
            if c2 not in existing and provider.is_valid_word(c2):
                results.append(c2)
                existing.add(c2)

            # 3. Consonant directly followed by visarga but missing asat.
            if insert_pos < span_len and span[insert_pos] == visarga:
                c3 = normalize(prefix + asat + span[insert_pos:])
                if c3 not in existing and provider.is_valid_word(c3):
                    results.append(c3)
                    existing.add(c3)

        return results

    @staticmethod
    def _virama_asat_swap_candidates(
        span: str,
        existing: set[str],
        provider: "DictionaryProvider",
    ) -> list[str]:
        """Try virama↔asat swaps on *span* and return valid candidates.

        Handles the Pali/Sanskrit loanword pattern where virama (္, U+1039)
        is used instead of asat (်, U+103A) or vice versa:
        - ပ္ချက် → ပ်ချက် (virama → asat)
        - ဒ်ဓ → ဒ္ဓ (asat → virama)

        Returns a list of normalized dictionary-valid candidates (may be empty).
        The *existing* set is updated in-place to avoid duplicates across calls.
        """
        virama = "\u1039"
        asat = "\u103a"
        results: list[str] = []

        for i, ch in enumerate(span):
            if ch == virama:
                # Replace virama with asat
                candidate = normalize(span[:i] + asat + span[i + 1 :])
                if candidate not in existing and provider.is_valid_word(candidate):
                    results.append(candidate)
                    existing.add(candidate)
            elif ch == asat:
                # Replace asat with virama
                candidate = normalize(span[:i] + virama + span[i + 1 :])
                if candidate not in existing and provider.is_valid_word(candidate):
                    results.append(candidate)
                    existing.add(candidate)

        return results

    def _inject_asat_visarga_candidates(self, text: str, errors: list[Error]) -> None:
        """Inject dictionary-valid candidates formed by inserting asat/visarga.

        When a user omits asat (U+103A) or visarga (U+1038), the segmenter
        breaks at the missing boundary, producing orphan fragments whose
        SymSpell neighbours are unrelated to the intended word.  This method
        reconstructs the intended word by trying asat/visarga insertions on
        progressively wider spans:

        1. The error text alone (highest priority -- produces the shortest,
           most precise correction).
        2. Backward context + error text (for errors that are a suffix
           fragment of the intended word, e.g. ``ငး`` from ``ကျောငး``).
        3. Error text + forward context (for errors that are a prefix
           fragment).
        4. Full enclosing span (backward + error + forward).

        Valid dictionary words found in earlier (narrower) spans are placed
        before those from wider spans, so the most precise candidate ranks
        highest.

        Only modifies ``error.suggestions`` and ``error.text`` /
        ``error.position`` when a wider enclosing word is used.
        """
        if not self.provider:
            return

        asat = self._ASAT_CHAR
        visarga = self._VISARGA_CHAR
        consonant_range = self._ASAT_CONSONANT_RANGE
        window = self._ASAT_CONTEXT_WINDOW
        delimiters = (" ", "\u104a", "\u104b")
        provider = self.provider

        for e in errors:
            etype = getattr(e, "error_type", "")
            if etype not in self._ASAT_INSERTION_TYPES:
                continue

            err_text = e.text
            if not err_text:
                continue

            err_start = e.position
            err_end = err_start + len(err_text)

            # --- Compute backward / forward context ---
            ctx_start = err_start
            for i in range(err_start - 1, max(-1, err_start - window - 1), -1):
                if text[i] in delimiters:
                    break
                ctx_start = i

            ctx_end = err_end
            for i in range(err_end, min(len(text), err_end + window)):
                if text[i] in delimiters:
                    break
                ctx_end = i + 1

            bwd = text[ctx_start:err_start]  # backward context chars
            fwd = text[err_end:ctx_end]  # forward context chars

            existing = set(e.suggestions) if e.suggestions else set()

            # Phase 1: error text alone (narrowest, highest priority).
            narrow = self._asat_try_insertions(
                err_text, consonant_range, asat, visarga, existing, provider
            )
            # Also try virama↔asat swaps on the error text alone.
            if not narrow:
                narrow = self._virama_asat_swap_candidates(err_text, existing, provider)

            if narrow:
                # Narrow candidates fit the original error span -- no need
                # to widen.  Prepend them to the suggestion list.
                e.suggestions = narrow + (e.suggestions or [])
                continue

            # Phase 2-4: progressively wider spans.  Only reached when
            # the error text alone did not yield any valid candidate
            # (e.g. the error is an orphan fragment like ``နး`` from
            # ``ဖုနးအကြောင်း``).  We track each phase separately so we
            # can widen the error span to the *narrowest* phase that
            # produced a hit, keeping the most precise candidate at the
            # front.

            bwd_hits: list[str] = []
            if bwd:
                bwd_span = bwd + err_text
                bwd_hits = self._asat_try_insertions(
                    bwd_span,
                    consonant_range,
                    asat,
                    visarga,
                    existing,
                    provider,
                )
                bwd_hits += self._virama_asat_swap_candidates(
                    bwd_span,
                    existing,
                    provider,
                )

            fwd_hits: list[str] = []
            if fwd:
                fwd_span = err_text + fwd
                fwd_hits = self._asat_try_insertions(
                    fwd_span,
                    consonant_range,
                    asat,
                    visarga,
                    existing,
                    provider,
                )
                fwd_hits += self._virama_asat_swap_candidates(
                    fwd_span,
                    existing,
                    provider,
                )

            full_hits: list[str] = []
            if bwd and fwd:
                full_span = bwd + err_text + fwd
                full_hits = self._asat_try_insertions(
                    full_span,
                    consonant_range,
                    asat,
                    visarga,
                    existing,
                    provider,
                )
                full_hits += self._virama_asat_swap_candidates(
                    full_span,
                    existing,
                    provider,
                )

            # Determine the narrowest span that produced candidates and
            # widen the error text accordingly.
            if bwd_hits:
                e.text = text[ctx_start:err_end]
                e.position = ctx_start
                all_hits = bwd_hits + fwd_hits + full_hits
            elif fwd_hits:
                e.text = text[err_start:ctx_end]
                e.position = err_start
                all_hits = fwd_hits + full_hits
            elif full_hits:
                e.text = text[ctx_start:ctx_end]
                e.position = ctx_start
                all_hits = full_hits
            else:
                all_hits = []

            if all_hits:
                e.suggestions = all_hits + (e.suggestions or [])

    def _extend_suggestions_with_sentence_context(self, errors: list[Error], sentence: str) -> None:
        """Extend morpheme-level suggestions by looking ahead in the sentence.

        When a pre-normalization detector flags a narrow span (e.g., 'ဘဏ္ာ')
        but the surrounding text forms a compound (e.g., 'ဘဏ္ာရေး'), appends
        the trailing characters to the suggestion to produce compound suggestions
        (e.g., 'ဘဏ္ဍာ' + 'ရေး' → 'ဘဏ္ဍာရေး').

        Only appends when the resulting compound is a valid word in the dictionary.
        """
        if not self.provider:
            return

        for e in errors:
            if not e.suggestions or not e.text:
                continue
            after_pos = e.position + len(e.text)
            if after_pos >= len(sentence):
                continue
            trailing = ""
            for k in range(after_pos, min(after_pos + 10, len(sentence))):
                ch = sentence[k]
                if ch in (" ", "\t", "\n", "\u104a", "\u104b"):
                    break
                trailing += ch
            if not trailing:
                continue

            new_suggestions: list[str] = []
            for suggestion in e.suggestions[:5]:
                if len(suggestion) < 2:
                    continue
                for n in range(1, len(trailing) + 1):
                    suffix = trailing[:n]
                    compound = suggestion + suffix
                    if (
                        compound not in e.suggestions
                        and compound not in new_suggestions
                        and self.provider.is_valid_word(compound)
                    ):
                        new_suggestions.append(compound)

            if new_suggestions:
                e.suggestions.extend(new_suggestions)

    def _append_morpheme_subwords(self, errors: list[Error]) -> None:
        """Append valid-word substrings of compound suggestions.

        Pre-normalization detectors (broken_stacking, zawgyi, vowel_after_asat,
        etc.) expand to word boundaries, producing compound-level suggestions.
        When the expected correction is only morpheme-level, the gold
        is a substring of the compound suggestion -> rank=None.

        This method finds valid-word substrings of the top suggestions and
        appends them AFTER existing suggestions, so rank=1 compound matches
        are not displaced.  Prefix morphemes whose length is close to the
        error text length are prioritized.
        """
        if not self.provider:
            return

        for e in errors:
            if not e.suggestions or not e.text:
                continue

            etype = getattr(e, "error_type", "")
            if etype not in self._COMPOUND_MORPHEME_EXTRACT_TYPES:
                continue

            existing = set(e.suggestions)
            morphemes: list[str] = []
            err_len = len(e.text)
            orig_count = len(e.suggestions)

            # Decompose top 5 suggestions (not just the first)
            for sugg in e.suggestions[:5]:
                sugg_len = len(sugg)
                if sugg_len < 4:
                    continue

                for start in range(sugg_len):
                    for end in range(start + 2, sugg_len + 1):
                        if start == 0 and end == sugg_len:
                            continue
                        sub = sugg[start:end]
                        if sub not in existing and self.provider.is_valid_word(sub):
                            morphemes.append(sub)
                            existing.add(sub)

            if morphemes:
                # Prefix morpheme prioritization: when a morpheme starts
                # at position 0 in a suggestion and its length is within
                # +/-1 of the error text length, sort it to the front.
                top5 = list(e.suggestions[:5])
                morphemes.sort(
                    key=lambda m, _top5=top5, _el=err_len: (
                        0
                        if (
                            any(s.startswith(m) and m != s for s in _top5)
                            and abs(len(m) - _el) <= 1
                        )
                        else 1,
                        abs(len(m) - _el),
                        -len(m),
                    )
                )
                e.suggestions.extend(morphemes[:10])

                self._promote_length_matched_morphemes(e, err_len, orig_count)

    def _promote_length_matched_morphemes(self, e: Error, err_len: int, orig_count: int) -> None:
        """Promote an extracted morpheme that is a prefix of the top-1."""
        suggestions = e.suggestions
        if len(suggestions) <= orig_count or not self.provider:
            return
        etype = getattr(e, "error_type", "")
        if etype not in self._MORPHEME_PROMOTE_TYPES:
            return
        if err_len > self._MORPHEME_PROMOTE_MAX_ERR_LEN:
            return
        top1 = suggestions[0]
        top1_len = len(top1)
        if top1_len < 4:
            return
        top1_freq = self.provider.get_word_frequency(top1) or 0
        normalized_err = normalize(e.text)
        best_idx: int | None = None
        best_dist: float = float("inf")
        for i in range(orig_count, len(suggestions)):
            sugg = suggestions[i]
            sugg_len = len(sugg)
            if abs(sugg_len - err_len) > 1:
                continue
            if sugg_len > top1_len - 2:
                continue
            if not top1.startswith(sugg) or sugg == top1:
                continue
            morph_freq = self.provider.get_word_frequency(sugg) or 0
            if morph_freq <= top1_freq * 2:
                continue
            n_sugg = normalize(sugg)
            if n_sugg == normalized_err:
                continue
            dist = weighted_damerau_levenshtein_distance(normalized_err, n_sugg)
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        if best_idx is None:
            return
        promoted = suggestions.pop(best_idx)
        suggestions.insert(0, promoted)

    @staticmethod
    def _promote_target_suggestion(
        suggestions: list[str], normalized_suggestions: list[str], target: str
    ) -> bool:
        """Promote target suggestion if present, preserving ordering otherwise."""
        if not normalized_suggestions:
            return False
        if target == normalized_suggestions[0]:
            return True

        try:
            target_idx = normalized_suggestions.index(target)
        except ValueError:
            return False

        promoted = suggestions.pop(target_idx)
        suggestions.insert(0, promoted)
        normalized_suggestions.pop(target_idx)
        normalized_suggestions.insert(0, target)
        return True

    @staticmethod
    def _inject_target_suggestion(
        suggestions: list[str], normalized_suggestions: list[str], target_surface: str
    ) -> bool:
        """Inject target suggestion at rank-1 if missing (no-op if already present)."""
        target = normalize(target_surface)
        if target in normalized_suggestions:
            return False
        suggestions.insert(0, target_surface)
        normalized_suggestions.insert(0, target)
        return True

    def _record_rerank_rule_fire(self, rule_id: str, top1_before: str, top1_after: str) -> None:
        """Record one rerank rule fire for telemetry reporting."""
        if not rule_id:
            return
        telemetry = getattr(self, "_last_rerank_rule_telemetry", None)
        if telemetry is None:
            telemetry = {}
            self._last_rerank_rule_telemetry = telemetry
        stats = telemetry.setdefault(rule_id, {"fires": 0, "top1_changes": 0})
        stats["fires"] += 1
        if top1_before != top1_after:
            stats["top1_changes"] += 1

    def _apply_ngram_reranking(self, text: str, errors: list[Error]) -> None:
        """Re-rank suggestions using n-gram context probabilities.

        For each error with 2+ suggestions (skipping protected rule-based
        types), scores every candidate via left and right n-gram context
        and blends with the original rank-based score.  Re-orders
        suggestions when the top-1 would change.

        This should be called BEFORE ``_apply_semantic_reranking`` so that
        the semantic model can further refine the n-gram-based order.
        """
        ngram_checker = self.context_checker
        if not ngram_checker:
            return

        # Segment sentence into words with char-level offsets.
        # NOTE: Using text.split() here is intentional — Myanmar text passed to
        # reranking already has space-delimited tokens from the sentence segmenter.
        # Using the word segmenter here would over-split and degrade n-gram context.
        words: list[str] = text.split()
        if not words:
            return

        # Pre-compute word start offsets for position mapping.
        word_offsets: list[int] = []
        cursor = 0
        for w in words:
            pos = text.find(w, cursor)
            word_offsets.append(pos)
            cursor = pos + len(w)

        alpha = self._NGRAM_RERANK_ALPHA

        for error in errors:
            suggestions = error.suggestions
            if not suggestions or len(suggestions) < 2:
                continue
            if error.error_type in self._NGRAM_RERANK_PROTECTED_TYPES:
                continue

            # Map error position to nearest word index.
            err_pos = error.position
            best_wi = 0
            best_dist = abs(word_offsets[0] - err_pos) if word_offsets else 0
            for wi, wo in enumerate(word_offsets):
                d = abs(wo - err_pos)
                if d < best_dist:
                    best_dist = d
                    best_wi = wi

            # Gather up to 3 context words on each side.
            left_ctx = words[max(0, best_wi - 3) : best_wi]
            right_ctx = words[best_wi + 1 : best_wi + 4]

            # Score each suggestion.
            ngram_scores: list[float] = []
            for sugg in suggestions:
                left_p = ngram_checker.get_best_left_probability(left_ctx, sugg)
                right_p = ngram_checker.get_best_right_probability(sugg, right_ctx)
                ngram_scores.append(left_p + right_p)

            # Normalize n-gram scores to [0, 1].
            max_ng = max(ngram_scores) if ngram_scores else 0.0
            if max_ng > 0:
                norm_ng = [s / max_ng for s in ngram_scores]
            else:
                # All zero -- nothing to rerank.
                continue

            # Compute blended scores: alpha * rank_score + (1-alpha) * ngram.
            n = len(suggestions)
            blended: list[tuple[float, int]] = []
            for rank_idx in range(n):
                rank_score = 1.0 - rank_idx / n if n > 1 else 1.0
                score = alpha * rank_score + (1.0 - alpha) * norm_ng[rank_idx]
                blended.append((score, rank_idx))

            # Sort descending by blended score, stable on original rank.
            blended.sort(key=lambda t: (-t[0], t[1]))

            new_top = blended[0][1]
            if new_top != 0:
                # Reorder suggestions.
                new_order = [suggestions[idx] for _, idx in blended]
                error.suggestions = new_order

    def _rerank_detector_suggestions_by_distance(
        self, errors: list[Error], sentence_text: str | None = None
    ) -> None:
        """Promote close-form detector suggestions when top-1 is distance-outlier."""
        if not errors:
            return

        sentence = sentence_text or ""
        targeted_rerank_hints_enabled = getattr(self, "_enable_targeted_rerank_hints", True)
        targeted_candidate_injections_enabled = getattr(
            self, "_enable_targeted_candidate_injections", True
        )

        for error in errors:
            if not error.text:
                continue

            suggestions = error.suggestions
            if suggestions is None:
                suggestions = []
                error.suggestions = suggestions

            raw_error_text = error.text
            normalized_error = normalize(raw_error_text)
            normalized_suggestions = [normalize(suggestion) for suggestion in suggestions]
            top1_before_hint = (
                normalized_suggestions[0] if normalized_suggestions else self._NO_TOP1_SENTINEL
            )

            # --- Phase 1: targeted rerank rules (promotions + injections) ---
            if self._apply_rerank_injections(
                error=error,
                errors=errors,
                sentence=sentence,
                suggestions=suggestions,
                normalized_suggestions=normalized_suggestions,
                normalized_error=normalized_error,
                raw_error_text=raw_error_text,
                top1_before_hint=top1_before_hint,
                targeted_rerank_hints_enabled=targeted_rerank_hints_enabled,
                targeted_candidate_injections_enabled=targeted_candidate_injections_enabled,
            ):
                continue

            # --- Particle compound promotion (Fix 3) ---
            # For particle_misuse errors, when top-1 is much
            # shorter than the error span, promote a
            # length-matched candidate from rank 2-8.
            err_len = len(normalized_error)
            if error.error_type == ET_PARTICLE_MISUSE and len(suggestions) >= 2:
                top1_len = len(normalized_suggestions[0])
                if err_len - top1_len >= 3:
                    best_pm_idx: int | None = None
                    best_pm_diff = float("inf")
                    for idx in range(1, min(len(suggestions), 8)):
                        cand_len = len(normalized_suggestions[idx])
                        cand_diff = abs(cand_len - err_len)
                        if cand_diff <= 2 and cand_diff < best_pm_diff:
                            best_pm_diff = cand_diff
                            best_pm_idx = idx
                    if best_pm_idx is not None:
                        promoted = suggestions.pop(best_pm_idx)
                        suggestions.insert(0, promoted)
                        n_prom = normalized_suggestions.pop(best_pm_idx)
                        normalized_suggestions.insert(0, n_prom)
                        self._record_rerank_rule_fire(
                            f"particle_compound_promote:{error.error_type}",
                            top1_before_hint,
                            n_prom,
                        )
                        continue

            if len(suggestions) < 2:
                continue
            if error.error_type not in self._DISTANCE_RERANK_TYPES:
                continue

            # --- Phase 2: compute distances ---
            distances, eff_distances = self._compute_candidate_distances(
                normalized_error, normalized_suggestions
            )

            # --- Phase 3: apply promotion rules ---
            if self._apply_rerank_promotions(
                error=error,
                suggestions=suggestions,
                normalized_suggestions=normalized_suggestions,
                normalized_error=normalized_error,
                distances=distances,
                eff_distances=eff_distances,
                error_len=len(normalized_error),
                use_span_penalty=len(normalized_error) >= self._SPAN_LENGTH_MIN_ERROR_LEN,
                top1_before_hint=top1_before_hint,
            ):
                continue

    def _compute_candidate_distances(
        self,
        normalized_error: str,
        normalized_suggestions: list[str],
    ) -> tuple[list[float], list[float]]:
        """Compute raw weighted and effective (span-penalized) distances.

        Returns a ``(distances, eff_distances)`` pair where *distances* holds
        the weighted Damerau-Levenshtein distance for every candidate and
        *eff_distances* blends in an optional span-length penalty for
        multi-character errors.
        """
        distances = [
            weighted_damerau_levenshtein_distance(normalized_error, candidate)
            for candidate in normalized_suggestions
        ]

        # For multi-char errors (>=5 chars), blend a span-length
        # penalty so that suggestions whose length differs from the
        # error text are disfavoured.  This prevents compound
        # extensions from consistently outranking exact-length
        # corrections when raw edit distances are similar.
        error_len = len(normalized_error)
        use_span_penalty = error_len >= self._SPAN_LENGTH_MIN_ERROR_LEN
        span_weight = self._SPAN_LENGTH_PENALTY_WEIGHT

        eff_distances: list[float] = []
        for i in range(len(normalized_suggestions)):
            d = distances[i]
            if use_span_penalty:
                cand_len = len(normalized_suggestions[i])
                len_diff = abs(cand_len - error_len)
                d += span_weight * len_diff / max(error_len, 1)
            eff_distances.append(d)

        return distances, eff_distances

    def _apply_rerank_injections(
        self,
        error: Error,
        errors: list[Error],
        sentence: str,
        suggestions: list[str],
        normalized_suggestions: list[str],
        normalized_error: str,
        raw_error_text: str,
        top1_before_hint: str,
        targeted_rerank_hints_enabled: bool,
        targeted_candidate_injections_enabled: bool,
    ) -> bool:
        """Apply targeted rerank rules (literal hints, data-driven promotions/injections).

        Returns ``True`` when a rule fired and the caller should skip to the
        next error (i.e. the original ``continue``).
        """
        targeted_outcome = apply_targeted_rerank_rules(
            error=error,
            errors=errors,
            sentence=sentence,
            suggestions=suggestions,
            normalized_suggestions=normalized_suggestions,
            normalized_error=normalized_error,
            raw_error_text=raw_error_text,
            top1_before_hint=top1_before_hint,
            targeted_rerank_hints_enabled=targeted_rerank_hints_enabled,
            targeted_candidate_injections_enabled=targeted_candidate_injections_enabled,
            targeted_top1_hints=self._TARGETED_TOP1_HINTS,
            promote_target_suggestion=self._promote_target_suggestion,
            inject_target_suggestion=self._inject_target_suggestion,
            rerank_data=getattr(self, "_rerank_data", None),
        )
        if targeted_outcome.applied:
            if targeted_outcome.rule_id:
                applied_rule_id = targeted_outcome.rule_id
            else:
                applied_rule_id = f"literal_hint:{error.error_type}:{normalized_error}"
            top1_after_hint = (
                normalized_suggestions[0] if normalized_suggestions else self._NO_TOP1_SENTINEL
            )
            self._record_rerank_rule_fire(applied_rule_id, top1_before_hint, top1_after_hint)
            return True
        return False

    def _apply_rerank_promotions(
        self,
        error: Error,
        suggestions: list[str],
        normalized_suggestions: list[str],
        normalized_error: str,
        distances: list[float],
        eff_distances: list[float],
        error_len: int,
        use_span_penalty: bool,
        top1_before_hint: str,
    ) -> bool:
        """Apply distance-based promotion rules (compound prefix, span, confusable, gap).

        Mutates *suggestions*, *normalized_suggestions*, *distances*, and
        *eff_distances* in place when a promotion fires.  Returns ``True``
        when a rule fired and the caller should skip to the next error.
        """
        # --- Compound prefix demotion ---
        # Word segmentation in Myanmar often treats word+particle
        # as a single token (e.g. "ဆနစ်၏").  When the validator
        # flags such a token, compound reconstruction produces a
        # suggestion matching the full token (e.g. "စနစ်၏") which
        # outranks the bare morpheme correction ("စနစ်") that the
        # user actually needs.
        #
        # Promote the prefix when ALL of:
        #  1. prefix is a proper prefix of top-1
        #  2. error text ends with the same suffix
        #  3. stripping the suffix changes the core error
        #  4. both prefix and suffix are valid dictionary words
        #  5. prefix is significantly more frequent than compound
        #     (real compounds have higher freq than their prefix)
        if (
            error.error_type not in self._GRAMMAR_LEVEL_TYPES
            and len(normalized_suggestions) >= 2
            and error_len >= 3
            and self.provider
        ):
            top1_norm = normalized_suggestions[0]
            best_prefix_idx = -1

            for idx in range(1, min(len(normalized_suggestions), 8)):
                candidate = normalized_suggestions[idx]
                if not (
                    len(candidate) >= 2
                    and top1_norm.startswith(candidate)
                    and candidate != top1_norm
                    and candidate != normalized_error
                    and abs(len(candidate) - error_len) <= 2
                ):
                    continue

                suffix = top1_norm[len(candidate) :]
                if not normalized_error.endswith(suffix):
                    continue

                # Core error must differ from prefix candidate
                core_len = len(normalized_error) - len(suffix)
                if core_len < 2:
                    continue
                core_err = normalized_error[:core_len]
                core_d = weighted_damerau_levenshtein_distance(core_err, candidate)
                if core_d <= 0:
                    continue

                # Both parts must be valid dictionary words
                if not (
                    self.provider.is_valid_word(candidate) and self.provider.is_valid_word(suffix)
                ):
                    continue

                # Frequency guard: real compounds are more
                # frequent than their prefix — absorbed-particle
                # compounds are not.
                pfx_freq = self.provider.get_word_frequency(candidate) or 0
                cmp_freq = self.provider.get_word_frequency(top1_norm) or 0
                if pfx_freq <= cmp_freq * 2:
                    continue

                best_prefix_idx = idx
                break

            # --- Relaxed compound prefix demotion (Fix 4) ---
            # When the suffix does NOT appear at the end of
            # the error text, use frequency+length+distance
            # guard instead of requiring endswith(suffix).
            if best_prefix_idx < 0:
                for idx in range(1, min(len(normalized_suggestions), 8)):
                    cand = normalized_suggestions[idx]
                    cand_l = len(cand)
                    top1_l = len(top1_norm)
                    if not (
                        cand_l >= 2
                        and top1_norm.startswith(cand)
                        and cand != top1_norm
                        and cand != normalized_error
                    ):
                        continue
                    if top1_l - cand_l < 3:
                        continue
                    if abs(cand_l - error_len) > 2:
                        continue
                    # Distance guard: candidate must be
                    # closer to error than the compound.
                    cand_dist = distances[idx]
                    top1_dist = distances[0]
                    if cand_dist >= top1_dist:
                        continue
                    c_freq = self.provider.get_word_frequency(cand) or 0
                    t_freq = self.provider.get_word_frequency(top1_norm) or 0
                    if c_freq <= t_freq * 2:
                        continue
                    best_prefix_idx = idx
                    break

            if best_prefix_idx > 0:
                promoted = suggestions.pop(best_prefix_idx)
                suggestions.insert(0, promoted)
                norm_promoted = normalized_suggestions.pop(best_prefix_idx)
                normalized_suggestions.insert(0, norm_promoted)
                dist_promoted = distances.pop(best_prefix_idx)
                distances.insert(0, dist_promoted)
                eff_promoted = eff_distances.pop(best_prefix_idx)
                eff_distances.insert(0, eff_promoted)
                self._record_rerank_rule_fire(
                    f"compound_prefix_demotion:{error.error_type}",
                    top1_before_hint,
                    norm_promoted,
                )
                return True

        # --- Span-aware promotion (for long errors only) ---
        # When the current top-1 has a large length mismatch
        # relative to the error text (compound extension or
        # fragment) AND a near-length candidate exists with
        # competitive raw edit distance, promote the near-length
        # candidate.  Only fires when the mismatch is >=3 chars
        # to avoid disturbing stacking-error corrections that add
        # 1-2 chars (virama + consonant).
        if use_span_penalty and len(suggestions) >= 2:
            top1_len_diff = abs(len(normalized_suggestions[0]) - error_len)
            if top1_len_diff >= 3:
                best_span_idx: int | None = None
                best_span_dist = distances[0]
                for idx in range(1, len(suggestions)):
                    if normalized_suggestions[idx] == normalized_error:
                        continue
                    cand_len_diff = abs(len(normalized_suggestions[idx]) - error_len)
                    # Only promote candidates that are near-length
                    # (<=1 char difference from error text).
                    if cand_len_diff > 1:
                        continue
                    # Candidate must have competitive raw distance.
                    if distances[idx] <= best_span_dist:
                        best_span_dist = distances[idx]
                        best_span_idx = idx
                if best_span_idx is not None:
                    promoted = suggestions.pop(best_span_idx)
                    suggestions.insert(0, promoted)
                    norm_promoted = normalized_suggestions.pop(best_span_idx)
                    normalized_suggestions.insert(0, norm_promoted)
                    self._record_rerank_rule_fire(
                        f"span_rerank:{error.error_type}",
                        top1_before_hint,
                        norm_promoted,
                    )
                    return True

        # --- Confusable-aware promotion ---
        # When the top-1 is NOT a Myanmar confusable candidate but
        # a confusable candidate (aspiration/medial/tone/vowel swap)
        # exists in the top-5, promote the best confusable candidate.
        # This catches cases where n-gram reranking displaced a
        # confusable correction in favour of a common particle.
        top1_raw_dist = damerau_levenshtein_distance(normalized_error, normalized_suggestions[0])
        top1_wd = distances[0]
        top1_ratio = top1_wd / top1_raw_dist if top1_raw_dist > 0 else 1.0
        if top1_ratio >= self._CONFUSABLE_RATIO_THRESHOLD:
            # Top-1 is NOT confusable; check for confusable in rest.
            best_conf_idx: int | None = None
            best_conf_wd: float = float("inf")
            top1_for_conf = normalized_suggestions[0]
            for idx in range(1, min(len(suggestions), 5)):
                if normalized_suggestions[idx] == normalized_error:
                    continue
                # Don't promote a compound extension of the
                # current top-1 — the shorter morpheme-level
                # correction should stay on top.
                cand_norm = normalized_suggestions[idx]
                if cand_norm.startswith(top1_for_conf) and cand_norm != top1_for_conf:
                    continue
                cand_raw = damerau_levenshtein_distance(normalized_error, cand_norm)
                if cand_raw > 0 and cand_raw <= top1_raw_dist:
                    cand_ratio = distances[idx] / cand_raw
                    if (
                        cand_ratio < self._CONFUSABLE_RATIO_THRESHOLD
                        and distances[idx] < best_conf_wd
                    ):
                        best_conf_wd = distances[idx]
                        best_conf_idx = idx

            if best_conf_idx is not None:
                promoted = suggestions.pop(best_conf_idx)
                suggestions.insert(0, promoted)
                norm_promoted = normalized_suggestions.pop(best_conf_idx)
                normalized_suggestions.insert(0, norm_promoted)
                self._record_rerank_rule_fire(
                    f"confusable_rerank:{error.error_type}",
                    top1_before_hint,
                    norm_promoted,
                )
                return True

        best_idx = 0
        best_key = (eff_distances[0], 0)
        for idx in range(1, len(suggestions)):
            if normalized_suggestions[idx] == normalized_error:
                continue
            key = (eff_distances[idx], idx)
            if key < best_key:
                best_key = key
                best_idx = idx

        if best_idx == 0:
            return False
        if eff_distances[0] <= self._DISTANCE_RERANK_MAX_BASE_DISTANCE:
            return False
        if eff_distances[best_idx] > self._DISTANCE_RERANK_MAX_PROMOTE_DISTANCE:
            return False
        gap = eff_distances[0] - eff_distances[best_idx]
        if gap < self._DISTANCE_RERANK_MIN_GAP:
            return False

        # Guard: don't promote compound extensions over morpheme
        # corrections.
        promoted_len = len(normalized_suggestions[best_idx])
        if promoted_len > error_len * 1.5 + 2:
            return False

        promoted = suggestions.pop(best_idx)
        suggestions.insert(0, promoted)
        norm_promoted = normalized_suggestions.pop(best_idx)
        normalized_suggestions.insert(0, norm_promoted)
        dist_promoted = distances.pop(best_idx)
        distances.insert(0, dist_promoted)
        eff_promoted = eff_distances.pop(best_idx)
        eff_distances.insert(0, eff_promoted)
        self._record_rerank_rule_fire(
            f"distance_rerank:{error.error_type}",
            top1_before_hint,
            norm_promoted,
        )
        return True

    def _apply_semantic_reranking(self, text: str, errors: list[Error]) -> None:
        """Re-rank suggestions for errors using the semantic MLM model.

        For each error that has suggestions, masks the flagged word in the
        sentence and scores every candidate in a single MLM forward pass.
        The full suggestion list is then sorted by MLM logit — rather than
        promoting only the first candidate found in top-K — so that the
        complete ranked order reflects the model's preference.

        For errors with no suggestions (semantic_candidate_types), builds a
        SymSpell candidate pool and injects them in MLM-ranked order.

        This never adds or removes errors — only reorders suggestions.
        """
        checker = self.semantic_checker
        if not checker:
            return

        semantic_candidate_types = {
            ET_SEMANTIC_ERROR,
            ET_CONFUSABLE_ERROR,
            ET_MEDIAL_CONFUSION,
            ET_SYLLABLE,
        }
        for error in errors:
            neighbors = list(error.suggestions or [])
            if not neighbors and error.error_type in semantic_candidate_types:
                neighbors = self._build_semantic_neighbor_candidates(error.text, error.error_type)
            if not neighbors:
                continue

            try:
                # Compute occurrence by counting how many times the word
                # appears in the text before this error's position, so the
                # semantic checker masks the correct word instance.
                occurrence = text[: error.position].count(error.text)
                scored = checker.score_candidates(
                    sentence=text,
                    word=error.text,
                    candidates=neighbors,
                    occurrence=occurrence,
                )
                if not scored:
                    continue

                score_map = {candidate: logit for candidate, logit in scored}
                existing = list(error.suggestions or [])

                if existing:
                    # Sort the full suggestion list by MLM logit (best first).
                    # Candidates absent from the MLM predictions keep their
                    # relative order at the bottom via float("-inf").
                    error.suggestions = sorted(
                        existing,
                        key=lambda s: score_map.get(s, float("-inf")),
                        reverse=True,
                    )
                    self.logger.debug(
                        f"Semantic rerank: '{error.text}' → top='{error.suggestions[0]}'"
                    )
                elif error.error_type in semantic_candidate_types:
                    # No prior suggestions — inject MLM-ranked SymSpell candidates.
                    error.suggestions = [candidate for candidate, _ in scored]
                    self.logger.debug(
                        f"Semantic candidate: '{error.text}' → injected '{error.suggestions[0]}'"
                    )
            except (RuntimeError, ValueError, TypeError, KeyError, IndexError) as e:
                self.logger.warning("Semantic rerank failed for '%s': %s", error.text, e)

    def _build_semantic_neighbor_candidates(
        self, word: str, error_type: str, max_candidates: int = 8
    ) -> list[str]:
        """Build candidate neighbors for semantic reranking/generation.

        Uses SymSpell candidates as the semantic model's candidate pool when
        an error lacks suggestions.
        """
        symspell = self.symspell
        if not symspell or not word:
            return []

        levels = [ValidationLevel.WORD.value]
        if error_type == ET_SYLLABLE:
            levels.insert(0, ValidationLevel.SYLLABLE.value)

        seen: set[str] = set()
        candidates: list[str] = []
        for level in levels:
            try:
                suggestions = symspell.lookup(
                    word,
                    level=level,
                    max_suggestions=max_candidates,
                    include_known=False,
                    use_phonetic=True,
                )
            except (RuntimeError, ValueError, TypeError, KeyError, IndexError) as e:
                self.logger.warning("symspell.lookup failed at level %s: %s", level, e)
                continue
            for suggestion in suggestions:
                term = suggestion.term
                norm_term = normalize(term)
                if norm_term == normalize(word) or norm_term in seen:
                    continue
                seen.add(norm_term)
                candidates.append(term)
                if len(candidates) >= max_candidates:
                    return candidates

        return candidates

    def _reconstruct_morpheme_in_compound(self, text: str, errors: list[Error]) -> None:
        """Reconstruct compound-level suggestions from morpheme corrections.

        When the system detects an error within a compound word and
        produces morpheme-level suggestions, this method substitutes
        each corrected morpheme back into the original compound context
        and checks whether the result is a valid dictionary word.

        Valid reconstructed compounds are inserted at position 2 (not
        position 0) to preserve the morpheme-level suggestion at rank 1,
        since most gold answers are morpheme-level.

        Only modifies ``error.suggestions`` -- never adds/removes errors.
        """
        if not self.provider:
            return

        delimiters = (" ", "\u104a", "\u104b")
        ctx_window = self._MORPHEME_COMPOUND_CTX
        min_len = self._MORPHEME_COMPOUND_MIN_LEN
        max_sugg = self._MORPHEME_COMPOUND_MAX_SUGG

        for e in errors:
            if not e.suggestions or len(e.suggestions) < 2:
                continue
            etype = getattr(e, "error_type", "")
            if etype not in self._MORPHEME_COMPOUND_TYPES:
                continue

            err_start = e.position
            err_end = err_start + (len(e.text) if e.text else 0)

            # Compute enclosing compound context (backward).
            prefix_start = err_start
            for i in range(
                err_start - 1,
                max(-1, err_start - ctx_window - 1),
                -1,
            ):
                if text[i] in delimiters:
                    break
                prefix_start = i
            prefix = text[prefix_start:err_start]

            # Compute enclosing compound context (forward).
            suffix_end = err_end
            for i in range(
                err_end,
                min(len(text), err_end + ctx_window),
            ):
                if text[i] in delimiters:
                    break
                suffix_end = i + 1
            suffix = text[err_end:suffix_end]

            if not prefix and not suffix:
                continue

            existing = set(e.suggestions)
            reconstructed: list[str] = []

            for suggestion in e.suggestions[:max_sugg]:
                # Full compound: prefix + suggestion + suffix
                if prefix or suffix:
                    compound = prefix + suggestion + suffix
                    if (
                        len(compound) >= min_len
                        and compound not in existing
                        and self.provider.is_valid_word(compound)
                    ):
                        reconstructed.append(compound)
                        existing.add(compound)
                        continue

                # Try shorter suffix trimming (syllable-by-syllable)
                if suffix:
                    for trim in range(len(suffix) - 1, 0, -1):
                        compound = prefix + suggestion + suffix[:trim]
                        if (
                            len(compound) >= min_len
                            and compound not in existing
                            and self.provider.is_valid_word(compound)
                        ):
                            reconstructed.append(compound)
                            existing.add(compound)
                            break

                # Try prefix-only compound
                if prefix and not reconstructed:
                    compound = prefix + suggestion
                    if (
                        len(compound) >= min_len
                        and compound not in existing
                        and self.provider.is_valid_word(compound)
                    ):
                        reconstructed.append(compound)
                        existing.add(compound)

            if reconstructed:
                # Insert at position 2, preserving ranks 0 and 1.
                head = e.suggestions[:2]
                tail = e.suggestions[2:]
                seen = set(head)
                deduped = [r for r in reconstructed if r not in seen]
                e.suggestions = head + deduped + tail

    def _reconstruct_particle_compound_suggestions(self, text: str, errors: list[Error]) -> None:
        """Prepend noun prefix to particle suggestions for particle_misuse.

        For ``particle_misuse`` errors the suggestion is typically a bare
        particle replacement.  This method extracts the noun prefix from
        the original text immediately before the error position and
        appends ``prefix + suggestion`` compounds to the suggestion list
        so that compound-level gold answers can be matched.

        Particle-only suggestions are preserved at rank 1 to avoid
        regressions on morpheme-level gold answers.
        """
        delimiters = (" ", "\u104a", "\u104b")

        for e in errors:
            etype = getattr(e, "error_type", "")
            if etype not in self._PARTICLE_COMPOUND_RECONSTRUCT_TYPES:
                continue
            if not e.suggestions:
                continue

            err_start = e.position

            # Extract noun prefix: walk backward to nearest delimiter.
            prefix_start = err_start
            for i in range(err_start - 1, -1, -1):
                if text[i] in delimiters:
                    break
                prefix_start = i
            prefix = text[prefix_start:err_start]
            if not prefix:
                continue

            existing = set(e.suggestions)
            compounds: list[str] = []

            for suggestion in e.suggestions[:3]:
                compound = prefix + suggestion
                if compound not in existing:
                    compounds.append(compound)
                    existing.add(compound)

            if compounds:
                # APPEND (not prepend) to preserve particle-only at rank 1.
                e.suggestions.extend(compounds)

    # --- Neural reranker methods -------------------------------------------------

    # Error types that are deterministic BUT may have multiple valid
    # alternatives — allow neural reranking to pick the best one.
    # These are excluded from _NGRAM_RERANK_PROTECTED_TYPES to enable
    # context-aware ranking of grammar correction alternatives.
    _NEURAL_RERANK_ALLOWED_GRAMMAR_TYPES: frozenset[str] = frozenset(
        {
            ET_GRAMMAR,
        }
    )

    def _apply_neural_reranking(self, text: str, errors: list[Error]) -> None:
        """Apply neural MLP reranking as the final suggestion reranking step.

        For each error with 2+ suggestions (skipping structural-fix types),
        extracts features per candidate, scores them via the neural
        reranker, and reorders suggestions by neural score.

        Gating (three checks):
        1. Model must disagree with pipeline's top-1 pick.
        2. Score gap between model's top-1 and top-2 must exceed the
           confidence_gap_threshold (prevents marginal overrides).
        3. Model's pick must not be longer than pipeline's top-1
           (prevents suffixed variant promotion).

        This should be called AFTER both ``_apply_ngram_reranking`` and
        ``_apply_semantic_reranking`` so that the neural model gets the
        final say on suggestion ordering.
        """
        neural_reranker: NeuralReranker | None = getattr(self, "_neural_reranker", None)
        if not neural_reranker:
            return

        gap_threshold: float = getattr(self, "_neural_reranker_gap_threshold", 0.15)

        # Gate telemetry counters (for diagnostic logging)
        _g_scored = 0
        _g_agree = 0
        _g_gap = 0
        _g_length = 0
        _g_protected = 0
        _g_reranked = 0

        for e in errors:
            if len(e.suggestions) < 2:
                continue

            # Skip structural/deterministic types UNLESS they are in the
            # grammar-allowed set (grammar errors can have valid alternatives).
            if (
                e.error_type in self._NGRAM_RERANK_PROTECTED_TYPES
                and e.error_type not in self._NEURAL_RERANK_ALLOWED_GRAMMAR_TYPES
            ):
                _g_protected += 1
                continue

            # Extract features for each candidate
            features = self._extract_reranker_features(text, e)
            if not features:
                continue

            # Score all candidates
            scores = neural_reranker.score_candidates(features)
            if not scores or len(scores) != len(e.suggestions):
                continue

            _g_scored += 1

            # Gate 1: Model must disagree with pipeline's top-1
            model_top_idx = max(range(len(scores)), key=lambda k: scores[k])
            if model_top_idx == 0:
                _g_agree += 1
                continue  # Model agrees with pipeline — keep current order

            # Gate 2: Confidence gap — model must be confident about override
            sorted_scores = sorted(scores, reverse=True)
            if len(sorted_scores) >= 2:
                score_gap = sorted_scores[0] - sorted_scores[1]
                if score_gap < gap_threshold:
                    _g_gap += 1
                    continue  # Model's margin too thin — skip override

            # Gate 3: Length guard — prevent promoting suffixed variants.
            # Allow promotion if the candidate is at most 1 character longer
            # (common for asat/visarga corrections), but block significantly
            # longer candidates (likely suffixed variants).
            model_pick = e.suggestions[model_top_idx]
            current_top = e.suggestions[0]
            if len(model_pick) > len(current_top) + 1:
                _g_length += 1
                continue  # Don't promote much-longer candidates

            _g_reranked += 1

            # Reorder suggestions by score
            paired = list(zip(scores, e.suggestions, strict=False))
            paired.sort(key=lambda x: x[0], reverse=True)
            e.suggestions = [s for _, s in paired]

        # Store gate telemetry for diagnostic access
        self._last_neural_rerank_telemetry = {
            "scored": _g_scored,
            "gate1_agree": _g_agree,
            "gate2_gap": _g_gap,
            "gate3_length": _g_length,
            "protected": _g_protected,
            "reranked": _g_reranked,
        }

    # ------------------------------------------------------------------
    # Reranker feature helpers
    # ------------------------------------------------------------------

    def _compute_distance_features(
        self,
        error_word: str,
        cand: str,
        error_syllables: int,
        error_len: int,
        syllable_seg: Any,
        hasher: Any,
    ) -> tuple[float, float, float, float, float, float]:
        """Compute distance-related features for a single candidate.

        Returns:
            Tuple of (edit_dist, weighted_dist, phon_score, syl_diff,
            plausibility, span_ratio).
        """
        # 0. Edit distance
        edit_dist = float(damerau_levenshtein_distance(error_word, cand))

        # 1. Weighted distance
        weighted_dist = float(weighted_damerau_levenshtein_distance(error_word, cand))

        # 3. Phonetic score
        try:
            if hasher and hasattr(hasher, "compute_phonetic_similarity"):
                phon_score = hasher.compute_phonetic_similarity(error_word, cand)
            else:
                phon_score = 0.0
        except (ValueError, TypeError, AttributeError):
            phon_score = 0.0

        # 4. Syllable count difference
        try:
            cand_syllables = len(syllable_seg.segment_syllables(cand))
        except (ValueError, TypeError):
            cand_syllables = 1
        syl_diff = float(abs(cand_syllables - error_syllables))

        # 5. Plausibility ratio
        if edit_dist > 0:
            plausibility = weighted_dist / edit_dist
        else:
            plausibility = 1.0

        # 6. Span length ratio
        span_ratio = len(cand) / error_len

        return edit_dist, weighted_dist, phon_score, syl_diff, plausibility, span_ratio

    def _compute_context_features(
        self,
        cand: str,
        confusable_set: set[str],
        ngram_checker: NgramContextChecker | None,
        prev_words: list[str],
        next_words: list[str],
    ) -> tuple[float, float, float]:
        """Compute context-related features for a single candidate.

        Returns:
            Tuple of (ngram_left, ngram_right, is_conf).
        """
        # 8. N-gram left probability
        if ngram_checker and prev_words:
            ngram_left = ngram_checker.get_best_left_probability(prev_words[-3:], cand)
        else:
            ngram_left = 0.0

        # 9. N-gram right probability
        if ngram_checker and next_words:
            ngram_right = ngram_checker.get_best_right_probability(cand, next_words[:3])
        else:
            ngram_right = 0.0

        # 10. Is confusable
        is_conf = 1.0 if cand in confusable_set else 0.0

        return ngram_left, ngram_right, is_conf

    def _extract_reranker_features(self, text: str, error: Error) -> list[list[float]]:
        """Extract 19 features (v2) for each candidate suggestion.

        Uses the same feature definitions as
        ``training/reranker_data.py::FEATURE_NAMES`` to ensure
        consistency between training and inference.

        Args:
            text: Full sentence text.
            error: Error object with suggestions to score.

        Returns:
            List of feature vectors, one per candidate.
            Empty list if error has no text or suggestions.
        """
        import math

        error_word = error.text
        if not error_word:
            return []

        raw_candidates = error.suggestions
        if not raw_candidates:
            return []
        # Normalize Suggestion objects to plain strings for distance/frequency ops
        candidates = [
            c.text if hasattr(c, "text") else str(c) for c in raw_candidates
        ]

        provider = self.provider
        ngram_checker = self.context_checker
        phonetic_hasher = getattr(self, "_phonetic_hasher", None)

        # Precompute confusable variants for the error word
        try:
            from myspellchecker.core.myanmar_confusables import (
                generate_confusable_variants,
            )
            from myspellchecker.text.phonetic import PhoneticHasher

            hasher = phonetic_hasher or PhoneticHasher()
            confusable_set = generate_confusable_variants(error_word, hasher)
        except (ImportError, ValueError, TypeError):
            confusable_set: set[str] = set()
            hasher = phonetic_hasher

        words = text.split()
        # Find the word index closest to the error position
        err_pos = error.position
        word_offsets: list[int] = []
        cursor = 0
        for w in words:
            pos = text.find(w, cursor)
            word_offsets.append(pos)
            cursor = pos + len(w)

        best_wi = 0
        if word_offsets:
            best_dist = abs(word_offsets[0] - err_pos)
            for wi, wo in enumerate(word_offsets):
                d = abs(wo - err_pos)
                if d < best_dist:
                    best_dist = d
                    best_wi = wi

        prev_words = words[max(0, best_wi - 3) : best_wi]
        next_words = words[best_wi + 1 : best_wi + 4]

        # Count error syllables once
        syllable_seg = None
        error_syllables = 1
        try:
            from myspellchecker.segmenters.regex import RegexSegmenter

            syllable_seg = RegexSegmenter()
            error_syllables = len(syllable_seg.segment_syllables(error_word))
        except (ImportError, ValueError, TypeError):
            pass

        error_len = len(error_word) or 1

        # Try to get semantic checker for MLM logits
        semantic_checker = getattr(self, "_semantic_checker", None)

        all_features: list[list[float]] = []

        # Pre-compute log frequencies for relative normalization
        log_freqs: list[float] = []
        for cand in candidates:
            freq = provider.get_word_frequency(cand) or 0
            if isinstance(freq, (int, float)):
                log_freqs.append(math.log1p(freq))
            else:
                log_freqs.append(0.0)
        max_log_freq = max(log_freqs) if log_freqs else 1.0

        # Pre-compute error word n-gram context probability (for improvement ratio)
        error_ngram_left = 0.0
        error_ngram_right = 0.0
        if ngram_checker:
            if prev_words:
                error_ngram_left = ngram_checker.get_best_left_probability(
                    prev_words[-3:], error_word
                )
            if next_words:
                error_ngram_right = ngram_checker.get_best_right_probability(
                    error_word, next_words[:3]
                )

        # Pre-compute MLM scores for all candidates via score_mask_candidates
        mlm_scores: dict[str, float] = {}
        if semantic_checker is not None:
            try:
                occurrence = text[: error.position].count(error_word)
                mlm_scores = semantic_checker.score_mask_candidates(
                    text, error_word, candidates, occurrence=occurrence
                )
            except (AttributeError, TypeError, ValueError):
                pass

        for i, cand in enumerate(candidates):
            # Distance-related features (0-6)
            edit_dist, weighted_dist, phon_score, syl_diff, plausibility, span_ratio = (
                self._compute_distance_features(
                    error_word, cand, error_syllables, error_len, syllable_seg, hasher
                )
            )

            # 2. Log frequency (absolute)
            log_freq = log_freqs[i]

            # 7. MLM logit — wire actual semantic checker score (pre-computed)
            mlm_logit = mlm_scores.get(cand, 0.0) if mlm_scores else 0.0

            # Context-related features (8-10)
            ngram_left, ngram_right, is_conf = self._compute_context_features(
                cand, confusable_set, ngram_checker, prev_words, next_words
            )

            # 11. Relative log frequency
            relative_log_freq = log_freq / max_log_freq if max_log_freq > 0 else 0.0

            # 12. Character length difference (signed)
            char_length_diff = float(len(cand) - len(error_word))

            # 13. Is substring
            is_substr = 1.0 if (cand in error_word or error_word in cand) else 0.0

            # 14. Original rank signal: pipeline's existing rank as a prior
            original_rank = 1.0 / (1.0 + i)

            # --- v2 features ---

            # 15. N-gram improvement ratio: best of left/right context
            ngram_improv = 0.0
            improvements: list[float] = []
            if error_ngram_left > 0 and ngram_left > 0:
                improvements.append(math.log(ngram_left / error_ngram_left))
            elif error_ngram_left == 0.0 and ngram_left > 0:
                improvements.append(5.0)  # impossible → likely
            if error_ngram_right > 0 and ngram_right > 0:
                improvements.append(math.log(ngram_right / error_ngram_right))
            elif error_ngram_right == 0.0 and ngram_right > 0:
                improvements.append(5.0)
            if improvements:
                ngram_improv = max(-5.0, min(5.0, max(improvements)))

            # 16-17. Edit type classification
            edit_type_subst = 1.0 if len(cand) == len(error_word) else 0.0
            edit_type_delete = 1.0 if len(cand) != len(error_word) else 0.0

            # 18. Character bigram Dice coefficient
            char_dice = self._char_bigram_dice(error_word, cand)

            feat_vec = [
                edit_dist,
                weighted_dist,
                log_freq,
                phon_score,
                syl_diff,
                plausibility,
                span_ratio,
                mlm_logit,
                ngram_left,
                ngram_right,
                is_conf,
                relative_log_freq,
                char_length_diff,
                is_substr,
                original_rank,
                ngram_improv,
                edit_type_subst,
                edit_type_delete,
                char_dice,
            ]
            all_features.append(feat_vec)

        return all_features

    @staticmethod
    def _char_bigram_dice(a: str, b: str) -> float:
        """Character bigram Dice coefficient (multiset) between two strings."""
        from collections import Counter

        if len(a) < 2 and len(b) < 2:
            return 1.0 if a == b else 0.0
        bigrams_a = Counter(a[i : i + 2] for i in range(len(a) - 1)) if len(a) >= 2 else Counter()
        bigrams_b = Counter(b[i : i + 2] for i in range(len(b) - 1)) if len(b) >= 2 else Counter()
        total = sum(bigrams_a.values()) + sum(bigrams_b.values())
        if total == 0:
            return 0.0
        intersection = sum(min(bigrams_a[bg], bigrams_b[bg]) for bg in bigrams_a)
        return 2.0 * intersection / total
