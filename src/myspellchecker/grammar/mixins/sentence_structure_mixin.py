"""Sentence-level structure check mixin for SyntacticRuleChecker.

Provides sentence-level validation methods: overall structure checks,
boundary constraints, and tense agreement detection.

Extracted from ``engine.py`` to reduce file size.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from myspellchecker.grammar.patterns import (
    QUESTION_PARTICLES,
    detect_sentence_type,
    get_dangling_completion_suggestions,
    get_question_completion_suggestions,
    has_dangling_completion_template,
    has_question_particle_context,
    is_second_person_modal_future_question,
)

if TYPE_CHECKING:
    from myspellchecker.core.config import GrammarEngineConfig
    from myspellchecker.grammar.config import GrammarRuleConfig


class SentenceStructureMixin:
    """Mixin providing sentence-level structure check methods.

    Validates sentence-final particles, POS sequences, subject markers,
    question markers, boundary constraints, and tense agreement.
    """

    # --- Type stubs for attributes provided by SyntacticRuleChecker ---
    config: "GrammarRuleConfig"
    grammar_config: "GrammarEngineConfig"

    # --- Type stubs for methods provided by POSTagMixin (resolved via MRO) ---
    def _has_tag(self, pos_string: str | None, target_tag: str) -> bool: ...  # type: ignore[empty-body]
    def _get_primary_tag(self, pos_string: str | None, word: str | None = None) -> str: ...  # type: ignore[empty-body]
    def _is_particle_like_tag(self, tag: str) -> bool: ...  # type: ignore[empty-body]

    # Future time words (including common misspellings that remain valid dictionary entries)
    _FUTURE_TIME_WORDS: frozenset[str] = frozenset(
        {
            "\u1019\u1014\u1000\u103a\u1016\u103c\u1014\u103a",  # tomorrow, standard
            "\u1019\u1014\u1000\u103a\u1016\u1014\u103a",  # tomorrow, variant
            "\u1014\u1031\u102c\u1000\u103a\u1014\u1031\u1037",  # next day
            "\u1014\u1031\u102c\u1000\u103a\u101c",  # next month
            "\u1014\u1031\u102c\u1000\u103a\u1014\u103e\u1005\u103a",  # next year
            "\u1021\u1014\u102c\u1002\u1010\u103a",  # future
        }
    )

    # Past tense markers -> their future counterparts
    _PAST_TO_FUTURE: dict[str, str] = {
        "\u1010\u101a\u103a": "\u1019\u101a\u103a",  # past -> future
        "\u101e\u100a\u103a": "\u1019\u100a\u103a",  # past -> future
    }

    # Past time words that indicate a past-time context
    _PAST_TIME_WORDS: frozenset[str] = frozenset(
        {
            "\u1019\u1014\u1031\u1037\u1000",  # မနေ့က yesterday+က
            "\u1021\u101b\u1004\u103a\u1010\u102f\u1014\u103a\u1038\u1000",  # အရင်တုန်းက
            "\u1010\u102f\u1014\u103a\u1038\u1000",  # တုန်းက back-then+က
            "\u1019\u1014\u1031\u1037",  # မနေ့ yesterday
            "\u101a\u1001\u1004\u103a",  # ယခင် previously
            "\u1021\u101b\u1004\u103a",  # အရင် before/earlier
            "\u101c\u103d\u1014\u103a\u1001\u1032\u1037\u101e\u1031\u102c",  # လွန်ခဲ့သော
        }
    )

    # Future markers -> their past counterparts
    _FUTURE_TO_PAST: dict[str, str] = {
        "\u1019\u101a\u103a": "\u1010\u101a\u103a",  # မယ် -> တယ်
        "\u1019\u100a\u103a": "\u101e\u100a\u103a",  # မည် -> သည်
        # တော့မယ် -> ခဲ့တယ်
        "\u1010\u1031\u102c\u1037\u1019\u101a\u103a": "\u1001\u1032\u1037\u1010\u101a\u103a",
        # လိမ့်မယ် -> ခဲ့တယ်
        "\u101c\u102d\u1019\u1037\u103a\u1019\u101a\u103a": "\u1001\u1032\u1037\u1010\u101a\u103a",
        # လိမ့်မည် -> ခဲ့သည်
        "\u101c\u102d\u1019\u1037\u103a\u1019\u100a\u103a": "\u1001\u1032\u1037\u101e\u100a\u103a",
    }

    # Suffix-based past → future rewrite table (sorted LONGEST-FIRST)
    _PAST_SUFFIX_TO_FUTURE: list[tuple[str, str]] = [
        (
            "\u1001\u1032\u1037\u1015\u102b\u101e\u100a\u103a",  # ခဲ့ပါသည်
            "\u1015\u102b\u1019\u100a\u103a",  # ပါမည်
        ),
        (
            "\u1001\u1032\u1037\u1015\u102b\u1010\u101a\u103a",  # ခဲ့ပါတယ်
            "\u1015\u102b\u1019\u101a\u103a",  # ပါမယ်
        ),
        (
            "\u1001\u1032\u1037\u101e\u100a\u103a",  # ခဲ့သည်
            "\u1019\u100a\u103a",  # မည်
        ),
        (
            "\u1001\u1032\u1037\u1010\u101a\u103a",  # ခဲ့တယ်
            "\u1019\u101a\u103a",  # မယ်
        ),
    ]

    # Suffix-based future → past rewrite table (sorted LONGEST-FIRST)
    _FUTURE_SUFFIX_TO_PAST: list[tuple[str, str]] = [
        (
            "\u1015\u102b\u1019\u100a\u103a",  # ပါမည်
            "\u1001\u1032\u1037\u1015\u102b\u101e\u100a\u103a",  # ခဲ့ပါသည်
        ),
        (
            "\u1015\u102b\u1019\u101a\u103a",  # ပါမယ်
            "\u1001\u1032\u1037\u1015\u102b\u1010\u101a\u103a",  # ခဲ့ပါတယ်
        ),
        (
            "\u1019\u100a\u103a",  # မည်
            "\u1001\u1032\u1037\u101e\u100a\u103a",  # ခဲ့သည်
        ),
        (
            "\u1019\u101a\u103a",  # မယ်
            "\u1001\u1032\u1037\u1010\u101a\u103a",  # ခဲ့တယ်
        ),
    ]

    # Suffix-based past → present rewrite table
    _PAST_SUFFIX_TO_PRESENT: list[tuple[str, str]] = [
        (
            "\u1001\u1032\u1037\u101e\u100a\u103a",  # ခဲ့သည်
            "\u1014\u1031\u101e\u100a\u103a",  # နေသည်
        ),
        (
            "\u1001\u1032\u1037\u1010\u101a\u103a",  # ခဲ့တယ်
            "\u1014\u1031\u1010\u101a\u103a",  # နေတယ်
        ),
    ]

    # Present time words
    _PRESENT_TIME_WORDS: frozenset[str] = frozenset(
        {
            "\u101c\u1000\u103a\u101b\u103e\u102d",  # လက်ရှိ
            "\u101a\u1001\u102f",  # ယခု
            "\u1021\u1001\u102f",  # အခု
        }
    )

    # Normalized present time prefixes for prefix-based matching
    _PRESENT_TIME_PREFIXES: tuple[str, ...] = (
        "\u101c\u1000\u103a\u101b\u103e\u102d",  # လက်ရှိ
        "\u101a\u1001\u102f",  # ယခု
        "\u1021\u1001\u102f",  # အခု
    )

    # Past time prefixes for prefix-based matching (normalized)
    _PAST_TIME_PREFIXES: tuple[str, ...] = (
        "\u1015\u103c\u102e\u1038\u1001\u1032\u1037\u101e\u100a\u1037\u103a",  # ပြီးခဲ့သည့်
        "\u1015\u103c\u102e\u1038\u1001\u1032\u1037\u101e\u1031\u102c",  # ပြီးခဲ့သော
        "\u101c\u103d\u1014\u103a\u1001\u1032\u1037\u101e\u1031\u102c",  # လွန်ခဲ့သော
    )

    # Quote particles that form a boundary between quoted and quoting clauses
    _QUOTE_PARTICLES: frozenset[str] = frozenset(
        {
            "\u101f\u102f",  # ဟု
            "\u101f\u1030",  # ဟူ
        }
    )

    def _check_sentence_structure(
        self, words: list[str], pos_tags: list[str | None]
    ) -> list[tuple[int, str, str, float]]:
        """
        Check overall sentence structure for grammatical issues.

        Validates:
        1. Sentence-final particle usage
        2. Invalid POS sequences
        3. Missing essential particles (subject/object markers)
        4. Unusual word order patterns

        Args:
            words: List of words in the sentence.
            pos_tags: List of POS tags corresponding to words.

        Returns:
            List of (index, error_word, suggestion, confidence) tuples.
        """
        errors: list[tuple[int, str, str, float]] = []

        if not words:
            return errors

        if not pos_tags or len(pos_tags) != len(words):
            return errors

        # Skip boundary checks on single-word fragments (e.g., noun phrases)
        if len(words) <= 1:
            return errors

        # Rule 1: Check sentence-final particle
        last_word = words[-1]
        last_pos = pos_tags[-1]
        sentence_type = detect_sentence_type(words)
        implicit_question = is_second_person_modal_future_question(words)
        question_context = sentence_type == "question" or implicit_question
        has_question_particle = has_question_particle_context(words)

        is_final_particle = self.config.is_sentence_final(last_word)
        question_fallback_added = False

        if not is_final_particle:
            if self._has_tag(last_pos, "V") and not self._has_tag(last_pos, "N"):
                # Verb at end without sentence-final particle.
                # Only flag unambiguous verbs -- N|V words are often noun phrases.
                # Skip question particles -- they are valid sentence-final.
                # Also skip imperative auxiliary verbs that validly end commands.
                _IMPERATIVE_FINALS = frozenset({"လိုက်", "ကြ", "ပေး", "ထား", "စမ်း", "ကြည့်"})
                # Imperative context: verb after resultative complement
                _IMPERATIVE_CONTEXTS = frozenset({"အောင်"})
                prev_word_sfp = words[-2] if len(words) >= 2 else ""
                is_imperative_ctx = prev_word_sfp in _IMPERATIVE_CONTEXTS
                if (
                    last_word in QUESTION_PARTICLES
                    or last_word in _IMPERATIVE_FINALS
                    or is_imperative_ctx
                ):
                    pass
                elif question_context and not has_question_particle:
                    question_suggestions = get_question_completion_suggestions(
                        last_word,
                        words,
                        prefer_yes_no=implicit_question,
                        phrase_first=True,
                    )
                    if question_suggestions:
                        errors.append(
                            (
                                len(words) - 1,
                                last_word,
                                question_suggestions[0],
                                self.grammar_config.question_confidence,
                            )
                        )
                        question_fallback_added = True
                else:
                    errors.append(
                        (
                            len(words) - 1,
                            last_word,
                            f"{last_word}\u1010\u101a\u103a",
                            self.grammar_config.pos_sequence_confidence,
                        )
                    )
            elif self._has_tag(last_pos, "ADJ") and not self._has_tag(last_pos, "V"):
                # Rule 1b: Sentence ends with bare adjective without verb/particle.
                # Exception: Demonstrative pronouns commonly appear at sentence
                # boundaries and naturally precede nouns.
                _DEMONSTRATIVES = frozenset(
                    {
                        "\u1012\u102e",
                        "\u101f\u102d\u102f\u1038",
                        "\u101f\u102d\u102f",
                        "\u1011\u102d\u102f",
                        "\u1021\u1032\u1012\u102e",
                        "\u1024",
                        "\u101a\u1004\u103a\u1038",
                    }
                )
                if last_word not in _DEMONSTRATIVES:
                    errors.append(
                        (
                            len(words) - 1,
                            last_word,
                            f"{last_word} \u1016\u103c\u1005\u103a\u101e\u100a\u103a",
                            self.grammar_config.pos_sequence_confidence,
                        )
                    )

        # Rule 2: Check for invalid POS sequences
        for i in range(1, len(words)):
            prev_pos = pos_tags[i - 1]
            curr_pos = pos_tags[i]
            prev_word = words[i - 1]
            curr_word = words[i]

            # Use config-based sequence checking
            prev_tag_precise = self._get_primary_tag(prev_pos, prev_word)
            curr_tag_precise = self._get_primary_tag(curr_pos, curr_word)

            sequence_error = self.config.get_invalid_sequence_error(
                prev_tag_precise, curr_tag_precise
            )

            # If no error with precise tags, check with generic tags (fallback)
            if not sequence_error and (
                self._is_particle_like_tag(prev_tag_precise)
                or self._is_particle_like_tag(curr_tag_precise)
            ):
                prev_tag_generic = (
                    "P" if self._is_particle_like_tag(prev_tag_precise) else prev_tag_precise
                )
                curr_tag_generic = (
                    "P" if self._is_particle_like_tag(curr_tag_precise) else curr_tag_precise
                )
                sequence_error = self.config.get_invalid_sequence_error(
                    prev_tag_generic, curr_tag_generic
                )

            if sequence_error:
                # Handle both dict (production) and string (legacy/test) returns
                if isinstance(sequence_error, str):
                    errors.append(
                        (i, words[i], words[i], self.grammar_config.pos_sequence_confidence)
                    )
                    continue

                # Check for word-level exceptions (e.g., V-V with auxiliary verbs)
                exceptions = sequence_error.get("exceptions", [])
                if exceptions and (curr_word in exceptions or prev_word in exceptions):
                    continue

                # Check for tag-level exceptions
                _valid_follower_tags = frozenset(
                    {
                        "P_SENT",
                        "P_NEG_SENT",
                        "P_POL",
                        "P_TOP",
                        "P_ADD",
                        "P_EMPH",
                    }
                )
                _valid_preceding_tags = frozenset(
                    {
                        "P_CAUS",
                        "P_COND",
                        "P_SEQ",
                        "P_POL",
                        "P_TOP",
                        "P_ADD",
                        "P_EMPH",
                        "P_SENT",
                        "P_NEG_SENT",
                    }
                )
                if (
                    curr_tag_precise in _valid_follower_tags
                    or prev_tag_precise in _valid_preceding_tags
                ):
                    continue

                severity = sequence_error.get("severity", "warning")
                if severity == "error":
                    msg = sequence_error.get(
                        "message",
                        f"Invalid POS sequence: {prev_tag_precise}-{curr_tag_precise}",
                    )
                    errors.append((i, words[i], msg, self.grammar_config.pos_sequence_confidence))

        # Rule 3: Check for subject marker after initial noun
        # Only fire for sentences with 3+ words — 2-word "N V" is valid minimal SOV
        if len(words) >= 3:
            first_pos = pos_tags[0]
            second_word = words[1]
            second_pos = pos_tags[1]

            # If first word is a noun and second isn't a subject marker
            if self._has_tag(first_pos, "N"):
                subject_markers = {
                    "\u1000",
                    "\u101e\u100a\u103a",
                    "\u101f\u102c",
                }
                if second_word not in subject_markers:
                    # Check if second word is a verb (missing subject marker)
                    if self._has_tag(second_pos, "V"):
                        errors.append(
                            (
                                0,
                                words[0],
                                f"{words[0]}\u1000",
                                self.grammar_config.low_confidence_threshold,
                            )
                        )

        # Rule 4: Question marker fallback for question-like contexts
        if (
            question_context
            and not has_question_particle
            and not is_final_particle
            and not question_fallback_added
        ):
            if last_word not in QUESTION_PARTICLES:
                question_suggestions = get_question_completion_suggestions(
                    last_word,
                    words,
                    prefer_yes_no=implicit_question,
                    phrase_first=True,
                )
                if question_suggestions:
                    errors.append(
                        (
                            len(words) - 1,
                            last_word,
                            question_suggestions[0],
                            self.grammar_config.question_confidence,
                        )
                    )

        return errors

    def _check_sentence_boundaries(
        self, words: list[str], pos_tags: list[str | None]
    ) -> list[tuple[int, str, str, float]]:
        """
        Check sentence start and end constraints.

        Args:
            words: List of words in the sentence.
            pos_tags: List of POS tags corresponding to words.

        Returns:
            List of (index, error_word, suggestion, confidence) tuples.
        """
        errors: list[tuple[int, str, str, float]] = []
        if not words:
            return errors

        if not pos_tags or len(pos_tags) != len(words):
            return errors

        # Check start constraint
        start_word = words[0]
        start_pos = pos_tags[0]
        start_tag_precise = self._get_primary_tag(start_pos, start_word)

        for rule in self.config.sentence_start_constraints:
            forbidden_words = rule.get("forbidden_words", [])
            forbidden_tags = rule.get("forbidden_tags", [])

            if start_word in forbidden_words or start_tag_precise in forbidden_tags:
                errors.append((0, start_word, "Invalid start", rule.get("confidence", 0.9)))
                break

        # Check end constraint
        end_idx = len(words) - 1
        end_word = words[end_idx]
        end_pos = pos_tags[end_idx]
        end_tag_precise = self._get_primary_tag(end_pos, end_word)

        # Skip end constraint if the word is a known sentence-final particle.
        if not self.config.is_sentence_final(end_word):
            for rule in self.config.sentence_end_constraints:
                forbidden_words = rule.get("forbidden_words", [])
                forbidden_tags = rule.get("forbidden_tags", [])

                if end_word in forbidden_words or end_tag_precise in forbidden_tags:
                    enabled = self.grammar_config.enable_targeted_grammar_completion_templates
                    completion_suggestions = get_dangling_completion_suggestions(
                        end_word, enabled=enabled
                    )
                    if end_idx > 0:
                        combined_ending = words[end_idx - 1] + end_word
                        if has_dangling_completion_template(combined_ending, enabled=enabled):
                            completion_suggestions = get_dangling_completion_suggestions(
                                combined_ending, enabled=enabled
                            )
                    errors.append(
                        (
                            end_idx,
                            end_word,
                            completion_suggestions[0],
                            rule.get("confidence", 0.9),
                        )
                    )
                    break

        return errors

    @staticmethod
    def _has_time_context(
        words: list[str],
        word_set: frozenset[str],
        prefix_tuple: tuple[str, ...],
    ) -> tuple[bool, int]:
        """Check whether any word in *words* signals a time context.

        Performs two checks:
        1. Exact membership in *word_set*.
        2. Prefix-based matching against *prefix_tuple*.

        Returns:
            ``(found, position)`` where *position* is the index of the
            first matching word, or ``-1`` if not found.
        """
        for i, w in enumerate(words):
            if w in word_set:
                return True, i
            for pfx in prefix_tuple:
                if w.startswith(pfx):
                    return True, i
        return False, -1

    def _match_tense_near_end(
        self,
        words: list[str],
        standalone_map: dict[str, str],
        suffix_table: list[tuple[str, str]],
        search_bound: int | None = None,
    ) -> tuple[int, str, str] | None:
        """Find a tense marker near the end of *words* and return a rewrite.

        Checks in two passes:
        1. **Standalone particle match** — the word itself is a key in
           *standalone_map*.
        2. **Suffix-based scan** — iterate ALL words (right-to-left) and
           check if any word ends with a suffix from *suffix_table*.
           The table must be sorted LONGEST-FIRST.

        Args:
            words: Word list of the sentence.
            standalone_map: ``{past_particle: future_particle, ...}``.
            suffix_table: ``[(old_suffix, new_suffix), ...]`` sorted
                longest-first.
            search_bound: If given, only scan ``words[:search_bound]``
                (exclusive upper bound).

        Returns:
            ``(index, old_word, suggestion)`` or ``None``.
        """
        end = search_bound if search_bound is not None else len(words)

        # Pass 1: standalone particle match (last 2 words within bound)
        for offset in range(1, min(3, end + 1)):
            idx = end - offset
            if idx < 0:
                break
            word = words[idx]
            if word in standalone_map:
                return idx, word, standalone_map[word]

        # Pass 2: suffix-based scan (all words right-to-left within bound)
        for idx in range(end - 1, -1, -1):
            word = words[idx]
            for old_suffix, new_suffix in suffix_table:
                if word.endswith(old_suffix):
                    stem = word[: len(word) - len(old_suffix)]
                    return idx, word, stem + new_suffix

        return None

    def _check_tense_agreement(self, words: list[str]) -> list[tuple[int, str, str, float]]:
        """
        Check for tense agreement between time words and sentence-ending markers.

        Detects three mismatch patterns:
        1. Future time word (e.g., "tomorrow") + past/present tense ender
           -> suggests the future form.
        2. Past time word (e.g., "yesterday") + future marker
           -> suggests the past form.
        3. Present time word (e.g., "now") + past tense ender
           -> suggests the present form.

        A quote boundary guard prevents false positives when a time word
        and a tense marker are separated by a quote particle (ဟု / ဟူ).

        Returns:
            List of (index, error_word, suggestion, confidence) tuples.
        """
        if len(words) < 3:
            return []

        has_future_time = any(w in self._FUTURE_TIME_WORDS for w in words)
        has_past_time, past_time_pos = self._has_time_context(
            words, self._PAST_TIME_WORDS, self._PAST_TIME_PREFIXES
        )
        has_present_time, present_time_pos = self._has_time_context(
            words, self._PRESENT_TIME_WORDS, self._PRESENT_TIME_PREFIXES
        )

        if not has_future_time and not has_past_time and not has_present_time:
            return []

        # Past aspect markers that carry the actual tense information.
        _PAST_ASPECT_MARKERS = frozenset(
            {
                "\u1001\u1032\u1037",  # past aspect marker (ခဲ့)
            }
        )

        # --- Quote boundary guard ---
        # Find the last quote particle (ဟု/ဟူ) position.
        # If the quote particle separates the time word from the tense
        # marker, skip the detection — the tense marker belongs to the
        # outer (quoting) clause, not the inner (quoted) clause.
        last_quote_pos: int = -1
        for qi in range(len(words) - 1, -1, -1):
            if words[qi] in self._QUOTE_PARTICLES:
                last_quote_pos = qi
                break

        def _compute_search_bound(time_word_pos: int) -> int | None:
            """Return search_bound if a quote particle separates contexts."""
            if last_quote_pos > time_word_pos:
                return last_quote_pos
            return None

        # --- Direction 1: future time + past tense marker ---
        if has_future_time:
            # Determine the time word position for quote guard
            ft_pos = -1
            for i, w in enumerate(words):
                if w in self._FUTURE_TIME_WORDS:
                    ft_pos = i
                    break
            search_bound = _compute_search_bound(ft_pos)

            # Standalone check (with past aspect guard)
            end = search_bound if search_bound is not None else len(words)
            for offset in range(1, min(3, end + 1)):
                idx = end - offset
                if idx < 0:
                    break
                word = words[idx]
                if word in self._PAST_TO_FUTURE:
                    # Skip if preceded by a past aspect marker
                    if idx > 0 and words[idx - 1] in _PAST_ASPECT_MARKERS:
                        continue
                    if is_second_person_modal_future_question(words):
                        question_suggestions = get_question_completion_suggestions(
                            word,
                            words,
                            prefer_yes_no=True,
                            phrase_first=False,
                        )
                        if question_suggestions:
                            return [(idx, word, question_suggestions[0], 0.8)]
                    future_form = self._PAST_TO_FUTURE[word]
                    return [(idx, word, future_form, 0.8)]

            # Suffix-based scan for direction 1
            match = self._match_tense_near_end(
                words,
                standalone_map={},
                suffix_table=self._PAST_SUFFIX_TO_FUTURE,
                search_bound=search_bound,
            )
            if match:
                return [(match[0], match[1], match[2], 0.8)]

        # --- Direction 2: past time + future marker ---
        if has_past_time:
            search_bound = _compute_search_bound(past_time_pos)
            match = self._match_tense_near_end(
                words,
                standalone_map=self._FUTURE_TO_PAST,
                suffix_table=self._FUTURE_SUFFIX_TO_PAST,
                search_bound=search_bound,
            )
            if match:
                return [(match[0], match[1], match[2], 0.8)]

        # --- Direction 3: present time + past tense ender ---
        if has_present_time:
            search_bound = _compute_search_bound(present_time_pos)
            match = self._match_tense_near_end(
                words,
                standalone_map={},
                suffix_table=self._PAST_SUFFIX_TO_PRESENT,
                search_bound=search_bound,
            )
            if match:
                return [(match[0], match[1], match[2], 0.8)]

        return []
