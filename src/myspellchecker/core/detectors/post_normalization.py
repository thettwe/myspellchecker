"""Post-normalization word and particle level detectors.

Detectors for medial confusion, particle confusion, compound confusion,
broken stacking, punctuation errors, and related character/word-level issues.
These run on normalized text and mutate the errors list in place.

Data for homophone/aukmyit confusion is loaded from
``rules/homophone_confusion.yaml`` at module level, with fallback to
hardcoded defaults if the YAML file is missing or invalid.

Extracted from ``spellchecker.py`` to reduce file size while preserving
the exact same method signatures and behaviour.

Method groups are factored into sub-mixins under ``post_norm_mixins``:
- MedialConfusionMixin: medial ya-pin/ya-yit confusion detection
- ParticleDetectionMixin: particle-related detection methods
- CompoundDetectionMixin: compound confusion and broken compound detection
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from myspellchecker.core.constants import (
    ET_BROKEN_STACKING,
    ET_CONFUSABLE_ERROR,
    ET_DUPLICATE_PUNCTUATION,
    ET_MISSING_PUNCTUATION,
    ET_REGISTER_MIXING,
    ET_VOWEL_AFTER_ASAT,
    ET_WORD,
    ET_WRONG_PUNCTUATION,
    LEXICALIZED_COMPOUND_MIN_FREQ,
)

if TYPE_CHECKING:
    from myspellchecker.providers.base import DictionaryProvider
from myspellchecker.core.detector_data import (
    TEXT_DETECTOR_CONFIDENCES,
)
from myspellchecker.core.detector_data import (
    norm_dict as _norm_dict,
)
from myspellchecker.core.detector_data import (
    norm_dict_context as _norm_dict_context,
)
from myspellchecker.core.detectors.post_norm_mixins import (
    CollocationDetectionMixin,
    CompoundDetectionMixin,
    MedialConfusionMixin,
    ParticleDetectionMixin,
)
from myspellchecker.core.detectors.utils import (
    get_existing_positions,
    iter_occurrences,
    try_replace_syllable_error,
)
from myspellchecker.core.response import Error, SyllableError
from myspellchecker.text.normalize import normalize
from myspellchecker.utils.logging_utils import get_logger

logger = get_logger(__name__)

# (insert_char, trigger_chars) pairs for missing diacritic detection.
_DIACRITIC_INSERTIONS: list[tuple[str, frozenset[str]]] = [
    ("\u1036", frozenset({"\u102f", "\u1030"})),  # anusvara after U/UU
    ("\u1037", frozenset({"\u103a"})),  # dot-below after asat
]

# ── Hardcoded defaults (fallback when YAML is unavailable) ──

_DEFAULT_AUKMYIT_CONTEXT: dict[str, tuple[str, tuple[str, ...]]] = _norm_dict_context(
    {
        "ထည်": ("ထည့်", ("လိုက်", "ပေး", "သွင်း", "ထား", "ပါ", "မယ်")),
    }
)

_DEFAULT_EXTRA_AUKMYIT_CONTEXT: dict[str, tuple[str, tuple[str, ...]]] = _norm_dict_context(
    {
        "ပြော့": (
            "ပြော",
            ("ကြား", "ပြ", "ဆို", "ခွင့်", "ရေး", "သည်", "ခဲ့", "မည်", "ပါ"),
        ),
    }
)

_DEFAULT_HOMOPHONE_LEFT_CONTEXT: dict[str, tuple[str, tuple[str, ...]]] = _norm_dict_context(
    {
        "ဖက်": (
            "ဖတ်",
            (
                "စာအုပ်",
                "စာ",
                "သတင်းစာ",
                "ဂျာနယ်",
                "မဂ္ဂဇင်း",
                "မှတ်တမ်း",
                "အမိန့်စာ",
                "စာချုပ်",
                "ဥပဒေ",
                "စာမျက်နှာ",
                "သင်ခန်းစာ",
                "ဘာသာစကား",
            ),
        ),
    }
)

_DEFAULT_HOMOPHONE_LEFT_SUFFIXES: dict[str, tuple[str, tuple[str, ...]]] = _norm_dict_context(
    {
        "ဖက်": ("ဖတ်", ("စာ", "တမ်း", "ချုပ်", "အုပ်")),
    }
)

# ── YAML loading ──

_YAML_PATH = Path(__file__).resolve().parent.parent.parent / "rules" / "homophone_confusion.yaml"


def _load_homophone_confusion() -> tuple[
    dict[str, tuple[str, tuple[str, ...]]],
    dict[str, tuple[str, tuple[str, ...]]],
    dict[str, tuple[str, tuple[str, ...]]],
    dict[str, tuple[str, tuple[str, ...]]],
]:
    """Load homophone confusion dicts from YAML with fallback.

    Returns:
        Tuple of (aukmyit_context, extra_aukmyit_context,
        left_context_homophones, left_suffix_triggers),
        all with keys/values normalized via _norm_dict_context.
    """
    if not _YAML_PATH.exists():
        logger.debug(
            "Homophone confusion YAML not found at %s, using defaults",
            _YAML_PATH,
        )
        return (
            _DEFAULT_AUKMYIT_CONTEXT,
            _DEFAULT_EXTRA_AUKMYIT_CONTEXT,
            _DEFAULT_HOMOPHONE_LEFT_CONTEXT,
            _DEFAULT_HOMOPHONE_LEFT_SUFFIXES,
        )

    try:
        import yaml

        with open(_YAML_PATH, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not data or not isinstance(data, dict):
            logger.warning("Homophone confusion YAML empty or invalid, using defaults")
            return (
                _DEFAULT_AUKMYIT_CONTEXT,
                _DEFAULT_EXTRA_AUKMYIT_CONTEXT,
                _DEFAULT_HOMOPHONE_LEFT_CONTEXT,
                _DEFAULT_HOMOPHONE_LEFT_SUFFIXES,
            )

        def _parse_context_section(
            raw: dict[str, dict[str, str | list[str]]] | None,
            trigger_key: str = "triggers",
        ) -> dict[str, tuple[str, tuple[str, ...]]] | None:
            """Parse a YAML section with correct/triggers structure."""
            if not isinstance(raw, dict) or not raw:
                return None
            parsed: dict[str, tuple[str, tuple[str, ...]]] = {}
            for k, v in raw.items():
                if isinstance(v, dict):
                    correct = v.get("correct", "")
                    triggers = v.get(trigger_key, [])
                    if correct and triggers:
                        parsed[k] = (correct, tuple(triggers))
            return parsed if parsed else None

        # -- Aukmyit context (missing dot-below) --
        raw_aukmyit = data.get("aukmyit_context", {})
        parsed = _parse_context_section(raw_aukmyit)
        aukmyit = _norm_dict_context(parsed) if parsed else _DEFAULT_AUKMYIT_CONTEXT

        # -- Extra aukmyit context (extra dot-below) --
        raw_extra = data.get("extra_aukmyit_context", {})
        parsed = _parse_context_section(raw_extra)
        extra_aukmyit = _norm_dict_context(parsed) if parsed else _DEFAULT_EXTRA_AUKMYIT_CONTEXT

        # -- Left-context homophones --
        raw_left = data.get("left_context_homophones", {})
        parsed = _parse_context_section(raw_left)
        left_context = _norm_dict_context(parsed) if parsed else _DEFAULT_HOMOPHONE_LEFT_CONTEXT

        # -- Left-suffix triggers --
        raw_suffix = data.get("left_suffix_triggers", {})
        suffix_parsed: dict[str, tuple[str, tuple[str, ...]]] | None = None
        if isinstance(raw_suffix, dict) and raw_suffix:
            suffix_parsed_tmp: dict[str, tuple[str, tuple[str, ...]]] = {}
            for k, v in raw_suffix.items():
                if isinstance(v, dict):
                    correct = v.get("correct", "")
                    suffixes = v.get("suffixes", [])
                    if correct and suffixes:
                        suffix_parsed_tmp[k] = (correct, tuple(suffixes))
            suffix_parsed = suffix_parsed_tmp if suffix_parsed_tmp else None
        left_suffixes = (
            _norm_dict_context(suffix_parsed) if suffix_parsed else _DEFAULT_HOMOPHONE_LEFT_SUFFIXES
        )

        logger.debug(
            "Loaded homophone confusion rules from YAML: "
            "%d aukmyit, %d extra_aukmyit, %d left_context, %d left_suffix",
            len(aukmyit),
            len(extra_aukmyit),
            len(left_context),
            len(left_suffixes),
        )
        return aukmyit, extra_aukmyit, left_context, left_suffixes

    except Exception:
        logger.warning(
            "Failed to load homophone confusion YAML, using defaults",
            exc_info=True,
        )
        return (
            _DEFAULT_AUKMYIT_CONTEXT,
            _DEFAULT_EXTRA_AUKMYIT_CONTEXT,
            _DEFAULT_HOMOPHONE_LEFT_CONTEXT,
            _DEFAULT_HOMOPHONE_LEFT_SUFFIXES,
        )


# Load at module level (once, at import time)
(
    _AUKMYIT_CONTEXT,
    _EXTRA_AUKMYIT_CONTEXT,
    _HOMOPHONE_LEFT_CONTEXT,
    _HOMOPHONE_LEFT_SUFFIXES,
) = _load_homophone_confusion()


class PostNormalizationDetectorsMixin(
    MedialConfusionMixin,
    ParticleDetectionMixin,
    CompoundDetectionMixin,
    CollocationDetectionMixin,
):
    """Mixin providing post-normalization word/particle detector methods.

    These detectors run on normalized text and mutate the errors list.
    They cover medial confusion, particle typos, compound word issues,
    collocation errors, broken/missing stacking, and punctuation errors.

    Data for homophone/aukmyit confusion is loaded from
    ``rules/homophone_confusion.yaml`` at module level, falling back
    to hardcoded defaults if the YAML is unavailable.

    Method groups are factored into sub-mixins:
    - MedialConfusionMixin: ``_detect_medial_confusion``,
      ``_detect_colloquial_contractions``
    - ParticleDetectionMixin: ``_detect_missing_asat``,
      ``_detect_missing_visarga_suffix``, ``_detect_particle_confusion``,
      ``_detect_sequential_particle_confusion``,
      ``_detect_particle_misuse``,
      ``_detect_ha_htoe_particle_typos``, ``_detect_dangling_particles``
    - CompoundDetectionMixin: ``_detect_compound_confusion_typos``,
      ``_detect_broken_compound_space``
    - CollocationDetectionMixin: ``_detect_collocation_errors``
    """

    # --- Type stubs for attributes provided by SpellChecker or sibling mixins ---
    provider: "DictionaryProvider"
    _COLLOQUIAL_ENDINGS_WITH_STRIPPED: frozenset[str]  # from SentenceDetectorsMixin
    _FORMAL_ENDINGS_WITH_STRIPPED: frozenset[str]  # from SentenceDetectorsMixin
    _VOWEL_SIGNS: frozenset[str]  # from PreNormalizationDetectorsMixin

    # ----- Dot-below (aukmyit) confusion: context-dependent -----
    # Words where missing dot-below (့ U+1037) creates a valid but wrong word.
    # key = wrong form, value = (correct form, tuple of right-context triggers)
    # Only flagged when followed by one of the context triggers.
    _AUKMYIT_CONTEXT: dict[str, tuple[str, tuple[str, ...]]] = _AUKMYIT_CONTEXT

    # ----- Extra dot-below (aukmyit) confusion: context-dependent -----
    # Words where an EXTRA dot-below (့) creates a valid but wrong word.
    # key = wrong form (with dot-below), value = (correct form, tuple of right-context triggers)
    # Inverse of _AUKMYIT_CONTEXT (which handles MISSING dot-below).
    _EXTRA_AUKMYIT_CONTEXT: dict[str, tuple[str, tuple[str, ...]]] = _EXTRA_AUKMYIT_CONTEXT

    # ----- Homophone context: left-context based -----
    # Homophones where the correct form depends on preceding word.
    # key = wrong form, value = (correct form, tuple of left-context triggers)
    # Flagged when preceded by one of the context triggers.
    # Also supports suffix-based trigger generalization via _HOMOPHONE_LEFT_SUFFIXES.
    _HOMOPHONE_LEFT_CONTEXT: dict[str, tuple[str, tuple[str, ...]]] = _HOMOPHONE_LEFT_CONTEXT

    # Suffix-based trigger generalization for homophone left context.
    # If any preceding word ends with one of these suffixes, it counts as a trigger.
    # Maps wrong_form -> (correct_form, tuple of suffix triggers)
    _HOMOPHONE_LEFT_SUFFIXES: dict[str, tuple[str, tuple[str, ...]]] = _HOMOPHONE_LEFT_SUFFIXES

    # Verb prefixes that should be stripped before homophone left-context lookup.
    # e.g., မဖက် (negation + hug) -> strip မ -> check context for ဖက်
    _VERB_PREFIXES: frozenset[str] = frozenset({normalize("မ")})

    def _detect_aukmyit_confusion(self, text: str, errors: list[Error]) -> None:
        """Detect missing dot-below (့) when context disambiguates.

        ထည် (fabric) vs ထည့် (add/insert): both valid words, but when followed
        by action verbs like လိုက်/ပေး/သွင်း, it always means "add".
        """
        existing_positions = get_existing_positions(errors)
        for wrong, (correct, triggers) in self._AUKMYIT_CONTEXT.items():
            for idx, end in iter_occurrences(text, wrong):
                # Check right context: next content (optionally after space) is a trigger
                tail = text[end:].lstrip()
                if not any(tail.startswith(t) for t in triggers):
                    continue
                new_err = SyllableError(
                    text=wrong,
                    position=idx,
                    suggestions=[correct],
                    confidence=TEXT_DETECTOR_CONFIDENCES.get("aukmyit_confusion", 0.88),
                    error_type=ET_CONFUSABLE_ERROR,
                )
                if idx in existing_positions:
                    try_replace_syllable_error(errors, idx, new_err)
                    continue
                errors.append(new_err)

    def _detect_extra_aukmyit_confusion(self, text: str, errors: list[Error]) -> None:
        """Detect spurious dot-below (့) when context disambiguates.

        ပြော့ (soft/mild) vs ပြော (say/tell): both valid words, but when followed
        by communication verbs like ကြား/ပြ/ဆို, it always means "say".
        Inverse of _detect_aukmyit_confusion (which handles MISSING dot-below).
        """
        existing_positions = get_existing_positions(errors)
        for wrong, (correct, triggers) in self._EXTRA_AUKMYIT_CONTEXT.items():
            for idx, end in iter_occurrences(text, wrong):
                # Check right context: next content (optionally after space) is a trigger
                tail = text[end:].lstrip()
                if not any(tail.startswith(t) for t in triggers):
                    continue
                new_err = SyllableError(
                    text=wrong,
                    position=idx,
                    suggestions=[correct],
                    confidence=TEXT_DETECTOR_CONFIDENCES.get("aukmyit_confusion", 0.88),
                    error_type=ET_CONFUSABLE_ERROR,
                )
                if idx in existing_positions:
                    try_replace_syllable_error(errors, idx, new_err)
                    continue
                errors.append(new_err)

    def _detect_homophone_left_context(self, text: str, errors: list[Error]) -> None:
        """Detect homophones disambiguated by left context.

        ဖက် (hug/fold) vs ဖတ် (read): both valid, but after စာ/စာအုပ် it's always reading.

        Uses a multi-strategy approach:
        1. Exact match: preceding word exactly matches a trigger
        2. Suffix match: preceding word ends with a suffix trigger (generalizable)
        3. Windowed scan: scans up to N tokens back (not just immediate neighbor)
        4. Prefix stripping: strips verb prefixes (e.g., negation မ) before lookup
        """
        existing_positions = get_existing_positions(errors)
        for wrong, (correct, triggers) in self._HOMOPHONE_LEFT_CONTEXT.items():
            suffix_triggers: tuple[str, ...] = ()
            if wrong in self._HOMOPHONE_LEFT_SUFFIXES:
                _, suffix_triggers = self._HOMOPHONE_LEFT_SUFFIXES[wrong]

            for idx, _end in iter_occurrences(text, wrong):
                # Also handle prefix-stripped forms (e.g., မဖက် → strip မ → ဖက်)
                actual_wrong = wrong
                actual_idx = idx
                for prefix in self._VERB_PREFIXES:
                    if idx >= len(prefix) and text[idx - len(prefix) : idx] == prefix:
                        actual_idx = idx - len(prefix)
                        actual_wrong = prefix + wrong
                        break

                # Windowed left-context scan: check tokens within a clause window
                head = text[:actual_idx].rstrip()
                # Split into tokens (space-separated) and scan up to 5 tokens back
                tokens = head.split()
                window = tokens[-5:] if len(tokens) > 5 else tokens

                matched = False
                for token in reversed(window):
                    # Exact trigger match (endswith handles particles at end)
                    if any(token.endswith(t) for t in triggers):
                        matched = True
                        break
                    # Exact trigger contained in token (handles particles appended)
                    if any(t in token for t in triggers):
                        matched = True
                        break
                    # Suffix-based trigger generalization (substring: handles
                    # particles appended after the suffix, e.g., မှတ်တမ်းကို)
                    if suffix_triggers and any(s in token for s in suffix_triggers):
                        matched = True
                        break

                if not matched:
                    continue

                new_err = SyllableError(
                    text=actual_wrong,
                    position=actual_idx,
                    suggestions=[correct if actual_wrong == wrong else prefix + correct],
                    confidence=TEXT_DETECTOR_CONFIDENCES.get("homophone_left_context", 0.88),
                    error_type=ET_CONFUSABLE_ERROR,
                )
                if idx in existing_positions or actual_idx in existing_positions:
                    # Try replacing at both possible positions (idx for
                    # the wrong-form match, actual_idx for prefix-stripped).
                    replaced = try_replace_syllable_error(errors, idx, new_err)
                    if not replaced and actual_idx != idx:
                        try_replace_syllable_error(errors, actual_idx, new_err)
                    continue
                errors.append(new_err)

    def _detect_missing_diacritic_in_compound(self, text: str, errors: list[Error]) -> None:
        """Detect invalid words that become valid with anusvara (ံ) or dot-below (့).

        Catches patterns like အသုးပြု → အသုံးပြု where the anusvara is missing.
        The word validator's multi-syllable skip heuristic may skip these tokens
        because most constituent syllables are valid independently.

        Works at prefix level: for compound tokens like ``အသုးပြုပါ``, checks
        if inserting the diacritic makes a PREFIX valid (e.g., ``အသုံး`` freq=49K)
        even when the full modified token isn't a single dictionary entry.
        """
        if not self.provider:
            return

        from myspellchecker.core.detectors.utils import get_tokenized

        tokenized = get_tokenized(self, text)

        for span in tokenized:
            token = span.text
            pos = span.position

            # Skip positions that already have a *text-level* error (high confidence).
            # Generic syllable/word errors are OK to override with our more specific
            # diacritic suggestion.
            existing_at_pos = [e for e in errors if e.position == pos]
            if existing_at_pos and any(
                getattr(e, "error_type", "") not in ("invalid_syllable", "invalid_word")
                for e in existing_at_pos
            ):
                continue
            if not any("\u1000" <= ch <= "\u109f" for ch in token):
                continue
            if self.provider.is_valid_word(token):
                continue

            found = False
            for insert_char, triggers in _DIACRITIC_INSERTIONS:
                if found:
                    break
                for i in range(1, len(token)):
                    if token[i - 1] not in triggers:
                        continue

                    modified = token[:i] + insert_char + token[i:]

                    # Prefer prefix-level detection (compound case):
                    # e.g., "အသုံးပြုပါ" → prefix "အသုံး" is more targeted
                    # than flagging the whole compound token.
                    for j in range(i + 2, len(modified)):
                        prefix = modified[:j]
                        orig_prefix = token[: j - 1]  # -1 for inserted char
                        if self.provider.is_valid_word(orig_prefix):
                            continue  # original prefix was already valid
                        if not self.provider.is_valid_word(prefix):
                            continue
                        freq = self.provider.get_word_frequency(prefix)
                        if freq >= LEXICALIZED_COMPOUND_MIN_FREQ:
                            errors.append(
                                SyllableError(
                                    text=orig_prefix,
                                    position=pos,
                                    suggestions=[prefix],
                                    confidence=TEXT_DETECTOR_CONFIDENCES[
                                        "missing_diacritic_compound"
                                    ],
                                    error_type=ET_WORD,
                                )
                            )
                            found = True
                            break
                    if found:
                        break

                    # Fallback: whole modified token (single-word case)
                    if self.provider.is_valid_word(modified):
                        freq = self.provider.get_word_frequency(modified)
                        if freq >= LEXICALIZED_COMPOUND_MIN_FREQ:
                            errors.append(
                                SyllableError(
                                    text=token,
                                    position=pos,
                                    suggestions=[modified],
                                    confidence=TEXT_DETECTOR_CONFIDENCES[
                                        "missing_diacritic_compound"
                                    ],
                                    error_type=ET_WORD,
                                )
                            )
                            found = True
                            break

            # Remove generic syllable/word errors at this position
            # that are superseded by our more specific diacritic error.
            if found and existing_at_pos:
                for old in existing_at_pos:
                    if old in errors:
                        errors.remove(old)

    def _detect_unknown_compound_segments(self, text: str, errors: list[Error]) -> None:
        """Detect freq=0 word segments where at least one syllable is not a valid word.

        When the Viterbi word segmenter merges misspelled syllables into a
        compound chunk (e.g. မျန်မာဆာ from မျန်+မာ+ဆာ), the chunk has
        freq=0 and is not in the word table. The WordValidator's merge
        heuristic skips such chunks because most syllable parts are valid
        words individually. This detector catches those cases by checking
        for unknown compound segments with at least one invalid syllable
        part, generating SymSpell suggestions for the whole chunk.
        """
        if not self.provider or not hasattr(self, "segmenter") or not hasattr(self, "symspell"):
            return

        segmenter = self.segmenter
        provider = self.provider
        symspell = getattr(self, "symspell", None)

        try:
            words = segmenter.segment_words(text)
        except Exception:
            return
        cursor = 0

        for word in words:
            if not word or not word.strip():
                continue
            # Find position in text
            pos = text.find(word, cursor)
            if pos < 0:
                continue
            cursor = pos + len(word)

            # Skip tokens that aren't purely Myanmar script — mixed-script
            # tokens (e.g., "မြန်-ဂ" with ASCII hyphen) are not compounds
            if not all("\u1000" <= ch <= "\u109f" for ch in word):
                continue

            # Only check multi-syllable words
            syllables = segmenter.segment_syllables(word)
            if len(syllables) < 2:
                continue

            # Only check words NOT in dictionary with freq=0
            if provider.is_valid_word(word) or provider.get_word_frequency(word) > 0:
                continue

            # Skip if there's already an error at this position
            if any(e.position == pos for e in errors):
                continue

            # Check if at least one syllable part is NOT a valid word
            # (distinguishes genuinely misspelled compounds from segmenter
            # merges of valid words)
            invalid_parts = [s for s in syllables if not provider.is_valid_word(s)]
            if not invalid_parts:
                continue

            # Generate suggestions via SymSpell word-level lookup
            suggestions: list[str] = []
            if symspell is not None:
                from myspellchecker.core.constants import ValidationLevel

                symspell_results = symspell.lookup(
                    word,
                    level=ValidationLevel.WORD.value,
                    max_suggestions=5,
                    use_phonetic=False,
                )
                suggestions = [s.term for s in symspell_results if s.term != word]

            if suggestions:
                errors.append(
                    SyllableError(
                        text=word,
                        position=pos,
                        suggestions=suggestions,
                        confidence=TEXT_DETECTOR_CONFIDENCES.get("unknown_compound_segment", 0.85),
                        error_type=ET_WORD,
                    )
                )

    # ---- Punctuation error detection ----

    _DOUBLE_FULLSTOP_RE = re.compile("။{2,}")
    _ENGLISH_Q_AFTER_MYANMAR_RE = re.compile(r"[\u1000-\u109F]\?")

    def _detect_punctuation_errors(self, text: str, errors: list[Error]) -> None:
        """Detect Myanmar punctuation errors.

        Three detections:
        1. Double ။ (always-on) — duplicate_punctuation
        2. English ? after Myanmar text (always-on) — wrong_punctuation
        3. Missing ။ at text end (conditional) — missing_punctuation
           Only fires when text already contains ။ elsewhere (inconsistency).
        """
        from myspellchecker.grammar.patterns import ends_with_sfp

        existing_positions = get_existing_positions(errors)

        def _register_mixing_at(pos: int) -> bool:
            return any(e.position == pos and e.error_type == ET_REGISTER_MIXING for e in errors)

        # Detection 1: Double ။ (။။ or more)
        for m in self._DOUBLE_FULLSTOP_RE.finditer(text):
            pos = m.start()
            if pos not in existing_positions:
                errors.append(
                    SyllableError(
                        text=m.group(),
                        position=pos,
                        suggestions=["။"],
                        confidence=TEXT_DETECTOR_CONFIDENCES["duplicate_punctuation"],
                        error_type=ET_DUPLICATE_PUNCTUATION,
                    )
                )
                existing_positions.add(pos)

        # Detection 2: English ? after Myanmar text
        for m in self._ENGLISH_Q_AFTER_MYANMAR_RE.finditer(text):
            q_pos = m.start() + 1  # position of the ?
            if q_pos not in existing_positions:
                errors.append(
                    SyllableError(
                        text="?",
                        position=q_pos,
                        suggestions=["။"],
                        confidence=TEXT_DETECTOR_CONFIDENCES["wrong_punctuation"],
                        error_type=ET_WRONG_PUNCTUATION,
                    )
                )
                existing_positions.add(q_pos)

        # Detection 2.5: Missing clause boundary punctuation in no-။ texts.
        # Targets high-impact missing-boundary cases such as:
        #   ...တယ် နောက်မှ ...
        #   ...ပြီး ကျွန်တော် ...
        from myspellchecker.core.detectors.utils import get_tokenized

        tokenized_p = get_tokenized(self, text)
        tokens = tokenized_p.tokens
        if len(tokens) >= 2:
            token_positions = tokenized_p.positions

            boundary_endings = (
                normalize("ပါသည်"),
                normalize("ပါတယ်"),
                normalize("ပါမယ်"),
                normalize("မယ်"),
                normalize("တယ်"),
                normalize("သည်"),
                normalize("ပြီ"),
                normalize("ပါပြီ"),
            )
            temporal_starters = {normalize("နောက်မှ")}
            pronoun_starters = {
                normalize("ငါ"),
                normalize("ကျွန်တော်"),
                normalize("ကျွန်မ"),
                normalize("ကျွန်ုပ်"),
                normalize("နင်"),
            }
            clause_linker = normalize("ပြီး")

            for i in range(len(tokens) - 1):
                current = tokens[i]
                nxt = tokens[i + 1]
                current_pos = token_positions[i]

                if current.endswith("။"):
                    continue

                for ending in boundary_endings:
                    matches_ending = current == ending or (
                        len(current) > len(ending) and current.endswith(ending)
                    )
                    if matches_ending:
                        if nxt in temporal_starters or nxt in pronoun_starters:
                            err_pos = current_pos + len(current) - len(ending)
                            if err_pos not in existing_positions or _register_mixing_at(err_pos):
                                span_text = f"{ending} {nxt}"
                                suggestions = [f"{ending}။ {nxt}"]
                                if nxt == normalize("နောက်မှ"):
                                    suggestions.insert(0, f"{ending}။ {normalize('နောက်')}")
                                # Prefer punctuation repair over weaker register
                                # warnings at the same suffix boundary.
                                errors[:] = [
                                    e
                                    for e in errors
                                    if not (
                                        e.position == err_pos and e.error_type == ET_REGISTER_MIXING
                                    )
                                ]
                                errors.append(
                                    SyllableError(
                                        text=span_text,
                                        position=err_pos,
                                        suggestions=suggestions,
                                        confidence=TEXT_DETECTOR_CONFIDENCES["missing_punctuation"],
                                        error_type=ET_MISSING_PUNCTUATION,
                                    )
                                )
                                existing_positions.add(err_pos)
                        break

                if len(current) > len(clause_linker) and current.endswith(clause_linker):
                    if nxt in pronoun_starters:
                        err_pos = current_pos + len(current) - len(clause_linker)
                        if err_pos not in existing_positions or _register_mixing_at(err_pos):
                            span_text = f"{clause_linker} {nxt}"
                            suggestions = [f"{clause_linker}။ {nxt}"]
                            if nxt == normalize("ကျွန်တော်"):
                                suggestions.insert(0, f"{clause_linker}။ {normalize('ကျွန်')}")
                            errors[:] = [
                                e
                                for e in errors
                                if not (
                                    e.position == err_pos and e.error_type == ET_REGISTER_MIXING
                                )
                            ]
                            errors.append(
                                SyllableError(
                                    text=span_text,
                                    position=err_pos,
                                    suggestions=suggestions,
                                    confidence=TEXT_DETECTOR_CONFIDENCES["missing_punctuation"],
                                    error_type=ET_MISSING_PUNCTUATION,
                                )
                            )
                            existing_positions.add(err_pos)

        # Detection 3: Missing ။ at text end (inconsistency mode)
        # Only trigger when text already contains at least one ။
        if "။" not in text:
            return

        stripped = text.rstrip()
        if not stripped or stripped.endswith("။"):
            return

        result = ends_with_sfp(stripped)
        if result is not None:
            sfp, sfp_pos = result
            # Position is at the SFP — suggest appending ။
            error_pos = sfp_pos
            if error_pos not in existing_positions:
                errors.append(
                    SyllableError(
                        text=sfp,
                        position=error_pos,
                        suggestions=[sfp + "။"],
                        confidence=TEXT_DETECTOR_CONFIDENCES["missing_punctuation"],
                        error_type=ET_MISSING_PUNCTUATION,
                    )
                )

    def _detect_vowel_after_asat(self, text: str, errors: list[Error]) -> None:
        """Detect asat (U+103A) followed by a vowel sign — spurious vowel insertion.

        Example: ကျွန်ုတော် (vowel-u after asat on န) → ကျွန်တော်
        Scans raw text for the pattern, finds word boundaries, and suggests
        the word with the spurious vowel removed.
        """
        asat = "\u103a"

        i = 0
        while i < len(text) - 1:
            if text[i] == asat and text[i + 1] in self._VOWEL_SIGNS:
                # Found asat+vowel — expand to word boundaries
                # Walk backward to find word start (Myanmar consonant or related char)
                start = i
                while start > 0 and text[start - 1] not in (" ", "\u104a", "\u104b"):
                    start -= 1
                # Walk forward to find word end
                end = i + 2  # past the vowel sign
                while end < len(text) and text[end] not in (" ", "\u104a", "\u104b"):
                    end += 1

                word = text[start:end]
                # Remove the spurious vowel (position relative to word)
                vowel_offset = (i + 1) - start
                corrected = word[:vowel_offset] + word[vowel_offset + 1 :]
                # Collapse double asat left behind after vowel removal
                # e.g. ကျွန်ု်တော် → remove ု → ကျွန််တော် → ကျွန်တော်
                corrected = corrected.replace("\u103a\u103a", "\u103a")

                # Always append — position dedup handles conflicts with
                # syllable errors (conf=1.0 + longer text wins).
                if corrected != word:
                    errors.append(
                        SyllableError(
                            text=word,
                            position=start,
                            suggestions=[corrected],
                            confidence=TEXT_DETECTOR_CONFIDENCES["vowel_after_asat"],
                            error_type=ET_VOWEL_AFTER_ASAT,
                        )
                    )
                # Skip past this match
                i = end
            else:
                i += 1

    # Myanmar consonant range for stacking detection
    _CONSONANT_RANGE = range(0x1000, 0x1022)  # U+1000..U+1021

    def _detect_broken_stacking(self, text: str, errors: list[Error]) -> None:
        """Detect asat (U+103A) used instead of virama (U+1039) before stacked consonant.

        Common in Pali/Sanskrit loanwords where users type the visible asat
        instead of the stacking virama.  Examples:
          ဗုဒ်ဓ → ဗုဒ္ဓ  (Buddha)
          ကိစ်စ → ကိစ္စ  (matter)

        Also detects missing kinzi virama:
          သင်ဘော → သင်္ဘော  (ship)  — င + asat needs virama before ဘ
        """
        asat = "\u103a"
        virama = "\u1039"

        occupied = set()
        for e in errors:
            for p in range(e.position, e.position + len(e.text)):
                occupied.add(p)

        i = 0
        while i < len(text) - 2:
            if text[i] != asat:
                i += 1
                continue

            # Must have consonant before and after: C + asat + C
            prev_cp = ord(text[i - 1]) if i > 0 else 0
            next_cp = ord(text[i + 1])
            if prev_cp not in self._CONSONANT_RANGE or next_cp not in self._CONSONANT_RANGE:
                i += 1
                continue

            # Skip if already virama-stacked (C + asat + virama + C = kinzi)
            if i + 2 < len(text) and text[i + 1] == virama:
                i += 2
                continue

            # Expand to word boundaries
            start = i - 1
            while start > 0 and text[start - 1] not in (" ", "\u104a", "\u104b"):
                start -= 1
            end = i + 2
            while end < len(text) and text[end] not in (" ", "\u104a", "\u104b"):
                end += 1

            # Check if position already has an error — if so, we may
            # replace shorter errors (e.g., colloquial "သိပ်" with stacking "သိပ်ပံ")
            has_overlap = any(p in occupied for p in range(start, end))

            # Try progressively narrower substrings around the stacking.
            # The segmenter may have merged a prefix (e.g. "ဒီ" + "ကိစ်စ")
            # so the full word won't be in the dictionary after the virama fix,
            # but a narrower substring (e.g. "ကိစ္စ") will be.
            found = False
            # Try progressively narrower substrings: shrink START and END
            # to handle compounds where the stacking word is a prefix
            # (e.g., "သင်ဘောဆိပ်ကမ်း" contains kinzi "သင်္ဘော" as prefix).
            for sub_start in range(start, i):
                for sub_end in range(end, i + 1, -1):
                    sub = text[sub_start:sub_end]
                    sub_asat = i - sub_start

                    # Skip if the original substring is already a valid word —
                    # e.g. "သင်ခန်းစာ" (lesson) contains င+်+ခ but is correct.
                    if self.provider and self.provider.is_valid_word(sub):
                        continue

                    # Strategy 1: Replace asat with virama (regular stacking)
                    candidate = sub[:sub_asat] + virama + sub[sub_asat + 1 :]
                    if self.provider and self.provider.is_valid_word(candidate):
                        new_err = SyllableError(
                            text=sub,
                            position=sub_start,
                            suggestions=[candidate],
                            confidence=TEXT_DETECTOR_CONFIDENCES["broken_stacking"],
                            error_type=ET_BROKEN_STACKING,
                        )
                        if has_overlap:
                            errors[:] = [
                                e
                                for e in errors
                                if not (
                                    e.position >= sub_start
                                    and e.position + len(e.text or "") <= sub_start + len(sub)
                                )
                            ]
                        errors.append(new_err)
                        for p in range(sub_start, sub_start + len(sub)):
                            occupied.add(p)
                        found = True
                        break

                    # Strategy 2: Insert virama after asat (kinzi: င + ် + C)
                    if prev_cp == 0x1004:  # င (nga)
                        kinzi_candidate = sub[: sub_asat + 1] + virama + sub[sub_asat + 1 :]
                        if self.provider and self.provider.is_valid_word(kinzi_candidate):
                            new_err = SyllableError(
                                text=sub,
                                position=sub_start,
                                suggestions=[kinzi_candidate],
                                confidence=TEXT_DETECTOR_CONFIDENCES["broken_stacking"],
                                error_type=ET_BROKEN_STACKING,
                            )
                            if has_overlap:
                                errors[:] = [
                                    e
                                    for e in errors
                                    if not (
                                        e.position >= sub_start
                                        and e.position + len(e.text or "") <= sub_start + len(sub)
                                    )
                                ]
                            errors.append(new_err)
                            for p in range(sub_start, sub_start + len(sub)):
                                occupied.add(p)
                            found = True
                            break
                if found:
                    break

            i = end if found else i + 1

    # ------------------------------------------------------------------ #
    # Missing Pali/Sanskrit stacking detection                            #
    # ------------------------------------------------------------------ #

    # Known Pali/Sanskrit broken stacking forms.
    # Maps: broken (unstacked) form → correct (stacked) form.
    #
    # Three categories:
    # 1. Missing virama between adjacent consonants: C1+C2 → C1+္+C2
    #    e.g. စကန့် → စက္ကန့်  (the virama+consonant is entirely absent)
    # 2. Valid-but-wrong unstacked variant vs correct stacked form:
    #    e.g. ဗုဒဘာသာ (freq=89) → ဗုဒ္ဓဘာသာ (freq=28375)
    # 3. Over-stacking: virama used where asat is correct:
    #    e.g. လက္ခံချက် → လက်ခံချက်  (already handled by _detect_broken_stacking)
    #
    # Keys must be post-normalization forms (wrapped via _norm_dict).
    _PALI_STACKING_CORRECTIONS: dict[str, str | list[str]] = _norm_dict(
        {
            # --- Category 1: Missing virama (consonant not stacked) ---
            # စကန့် → စက္ကန့် (second/moment) — missing က္က stacking
            "စကန့်": "စက္ကန့်",
            # --- Category 2: Valid-but-wrong unstacked Pali/Sanskrit ---
            # ဗုဒဘာသာ → ဗုဒ္ဓဘာသာ (Buddhism) — ဒ should stack as ဒ္ဓ
            "ဗုဒဘာသာ": "ဗုဒ္ဓဘာသာ",
            # သတတဝါ → သတ္တဝါ (creature/being) — unstacked form
            "သတတဝါ": "သတ္တဝါ",
            # --- Category 3: Asat used instead of virama (common typo) ---
            # These supplement _detect_broken_stacking for cases where the
            # broken form is a valid word in the DB (freq > 0), causing the
            # generic detector to skip it.
            # သတ်တဝါ → သတ္တဝါ (creature/being) — valid word with asat
            "သတ်တဝါ": "သတ္တဝါ",
        }
    )

    def _detect_missing_stacking(self, text: str, errors: list[Error]) -> None:
        """Detect broken Pali/Sanskrit stacking via known-form dictionary.

        Handles cases that ``_detect_broken_stacking`` cannot:
        - Missing virama entirely (no asat trigger): စကန့် → စက္ကန့်
        - Valid-but-wrong unstacked form in DB: ဗုဒဘာသာ → ဗုဒ္ဓဘာသာ
        - Asat-for-virama where broken form is valid: သတ်တဝါ → သတ္တဝါ

        Uses a dictionary of known broken → correct mappings for common
        Pali/Sanskrit loanwords.
        """
        occupied: set[int] = set()
        for e in errors:
            for p in range(e.position, e.position + len(e.text or "")):
                occupied.add(p)

        for broken, correction in self._PALI_STACKING_CORRECTIONS.items():
            corrections = correction if isinstance(correction, list) else [correction]
            for idx, end in iter_occurrences(text, broken):
                # Skip if fully covered by an existing preserved-type error
                if all(p in occupied for p in range(idx, end)):
                    continue

                new_err = SyllableError(
                    text=broken,
                    position=idx,
                    suggestions=list(corrections),
                    confidence=TEXT_DETECTOR_CONFIDENCES.get("missing_stacking", 0.90),
                    error_type=ET_BROKEN_STACKING,
                )

                # If position already has an error, try to replace it
                has_overlap = any(p in occupied for p in range(idx, end))
                if has_overlap:
                    # Remove any shorter/weaker errors fully inside our span
                    errors[:] = [
                        e
                        for e in errors
                        if not (e.position >= idx and e.position + len(e.text or "") <= end)
                    ]

                errors.append(new_err)
                for p in range(idx, end):
                    occupied.add(p)
