"""Register mixing detection mixin for SentenceDetectorsMixin.

Provides ``_detect_register_mixing`` and its associated class-level
data constants (register endings, pronouns, conversion maps).

Data is loaded from ``rules/register.yaml`` at module level, with
fallback to hardcoded defaults if the YAML file is missing or invalid.

Extracted from ``sentence_detectors.py`` to reduce file size while
preserving the exact same method signatures and behaviour.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from myspellchecker.core.constants import ET_REGISTER_MIXING, LEXICALIZED_COMPOUND_MIN_FREQ

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

# ── Hardcoded defaults (fallback when YAML is unavailable) ──

_DEFAULT_LITERARY_PARTICLES: frozenset[str] = _norm_set(
    {
        "၏",  # possessive (literary)
        "တွင်",  # locative (literary)
        "သော",  # attributive (literary)
        "ဖြင့်",  # instrumental (literary)
        "၌",  # locative (literary, archaic)
        "အား",  # dative (literary, when postposition)
        "သော်လည်း",  # concessive conjunction (literary "although")
    }
)

_DEFAULT_LITERARY_ADVERBS: frozenset[str] = _norm_set(
    {
        "ယနေ့",  # today (formal); colloquial = ဒီနေ့
        "ယခု",  # now (formal); colloquial = အခု
        "ယခင်",  # before (formal); colloquial = အရင်
    }
)

_DEFAULT_LITERARY_DETERMINERS: frozenset[str] = _norm_set(
    {
        "ယင်း",  # that/those (formal demonstrative)
        "ထို",  # that (literary demonstrative)
        "ဤ",  # this (literary demonstrative)
        "ယင်းသို့",  # thus/in that way (literary)
    }
)

_DEFAULT_FIRST_PERSON_FORMS: frozenset[str] = _norm_set(
    {
        "ကျွန်တော်",
        "ကျွန်မ",
        "ကျွန်ုပ်",
        "ကျွန်တော်တို့",
        "ကျွန်မတို့",
        "ငါ",
        "ငါတို့",
    }
)

_DEFAULT_COLLOQUIAL_PRONOUNS: frozenset[str] = _norm_set({"ငါ", "ငါတို့", "နင်", "နင်တို့"})

_DEFAULT_COLLOQUIAL_DISCOURSE_PARTICLES: frozenset[str] = _norm_set(
    {"နော်", "ဟ", "ဟေ့", "ဒါကြောင့်", "ပေမဲ့", "ဒါပေမဲ့"}
)

_DEFAULT_FORMAL_PRONOUNS: frozenset[str] = _norm_set({"ကျွန်တော်", "ကျွန်မ", "ကျွန်ုပ်"})

_DEFAULT_FORMAL_CONTEXT_WORDS: frozenset[str] = _norm_set({"အစီရင်ခံစာ", "အစည်းအဝေး", "လွှတ်တော်"})

_DEFAULT_FORMAL_CONTEXT_STEMS: frozenset[str] = _norm_set({"တင်ပြ", "ကျင်းပ"})

_DEFAULT_FORMAL_FULL_FORM_STEMS: frozenset[str] = _norm_set({"တင်ပြ"})

# ── YAML loading ──

_YAML_PATH = Path(__file__).resolve().parent.parent.parent.parent / "rules" / "register.yaml"


def _load_register_data() -> tuple[
    frozenset[str],  # literary_particles
    frozenset[str],  # literary_adverbs
    frozenset[str],  # literary_determiners
    frozenset[str],  # first_person_forms
    frozenset[str],  # colloquial_pronouns
    frozenset[str],  # colloquial_discourse_particles
    frozenset[str],  # formal_pronouns
    frozenset[str],  # formal_context_words
    frozenset[str],  # formal_context_stems
    frozenset[str],  # formal_full_form_stems
]:
    """Load register mixing data from YAML with fallback to defaults.

    Returns:
        Tuple of frozensets for each register data category,
        all normalized via _norm_set.
    """
    defaults = (
        _DEFAULT_LITERARY_PARTICLES,
        _DEFAULT_LITERARY_ADVERBS,
        _DEFAULT_LITERARY_DETERMINERS,
        _DEFAULT_FIRST_PERSON_FORMS,
        _DEFAULT_COLLOQUIAL_PRONOUNS,
        _DEFAULT_COLLOQUIAL_DISCOURSE_PARTICLES,
        _DEFAULT_FORMAL_PRONOUNS,
        _DEFAULT_FORMAL_CONTEXT_WORDS,
        _DEFAULT_FORMAL_CONTEXT_STEMS,
        _DEFAULT_FORMAL_FULL_FORM_STEMS,
    )

    if not _YAML_PATH.exists():
        logger.debug(
            "Register YAML not found at %s, using defaults",
            _YAML_PATH,
        )
        return defaults

    try:
        import yaml

        with open(_YAML_PATH, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not data or not isinstance(data, dict):
            logger.warning("Register YAML empty or invalid, using defaults")
            return defaults

        # -- Literary markers --
        lit = data.get("literary_markers", {})
        if isinstance(lit, dict):
            literary_particles = (
                _norm_set(lit["particles"])
                if isinstance(lit.get("particles"), list) and lit["particles"]
                else _DEFAULT_LITERARY_PARTICLES
            )
            literary_adverbs = (
                _norm_set(lit["adverbs"])
                if isinstance(lit.get("adverbs"), list) and lit["adverbs"]
                else _DEFAULT_LITERARY_ADVERBS
            )
            literary_determiners = (
                _norm_set(lit["determiners"])
                if isinstance(lit.get("determiners"), list) and lit["determiners"]
                else _DEFAULT_LITERARY_DETERMINERS
            )
        else:
            literary_particles = _DEFAULT_LITERARY_PARTICLES
            literary_adverbs = _DEFAULT_LITERARY_ADVERBS
            literary_determiners = _DEFAULT_LITERARY_DETERMINERS

        # -- Pronouns --
        pron = data.get("pronouns", {})
        if isinstance(pron, dict):
            colloquial_pronouns = (
                _norm_set(pron["colloquial"])
                if isinstance(pron.get("colloquial"), list) and pron["colloquial"]
                else _DEFAULT_COLLOQUIAL_PRONOUNS
            )
            formal_pronouns = (
                _norm_set(pron["formal"])
                if isinstance(pron.get("formal"), list) and pron["formal"]
                else _DEFAULT_FORMAL_PRONOUNS
            )
            first_person_all = (
                _norm_set(pron["first_person_all"])
                if isinstance(pron.get("first_person_all"), list) and pron["first_person_all"]
                else _DEFAULT_FIRST_PERSON_FORMS
            )
            discourse_particles = (
                _norm_set(pron["discourse_particles"])
                if isinstance(pron.get("discourse_particles"), list) and pron["discourse_particles"]
                else _DEFAULT_COLLOQUIAL_DISCOURSE_PARTICLES
            )
        else:
            colloquial_pronouns = _DEFAULT_COLLOQUIAL_PRONOUNS
            formal_pronouns = _DEFAULT_FORMAL_PRONOUNS
            first_person_all = _DEFAULT_FIRST_PERSON_FORMS
            discourse_particles = _DEFAULT_COLLOQUIAL_DISCOURSE_PARTICLES

        # -- Formal context --
        fctx = data.get("formal_context", {})
        if isinstance(fctx, dict):
            formal_context_words = (
                _norm_set(fctx["words"])
                if isinstance(fctx.get("words"), list) and fctx["words"]
                else _DEFAULT_FORMAL_CONTEXT_WORDS
            )
            formal_context_stems = (
                _norm_set(fctx["stems"])
                if isinstance(fctx.get("stems"), list) and fctx["stems"]
                else _DEFAULT_FORMAL_CONTEXT_STEMS
            )
            formal_full_form_stems = (
                _norm_set(fctx["full_form_stems"])
                if isinstance(fctx.get("full_form_stems"), list) and fctx["full_form_stems"]
                else _DEFAULT_FORMAL_FULL_FORM_STEMS
            )
        else:
            formal_context_words = _DEFAULT_FORMAL_CONTEXT_WORDS
            formal_context_stems = _DEFAULT_FORMAL_CONTEXT_STEMS
            formal_full_form_stems = _DEFAULT_FORMAL_FULL_FORM_STEMS

        logger.debug(
            "Loaded register data from YAML: "
            "%d literary_particles, %d literary_adverbs, %d literary_determiners, "
            "%d colloquial_pronouns, %d formal_pronouns",
            len(literary_particles),
            len(literary_adverbs),
            len(literary_determiners),
            len(colloquial_pronouns),
            len(formal_pronouns),
        )
        return (
            literary_particles,
            literary_adverbs,
            literary_determiners,
            first_person_all,
            colloquial_pronouns,
            discourse_particles,
            formal_pronouns,
            formal_context_words,
            formal_context_stems,
            formal_full_form_stems,
        )

    except Exception:
        logger.warning(
            "Failed to load register YAML, using defaults",
            exc_info=True,
        )
        return defaults


# Load at module level (once, at import time)
(
    _LITERARY_PARTICLES,
    _LITERARY_ADVERBS,
    _LITERARY_DETERMINERS,
    _FIRST_PERSON_FORMS,
    _COLLOQUIAL_PRONOUNS,
    _COLLOQUIAL_DISCOURSE_PARTICLES,
    _FORMAL_PRONOUNS,
    _FORMAL_CONTEXT_WORDS,
    _FORMAL_CONTEXT_STEMS,
    _FORMAL_FULL_FORM_STEMS,
) = _load_register_data()


class RegisterMixingMixin:
    """Mixin providing register mixing detection and its data constants.

    Detects mixed formal/colloquial register across the entire text.
    """

    # --- Type stubs for attributes provided by SpellChecker or sibling mixins ---
    provider: "DictionaryProvider"
    _FORMAL_ENDINGS: frozenset[str]
    _COLLOQUIAL_ENDINGS: frozenset[str]
    _POLITE_ENDINGS: frozenset[str]
    _FORMAL_ENDINGS_WITH_STRIPPED: frozenset[str]
    _POLITE_ENDINGS_WITH_STRIPPED: frozenset[str]
    _COLLOQUIAL_ENDINGS_WITH_STRIPPED: frozenset[str]

    # ----- Register pronoun sets -----
    # Loaded from rules/register.yaml at module level (with hardcoded fallbacks).
    # Colloquial pronouns (first-person) used exclusively in informal register.
    # Note: ကျွန်တော်/ကျွန်မ are NOT included as formal — they're used across
    # both registers in practice (e.g., "ကျွန်တော် ကျန်းမာပါတယ်" is natural).
    # Only ငါ is strictly informal and signals register mismatch.
    _COLLOQUIAL_PRONOUNS: frozenset[str] = _COLLOQUIAL_PRONOUNS

    # Discourse particles that are exclusively informal/colloquial.
    # Used in register mixing detector to detect formal+colloquial clash.
    _COLLOQUIAL_DISCOURSE_PARTICLES: frozenset[str] = _COLLOQUIAL_DISCOURSE_PARTICLES
    # Formal first-person pronouns and official-reporting lexical cues.
    _FORMAL_PRONOUNS: frozenset[str] = _FORMAL_PRONOUNS
    _FORMAL_CONTEXT_WORDS: frozenset[str] = _FORMAL_CONTEXT_WORDS
    _FORMAL_CONTEXT_STEMS: frozenset[str] = _FORMAL_CONTEXT_STEMS
    # Subset of formal stems that should trigger full-token rewrite
    # (e.g., "တင်ပြတယ်" -> "တင်ပြပါသည်"), not suffix-only replacement.
    _FORMAL_FULL_FORM_STEMS: frozenset[str] = _FORMAL_FULL_FORM_STEMS

    def _get_word_register(self, word: str) -> str | None:
        """Get register tag for a word from DB (if available).

        Returns 'formal', 'informal', 'neutral', or None if not in DB.
        Provides data-driven register signal for content words beyond
        the closed-class particle/pronoun inventories.
        """
        if hasattr(self.provider, "get_register_tag"):
            return self.provider.get_register_tag(word)
        return None

    def _detect_register_mixing(self, text: str, errors: list[Error]) -> None:
        """Detect mixed formal/colloquial register across the entire text.

        Myanmar sentence-final particles are the primary register markers.
        Mixing formal endings (သည်, ပါသည်) with colloquial endings
        (တယ်, ပါတယ်) in the same text is a register consistency error.

        This runs on the full text (not per-sentence) because the sentence
        segmenter splits at sentence-final particles, making per-sentence
        checks unable to detect cross-sentence register mixing.
        """
        from myspellchecker.core.detectors.utils import get_tokenized

        tokenized = get_tokenized(self, text)
        if len(tokenized) < 2:
            return

        tokens = tokenized.tokens

        # (token_pos, token, matched_ending) — store the ending for suffix position
        # Suffix positions: sentence-final particles (သည်, တယ်, etc.)
        formal_suffix_pos: list[tuple[int, str, str]] = []
        polite_suffix_pos: list[tuple[int, str, str]] = []
        colloquial_suffix_pos: list[tuple[int, str, str]] = []
        # Pronoun positions: strictly informal pronouns (ငါ, ငါတို့)
        colloquial_pronoun_pos: list[tuple[int, str, str]] = []

        for span in tokenized:
            token = span.text
            pos = span.position

            # Check colloquial pronouns (exact match or possessive suffix match).
            # ငါ/နင် are strictly informal and signal register mismatch
            # when paired with formal endings.
            # Suffix match: ငါ့ (possessive dot-below), ငါ့ကို etc.
            # Guard: ငါး (five) starts with ငါ but is NOT a pronoun.
            # Only accept suffix if next char is dot-below ့ (U+1037, possessive)
            # or asat ် (U+103A) — these form possessive/inflected pronoun forms.
            # Visarga း (U+1038) and other diacritics form different words.
            matched_pronoun = None
            _PRONOUN_SUFFIX_CHARS = ("\u1037", "\u103a")  # dot-below, asat
            for p in self._COLLOQUIAL_PRONOUNS:
                if token == p:
                    matched_pronoun = p
                    break
                if token.startswith(p) and len(token) > len(p):
                    next_ch = token[len(p)]
                    if next_ch in _PRONOUN_SUFFIX_CHARS:
                        matched_pronoun = p
                        break
            if matched_pronoun:
                colloquial_pronoun_pos.append((pos, token, matched_pronoun))
                continue

            # Check colloquial discourse particles (exact match only).
            # နော် (tag question) is informal and signals register mismatch
            # when paired with formal endings like သည်/ပါသည်.
            if token in self._COLLOQUIAL_DISCOURSE_PARTICLES:
                colloquial_pronoun_pos.append((pos, token, token))
                continue

            # Check exact match or suffix match (longest first)
            # Includes asat-stripped variants for missing-asat typos
            # Skip suffix matches for lexicalized compound nouns (e.g.,
            # တပ်မတော်သည် freq=6K where သည် is part of the noun).
            # Low-frequency entries (<1000) are segmentation artifacts
            # and should still undergo register suffix detection.
            is_lexicalized = (
                len(token) > 3
                and hasattr(self.provider, "get_word_frequency")
                and self.provider.get_word_frequency(token) >= LEXICALIZED_COMPOUND_MIN_FREQ
            )
            # Sentence-final tokens always undergo SFP suffix matching
            # regardless of lexicalization (e.g., နေတယ် at end).
            is_final_token = token == tokens[-1]
            for endings, dest in [
                (self._FORMAL_ENDINGS_WITH_STRIPPED, formal_suffix_pos),
                (self._POLITE_ENDINGS_WITH_STRIPPED, polite_suffix_pos),
                (self._COLLOQUIAL_ENDINGS_WITH_STRIPPED, colloquial_suffix_pos),
            ]:
                for ending in sorted(endings, key=len, reverse=True):
                    if token == ending:
                        dest.append((pos, token, ending))
                        break
                    if (
                        (not is_lexicalized or is_final_token)
                        and len(ending) < len(token)
                        and token.endswith(ending)
                    ):
                        dest.append((pos, token, ending))
                        break

        # Literary particles may be suffixed to words (e.g., ဆရာ၏, သားတွင်)
        # so use suffix matching, not just exact match.
        literary_count = 0
        for t in tokens:
            if t in _LITERARY_PARTICLES:
                literary_count += 1
            else:
                for lp in _LITERARY_PARTICLES:
                    if t.endswith(lp):
                        literary_count += 1
                        break
        literary_adverb_count = sum(1 for t in tokens if t in _LITERARY_ADVERBS)
        # Prefix-match literary determiners: ယင်း, ထို, ဤ often appear as
        # prefixes in compound tokens (e.g., ယင်းဥပဒေ, ထိုနေ့).
        literary_determiner_count = sum(
            1
            for t in tokens
            if t in _LITERARY_DETERMINERS
            or any(t.startswith(d) and len(t) > len(d) for d in _LITERARY_DETERMINERS)
        )
        formal_pronoun_count = sum(1 for t in tokens if t in self._FORMAL_PRONOUNS)
        formal_context_count = 0
        for t in tokens:
            if t in self._FORMAL_CONTEXT_WORDS:
                formal_context_count += 1
                continue
            for stem in self._FORMAL_CONTEXT_STEMS:
                if t == stem or (len(t) > len(stem) and t.startswith(stem)):
                    formal_context_count += 1
                    break
            # Note: DB register_tags are available via _get_word_register()
            # but not used for formal_context_count — that counter is tuned
            # for high-confidence hardcoded keywords. DB tags are a weaker
            # signal better suited for future confidence-weighted scoring.

        # Both registers must be present (from any signal) for mixing
        has_formal = len(formal_suffix_pos) > 0
        has_polite = len(polite_suffix_pos) > 0
        # Literary particles are strong formal signals when 2+ appear.
        # Literary adverbs alone (e.g., "ယနေ့") are too weak as a formal signal
        # and frequently appear in colloquial sentences.
        # Literary particles are strong formal signals:
        # - 3+ literary particles: standalone formal signal (no SFP needed)
        # - 2 literary particles: need formal SFP corroboration (avoids FP
        #   when only locatives like တွင်/၌ appear in colloquial text)
        # Discourse particles (ပေမဲ့, ဟေ့, etc.) are strong colloquial signals
        # that lower the literary threshold needed for formal detection.
        has_discourse_particle = len(colloquial_pronoun_pos) > 0 and any(
            token in self._COLLOQUIAL_DISCOURSE_PARTICLES for _, token, _ in colloquial_pronoun_pos
        )
        has_literary_formal = (
            literary_count >= 3
            or (literary_count >= 2 and has_formal)
            or (literary_count >= 1 and has_discourse_particle)
        )
        has_contextual_formal = (
            formal_context_count >= 2
            or (formal_context_count >= 1 and formal_pronoun_count >= 1)
            or literary_determiner_count >= 1
        )
        has_first_person = any(t in _FIRST_PERSON_FORMS for t in tokens)
        has_colloquial = len(colloquial_suffix_pos) > 0 or len(colloquial_pronoun_pos) > 0

        # Polite + Casual without formal = acceptable, no mixing error
        if not (has_formal or has_literary_formal or has_contextual_formal):
            return  # No formal signal, no mixing to detect

        # Must have some non-formal signal to flag
        if not (has_colloquial or has_polite):
            return  # No mixing

        existing_positions = get_existing_positions(errors)

        # Register conversion maps for suggestions (list of alternatives)
        _formal_to_colloquial: dict[str, list[str]] = {
            "ပါသည်": ["ပါတယ်"],
            "သည်": ["တယ်"],
            "ပါမည်": ["ပါမယ်"],
            "မည်": ["မယ်"],
        }
        # Suggest both plain and polite formal equivalents
        _colloquial_to_formal: dict[str, list[str]] = {
            "ပါတယ်": ["ပါသည်"],
            "တယ်": ["ပါသည်", "သည်"],
            "ပါမယ်": ["ပါမည်"],
            "မယ်": ["မည်", "ပါမည်"],
            # Pronouns
            "ငါ": ["ကျွန်တော်"],
            "ငါတို့": ["ကျွန်တော်တို့"],
            "နင်": ["ကျွန်မ", "ကျွန်တော်", "ခင်ဗျား"],
            "နင်တို့": ["ခင်ဗျားတို့"],
            # Discourse particles
            "နော်": ["ပါသည်", "ပါ"],
            "ဟ": ["ပါ", "ရှင်"],
            "ဟေ့": [],
            "ဒါကြောင့်": ["ထို့ကြောင့်"],
            "ပေမဲ့": ["သို့သော်"],
            "ဒါပေမဲ့": ["သို့သော်"],
        }

        def _add_register_errors(
            positions: list[tuple[int, str, str]],
            is_formal: bool,
            confidence_override: float | None = None,
        ) -> None:
            base_confidence = (
                confidence_override
                if confidence_override is not None
                else TEXT_DETECTOR_CONFIDENCES["register_mixing"]
            )
            convert = _formal_to_colloquial if is_formal else _colloquial_to_formal
            full_form_suffix_map: dict[str, list[str]] = {
                normalize("တယ်"): [normalize("ပါသည်"), normalize("သည်")],
                normalize("ပါတယ်"): [normalize("ပါသည်")],
                normalize("မယ်"): [normalize("မည်"), normalize("ပါမည်")],
                normalize("ပါမယ်"): [normalize("ပါမည်")],
            }
            short_colloquial_suffixes = {
                normalize("တယ်"),
                normalize("ပါတယ်"),
            }

            def _should_emit_full_form(token_text: str, matched_ending: str) -> bool:
                if is_formal or matched_ending not in full_form_suffix_map:
                    return False
                if len(token_text) <= len(matched_ending):
                    return False

                verb_stem = token_text[: -len(matched_ending)]
                if any(
                    verb_stem == stem or verb_stem.startswith(stem)
                    for stem in self._FORMAL_FULL_FORM_STEMS
                ):
                    return True

                return matched_ending in short_colloquial_suffixes and len(verb_stem) <= 4

            for token_pos, token, ending in positions:
                suffix_pos = token_pos + len(token) - len(ending)
                if _should_emit_full_form(token, ending):
                    if token_pos not in existing_positions:
                        verb_stem = token[: -len(ending)]
                        full_suggestions = [
                            verb_stem + suffix for suffix in full_form_suffix_map[ending]
                        ]
                        errors.append(
                            SyllableError(
                                text=token,
                                position=token_pos,
                                suggestions=full_suggestions,
                                confidence=base_confidence,
                                error_type=ET_REGISTER_MIXING,
                            )
                        )
                    continue

                if suffix_pos not in existing_positions:
                    suggestions = convert.get(ending, [])
                    # In strongly literary formal context without first-person
                    # self-reference, plain formal "သည်" is preferred over polite
                    # "ပါသည်" for colloquial "တယ်" conversion.
                    if (
                        not is_formal
                        and ending == normalize("တယ်")
                        and has_literary_formal
                        and not has_first_person
                    ):
                        normalized = [normalize(s) for s in suggestions]
                        if normalize("ပါသည်") in normalized and normalize("သည်") in normalized:
                            suggestions = ["သည်", "ပါသည်"]
                    errors.append(
                        SyllableError(
                            text=ending,
                            position=suffix_pos,
                            suggestions=list(suggestions),
                            confidence=base_confidence,
                            error_type=ET_REGISTER_MIXING,
                        )
                    )

        # Determine dominant register using total counts (suffix + pronoun).
        # Flag only the minority side to avoid double-flagging.
        # When counts are equal, default to formal (standard written Myanmar).
        # Literary markers always push toward formal dominance.
        total_formal = (
            len(formal_suffix_pos)
            + literary_count
            + literary_adverb_count
            + literary_determiner_count
            + formal_pronoun_count
            + formal_context_count
        )
        total_colloquial = len(colloquial_suffix_pos) + len(colloquial_pronoun_pos)

        # Formal + Polite (no casual) = lower confidence warning.
        # Formal + Casual = full confidence error (regardless of polite).
        # Polite + Casual = acceptable (already returned above).
        _POLITE_CONFIDENCE_FACTOR = 0.76
        polite_only_mixing = has_polite and not has_colloquial
        reduced_confidence: float | None = (
            TEXT_DETECTOR_CONFIDENCES["register_mixing"] * _POLITE_CONFIDENCE_FACTOR
            if polite_only_mixing
            else None
        )

        if total_formal >= total_colloquial:
            # Formal dominant — flag colloquial suffixes and pronouns
            _add_register_errors(colloquial_suffix_pos, is_formal=False)
            _add_register_errors(colloquial_pronoun_pos, is_formal=False)
            # Formal + Polite only: flag polite positions with reduced confidence
            if polite_only_mixing:
                _add_register_errors(
                    polite_suffix_pos, is_formal=False, confidence_override=reduced_confidence
                )
        else:
            # Colloquial dominant — flag formal suffixes
            _add_register_errors(formal_suffix_pos, is_formal=True)
            # Polite-only mixing with colloquial dominant shouldn't happen
            # (polite_only_mixing implies no colloquial), but guard anyway
            if polite_only_mixing:
                _add_register_errors(
                    formal_suffix_pos,
                    is_formal=True,
                    confidence_override=reduced_confidence,
                )
