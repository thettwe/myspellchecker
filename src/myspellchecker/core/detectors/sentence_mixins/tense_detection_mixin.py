"""Tense mismatch detection mixin for SentenceDetectorsMixin.

Provides ``_detect_tense_mismatch`` and its associated class-level
data constants (temporal adverbs, aspect markers, particle corrections).

Data is loaded from ``rules/tense_markers.yaml`` at module level, with
fallback to hardcoded defaults if the YAML file is missing or invalid.

Extracted from ``sentence_detectors.py`` to reduce file size while
preserving the exact same method signatures and behaviour.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from myspellchecker.core.constants import ET_ASPECT_ADVERB_CONFLICT, ET_TENSE_MISMATCH

if TYPE_CHECKING:
    from myspellchecker.providers.base import DictionaryProvider
from myspellchecker.core.detector_data import (
    TEXT_DETECTOR_CONFIDENCES,
)
from myspellchecker.core.detector_data import (
    norm_dict as _norm_dict,
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

_DEFAULT_FUTURE_ADVERBS: frozenset[str] = _norm_set(
    {
        "မနက်ဖြန်",
        "မနက်ဖန်",  # common misspelling (missing ya-yit)
        "နောက်နေ့",
        "နောက်ရက်",
        "နောက်အပတ်",
        "နောက်လ",
        "နောက်နှစ်",
        "မကြာခင်",
        "မကြာမီ",
        "နောင်",
        "သန်ဖြန်",
        "လာမယ့်နှစ်",
        "လာမယ့်လ",
        "လာမယ့်အပတ်",
        "လာမည့်နှစ်",
        "လာမည့်လ",
        "လာမည့်အပတ်",
    }
)

_DEFAULT_PAST_ADVERBS: frozenset[str] = _norm_set(
    {
        "မနေ့က",
        "မနေ့",
        "တစ်နေ့က",
        "အရင်",
        "အရင်က",
        "အရင်တုန်းက",
        "တုန်းက",
        "ယခင်",
        "ယခင်က",
        "တမြန်နေ့က",
        "ယနေ့နံနက်က",
        "ယနေ့က",
        "ပြီးခဲ့သော",
        "ပြီးခဲ့သည့်",
        "ပြီးခဲ့တဲ့",
        "လွန်ခဲ့သော",
        "လွန်ခဲ့သည့်",
        "လွန်ခဲ့တဲ့",
    }
)

_DEFAULT_HABITUAL_ADVERBS: frozenset[str] = _norm_set(
    {
        "နေ့တိုင်း",
        "အမြဲ",
        "အမြဲတမ်း",
        "ပုံမှန်",
        "မကြာခဏ",
        "နှစ်တိုင်း",
        "လတိုင်း",
        "အပတ်တိုင်း",
        "မနက်တိုင်း",
        "ညတိုင်း",
        "ညနေတိုင်း",
    }
)

_DEFAULT_PRESENT_ADVERBS: frozenset[str] = _norm_set(
    {
        "အခု",
        "အခုတော့",
        "ယခု",
        "ယခုတော့",
        "လက်ရှိ",
        "လက်ရှိမှာ",
        "လက်ရှိတွင်",
    }
)

_DEFAULT_PAST_PRESENT_TO_FUTURE: dict[str, str | list[str]] = _norm_dict(
    {
        "တယ်": "မယ်",
        "ပါတယ်": "ပါမယ်",
        "သည်": "မည်",
        "ပါသည်": "ပါမည်",
    }
)

_DEFAULT_FUTURE_TO_PAST_PRESENT: dict[str, str | list[str]] = _norm_dict(
    {
        "မယ်": "တယ်",
        "ပါမယ်": "ပါတယ်",
        "မည်": "သည်",
        "ပါမည်": "ပါသည်",
        "လိမ့်မည်": "သည်",
        "လိမ့်မယ်": "တယ်",
    }
)

# ── YAML loading ──

_YAML_PATH = Path(__file__).resolve().parent.parent.parent.parent / "rules" / "tense_markers.yaml"


def _load_tense_markers() -> tuple[
    frozenset[str],
    frozenset[str],
    frozenset[str],
    frozenset[str],
    dict[str, str | list[str]],
    dict[str, str | list[str]],
]:
    """Load tense marker data from YAML with fallback.

    Returns:
        Tuple of (future_adverbs, past_adverbs, habitual_adverbs,
        present_adverbs, past_present_to_future, future_to_past_present),
        all normalized via _norm_set/_norm_dict.
    """
    if not _YAML_PATH.exists():
        logger.debug(
            "Tense markers YAML not found at %s, using defaults",
            _YAML_PATH,
        )
        return (
            _DEFAULT_FUTURE_ADVERBS,
            _DEFAULT_PAST_ADVERBS,
            _DEFAULT_HABITUAL_ADVERBS,
            _DEFAULT_PRESENT_ADVERBS,
            _DEFAULT_PAST_PRESENT_TO_FUTURE,
            _DEFAULT_FUTURE_TO_PAST_PRESENT,
        )

    try:
        import yaml  # type: ignore[import-untyped]

        with open(_YAML_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not data or not isinstance(data, dict):
            logger.warning("Tense markers YAML empty or invalid, using defaults")
            return (
                _DEFAULT_FUTURE_ADVERBS,
                _DEFAULT_PAST_ADVERBS,
                _DEFAULT_HABITUAL_ADVERBS,
                _DEFAULT_PRESENT_ADVERBS,
                _DEFAULT_PAST_PRESENT_TO_FUTURE,
                _DEFAULT_FUTURE_TO_PAST_PRESENT,
            )

        # -- Temporal adverbs --
        raw_adverbs = data.get("temporal_adverbs", {})
        if isinstance(raw_adverbs, dict):
            raw_future = raw_adverbs.get("future", [])
            future = (
                _norm_set(raw_future)
                if isinstance(raw_future, list) and raw_future
                else _DEFAULT_FUTURE_ADVERBS
            )

            raw_past = raw_adverbs.get("past", [])
            past = (
                _norm_set(raw_past)
                if isinstance(raw_past, list) and raw_past
                else _DEFAULT_PAST_ADVERBS
            )

            raw_habitual = raw_adverbs.get("habitual", [])
            habitual = (
                _norm_set(raw_habitual)
                if isinstance(raw_habitual, list) and raw_habitual
                else _DEFAULT_HABITUAL_ADVERBS
            )

            raw_present = raw_adverbs.get("present", [])
            present = (
                _norm_set(raw_present)
                if isinstance(raw_present, list) and raw_present
                else _DEFAULT_PRESENT_ADVERBS
            )
        else:
            future = _DEFAULT_FUTURE_ADVERBS
            past = _DEFAULT_PAST_ADVERBS
            habitual = _DEFAULT_HABITUAL_ADVERBS
            present = _DEFAULT_PRESENT_ADVERBS

        # -- SFP corrections --
        raw_sfp = data.get("sfp_corrections", {})
        if isinstance(raw_sfp, dict):
            raw_pp2f = raw_sfp.get("past_present_to_future", {})
            pp2f = (
                _norm_dict(raw_pp2f)
                if isinstance(raw_pp2f, dict) and raw_pp2f
                else _DEFAULT_PAST_PRESENT_TO_FUTURE
            )

            raw_f2pp = raw_sfp.get("future_to_past_present", {})
            f2pp = (
                _norm_dict(raw_f2pp)
                if isinstance(raw_f2pp, dict) and raw_f2pp
                else _DEFAULT_FUTURE_TO_PAST_PRESENT
            )
        else:
            pp2f = _DEFAULT_PAST_PRESENT_TO_FUTURE
            f2pp = _DEFAULT_FUTURE_TO_PAST_PRESENT

        logger.debug(
            "Loaded tense markers from YAML: "
            "%d future, %d past, %d habitual, %d present adverbs, "
            "%d pp2f, %d f2pp corrections",
            len(future),
            len(past),
            len(habitual),
            len(present),
            len(pp2f),
            len(f2pp),
        )
        return future, past, habitual, present, pp2f, f2pp

    except Exception:
        logger.warning(
            "Failed to load tense markers YAML, using defaults",
            exc_info=True,
        )
        return (
            _DEFAULT_FUTURE_ADVERBS,
            _DEFAULT_PAST_ADVERBS,
            _DEFAULT_HABITUAL_ADVERBS,
            _DEFAULT_PRESENT_ADVERBS,
            _DEFAULT_PAST_PRESENT_TO_FUTURE,
            _DEFAULT_FUTURE_TO_PAST_PRESENT,
        )


# Load at module level (once, at import time)
(
    _FUTURE_ADVERBS,
    _PAST_ADVERBS,
    _HABITUAL_ADVERBS,
    _PRESENT_ADVERBS,
    _PAST_PRESENT_TO_FUTURE,
    _FUTURE_TO_PAST_PRESENT,
) = _load_tense_markers()


class TenseDetectionMixin:
    """Mixin providing tense/aspect mismatch detection.

    Detects conflicts between temporal adverbs and verb markers
    (embedded aspect markers and sentence-final particles).

    Data is loaded from ``rules/tense_markers.yaml`` at module level,
    falling back to hardcoded defaults if the YAML is unavailable.
    """

    # --- Type stubs for attributes provided by SpellChecker or sibling mixins ---
    provider: "DictionaryProvider"
    _COLLOQUIAL_PRONOUNS: frozenset[str]

    # Temporal adverbs by tense category
    _FUTURE_ADVERBS: frozenset[str] = _FUTURE_ADVERBS
    _PAST_ADVERBS: frozenset[str] = _PAST_ADVERBS
    _HABITUAL_ADVERBS: frozenset[str] = _HABITUAL_ADVERBS
    _PRESENT_ADVERBS: frozenset[str] = _PRESENT_ADVERBS
    # Embedded aspect markers — these sit INSIDE verb compounds
    # and indicate specific tense/aspect, distinct from SFP endings.
    _PAST_ASPECT_MARKER = normalize("ခဲ့")  # specific past event
    _FUTURE_ASPECT_MARKER = normalize("မည်")  # future intent
    _PAST_VERB_SUFFIXES_FOR_PRESENT_CONFLICT: tuple[str, ...] = tuple(
        normalize(s) for s in ("ခဲ့", "ခဲ့သည်", "ခဲ့တယ်", "ခဲ့ပါသည်", "ခဲ့ပါပြီ")
    )
    # Particle correction maps (SFP-level fallback)
    _PAST_PRESENT_TO_FUTURE: dict[str, str | list[str]] = _PAST_PRESENT_TO_FUTURE
    _FUTURE_TO_PAST_PRESENT: dict[str, str | list[str]] = _FUTURE_TO_PAST_PRESENT

    # ── Full-form suffix rewrite tables (longest-first) ──
    # Used to build complete corrected verb forms instead of bare markers.

    # Past suffix → future suffix (for future-adverb + past-marker conflicts)
    _PAST_SUFFIX_TO_FUTURE: list[tuple[str, str]] = [
        (normalize(old), normalize(new))
        for old, new in [
            ("ခဲ့ပါသည်", "ပါမည်"),
            ("ခဲ့ပါတယ်", "ပါမယ်"),
            ("ခဲ့သည်", "မည်"),
            ("ခဲ့တယ်", "မယ်"),
            ("ခဲ့", ""),  # bare ခဲ့ → just remove
        ]
    ]

    # Future suffix → past suffix (for past-adverb + future-marker conflicts)
    _FUTURE_SUFFIX_TO_PAST: list[tuple[str, str]] = [
        (normalize(old), normalize(new))
        for old, new in [
            ("ပါမည်", "ခဲ့ပါသည်"),
            ("ပါမယ်", "ခဲ့ပါတယ်"),
            ("မည်", "ခဲ့သည်"),
            ("မယ်", "ခဲ့တယ်"),
        ]
    ]

    # Past suffix → present suffix (for present-adverb + past-marker)
    _PAST_SUFFIX_TO_PRESENT: list[tuple[str, str]] = [
        (normalize(old), normalize(new))
        for old, new in [
            ("ခဲ့ပါသည်", "နေပါသည်"),
            ("ခဲ့ပါတယ်", "နေပါတယ်"),
            ("ခဲ့သည်", "နေသည်"),
            ("ခဲ့တယ်", "နေတယ်"),
            ("ခဲ့", "နေ"),  # bare ခဲ့ → နေ
        ]
    ]

    # Myanmar sentence-ending punctuation that may trail verb tokens.
    _TRAILING_PUNCT: frozenset[str] = frozenset({"။", "၊"})

    @staticmethod
    def _build_full_form_suggestion(
        token: str,
        suffix_table: list[tuple[str, str]],
    ) -> str | None:
        """Build a full corrected verb form using suffix rewrite tables.

        Scans *suffix_table* (sorted longest-first) to find the first
        suffix that matches *token*, strips it, and appends the
        replacement suffix.  Trailing Myanmar punctuation (``။``,
        ``၊``) is stripped before matching and re-appended after.

        Args:
            token: The verb compound token (e.g., ``တွေ့ခဲ့တယ်``).
            suffix_table: List of ``(old_suffix, new_suffix)`` pairs
                sorted longest-first.

        Returns:
            Full corrected form (e.g., ``တွေ့မယ်``), or ``None`` if
            no suffix matched.
        """
        # Strip trailing punctuation for matching — punctuation is NOT
        # included in the returned corrected form because it's not part
        # of the linguistic correction.
        bare = token
        while bare and bare[-1] in TenseDetectionMixin._TRAILING_PUNCT:
            bare = bare[:-1]

        for old_suffix, new_suffix in suffix_table:
            if bare.endswith(old_suffix) and len(old_suffix) < len(bare):
                stem = bare[: len(bare) - len(old_suffix)]
                return stem + new_suffix
        return None

    def _detect_tense_mismatch(self, text: str, errors: list[Error]) -> None:
        """Detect tense/aspect mismatch between temporal adverbs and verb markers.

        Checks for:
        1. Embedded aspect markers (ခဲ့, မည်) conflicting with temporal adverbs
        2. SFP endings (တယ်, မယ်) conflicting with temporal adverbs (fallback)
        3. Habitual adverbs (နေ့တိုင်း) with specific-past marker ခဲ့
        """
        from myspellchecker.core.detectors.utils import get_tokenized

        tokenized = get_tokenized(self, text)
        if len(tokenized) < 2:
            return

        tokens = tokenized.tokens
        token_positions = tokenized.positions

        from myspellchecker.grammar.patterns import (
            get_question_completion_suggestions,
            is_second_person_modal_future_question,
        )

        # Classify temporal adverbs (prefix match for suffixed multi-word forms
        # like လာမည့်နှစ်တွင် which contain the adverb as a prefix)
        found_future_adverb = any(
            any(t == adv or t.startswith(adv) for adv in self._FUTURE_ADVERBS) for t in tokens
        )
        found_past_adverb = any(
            any(t == adv or t.startswith(adv) for adv in self._PAST_ADVERBS) for t in tokens
        )
        found_habitual_adverb = any(t in self._HABITUAL_ADVERBS for t in tokens)
        found_present_adverb = any(t in self._PRESENT_ADVERBS for t in tokens)
        implicit_second_person_question = is_second_person_modal_future_question(tokens) or (
            found_future_adverb
            and any(t in self._COLLOQUIAL_PRONOUNS for t in tokens)
            and any(normalize("နိုင်") in t for t in tokens)
        )

        if (
            not found_future_adverb
            and not found_past_adverb
            and not found_habitual_adverb
            and not found_present_adverb
        ):
            return

        existing_positions = get_existing_positions(errors)

        # Phase 1: Look for embedded aspect markers inside verb tokens
        # This takes priority over SFP-level detection (more precise).
        for i in range(len(tokens) - 1, -1, -1):
            token = tokens[i]
            token_pos = token_positions[i]

            if found_present_adverb and normalize("နေခဲ့") in token:
                # Present-now contexts should not use progressive+past cluster နေခဲ့.
                marker = normalize("နေခဲ့")
                marker_offset = token.find(marker)
                err_pos = token_pos + marker_offset
                if err_pos not in existing_positions:
                    errors.append(
                        SyllableError(
                            text=marker,
                            position=err_pos,
                            suggestions=[normalize("နေ")],
                            confidence=TEXT_DETECTOR_CONFIDENCES["aspect_adverb_conflict"],
                            error_type=ET_ASPECT_ADVERB_CONFLICT,
                        )
                    )
                return

            if (
                found_present_adverb
                and self._PAST_ASPECT_MARKER in token
                and any(
                    token.endswith(suffix)
                    for suffix in self._PAST_VERB_SUFFIXES_FOR_PRESENT_CONFLICT
                )
            ):
                # Present-now contexts should avoid specific past marker ခဲ့.
                # Report the suffix portion as error text, suggest replacement.
                # E.g., စစ်ဆေးခဲ့သည် → err_text=ခဲ့သည်, sugg=နေသည်
                marker_offset = token.find(self._PAST_ASPECT_MARKER)
                err_pos = token_pos + marker_offset
                err_suffix = token[marker_offset:]
                # Find matching suffix replacement
                sugg_present = normalize("နေ")
                for old_sfx, new_sfx in self._PAST_SUFFIX_TO_PRESENT:
                    if err_suffix == old_sfx or (
                        err_suffix.endswith(old_sfx) and len(old_sfx) == len(err_suffix)
                    ):
                        sugg_present = new_sfx
                        break
                if err_pos not in existing_positions:
                    errors.append(
                        SyllableError(
                            text=err_suffix,
                            position=err_pos,
                            suggestions=[sugg_present],
                            confidence=TEXT_DETECTOR_CONFIDENCES["aspect_adverb_conflict"],
                            error_type=ET_ASPECT_ADVERB_CONFLICT,
                        )
                    )
                return

            if found_future_adverb and self._PAST_ASPECT_MARKER in token:
                # Future adverb + past aspect ခဲ့ → build full-form suggestion.
                # E.g., တွေ့ခဲ့တယ် → တွေ့မယ်, စတင်ခဲ့သည် → စတင်မည်
                full_form = self._build_full_form_suggestion(token, self._PAST_SUFFIX_TO_FUTURE)
                if token_pos not in existing_positions:
                    if full_form:
                        sugg_list_fut: list[str] = [full_form]
                        err_text_fut = token
                    else:
                        sugg_list_fut = []
                        err_text_fut = self._PAST_ASPECT_MARKER
                    errors.append(
                        SyllableError(
                            text=err_text_fut,
                            position=token_pos,
                            suggestions=sugg_list_fut,
                            confidence=TEXT_DETECTOR_CONFIDENCES["tense_mismatch_future"],
                            error_type=ET_TENSE_MISMATCH,
                        )
                    )
                return

            if found_past_adverb and self._FUTURE_ASPECT_MARKER in token:
                # Past adverb + future aspect မည် → build full-form suggestion.
                # E.g., နမူနာပြမည် → နမူနာပြခဲ့သည်
                full_form = self._build_full_form_suggestion(token, self._FUTURE_SUFFIX_TO_PAST)
                if token_pos not in existing_positions:
                    if full_form:
                        sugg_list_past: list[str] = [full_form]
                        err_text_past = token
                    else:
                        err_text_past = self._FUTURE_ASPECT_MARKER
                        sugg_list_past = [self._PAST_ASPECT_MARKER]
                    errors.append(
                        SyllableError(
                            text=err_text_past,
                            position=token_pos,
                            suggestions=sugg_list_past,
                            confidence=TEXT_DETECTOR_CONFIDENCES["tense_mismatch_past"],
                            error_type=ET_TENSE_MISMATCH,
                        )
                    )
                return

            if found_habitual_adverb and self._PAST_ASPECT_MARKER in token:
                # Habitual adverb + specific-past ခဲ့ → aspect conflict
                marker_offset = token.find(self._PAST_ASPECT_MARKER)
                err_pos = token_pos + marker_offset
                if err_pos not in existing_positions:
                    errors.append(
                        SyllableError(
                            text=self._PAST_ASPECT_MARKER,
                            position=err_pos,
                            suggestions=[],  # detection-only
                            confidence=TEXT_DETECTOR_CONFIDENCES["aspect_adverb_conflict"],
                            error_type=ET_ASPECT_ADVERB_CONFLICT,
                        )
                    )
                return

        # Phase 2: Fallback — check SFP endings when no embedded marker found
        for i in range(len(tokens) - 1, -1, -1):
            token = tokens[i]
            token_pos = token_positions[i]

            if found_future_adverb:
                for particle, correction in sorted(
                    self._PAST_PRESENT_TO_FUTURE.items(), key=lambda x: len(x[0]), reverse=True
                ):
                    if token == particle or (
                        len(particle) < len(token) and token.endswith(particle)
                    ):
                        # Skip if the token already contains the future aspect
                        # marker — the future tense is already expressed via
                        # the embedded marker, and the SFP is just a copula.
                        # E.g., ထုတ်ပြန်မည်ဖြစ်သည် has both မည် and သည်,
                        # where မည် provides future and ဖြစ်သည် is copula.
                        if len(particle) < len(token) and self._FUTURE_ASPECT_MARKER in token:
                            continue
                        err_pos = token_pos + len(token) - len(particle)
                        if err_pos not in existing_positions:
                            sugg_tm = [correction] if isinstance(correction, str) else correction
                            if implicit_second_person_question and particle in {"တယ်", "သည်"}:
                                question_suggestions = get_question_completion_suggestions(
                                    particle,
                                    tokens,
                                    prefer_yes_no=True,
                                    phrase_first=False,
                                )
                                if question_suggestions:
                                    sugg_tm = [question_suggestions[0]]
                            errors.append(
                                SyllableError(
                                    text=particle,
                                    position=err_pos,
                                    suggestions=sugg_tm,
                                    confidence=TEXT_DETECTOR_CONFIDENCES["tense_mismatch_future"],
                                    error_type=ET_TENSE_MISMATCH,
                                )
                            )
                        return

            elif found_past_adverb:
                for particle, correction in sorted(
                    self._FUTURE_TO_PAST_PRESENT.items(),
                    key=lambda x: len(x[0]),
                    reverse=True,
                ):
                    if token == particle or (
                        len(particle) < len(token) and token.endswith(particle)
                    ):
                        # Symmetric guard: skip if the token already contains
                        # the past aspect marker (ခဲ့).
                        if len(particle) < len(token) and self._PAST_ASPECT_MARKER in token:
                            continue
                        err_pos = token_pos + len(token) - len(particle)
                        if err_pos not in existing_positions:
                            sugg_tp = [correction] if isinstance(correction, str) else correction
                            errors.append(
                                SyllableError(
                                    text=particle,
                                    position=err_pos,
                                    suggestions=sugg_tp,
                                    confidence=TEXT_DETECTOR_CONFIDENCES["tense_mismatch_past"],
                                    error_type=ET_TENSE_MISMATCH,
                                )
                            )
                        return
