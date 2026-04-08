"""Medial confusion detection mixin for PostNormalizationDetectorsMixin.

Provides ``_detect_medial_confusion``, ``_detect_colloquial_contractions``,
and their associated data dicts.

Data is loaded from ``rules/medial_confusion.yaml`` at module level, with
fallback to hardcoded defaults if the YAML file is missing or invalid.

Extracted from ``post_normalization.py`` to reduce file size while
preserving the exact same method signatures and behaviour.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from myspellchecker.core.constants import (
    ET_COLLOQUIAL_CONTRACTION,
    ET_MEDIAL_CONFUSION,
    ET_SYLLABLE,
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
from myspellchecker.core.detectors.utils import (
    get_existing_positions,
    iter_occurrences,
    try_replace_syllable_error,
)
from myspellchecker.core.response import Error, SyllableError
from myspellchecker.utils.logging_utils import get_logger

logger = get_logger(__name__)

# ── Hardcoded defaults (fallback when YAML is unavailable) ──

_DEFAULT_UNCONDITIONAL: dict[str, str | list[str]] = _norm_dict(
    {
        "ကြေးဇူး": "ကျေးဇူး",
        "ကျိုးစား": "ကြိုးစား",
        "ဖေဆို": ["ဖြေ", "ဖြေဆို"],
        "ကြောင်းသား": ["ကျောင်းသား", "ကျောင်း"],
        "ကြောင်းပို့": ["ကျောင်း", "ကျောင်းပို့"],
        "ကျန်မာ": ["ကျန်း", "ကျန်းမာ"],
        "စာဖက်": ["ဖတ်", "စာဖတ်"],
        "ထွက်ခြာ": ["ခွာ", "ထွက်ခွာ"],
        "ကွန်ပျုတာ": "ကွန်ပျူတာ",
        "ပတ်ဝန်ကျင်": "ပတ်ဝန်းကျင်",
        "သိချင်း": ["သီချင်း", "သီ"],
        "ပျည်ထောင်စု": "ပြည်ထောင်စု",
        "ပျည်နယ်": "ပြည်နယ်",
        "ပျည်တွင်း": "ပြည်တွင်း",
        "ပျည်ပ": "ပြည်ပ",
        "ပျည်": "ပြည်",
        "ကုညီ": "ကူညီ",
        "ဟင်သီးဟင်ရွက်": "ဟင်းသီးဟင်းရွက်",
        "ပုဂံခတ်": ["ခေတ်", "ပုဂံခေတ်"],
        "ကျြန်တော်": ["ကျွန်တော်", "ကျွန်"],
        "ကျြန်": "ကျွန်",
        "မျြန်မာ့": "မြန်မာ့",
        "မျြန်မာ": "မြန်မာ",
        "မျြန်မြန်": "မြန်မြန်",
        "မျြန်": "မြန်",
        "မျြတ်စွာ": "မြတ်စွာ",
        "မျြတ်": "မြတ်",
        "ပြော့ပြ": ["ပြော", "ပြောပြ"],
        "အအေး": "အေး",
        "ကွောင်း": "ကောင်း",
        "ပြီးနှင့်": "ပြီးတော့",
        "နေ့ လည်": "နေ့လည်",
        "ဖက်နေ": ["ဖတ်", "ဖတ်နေ"],
        "ကျေညာ": "ကြေညာ",  # announce (ya-pin→ya-yit)
        "ပျောင်း": "ပြောင်း",  # change (ya-yit→ya-pin)
        # --- Loanword misspellings ---
        "ကွန်ပြူတာ": "ကွန်ပျူတာ",  # computer (medial ြ→ျ)
        "ပရောဂျတ်": "ပရောဂျက်",  # project (wrong final)
        "အင်တာနတ်": "အင်တာနက်",  # internet (wrong final)
        "ဘက်ဂျက်": "ဘတ်ဂျက်",  # budget (wrong consonant)
        "ဒီမိုကရီစီ": "ဒီမိုကရေစီ",  # democracy (wrong vowel)
        "ဆော့ဝ်ဖဲ": "ဆော့ဖ်ဝဲ",  # software (syllable reorder)
        "ဗီတမင်": "ဗီတာမင်",  # vitamin (missing vowel AA)
        "ဟိုတေလ်": "ဟိုတယ်",  # hotel (wrong form)
        # --- Vowel / diacritic confusion in common words ---
        "ဒီဇိင်း": "ဒီဇိုင်း",  # design (missing vowel ု)
        "စိးပွားရေး": "စီးပွားရေး",  # economy (wrong vowel ိ→ီ)
        "ကျာင်း": "ကျောင်း",  # school (missing vowel)
        "ကြိးကြပ်": "ကြီးကြပ်",  # supervise (wrong vowel ိ→ီ)
        "ခွင့်ပြူချက်": "ခွင့်ပြုချက်",  # permission (wrong vowel ူ→ု)
        "မျို့တော်": "မြို့တော်",  # capital city (medial ျ→ြ)
    }
)

_DEFAULT_CONTEXTUAL: dict[str, tuple[str, tuple[str, ...]]] = _norm_dict_context(
    {
        "ကြောင်း": (
            "ကျောင်း",
            ("ကို", "သား", "ပို့", "အုပ်", "တိုက်"),
        ),
        "ကြး": ("ကျေး", ("ဇူး",)),
    }
)

_DEFAULT_COLLOQUIAL: dict[str, str | list[str]] = _norm_dict(
    {
        "ကျနော်": "ကျွန်တော်",
    }
)

# ── YAML loading ──

_YAML_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent / "rules" / "medial_confusion.yaml"
)


def _load_medial_confusion() -> tuple[
    dict[str, str | list[str]],
    dict[str, tuple[str, tuple[str, ...]]],
    dict[str, str | list[str]],
]:
    """Load medial confusion dicts from YAML with fallback.

    Returns:
        Tuple of (unconditional, contextual, colloquial) dicts,
        all with keys/values normalized via _norm_dict/_norm_dict_context.
    """
    if not _YAML_PATH.exists():
        logger.debug(
            "Medial confusion YAML not found at %s, using defaults",
            _YAML_PATH,
        )
        return (
            _DEFAULT_UNCONDITIONAL,
            _DEFAULT_CONTEXTUAL,
            _DEFAULT_COLLOQUIAL,
        )

    try:
        import yaml

        with open(_YAML_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not data or not isinstance(data, dict):
            logger.warning("Medial confusion YAML empty or invalid, using defaults")
            return (
                _DEFAULT_UNCONDITIONAL,
                _DEFAULT_CONTEXTUAL,
                _DEFAULT_COLLOQUIAL,
            )

        # -- Unconditional --
        raw_unconditional = data.get("unconditional", {})
        if isinstance(raw_unconditional, dict) and raw_unconditional:
            unconditional = _norm_dict(raw_unconditional)
        else:
            unconditional = _DEFAULT_UNCONDITIONAL

        # -- Contextual --
        raw_contextual = data.get("contextual", {})
        if isinstance(raw_contextual, dict) and raw_contextual:
            contextual_parsed: dict[str, tuple[str, tuple[str, ...]]] = {}
            for k, v in raw_contextual.items():
                if isinstance(v, dict):
                    correct = v.get("correct", "")
                    contexts = v.get("contexts", [])
                    if correct and contexts:
                        contextual_parsed[k] = (
                            correct,
                            tuple(contexts),
                        )
            contextual = _norm_dict_context(contextual_parsed)
        else:
            contextual = _DEFAULT_CONTEXTUAL

        # -- Colloquial contractions --
        raw_colloquial = data.get("colloquial_contractions", {})
        if isinstance(raw_colloquial, dict) and raw_colloquial:
            colloquial = _norm_dict(raw_colloquial)
        else:
            colloquial = _DEFAULT_COLLOQUIAL

        logger.debug(
            "Loaded medial confusion rules from YAML: "
            "%d unconditional, %d contextual, %d colloquial",
            len(unconditional),
            len(contextual),
            len(colloquial),
        )
        return unconditional, contextual, colloquial

    except Exception:
        logger.warning(
            "Failed to load medial confusion YAML, using defaults",
            exc_info=True,
        )
        return (
            _DEFAULT_UNCONDITIONAL,
            _DEFAULT_CONTEXTUAL,
            _DEFAULT_COLLOQUIAL,
        )


# Load at module level (once, at import time)
(
    _MEDIAL_CONFUSION_UNCONDITIONAL,
    _MEDIAL_CONFUSION_CONTEXTUAL,
    _COLLOQUIAL_CONTRACTIONS,
) = _load_medial_confusion()


class MedialConfusionMixin:
    """Mixin providing medial ya-pin/ya-yit confusion detection.

    Detects unconditional and contextual medial confusion patterns.
    Data is loaded from ``rules/medial_confusion.yaml`` at module level,
    falling back to hardcoded defaults if the YAML is unavailable.
    """

    # --- Type stubs for attributes provided by SpellChecker ---
    provider: "DictionaryProvider"

    # ----- Medial Ya-pin / Ya-yit confusion detection -----
    # Entries: (wrong form, correct form).  Two categories:
    #  1. Unconditional -- the wrong form is NOT a valid compound word.
    #  2. Contextual -- both forms are valid; trigger only when a
    #     trailing context token appears.
    _MEDIAL_CONFUSION_UNCONDITIONAL: dict[str, str | list[str]] = _MEDIAL_CONFUSION_UNCONDITIONAL

    _MEDIAL_CONFUSION_CONTEXTUAL: dict[str, tuple[str, tuple[str, ...]]] = (
        _MEDIAL_CONFUSION_CONTEXTUAL
    )

    # ----- Colloquial contraction detection -----
    # Colloquial forms that the segmenter may split, bypassing the
    # word-level colloquial checker.
    _COLLOQUIAL_CONTRACTIONS: dict[str, str | list[str]] = _COLLOQUIAL_CONTRACTIONS

    def _detect_medial_confusion(self, text: str, errors: list[Error]) -> None:
        """Detect ya-pin / ya-yit medial confusion.

        Searches the text for known medial-confusion patterns:
        - Unconditional: wrong compound that is NOT a valid word.
        - Contextual: both forms are valid words, but surrounding
          tokens disambiguate.
        """
        existing_positions = get_existing_positions(errors)

        # 1. Unconditional patterns (invalid compound -- always wrong)
        for wrong, correct_val in self._MEDIAL_CONFUSION_UNCONDITIONAL.items():
            # Normalize to list of suggestions
            corrections = correct_val if isinstance(correct_val, list) else [correct_val]
            for idx, _end in iter_occurrences(text, wrong):
                # If an invalid_syllable already exists at this
                # position, replace it with medial_confusion error
                # (medial_confusion survives
                # filter_syllable_errors_in_valid_words,
                # invalid_syllable does not when inside a valid
                # compound word).
                if idx in existing_positions:
                    replaced = False
                    for i, e in enumerate(errors):
                        if e.position == idx and e.error_type == ET_SYLLABLE:
                            errors[i] = SyllableError(
                                text=wrong,
                                position=idx,
                                suggestions=list(corrections),
                                confidence=TEXT_DETECTOR_CONFIDENCES[
                                    "medial_confusion_unconditional"
                                ],
                                error_type=ET_MEDIAL_CONFUSION,
                            )
                            replaced = True
                            break
                        if e.position == idx and e.error_type == ET_MEDIAL_CONFUSION:
                            # Existing medial_confusion from syllable
                            # validator -- append unconditional
                            # corrections as extra suggestions
                            for c in corrections:
                                if c not in (e.suggestions or []):
                                    e.suggestions = (e.suggestions or []) + [c]
                            replaced = True
                            break
                    if replaced:
                        continue
                    # Non-syllable error at this position -- skip
                    continue
                errors.append(
                    SyllableError(
                        text=wrong,
                        position=idx,
                        suggestions=list(corrections),
                        confidence=TEXT_DETECTOR_CONFIDENCES["medial_confusion_unconditional"],
                        error_type=ET_MEDIAL_CONFUSION,
                    )
                )

        # 2. Contextual patterns (wrong form + required trailing context)
        for wrong, (correct, trailing) in self._MEDIAL_CONFUSION_CONTEXTUAL.items():
            for idx, end in iter_occurrences(text, wrong):
                # Must start at word boundary (start of text or
                # after space) to avoid matching inside compounds
                # like သမိုင်းကြောင်းကို
                if idx > 0 and text[idx - 1] != " ":
                    continue
                if trailing:
                    tail = text[end:]
                    if not any(tail.startswith(t) for t in trailing):
                        continue
                new_err = SyllableError(
                    text=wrong,
                    position=idx,
                    suggestions=[correct],
                    confidence=TEXT_DETECTOR_CONFIDENCES["medial_confusion_contextual"],
                    error_type=ET_MEDIAL_CONFUSION,
                )
                if idx in existing_positions:
                    # Replace invalid_syllable (would be filtered out later)
                    # with medial_confusion (preserved). Skip if a preserved
                    # error type already occupies this position.
                    try_replace_syllable_error(errors, idx, new_err)
                    continue
                errors.append(new_err)

    def _detect_colloquial_contractions(self, text: str, errors: list[Error]) -> None:
        """Detect colloquial contractions via direct text search.

        The word-level colloquial checker depends on the segmenter
        producing the contraction as a single token. When the
        segmenter splits it (e.g. ကျနော် → ကျ + နော်), the
        word-level check misses it. This method searches the text
        directly and is immune to segmenter behaviour.

        Replaces any shorter syllable-fragment errors that overlap
        the same position.
        """
        for wrong, correct in self._COLLOQUIAL_CONTRACTIONS.items():
            for idx, end in iter_occurrences(text, wrong):
                # Word boundary: start of text or after space
                if idx > 0 and text[idx - 1] != " ":
                    continue
                # Word boundary: end of text or before space
                if end < len(text) and text[end] != " ":
                    continue
                sugg_cc = [correct] if isinstance(correct, str) else correct
                new_err = SyllableError(
                    text=wrong,
                    position=idx,
                    suggestions=sugg_cc,
                    confidence=TEXT_DETECTOR_CONFIDENCES["colloquial_contraction"],
                    error_type=ET_COLLOQUIAL_CONTRACTION,
                )
                # Replace any shorter overlapping errors at same pos
                replaced = False
                for i, e in enumerate(errors):
                    if e.position == idx and len(e.text) < len(wrong):
                        errors[i] = new_err
                        replaced = True
                        break
                if not replaced and idx not in {e.position for e in errors}:
                    errors.append(new_err)
