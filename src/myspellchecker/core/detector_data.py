"""
Shared data constants for text-level detectors.

Contains normalization helpers and module-level constants used across
SpellChecker's detection methods.
"""

from collections.abc import Iterable
from pathlib import Path

from myspellchecker.text.normalize import normalize
from myspellchecker.utils.logging_utils import get_logger

logger = get_logger(__name__)

__all__ = [
    "STRUCTURAL_ERROR_TYPES",
    "TEXT_DETECTOR_CONFIDENCES",
    "norm_set",
    "norm_dict",
    "norm_dict_tuple",
    "norm_dict_context",
]


def norm_set(items: Iterable[str]) -> frozenset[str]:
    """Build a frozenset with all entries Myanmar-normalized."""
    return frozenset(normalize(s) for s in items)


def norm_dict(d: dict[str, str | list[str]]) -> dict[str, str | list[str]]:
    """Normalize both keys and values (supports str or list[str] values)."""
    result: dict[str, str | list[str]] = {}
    for k, v in d.items():
        nk = normalize(k)
        if isinstance(v, list):
            result[nk] = [normalize(s) for s in v]
        else:
            result[nk] = normalize(v)
    return result


def norm_dict_tuple(d: dict[str, tuple[str, str]]) -> dict[str, tuple[str, str]]:
    """Normalize keys and tuple values."""
    return {normalize(k): (normalize(v[0]), normalize(v[1])) for k, v in d.items()}


def norm_dict_context(
    d: dict[str, tuple[str, tuple[str, ...]]],
) -> dict[str, tuple[str, tuple[str, ...]]]:
    """Normalize keys and correction, keep context triggers normalized."""
    return {
        normalize(k): (normalize(v[0]), tuple(normalize(t) for t in v[1])) for k, v in d.items()
    }


# Error types that indicate structural degradation (not just character-level typos).
# Used to gate structural detectors: typo-heavy but structurally clear sentences
# should still get structural checks, while structurally garbled text should not.
STRUCTURAL_ERROR_TYPES = frozenset(
    {
        "dangling_particle",
        "dangling_word",
        "missing_conjunction",
        "tense_mismatch",
        "register_mixing",
        "broken_compound",
        "broken_stacking",
        "broken_compound_space",
        "negation_sfp_mismatch",
        "merged_sfp_conjunction",
        "aspect_adverb_conflict",
    }
)

# Default confidence scores for text-level detectors (fallback).
# These are internal heuristic values, not user-tunable config.
_DEFAULT_CONFIDENCES: dict[str, float] = {
    "zawgyi_detected": 0.95,
    "zero_width_chars": 0.95,
    "missing_asat": 0.90,
    "missing_visarga": 0.90,
    "medial_confusion_unconditional": 0.85,
    "medial_confusion_contextual": 0.80,
    "colloquial_contraction": 0.85,
    "particle_confusion": 0.80,
    "ha_htoe_confusion": 0.70,
    "dangling_particle": 0.75,
    "dangling_word": 0.80,
    "missing_conjunction": 0.80,
    "tense_mismatch_future": 0.90,
    "tense_mismatch_past": 0.85,
    "register_mixing": 0.85,
    "vowel_after_asat": 1.0,
    "broken_virama": 0.95,
    "broken_stacking": 0.90,
    "broken_compound_space": 0.90,
    "medial_order": 0.95,
    "vowel_medial_reorder": 0.95,
    "duplicate_diacritic": 1.0,
    "leading_vowel_e": 0.95,
    "incomplete_stacking": 0.90,
    "missing_stacking": 0.90,
    "negation_sfp_mismatch": 0.95,
    "merged_sfp_conjunction": 0.90,
    "aspect_adverb_conflict": 0.85,
    "semantic_implausibility": 0.72,
    "duplicate_punctuation": 0.99,
    "wrong_punctuation": 0.98,
    "missing_punctuation": 0.90,
    # Syllable validator confidence values
    "broken_virama_fix": 0.90,
    "digit_letter_confusion": 0.95,
    "aspiration_confusion": 0.85,
    "missing_diacritic_compound": 0.85,
    "unknown_compound_segment": 0.85,
    "collocation_error": 0.85,
    "particle_misuse": 0.80,
    "informal_with_honorific": 0.85,
    "informal_h_after_completive": 0.85,
    "broken_compound_morpheme": 0.88,
    "missegmented_confusable": 0.88,
}

_DETECTOR_CONFIDENCES_PATH = (
    Path(__file__).resolve().parent.parent / "rules" / "detector_confidences.yaml"
)


def _load_detector_confidences() -> dict[str, float]:
    """Load detector confidences from YAML, falling back to defaults."""
    if _DETECTOR_CONFIDENCES_PATH.exists():
        try:
            import yaml

            with open(_DETECTOR_CONFIDENCES_PATH, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if data and "confidences" in data:
                logger.debug(
                    "Loaded detector confidences from %s",
                    _DETECTOR_CONFIDENCES_PATH,
                )
                return data["confidences"]
        except Exception:
            logger.warning(
                "Failed to load detector confidences from %s, using defaults",
                _DETECTOR_CONFIDENCES_PATH,
            )
    return _DEFAULT_CONFIDENCES


TEXT_DETECTOR_CONFIDENCES: dict[str, float] = _load_detector_confidences()
