"""Loan word variant lookup for Myanmar spell checking.

Loads transliteration variants from rules/loan_words.yaml and provides
bidirectional lookup: variant -> standard form(s), and standard -> variants.
"""

from __future__ import annotations

import importlib.resources
from functools import lru_cache

from myspellchecker.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Section keys in loan_words.yaml that contain entry lists.
_ENTRY_SECTIONS = ("english", "pali_sanskrit", "other")


@lru_cache(maxsize=1)
def _load_loan_word_data() -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Load loan word variants from YAML.

    Returns:
        Tuple of:
        - variant_to_standard: maps each variant to its standard form(s)
        - standard_to_variants: maps each standard form to its variants
    """
    variant_to_standard: dict[str, set[str]] = {}
    standard_to_variants: dict[str, set[str]] = {}

    try:
        import yaml

        yaml_path = importlib.resources.files("myspellchecker.rules").joinpath("loan_words.yaml")
        with importlib.resources.as_file(yaml_path) as path:
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
    except Exception:
        logger.debug("Failed to load loan_words.yaml", exc_info=True)
        return variant_to_standard, standard_to_variants

    if not data:
        return variant_to_standard, standard_to_variants

    for section_key in _ENTRY_SECTIONS:
        entries = data.get(section_key, [])
        if not isinstance(entries, list):
            continue
        for entry in entries:
            standard = entry.get("standard", "")
            variants = entry.get("variants", [])
            if not standard or not variants:
                continue

            if standard not in standard_to_variants:
                standard_to_variants[standard] = set()

            for variant in variants:
                if not variant:
                    continue
                standard_to_variants[standard].add(variant)
                if variant not in variant_to_standard:
                    variant_to_standard[variant] = set()
                variant_to_standard[variant].add(standard)

    return variant_to_standard, standard_to_variants


_EMPTY_FROZENSET: frozenset[str] = frozenset()


def get_loan_word_standard(word: str) -> frozenset[str]:
    """Get standard form(s) for a loan word variant."""
    v2s, _ = _load_loan_word_data()
    result = v2s.get(word)
    return frozenset(result) if result else _EMPTY_FROZENSET


def get_loan_word_variants(word: str) -> frozenset[str]:
    """Get known variant spellings for a standard loan word."""
    _, s2v = _load_loan_word_data()
    result = s2v.get(word)
    return frozenset(result) if result else _EMPTY_FROZENSET


def is_loan_word_variant(word: str) -> bool:
    """Check if a word is a known loan word variant (non-standard spelling)."""
    v2s, _ = _load_loan_word_data()
    return word in v2s
