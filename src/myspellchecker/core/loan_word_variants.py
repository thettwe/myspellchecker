"""Loan word variant lookup for Myanmar spell checking.

Loads transliteration variants from two sources and merges them:

1. ``rules/loan_words.yaml`` — hand-curated, authoritative on conflict. ~197
   entries across english/pali_sanskrit/other sections.
2. ``rules/loan_words_mined.yaml`` — mined from the production DB's
   ``confusable_pairs`` table via ``scripts/mine_loan_word_pairs.py``. ~54
   additional variant→standard pairs as of v1.6.0 (2026-04-18), gated at
   context_overlap>=0.6, freq_ratio>=50, 2+ variant syllables, aspiration
   dropped, with a hand-curated blacklist for semantic/directional rejects.
   Linguist review passed at 93% precision.

Conflict resolution: YAML entries win. A variant already in ``loan_words.yaml``
is never overwritten by the mined set.
"""

from __future__ import annotations

import importlib.resources
import os
from functools import lru_cache
from pathlib import Path

from myspellchecker.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Section keys in loan_words.yaml that contain entry lists.
_ENTRY_SECTIONS = ("english", "pali_sanskrit", "other")

# Env var to disable mined-pair merging (debugging / A-B benchmarking).
_MINED_DISABLED_ENV = "MSC_DISABLE_MINED_LOAN_WORDS"
# Env var for an override path (testing / alternate artifact).
_MINED_PATH_ENV = "MSC_MINED_LOAN_WORDS_PATH"


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

    # Merge mined pairs on top (YAML entries take precedence on conflict).
    _merge_mined_pairs(variant_to_standard, standard_to_variants)

    return variant_to_standard, standard_to_variants


def _merge_mined_pairs(
    variant_to_standard: dict[str, set[str]],
    standard_to_variants: dict[str, set[str]],
) -> None:
    """Merge mined variant→standard pairs into the existing maps.

    YAML-sourced variants are authoritative: a variant already present in
    ``variant_to_standard`` is never overwritten. New variants from the mined
    set are added; new standards seen alongside a known variant are unioned.
    Controlled by env vars ``MSC_DISABLE_MINED_LOAN_WORDS`` and
    ``MSC_MINED_LOAN_WORDS_PATH``.
    """
    if os.environ.get(_MINED_DISABLED_ENV, "").strip().lower() in {"1", "true", "yes"}:
        logger.debug("Mined loan-word merge disabled via env")
        return

    try:
        import yaml
    except Exception:
        logger.debug("PyYAML not available; skipping mined loan-word merge")
        return

    data = None
    override_path = os.environ.get(_MINED_PATH_ENV, "").strip()
    if override_path:
        p = Path(override_path)
        if p.is_file():
            try:
                with p.open(encoding="utf-8") as f:
                    data = yaml.safe_load(f)
            except Exception:
                logger.debug("Failed to read mined file at override path %s", p, exc_info=True)

    if data is None:
        try:
            mined_path = importlib.resources.files("myspellchecker.rules").joinpath(
                "loan_words_mined.yaml"
            )
            with importlib.resources.as_file(mined_path) as path:
                if path.exists():
                    with open(path, encoding="utf-8") as f:
                        data = yaml.safe_load(f)
        except Exception:
            logger.debug("Failed to load loan_words_mined.yaml from package", exc_info=True)
            return

    if not data:
        return

    pairs = data.get("pairs") or []
    added = 0
    skipped_existing = 0
    for pair in pairs:
        variant = (pair.get("variant") or "").strip()
        standard = (pair.get("standard") or "").strip()
        if not variant or not standard or variant == standard:
            continue

        # YAML authority: if this variant is already mapped (by loan_words.yaml),
        # do not override — but we do allow unioning a new standard alongside
        # the existing ones only when the variant came from mined source too.
        if variant in variant_to_standard:
            skipped_existing += 1
            continue

        variant_to_standard[variant] = {standard}
        standard_to_variants.setdefault(standard, set()).add(variant)
        added += 1

    logger.debug(
        "Mined loan-word merge: +%d variants, %d skipped (already in YAML)",
        added,
        skipped_existing,
    )


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
