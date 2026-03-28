"""
Helper utilities for SemanticValidationStrategy.

Contains candidate generation, scoring helpers, and variant generation
logic extracted from the main strategy class to improve maintainability.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from myspellchecker.core.config.algorithm_configs import SemanticStrategyConfig
from myspellchecker.core.constants import PARTICLE_CONFUSABLES, VARIANT_BLOCKLIST
from myspellchecker.core.validation_strategies.confusable_strategy import (
    generate_confusable_variants,
)
from myspellchecker.text.phonetic import PhoneticHasher

if TYPE_CHECKING:
    from myspellchecker.providers.interfaces import DictionaryProvider

# ── Constants ──────────────────────────────────────────────────────────

STOP_CODA_PAIRS: tuple[tuple[str, str], ...] = (
    ("\u1000\u103a", "\u1010\u103a"),  # က် / တ်
    ("\u1000\u103a", "\u1015\u103a"),  # က် / ပ်
    ("\u1010\u103a", "\u1015\u103a"),  # တ် / ပ်
)

INITIAL_SWAP_MAP: dict[str, tuple[str, ...]] = {
    "\u1000": ("\u1001",),  # က → ခ
    "\u1001": ("\u1000",),  # ခ → က
    "\u1002": ("\u1003",),  # ဂ → ဃ
    "\u1003": ("\u1002",),  # ဃ → ဂ
    "\u1005": ("\u1006",),  # စ → ဆ
    "\u1006": ("\u1005",),  # ဆ → စ
    "\u1010": ("\u1011",),  # တ → ထ
    "\u1011": ("\u1010",),  # ထ → တ
    "\u1015": ("\u1016", "\u1017"),  # ပ → ဖ, ဗ
    "\u1016": ("\u1015", "\u1018"),  # ဖ → ပ, ဘ
    "\u1017": ("\u1015",),  # ဗ → ပ
    "\u1018": ("\u1016",),  # ဘ → ဖ
    "\u1014": ("\u1014\u103e",),  # န → နှ
    "\u101b": ("\u101b\u103e",),  # ရ → ရှ
    "\u101c": ("\u101c\u103e",),  # လ → လှ
}

# Suffixes that can be detached to form candidate stems.
DETACHABLE_SUFFIXES: tuple[str, ...] = (
    "\u1010\u103d\u1004\u103a",  # တွင်
    "\u101e\u100a\u103a",  # သည်
    "\u1019\u103e\u102c",  # မှာ
    "\u1015\u103c\u102e\u1038",  # ပြီး
    "\u1010\u1031\u102c\u103a",  # တော့  (using NFC forms — see normalization)
    "\u101c\u100a\u103a\u1038",  # လည်း
    "\u1010\u101a\u103a",  # တယ်
    "\u1019\u101a\u103a",  # မယ်
    "\u1014\u1032\u1037",  # နဲ့
    "\u1000\u102d\u102f",  # ကို
    "\u1019\u103e",  # မှ
    "\u1000",  # က
    "\u1015\u102b",  # ပါ
)


# ── Free functions ─────────────────────────────────────────────────────


def char_overlap_similarity(left: str, right: str) -> float:
    """Symmetric set-based overlap similarity used as a structural sanity guard."""
    if not left or not right:
        return 0.0
    left_set = set(left)
    right_set = set(right)
    common = len(left_set & right_set)
    return common / max(len(left_set), len(right_set))


def contrast_margin(
    word: str,
    frequency: int,
    config: SemanticStrategyConfig,
) -> float:
    """Dynamic margin for candidate-vs-current contrast decisions."""
    margin = config.contrast_base_margin
    if len(word) <= 2:
        margin += config.margin_boost_short_word
    if frequency >= config.scan_freq_threshold:
        margin += config.margin_boost_high_freq
    elif frequency >= 10_000:
        margin += config.margin_boost_mid_freq
    return margin


def generate_mark_variants(word: str) -> set[str]:
    """Generate basic visarga/asat toggling variants."""
    variants: set[str] = set()
    if not word:
        return variants

    if word.endswith("\u1038"):  # း
        variants.add(word[:-1])
    else:
        variants.add(word + "\u1038")

    if word.endswith("\u103a"):  # ်
        variants.add(word[:-1])
    else:
        variants.add(word + "\u103a")

    # Common tonal/asat sequence variants.
    if word.endswith("\u1014\u103a"):  # န်
        variants.add(word + "\u1038")
    if word.endswith("\u1019\u103a"):  # မ်
        variants.add(word + "\u1038")

    return variants


def generate_stop_coda_variants(word: str) -> set[str]:
    """Generate final stop-coda confusion variants (က်/တ်/ပ်)."""
    variants: set[str] = set()
    if len(word) < 2:
        return variants
    for first, second in STOP_CODA_PAIRS:
        if word.endswith(first):
            variants.add(word[: -len(first)] + second)
        if word.endswith(second):
            variants.add(word[: -len(second)] + first)
    return variants


def generate_initial_swap_variants(word: str) -> set[str]:
    """Generate generalized initial-consonant confusion variants."""
    variants: set[str] = set()
    if not word:
        return variants

    for source, replacements in INITIAL_SWAP_MAP.items():
        if source.endswith("\u103e"):  # ှ
            if word.startswith(source):
                tail = word[len(source) :]
                for replacement in replacements:
                    variants.add(replacement + tail)
            continue

        if word.startswith(source):
            tail = word[len(source) :]
            for replacement in replacements:
                variants.add(replacement + tail)

    return variants


def filter_and_rank_candidates(
    *,
    word: str,
    candidates: set[str],
    limit: int,
    provider: "DictionaryProvider | None",
) -> list[str]:
    """Filter to dictionary-valid candidates and rank by frequency."""
    if not provider:
        return []

    scored: list[tuple[int, str]] = []
    for candidate in candidates:
        if (
            not candidate
            or candidate == word
            or candidate in VARIANT_BLOCKLIST
            or len(candidate) < 2
        ):
            continue
        if not provider.is_valid_word(candidate):
            continue
        freq = 0
        if hasattr(provider, "get_word_frequency"):
            value = provider.get_word_frequency(candidate)
            if isinstance(value, (int, float)):
                freq = int(value)
        scored.append((freq, candidate))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [candidate for _freq, candidate in scored[:limit]]


def build_contrast_candidate_pool(
    word: str,
    *,
    provider: "DictionaryProvider | None",
    hasher: PhoneticHasher,
    config: SemanticStrategyConfig,
) -> list[str]:
    """Build generalized fallback candidate pool for one token."""
    if not provider:
        return []

    candidates: set[str] = set()

    try:
        candidates.update(generate_confusable_variants(word, hasher))
    except (RuntimeError, ValueError, TypeError, AttributeError):
        pass

    if word in PARTICLE_CONFUSABLES:
        candidates.update(PARTICLE_CONFUSABLES[word])

    candidates.update(generate_mark_variants(word))
    candidates.update(generate_stop_coda_variants(word))

    # Simple de-affix candidates
    if word.startswith("\u1019") and len(word) > 2:  # မ prefix
        candidates.add(word[1:])

    for suffix in DETACHABLE_SUFFIXES:
        if word.endswith(suffix) and len(word) > len(suffix) + 1:
            candidates.add(word[: -len(suffix)])

    return filter_and_rank_candidates(
        word=word,
        candidates=candidates,
        limit=config.contrast_max_candidates,
        provider=provider,
    )


def build_escalation_candidate_pool(
    word: str,
    *,
    base_pool: set[str],
    provider: "DictionaryProvider | None",
    config: SemanticStrategyConfig,
) -> list[str]:
    """Build expanded candidate pool for escalation second pass."""
    expanded = set(base_pool)
    expanded.update(generate_initial_swap_variants(word))
    expanded.update(generate_mark_variants(word))
    expanded.update(generate_stop_coda_variants(word))

    # Short-token fallback: inject ha-htoe on initial consonant.
    if word and len(word) <= 2:
        first = word[0]
        first_cp = ord(first)
        if 0x1000 <= first_cp <= 0x1021:
            expanded.add(first + "\u103e" + word[1:])  # ှ

    return filter_and_rank_candidates(
        word=word,
        candidates=expanded,
        limit=config.escalation_max_candidates,
        provider=provider,
    )


def should_run_escalation_pass(
    *,
    word: str,
    current_freq: int,
    is_valid_current: bool,
    config: SemanticStrategyConfig,
) -> bool:
    """Decide whether to run expensive second-pass candidate scoring."""
    if len(word) <= 3:
        return True
    if not is_valid_current:
        return True
    return current_freq <= config.escalation_valid_word_freq_cap
