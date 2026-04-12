"""Generalized token-refinement helpers for validation-time segmentation.

This module implements a lightweight token-lattice style refinement pass that
operates on already-segmented tokens. It is designed to expose hidden error
spans in merged tokens (e.g., particle attachment, negation attachment)
without replacing the segmenter.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from myspellchecker.core.config.algorithm_configs import TokenRefinementConfig

# Common detachable clitics/particles in Burmese.
# Kept intentionally small and high-precision to avoid FP blow-ups.
_DETACHABLE_SUFFIXES: tuple[str, ...] = (
    "တို့",
    "နှင့်",
    "အတွက်",
    "တွင်",
    "သည်",
    "မှာ",
    "သို့",
    "ဖြင့်",
    "တော့",
    "လည်း",
    "ပြီး",
    "တယ်",
    "မယ်",
    "နဲ့",
    "ကို",
    "မှ",
    "က",
    "ပါ",
    "ပဲ",
    "ရော",
    "ဘူး",
    "ဟာ",
)

# Common typo variants of frequent suffixes.
_DETACHABLE_SUFFIX_VARIANTS: tuple[str, ...] = (
    "မာ",  # မှာ missing ှ
    "တယ",  # တယ် missing ်
    "မယ",  # မယ် missing ်
    "သည",  # သည် missing ်
)

_NEGATION_PREFIX = "မ"

# Module-level default config singleton (avoids re-instantiation).
_default_config = TokenRefinementConfig()


def refine_validation_tokens(
    words: list[str],
    *,
    is_valid_word: Callable[[str], bool] | None = None,
    get_word_frequency: Callable[[str], int] | None = None,
    get_bigram_probability: Callable[[str, str], float] | None = None,
    config: TokenRefinementConfig | None = None,
) -> list[str]:
    """Refine token boundaries for validation-time processing.

    The original token stream is always available as fallback. Each token is
    locally evaluated against split candidates; only candidates that clearly
    improve lexical/context plausibility are selected.
    """
    if not words:
        return []

    if is_valid_word is None and get_word_frequency is None:
        # No lexical signals available: keep original segmentation.
        return list(words)

    cfg = config or _default_config
    refined: list[str] = []

    for idx, token in enumerate(words):
        if not _is_refinement_candidate(token, cfg):
            refined.append(token)
            continue

        if _is_high_confidence_known(token, is_valid_word, get_word_frequency, cfg):
            refined.append(token)
            continue

        left = refined[-1] if refined else None
        right = words[idx + 1] if idx + 1 < len(words) else None

        base_score = _score_parts(
            [token],
            left=left,
            right=right,
            is_valid_word=is_valid_word,
            get_word_frequency=get_word_frequency,
            get_bigram_probability=get_bigram_probability,
            config=cfg,
        )

        best_parts = [token]
        best_score = base_score

        for candidate_parts in _generate_split_candidates(token):
            if not _passes_structure_guard(
                candidate_parts,
                is_valid_word=is_valid_word,
                get_word_frequency=get_word_frequency,
            ):
                continue

            candidate_score = _score_parts(
                candidate_parts,
                left=left,
                right=right,
                is_valid_word=is_valid_word,
                get_word_frequency=get_word_frequency,
                get_bigram_probability=get_bigram_probability,
                config=cfg,
            )
            if candidate_score > best_score:
                best_score = candidate_score
                best_parts = candidate_parts

        if best_parts != [token] and best_score >= base_score + cfg.min_score_gain:
            refined.extend(best_parts)
        else:
            refined.append(token)

    return refined


def build_validation_token_paths(
    words: list[str],
    *,
    is_valid_word: Callable[[str], bool] | None = None,
    get_word_frequency: Callable[[str], int] | None = None,
    get_bigram_probability: Callable[[str, str], float] | None = None,
    segment_syllables: Callable[[str], list[str]] | None = None,
    config: TokenRefinementConfig | None = None,
) -> list[list[str]]:
    """Build up to two validation-time token paths (Boundary-Lattice V2).

    Path 0 is the refined base path. Path 1 is the best alternate split path
    when a high-gain split candidate exists.
    """
    if not words:
        return []

    cfg = config or _default_config
    original_path = list(words)
    refined_path = refine_validation_tokens(
        words,
        is_valid_word=is_valid_word,
        get_word_frequency=get_word_frequency,
        get_bigram_probability=get_bigram_probability,
        config=cfg,
    )
    if cfg.lattice_max_paths <= 1:
        return [original_path]

    best_alt_idx = -1
    best_alt_parts: list[str] | None = None
    best_gain = 0.0

    for idx, token in enumerate(refined_path):
        if not _is_refinement_candidate(token, cfg):
            continue
        if _is_high_confidence_known(token, is_valid_word, get_word_frequency, cfg):
            continue

        left = refined_path[idx - 1] if idx > 0 else None
        right = refined_path[idx + 1] if idx + 1 < len(refined_path) else None

        base_score = _score_parts(
            [token],
            left=left,
            right=right,
            is_valid_word=is_valid_word,
            get_word_frequency=get_word_frequency,
            get_bigram_probability=get_bigram_probability,
            config=cfg,
        )

        candidate_sets = _generate_split_candidates(token)
        candidate_sets.extend(
            _generate_syllable_split_candidates(
                token, segment_syllables=segment_syllables, config=cfg
            )
        )
        for parts in candidate_sets:
            if not _passes_structure_guard(
                parts,
                is_valid_word=is_valid_word,
                get_word_frequency=get_word_frequency,
            ):
                continue
            candidate_score = _score_parts(
                parts,
                left=left,
                right=right,
                is_valid_word=is_valid_word,
                get_word_frequency=get_word_frequency,
                get_bigram_probability=get_bigram_probability,
                config=cfg,
            )
            gain = candidate_score - base_score
            if gain > best_gain and gain >= cfg.min_score_gain:
                best_gain = gain
                best_alt_idx = idx
                best_alt_parts = parts

    candidate_paths: list[list[str]] = []
    if refined_path != original_path:
        candidate_paths.append(refined_path)
    if best_alt_parts is not None and best_alt_idx >= 0:
        alt_path = list(refined_path[:best_alt_idx])
        alt_path.extend(best_alt_parts)
        alt_path.extend(refined_path[best_alt_idx + 1 :])
        if alt_path != refined_path:
            candidate_paths.append(alt_path)

    paths = [original_path]
    if candidate_paths:
        best_candidate = max(
            candidate_paths,
            key=lambda path: _score_token_sequence(
                path,
                is_valid_word=is_valid_word,
                get_word_frequency=get_word_frequency,
                get_bigram_probability=get_bigram_probability,
                config=cfg,
            ),
        )
        if best_candidate != original_path:
            paths.append(best_candidate)

    return paths[: cfg.lattice_max_paths]


def _is_refinement_candidate(token: str, config: TokenRefinementConfig) -> bool:
    if not token or len(token) < config.min_token_len:
        return False
    # Skip obvious punctuation/whitespace-only tokens.
    if not token.strip():
        return False
    # Keep numeric/mixed-numeric tokens untouched.
    if any(ch.isdigit() for ch in token):
        return False
    return True


def _is_high_confidence_known(
    token: str,
    is_valid_word: Callable[[str], bool] | None,
    get_word_frequency: Callable[[str], int] | None,
    config: TokenRefinementConfig,
) -> bool:
    known = bool(is_valid_word(token)) if is_valid_word is not None else False
    freq = _safe_freq(get_word_frequency, token)
    return known and freq >= config.keep_if_freq_at_least


def _generate_split_candidates(token: str) -> list[list[str]]:
    candidates: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()

    def _add(parts: list[str]) -> None:
        if len(parts) < 2 or len(parts) > 3:
            return
        if any(not p for p in parts):
            return
        key = tuple(parts)
        if key in seen:
            return
        seen.add(key)
        candidates.append(parts)

    suffixes = sorted(
        set(_DETACHABLE_SUFFIXES) | set(_DETACHABLE_SUFFIX_VARIANTS),
        key=len,
        reverse=True,
    )

    # Suffix detach: stem + suffix
    for suffix in suffixes:
        if token.endswith(suffix) and len(token) > len(suffix):
            stem = token[: -len(suffix)]
            _add([stem, suffix])
            # Negation + stem + suffix
            if (
                stem.startswith(_NEGATION_PREFIX)
                and len(stem) > 1
                and _looks_like_negation_stem(stem[1:])
            ):
                _add([_NEGATION_PREFIX, stem[1:], suffix])

    # Negation prefix detach: မ + stem
    if token.startswith(_NEGATION_PREFIX) and len(token) > 1:
        rest = token[1:]
        if _looks_like_negation_stem(rest):
            _add([_NEGATION_PREFIX, rest])
            for suffix in suffixes:
                if rest.endswith(suffix) and len(rest) > len(suffix):
                    _add([_NEGATION_PREFIX, rest[: -len(suffix)], suffix])

    return candidates


def _looks_like_negation_stem(stem: str) -> bool:
    """Heuristic guard to avoid splitting non-negation words starting with မ."""
    if not stem:
        return False
    first = ord(stem[0])
    # Require base consonant as stem head. This blocks words like မောင်...
    # where the next code point is a pre-vowel mark.
    return 0x1000 <= first <= 0x1021


def _generate_syllable_split_candidates(
    token: str,
    *,
    segment_syllables: Callable[[str], list[str]] | None,
    config: TokenRefinementConfig,
) -> list[list[str]]:
    """Generate split candidates by syllable boundaries for merged/OOV tokens."""
    if segment_syllables is None:
        return []
    if len(token) < config.syllable_split_min_token_len:
        return []

    try:
        syllables = [s for s in segment_syllables(token) if s]
    except (RuntimeError, ValueError, TypeError, AttributeError):
        return []

    if len(syllables) < 2 or len(syllables) > config.syllable_split_max_syllables:
        return []

    candidates: list[list[str]] = []
    for split_at in range(1, len(syllables)):
        left = "".join(syllables[:split_at])
        right = "".join(syllables[split_at:])
        if not left or not right:
            continue
        # Avoid degenerate one-char tails from noisy syllable segmentation.
        if len(left) < 2 and len(right) < 2:
            continue
        parts = [left, right]
        if parts not in candidates:
            candidates.append(parts)
    return candidates


def _passes_structure_guard(
    parts: list[str],
    *,
    is_valid_word: Callable[[str], bool] | None,
    get_word_frequency: Callable[[str], int] | None,
) -> bool:
    function_parts = 0
    known_parts = 0
    unknown_content_parts = 0

    for part in parts:
        known = _is_known(part, is_valid_word, get_word_frequency)
        function_like = _is_function_like(part)
        if function_like:
            function_parts += 1
        if known:
            known_parts += 1
        elif not function_like and len(part) >= 2:
            unknown_content_parts += 1

    # Guard 1: only detach around function-like anchors.
    if function_parts == 0:
        return False
    # Guard 2: at most one unknown content segment.
    if unknown_content_parts > 1:
        return False
    return True


def _score_parts(
    parts: list[str],
    *,
    left: str | None,
    right: str | None,
    is_valid_word: Callable[[str], bool] | None,
    get_word_frequency: Callable[[str], int] | None,
    get_bigram_probability: Callable[[str, str], float] | None,
    config: TokenRefinementConfig | None = None,
) -> float:
    cfg = config or _default_config
    score = 0.0

    for part in parts:
        known = _is_known(part, is_valid_word, get_word_frequency)
        freq = _safe_freq(get_word_frequency, part)
        function_like = _is_function_like(part)

        if known or function_like:
            score += cfg.known_part_score
        elif len(part) >= 2:
            score -= cfg.unknown_long_part_penalty
        else:
            score -= 0.75

        if function_like:
            score += cfg.suffix_score_boost
        if freq > 0:
            score += min(math.log10(freq + 1), 5.0) * 0.18

    score -= cfg.split_complexity_penalty * max(0, len(parts) - 1)

    if get_bigram_probability is not None:
        if left:
            score += _scaled_bigram(get_bigram_probability(left, parts[0]), config=cfg)
        if right:
            score += _scaled_bigram(get_bigram_probability(parts[-1], right), config=cfg)
        for first, second in zip(parts, parts[1:], strict=False):
            score += _scaled_bigram(get_bigram_probability(first, second), config=cfg)

    return score


def _score_token_sequence(
    tokens: list[str],
    *,
    is_valid_word: Callable[[str], bool] | None,
    get_word_frequency: Callable[[str], int] | None,
    get_bigram_probability: Callable[[str, str], float] | None,
    config: TokenRefinementConfig | None = None,
) -> float:
    """Score a full token sequence for candidate path selection."""
    cfg = config or _default_config
    score = 0.0
    for idx, token in enumerate(tokens):
        left = tokens[idx - 1] if idx > 0 else None
        right = tokens[idx + 1] if idx + 1 < len(tokens) else None
        score += _score_parts(
            [token],
            left=left,
            right=right,
            is_valid_word=is_valid_word,
            get_word_frequency=get_word_frequency,
            get_bigram_probability=get_bigram_probability,
            config=cfg,
        )
    return score


def _scaled_bigram(probability: float, *, config: TokenRefinementConfig | None = None) -> float:
    if not isinstance(probability, (int, float)) or probability <= 0.0:
        return 0.0
    cfg = config or _default_config
    return min(float(probability) * cfg.bigram_scale, 0.7)


def _is_known(
    token: str,
    is_valid_word: Callable[[str], bool] | None,
    get_word_frequency: Callable[[str], int] | None,
) -> bool:
    if is_valid_word is not None and is_valid_word(token):
        return True
    return _safe_freq(get_word_frequency, token) > 0


def _safe_freq(get_word_frequency: Callable[[str], int] | None, token: str) -> int:
    if get_word_frequency is None:
        return 0
    try:
        value = get_word_frequency(token)
    except (RuntimeError, ValueError, TypeError, AttributeError):
        return 0
    return int(value) if isinstance(value, (int, float)) and value > 0 else 0


def _is_function_like(token: str) -> bool:
    return (
        token == _NEGATION_PREFIX
        or token in _DETACHABLE_SUFFIXES
        or token in _DETACHABLE_SUFFIX_VARIANTS
    )
