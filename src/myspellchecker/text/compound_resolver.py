"""
Compound word resolver for validating productive Myanmar compounds.

Myanmar frequently creates valid words through compounding:
- N+N: ကျောင်း + သား → ကျောင်းသား ("student")
- V+V: စား + သောက် → စားသောက် ("eat and drink")
- N+V, V+N, ADJ+N patterns

This resolver validates OOV words by splitting them into known dictionary
morphemes using dynamic programming for optimal segmentation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from myspellchecker.text.types import DictionaryCheck, FrequencyCheck, POSCheck
from myspellchecker.utils.cache import LRUCache
from myspellchecker.utils.logging_utils import get_logger

if TYPE_CHECKING:
    from myspellchecker.core.config.algorithm_configs import CompoundResolverConfig
    from myspellchecker.segmenters.base import Segmenter

logger = get_logger(__name__)

__all__ = [
    "ALL_ALLOWED_PATTERNS",
    "BLOCKED_PATTERNS",
    "MORPHOTACTIC_BONUSES",
    "CompoundResolver",
    "CompoundSplit",
]

# Default path for morphotactics YAML
DEFAULT_MORPHOTACTICS_PATH = Path(__file__).parent.parent / "rules" / "morphotactics.yaml"


def _load_morphotactics_yaml() -> dict | None:
    """Load morphotactics.yaml if available."""
    try:
        import yaml

        if DEFAULT_MORPHOTACTICS_PATH.exists():
            with open(DEFAULT_MORPHOTACTICS_PATH, encoding="utf-8") as f:
                return yaml.safe_load(f)
    except ImportError:
        pass
    except Exception:
        logger.warning("Failed to load morphotactics YAML", exc_info=True)
    return None


@dataclass(frozen=True)
class CompoundSplit:
    """Result of compound word resolution.

    Attributes:
        word: The original compound word.
        parts: List of morpheme strings the word splits into.
        part_pos: POS tag for each part (None if unknown).
        pattern: Compound pattern string (e.g., "N+N").
        confidence: Confidence score for this split.
        is_valid: Whether this is a valid compound split.
    """

    word: str
    parts: list[str] = field(default_factory=list)
    part_pos: list[str | None] = field(default_factory=list)
    pattern: str = ""
    confidence: float = 0.0
    is_valid: bool = False


# --- Hardcoded defaults (fallback when YAML unavailable) ---
_DEFAULT_ALLOWED_PATTERNS: frozenset[str] = frozenset(
    {
        "N+N",
        "V+V",
        "N+V",
        "V+N",
        "ADJ+N",
        "N+ADJ",
        "ADJ+ADJ",
        "V+ADJ",
        "ADJ+V",
        "ADV+V",
        "TN+N",
        "ADV+N",
        "ADV+ADV",
    }
)

_DEFAULT_MORPHOTACTIC_BONUSES: dict[str, float] = {
    "N+N": 0.10,
    "ADJ+N": 0.08,
    "N+ADJ": 0.08,
    "V+V": 0.05,
    "N+V": 0.03,
    "V+N": 0.03,
    "ADJ+ADJ": 0.05,
    "V+ADJ": 0.03,
    "ADJ+V": 0.05,
    "ADV+V": 0.03,
    "TN+N": 0.05,
    "ADV+N": 0.03,
    "ADV+ADV": 0.05,
}


def _load_patterns_and_bonuses() -> tuple[frozenset[str], frozenset[str], dict[str, float]]:
    """Load compound patterns and bonuses from morphotactics.yaml.

    Returns (allowed_patterns, blocked_patterns, bonuses) with
    hardcoded fallback if YAML is unavailable.
    """
    yaml_data = _load_morphotactics_yaml()
    if yaml_data is None:
        return _DEFAULT_ALLOWED_PATTERNS, frozenset(), dict(_DEFAULT_MORPHOTACTIC_BONUSES)

    # Load allowed patterns from YAML
    allowed = set()
    for entry in yaml_data.get("compound_patterns", []):
        if entry.get("enabled", True):
            allowed.add(entry["pattern"])
    if not allowed:
        allowed = set(_DEFAULT_ALLOWED_PATTERNS)

    # Load blocked patterns
    blocked = frozenset(yaml_data.get("blocked_patterns", []))

    # Load morphotactic bonuses
    bonuses = yaml_data.get("morphotactic_bonuses", None)
    if bonuses is None:
        bonuses = dict(_DEFAULT_MORPHOTACTIC_BONUSES)

    return frozenset(allowed), blocked, bonuses


# Module-level constants loaded from YAML (with hardcoded fallback)
ALL_ALLOWED_PATTERNS, BLOCKED_PATTERNS, MORPHOTACTIC_BONUSES = _load_patterns_and_bonuses()


class CompoundResolver:
    """Resolver for validating productive Myanmar compound words.

    Uses dynamic programming for optimal segmentation into known dictionary
    morphemes with valid POS patterns.

    Args:
        segmenter: Segmenter for syllable segmentation.
        min_morpheme_frequency: Minimum frequency for each morpheme (default: 10).
        max_parts: Maximum number of parts in a compound (default: 4).
        allowed_patterns: Set of allowed POS patterns.
        cache_size: Maximum cache entries (default: 1024).
        config: Optional CompoundResolverConfig for algorithm tuning parameters.
            When provided, overrides min_morpheme_frequency, max_parts, and
            cache_size with config values.
    """

    def __init__(
        self,
        segmenter: Segmenter,
        min_morpheme_frequency: int = 10,
        max_parts: int = 4,
        allowed_patterns: frozenset[str] = ALL_ALLOWED_PATTERNS,
        cache_size: int = 1024,
        config: CompoundResolverConfig | None = None,
    ):
        """Initialize the compound resolver.

        When ``config`` is provided, its values override the corresponding
        positional parameters (min_morpheme_frequency, max_parts, cache_size).
        """
        if config is not None:
            self.min_morpheme_frequency = config.min_morpheme_frequency
            self.max_parts = config.max_parts
            cache_size = config.cache_size
        else:
            self.min_morpheme_frequency = min_morpheme_frequency
            self.max_parts = max_parts

        self.segmenter = segmenter
        self.allowed_patterns = allowed_patterns
        self.blocked_patterns = BLOCKED_PATTERNS
        self.cache_size = cache_size
        self._config = config
        self._cache: LRUCache[CompoundSplit | None] = LRUCache(maxsize=cache_size)

    def resolve(
        self,
        word: str,
        dictionary_check: DictionaryCheck,
        frequency_check: FrequencyCheck,
        pos_check: POSCheck,
    ) -> CompoundSplit | None:
        """Resolve an OOV word into valid compound morphemes.

        Uses DP segmentation to find the optimal split into known morphemes.

        Args:
            word: The OOV word to analyze.
            dictionary_check: Callable returning True if word is in dictionary.
            frequency_check: Callable returning word frequency count.
            pos_check: Callable returning POS tag string or None.

        Returns:
            CompoundSplit if valid compound detected, None otherwise.
        """
        _sentinel = object()
        cached = self._cache.get(word, _sentinel)
        if cached is not _sentinel:
            return cached

        result = self._resolve_impl(word, dictionary_check, frequency_check, pos_check)
        self._cache.set(word, result)
        return result

    def _resolve_impl(
        self,
        word: str,
        dictionary_check: DictionaryCheck,
        frequency_check: FrequencyCheck,
        pos_check: POSCheck,
    ) -> CompoundSplit | None:
        """Segment word into syllables and delegate to DP splitting.

        Returns None if the word has fewer than 2 syllables.
        """
        syllables = self.segmenter.segment_syllables(word)
        n = len(syllables)

        # Need at least 2 syllables for a compound
        if n < 2:
            return None

        return self._dp_split(syllables, dictionary_check, frequency_check, pos_check)

    def _dp_split(
        self,
        syllables: list[str],
        dictionary_check: DictionaryCheck,
        frequency_check: FrequencyCheck,
        pos_check: POSCheck,
    ) -> CompoundSplit | None:
        """Find the optimal compound split using dynamic programming.

        Builds a DP table where ``dp[i]`` stores the best-scoring segmentation
        of ``syllables[0:i]``. For each position *i*, every start position
        *j < i* is tried: the candidate morpheme ``join(syllables[j:i])`` is
        validated against the dictionary, frequency floor, and morphotactic
        compatibility before extending ``dp[j]``.

        Args:
            syllables: Pre-segmented syllable list for the word.
            dictionary_check: Returns True if a morpheme is in the dictionary.
            frequency_check: Returns corpus frequency for a morpheme.
            pos_check: Returns POS tag string or None for a morpheme.

        Returns:
            CompoundSplit if a valid split with 2+ parts is found, else None.
        """
        n = len(syllables)

        # Pre-compute all possible substring concatenations to avoid
        # O(N^2) string creation inside the DP loop.
        substr_cache: dict[tuple[int, int], str] = {}
        for i in range(n + 1):
            accumulated = ""
            for j in range(i - 1, -1, -1):
                accumulated = syllables[j] + accumulated
                substr_cache[(j, i)] = accumulated

        # Local memoization to avoid redundant provider calls in the DP loop.
        # The same candidate morpheme may be checked from multiple DP states.
        dict_memo: dict[str, bool] = {}
        freq_memo: dict[str, int] = {}
        pos_memo: dict[str, str | None] = {}

        def _dict_check(candidate: str) -> bool:
            """Memoized dictionary validity check."""
            if candidate not in dict_memo:
                dict_memo[candidate] = dictionary_check(candidate)
            return dict_memo[candidate]

        def _freq_check(candidate: str) -> int:
            """Memoized frequency lookup."""
            if candidate not in freq_memo:
                freq_memo[candidate] = frequency_check(candidate)
            return freq_memo[candidate]

        def _pos_check(candidate: str) -> str | None:
            """Memoized POS tag lookup."""
            if candidate not in pos_memo:
                pos_memo[candidate] = pos_check(candidate)
            return pos_memo[candidate]

        # dp[i] = (score, parts, part_pos, freqs) or None if no valid split ending at i
        # dp[0] = empty split (base case)
        dp: list[tuple[float, list[str], list[str | None], list[int]] | None] = [None] * (n + 1)
        dp[0] = (0.0, [], [], [])

        for i in range(1, n + 1):
            for j in range(max(0, i - n), i):  # All start positions
                if dp[j] is None:
                    continue

                dp_j = dp[j]
                assert dp_j is not None
                prev_score, prev_parts, prev_pos, prev_freqs = dp_j

                # Check max parts constraint
                if len(prev_parts) >= self.max_parts:
                    continue

                # Look up pre-computed candidate morpheme for syllables[j:i]
                candidate = substr_cache[(j, i)]

                # Candidate must be in dictionary
                if not _dict_check(candidate):
                    continue

                # Must meet frequency floor
                freq = _freq_check(candidate)
                if freq < self.min_morpheme_frequency:
                    continue

                # Get POS tag
                pos = _pos_check(candidate)

                # Check morphotactic compatibility with previous part
                if prev_pos:
                    last_pos = prev_pos[-1]
                    pattern = self._make_pattern(last_pos, pos)

                    # Blocked patterns are always rejected
                    if pattern in self.blocked_patterns:
                        continue

                    # Must match an allowed pattern
                    if pattern not in self.allowed_patterns:
                        continue

                # Compute score for this extension
                freq_score = math.log1p(freq)

                # Morphotactic bonus
                morph_bonus = 0.0
                if prev_pos:
                    last_pos = prev_pos[-1]
                    pattern = self._make_pattern(last_pos, pos)
                    morph_bonus = MORPHOTACTIC_BONUSES.get(pattern, 0.0)

                # Fewer-parts penalty (prefer 2 parts over 3)
                # Each additional part beyond the first gets a significant penalty
                penalty_mult = (
                    self._config.parts_penalty_multiplier if self._config is not None else 2.0
                )
                parts_penalty = penalty_mult * len(prev_parts)

                score = prev_score + freq_score + morph_bonus - parts_penalty

                new_parts = prev_parts + [candidate]
                new_pos = prev_pos + [pos]
                new_freqs = prev_freqs + [freq]

                # Update dp[i] if this is better than current
                if dp[i] is None or score > dp[i][0]:  # type: ignore[index]
                    dp[i] = (score, new_parts, new_pos, new_freqs)

        # Extract result from dp[n]
        if dp[n] is None:
            return None

        dp_n = dp[n]
        assert dp_n is not None
        _, parts, part_pos, freqs = dp_n

        # Must have at least 2 parts to be a compound
        if len(parts) < 2:
            return None

        # Build pattern string
        pattern = "+".join(p if p else "?" for p in part_pos)

        return CompoundSplit(
            word="".join(syllables),
            parts=parts,
            part_pos=part_pos,
            pattern=pattern,
            confidence=self._compute_confidence(*freqs),
            is_valid=True,
        )

    def _make_pattern(self, left_pos: str | None, right_pos: str | None) -> str:
        """Create a pattern string from two POS tags (e.g., "N+V")."""
        left = left_pos if left_pos else "?"
        right = right_pos if right_pos else "?"
        return f"{left}+{right}"

    def _compute_confidence(self, *frequencies: int) -> float:
        """Compute confidence score based on morpheme frequencies.

        Higher minimum frequency across parts yields higher confidence.
        Extra parts beyond two incur a small penalty.

        Returns:
            Confidence score clamped to [0.5, config.confidence_cap].
        """
        if not frequencies:
            return 0.0

        cfg = self._config
        # Base confidence
        base = cfg.base_confidence if cfg is not None else 0.85
        # Boost for high-frequency morphemes
        min_freq = min(frequencies)
        high_thresh = cfg.high_freq_threshold if cfg is not None else 100
        med_thresh = cfg.medium_freq_threshold if cfg is not None else 50
        high_boost = cfg.high_freq_boost if cfg is not None else 0.05
        med_boost = cfg.medium_freq_boost if cfg is not None else 0.03
        extra_pen = cfg.extra_parts_penalty if cfg is not None else 0.05

        cap_high = cfg.confidence_cap if cfg is not None else 0.95
        cap_mid = cfg.confidence_cap_mid if cfg is not None else 0.93

        if min_freq >= high_thresh:
            base = min(base + high_boost, cap_high)
        elif min_freq >= med_thresh:
            base = min(base + med_boost, cap_mid)
        # Penalty for many parts
        if len(frequencies) > 2:
            base -= extra_pen * (len(frequencies) - 2)
        return max(base, 0.5)
