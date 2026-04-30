"""
Broken Compound Validation Strategy.

Detects compound words that were incorrectly split by a space.
For example: "မနက် ဖြန်" should be "မနက်ဖြန်" (tomorrow).

This is the inverse of MergedWordChecker, which detects wrongly-merged words.

The strategy uses a two-layer approach:
1. **Morphological validation** — curated rules from compound_morphology.yaml
   provide high-confidence detection of mandatory compounds and suppress
   false positives from known verb+particle sequences.
2. **Frequency heuristic** — statistical fallback that checks if the
   concatenated form is significantly more common than the rarer component.

Priority: 25 (after SyntacticValidation 20, before POS 30)
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from myspellchecker.core.config.algorithm_configs import BrokenCompoundStrategyConfig
from myspellchecker.core.constants import ET_BROKEN_COMPOUND
from myspellchecker.core.response import Error, WordError
from myspellchecker.core.validation_strategies.base import ValidationContext, ValidationStrategy
from myspellchecker.utils.logging_utils import get_logger

if TYPE_CHECKING:
    from myspellchecker.providers.interfaces import WordRepository

logger = get_logger(__name__)

_COMPOUND_MORPHOLOGY_PATH = (
    Path(__file__).resolve().parent.parent.parent / "rules" / "compound_morphology.yaml"
)

# Closed-class title/role suffixes in Myanmar.
# When these follow a role/title noun and the concatenation is a valid
# dictionary word, the pair is a mandatory compound.
# Excludes ambiguous suffixes (သူ=pronoun, မ=negation prefix).
_TITLE_SUFFIXES: frozenset[str] = frozenset(
    {
        "ကြီး",  # senior/chief (ဝန်ကြီး, ဗိုလ်ကြီး, အုပ်ကြီး)
        "ငယ်",  # junior/small (လူငယ်, ဗိုလ်ငယ်)
        "သား",  # male/son (ကျောင်းသား, စစ်သား)
        "ဝန်",  # minister/official (ဆရာဝန်)
        "တော်",  # royal (မြို့တော်, ဘုရားတော်)
        "တန်း",  # rank/class (အထက်တန်း, အလယ်တန်း)
        "မှူး",  # head/chief (ရဲမှူး, အုပ်ချုပ်ရေးမှူး)
    }
)

_TITLE_SUFFIX_CONFIDENCE = 0.88

# Module-level cache for compound morphology data (loaded once, shared).
_morphology_data: _CompoundMorphologyData | None = None
_morphology_lock = threading.Lock()


class _CompoundMorphologyData:
    """Parsed and indexed compound morphology rules for O(1) lookup."""

    __slots__ = (
        "mandatory_by_parts",
        "false_compound_keys",
        "a_prefix_pattern_enabled",
    )

    def __init__(self, raw: dict[str, Any]) -> None:
        # mandatory_compounds: map (part1, part2) -> compound string
        self.mandatory_by_parts: dict[tuple[str, str], str] = {}
        for entry in raw.get("mandatory_compounds", []):
            parts = entry.get("parts", [])
            compound = entry.get("compound", "")
            if len(parts) == 2 and compound:
                self.mandatory_by_parts[(parts[0], parts[1])] = compound

        # false_compounds: set of (part1, part2) tuples to suppress
        self.false_compound_keys: set[tuple[str, str]] = set()
        for entry in raw.get("false_compounds", []):
            parts = entry.get("parts", [])
            if len(parts) == 2:
                self.false_compound_keys.add((parts[0], parts[1]))

        # compound_patterns: check for အ-prefix nominalization pattern
        self.a_prefix_pattern_enabled = False
        for pat in raw.get("compound_patterns", []):
            if pat.get("pattern_id") == "a_prefix_nominalization":
                self.a_prefix_pattern_enabled = True
                break


def _load_morphology_data() -> _CompoundMorphologyData | None:
    """Load and cache compound morphology data from YAML (thread-safe)."""
    global _morphology_data
    if _morphology_data is not None:
        return _morphology_data
    with _morphology_lock:
        if _morphology_data is not None:
            return _morphology_data
        if not _COMPOUND_MORPHOLOGY_PATH.exists():
            logger.debug(
                "Compound morphology rules not found: %s",
                _COMPOUND_MORPHOLOGY_PATH,
            )
            return None
        try:
            with open(_COMPOUND_MORPHOLOGY_PATH, encoding="utf-8") as f:
                raw = yaml.safe_load(f)
            if isinstance(raw, dict):
                _morphology_data = _CompoundMorphologyData(raw)
                logger.debug(
                    "Loaded compound morphology: %d mandatory, %d false",
                    len(_morphology_data.mandatory_by_parts),
                    len(_morphology_data.false_compound_keys),
                )
                return _morphology_data
        except (yaml.YAMLError, OSError) as e:
            logger.warning(
                "Failed to load compound morphology from %s: %s",
                _COMPOUND_MORPHOLOGY_PATH,
                e,
            )
    return None


class BrokenCompoundStrategy(ValidationStrategy):
    """
    Detect compound words that were incorrectly split by a space.

    When adjacent words W_i and W_{i+1} concatenate to a valid dictionary word
    that is much more common than either individual word, this likely indicates
    a broken compound. The strategy flags the rarer component and suggests
    the merged compound form.

    Priority: 25
    """

    # Higher confidence for morphology-backed detections vs frequency heuristic.
    _MORPHOLOGY_CONFIDENCE = 0.92

    def __init__(
        self,
        provider: "WordRepository",
        rare_threshold: int | None = None,
        compound_min_frequency: int | None = None,
        compound_ratio: float | None = None,
        confidence: float | None = None,
        config: BrokenCompoundStrategyConfig | None = None,
    ):
        """
        Initialize broken compound strategy.

        Args:
            provider: Word repository with is_valid_word and get_word_frequency.
            rare_threshold: Maximum frequency for a word to be considered "rare".
                Words with frequency below this are candidates for compound merging.
            compound_min_frequency: Minimum frequency for the compound to be flagged.
                Prevents merging into rare compounds.
            compound_ratio: Minimum ratio of compound_freq / rare_word_freq.
                Higher values = more conservative (fewer flags).
            confidence: Confidence score for broken compound errors.
            config: BrokenCompoundStrategyConfig with all thresholds.
                When provided, its values are used as defaults that explicit
                kwargs can override.
        """
        self._config = config or BrokenCompoundStrategyConfig()
        self.provider = provider
        self.rare_threshold = (
            rare_threshold if rare_threshold is not None else self._config.rare_threshold
        )
        self.compound_min_frequency = (
            compound_min_frequency
            if compound_min_frequency is not None
            else self._config.compound_min_frequency
        )
        self.compound_ratio = (
            compound_ratio if compound_ratio is not None else self._config.compound_ratio
        )
        self.confidence = confidence if confidence is not None else self._config.confidence
        self._both_high_freq = self._config.both_high_freq
        self._min_compound_len = self._config.min_compound_len
        self.logger = logger
        self._morphology = _load_morphology_data()

    def validate(self, context: ValidationContext) -> list[Error]:
        """Validate word pairs for broken compound errors."""
        if len(context.words) < 2:
            return []

        if not hasattr(self.provider, "is_valid_word") or not hasattr(
            self.provider, "get_word_frequency"
        ):
            return []

        # POS tags for V+particle detection (generalizes false_compound_keys)
        has_pos = bool(context.pos_tags) and len(context.pos_tags) == len(context.words)

        errors: list[Error] = []

        try:
            for i in range(len(context.words) - 1):
                # Skip if either position is already flagged
                pos_i = context.word_positions[i]
                pos_next = context.word_positions[i + 1]
                if pos_i in context.existing_errors or pos_next in context.existing_errors:
                    continue

                w1 = context.words[i]
                w2 = context.words[i + 1]

                # Skip names — unless morphology confirms a mandatory compound
                if context.is_name_mask[i] or context.is_name_mask[i + 1]:
                    morpho_override = self._check_morphology(w1, w2)
                    if morpho_override is None or morpho_override == "false_compound":
                        continue

                # Skip Pali/Sanskrit stacking fragments (virama U+1039)
                # The segmenter splits stacking words like ဗုဒ္ဓ into fragments
                # that falsely appear as broken compounds
                if "\u1039" in w1 or "\u1039" in w2:
                    continue

                # Guard: skip if the two "words" are actually adjacent in the
                # original text with no whitespace between them. This happens
                # when the segmenter splits a valid compound like ကျောင်းသား
                # into its component syllables. The user didn't insert a space,
                # so there is no broken compound to fix.
                if pos_next == pos_i + len(w1):
                    continue

                # ── Reduplication guard ──
                # Same word repeated with space (e.g. "တိုင် တိုင်") is
                # emphatic/adverbial reduplication, not a broken compound.
                if w1 == w2:
                    continue

                # ── POS-based V+particle detection ──
                # Generalizes false_compound_keys to ALL verb+particle
                # combinations. POS tags use: PART (particle), PPM
                # (postpositional marker). Blocklist always runs as backup
                # to guard against POS tagger errors (~15%).
                if has_pos and context.pos_tags[i + 1] in ("PART", "PPM"):
                    continue

                # ── Layer 1: Morphological validation (curated rules) ──
                morpho_result = self._check_morphology(w1, w2)
                if morpho_result == "false_compound":
                    # Known verb+particle or similar — skip entirely
                    continue
                if morpho_result is not None:
                    error = self._build_error(
                        context, i, w1, w2, morpho_result, self._MORPHOLOGY_CONFIDENCE
                    )
                    if error is not None:
                        errors.append(error)
                        self._mark_positions(
                            context,
                            pos_i,
                            pos_next,
                            morpho_result,
                            self._MORPHOLOGY_CONFIDENCE,
                        )
                    continue

                # ── Layer 1b: Title/suffix pattern (closed-class) ──
                # When w2 is a title suffix and concatenation is valid,
                # this is a mandatory compound regardless of frequency.
                if w2 in _TITLE_SUFFIXES:
                    compound_ts = w1 + w2
                    if hasattr(self.provider, "is_valid_word") and self.provider.is_valid_word(
                        compound_ts
                    ):
                        error = self._build_error(
                            context, i, w1, w2, compound_ts, _TITLE_SUFFIX_CONFIDENCE
                        )
                        if error is not None:
                            errors.append(error)
                            self._mark_positions(
                                context,
                                pos_i,
                                pos_next,
                                compound_ts,
                                _TITLE_SUFFIX_CONFIDENCE,
                            )
                        continue

                # ── Layer 2: Frequency heuristic (statistical fallback) ──

                # Both must be valid individual words (we're detecting split, not typo)
                if not self.provider.is_valid_word(w1) or not self.provider.is_valid_word(w2):
                    continue

                # At least one must be rare
                freq1 = self.provider.get_word_frequency(w1)
                freq2 = self.provider.get_word_frequency(w2)
                rare_freq = min(freq1, freq2)
                if rare_freq >= self.rare_threshold:
                    continue

                # Both-high-freq guard: when both tokens are well-established
                # multi-syllable compounds (freq >= both_high_freq, len >= min_compound_len),
                # their adjacency is intentional — don't suggest merging.
                if (
                    freq1 >= self._both_high_freq
                    and freq2 >= self._both_high_freq
                    and len(w1) >= self._min_compound_len
                    and len(w2) >= self._min_compound_len
                ):
                    continue

                # Guard: zero-frequency but valid words are curated dictionary
                # entries (e.g., adjective stems like လှပ). These are valid
                # standalone forms, not segmenter artifacts. Don't flag them
                # as broken compounds — their adjacency with particles like
                # သော is natural grammar, not a space error.
                if rare_freq == 0:
                    continue

                compound = w1 + w2
                if not self.provider.is_valid_word(compound):
                    continue

                compound_freq = self.provider.get_word_frequency(compound)
                if compound_freq < self.compound_min_frequency:
                    continue

                # Compound must be significantly more common than the rare word
                if rare_freq > 0 and compound_freq / rare_freq < self.compound_ratio:
                    continue

                error = self._build_error(context, i, w1, w2, compound, self.confidence)
                if error is not None:
                    errors.append(error)
                    self._mark_positions(context, pos_i, pos_next, compound, self.confidence)

        except (RuntimeError, ValueError, KeyError, IndexError, AttributeError, TypeError) as e:
            self.logger.error(f"Error in broken compound validation: {e}", exc_info=True)

        return errors

    # ── Morphological helpers ──

    def _check_morphology(self, w1: str, w2: str) -> str | None:
        """Check morphological rules for a word pair.

        Returns:
            - The compound string if (w1, w2) is a mandatory compound.
            - ``"false_compound"`` if the pair should be suppressed.
            - ``None`` if no morphological rule applies (fall through to
              frequency heuristic).
        """
        if self._morphology is None:
            return None

        pair = (w1, w2)

        # 1. False compound suppression (verb+particle, etc.)
        if pair in self._morphology.false_compound_keys:
            return "false_compound"

        # 2. Mandatory compound lookup
        compound = self._morphology.mandatory_by_parts.get(pair)
        if compound is not None:
            return compound

        # 3. Productive pattern: အ-prefix nominalization
        #    If w1 == "အ" and w2 is a valid word, the combined form is
        #    a nominalization that must stay joined.
        if (
            self._morphology.a_prefix_pattern_enabled
            and w1 == "\u1021"  # အ
            and hasattr(self.provider, "is_valid_word")
            and self.provider.is_valid_word(w1 + w2)
        ):
            return w1 + w2

        return None

    def _build_error(
        self,
        context: ValidationContext,
        i: int,
        w1: str,
        w2: str,
        compound: str,
        confidence: float,
    ) -> WordError | None:
        """Build a WordError spanning both words of a broken compound."""
        pos_i = context.word_positions[i]
        # Convert absolute word_positions to sentence-local offsets.
        # word_positions stores sentence_offset + local_idx, so derive
        # the base offset from the first word.
        first_local = context.sentence.find(context.words[0])
        sentence_base = context.word_positions[0] - max(first_local, 0)
        w1_local = pos_i - sentence_base
        if i + 1 < len(context.word_positions):
            w2_local = context.word_positions[i + 1] - sentence_base
            end = w2_local + len(w2)
            if 0 <= w1_local < len(context.sentence) and end <= len(context.sentence):
                span_text = context.sentence[w1_local:end]
            else:
                span_text = w1 + " " + w2
        else:
            span_text = w1 + " " + w2

        return WordError(
            text=span_text,
            position=pos_i,
            error_type=ET_BROKEN_COMPOUND,
            suggestions=[compound],
            confidence=confidence,
        )

    @staticmethod
    def _mark_positions(
        context: ValidationContext,
        pos_i: int,
        pos_next: int,
        compound: str,
        confidence: float,
    ) -> None:
        """Mark both word positions as flagged to prevent downstream duplicates."""
        context.existing_errors[pos_i] = ET_BROKEN_COMPOUND
        context.existing_suggestions[pos_i] = [compound]
        context.existing_confidences[pos_i] = confidence
        context.existing_errors[pos_next] = ET_BROKEN_COMPOUND

    def priority(self) -> int:
        """Return strategy execution priority (25)."""
        return 25

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"BrokenCompoundStrategy(priority={self.priority()}, "
            f"rare_threshold={self.rare_threshold}, ratio={self.compound_ratio}x)"
        )
