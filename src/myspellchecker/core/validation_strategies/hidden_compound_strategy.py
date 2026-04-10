"""
Hidden Compound Typo Validation Strategy.

Detects compound-word typos that are hidden by segmenter over-splitting.

Canonical example: the input ``ခုန်ကျစရိတ်`` (intended: ``ကုန်ကျစရိတ်``,
"costs/expenses") is segmented as ``['ခုန်', 'ကျ', 'စရိတ်']`` — each piece is
a valid standalone word, so no upstream strategy flags it. The correct
compound ``ကုန်ကျစရိတ်`` has frequency 27677 in the production dictionary.

The strategy recovers these errors by:

1. Walking token windows where both ``w_i`` and ``w_{i+1}`` are curated
   vocabulary.
2. Generating confusable variants of ``w_i`` via
   :func:`generate_confusable_variants` (phonetic / tonal / medial / nasal /
   stop-coda / stacking / kinzi).
3. Checking whether ``variant + w_{i+1}`` is a high-frequency dictionary
   compound.
4. For ``freq=0`` subsumed compounds (e.g. ``ကုန်ကျ`` is valid but freq=0
   because the corpus only attests the larger ``ကုန်ကျစရိတ်``), extending
   the window by one token and checking the trigram variant.
5. Emitting a multi-token-span :class:`WordError` via
   :data:`~myspellchecker.core.constants.ET_HIDDEN_COMPOUND_TYPO` without
   calling ``_mark_positions`` — downstream strategies keep running at the
   same position; a post-processing suppression rule (to be added in
   Sprint C) handles deduplication.

Priority: **23** (structural phase, before StatisticalConfusable 24 and
BrokenCompound 25, surviving the fast-path cutoff at 25).

See ``~/Documents/myspellchecker/Workstreams/v1.5.0/hidden-compound-typo-plan.md``
for the full plan.
"""

from __future__ import annotations

import functools
import math
from typing import TYPE_CHECKING

from myspellchecker.core.constants import ET_HIDDEN_COMPOUND_TYPO
from myspellchecker.core.response import Error, Suggestion, WordError
from myspellchecker.core.validation_strategies.base import ValidationContext, ValidationStrategy
from myspellchecker.utils.logging_utils import get_logger

if TYPE_CHECKING:
    from myspellchecker.providers.interfaces import WordRepository
    from myspellchecker.text.phonetic import PhoneticHasher

logger = get_logger(__name__)

_PRIORITY = 23

# Typo-prone characters. A candidate token must contain at least one of these
# to be considered as a typo source. This short-circuits tokens composed only
# of particles/vowels/suffixes that are never the subject of confusable typos.
_TYPO_PRONE_CHARS: frozenset[str] = frozenset(
    [
        # Aspirated / unaspirated consonant pairs
        "ခ",
        "က",
        "ဂ",
        "ဃ",
        "ပ",
        "ဖ",
        "ဗ",
        "ဘ",
        "ဒ",
        "ဓ",
        "တ",
        "ထ",
        "ဇ",
        "ဈ",
        "ဆ",
        "စ",
        # Medials
        "ျ",
        "ြ",
        "ွ",
        "ှ",
        # Rhotic / liquid confusion
        "ရ",
        "ယ",
        "လ",
        # Nasal confusion
        "န",
        "မ",
        "ံ",
        # Retroflex / dental confusion
        "ည",
        "ဏ",
        "ဋ",
        "ဍ",
        # Vowel length confusion
        "ိ",
        "ီ",
        "ု",
        "ူ",
        "ေ",
        "ဲ",
        "ော",
        "ို",
    ]
)


class HiddenCompoundStrategy(ValidationStrategy):
    """
    Detect hidden compound typos exposed by segmenter over-splitting.

    See the module docstring for the full algorithm. Sprint B implements
    bigram detection + trigram lookahead via cached variant generation
    and bulk dictionary lookups. Error emission uses a multi-token span
    with :data:`ET_HIDDEN_COMPOUND_TYPO` and does **not** pre-empt
    downstream strategies via ``_mark_positions``.

    Args:
        provider: DictionaryProvider for word / frequency / vocabulary lookups.
        hasher: PhoneticHasher instance (held by the strategy so the
            LRU-cached variant method can key on word-only without making
            the hasher a cache key — PhoneticHasher is not hashable).
        enabled: Master on/off switch (default False during Sprint A rollout).
        max_token_syllables: Maximum syllables in a candidate token.
        max_variants_per_token: Cap on variants per token after sorting.
        compound_min_frequency: Minimum dict frequency to flag a compound.
        confidence_floor: Minimum confidence to emit an error.
        enable_trigram_lookahead: If True, freq=0 subsumed bigram triggers a
            trigram lookahead.
        variant_cache_size: LRU cache size for ``_cached_variants``.
        require_typo_prone_chars: If True, candidate tokens must contain at
            least one character from ``_TYPO_PRONE_CHARS``.
        curated_only: If True, both ``w_i`` and ``w_{i+1}`` must satisfy
            ``provider.is_valid_vocabulary`` (stricter than ``is_valid_word``).
    """

    def __init__(
        self,
        provider: WordRepository,
        hasher: PhoneticHasher | None,
        *,
        enabled: bool = False,
        max_token_syllables: int = 3,
        max_variants_per_token: int = 20,
        compound_min_frequency: int = 100,
        confidence_floor: float = 0.75,
        enable_trigram_lookahead: bool = True,
        variant_cache_size: int = 8192,
        require_typo_prone_chars: bool = True,
        curated_only: bool = True,
    ) -> None:
        self.provider = provider
        self._hasher = hasher
        self.enabled = enabled
        self.max_token_syllables = max_token_syllables
        self.max_variants_per_token = max_variants_per_token
        self.compound_min_frequency = compound_min_frequency
        self.confidence_floor = confidence_floor
        self.enable_trigram_lookahead = enable_trigram_lookahead
        self.variant_cache_size = variant_cache_size
        self.require_typo_prone_chars = require_typo_prone_chars
        self.curated_only = curated_only
        self.logger = logger

        # Lazy-load SyllableTokenizer so tests that mock the provider without
        # a real tokenizer environment still work.
        from myspellchecker.tokenizers.syllable import SyllableTokenizer

        self._syllable_tokenizer = SyllableTokenizer()

        # Bound LRU cache — wraps the instance method so ``self`` is bound and
        # does not appear in the cache key. Keying on ``word`` only avoids the
        # PhoneticHasher hashability problem.
        self._cached_variants = functools.lru_cache(maxsize=self.variant_cache_size)(
            self._compute_variants
        )

    def priority(self) -> int:
        """Return strategy execution priority (23).

        Placed before StatisticalConfusable (24) and BrokenCompound (25) so
        the strategy runs inside the structural phase (priority <= 25) and
        survives the fast-path cutoff at
        :attr:`ContextValidator._FAST_PATH_PRIORITY_CUTOFF`. Hidden-compound
        failures predominantly occur on structurally-clean sentences.
        """
        return _PRIORITY

    # ── Core walk ──────────────────────────────────────────────────────

    def validate(self, context: ValidationContext) -> list[Error]:
        """Walk the token sequence and emit hidden-compound typo errors.

        For each adjacent pair (i, i+1), the typo may live in either token:
        - Forward: the typo is ``w_i`` and the anchor is ``w_{i+1}`` (+optional
          ``w_{i+2}`` for the freq=0 trigram lookahead).
        - Backward: the typo is ``w_{i+1}`` and the anchor is ``w_i``
          (+optional ``w_{i-1}`` for a left-side trigram lookahead, if we can
          prefix the pair with a preceding token).

        Both directions are tested for each pair. The emitted error is placed
        at the typo position (which is the LEFT-most position of the span)
        so downstream UI highlights the actual misspelled start.
        """
        if not self.enabled:
            return []
        if self._hasher is None:
            return []

        words = context.words
        n = len(words)
        if n < 2:
            return []

        errors: list[Error] = []
        emitted_starts: set[int] = set()  # dedup within this strategy

        try:
            for i in range(n - 1):
                pair_start = i
                pair_end = i + 1

                # Both tokens must be at least dictionary-valid words.
                # The typo-side check in _try_direction enforces the stricter
                # curated_only filter on the actual typo source.
                if not (
                    self.provider.is_valid_word(words[pair_start])
                    and self.provider.is_valid_word(words[pair_end])
                ):
                    continue

                # Adjacency check: the two tokens must be contiguous in the
                # source text. A gap (space, punctuation, newline) means the
                # user wrote them as SEPARATE words — merging them into a
                # "hidden compound" is a false positive.
                expected_next = (
                    context.word_positions[pair_start] + len(words[pair_start])
                )
                actual_next = context.word_positions[pair_end]
                if actual_next != expected_next:
                    continue

                # Original-pair attestation: if w_i+w_next is already a
                # high-frequency dictionary word, the segmenter split was
                # cosmetic and nothing is wrong.
                original_bigram = words[pair_start] + words[pair_end]
                if (
                    self.provider.is_valid_word(original_bigram)
                    and self.provider.get_word_frequency(original_bigram)
                    >= self.compound_min_frequency
                ):
                    continue

                # ── Forward direction: typo at pair_start ─────────────
                forward_error = self._try_direction(
                    context=context,
                    typo_idx=pair_start,
                    anchor_idx=pair_end,
                    lookahead_idx=pair_end + 1 if pair_end + 1 < n else None,
                )
                if forward_error is not None and forward_error.position not in emitted_starts:
                    errors.append(forward_error)
                    emitted_starts.add(forward_error.position)
                    continue  # one error per pair; forward took precedence

                # ── Backward direction: typo at pair_end ──────────────
                backward_error = self._try_direction(
                    context=context,
                    typo_idx=pair_end,
                    anchor_idx=pair_start,
                    lookahead_idx=pair_start - 1 if pair_start - 1 >= 0 else None,
                )
                if backward_error is not None and backward_error.position not in emitted_starts:
                    errors.append(backward_error)
                    emitted_starts.add(backward_error.position)
        except (RuntimeError, ValueError, KeyError, IndexError, AttributeError, TypeError) as e:
            self.logger.error(f"Error in hidden compound validation: {e}", exc_info=True)

        return errors

    def _try_direction(
        self,
        *,
        context: ValidationContext,
        typo_idx: int,
        anchor_idx: int,
        lookahead_idx: int | None,
    ) -> WordError | None:
        """Try to flag a hidden compound typo at ``typo_idx``.

        ``anchor_idx`` is the adjacent (always-valid) token that contributes
        its surface form to the compound candidate. ``lookahead_idx``, if
        provided, enables trigram subsumption checks for freq=0 valid dict
        entries — it is the position ON THE ANCHOR SIDE (further from the
        typo), not adjacent to the typo.
        """
        words = context.words
        n = len(words)
        w_typo = words[typo_idx]
        w_anchor = words[anchor_idx]

        # Typo-side gating (character filter, length, curated).
        if not self._is_candidate_token(w_typo):
            return None

        # Skip if already flagged upstream.
        pos_typo = context.word_positions[typo_idx]
        if pos_typo in context.existing_errors:
            return None

        # Determine direction (forward means anchor on the right).
        forward = anchor_idx > typo_idx

        # Subsumed-token check: if w_typo is already part of a valid
        # compound with its OPPOSITE-SIDE neighbor (the one NOT included in
        # our bigram window), we must skip. E.g., for "ကဖေးမှာ" split as
        # ['က','ဖေး','မှာ'], trying (typo='ဖေး', anchor='မှာ') forward
        # should not fire because 'က'+'ဖေး' = 'ကဖေး' is a valid dict
        # compound ("cafe"). The typo-token is already covered by an
        # existing compound and is not a standalone typo source.
        if forward:
            # Anchor is on the right, so check the LEFT neighbor of w_typo.
            neighbor_idx = typo_idx - 1
            if neighbor_idx >= 0:
                neighbor = words[neighbor_idx]
                # Only count if the two are adjacent in source (no gap).
                expected_next = (
                    context.word_positions[neighbor_idx] + len(neighbor)
                )
                if expected_next == pos_typo:
                    subsumed_left = neighbor + w_typo
                    if self.provider.is_valid_word(subsumed_left):
                        return None
        else:
            # Anchor is on the left, so check the RIGHT neighbor of w_typo.
            neighbor_idx = typo_idx + 1
            if neighbor_idx < n:
                neighbor = words[neighbor_idx]
                expected_next = pos_typo + len(w_typo)
                if context.word_positions[neighbor_idx] == expected_next:
                    subsumed_right = w_typo + neighbor
                    if self.provider.is_valid_word(subsumed_right):
                        return None

        variants_list = sorted(self._cached_variants(w_typo))
        if not variants_list:
            return None
        variants_list = variants_list[: self.max_variants_per_token]

        # Build bigram + optional trigram candidates.
        if forward:
            bigram_candidates = [v + w_anchor for v in variants_list]
        else:
            bigram_candidates = [w_anchor + v for v in variants_list]

        bigram_valid = self.provider.is_valid_words_bulk(bigram_candidates)

        lookahead_enabled = self.enable_trigram_lookahead and lookahead_idx is not None
        trigram_candidates: list[str] = []
        if lookahead_enabled and lookahead_idx is not None:
            w_look = words[lookahead_idx]
            if forward:
                # [typo, anchor, lookahead] → variant + anchor + lookahead
                trigram_candidates = [v + w_anchor + w_look for v in variants_list]
            else:
                # [lookahead, anchor, typo] → lookahead + anchor + variant
                trigram_candidates = [w_look + w_anchor + v for v in variants_list]
        trigram_valid = (
            self.provider.is_valid_words_bulk(trigram_candidates) if trigram_candidates else {}
        )

        # best tracks: (compound, evidence_freq, variant, subsumed)
        # subsumed=False: direct bigram hit (gold typically wants minimal
        #                 single-token edit like 'စရာ' → 'ဆရာ')
        # subsumed=True:  freq=0 bigram, verified via trigram lookahead
        #                 (gold typically wants the full bigram like
        #                 'ခုန်ကျ' → 'ကုန်ကျ')
        best: tuple[str, int, str, bool] | None = None
        for idx, v in enumerate(variants_list):
            bi = bigram_candidates[idx]
            if not bigram_valid.get(bi):
                continue
            f = self.provider.get_word_frequency(bi)
            if f >= self.compound_min_frequency:
                best = self._pick_better(best, (bi, f, v, False))
                continue
            # freq=0 subsumed compound → trigram verification.
            if f == 0 and trigram_candidates and idx < len(trigram_candidates):
                tri = trigram_candidates[idx]
                if trigram_valid.get(tri):
                    f2 = self.provider.get_word_frequency(tri)
                    if f2 >= self.compound_min_frequency:
                        best = self._pick_better(best, (bi, f2, v, True))

        if best is None:
            return None

        compound, compound_freq, variant, subsumed = best
        confidence = self._compute_confidence(
            w_i=w_typo,
            variant=variant,
            compound_freq=compound_freq,
            subsumed=subsumed,
        )
        if confidence < self.confidence_floor:
            return None

        # Format S — canonical single-token gold format
        # (decided via /octo:debate Round 3, 2-1 vote):
        # Always emit the minimal-edit single-token variant as the primary
        # suggestion. The full compound is kept as the secondary for users
        # who want to see the context, but benchmark Top1 and user-facing
        # auto-fix target the single-token correction.
        primary, secondary = variant, compound

        return self._build_typo_error(
            context=context,
            typo_idx=typo_idx,
            primary=primary,
            secondary=secondary,
            confidence=confidence,
        )

    # ── Candidate filter ───────────────────────────────────────────────

    def _is_candidate_token(self, word: str) -> bool:
        """Decide whether ``word`` should be tested as a typo source."""
        if not word:
            return False
        if self.curated_only and not self.provider.is_valid_vocabulary(word):
            return False
        # Length cap based on syllable count.
        syllables = self._syllable_tokenizer.tokenize(word)
        if len(syllables) > self.max_token_syllables:
            return False
        # Must contain at least one typo-prone character.
        if self.require_typo_prone_chars and not any(c in _TYPO_PRONE_CHARS for c in word):
            return False
        return True

    # ── Variant cache ──────────────────────────────────────────────────

    def _compute_variants(self, word: str) -> frozenset[str]:
        """Backing method for ``self._cached_variants``.

        Bound instance method so the LRU cache key is ``(word,)`` only —
        ``self`` is part of the bound descriptor and does not enter the key.
        This avoids the PhoneticHasher hashability problem.
        """
        from myspellchecker.core.myanmar_confusables import generate_confusable_variants

        assert self._hasher is not None  # validate() already gated on this
        variants = generate_confusable_variants(word, self._hasher)
        # Drop the original if the generator returned it (belt and braces).
        variants.discard(word)
        return frozenset(variants)

    @staticmethod
    def _pick_better(
        current: tuple[str, int, str, bool] | None,
        candidate: tuple[str, int, str, bool],
    ) -> tuple[str, int, str, bool]:
        if current is None:
            return candidate
        # Prefer higher-frequency evidence.
        if candidate[1] > current[1]:
            return candidate
        return current

    # ── Confidence scoring ─────────────────────────────────────────────

    def _compute_confidence(
        self,
        *,
        w_i: str,
        variant: str,
        compound_freq: int,
        subsumed: bool,
    ) -> float:
        """Confidence score in [0.0, 1.0]."""
        # Log-freq base normalized so 10^5 ≈ 1.0.
        base = min(1.0, math.log10(max(compound_freq, 1)) / 5.0)

        # Edit-distance penalty: prefer edit = 1.
        from myspellchecker.algorithms.distance.edit_distance import (
            weighted_damerau_levenshtein_distance,
        )

        ed = weighted_damerau_levenshtein_distance(w_i, variant)
        if ed <= 1.0:
            ed_penalty = 1.0
        elif ed <= 2.0:
            ed_penalty = 0.85
        else:
            ed_penalty = 0.5

        # Subsumed-compound bonus: trigram verification is stronger evidence
        # than a direct bigram hit.
        subsumed_bonus = 1.1 if subsumed else 1.0

        # Ratio bonus: compound should be more frequent than either unigram.
        ratio_bonus = 1.0
        wi_freq = max(self.provider.get_word_frequency(w_i), 1)
        v_freq = max(self.provider.get_word_frequency(variant), 1)
        if compound_freq / max(wi_freq, v_freq) >= 2.0:
            ratio_bonus = 1.05

        return min(1.0, base * ed_penalty * subsumed_bonus * ratio_bonus)

    # ── Error construction ─────────────────────────────────────────────

    def _build_typo_error(
        self,
        *,
        context: ValidationContext,
        typo_idx: int,
        primary: str,
        secondary: str,
        confidence: float,
    ) -> WordError | None:
        """Build a single-token WordError with dual suggestions.

        The span covers only the mistyped token. The suggestion list has
        two entries:
          1. ``primary`` — the preferred correction (ordering decided by
             the caller based on subsumed vs direct-bigram evidence).
          2. ``secondary`` — the alternative correction, still in the list
             so benchmark matchers that scan top-k can find the gold.
        """
        if typo_idx < 0 or typo_idx >= len(context.word_positions):
            return None

        pos_typo = context.word_positions[typo_idx]
        first_local = context.sentence.find(context.words[0]) if context.words else 0
        sentence_base = context.word_positions[0] - max(first_local, 0)
        w_start = pos_typo - sentence_base
        w_end = w_start + len(context.words[typo_idx])

        if 0 <= w_start < len(context.sentence) and w_end <= len(context.sentence):
            span_text = context.sentence[w_start:w_end]
        else:
            span_text = context.words[typo_idx]

        return WordError(
            text=span_text,
            position=pos_typo,
            error_type=ET_HIDDEN_COMPOUND_TYPO,
            suggestions=[Suggestion(text=primary), Suggestion(text=secondary)],
            confidence=confidence,
        )

    def __repr__(self) -> str:
        return (
            f"HiddenCompoundStrategy(priority={self.priority()}, "
            f"enabled={self.enabled}, "
            f"max_token_syllables={self.max_token_syllables}, "
            f"confidence_floor={self.confidence_floor})"
        )
