"""MLM span-mask candidate generator.

Wraps the existing ``semantic-v2.4`` RoBERTa MLM as a production
*candidate generator* (not a scorer). For each Myanmar token in the
sentence, masks the token, asks the MLM for its top-K predictions, filters
the predictions to dictionary words within an edit-distance budget of the
typo, and gates the emission on a logit margin ``score(candidate) −
score(typo) ≥ margin``.

The backing probe (2026-04-18) measured:

- 67.9% top-10 recall on the 187 pure real-word-confusion FNs.
- 38.9% top-10 recall on all 864 spelling FNs.
- Sweet spot (K=10, margin=2.0) → **+297 TP at 8% position-level FP rate**,
  Δcomposite +0.039 in simulation.

This strategy is the production realisation of that probe, behind the
``use_mlm_span_mask_candgen`` flag (default off) pending the
``mlm-cg-benchmark-01`` gate.

Priority **46** — after the structural phase and the confusable strategies
at 24/25, but before :class:`ConfusableCompoundClassifierStrategy` (47) so
that the MLM's proposal survives the classifier's gate.

See ``[[MLM Candidate Generation Probe 2026-04-18]]`` and the companion
workstream doc at ``10_Workstreams/Active/mlm-candidate-generator.md``.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from myspellchecker.algorithms.distance.edit_distance import (
    weighted_damerau_levenshtein_distance,
)
from myspellchecker.core.constants import ET_WORD
from myspellchecker.core.response import Error, Suggestion, WordError
from myspellchecker.core.validation_strategies.base import (
    ValidationContext,
    ValidationStrategy,
)
from myspellchecker.text.phonetic_data import is_colloquial_variant
from myspellchecker.utils.logging_utils import get_logger

if TYPE_CHECKING:
    from myspellchecker.algorithms.semantic_checker import SemanticChecker
    from myspellchecker.providers.interfaces import WordRepository

logger = get_logger(__name__)

_PRIORITY = 46

# Myanmar token regex: whitespace/punctuation-delimited Myanmar-script
# spans. Extended-A / B ranges are included so Shan/Mon tokens pass through
# without being attacked. Mirrors the raw-token regex used elsewhere.
_RAW_TOKEN_REGEX = re.compile(r"[\u1000-\u109F\uA9E0-\uA9FF\uAA60-\uAA7F]+")


class MLMSpanMaskCandGenStrategy(ValidationStrategy):
    """MLM-as-candidate-generator over span-masked Myanmar tokens.

    Per-token algorithm:

    1. Iterate raw Myanmar tokens in ``context.sentence`` (``_RAW_TOKEN_REGEX``).
    2. Skip if token is empty, already flagged (``context.existing_errors``),
       not in the dictionary, frequent above ``skip_above_freq``, on the
       colloquial whitelist, or inside a name-masked span.
    3. Call ``semantic_checker.predict_mask(sentence, token, top_k)`` to
       decode the MLM's top candidates at the masked position.
    4. Filter candidates to dictionary words at edit distance in
       ``[1, max_edit_distance]`` of the token.
    5. Score the typo at the same masked position via
       ``semantic_checker.score_mask_candidates(sentence, token, [token])``
       (same logits cache as step 3 — one forward pass per token).
    6. If the best filtered candidate's score exceeds
       ``score(token) + margin``, emit a :class:`WordError` suggesting it.

    Args:
        semantic_checker: Configured :class:`SemanticChecker` instance
            (with a loaded ONNX session). When ``None`` or its session is
            unavailable, the strategy becomes a no-op — graceful
            degradation on systems without the model asset.
        provider: Dictionary provider for ``is_valid_word`` /
            ``get_word_frequency`` lookups.
        enabled: Master on/off switch. Wired from
            :attr:`ValidationConfig.use_mlm_span_mask_candgen`.
        top_k: Number of MLM top predictions to decode per token.
        margin: Required ``score(candidate) − score(typo)`` margin.
        max_edit_distance: Upper bound on accepted edit distance.
        skip_above_freq: Do not probe tokens with frequency above this cap.
        min_token_length: Skip tokens shorter than this. MLM decoding on
            single-char tokens is high-variance and FP-prone.
        confidence: Confidence stamped on emitted errors.
    """

    def __init__(
        self,
        semantic_checker: SemanticChecker | None,
        provider: WordRepository,
        *,
        enabled: bool = False,
        top_k: int = 10,
        margin: float = 2.0,
        max_edit_distance: int = 2,
        skip_above_freq: int = 50_000,
        min_token_length: int = 2,
        confidence: float = 0.75,
    ) -> None:
        self.semantic_checker = semantic_checker
        self.provider = provider
        self.enabled = enabled
        self.top_k = top_k
        self.margin = margin
        self.max_edit_distance = max_edit_distance
        self.skip_above_freq = skip_above_freq
        self.min_token_length = min_token_length
        self.confidence = confidence

    def priority(self) -> int:
        """Return strategy execution priority (46)."""
        return _PRIORITY

    def validate(self, context: ValidationContext) -> list[Error]:
        """Emit :class:`WordError` for tokens where the MLM proposes a better fit."""
        if not self.enabled or self.semantic_checker is None:
            return []

        sentence = context.sentence
        if not sentence:
            return []

        sentence_base = self._resolve_sentence_base(context)
        errors: list[Error] = []

        for match in _RAW_TOKEN_REGEX.finditer(sentence):
            raw_token = match.group(0)
            local_start = match.start()
            local_end = match.end()
            abs_start = sentence_base + local_start

            if abs_start in context.existing_errors:
                continue
            if not self._should_probe(raw_token, context, local_start, local_end):
                continue

            candidate = self._best_candidate(sentence, raw_token)
            if candidate is None:
                continue

            suggestion_text, cand_score, token_score, cand_ed = candidate
            error = WordError(
                text=raw_token,
                position=abs_start,
                error_type=ET_WORD,
                suggestions=[Suggestion(text=suggestion_text, source="mlm_span_mask_candgen")],
                confidence=self.confidence,
            )
            errors.append(error)
            context.existing_errors[abs_start] = ET_WORD
            context.existing_confidences[abs_start] = self.confidence
            context.existing_suggestions[abs_start] = [suggestion_text]
            logger.debug(
                "mlm_span_mask_candgen: %s (score=%.3f) -> %s (score=%.3f) delta=%.3f ed=%d",
                raw_token,
                token_score,
                suggestion_text,
                cand_score,
                cand_score - token_score,
                cand_ed,
            )

        return errors

    def _should_probe(
        self,
        raw_token: str,
        context: ValidationContext,
        local_start: int,
        local_end: int,
    ) -> bool:
        """Return True if ``raw_token`` is a viable probe target."""
        if not raw_token:
            return False
        if len(raw_token) < self.min_token_length:
            return False
        # Only score real-word confusions — the token must itself be a
        # valid dictionary word. OOV handling is owned by the raw-token
        # probe / SymSpell strategies.
        if not self.provider.is_valid_word(raw_token):
            return False
        # High-frequency tokens are unlikely to be typos; skip to bound FPR.
        token_freq = self.provider.get_word_frequency(raw_token) or 0
        if token_freq > self.skip_above_freq:
            return False
        if is_colloquial_variant(raw_token):
            return False
        if self._overlaps_name(context, local_start, local_end):
            return False
        return True

    def _best_candidate(
        self,
        sentence: str,
        raw_token: str,
    ) -> tuple[str, float, float, int] | None:
        """Probe the MLM at ``raw_token``'s position and return the best hit.

        Returns ``(candidate, candidate_score, token_score, edit_distance)``
        or ``None`` when no candidate clears the guards.
        """
        assert self.semantic_checker is not None  # typing guard
        try:
            predictions = self.semantic_checker.predict_mask(sentence, raw_token, top_k=self.top_k)
        except (RuntimeError, ValueError, KeyError):
            logger.debug("MLM predict_mask failed for %r", raw_token, exc_info=True)
            return None

        if not predictions:
            return None

        # Filter predictions to in-dict words within edit-distance budget.
        filtered: list[tuple[str, float, int]] = []
        for cand_word, cand_score in predictions:
            if cand_word == raw_token:
                continue
            if not cand_word:
                continue
            ed = weighted_damerau_levenshtein_distance(raw_token, cand_word)
            if ed == 0 or ed > self.max_edit_distance:
                continue
            if not self.provider.is_valid_word(cand_word):
                continue
            filtered.append((cand_word, float(cand_score), int(ed)))

        if not filtered:
            return None

        # Score the typo at the same position (shares logits cache → no
        # extra forward pass).
        try:
            scores = self.semantic_checker.score_mask_candidates(sentence, raw_token, [raw_token])
        except (RuntimeError, ValueError, KeyError):
            logger.debug("MLM score_mask_candidates failed for %r", raw_token, exc_info=True)
            return None

        token_score = scores.get(raw_token)
        if token_score is None:
            return None

        # Pick highest-scoring filtered candidate; gate on the margin.
        filtered.sort(key=lambda row: row[1], reverse=True)
        best_word, best_score, best_ed = filtered[0]
        if best_score - token_score < self.margin:
            return None
        return best_word, best_score, float(token_score), best_ed

    @staticmethod
    def _resolve_sentence_base(context: ValidationContext) -> int:
        """Return the absolute offset of ``context.sentence`` in the full text.

        Mirrors :meth:`PreSegmenterRawProbeStrategy._resolve_sentence_base`.
        """
        if not context.words or not context.word_positions:
            return 0
        first_local = context.sentence.find(context.words[0]) if context.sentence else 0
        if first_local < 0:
            first_local = 0
        return context.word_positions[0] - first_local

    @staticmethod
    def _overlaps_name(
        context: ValidationContext,
        local_start: int,
        local_end: int,
    ) -> bool:
        """Return True if any name-masked word overlaps ``[local_start, local_end)``."""
        if not context.is_name_mask:
            return False
        sentence_base = MLMSpanMaskCandGenStrategy._resolve_sentence_base(context)
        for idx, word in enumerate(context.words):
            if idx >= len(context.is_name_mask) or not context.is_name_mask[idx]:
                continue
            word_local_start = context.word_positions[idx] - sentence_base
            word_local_end = word_local_start + len(word)
            if word_local_start < local_end and word_local_end > local_start:
                return True
        return False

    def __repr__(self) -> str:
        return (
            f"MLMSpanMaskCandGenStrategy(priority={self.priority()}, "
            f"enabled={self.enabled}, top_k={self.top_k}, margin={self.margin}, "
            f"max_edit_distance={self.max_edit_distance})"
        )
