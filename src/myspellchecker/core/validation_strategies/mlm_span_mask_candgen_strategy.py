"""MLM span-mask candidate generator.

Wraps the existing ``semantic-v2.4`` RoBERTa MLM as a production
*candidate generator* (not a scorer). For each Myanmar token in the
sentence, masks the token, asks the MLM for its top-K predictions, filters
the predictions to dictionary words within an edit-distance budget of the
typo, and gates the emission on a logit margin ``score(candidate) −
score(typo) ≥ margin``.

Gated by :attr:`ValidationConfig.use_mlm_span_mask_candgen` (default off)
pending a benchmark gate that measures composite + FPR impact.

Priority **46** — after the structural phase and the confusable strategies
at 24/25, but before :class:`ConfusableCompoundClassifierStrategy` (47) so
that the MLM's proposal survives the classifier's gate.
"""

from __future__ import annotations

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
        words = context.words
        if not sentence or not words:
            return []

        errors: list[Error] = []

        for idx, word in enumerate(words):
            if idx >= len(context.word_positions):
                continue
            position = context.word_positions[idx]

            if position in context.existing_errors:
                continue
            if not self._should_probe(word, context, idx):
                continue

            candidate = self._best_candidate(sentence, word)
            if candidate is None:
                continue

            suggestion_text, cand_score, token_score, cand_ed = candidate
            error = WordError(
                text=word,
                position=position,
                error_type=ET_WORD,
                suggestions=[Suggestion(text=suggestion_text, source="mlm_span_mask_candgen")],
                confidence=self.confidence,
            )
            errors.append(error)
            context.existing_errors[position] = ET_WORD
            context.existing_confidences[position] = self.confidence
            context.existing_suggestions[position] = [suggestion_text]
            logger.debug(
                "mlm_span_mask_candgen: %s (score=%.3f) -> %s (score=%.3f) delta=%.3f ed=%d",
                word,
                token_score,
                suggestion_text,
                cand_score,
                cand_score - token_score,
                cand_ed,
            )

        return errors

    def _should_probe(
        self,
        word: str,
        context: ValidationContext,
        idx: int,
    ) -> bool:
        """Return True if the segmented ``word`` is a viable probe target."""
        if not word:
            return False
        if len(word) < self.min_token_length:
            return False
        # Require at least one Myanmar character; Latin / punctuation tokens
        # pass through.
        if not any("\u1000" <= ch <= "\u109f" for ch in word):
            return False
        # Only score real-word confusions — the token must itself be a
        # valid dictionary word. OOV handling is owned by the raw-token
        # probe / SymSpell strategies.
        if not self.provider.is_valid_word(word):
            return False
        # High-frequency tokens are unlikely to be typos; skip to bound FPR.
        token_freq = self.provider.get_word_frequency(word) or 0
        if token_freq > self.skip_above_freq:
            return False
        if is_colloquial_variant(word):
            return False
        # Skip tokens flagged as proper names.
        if context.is_name_mask and idx < len(context.is_name_mask) and context.is_name_mask[idx]:
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

    def __repr__(self) -> str:
        return (
            f"MLMSpanMaskCandGenStrategy(priority={self.priority()}, "
            f"enabled={self.enabled}, top_k={self.top_k}, margin={self.margin}, "
            f"max_edit_distance={self.max_edit_distance})"
        )
