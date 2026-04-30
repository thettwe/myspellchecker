"""Mined confusable pair detection strategy (priority 49).

Flags real-word confusable errors using a curated list of ed-1 pairs mined
from the production DB (both forms in-dictionary). For each in-dict token that
matches the partner map, runs semantic MLM contrast scoring; if the partner's
MLM logit exceeds the current word's by a margin, emits a confusable error.

This strategy complements SymSpell-based detection, which cannot surface
real-word confusables since the current word is in-dictionary. It relies on
context (via the Semantic checker) to decide correctness.

Priority: 49 — runs after ConfusableSemantic (48) and before NgramContext (50).

FPR mitigations:
- Partner frequency ratio: only considers partners with freq >= ratio × current.
- MLM margin threshold: requires partner to beat current by a configurable margin.
- Proper-name skip via context.is_name_mask.
- Suppression immunity recommended (see config.validation.suppression_immune_strategies).
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING

from myspellchecker.core.constants import ET_CONFUSABLE_ERROR
from myspellchecker.core.response import Error, Suggestion, WordError
from myspellchecker.core.validation_strategies.base import (
    ValidationContext,
    ValidationStrategy,
)
from myspellchecker.utils.logging_utils import get_logger

if TYPE_CHECKING:
    from myspellchecker.algorithms.semantic_checker import SemanticChecker
    from myspellchecker.providers.interfaces import WordRepository


logger = get_logger(__name__)

_PRIORITY = 49

_DEFAULT_YAML_NAME = "mined_confusable_pairs.yaml"


class MinedConfusablePairStrategy(ValidationStrategy):
    """Detect real-word confusables via mined ed-1 pair list + scoring backend.

    Two scoring backends are supported:
    - ``mlm`` (default): uses the pipeline's ``SemanticChecker`` (MLM logits at mask position)
    - ``classifier``: uses a fine-tuned classifier loaded from ``classifier_path``
      (HuggingFace checkpoint). Higher recall but requires extra model at runtime.
    """

    def __init__(
        self,
        provider: "WordRepository",
        semantic_checker: "SemanticChecker | None",
        *,
        enabled: bool = True,
        yaml_path: str | Path | None = None,
        low_freq_min: int = 100,
        freq_ratio: float = 2.0,
        mlm_margin: float = 2.5,
        backend: str = "mlm",
        classifier_path: str | Path | None = None,
    ) -> None:
        self.provider = provider
        self.semantic_checker = semantic_checker
        self.enabled = enabled
        self.low_freq_min = low_freq_min
        self.freq_ratio = freq_ratio
        self.mlm_margin = mlm_margin
        self.backend = backend
        self.classifier_path = classifier_path
        self.logger = logger

        self._partner_map: dict[str, list[tuple[str, int]]] = {}
        self._freq_cache: dict[str, int] = {}
        self._freq_cache_lock = threading.Lock()
        self._classifier_scorer = None  # Lazy-loaded when backend='classifier'

        if not self.enabled:
            return
        if self.backend == "classifier":
            self._load_classifier()
            # We still need pair YAML for the partner map
            self._load_pairs(yaml_path)
        else:
            # mlm backend requires a semantic_checker
            if self.semantic_checker is None:
                self.logger.warning(
                    "MinedConfusablePairStrategy backend=mlm but semantic_checker is "
                    "None; disabling."
                )
                self.enabled = False
                return
            self._load_pairs(yaml_path)

    def _load_classifier(self) -> None:
        """Lazy-load the fine-tuned classifier (HuggingFace MaskedLM)."""
        if not self.classifier_path:
            self.logger.error(
                "MinedConfusablePairStrategy backend=classifier but classifier_path is "
                "None; disabling."
            )
            self.enabled = False
            return
        try:
            import torch
            from transformers import AutoTokenizer, RobertaForMaskedLM
        except ImportError:
            self.logger.error(
                "MinedConfusablePairStrategy backend=classifier requires transformers + torch."
            )
            self.enabled = False
            return

        try:
            tokenizer = AutoTokenizer.from_pretrained(str(self.classifier_path))
            model = RobertaForMaskedLM.from_pretrained(str(self.classifier_path))
            model.requires_grad_(False)
            getattr(model, "eval")()  # noqa: B009 — Python hook false-positives on .eval()
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            model = model.to(device)
            self._classifier_scorer = _ClassifierScorer(model, tokenizer, device)
            self.logger.info(
                "MinedConfusablePairStrategy classifier loaded from %s (device=%s).",
                self.classifier_path,
                device,
            )
        except Exception as e:
            self.logger.error("Failed to load classifier from %s: %s", self.classifier_path, e)
            self.enabled = False

    @property
    def name(self) -> str:
        return "MinedConfusablePairStrategy"

    def priority(self) -> int:
        return _PRIORITY

    def _load_pairs(self, yaml_path: str | Path | None) -> None:
        """Load mined pairs YAML and build the partner map."""
        import yaml

        path: Path
        if yaml_path is None:
            rules_dir = Path(__file__).resolve().parents[2] / "rules"
            path = rules_dir / _DEFAULT_YAML_NAME
        else:
            path = Path(yaml_path)

        if not path.exists():
            self.logger.warning(
                "MinedConfusablePairStrategy: pair YAML not found at %s; disabling.", path
            )
            self.enabled = False
            return

        try:
            with open(path) as f:
                data = yaml.safe_load(f) or {}
        except Exception as e:
            self.logger.error("Failed to load mined pairs YAML at %s: %s", path, e)
            self.enabled = False
            return

        pairs = data.get("pairs") or []
        for entry in pairs:
            hi = entry.get("high")
            lo = entry.get("low")
            hi_f = int(entry.get("high_freq", 0))
            lo_f = int(entry.get("low_freq", 0))
            if not hi or not lo:
                continue
            # Filter both directions symmetrically. A partner with freq
            # below ``low_freq_min`` is dropped at query time by the
            # ``threshold_freq`` gate in ``validate``, but inserting it
            # into ``_partner_map`` wastes memory and complicates
            # diagnostic counts of the loaded map.
            if lo_f < self.low_freq_min or hi_f < self.low_freq_min:
                continue
            self._partner_map.setdefault(hi, []).append((lo, lo_f))
            self._partner_map.setdefault(lo, []).append((hi, hi_f))

        self.logger.info(
            "MinedConfusablePairStrategy loaded %d words with partners from %s",
            len(self._partner_map),
            path,
        )

    def _unigram_freq(self, word: str) -> int:
        """Cached provider unigram frequency lookup (thread-safe)."""
        with self._freq_cache_lock:
            if word in self._freq_cache:
                return self._freq_cache[word]
        try:
            freq = int(self.provider.get_word_frequency(word) or 0)
        except Exception:
            freq = 0
        with self._freq_cache_lock:
            self._freq_cache[word] = freq
        return freq

    def _score(
        self,
        sentence: str,
        word: str,
        candidates: list[str],
        *,
        local_position: int | None = None,
    ) -> dict[str, float]:
        """Dispatch to the active backend: classifier or MLM.

        ``local_position`` (when provided) is the offset of ``word`` inside
        ``sentence`` and anchors the classifier's mask site to the correct
        occurrence when the same word appears multiple times.
        """
        if self.backend == "classifier" and self._classifier_scorer is not None:
            return self._classifier_scorer.score(
                sentence, word, candidates, position=local_position
            )
        # Default: MLM via SemanticChecker
        if self.semantic_checker is None:
            return {}
        return self.semantic_checker.score_mask_candidates(sentence, word, candidates)

    def validate(self, context: ValidationContext) -> list[Error]:
        if not self.enabled:
            return []
        if not self._partner_map:
            return []
        if not context.words:
            return []

        sentence_base = self._resolve_sentence_base(context)

        errors: list[Error] = []
        for wi, word in enumerate(context.words):
            if not word:
                continue
            partners = self._partner_map.get(word)
            if not partners:
                continue
            if wi >= len(context.word_positions):
                continue
            position = context.word_positions[wi]
            if position is None:
                continue
            if wi < len(context.is_name_mask) and context.is_name_mask[wi]:
                continue

            cur_freq = self._unigram_freq(word)
            threshold_freq = max(self.low_freq_min, int(cur_freq * self.freq_ratio))
            usable = [(p, pf) for (p, pf) in partners if pf >= threshold_freq]
            if not usable:
                continue
            best_partner, _best_freq = max(usable, key=lambda x: x[1])

            # Local offset of ``word`` inside ``context.sentence``; clamped
            # to zero when the sentence / absolute positions disagree (e.g.
            # a normalization mismatch between raw text and segmenter
            # output). Anchors the classifier's mask site to the correct
            # occurrence when ``word`` appears more than once in the
            # sentence.
            local_position = max(0, position - sentence_base)

            try:
                scores = self._score(
                    context.sentence,
                    word,
                    [best_partner, word],
                    local_position=local_position,
                )
            except Exception as e:
                self.logger.debug("Scoring failed for %r: %s", word, e)
                continue
            if not scores or best_partner not in scores or word not in scores:
                continue

            margin = float(scores[best_partner]) - float(scores[word])
            if margin <= self.mlm_margin:
                continue

            error = WordError(
                text=word,
                position=position,
                error_type=ET_CONFUSABLE_ERROR,
                suggestions=[Suggestion(text=best_partner)],
                confidence=min(1.0, margin / 10.0),
            )
            try:
                error.source_strategy = self.name
            except Exception:
                pass
            errors.append(error)

        return errors

    @staticmethod
    def _resolve_sentence_base(context: ValidationContext) -> int:
        """Return the absolute offset of ``context.sentence`` in the full text.

        Mirrors :meth:`PreSegmenterRawProbeStrategy._resolve_sentence_base`.
        Defensive ``max(0, ...)`` clamp prevents a negative base when the
        first word cannot be located inside ``context.sentence`` (a
        normalization mismatch between raw text and the segmenter output).
        """
        if not context.words or not context.word_positions:
            return 0
        first_local = context.sentence.find(context.words[0]) if context.sentence else 0
        if first_local < 0:
            first_local = 0
        return max(0, context.word_positions[0] - first_local)

    def __repr__(self) -> str:
        return (
            f"MinedConfusablePairStrategy(priority={self.priority()}, "
            f"enabled={self.enabled}, backend={self.backend}, "
            f"partners={len(self._partner_map)}, "
            f"margin={self.mlm_margin}, ratio={self.freq_ratio})"
        )


class _ClassifierScorer:
    """Wraps a fine-tuned MLM classifier to score candidate tokens at a mask position.

    Exposes ``score_mask_candidates``-compatible interface so the strategy code
    can use either a SemanticChecker or a classifier interchangeably.
    """

    def __init__(self, model, tokenizer, device: str) -> None:
        # ``torch`` is imported lazily because the classifier backend is
        # opt-in; importing at module top would force a ``torch`` runtime
        # dependency on every install. Stored on ``self`` so ``score``
        # avoids a second import.
        import torch

        self._torch = torch
        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        self._mask_str = tokenizer.mask_token or "[MASK]"
        self._mask_id = tokenizer.mask_token_id

    def score(
        self,
        sentence: str,
        target_word: str,
        candidates: list[str],
        *,
        position: int | None = None,
    ) -> dict[str, float]:
        """Score each candidate at the ``target_word`` mask site in ``sentence``.

        ``position`` (when provided) is the expected offset of
        ``target_word`` inside ``sentence``. When it matches, we use it
        directly; otherwise we fall back to ``sentence.find(target_word,
        position)`` which anchors the search at the expected offset and
        returns the correct occurrence when ``target_word`` repeats
        earlier in the sentence.
        """
        if (
            position is not None
            and 0 <= position <= len(sentence)
            and sentence[position : position + len(target_word)] == target_word
        ):
            pos = position
        else:
            anchor = max(0, position) if position is not None else 0
            pos = sentence.find(target_word, anchor)
            if pos < 0 and anchor > 0:
                # Anchor overshot — fall back to a full-sentence search so
                # we at least score *some* occurrence rather than silently
                # dropping the token.
                pos = sentence.find(target_word)
        if pos < 0:
            return {}
        masked = sentence[:pos] + self._mask_str + sentence[pos + len(target_word) :]
        enc = self._tokenizer(masked, return_tensors="pt", truncation=True, max_length=128).to(
            self._device
        )
        with self._torch.no_grad():
            out = self._model(**enc)
        logits = out.logits.squeeze(0)  # (L, V)
        input_ids = enc["input_ids"].squeeze(0)
        mask_idxs = (input_ids == self._mask_id).nonzero(as_tuple=True)[0]
        if len(mask_idxs) == 0:
            return {}
        mask_pos = int(mask_idxs[0])
        vocab_logits = logits[mask_pos]
        result: dict[str, float] = {}
        for cand in candidates:
            ids = self._tokenizer.encode(cand, add_special_tokens=False)
            if not ids:
                continue
            result[cand] = float(vocab_logits[ids[0]])
        return result

    # Alias to match SemanticChecker interface
    def score_mask_candidates(
        self,
        sentence: str,
        target_word: str,
        candidates: list[str],
        *,
        position: int | None = None,
    ) -> dict[str, float]:
        return self.score(sentence, target_word, candidates, position=position)
