"""Frozen-encoder + thin-Linear-head syllable span probe.

Production module for the v1.7.x neural enhancement. Wraps a frozen BERT-class
encoder with a single Linear layer that emits per-syllable binary span scores.
Trained for ~5 minutes on 50K examples, head-only (no encoder fine-tuning).

Run-time inference helpers project per-syllable scores onto words via:
  - direct char-span overlap, OR
  - whitespace-adjacency (high-prob whitespace syllable attaches to the
    preceding Myanmar word — the broken_compound signal).

Validated artifact: ``models/probe-syllable-span-v1/`` (head.pt + config.json).
See ``30_Audits/Probe Hybrid Ships at +0.0067 2026-05-03.md``.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from myspellchecker.utils.logging_utils import get_logger

if TYPE_CHECKING:  # pragma: no cover - type-only imports
    pass

logger = get_logger(__name__)


def _detect_device() -> str:
    """Return the best available torch device string."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class FrozenSyllableSpanProbe:
    """Frozen BERT encoder + thin per-syllable binary head."""

    def __init__(self, encoder_path: str | Path):
        import torch
        import torch.nn as nn
        from transformers import AutoModel

        self.encoder = AutoModel.from_pretrained(str(encoder_path))
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.head = nn.Linear(self.encoder.config.hidden_size, 1)
        self._torch = torch
        self._nn = nn

    def to(self, device: str) -> "FrozenSyllableSpanProbe":
        self.encoder.to(device)
        self.head.to(device)
        return self

    def eval(self) -> "FrozenSyllableSpanProbe":
        self.encoder.eval()
        self.head.eval()
        return self

    @property
    def hidden_size(self) -> int:
        return int(self.encoder.config.hidden_size)

    def predict_logits(self, input_ids, attention_mask, syl_to_subword_mask):
        """Return per-syllable logits (B, S).

        Args:
            input_ids: (B, T) tensor of subword ids.
            attention_mask: (B, T) tensor of 0/1.
            syl_to_subword_mask: (B, S, T) float tensor; per syllable, mask
                of which subwords belong to it (overlap-based).
        """
        torch = self._torch
        with torch.no_grad():
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            hidden = out.last_hidden_state  # (B, T, H)
            mask = syl_to_subword_mask.float()
            denom = mask.sum(dim=-1, keepdim=True).clamp(min=1)
            syl_hidden = torch.einsum("bst,bth->bsh", mask, hidden) / denom
            return self.head(syl_hidden).squeeze(-1)


@dataclass
class _SyllableSpan:
    """Helper: a syllable's text and char span in the input."""

    text: str
    start: int
    end: int


class ProbeInferenceEngine:
    """High-level inference helper used by validation strategies.

    Loads probe + tokenizer + syllable segmenter once and exposes a single
    ``score_sentence(text)`` that returns per-syllable probabilities and the
    underlying syllable spans.
    """

    def __init__(
        self,
        model_path: str | Path,
        device: str | None = None,
        max_length: int = 256,
    ):
        import torch
        from transformers import AutoTokenizer

        from myspellchecker.segmenters.regex import RegexSegmenter

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Probe model directory not found: {model_path}. Expected head.pt + config.json."
            )

        config_path = model_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Missing probe config.json at {config_path}")
        cfg = json.loads(config_path.read_text())
        encoder_path = cfg["encoder"]

        head_path = model_path / "head.pt"
        if not head_path.exists():
            raise FileNotFoundError(f"Missing probe head.pt at {head_path}")

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(str(encoder_path))
        self.model = FrozenSyllableSpanProbe(encoder_path)
        self.model.head.load_state_dict(torch.load(str(head_path), map_location=device or "cpu"))
        self.device = device or _detect_device()
        self.model.to(self.device)
        self.model.eval()
        self.max_length = max_length
        self.segmenter = RegexSegmenter()
        # Memo of score_sentence results. The probe is frozen and inference
        # is deterministic per text, and the three probe strategies share
        # one engine — on the p95 latency tail ~69% of score_sentence calls
        # within a single check() are duplicates of the same sentence text.
        self._score_cache: OrderedDict[str, tuple[list[float], list[_SyllableSpan]]] = OrderedDict()
        self._SCORE_CACHE_MAX = 256
        logger.info(
            "ProbeInferenceEngine loaded: encoder=%s head=%s device=%s",
            encoder_path,
            head_path,
            self.device,
        )

    def score_sentence(self, text: str) -> tuple[list[float], list[_SyllableSpan]]:
        """Return (per-syllable probability list, syllable span list).

        Results are memoized per text (LRU, behavior-identical: the frozen
        probe is deterministic). The outer lists are copied per call so a
        caller mutating its result cannot poison the cache.
        """
        cached = self._score_cache.get(text)
        if cached is not None:
            self._score_cache.move_to_end(text)
            return list(cached[0]), list(cached[1])
        probs, spans = self._score_sentence_uncached(text)
        if len(self._score_cache) >= self._SCORE_CACHE_MAX:
            self._score_cache.popitem(last=False)
        self._score_cache[text] = (probs, spans)
        return list(probs), list(spans)

    def _score_sentence_uncached(self, text: str) -> tuple[list[float], list[_SyllableSpan]]:
        if not text:
            return [], []

        syllables = self.segmenter.segment_syllables(text)
        if not syllables:
            return [], []

        cursor = 0
        spans: list[_SyllableSpan] = []
        for s in syllables:
            idx = text.find(s, cursor)
            if idx == -1:
                idx = cursor
            spans.append(_SyllableSpan(text=s, start=idx, end=idx + len(s)))
            cursor = idx + len(s)

        enc = self.tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors=None,
            padding=False,
        )
        T = len(enc["input_ids"])
        S = len(spans)
        mask = np.zeros((S, T), dtype=np.float32)
        for t, (cs, ce) in enumerate(enc["offset_mapping"]):
            if cs == ce:
                continue
            for s_idx, span in enumerate(spans):
                if cs < span.end and ce > span.start:
                    mask[s_idx, t] = 1.0
        valid = mask.sum(axis=1) > 0
        if not valid.any():
            return [0.0] * S, spans

        torch = self._torch
        input_ids_t = torch.tensor(enc["input_ids"]).unsqueeze(0).to(self.device)
        am_t = torch.tensor(enc["attention_mask"]).unsqueeze(0).to(self.device)
        mask_t = torch.from_numpy(mask).unsqueeze(0).to(self.device)
        logits = self.model.predict_logits(input_ids_t, am_t, mask_t)
        probs = torch.sigmoid(logits[0]).cpu().numpy()
        probs[~valid] = 0.0
        return probs.tolist(), spans
