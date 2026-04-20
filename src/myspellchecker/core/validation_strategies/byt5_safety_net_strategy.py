"""ByT5 generator safety-net strategy.

A last-chance structural-typo rescue that runs only when every other
strategy has declined to flag the sentence. Uses a fine-tuned ByT5-small
seq2seq model to propose whole-sentence corrections; proposed edits are
gated by (a) dictionary membership and (b) an MLM plausibility margin
before being emitted as :data:`ET_WORD` errors.

Trigger gate (both must be true):
    1. ``len(context.existing_errors) == 0`` -- pipeline flagged nothing.
    2. Sentence contains ``>= min_typo_prone_chars`` characters from the
       shared ``_TYPO_PRONE_CHARS`` set (reused from SyllableWindowOOV).

For every proposed word-level edit we require:
    * The proposed replacement is an in-dict word
      (``provider.get_word_frequency(...) > 0``).
    * The proposed replacement beats the original at the masked position
      by at least ``mlm_gate_margin`` logits (via
      :meth:`SemanticChecker.score_mask_candidates`).

Runtime:
    * ONNX backend is preferred when the export is usable; falls back to
      PyTorch (MPS/CPU) automatically. ONNX export of T5 decoders with
      relative-position bias is fragile across torch/onnx versions -- the
      PyTorch fallback is the canonical production path until a clean
      export pipeline is available.

Priority: 80 (after every other strategy).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from myspellchecker.core.constants import ET_WORD
from myspellchecker.core.response import Error, Suggestion, WordError
from myspellchecker.core.validation_strategies.base import ValidationContext, ValidationStrategy
from myspellchecker.core.validation_strategies.syllable_window_oov_strategy import (
    _TYPO_PRONE_CHARS,
)
from myspellchecker.tokenizers.syllable import SyllableTokenizer
from myspellchecker.utils.logging_utils import get_logger

if TYPE_CHECKING:
    from myspellchecker.algorithms.semantic_checker import SemanticChecker
    from myspellchecker.providers.interfaces import WordRepository

logger = get_logger(__name__)

_PRIORITY = 80
_TASK_PREFIX = "correct: "


@dataclass
class _Edit:
    """One proposed word-level replacement extracted from a diff."""

    position: int
    original: str
    replacement: str


class ByT5SafetyNetStrategy(ValidationStrategy):
    """Rescue structural typos that survive the rest of the pipeline.

    See the module docstring for gating, inference, and priority details.
    """

    # Signals to ContextValidator that this strategy must bypass the
    # fast-path exit — its job is precisely to fire on sentences that
    # structural strategies declined to flag.
    is_safety_net: bool = True

    def __init__(
        self,
        provider: "WordRepository",
        model_path: str,
        *,
        semantic_checker: "SemanticChecker | None" = None,
        enabled: bool = True,
        mlm_gate_margin: float = 2.0,
        min_typo_prone_chars: int = 2,
        max_sentence_chars: int = 400,
        max_new_tokens_slack: int = 20,
        confidence: float = 0.75,
        max_existing_errors: int = 2,
    ) -> None:
        self.provider = provider
        self.model_path = model_path
        self.semantic_checker = semantic_checker
        self.enabled = enabled
        self.mlm_gate_margin = float(mlm_gate_margin)
        self.min_typo_prone_chars = int(min_typo_prone_chars)
        self.max_sentence_chars = int(max_sentence_chars)
        self.max_new_tokens_slack = int(max_new_tokens_slack)
        self.confidence = float(confidence)
        self.max_existing_errors = int(max_existing_errors)
        self._syllable_tokenizer = SyllableTokenizer()
        self.logger = logger

        self._backend: str | None = None  # "onnx" | "torch"
        self._tokenizer: Any = None
        self._pt_model: Any = None
        self._pt_device: str = "cpu"
        self._onnx_enc: Any = None
        self._onnx_dec: Any = None
        self._onnx_meta: dict[str, Any] = {}
        self._load_model(model_path)

    def priority(self) -> int:
        return _PRIORITY

    def _load_model(self, model_path: str) -> None:
        """Load ONNX if available, else PyTorch. Never raise at init."""
        p = Path(model_path)
        if not p.exists():
            self.logger.warning(f"ByT5 safety net: model path not found: {model_path}")
            self.enabled = False
            return

        enc_path = p / "encoder.onnx"
        dec_path = p / "decoder.onnx"
        meta_path = p / "onnx_meta.json"
        tried_onnx = enc_path.exists() and dec_path.exists()
        if tried_onnx:
            try:
                import json

                import onnxruntime as ort

                so = ort.SessionOptions()
                so.intra_op_num_threads = 4
                self._onnx_enc = ort.InferenceSession(str(enc_path), sess_options=so)
                self._onnx_dec = ort.InferenceSession(str(dec_path), sess_options=so)
                self._onnx_meta = (
                    json.loads(meta_path.read_text())
                    if meta_path.exists()
                    else {"decoder_start_token_id": 0, "eos_token_id": 1}
                )
                from transformers import ByT5Tokenizer

                self._tokenizer = ByT5Tokenizer()
                self._onnx_smoke_test()
                self._backend = "onnx"
                self.logger.info(f"ByT5 safety net: ONNX backend loaded from {p}")
                return
            except (RuntimeError, OSError, ValueError) as exc:
                self.logger.warning(
                    f"ByT5 safety net: ONNX backend failed ({exc!r}); falling back to PyTorch."
                )
                self._onnx_enc = None
                self._onnx_dec = None

        try:
            import torch
            from transformers import ByT5Tokenizer, T5ForConditionalGeneration

            pt_src = (
                p if (p / "model.safetensors").exists() or (p / "pytorch_model.bin").exists() else p
            )
            self._pt_model = T5ForConditionalGeneration.from_pretrained(pt_src)
            self._pt_model.eval()
            self._pt_device = "mps" if torch.backends.mps.is_available() else "cpu"
            self._pt_model.to(self._pt_device)
            self._tokenizer = ByT5Tokenizer()
            self._backend = "torch"
            self.logger.info(
                f"ByT5 safety net: PyTorch backend loaded on {self._pt_device} from {pt_src}"
            )
        except (ImportError, RuntimeError, OSError, ValueError) as exc:
            self.logger.warning(f"ByT5 safety net: PyTorch fallback also failed: {exc!r}")
            self.enabled = False

    def _onnx_smoke_test(self) -> None:
        """Minimal end-to-end ONNX call to surface broadcast errors at init.

        Exercises (a) src_len != dummy export length and (b) tgt_len > 1
        to catch T5 relative-position-bias exports that baked a fixed
        dim from a single dummy trace.
        """
        import numpy as np

        # src_len=32, tgt_len=2: both dims differ from the dummy export.
        ids = np.full((1, 32), ord("a"), dtype=np.int64)
        ids[0, -1] = 1  # eos
        mask = np.ones_like(ids, dtype=np.int64)
        enc_out = self._onnx_enc.run(None, {"input_ids": ids, "attention_mask": mask})[0]
        dec_ids = np.zeros((1, 2), dtype=np.int64)
        dec_ids[0, 0] = int(self._onnx_meta.get("decoder_start_token_id", 0))
        self._onnx_dec.run(
            None,
            {
                "decoder_input_ids": dec_ids,
                "encoder_hidden_states": enc_out.astype(np.float32),
                "encoder_attention_mask": mask,
            },
        )

    def validate(self, context: ValidationContext) -> list[Error]:
        if not self.enabled or self._backend is None:
            return []
        # The safety-net's job is to fire on sentences the rest of the pipeline
        # did NOT meaningfully flag. `context.existing_errors` is intermediate
        # state populated by earlier strategies; many of those entries are
        # suppressed later by the spellchecker's post-processing. Treat a
        # sentence as eligible when intermediate errors are sparse — up to the
        # configured `max_existing_errors` threshold.
        if len(context.existing_errors) > self.max_existing_errors:
            return []
        if not context.words:
            return []

        sentence = context.sentence or ""
        if not sentence or len(sentence) > self.max_sentence_chars:
            return []

        prone = sum(1 for c in sentence if c in _TYPO_PRONE_CHARS)
        if prone < self.min_typo_prone_chars:
            return []

        try:
            corrected = self._generate(sentence)
        except (RuntimeError, ValueError, KeyError) as exc:
            self.logger.debug(f"ByT5 safety net: generation failed: {exc!r}")
            return []
        if not corrected or corrected.strip() == sentence.strip():
            return []

        edits = self._extract_edits(context, corrected)
        if not edits:
            return []

        return [
            err for err in (self._verify_and_build(context, e) for e in edits) if err is not None
        ]

    def _generate(self, sentence: str) -> str:
        prompt = _TASK_PREFIX + sentence
        if self._backend == "onnx":
            return self._generate_onnx(prompt, sentence)
        return self._generate_torch(prompt, sentence)

    def _generate_torch(self, prompt: str, sentence: str) -> str:
        import torch

        ids = self._tokenizer(prompt, return_tensors="pt").input_ids.to(self._pt_device)
        max_new_tokens = ids.shape[1] + self.max_new_tokens_slack
        with torch.no_grad():
            out = self._pt_model.generate(
                ids,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                do_sample=False,
            )
        return self._tokenizer.decode(out[0], skip_special_tokens=True)

    def _generate_onnx(self, prompt: str, sentence: str) -> str:
        import numpy as np

        encoded = self._tokenizer(prompt, return_tensors="np")
        input_ids = encoded["input_ids"].astype(np.int64)
        attn = encoded["attention_mask"].astype(np.int64)
        enc_hidden = self._onnx_enc.run(None, {"input_ids": input_ids, "attention_mask": attn})[
            0
        ].astype(np.float32)

        dec_start = int(self._onnx_meta.get("decoder_start_token_id", 0))
        eos_id = self._onnx_meta.get("eos_token_id", 1)
        if isinstance(eos_id, list):
            eos_set = {int(x) for x in eos_id}
        else:
            eos_set = {int(eos_id)}

        decoder_ids = np.array([[dec_start]], dtype=np.int64)
        max_new = input_ids.shape[1] + self.max_new_tokens_slack
        generated: list[int] = []
        for _ in range(max_new):
            logits = self._onnx_dec.run(
                None,
                {
                    "decoder_input_ids": decoder_ids,
                    "encoder_hidden_states": enc_hidden,
                    "encoder_attention_mask": attn,
                },
            )[0]
            next_id = int(logits[0, -1].argmax())
            if next_id in eos_set:
                break
            generated.append(next_id)
            decoder_ids = np.concatenate([decoder_ids, [[next_id]]], axis=1)
        return self._tokenizer.decode(generated, skip_special_tokens=True)

    def _extract_edits(self, context: ValidationContext, corrected: str) -> list[_Edit]:
        """Return word-level replacements via whitespace-token alignment.

        ByT5 outputs a whitespace-tokenized sentence. We compare it against the
        *whitespace tokens* of the source sentence — not ``context.words``,
        which comes from the compound-aware segmenter and doesn't match the
        model's tokenization scheme. For each token that differs, we extract
        the *minimal* substring that changed (common prefix/suffix trimmed)
        so the downstream in-dict and MLM gates see a clean single-word edit
        rather than a whole compound.
        """
        sentence = context.sentence or ""
        source_tokens = sentence.split()
        corrected_tokens = corrected.strip().split()
        if len(corrected_tokens) != len(source_tokens):
            return []

        edits: list[_Edit] = []
        cursor = 0
        for src, tgt in zip(source_tokens, corrected_tokens, strict=False):
            tok_pos = sentence.find(src, cursor)
            if tok_pos < 0:
                cursor = cursor + len(src)
                continue
            cursor = tok_pos + len(src)
            if src == tgt or not tgt.strip():
                continue
            if abs(len(tgt) - len(src)) > max(4, len(src) // 2):
                continue

            # Trim common prefix / suffix to isolate the changed region.
            pre = 0
            while pre < len(src) and pre < len(tgt) and src[pre] == tgt[pre]:
                pre += 1
            suf = 0
            while (
                suf < len(src) - pre
                and suf < len(tgt) - pre
                and src[len(src) - 1 - suf] == tgt[len(tgt) - 1 - suf]
            ):
                suf += 1

            min_src = src[pre : len(src) - suf]
            min_tgt = tgt[pre : len(tgt) - suf]
            if not min_src or not min_tgt:
                # Whole-token deletion/insertion — too aggressive; skip.
                continue

            # Grow the minimal diff out to a syllable-aligned word so the
            # downstream in-dict gate sees a real word, not a fragment. We
            # tokenize `tgt` into syllables and greedily enlarge: start from
            # the syllable covering the change, then extend by one syllable
            # on either side until a dictionary word is matched, or until we
            # hit the original token boundary.
            tgt_sylls = self._syllable_tokenizer.tokenize(tgt)
            src_sylls = self._syllable_tokenizer.tokenize(src)
            left_pre, _right_suf = pre, suf
            enlarged_src = min_src
            enlarged_tgt = min_tgt
            if tgt_sylls and src_sylls and sum(map(len, tgt_sylls)) == len(tgt):
                # Find syllable index spanning the changed region.
                offset = 0
                change_syll_idx = 0
                for i, s in enumerate(tgt_sylls):
                    if offset + len(s) > pre:
                        change_syll_idx = i
                        break
                    offset += len(s)
                # Try syllable-level windows centred on change_syll_idx.
                for radius in range(0, max(len(tgt_sylls), len(src_sylls))):
                    lo = max(0, change_syll_idx - radius)
                    hi_t = min(len(tgt_sylls), change_syll_idx + radius + 1)
                    hi_s = min(len(src_sylls), change_syll_idx + radius + 1)
                    cand_tgt = "".join(tgt_sylls[lo:hi_t])
                    cand_src = "".join(src_sylls[lo:hi_s])
                    if self._is_in_dict(cand_tgt):
                        prefix_len = sum(len(tgt_sylls[i]) for i in range(lo))
                        enlarged_src = cand_src
                        enlarged_tgt = cand_tgt
                        left_pre = prefix_len
                        break

            edits.append(
                _Edit(
                    position=tok_pos + left_pre,
                    original=enlarged_src,
                    replacement=enlarged_tgt,
                )
            )
        return edits

    def _verify_and_build(self, context: ValidationContext, edit: _Edit) -> WordError | None:
        if not self._is_in_dict(edit.replacement):
            return None

        if self.semantic_checker is not None and not self._mlm_gate(context, edit):
            return None

        return WordError(
            text=edit.original,
            position=edit.position,
            error_type=ET_WORD,
            suggestions=[Suggestion(text=edit.replacement)],
            confidence=self.confidence,
            syllable_count=0,
            source_strategy="byt5_safety_net",
        )

    def _is_in_dict(self, word: str) -> bool:
        try:
            freq = self.provider.get_word_frequency(word)
        except (AttributeError, RuntimeError, ValueError):
            return False
        return bool(freq and freq > 0)

    def _mlm_gate(self, context: ValidationContext, edit: _Edit) -> bool:
        sentence = context.sentence or ""
        if not sentence or edit.original not in sentence:
            return False
        try:
            scores = self.semantic_checker.score_mask_candidates(
                sentence=sentence,
                target_word=edit.original,
                candidates=[edit.original, edit.replacement],
            )
        except (RuntimeError, ValueError, KeyError, IndexError, AttributeError):
            return False
        if not scores:
            return False
        orig_score = scores.get(edit.original)
        repl_score = scores.get(edit.replacement)
        if orig_score is None or repl_score is None:
            return False
        return (repl_score - orig_score) >= self.mlm_gate_margin

    def __repr__(self) -> str:
        return (
            f"ByT5SafetyNetStrategy(priority={self.priority()}, "
            f"enabled={self.enabled}, backend={self._backend}, "
            f"mlm_gate_margin={self.mlm_gate_margin}, "
            f"min_typo_prone_chars={self.min_typo_prone_chars})"
        )
