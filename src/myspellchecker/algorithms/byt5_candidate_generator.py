"""ByT5-based top-K spelling candidate generator.

Reuses the existing v1 ByT5 ONNX model (``models/byt5-v1-onnx-int8/``) as a
candidate generator — NOT as a safety-net detector. Called only on tokens
the pipeline already suspects are erroneous. Produces top-K plausible
corrections for the suspect token given its sentence context.

Distinguishes from ``ByT5SafetyNetStrategy`` in purpose and integration:

- Safety-net runs on every "clean" sentence, one greedy decode, emits as
  errors. Slow and low-yield at scale.
- Candidate-generator runs only on pre-flagged tokens, beam-searches top-K,
  feeds the candidates into the ranker as a secondary source. The ranker
  (not the generator) decides what to emit.

Workstream: byt5-candidate-generator / Task: byt5gen-wrapper-01

Model I/O contract (matches the safety-net generator's): input is the full
Myanmar sentence with the typo in place; output is the corrected sentence
(byte-level, whitespace-tokenized at the sentence level). We run beam
search to get the top-K candidate corrections, then align by whitespace
token index to extract the candidate for the specific word position.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from myspellchecker.utils.logging_utils import get_logger

logger = get_logger(__name__)


class ByT5CandidateGenerator:
    """Top-K spell-correction candidate generator backed by ByT5 ONNX model.

    Not thread-safe: holds ONNX InferenceSession instances. Instantiate
    once per process.
    """

    DEFAULT_BEAM_WIDTH = 3
    DEFAULT_K = 5
    DEFAULT_MAX_LENGTH = 100

    def __init__(
        self,
        model_path: str | Path,
        *,
        beam_width: int = DEFAULT_BEAM_WIDTH,
        max_length: int = DEFAULT_MAX_LENGTH,
    ) -> None:
        self.model_path = Path(model_path)
        self.beam_width = int(beam_width)
        self.max_length = int(max_length)

        self.enabled = False
        self._onnx_enc: Any = None
        self._onnx_dec: Any = None
        self._onnx_meta: dict[str, Any] = {}
        self._tokenizer: Any = None
        self._load()

    def _load(self) -> None:
        p = self.model_path
        enc_path = p / "encoder.onnx"
        dec_path = p / "decoder.onnx"
        meta_path = p / "onnx_meta.json"
        if not (enc_path.exists() and dec_path.exists()):
            logger.warning("ByT5CandidateGenerator: model files missing under %s", p)
            return

        try:
            import onnxruntime as ort
            from transformers import ByT5Tokenizer

            so = ort.SessionOptions()
            so.intra_op_num_threads = 4
            self._onnx_enc = ort.InferenceSession(str(enc_path), sess_options=so)
            self._onnx_dec = ort.InferenceSession(str(dec_path), sess_options=so)
            self._onnx_meta = (
                json.loads(meta_path.read_text())
                if meta_path.exists()
                else {"decoder_start_token_id": 0, "eos_token_id": 1}
            )
            self._tokenizer = ByT5Tokenizer()
            self.enabled = True
            logger.info(
                "ByT5CandidateGenerator: loaded from %s (beam=%d, max_len=%d)",
                p,
                self.beam_width,
                self.max_length,
            )
        except (ImportError, RuntimeError, OSError, ValueError) as exc:
            logger.warning("ByT5CandidateGenerator: load failed: %r", exc)
            self.enabled = False

    def generate_sentences(
        self,
        context: str,
        *,
        k: int = DEFAULT_K,
    ) -> list[tuple[str, float]]:
        """Return top-K decoded sentences with log-probabilities.

        Lower-level than :meth:`generate` — returns the full decoded output
        from each beam, no word-level extraction. Useful for audit / recall
        measurement where we want to check if the gold form appears anywhere
        in the model output rather than only at a specific token index.
        """
        if not self.enabled or not context:
            return []
        try:
            beams = self._beam_search(context, k=max(k, self.beam_width))
        except Exception as exc:
            logger.debug("ByT5CandidateGenerator: beam failed: %r", exc)
            return []
        out: list[tuple[str, float]] = []
        for decoded_ids, logprob in beams:
            try:
                text = self._tokenizer.decode(decoded_ids, skip_special_tokens=True)
            except Exception:
                continue
            out.append((text.strip(), logprob))
        # Dedupe sentences, keep highest-logprob occurrence.
        seen: dict[str, float] = {}
        for text, lp in out:
            if text not in seen or lp > seen[text]:
                seen[text] = lp
        ranked = sorted(seen.items(), key=lambda kv: -kv[1])
        return ranked[:k]

    def generate(
        self,
        word: str,
        context: str,
        *,
        k: int = DEFAULT_K,
        occurrence: int = 0,
    ) -> list[tuple[str, float]]:
        """Return top-K candidate corrections for ``word`` given ``context``.

        Args:
            word: The suspect token (as it appears in context).
            context: The full sentence containing the suspect token.
            k: Maximum number of candidates to return (after dedup).
            occurrence: If ``word`` appears multiple times in context, pick
                the ``occurrence``-th one (0-indexed).

        Returns:
            List of (candidate, log_prob) ordered by descending log_prob.
            Empty list if generator is disabled or no candidates align.
        """
        if not self.enabled or not word or not context:
            return []

        # Find the word-token index in whitespace-tokenized context.
        src_tokens = context.split()
        positions = [i for i, t in enumerate(src_tokens) if word in t]
        if not positions:
            return []
        token_idx = positions[min(occurrence, len(positions) - 1)]

        try:
            beams = self._beam_search(context, k=max(k, self.beam_width))
        except Exception as exc:
            logger.debug("ByT5CandidateGenerator: beam failed: %r", exc)
            return []

        # For each completed beam, decode and extract the candidate at token_idx.
        seen: dict[str, float] = {}
        for decoded_ids, logprob in beams:
            try:
                text = self._tokenizer.decode(decoded_ids, skip_special_tokens=True)
            except Exception:
                continue
            out_tokens = text.strip().split()
            if not out_tokens or token_idx >= len(out_tokens):
                continue
            candidate = out_tokens[token_idx]
            # Minimal sanity: skip empty, skip identity match.
            if not candidate or candidate == word:
                continue
            # Dedupe on candidate string, keeping highest-logprob occurrence.
            if candidate not in seen or logprob > seen[candidate]:
                seen[candidate] = logprob

        ranked = sorted(seen.items(), key=lambda kv: -kv[1])
        return ranked[:k]

    def _beam_search(self, sentence: str, *, k: int) -> list[tuple[list[int], float]]:
        """Beam-search decode. Returns list of (token_ids, cumulative_logprob).

        Encoder is run once; decoder is called step-by-step with the current
        beam states. We keep ``self.beam_width`` live beams and collect
        completed ones (hit EOS) until the budget is exhausted.
        """
        import numpy as np

        encoded = self._tokenizer(sentence, return_tensors="np")
        input_ids = encoded["input_ids"].astype(np.int64)
        attn = encoded["attention_mask"].astype(np.int64)
        enc_hidden = self._onnx_enc.run(None, {"input_ids": input_ids, "attention_mask": attn})[
            0
        ].astype(np.float32)

        dec_start = int(self._onnx_meta.get("decoder_start_token_id", 0))
        eos_id = self._onnx_meta.get("eos_token_id", 1)
        eos_set = {int(x) for x in (eos_id if isinstance(eos_id, list) else [eos_id])}

        beam_width = max(self.beam_width, k)

        # beams: list of (tokens: list[int], logprob: float)
        beams: list[tuple[list[int], float]] = [([dec_start], 0.0)]
        completed: list[tuple[list[int], float]] = []

        max_new = min(self.max_length, input_ids.shape[1] + 40)

        for _ in range(max_new):
            candidates: list[tuple[list[int], float]] = []
            for tokens, logprob in beams:
                dec_in = np.array([tokens], dtype=np.int64)
                logits = self._onnx_dec.run(
                    None,
                    {
                        "decoder_input_ids": dec_in,
                        "encoder_hidden_states": enc_hidden,
                        "encoder_attention_mask": attn,
                    },
                )[0]
                last_logits = logits[0, -1]
                # Log-softmax.
                max_l = float(np.max(last_logits))
                log_z = max_l + math.log(
                    float(np.sum(np.exp(last_logits - max_l).astype(np.float64)))
                )
                # Top-k next tokens for this beam.
                top_idx = np.argpartition(last_logits, -beam_width)[-beam_width:]
                for next_id in top_idx:
                    nid = int(next_id)
                    new_lp = logprob + float(last_logits[nid]) - log_z
                    new_tokens = tokens + [nid]
                    if nid in eos_set:
                        completed.append((tokens[1:], new_lp))  # drop start token
                    else:
                        candidates.append((new_tokens, new_lp))

            if not candidates:
                break
            # Keep top-beam_width by logprob.
            candidates.sort(key=lambda x: -x[1])
            beams = candidates[:beam_width]

            # Early stop: if we have enough completed and the best live beam
            # is worse than the worst completed beam, no point continuing.
            if len(completed) >= k and completed:
                completed.sort(key=lambda x: -x[1])
                worst_completed = completed[k - 1][1] if len(completed) >= k else float("-inf")
                if beams[0][1] < worst_completed:
                    break

        # Unfinished beams count too — strip the start token to match.
        for tokens, lp in beams:
            completed.append((tokens[1:], lp))

        completed.sort(key=lambda x: -x[1])
        return completed[: max(k, beam_width)]
