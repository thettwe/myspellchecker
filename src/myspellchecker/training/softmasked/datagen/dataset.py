"""PyTorch dataset reading pairs.jsonl produced by ``corrupt_pairs.py``.

For training, we tokenize on-the-fly because:
  - ``sample_clean.py`` output isn't BERT-tokenized
  - Soft-Masked needs char-aligned span info to build the per-position
    detect_target tensor (1 = position has an error, 0 = clean)
  - ``correct_target`` is the GOLD token sequence at positions covering the
    corrupted span. Positions OUTSIDE the span are set to -100 (PyTorch's
    ``ignore_index`` for cross-entropy) so the corrector is supervised only
    where we have ground truth — naive truncation against PAD tokens (the
    pre-fix bug) is gone.

Schema in pairs.jsonl (per line):
  {"clean", "corrupted", "gold", "erroneous", "span_start", "span_end",
   "subtype", "bucket", "rationale", "source", "clean_idx"}

Ingestion-time validations (added 2026-05-04 after second review pass):
  - V1: ``corrupted[s:e] == erroneous`` — guards against generators that
    return offsets in wrong units (visual clusters, bytes) or hallucinate.
    Burmese is multi-codepoint per visual cluster so this is high-risk.
  - V2: ``clean[:s] == corrupted[:s]`` and ``clean[s:?] == gold`` — guards
    against generators that violate the "byte-identical outside span" rule.
  - V3: ``s < e`` — degenerate zero-width spans produce all-zero
    detect_target + all-IGNORE_INDEX correct_target (wasted batch slot,
    drags the BCE class prior toward 0). Drop them.
  - V4: corrupted-span fits inside the tokenizer's max_length window.
    If the span gets truncated out, the row would have all-zero
    detect_target. Drop them.
  - V5 (added 2026-05-05): ``NFC(gold) != NFC(erroneous)`` — drops rows
    where gold and erroneous differ only in codepoint order (e.g. virama
    and dot-below-dot swap, or U+1026 vs U+1025+U+102E for "ဦ"). The
    tokenizer normalizes these to identical input_ids and identical
    correct_target, so the corrector gets zero signal while the detector
    still learns "this position is an error" — pure noise.

Whitespace normalization (added 2026-05-05, v1.9 architecture decision)
-----------------------------------------------------------------------
With ``normalize_whitespace=True`` (default), all U+0020 are stripped from
``clean`` and ``corrupted`` BEFORE validation and tokenization, and the
span (s, e) is recomputed in the stripped coordinate system. This matches
the v1.9 architecture: Soft-Masked BERT operates on whitespace-free,
length-preserving Burmese; spacing is handled by the existing pre-pass
segmenter, not by this model. See:
[[60_Decisions/Soft-Masked v1.9 Whitespace-Stripped Architecture 2026-05-05]]

Rows that fail any validation are skipped and counted in
``self.skip_counts`` for telemetry. ``RAISE_ON_INVALID=True`` switches to
hard-fail mode for catching upstream regressions early.
"""

from __future__ import annotations

import json
import unicodedata
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import IterableDataset

# CE ignore_index — positions with this label contribute 0 to cross-entropy loss
IGNORE_INDEX = -100


class SoftMaskedPairDataset(IterableDataset):
    """Streams (corrupted, gold) pairs from JSONL, tokenizing per-batch.

    Yields dicts with keys: input_ids, attention_mask, [token_type_ids],
    detect_target, correct_target.

    Worker sharding: when ``DataLoader(num_workers > 0)``, each worker
    processes lines where ``(line_idx - skip) % num_workers == worker_id``,
    so the dataset isn't replicated across workers (a bug Codex flagged).
    """

    def __init__(
        self,
        path: str | Path,
        tokenizer: Any,
        *,
        max_length: int = 256,
        skip: int = 0,
        limit: int | None = None,
        raise_on_invalid: bool = False,
        normalize_whitespace: bool = True,
    ) -> None:
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.skip = skip
        self.limit = limit
        self.raise_on_invalid = raise_on_invalid
        self.normalize_whitespace = normalize_whitespace
        # Per-reason counters for skipped rows (telemetry).
        self.skip_counts: Counter[str] = Counter()

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        n = 0
        emitted = 0
        with self.path.open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                # Apply pre-skip first
                if line_idx < self.skip:
                    continue
                # Worker sharding
                if (line_idx - self.skip) % num_workers != worker_id:
                    continue
                if self.limit is not None and emitted >= self.limit:
                    break
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                sample = self._build_sample(rec)
                if sample is not None:
                    emitted += 1
                    n += 1
                    yield sample

    def _skip(self, reason: str, rec: dict[str, Any]) -> None:
        """Record a skipped row. Optionally raise (debug mode)."""
        self.skip_counts[reason] += 1
        if self.raise_on_invalid:
            raise ValueError(f"dataset row failed {reason}: {rec!r}")

    def _build_sample(self, rec: dict[str, Any]) -> dict[str, torch.Tensor] | None:
        corrupted = rec.get("corrupted")
        gold_word = rec.get("gold")
        s = rec.get("span_start")
        e = rec.get("span_end")
        erroneous = rec.get("erroneous")
        clean = rec.get("clean")
        if not corrupted or not gold_word or s is None or e is None:
            self._skip("missing_field", rec)
            return None

        # V3: degenerate zero-width spans produce no useful supervision.
        if not (isinstance(s, int) and isinstance(e, int) and 0 <= s < e <= len(corrupted)):
            self._skip("invalid_span_bounds", rec)
            return None

        # Whitespace normalization (v1.9). Strip U+0020 from corrupted, clean,
        # gold_word, erroneous, and recompute (s, e) relative to the stripped
        # corrupted form. This is done BEFORE V1/V2 so all subsequent checks
        # operate in the canonical whitespace-free coordinate system.
        if self.normalize_whitespace:
            spaces_before_s = corrupted[:s].count(" ")
            spaces_in_span = corrupted[s:e].count(" ")
            corrupted = corrupted.replace(" ", "")
            if clean is not None:
                clean = clean.replace(" ", "")
            gold_word = gold_word.replace(" ", "")
            if erroneous is not None:
                erroneous = erroneous.replace(" ", "")
            new_s = s - spaces_before_s
            new_e = new_s + (e - s) - spaces_in_span
            s, e = new_s, new_e
            # Re-check bounds after recompute (defensive — strip should not
            # produce invalid spans, but Gemini occasionally reports spans
            # whose internal whitespace structure is inconsistent).
            if not (0 <= s < e <= len(corrupted)):
                self._skip("invalid_span_bounds_after_strip", rec)
                return None

        # V1: span/erroneous offset agreement. Guards against generators that
        # return offsets in wrong units (Burmese visual-cluster vs codepoint)
        # or hallucinate the span. ``erroneous`` is optional in older data;
        # only check when present.
        if erroneous is not None and corrupted[s:e] != erroneous:
            self._skip("span_erroneous_mismatch", rec)
            return None

        # V5: NFC-equivalence guard. Drop rows where gold and erroneous
        # differ only in codepoint order — they tokenize identically so
        # the corrector gets no signal but the detector still learns noise.
        if erroneous is not None and unicodedata.normalize(
            "NFC", erroneous
        ) == unicodedata.normalize("NFC", gold_word):
            self._skip("nfc_equivalent", rec)
            return None

        # V2: outside-span byte-identical guarantee — corrupted and clean
        # must agree on the prefix and suffix; corrupted[s:e] is the
        # erroneous form, clean[s:s+len(gold)] is the gold form. Only
        # validate when ``clean`` is provided (older datasets may omit it).
        # When normalize_whitespace=True, both have already been stripped
        # so this passes more rows (typist-spacing differences are no
        # longer differences in the stripped coordinate system).
        if clean is not None:
            if corrupted[:s] != clean[:s]:
                self._skip("clean_prefix_mismatch", rec)
                return None
            gold_e_in_clean = s + len(gold_word)
            if gold_e_in_clean > len(clean) or clean[s:gold_e_in_clean] != gold_word:
                self._skip("clean_gold_mismatch", rec)
                return None
            if corrupted[e:] != clean[gold_e_in_clean:]:
                self._skip("clean_suffix_mismatch", rec)
                return None

        corr_enc = self.tokenizer(
            corrupted,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
        )
        # V4: span-fits-in-window check. After truncation, if no token
        # offset overlaps [s, e), the row would have all-zero detect_target
        # and supervise nothing — drop it.
        corr_offsets_check = corr_enc["offset_mapping"][0].tolist()
        any_overlap = any(
            (a < e and b > s) and not (a == 0 and b == 0) for a, b in corr_offsets_check
        )
        if not any_overlap:
            self._skip("span_truncated_out", rec)
            return None

        # Build a "gold-replaced" version of the sentence for span-aligned
        # gold-token lookup. When ``clean`` is supplied AND passed V2, use
        # it directly so we never have to trust a reconstruction. Otherwise
        # fall back to splice-reconstruction (with the surrounding context
        # from corrupted) — this still works thanks to V1 + V2 guards.
        if clean is not None:
            gold_sent = clean
        else:
            gold_sent = corrupted[:s] + gold_word + corrupted[e:]
        gold_e = s + len(gold_word)
        gold_enc = self.tokenizer(
            gold_sent,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
        )

        seq_len = corr_enc["input_ids"].shape[1]
        corr_offsets = corr_enc["offset_mapping"][0].tolist()
        gold_offsets = gold_enc["offset_mapping"][0].tolist()
        gold_input_ids = gold_enc["input_ids"][0]

        # detect_target: 1 at corrupted-span-overlapping positions, 0 elsewhere
        detect_target = torch.zeros(seq_len, dtype=torch.long)
        corr_span_positions: list[int] = []
        for i, (a, b) in enumerate(corr_offsets):
            if a == 0 and b == 0:
                continue  # special tokens
            if a < e and b > s:
                detect_target[i] = 1
                corr_span_positions.append(i)

        # correct_target: IGNORE_INDEX everywhere except where we know gold
        correct_target = torch.full((seq_len,), IGNORE_INDEX, dtype=torch.long)
        gold_span_positions: list[int] = []
        for i, (a, b) in enumerate(gold_offsets):
            if a == 0 and b == 0:
                continue
            if a < gold_e and b > s:
                gold_span_positions.append(i)

        # Match corrupted-span positions to gold-span positions by sequential
        # index. When token counts diverge (e.g. corruption added a syllable),
        # any extra positions on either side stay as IGNORE_INDEX — the loss
        # ignores them rather than fitting noise.
        for ci, gi in zip(corr_span_positions, gold_span_positions, strict=False):
            correct_target[ci] = int(gold_input_ids[gi].item())

        # Out-of-span positions (where corrupted == clean) could also be
        # supervised (using clean_input_ids at the equivalent offset), but
        # that requires diff-match-patch alignment to handle the offset
        # shift caused by the corruption. For the smoke baseline we accept
        # span-only supervision — it's a weaker signal but it's CORRECT.
        # Future work: add diff-aligned outside-span supervision in v2.

        out: dict[str, torch.Tensor] = {
            "input_ids": corr_enc["input_ids"][0],
            "attention_mask": corr_enc["attention_mask"][0],
            "detect_target": detect_target,
            "correct_target": correct_target,
        }
        if "token_type_ids" in corr_enc:
            out["token_type_ids"] = corr_enc["token_type_ids"][0]
        return out


def collate_pairs(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Stack same-length (already padded) pair dicts into a batch."""
    keys = ("input_ids", "attention_mask", "detect_target", "correct_target")
    out = {k: torch.stack([b[k] for b in batch]) for k in keys}
    if "token_type_ids" in batch[0]:
        out["token_type_ids"] = torch.stack([b["token_type_ids"] for b in batch])
    return out
