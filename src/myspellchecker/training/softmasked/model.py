"""Soft-Masked BERT end-to-end model (Zhang et al. ACL 2020).

Connects the four pieces with the corrected pipeline (post second review
2026-05-04):

    input_ids                              mask_ids = full([MASK]_id)
       │                                            │
       ▼                                            ▼
    [encoder.embed]  ── e_in_full ──┐    [encoder.embed]  ── e_mask_full ──┐
       │                            │                                      │
       ▼                            │                                      │
    [BiGRUDetector] → logits → σ → p│                                      │
       │                            │                                      │
       ▼                            ▼                                      ▼
    e_soft_full = (1 - p) * e_in_full + p * e_mask_full
       │
       ▼
    [BERT transformer body] → sequence_output
       │
       ▼
    final = sequence_output + e_in_full      ◄── residual (ACL 2020 §3.3)
       │                          (same dropout mask as the BERT input branch)
       ▼
    [MLM head] → per-position vocab logits

Architectural fix history:

  - **B1 (carried forward)**: the mask-token embedding is fetched dynamically
    on every forward via ``encoder.embed(mask_ids)`` instead of a cached
    snapshot, so encoder updates flow through.

  - **B2 (revised — full-embedding soft-mask)**: interpolation now happens
    AT THE FULL-EMBEDDING LEVEL (matches ACL 2020 §3.2 eq. 3). The
    ``e_mask_full`` branch runs the embedding stack on a tensor of [MASK]
    ids spanning the FULL sequence, so each position gets the mask token's
    word embedding paired with its OWN position embedding — no position-0
    leak. The earlier "word-only interpolation + re-embed" path is gone.

  - **B3 (carried forward)**: residual ``+ e_in_full`` lives AFTER the BERT
    body, before the MLM head — lets the corrector "choose not to correct"
    at low-p positions.

  - **D1 (new — single embedding dropout per branch)**: the input embedding
    ``e_in_full`` is computed ONCE and reused for both (a) the soft-mask
    base and (b) the post-BERT residual. The previous code computed
    ``e_in_full`` separately from the soft-mask path's ``e_soft_full``,
    which in train mode meant two different dropout masks on what should
    be a shared embedding — a needless source of training noise.
"""

from __future__ import annotations

import torch
from torch import nn

from .corrector import SoftMaskedCorrector
from .detector import BiGRUDetector
from .encoder import SoftMaskedEncoder


class SoftMaskedBERT(nn.Module):
    """End-to-end Soft-Masked BERT.

    The encoder + corrector share a single HuggingFace ``AutoModelForMaskedLM``
    instance (the encoder owns it). The detector is its own small Bi-GRU.

    Parameters
    ----------
    encoder : SoftMaskedEncoder
        Loaded gklmip-bert (or compatible BERT-family checkpoint).
    detector : BiGRUDetector or None
        If None, instantiated with default Bi-GRU dim (256/direction).
    """

    def __init__(
        self,
        encoder: SoftMaskedEncoder,
        detector: BiGRUDetector | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        # Register the underlying HF MaskedLM as a tracked submodule so its
        # parameters are visible to ``parameters()``, ``state_dict()``, optimizer
        # construction, LoRA wrapping, etc.
        self.bert = encoder.inner_model
        self.detector = detector or BiGRUDetector(embedding_dim=encoder.hidden_size)
        self.corrector = SoftMaskedCorrector(encoder)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """End-to-end forward.

        Parameters
        ----------
        input_ids : ``[batch, seq]``
        attention_mask : ``[batch, seq]``
        token_type_ids : ``[batch, seq]`` or None

        Returns
        -------
        dict with:
          - ``detect_logits`` : ``[batch, seq]`` raw detector pre-sigmoid logits
                                (use ``binary_cross_entropy_with_logits`` for stability)
          - ``p_error`` : ``[batch, seq]`` sigmoid(detect_logits) in [0, 1]
          - ``logits`` : ``[batch, seq, vocab]`` corrector output
        """
        # 1. Full input embedding (word + position + token_type + LN + dropout).
        #    Computed ONCE and reused for the detector input, the soft-mask
        #    base, AND the post-BERT residual — single dropout pass, no
        #    train-mode divergence between residual and BERT input.
        e_in_full = self.encoder.embed(input_ids, token_type_ids=token_type_ids)

        # 2. Detector reads the full input embedding
        detect_logits = self.detector(e_in_full, attention_mask=attention_mask)
        p_error = torch.sigmoid(detect_logits)

        # 3. Mask embedding at every position. Build a [MASK]-filled id
        #    tensor of the same shape as input_ids and run it through the
        #    same ``embed`` call. Each position therefore gets the mask
        #    token's WORD embedding paired with its OWN position embedding
        #    — fixes B2 (no position-0 leak) without needing a second
        #    re-embedding pass on word-only interpolation.
        mask_ids = torch.full_like(input_ids, fill_value=self.encoder.mask_token_id)
        e_mask_full = self.encoder.embed(mask_ids, token_type_ids=token_type_ids)

        # 4. Soft-mask interpolation at the FULL-embedding level (ACL 2020
        #    §3.2 eq. 3): e' = (1 - p) * e_in + p * e_mask
        p = p_error.unsqueeze(-1)  # [batch, seq, 1]
        e_soft_full = (1.0 - p) * e_in_full + p * e_mask_full

        # 5. BERT transformer body → hidden states (NO MLM head yet)
        sequence_output = self.corrector.hidden_states(e_soft_full, attention_mask)

        # 6. Residual: ACL 2020 §3.3 — add the SAME input embedding used in
        #    step 1 to the BERT output. Identical dropout mask as the BERT
        #    input branch.
        final = sequence_output + e_in_full

        # 7. MLM head → per-position vocab logits
        logits = self.corrector.mlm_head(final)

        return {
            "detect_logits": detect_logits,
            "p_error": p_error,
            "logits": logits,
        }

    def n_params(self) -> dict[str, int]:
        """Parameter counts per submodule for budgeting."""
        return {
            "encoder": sum(p.numel() for p in self.encoder.inner_model.parameters()),
            "detector": self.detector.n_params(),
            # Corrector shares parameters with encoder.inner_model — don't double-count
            "corrector_shared_with_encoder": 0,
            "total_unique": (
                sum(p.numel() for p in self.encoder.inner_model.parameters())
                + self.detector.n_params()
            ),
        }
