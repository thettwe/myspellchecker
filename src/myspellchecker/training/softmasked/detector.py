"""Bi-GRU detector network for Soft-Masked BERT (Zhang et al. ACL 2020).

Reads the encoder's full embedding-layer output and predicts a per-position
error LOGIT via a bidirectional GRU + linear head. Caller applies sigmoid
to get probabilities.

In the Soft-Masked architecture, ``sigmoid(logits)`` is the gating ``p_i``
that controls soft-mask interpolation:

    word_emb'_i = (1 - p_i) * word_emb_i + p_i * word_emb_[MASK]

When the detector is confident the position is correct (``p_i ≈ 0``), the
word embedding flows through unchanged; when it suspects an error
(``p_i ≈ 1``), the embedding is overwritten with the [MASK] word embedding
and the corrector decodes gold from context. The post-BERT residual added
in ``model.py`` lets the corrector preserve the original input when
detection confidence is low.

Returning raw logits (not sigmoid-then-inverted) is more numerically
stable for BCE loss than the previous probability-then-inverse-sigmoid
detour.
"""

from __future__ import annotations

import torch
from torch import nn


class BiGRUDetector(nn.Module):
    """Bidirectional GRU + linear head over per-position embeddings.

    Returns RAW logits (no sigmoid). Apply ``torch.sigmoid`` for [0, 1]
    probabilities, or use ``loss(logits, target, ...)`` which calls
    ``binary_cross_entropy_with_logits`` directly (numerically stable).

    Architecture per ACL 2020 §3.2:
      - Bi-GRU, 256 hidden per direction (concat → 512), 1 layer, dropout 0.1
      - Linear(2*hidden_per_direction, 1) → per-position logit
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_per_direction: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_per_direction = hidden_per_direction
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_per_direction,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(2 * hidden_per_direction, 1)

    def forward(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run Bi-GRU + linear head.

        Parameters
        ----------
        embeddings : ``[batch, seq, embedding_dim]``
            Full embedding output (with position + token_type + LN + dropout).
        attention_mask : ``[batch, seq]`` or None
            Used during training to mask BCE loss; the forward returns
            logits for ALL positions, so callers apply the mask at the
            loss level.

        Returns
        -------
        logits : ``[batch, seq]``
            Raw pre-sigmoid logits. Apply ``torch.sigmoid`` for probabilities.
        """
        gru_out, _ = self.gru(embeddings)  # [batch, seq, 2*hidden]
        gru_out = self.dropout(gru_out)
        return self.head(gru_out).squeeze(-1)  # [batch, seq]

    def loss(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        pos_weight: float | None = None,
    ) -> torch.Tensor:
        """BCE-with-logits loss, masked by attention_mask, optional pos_weight.

        Parameters
        ----------
        logits : ``[batch, seq]``
            Raw detector forward output (BEFORE sigmoid).
        target : ``[batch, seq]``
            Per-position 0/1 ground-truth (1 = position has an error).
        attention_mask : ``[batch, seq]`` or None
            Mask out padding positions from loss.
        pos_weight : float or None
            Class-prior weighting on the positive class. Errors are sparse
            (~1-5% of tokens in typical spell-check data), so pos_weight in
            10-30 range is reasonable. ``None`` = no weighting.
        """
        if pos_weight is not None:
            pw = torch.tensor(pos_weight, device=logits.device, dtype=logits.dtype)
            loss_per_position = nn.functional.binary_cross_entropy_with_logits(
                logits, target.float(), pos_weight=pw, reduction="none"
            )
        else:
            loss_per_position = nn.functional.binary_cross_entropy_with_logits(
                logits, target.float(), reduction="none"
            )
        if attention_mask is not None:
            loss_per_position = loss_per_position * attention_mask.float()
            denom = attention_mask.float().sum().clamp(min=1.0)
            return loss_per_position.sum() / denom
        return loss_per_position.mean()

    @property
    def output_dim(self) -> int:
        """Convenience: total Bi-GRU output channels (sum of both directions)."""
        return 2 * self.hidden_per_direction

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
