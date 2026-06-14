"""BERT corrector head for the Soft-Masked BERT skeleton.

Given a fully-embedded sequence (output of the soft-mask interpolation +
re-embedding stack from ``model.py``), runs the BERT transformer body to
produce per-position hidden states, then applies the residual connection
to the original input embedding, then runs the MLM head to produce
per-position vocab logits.

Why we bypass the standard ``model.forward(input_ids=...)`` path: Soft-
Masked BERT performs soft-mask interpolation on word embeddings BEFORE the
transformer blocks AND adds a residual connection from the original input
embedding AFTER the transformer blocks (Zhang et al. 2020 §3.3). Neither
hook is exposed by the HuggingFace forward API, so we walk the BERT
internals directly.

Supports BERT, RoBERTa, ELECTRA structurally — backbone resolved dynamically.
"""

from __future__ import annotations

from typing import Any

import torch

from .encoder import SoftMaskedEncoder


class SoftMaskedCorrector:
    """Runs BERT transformer body + residual + MLM head.

    Wraps the same ``AutoModelForMaskedLM`` instance owned by ``SoftMaskedEncoder``.
    """

    def __init__(self, encoder: SoftMaskedEncoder) -> None:
        self.encoder = encoder
        self._backbone, self._mlm_head = self._resolve_backbone_and_head(encoder.inner_model)

    @property
    def hidden_size(self) -> int:
        return self.encoder.hidden_size

    @property
    def vocab_size(self) -> int:
        return self.encoder.vocab_size

    def hidden_states(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run the BERT transformer blocks on a pre-computed embedding tensor.

        Returns ``[batch, seq, hidden]`` sequence_output — the hidden states
        before the MLM head. Used by ``model.py`` to add the residual
        connection from the original input embedding before the MLM head.
        """
        ext_mask = self.encoder.inner_model.get_extended_attention_mask(
            attention_mask, embeddings.shape[:-1]
        )
        encoder_out = self._backbone.encoder(
            hidden_states=embeddings,
            attention_mask=ext_mask,
        )
        return encoder_out[0]

    def mlm_head(self, sequence_output: torch.Tensor) -> torch.Tensor:
        """Apply the MLM prediction head to a sequence-output tensor.

        Returns ``[batch, seq, vocab]`` logits.
        """
        return self._mlm_head(sequence_output)

    def forward(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Convenience: run transformer + MLM head WITHOUT residual.

        Used by ablations and the equivalence test. Production forward path
        in ``model.py`` calls ``hidden_states`` + ``mlm_head`` separately
        with the residual added in between.
        """
        return self.mlm_head(self.hidden_states(embeddings, attention_mask))

    # ----- internals -------------------------------------------------------

    @staticmethod
    def _resolve_backbone_and_head(model: Any) -> tuple[Any, Any]:
        """Find the (backbone, mlm_head) pair across BERT/RoBERTa/ELECTRA."""
        # BERT: model.bert + model.cls (BertOnlyMLMHead — Linear→GELU→Linear+vocab)
        if hasattr(model, "bert") and hasattr(model, "cls"):
            return model.bert, model.cls
        # RoBERTa: model.roberta + model.lm_head
        if hasattr(model, "roberta") and hasattr(model, "lm_head"):
            return model.roberta, model.lm_head
        # ELECTRA generator: model.electra + generator_lm_head (rare; discriminator has no MLM)
        if hasattr(model, "electra") and hasattr(model, "generator_lm_head"):
            return model.electra, model.generator_lm_head
        raise AttributeError(
            "Could not resolve (backbone, mlm_head) on model. Supported: "
            "BERT (model.bert + model.cls), RoBERTa (model.roberta + model.lm_head), "
            "ELECTRA generator (model.electra + model.generator_lm_head)."
        )
