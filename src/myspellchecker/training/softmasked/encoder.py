"""Encoder loader for the Soft-Masked BERT skeleton.

Soft-Masked BERT (Zhang et al. ACL 2020) splits a single BERT checkpoint into
two consumers:

  1. **Detector input** — the full embedding-layer output (word + position +
     token-type, after layernorm + dropout) feeds the Bi-GRU detector to
     predict per-position p(error).
  2. **Corrector** — soft-masked WORD embeddings get re-embedded (position +
     token_type added afresh) and fed through the BERT transformer blocks +
     MLM head, with a residual connection from the original input embedding
     added after the transformer.

Both consumers share the same loaded HuggingFace model. This module is the
loader/owner; ``corrector.py`` consumes the same instance.

Encoder is locked to GKLMIP-BERT per [[Base Model Audit Verdict 2026-05-04]].
The AutoModel abstraction is kept so a prep-ws4 continued-pretrain checkpoint
(or, if the prep-ws5 activation gate fires, a different bert-family encoder)
drops in without code changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer

DEFAULT_MODEL_PATH = "models/gklmip-bert-myanmar-fixed"


@dataclass
class TokenizerOutput:
    """Container for a tokenized batch + the embedding output."""

    input_ids: torch.Tensor  # [batch, seq]
    attention_mask: torch.Tensor  # [batch, seq]
    token_type_ids: torch.Tensor | None  # [batch, seq] or None
    offset_mapping: torch.Tensor | None  # [batch, seq, 2] or None


class SoftMaskedEncoder:
    """Owns the BERT MLM checkpoint, exposes the embedding submodule + helpers.

    Methods
    -------
    tokenize(texts, max_length, return_offsets) -> TokenizerOutput
        Standard HuggingFace tokenization.
    embed(input_ids, token_type_ids) -> embeddings
        Runs the model's full embedding stack. Output is what the detector
        reads. Equivalent to the BERT embedding layer (word + position +
        token_type + layernorm + dropout).
    embed_from_inputs(inputs_embeds, token_type_ids) -> embeddings
        Runs the embedding stack with a CUSTOM word-embedding tensor (used
        for soft-mask interpolation). Position + token_type embeddings are
        added by HuggingFace's BertEmbeddings.forward when ``inputs_embeds``
        is provided.
    word_embedding_table -> nn.Embedding
        Direct access to the word-embeddings parameter (for mask-token lookup
        without position info — fixes the position-0 leak bug).
    inner_model -> the wrapped AutoModelForMaskedLM
        Used by ``Corrector`` for transformer body + MLM head access.
    """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        device: str | torch.device | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.config = AutoConfig.from_pretrained(str(self.model_path))
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), use_fast=True)
        self.model = AutoModelForMaskedLM.from_pretrained(str(self.model_path))
        self.model.train(False)  # PyTorch inference mode by default
        if device is not None:
            self.model.to(device)
        self.device = next(self.model.parameters()).device
        # BERT exposes ``model.bert.embeddings``; RoBERTa exposes ``model.roberta.embeddings``.
        # Resolve dynamically so a checkpoint swap doesn't require code changes.
        self._embedding_module = self._resolve_embedding_module(self.model)
        self._mask_token_id = self.tokenizer.mask_token_id
        if self._mask_token_id is None:
            raise ValueError(f"tokenizer at {model_path} has no mask_token_id")

    # ----- public API ------------------------------------------------------

    @property
    def hidden_size(self) -> int:
        return int(self.config.hidden_size)

    @property
    def vocab_size(self) -> int:
        return int(self.config.vocab_size)

    @property
    def mask_token_id(self) -> int:
        return self._mask_token_id

    @property
    def inner_model(self) -> Any:
        """The HuggingFace MaskedLM model — used by the corrector forward path."""
        return self.model

    @property
    def word_embedding_table(self) -> nn.Embedding:
        """The word_embeddings ``nn.Embedding`` submodule.

        Use this for mask-token lookup without position info — the cached
        ``mask_embedding()`` approach in earlier versions captured a stale
        vector AND included position-0 positional info, both of which
        broke the soft-mask interpolation.
        """
        return self._embedding_module.word_embeddings

    def tokenize(
        self,
        texts: str | list[str],
        *,
        max_length: int = 256,
        return_offsets: bool = False,
        padding: bool = True,
        truncation: bool = True,
    ) -> TokenizerOutput:
        if isinstance(texts, str):
            texts = [texts]
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_offsets_mapping=return_offsets,
        )
        return TokenizerOutput(
            input_ids=enc["input_ids"].to(self.device),
            attention_mask=enc["attention_mask"].to(self.device),
            token_type_ids=enc.get("token_type_ids").to(self.device)
            if "token_type_ids" in enc and enc["token_type_ids"] is not None
            else None,
            offset_mapping=enc.get("offset_mapping").to(self.device)
            if "offset_mapping" in enc and enc["offset_mapping"] is not None
            else None,
        )

    def embed(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the embedding layer (word + position + token_type + layernorm + dropout).

        Returns ``[batch, seq, hidden]`` — the full embedding-layer output
        the detector reads.
        """
        kwargs: dict[str, Any] = {"input_ids": input_ids}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        return self._embedding_module(**kwargs)

    def embed_from_inputs(
        self,
        inputs_embeds: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the embedding stack with custom word embeddings.

        Pass-through to HuggingFace ``BertEmbeddings.forward(inputs_embeds=...)``
        which adds position + token_type embeddings, layernorm, and dropout
        on top of the supplied word-embedding tensor. Used to feed soft-
        masked word embeddings through the rest of the embedding stack
        without re-embedding the input_ids.
        """
        kwargs: dict[str, Any] = {"inputs_embeds": inputs_embeds}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        return self._embedding_module(**kwargs)

    # ----- internals -------------------------------------------------------

    @staticmethod
    def _resolve_embedding_module(model: Any) -> Any:
        """Locate the embedding submodule across BERT/RoBERTa-family checkpoints."""
        for attr in ("bert", "roberta", "electra"):
            backbone = getattr(model, attr, None)
            if backbone is not None and hasattr(backbone, "embeddings"):
                return backbone.embeddings
        # Fallback: try a generic ``embeddings`` attribute on the top model
        if hasattr(model, "embeddings"):
            return model.embeddings
        raise AttributeError(
            "Could not locate embedding submodule on model — supported: "
            "BERT (model.bert.embeddings), RoBERTa (model.roberta.embeddings), "
            "ELECTRA (model.electra.embeddings)"
        )
