"""Shared inference backends for ONNX and PyTorch model loading."""

from __future__ import annotations

from typing import Any, cast

import numpy as np

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

__all__ = [
    "EncodingResult",
    "HFTokenizerWrapper",
    "PyTorchInferenceSession",
    "RawTokenizersWrapper",
]


class EncodingResult:
    """Unified encoding result for both tokenizer formats."""

    __slots__ = ("ids", "offsets")

    def __init__(self, ids: list[int], offsets: list[tuple[int, int]]):
        self.ids = ids
        self.offsets = offsets


class HFTokenizerWrapper:
    """
    Wrapper to unify HuggingFace transformers tokenizer API with tokenizers library.

    This allows SemanticChecker to work with both:
    - Custom tokenizer.json files (from tokenizers library)
    - HuggingFace pretrained tokenizers (XLM-RoBERTa, mBERT, etc.)
    """

    def __init__(self, hf_tokenizer: Any):
        """
        Initialize wrapper with a HuggingFace tokenizer.

        Args:
            hf_tokenizer: A tokenizer from transformers.AutoTokenizer
        """
        self._tok = hf_tokenizer
        self._offset_warning_shown = False

    def encode(self, text: str) -> EncodingResult:
        """
        Encode text and return unified result.

        Args:
            text: Input text to tokenize.

        Returns:
            EncodingResult with ids and offsets.
        """
        encoded = self._tok(text, return_offsets_mapping=True)
        offsets = encoded.get("offset_mapping")

        # Warn if tokenizer doesn't support offset_mapping
        if offsets is None:
            if not self._offset_warning_shown:
                import logging

                logging.getLogger(__name__).warning(
                    "Tokenizer does not support offset_mapping. "
                    "Myanmar word alignment will fall back to string replacement."
                )
                self._offset_warning_shown = True
            offsets = []

        return EncodingResult(
            ids=encoded["input_ids"],
            offsets=offsets,
        )

    def token_to_id(self, token: str) -> int | None:
        """Convert token string to ID."""
        token_id = self._tok.convert_tokens_to_ids(token)
        # HuggingFace returns unk_token_id for unknown tokens
        if token_id == self._tok.unk_token_id:
            return None
        return cast(int, token_id)

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs to string."""
        return cast(str, self._tok.decode(ids, skip_special_tokens=True))


class RawTokenizersWrapper:
    """
    Adapter for tokenizers.Tokenizer (raw tokenizers library) to the same
    interface as HFTokenizerWrapper.

    Used when AutoTokenizer cannot load the tokenizer (e.g. tokenizer_class
    is "TokenizersBackend" — a raw tokenizers.Tokenizer serialized as JSON).
    """

    def __init__(self, tok: Any):
        self._tok = tok

    def encode(self, text: str) -> EncodingResult:
        encoding = self._tok.encode(text)
        return EncodingResult(ids=encoding.ids, offsets=encoding.offsets)

    def token_to_id(self, token: str) -> int | None:
        return cast(int | None, self._tok.token_to_id(token))

    def decode(self, ids: list[int]) -> str:
        return cast(str, self._tok.decode(ids))


class PyTorchInferenceSession:
    """
    PyTorch-based inference session as alternative to ONNX Runtime.

    This provides compatibility when ONNX Runtime is not available
    (e.g., Python 3.14 where onnxruntime doesn't have wheels yet).
    """

    def __init__(self, model: Any, device: str = "cpu"):
        """
        Initialize with a PyTorch model.

        Args:
            model: A PyTorch nn.Module (e.g., from transformers)
            device: Device to run inference on ("cpu" or "cuda")
        """
        self.model = model
        self.device = device
        self.model.eval()
        self.model.to(device)

        # Create input/output info to match ONNX Runtime API
        self._inputs = [type("Input", (), {"name": "input_ids"})()]
        self._outputs = [type("Output", (), {"name": "logits"})()]

    def get_inputs(self) -> list[Any]:
        """Return list of input specifications (ONNX Runtime compatible API)."""
        return self._inputs

    def get_outputs(self) -> list[Any]:
        """Return list of output specifications (ONNX Runtime compatible API)."""
        return self._outputs

    def run(self, output_names: list[str] | None, input_dict: dict) -> list[np.ndarray]:
        """
        Run inference (ONNX Runtime compatible API).

        Args:
            output_names: Names of outputs to return (ignored, returns all)
            input_dict: Dictionary mapping input names to numpy arrays

        Returns:
            List of output numpy arrays
        """
        if torch is None:
            raise ImportError("PyTorch is required for PyTorchInferenceSession")

        with torch.no_grad():
            # Convert numpy arrays to torch tensors
            input_ids = torch.tensor(input_dict["input_ids"], dtype=torch.long, device=self.device)

            attention_mask = None
            if "attention_mask" in input_dict:
                attention_mask = torch.tensor(
                    input_dict["attention_mask"], dtype=torch.long, device=self.device
                )

            # Run forward pass
            if attention_mask is not None:
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                outputs = self.model(input_ids=input_ids)

            # Get logits from model output
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

            return [logits.cpu().numpy()]
