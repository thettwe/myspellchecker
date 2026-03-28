"""
ONNX Export module.

Converts PyTorch Transformer models to optimized ONNX format.
"""

import logging
import os
import shutil
import sys
import warnings
from io import StringIO
from pathlib import Path

from myspellchecker.core.constants import DEFAULT_TOKENIZER_FILE
from myspellchecker.training.constants import DEFAULT_DUMMY_TEXT, DEFAULT_OPSET_VERSION
from myspellchecker.utils.logging_utils import get_logger

try:
    import onnxruntime
    import torch
    from onnxruntime.quantization import QuantType, quant_pre_process, quantize_dynamic
    from transformers import (
        AutoModelForMaskedLM,
        PreTrainedTokenizerFast,
    )

except ImportError:
    torch = None  # type: ignore
    onnxruntime = None  # type: ignore
    quantize_dynamic = None  # type: ignore
    quant_pre_process = None  # type: ignore


class ONNXExporter:
    """
    Handles exporting PyTorch models to ONNX.
    """

    def __init__(self):
        self.logger = get_logger(__name__)
        if torch is None or onnxruntime is None:
            raise ImportError(
                "Exporting requires 'torch', 'transformers', and 'onnxruntime'. "
                "Please install via pip install 'myspellchecker[train,ai]'"
            )

    def export(
        self,
        model_dir: str,
        output_dir: str,
        opset_version: int = DEFAULT_OPSET_VERSION,
        quantize: bool = True,
    ) -> str:
        """
        Convert a Hugging Face PyTorch model to ONNX.

        Args:
            model_dir: Directory containing the PyTorch model (pytorch_model.bin).
            output_dir: Directory to save the .onnx file.
            opset_version: ONNX opset version (default: 18).
            quantize: Whether to apply dynamic quantization (int8).

        Returns:
            Path to the final .onnx file.
        """
        self.logger.info(f"Exporting model from {model_dir} to ONNX...")
        os.makedirs(output_dir, exist_ok=True)

        # 1. Load Model & Tokenizer
        try:
            tokenizer = PreTrainedTokenizerFast.from_pretrained(model_dir)
            model = AutoModelForMaskedLM.from_pretrained(model_dir)
            model.eval()
        except (OSError, RuntimeError) as e:
            raise ValueError(f"Failed to load model from {model_dir}: {e}") from e

        # 2. Create Dummy Input
        # We need a sample input to trace the graph.
        # Using Myanmar text to ensure proper optimization for Myanmar character sequences.
        text = DEFAULT_DUMMY_TEXT
        inputs = tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # 3. Define Output Path
        onnx_path = os.path.join(output_dir, "model.onnx")

        # 4. Export
        # We use dynamic axes so the model accepts any sequence length.
        symbolic_names = {0: "batch_size", 1: "sequence_length"}
        self._export_onnx(
            model=model,
            input_tuple=(input_ids, attention_mask),
            onnx_path=onnx_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": symbolic_names,
                "attention_mask": symbolic_names,
                "logits": symbolic_names,
            },
            opset_version=opset_version,
        )

        # 5. Quantization (Optional but recommended)
        if quantize:
            self._quantize(onnx_path, output_dir)

        # 6. Copy Tokenizer JSON for convenience
        # The user needs tokenizer.json next to model.onnx
        src_tokenizer = Path(model_dir) / DEFAULT_TOKENIZER_FILE
        dst_tokenizer = Path(output_dir) / DEFAULT_TOKENIZER_FILE
        if src_tokenizer.exists():
            shutil.copy(src_tokenizer, dst_tokenizer)

        return onnx_path

    def _export_onnx(
        self,
        model,
        input_tuple: tuple,
        onnx_path: str,
        input_names: list[str],
        output_names: list[str],
        dynamic_axes: dict,
        opset_version: int = DEFAULT_OPSET_VERSION,
    ) -> None:
        """Export a PyTorch model to ONNX with suppressed warnings."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")

            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout = StringIO()
            sys.stderr = StringIO()

            try:
                torch.onnx.export(
                    model,
                    input_tuple,
                    onnx_path,
                    opset_version=opset_version,
                    do_constant_folding=True,
                    input_names=input_names,
                    output_names=output_names,
                    dynamic_axes=dynamic_axes,
                    dynamo=False,
                )
            finally:
                sys.stdout, sys.stderr = old_stdout, old_stderr

        self.logger.info(f"Base ONNX model exported to {onnx_path}")

    def _quantize(self, onnx_path: str, output_dir: str) -> None:
        """Apply dynamic int8 quantization to an ONNX model."""
        quantized_path = os.path.join(output_dir, "model.quant.onnx")
        preprocessed_path = os.path.join(output_dir, "model.preprocess.onnx")
        self.logger.info("Quantizing model to Int8...")

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")

                root_logger = logging.getLogger()
                prev_root_level = root_logger.level
                root_logger.setLevel(logging.ERROR)

                old_stdout, old_stderr = sys.stdout, sys.stderr
                sys.stdout = StringIO()
                sys.stderr = StringIO()

                try:
                    quantize_input = onnx_path
                    try:
                        quant_pre_process(
                            input_model_path=onnx_path,
                            output_model_path=preprocessed_path,
                        )
                        quantize_input = preprocessed_path
                    except Exception as e:
                        self.logger.debug("ONNX preprocessing skipped: %s", e)

                    quantize_dynamic(
                        model_input=quantize_input,
                        model_output=quantized_path,
                        weight_type=QuantType.QUInt8,
                    )
                finally:
                    sys.stdout, sys.stderr = old_stdout, old_stderr
                    root_logger.setLevel(prev_root_level)

            Path(preprocessed_path).unlink(missing_ok=True)

            base_path = os.path.join(output_dir, "model.base.onnx")
            shutil.move(onnx_path, base_path)
            shutil.move(quantized_path, onnx_path)

            self.logger.info(f"Quantized model saved to {onnx_path}")
            self.logger.info(f"Original fp32 model preserved at {base_path}")
        except (RuntimeError, ValueError, OSError) as e:
            self.logger.error(f"Quantization failed: {e}")
            self.logger.info("Falling back to non-quantized model")
            Path(preprocessed_path).unlink(missing_ok=True)
