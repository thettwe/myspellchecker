"""
Semantic Context Checker using Masked Language Modeling (BERT/RoBERTa).

This module implements AI-powered context checking. It uses a pre-trained
Transformer model to predict the most likely words in a sentence, helping
to catch semantic errors that statistical N-grams might miss.

Architecture:
    - Tokenizer: ByteLevelBPE (custom) or HuggingFace (XLM-RoBERTa, mBERT, etc.)
    - Model: ONNX Runtime (preferred) or PyTorch (fallback)

Design Pattern:
    - **Factory Pattern**: Supports multiple initialization modes (path or pre-loaded)
    - **Adapter Pattern**: HFTokenizerWrapper adapts HuggingFace tokenizers
    - **Backend Abstraction**: PyTorchInferenceSession provides ONNX-compatible API

Inference Backends:
    - ONNX Runtime: Fast, quantized models, production-ready
    - PyTorch: Slower but more compatible, useful for testing/development

Result Types:
    - predict_mask returns: list[tuple[str, float]] - (word, score) pairs, higher = better
    - is_semantic_error returns: str | None - suggested replacement or None if correct
    - scan_sentence returns: list[tuple[int, str, list[str], float]] - error positions

Parameter Validation:
    - model_path/tokenizer_path OR model/tokenizer must be provided (validated at init)
    - Mask token availability validated at initialization
    - Model dimensions validated via test inference at initialization

Myanmar-specific handling:
    - BPE tokens may not align with Myanmar word boundaries
    - Word alignment maps Myanmar words to their corresponding BPE token spans
    - Multi-token masking ensures complete words are masked, not partial tokens

Caching:
    - Encoding cache: LRU cache (512 entries) for tokenization results
    - Alignment cache: LRU cache (256 entries) for word-token alignment
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from myspellchecker.algorithms.inference_backends import (
    EncodingResult,
    HFTokenizerWrapper,
    PyTorchInferenceSession,
)
from myspellchecker.core.constants import CONFIDENCE_FLOOR, get_myanmar_char_set, is_myanmar_text
from myspellchecker.core.exceptions import InferenceError, ModelLoadError
from myspellchecker.utils.cache import LRUCache
from myspellchecker.utils.logging_utils import get_logger

# Try importing inference backends
try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

try:
    from tokenizers import Tokenizer
except ImportError:
    Tokenizer = None


__all__ = [
    "SemanticChecker",
]

# Cache sizes for SemanticChecker
# Sized for typical document processing (see also joint_segment_tagger.py)
ENCODING_CACHE_SIZE = 512  # Tokenization encoding results
ALIGNMENT_CACHE_SIZE = 256  # Word-token alignment results
LOGITS_CACHE_SIZE = 16  # Raw ONNX logits (eliminates duplicate forward passes)


class SemanticChecker:
    """
    AI-powered semantic spell checker.

    Uses a Masked Language Model to predict contextually appropriate words.
    """

    # --- Model logit scale factors for confidence calibration ---
    # Different transformer architectures produce logits on different scales.
    # Order matters: more specific patterns must come before generic ones
    # (e.g., "xlm-roberta" before "roberta", "distil" before "bert", "albert" before "bert").
    _MODEL_LOGIT_SCALES: dict[str, float] = {
        "xlm-roberta": 10.0,  # XLM-RoBERTa: ~5-15 range for top predictions
        "xlm": 10.0,  # Other XLM models
        "distil": 30.0,  # DistilBERT: ~15-50 range
        "albert": 40.0,  # ALBERT (must precede "bert")
        "bert": 50.0,  # Standard BERT/mBERT: ~20-100 range
        "roberta": 15.0,  # RoBERTa (non-XLM, matched after xlm-roberta)
    }
    _DEFAULT_LOGIT_SCALE: float = 10.0  # Fallback for unknown model architectures

    # --- Confidence calibration constants ---
    # Sigmoid-like transformation to bound raw logits between 0 and 1
    _CONFIDENCE_HIGH_THRESHOLD: float = 1.0  # Normalized score above which saturation applies
    _CONFIDENCE_SATURATION_NUMERATOR: float = 0.2  # Controls approach rate to 1.0
    _CONFIDENCE_SATURATION_OFFSET: float = 1.0  # Denominator offset in saturation formula
    _CONFIDENCE_LOW_BASE: float = 0.3  # Base confidence in linear region
    _CONFIDENCE_LOW_SCALE: float = 0.5  # Linear scaling factor in low-confidence region

    # --- Prefix skip margins ---
    # How close a prefix prediction score must be to top score to skip flagging
    _PREFIX_SKIP_MARGIN_SHORT: float = 0.25  # Single-char words (len <= 1)
    _PREFIX_SKIP_MARGIN_MEDIUM: float = 0.45  # Two-char words (len == 2)
    _PREFIX_SKIP_MARGIN_LONG: float = 1.0  # Longer words (len >= 3)
    _PREFIX_COMPETING_NON_PREFIX_GAP: float = (
        0.25  # Gap below best prefix for non-prefix competitor
    )

    # --- ONNX compatibility bounds ---
    _ONNX_MIN_SUPPORTED_OPSET: int = 7  # Minimum opset version for onnxruntime 1.16+
    _ONNX_MAX_SUPPORTED_OPSET: int = 20  # Maximum tested opset version

    # --- Model validation thresholds ---
    _MIN_VOCAB_SIZE: int = 1000  # Minimum vocabulary size for a valid MLM model

    # --- Beam search parameters ---
    _BEAM_WIDTH_MULTIPLIER: int = 3  # Multiplied by top_k for beam width
    _BEAM_WIDTH_CAP: int = 20  # Maximum beam width to prevent combinatorial explosion
    _BEAM_DEDUP_MULTIPLIER: int = 2  # Extra candidates to generate for deduplication

    # --- Multi-token decoding ---
    _MULTI_TOKEN_CANDIDATE_MULTIPLIER: int = 2  # Extra candidates per masked position

    # --- Sentence scanning constants ---
    _SCAN_PRESENCE_TOP_N: int = 5  # Top-N predictions to check for word presence
    _SCAN_SUGGESTION_POOL_SIZE: int = 10  # How many predictions to consider for suggestions
    _SCAN_MAX_SUGGESTIONS: int = 5  # Maximum suggestions to return per error
    _SCAN_MIN_SUGGESTION_LEN: int = 2  # Minimum character length for a valid suggestion
    _SCAN_MIN_MYANMAR_CHAR_RATIO: float = 0.5  # Minimum Myanmar character ratio (>50%)
    _SCAN_SIMILARITY_TOP_N: int = 3  # Suggestions to check for similarity boost
    _SCAN_CONFIDENCE_BASE_WEIGHT: float = 0.5  # Base weight in similarity-boosted confidence
    _SCAN_CONFIDENCE_SIMILARITY_WEIGHT: float = 0.5  # Similarity weight in confidence formula

    # --- Score candidates ---
    _SCORE_FLOOR_OFFSET: float = 1.0  # Subtracted from min logit for absent candidates
    _SCORE_DEFAULT_TOP_K: int = 200  # Default top_k for score_candidates predictions

    # --- Semantic error detection ---
    _ERROR_CHECK_TOP_N: int = 5  # Top-N predictions for prefix evidence in is_semantic_error

    # --- Short word threshold ---
    _SHORT_WORD_MAX_LEN: int = 2  # Words at or below this length are "short"

    def __init__(
        self,
        model_path: str | None = None,
        tokenizer_path: str | None = None,
        model: Any | None = None,
        tokenizer: Any | None = None,
        num_threads: int = 1,
        predict_top_k: int = 5,
        check_top_k: int = 10,
        use_pytorch: bool = False,
        allow_extended_myanmar: bool = False,
        cache_config: Any | None = None,
        semantic_config: Any | None = None,
    ):
        """
        Initialize the Semantic Checker.

        Args:
            model_path: Path to the model file. Can be either:
                - An .onnx file for ONNX Runtime
                - A HuggingFace model name (e.g., "xlm-roberta-base") for PyTorch
            tokenizer_path: Path to tokenizer. Can be either:
                - A file path to tokenizer.json (custom tokenizer format)
                - A directory path containing HuggingFace tokenizer files
                  (XLM-RoBERTa, mBERT, etc.)
            model: Pre-loaded inference session (ONNX or PyTorchInferenceSession).
            tokenizer: Pre-loaded Tokenizer or HFTokenizerWrapper object.
            num_threads: Number of threads for inference (default: 1).
            predict_top_k: Default number of predictions for predict_mask (default: 5).
            check_top_k: Number of predictions to check for error detection (default: 10).
            use_pytorch: Force PyTorch backend instead of ONNX (default: False).
                         Auto-detected if model_path is not .onnx file.
            allow_extended_myanmar: Allow Extended Myanmar characters for non-Burmese
                         Myanmar-script languages, including:
                         - Extended Core Block (U+1050-U+109F)
                         - Extended-A (U+AA60-U+AA7F)
                         - Extended-B (U+A9E0-U+A9FF)
                         - Non-standard core chars (U+1022, U+1028, U+1033-U+1035)
                         When False (default), enforces strict Burmese-only scope.
            cache_config: Optional AlgorithmCacheConfig for configuring cache sizes.
        """
        self.logger = get_logger(__name__)
        self.predict_top_k = predict_top_k
        self.check_top_k = check_top_k
        self.allow_extended_myanmar = allow_extended_myanmar

        if model and tokenizer:
            self.session = model
            self.tokenizer = tokenizer
            self.logger.info("Using provided Semantic Model and Tokenizer objects")
        elif model_path and tokenizer_path:
            if not os.path.exists(tokenizer_path) and not self._is_hf_model_name(tokenizer_path):
                raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")

            self.logger.info(f"Loading Semantic Model: {model_path}")

            # Load Tokenizer - supports both formats
            self._load_tokenizer(tokenizer_path)

            # Determine inference backend
            is_onnx_model = model_path.endswith(".onnx") and os.path.exists(model_path)

            if is_onnx_model and not use_pytorch:
                # ONNX Runtime backend
                self._load_onnx_model(model_path, num_threads)
            else:
                # PyTorch backend (for HuggingFace model names or when use_pytorch=True)
                self._load_pytorch_model(model_path)
        else:
            raise ModelLoadError(
                "Either (model_path, tokenizer_path) OR (model, tokenizer) must be provided."
            )

        self.mask_token = "<mask>"  # RoBERTa default, can be [MASK] for BERT
        self.mask_token_id = self.tokenizer.token_to_id(self.mask_token)

        if self.mask_token_id is None:
            # Fallback to BERT style
            self.mask_token = "[MASK]"
            self.mask_token_id = self.tokenizer.token_to_id(self.mask_token)

        # Validate mask token is available - critical for semantic checking
        if self.mask_token_id is None:
            raise ModelLoadError(
                "Tokenizer does not have a mask token. "
                "Tried '<mask>' and '[MASK]'. "
                "Ensure the tokenizer supports masked language modeling."
            )

        # Validate ONNX model has expected inputs/outputs
        session_inputs = self.session.get_inputs()
        session_outputs = self.session.get_outputs()
        if not session_inputs:
            raise ModelLoadError(
                "ONNX model has no inputs. Ensure the model is a valid masked language model."
            )
        if not session_outputs:
            raise ModelLoadError(
                "ONNX model has no outputs. Ensure the model is a valid masked language model."
            )
        self.input_name = session_inputs[0].name
        self.output_name = session_outputs[0].name

        # Validate embedding dimensions
        self._validate_model_dimensions()

        # Cache whether model expects attention_mask (avoids repeated get_inputs calls)
        self._has_attention_mask_input = len(session_inputs) > 1

        # Detect max sequence length from model input shape if available,
        # otherwise default to 512 (standard transformer limit).
        self._max_seq_len = self._detect_max_seq_len(session_inputs)

        # Myanmar-specific: track whether to use word-aligned masking
        self.use_word_alignment = True

        # Track model name for confidence calibration.
        # When using pre-loaded objects without model_path, _model_name defaults to ""
        # and _get_model_logit_scale() falls back to _DEFAULT_LOGIT_SCALE.
        # Callers using pre-loaded objects should pass model_path for correct calibration.
        self._model_name = self._extract_model_name(model_path or "")

        # Cache sizes: use config if provided, otherwise module-level defaults
        _enc_size = ENCODING_CACHE_SIZE
        _align_size = ALIGNMENT_CACHE_SIZE
        if cache_config is not None:
            _enc_size = getattr(cache_config, "semantic_encoding_cache_size", _enc_size)
            _align_size = getattr(cache_config, "semantic_alignment_cache_size", _align_size)

        # Embedding cache for tokenization results.
        self._encoding_cache: LRUCache[tuple[list[int], list[tuple[int, int]]]] = LRUCache(
            maxsize=_enc_size
        )

        # Cache for word alignment results.
        self._alignment_cache: LRUCache[tuple[list[int], int, int]] = LRUCache(maxsize=_align_size)

        # Cache for raw ONNX logits — keyed by (sentence, target_word, occurrence).
        # Eliminates duplicate forward passes when predict_mask() and
        # score_mask_candidates() are called for the same masked position.
        _logits_size = LOGITS_CACHE_SIZE
        if cache_config is not None:
            _logits_size = getattr(cache_config, "semantic_logits_cache_size", _logits_size)
        self._logits_cache: LRUCache[tuple[np.ndarray | None, list[int]]] = LRUCache(
            maxsize=_logits_size
        )

        # Override class-level constants from SemanticConfig if provided.
        # Class-level values remain as fallback defaults for backward compatibility.
        if semantic_config is not None:
            self._CONFIDENCE_HIGH_THRESHOLD = semantic_config.confidence_high_threshold
            self._CONFIDENCE_SATURATION_NUMERATOR = semantic_config.confidence_saturation_numerator
            self._CONFIDENCE_SATURATION_OFFSET = semantic_config.confidence_saturation_offset
            self._CONFIDENCE_LOW_BASE = semantic_config.confidence_low_base
            self._CONFIDENCE_LOW_SCALE = semantic_config.confidence_low_scale
            self._PREFIX_SKIP_MARGIN_SHORT = semantic_config.prefix_skip_margin_short
            self._PREFIX_SKIP_MARGIN_MEDIUM = semantic_config.prefix_skip_margin_medium
            self._PREFIX_SKIP_MARGIN_LONG = semantic_config.prefix_skip_margin_long
            self._PREFIX_COMPETING_NON_PREFIX_GAP = semantic_config.prefix_competing_non_prefix_gap
            self._BEAM_WIDTH_MULTIPLIER = semantic_config.beam_width_multiplier
            self._BEAM_WIDTH_CAP = semantic_config.beam_width_cap
            self._BEAM_DEDUP_MULTIPLIER = semantic_config.beam_dedup_multiplier
            self._MULTI_TOKEN_CANDIDATE_MULTIPLIER = (
                semantic_config.multi_token_candidate_multiplier
            )
            self._SCAN_PRESENCE_TOP_N = semantic_config.scan_presence_top_n
            self._SCAN_SUGGESTION_POOL_SIZE = semantic_config.scan_suggestion_pool_size
            self._SCAN_MAX_SUGGESTIONS = semantic_config.scan_max_suggestions
            self._SCAN_MIN_SUGGESTION_LEN = semantic_config.scan_min_suggestion_len
            self._SCAN_MIN_MYANMAR_CHAR_RATIO = semantic_config.scan_min_myanmar_char_ratio
            self._SCAN_CONFIDENCE_BASE_WEIGHT = semantic_config.scan_confidence_base_weight
            self._SCAN_CONFIDENCE_SIMILARITY_WEIGHT = (
                semantic_config.scan_confidence_similarity_weight
            )
            self._SCORE_FLOOR_OFFSET = semantic_config.score_floor_offset
            self._SCORE_DEFAULT_TOP_K = semantic_config.score_default_top_k
            self._ERROR_CHECK_TOP_N = semantic_config.error_check_top_n
            self._SHORT_WORD_MAX_LEN = semantic_config.short_word_max_len

    def clear_inference_cache(self) -> None:
        """Clear the logits cache between documents to free memory."""
        self._logits_cache.clear()

    def has_cached_logits(self, sentence: str, word: str, occurrence: int = 0) -> bool:
        """Check whether logits for the given (sentence, word, occurrence) are cached.

        This allows callers (e.g. SemanticValidationStrategy) to skip a
        redundant ``predict_mask`` call when ConfusableSemanticStrategy has
        already analyzed the same position and found it clean.
        """
        return (sentence, word, occurrence) in self._logits_cache

    def _is_hf_model_name(self, name: str) -> bool:
        """Check if the name looks like a HuggingFace model name."""
        # HuggingFace model names contain / or are known model names
        hf_patterns = [
            "/",  # org/model format
            "xlm-roberta",
            "bert",
            "roberta",
            "distilbert",
            "albert",
        ]
        name_lower = name.lower()
        return any(pattern in name_lower for pattern in hf_patterns)

    def _extract_model_name(self, model_path: str) -> str:
        """
        Extract model name from path for confidence calibration.

        Args:
            model_path: Path to model or HuggingFace model name.

        Returns:
            Lowercase model name string.
        """
        import os

        name = os.path.basename(model_path).lower()
        if "/" in model_path:
            # HuggingFace format: org/model
            name = model_path.split("/")[-1].lower()
        return name

    def _get_model_logit_scale(self) -> float:
        """
        Get appropriate logit scale for the current model.

        Different transformer models produce logits on different scales:
        - XLM-RoBERTa: ~5-15 range for top predictions
        - BERT/mBERT: ~20-100 range for top predictions
        - DistilBERT: ~15-50 range

        Returns:
            Appropriate scale factor for normalizing confidence.
        """
        model_name = self._model_name.lower()

        # Match against known model families (order matters: check specific before generic)
        for key, scale in self._MODEL_LOGIT_SCALES.items():
            if key in model_name:
                return scale

        return self._DEFAULT_LOGIT_SCALE

    def _calibrate_confidence(self, raw_score: float, logit_scale: float) -> float:
        """
        Calibrate raw model score to normalized confidence.

        Applies sigmoid-like transformation to convert raw logits
        to a bounded confidence score between 0 and 1.

        Args:
            raw_score: Raw logit score from model.
            logit_scale: Model-specific scale factor.

        Returns:
            Calibrated confidence score between 0.0 and 1.0.
        """
        if raw_score <= 0:
            return CONFIDENCE_FLOOR  # Minimum confidence for non-positive scores

        # Normalize by scale and apply soft saturation
        normalized = raw_score / logit_scale

        # Sigmoid-like transformation with smooth transition
        # This prevents overconfidence from very high logits
        if normalized >= self._CONFIDENCE_HIGH_THRESHOLD:
            # High confidence region - asymptotically approach 1.0
            confidence = 1.0 - (
                self._CONFIDENCE_SATURATION_NUMERATOR
                / (self._CONFIDENCE_SATURATION_OFFSET + normalized)
            )
        else:
            # Lower confidence region - linear-ish scaling
            confidence = self._CONFIDENCE_LOW_BASE + (self._CONFIDENCE_LOW_SCALE * normalized)

        return min(1.0, max(CONFIDENCE_FLOOR, confidence))

    def _load_tokenizer(self, tokenizer_path: str) -> None:
        """Load tokenizer from path or HuggingFace model name."""
        if os.path.isdir(tokenizer_path):
            # HuggingFace format directory — try AutoTokenizer first,
            # fall back to raw tokenizer.json if config is incompatible
            try:
                from transformers import AutoTokenizer

                hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
                self.tokenizer = HFTokenizerWrapper(hf_tokenizer)
                self.logger.info("Loaded HuggingFace tokenizer from directory")
            except ImportError as e:
                raise ImportError(
                    "HuggingFace tokenizer format requires 'transformers'. "
                    "Install via 'pip install transformers'."
                ) from e
            except (ValueError, OSError) as e:
                # AutoTokenizer failed (e.g., unrecognized tokenizer_class).
                # Fall back to loading tokenizer.json directly if present.
                fallback = os.path.join(tokenizer_path, "tokenizer.json")
                if os.path.isfile(fallback) and Tokenizer is not None:
                    self.tokenizer = Tokenizer.from_file(fallback)
                    self.logger.info("AutoTokenizer failed (%s), loaded tokenizer.json fallback", e)
                else:
                    raise
        elif os.path.isfile(tokenizer_path):
            # Custom tokenizer.json file format
            if Tokenizer is None:
                raise ImportError(
                    "Custom tokenizer format requires 'tokenizers'. "
                    "Install via 'pip install tokenizers'."
                )
            self.tokenizer = Tokenizer.from_file(tokenizer_path)
            self.logger.info("Loaded custom tokenizer.json format")
        elif self._is_hf_model_name(tokenizer_path):
            # HuggingFace model name (e.g., "xlm-roberta-base")
            try:
                from transformers import AutoTokenizer

                hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
                self.tokenizer = HFTokenizerWrapper(hf_tokenizer)
                self.logger.info(f"Loaded HuggingFace tokenizer: {tokenizer_path}")
            except ImportError as e:
                raise ImportError(
                    "HuggingFace tokenizer format requires 'transformers'. "
                    "Install via 'pip install transformers'."
                ) from e
        else:
            raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")

    def _check_onnx_compatibility(self, model_path: str) -> None:
        """
        Check ONNX model version compatibility with installed runtime.

        Validates that the model's opset version is compatible with the
        installed ONNX Runtime version.

        Args:
            model_path: Path to the ONNX model file.

        Raises:
            ValueError: If model version is incompatible with runtime.
        """
        try:
            import onnx

            model = onnx.load(model_path, load_external_data=False)
            model_opset = model.opset_import[0].version if model.opset_import else 0
            ir_version = model.ir_version

            # Get supported opset range from ONNX Runtime
            min_supported_opset = self._ONNX_MIN_SUPPORTED_OPSET
            max_supported_opset = self._ONNX_MAX_SUPPORTED_OPSET

            if model_opset < min_supported_opset:
                raise ModelLoadError(
                    f"ONNX model opset {model_opset} is too old. "
                    f"Minimum supported opset is {min_supported_opset}. "
                    f"Please re-export the model with a newer opset version."
                )

            if model_opset > max_supported_opset:
                self.logger.warning(
                    f"ONNX model opset {model_opset} is newer than tested opset "
                    f"{max_supported_opset}. Some operations may not be supported. "
                    f"Consider upgrading onnxruntime."
                )

            self.logger.debug(f"ONNX model: opset={model_opset}, ir_version={ir_version}")

        except ImportError:
            # onnx package not installed, skip version check
            self.logger.debug("ONNX package not installed, skipping version compatibility check")
        except (OSError, ValueError, RuntimeError) as e:
            # Non-fatal: log warning but continue loading
            self.logger.warning(f"Could not verify ONNX model version: {e}")

    def _load_onnx_model(self, model_path: str, num_threads: int) -> None:
        """Load ONNX model using ONNX Runtime."""
        if ort is None:
            raise ImportError(
                "ONNX Runtime is not available. "
                "Either install onnxruntime or use use_pytorch=True. "
                "Install via 'pip install onnxruntime'."
            )

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        # Check model version compatibility
        self._check_onnx_compatibility(model_path)

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = num_threads
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(model_path, sess_options)
        self.logger.info("Loaded ONNX model with ONNX Runtime backend")

    def _load_pytorch_model(self, model_path: str) -> None:
        """Load PyTorch model (HuggingFace transformers)."""
        if torch is None:
            raise ImportError("PyTorch is not available. Install via 'pip install torch'.")

        try:
            from transformers import AutoModelForMaskedLM

            # Load from HuggingFace model name or local path
            model = AutoModelForMaskedLM.from_pretrained(model_path)
            self.session = PyTorchInferenceSession(model)
            self.logger.info(f"Loaded PyTorch model: {model_path}")
        except ImportError as e:
            raise ImportError(
                "PyTorch backend requires 'transformers'. Install via 'pip install transformers'."
            ) from e

    def _validate_model_dimensions(self) -> None:
        """
        Validate model output dimensions for embedding compatibility.

        Performs a test inference with a simple input to verify:
        1. Model produces output in expected shape [batch, seq_len, vocab_size]
        2. Vocabulary size matches tokenizer vocabulary
        3. Sequence length handling is reasonable

        Raises:
            ValueError: If model dimensions are incompatible or unexpected.
        """
        try:
            # Create a simple test input with mask token
            test_text = f"test {self.mask_token} text"
            encoded = self.tokenizer.encode(test_text)
            input_ids = np.array([encoded.ids], dtype=np.int64)

            # Run test inference
            inputs = {self.input_name: input_ids}

            # Add attention mask if model expects it
            if len(self.session.get_inputs()) > 1:
                attention_mask = np.ones((1, len(encoded.ids)), dtype=np.int64)
                inputs["attention_mask"] = attention_mask

            logits = self.session.run([self.output_name], inputs)[0]

            # Validate output shape: [batch_size, seq_len, vocab_size]
            if logits.ndim != 3:
                raise InferenceError(
                    f"Model output has unexpected dimensions: {logits.ndim}. "
                    f"Expected 3D tensor [batch, seq_len, vocab_size]. "
                    f"Got shape: {logits.shape}"
                )

            batch_size, seq_len, vocab_size = logits.shape

            # Validate batch size
            if batch_size != 1:
                self.logger.warning(f"Unexpected batch size in output: {batch_size}. Expected 1.")

            # Validate sequence length matches input
            if seq_len != len(encoded.ids):
                self.logger.warning(
                    f"Sequence length mismatch: output has {seq_len} tokens, "
                    f"but input had {len(encoded.ids)} tokens. "
                    f"This may indicate padding or truncation differences."
                )

            # Validate vocab size is reasonable
            if vocab_size < self._MIN_VOCAB_SIZE:
                raise InferenceError(
                    f"Model vocabulary size ({vocab_size}) is unexpectedly small. "
                    f"This may indicate a model loading issue or incompatible model type."
                )

            # Log dimension info for debugging
            self.logger.debug(
                f"Model dimensions validated: vocab_size={vocab_size}, test_seq_len={seq_len}"
            )

            # Store vocab size for later validation
            self._vocab_size = vocab_size

        except RuntimeError as e:
            raise InferenceError(
                f"Model dimension validation failed during test inference: {e}. "
                f"Ensure the model is a valid masked language model (MLM) "
                f"that produces logits over vocabulary."
            ) from e
        except (IndexError, KeyError) as e:
            raise InferenceError(
                f"Model output structure is incompatible: {e}. "
                f"Expected output tensor with shape [batch, seq_len, vocab_size]."
            ) from e

    # --- Default max sequence length for pre-allocated buffers ---
    _DEFAULT_MAX_SEQ_LEN: int = 512

    @staticmethod
    def _detect_max_seq_len(session_inputs: list[Any]) -> int:
        """Extract max sequence length from ONNX model input shape.

        ONNX models typically declare input shapes like ``[batch, seq_len]``
        where ``seq_len`` may be a concrete integer (e.g. 512) or a dynamic
        string symbol (e.g. ``"sequence"``).  When a concrete value is
        available, use it; otherwise fall back to 512.

        Args:
            session_inputs: List of ONNX input descriptors from
                ``session.get_inputs()``.

        Returns:
            Maximum sequence length for buffer pre-allocation.
        """
        if session_inputs:
            shape = getattr(session_inputs[0], "shape", None)
            if isinstance(shape, (list, tuple)) and len(shape) >= 2:
                seq_dim = shape[1]
                if isinstance(seq_dim, int) and seq_dim > 0:
                    return seq_dim
        return SemanticChecker._DEFAULT_MAX_SEQ_LEN

    def _prepare_single_inference_inputs(
        self, token_ids: list[int]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build numpy arrays for a single-sequence inference call.

        Allocates fresh arrays per call to ensure thread safety under
        concurrent ``check_batch_async`` usage.  Truncates to the model's
        max sequence length to prevent ONNX runtime errors on long inputs.

        Args:
            token_ids: 1-D list of integer token IDs.

        Returns:
            ``(input_ids, attention_mask)`` -- contiguous arrays of shape
            ``(1, seq_len)``.
        """
        if len(token_ids) > self._max_seq_len:
            token_ids = token_ids[: self._max_seq_len]
        input_ids = np.array([token_ids], dtype=np.int64)
        attention_mask = np.ones((1, len(token_ids)), dtype=np.int64)
        return input_ids, attention_mask

    def _is_myanmar_text(self, text: str) -> bool:
        """Check if text contains Myanmar characters, respecting config.

        Uses the shared is_myanmar_text helper which respects the
        allow_extended_myanmar config flag.
        """
        return is_myanmar_text(text, allow_extended=self.allow_extended_myanmar)

    def _cached_encode(self, text: str) -> EncodingResult:
        """
        Encode text with LRU caching.

        Thread-safe caching of tokenization results to avoid
        recomputation for the same text. Uses LRUCache for proper
        LRU eviction instead of FIFO.

        Args:
            text: Input text to tokenize.

        Returns:
            EncodingResult with ids and offsets.
        """
        # Check cache first (LRUCache is thread-safe)
        cached = self._encoding_cache.get(text)
        if cached is not None:
            # Reconstruct EncodingResult from cached tuple
            return EncodingResult(ids=list(cached[0]), offsets=list(cached[1]))

        # Encode and cache
        result: EncodingResult = self.tokenizer.encode(text)

        # Store as tuple for LRUCache (LRUCache handles eviction automatically)
        self._encoding_cache.set(text, (list(result.ids), list(result.offsets)))

        return result

    def close(self) -> None:
        """Release the inference session and free native resources.

        Sets self.session to None so the GC can collect the underlying
        ONNX InferenceSession or PyTorchInferenceSession, and clears
        internal caches.  Idempotent.
        """
        self.session = None  # type: ignore[assignment]
        self._encoding_cache.clear()
        self._alignment_cache.clear()

    def cache_stats(self) -> dict[str, dict[str, Any]]:
        """
        Get statistics for all caches.

        Returns:
            Dictionary with stats for encoding and alignment caches.
            Each cache stats include: size, maxsize, hits, misses, hit_rate.
        """
        return {
            "encoding": self._encoding_cache.stats(),
            "alignment": self._alignment_cache.stats(),
        }

    def _find_all_word_positions(self, sentence: str, target_word: str) -> list[int]:
        """
        Find all positions of a word in a sentence.

        Args:
            sentence: Full sentence to search
            target_word: Word to find

        Returns:
            List of starting positions of each occurrence
        """
        if not target_word:
            return []

        # Prefer token-boundary matches when available (space/punctuation delimited
        # text), but fall back to raw substring matches for unsegmented Burmese.
        boundary_positions: list[int] = []
        all_positions: list[int] = []
        start = 0
        while True:
            pos = sentence.find(target_word, start)
            if pos == -1:
                break
            all_positions.append(pos)
            end = pos + len(target_word)
            if self._is_token_boundary(sentence, pos, end):
                boundary_positions.append(pos)
            # Advance by word length to avoid overlapping pseudo-matches.
            start = end
        return boundary_positions or all_positions

    @staticmethod
    def _is_token_boundary(sentence: str, start: int, end: int) -> bool:
        """Check whether [start:end] is bounded by non-word-like characters.

        Uses Myanmar Unicode range awareness instead of ``isalnum()`` which
        misclassifies dependent characters (medials, vowels, tone marks) as
        non-alphanumeric, causing incorrect boundary detection.
        """
        if start < 0 or end > len(sentence) or start >= end:
            return False

        def _is_myanmar_char(ch: str) -> bool:
            cp = ord(ch)
            return 0x1000 <= cp <= 0x109F

        # A boundary exists when the adjacent character is NOT part of a
        # Myanmar syllable (space, punctuation, non-Myanmar script, etc.).
        left_ok = start == 0 or not (
            sentence[start - 1].isalnum() or _is_myanmar_char(sentence[start - 1])
        )
        right_ok = end == len(sentence) or not (
            sentence[end].isalnum() or _is_myanmar_char(sentence[end])
        )
        return left_ok and right_ok

    def _mask_target_occurrence(
        self, sentence: str, target_word: str, occurrence: int, mask_token: str
    ) -> str:
        """
        Replace the specific occurrence of target_word with mask_token.

        If the requested occurrence is not found, returns original sentence.
        """
        positions = self._find_all_word_positions(sentence, target_word)
        if occurrence < 0 or occurrence >= len(positions):
            return sentence

        start = positions[occurrence]
        end = start + len(target_word)
        return f"{sentence[:start]}{mask_token}{sentence[end:]}"

    def _get_word_token_alignment(
        self, sentence: str, target_word: str, occurrence: int = 0
    ) -> tuple[list[int], int, int]:
        """
        Get alignment between Myanmar words and BPE tokens.

        For Myanmar text, BPE tokens often split words at arbitrary points.
        This method finds the token indices that correspond to a target word.

        Uses LRUCache for repeated alignments of the same sentence/word pair.

        Improved to handle multiple occurrences of the same word:
        - occurrence=0 returns the first occurrence (default)
        - occurrence=1 returns the second occurrence, etc.
        - If occurrence index is out of range, returns (-1, -1)

        Args:
            sentence: Full sentence
            target_word: Word to find alignment for
            occurrence: Which occurrence to find (0-indexed, default 0)

        Returns:
            Tuple of (token_ids, start_token_idx, end_token_idx) where
            start/end are the token indices spanning the target word.
            Returns (-1, -1) if word not found.
        """
        # Check alignment cache first (use string key for LRUCache)
        cache_key = f"{sentence}:{target_word}:{occurrence}"
        cached = self._alignment_cache.get(cache_key)
        if cached is not None:
            return cached

        # Use cached encoding
        encoded = self._cached_encode(sentence)
        token_ids = encoded.ids
        offsets = encoded.offsets  # List of (start_char, end_char) tuples

        # Find all positions of target_word in sentence
        positions = self._find_all_word_positions(sentence, target_word)

        if not positions or occurrence >= len(positions):
            result = (token_ids, -1, -1)
        else:
            word_start = positions[occurrence]
            word_end = word_start + len(target_word)

            # Find tokens that overlap with the word's character span
            start_token_idx = -1
            end_token_idx = -1

            for i, (tok_start, tok_end) in enumerate(offsets):
                # Check if token overlaps with word span
                if tok_start < word_end and tok_end > word_start:
                    if start_token_idx == -1:
                        start_token_idx = i
                    end_token_idx = i + 1  # exclusive end

            result = (token_ids, start_token_idx, end_token_idx)

        # Cache the result (LRUCache handles eviction automatically)
        self._alignment_cache.set(cache_key, result)

        return result

    def _create_word_aligned_mask(
        self, sentence: str, target_word: str, occurrence: int = 0
    ) -> tuple[np.ndarray | None, np.ndarray | None, list[int]]:
        """
        Create input with all tokens of target word masked.

        For Myanmar, we mask all BPE tokens that form the word, not just
        a single token position. This ensures complete word prediction.

        Args:
            sentence: Full sentence
            target_word: Word to mask
            occurrence: Which occurrence to mask (0-indexed, default 0)

        Returns:
            Tuple of (input_ids, attention_mask, mask_indices) where
            mask_indices contains all masked token positions.
        """
        token_ids, start_idx, end_idx = self._get_word_token_alignment(
            sentence, target_word, occurrence
        )

        if start_idx == -1 or end_idx == -1:
            # Fallback: word not found in tokenization
            return None, None, []

        # Create masked input
        input_ids = list(token_ids)
        mask_indices = []

        for i in range(start_idx, end_idx):
            if i >= self._max_seq_len:
                break
            input_ids[i] = self.mask_token_id
            mask_indices.append(i)

        if not mask_indices:
            return None, None, []

        # Reuse pre-allocated buffers instead of allocating new arrays
        input_ids_np, attention_mask_np = self._prepare_single_inference_inputs(input_ids)

        return input_ids_np, attention_mask_np, mask_indices

    def _beam_search_multi_token(
        self, all_position_preds: list[list[tuple[int, float]]], top_k: int
    ) -> list[tuple[str, float]]:
        """
        Perform beam search to find optimal token combinations.

        Instead of diagonal selection ([0][0], [1][1], [2][2]), this searches
        all possible combinations to find the best scoring paths. For example,
        the best combination might be [0][0], [1][2], [2][0] instead of [0][0], [1][1], [2][2].

        Args:
            all_position_preds: List of predictions for each masked position
                               Each element is [(token_id, score), ...]
            top_k: Number of final results to return

        Returns:
            List of (word, avg_score) tuples, sorted by score
        """
        # Beam width: balance between accuracy and performance
        beam_width = min(top_k * self._BEAM_WIDTH_MULTIPLIER, self._BEAM_WIDTH_CAP)

        # Initialize beam with first position's predictions
        # Each beam entry: (token_ids_list, cumulative_score)
        beam: list[tuple[list[int], float]] = []
        for token_id, score in all_position_preds[0][:beam_width]:
            beam.append(([token_id], score))

        # Extend beam through remaining positions
        for pos_idx in range(1, len(all_position_preds)):
            pos_preds = all_position_preds[pos_idx]
            new_beam = []

            # For each current path in beam
            for path_ids, path_score in beam:
                # Try extending with each candidate from current position
                for token_id, token_score in pos_preds[:beam_width]:
                    new_ids = path_ids + [token_id]
                    new_score = path_score + token_score
                    new_beam.append((new_ids, new_score))

            # Prune: keep only top beam_width paths
            new_beam.sort(key=lambda x: x[1], reverse=True)
            beam = new_beam[:beam_width]

        # Decode top paths into words
        results = []
        seen_words = set()

        # Generate more candidates than needed to account for duplicates/invalids
        for path_ids, path_score in beam[: top_k * self._BEAM_DEDUP_MULTIPLIER]:
            word = self.tokenizer.decode(path_ids).strip()
            # Clean up BPE artifacts
            word = word.replace("Ġ", "").replace("▁", "").strip()

            if word and word not in seen_words:
                # Average score across positions
                num_positions = len(all_position_preds)
                avg_score = path_score / num_positions if num_positions > 0 else 0.0
                results.append((word, avg_score))
                seen_words.add(word)

                if len(results) >= top_k:
                    break

        return results

    def _decode_multi_token_prediction(
        self, logits: np.ndarray, mask_indices: list[int], top_k: int
    ) -> list[tuple[str, float]]:
        """
        Decode predictions from multiple masked token positions.

        For Myanmar words that span multiple BPE tokens, we need to combine
        the predictions from each masked position.

        Args:
            logits: Model output logits [batch, seq_len, vocab]
            mask_indices: List of masked token positions
            top_k: Number of predictions to return

        Returns:
            List of (word, score) tuples
        """
        if len(mask_indices) == 1:
            # Single token: standard decoding
            mask_logits = logits[0, mask_indices[0], :]
            # argpartition is O(n) vs argsort O(n log n) — only need top-K
            if top_k < len(mask_logits):
                part_indices = np.argpartition(mask_logits, -top_k)[-top_k:]
                top_indices = part_indices[np.argsort(mask_logits[part_indices])][::-1]
            else:
                top_indices = np.argsort(mask_logits)[::-1]

            results = []
            for idx in top_indices:
                token = self.tokenizer.decode([idx]).strip()
                # Clean up BPE artifacts (RoBERTa Ġ prefix, SentencePiece ▁ prefix)
                token = token.replace("Ġ", "").replace("▁", "").strip()
                score = float(mask_logits[idx])
                if token:
                    results.append((token, score))
            return results

        # Multi-token word: get top predictions for each position
        # Then combine them into complete words
        all_position_preds: list[list[tuple[int, float]]] = []

        for mask_idx in mask_indices:
            # Bounds check: ensure mask_idx is within logits sequence length
            if mask_idx < 0 or mask_idx >= logits.shape[1]:
                # Position misalignment — return empty rather than garbled results
                return []
            mask_logits = logits[0, mask_idx, :]
            # Get more candidates for multi-token combinations
            n_candidates = top_k * self._MULTI_TOKEN_CANDIDATE_MULTIPLIER
            if n_candidates < len(mask_logits):
                part_idx = np.argpartition(mask_logits, -n_candidates)[-n_candidates:]
                top_indices = part_idx[np.argsort(mask_logits[part_idx])][::-1]
            else:
                top_indices = np.argsort(mask_logits)[::-1]
            position_preds = [(int(idx), float(mask_logits[idx])) for idx in top_indices]
            all_position_preds.append(position_preds)

        # Guard against empty all_position_preds
        if not all_position_preds:
            return []

        # Use beam search for multi-token combinations
        # This finds optimal token combinations instead of diagonal selection
        results = self._beam_search_multi_token(all_position_preds, top_k)

        # Also add single-token predictions that might form valid words
        # (in case the word could be represented by a single token)
        if len(mask_indices) > 1:
            seen_words = {w for w, _ in results}
            first_logits = logits[0, mask_indices[0], :]
            extra_indices = np.argsort(first_logits)[-top_k:][::-1]
            for idx in extra_indices:
                token = self.tokenizer.decode([idx]).strip()
                token = token.replace("Ġ", "").replace("▁", "").strip()
                if token and self._is_myanmar_text(token) and token not in seen_words:
                    score = float(first_logits[idx])
                    results.append((token, score))
                    seen_words.add(token)

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def predict_mask(
        self, sentence: str, target_word: str, top_k: int | None = None, occurrence: int = 0
    ) -> list[tuple[str, float]]:
        """
        Predict the most likely words for a masked position.

        For Myanmar text, uses word-aligned masking to handle BPE tokenization
        that may split words into multiple tokens.

        Shares cached ONNX logits with score_mask_candidates() via
        _get_mask_logits(), so back-to-back calls for the same
        (sentence, target_word, occurrence) cost only one forward pass.

        Args:
            sentence: Full sentence containing the target word.
            target_word: The word to mask and predict.
            top_k: Number of predictions to return (uses default if None).
            occurrence: Which occurrence to mask (0-indexed, default 0).

        Returns:
            List of (word, score) tuples.
        """
        limit = top_k if top_k is not None else self.predict_top_k

        logits, mask_indices = self._get_mask_logits(sentence, target_word, occurrence)

        if logits is None or not mask_indices:
            return []
        if logits.ndim != 3 or logits.shape[0] == 0 or logits.shape[2] == 0:
            return []
        if any(m < 0 or m >= logits.shape[1] for m in mask_indices):
            return []

        return self._decode_multi_token_prediction(logits, mask_indices, limit)

    def batch_get_mask_logits(
        self,
        sentence: str,
        targets: list[tuple[str, int]],
    ) -> None:
        """Pre-warm the logits cache for multiple words in the same sentence.

        Prepares masked inputs for all targets, runs a single batched ONNX
        forward pass, and stores results in ``_logits_cache``.  Subsequent
        calls to ``predict_mask`` / ``score_mask_candidates`` for these
        targets will be cache hits (no ONNX call).

        Args:
            sentence: The sentence containing all target words.
            targets: List of ``(word, occurrence)`` pairs to pre-compute.
        """
        if not targets:
            return

        # Separate targets into those already cached vs needing inference.
        uncached: list[tuple[int, str, int]] = []  # (orig_idx, word, occurrence)
        for idx, (word, occ) in enumerate(targets):
            cache_key = (sentence, word, occ)
            if cache_key not in self._logits_cache:
                uncached.append((idx, word, occ))

        if not uncached:
            return  # Everything already cached.

        # --- Prepare masked inputs for aligned path (Myanmar text) ---
        # All words share the same base tokenization, so sequence lengths
        # are identical and we can stack without padding.
        batch_ids_list: list[list[int]] = []
        batch_mask_indices: list[list[int]] = []
        batch_orig_indices: list[int] = []
        batch_words: list[str] = []
        batch_occs: list[int] = []
        fallback_items: list[tuple[int, str, int]] = []

        for orig_idx, word, occ in uncached:
            if self.use_word_alignment and self._is_myanmar_text(word):
                token_ids, start_idx, end_idx = self._get_word_token_alignment(sentence, word, occ)
                if start_idx != -1 and end_idx != -1:
                    masked_ids = list(token_ids)
                    mask_indices = []
                    for i in range(start_idx, end_idx):
                        masked_ids[i] = self.mask_token_id
                        mask_indices.append(i)
                    batch_ids_list.append(masked_ids)
                    batch_mask_indices.append(mask_indices)
                    batch_orig_indices.append(orig_idx)
                    batch_words.append(word)
                    batch_occs.append(occ)
                    continue

            # Alignment failed or non-Myanmar — fall back to individual call.
            fallback_items.append((orig_idx, word, occ))

        # --- Batched ONNX forward pass ---
        if batch_ids_list:
            input_ids = np.array(batch_ids_list, dtype=np.int64)
            attention_mask = np.ones_like(input_ids, dtype=np.int64)

            inputs = {self.input_name: input_ids}
            if self._has_attention_mask_input:
                inputs["attention_mask"] = attention_mask

            all_logits = self.session.run([self.output_name], inputs)[0]
            # all_logits shape: (batch_size, seq_len, vocab_size)

            # Distribute results into cache — store per-item slice with batch dim.
            for batch_i in range(len(batch_ids_list)):
                word = batch_words[batch_i]
                occ = batch_occs[batch_i]
                # Keep batch dimension for compatibility with predict_mask decoder.
                logits_slice = all_logits[batch_i : batch_i + 1]
                mask_idx = batch_mask_indices[batch_i]
                self._logits_cache.set((sentence, word, occ), (logits_slice, mask_idx))

        # --- Handle fallbacks individually ---
        for _orig_idx, word, occ in fallback_items:
            self._get_mask_logits(sentence, word, occ)

    def _get_mask_logits(
        self, sentence: str, target_word: str, occurrence: int = 0
    ) -> tuple[np.ndarray | None, list[int]]:
        """Return raw logits and mask positions for a masked target occurrence.

        Results are cached by (sentence, target_word, occurrence) so that
        back-to-back calls from predict_mask() and score_mask_candidates()
        for the same masked position share a single ONNX forward pass.
        """
        cache_key = (sentence, target_word, occurrence)
        cached = self._logits_cache.get(cache_key)
        if cached is not None:
            return cached

        result = self._run_mask_inference(sentence, target_word, occurrence)
        self._logits_cache.set(cache_key, result)
        return result

    def _run_mask_inference(
        self, sentence: str, target_word: str, occurrence: int = 0
    ) -> tuple[np.ndarray | None, list[int]]:
        """Execute ONNX forward pass for a masked target (uncached).

        Uses pre-allocated buffers via ``_prepare_single_inference_inputs``
        to avoid per-call numpy array allocation.  Gracefully returns
        ``(None, [])`` when the ONNX runtime encounters out-of-range token
        IDs (e.g. on very long inputs that exceed the model vocabulary).
        """
        try:
            return self._run_mask_inference_inner(sentence, target_word, occurrence)
        except Exception:
            return None, []

    def _run_mask_inference_inner(
        self, sentence: str, target_word: str, occurrence: int = 0
    ) -> tuple[np.ndarray | None, list[int]]:
        """Inner implementation — may raise on ONNX runtime errors."""
        use_alignment = self.use_word_alignment and self._is_myanmar_text(target_word)
        if use_alignment:
            input_ids, attention_mask, mask_indices = self._create_word_aligned_mask(
                sentence, target_word, occurrence
            )
            if input_ids is not None and mask_indices:
                inputs = {self.input_name: input_ids}
                if self._has_attention_mask_input and attention_mask is not None:
                    inputs["attention_mask"] = attention_mask
                logits = self.session.run([self.output_name], inputs)[0]
                return logits, mask_indices

        masked_sentence = self._mask_target_occurrence(
            sentence, target_word, occurrence, self.mask_token
        )
        if masked_sentence == sentence:
            masked_sentence = sentence.replace(target_word, self.mask_token, 1)

        encoded = self._cached_encode(masked_sentence)

        # EncodingResult has __slots__ = ("ids", "offsets") -- no attention_mask.
        # Reuse pre-allocated buffers for efficiency.
        input_ids, attention_mask = self._prepare_single_inference_inputs(encoded.ids)

        mask_indices_arr = np.where(input_ids == self.mask_token_id)
        if len(mask_indices_arr[1]) == 0:
            return None, []
        mask_idx = int(mask_indices_arr[1][0])

        inputs = {self.input_name: input_ids}
        if self._has_attention_mask_input:
            inputs["attention_mask"] = attention_mask
        logits = self.session.run([self.output_name], inputs)[0]
        return logits, [mask_idx]

    def _encode_candidate_token_ids(self, candidate: str) -> list[int]:
        """Encode candidate text and remove obvious special tokens."""
        if not candidate:
            return []

        try:
            encoded = self.tokenizer.encode(candidate)
        except (RuntimeError, ValueError, TypeError, AttributeError):
            return []

        ids = list(getattr(encoded, "ids", []) or [])
        if not ids:
            return []

        special_tokens = ("<s>", "</s>", "[CLS]", "[SEP]", "<pad>", "[PAD]")
        special_ids = {
            sid
            for sid in (self.tokenizer.token_to_id(tok) for tok in special_tokens)
            if isinstance(sid, int) and sid >= 0
        }
        filtered = [tid for tid in ids if tid not in special_ids]
        return filtered or ids

    def score_mask_candidates(
        self,
        sentence: str,
        target_word: str,
        candidates: list[str],
        occurrence: int = 0,
    ) -> dict[str, float]:
        """
        Score explicit candidate words for a masked target position.

        Unlike predict_mask(top_k), this method can score candidates even when
        they are not among the model's top-k decoded outputs.
        """
        if not candidates:
            return {}

        logits, mask_indices = self._get_mask_logits(sentence, target_word, occurrence=occurrence)
        if logits is None or not mask_indices or logits.ndim != 3:
            return {}

        scores: dict[str, float] = {}
        seq_len = logits.shape[1]
        if any(m < 0 or m >= seq_len for m in mask_indices):
            return {}

        unique_candidates = list(dict.fromkeys(candidates))
        num_masks = len(mask_indices)
        for candidate in unique_candidates:
            token_ids = self._encode_candidate_token_ids(candidate)
            if not token_ids:
                continue

            if num_masks == 1:
                mapped = [token_ids[0]]
            else:
                if len(token_ids) < num_masks:
                    continue
                mapped = token_ids[:num_masks]

            total = 0.0
            valid = True
            for mask_pos, token_id in zip(mask_indices, mapped, strict=False):
                if token_id < 0 or token_id >= logits.shape[2]:
                    valid = False
                    break
                total += float(logits[0, mask_pos, token_id])
            if not valid:
                continue

            scores[candidate] = total / num_masks

        return scores

    @classmethod
    def _prefix_skip_margin(cls, word: str) -> float:
        """Return margin used for prefix-based skip decisions."""
        if len(word) <= 1:
            return cls._PREFIX_SKIP_MARGIN_SHORT
        if len(word) <= cls._SHORT_WORD_MAX_LEN:
            return cls._PREFIX_SKIP_MARGIN_MEDIUM
        return cls._PREFIX_SKIP_MARGIN_LONG

    def _should_skip_due_to_prefix_evidence(
        self,
        word: str,
        predictions: list[tuple[str, float]],
        top_n: int = 5,
    ) -> bool:
        """Determine whether prefix evidence is strong enough to skip flagging."""
        if not predictions:
            return False

        top_preds = predictions[:top_n]
        top_score = float(top_preds[0][1])
        margin = self._prefix_skip_margin(word)
        is_short_word = len(word) <= self._SHORT_WORD_MAX_LEN

        # Case 1: predicted compounds that start with the target word.
        prefix_scores = [
            float(score)
            for pred_word, score in top_preds
            if pred_word.startswith(word) and pred_word != word
        ]
        if prefix_scores:
            best_prefix_score = max(prefix_scores)
            if is_short_word:
                # For short words, only skip when prefix evidence is almost tied
                # with top prediction and no strong non-prefix competitor exists.
                competing_non_prefix = max(
                    (
                        float(score)
                        for pred_word, score in top_preds
                        if not pred_word.startswith(word) and not word.startswith(pred_word)
                    ),
                    default=float("-inf"),
                )
                if (
                    top_score - best_prefix_score <= margin
                    and competing_non_prefix
                    < best_prefix_score - self._PREFIX_COMPETING_NON_PREFIX_GAP
                ):
                    return True
            elif top_score - best_prefix_score <= margin:
                return True

        # For short ambiguous words, avoid the "word extends predicted base"
        # shortcut — it suppresses real one-char/short confusions.
        if is_short_word:
            return False

        # Case 2: target word extends a predicted base token (BPE-like extension).
        morpheme_boundary_chars = frozenset({"\u1038", "\u1037", "\u103a"})  # း ့ ်
        for pred_word, score in top_preds:
            if not pred_word or not word.startswith(pred_word):
                continue
            suffix = word[len(pred_word) :]
            if suffix and suffix[0] in morpheme_boundary_chars:
                continue
            if top_score - float(score) <= margin:
                return True

        return False

    def is_semantic_error(
        self, sentence: str, word: str, neighbors: list[str], occurrence: int = 0
    ) -> str | None:
        """
        Check if a word is a semantic error using AI.

        Args:
            sentence: Full context.
            word: Suspicious word.
            neighbors: List of phonetic/edit-distance neighbors to check against.

        Returns:
            The suggestion (replacement word) if AI is confident it's an error.
            None if the word seems correct or AI is unsure.
        """
        # Run prediction
        predictions = self.predict_mask(
            sentence, word, top_k=self.check_top_k, occurrence=occurrence
        )
        pred_words = {p[0] for p in predictions}

        # Logic:
        # 1. If 'word' is in top predictions, it's likely Correct. Return None.
        #    Also check prefix match — BPE beam search often predicts compound
        #    words (e.g. ကျွန်တော်) that start with the target (ကျွန်).
        if word in pred_words:
            return None
        if self._should_skip_due_to_prefix_evidence(
            word, predictions, top_n=min(self._ERROR_CHECK_TOP_N, len(predictions))
        ):
            return None

        # 2. If 'word' is NOT in predictions, but a 'neighbor' IS...
        #    First try exact match (strongest signal).
        for neighbor in neighbors:
            if neighbor in pred_words:
                return neighbor

        #    Then try prefix match — the model may predict a compound that
        #    starts with the neighbor (e.g. predicts ကျွန်တော် for neighbor ကျွန်).
        for neighbor in neighbors:
            if any(pw.startswith(neighbor) for pw in pred_words):
                return neighbor

        return None

    def score_candidates(
        self,
        sentence: str,
        word: str,
        candidates: list[str],
        top_k: int | None = None,
        occurrence: int = 0,
    ) -> list[tuple[str, float]]:
        """Score each candidate word by its MLM logit at the masked position.

        Single forward pass: masks ``word`` in ``sentence``, runs inference,
        then returns the raw logit for each candidate from the output
        distribution.  Candidates not found in the top-K predictions receive
        a floor score (lowest logit in predictions minus 1.0) so that the
        ranked order is still meaningful.

        Use this instead of ``is_semantic_error`` when you need the full ranked
        list rather than a single best correction.

        Args:
            sentence: Full sentence containing the word.
            word: The word to mask (typically the error word).
            candidates: Candidate corrections to score.
            top_k: How many top predictions to retrieve per forward pass.
                Higher values improve candidate coverage at negligible cost.
            occurrence: Which occurrence of ``word`` to mask (0-indexed).

        Returns:
            List of (candidate, logit) sorted best-first (highest logit first).
            Empty list if the model is unavailable or inference fails.
        """
        if not candidates:
            return []

        effective_top_k = top_k if top_k is not None else self._SCORE_DEFAULT_TOP_K
        predictions = self.predict_mask(
            sentence, word, top_k=effective_top_k, occurrence=occurrence
        )
        if not predictions:
            return [(c, 0.0) for c in candidates]

        # Build lookup: predicted word → logit
        score_lookup: dict[str, float] = {pred_word: logit for pred_word, logit in predictions}

        # Floor for candidates absent from the prediction window
        floor_logit = min(logit for _, logit in predictions) - self._SCORE_FLOOR_OFFSET

        # Collect candidates not found in decoded predictions for direct
        # logit scoring via score_mask_candidates.
        missing_candidates = [c for c in candidates if c not in score_lookup]
        direct_scores: dict[str, float] = {}
        if missing_candidates:
            direct_scores = self.score_mask_candidates(
                sentence, word, missing_candidates, occurrence=occurrence
            )

        scored: list[tuple[str, float]] = []
        for candidate in candidates:
            if candidate in score_lookup:
                logit = score_lookup[candidate]
            elif candidate in direct_scores:
                logit = direct_scores[candidate]
            else:
                # Prefix match: model may predict a compound starting with candidate
                matched_logit = next(
                    (lg for pw, lg in predictions if pw.startswith(candidate)),
                    floor_logit,
                )
                logit = matched_logit
            scored.append((candidate, logit))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def scan_sentence(
        self,
        sentence: str,
        words: list[str],
        min_word_len: int = 2,
        confidence_threshold: float = 0.3,
    ) -> list[tuple[int, str, list[str], float]]:
        """
        Proactively scan a sentence for semantic errors using AI.

        This method checks each word by masking it and seeing if the model
        predicts it among the top candidates. If not, it's a potential error.

        Args:
            sentence: The full sentence text.
            words: List of segmented words in the sentence.
            min_word_len: Minimum word length to check (skip short particles).
            confidence_threshold: Minimum confidence to report an error.

        Returns:
            List of tuples: (word_index, error_word, suggestions, confidence)
            where confidence is based on how far down the prediction list
            the original word appears (or 1.0 if not found at all).
        """
        if self.session is None or self.tokenizer is None:
            return []

        errors = []
        # Track word occurrences in O(n) instead of O(n^2)
        word_occurrence_counts: dict[str, int] = {}

        for idx, word in enumerate(words):
            # Skip very short words (particles, etc.)
            if len(word) < min_word_len:
                word_occurrence_counts[word] = word_occurrence_counts.get(word, 0) + 1
                continue

            # Skip punctuation and non-Myanmar text
            # Use dynamic character set based on allow_extended_myanmar config
            valid_chars = get_myanmar_char_set(self.allow_extended_myanmar)
            if not any(c in valid_chars for c in word):
                word_occurrence_counts[word] = word_occurrence_counts.get(word, 0) + 1
                continue

            try:
                # Track occurrence for duplicate words
                occurrence = word_occurrence_counts.get(word, 0)

                # Get model predictions for this word position
                predictions = self.predict_mask(
                    sentence, word, top_k=self.check_top_k, occurrence=occurrence
                )

                if not predictions:
                    continue

                pred_words = [p[0] for p in predictions]

                # Check top-N only for "word present" (tighter gate).
                # Full top-K is still used for suggestion generation below.
                top_n_words = pred_words[: self._SCAN_PRESENCE_TOP_N]

                # Check if original word is in top-N predictions (exact or prefix match)
                # Prefix match is critical for BPE models where the model may
                # predict compound words that start with the target word
                # (e.g., predicts ကျွန်တော် when target is ကျွန်)
                if word in top_n_words:
                    # Word is predicted by model - likely correct
                    continue
                if self._should_skip_due_to_prefix_evidence(
                    word, predictions, top_n=self._SCAN_PRESENCE_TOP_N
                ):
                    continue

                # Filter suggestions to only VALID Myanmar words
                # A valid suggestion must:
                # 1. Contain mostly Myanmar characters (>50%)
                # 2. Be at least _SCAN_MIN_SUGGESTION_LEN chars long
                # 3. Not be mostly punctuation/symbols
                suggestions = []
                suggestion_valid_chars = get_myanmar_char_set(self.allow_extended_myanmar)
                for p_word, _p_score in predictions[: self._SCAN_SUGGESTION_POOL_SIZE]:
                    if len(p_word) < self._SCAN_MIN_SUGGESTION_LEN:
                        continue
                    # Use dynamic character set based on allow_extended_myanmar config
                    myanmar_chars = sum(1 for c in p_word if c in suggestion_valid_chars)
                    total_chars = len(p_word.replace(" ", ""))
                    if (
                        total_chars > 0
                        and myanmar_chars / total_chars >= self._SCAN_MIN_MYANMAR_CHAR_RATIO
                    ):
                        suggestions.append(p_word)
                    if len(suggestions) >= self._SCAN_MAX_SUGGESTIONS:
                        break

                # Only flag error if we have at least one valid Myanmar suggestion
                # This filters out cases where the model doesn't understand Myanmar
                if not suggestions:
                    continue

                # Calculate confidence based on whether suggestions look reasonable
                # Higher confidence if suggestions share syllables with original word
                top_score = predictions[0][1] if predictions else 0
                # Use model-aware logit scaling for confidence calibration
                logit_scale = self._get_model_logit_scale()
                base_confidence = self._calibrate_confidence(top_score, logit_scale)

                # Boost confidence if suggestions are similar to original (phonetically related)
                best_similarity = 0.0
                for sugg in suggestions[: self._SCAN_SIMILARITY_TOP_N]:
                    # Simple character overlap check
                    common = sum(1 for c in sugg if c in word)
                    similarity = common / max(len(word), len(sugg)) if word and sugg else 0.0
                    best_similarity = max(best_similarity, similarity)

                # Confidence is boosted by similarity (suggests phonetic confusion)
                confidence = base_confidence * (
                    self._SCAN_CONFIDENCE_BASE_WEIGHT
                    + self._SCAN_CONFIDENCE_SIMILARITY_WEIGHT * best_similarity
                )

                if confidence >= confidence_threshold:
                    errors.append((idx, word, suggestions, confidence))

            except (RuntimeError, ValueError, IndexError, KeyError, TypeError) as e:
                # Skip words that cause issues
                self.logger.debug(f"Semantic check failed for word '{word}': {e}")
            finally:
                word_occurrence_counts[word] = word_occurrence_counts.get(word, 0) + 1

        return errors
