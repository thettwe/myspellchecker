"""
Transformer-based word segmenter using HuggingFace models.

This module provides a neural word segmenter using a pre-trained transformer model
from HuggingFace. It uses token classification with B/I labels to identify word
boundaries in Myanmar text.

Features:
- High accuracy (96.17%) on Myanmar text segmentation
- Batch processing for GPU efficiency
- Support for custom fine-tuned models
- Device selection (CPU / CUDA / MPS)
- Lazy initialization for deferred model loading

Requirements:
    - transformers>=4.30.0
    - torch>=2.0.0

Install with: pip install myspellchecker[transformers]

Default Model Attribution:
    The default model used is "chuuhtetnaing/myanmar-text-segmentation-model"
    from Hugging Face.

    Model: https://huggingface.co/chuuhtetnaing/myanmar-text-segmentation-model
    Author: Chuu Htet Naing
    Base: XLM-RoBERTa fine-tuned for token classification (word boundaries)
    Accuracy: 96.17% | F1: 78.66% (best checkpoint at step 50000)
    Labels: B (beginning of word), I (inside/continuation of word)

    We gratefully acknowledge Chuu Htet Naing for making this model
    publicly available, enabling high-accuracy Myanmar word segmentation.

Example:
    >>> from myspellchecker.tokenizers.transformer_word_segmenter import (
    ...     TransformerWordSegmenter,
    ... )
    >>>
    >>> # Use default Myanmar segmentation model
    >>> segmenter = TransformerWordSegmenter()
    >>> segmenter.segment("မြန်မာနိုင်ငံသည်")
    ['မြန်မာ', 'နိုင်ငံ', 'သည်']
    >>>
    >>> # Use custom model on GPU
    >>> segmenter = TransformerWordSegmenter(
    ...     model_name="path/to/my/model",
    ...     device=0  # Use GPU
    ... )
"""

from __future__ import annotations

from myspellchecker.core.config.algorithm_configs import (
    TransformerSegmenterConfig,
)
from myspellchecker.core.exceptions import TokenizationError
from myspellchecker.utils.logging_utils import get_logger

# Try to import transformers (optional dependency)
try:
    from transformers import pipeline as hf_pipeline

    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False

# Module logger
logger = get_logger(__name__)

# Default config — single source of truth for defaults.
_DEFAULT_SEGMENTER_CONFIG = TransformerSegmenterConfig()


class TransformerWordSegmenter:
    """
    Transformer-based word segmenter using HuggingFace token classification.

    Uses a pre-trained XLM-RoBERTa model fine-tuned for Myanmar word boundary
    detection via B/I (Beginning/Inside) labeling.

    Model Attribution:
        Model: https://huggingface.co/chuuhtetnaing/myanmar-text-segmentation-model
        Author: Chuu Htet Naing
        License: See model page for license details

    Attributes:
        model_name: HuggingFace model ID or local path
        device: Device for inference (-1=CPU, 0+=GPU index)
        batch_size: Batch size for batch segmentation
        max_length: Maximum sequence length for the model

    Performance:
        - CPU: ~1K sentences/second (short sentences)
        - GPU: ~10K sentences/second
        - Memory: ~1.1GB for model + ~100MB buffer

    Note:
        This segmenter is NOT fork-safe when using GPU (CUDA context issues).
        Use in main process or with multiprocessing.spawn instead of fork.

    Example:
        >>> segmenter = TransformerWordSegmenter()
        >>> segmenter.segment("မြန်မာနိုင်ငံသည်")
        ['မြန်မာ', 'နိုင်ငံ', 'သည်']
        >>>
        >>> # Batch processing
        >>> segmenter.segment_batch(["မြန်မာနိုင်ငံ", "ကျွန်တော်သွားပါမယ်"])
        [['မြန်မာ', 'နိုင်ငံ'], ['ကျွန်တော်', 'သွား', 'ပါ', 'မယ်']]
    """

    DEFAULT_MODEL = _DEFAULT_SEGMENTER_CONFIG.model_name

    def __init__(
        self,
        model_name: str | None = None,
        device: int = _DEFAULT_SEGMENTER_CONFIG.device,
        batch_size: int = _DEFAULT_SEGMENTER_CONFIG.batch_size,
        max_length: int = _DEFAULT_SEGMENTER_CONFIG.max_length,
        cache_dir: str | None = None,
        **pipeline_kwargs,
    ) -> None:
        """
        Initialize transformer-based word segmenter.

        Args:
            model_name: HuggingFace model ID or local path.
                       Default: "chuuhtetnaing/myanmar-text-segmentation-model"
            device: Device for inference. -1 for CPU, 0+ for GPU index.
                   Default: -1 (CPU)
            batch_size: Batch size for batch segmentation (default: 32)
            max_length: Maximum sequence length (default: 512)
            cache_dir: Directory for caching downloaded models (optional)
            **pipeline_kwargs: Additional arguments passed to transformers.pipeline

        Raises:
            ImportError: If transformers package is not installed
            ValueError: If model cannot be loaded
        """
        if not _HAS_TRANSFORMERS:
            raise ImportError(
                "Transformer-based word segmentation requires the 'transformers' library.\n"
                "Install with: pip install myspellchecker[transformers]\n\n"
                "Alternatively, use the default myword segmentation engine:\n"
                "  DefaultSegmenter(word_engine='myword')"
            )

        self.model_name = model_name or self.DEFAULT_MODEL
        self.max_length = max_length

        # Auto-tune batch size based on device if not explicitly set
        # CPU benefits from larger batches (less Python overhead per batch)
        # GPU/MPS can handle large batches efficiently
        if batch_size == 32 and device == -1:
            batch_size = 64  # Better CPU throughput
        self.batch_size = batch_size

        # Validate and potentially adjust device
        if device >= 0:
            try:
                import torch

                if torch.cuda.is_available():
                    if device >= torch.cuda.device_count():
                        available = torch.cuda.device_count()
                        logger.warning(
                            f"GPU {device} not available (have {available} GPUs), "
                            f"falling back to CPU (device=-1)"
                        )
                        device = -1
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    # MPS (Apple Silicon) - use device="mps" via pipeline
                    logger.info("Using MPS (Apple Silicon) for inference")
                    # HuggingFace pipeline accepts device="mps" or device=0 on MPS
                    # We keep device=0 which the pipeline will map to MPS
                else:
                    logger.warning("No GPU available, falling back to CPU (device=-1)")
                    device = -1
            except ImportError:
                logger.warning("PyTorch not available, using CPU (device=-1)")
                device = -1

        self.device = device

        # Initialize HuggingFace pipeline
        # Note: TokenClassificationPipeline does not accept truncation/max_length
        # as constructor or call kwargs. We configure the tokenizer directly after
        # pipeline creation to enforce truncation at max_length.
        try:
            self._pipeline = hf_pipeline(
                "token-classification",
                model=self.model_name,
                aggregation_strategy="simple",
                device=device,
                model_kwargs={"cache_dir": cache_dir} if cache_dir else {},
                **pipeline_kwargs,
            )
            # Configure tokenizer truncation directly
            if self._pipeline.tokenizer is not None:
                self._pipeline.tokenizer.model_max_length = max_length
        except (OSError, RuntimeError, ImportError) as e:
            raise TokenizationError(
                f"Failed to load model '{self.model_name}': {e}\n\n"
                f"Possible solutions:\n"
                f"1. Check model name/path is correct\n"
                f"2. Ensure model is compatible with token-classification task\n"
                f"3. Try downloading model manually first\n"
                f"4. Check internet connection if loading from HuggingFace Hub"
            ) from e

    def segment(self, text: str) -> list[str]:
        """
        Segment a single text into words using B/I tag merging.

        The model labels each token as B (beginning of a new word) or
        I (inside/continuation of the current word). Adjacent B+I tokens
        are merged to form complete words.

        Args:
            text: Myanmar text to segment.

        Returns:
            List of segmented words.

        Example:
            >>> segmenter = TransformerWordSegmenter()
            >>> segmenter.segment("မြန်မာနိုင်ငံသည်")
            ['မြန်မာ', 'နိုင်ငံ', 'သည်']
        """
        if not text or not text.strip():
            return []

        try:
            results = self._pipeline(text)
            return self._merge_bi_tags(results)
        except RuntimeError as e:
            error_str = str(e).lower()
            if "out of memory" in error_str or "cuda" in error_str:
                logger.warning(
                    f"GPU memory exhausted during word segmentation: {e}. "
                    f"Consider using device=-1 (CPU) or reducing max_length."
                )
            else:
                logger.error(f"Transformer word segmentation failed: {e}")
            raise
        except (KeyError, IndexError) as e:
            logger.error(f"Transformer word segmentation failed for text '{text[:50]}...': {e}")
            raise

    def segment_batch(self, texts: list[str]) -> list[list[str]]:
        """
        Segment multiple texts into words using batch processing.

        More efficient than calling segment() in a loop, especially
        when using GPU, as it batches inference calls.

        Args:
            texts: List of Myanmar texts to segment.

        Returns:
            List of word lists, one per input text.

        Example:
            >>> segmenter = TransformerWordSegmenter()
            >>> segmenter.segment_batch(["မြန်မာနိုင်ငံ", "စာရေးသည်"])
            [['မြန်မာ', 'နိုင်ငံ'], ['စာ', 'ရေး', 'သည်']]
        """
        if not texts:
            return []

        # Filter empty texts and track their indices
        non_empty_indices = []
        non_empty_texts = []
        for i, text in enumerate(texts):
            if text and text.strip():
                non_empty_indices.append(i)
                non_empty_texts.append(text)

        if not non_empty_texts:
            return [[] for _ in texts]

        try:
            # Use pipeline batch processing
            all_results = self._pipeline(non_empty_texts, batch_size=self.batch_size)

            # Merge B/I tags for each text
            non_empty_words = []
            for results in all_results:
                words = self._merge_bi_tags(results)
                non_empty_words.append(words)

            # Reconstruct full results list with empty lists for empty texts
            all_words: list[list[str]] = [[] for _ in texts]
            for idx, words in zip(non_empty_indices, non_empty_words, strict=False):
                all_words[idx] = words

            return all_words

        except (RuntimeError, KeyError, IndexError) as e:
            # Fallback: process each text individually
            logger.debug(f"Transformer batch segmentation failed: {e}")
            result = []
            for text in texts:
                try:
                    result.append(self.segment(text))
                except (RuntimeError, ValueError, KeyError, IndexError):
                    result.append([])
            return result

    def _merge_bi_tags(self, results: list[dict]) -> list[str]:
        """
        Merge B/I tagged tokens into complete words.

        The model outputs tokens labeled as:
        - B: Beginning of a new word
        - I: Inside/continuation of the current word

        This method groups consecutive B+I* sequences into words.

        Args:
            results: Pipeline output with 'entity_group' and 'word' fields.
                Each result dict has:
                - entity_group: "B" or "I"
                - word: The token text
                - start: Character start offset
                - end: Character end offset
                - score: Confidence score

        Returns:
            List of merged words.
        """
        if not results:
            return []

        words = []
        current_word = ""

        for result in results:
            tag = result.get("entity_group", "").upper()
            token = result.get("word", "").strip()

            if not token:
                continue

            if tag == "B":
                # Start of a new word - flush current word
                if current_word:
                    words.append(current_word)
                current_word = token
            elif tag == "I":
                # Continuation of current word
                if current_word:
                    current_word += token
                else:
                    # I without preceding B - treat as new word start
                    current_word = token
            else:
                # Unknown tag - treat as B (new word start)
                if current_word:
                    words.append(current_word)
                current_word = token

        # Flush remaining word
        if current_word:
            words.append(current_word)

        return words

    @property
    def is_fork_safe(self) -> bool:
        """
        Check if the segmenter is fork-safe for multiprocessing.

        Returns False for GPU mode because CUDA contexts cannot be safely forked.
        Returns True for CPU mode (device=-1) which is fork-safe.
        Use multiprocessing.spawn instead of fork if using GPU.
        """
        return self.device < 0
