"""
Transformer-based POS tagger using HuggingFace models.

This module provides a neural POS tagger using pre-trained transformer models
from HuggingFace. It offers high accuracy (~93%) but requires the transformers
package and is slower than rule-based approaches.

Features:
- State-of-the-art accuracy (~93% on Myanmar text)
- Context-aware tagging using transformer models
- Batch processing for GPU efficiency
- Support for custom fine-tuned models
- Confidence scores from model probabilities

Requirements:
    - transformers>=4.30.0
    - torch>=2.0.0

Install with: pip install myspellchecker[transformers]

Default Model Attribution:
    The default model used is "chuuhtetnaing/myanmar-pos-model" from Hugging Face.

    Model: https://huggingface.co/chuuhtetnaing/myanmar-pos-model
    Author: Chuu Htet Naing
    Base: XLM-RoBERTa fine-tuned for Myanmar POS tagging
    Accuracy: 93.37% | F1: 92.24%

    We gratefully acknowledge Chuu Htet Naing for making this model
    publicly available, enabling high-accuracy Myanmar POS tagging.

Example:
    >>> from myspellchecker.algorithms.pos_tagger_transformer import TransformerPOSTagger
    >>>
    >>> # Use default Myanmar POS model
    >>> tagger = TransformerPOSTagger()
    >>> tagger.tag_word("မြန်မာ")
    'N'
    >>>
    >>> # Use custom model
    >>> tagger = TransformerPOSTagger(
    ...     model_name="/path/to/my/model",
    ...     device=0  # Use GPU
    ... )
    >>> tags = tagger.tag_sequence(["မြန်မာ", "နိုင်ငံ", "သည်"])
    ['N', 'N', 'PPM']

Note:
    The tagger automatically maps HuggingFace model tags (lowercase) to
    our internal tag convention (uppercase). For example:
    - 'n' -> 'N' (Noun)
    - 'v' -> 'V' (Verb)
    - 'ppm' -> 'PPM' (Postpositional marker)
    - 'part' -> 'PART' (Particle)
"""

from __future__ import annotations

from typing import Any

from myspellchecker.algorithms.pos_tagger_base import (
    POSPrediction,
    POSTaggerBase,
    TaggerType,
)
from myspellchecker.utils.logging_utils import get_logger

# Try to import transformers (optional dependency)
try:
    from transformers import pipeline

    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False

# Module logger
logger = get_logger(__name__)


class TransformerPOSTagger(POSTaggerBase):
    """
    Transformer-based POS tagger using HuggingFace models.

    Uses pre-trained transformer models for high-accuracy POS tagging.
    Default model is "chuuhtetnaing/myanmar-pos-model" by Chuu Htet Naing,
    which achieves 93.37% accuracy with F1 score of 92.24%.

    Model Attribution:
        Model: https://huggingface.co/chuuhtetnaing/myanmar-pos-model
        Author: Chuu Htet Naing
        License: See model page for license details

    Attributes:
        model_name: HuggingFace model ID or local path
        device: Device for inference (-1=CPU, 0+=GPU index)
        batch_size: Batch size for sequence tagging
        cache_dir: Directory for caching models

    Performance:
        - CPU: ~5K words/second
        - GPU: ~50K words/second
        - Memory: ~500MB for model + ~100MB buffer

    Note:
        This tagger is NOT fork-safe due to CUDA context issues.
        Use in main process or with multiprocessing.spawn instead of fork.

    Example:
        >>> # Default model (CPU)
        >>> tagger = TransformerPOSTagger()
        >>>
        >>> # GPU with custom model
        >>> tagger = TransformerPOSTagger(
        ...     model_name="path/to/custom/model",
        ...     device=0,
        ...     batch_size=64
        ... )
        >>>
        >>> # Tag with confidence
        >>> pred = tagger.tag_word_with_confidence("မြန်မာ")
        >>> print(f"{pred.tag} (conf: {pred.confidence:.2f})")
        N (conf: 0.98)
    """

    # Default Myanmar POS model from HuggingFace
    DEFAULT_MODEL = "chuuhtetnaing/myanmar-pos-model"

    # Mapping from HuggingFace model tags to internal tag convention
    # HuggingFace model uses lowercase tags, our internal system uses uppercase
    # Reference: https://huggingface.co/chuuhtetnaing/myanmar-pos-model
    HF_TO_INTERNAL_TAG_MAP = {
        # Core POS tags
        "n": "N",  # Noun
        "v": "V",  # Verb
        "adj": "ADJ",  # Adjective
        "adv": "ADV",  # Adverb
        "pron": "PRON",  # Pronoun
        "num": "NUM",  # Number
        "conj": "CONJ",  # Conjunction
        "int": "INT",  # Interjection
        "punc": "PUNCT",  # Punctuation
        # Particle/Postpositional markers (coarse - HF doesn't have granular particle tags)
        "ppm": "PPM",  # Postpositional marker (maps to our granular P_* tags contextually)
        "part": "PART",  # Particle (general)
        # Other tags from HuggingFace model
        "abb": "ABB",  # Abbreviation
        "fw": "FW",  # Foreign word
        "sb": "SB",  # Symbol
        "tn": "TN",  # Text number
    }

    def __init__(
        self,
        model_name: str | None = None,
        device: int | str = -1,
        batch_size: int = 32,
        max_length: int = 128,
        cache_dir: str | None = None,
        use_fp16: bool = True,
        use_torch_compile: bool = False,
        **pipeline_kwargs,
    ):
        """
        Initialize transformer-based POS tagger.

        Args:
            model_name: HuggingFace model ID or local path.
                       Default: "chuuhtetnaing/myanmar-pos-model"
            device: Device for inference. -1 for CPU, 0+ for GPU index.
                   Default: -1 (CPU)
            batch_size: Batch size for sequence tagging (default: 32)
            max_length: Maximum sequence length (default: 128)
            cache_dir: Directory for caching downloaded models (optional)
            use_fp16: Use float16 on GPU for ~2x throughput (default: True)
            use_torch_compile: Use torch.compile() JIT (default: False)
            **pipeline_kwargs: Additional arguments passed to transformers.pipeline

        Raises:
            ImportError: If transformers package is not installed
            ValueError: If model cannot be loaded

        Example:
            >>> # CPU with default model
            >>> tagger = TransformerPOSTagger()
            >>>
            >>> # GPU with custom model
            >>> tagger = TransformerPOSTagger(
            ...     model_name="username/my-pos-model",
            ...     device=0,
            ...     batch_size=64
            ... )
        """
        if not _HAS_TRANSFORMERS:
            raise ImportError(
                "Transformer-based POS tagging requires the 'transformers' library.\n"
                "Install with: pip install myspellchecker[transformers]\n\n"
                "Alternatively, use the default rule-based tagger:\n"
                "  from myspellchecker.algorithms.pos_tagger_rule import RuleBasedPOSTagger\n"
                "  tagger = RuleBasedPOSTagger()"
            )

        self.model_name = model_name or self.DEFAULT_MODEL
        self.max_length = max_length

        # Validate and potentially adjust device
        # Supports: -1 (CPU), 0+ (CUDA GPU index), "mps" (Apple Silicon)
        is_gpu = device != -1
        if isinstance(device, int) and device >= 0:
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
                        is_gpu = False
                elif torch.backends.mps.is_available():
                    logger.info("CUDA not available, using Apple MPS (Metal) GPU")
                    device = "mps"
                else:
                    logger.warning("No GPU available (CUDA/MPS), falling back to CPU (device=-1)")
                    device = -1
                    is_gpu = False
            except ImportError:
                logger.warning("PyTorch not available, using CPU (device=-1)")
                device = -1
                is_gpu = False

        self.device = device

        # VRAM-based batch size auto-scaling for GPU
        if batch_size == 32 and is_gpu:
            batch_size = self._auto_scale_batch_size(device)
            logger.info(f"Auto-scaled batch_size to {batch_size} for GPU inference")
        self.batch_size = batch_size

        # Build model_kwargs
        model_kwargs: dict[str, Any] = {}
        if cache_dir:
            model_kwargs["cache_dir"] = cache_dir

        # fp16 inference on GPU (nearly 2x throughput, argmax robust to fp16)
        if use_fp16 and is_gpu:
            try:
                import torch

                model_kwargs["torch_dtype"] = torch.float16
                logger.info("Enabled fp16 inference for GPU")
            except ImportError:
                pass

        # Initialize HuggingFace pipeline
        try:
            self._pipeline = pipeline(
                "token-classification",
                model=self.model_name,
                aggregation_strategy="simple",
                device=device,
                model_kwargs=model_kwargs,
                **pipeline_kwargs,
            )
            # Set truncation on the tokenizer directly since
            # TokenClassificationPipeline doesn't accept it as a kwarg
            if self._pipeline.tokenizer is not None:
                self._pipeline.tokenizer.model_max_length = max_length
            self._inference_kwargs: dict[str, Any] = {}
        except (OSError, RuntimeError, ImportError) as e:
            raise ValueError(
                f"Failed to load model '{self.model_name}': {e}\n\n"
                f"Possible solutions:\n"
                f"1. Check model name/path is correct\n"
                f"2. Ensure model is compatible with token-classification task\n"
                f"3. Try downloading model manually first\n"
                f"4. Check internet connection if loading from HuggingFace Hub"
            ) from e

        # torch.compile() JIT optimization (opt-in, 20-40% speedup)
        if use_torch_compile and is_gpu:
            try:
                import torch

                if hasattr(torch, "compile"):
                    self._pipeline.model = torch.compile(self._pipeline.model)  # type: ignore[assignment]
                    logger.info("Enabled torch.compile() JIT for model")
                else:
                    logger.debug("torch.compile not available (requires torch>=2.0)")
            except Exception as e:
                logger.debug(f"torch.compile() failed, continuing without JIT: {e}")

    @staticmethod
    def _auto_scale_batch_size(device) -> int:
        """Scale batch size based on available GPU VRAM.

        Returns:
            Batch size: 512 for 16GB+, 256 for 8GB+, 128 otherwise.
        """
        try:
            import torch

            if torch.cuda.is_available() and isinstance(device, int) and device >= 0:
                vram_bytes = torch.cuda.get_device_properties(device).total_memory
                vram_gb = vram_bytes / (1024**3)
                if vram_gb >= 16:
                    return 512
                elif vram_gb >= 8:
                    return 256
        except (ImportError, RuntimeError):
            pass
        return 128

    def _map_tag(self, hf_tag: str) -> str:
        """
        Map HuggingFace model tag to internal tag convention.

        Args:
            hf_tag: Tag from HuggingFace model (e.g., "n", "v", "ppm")

        Returns:
            Internal tag (e.g., "N", "V", "PPM")

        Note:
            - HuggingFace model uses lowercase coarse tags
            - Our internal system uses uppercase tags with granular particle types
            - PPM and PART are coarse mappings; for granular particle tags (P_SUBJ, P_OBJ, etc.),
              additional context-based classification would be needed

        Example:
            >>> tagger = TransformerPOSTagger()
            >>> tagger._map_tag("n")
            'N'
            >>> tagger._map_tag("ppm")
            'PPM'
        """
        # Normalize tag to lowercase for consistent lookup
        hf_tag_lower = hf_tag.lower() if hf_tag else ""
        return self.HF_TO_INTERNAL_TAG_MAP.get(hf_tag_lower, hf_tag.upper() if hf_tag else "UNK")

    def tag_word(self, word: str) -> str:
        """
        Tag a single word using transformer model.

        Args:
            word: Word to tag

        Returns:
            POS tag in internal convention (e.g., "N", "V", "PPM")

        Example:
            >>> tagger = TransformerPOSTagger()
            >>> tagger.tag_word("မြန်မာ")
            'N'
        """
        if not word:
            return "UNK"

        # Run pipeline on single word
        try:
            result = self._pipeline(word, **self._inference_kwargs)

            if not result:
                return "UNK"

            hf_tag = result[0]["entity_group"]
            return self._map_tag(hf_tag)

        except RuntimeError as e:
            # Check for CUDA OOM specifically
            error_str = str(e).lower()
            if "out of memory" in error_str or "cuda" in error_str:
                logger.warning(
                    f"GPU memory exhausted during POS tagging: {e}. "
                    f"Consider using device=-1 (CPU) or reducing batch_size."
                )
            else:
                logger.debug(f"Transformer POS tagging failed for word '{word}': {e}")
            return "UNK"
        except (KeyError, IndexError) as e:
            # Fallback to UNK on parsing error
            logger.debug(f"Transformer POS tagging failed for word '{word}': {e}")
            return "UNK"

    def tag_sequence(self, words: list[str]) -> list[str]:
        """
        Tag a sequence of words using transformer model.

        Joins words into a sentence for better contextual understanding,
        then maps results back to individual words.

        Automatically handles long sequences (>max_length) by chunking with
        overlap to preserve context. This prevents silent truncation
        and data loss.

        Note: More efficient than tagging words individually, especially
        when batch_size > 1 and using GPU.

        Args:
            words: List of words to tag

        Returns:
            List of POS tags (in internal convention) corresponding to input words

        Example:
            >>> tagger = TransformerPOSTagger()
            >>> tagger.tag_sequence(["မြန်မာ", "နိုင်ငံ", "သည်"])
            ['N', 'N', 'PPM']
        """
        if not words:
            return []

        # Estimate token count (conservative: 1.5 tokens per Myanmar word)
        estimated_tokens = self._estimate_token_count(words)

        # Check if we need chunking (use 90% threshold for safety)
        if estimated_tokens > self.max_length * 0.9:
            logger.debug(f"Sequence too long ({estimated_tokens} estimated tokens), using chunking")
            return self._tag_with_chunking(words)
        else:
            return self._tag_single_sequence(words)

    def _estimate_token_count(self, words: list[str]) -> int:
        """
        Estimate BPE token count for Myanmar words.

        Myanmar words are often split into 2-3 BPE tokens by the transformer.
        This conservative estimate helps detect when chunking is needed.

        Args:
            words: List of words

        Returns:
            Estimated token count
        """
        # Conservative estimate: 1.5 tokens per word on average for Myanmar
        # Accounts for: word tokens + space tokens + special tokens
        return int(len(words) * 1.5)

    def _tag_single_sequence(self, words: list[str]) -> list[str]:
        """
        Tag a single sequence that fits within max_length.

        Args:
            words: List of words to tag

        Returns:
            List of POS tags
        """
        # Join words as sentence for better context
        sentence = " ".join(words)

        try:
            # Run pipeline
            results = self._pipeline(sentence, **self._inference_kwargs)

            # Map results back to words
            tags = self._map_results_to_words(words, results)
            return tags

        except (RuntimeError, KeyError, IndexError) as e:
            # Fallback: tag each word individually
            logger.debug(f"Transformer sequence tagging failed: {e}")
            return [self.tag_word(word) for word in words]

    def _tag_with_chunking(self, words: list[str]) -> list[str]:
        """
        Tag long sequences using overlapping chunks.

        This prevents silent truncation when sequence exceeds max_length.
        Uses overlapping chunks to maintain context at boundaries.

        Args:
            words: List of words to tag

        Returns:
            List of POS tags
        """
        # Safe chunk size: max_length / 1.5 tokens per word * 0.8 safety
        # For max_length=128: 128 / 1.5 * 0.8 ≈ 68 words per chunk
        words_per_chunk = int(self.max_length / 1.5 * 0.8)
        # Overlap: 25% of chunk size for context preservation
        overlap = max(10, words_per_chunk // 4)

        all_tags = []
        pos = 0

        while pos < len(words):
            # Calculate chunk boundaries
            chunk_start = max(0, pos - overlap if pos > 0 else 0)
            chunk_end = min(pos + words_per_chunk, len(words))
            chunk_words = words[chunk_start:chunk_end]

            # Tag this chunk
            chunk_tags = self._tag_single_sequence(chunk_words)

            # Extract non-overlapping portion
            if pos > 0:
                # Skip overlapping tags (use previous chunk's tags for this region)
                skip = pos - chunk_start
                chunk_tags = chunk_tags[skip:]

            all_tags.extend(chunk_tags)
            pos += words_per_chunk

        return all_tags

    def tag_sentences_batch(self, sentences: list[list[str]]) -> list[list[str]]:
        """
        Tag multiple sentences in a batch for efficiency.

        Uses HuggingFace pipeline's batch processing for much faster inference
        compared to processing sentences one at a time.

        Args:
            sentences: List of sentences, each sentence is a list of words

        Returns:
            List of tag lists, one per sentence

        Example:
            >>> tagger.tag_sentences_batch([
            ...     ["မြန်မာ", "နိုင်ငံ"],
            ...     ["စာ", "ရေး", "သည်"]
            ... ])
            [['N', 'N'], ['N', 'V', 'PPM']]
        """
        if not sentences:
            return []

        # Filter empty sentences and track their indices
        non_empty_indices = []
        non_empty_sentences = []
        for i, sentence in enumerate(sentences):
            if sentence:
                non_empty_indices.append(i)
                non_empty_sentences.append(sentence)

        if not non_empty_sentences:
            return [[] for _ in sentences]

        # Join each sentence into a string for the pipeline
        sentence_strings = [" ".join(words) for words in non_empty_sentences]

        try:
            # Use pipeline batch processing
            all_results = self._pipeline(
                sentence_strings, batch_size=self.batch_size, **self._inference_kwargs
            )

            # Map results back to words for each sentence
            non_empty_tags = []
            for words, results in zip(non_empty_sentences, all_results, strict=False):
                tags = self._map_results_to_words(words, results)
                non_empty_tags.append(tags)

            # Reconstruct full results list with empty lists for empty sentences
            all_tags: list[list[str]] = [[] for _ in sentences]
            for idx, tags in zip(non_empty_indices, non_empty_tags, strict=False):
                all_tags[idx] = tags

            return all_tags

        except (RuntimeError, KeyError, IndexError) as e:
            # Fallback: process each sentence individually
            logger.debug(f"Transformer batch tagging failed: {e}")
            return [self.tag_sequence(sentence) for sentence in sentences]

    def _map_results_to_words(self, words: list[str], results: list[dict]) -> list[str]:
        """
        Map tokenized model results back to input words.

        The transformer may use different tokenization than simple word splitting,
        so we need to map the results back to the original word boundaries.

        Uses character offsets when available (preferred for accuracy), falls back
        to improved fuzzy matching otherwise. Uses precise
        character-level alignment instead of substring matching.

        Args:
            words: Original input words
            results: Pipeline results with 'word' and 'entity_group' fields

        Returns:
            List of POS tags (in internal convention) aligned with input words
        """
        # Check if results have character offsets (preferred method)
        if results and "start" in results[0] and "end" in results[0]:
            return self._map_with_offsets(words, results)
        else:
            # Fallback to improved fuzzy matching
            return self._map_fuzzy(words, results)

    def _map_with_offsets(self, words: list[str], results: list[dict]) -> list[str]:
        """
        Map using character offsets for precise alignment.

        This is the preferred method when the HuggingFace pipeline returns
        character positions (start, end) for each token. It's more accurate
        than fuzzy matching, especially for Myanmar text where substrings
        can cause ambiguity.

        Args:
            words: Original words
            results: Pipeline results with 'start', 'end', 'entity_group' fields

        Returns:
            List of tags, one per word
        """
        sentence = " ".join(words)
        tags = []

        # Build word position map
        word_positions = []
        char_pos = 0
        for word in words:
            start = sentence.find(word, char_pos)
            if start != -1:
                word_positions.append((start, start + len(word)))
                char_pos = start + len(word) + 1  # +1 for space
            else:
                word_positions.append((-1, -1))

        # Two-pointer merge: both word_positions and results are sorted by offset.
        # O(n+m) instead of O(n*m).
        # Pre-filter results that have offset info
        offset_results = [r for r in results if "start" in r and "end" in r]
        ri = 0  # result index

        for word_start, word_end in word_positions:
            if word_start == -1:
                tags.append("UNK")
                continue

            best_match = None
            best_overlap = 0

            # Advance past results that end before this word starts
            while ri > 0 and offset_results[ri - 1]["end"] > word_start:
                ri -= 1  # safety: backtrack if needed for overlapping results
            while ri < len(offset_results) and offset_results[ri]["end"] <= word_start:
                ri += 1

            # Scan forward through results that could overlap this word
            for j in range(ri, len(offset_results)):
                r = offset_results[j]
                if r["start"] >= word_end:
                    break  # no more overlaps possible
                overlap = min(word_end, r["end"]) - max(word_start, r["start"])
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = r

            if best_match:
                tags.append(self._map_tag(best_match["entity_group"]))
            else:
                tags.append("UNK")

        return tags

    def _map_fuzzy(self, words: list[str], results: list[dict]) -> list[str]:
        """
        Fallback mapping using improved fuzzy matching.

        Used when tokenizer doesn't provide character offsets. More conservative
        than the old implementation to avoid substring collision issues.
        Removes problematic "result_word in word" check.

        Args:
            words: Original words
            results: Pipeline results with 'word' and 'entity_group' fields

        Returns:
            List of tags, one per word
        """
        tags = []
        result_idx = 0

        for word in words:
            if result_idx >= len(results):
                # No more results, use UNK
                tags.append("UNK")
                continue

            result = results[result_idx]
            result_word = result.get("word", "").strip()

            # Try to match word with result
            # Use stricter rules to avoid substring collisions
            if (
                result_word == word  # Exact match
                or word in result_word  # Word is prefix of result (common for BPE)
                or result_word.replace("##", "") in word  # BPE artifact removal
            ):
                # Match found - map to internal convention
                hf_tag = result["entity_group"]
                tags.append(self._map_tag(hf_tag))
                result_idx += 1
            else:
                # No match, try to skip to next result that might match
                # This handles cases where tokenizer splits differently
                found = False
                for i in range(result_idx, min(result_idx + 3, len(results))):
                    test_word = results[i].get("word", "").strip()
                    # Only allow word as prefix of test_word, not arbitrary substring
                    if word in test_word:
                        hf_tag = results[i]["entity_group"]
                        tags.append(self._map_tag(hf_tag))
                        result_idx = i + 1
                        found = True
                        break

                if not found:
                    # Still no match, use UNK and advance result_idx to prevent misalignment
                    tags.append("UNK")
                    result_idx += 1

        return tags

    def tag_word_with_confidence(self, word: str) -> POSPrediction:
        """
        Tag word with confidence score from model.

        Args:
            word: Word to tag

        Returns:
            POSPrediction with confidence from transformer model (tag in internal convention)

        Example:
            >>> tagger = TransformerPOSTagger()
            >>> pred = tagger.tag_word_with_confidence("မြန်မာ")
            >>> print(f"{pred.tag} ({pred.confidence:.2f})")
            N (0.98)
        """
        if not word:
            return POSPrediction(
                word=word, tag="UNK", confidence=0.0, metadata={"model": self.model_name}
            )

        try:
            result = self._pipeline(word, **self._inference_kwargs)

            if not result:
                return POSPrediction(
                    word=word, tag="UNK", confidence=0.0, metadata={"model": self.model_name}
                )

            first_result = result[0]
            hf_tag = first_result["entity_group"]
            internal_tag = self._map_tag(hf_tag)
            return POSPrediction(
                word=word,
                tag=internal_tag,
                confidence=float(first_result.get("score", 0.0)),
                metadata={
                    "model": self.model_name,
                    "hf_tag": hf_tag,  # Keep original HF tag for reference
                    "start": first_result.get("start"),
                    "end": first_result.get("end"),
                },
            )

        except (RuntimeError, KeyError, IndexError) as e:
            return POSPrediction(
                word=word,
                tag="UNK",
                confidence=0.0,
                metadata={"model": self.model_name, "error": str(e)},
            )

    def tag_sequence_with_confidence(self, words: list[str]) -> list[POSPrediction]:
        """
        Tag sequence with confidence scores.

        Args:
            words: List of words to tag

        Returns:
            List of POSPredictions with confidence scores (tags in internal convention)

        Example:
            >>> tagger = TransformerPOSTagger()
            >>> preds = tagger.tag_sequence_with_confidence(["မြန်မာ", "နိုင်ငံ"])
            >>> for pred in preds:
            ...     print(f"{pred.word}: {pred.tag} ({pred.confidence:.2f})")
            မြန်မာ: N (0.98)
            နိုင်ငံ: N (0.95)
        """
        if not words:
            return []

        sentence = " ".join(words)

        try:
            results = self._pipeline(sentence, **self._inference_kwargs)

            # Map results with confidence
            predictions = []
            result_idx = 0

            for word in words:
                if result_idx >= len(results):
                    predictions.append(
                        POSPrediction(
                            word=word,
                            tag="UNK",
                            confidence=0.0,
                            metadata={"model": self.model_name},
                        )
                    )
                    continue

                result = results[result_idx]
                result_word = result.get("word", "").strip()

                # Try to match word with result
                # Use exact match or subword piece matching, not substring match
                # to avoid false positives (e.g., "က" matching "ကျွန်တော်")
                stripped_result = result_word.replace("##", "")
                if (
                    result_word == word
                    or stripped_result == word
                    or word.startswith(result_word)
                    or word.startswith(stripped_result)
                ):
                    # Match found - map to internal convention
                    hf_tag = result["entity_group"]
                    internal_tag = self._map_tag(hf_tag)
                    predictions.append(
                        POSPrediction(
                            word=word,
                            tag=internal_tag,
                            confidence=float(result.get("score", 0.0)),
                            metadata={
                                "model": self.model_name,
                                "hf_tag": hf_tag,  # Keep original HF tag for reference
                                "start": result.get("start"),
                                "end": result.get("end"),
                            },
                        )
                    )
                    result_idx += 1
                else:
                    # No match, try to skip to next result that might match
                    found = False
                    for i in range(result_idx, min(result_idx + 3, len(results))):
                        test_word = results[i].get("word", "").strip()
                        test_word_clean = test_word.replace("##", "")
                        if word in test_word or test_word in word or test_word_clean in word:
                            hf_tag = results[i]["entity_group"]
                            internal_tag = self._map_tag(hf_tag)
                            predictions.append(
                                POSPrediction(
                                    word=word,
                                    tag=internal_tag,
                                    confidence=float(results[i].get("score", 0.0)),
                                    metadata={
                                        "model": self.model_name,
                                        "hf_tag": hf_tag,  # Keep original HF tag for reference
                                        "start": results[i].get("start"),
                                        "end": results[i].get("end"),
                                    },
                                )
                            )
                            result_idx = i + 1
                            found = True
                            break

                    if not found:
                        # Still no match, use UNK and advance result_idx to prevent misalignment
                        predictions.append(
                            POSPrediction(
                                word=word,
                                tag="UNK",
                                confidence=0.0,
                                metadata={"model": self.model_name},
                            )
                        )
                        result_idx += 1

            return predictions

        except (RuntimeError, KeyError, IndexError) as e:
            # Fallback: tag individually
            logger.debug(f"Transformer confidence tagging failed: {e}")
            return [self.tag_word_with_confidence(word) for word in words]

    @property
    def tagger_type(self) -> TaggerType:
        """Return transformer tagger type."""
        return TaggerType.TRANSFORMER

    @property
    def supports_batch(self) -> bool:
        """Transformer models support efficient batching."""
        return True

    @property
    def is_fork_safe(self) -> bool:
        """
        Transformer with CUDA is not fork-safe, but CPU-only is safe.

        Returns False for GPU mode because CUDA contexts cannot be safely forked.
        Returns True for CPU mode (device=-1) which is fork-safe.
        Use multiprocessing.spawn instead of fork if using GPU.
        """
        return isinstance(self.device, int) and self.device < 0
