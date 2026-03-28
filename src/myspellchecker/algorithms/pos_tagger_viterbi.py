"""
Viterbi POS tagger adapter implementing POSTaggerBase interface.

This module provides a wrapper around the existing ViterbiTagger class,
adapting it to the pluggable POSTaggerBase interface. It uses HMM-based
sequence tagging with trigram probabilities for context-aware POS tagging.

Features:
- Context-aware sequence tagging using Viterbi algorithm
- Trigram HMM with transition and emission probabilities
- Beam pruning for performance optimization
- Morphological fallback for OOV words
- Cython acceleration when available

Performance:
- CPU: ~20K words/second
- Accuracy: ~85% (with proper probability tables)
- Memory: ~50MB for probability tables

Requirements:
    - DictionaryProvider with POS probability tables
    - No external dependencies (pure Python + optional Cython)

Example:
    >>> from myspellchecker.providers import SQLiteProvider
    >>> from myspellchecker.algorithms.pos_tagger_viterbi import ViterbiPOSTaggerAdapter
    >>>
    >>> # Initialize provider with probability tables
    >>> provider = SQLiteProvider("mydict.db")
    >>>
    >>> # Create Viterbi tagger
    >>> tagger = ViterbiPOSTaggerAdapter(
    ...     provider=provider,
    ...     beam_width=10,
    ...     emission_weight=1.2
    ... )
    >>> tagger.tag_sequence(["မြန်မာ", "နိုင်ငံ", "သည်"])
    ['n', 'n', 'ppm']
"""

from __future__ import annotations

from myspellchecker.algorithms.pos_tagger_base import (
    POSPrediction,
    POSTaggerBase,
    TaggerType,
)
from myspellchecker.algorithms.viterbi import ViterbiTagger
from myspellchecker.providers import DictionaryProvider
from myspellchecker.utils.logging_utils import get_logger


class ViterbiPOSTaggerAdapter(POSTaggerBase):
    """
    Adapter wrapping ViterbiTagger to implement POSTaggerBase interface.

    This adapter provides a pluggable interface to the existing ViterbiTagger,
    enabling it to be used interchangeably with other POS tagger implementations
    through the factory pattern.

    Attributes:
        provider: DictionaryProvider for word lookups and probability tables
        beam_width: Number of top states to keep per position (default: 10)
        emission_weight: Weight for emission probabilities (default: 1.2)
        unknown_tag: Tag for completely unknown words (default: "UNK")

    Note:
        The Viterbi tagger requires probability tables from the provider.
        If tables are not available, it will fall back to morphological analysis.

    Example:
        >>> # Default initialization
        >>> tagger = ViterbiPOSTaggerAdapter(provider=provider)
        >>>
        >>> # Custom configuration
        >>> tagger = ViterbiPOSTaggerAdapter(
        ...     provider=provider,
        ...     beam_width=15,
        ...     emission_weight=1.5,
        ...     use_morphology_fallback=True
        ... )
        >>>
        >>> # Tag sequence
        >>> tags = tagger.tag_sequence(["word1", "word2", "word3"])
    """

    logger = get_logger(__name__)

    def __init__(
        self,
        provider: DictionaryProvider,
        pos_bigram_probs: dict[tuple[str, str], float] | None = None,
        pos_trigram_probs: dict[tuple[str, str, str], float] | None = None,
        pos_unigram_probs: dict[str, float] | None = None,
        unknown_tag: str = "UNK",
        min_prob: float = 1e-10,
        beam_width: int = 10,
        emission_weight: float = 1.2,
        *,
        use_morphology_fallback: bool = True,
    ):
        """
        Initialize Viterbi POS tagger adapter.

        Args:
            provider: DictionaryProvider for word lookups and probability tables
            pos_bigram_probs: Bigram transition probabilities P(tag2 | tag1).
                            If None, will attempt to load from provider.
            pos_trigram_probs: Trigram transition probabilities P(tag3 | tag1, tag2).
                             If None, will attempt to load from provider.
            pos_unigram_probs: Unigram tag probabilities P(tag).
                             If None, will attempt to load from provider.
            unknown_tag: Tag to return for completely unknown words (default: "UNK")
            min_prob: Minimum probability to prevent underflow (default: 1e-10)
            beam_width: Number of top states to keep per position for beam pruning.
                       Higher values increase accuracy but decrease speed. (default: 10)
            emission_weight: Weight for emission probabilities. Values > 1.0 give
                           more weight to word-tag associations. (default: 1.2)
            use_morphology_fallback: Whether to use MorphologyAnalyzer for
                                       OOV words (default: True)

        Example:
            >>> # Default initialization (load probs from provider)
            >>> tagger = ViterbiPOSTaggerAdapter(provider=provider)
            >>>
            >>> # Custom probability tables
            >>> bigram_probs = {("n", "v"): 0.3, ("v", "n"): 0.2}
            >>> trigram_probs = {("n", "v", "n"): 0.15}
            >>> tagger = ViterbiPOSTaggerAdapter(
            ...     provider=provider,
            ...     pos_bigram_probs=bigram_probs,
            ...     pos_trigram_probs=trigram_probs
            ... )
        """
        self.provider = provider
        self.unknown_tag = unknown_tag

        # Load probability tables from provider if not provided
        if pos_bigram_probs is None:
            pos_bigram_probs = self._load_bigram_probs()
        if pos_trigram_probs is None:
            pos_trigram_probs = self._load_trigram_probs()
        if pos_unigram_probs is None:
            pos_unigram_probs = self._load_unigram_probs()

        # Initialize wrapped ViterbiTagger
        self._viterbi_tagger = ViterbiTagger(
            provider=provider,
            pos_bigram_probs=pos_bigram_probs,
            pos_trigram_probs=pos_trigram_probs,
            pos_unigram_probs=pos_unigram_probs,
            unknown_word_tag=unknown_tag,
            min_prob=min_prob,
            beam_width=beam_width,
            emission_weight=emission_weight,
            use_morphology_fallback=use_morphology_fallback,
        )

    def _load_bigram_probs(self) -> dict[tuple[str, str], float]:
        """
        Load bigram transition probabilities from provider.

        Returns:
            Dictionary mapping (tag1, tag2) tuples to probabilities.
            Returns empty dict if provider doesn't support probability tables.
        """
        if hasattr(self.provider, "get_pos_bigram_probabilities"):
            probs = self.provider.get_pos_bigram_probabilities()
            self.logger.debug(f"Loaded {len(probs)} bigram probabilities from provider")
            return probs
        self.logger.debug("Provider does not support bigram probabilities, using empty dict")
        return {}

    def _load_trigram_probs(self) -> dict[tuple[str, str, str], float]:
        """
        Load trigram transition probabilities from provider.

        Returns:
            Dictionary mapping (tag1, tag2, tag3) tuples to probabilities.
            Returns empty dict if provider doesn't support probability tables.
        """
        if hasattr(self.provider, "get_pos_trigram_probabilities"):
            probs = self.provider.get_pos_trigram_probabilities()
            self.logger.debug(f"Loaded {len(probs)} trigram probabilities from provider")
            return probs
        self.logger.debug("Provider does not support trigram probabilities, using empty dict")
        return {}

    def _load_unigram_probs(self) -> dict[str, float]:
        """
        Load unigram tag probabilities from provider.

        Returns:
            Dictionary mapping tags to probabilities.
            Returns empty dict if provider doesn't support probability tables.
        """
        if hasattr(self.provider, "get_pos_unigram_probabilities"):
            probs = self.provider.get_pos_unigram_probabilities()
            self.logger.debug(f"Loaded {len(probs)} unigram probabilities from provider")
            return probs
        self.logger.debug("Provider does not support unigram probabilities, using empty dict")
        return {}

    def tag_word(self, word: str) -> str:
        """
        Tag a single word using Viterbi algorithm.

        Note: For single words, Viterbi degenerates to unigram-based tagging
        since there's no sequence context. Consider using tag_sequence() for
        better accuracy when context is available.

        Args:
            word: Word to tag

        Returns:
            POS tag (e.g., "n", "v", "ppm")

        Example:
            >>> tagger = ViterbiPOSTaggerAdapter(provider=provider)
            >>> tagger.tag_word("မြန်မာ")
            'n'
        """
        if not word:
            return self.unknown_tag

        # Tag as single-word sequence
        tags = self._viterbi_tagger.tag_sequence([word])
        return tags[0] if tags else self.unknown_tag

    def tag_sequence(self, words: list[str]) -> list[str]:
        """
        Tag a sequence of words using Viterbi algorithm.

        Uses HMM with trigram transition probabilities and emission scores
        to find the most likely POS tag sequence. More accurate than tagging
        words individually due to contextual information.

        Args:
            words: List of words to tag

        Returns:
            List of POS tags corresponding to input words

        Example:
            >>> tagger = ViterbiPOSTaggerAdapter(provider=provider)
            >>> tagger.tag_sequence(["မြန်မာ", "နိုင်ငံ", "သည်"])
            ['n', 'n', 'ppm']
        """
        if not words:
            return []

        return self._viterbi_tagger.tag_sequence(words)

    def tag_word_with_confidence(self, word: str) -> POSPrediction:
        """
        Tag word with confidence score.

        Note: Viterbi algorithm doesn't directly provide confidence scores.
        This method returns confidence=1.0 for consistency with the interface.
        For true confidence estimates, consider using a probabilistic model.

        Args:
            word: Word to tag

        Returns:
            POSPrediction with tag and confidence score

        Example:
            >>> tagger = ViterbiPOSTaggerAdapter(provider=provider)
            >>> pred = tagger.tag_word_with_confidence("မြန်မာ")
            >>> print(f"{pred.tag} (conf: {pred.confidence:.2f})")
            n (conf: 1.00)
        """
        tag = self.tag_word(word)

        return POSPrediction(
            word=word,
            tag=tag,
            confidence=1.0,  # Viterbi doesn't provide direct confidence scores
            metadata={
                "method": "viterbi",
                "beam_width": self._viterbi_tagger.beam_width,
                "emission_weight": self._viterbi_tagger.emission_weight,
            },
        )

    def tag_sequence_with_confidence(self, words: list[str]) -> list[POSPrediction]:
        """
        Tag sequence with per-position confidence scores.

        Uses Viterbi lattice-based marginal probabilities to compute real
        confidence scores instead of returning hardcoded 1.0. The confidence
        is the marginal probability P(tag_i | word_sequence) computed via
        log-sum-exp normalization over the Viterbi lattice.

        Args:
            words: List of words to tag

        Returns:
            List of POSPredictions with tags and marginal confidence scores

        Example:
            >>> tagger = ViterbiPOSTaggerAdapter(provider=provider)
            >>> preds = tagger.tag_sequence_with_confidence(["မြန်မာ", "နိုင်ငံ"])
            >>> for pred in preds:
            ...     print(f"{pred.word}: {pred.tag} ({pred.confidence:.2f})")
            မြန်မာ: n (0.95)
            နိုင်ငံ: n (0.88)
        """
        if not words:
            return []

        tags = self.tag_sequence(words)

        # Compute marginal probabilities for confidence scores
        try:
            marginals = self._viterbi_tagger.compute_marginals(words)
        except (RuntimeError, ValueError, TypeError):
            marginals = None

        predictions = []
        for i, (word, tag) in enumerate(zip(words, tags, strict=False)):
            # Get marginal confidence for the MAP tag at this position
            if marginals and i < len(marginals):
                tag_lower = tag.lower()
                confidence = marginals[i].get(tag_lower, marginals[i].get(tag, 1.0))
            else:
                confidence = 1.0

            predictions.append(
                POSPrediction(
                    word=word,
                    tag=tag,
                    confidence=confidence,
                    metadata={
                        "method": "viterbi_marginal",
                        "beam_width": self._viterbi_tagger.beam_width,
                        "position": i,
                    },
                )
            )

        return predictions

    @property
    def tagger_type(self) -> TaggerType:
        """Return Viterbi tagger type."""
        return TaggerType.VITERBI

    @property
    def supports_batch(self) -> bool:
        """
        Viterbi tagger does not benefit from batching.

        Viterbi processes sequences individually, so batch processing
        doesn't provide performance benefits over sequential processing.
        """
        return False

    @property
    def is_fork_safe(self) -> bool:
        """
        Viterbi tagger fork-safety depends on Cython extension.

        Returns True for pure Python implementation, but may be False
        if Cython extensions are used. Generally safe for multiprocessing.
        """
        # Pure Python Viterbi is fork-safe
        # Cython version is also fork-safe (no CUDA like transformers)
        return True
