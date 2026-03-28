"""
Abstract base class for POS taggers.

This module defines the interface that all POS tagger implementations must follow,
enabling pluggable backends for Part-of-Speech tagging.

Supports multiple tagger types:
- RuleBased: Morphological suffix analysis (fast, no dependencies)
- Viterbi: HMM-based sequence tagging (context-aware)
- Transformer: Neural models from HuggingFace (high accuracy)
- Custom: User-provided implementations

Example:
    >>> from myspellchecker.algorithms.pos_tagger_factory import POSTaggerFactory
    >>>
    >>> # Create rule-based tagger (default)
    >>> tagger = POSTaggerFactory.create(tagger_type="rule_based")
    >>> tags = tagger.tag_sequence(["မြန်မာ", "နိုင်ငံ"])
    >>>
    >>> # Create transformer tagger
    >>> tagger = POSTaggerFactory.create(
    ...     tagger_type="transformer",
    ...     model_name="chuuhtetnaing/myanmar-pos-model"
    ... )
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


class TaggerType(Enum):
    """
    Available POS tagger implementation types.

    Attributes:
        RULE_BASED: Morphological suffix analysis (default)
        VITERBI: HMM-based sequence tagging
        TRANSFORMER: Neural models from HuggingFace
        CUSTOM: User-provided implementation
    """

    RULE_BASED = "rule_based"
    VITERBI = "viterbi"
    TRANSFORMER = "transformer"
    CUSTOM = "custom"


@dataclass
class POSPrediction:
    """
    POS prediction result with confidence score.

    Attributes:
        word: The input word
        tag: Predicted POS tag (e.g., "N", "V", "P_SENT")
        confidence: Confidence score between 0.0 and 1.0
        metadata: Additional tagger-specific information (optional)

    Example:
        >>> prediction = POSPrediction(
        ...     word="မြန်မာ",
        ...     tag="N",
        ...     confidence=0.95,
        ...     metadata={"method": "transformer"}
        ... )
    """

    word: str
    tag: str
    confidence: float
    metadata: dict | None = None

    def __repr__(self) -> str:
        return (
            f"POSPrediction(word='{self.word}', tag='{self.tag}', confidence={self.confidence:.2f})"
        )


class POSTaggerBase(ABC):
    """
    Abstract base class for all POS taggers.

    All POS tagger implementations must inherit from this class and implement
    the required abstract methods.

    Required Methods:
        - tag_word(word) -> str: Tag a single word
        - tag_sequence(words) -> list[str]: Tag a sequence of words
        - tagger_type -> TaggerType: Return the tagger type identifier

    Optional Methods (with default implementations):
        - tag_word_with_confidence(word) -> POSPrediction
        - tag_sequence_with_confidence(words) -> list[POSPrediction]

    Properties:
        - supports_batch: Whether the tagger supports efficient batch processing
        - is_fork_safe: Whether the tagger can be used in forked processes

    Example:
        >>> class MyCustomTagger(POSTaggerBase):
        ...     def tag_word(self, word: str) -> str:
        ...         return "N"  # Simple implementation
        ...
        ...     def tag_sequence(self, words: list[str]) -> list[str]:
        ...         return ["N"] * len(words)
        ...
        ...     @property
        ...     def tagger_type(self) -> TaggerType:
        ...         return TaggerType.CUSTOM
    """

    @abstractmethod
    def tag_word(self, word: str) -> str:
        """
        Tag a single word with its POS tag.

        Args:
            word: The word to tag

        Returns:
            POS tag string (e.g., "N", "V", "P_SENT", "ADJ")

        Example:
            >>> tagger.tag_word("မြန်မာ")
            'N'
        """

    @abstractmethod
    def tag_sequence(self, words: list[str]) -> list[str]:
        """
        Tag a sequence of words with their POS tags.

        This method may use contextual information to improve tagging accuracy.
        The returned list must have the same length as the input list.

        Args:
            words: List of words to tag

        Returns:
            List of POS tags corresponding to input words

        Example:
            >>> tagger.tag_sequence(["မြန်မာ", "နိုင်ငံ", "သည်"])
            ['N', 'N', 'P_SENT']
        """

    def tag_word_with_confidence(self, word: str) -> POSPrediction:
        """
        Tag a word and return prediction with confidence score.

        Default implementation returns confidence=1.0. Subclasses should override
        for better confidence estimation.

        Args:
            word: The word to tag

        Returns:
            POSPrediction with tag and confidence score

        Example:
            >>> pred = tagger.tag_word_with_confidence("မြန်မာ")
            >>> print(f"{pred.word}: {pred.tag} ({pred.confidence:.2f})")
            မြန်မာ: N (0.95)
        """
        tag = self.tag_word(word)
        return POSPrediction(
            word=word, tag=tag, confidence=1.0, metadata={"method": self.__class__.__name__}
        )

    def tag_sequence_with_confidence(self, words: list[str]) -> list[POSPrediction]:
        """
        Tag a sequence and return predictions with confidence scores.

        Default implementation returns confidence=1.0 for all words. Subclasses
        should override for better confidence estimation.

        Args:
            words: List of words to tag

        Returns:
            List of POSPredictions with tags and confidence scores

        Example:
            >>> preds = tagger.tag_sequence_with_confidence(["မြန်မာ", "နိုင်ငံ"])
            >>> for pred in preds:
            ...     print(f"{pred.word}: {pred.tag} ({pred.confidence:.2f})")
            မြန်မာ: N (0.95)
            နိုင်ငံ: N (0.92)
        """
        tags = self.tag_sequence(words)
        return [
            POSPrediction(
                word=word, tag=tag, confidence=1.0, metadata={"method": self.__class__.__name__}
            )
            for word, tag in zip(words, tags, strict=False)
        ]

    def tag_sentences_batch(self, sentences: list[list[str]]) -> list[list[str]]:
        """
        Tag multiple sentences in a batch for efficiency.

        Default implementation calls tag_sequence for each sentence.
        Subclasses (especially transformer-based) should override for
        better batch processing performance.

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
        return [self.tag_sequence(sentence) for sentence in sentences]

    @property
    @abstractmethod
    def tagger_type(self) -> TaggerType:
        """
        Return the tagger type identifier.

        Returns:
            TaggerType enum value

        Example:
            >>> tagger.tagger_type
            <TaggerType.RULE_BASED: 'rule_based'>
        """

    @property
    def supports_batch(self) -> bool:
        """
        Whether this tagger supports efficient batch processing.

        Returns:
            True if the tagger can process multiple sequences efficiently,
            False otherwise (default)

        Note:
            Transformer-based taggers typically return True as they benefit
            from batch processing on GPU. Rule-based taggers typically return
            False as they process each word independently.
        """
        return False

    @property
    def is_fork_safe(self) -> bool:
        """
        Whether this tagger can be used in forked processes.

        Returns:
            True if the tagger is fork-safe (default), False otherwise

        Note:
            Transformer-based taggers with CUDA typically return False due to
            CUDA context issues in forked processes. Pure Python taggers
            typically return True.

        Example:
            >>> if tagger.is_fork_safe:
            ...     # Can use in multiprocessing.Pool
            ...     with Pool() as pool:
            ...         results = pool.map(tagger.tag_word, words)
            ... else:
            ...     # Must process in main process
            ...     results = [tagger.tag_word(w) for w in words]
        """
        return True
