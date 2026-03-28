"""
Abstract base class for Myanmar text segmentation.

This module defines the Segmenter interface that all segmentation
implementations must follow.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

__all__ = [
    "Segmenter",
]


class Segmenter(ABC):
    """
    Abstract interface for Myanmar text segmentation.

    Myanmar text is written in scriptio continua (continuous text without spaces),
    requiring specialized segmentation algorithms. Implementations of this interface
    handle breaking text into syllables and words.

    Thread Safety:
        Implementations should be thread-safe for read operations.
    """

    @abstractmethod
    def segment_syllables(self, text: str) -> list[str]:
        """
        Segment Myanmar text into syllables using deterministic rules.

        Syllable segmentation in Myanmar is rule-based and achieves very high
        accuracy (>99.9%) because syllable boundaries follow consistent patterns
        in the Myanmar script.

        Args:
            text: Raw Myanmar text (Unicode) in scriptio continua format.
                  May contain mixed content (Myanmar + punctuation + numbers).

        Returns:
            List of Myanmar syllables (Unicode strings) preserving order.
            Non-Myanmar content (punctuation, numbers, spaces) may be
            preserved or filtered depending on implementation.

        Raises:
            ValueError: If text is empty or invalid.
            TypeError: If text is not a string.

        Expected Performance:
            <10ms per sentence (20-30 syllables)

        Expected Accuracy:
            >99.9% on well-formed Myanmar text

        Example:
            >>> segmenter = DefaultSegmenter()
            >>> segmenter.segment_syllables("မြနမ်ာနိုငံ")
            ["မြန်", "မာ", "နို", "ငံ"]

            >>> segmenter.segment_syllables("သူသွားသည်။")
            ["သူ", "သွား", "သည်", "။"]

        Notes:
            - Input text should be Unicode-normalized before segmentation
            - Myanmar3 normalization is recommended for handling encoding issues
            - Diacritics and tone marks must be handled correctly
        """
        raise NotImplementedError

    @abstractmethod
    def segment_words(self, text: str) -> list[str]:
        """
        Segment Myanmar text into words using statistical models.

        Word segmentation in Myanmar is more complex than syllable segmentation
        because word boundaries are ambiguous and context-dependent. This method
        typically uses CRF, Viterbi, or Transformer models trained on annotated corpora.

        Args:
            text: Raw Myanmar text (Unicode) in scriptio continua format.

        Returns:
            List of Myanmar words (Unicode strings) preserving order.
            Words may consist of one or more syllables.

        Raises:
            ValueError: If text is empty or invalid.
            TypeError: If text is not a string.

        Expected Performance:
            <50ms per sentence (10-15 words)

        Expected Accuracy:
            85-95% (inherently ambiguous - even native speakers may disagree)

        Example:
            >>> segmenter = DefaultSegmenter()
            >>> segmenter.segment_words("မြနမ်ာနိုငံ")
            ["မြနမ်ာ", "နိုငံ"]

            >>> segmenter.segment_words("သူသွားသည်။")
            ["သူ", "သွား", "သည်", "။"]

        Notes:
            - Word segmentation is probabilistic and may have multiple valid outputs
            - Different models (CRF vs Viterbi vs Transformer) may produce different results
            - Quality depends heavily on training data
        """
        raise NotImplementedError

    @abstractmethod
    def segment_sentences(self, text: str) -> list[str]:
        """
        Segment Myanmar text into sentences.

        Args:
            text: Raw Myanmar text (Unicode).

        Returns:
            List of Myanmar sentences.

        Raises:
            ValueError: If text is empty or invalid.
            TypeError: If text is not a string.
        """
        raise NotImplementedError

    def segment_and_tag(self, text: str) -> tuple[list[str], list[str]]:
        """
        Perform joint word segmentation and POS tagging.

        This is an optional method that combines word segmentation with
        POS tagging in a single pass. Not all segmenters support this;
        the default implementation raises NotImplementedError.

        For joint segmentation-tagging, consider using:
        - SpellChecker.segment_and_tag() with config.joint.enabled=True
        - JointSegmentTagger directly

        Args:
            text: Raw Myanmar text (Unicode).

        Returns:
            Tuple of (words, tags) where:
            - words: List of segmented words
            - tags: List of POS tags (same length as words)

        Raises:
            NotImplementedError: If the segmenter doesn't support joint mode.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support joint segmentation-tagging. "
            "Use SpellChecker.segment_and_tag() with config.joint.enabled=True instead. "
            "See https://docs.myspellchecker.com/features/pos-tagging"
            "#joint-segmentation-and-tagging for details."
        )
