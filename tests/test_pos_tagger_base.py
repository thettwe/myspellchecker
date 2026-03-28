"""
Unit tests for POS tagger base classes and interfaces.

Tests the abstract base class, dataclasses, and enums.
"""

from typing import List

import pytest

from myspellchecker.algorithms.pos_tagger_base import (
    POSPrediction,
    POSTaggerBase,
    TaggerType,
)


class TestTaggerType:
    """Test TaggerType enum."""

    def test_enum_values(self):
        """Test that all expected tagger types are defined."""
        assert TaggerType.RULE_BASED.value == "rule_based"
        assert TaggerType.VITERBI.value == "viterbi"
        assert TaggerType.TRANSFORMER.value == "transformer"
        assert TaggerType.CUSTOM.value == "custom"

    def test_enum_membership(self):
        """Test enum membership checks."""
        assert TaggerType.RULE_BASED in TaggerType
        assert TaggerType.VITERBI in TaggerType
        assert TaggerType.TRANSFORMER in TaggerType
        assert TaggerType.CUSTOM in TaggerType


class TestPOSPrediction:
    """Test POSPrediction dataclass."""

    def test_basic_prediction(self):
        """Test basic prediction creation."""
        pred = POSPrediction(
            word="မြန်မာ",
            tag="N",
            confidence=0.95,
        )

        assert pred.word == "မြန်မာ"
        assert pred.tag == "N"
        assert pred.confidence == 0.95
        assert pred.metadata is None

    def test_prediction_with_metadata(self):
        """Test prediction with metadata."""
        metadata = {"source": "transformer", "alternatives": ["V", "ADJ"]}
        pred = POSPrediction(
            word="ကောင်း",
            tag="ADJ",
            confidence=0.87,
            metadata=metadata,
        )

        assert pred.word == "ကောင်း"
        assert pred.tag == "ADJ"
        assert pred.confidence == 0.87
        assert pred.metadata == metadata
        assert pred.metadata["source"] == "transformer"

    def test_prediction_equality(self):
        """Test prediction equality comparison."""
        pred1 = POSPrediction(word="test", tag="N", confidence=0.9)
        pred2 = POSPrediction(word="test", tag="N", confidence=0.9)
        pred3 = POSPrediction(word="test", tag="V", confidence=0.9)

        assert pred1 == pred2
        assert pred1 != pred3


class SimpleTagger(POSTaggerBase):
    """Simple concrete implementation for testing abstract base."""

    def __init__(self, default_tag: str = "N"):
        self.default_tag = default_tag

    def tag_word(self, word: str) -> str:
        """Return default tag."""
        return self.default_tag

    def tag_sequence(self, words: List[str]) -> List[str]:
        """Return default tags for all words."""
        return [self.default_tag] * len(words)

    @property
    def tagger_type(self) -> TaggerType:
        """Return CUSTOM type."""
        return TaggerType.CUSTOM


class TestPOSTaggerBase:
    """Test POSTaggerBase abstract class."""

    def test_concrete_implementation(self):
        """Test that concrete implementation works."""
        tagger = SimpleTagger(default_tag="V")

        assert tagger.tag_word("test") == "V"
        assert tagger.tag_sequence(["a", "b", "c"]) == ["V", "V", "V"]
        assert tagger.tagger_type == TaggerType.CUSTOM

    def test_tag_word_with_confidence(self):
        """Test default tag_word_with_confidence implementation."""
        tagger = SimpleTagger(default_tag="N")
        prediction = tagger.tag_word_with_confidence("မြန်မာ")

        assert isinstance(prediction, POSPrediction)
        assert prediction.word == "မြန်မာ"
        assert prediction.tag == "N"
        assert prediction.confidence == 1.0
        # Default implementation adds metadata with method name
        assert prediction.metadata is not None
        assert "method" in prediction.metadata

    def test_tag_sequence_with_confidence(self):
        """Test default tag_sequence_with_confidence implementation."""
        tagger = SimpleTagger(default_tag="ADJ")
        words = ["ကောင်း", "လှ", "သန့်"]
        predictions = tagger.tag_sequence_with_confidence(words)

        assert len(predictions) == 3
        assert all(isinstance(p, POSPrediction) for p in predictions)
        assert [p.word for p in predictions] == words
        assert all(p.tag == "ADJ" for p in predictions)
        assert all(p.confidence == 1.0 for p in predictions)

    def test_supports_batch_default(self):
        """Test that supports_batch defaults to False."""
        tagger = SimpleTagger()
        assert tagger.supports_batch is False

    def test_is_fork_safe_default(self):
        """Test that is_fork_safe defaults to True."""
        tagger = SimpleTagger()
        assert tagger.is_fork_safe is True

    def test_cannot_instantiate_abstract_class(self):
        """Test that POSTaggerBase cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            POSTaggerBase()  # type: ignore


class BatchSupportingTagger(SimpleTagger):
    """Tagger that supports batch processing."""

    @property
    def supports_batch(self) -> bool:
        return True


class NonForkSafeTagger(SimpleTagger):
    """Tagger that is not fork-safe (e.g., uses CUDA)."""

    @property
    def is_fork_safe(self) -> bool:
        return False


class TestTaggerProperties:
    """Test tagger property overrides."""

    def test_batch_support_override(self):
        """Test overriding supports_batch property."""
        tagger = BatchSupportingTagger()
        assert tagger.supports_batch is True

    def test_fork_safety_override(self):
        """Test overriding is_fork_safe property."""
        tagger = NonForkSafeTagger()
        assert tagger.is_fork_safe is False

    def test_combined_property_overrides(self):
        """Test that properties can be independently overridden."""
        simple = SimpleTagger()
        batch = BatchSupportingTagger()
        non_fork = NonForkSafeTagger()

        # Simple tagger: no batch, fork-safe
        assert simple.supports_batch is False
        assert simple.is_fork_safe is True

        # Batch tagger: batch support, fork-safe
        assert batch.supports_batch is True
        assert batch.is_fork_safe is True

        # Non-fork tagger: no batch, not fork-safe
        assert non_fork.supports_batch is False
        assert non_fork.is_fork_safe is False
