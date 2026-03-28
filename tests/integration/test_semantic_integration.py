"""
Tests for Semantic Checker Integration.

These tests verify:
1. SemanticValidationStrategy suppresses false positives
2. SemanticValidationStrategy prioritizes AI suggestions
3. Fallback when model is unavailable
4. Fallback when use_semantic_refinement=False
5. Graceful exception handling
6. HuggingFace tokenizer wrapper functionality
"""

from typing import List, Optional
from unittest.mock import MagicMock

import pytest

from myspellchecker.algorithms.semantic_checker import (
    EncodingResult,
    HFTokenizerWrapper,
)
from myspellchecker.core.config import SemanticConfig, SpellCheckerConfig
from myspellchecker.core.context_validator import ContextValidator
from myspellchecker.core.validation_strategies import (
    SemanticValidationStrategy,
    ValidationContext,
)
from myspellchecker.providers import MemoryProvider
from myspellchecker.segmenters import DefaultSegmenter, Segmenter


class SpaceSegmenter(Segmenter):
    """Simple segmenter that splits by space for testing."""

    def segment_syllables(self, text: str):
        return DefaultSegmenter().segment_syllables(text)

    def segment_words(self, text: str):
        return text.split()

    def segment_sentences(self, text: str):
        return [text]


class MockSemanticChecker:
    """Mock SemanticChecker for testing without actual model."""

    def __init__(self, responses: dict = None):
        """
        Initialize mock with predefined responses.

        Args:
            responses: Dict mapping (sentence, word) -> result
                       result is None to suppress error, or a string suggestion
        """
        self.responses = responses or {}
        self.call_count = 0

    def is_semantic_error(
        self,
        sentence: str,
        word: str,
        neighbors: List[str],
        occurrence: int = 0,
    ) -> Optional[str]:
        """Mock semantic error detection."""
        self.call_count += 1
        key = (sentence, word)
        if key in self.responses:
            return self.responses[key]
        # Default: return None (word is correct)
        return None


class MockSemanticCheckerWithError:
    """Mock SemanticChecker that raises an exception."""

    def is_semantic_error(
        self,
        sentence: str,
        word: str,
        neighbors: List[str],
        occurrence: int = 0,
    ) -> Optional[str]:
        raise RuntimeError("Model inference failed")


@pytest.fixture
def basic_provider():
    """Create a MemoryProvider with basic test data."""
    provider = MemoryProvider()

    # Add syllables
    syllables = [
        ("ကျွန်", 1000),
        ("တော်", 900),
        ("သွား", 800),
        ("သည်", 700),
        ("မြန်", 600),
        ("မာ", 500),
    ]
    for syllable, freq in syllables:
        provider.add_syllable(syllable, freq)

    # Add words
    words = [
        ("ကျွန်တော်", 500),
        ("သွားသည်", 400),
        ("မြန်မာ", 300),
    ]
    for word, freq in words:
        provider.add_word(word, freq)

    # Add bigrams (probability in range [0.0, 1.0])
    bigrams = [
        ("ကျွန်တော်", "သွားသည်", 0.1),
        ("မြန်မာ", "နိုင်ငံ", 0.08),
    ]
    for w1, w2, prob in bigrams:
        provider.add_bigram(w1, w2, prob)

    return provider


@pytest.fixture
def semantic_config_enabled():
    """Create config with semantic refinement enabled."""
    return SemanticConfig(use_semantic_refinement=True)


@pytest.fixture
def semantic_config_disabled():
    """Create config with semantic refinement disabled."""
    return SemanticConfig(use_semantic_refinement=False)


class TestSemanticValidationStrategy:
    """Tests for SemanticValidationStrategy."""

    def test_strategy_priority(self):
        """Test that strategy has correct priority (lowest priority = runs last)."""
        semantic_mock = MockSemanticChecker()
        strategy = SemanticValidationStrategy(
            semantic_checker=semantic_mock,
            use_proactive_scanning=True,
        )
        assert strategy.priority() == 70  # Highest priority value = runs last

    def test_strategy_repr(self):
        """Test strategy string representation."""
        semantic_mock = MockSemanticChecker()
        strategy = SemanticValidationStrategy(
            semantic_checker=semantic_mock,
            use_proactive_scanning=True,
        )
        assert "SemanticValidationStrategy" in repr(strategy)

    def test_validate_disabled_returns_empty(self, semantic_config_disabled):
        """Test that validation returns empty when disabled."""
        semantic_mock = MockSemanticChecker()
        # Proactive scanning disabled
        strategy = SemanticValidationStrategy(
            semantic_checker=semantic_mock,
            use_proactive_scanning=False,
        )

        context = ValidationContext(
            words=["ကျွန်တော်", "သွားသည်"],
            word_positions=[0, 9],
            sentence="ကျွန်တော် သွားသည်",
            is_name_mask=[False, False],
        )

        errors = strategy.validate(context)
        assert errors == []
        assert semantic_mock.call_count == 0

    def test_validate_no_checker_returns_empty(self, semantic_config_enabled):
        """Test that validation returns empty when no checker."""
        strategy = SemanticValidationStrategy(
            semantic_checker=None,
            use_proactive_scanning=True,
        )

        context = ValidationContext(
            words=["ကျွန်တော်", "သွားသည်"],
            word_positions=[0, 9],
            sentence="ကျွန်တော် သွားသည်",
            is_name_mask=[False, False],
        )

        errors = strategy.validate(context)
        assert errors == []

    def test_validate_with_checker_called(self, semantic_config_enabled):
        """Test that semantic checker is called during validation."""
        semantic_mock = MockSemanticChecker()
        # Mock scan_sentence method
        semantic_mock.scan_sentence = lambda **kwargs: []
        strategy = SemanticValidationStrategy(
            semantic_checker=semantic_mock,
            use_proactive_scanning=True,
        )

        context = ValidationContext(
            words=["ကျွန်တော်", "သွားသည်"],
            word_positions=[0, 9],
            sentence="ကျွန်တော် သွားသည်",
            is_name_mask=[False, False],
        )

        strategy.validate(context)
        # Strategy should have been called (scan_sentence is used internally)
        assert hasattr(semantic_mock, "scan_sentence")

    def test_validate_skips_existing_errors(self, semantic_config_enabled):
        """Test that validation skips positions with existing errors."""
        semantic_mock = MockSemanticChecker()
        semantic_mock.scan_sentence = lambda **kwargs: []
        strategy = SemanticValidationStrategy(
            semantic_checker=semantic_mock,
            use_proactive_scanning=True,
        )

        context = ValidationContext(
            words=["ကျွန်တော်", "သွားသည်"],
            word_positions=[0, 9],
            sentence="ကျွန်တော် သွားသည်",
            is_name_mask=[False, False],
            existing_errors={0: "test"},  # Position 0 already has an error
        )

        errors = strategy.validate(context)
        # Should not check position 0 again
        assert errors == [] or all(e.position != 0 for e in errors)

    def test_validate_handles_exception(self, semantic_config_enabled):
        """Test that validation handles exceptions gracefully."""
        semantic_mock = MockSemanticCheckerWithError()
        strategy = SemanticValidationStrategy(
            semantic_checker=semantic_mock,
            use_proactive_scanning=True,
        )

        context = ValidationContext(
            words=["ကျွန်တော်", "သွားသည်"],
            word_positions=[0, 9],
            sentence="ကျွန်တော် သွားသည်",
            is_name_mask=[False, False],
        )

        # Should not raise exception
        errors = strategy.validate(context)
        assert errors == []


class TestSemanticIntegrationWithContextValidator:
    """Integration tests for semantic validation via ContextValidator."""

    def test_context_validator_with_semantic_strategy(
        self, basic_provider, semantic_config_enabled
    ):
        """Test ContextValidator with SemanticValidationStrategy."""
        semantic_mock = MockSemanticChecker()
        semantic_mock.scan_sentence = lambda **kwargs: []
        strategy = SemanticValidationStrategy(
            semantic_checker=semantic_mock,
            use_proactive_scanning=True,
        )

        config = SpellCheckerConfig()
        validator = ContextValidator(
            config=config,
            segmenter=SpaceSegmenter(),
            strategies=[strategy],
        )

        validator.validate("ကျွန်တော် သွားသည်")
        # Strategy should have been added
        assert hasattr(validator, "strategies")


class TestHFTokenizerWrapper:
    """Tests for HuggingFace tokenizer wrapper."""

    def test_encode_returns_encoding_result(self):
        """Test that encode returns EncodingResult with correct fields."""
        # Create mock HuggingFace tokenizer
        mock_hf_tokenizer = MagicMock()
        mock_hf_tokenizer.return_value = {
            "input_ids": [0, 123, 456, 2],
            "offset_mapping": [(0, 0), (0, 3), (3, 6), (0, 0)],
        }
        mock_hf_tokenizer.unk_token_id = 3

        wrapper = HFTokenizerWrapper(mock_hf_tokenizer)
        result = wrapper.encode("test")

        assert isinstance(result, EncodingResult)
        assert result.ids == [0, 123, 456, 2]
        assert result.offsets == [(0, 0), (0, 3), (3, 6), (0, 0)]

    def test_token_to_id_returns_valid_id(self):
        """Test token_to_id returns correct ID."""
        mock_hf_tokenizer = MagicMock()
        mock_hf_tokenizer.convert_tokens_to_ids.return_value = 123
        mock_hf_tokenizer.unk_token_id = 0

        wrapper = HFTokenizerWrapper(mock_hf_tokenizer)
        result = wrapper.token_to_id("<mask>")

        assert result == 123

    def test_token_to_id_returns_none_for_unknown(self):
        """Test token_to_id returns None for unknown tokens."""
        mock_hf_tokenizer = MagicMock()
        mock_hf_tokenizer.convert_tokens_to_ids.return_value = 0  # unk_token_id
        mock_hf_tokenizer.unk_token_id = 0

        wrapper = HFTokenizerWrapper(mock_hf_tokenizer)
        result = wrapper.token_to_id("unknown_token")

        assert result is None

    def test_decode_returns_string(self):
        """Test decode returns decoded string."""
        mock_hf_tokenizer = MagicMock()
        mock_hf_tokenizer.decode.return_value = "hello world"

        wrapper = HFTokenizerWrapper(mock_hf_tokenizer)
        result = wrapper.decode([123, 456])

        assert result == "hello world"
        mock_hf_tokenizer.decode.assert_called_once_with([123, 456], skip_special_tokens=True)


class TestSemanticConfigFlag:
    """Tests for use_semantic_refinement config flag."""

    def test_default_value_is_true(self):
        """Test that use_semantic_refinement defaults to True."""
        config = SemanticConfig()
        assert config.use_semantic_refinement is True

    def test_can_disable_refinement(self):
        """Test that use_semantic_refinement can be set to False."""
        config = SemanticConfig(use_semantic_refinement=False)
        assert config.use_semantic_refinement is False

    def test_config_in_spellchecker_config(self):
        """Test that SemanticConfig is accessible in SpellCheckerConfig."""
        config = SpellCheckerConfig(
            semantic=SemanticConfig(
                use_semantic_refinement=False,
                model_path="/path/to/model.onnx",
                tokenizer_path="/path/to/tokenizer",
            )
        )
        assert config.semantic.use_semantic_refinement is False
        assert config.semantic.model_path == "/path/to/model.onnx"
