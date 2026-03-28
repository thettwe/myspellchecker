"""
Unit tests for Response and Error classes.

Tests the core data structures used for spell checking results.
"""

import json
import threading
from typing import Any, Dict

from myspellchecker.core.constants import ErrorType
from myspellchecker.core.i18n import set_language
from myspellchecker.core.response import (
    ContextError,
    Error,
    GrammarError,
    Response,
    SyllableError,
    WordError,
)


class TestError:
    """Test suite for the base Error class."""

    def test_error_creation(self) -> None:
        """Test basic error object creation."""
        error = Error(
            text="မျန်",
            position=0,
            error_type="test_error",
            suggestions=["မြန်", "မျန်း"],
            confidence=0.95,
        )

        assert error.text == "မျန်"
        assert error.position == 0
        assert error.error_type == "test_error"
        assert error.suggestions == ["မြန်", "မျန်း"]
        assert error.confidence == 0.95

    def test_error_to_dict(self) -> None:
        """Test error conversion to dictionary."""
        error = Error(
            text="မျန်",
            position=5,
            error_type="test",
            suggestions=["မြန်"],
            confidence=1.0,
        )

        result: Dict[str, Any] = error.to_dict()

        assert isinstance(result, dict)
        assert result["text"] == "မျန်"
        assert result["position"] == 5
        assert result["error_type"] == "test"
        assert result["suggestions"] == ["မြန်"]
        assert result["confidence"] == 1.0
        assert "message" in result
        assert "action" in result

    def test_error_default_confidence(self) -> None:
        """Test that confidence defaults to 1.0."""
        error = Error(text="test", position=0, error_type="test", suggestions=[])

        assert error.confidence == 1.0


class TestSyllableError:
    """Test suite for SyllableError class."""

    def test_syllable_error_creation(self) -> None:
        """Test SyllableError object creation."""
        error = SyllableError(
            text="မျန်",
            position=0,
            suggestions=["မြန်", "မျန်း"],
            confidence=1.0,
        )

        assert error.text == "မျန်"
        assert error.error_type == "invalid_syllable"
        assert error.suggestions == ["မြန်", "မျန်း"]

    def test_syllable_error_to_dict(self) -> None:
        """Test SyllableError dictionary conversion."""
        error = SyllableError(text="မျန်", position=0, suggestions=["မြန်"])

        result = error.to_dict()

        assert result["error_type"] == "invalid_syllable"
        assert result["text"] == "မျန်"


class TestWordError:
    """Test suite for WordError class."""

    def test_word_error_creation(self) -> None:
        """Test WordError object creation."""
        error = WordError(
            text="မြန်စာ",
            position=0,
            suggestions=["မြနမ်ာ"],
            syllable_count=2,
        )
        assert error.text == "မြန်စာ"
        assert error.position == 0
        assert error.error_type == ErrorType.WORD.value
        assert error.suggestions == ["မြနမ်ာ"]
        assert error.confidence == 1.0
        assert error.syllable_count == 2

    def test_word_error_default_syllable_count(self) -> None:
        """Test that syllable_count defaults to 0."""
        error = WordError(text="test", position=0, suggestions=[])
        assert error.syllable_count == 0
        assert error.error_type == ErrorType.WORD.value


class TestContextError:
    """Test suite for ContextError class."""

    def test_context_error_creation(self) -> None:
        """Test ContextError object creation."""
        error = ContextError(
            text="ဘယ်",
            position=10,
            error_type="context_error",
            suggestions=["သွား"],
            probability=0.001,
            prev_word="သူ",
        )

        assert error.text == "ဘယ်"
        assert error.error_type == "context_error"
        assert error.probability == 0.001
        assert error.prev_word == "သူ"

    def test_context_error_defaults(self) -> None:
        """Test ContextError default values."""
        error = ContextError(text="test", position=0, error_type="context_error", suggestions=[])

        assert error.probability == 0.0
        assert error.prev_word == ""


class TestResponse:
    """Test suite for Response class."""

    def test_response_creation(self) -> None:
        """Test basic Response object creation."""
        errors = [SyllableError(text="မျန်", position=0, suggestions=["မြန်"])]

        response = Response(
            text="မျနမ်ာ",
            corrected_text="မြနမ်ာ",
            has_errors=True,
            level="syllable",
            errors=errors,
            metadata={"processing_time_ms": 15.2},
        )

        assert response.text == "မျနမ်ာ"
        assert response.corrected_text == "မြနမ်ာ"
        assert response.has_errors is True
        assert response.level == "syllable"
        assert len(response.errors) == 1
        assert response.metadata["processing_time_ms"] == 15.2

    def test_response_no_errors(self) -> None:
        """Test Response with no errors."""
        response = Response(
            text="မြနမ်ာ",
            corrected_text="မြနမ်ာ",
            has_errors=False,
            level="syllable",
            errors=[],
            metadata={},
        )

        assert response.has_errors is False
        assert len(response.errors) == 0

    def test_response_to_dict(self) -> None:
        """Test Response conversion to dictionary."""
        errors = [SyllableError(text="မျန်", position=0, suggestions=["မြန်"])]

        response = Response(
            text="မျနမ်ာ",
            corrected_text="မြနမ်ာ",
            has_errors=True,
            level="syllable",
            errors=errors,
            metadata={"count": 1},
        )

        result = response.to_dict()

        assert isinstance(result, dict)
        assert result["text"] == "မျနမ်ာ"
        assert result["has_errors"] is True
        assert result["level"] == "syllable"
        assert len(result["errors"]) == 1
        assert isinstance(result["errors"][0], dict)
        assert result["errors"][0]["error_type"] == "invalid_syllable"

    def test_response_to_json(self) -> None:
        """Test Response conversion to JSON string."""
        errors = [SyllableError(text="မျန်", position=0, suggestions=["မြန်"])]

        response = Response(
            text="မျနမ်ာ",
            corrected_text="မြနမ်ာ",
            has_errors=True,
            level="syllable",
            errors=errors,
            metadata={},
        )

        json_str = response.to_json()

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)
        assert parsed["text"] == "မျနမ်ာ"
        assert parsed["has_errors"] is True

    def test_response_to_json_preserves_unicode(self) -> None:
        """Test that to_json preserves Myanmar Unicode characters."""
        response = Response(
            text="မြနမ်ာ",
            corrected_text="မြနမ်ာ",
            has_errors=False,
            level="syllable",
            errors=[],
            metadata={},
        )

        json_str = response.to_json()

        # Myanmar characters should be preserved, not escaped
        assert "မြနမ်ာ" in json_str
        assert "\\u" not in json_str  # No Unicode escapes

    def test_response_to_json_compact(self) -> None:
        """Test Response conversion to compact JSON."""
        response = Response(
            text="test",
            corrected_text="test",
            has_errors=False,
            level="syllable",
            errors=[],
            metadata={},
        )

        json_str = response.to_json(indent=None)

        # Should be compact (no newlines from indentation)
        assert "\n" not in json_str

    def test_response_str_representation(self) -> None:
        """Test Response string representation."""
        response = Response(
            text="မြနမ်ာနိုငံသည်အရှေ့တောင်အာရှတွင်ရှိသည်",
            corrected_text="မြနမ်ာနိုငံသည်အရှေ့တောင်အာရှတွင်ရှိသည်",
            has_errors=False,
            level="syllable",
            errors=[],
            metadata={},
        )

        str_repr = str(response)

        assert "Response" in str_repr
        assert "has_errors=False" in str_repr
        assert "level='syllable'" in str_repr
        assert "error_count=0" in str_repr

    def test_response_with_multiple_errors(self) -> None:
        """Test Response with multiple errors of different types."""
        errors = [
            SyllableError(text="မျန်", position=0, suggestions=["မြန်"]),
            WordError(
                text="မြန်စာ",
                position=10,
                suggestions=["မြနမ်ာ"],
                syllable_count=2,
            ),
            ContextError(
                text="ဘယ်",
                position=20,
                error_type="context_error",
                suggestions=["သွား"],
                prev_word="သူ",
            ),
        ]

        response = Response(
            text="original",
            corrected_text="corrected",
            has_errors=True,
            level="word",
            errors=errors,
            metadata={},
        )

        assert len(response.errors) == 3
        assert isinstance(response.errors[0], SyllableError)
        assert isinstance(response.errors[1], WordError)
        assert isinstance(response.errors[2], ContextError)

        # Convert to dict and verify all errors preserved
        result = response.to_dict()
        assert len(result["errors"]) == 3
        assert result["errors"][0]["error_type"] == "invalid_syllable"
        assert result["errors"][1]["error_type"] == "invalid_word"
        assert result["errors"][2]["error_type"] == "context_error"


class TestErrorMessage:
    """Test suite for Error.message property and i18n integration."""

    def setup_method(self) -> None:
        """Reset language to English before each test."""
        set_language("en")

    def test_syllable_error_message_english(self) -> None:
        """Test SyllableError.message returns English text."""
        error = SyllableError(text="မျန်", position=0, suggestions=["မြန်"])
        assert error.message == "Invalid syllable"

    def test_word_error_message_english(self) -> None:
        """Test WordError.message returns English text."""
        error = WordError(text="test", position=0, suggestions=[])
        assert error.message == "Invalid word"

    def test_context_error_message_english(self) -> None:
        """Test ContextError.message returns English text."""
        error = ContextError(text="test", position=0, suggestions=[])
        assert error.message == "Unlikely word sequence"

    def test_grammar_error_message_english(self) -> None:
        """Test GrammarError.message returns English text."""
        error = GrammarError(text="test", position=0, suggestions=[])
        assert error.message == "Grammar error"

    def test_message_myanmar_language(self) -> None:
        """Test Error.message returns Myanmar text when language is set to 'my'."""
        set_language("my")
        error = SyllableError(text="မျန်", position=0, suggestions=[])
        assert error.message == "စာလုံးပေါင်း မမှန်ကန်ပါ"

    def test_message_unknown_error_type_returns_key(self) -> None:
        """Test that unknown error types return the key string as fallback."""
        error = Error(text="x", position=0, error_type="unknown_type", suggestions=[])
        assert error.message == "unknown_type"

    def test_to_dict_includes_message(self) -> None:
        """Test that to_dict() output includes the message field."""
        error = SyllableError(text="test", position=0, suggestions=[])
        d = error.to_dict()
        assert "message" in d
        assert d["message"] == "Invalid syllable"

    def test_word_error_to_dict_includes_message(self) -> None:
        """Test that WordError.to_dict() includes message via _add_action."""
        error = WordError(text="test", position=0, suggestions=[], syllable_count=1)
        d = error.to_dict()
        assert "message" in d
        assert d["message"] == "Invalid word"

    def test_context_error_to_dict_includes_message(self) -> None:
        """Test that ContextError.to_dict() includes message via _add_action."""
        error = ContextError(text="test", position=0, suggestions=[])
        d = error.to_dict()
        assert "message" in d

    def test_message_thread_safety(self) -> None:
        """Test that i18n language is thread-local."""
        results: dict[str, str] = {}
        barrier = threading.Barrier(2)

        def check_english() -> None:
            set_language("en")
            barrier.wait()
            error = SyllableError(text="x", position=0, suggestions=[])
            results["en"] = error.message

        def check_myanmar() -> None:
            set_language("my")
            barrier.wait()
            error = SyllableError(text="x", position=0, suggestions=[])
            results["my"] = error.message

        t1 = threading.Thread(target=check_english)
        t2 = threading.Thread(target=check_myanmar)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert results["en"] == "Invalid syllable"
        assert results["my"] == "စာလုံးပေါင်း မမှန်ကန်ပါ"

    def test_all_error_types_have_messages(self) -> None:
        """Test that all ErrorType values have i18n messages (not just fallback)."""
        set_language("en")
        for et in ErrorType:
            error = Error(text="x", position=0, error_type=et.value, suggestions=[])
            # The message should NOT be the raw snake_case key
            assert error.message != et.value, (
                f"ErrorType.{et.name} ({et.value}) has no i18n message"
            )
