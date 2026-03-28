"""Tests for response_builder module."""

from myspellchecker.core.response import ContextError, SyllableError, WordError
from myspellchecker.core.response_builder import build_response_metadata


class TestBuildResponseMetadata:
    """Tests for build_response_metadata function."""

    def test_empty_errors(self) -> None:
        """Should return correct metadata for no errors."""
        metadata = build_response_metadata([], ["syllable"], 0.05)
        assert metadata["total_errors"] == 0
        assert metadata["syllable_errors"] == 0
        assert metadata["word_errors"] == 0
        assert metadata["context_errors"] == 0
        assert metadata["semantic_errors"] == 0
        assert metadata["layers_applied"] == ["syllable"]
        assert metadata["processing_time"] == 0.05
        assert metadata["zawgyi_warning"] is None

    def test_syllable_errors_counted(self) -> None:
        """Should count syllable errors correctly."""
        errors = [
            SyllableError(position=0, text="abc", suggestions=[]),
            SyllableError(position=5, text="def", suggestions=[]),
        ]
        metadata = build_response_metadata(errors, ["syllable"], 0.1)
        assert metadata["total_errors"] == 2
        assert metadata["syllable_errors"] == 2
        assert metadata["word_errors"] == 0

    def test_word_errors_counted(self) -> None:
        """Should count word errors correctly."""
        errors = [
            WordError(position=0, text="abc", suggestions=[]),
        ]
        metadata = build_response_metadata(errors, ["word"], 0.1)
        assert metadata["total_errors"] == 1
        assert metadata["syllable_errors"] == 0
        assert metadata["word_errors"] == 1

    def test_context_errors_counted(self) -> None:
        """Should count context errors correctly."""
        errors = [
            ContextError(position=0, text="abc", suggestions=[], error_type="context_error"),
        ]
        metadata = build_response_metadata(errors, ["context"], 0.1)
        assert metadata["total_errors"] == 1
        assert metadata["context_errors"] == 1
        assert metadata["semantic_errors"] == 0

    def test_semantic_errors_counted(self) -> None:
        """Should count semantic errors correctly."""
        errors = [
            ContextError(position=0, text="abc", suggestions=[], error_type="semantic_error"),
        ]
        metadata = build_response_metadata(errors, ["semantic"], 0.1)
        assert metadata["total_errors"] == 1
        assert metadata["context_errors"] == 0
        assert metadata["semantic_errors"] == 1

    def test_mixed_errors(self) -> None:
        """Should handle mixed error types correctly."""
        errors = [
            SyllableError(position=0, text="a", suggestions=[]),
            WordError(position=2, text="b", suggestions=[]),
            ContextError(position=4, text="c", suggestions=[], error_type="context_error"),
            ContextError(position=6, text="d", suggestions=[], error_type="semantic_error"),
        ]
        metadata = build_response_metadata(errors, ["syllable", "word", "context"], 0.2)
        assert metadata["total_errors"] == 4
        assert metadata["syllable_errors"] == 1
        assert metadata["word_errors"] == 1
        assert metadata["context_errors"] == 1
        assert metadata["semantic_errors"] == 1

    def test_zawgyi_warning_included(self) -> None:
        """Should include Zawgyi warning when provided."""

        class MockZawgyiWarning:
            message = "Zawgyi detected"
            confidence = 0.95
            suggestion = "Convert to Unicode"

        warning = MockZawgyiWarning()
        metadata = build_response_metadata([], [], 0.05, zawgyi_warning=warning)
        assert metadata["zawgyi_warning"] is not None
        assert metadata["zawgyi_warning"]["message"] == "Zawgyi detected"
        assert metadata["zawgyi_warning"]["confidence"] == 0.95
        assert metadata["zawgyi_warning"]["suggestion"] == "Convert to Unicode"

    def test_layers_applied_preserved(self) -> None:
        """Should preserve layers_applied list."""
        layers = ["syllable", "word", "context", "semantic"]
        metadata = build_response_metadata([], layers, 0.1)
        assert metadata["layers_applied"] == layers
