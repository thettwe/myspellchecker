"""Tests for word-level suggestion lifting in the suggestion pipeline.

When a syllable error is inside a larger token, the system should
reconstruct full-word suggestions by prepending/appending adjacent
valid syllables from the enclosing token.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from myspellchecker.core.response import SyllableError
from myspellchecker.core.suggestion_pipeline import SuggestionPipelineMixin


class FakePipeline(SuggestionPipelineMixin):
    """Minimal concrete class for testing the mixin."""

    def __init__(self, valid_words: set[str] | None = None):
        self.provider = MagicMock()
        self.logger = MagicMock()
        self._semantic_checker = None
        self._symspell = None
        # Configure is_valid_word to check against the provided set
        _valid = valid_words or set()
        self.provider.is_valid_word.side_effect = lambda w: w in _valid

    @property
    def semantic_checker(self):
        return self._semantic_checker

    @property
    def symspell(self):
        return self._symspell


class TestBackwardReconstruction:
    """Test backward (prefix) word-level suggestion lifting."""

    def test_backward_lifting_missing_asat_end_of_word(self):
        """Error at end of word: 'ငး' in 'ကျောငး' -> 'ကျောင်း'."""
        pipeline = FakePipeline(valid_words={"ကျောင်း"})
        # Error 'ငး' at position 4 (ကျော = 4 chars), suggestion 'င်း'
        errors = [
            SyllableError(
                text="ငး",
                position=4,
                suggestions=["င်း"],
                confidence=0.9,
            )
        ]
        text = "ကျောငး"
        pipeline._reconstruct_compound_suggestions(text, errors)

        assert "ကျောင်း" in errors[0].suggestions
        # Lifted suggestion should be at the top
        assert errors[0].suggestions[0] == "ကျောင်း"

    def test_backward_lifting_missing_asat_middle(self):
        """Error in middle of compound: 'ညး' in 'နညးပညာ'."""
        pipeline = FakePipeline(valid_words={"နည်း", "နည်းပညာ"})
        errors = [
            SyllableError(
                text="ညး",
                position=1,
                suggestions=["ည်း"],
                confidence=0.9,
            )
        ]
        text = "နညးပညာ"
        pipeline._reconstruct_compound_suggestions(text, errors)

        # Should find both backward and bidirectional forms
        assert "နည်း" in errors[0].suggestions
        assert errors[0].suggestions[0] in ("နည်း", "နည်းပညာ")

    def test_backward_lifting_preserves_original_suggestions(self):
        """Original morpheme suggestions should still be present."""
        pipeline = FakePipeline(valid_words={"သက်တမ်း"})
        errors = [
            SyllableError(
                text="မး",
                position=4,
                suggestions=["မ်း", "မူး"],
                confidence=0.9,
            )
        ]
        text = "သက်တမး"
        pipeline._reconstruct_compound_suggestions(text, errors)

        assert "သက်တမ်း" in errors[0].suggestions
        assert "မ်း" in errors[0].suggestions
        assert "မူး" in errors[0].suggestions

    def test_no_backward_when_at_token_start(self):
        """No backward lifting when error is at the start of a token."""
        pipeline = FakePipeline(valid_words={"လူကြီး"})
        errors = [
            SyllableError(
                text="လု",
                position=0,
                suggestions=["လူ"],
                confidence=0.9,
            )
        ]
        text = "လုကြီး"
        pipeline._reconstruct_compound_suggestions(text, errors)

        # Forward reconstruction should still work
        assert "လူကြီး" in errors[0].suggestions

    def test_no_backward_across_space(self):
        """Backward lifting should not cross space boundaries."""
        pipeline = FakePipeline(valid_words=set())
        # 'ကျော ငး' -> 'ငး' at position 5 (4 chars + space)
        errors = [
            SyllableError(
                text="ငး",
                position=5,
                suggestions=["င်း"],
                confidence=0.9,
            )
        ]
        text = "ကျော ငး"
        pipeline._reconstruct_compound_suggestions(text, errors)

        # No lifting should happen (space separates tokens)
        assert errors[0].suggestions == ["င်း"]

    def test_backward_lifting_dedup(self):
        """Lifted suggestions should not duplicate existing ones."""
        pipeline = FakePipeline(valid_words={"ကျောင်း"})
        errors = [
            SyllableError(
                text="ငး",
                position=4,
                suggestions=["ကျောင်း", "င်း"],
                confidence=0.9,
            )
        ]
        text = "ကျောငး"
        pipeline._reconstruct_compound_suggestions(text, errors)

        # Should not have duplicates
        count = errors[0].suggestions.count("ကျောင်း")
        assert count == 1


class TestBidirectionalReconstruction:
    """Test bidirectional (prefix + suffix) reconstruction."""

    def test_bidirectional_lifting(self):
        """Error in middle with both prefix and suffix."""
        pipeline = FakePipeline(valid_words={"ပညာရေး"})
        errors = [
            SyllableError(
                text="ညာ",
                position=1,
                suggestions=["ညာ့"],
                confidence=0.9,
            )
        ]
        # Error 'ညာ' is in the middle of 'ပညာရေး'
        text = "ပညာရေး"
        pipeline._reconstruct_compound_suggestions(text, errors)

        # Check that original suggestion is preserved
        assert "ညာ့" in errors[0].suggestions


class TestForwardReconstruction:
    """Test that existing forward reconstruction still works."""

    def test_forward_only(self):
        """Forward reconstruction at token start."""
        pipeline = FakePipeline(valid_words={"လူကြီး"})
        errors = [
            SyllableError(
                text="လု",
                position=0,
                suggestions=["လူ"],
                confidence=0.9,
            )
        ]
        text = "လုကြီး"
        pipeline._reconstruct_compound_suggestions(text, errors)

        assert "လူကြီး" in errors[0].suggestions
        # Forward suggestions come after existing
        assert errors[0].suggestions[0] == "လူ"

    def test_no_suggestions_skipped(self):
        """Errors with no suggestions should be skipped."""
        pipeline = FakePipeline(valid_words={"ကျောင်း"})
        errors = [
            SyllableError(
                text="ငး",
                position=4,
                suggestions=[],
                confidence=0.9,
            )
        ]
        text = "ကျောငး"
        pipeline._reconstruct_compound_suggestions(text, errors)

        assert errors[0].suggestions == []

    def test_long_error_text_skipped(self):
        """Errors with text longer than 6 chars should be skipped."""
        pipeline = FakePipeline(valid_words={"ကျောင်းသားများ"})
        errors = [
            SyllableError(
                text="ကျောငးသားများ",
                position=0,
                suggestions=["ကျောင်းသားများ"],
                confidence=0.9,
            )
        ]
        text = "ကျောငးသားများ"
        pipeline._reconstruct_compound_suggestions(text, errors)

        # Should not attempt reconstruction for long error text
        assert errors[0].suggestions == ["ကျောင်းသားများ"]


class TestEdgeCases:
    """Test edge cases for the reconstruction logic."""

    def test_empty_errors_list(self):
        """Empty errors list should not crash."""
        pipeline = FakePipeline()
        pipeline._reconstruct_compound_suggestions("some text", [])

    def test_no_provider(self):
        """No provider should skip gracefully."""
        pipeline = FakePipeline()
        pipeline.provider = None
        errors = [
            SyllableError(
                text="ငး",
                position=4,
                suggestions=["င်း"],
                confidence=0.9,
            )
        ]
        pipeline._reconstruct_compound_suggestions("ကျောငး", errors)
        assert errors[0].suggestions == ["င်း"]

    def test_error_at_end_of_text(self):
        """Error at the very end of text (no forward chars)."""
        pipeline = FakePipeline(valid_words={"ကျောင်း"})
        errors = [
            SyllableError(
                text="ငး",
                position=4,
                suggestions=["င်း"],
                confidence=0.9,
            )
        ]
        text = "ကျောငး"
        pipeline._reconstruct_compound_suggestions(text, errors)

        assert "ကျောင်း" in errors[0].suggestions
        assert errors[0].suggestions[0] == "ကျောင်း"

    def test_error_at_start_of_text(self):
        """Error at the very start (no backward chars)."""
        pipeline = FakePipeline(valid_words={"င်းကျော"})
        errors = [
            SyllableError(
                text="ငး",
                position=0,
                suggestions=["င်း"],
                confidence=0.9,
            )
        ]
        text = "ငးကျော"
        pipeline._reconstruct_compound_suggestions(text, errors)

        # Only forward should work
        assert "င်းကျော" in errors[0].suggestions

    def test_multiple_errors_in_same_token(self):
        """Multiple errors should each get their own lifting."""
        pipeline = FakePipeline(valid_words={"ကျောင်း", "ကျောင်းသား"})
        errors = [
            SyllableError(
                text="ငး",
                position=4,
                suggestions=["င်း"],
                confidence=0.9,
            ),
            SyllableError(
                text="သာ",
                position=6,
                suggestions=["သား"],
                confidence=0.9,
            ),
        ]
        text = "ကျောငးသာ"
        pipeline._reconstruct_compound_suggestions(text, errors)

        assert "ကျောင်း" in errors[0].suggestions

    def test_punctuation_delimiter(self):
        """Backward lifting should stop at Myanmar punctuation."""
        pipeline = FakePipeline(valid_words=set())
        # 'ကျော' = 4 chars, + U+104A = pos 5 for 'ငး'
        errors = [
            SyllableError(
                text="ငး",
                position=5,
                suggestions=["င်း"],
                confidence=0.9,
            )
        ]
        # U+104A is Myanmar sign little section
        text = "ကျော\u104aငး"
        pipeline._reconstruct_compound_suggestions(text, errors)

        # Should not cross punctuation
        assert errors[0].suggestions == ["င်း"]
