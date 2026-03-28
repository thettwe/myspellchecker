"""Tests for suggestion_pipeline.py — suggestion reconstruction and reranking."""

from unittest.mock import MagicMock

from myspellchecker.core.response import Error
from myspellchecker.core.suggestion_pipeline import SuggestionPipelineMixin

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_error(
    text="ကို",
    position=0,
    suggestions=None,
    error_type="invalid_syllable",
    confidence=0.9,
):
    return Error(
        text=text,
        position=position,
        suggestions=suggestions or [],
        error_type=error_type,
        confidence=confidence,
    )


def _make_mixin(**overrides):
    """Create a minimal SuggestionPipelineMixin instance with mocked deps."""
    mixin = object.__new__(SuggestionPipelineMixin)
    provider = MagicMock()
    provider.is_valid_word.return_value = False
    mixin.provider = overrides.get("provider", provider)
    mixin.logger = MagicMock()
    return mixin


# ---------------------------------------------------------------------------
# _extract_adjacent_chars
# ---------------------------------------------------------------------------


class TestExtractAdjacentChars:
    def test_backward_extracts_chars_up_to_space(self):
        result = SuggestionPipelineMixin._extract_adjacent_chars(
            "ကြောင် မြန်", start=7, end=10, direction="backward"
        )
        assert result == ""  # space at position 6

    def test_forward_extracts_chars_up_to_delimiter(self):
        result = SuggestionPipelineMixin._extract_adjacent_chars(
            "ကို ပါ", start=0, end=3, direction="forward"
        )
        assert result == ""  # space after ကို

    def test_backward_extracts_full_prefix_without_delimiter(self):
        text = "ပြောကို"
        # "ကို" starts at some position, get backward chars
        result = SuggestionPipelineMixin._extract_adjacent_chars(
            text, start=4, end=7, direction="backward"
        )
        # Should return everything before position 4
        assert len(result) > 0

    def test_forward_extracts_full_suffix_without_delimiter(self):
        text = "ကိုပါ"
        result = SuggestionPipelineMixin._extract_adjacent_chars(
            text, start=0, end=3, direction="forward"
        )
        assert len(result) > 0


# ---------------------------------------------------------------------------
# _reconstruct_compound_suggestions
# ---------------------------------------------------------------------------


class TestReconstructCompoundSuggestions:
    def test_forward_extension_appends_valid_compound(self):
        provider = MagicMock()
        # First call for the original suggestion check, then True for compound
        provider.is_valid_word.side_effect = lambda w: w == "မြန်မာ"
        mixin = _make_mixin(provider=provider)
        e = _make_error(
            text="မြန်",
            position=0,
            suggestions=["မြန်"],
            error_type="invalid_syllable",
        )
        text = "မြန်မာ"
        mixin._reconstruct_compound_suggestions(text, [e])
        # Should have appended "မြန်မာ" to suggestions
        assert any("မာ" in s for s in e.suggestions)

    def test_no_modification_without_provider(self):
        mixin = _make_mixin(provider=None)
        e = _make_error(suggestions=["ခို"])
        original_sugs = list(e.suggestions)
        mixin._reconstruct_compound_suggestions("ကို", [e])
        assert e.suggestions == original_sugs

    def test_skips_errors_with_no_suggestions(self):
        mixin = _make_mixin()
        e = _make_error(suggestions=[])
        mixin._reconstruct_compound_suggestions("ကိုပါ", [e])
        assert e.suggestions == []

    def test_skips_long_error_text(self):
        mixin = _make_mixin()
        long_text = "ကိုယ့်ဘာသာကိုယ်"
        e = _make_error(text=long_text, suggestions=["test"])
        mixin._reconstruct_compound_suggestions(long_text, [e])
        # Should not add compounds for spans > 6 chars
        assert e.suggestions == ["test"]


# ---------------------------------------------------------------------------
# _asat_try_insertions (static)
# ---------------------------------------------------------------------------


class TestAsatTryInsertions:
    def test_inserts_asat_after_consonant_when_valid(self):
        provider = MagicMock()
        provider.is_valid_word.side_effect = lambda w: w == "က်"
        existing = set()
        results = SuggestionPipelineMixin._asat_try_insertions(
            span="က",
            consonant_range=range(0x1000, 0x1022),
            asat="\u103a",
            visarga="\u1038",
            existing=existing,
            provider=provider,
        )
        assert len(results) >= 1
        assert any("\u103a" in r for r in results)

    def test_skips_when_asat_already_present(self):
        provider = MagicMock()
        provider.is_valid_word.return_value = True
        existing = set()
        results = SuggestionPipelineMixin._asat_try_insertions(
            span="က်",
            consonant_range=range(0x1000, 0x1022),
            asat="\u103a",
            visarga="\u1038",
            existing=existing,
            provider=provider,
        )
        # Should skip insertion after က since it's already followed by asat
        assert all("\u103a\u103a" not in r for r in results)


# ---------------------------------------------------------------------------
# _virama_asat_swap_candidates (static)
# ---------------------------------------------------------------------------


class TestViramaAsatSwapCandidates:
    def test_swaps_virama_to_asat_when_valid(self):
        provider = MagicMock()
        provider.is_valid_word.side_effect = lambda w: "\u103a" in w
        existing = set()
        # Input with virama (္)
        span = "ဒ\u1039ဓ"
        results = SuggestionPipelineMixin._virama_asat_swap_candidates(span, existing, provider)
        assert len(results) >= 1

    def test_returns_empty_when_no_virama_or_asat(self):
        provider = MagicMock()
        existing = set()
        results = SuggestionPipelineMixin._virama_asat_swap_candidates("ကို", existing, provider)
        assert results == []


# ---------------------------------------------------------------------------
# _extend_suggestions_with_sentence_context
# ---------------------------------------------------------------------------


class TestExtendSuggestionsWithSentenceContext:
    def test_extends_morpheme_suggestion_to_valid_compound(self):
        provider = MagicMock()
        provider.is_valid_word.side_effect = lambda w: w == "ဘဏ္ဍာရေး"
        mixin = _make_mixin(provider=provider)
        e = _make_error(
            text="ဘဏ္ာ",
            position=0,
            suggestions=["ဘဏ္ဍာ"],
        )
        sentence = "ဘဏ္ာရေး"
        mixin._extend_suggestions_with_sentence_context([e], sentence)
        assert "ဘဏ္ဍာရေး" in e.suggestions

    def test_no_extension_at_end_of_sentence(self):
        mixin = _make_mixin()
        e = _make_error(text="ကို", position=0, suggestions=["ခို"])
        mixin._extend_suggestions_with_sentence_context([e], "ကို")
        # No trailing chars, no extension
        assert e.suggestions == ["ခို"]

    def test_no_crash_without_provider(self):
        mixin = _make_mixin(provider=None)
        e = _make_error(suggestions=["ခို"])
        mixin._extend_suggestions_with_sentence_context([e], "ကိုပါ")
        assert e.suggestions == ["ခို"]
