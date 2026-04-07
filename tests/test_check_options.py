"""
Unit tests for per-request CheckOptions.
"""

import asyncio

import pytest

from myspellchecker import CheckOptions, SpellChecker
from myspellchecker.core.config import SpellCheckerConfig
from myspellchecker.core.constants import ValidationLevel
from myspellchecker.core.response import GrammarError
from myspellchecker.providers import MemoryProvider


@pytest.fixture()
def checker():
    """Create a SpellChecker with an empty MemoryProvider for fast tests."""
    provider = MemoryProvider()
    config = SpellCheckerConfig(provider=provider)
    return SpellChecker(config=config)


class TestCheckOptionsNone:
    """Verify that options=None preserves default behaviour."""

    def test_options_none_is_default(self, checker):
        """check() with options=None should behave identically to no options."""
        text = "မြန်မာ"
        result_default = checker.check(text)
        result_none = checker.check(text, options=None)

        assert result_default.has_errors == result_none.has_errors
        assert result_default.level == result_none.level

    def test_default_check_options_fields_are_none(self):
        """All CheckOptions fields should default to None."""
        opts = CheckOptions()
        assert opts.context_checking is None
        assert opts.grammar_checking is None
        assert opts.max_suggestions is None
        assert opts.use_semantic is None


class TestMaxSuggestions:
    """Verify that max_suggestions limits suggestion lists."""

    def test_max_suggestions_limits_results(self, checker):
        """Suggestions should be truncated to max_suggestions."""
        text = "မြန်မာ"
        opts = CheckOptions(max_suggestions=1)
        result = checker.check(text, options=opts)
        for error in result.errors:
            assert len(error.suggestions) <= 1

    def test_max_suggestions_zero(self, checker):
        """max_suggestions=0 should empty all suggestion lists."""
        text = "မြန်မာ"
        opts = CheckOptions(max_suggestions=0)
        result = checker.check(text, options=opts)
        for error in result.errors:
            assert error.suggestions == []


class TestUseSemantic:
    """Verify that use_semantic in options takes precedence."""

    def test_use_semantic_override_from_options(self, checker):
        """options.use_semantic should override the use_semantic parameter."""
        text = "မြန်မာ"
        # Explicitly pass use_semantic=True at parameter level,
        # but override to False via options.
        opts = CheckOptions(use_semantic=False)
        result = checker.check(text, use_semantic=True, options=opts)
        # Semantic layer should not be in layers_applied because
        # options.use_semantic=False takes precedence.
        layers = result.metadata.get("layers_applied", [])
        assert "semantic" not in layers

    def test_use_semantic_none_in_options_falls_through(self, checker):
        """options.use_semantic=None should not override the parameter."""
        text = "မြန်မာ"
        opts = CheckOptions(use_semantic=None)
        result_param = checker.check(text, use_semantic=False)
        result_opts = checker.check(text, use_semantic=False, options=opts)
        assert result_param.has_errors == result_opts.has_errors


class TestContextChecking:
    """Verify that context_checking=False skips context validation."""

    def test_context_checking_disabled(self, checker):
        """context_checking=False should skip context layer."""
        text = "မြန်မာ"
        opts = CheckOptions(context_checking=False)
        result = checker.check(text, level=ValidationLevel.WORD, options=opts)
        layers = result.metadata.get("layers_applied", [])
        assert "context" not in layers

    def test_context_checking_true_allows_context(self, checker):
        """context_checking=True should allow context layer (default)."""
        text = "မြန်မာ"
        opts = CheckOptions(context_checking=True)
        result = checker.check(text, level=ValidationLevel.WORD, options=opts)
        # context layer should still run at WORD level
        layers = result.metadata.get("layers_applied", [])
        assert "context" in layers


class TestGrammarChecking:
    """Verify that grammar_checking=False filters grammar errors."""

    def test_grammar_checking_disabled_filters_grammar_errors(self, checker):
        """grammar_checking=False should remove GrammarError instances."""
        text = "မြန်မာ"
        opts = CheckOptions(grammar_checking=False)
        result = checker.check(text, options=opts)
        for error in result.errors:
            assert not isinstance(error, GrammarError)


class TestCheckOptionsBatchMethods:
    """Verify that batch methods forward options correctly."""

    def test_check_batch_with_options(self, checker):
        """check_batch should forward options to each check() call."""
        texts = ["မြန်မာ", "မြန်မာ"]
        opts = CheckOptions(max_suggestions=1)
        results = checker.check_batch(texts, options=opts)
        assert len(results) == 2
        for result in results:
            for error in result.errors:
                assert len(error.suggestions) <= 1

    def test_check_async_signature_accepts_options(self):
        """check_async signature accepts options parameter."""
        import inspect

        sig = inspect.signature(SpellChecker.check_async)
        assert "options" in sig.parameters

    def test_check_batch_async_signature_accepts_options(self):
        """check_batch_async signature accepts options parameter."""
        import inspect

        sig = inspect.signature(SpellChecker.check_batch_async)
        assert "options" in sig.parameters
