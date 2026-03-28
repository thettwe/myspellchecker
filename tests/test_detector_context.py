"""Tests for DetectorContext."""

from unittest.mock import MagicMock

import pytest

from myspellchecker.core.detectors.context import DetectorContext
from myspellchecker.core.detectors.tokenized_text import TokenizedText


class TestDetectorContext:
    def test_construction(self):
        tokenized = TokenizedText.from_text("test text")
        ctx = DetectorContext(
            provider=MagicMock(),
            segmenter=MagicMock(),
            symspell=None,
            semantic_checker=None,
            config=MagicMock(),
            tokenized=tokenized,
        )
        assert ctx.provider is not None
        assert ctx.symspell is None
        assert ctx.semantic_checker is None
        assert len(ctx.tokenized) == 2

    def test_frozen(self):
        tokenized = TokenizedText.from_text("test")
        ctx = DetectorContext(
            provider=MagicMock(),
            segmenter=MagicMock(),
            symspell=None,
            semantic_checker=None,
            config=MagicMock(),
            tokenized=tokenized,
        )
        with pytest.raises(AttributeError):
            ctx.provider = MagicMock()  # type: ignore[misc]

    def test_tokenized_access(self):
        tokenized = TokenizedText.from_text("ကျွန်တော် စာဖတ်တယ်")
        ctx = DetectorContext(
            provider=MagicMock(),
            segmenter=MagicMock(),
            symspell=MagicMock(),
            semantic_checker=MagicMock(),
            config=MagicMock(),
            tokenized=tokenized,
        )
        assert ctx.tokenized.tokens == ["ကျွန်တော်", "စာဖတ်တယ်"]
        assert ctx.tokenized[0].position == 0
