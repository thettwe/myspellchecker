"""Tests for TokenizedText and TokenSpan."""

import pytest

from myspellchecker.core.detectors.tokenized_text import TokenizedText, TokenSpan


class TestTokenSpan:
    def test_basic_properties(self):
        span = TokenSpan(text="hello", position=5)
        assert span.text == "hello"
        assert span.position == 5
        assert span.end == 10
        assert len(span) == 5

    def test_empty_text(self):
        span = TokenSpan(text="", position=0)
        assert span.end == 0
        assert len(span) == 0

    def test_immutable(self):
        span = TokenSpan(text="hello", position=0)
        with pytest.raises(AttributeError):
            span.text = "world"  # type: ignore[misc]
        with pytest.raises(AttributeError):
            span.position = 5  # type: ignore[misc]


class TestTokenizedText:
    def test_basic_english(self):
        t = TokenizedText.from_text("abc def ghi")
        assert len(t) == 3
        assert t.tokens == ["abc", "def", "ghi"]
        assert t.positions == [0, 4, 8]
        assert t.raw == "abc def ghi"

    def test_positions_match_manual(self):
        """Verify positions match the exact behavior of the old pattern."""
        text = "abc def ghi"
        t = TokenizedText.from_text(text)

        # Old pattern
        tokens = text.split()
        cursor = 0
        old_positions = []
        for token in tokens:
            pos = text.find(token, cursor)
            old_positions.append(pos)
            cursor = pos + len(token)

        assert t.tokens == tokens
        assert t.positions == old_positions

    def test_multi_space(self):
        """Double spaces should produce correct positions."""
        text = "abc  def"
        t = TokenizedText.from_text(text)
        assert len(t) == 2
        assert t[0].text == "abc"
        assert t[0].position == 0
        assert t[1].text == "def"
        assert t[1].position == 5  # after "abc  "

    def test_leading_trailing_space(self):
        text = "  abc def  "
        t = TokenizedText.from_text(text)
        assert len(t) == 2
        assert t[0].text == "abc"
        assert t[0].position == 2
        assert t[1].text == "def"
        assert t[1].position == 6

    def test_empty_text(self):
        t = TokenizedText.from_text("")
        assert len(t) == 0
        assert t.tokens == []
        assert t.positions == []
        assert not t

    def test_single_token(self):
        t = TokenizedText.from_text("hello")
        assert len(t) == 1
        assert t[0].text == "hello"
        assert t[0].position == 0
        assert t

    def test_myanmar_text(self):
        """Myanmar tokens should get correct byte-aware positions."""
        text = "ကျွန်တော် စာ ဖတ်တယ်"
        t = TokenizedText.from_text(text)
        assert len(t) == 3
        assert t[0].text == "ကျွန်တော်"
        assert t[0].position == 0
        assert t[1].text == "စာ"
        # Position should be len("ကျွန်တော် ") in characters
        assert text[t[1].position : t[1].end] == "စာ"
        assert text[t[2].position : t[2].end] == "ဖတ်တယ်"

    def test_iteration(self):
        t = TokenizedText.from_text("a b c")
        texts = [span.text for span in t]
        assert texts == ["a", "b", "c"]

    def test_getitem(self):
        t = TokenizedText.from_text("a b c")
        assert t[0].text == "a"
        assert t[2].text == "c"
        with pytest.raises(IndexError):
            _ = t[3]

    def test_immutable(self):
        t = TokenizedText.from_text("a b c")
        with pytest.raises(AttributeError):
            t.raw = "x"  # type: ignore[misc]

    def test_window_before(self):
        t = TokenizedText.from_text("a b c d e")
        before = t.window_before(3, size=2)
        assert len(before) == 2
        assert before[0].text == "b"
        assert before[1].text == "c"

    def test_window_before_at_start(self):
        t = TokenizedText.from_text("a b c")
        before = t.window_before(0, size=5)
        assert len(before) == 0

    def test_window_before_clamped(self):
        t = TokenizedText.from_text("a b c")
        before = t.window_before(1, size=5)
        assert len(before) == 1
        assert before[0].text == "a"

    def test_window_after(self):
        t = TokenizedText.from_text("a b c d e")
        after = t.window_after(1, size=2)
        assert len(after) == 2
        assert after[0].text == "c"
        assert after[1].text == "d"

    def test_window_after_at_end(self):
        t = TokenizedText.from_text("a b c")
        after = t.window_after(2, size=5)
        assert len(after) == 0

    def test_prev_next(self):
        t = TokenizedText.from_text("a b c")
        assert t.prev(0) is None
        assert t.prev(1).text == "a"
        assert t.next(1).text == "c"
        assert t.next(2) is None

    def test_backward_compat_tokens_positions(self):
        """The .tokens and .positions properties should match text.split() + find loop."""
        text = "Myanmar   text  with   various  spacing"
        t = TokenizedText.from_text(text)

        manual_tokens = text.split()
        manual_positions = []
        cursor = 0
        for token in manual_tokens:
            pos = text.find(token, cursor)
            manual_positions.append(pos)
            cursor = pos + len(token)

        assert t.tokens == manual_tokens
        assert t.positions == manual_positions
