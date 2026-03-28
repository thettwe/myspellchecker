"""Pre-computed space-delimited tokenization for detector methods.

Replaces the duplicated ``text.split()`` + ``text.find(token, cursor)``
pattern used across ~25 detector methods.  Immutable and computed once
per ``check()`` call, then passed to all detectors.

This is intentionally NOT a word segmenter -- it uses simple space-splitting
because that is what the detector methods already do.  Word segmentation
(Viterbi-based) is a separate concern handled by ``self.segmenter``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True, slots=True)
class TokenSpan:
    """A single space-delimited token with its position in the source text."""

    text: str
    position: int

    @property
    def end(self) -> int:
        """End position (exclusive) of this token in the source text."""
        return self.position + len(self.text)

    def __len__(self) -> int:
        return len(self.text)


@dataclass(frozen=True, slots=True)
class TokenizedText:
    """Pre-computed tokenization of a text string.

    Built once via ``from_text()``, then shared across all detector
    methods that need token-by-token iteration with positions.

    Immutable (frozen) to prevent accidental mutation.  The ``tokens``
    and ``positions`` properties provide backward-compatible list access
    for gradual migration of existing detector code.
    """

    raw: str
    spans: tuple[TokenSpan, ...]

    @classmethod
    def from_text(cls, text: str) -> TokenizedText:
        """Tokenize by space-splitting, computing positions once.

        Replicates the exact behavior of the duplicated pattern::

            tokens = text.split()
            cursor = 0
            for token in tokens:
                pos = text.find(token, cursor)
                ...
                cursor = pos + len(token)
        """
        result: list[TokenSpan] = []
        cursor = 0
        for token in text.split():
            pos = text.find(token, cursor)
            result.append(TokenSpan(text=token, position=pos))
            cursor = pos + len(token)
        return cls(raw=text, spans=tuple(result))

    def __len__(self) -> int:
        return len(self.spans)

    def __getitem__(self, index: int) -> TokenSpan:
        return self.spans[index]

    def __iter__(self) -> Iterator[TokenSpan]:
        return iter(self.spans)

    def __bool__(self) -> bool:
        return len(self.spans) > 0

    @property
    def tokens(self) -> list[str]:
        """Backward-compat: list of token text strings.

        Equivalent to ``text.split()`` but pre-computed.
        """
        return [s.text for s in self.spans]

    @property
    def positions(self) -> list[int]:
        """Backward-compat: list of token start positions.

        Equivalent to the ``text.find(token, cursor)`` loop.
        """
        return [s.position for s in self.spans]

    def window_before(self, index: int, size: int = 5) -> tuple[TokenSpan, ...]:
        """Return up to ``size`` token spans before ``index``."""
        start = max(0, index - size)
        return self.spans[start:index]

    def window_after(self, index: int, size: int = 5) -> tuple[TokenSpan, ...]:
        """Return up to ``size`` token spans after ``index``."""
        return self.spans[index + 1 : index + 1 + size]

    def prev(self, index: int) -> TokenSpan | None:
        """Return the token span before ``index``, or None."""
        return self.spans[index - 1] if index > 0 else None

    def next(self, index: int) -> TokenSpan | None:
        """Return the token span after ``index``, or None."""
        return self.spans[index + 1] if index + 1 < len(self.spans) else None
