"""Fixture tests for the cython top-K Viterbi enabler (cvt-01 of cython-viterbi-topk).

These tests are the spec for cvt-02 (`viterbi_topk` implementation). They are
EXPECTED TO FAIL until cvt-02 lands — the import of `viterbi_topk` inside each
test will raise ImportError, which pytest surfaces as a test failure (not a
collection error).

API under test:
    viterbi_topk(text: str, K: int, prev: str = "<S>", maxlen: int = 20)
        -> list[tuple[float, list[str]]]

    Returns up to K (score, words) tuples, sorted desc by score. K=1 must
    match the existing `viterbi(text)` byte-identically (full parity is
    enforced by cvt-04 on the 200-sentence clean slice; the fixtures here
    enforce it on small, hand-selected chunks).
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

# Real mmap (bypasses the session-scoped mock which substitutes a regex splitter).
# The mmap is cached on any dev machine that has ever built a benchmark; CI
# would need to re-cache. If the cached file is absent, the whole module
# skips — this is preferable to silently running against a 100-byte dummy.
_REAL_MMAP = Path.home() / ".cache" / "myspellchecker" / "resources" / "segmentation.mmap"


pytestmark = pytest.mark.skipif(
    not _REAL_MMAP.exists() or _REAL_MMAP.stat().st_size < 1_000_000,
    reason=(
        f"Real segmentation mmap not cached at {_REAL_MMAP}. "
        "Run `myspellchecker build --sample` or any benchmark to populate it."
    ),
)


@pytest.fixture(scope="module", autouse=True)
def _init_real_mmap():
    """Initialize the Cython word_segment module against the REAL production mmap.

    The conftest session-scoped fixture replaces `WordTokenizer._init_myword`
    with a dummy syllable splitter, but it does NOT patch the raw Cython
    `initialize_mmap` function. Calling it here wires the Cython module
    against real bigram/unigram tables so scores are meaningful.
    """
    from myspellchecker.tokenizers.cython.word_segment import initialize_mmap

    assert initialize_mmap(str(_REAL_MMAP)), "Failed to initialize real mmap"
    yield


# Fixture chunks — top-1 captured from the existing `viterbi()` on the real
# mmap. These anchors are what K=1 parity asserts against. The `top1_words`
# field is recomputed each test run (not hard-coded) so a future LM retrain
# doesn't break the suite for unrelated reasons; only the structural
# invariants are hard-coded.
FIXTURES = [
    # Single compound word, 2 syllables. Top-1 is the compound itself.
    pytest.param("မြန်မာ", id="compound_2syl_mranma"),
    # Single compound word, 3 syllables.
    pytest.param("မြန်မာစာ", id="compound_3syl_mranma_sa"),
    # Single compound word, different shape (medial + final stack).
    pytest.param("ကျောင်းသား", id="compound_2syl_student"),
]


# Tiny 2-char chunk where we can hand-bound the number of valid segmentations.
# Two distinct Myanmar characters concatenated → exactly two valid
# segmentations: ["ကခ"] and ["က", "ခ"]. With K=5 the result must be length 2.
TWO_CHAR_CHUNK = "ကခ"


class TestTopKParity:
    """K=1 parity: viterbi_topk(text, K=1) must equal viterbi(text) exactly."""

    @pytest.mark.parametrize("text", [f.values[0] for f in FIXTURES])
    def test_k1_matches_viterbi_exactly(self, text: str) -> None:
        from myspellchecker.tokenizers.cython.word_segment import (
            viterbi,
            viterbi_topk,
        )

        topk = viterbi_topk(text, 1)
        baseline = viterbi(text)

        assert len(topk) == 1, f"K=1 must yield exactly 1 result, got {len(topk)}"
        top_score, top_words = topk[0]
        base_score, base_words = baseline

        assert top_words == base_words, (
            f"K=1 words diverge from viterbi(): {top_words!r} vs {base_words!r}"
        )
        # Scores should be bit-identical; tolerate 0 epsilon since DP is deterministic.
        assert top_score == base_score, (
            f"K=1 score diverges from viterbi(): {top_score!r} vs {base_score!r}"
        )


class TestTopKStructuralInvariants:
    """Properties that must hold for any K >= 1 on any non-empty input."""

    @pytest.mark.parametrize("text", [f.values[0] for f in FIXTURES])
    def test_length_within_bound(self, text: str) -> None:
        from myspellchecker.tokenizers.cython.word_segment import viterbi_topk

        results = viterbi_topk(text, 5)
        assert 1 <= len(results) <= 5, f"len(results)={len(results)} not in [1,5]"

    @pytest.mark.parametrize("text", [f.values[0] for f in FIXTURES])
    def test_scores_are_finite(self, text: str) -> None:
        from myspellchecker.tokenizers.cython.word_segment import viterbi_topk

        results = viterbi_topk(text, 5)
        for score, words in results:
            assert math.isfinite(score), f"non-finite score {score} for {words!r}"

    @pytest.mark.parametrize("text", [f.values[0] for f in FIXTURES])
    def test_scores_non_increasing(self, text: str) -> None:
        from myspellchecker.tokenizers.cython.word_segment import viterbi_topk

        results = viterbi_topk(text, 5)
        scores = [s for s, _ in results]
        assert scores == sorted(scores, reverse=True), (
            f"scores not in non-increasing order: {scores}"
        )

    @pytest.mark.parametrize("text", [f.values[0] for f in FIXTURES])
    def test_all_segmentations_valid(self, text: str) -> None:
        from myspellchecker.tokenizers.cython.word_segment import viterbi_topk

        results = viterbi_topk(text, 5)
        for _score, words in results:
            assert "".join(words) == text, (
                f"invalid segmentation: {words!r} does not reassemble to {text!r}"
            )

    @pytest.mark.parametrize("text", [f.values[0] for f in FIXTURES])
    def test_all_segmentations_distinct(self, text: str) -> None:
        from myspellchecker.tokenizers.cython.word_segment import viterbi_topk

        results = viterbi_topk(text, 5)
        seen: set[tuple[str, ...]] = set()
        for _score, words in results:
            key = tuple(words)
            assert key not in seen, f"duplicate segmentation: {words!r}"
            seen.add(key)


class TestTopKEnumeration:
    """Bounded-enumeration test: a 2-character chunk has exactly 2 valid
    segmentations (merged and split). At K=5 we must see both — proving the
    top-K extraction actually enumerates alternates instead of just returning
    top-1 padded."""

    def test_two_char_chunk_yields_two_segmentations(self) -> None:
        from myspellchecker.tokenizers.cython.word_segment import viterbi_topk

        results = viterbi_topk(TWO_CHAR_CHUNK, 5)

        assert len(results) == 2, (
            f"2-char chunk has 2 possible segmentations; got {len(results)}: "
            f"{[w for _, w in results]}"
        )

        segmentations = {tuple(words) for _score, words in results}
        assert segmentations == {(TWO_CHAR_CHUNK,), tuple(TWO_CHAR_CHUNK)}, (
            f"expected both merged and split; got {segmentations}"
        )

    def test_two_char_chunk_k1_is_best_of_both(self) -> None:
        """K=1 on the 2-char chunk must match the existing viterbi (whichever
        of the two segmentations the LM prefers). This links the enumeration
        test back to the parity contract."""
        from myspellchecker.tokenizers.cython.word_segment import (
            viterbi,
            viterbi_topk,
        )

        topk = viterbi_topk(TWO_CHAR_CHUNK, 1)
        baseline = viterbi(TWO_CHAR_CHUNK)
        assert len(topk) == 1
        assert topk[0] == baseline
