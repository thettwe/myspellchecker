"""Fixture tests for the joint noisy-channel lattice decoder.

Verifies lattice construction, DP path finding, and merge-edge behavior
on hand-crafted inputs with known gold corrections.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from myspellchecker.segmenters.joint_decoder import (
    LatticeEdge,
    build_lattice,
    decode,
    lattice_dp,
)

_REAL_MMAP = Path.home() / ".cache" / "myspellchecker" / "resources" / "segmentation.mmap"


class TestLatticeDPUnit:
    """Unit tests for the DP solver on synthetic lattices."""

    def test_single_path(self) -> None:
        edges = [
            LatticeEdge(start=0, end=3, word="abc", score=-2.0, edge_type="base"),
            LatticeEdge(start=3, end=6, word="def", score=-3.0, edge_type="base"),
        ]
        score, path = lattice_dp(edges, 6)
        assert score == pytest.approx(-5.0)
        assert [e.word for e in path] == ["abc", "def"]

    def test_merge_wins_over_split(self) -> None:
        edges = [
            LatticeEdge(start=0, end=3, word="ab", score=-4.0, edge_type="base"),
            LatticeEdge(start=3, end=6, word="cd", score=-4.0, edge_type="base"),
            LatticeEdge(start=0, end=6, word="abcd", score=-3.0, edge_type="merge"),
        ]
        score, path = lattice_dp(edges, 6)
        assert len(path) == 1
        assert path[0].word == "abcd"
        assert path[0].edge_type == "merge"

    def test_split_wins_when_merge_score_low(self) -> None:
        edges = [
            LatticeEdge(start=0, end=3, word="ab", score=-2.0, edge_type="base"),
            LatticeEdge(start=3, end=6, word="cd", score=-2.0, edge_type="base"),
            LatticeEdge(start=0, end=6, word="abcd", score=-10.0, edge_type="merge"),
        ]
        score, path = lattice_dp(edges, 6)
        assert len(path) == 2
        assert [e.word for e in path] == ["ab", "cd"]

    def test_unreachable_end_returns_neg_inf(self) -> None:
        edges = [
            LatticeEdge(start=0, end=3, word="ab", score=-2.0, edge_type="base"),
        ]
        score, path = lattice_dp(edges, 6)
        assert score == -math.inf
        assert path == []

    def test_three_token_merge(self) -> None:
        edges = [
            LatticeEdge(start=0, end=2, word="a", score=-3.0, edge_type="base"),
            LatticeEdge(start=2, end=4, word="b", score=-3.0, edge_type="base"),
            LatticeEdge(start=4, end=6, word="c", score=-3.0, edge_type="base"),
            LatticeEdge(start=0, end=6, word="abc", score=-4.0, edge_type="merge"),
        ]
        score, path = lattice_dp(edges, 6)
        assert len(path) == 1
        assert path[0].word == "abc"


@pytest.mark.skipif(
    not _REAL_MMAP.exists() or _REAL_MMAP.stat().st_size < 1_000_000,
    reason=f"Real segmentation mmap not cached at {_REAL_MMAP}",
)
class TestBuildLatticeIntegration:
    """Integration tests requiring real mmap + SpellChecker initialization."""

    @pytest.fixture(autouse=True)
    def _init_checker(self) -> None:
        from myspellchecker.tokenizers.cython.word_segment import initialize_mmap

        assert initialize_mmap(str(_REAL_MMAP)), "Failed to initialize real mmap"

        from myspellchecker import SpellChecker
        from myspellchecker.core.config.main import SpellCheckerConfig
        from myspellchecker.providers.sqlite import SQLiteProvider

        cfg = SpellCheckerConfig()
        provider = SQLiteProvider(database_path="data/mySpellChecker_production.db")
        self.checker = SpellChecker(config=cfg, provider=provider)
        _ = self.checker.check("ကျွန်တော်")
        self.symspell = self.checker.symspell

    def test_build_lattice_has_base_and_merge_edges(self) -> None:
        from myspellchecker.tokenizers.cython.word_segment import viterbi_topk

        chunk = "မြန်မာစာ"
        topk = viterbi_topk(chunk, 3)
        edges = build_lattice(chunk, topk, self.symspell, merge_bonus=1.0)
        base_count = sum(1 for e in edges if e.edge_type == "base")
        assert base_count > 0, "must have base edges"
        assert all(e.start >= 0 and e.end <= len(chunk) for e in edges)
        for e in edges:
            assert e.word, "edge must have a word"
            assert math.isfinite(e.score), f"non-finite score {e.score}"

    def test_decode_path_covers_full_chunk(self) -> None:
        chunk = "ကျွန်တော်တို့"
        words, score, meta = decode(chunk, K=3, symspell=self.symspell)
        assert math.isfinite(score)
        assert meta["num_edges"] > 0
        assert len(words) >= 1

    def test_decode_recovers_over_split_correction(self) -> None:
        """The erroneous form 'စွမ်းဆောင်ရည' (missing asat) gets over-split
        by Viterbi into ['စွမ်းဆောင်', 'ရ', 'ည']. The lattice decoder should
        produce a merge edge with the gold correction 'စွမ်းဆောင်ရည်'."""
        chunk = "စွမ်းဆောင်ရည"
        words, score, meta = decode(chunk, K=5, symspell=self.symspell, merge_bonus=1.5)
        assert meta["merge_edges_in_path"] > 0, (
            f"Expected merge edge to fire; baseline={meta['baseline_words']}"
        )
        merge_word = meta["merges"][0]["word"]
        assert merge_word == "စွမ်းဆောင်ရည်", f"Expected gold correction, got {merge_word}"
