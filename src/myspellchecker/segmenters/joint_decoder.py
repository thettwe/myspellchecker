"""Joint noisy-channel lattice decoder for segmentation correction.

Augments K-best Viterbi paths with SymSpell merge edges, then runs DP
on the resulting lattice to find a segmentation that can recover
over-split compound errors.

This module is Stage 1 infrastructure for the segmenter-architecture
workstream. It is NOT wired into production — consumers call it
explicitly via decode().
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from myspellchecker.algorithms.symspell import SymSpell


@dataclass
class LatticeEdge:
    start: int
    end: int
    word: str
    score: float
    edge_type: str  # "base" | "merge"
    edit_distance: int = 0
    source_tokens: list[str] = field(default_factory=list)


def _word_score(word: str) -> float:
    """Get the unigram log-prob of a single word via the Cython Viterbi LM."""
    from myspellchecker.tokenizers.cython.word_segment import viterbi

    score, tokens = viterbi(word)
    if len(tokens) == 1 and tokens[0] == word:
        return score
    return score


def _is_lm_single_token(word: str) -> bool:
    """True if the LM recognizes word as a single token (not split)."""
    from myspellchecker.tokenizers.cython.word_segment import viterbi

    _, tokens = viterbi(word)
    return len(tokens) == 1 and tokens[0] == word


def build_lattice(
    chunk: str,
    topk_results: list[tuple[float, list[str]]],
    symspell: SymSpell,
    merge_bonus: float = 1.0,
    edit_penalty_weight: float = 0.5,
    merge_freq_floor: int = 100,
    max_merge_candidates: int = 3,
    max_edit_distance: int = 2,
    min_edit_distance: int = 0,
    source_freq_gate: int = 0,
    require_fragment: bool = False,
) -> list[LatticeEdge]:
    """Build a lattice from K-best paths + SymSpell merge edges.

    Returns a flat list of LatticeEdge objects forming a DAG over
    character positions [0, len(chunk)].
    """
    edges: list[LatticeEdge] = []
    seen_edges: set[tuple[int, int, str]] = set()

    # Step 1: Add base edges from all K-best paths
    for _path_score, words in topk_results:
        pos = 0
        for word in words:
            start = pos
            end = pos + len(word)
            key = (start, end, word)
            if key not in seen_edges:
                seen_edges.add(key)
                edges.append(
                    LatticeEdge(
                        start=start,
                        end=end,
                        word=word,
                        score=_word_score(word),
                        edge_type="base",
                    )
                )
            pos = end

    # Step 2: Add SymSpell merge edges for adjacent token pairs
    for _path_score, words in topk_results:
        pos = 0
        positions: list[tuple[int, int]] = []
        for word in words:
            positions.append((pos, pos + len(word)))
            pos += len(word)

        # 2-token merges
        for i in range(len(positions) - 1):
            s1, e1 = positions[i]
            s2, e2 = positions[i + 1]
            merged_text = chunk[s1:e2]

            _add_merge_edges(
                edges,
                seen_edges,
                symspell,
                s1,
                e2,
                merged_text,
                [words[i], words[i + 1]],
                merge_bonus,
                edit_penalty_weight,
                merge_freq_floor,
                max_merge_candidates,
                max_edit_distance,
                min_edit_distance,
                source_freq_gate,
                require_fragment,
            )

        # 3-token merges
        for i in range(len(positions) - 2):
            s1, _ = positions[i]
            _, e3 = positions[i + 2]
            merged_text = chunk[s1:e3]

            _add_merge_edges(
                edges,
                seen_edges,
                symspell,
                s1,
                e3,
                merged_text,
                [words[i], words[i + 1], words[i + 2]],
                merge_bonus,
                edit_penalty_weight,
                merge_freq_floor,
                max_merge_candidates,
                max_edit_distance,
                min_edit_distance,
                source_freq_gate,
                require_fragment,
            )

    return edges


def _get_word_freq(word: str, symspell: SymSpell) -> int:
    """Get dictionary frequency of a word. Returns 0 if OOV."""
    try:
        hits = symspell.lookup(word, level="word", max_suggestions=1, include_known=True)
        if hits and hits[0].edit_distance == 0:
            return hits[0].frequency
    except Exception:
        pass
    return 0


def _add_merge_edges(
    edges: list[LatticeEdge],
    seen_edges: set[tuple[int, int, str]],
    symspell: SymSpell,
    start: int,
    end: int,
    merged_text: str,
    source_tokens: list[str],
    merge_bonus: float,
    edit_penalty_weight: float,
    merge_freq_floor: int,
    max_merge_candidates: int,
    max_edit_distance: int = 2,
    min_edit_distance: int = 0,
    source_freq_gate: int = 0,
    require_fragment: bool = False,
) -> None:
    """Generate merge edges from SymSpell lookup on concatenated text."""
    # Fragment gate: only merge when at least one source token is not
    # recognized by the LM as a single token (suggesting it's a fragment).
    if require_fragment:
        if all(_is_lm_single_token(t) for t in source_tokens):
            return

    # Source-token frequency gate: only merge when at least one source token
    # is low-frequency (suggesting it's a fragment, not a real word).
    if source_freq_gate > 0:
        min_freq = min(_get_word_freq(t, symspell) for t in source_tokens)
        if min_freq >= source_freq_gate:
            return

    try:
        candidates = symspell.lookup(
            merged_text,
            level="word",
            max_suggestions=max_merge_candidates,
            include_known=True,
        )
    except Exception:
        return

    added = 0
    for sug in candidates:
        if sug.edit_distance > max_edit_distance:
            continue
        if sug.edit_distance < min_edit_distance:
            continue
        if sug.frequency < merge_freq_floor:
            continue

        key = (start, end, sug.term)
        if key in seen_edges:
            continue
        seen_edges.add(key)

        # Score: LM score of the SymSpell candidate + merge bonus - edit penalty
        lm_score = _word_score(sug.term)
        edge_score = lm_score + merge_bonus - edit_penalty_weight * sug.edit_distance

        edges.append(
            LatticeEdge(
                start=start,
                end=end,
                word=sug.term,
                score=edge_score,
                edge_type="merge",
                edit_distance=sug.edit_distance,
                source_tokens=source_tokens,
            )
        )
        added += 1
        if added >= max_merge_candidates:
            break


def lattice_dp(
    edges: list[LatticeEdge],
    chunk_len: int,
) -> tuple[float, list[LatticeEdge]]:
    """Viterbi-style DP on the lattice DAG to find the highest-scoring path.

    Returns (total_score, edge_path) for the best path from position 0
    to position chunk_len.
    """
    # best[pos] = (best_score_to_reach_pos, backpointer_edge)
    best: dict[int, tuple[float, LatticeEdge | None]] = {0: (0.0, None)}

    # Group edges by start position for efficient lookup
    edges_by_start: dict[int, list[LatticeEdge]] = {}
    for e in edges:
        edges_by_start.setdefault(e.start, []).append(e)

    # Forward pass
    for pos in sorted(edges_by_start.keys()):
        if pos not in best:
            continue
        prev_score, _ = best[pos]

        for edge in edges_by_start[pos]:
            candidate_score = prev_score + edge.score
            if edge.end not in best or candidate_score > best[edge.end][0]:
                best[edge.end] = (candidate_score, edge)

    # Backtrack
    if chunk_len not in best:
        return -math.inf, []

    path: list[LatticeEdge] = []
    pos = chunk_len
    while pos > 0:
        _, edge = best[pos]
        if edge is None:
            break
        path.append(edge)
        pos = edge.start
    path.reverse()

    total_score = best[chunk_len][0]
    return total_score, path


def decode(
    chunk: str,
    K: int,
    symspell: SymSpell,
    merge_bonus: float = 1.0,
    edit_penalty_weight: float = 0.5,
    merge_freq_floor: int = 100,
    max_merge_candidates: int = 3,
    max_edit_distance: int = 2,
    min_edit_distance: int = 0,
    source_freq_gate: int = 0,
    require_fragment: bool = False,
    min_score_gain: float = 0.0,
) -> tuple[list[str], float, dict]:
    """Run the joint lattice decoder on a single chunk.

    Returns (words, score, metadata) where:
      - words: list of tokens in the best path
      - score: total path score
      - metadata: dict with lattice stats and merge info
    """
    from myspellchecker.tokenizers.cython.word_segment import viterbi_topk

    topk = viterbi_topk(chunk, K)

    edges = build_lattice(
        chunk,
        topk,
        symspell,
        merge_bonus=merge_bonus,
        edit_penalty_weight=edit_penalty_weight,
        merge_freq_floor=merge_freq_floor,
        max_merge_candidates=max_merge_candidates,
        max_edit_distance=max_edit_distance,
        min_edit_distance=min_edit_distance,
        source_freq_gate=source_freq_gate,
        require_fragment=require_fragment,
    )

    total_score, path = lattice_dp(edges, len(chunk))

    baseline_words = topk[0][1] if topk else []
    baseline_score = topk[0][0] if topk else -math.inf

    score_gain = total_score - baseline_score
    has_merges = any(e.edge_type == "merge" for e in path)

    # Gate 1: reject base-only recombinations — only accept paths that
    # include at least one merge edge (the whole point of the decoder).
    # Gate 2: score-gain threshold — the merge path must beat the baseline
    # by a meaningful margin to avoid spurious corrections.
    use_lattice = has_merges and (min_score_gain <= 0.0 or score_gain >= min_score_gain)

    if use_lattice:
        words = [e.word for e in path]
        merge_edges_used = [e for e in path if e.edge_type == "merge"]
        effective_score = total_score
    else:
        words = baseline_words
        merge_edges_used: list[LatticeEdge] = []
        effective_score = baseline_score

    metadata = {
        "K": K,
        "num_paths": len(topk),
        "num_edges": len(edges),
        "num_base_edges": sum(1 for e in edges if e.edge_type == "base"),
        "num_merge_edges": sum(1 for e in edges if e.edge_type == "merge"),
        "merge_edges_in_path": len(merge_edges_used),
        "merges": [
            {
                "word": e.word,
                "source_tokens": e.source_tokens,
                "edit_distance": e.edit_distance,
                "score": round(e.score, 4),
            }
            for e in merge_edges_used
        ],
        "baseline_words": baseline_words,
        "baseline_score": round(baseline_score, 4),
        "lattice_score": round(total_score, 4),
        "score_gain": round(score_gain, 4),
        "changed": words != baseline_words,
    }

    return words, effective_score, metadata
