from unittest.mock import MagicMock

import pytest

from myspellchecker.algorithms.ngram_context_checker import NgramContextChecker
from myspellchecker.core.config import NgramContextConfig


@pytest.fixture
def mock_provider():
    provider = MagicMock()
    # Default behavior for frequencies/probabilities
    provider.get_bigram_probability.return_value = 0.0
    provider.get_trigram_probability.return_value = 0.0
    provider.get_word_frequency.return_value = 0
    provider.get_word_pos.return_value = None
    provider.get_top_continuations.return_value = []
    return provider


@pytest.fixture
def mock_symspell():
    symspell = MagicMock()
    symspell.lookup.return_value = []
    symspell.phonetic_hasher.get_phonetic_variants.return_value = set()
    return symspell


@pytest.fixture
def checker(mock_provider, mock_symspell):
    cfg = NgramContextConfig(
        bigram_threshold=0.01,
        trigram_threshold=0.005,
        right_context_threshold=0.1,
        min_unigram_threshold=10,
        unigram_denominator=1_000_000.0,
    )
    return NgramContextChecker(
        provider=mock_provider,
        config=cfg,
        symspell=mock_symspell,
    )


def test_get_smoothed_bigram_probability(checker, mock_provider):
    # Case 1: Bigram exists
    mock_provider.get_bigram_probability.return_value = 0.5
    assert checker.get_smoothed_bigram_probability("a", "b") == 0.5

    # Case 2: Unseen bigram, backoff to unigram
    mock_provider.get_bigram_probability.return_value = 0.0
    mock_provider.get_word_frequency.return_value = 1000  # unigram prob ~ 0.001
    # 0.4 * 0.001 = 0.0004
    assert checker.get_smoothed_bigram_probability("a", "b") == pytest.approx(0.0004)

    # Case 3: No unigram info
    mock_provider.get_word_frequency.return_value = 0
    assert checker.get_smoothed_bigram_probability("a", "b") == 0.0

    # Case 4: Smoothing disabled
    checker.use_smoothing = False
    assert checker.get_smoothed_bigram_probability("a", "b") == 0.0


def test_get_smoothed_trigram_probability(checker, mock_provider):
    # Case 1: Trigram exists
    mock_provider.get_trigram_probability.return_value = 0.5
    assert checker.get_smoothed_trigram_probability("a", "b", "c") == 0.5

    # Case 2: Unseen trigram, backoff to bigram
    mock_provider.get_trigram_probability.return_value = 0.0
    # Bigram ("b", "c")
    mock_provider.get_bigram_probability.side_effect = lambda w1, w2: (
        0.1 if (w1, w2) == ("b", "c") else 0.0
    )
    # 0.4 * 0.1 = 0.04
    assert checker.get_smoothed_trigram_probability("a", "b", "c") == pytest.approx(0.04)

    # Case 3: Unseen bigram, backoff to unigram
    # Clear side_effect from previous case
    mock_provider.get_bigram_probability.side_effect = None
    mock_provider.get_bigram_probability.return_value = 0.0

    mock_provider.get_word_frequency.return_value = 1000
    # 0.4^2 * 0.001 = 0.16 * 0.001 = 0.00016
    # Floor = trigram_threshold * backoff_floor_multiplier = 0.005 * 0.1 = 0.0005
    # max(0.00016, 0.0005) = 0.0005 (floor exceeds backoff_prob)
    assert checker.get_smoothed_trigram_probability("a", "b", "c") == pytest.approx(0.0005)


def test_is_contextual_error_trigram(checker, mock_provider):
    # Trigram logic
    checker.use_smoothing = False
    mock_provider.get_trigram_probability.return_value = 0.001
    # 0.001 < 0.005 threshold -> Error
    assert checker.is_contextual_error("b", "c", prev_prev_word="a") is True

    mock_provider.get_trigram_probability.return_value = 0.01
    # 0.01 > 0.005 -> Not error
    assert checker.is_contextual_error("b", "c", prev_prev_word="a") is False


def test_is_contextual_error_bidirectional(checker, mock_provider):
    # Left context weak, Right context strong
    # Left: P(curr|prev)
    # right_context_threshold = threshold * 10 = 0.01 * 10 = 0.1
    # So right_prob must be > 0.1 to rescue
    mock_provider.get_bigram_probability.side_effect = lambda w1, w2: (
        0.005
        if (w1, w2) == ("prev", "curr")
        else 0.15  # Must be > right_context_threshold (0.1)
        if (w1, w2) == ("curr", "next")
        else 0.0
    )

    # Left 0.005 < 0.01 (threshold) -> potential error
    # Right 0.15 > 0.1 (right_context_threshold) -> strong confirmation -> NOT error
    assert checker.is_contextual_error("prev", "curr", next_word="next") is False

    # Left weak, Right weak
    mock_provider.get_bigram_probability.side_effect = lambda w1, w2: 0.005
    assert checker.is_contextual_error("prev", "curr", next_word="next") is True


def test_is_contextual_error_unseen_backoff(checker, mock_provider):
    # Unseen bigram (P=0)
    mock_provider.get_bigram_probability.return_value = 0.0

    # Case A: Common word -> assumed valid
    mock_provider.get_word_frequency.return_value = 20  # > 10
    assert checker.is_contextual_error("prev", "common") is False

    # Case B: Rare word -> potential typo
    mock_provider.get_word_frequency.return_value = 5  # < 10

    # Setup SymSpell neighbor (edit_distance=1 to pass the close-neighbor filter)
    neighbor = MagicMock()
    neighbor.term = "correct"
    neighbor.edit_distance = 1
    checker.symspell.lookup.return_value = [neighbor]

    # Neighbor context probability is high
    # P(correct|prev) > threshold * multiplier
    # 0.01 * 10 = 0.1
    # lambda handles neighbor lookup
    def prob_side_effect(w1, w2):
        if w2 == "correct":
            return 0.2
        return 0.0

    mock_provider.get_bigram_probability.side_effect = prob_side_effect

    assert checker.is_contextual_error("prev", "rare") is True


def test_suggest_scoring_logic(checker, mock_provider):
    # Current prob
    # Use "cura" as candidate (distance 1 from "curr")
    candidate = "cura"

    mock_provider.get_bigram_probability.side_effect = lambda w1, w2: (
        0.001 if (w1, w2) == ("prev", "curr") else 0.5 if (w1, w2) == ("prev", candidate) else 0.0
    )

    checker.provider.get_top_continuations.return_value = [(candidate, 0.5)]

    suggestions = checker.suggest("prev", "curr")
    assert len(suggestions) == 1
    assert suggestions[0].term == candidate
    assert suggestions[0].score > -100  # Should be reasonably high


def test_generate_candidates_with_symspell(mock_provider, mock_symspell):
    checker = NgramContextChecker(provider=mock_provider, symspell=mock_symspell)
    mock_provider.get_top_continuations.return_value = [("prov_cand", 1)]
    sym_res = MagicMock()
    sym_res.term = "sym_cand"
    mock_symspell.lookup.return_value = [sym_res]
    candidates = checker._generate_candidates("prev", "curr", max_edit_distance=2)
    assert "prov_cand" in candidates
    assert "sym_cand" in candidates


def test_clear_context_cache(mock_provider):
    checker = NgramContextChecker(provider=mock_provider, config=NgramContextConfig())
    checker._bidir_prob_cache[("a", ("b",), ("c",))] = 0.5
    checker._ngram_context_cache[("x", "y")] = MagicMock()
    assert len(checker._bidir_prob_cache) > 0
    assert len(checker._ngram_context_cache) > 0
    checker.clear_context_cache()
    assert len(checker._bidir_prob_cache) == 0
    assert len(checker._ngram_context_cache) == 0


def test_suggest_pos_context_scoring(checker, mock_provider):
    # Enable POS scoring
    checker.pos_bigram_probs = {("N", "V"): 0.8}
    checker.pos_score_weight = 1.0

    candidate = "candidate"

    # Words and POS
    mock_provider.get_word_pos.side_effect = lambda w: (
        "N" if w == "prev" else "V" if w == candidate else None
    )

    # Bigram probs
    mock_provider.get_bigram_probability.side_effect = lambda w1, w2: (
        0.5 if (w1, w2) == ("prev", candidate) else 0.001
    )

    checker.provider.get_top_continuations.return_value = [(candidate, 0.5)]

    # Suggest with large max_edit_distance to allow "candidate" vs "curr"
    suggestions = checker.suggest("prev", "curr", max_edit_distance=10)

    # Should calculate pos_context_score
    # Left context: P(V|N) = 0.8
    # pos_distance_reduction should be applied
    assert len(suggestions) > 0
    assert suggestions[0].term == candidate
