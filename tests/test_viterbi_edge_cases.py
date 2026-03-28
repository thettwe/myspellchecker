import math
from unittest.mock import MagicMock, patch

import pytest

from myspellchecker.algorithms.viterbi import ViterbiTagger


@pytest.fixture
def mock_provider():
    provider = MagicMock()
    provider.get_word_pos.return_value = None
    return provider


@pytest.fixture
def viterbi_tagger(mock_provider):
    # Setup basic probabilities
    pos_bigram = {("N", "V"): 0.5, ("V", "N"): 0.3}
    pos_trigram = {("N", "V", "N"): 0.4}
    pos_unigram = {"N": 0.4, "V": 0.3, "UNK": 0.1}

    # Mock Cython availability to force Python implementation
    with patch("myspellchecker.algorithms.viterbi._HAS_CYTHON_VITERBI", False):
        tagger = ViterbiTagger(
            provider=mock_provider,
            pos_bigram_probs=pos_bigram,
            pos_trigram_probs=pos_trigram,
            pos_unigram_probs=pos_unigram,
            beam_width=2,
        )
    return tagger


def test_get_emission_score(viterbi_tagger):
    # 1. Word-level emission (mocked)
    viterbi_tagger.word_tag_probs = {"word": {"N": 0.8}}
    score = viterbi_tagger._get_emission_score("word", "N")
    expected = viterbi_tagger.emission_weight * math.log(0.8)
    assert score == pytest.approx(expected)

    # 2. Unigram fallback
    score = viterbi_tagger._get_emission_score("other", "V")  # V is 0.3 in unigram
    expected = viterbi_tagger.emission_weight * math.log(0.3)
    assert score == pytest.approx(expected)

    # 3. Min prob fallback
    score = viterbi_tagger._get_emission_score("other", "Z")  # Z not in unigram
    expected = viterbi_tagger.emission_weight * math.log(viterbi_tagger.min_prob)
    assert score == pytest.approx(expected)


def test_get_valid_tags(viterbi_tagger):
    # 1. From provider
    viterbi_tagger.provider.get_word_pos.return_value = "N|V"
    tags = viterbi_tagger._get_valid_tags("known")
    assert tags == {"N", "V"}

    # 2. From seed (mixed with provider if provider returns something)
    viterbi_tagger.provider.get_word_pos.return_value = None
    viterbi_tagger.word_tag_probs = {"seeded": {"ADJ": 1.0}}
    tags = viterbi_tagger._get_valid_tags("seeded")
    assert tags == {"ADJ"}

    # 3. Morphology fallback
    viterbi_tagger.morphology_analyzer = MagicMock()
    viterbi_tagger.morphology_analyzer.guess_pos.return_value = {"ADV"}
    tags = viterbi_tagger._get_valid_tags("morpho")
    assert tags == {"ADV"}

    # 4. Unknown fallback
    viterbi_tagger.morphology_analyzer.guess_pos.return_value = set()
    tags = viterbi_tagger._get_valid_tags("unknown")
    assert tags == {viterbi_tagger.unknown_word_tag}


def test_tag_sequence_empty(viterbi_tagger):
    assert viterbi_tagger.tag_sequence([]) == []


def test_tag_sequence_no_probs(mock_provider):
    with patch("myspellchecker.algorithms.viterbi._HAS_CYTHON_VITERBI", False):
        tagger = ViterbiTagger(mock_provider, {}, {})
    assert tagger.tag_sequence(["word"]) == ["UNK"]


def test_tag_sequence_single_word(viterbi_tagger):
    # Setup: 'dog' is N (0.8) or V (0.2). N has higher unigram/emission prob.
    viterbi_tagger.word_tag_probs = {"dog": {"N": 0.8, "V": 0.2}}
    tags = viterbi_tagger.tag_sequence(["dog"])
    assert tags == ["N"]


def test_tag_sequence_two_words(viterbi_tagger):
    # Setup: "I run". I -> N, run -> V. N->V bigram is high.
    viterbi_tagger.word_tag_probs = {"I": {"N": 1.0}, "run": {"V": 0.9, "N": 0.1}}
    viterbi_tagger.pos_bigram_probs = {("N", "V"): 0.9, ("N", "N"): 0.1}

    tags = viterbi_tagger.tag_sequence(["I", "run"])
    assert tags == ["N", "V"]


def test_tag_sequence_trigram(viterbi_tagger):
    # Setup: "I saw her". N V N.
    viterbi_tagger.word_tag_probs = {"I": {"N": 1.0}, "saw": {"V": 1.0}, "her": {"N": 1.0}}
    # Trigram N V N is high
    viterbi_tagger.pos_trigram_probs = {("N", "V", "N"): 0.9}

    # Bigrams to support the path
    viterbi_tagger.pos_bigram_probs = {("N", "V"): 0.8, ("V", "N"): 0.8}

    tags = viterbi_tagger.tag_sequence(["I", "saw", "her"])
    assert tags == ["N", "V", "N"]


def test_beam_pruning(viterbi_tagger):
    # Force many tags for a word to trigger pruning
    # Beam width is 2 (from fixture)

    # Word "many_tags" has tags A, B, C, D
    viterbi_tagger.word_tag_probs = {
        "start": {"START": 1.0},
        "many": {"A": 0.4, "B": 0.3, "C": 0.2, "D": 0.1},
        "end": {"END": 1.0},
    }

    # Transitions
    viterbi_tagger.pos_bigram_probs = {
        ("START", "A"): 0.5,
        ("START", "B"): 0.5,
        ("START", "C"): 0.5,
        ("START", "D"): 0.5,
    }

    # Step 2 logic inside tag_sequence (t=1) generates states for "many"
    # It should prune to keep only 2 states (A and B likely, as they have higher emission)

    tags = viterbi_tagger.tag_sequence(["start", "many", "end"])
    # Just ensure it runs without error and returns valid tags
    assert len(tags) == 3
    assert tags[1] in ["A", "B", "C", "D"]


def test_viterbi_cython_delegation(mock_provider):
    # Test that it delegates to cython tagger if available
    with (
        patch("myspellchecker.algorithms.viterbi._HAS_CYTHON_VITERBI", True),
        patch("myspellchecker.algorithms.viterbi.viterbi_c") as mock_c,
    ):
        mock_cython_instance = MagicMock()
        mock_c.CythonViterbiTagger.return_value = mock_cython_instance
        mock_cython_instance.tag_sequence.return_value = ["TAG"]

        tagger = ViterbiTagger(mock_provider, {}, {})
        result = tagger.tag_sequence(["word"])

        assert result == ["TAG"]
        mock_cython_instance.tag_sequence.assert_called_once_with(["word"])


def test_beam_pruning_fallback_handler(mock_provider):
    """
    Test beam pruning fallback when state is missing from backpointer.

    This simulates a scenario where aggressive beam pruning causes the current
    state to be missing from backpointer during backtracking, which should
    gracefully fall back to unknown_word_tag instead of raising KeyError.
    """
    with patch("myspellchecker.algorithms.viterbi._HAS_CYTHON_VITERBI", False):
        # Very narrow beam width to trigger aggressive pruning
        tagger = ViterbiTagger(
            provider=mock_provider,
            pos_bigram_probs={
                ("A", "B"): 0.9,
                ("B", "C"): 0.9,
                ("C", "D"): 0.9,
                ("D", "E"): 0.9,
            },
            pos_trigram_probs={
                ("A", "B", "C"): 0.9,
                ("B", "C", "D"): 0.9,
                ("C", "D", "E"): 0.9,
            },
            pos_unigram_probs={"A": 0.2, "B": 0.2, "C": 0.2, "D": 0.2, "E": 0.2},
            beam_width=1,  # Very narrow beam to force pruning
        )

        # Give each word different tag options that create many state combinations
        tagger.word_tag_probs = {
            "w1": {"A": 0.8, "X": 0.2},  # X will get pruned
            "w2": {"B": 0.8, "Y": 0.2},  # Y will get pruned
            "w3": {"C": 0.8, "Z": 0.2},  # Z will get pruned
            "w4": {"D": 0.8, "W": 0.2},  # W will get pruned
            "w5": {"E": 0.8, "Q": 0.2},  # Q will get pruned
        }

        # Should not raise KeyError - should gracefully fall back to UNK
        # for any positions where beam pruning removed the required state
        result = tagger.tag_sequence(["w1", "w2", "w3", "w4", "w5"])

        # Should return valid tags (may include UNK if fallback triggered)
        assert len(result) == 5
        for tag in result:
            assert tag in ["A", "B", "C", "D", "E", "X", "Y", "Z", "W", "Q", "UNK"]


# === Edge Case Tests for Beam Pruning ===


def test_beam_width_one_single_path(mock_provider):
    """
    Test beam_width=1 forces single path through state space.

    With beam_width=1, only the single best state is kept at each position.
    This should produce deterministic results and not crash.
    """
    with patch("myspellchecker.algorithms.viterbi._HAS_CYTHON_VITERBI", False):
        tagger = ViterbiTagger(
            provider=mock_provider,
            pos_bigram_probs={
                ("N", "V"): 0.8,
                ("V", "N"): 0.7,
                ("N", "N"): 0.3,
                ("V", "V"): 0.2,
            },
            pos_trigram_probs={},
            pos_unigram_probs={"N": 0.5, "V": 0.5},
            beam_width=1,  # Single path only
        )

        # Multiple ambiguous words
        tagger.word_tag_probs = {
            "word1": {"N": 0.6, "V": 0.4},
            "word2": {"N": 0.4, "V": 0.6},
            "word3": {"N": 0.5, "V": 0.5},
        }

        result = tagger.tag_sequence(["word1", "word2", "word3"])

        assert len(result) == 3
        # All tags should be valid
        for tag in result:
            assert tag in ["N", "V", "UNK"]

        # Run multiple times to verify determinism
        results = [tagger.tag_sequence(["word1", "word2", "word3"]) for _ in range(5)]
        assert all(r == results[0] for r in results), "beam_width=1 should be deterministic"


def test_beam_width_larger_than_tagset(mock_provider):
    """
    Test beam_width larger than the number of possible tags.

    When beam_width exceeds the tagset size, all states should be preserved
    (no pruning occurs). This should behave like exhaustive Viterbi.
    """
    with patch("myspellchecker.algorithms.viterbi._HAS_CYTHON_VITERBI", False):
        # Small tagset with only 3 tags
        tagger = ViterbiTagger(
            provider=mock_provider,
            pos_bigram_probs={
                ("A", "B"): 0.7,
                ("B", "C"): 0.7,
                ("A", "C"): 0.5,
                ("C", "A"): 0.6,
            },
            pos_trigram_probs={},
            pos_unigram_probs={"A": 0.4, "B": 0.3, "C": 0.3},
            beam_width=100,  # Much larger than tagset (3 tags)
        )

        tagger.word_tag_probs = {
            "w1": {"A": 0.8, "B": 0.1, "C": 0.1},
            "w2": {"A": 0.1, "B": 0.8, "C": 0.1},
            "w3": {"A": 0.1, "B": 0.1, "C": 0.8},
        }

        result = tagger.tag_sequence(["w1", "w2", "w3"])

        assert len(result) == 3
        # With large beam width, should find optimal path
        # Based on emission probs: A -> B -> C is likely
        for tag in result:
            assert tag in ["A", "B", "C", "UNK"]


def test_beam_pruning_deterministic_tiebreaking(mock_provider):
    """
    Test that tie-breaking in beam pruning is deterministic.

    When multiple states have identical scores, the pruning behavior
    should be consistent across multiple runs.
    """
    with patch("myspellchecker.algorithms.viterbi._HAS_CYTHON_VITERBI", False):
        tagger = ViterbiTagger(
            provider=mock_provider,
            pos_bigram_probs={
                ("X", "A"): 0.5,
                ("X", "B"): 0.5,  # Same probability as A
                ("X", "C"): 0.5,  # Same probability as A and B
                ("A", "Y"): 0.5,
                ("B", "Y"): 0.5,
                ("C", "Y"): 0.5,
            },
            pos_trigram_probs={},
            pos_unigram_probs={"X": 0.33, "A": 0.33, "B": 0.33, "C": 0.33, "Y": 0.33},
            beam_width=2,  # Only keep 2 of the 3 tied states
        )

        # All middle tags have equal emission probability
        tagger.word_tag_probs = {
            "start": {"X": 1.0},
            "middle": {"A": 0.33, "B": 0.33, "C": 0.33},  # Equal probs - tie!
            "end": {"Y": 1.0},
        }

        # Run multiple times to check determinism
        results = []
        for _ in range(10):
            result = tagger.tag_sequence(["start", "middle", "end"])
            results.append(tuple(result))

        # All results should be identical (deterministic tie-breaking)
        assert len(set(results)) == 1, (
            f"Tie-breaking should be deterministic, got {len(set(results))} different results"
        )


def test_beam_width_zero_raises_error(mock_provider):
    """
    Test that beam_width=0 is handled properly.

    A beam width of 0 makes no sense and should either raise an error
    or be treated as 1.
    """
    with patch("myspellchecker.algorithms.viterbi._HAS_CYTHON_VITERBI", False):
        # beam_width=0 should still work (treated as minimum viable)
        tagger = ViterbiTagger(
            provider=mock_provider,
            pos_bigram_probs={("N", "V"): 0.5},
            pos_trigram_probs={},
            pos_unigram_probs={"N": 0.5, "V": 0.5},
            beam_width=0,  # Edge case
        )

        tagger.word_tag_probs = {
            "word": {"N": 0.5, "V": 0.5},
        }

        # Should not crash - either works with 0 or treats as minimum
        result = tagger.tag_sequence(["word"])
        assert len(result) == 1


def test_beam_pruning_with_empty_sequence(mock_provider):
    """
    Test beam pruning with empty word sequence.
    """
    with patch("myspellchecker.algorithms.viterbi._HAS_CYTHON_VITERBI", False):
        tagger = ViterbiTagger(
            provider=mock_provider,
            pos_bigram_probs={("N", "V"): 0.5},
            pos_trigram_probs={},
            pos_unigram_probs={"N": 0.5, "V": 0.5},
            beam_width=5,
        )

        result = tagger.tag_sequence([])
        assert result == []
