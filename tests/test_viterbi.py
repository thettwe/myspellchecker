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
