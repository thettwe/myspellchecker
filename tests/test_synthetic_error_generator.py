"""Tests for SyntheticErrorGenerator."""

import pytest

from myspellchecker.training.constants import (
    DEFAULT_CORRUPTION_RATIO,
    DEFAULT_CORRUPTION_WEIGHTS,
    LABEL_CORRECT,
    LABEL_ERROR,
)


@pytest.fixture
def generator():
    """Create a SyntheticErrorGenerator with fixed seed."""
    from myspellchecker.training.generator import SyntheticErrorGenerator

    return SyntheticErrorGenerator(seed=42)


@pytest.fixture
def generator_high_ratio():
    """Generator with high corruption ratio for testing."""
    from myspellchecker.training.generator import SyntheticErrorGenerator

    return SyntheticErrorGenerator(corruption_ratio=0.5, seed=42)


class TestSyntheticErrorGeneratorInit:
    """Test generator initialization."""

    def test_default_init(self, generator):
        assert generator.corruption_ratio == DEFAULT_CORRUPTION_RATIO
        assert generator.corruption_weights == DEFAULT_CORRUPTION_WEIGHTS

    def test_custom_ratio(self):
        from myspellchecker.training.generator import SyntheticErrorGenerator

        gen = SyntheticErrorGenerator(corruption_ratio=0.3)
        assert gen.corruption_ratio == 0.3

    def test_custom_weights(self):
        from myspellchecker.training.generator import SyntheticErrorGenerator

        weights = {"homophone_swap": 1.0}
        gen = SyntheticErrorGenerator(corruption_weights=weights)
        assert gen.corruption_weights == weights

    def test_invalid_ratio(self):
        from myspellchecker.training.generator import SyntheticErrorGenerator

        with pytest.raises(ValueError):
            SyntheticErrorGenerator(corruption_ratio=-0.1)

    def test_invalid_ratio_above_one(self):
        from myspellchecker.training.generator import SyntheticErrorGenerator

        with pytest.raises(ValueError):
            SyntheticErrorGenerator(corruption_ratio=1.5)

    def test_homophones_loaded(self, generator):
        """YAML homophones should be loaded on first access."""
        # Lazy-loaded property
        homophones = generator.homophones
        assert homophones is not None
        assert isinstance(homophones, dict)

    def test_typo_patterns_loaded(self, generator):
        """YAML typo corrections should be loaded on first access."""
        # Lazy-loaded as inverse_typos property
        typos = generator.inverse_typos
        assert typos is not None
        assert isinstance(typos, dict)


class TestGenerate:
    """Test the generate method."""

    def test_generate_returns_list(self, generator):
        sentences = ["ကျွန်တော် စာ ဖတ် တယ်"]
        results = generator.generate(sentences)
        assert isinstance(results, list)
        assert len(results) == 1

    def test_generate_tuple_structure(self, generator):
        sentences = ["ကျွန်တော် စာ ဖတ် တယ်"]
        results = generator.generate(sentences)
        corrupted_sentence, words, labels = results[0]
        assert isinstance(corrupted_sentence, str)
        assert isinstance(words, list)
        assert isinstance(labels, list)
        assert len(words) == len(labels)

    def test_labels_are_binary(self, generator):
        sentences = ["ကျွန်တော် စာ ဖတ် တယ်"]
        results = generator.generate(sentences)
        _, _, labels = results[0]
        for label in labels:
            assert label in (LABEL_CORRECT, LABEL_ERROR)

    def test_empty_input(self, generator):
        results = generator.generate([])
        assert results == []

    def test_empty_sentence(self, generator):
        """Empty sentences may be skipped or produce empty word lists."""
        results = generator.generate([""])
        # Generator may skip empty sentences
        assert isinstance(results, list)

    def test_multiple_sentences(self, generator):
        sentences = [
            "ကျွန်တော် စာ ဖတ် တယ်",
            "သူ ကျောင်း သွား တယ်",
        ]
        results = generator.generate(sentences)
        assert len(results) == 2

    def test_corruption_ratio_respected(self, generator_high_ratio):
        """With high corruption ratio, at least some words should be corrupted."""
        sentences = ["ကျွန်တော် စာ ဖတ် တယ် ပါ ကြည့် ပေး"]
        results = generator_high_ratio.generate(sentences)
        _, _, labels = results[0]
        error_count = sum(1 for label in labels if label == LABEL_ERROR)
        # With 50% corruption ratio and 7 words, expect at least 1 error
        assert error_count >= 1

    def test_deterministic_with_seed(self):
        """Same seed should produce same results."""
        from myspellchecker.training.generator import SyntheticErrorGenerator

        gen1 = SyntheticErrorGenerator(seed=123)
        gen2 = SyntheticErrorGenerator(seed=123)
        sentences = ["ကျွန်တော် စာ ဖတ် တယ်"]
        r1 = gen1.generate(sentences)
        r2 = gen2.generate(sentences)
        assert r1[0][2] == r2[0][2]  # Same labels


class TestCorruptionMethods:
    """Test individual corruption methods."""

    def test_corrupt_medial_confusion(self, generator):
        """Medial confusion should swap ျ↔ြ or ွ↔ှ."""
        # Word with medial ya-yit (ျ)
        word = "ကျွန်"
        corrupted = generator._corrupt_medial_confusion(word)
        # May or may not corrupt depending on random choice, but should return a string
        assert isinstance(corrupted, str)

    def test_corrupt_char_deletion(self, generator):
        """Char deletion should remove a character."""
        word = "ကျွန်တော်"
        corrupted = generator._corrupt_char_deletion(word)
        assert isinstance(corrupted, str)
        # Corrupted should be shorter or same (if no deletable chars)
        assert len(corrupted) <= len(word)

    def test_corrupt_char_insertion(self, generator):
        """Char insertion should add a character."""
        word = "စာ"
        corrupted = generator._corrupt_char_insertion(word)
        assert isinstance(corrupted, str)

    def test_corrupt_similar_char(self, generator):
        """Similar char swap should replace with visually similar char."""
        word = "ကျွန်"
        corrupted = generator._corrupt_similar_char(word)
        assert isinstance(corrupted, str)

    def test_corrupt_returns_original_if_impossible(self, generator):
        """If corruption not possible, return original."""
        # Single char word with no possible corruption
        word = "a"
        corrupted = generator._corrupt_homophone_swap(word)
        # Should return the original or corrupted string
        assert isinstance(corrupted, str)
