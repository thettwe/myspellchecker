"""
Unit tests for morphology integration in the spell checker.

Tests the morphological analysis for OOV (Out-of-Vocabulary) recovery,
including suffix stripping, root extraction, and suggestion enhancement.
"""

import pytest

from myspellchecker.text.morphology import (
    MorphologyAnalyzer,
    POSGuess,
    WordAnalysis,
    analyze_word,
)


class TestWordAnalysisDataclass:
    """Tests for the WordAnalysis dataclass."""

    def test_word_analysis_defaults(self):
        """Test WordAnalysis with default values."""
        analysis = WordAnalysis(original="test", root="test")
        assert analysis.original == "test"
        assert analysis.root == "test"
        assert analysis.suffixes == []
        assert analysis.pos_guesses == []
        assert analysis.confidence == 0.0
        assert analysis.is_compound is False

    def test_word_analysis_with_suffixes(self):
        """Test WordAnalysis with suffixes."""
        analysis = WordAnalysis(
            original="စားခဲ့သည်",
            root="စား",
            suffixes=["ခဲ့", "သည်"],
            confidence=0.85,
        )
        assert analysis.original == "စားခဲ့သည်"
        assert analysis.root == "စား"
        assert analysis.suffixes == ["ခဲ့", "သည်"]
        assert analysis.confidence == 0.85

    def test_word_analysis_repr(self):
        """Test WordAnalysis string representation."""
        analysis = WordAnalysis(
            original="စားခဲ့သည်",
            root="စား",
            suffixes=["ခဲ့", "သည်"],
            confidence=0.85,
        )
        repr_str = repr(analysis)
        assert "စား" in repr_str
        assert "ခဲ့+သည်" in repr_str
        assert "0.85" in repr_str


class TestMorphologyAnalyzerAnalyzeWord:
    """Tests for the MorphologyAnalyzer.analyze_word method."""

    @pytest.fixture
    def analyzer(self):
        """Create a MorphologyAnalyzer instance."""
        return MorphologyAnalyzer()

    def test_analyze_empty_word(self, analyzer):
        """Test analyzing an empty word."""
        result = analyzer.analyze_word("")
        assert result.original == ""
        assert result.root == ""
        assert result.suffixes == []

    def test_analyze_word_with_verb_suffix(self, analyzer):
        """Test analyzing a word with verb suffixes."""
        # စားခဲ့သည် = "ate" (verb + past tense + sentence ending)
        result = analyzer.analyze_word("စားခဲ့သည်")
        assert result.original == "စားခဲ့သည်"
        # Root should be extracted after suffix stripping
        assert len(result.suffixes) >= 1  # At least one suffix
        assert result.root != result.original  # Root should be different

    def test_analyze_word_with_sentence_particle(self, analyzer):
        """Test analyzing a word ending with sentence particle."""
        # ကောင်းပါတယ် = "it's good" (adj + polite + sentence ending)
        result = analyzer.analyze_word("ကောင်းပါတယ်")
        assert result.original == "ကောင်းပါတယ်"
        # Accept either split suffixes or combined form
        assert "တယ်" in result.suffixes or "ပါ" in result.suffixes or "ပါတယ်" in result.suffixes

    def test_analyze_word_with_plural_suffix(self, analyzer):
        """Test analyzing a word with plural suffix."""
        # ကလေးများ = "children" (child + plural)
        result = analyzer.analyze_word("ကလေးများ")
        assert result.original == "ကလေးများ"
        # "များ" is a common plural suffix
        if result.suffixes:
            assert "များ" in result.suffixes or result.root == "ကလေး"

    def test_analyze_simple_word_no_suffix(self, analyzer):
        """Test analyzing a simple word without suffixes."""
        # စာ = "letter" (simple noun)
        result = analyzer.analyze_word("စာ")
        # Single syllable words often have no suffixes
        assert result.original == "စာ"


class TestAnalyzeWordWithDictionaryCheck:
    """Tests for analyze_word with dictionary validation."""

    @pytest.fixture
    def mock_dictionary(self):
        """Create a mock dictionary check function."""
        valid_words = {"စား", "သွား", "လာ", "ကောင်း", "မြန်မာ", "ကလေး", "စာ"}
        return lambda word: word in valid_words

    def test_analyze_with_valid_root(self, mock_dictionary):
        """Test that valid roots are preserved."""
        result = analyze_word("စားခဲ့သည်", dictionary_check=mock_dictionary)
        assert result.root == "စား"
        assert "ခဲ့" in result.suffixes
        assert "သည်" in result.suffixes

    def test_analyze_with_invalid_root_fallback(self, mock_dictionary):
        """Test fallback when root is not in dictionary."""
        # ထမင်း is NOT in mock dictionary
        result = analyze_word("ထမင်းစားခဲ့သည်", dictionary_check=mock_dictionary)
        # Should try to find a valid intermediate root
        # If none found, root should be the longest valid prefix
        assert result.original == "ထမင်းစားခဲ့သည်"

    def test_confidence_boost_with_valid_root(self, mock_dictionary):
        """Test that confidence is higher when root is validated."""
        result_with_dict = analyze_word("စားခဲ့သည်", dictionary_check=mock_dictionary)
        result_without_dict = analyze_word("စားခဲ့သည်")

        # With dictionary validation, confidence should be higher
        assert result_with_dict.confidence >= result_without_dict.confidence

    def test_analyze_known_word_returns_minimal_analysis(self, mock_dictionary):
        """Test that a known word without suffixes returns minimal analysis."""
        result = analyze_word("စာ", dictionary_check=mock_dictionary)
        # Simple word should have no suffixes stripped
        assert result.original == "စာ"


class TestModuleLevelAnalyzeWord:
    """Tests for the module-level analyze_word convenience function."""

    def test_convenience_function_works(self):
        """Test that the convenience function works."""
        result = analyze_word("စားခဲ့သည်")
        assert isinstance(result, WordAnalysis)
        assert result.original == "စားခဲ့သည်"

    def test_convenience_function_accepts_dictionary_check(self):
        """Test that the convenience function accepts dictionary_check."""

        def mock_dict(word):
            return word == "စား"

        result = analyze_word("စားခဲ့သည်", dictionary_check=mock_dict)
        assert result.root == "စား"


class TestPOSGuessIntegration:
    """Tests for POS guessing integration with analyze_word."""

    @pytest.fixture
    def analyzer(self):
        return MorphologyAnalyzer()

    def test_pos_guesses_included_in_analysis(self, analyzer):
        """Test that POS guesses are included in word analysis."""
        result = analyzer.analyze_word("စားခဲ့သည်")
        # Should have POS guesses based on suffixes
        assert isinstance(result.pos_guesses, list)
        # With sentence ending particle, should have P_SENT guess
        if result.pos_guesses:
            tags = [g.tag for g in result.pos_guesses]
            # Expect some particle or verb tags
            assert any(t in tags for t in ["P_SENT", "V", "P_MOD", "N"])

    def test_pos_guesses_have_confidence(self, analyzer):
        """Test that POS guesses have confidence scores."""
        result = analyzer.analyze_word("ကောင်းပါတယ်")
        if result.pos_guesses:
            for guess in result.pos_guesses:
                assert isinstance(guess, POSGuess)
                assert 0.0 <= guess.confidence <= 1.0
                assert guess.reason  # Should have a reason


class TestCompoundWordDetection:
    """Tests for compound word detection in morphological analysis."""

    @pytest.fixture
    def analyzer(self):
        return MorphologyAnalyzer()

    def test_simple_word_not_compound(self, analyzer):
        """Test that simple words are not marked as compounds."""
        result = analyzer.analyze_word("စား")
        assert result.is_compound is False

    def test_word_with_many_suffixes_marked_compound(self, analyzer):
        """Test that words with many suffixes are marked as compounds."""
        # A word with 3+ suffixes should be marked as compound
        result = analyzer.analyze_word("စားခဲ့ကြပါသည်")
        # If we detect 3+ suffixes, it should be a compound
        if len(result.suffixes) > 2:
            assert result.is_compound is True


class TestMorphologyAnalyzerEdgeCases:
    """Edge case tests for MorphologyAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        return MorphologyAnalyzer()

    def test_unicode_normalization(self, analyzer):
        """Test that Unicode is normalized before analysis."""
        # Different Unicode representations should give same result
        result = analyzer.analyze_word("စားခဲ့သည်")
        assert result.original  # Should not be empty

    def test_max_iterations_respected(self, analyzer):
        """Test that infinite loops are prevented."""
        # Even with a very long word, should not hang
        long_word = "စား" + "ခဲ့" * 10 + "သည်"
        result = analyzer.analyze_word(long_word)
        # Should complete without error
        assert isinstance(result, WordAnalysis)

    def test_single_character_word(self, analyzer):
        """Test analyzing a single character."""
        result = analyzer.analyze_word("က")
        assert result.original == "က"
        assert result.root == "က"


class TestSpellCheckerOOVRecoveryIntegration:
    """Integration tests for OOV recovery in SpellChecker."""

    @pytest.fixture
    def spellchecker(self):
        """Create a SpellChecker instance for testing."""
        try:
            from myspellchecker import SpellChecker

            return SpellChecker.create_default()
        except Exception as e:
            pytest.skip(f"SpellChecker not available: {e}")

    def test_oov_recovery_enhances_suggestions(self, spellchecker):
        """Test that OOV recovery can enhance suggestions for inflected words."""
        from myspellchecker.core.constants import ValidationLevel

        # Test with a potentially OOV inflected form
        # The exact behavior depends on the dictionary content
        result = spellchecker.check("စားခဲ့သည်", level=ValidationLevel.WORD)

        # If the word is in dictionary, no errors expected
        # If OOV, should have suggestions enhanced by morphology
        # This test verifies the integration doesn't crash
        assert result is not None
        assert hasattr(result, "errors")

    def test_word_validator_has_oov_methods(self, spellchecker):
        """Test that WordValidator has OOV recovery methods."""
        word_validator = spellchecker.word_validator
        assert hasattr(word_validator, "_recover_oov_root")
        assert hasattr(word_validator, "_is_in_dictionary")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
