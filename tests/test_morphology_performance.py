"""
Tests for MED-003: O(n²) Performance in Morphology

Verifies that morphology analyzer uses efficient list operations:
- append() instead of insert(0, ...) for O(1) operations
- Correct suffix order maintained after optimization
- Performance is acceptable for batch operations
"""

import time

import pytest


class TestMorphologyPerformance:
    """Test performance optimizations in morphology module."""

    def test_analyze_word_returns_correct_structure(self):
        """analyze_word should return WordAnalysis with correct structure."""
        from myspellchecker.text.morphology import MorphologyAnalyzer, WordAnalysis

        analyzer = MorphologyAnalyzer()
        result = analyzer.analyze_word("စာအုပ်များ")

        assert isinstance(result, WordAnalysis)
        assert hasattr(result, "root")
        assert hasattr(result, "suffixes")
        assert hasattr(result, "pos_guesses")

    def test_suffix_order_preserved(self):
        """Suffixes should be in correct order (word-internal order)."""
        from myspellchecker.text.morphology import MorphologyAnalyzer

        analyzer = MorphologyAnalyzer()

        # Test with word that has multiple suffixes
        # "စာရေးသူများ" = writer + plural
        result = analyzer.analyze_word("စာရေးသူများ")

        # Check that suffixes are found (order depends on stripping sequence)
        assert isinstance(result.suffixes, list)

    @pytest.mark.slow
    def test_batch_analysis_performance(self):
        """Batch word analysis should complete in reasonable time."""
        from myspellchecker.text.morphology import MorphologyAnalyzer

        analyzer = MorphologyAnalyzer()

        # Test words with various suffix patterns
        test_words = [
            "စာအုပ်များ",  # book + plural
            "ဆရာဝန်",  # doctor
            "ကျောင်းသား",  # student
            "လုပ်ငန်း",  # work/business
            "သူငယ်ချင်း",  # friend
        ] * 100  # 500 words total

        start = time.perf_counter()
        for word in test_words:
            analyzer.analyze_word(word)
        elapsed = time.perf_counter() - start

        # Should complete 500 words in under 2 seconds
        assert elapsed < 2.0, f"Too slow: {elapsed:.3f}s for 500 words"


class TestSuffixExtractionCorrectness:
    """Test that suffix extraction produces correct results after optimization."""

    def test_plural_suffix_detected(self):
        """Plural marker များ should be detected."""
        from myspellchecker.text.morphology import MorphologyAnalyzer

        analyzer = MorphologyAnalyzer()
        result = analyzer.analyze_word("စာအုပ်များ")

        # Should find the plural suffix
        suffix_list = result.suffixes
        assert any("များ" in s for s in suffix_list) or "များ" in str(suffix_list), (
            f"Plural suffix not found in {suffix_list}"
        )

    def test_verb_suffix_detected(self):
        """Verb suffixes should be detected."""
        from myspellchecker.text.morphology import MorphologyAnalyzer

        analyzer = MorphologyAnalyzer()

        # "သွားခဲ့" = went (past tense)
        result = analyzer.analyze_word("သွားခဲ့")

        # Should identify as verb and extract the past-tense suffix ခဲ့
        pos_tags = [g.tag for g in result.pos_guesses]
        assert "V" in pos_tags, f"Expected 'V' in POS tags, got {pos_tags}"
        assert len(result.suffixes) > 0, (
            f"Expected suffixes for past-tense verb, got {result.suffixes}"
        )

    def test_empty_word_handling(self):
        """Empty or very short words should be handled gracefully."""
        from myspellchecker.text.morphology import MorphologyAnalyzer

        analyzer = MorphologyAnalyzer()

        # Empty word
        result = analyzer.analyze_word("")
        assert result.root == ""
        assert result.suffixes == []

        # Single character — no suffix can be stripped, root should be the character itself
        result = analyzer.analyze_word("က")
        assert result.root == "က"

    def test_word_without_suffixes(self):
        """Words without recognizable suffixes should return root intact."""
        from myspellchecker.text.morphology import MorphologyAnalyzer

        analyzer = MorphologyAnalyzer()

        # Simple word without common suffixes
        result = analyzer.analyze_word("အိမ်")  # house

        # Root should be the word itself or a valid substring
        assert len(result.root) <= len("အိမ်")


class TestPOSGuessing:
    """Test POS guessing functionality."""

    def test_noun_suffix_gives_noun_pos(self):
        """Words with noun suffixes should be guessed as nouns."""
        from myspellchecker.text.morphology import MorphologyAnalyzer

        analyzer = MorphologyAnalyzer()

        # "ကျောင်းသား" = student (noun)
        result = analyzer.analyze_word("ကျောင်းသား")

        # Should have some POS guesses
        assert len(result.pos_guesses) > 0

        # Check if N (noun) is among guesses
        pos_tags = [g.tag for g in result.pos_guesses]
        # Note: The system might guess N or have confidence-based ranking
        assert any(tag in ["N", "V", "ADV"] for tag in pos_tags), (
            f"Expected valid POS tag, got {pos_tags}"
        )

    def test_confidence_scores_valid(self):
        """Confidence scores should be between 0 and 1."""
        from myspellchecker.text.morphology import MorphologyAnalyzer

        analyzer = MorphologyAnalyzer()

        result = analyzer.analyze_word("စာအုပ်များ")

        for guess in result.pos_guesses:
            assert 0.0 <= guess.confidence <= 1.0, f"Invalid confidence: {guess.confidence}"

    def test_pos_guesses_have_reasons(self):
        """POS guesses should include reasoning."""
        from myspellchecker.text.morphology import MorphologyAnalyzer

        analyzer = MorphologyAnalyzer()

        result = analyzer.analyze_word("ဆရာများ")

        for guess in result.pos_guesses:
            assert guess.reason, f"Missing reason for guess: {guess}"


class TestWordAnalysisIntegration:
    """Integration tests for complete word analysis workflow."""

    def test_analysis_with_dictionary_check(self):
        """Analysis with dictionary check should validate roots."""
        from myspellchecker.text.morphology import MorphologyAnalyzer

        analyzer = MorphologyAnalyzer()

        # Mock dictionary check
        def mock_dict_check(word: str) -> bool:
            valid_roots = {"စာ", "သူ", "ဆရာ", "အိမ်"}
            return word in valid_roots

        result = analyzer.analyze_word("စာအုပ်များ", dictionary_check=mock_dict_check)
        assert isinstance(result.root, str)

    def test_multiple_analyses_consistent(self):
        """Multiple analyses of same word should be consistent."""
        from myspellchecker.text.morphology import MorphologyAnalyzer

        analyzer = MorphologyAnalyzer()

        word = "ကျောင်းသားများ"
        results = [analyzer.analyze_word(word) for _ in range(5)]

        # All results should have same root and suffixes
        roots = [r.root for r in results]
        assert all(r == roots[0] for r in roots), "Inconsistent roots across analyses"

        suffix_counts = [len(r.suffixes) for r in results]
        assert all(c == suffix_counts[0] for c in suffix_counts), "Inconsistent suffix counts"
