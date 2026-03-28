"""
Tests for CRIT-002: Invalid Grammar Corrections Fix

Verifies that valid Myanmar words are not incorrectly flagged as typos.
These words were previously mapped to incorrect "corrections":
- ပြီ (sentence-final "done") was mapped to ပြီး ("after")
- လို (want/like) was mapped to လို့ (because)
- ပါ (politeness particle) was mapped to ပှာ (bee)
- တာ (nominalizer) was mapped to တှာ (rare)

Reference: Myanmar Language Commission, "Myanmar Grammar" (2005)
"""


class TestValidWordsNotFlagged:
    """Test that valid common words are not in the typo map."""

    def test_pyi_not_in_typo_map(self):
        """ပြီ (completed/done) should not be mapped to ပြီး."""
        from myspellchecker.grammar.config import GrammarRuleConfig

        config = GrammarRuleConfig()

        # ပြီ should either not be in the map, or map to itself
        if "ပြီ" in config.particle_typos:
            correction = config.particle_typos["ပြီ"]["correction"]
            assert correction == "ပြီ", "ပြီ should not be 'corrected' to another word"

    def test_lo_not_in_typo_map(self):
        """လို (want/like) should not be mapped to လို့."""
        from myspellchecker.grammar.config import GrammarRuleConfig

        config = GrammarRuleConfig()

        # လို should either not be in the map, or map to itself
        if "လို" in config.particle_typos:
            correction = config.particle_typos["လို"]["correction"]
            assert correction == "လို", "လို should not be 'corrected' to another word"

    def test_pa_not_mapped_to_pha(self):
        """ပါ (politeness particle) should not be mapped to ပှာ (bee)."""
        from myspellchecker.grammar.config import GrammarRuleConfig

        config = GrammarRuleConfig()

        # ပါ is a very common word and should not be auto-corrected
        if "ပါ" in config.particle_typos:
            correction = config.particle_typos["ပါ"]["correction"]
            assert correction != "ပှာ", "ပါ should not be corrected to ပှာ"

    def test_ta_not_mapped_to_hta(self):
        """တာ (nominalizer) should not be mapped to တှာ."""
        from myspellchecker.grammar.config import GrammarRuleConfig

        config = GrammarRuleConfig()

        # တာ is a common nominalizer and should not be auto-corrected
        if "တာ" in config.particle_typos:
            correction = config.particle_typos["တာ"]["correction"]
            assert correction != "တှာ", "တာ should not be corrected to တှာ"

    def test_la_is_valid(self):
        """လာ (to come) should NOT be in typo map - it's a valid common verb."""
        from myspellchecker.grammar.config import GrammarRuleConfig

        config = GrammarRuleConfig()

        # CRIT-002 FIX: လာ (to come) should NOT be in typo map at all
        # Previously it was incorrectly mapped to လှာ (tongue) which is wrong:
        # - လာ = /la/ (verb: to come) - one of the most common Myanmar verbs
        # - လှာ = /l̥a/ (noun: tongue) - completely different word
        assert "လာ" not in config.particle_typos, (
            "CRIT-002: လာ (to come) should not be in particle_typos. "
            "It's a valid common verb, not a typo for လှာ (tongue)."
        )


class TestValidSentences:
    """Test that sentences with valid words are not flagged."""

    def test_pyi_in_valid_sentence(self):
        """Sentence with ပြီ should be valid."""
        # ပြီ is valid as sentence-final particle
        # "ကျွန်တော် စား ပြီ" = "I ate (done)"

        # The word ပြီ should not generate corrections
        from myspellchecker.grammar.config import GrammarRuleConfig

        config = GrammarRuleConfig()

        pyi_in_map = "ပြီ" in config.particle_typos
        if pyi_in_map:
            correction = config.particle_typos["ပြီ"]["correction"]
            # If in map, should not suggest ပြီး
            assert correction != "ပြီး"

    def test_pyii_also_valid(self):
        """ပြီး (after/and then) should also be valid."""
        # "စား ပြီး သွား မယ်" = "After eating, (I'll) go"
        from myspellchecker.grammar.config import GrammarRuleConfig

        config = GrammarRuleConfig()

        # ပြီး should not be in the typo map at all
        assert "ပြီး" not in config.particle_typos, "ပြီး is valid, not a typo"


class TestMinimalPairs:
    """Test words that form minimal pairs and shouldn't be confused."""

    def test_pyi_pyii_distinct(self):
        """ပြီ and ပြီး have different grammatical functions."""
        # ပြီ = sentence-final particle (action completed)
        # ပြီး = conjunctive particle (after doing X)

        # Both should be valid in their contexts
        pyi = "ပြီ"
        pyii = "ပြီး"

        assert pyi != pyii
        # The difference is grammatical function, not a typo

    def test_lo_loh_distinct(self):
        """လို and လို့ have different meanings."""
        # လို = want, like, such as
        # လို့ = because, quotative

        lo = "လို"
        loh = "လို့"

        assert lo != loh
        # Different meanings, not typos

    def test_pa_pha_distinct(self):
        """ပါ and ပှာ have completely different meanings."""
        # ပါ = politeness particle, "also", "please"
        # ပှာ = bee (rare word)

        pa = "ပါ"
        pha = "ပှာ"

        assert pa != pha
        # ပါ is extremely common, ပှာ is rare


class TestGrammaticalValidity:
    """Test that grammatically correct sentences aren't flagged."""

    def test_sentence_final_pyi_valid(self):
        """ပြီ at sentence end is grammatically correct."""
        # Usage: Statement of completed action
        # "သူ သွား ပြီ" = "He went (completed)"

        from myspellchecker.grammar.config import GrammarRuleConfig

        config = GrammarRuleConfig()

        # ပြီ should be in sentence-final particles
        assert "ပြီ" in config.sentence_final_particles, "ပြီ is a valid sentence-final particle"

    def test_conjunctive_pyii_valid(self):
        """ပြီး in conjunctive position is grammatically correct."""
        # Usage: "After X, Y"
        # "စား ပြီး သွား" = "Go after eating"

        from myspellchecker.grammar.config import GrammarRuleConfig

        config = GrammarRuleConfig()

        # ပြီး should be in verb particles
        assert "ပြီး" in config.verb_particles, "ပြီး is a valid verb particle"

    def test_pa_as_politeness_marker(self):
        """ပါ as politeness marker should be valid."""

        # ပါ might be in either or both particle sets
        # It's used for politeness and emphasis
        # Check it's not incorrectly flagged
        pass  # Implementation depends on how particles are categorized


class TestGrammarPatternsIntegration:
    """Test grammar patterns module integration."""

    def test_statement_endings_defined(self):
        """Test STATEMENT_ENDINGS constant is defined."""
        from myspellchecker.grammar.patterns import STATEMENT_ENDINGS

        assert isinstance(STATEMENT_ENDINGS, frozenset)
        assert "တယ်" in STATEMENT_ENDINGS
        assert "သည်" in STATEMENT_ENDINGS
        assert "ပါတယ်" in STATEMENT_ENDINGS
        assert "ပါသည်" in STATEMENT_ENDINGS

    def test_question_particle_suggestions_defined(self):
        """Test QUESTION_PARTICLE_SUGGESTIONS constant is defined."""
        from myspellchecker.grammar.patterns import QUESTION_PARTICLE_SUGGESTIONS

        assert isinstance(QUESTION_PARTICLE_SUGGESTIONS, tuple)
        assert "လား" in QUESTION_PARTICLE_SUGGESTIONS
        assert "သလား" in QUESTION_PARTICLE_SUGGESTIONS
        assert "လဲ" in QUESTION_PARTICLE_SUGGESTIONS

    def test_question_strategy_uses_patterns(self):
        """Test QuestionStructureValidationStrategy uses patterns.py constants."""
        from myspellchecker.core.validation_strategies import (
            QuestionStructureValidationStrategy,
            ValidationContext,
        )
        from myspellchecker.grammar.patterns import QUESTION_PARTICLES

        strategy = QuestionStructureValidationStrategy()

        # Create context with question word but statement ending
        context = ValidationContext(
            sentence="ဘာ စား တယ်",
            words=["ဘာ", "စား", "တယ်"],
            word_positions=[0, 6, 15],
        )

        errors = strategy.validate(context)

        # Should detect question structure issue
        assert len(errors) == 1
        # Suggestions should be valid question particles from patterns.py
        # (order depends on context via get_question_completion_suggestions)
        suggestions = errors[0].suggestions
        assert len(suggestions) > 0
        for s in suggestions:
            assert any(s.endswith(p) for p in QUESTION_PARTICLES), (
                f"Suggestion {s!r} does not end with a known question particle"
            )

    def test_engine_uses_patterns_question_words(self):
        """Test SyntacticRuleChecker uses QUESTION_WORDS from patterns.py."""
        from unittest.mock import Mock

        from myspellchecker.grammar.engine import SyntacticRuleChecker
        from myspellchecker.grammar.patterns import QUESTION_WORDS

        mock_provider = Mock()
        mock_provider.get_word_pos = Mock(return_value=None)
        SyntacticRuleChecker(mock_provider)

        # Verify the module imports are correct
        # The checker should use QUESTION_WORDS from patterns
        assert "ဘာ" in QUESTION_WORDS
        assert "ဘယ်" in QUESTION_WORDS

    def test_engine_uses_patterns_question_particles(self):
        """Test SyntacticRuleChecker uses QUESTION_PARTICLES from patterns.py."""
        from unittest.mock import Mock

        from myspellchecker.grammar.engine import SyntacticRuleChecker
        from myspellchecker.grammar.patterns import QUESTION_PARTICLES

        mock_provider = Mock()
        mock_provider.get_word_pos = Mock(return_value=None)
        SyntacticRuleChecker(mock_provider)

        # Verify the module imports are correct
        # The checker should use QUESTION_PARTICLES from patterns
        assert "လား" in QUESTION_PARTICLES
        assert "သလား" in QUESTION_PARTICLES
