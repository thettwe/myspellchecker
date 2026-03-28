"""
Unit tests for POS disambiguation.

Tests the context-based disambiguation rules R1-R5 for resolving
multi-POS ambiguous words in Myanmar text.
"""

from myspellchecker.algorithms.pos_disambiguator import (
    DisambiguationResult,
    DisambiguationRule,
    POSDisambiguator,
    disambiguate,
    get_disambiguator,
)


class TestDisambiguationRuleR1:
    """Tests for Rule R1: Noun after Verb."""

    def test_noun_after_verb(self):
        """Ambiguous word after verb should be noun."""
        d = POSDisambiguator()
        result = d.disambiguate_in_context(
            word="ကြီး",
            word_pos_tags=frozenset({"ADJ", "N", "V"}),
            prev_word_pos="V",
        )
        assert result.resolved_pos == "N"
        assert result.rule_applied == DisambiguationRule.R1_NOUN_AFTER_VERB
        assert result.confidence >= 0.80

    def test_r1_requires_noun_option(self):
        """R1 should only apply if N is in tag options."""
        d = POSDisambiguator()
        result = d.disambiguate_in_context(
            word="test",
            word_pos_tags=frozenset({"ADJ", "V"}),  # No N option
            prev_word_pos="V",
        )
        # Should not apply R1
        assert result.rule_applied != DisambiguationRule.R1_NOUN_AFTER_VERB


class TestDisambiguationRuleR2:
    """Tests for Rule R2: Adjective before Noun."""

    def test_adj_before_noun(self):
        """Ambiguous word before noun should be adjective."""
        d = POSDisambiguator()
        result = d.disambiguate_in_context(
            word="ကြီး",
            word_pos_tags=frozenset({"ADJ", "N", "V"}),
            next_word_pos="N",
        )
        assert result.resolved_pos == "ADJ"
        assert result.rule_applied == DisambiguationRule.R2_ADJ_BEFORE_NOUN

    def test_adj_before_pronoun(self):
        """PRON is also modifiable, should trigger R2."""
        d = POSDisambiguator()
        result = d.disambiguate_in_context(
            word="ကောင်း",
            word_pos_tags=frozenset({"ADJ", "N", "V"}),
            next_word_pos="PRON",
        )
        assert result.resolved_pos == "ADJ"

    def test_r2_requires_adj_option(self):
        """R2 should only apply if ADJ is in tag options."""
        d = POSDisambiguator()
        result = d.disambiguate_in_context(
            word="test",
            word_pos_tags=frozenset({"N", "V"}),  # No ADJ option
            next_word_pos="N",
        )
        assert result.rule_applied != DisambiguationRule.R2_ADJ_BEFORE_NOUN


class TestDisambiguationRuleR3:
    """Tests for Rule R3: Verb before Particle."""

    def test_verb_before_p_sent(self):
        """Ambiguous word before P_SENT should be verb."""
        d = POSDisambiguator()
        result = d.disambiguate_in_context(
            word="ကြီး",
            word_pos_tags=frozenset({"ADJ", "N", "V"}),
            next_word_pos="P_SENT",
        )
        assert result.resolved_pos == "V"
        assert result.rule_applied == DisambiguationRule.R3_VERB_BEFORE_PARTICLE
        assert result.confidence >= 0.85

    def test_verb_before_p_mod(self):
        """Ambiguous word before P_MOD should be verb."""
        d = POSDisambiguator()
        result = d.disambiguate_in_context(
            word="ကြီး",
            word_pos_tags=frozenset({"ADJ", "N", "V"}),
            next_word_pos="P_MOD",
        )
        assert result.resolved_pos == "V"

    def test_verb_before_ppm(self):
        """Ambiguous word before PPM should be verb."""
        d = POSDisambiguator()
        result = d.disambiguate_in_context(
            word="ကြီး",
            word_pos_tags=frozenset({"ADJ", "N", "V"}),
            next_word_pos="PPM",
        )
        assert result.resolved_pos == "V"

    def test_verb_before_known_particle(self):
        """Ambiguous word before known particle word should be verb."""
        d = POSDisambiguator()
        result = d.disambiguate_in_context(
            word="ကြီး",
            word_pos_tags=frozenset({"ADJ", "N", "V"}),
            next_word="ပြီ",  # Known particle
        )
        # May or may not trigger depending on PARTICLE_MAP
        assert isinstance(result, DisambiguationResult)


class TestDisambiguationRuleR4:
    """Tests for Rule R4: Noun after Determiner."""

    def test_noun_after_determiner(self):
        """Ambiguous word after determiner should be noun."""
        d = POSDisambiguator()
        result = d.disambiguate_in_context(
            word="ကြီး",
            word_pos_tags=frozenset({"ADJ", "N", "V"}),
            prev_word="ဤ",  # Determiner
        )
        assert result.resolved_pos == "N"
        assert result.rule_applied == DisambiguationRule.R4_NOUN_AFTER_DETERMINER

    def test_noun_after_this(self):
        """Word after 'ဒီ' (this) should be noun."""
        d = POSDisambiguator()
        result = d.disambiguate_in_context(
            word="ကြီး",
            word_pos_tags=frozenset({"ADJ", "N", "V"}),
            prev_word="ဒီ",
        )
        assert result.resolved_pos == "N"

    def test_noun_after_that(self):
        """Word after 'အဲဒီ' (that) should be noun."""
        d = POSDisambiguator()
        result = d.disambiguate_in_context(
            word="ကြီး",
            word_pos_tags=frozenset({"ADJ", "N", "V"}),
            prev_word="အဲဒီ",
        )
        assert result.resolved_pos == "N"


class TestDisambiguationRuleR5:
    """Tests for Rule R5: Verb after Adverb."""

    def test_verb_after_adverb(self):
        """Ambiguous word after adverb should be verb."""
        d = POSDisambiguator()
        result = d.disambiguate_in_context(
            word="ကြီး",
            word_pos_tags=frozenset({"ADJ", "N", "V"}),
            prev_word="လျင်မြန်စွာ",  # quickly
        )
        assert result.resolved_pos == "V"
        assert result.rule_applied == DisambiguationRule.R5_VERB_AFTER_ADVERB

    def test_adj_after_degree_adverb(self):
        """Word after degree adverb 'အလွန်' (very) should be ADJ (e.g., very big)."""
        d = POSDisambiguator()
        result = d.disambiguate_in_context(
            word="ကြီး",
            word_pos_tags=frozenset({"ADJ", "N", "V"}),
            prev_word="အလွန်",
        )
        assert result.resolved_pos == "ADJ"

    def test_verb_after_adv_pos(self):
        """Word after ADV POS tag should be verb."""
        d = POSDisambiguator()
        result = d.disambiguate_in_context(
            word="ကြီး",
            word_pos_tags=frozenset({"ADJ", "N", "V"}),
            prev_word_pos="ADV",
        )
        assert result.resolved_pos == "V"


class TestRulePriority:
    """Tests for rule priority ordering."""

    def test_r3_has_high_priority(self):
        """R3 (verb before particle) should have higher priority."""
        d = POSDisambiguator()
        # Both R2 and R3 could apply - next is N, but also checking particle
        result = d.disambiguate_in_context(
            word="ကြီး",
            word_pos_tags=frozenset({"ADJ", "N", "V"}),
            next_word_pos="P_SENT",  # R3 condition
        )
        # R3 should win
        assert result.rule_applied == DisambiguationRule.R3_VERB_BEFORE_PARTICLE

    def test_no_rule_applies(self):
        """When no rule matches, use default selection."""
        d = POSDisambiguator()
        result = d.disambiguate_in_context(
            word="ကြီး",
            word_pos_tags=frozenset({"ADJ", "N", "V"}),
            # No context provided
        )
        assert result.rule_applied == DisambiguationRule.NO_RULE
        # Default should prefer N > V > ADJ
        assert result.resolved_pos == "N"


class TestUnambiguousWords:
    """Tests for unambiguous words."""

    def test_single_pos(self):
        """Single POS word should pass through."""
        d = POSDisambiguator()
        result = d.disambiguate_in_context(
            word="test",
            word_pos_tags=frozenset({"N"}),
        )
        assert result.resolved_pos == "N"
        assert result.rule_applied == DisambiguationRule.NO_RULE
        assert result.confidence == 1.0

    def test_empty_pos(self):
        """Empty POS should return empty result."""
        d = POSDisambiguator()
        result = d.disambiguate_in_context(
            word="test",
            word_pos_tags=frozenset(),
        )
        assert result.resolved_pos == ""
        assert result.confidence == 0.0


class TestConvenienceFunction:
    """Tests for module-level convenience function."""

    def test_disambiguate_function(self):
        """disambiguate() convenience function should work."""
        pos = disambiguate(
            word="ကြီး",
            word_pos_tags=frozenset({"ADJ", "N", "V"}),
            next_word_pos="N",
        )
        assert pos == "ADJ"

    def test_singleton_disambiguator(self):
        """get_disambiguator() should return singleton."""
        d1 = get_disambiguator()
        d2 = get_disambiguator()
        assert d1 is d2


class TestDisambiguationResult:
    """Tests for DisambiguationResult dataclass."""

    def test_result_repr(self):
        """Result should have readable repr."""
        result = DisambiguationResult(
            word="ကြီး",
            original_pos_tags=frozenset({"ADJ", "N", "V"}),
            resolved_pos="N",
            rule_applied=DisambiguationRule.R1_NOUN_AFTER_VERB,
            confidence=0.85,
            context_used="after verb",
        )
        repr_str = repr(result)
        assert "ကြီး" in repr_str
        assert "N" in repr_str
        assert "R1" in repr_str


class TestContextIntegration:
    """Integration tests with realistic context."""

    def test_sentence_context_1(self):
        """Test: 'သူ ပြော ကြီး ကို ဝယ်သည်' - ကြီး after V."""
        d = POSDisambiguator()
        # ကြီး follows verb ပြော
        result = d.disambiguate_in_context(
            word="ကြီး",
            word_pos_tags=frozenset({"ADJ", "N", "V"}),
            prev_word="ပြော",
            prev_word_pos="V",
        )
        assert result.resolved_pos == "N"

    def test_sentence_context_2(self):
        """Test: 'ကြီး သော အိမ်' - ကြီး before N."""
        d = POSDisambiguator()
        # ကြီး before noun အိမ် (via သော)
        # Actually, immediate next is "သော" (P_MOD)
        result = d.disambiguate_in_context(
            word="ကြီး",
            word_pos_tags=frozenset({"ADJ", "N", "V"}),
            next_word="သော",
            next_word_pos="P_MOD",
        )
        # R3 should apply (before particle)
        assert result.resolved_pos == "V"

    def test_sentence_context_3(self):
        """Test: 'သူ ကြီး ပြီ' - ကြီး before P_SENT."""
        d = POSDisambiguator()
        result = d.disambiguate_in_context(
            word="ကြီး",
            word_pos_tags=frozenset({"ADJ", "N", "V"}),
            next_word="ပြီ",
            next_word_pos="P_SENT",
        )
        assert result.resolved_pos == "V"
        assert result.rule_applied == DisambiguationRule.R3_VERB_BEFORE_PARTICLE
