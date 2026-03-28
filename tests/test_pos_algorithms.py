"""
Unit tests for POS algorithm modules.

Tests cover:
- pos_inference.py: POSInferenceEngine, POSInferenceResult, BatchInferenceStats
- pos_disambiguator.py: POSDisambiguator, DisambiguationResult, disambiguation rules

All tests use the actual implementations (no heavy ML dependencies required).
"""

import pytest

# ============================================================================
# POS Inference Tests
# ============================================================================


class TestInferenceSource:
    """Test InferenceSource enum."""

    def test_all_sources_defined(self):
        """Test all inference sources are defined."""
        from myspellchecker.algorithms.pos_inference import InferenceSource

        expected_sources = [
            "DATABASE",
            "AMBIGUOUS_REGISTRY",
            "NUMERAL_DETECTION",
            "PREFIX_PATTERN",
            "PROPER_NOUN_SUFFIX",
            "SUFFIX_PATTERN",
            "MORPHOLOGICAL",
            "UNKNOWN",
        ]

        for source in expected_sources:
            assert hasattr(InferenceSource, source)

    def test_source_values(self):
        """Test inference source values."""
        from myspellchecker.algorithms.pos_inference import InferenceSource

        assert InferenceSource.DATABASE.value == "database"
        assert InferenceSource.NUMERAL_DETECTION.value == "numeral_detection"
        assert InferenceSource.UNKNOWN.value == "unknown"


class TestPOSInferenceResult:
    """Test POSInferenceResult dataclass."""

    def test_result_creation_with_defaults(self):
        """Test POSInferenceResult creation with default values."""
        from myspellchecker.algorithms.pos_inference import (
            InferenceSource,
            POSInferenceResult,
        )

        result = POSInferenceResult(word="test")

        assert result.word == "test"
        assert result.inferred_pos is None
        assert result.all_pos_tags == frozenset()
        assert result.confidence == 0.0
        assert result.source == InferenceSource.UNKNOWN
        assert result.is_ambiguous is False
        assert result.requires_context is False
        assert result.details == {}

    def test_result_creation_with_values(self):
        """Test POSInferenceResult creation with custom values."""
        from myspellchecker.algorithms.pos_inference import (
            InferenceSource,
            POSInferenceResult,
        )

        result = POSInferenceResult(
            word="ကြီး",
            inferred_pos="N",
            all_pos_tags=frozenset({"N", "V", "ADJ"}),
            confidence=0.85,
            source=InferenceSource.AMBIGUOUS_REGISTRY,
            is_ambiguous=True,
            requires_context=True,
            details={"possible_tags": ["ADJ", "N", "V"]},
        )

        assert result.word == "ကြီး"
        assert result.inferred_pos == "N"
        assert result.all_pos_tags == frozenset({"N", "V", "ADJ"})
        assert result.confidence == 0.85
        assert result.source == InferenceSource.AMBIGUOUS_REGISTRY
        assert result.is_ambiguous is True
        assert result.requires_context is True

    def test_to_multi_pos_string_empty(self):
        """Test to_multi_pos_string with no tags."""
        from myspellchecker.algorithms.pos_inference import POSInferenceResult

        result = POSInferenceResult(word="test")
        assert result.to_multi_pos_string() is None

    def test_to_multi_pos_string_single_tag(self):
        """Test to_multi_pos_string with single tag."""
        from myspellchecker.algorithms.pos_inference import POSInferenceResult

        result = POSInferenceResult(word="test", all_pos_tags=frozenset({"N"}))
        assert result.to_multi_pos_string() == "N"

    def test_to_multi_pos_string_multiple_tags(self):
        """Test to_multi_pos_string with multiple tags (sorted)."""
        from myspellchecker.algorithms.pos_inference import POSInferenceResult

        result = POSInferenceResult(word="test", all_pos_tags=frozenset({"V", "N", "ADJ"}))
        # Should be sorted alphabetically
        assert result.to_multi_pos_string() == "ADJ|N|V"

    def test_result_repr(self):
        """Test POSInferenceResult __repr__."""
        from myspellchecker.algorithms.pos_inference import (
            InferenceSource,
            POSInferenceResult,
        )

        result = POSInferenceResult(
            word="test",
            inferred_pos="N",
            all_pos_tags=frozenset({"N"}),
            confidence=0.95,
            source=InferenceSource.NUMERAL_DETECTION,
        )

        repr_str = repr(result)
        assert "word='test'" in repr_str
        assert "pos='N'" in repr_str
        assert "0.95" in repr_str


class TestBatchInferenceStats:
    """Test BatchInferenceStats dataclass."""

    def test_stats_creation_with_defaults(self):
        """Test BatchInferenceStats creation with defaults."""
        from myspellchecker.algorithms.pos_inference import BatchInferenceStats

        stats = BatchInferenceStats()

        assert stats.total_words == 0
        assert stats.inferred_count == 0
        assert stats.ambiguous_count == 0
        assert stats.numeral_count == 0
        assert stats.prefix_count == 0
        assert stats.suffix_count == 0
        assert stats.unknown_count == 0
        assert stats.avg_confidence == 0.0

    def test_stats_repr(self):
        """Test BatchInferenceStats __repr__."""
        from myspellchecker.algorithms.pos_inference import BatchInferenceStats

        stats = BatchInferenceStats(
            total_words=100, inferred_count=80, ambiguous_count=20, unknown_count=10
        )

        repr_str = repr(stats)
        assert "total=100" in repr_str
        assert "inferred=80" in repr_str
        assert "ambiguous=20" in repr_str
        assert "unknown=10" in repr_str


class TestPOSInferenceEngine:
    """Test POSInferenceEngine class."""

    @pytest.fixture
    def engine(self):
        """Create a POSInferenceEngine instance."""
        from myspellchecker.algorithms.pos_inference import POSInferenceEngine

        return POSInferenceEngine()

    def test_engine_initialization(self, engine):
        """Test engine initialization."""
        assert engine is not None
        assert engine.ambiguous_words is not None
        assert engine.prefix_patterns is not None
        assert engine.proper_noun_suffixes is not None

    def test_infer_pos_empty_word(self, engine):
        """Test infer_pos with empty word."""
        result = engine.infer_pos("")
        assert result.word == ""
        assert result.inferred_pos is None

    def test_infer_pos_numeral_digits(self, engine):
        """Test infer_pos with Myanmar numeral digits."""
        from myspellchecker.algorithms.pos_inference import InferenceSource

        # Myanmar digits
        result = engine.infer_pos("၁၂၃")
        assert result.inferred_pos == "NUM"
        assert result.source == InferenceSource.NUMERAL_DETECTION
        assert result.confidence >= 0.95
        assert result.is_ambiguous is False

    def test_infer_pos_batch_empty(self, engine):
        """Test infer_pos_batch with empty list."""
        results, stats = engine.infer_pos_batch([])

        assert len(results) == 0
        assert stats.total_words == 0

    def test_infer_pos_batch_with_existing_pos(self, engine):
        """Test infer_pos_batch skips words with existing POS."""
        words = ["word1", "word2", "word3"]
        existing_pos = {"word1": "N", "word3": "V"}

        results, stats = engine.infer_pos_batch(words, existing_pos)

        # Should only process word2 (the one without existing POS)
        assert stats.total_words == 1

    def test_is_ambiguous_word(self, engine):
        """Test is_ambiguous_word method."""
        # This depends on config, but we can test the method exists
        # and returns a boolean
        result = engine.is_ambiguous_word("ကြီး")
        assert isinstance(result, bool)


class TestPOSInferenceModuleFunctions:
    """Test module-level functions."""

    def test_get_pos_inference_engine_singleton(self):
        """Test get_pos_inference_engine returns singleton."""
        from myspellchecker.algorithms.pos_inference import get_pos_inference_engine

        engine1 = get_pos_inference_engine()
        engine2 = get_pos_inference_engine()

        assert engine1 is engine2

    def test_infer_pos_function(self):
        """Test module-level infer_pos function."""
        from myspellchecker.algorithms.pos_inference import (
            POSInferenceResult,
            infer_pos,
        )

        result = infer_pos("၁၂၃")
        assert isinstance(result, POSInferenceResult)
        assert result.inferred_pos == "NUM"

    def test_infer_pos_batch_function(self):
        """Test module-level infer_pos_batch function."""
        from myspellchecker.algorithms.pos_inference import (
            BatchInferenceStats,
            infer_pos_batch,
        )

        results, stats = infer_pos_batch(["၁၂၃"])

        assert isinstance(stats, BatchInferenceStats)
        assert len(results) == 1


# ============================================================================
# POS Disambiguator Tests
# ============================================================================


class TestDisambiguationRule:
    """Test DisambiguationRule enum."""

    def test_all_rules_defined(self):
        """Test all disambiguation rules are defined."""
        from myspellchecker.algorithms.pos_disambiguator import DisambiguationRule

        expected_rules = [
            "R1_NOUN_AFTER_VERB",
            "R2_ADJ_BEFORE_NOUN",
            "R3_VERB_BEFORE_PARTICLE",
            "R4_NOUN_AFTER_DETERMINER",
            "R5_VERB_AFTER_ADVERB",
            "NO_RULE",
        ]

        for rule in expected_rules:
            assert hasattr(DisambiguationRule, rule)

    def test_rule_values(self):
        """Test rule values."""
        from myspellchecker.algorithms.pos_disambiguator import DisambiguationRule

        assert DisambiguationRule.R1_NOUN_AFTER_VERB.value == "R1"
        assert DisambiguationRule.R2_ADJ_BEFORE_NOUN.value == "R2"
        assert DisambiguationRule.R3_VERB_BEFORE_PARTICLE.value == "R3"
        assert DisambiguationRule.R4_NOUN_AFTER_DETERMINER.value == "R4"
        assert DisambiguationRule.R5_VERB_AFTER_ADVERB.value == "R5"
        assert DisambiguationRule.NO_RULE.value == "none"


class TestDisambiguationResult:
    """Test DisambiguationResult dataclass."""

    def test_result_creation(self):
        """Test DisambiguationResult creation."""
        from myspellchecker.algorithms.pos_disambiguator import (
            DisambiguationResult,
            DisambiguationRule,
        )

        result = DisambiguationResult(
            word="ကြီး",
            original_pos_tags=frozenset({"N", "V", "ADJ"}),
            resolved_pos="N",
            rule_applied=DisambiguationRule.R1_NOUN_AFTER_VERB,
            confidence=0.85,
            context_used="after verb",
        )

        assert result.word == "ကြီး"
        assert result.original_pos_tags == frozenset({"N", "V", "ADJ"})
        assert result.resolved_pos == "N"
        assert result.rule_applied == DisambiguationRule.R1_NOUN_AFTER_VERB
        assert result.confidence == 0.85
        assert result.context_used == "after verb"

    def test_result_repr(self):
        """Test DisambiguationResult __repr__."""
        from myspellchecker.algorithms.pos_disambiguator import (
            DisambiguationResult,
            DisambiguationRule,
        )

        result = DisambiguationResult(
            word="test",
            original_pos_tags=frozenset({"N"}),
            resolved_pos="N",
            rule_applied=DisambiguationRule.NO_RULE,
            confidence=1.0,
            context_used="unambiguous",
        )

        repr_str = repr(result)
        assert "word='test'" in repr_str
        assert "pos='N'" in repr_str
        assert "1.00" in repr_str


class TestDisambiguatorConstants:
    """Test disambiguator module constants."""

    def test_determiners_defined(self):
        """Test DETERMINERS constant."""
        from myspellchecker.algorithms.pos_disambiguator import DETERMINERS

        assert isinstance(DETERMINERS, frozenset)
        assert len(DETERMINERS) > 0
        # Check some expected determiners
        assert "ဤ" in DETERMINERS  # "this"
        assert "ထို" in DETERMINERS  # "that"

    def test_adverb_markers_defined(self):
        """Test ADVERB_MARKERS constant."""
        from myspellchecker.algorithms.pos_disambiguator import ADVERB_MARKERS

        assert isinstance(ADVERB_MARKERS, frozenset)
        assert len(ADVERB_MARKERS) > 0
        assert "အလွန်" in ADVERB_MARKERS  # "very"

    def test_particle_pos_tags_defined(self):
        """Test PARTICLE_POS_TAGS constant."""
        from myspellchecker.algorithms.pos_disambiguator import PARTICLE_POS_TAGS

        assert isinstance(PARTICLE_POS_TAGS, frozenset)
        assert "P_SENT" in PARTICLE_POS_TAGS
        assert "P_MOD" in PARTICLE_POS_TAGS
        assert "PPM" in PARTICLE_POS_TAGS

    def test_verb_tags_defined(self):
        """Test VERB_TAGS constant."""
        from myspellchecker.algorithms.pos_disambiguator import VERB_TAGS

        assert isinstance(VERB_TAGS, frozenset)
        assert "V" in VERB_TAGS
        assert "AUX" in VERB_TAGS


class TestPOSDisambiguator:
    """Test POSDisambiguator class."""

    @pytest.fixture
    def disambiguator(self):
        """Create a POSDisambiguator instance."""
        from myspellchecker.algorithms.pos_disambiguator import POSDisambiguator

        return POSDisambiguator()

    def test_disambiguator_initialization(self, disambiguator):
        """Test disambiguator initialization."""
        assert disambiguator is not None
        assert disambiguator.grammar_config is not None
        assert len(disambiguator._rules) == 5

    def test_disambiguate_unambiguous_word(self, disambiguator):
        """Test disambiguation of unambiguous word (single POS)."""
        from myspellchecker.algorithms.pos_disambiguator import DisambiguationRule

        result = disambiguator.disambiguate_in_context(
            word="test",
            word_pos_tags=frozenset({"N"}),
        )

        assert result.resolved_pos == "N"
        assert result.rule_applied == DisambiguationRule.NO_RULE
        assert result.confidence == 1.0
        assert result.context_used == "unambiguous"

    def test_disambiguate_empty_pos_tags(self, disambiguator):
        """Test disambiguation with empty POS tags."""
        from myspellchecker.algorithms.pos_disambiguator import DisambiguationRule

        result = disambiguator.disambiguate_in_context(
            word="test",
            word_pos_tags=frozenset(),
        )

        assert result.resolved_pos == ""
        assert result.rule_applied == DisambiguationRule.NO_RULE
        assert result.confidence == 0.0

    def test_rule_r1_noun_after_verb(self, disambiguator):
        """Test R1: Noun after Verb rule."""
        from myspellchecker.algorithms.pos_disambiguator import DisambiguationRule

        result = disambiguator.disambiguate_in_context(
            word="ကြီး",
            word_pos_tags=frozenset({"N", "V", "ADJ"}),
            prev_word_pos="V",
        )

        assert result.resolved_pos == "N"
        assert result.rule_applied == DisambiguationRule.R1_NOUN_AFTER_VERB
        assert result.confidence == 0.85

    def test_rule_r2_adj_before_noun(self, disambiguator):
        """Test R2: Adjective before Noun rule."""
        from myspellchecker.algorithms.pos_disambiguator import DisambiguationRule

        result = disambiguator.disambiguate_in_context(
            word="ကြီး",
            word_pos_tags=frozenset({"N", "V", "ADJ"}),
            next_word_pos="N",
        )

        assert result.resolved_pos == "ADJ"
        assert result.rule_applied == DisambiguationRule.R2_ADJ_BEFORE_NOUN
        assert result.confidence == 0.80

    def test_rule_r3_verb_before_particle(self, disambiguator):
        """Test R3: Verb before Particle rule."""
        from myspellchecker.algorithms.pos_disambiguator import DisambiguationRule

        result = disambiguator.disambiguate_in_context(
            word="ကြီး",
            word_pos_tags=frozenset({"N", "V", "ADJ"}),
            next_word_pos="P_SENT",
        )

        assert result.resolved_pos == "V"
        assert result.rule_applied == DisambiguationRule.R3_VERB_BEFORE_PARTICLE
        assert result.confidence == 0.90

    def test_rule_r4_noun_after_determiner(self, disambiguator):
        """Test R4: Noun after Determiner rule."""
        from myspellchecker.algorithms.pos_disambiguator import DisambiguationRule

        result = disambiguator.disambiguate_in_context(
            word="ကြီး",
            word_pos_tags=frozenset({"N", "V", "ADJ"}),
            prev_word="ဤ",  # "this" determiner
        )

        assert result.resolved_pos == "N"
        assert result.rule_applied == DisambiguationRule.R4_NOUN_AFTER_DETERMINER
        assert result.confidence == 0.88

    def test_rule_r5_adj_after_degree_adverb(self, disambiguator):
        """Test R5: ADJ after degree adverb (e.g., အလွန် ကြီး = very big)."""
        from myspellchecker.algorithms.pos_disambiguator import DisambiguationRule

        result = disambiguator.disambiguate_in_context(
            word="ကြီး",
            word_pos_tags=frozenset({"N", "V", "ADJ"}),
            prev_word="အလွန်",  # "very" - degree adverb
        )

        assert result.resolved_pos == "ADJ"
        assert result.rule_applied == DisambiguationRule.R5_VERB_AFTER_ADVERB
        assert result.confidence == 0.85

    def test_rule_r5_verb_after_manner_adverb(self, disambiguator):
        """Test R5: Verb after manner adverb (e.g., လျင်မြန်စွာ ရေး = quickly write)."""
        from myspellchecker.algorithms.pos_disambiguator import DisambiguationRule

        result = disambiguator.disambiguate_in_context(
            word="ရေး",
            word_pos_tags=frozenset({"N", "V"}),
            prev_word="လျင်မြန်စွာ",  # "quickly" - manner adverb
        )

        assert result.resolved_pos == "V"
        assert result.rule_applied == DisambiguationRule.R5_VERB_AFTER_ADVERB
        assert result.confidence == 0.85

    def test_no_rule_matches_uses_default(self, disambiguator):
        """Test default POS selection when no rule matches."""
        from myspellchecker.algorithms.pos_disambiguator import DisambiguationRule

        result = disambiguator.disambiguate_in_context(
            word="ကြီး",
            word_pos_tags=frozenset({"N", "V", "ADJ"}),
            # No context that triggers any rule
        )

        # Should use default selection (N > V > ADJ)
        assert result.resolved_pos == "N"
        assert result.rule_applied == DisambiguationRule.NO_RULE
        assert result.confidence == 0.5

    def test_rule_priority_r3_over_r1(self, disambiguator):
        """Test R3 has priority over R1."""
        from myspellchecker.algorithms.pos_disambiguator import DisambiguationRule

        # Both R1 and R3 could apply, but R3 has higher priority
        result = disambiguator.disambiguate_in_context(
            word="ကြီး",
            word_pos_tags=frozenset({"N", "V"}),
            prev_word_pos="V",  # R1 could apply
            next_word_pos="P_SENT",  # R3 could apply
        )

        # R3 should win
        assert result.resolved_pos == "V"
        assert result.rule_applied == DisambiguationRule.R3_VERB_BEFORE_PARTICLE

    def test_select_default_pos_priority(self, disambiguator):
        """Test _select_default_pos priority order."""
        # N should have highest priority for default
        result = disambiguator._select_default_pos(frozenset({"ADJ", "V", "N"}))
        assert result == "N"


class TestDisambiguatorModuleFunctions:
    """Test module-level functions."""

    def test_get_disambiguator_singleton(self):
        """Test get_disambiguator returns singleton (thread-safe)."""
        from myspellchecker.algorithms.pos_disambiguator import get_disambiguator

        d1 = get_disambiguator()
        d2 = get_disambiguator()

        assert d1 is d2

    def test_disambiguate_function(self):
        """Test module-level disambiguate function."""
        from myspellchecker.algorithms.pos_disambiguator import disambiguate

        # With R1 context
        result = disambiguate(
            word="ကြီး",
            word_pos_tags=frozenset({"N", "V"}),
            prev_word_pos="V",
        )

        assert result == "N"

    def test_disambiguate_function_with_next_pos(self):
        """Test disambiguate function with next_word_pos."""
        from myspellchecker.algorithms.pos_disambiguator import disambiguate

        # With R3 context
        result = disambiguate(
            word="ကြီး",
            word_pos_tags=frozenset({"N", "V"}),
            next_word_pos="P_SENT",
        )

        assert result == "V"


class TestPOSAlgorithmsEdgeCases:
    """Test edge cases for both modules."""

    def test_inference_with_single_char_word(self):
        """Test inference with single character word."""
        from myspellchecker.algorithms.pos_inference import POSInferenceEngine

        engine = POSInferenceEngine()
        result = engine.infer_pos("က")

        # Should not crash, may return unknown
        assert result is not None

    def test_disambiguation_with_all_rules_not_applicable(self):
        """Test disambiguation when word doesn't have required POS for any rule."""
        from myspellchecker.algorithms.pos_disambiguator import (
            DisambiguationRule,
            POSDisambiguator,
        )

        disambiguator = POSDisambiguator()

        # Only CONJ and INT - none of the rules apply to these
        result = disambiguator.disambiguate_in_context(
            word="test",
            word_pos_tags=frozenset({"CONJ", "INT"}),
            prev_word_pos="V",  # R1 needs N in tags
            next_word_pos="N",  # R2 needs ADJ in tags
        )

        assert result.rule_applied == DisambiguationRule.NO_RULE
        # Should select first in sorted order if no priority matches
        assert result.resolved_pos in {"CONJ", "INT"}

    def test_inference_batch_with_large_list(self):
        """Test batch inference with many words."""
        from myspellchecker.algorithms.pos_inference import POSInferenceEngine

        engine = POSInferenceEngine()
        words = ["word"] * 100

        results, stats = engine.infer_pos_batch(words)

        assert stats.total_words == 100

    def test_disambiguation_r1_needs_n_in_tags(self):
        """Test R1 only applies if N is in possible tags."""
        from myspellchecker.algorithms.pos_disambiguator import (
            DisambiguationRule,
            POSDisambiguator,
        )

        disambiguator = POSDisambiguator()

        # Prev word is V, but N is not in tags
        result = disambiguator.disambiguate_in_context(
            word="test",
            word_pos_tags=frozenset({"V", "ADJ"}),  # No N
            prev_word_pos="V",
        )

        # R1 should not apply since N is not an option
        assert result.rule_applied != DisambiguationRule.R1_NOUN_AFTER_VERB

    def test_disambiguation_r3_with_pppm(self):
        """Test R3 applies with PPM particle tag."""
        from myspellchecker.algorithms.pos_disambiguator import (
            DisambiguationRule,
            POSDisambiguator,
        )

        disambiguator = POSDisambiguator()

        result = disambiguator.disambiguate_in_context(
            word="test",
            word_pos_tags=frozenset({"V", "N"}),
            next_word_pos="PPM",
        )

        assert result.resolved_pos == "V"
        assert result.rule_applied == DisambiguationRule.R3_VERB_BEFORE_PARTICLE

    def test_r5_with_adv_pos_tag(self):
        """Test R5 applies with ADV POS tag as prev_word_pos."""
        from myspellchecker.algorithms.pos_disambiguator import (
            DisambiguationRule,
            POSDisambiguator,
        )

        disambiguator = POSDisambiguator()

        result = disambiguator.disambiguate_in_context(
            word="test",
            word_pos_tags=frozenset({"V", "N"}),
            prev_word_pos="ADV",
        )

        assert result.resolved_pos == "V"
        assert result.rule_applied == DisambiguationRule.R5_VERB_AFTER_ADVERB


class TestPrefixPatterns:
    """Tests for prefix pattern inference."""

    @pytest.fixture
    def engine(self):
        """Create a POSInferenceEngine instance."""
        from myspellchecker.algorithms.pos_inference import POSInferenceEngine

        return POSInferenceEngine()

    def test_prefix_patterns_loaded(self, engine):
        """Test prefix patterns are loaded from config."""
        # Engine should have prefix patterns loaded
        assert engine.prefix_patterns is not None
        # The patterns dict may be empty or populated depending on config

    def test_proper_noun_suffixes_loaded(self, engine):
        """Test proper noun suffixes are loaded from config."""
        assert engine.proper_noun_suffixes is not None

    def test_config_loading(self, engine):
        """Test _load_from_config runs without error."""
        # The config is loaded during __init__, verify engine is functional
        result = engine.infer_pos("test")
        assert result is not None


class TestMorphologicalInference:
    """Tests for morphological inference path."""

    @pytest.fixture
    def engine(self):
        """Create a POSInferenceEngine instance."""
        from myspellchecker.algorithms.pos_inference import POSInferenceEngine

        return POSInferenceEngine()

    def test_morphological_inference_called_for_unknown_word(self, engine):
        """Test morphological inference is used for unknown words."""

        # A word that's not a numeral, not in ambiguous registry,
        # doesn't match prefix/suffix patterns
        result = engine.infer_pos("xyz")

        # Should have tried morphological inference
        # Result depends on the analyzer, but shouldn't crash
        assert result is not None
        assert result.word == "xyz"

    def test_inference_with_myanmar_word(self, engine):
        """Test inference with a Myanmar word."""
        # Test with a common Myanmar word
        result = engine.infer_pos("သည်")
        assert result is not None
        assert result.word == "သည်"


class TestBatchInferenceDetailedStats:
    """Detailed tests for batch inference statistics."""

    @pytest.fixture
    def engine(self):
        """Create a POSInferenceEngine instance."""
        from myspellchecker.algorithms.pos_inference import POSInferenceEngine

        return POSInferenceEngine()

    def test_batch_stats_avg_confidence_calculation(self, engine):
        """Test average confidence is calculated correctly."""
        # Include numerals which have high confidence
        words = ["၁၂၃", "၄၅၆"]

        results, stats = engine.infer_pos_batch(words)

        # Both should be detected as numerals with high confidence
        assert stats.inferred_count == 2
        assert stats.numeral_count == 2
        assert stats.avg_confidence >= 0.95

    def test_batch_empty_existing_pos_skipped(self, engine):
        """Test words with empty string POS are still processed."""
        words = ["word1"]
        existing_pos = {"word1": ""}  # Empty string means untagged

        results, stats = engine.infer_pos_batch(words, existing_pos)

        # word1 should be processed since its POS is empty
        assert stats.total_words == 1


class TestDisambiguatorWithContext:
    """Tests for disambiguation with full context."""

    @pytest.fixture
    def disambiguator(self):
        """Create a POSDisambiguator instance."""
        from myspellchecker.algorithms.pos_disambiguator import POSDisambiguator

        return POSDisambiguator()

    def test_r2_with_pron_as_next(self, disambiguator):
        """Test R2 applies with PRON as next word POS."""
        from myspellchecker.algorithms.pos_disambiguator import DisambiguationRule

        result = disambiguator.disambiguate_in_context(
            word="test",
            word_pos_tags=frozenset({"ADJ", "V"}),
            next_word_pos="PRON",  # PRON is in MODIFIABLE_TAGS
        )

        assert result.resolved_pos == "ADJ"
        assert result.rule_applied == DisambiguationRule.R2_ADJ_BEFORE_NOUN

    def test_r1_with_aux_as_prev(self, disambiguator):
        """Test R1 applies with AUX as prev word POS."""
        from myspellchecker.algorithms.pos_disambiguator import DisambiguationRule

        result = disambiguator.disambiguate_in_context(
            word="test",
            word_pos_tags=frozenset({"N", "V"}),
            prev_word_pos="AUX",  # AUX is in VERB_TAGS
        )

        assert result.resolved_pos == "N"
        assert result.rule_applied == DisambiguationRule.R1_NOUN_AFTER_VERB

    def test_r3_with_p_mod_as_next(self, disambiguator):
        """Test R3 applies with P_MOD as next word POS."""
        from myspellchecker.algorithms.pos_disambiguator import DisambiguationRule

        result = disambiguator.disambiguate_in_context(
            word="test",
            word_pos_tags=frozenset({"V", "N"}),
            next_word_pos="P_MOD",  # P_MOD is in PARTICLE_POS_TAGS
        )

        assert result.resolved_pos == "V"
        assert result.rule_applied == DisambiguationRule.R3_VERB_BEFORE_PARTICLE

    def test_determiners_all_trigger_r4(self, disambiguator):
        """Test various determiners trigger R4."""
        from myspellchecker.algorithms.pos_disambiguator import (
            DETERMINERS,
            DisambiguationRule,
        )

        for det in list(DETERMINERS)[:3]:  # Test first 3
            result = disambiguator.disambiguate_in_context(
                word="test",
                word_pos_tags=frozenset({"N", "V"}),
                prev_word=det,
            )

            assert result.resolved_pos == "N"
            assert result.rule_applied == DisambiguationRule.R4_NOUN_AFTER_DETERMINER

    def test_adverb_markers_all_trigger_r5(self, disambiguator):
        """Test various adverb markers trigger R5."""
        from myspellchecker.algorithms.pos_disambiguator import (
            ADVERB_MARKERS,
            DisambiguationRule,
        )

        for adv in list(ADVERB_MARKERS)[:3]:  # Test first 3
            result = disambiguator.disambiguate_in_context(
                word="test",
                word_pos_tags=frozenset({"V", "N"}),
                prev_word=adv,
            )

            assert result.resolved_pos == "V"
            assert result.rule_applied == DisambiguationRule.R5_VERB_AFTER_ADVERB


class TestInferenceSourceMapping:
    """Tests for source mapping in morphological inference."""

    @pytest.fixture
    def engine(self):
        """Create a POSInferenceEngine instance."""
        from myspellchecker.algorithms.pos_inference import POSInferenceEngine

        return POSInferenceEngine()

    def test_select_primary_tag_all_priorities(self, engine):
        """Test _select_primary_tag covers all priority levels."""
        # Test each priority level
        assert engine._select_primary_tag(frozenset({"N", "CONJ"})) == "N"
        assert engine._select_primary_tag(frozenset({"V", "CONJ"})) == "V"
        assert engine._select_primary_tag(frozenset({"ADJ", "CONJ"})) == "ADJ"
        assert engine._select_primary_tag(frozenset({"ADV", "CONJ"})) == "ADV"
        assert engine._select_primary_tag(frozenset({"CONJ", "INT"})) == "CONJ"
        assert engine._select_primary_tag(frozenset({"PRON", "INT"})) == "PRON"
        assert engine._select_primary_tag(frozenset({"INT", "X"})) == "INT"


class TestThreadSafety:
    """Tests for thread-safe singleton patterns."""

    def test_disambiguator_singleton_thread_safe(self):
        """Test get_disambiguator is thread-safe."""
        import threading

        from myspellchecker.algorithms.pos_disambiguator import get_disambiguator

        results = []

        def get_instance():
            d = get_disambiguator()
            results.append(id(d))

        threads = [threading.Thread(target=get_instance) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All threads should get the same instance
        assert len(set(results)) == 1
