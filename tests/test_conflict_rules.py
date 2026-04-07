"""Unit tests for the validation pipeline conflict resolution rules."""

from myspellchecker.core.validation_strategies.conflict_rules import (
    STRATEGY_OVERRIDE_RULES,
    should_skip_position,
)


class TestStrategyOverrideRules:
    """Tests for the STRATEGY_OVERRIDE_RULES constant."""

    def test_statistical_confusable_overrides_pos(self):
        assert "pos_sequence_error" in STRATEGY_OVERRIDE_RULES["StatisticalConfusableStrategy"]

    def test_statistical_confusable_does_not_override_others(self):
        rules = STRATEGY_OVERRIDE_RULES["StatisticalConfusableStrategy"]
        assert "context_probability" not in rules
        assert "confusable_error" not in rules
        assert "tone_ambiguity" not in rules

    def test_confusable_semantic_overrides_pos_and_context(self):
        rules = STRATEGY_OVERRIDE_RULES["ConfusableSemanticStrategy"]
        assert "pos_sequence_error" in rules
        assert "context_probability" in rules

    def test_confusable_semantic_does_not_override_deterministic(self):
        rules = STRATEGY_OVERRIDE_RULES["ConfusableSemanticStrategy"]
        assert "tone_ambiguity" not in rules
        assert "confusable_error" not in rules

    def test_semantic_overrides_pos_and_context(self):
        rules = STRATEGY_OVERRIDE_RULES["SemanticValidationStrategy"]
        assert "pos_sequence_error" in rules
        assert "context_probability" in rules

    def test_semantic_does_not_override_confusable(self):
        rules = STRATEGY_OVERRIDE_RULES["SemanticValidationStrategy"]
        assert "confusable_error" not in rules

    def test_exactly_three_strategies_have_rules(self):
        assert len(STRATEGY_OVERRIDE_RULES) == 3


class TestShouldSkipPosition:
    """Tests for should_skip_position()."""

    # -- Unclaimed positions --

    def test_unclaimed_position_never_skips(self):
        assert should_skip_position("AnyStrategy", 0, {}) is False

    def test_unclaimed_position_with_other_claimed(self):
        errors = {10: "pos_sequence_error"}
        assert should_skip_position("AnyStrategy", 0, errors) is False

    # -- StatisticalConfusable --

    def test_stat_confusable_overrides_pos_sequence(self):
        errors = {0: "pos_sequence_error"}
        assert should_skip_position("StatisticalConfusableStrategy", 0, errors) is False

    def test_stat_confusable_skips_confusable_error(self):
        errors = {0: "confusable_error"}
        assert should_skip_position("StatisticalConfusableStrategy", 0, errors) is True

    def test_stat_confusable_skips_tone_ambiguity(self):
        errors = {0: "tone_ambiguity"}
        assert should_skip_position("StatisticalConfusableStrategy", 0, errors) is True

    # -- ConfusableSemantic --

    def test_confusable_semantic_overrides_pos_sequence(self):
        errors = {0: "pos_sequence_error"}
        assert should_skip_position("ConfusableSemanticStrategy", 0, errors) is False

    def test_confusable_semantic_overrides_context_probability(self):
        errors = {0: "context_probability"}
        assert should_skip_position("ConfusableSemanticStrategy", 0, errors) is False

    def test_confusable_semantic_skips_tone(self):
        errors = {0: "tone_ambiguity"}
        assert should_skip_position("ConfusableSemanticStrategy", 0, errors) is True

    def test_confusable_semantic_skips_confusable_error(self):
        errors = {0: "confusable_error"}
        assert should_skip_position("ConfusableSemanticStrategy", 0, errors) is True

    # -- Semantic --

    def test_semantic_overrides_pos_sequence(self):
        errors = {0: "pos_sequence_error"}
        assert should_skip_position("SemanticValidationStrategy", 0, errors) is False

    def test_semantic_overrides_context_probability(self):
        errors = {0: "context_probability"}
        assert should_skip_position("SemanticValidationStrategy", 0, errors) is False

    def test_semantic_skips_confusable(self):
        errors = {0: "confusable_error"}
        assert should_skip_position("SemanticValidationStrategy", 0, errors) is True

    def test_semantic_skips_homophone(self):
        errors = {0: "homophone_error"}
        assert should_skip_position("SemanticValidationStrategy", 0, errors) is True

    # -- Unknown strategies --

    def test_unknown_strategy_always_skips_claimed(self):
        errors = {0: "pos_sequence_error"}
        assert should_skip_position("UnknownStrategy", 0, errors) is True

    def test_unknown_strategy_skips_any_type(self):
        for error_type in ["pos_sequence_error", "context_probability", "confusable_error"]:
            errors = {0: error_type}
            assert should_skip_position("MadeUpStrategy", 0, errors) is True

    # -- Strategies not in override rules (should always skip) --

    def test_pos_strategy_always_skips(self):
        errors = {0: "confusable_error"}
        assert should_skip_position("POSSequenceValidationStrategy", 0, errors) is True

    def test_tone_strategy_always_skips(self):
        errors = {0: "context_probability"}
        assert should_skip_position("ToneValidationStrategy", 0, errors) is True

    def test_homophone_strategy_always_skips(self):
        errors = {0: "pos_sequence_error"}
        assert should_skip_position("HomophoneValidationStrategy", 0, errors) is True

    # -- Multiple positions --

    def test_mixed_positions(self):
        errors = {0: "pos_sequence_error", 10: "confusable_error", 20: "tone_ambiguity"}
        # ConfusableSemantic can override pos_sequence_error at 0
        assert should_skip_position("ConfusableSemanticStrategy", 0, errors) is False
        # ConfusableSemantic cannot override confusable_error at 10
        assert should_skip_position("ConfusableSemanticStrategy", 10, errors) is True
        # Position 30 is unclaimed
        assert should_skip_position("ConfusableSemanticStrategy", 30, errors) is False
