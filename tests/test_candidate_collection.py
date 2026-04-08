"""Integration tests for ErrorCandidate collection in the validation pipeline."""

from unittest.mock import MagicMock

from myspellchecker.core.config import SpellCheckerConfig
from myspellchecker.core.config.validation_configs import ValidationConfig
from myspellchecker.core.context_validator import ContextValidator
from myspellchecker.core.response import Error
from myspellchecker.core.validation_strategies import (
    ErrorCandidate,
    ValidationContext,
    ValidationStrategy,
)


class _ClaimingStrategy(ValidationStrategy):
    """Strategy that claims a position with a specific error type."""

    def __init__(self, priority_val: int, error_type: str, confidence: float = 0.8):
        self._priority = priority_val
        self._error_type = error_type
        self._confidence = confidence

    def validate(self, context: ValidationContext) -> list[Error]:
        if not context.words:
            return []
        pos = context.word_positions[0]
        error = Error(
            text=context.words[0],
            position=pos,
            error_type=self._error_type,
            suggestions=["suggestion_from_" + self.__class__.__name__],
            confidence=self._confidence,
        )
        context.existing_errors[pos] = self._error_type
        context.existing_confidences[pos] = self._confidence
        return [error]

    def priority(self) -> int:
        return self._priority


class _POSLikeStrategy(_ClaimingStrategy):
    """Simulates POSSequenceValidationStrategy claiming a position."""

    def __init__(self):
        super().__init__(30, "pos_sequence_error", 0.7)


class _ConfusableLikeStrategy(_ClaimingStrategy):
    """Simulates ConfusableSemanticStrategy that skips claimed positions."""

    def __init__(self):
        super().__init__(48, "confusable_error", 0.85)

    def validate(self, context: ValidationContext) -> list[Error]:
        if not context.words:
            return []
        pos = context.word_positions[0]
        # Simulate the skip check (position already claimed by earlier strategy)
        if pos in context.existing_errors:
            return []
        return super().validate(context)


def _make_validator(*strategies: ValidationStrategy) -> ContextValidator:
    config = SpellCheckerConfig(
        use_ner=False,
        validation=ValidationConfig(
            raise_on_strategy_error=True,
            enable_fast_path=False,
        ),
    )
    segmenter = MagicMock()
    segmenter.segment_sentences.return_value = ["\u1000\u1005\u102c\u1038"]
    segmenter.segment_words.return_value = ["\u1000\u1005\u102c\u1038"]
    return ContextValidator(config, segmenter, strategies=list(strategies))


class TestErrorCandidateDataclass:
    """Tests for the ErrorCandidate dataclass itself."""

    def test_creation_with_all_fields(self):
        c = ErrorCandidate(
            strategy_name="TestStrategy",
            error_type="test_error",
            confidence=0.85,
            suggestion="fix",
            evidence="test reason",
        )
        assert c.strategy_name == "TestStrategy"
        assert c.error_type == "test_error"
        assert c.confidence == 0.85
        assert c.suggestion == "fix"
        assert c.evidence == "test reason"

    def test_creation_with_defaults(self):
        c = ErrorCandidate(
            strategy_name="TestStrategy",
            error_type="test_error",
            confidence=0.5,
        )
        assert c.suggestion is None
        assert c.evidence == ""

    def test_in_validation_context(self):
        ctx = ValidationContext(
            sentence="test",
            words=["test"],
            word_positions=[0],
        )
        assert ctx.error_candidates == {}
        c = ErrorCandidate("S", "e", 0.5)
        ctx.error_candidates.setdefault(0, []).append(c)
        assert len(ctx.error_candidates[0]) == 1


class TestCandidateCollection:
    """Tests that candidates are collected during strategy execution."""

    def test_single_strategy_emits_candidate(self):
        strategy = _ClaimingStrategy(10, "test_error", 0.8)
        validator = _make_validator(strategy)

        # Access _execute_strategies directly for precise testing
        ctx = ValidationContext(
            sentence="\u1000\u1005\u102c\u1038",
            words=["\u1000\u1005\u102c\u1038"],
            word_positions=[0],
        )
        errors = validator._execute_strategies(
            context=ctx,
            strategy_debug_telemetry=None,
        )

        assert len(errors) == 1
        assert 0 in ctx.error_candidates
        assert len(ctx.error_candidates[0]) == 1
        assert ctx.error_candidates[0][0].error_type == "test_error"
        assert ctx.error_candidates[0][0].confidence == 0.8

    def test_two_strategies_two_positions(self):
        """Two strategies claiming different positions each emit a candidate."""

        class _SecondPositionStrategy(ValidationStrategy):
            def validate(self, context: ValidationContext) -> list[Error]:
                if len(context.words) < 2:
                    return []
                pos = context.word_positions[1]
                error = Error(
                    text=context.words[1],
                    position=pos,
                    error_type="second_error",
                    suggestions=["fix2"],
                    confidence=0.9,
                )
                context.existing_errors[pos] = "second_error"
                return [error]

            def priority(self) -> int:
                return 50

        s1 = _ClaimingStrategy(10, "first_error", 0.7)
        s2 = _SecondPositionStrategy()

        ctx = ValidationContext(
            sentence="\u1000 \u1001",
            words=["\u1000", "\u1001"],
            word_positions=[0, 3],
        )
        validator = _make_validator(s1, s2)
        errors = validator._execute_strategies(context=ctx, strategy_debug_telemetry=None)

        assert len(errors) == 2
        assert 0 in ctx.error_candidates
        assert 3 in ctx.error_candidates
        assert ctx.error_candidates[0][0].error_type == "first_error"
        assert ctx.error_candidates[3][0].error_type == "second_error"

    def test_candidate_suggestion_captured(self):
        strategy = _ClaimingStrategy(10, "test_error", 0.8)
        ctx = ValidationContext(
            sentence="\u1000\u1005\u102c\u1038",
            words=["\u1000\u1005\u102c\u1038"],
            word_positions=[0],
        )
        validator = _make_validator(strategy)
        validator._execute_strategies(context=ctx, strategy_debug_telemetry=None)

        candidate = ctx.error_candidates[0][0]
        assert candidate.suggestion == "suggestion_from__ClaimingStrategy"

    def test_no_errors_means_no_candidates(self):
        class _NoErrorStrategy(ValidationStrategy):
            def validate(self, context: ValidationContext) -> list[Error]:
                return []

            def priority(self) -> int:
                return 10

        ctx = ValidationContext(
            sentence="\u1000\u1005\u102c\u1038",
            words=["\u1000\u1005\u102c\u1038"],
            word_positions=[0],
        )
        validator = _make_validator(_NoErrorStrategy())
        validator._execute_strategies(context=ctx, strategy_debug_telemetry=None)

        assert ctx.error_candidates == {}
