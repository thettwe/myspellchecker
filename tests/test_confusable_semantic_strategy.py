"""Tests for ConfusableSemanticStrategy (MLM-enhanced confusable detection)."""

from unittest.mock import MagicMock

from myspellchecker.core.validation_strategies.base import ValidationContext
from myspellchecker.core.validation_strategies.confusable_semantic_strategy import (
    ConfusableSemanticStrategy,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_context(
    sentence: str,
    words: list[str],
    positions: list[int],
    *,
    is_name_mask: list[bool] | None = None,
    existing_errors: dict[int, str] | None = None,
) -> ValidationContext:
    """Build a ValidationContext with sensible defaults."""
    return ValidationContext(
        sentence=sentence,
        words=words,
        word_positions=positions,
        is_name_mask=is_name_mask or [False] * len(words),
        existing_errors=existing_errors or {},
    )


def _make_provider(
    *,
    valid_words: set[str] | None = None,
    frequencies: dict[str, int] | None = None,
) -> MagicMock:
    """Create a mock provider."""
    provider = MagicMock()
    valid = valid_words or set()
    freqs = frequencies or {}
    provider.is_valid_word.side_effect = lambda w: w in valid
    provider.get_word_frequency.side_effect = lambda w: freqs.get(w, 0)
    return provider


def _make_semantic_checker(
    predictions: list[tuple[str, float]] | None = None,
    explicit_scores: dict[str, float] | None = None,
) -> MagicMock:
    """Create a mock SemanticChecker with predict_mask results."""
    checker = MagicMock()
    checker.predict_mask.return_value = predictions or []
    checker.score_mask_candidates.return_value = explicit_scores or {}
    return checker


# ===========================================================================
# Initialization and basic properties
# ===========================================================================


class TestConfusableSemanticStrategyInit:
    """Tests for strategy initialization and basic properties."""

    def test_priority(self):
        """Priority must be 48."""
        strategy = ConfusableSemanticStrategy(semantic_checker=MagicMock(), provider=MagicMock())
        assert strategy.priority() == 48

    def test_repr(self):
        """__repr__ includes class name and key parameters."""
        strategy = ConfusableSemanticStrategy(
            semantic_checker=MagicMock(),
            provider=MagicMock(),
            confidence=0.85,
            logit_diff_threshold=4.0,
        )
        r = repr(strategy)
        assert "ConfusableSemanticStrategy" in r
        assert "0.85" in r
        assert "4.0" in r

    def test_close_delegates_to_semantic_checker(self):
        """close() calls semantic_checker.close() if available."""
        checker = MagicMock()
        strategy = ConfusableSemanticStrategy(semantic_checker=checker, provider=MagicMock())
        strategy.close()
        checker.close.assert_called_once()

    def test_close_with_none_checker(self):
        """close() does not crash when semantic_checker is None."""
        strategy = ConfusableSemanticStrategy(semantic_checker=None, provider=MagicMock())
        strategy.close()  # should not raise


# ===========================================================================
# Validate: early returns
# ===========================================================================


class TestConfusableSemanticValidateEarlyReturns:
    """Tests for early-return conditions in validate()."""

    def test_empty_when_no_semantic_checker(self):
        """Returns [] when semantic_checker is None."""
        strategy = ConfusableSemanticStrategy(semantic_checker=None, provider=MagicMock())
        ctx = _make_context("w1 w2", ["w1", "w2"], [0, 3])
        assert strategy.validate(ctx) == []

    def test_empty_when_single_word(self):
        """Returns [] for sentences with fewer than 2 words."""
        strategy = ConfusableSemanticStrategy(semantic_checker=MagicMock(), provider=MagicMock())
        ctx = _make_context("word", ["word"], [0])
        assert strategy.validate(ctx) == []

    def test_empty_when_provider_lacks_is_valid_word(self):
        """Returns [] when provider has no is_valid_word."""
        provider = MagicMock(spec=[])
        strategy = ConfusableSemanticStrategy(semantic_checker=MagicMock(), provider=provider)
        ctx = _make_context("a b", ["a", "b"], [0, 2])
        assert strategy.validate(ctx) == []


# ===========================================================================
# Validate: skipping logic
# ===========================================================================


class TestConfusableSemanticSkipping:
    """Tests for words that should be skipped during validation."""

    def test_skip_names(self):
        """Words marked as names are skipped."""
        provider = _make_provider(valid_words={"w1", "w2"})
        strategy = ConfusableSemanticStrategy(
            semantic_checker=_make_semantic_checker(),
            provider=provider,
        )
        ctx = _make_context("w1 w2", ["w1", "w2"], [0, 3], is_name_mask=[True, True])
        assert strategy.validate(ctx) == []

    def test_skip_existing_errors(self):
        """Positions already in existing_errors are skipped."""
        provider = _make_provider(valid_words={"w1", "w2"})
        strategy = ConfusableSemanticStrategy(
            semantic_checker=_make_semantic_checker(),
            provider=provider,
        )
        ctx = _make_context("w1 w2", ["w1", "w2"], [0, 3], existing_errors={0: "test", 3: "test"})
        assert strategy.validate(ctx) == []

    def test_skip_invalid_words(self):
        """Invalid words (not in dictionary) are skipped."""
        provider = _make_provider(valid_words=set())
        strategy = ConfusableSemanticStrategy(
            semantic_checker=_make_semantic_checker(),
            provider=provider,
        )
        ctx = _make_context("aa bb", ["aa", "bb"], [0, 3])
        assert strategy.validate(ctx) == []

    def test_skip_short_words(self):
        """Words shorter than min_word_length are skipped (unless particle confusable)."""
        provider = _make_provider(valid_words={"a", "bb"})
        strategy = ConfusableSemanticStrategy(
            semantic_checker=_make_semantic_checker(),
            provider=provider,
            min_word_length=2,
        )
        ctx = _make_context("a bb", ["a", "bb"], [0, 2])
        assert strategy.validate(ctx) == []


# ===========================================================================
# Validate: main flow
# ===========================================================================


class TestConfusableSemanticValidateFlow:
    """Tests for the main validation flow."""

    def test_variant_flagged_when_logit_diff_exceeds_threshold(self):
        """A confusable error is created when MLM strongly prefers the variant."""
        # ကြောင်း (cat) vs ကျောင်း (school) — medial ျ↔ြ swap
        current = "\u1000\u103c\u1031\u102c\u1004\u103a\u1038"  # ကြောင်း
        variant = "\u1000\u103b\u1031\u102c\u1004\u103a\u1038"  # ကျောင်း
        prev = "\u1000\u103b\u103d\u1014\u103a\u1019"  # ကျွန်မ
        nxt = "\u101e\u103d\u102c\u1038"  # သွား

        provider = _make_provider(
            valid_words={current, variant, prev, nxt},
            frequencies={current: 500, variant: 800},
        )
        # MLM strongly predicts variant over current
        checker = _make_semantic_checker(predictions=[(variant, 8.5), (current, 2.0), (prev, 1.0)])

        strategy = ConfusableSemanticStrategy(
            semantic_checker=checker,
            provider=provider,
            logit_diff_threshold_medial=2.0,
        )

        sentence = f"{prev} {current} {nxt}"
        ctx = _make_context(sentence, [prev, current, nxt], [0, 20, 40])
        errors = strategy.validate(ctx)

        assert len(errors) == 1
        assert errors[0].text == current
        assert variant in errors[0].suggestions
        assert errors[0].error_type == "confusable_error"
        assert 20 in ctx.existing_errors

    def test_no_error_when_logit_diff_below_threshold(self):
        """No error when MLM difference is below the threshold."""
        current = "\u1019\u102c"  # မာ
        variant = "\u1019\u103e\u102c"  # မှာ
        prev = "X"
        nxt = "Y"

        provider = _make_provider(
            valid_words={current, variant, prev, nxt},
            frequencies={current: 100, variant: 200},
        )
        # MLM scores nearly equal — diff below default threshold of 3.0
        checker = _make_semantic_checker(predictions=[(variant, 5.0), (current, 4.0)])

        strategy = ConfusableSemanticStrategy(
            semantic_checker=checker,
            provider=provider,
            logit_diff_threshold=3.0,
        )
        ctx = _make_context("X curr Y", [prev, current, nxt], [0, 2, 7])
        errors = strategy.validate(ctx)
        assert errors == []

    def test_no_error_when_no_valid_variants(self):
        """No error when all generated variants are invalid."""
        current = "\u1019\u102c"  # မာ
        prev = "X"
        nxt = "Y"

        # Only current is valid — no variants in dictionary
        provider = _make_provider(
            valid_words={current, prev, nxt},
            frequencies={current: 100},
        )
        checker = _make_semantic_checker(predictions=[("something", 10.0)])

        strategy = ConfusableSemanticStrategy(semantic_checker=checker, provider=provider)
        ctx = _make_context("X curr Y", [prev, current, nxt], [0, 2, 7])
        errors = strategy.validate(ctx)
        assert errors == []

    def test_no_error_when_predictions_empty(self):
        """No error when predict_mask returns empty list."""
        current = "\u1019\u102c"  # မာ
        variant = "\u1019\u103e\u102c"  # မှာ
        prev = "X"
        nxt = "Y"

        provider = _make_provider(
            valid_words={current, variant, prev, nxt},
            frequencies={current: 100, variant: 200},
        )
        checker = _make_semantic_checker(predictions=[])

        strategy = ConfusableSemanticStrategy(semantic_checker=checker, provider=provider)
        ctx = _make_context("X curr Y", [prev, current, nxt], [0, 2, 7])
        errors = strategy.validate(ctx)
        assert errors == []

    def test_variant_not_in_predictions_skipped(self):
        """Variants not appearing in MLM top-K predictions are skipped."""
        current = "\u1019\u102c"  # မာ
        variant = "\u1019\u103e\u102c"  # မှာ
        prev = "X"
        nxt = "Y"

        provider = _make_provider(
            valid_words={current, variant, prev, nxt},
            frequencies={current: 100, variant: 200},
        )
        # MLM predicts neither current nor variant
        checker = _make_semantic_checker(predictions=[("unrelated", 10.0)])

        strategy = ConfusableSemanticStrategy(semantic_checker=checker, provider=provider)
        ctx = _make_context("X curr Y", [prev, current, nxt], [0, 2, 7])
        errors = strategy.validate(ctx)
        assert errors == []

    def test_explicit_non_topk_scoring_penalized_for_non_homophones(self):
        """Explicit-only non-top-K variants need much stronger evidence."""
        current = "\u1019\u102c"  # မာ
        variant = "\u1019\u103e\u102c"  # မှာ
        prev = "X"
        nxt = "Y"

        provider = _make_provider(
            valid_words={current, variant, prev, nxt},
            frequencies={current: 100, variant: 200},
        )
        checker = _make_semantic_checker(
            predictions=[("unrelated", 10.0)],
            explicit_scores={current: 1.0, variant: 8.0},
        )

        strategy = ConfusableSemanticStrategy(
            semantic_checker=checker,
            provider=provider,
            logit_diff_threshold=3.0,
        )
        ctx = _make_context("X curr Y", [prev, current, nxt], [0, 2, 7])
        errors = strategy.validate(ctx)
        assert errors == []

    def test_explicit_non_topk_homophone_can_still_flag(self):
        """Known homophones keep a lighter explicit-only penalty."""
        current = "\u1001\u103d\u1032"  # ခွဲ
        variant = "\u1000\u103d\u1032"  # ကွဲ
        prev = "A"
        nxt = "B"

        provider = _make_provider(
            valid_words={current, variant, prev, nxt},
            frequencies={current: 5000, variant: 9000},
        )
        checker = _make_semantic_checker(
            predictions=[("unrelated", 10.0)],
            explicit_scores={current: 1.0, variant: 7.0},
        )

        strategy = ConfusableSemanticStrategy(
            semantic_checker=checker,
            provider=provider,
            homophone_map={current: {variant}},
        )
        ctx = _make_context("A ခွဲ B", [prev, current, nxt], [0, 2, 7])
        errors = strategy.validate(ctx)

        assert len(errors) == 1
        assert errors[0].text == current
        assert variant in errors[0].suggestions

    def test_exception_handling(self):
        """Exceptions in validate are caught gracefully."""
        provider = MagicMock()
        provider.is_valid_word.side_effect = RuntimeError("boom")
        checker = MagicMock()

        strategy = ConfusableSemanticStrategy(semantic_checker=checker, provider=provider)
        ctx = _make_context("w1 w2", ["w1", "w2"], [0, 3])
        errors = strategy.validate(ctx)
        assert errors == []


# ===========================================================================
# Threshold logic
# ===========================================================================


class TestGetThreshold:
    """Tests for _get_threshold() and its layered penalty system."""

    def _make_strategy(self, **kwargs) -> ConfusableSemanticStrategy:
        defaults = dict(
            semantic_checker=MagicMock(),
            provider=_make_provider(),
        )
        defaults.update(kwargs)
        return ConfusableSemanticStrategy(**defaults)

    def test_default_threshold(self):
        """Default base threshold is 3.0."""
        strategy = self._make_strategy(logit_diff_threshold=3.0)
        threshold, _ = strategy._get_threshold(
            "word",
            "variant",
            current_in_topk=False,
            is_high_freq=False,
        )
        assert threshold == 3.0

    def test_high_freq_threshold(self):
        """High-frequency words get threshold 6.0."""
        strategy = self._make_strategy(high_freq_logit_diff=6.0)
        threshold, _ = strategy._get_threshold(
            "word",
            "variant",
            current_in_topk=False,
            is_high_freq=True,
        )
        assert threshold >= 6.0

    def test_current_in_topk_threshold(self):
        """When current word is in top-K, threshold is 5.0."""
        strategy = self._make_strategy(logit_diff_threshold_current_in_topk=5.0)
        threshold, _ = strategy._get_threshold(
            "word",
            "variant",
            current_in_topk=True,
            is_high_freq=False,
        )
        assert threshold >= 5.0

    def test_medial_confusable_threshold(self):
        """Medial ျ↔ြ swap gets lower threshold (2.0)."""
        word = "\u1000\u103b\u1031\u102c\u1004\u103a\u1038"  # ကျောင်း
        variant = "\u1000\u103c\u1031\u102c\u1004\u103a\u1038"  # ကြောင်း
        strategy = self._make_strategy(logit_diff_threshold_medial=2.0)
        threshold, _ = strategy._get_threshold(
            word,
            variant,
            current_in_topk=False,
            is_high_freq=False,
        )
        assert threshold == 2.0

    def test_freq_ratio_penalty_high(self):
        """Frequency ratio >5x adds freq_ratio_penalty_high to threshold."""
        provider = _make_provider(
            frequencies={"word": 100, "variant": 600},
        )
        strategy = self._make_strategy(
            provider=provider,
            logit_diff_threshold=3.0,
            freq_ratio_penalty_high=3.0,
        )
        threshold, _ = strategy._get_threshold(
            "word",
            "variant",
            current_in_topk=False,
            is_high_freq=False,
            word_freq=100,
        )
        # freq_ratio = 600/100 = 6 > 5 → +3.0 penalty
        assert threshold >= 6.0

    def test_freq_ratio_penalty_mid(self):
        """Frequency ratio >2x but <=5x adds freq_ratio_penalty_mid."""
        provider = _make_provider(
            frequencies={"word": 100, "variant": 350},
        )
        strategy = self._make_strategy(
            provider=provider,
            logit_diff_threshold=3.0,
            freq_ratio_penalty_mid=1.5,
        )
        threshold, _ = strategy._get_threshold(
            "word",
            "variant",
            current_in_topk=False,
            is_high_freq=False,
            word_freq=100,
        )
        # freq_ratio = 350/100 = 3.5 > 2 → +1.5 penalty
        assert threshold >= 4.5

    def test_reverse_freq_ratio_penalty(self):
        """When current word >>50x more frequent than variant, penalty added."""
        provider = _make_provider(
            frequencies={"word": 100000, "variant": 10},
        )
        strategy = self._make_strategy(
            provider=provider,
            logit_diff_threshold=3.0,
            freq_ratio_penalty_high=3.0,
        )
        threshold, _ = strategy._get_threshold(
            "word",
            "variant",
            current_in_topk=False,
            is_high_freq=False,
            word_freq=100000,
        )
        # reverse_ratio = 100000/10 = 10000 > 50 → +3.0
        assert threshold >= 6.0

    def test_visarga_pair_penalty(self):
        """Visarga-only pair adds visarga_penalty."""
        word = "\u1015\u103c\u102e"  # ပြီ
        variant = "\u1015\u103c\u102e\u1038"  # ပြီး
        strategy = self._make_strategy(
            logit_diff_threshold=3.0,
            visarga_penalty=2.0,
        )
        threshold, _ = strategy._get_threshold(
            word,
            variant,
            current_in_topk=False,
            is_high_freq=False,
        )
        assert threshold >= 5.0  # 3.0 + 2.0

    def test_high_freq_visarga_pair_hard_block_by_default(self):
        """High-freq visarga pair returns inf by default (hard block enabled)."""
        word = "\u1015\u103c\u102e"  # ပြီ
        variant = "\u1015\u103c\u102e\u1038"  # ပြီး
        provider = _make_provider(
            frequencies={word: 60000, variant: 60000},
        )
        strategy = self._make_strategy(
            provider=provider,
            high_freq_threshold=50000,
        )
        threshold, _ = strategy._get_threshold(
            word,
            variant,
            current_in_topk=False,
            is_high_freq=False,
            word_freq=60000,
        )
        # Default: hard block enabled → inf threshold for high-freq visarga pairs
        assert threshold == float("inf")

    def test_high_freq_visarga_pair_no_hard_block_when_disabled(self):
        """High-freq visarga pair uses finite threshold when hard block disabled."""
        word = "\u1015\u103c\u102e"  # ပြီ
        variant = "\u1015\u103c\u102e\u1038"  # ပြီး
        provider = _make_provider(
            frequencies={word: 60000, variant: 60000},
        )
        strategy = self._make_strategy(
            provider=provider,
            high_freq_threshold=50000,
            visarga_high_freq_hard_block=False,
        )
        threshold, _ = strategy._get_threshold(
            word,
            variant,
            current_in_topk=False,
            is_high_freq=False,
            word_freq=60000,
        )
        # Hard block disabled: high but finite threshold
        assert threshold < float("inf")
        assert threshold >= 5.0  # base (3.0) + visarga_penalty (2.0)

    def test_high_freq_visarga_pair_hard_block_when_enabled(self):
        """High-freq visarga pair returns inf when hard block is enabled."""
        word = "\u1015\u103c\u102e"  # ပြီ
        variant = "\u1015\u103c\u102e\u1038"  # ပြီး
        provider = _make_provider(
            frequencies={word: 60000, variant: 60000},
        )
        strategy = self._make_strategy(
            provider=provider,
            high_freq_threshold=50000,
            visarga_high_freq_hard_block=True,
        )
        threshold, _ = strategy._get_threshold(
            word,
            variant,
            current_in_topk=False,
            is_high_freq=False,
            word_freq=60000,
        )
        assert threshold == float("inf")

    def test_sentence_final_penalty(self):
        """Sentence-final position adds sentence_final_penalty."""
        strategy = self._make_strategy(
            logit_diff_threshold=3.0,
            sentence_final_penalty=1.0,
        )
        threshold, _ = strategy._get_threshold(
            "word",
            "variant",
            current_in_topk=False,
            is_high_freq=False,
            is_sentence_final=True,
        )
        assert threshold >= 4.0

    def test_max_threshold_cap(self):
        """Stacked penalties are capped at max_threshold."""
        provider = _make_provider(
            frequencies={"word": 100000, "variant": 800000},
        )
        strategy = self._make_strategy(
            provider=provider,
            high_freq_logit_diff=6.0,
            freq_ratio_penalty_high=2.0,
            sentence_final_penalty=0.5,
            max_threshold=8.0,
        )
        threshold, _ = strategy._get_threshold(
            "word",
            "variant",
            current_in_topk=False,
            is_high_freq=True,
            word_freq=100000,
            is_sentence_final=True,
        )
        # Without cap: 6.0 + 2.0 + 0.5 = 8.5, with cap: 8.0
        assert threshold == 8.0

    def test_max_threshold_disabled(self):
        """max_threshold=0 disables capping."""
        provider = _make_provider(
            frequencies={"word": 100000, "variant": 800000},
        )
        strategy = self._make_strategy(
            provider=provider,
            high_freq_logit_diff=6.0,
            freq_ratio_penalty_high=3.0,
            sentence_final_penalty=1.0,
            max_threshold=0.0,
        )
        threshold, _ = strategy._get_threshold(
            "word",
            "variant",
            current_in_topk=False,
            is_high_freq=True,
            word_freq=100000,
            is_sentence_final=True,
        )
        # No cap: 6.0 + 3.0 + 1.0 = 10.0
        assert threshold >= 10.0

    def test_reverse_ratio_skipped_for_low_freq_words(self):
        """Reverse ratio penalty skipped when word_freq < reverse_ratio_min_freq."""
        provider = _make_provider(
            frequencies={"word": 3000, "variant": 10},
        )
        strategy = self._make_strategy(
            provider=provider,
            logit_diff_threshold=3.0,
            freq_ratio_penalty_high=2.0,
            reverse_ratio_min_freq=50000,
        )
        threshold, _ = strategy._get_threshold(
            "word",
            "variant",
            current_in_topk=False,
            is_high_freq=False,
            word_freq=3000,
        )
        # reverse_ratio = 3000/10 = 300 > 50, but word_freq (3000) < 50000
        # So penalty NOT applied, threshold stays at 3.0
        assert threshold == 3.0

    def test_particle_confusable_override(self):
        """Ultra-high-freq single-char particle confusables use threshold 5.0."""
        # "က" and "ကို" are in PARTICLE_CONFUSABLES
        word = "\u1000"  # က
        variant = "\u1000\u102d\u102f"  # ကို
        provider = _make_provider(
            frequencies={word: 1000000, variant: 1000000},
        )
        strategy = self._make_strategy(
            provider=provider,
            high_freq_logit_diff=6.0,
        )
        threshold, _ = strategy._get_threshold(
            word,
            variant,
            current_in_topk=True,
            is_high_freq=True,
            word_freq=1000000,
        )
        # Single-char + both sides high-freq => 5.0 guard
        assert threshold == 5.0

    def test_particle_confusable_multichar_keeps_relaxed_threshold(self):
        """Multi-char particle confusables keep the 3.5 base threshold."""
        word = "\u101c\u102d\u102f"  # လို
        variant = "\u101c\u102d\u102f\u1037"  # လို့
        provider = _make_provider(
            frequencies={word: 1000000, variant: 1000000},
        )
        strategy = self._make_strategy(provider=provider)
        threshold, _ = strategy._get_threshold(
            word,
            variant,
            current_in_topk=False,
            is_high_freq=False,
            word_freq=1000000,
        )
        assert threshold == 3.5

    def test_particle_confusable_zero_freq_variant(self):
        """Particle confusable with zero-freq variant uses threshold 5.0."""
        word = "\u1019\u103e\u102c"  # မှာ
        variant = "\u1019\u103e"  # မှ
        provider = _make_provider(
            frequencies={word: 1000000, variant: 0},
        )
        strategy = self._make_strategy(provider=provider)
        threshold, _ = strategy._get_threshold(
            word,
            variant,
            current_in_topk=False,
            is_high_freq=False,
            word_freq=1000000,
        )
        # Zero-freq variant particle confusable → 5.0
        assert threshold == 5.0


# ===========================================================================
# Variant ranking penalties
# ===========================================================================


class TestFindBestVariant:
    """Tests for _find_best_variant() post-threshold penalties."""

    def test_non_boundary_occurrence_adds_penalty(self):
        """Non-boundary masked occurrences need extra margin."""
        word = "\u101c\u102c"  # လာ
        variant = "\u101c"  # လ
        provider = _make_provider(
            frequencies={word: 825116, variant: 422978},
        )
        strategy = ConfusableSemanticStrategy(
            semantic_checker=MagicMock(),
            provider=provider,
        )

        pred_map = {word: 19.0, variant: 10.0}
        explicit_scores = {word: 1.0, variant: 7.0}

        boundary_variant = strategy._find_best_variant(
            word=word,
            valid_variants={variant},
            pred_map=pred_map,
            explicit_scores=explicit_scores,
            current_score=1.0,
            current_in_topk=True,
            is_high_freq=True,
            word_freq=825116,
            is_sentence_final=False,
            is_boundary_occurrence=True,
        )
        non_boundary_variant = strategy._find_best_variant(
            word=word,
            valid_variants={variant},
            pred_map=pred_map,
            explicit_scores=explicit_scores,
            current_score=1.0,
            current_in_topk=True,
            is_high_freq=True,
            word_freq=825116,
            is_sentence_final=False,
            is_boundary_occurrence=False,
        )

        assert boundary_variant == variant
        assert non_boundary_variant is None


# ===========================================================================
# Static helper methods
# ===========================================================================


class TestStaticHelpers:
    """Tests for module-level confusable helper functions."""

    def test_is_medial_confusable_ya_yit_to_pin(self):
        """Detects ျ→ြ swap."""
        from myspellchecker.core.validation_strategies.confusable_helpers import (
            is_medial_confusable,
        )

        word = "\u1000\u103b\u102c"  # ကျာ
        variant = "\u1000\u103c\u102c"  # ကြာ
        assert is_medial_confusable(word, variant) is True

    def test_is_medial_confusable_ya_pin_to_yit(self):
        """Detects ြ→ျ swap."""
        from myspellchecker.core.validation_strategies.confusable_helpers import (
            is_medial_confusable,
        )

        word = "\u1000\u103c\u102c"  # ကြာ
        variant = "\u1000\u103b\u102c"  # ကျာ
        assert is_medial_confusable(word, variant) is True

    def test_is_medial_confusable_no_swap(self):
        """Returns False when difference is not a medial swap."""
        from myspellchecker.core.validation_strategies.confusable_helpers import (
            is_medial_confusable,
        )

        assert is_medial_confusable("abc", "def") is False

    def test_is_medial_confusable_both_have_same_medial(self):
        """Returns False when both words have the same medial."""
        from myspellchecker.core.validation_strategies.confusable_helpers import (
            is_medial_confusable,
        )

        word = "\u1000\u103b\u102c"  # ကျာ
        variant = "\u1000\u103b\u102e"  # ကျီ
        assert is_medial_confusable(word, variant) is False

    def test_is_visarga_only_pair_addition(self):
        """Detects visarga addition."""
        from myspellchecker.core.validation_strategies.confusable_helpers import (
            is_visarga_only_pair,
        )

        word = "\u1015\u103c\u102e"  # ပြီ
        variant = "\u1015\u103c\u102e\u1038"  # ပြီး
        assert is_visarga_only_pair(word, variant) is True

    def test_is_visarga_only_pair_removal(self):
        """Detects visarga removal."""
        from myspellchecker.core.validation_strategies.confusable_helpers import (
            is_visarga_only_pair,
        )

        word = "\u1015\u103c\u102e\u1038"  # ပြီး
        variant = "\u1015\u103c\u102e"  # ပြီ
        assert is_visarga_only_pair(word, variant) is True

    def test_is_visarga_only_pair_false(self):
        """Returns False for non-visarga differences."""
        from myspellchecker.core.validation_strategies.confusable_helpers import (
            is_visarga_only_pair,
        )

        assert is_visarga_only_pair("abc", "def") is False

    def test_is_tone_marker_only_pair_visarga(self):
        """Tone marker pair covers visarga."""
        from myspellchecker.core.validation_strategies.confusable_helpers import (
            is_tone_marker_only_pair,
        )

        word = "\u1015\u103c\u102e"
        variant = "\u1015\u103c\u102e\u1038"
        assert is_tone_marker_only_pair(word, variant) is True

    def test_is_tone_marker_only_pair_aukmyit(self):
        """Tone marker pair covers aukmyit (dot-below)."""
        from myspellchecker.core.validation_strategies.confusable_helpers import (
            is_tone_marker_only_pair,
        )

        word = "\u101c\u102d\u102f"  # လို
        variant = "\u101c\u102d\u102f\u1037"  # လို့
        assert is_tone_marker_only_pair(word, variant) is True

    def test_is_tone_marker_only_pair_false(self):
        """Returns False when difference is not a tone marker."""
        from myspellchecker.core.validation_strategies.confusable_helpers import (
            is_tone_marker_only_pair,
        )

        assert is_tone_marker_only_pair("abc", "xyz") is False

    def test_is_particle_confusable_true(self):
        """Recognizes known particle confusable pairs."""
        from myspellchecker.core.validation_strategies.confusable_helpers import (
            is_particle_confusable,
        )

        assert (
            is_particle_confusable(
                "\u1000",
                "\u1000\u102d\u102f",  # က → ကို
            )
            is True
        )

    def test_is_particle_confusable_false(self):
        """Returns False for non-particle pairs."""
        from myspellchecker.core.validation_strategies.confusable_helpers import (
            is_particle_confusable,
        )

        assert is_particle_confusable("abc", "def") is False
