"""Tests for detector threshold dataclasses.

Verifies that:
1. Mixin classes expose the correct threshold dataclass as a class-level default.
2. SpellChecker instances inherit mixin threshold defaults.
"""

from __future__ import annotations

from myspellchecker.core.config import SpellCheckerConfig
from myspellchecker.core.constants.detector_thresholds import (
    DEFAULT_COMPOUND_THRESHOLDS,
    DEFAULT_PARTICLE_THRESHOLDS,
)
from myspellchecker.core.detectors.post_norm_mixins.compound_detection_mixin import (
    CompoundDetectionMixin,
)
from myspellchecker.core.detectors.post_norm_mixins.particle_detection_mixin import (
    ParticleDetectionMixin,
)


class TestMixinClassDefaults:
    """Mixin classes expose the correct threshold dataclass."""

    def test_compound_mixin_has_default_thresholds(self) -> None:
        assert CompoundDetectionMixin._compound_thresholds is DEFAULT_COMPOUND_THRESHOLDS

    def test_particle_mixin_has_default_thresholds(self) -> None:
        assert ParticleDetectionMixin._particle_thresholds is DEFAULT_PARTICLE_THRESHOLDS


class TestSpellCheckerInheritsThresholds:
    """SpellChecker instances inherit mixin threshold defaults."""

    def test_inherits_compound_thresholds(self) -> None:
        config = SpellCheckerConfig(fallback_to_empty_provider=True)
        from myspellchecker.core.spellchecker import SpellChecker

        checker = SpellChecker(config=config)
        assert checker._compound_thresholds is DEFAULT_COMPOUND_THRESHOLDS

    def test_inherits_particle_thresholds(self) -> None:
        config = SpellCheckerConfig(fallback_to_empty_provider=True)
        from myspellchecker.core.spellchecker import SpellChecker

        checker = SpellChecker(config=config)
        assert checker._particle_thresholds is DEFAULT_PARTICLE_THRESHOLDS

    def test_compound_threshold_values_accessible(self) -> None:
        """Verify threshold values are accessible through the dataclass."""
        config = SpellCheckerConfig(fallback_to_empty_provider=True)
        from myspellchecker.core.spellchecker import SpellChecker

        checker = SpellChecker(config=config)
        t = checker._compound_thresholds
        assert t.rejoin_min_correction_freq == 5000
        assert t.variant_min_freq_ratio == 4.0
        assert t.visarga_confidence_max == 0.95

    def test_particle_threshold_values_accessible(self) -> None:
        """Verify threshold values are accessible through the dataclass."""
        config = SpellCheckerConfig(fallback_to_empty_provider=True)
        from myspellchecker.core.spellchecker import SpellChecker

        checker = SpellChecker(config=config)
        t = checker._particle_thresholds
        assert t.missing_asat_standalone_min_freq == 500
        assert t.structural_error_max_count == 3
        assert t.visarga_compound_min_ratio == 10.0
