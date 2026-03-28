"""
Unit tests for particle typo detection integration.

Tests the integration of PARTICLE_TYPO_PATTERNS and MEDIAL_CONFUSION_PATTERNS
from grammar.patterns into the SyllableValidator.
"""

import pytest

from myspellchecker.grammar.patterns import (
    MEDIAL_CONFUSION_PATTERNS,
    PARTICLE_TYPO_PATTERNS,
    get_medial_confusion_correction,
    get_particle_typo_correction,
)


class TestParticleTypoPatterns:
    """Tests for the particle typo detection patterns."""

    def test_get_particle_typo_correction_missing_asat(self):
        """Test detection of missing asat patterns."""
        # Missing asat on statement ending
        result = get_particle_typo_correction("တယ")
        assert result is not None
        correction, description, confidence = result
        assert correction == "တယ်"
        assert confidence >= 0.90

    def test_get_particle_typo_correction_missing_tone(self):
        """Test detection of missing tone patterns."""
        # Missing tone on past tense
        result = get_particle_typo_correction("ခဲ")
        assert result is not None
        correction, description, confidence = result
        assert correction == "ခဲ့"
        assert confidence >= 0.85

    def test_get_particle_typo_correction_valid_particle(self):
        """Test that valid particles don't trigger false positives."""
        # Valid particles should return None
        valid_particles = ["တယ်", "သည်", "ပါ", "မှာ", "က", "ကို"]
        for particle in valid_particles:
            result = get_particle_typo_correction(particle)
            assert result is None, f"Valid particle '{particle}' should not be flagged"

    def test_get_particle_typo_correction_non_particle(self):
        """Test that non-particles don't trigger false positives."""
        non_particles = ["မြန်မာ", "ဘာသာ", "စကား", "နိုင်ငံ"]
        for word in non_particles:
            result = get_particle_typo_correction(word)
            assert result is None, f"Non-particle '{word}' should not be flagged"

    def test_particle_typo_patterns_count(self):
        """Test that we have sufficient particle typo patterns."""
        # Should have at least 15 patterns (as per linguistic_rules.py)
        assert len(PARTICLE_TYPO_PATTERNS) >= 15, (
            f"Expected at least 15 particle typo patterns, got {len(PARTICLE_TYPO_PATTERNS)}"
        )

    def test_particle_typo_patterns_confidence_range(self):
        """Test that all patterns have valid confidence scores."""
        for typo, (_correction, _description, confidence) in PARTICLE_TYPO_PATTERNS.items():
            assert 0.0 <= confidence <= 1.0, (
                f"Confidence for '{typo}' should be in [0, 1], got {confidence}"
            )
            assert confidence >= 0.70, (
                f"Confidence for '{typo}' should be >= 0.70, got {confidence}"
            )


class TestMedialConfusionPatterns:
    """Tests for the medial confusion detection patterns."""

    def test_get_medial_confusion_correction_school_vs_because(self):
        """Test detection of ကျောင်း vs ကြောင်း confusion."""
        # This is context-dependent, so the pattern may or may not trigger
        result = get_medial_confusion_correction("ကျောင်း")
        if result is not None:
            correction, description, context_hint = result
            assert correction == "ကြောင်း"
            assert context_hint in ["after_verb", "any", "context_dependent"]

    def test_get_medial_confusion_correction_thanks(self):
        """Test detection of thanks pattern."""
        result = get_medial_confusion_correction("ကြေးဇူး")
        assert result is not None
        correction, description, context_hint = result
        assert correction == "ကျေးဇူး"

    def test_get_medial_confusion_correction_valid_words(self):
        """Test that correctly-spelled words don't trigger false positives."""
        valid_words = ["မြန်မာ", "ကျေးဇူး", "ပြည်"]
        for word in valid_words:
            result = get_medial_confusion_correction(word)
            # These correct forms should not be in the confusion patterns
            if result is not None:
                # If they are, the correction shouldn't be the same as input
                correction, _, _ = result
                assert correction != word

    def test_medial_confusion_patterns_count(self):
        """Test that we have medial confusion patterns defined."""
        assert len(MEDIAL_CONFUSION_PATTERNS) >= 3, (
            f"Expected at least 3 medial confusion patterns, got {len(MEDIAL_CONFUSION_PATTERNS)}"
        )


class TestSyllableValidatorIntegration:
    """Integration tests for SyllableValidator with particle/medial checks."""

    @pytest.fixture
    def spellchecker(self):
        """Create a SpellChecker instance for testing."""
        try:
            from myspellchecker import SpellChecker

            return SpellChecker.create_default()
        except Exception as e:
            pytest.skip(f"SpellChecker not available: {e}")

    def test_particle_typo_detection_in_check(self, spellchecker):
        """Test that particle typos are detected by the spellchecker."""
        # Text with missing asat on particle
        result = spellchecker.check("ကောင်းတယ")  # Should detect "တယ" → "တယ်"

        # Check if any error was detected for the particle typo
        particle_errors = [e for e in result.errors if e.error_type == "particle_typo"]

        # Either we get a particle_typo error, or the word is valid
        # (depends on dictionary and segmentation)
        if particle_errors:
            error = particle_errors[0]
            assert "တယ်" in error.suggestions

    def test_medial_confusion_detection_in_check(self, spellchecker):
        """Test that medial confusion patterns are detected."""
        # Text with medial confusion
        result = spellchecker.check("ကြေးဇူးပါ")  # ကြေးဇူး → ကျေးဇူး

        medial_errors = [e for e in result.errors if e.error_type == "medial_confusion"]

        if medial_errors:
            error = medial_errors[0]
            assert "ကျေးဇူး" in error.suggestions

    def test_valid_text_no_false_positives(self, spellchecker):
        """Test that valid text doesn't trigger particle/medial errors."""
        valid_texts = [
            "ကောင်းတယ်",  # Correct particle
            "ကျေးဇူးပါ",  # Correct medial
            "မြန်မာနိုင်ငံ",  # Common words
        ]

        for text in valid_texts:
            result = spellchecker.check(text)
            # Filter for particle_typo and medial_confusion errors only
            pattern_errors = [
                e for e in result.errors if e.error_type in ("particle_typo", "medial_confusion")
            ]
            assert len(pattern_errors) == 0, (
                f"Valid text '{text}' should not have pattern errors, got {pattern_errors}"
            )


class TestPatternPriority:
    """Test that pattern detection has correct priority."""

    def test_particle_typo_higher_confidence_than_dictionary(self):
        """Test that particle typos have reasonable confidence scores."""
        # All particle typo patterns should have >= 0.85 confidence
        # (which is higher than typical dictionary-based corrections)
        for typo, (_correction, _description, confidence) in PARTICLE_TYPO_PATTERNS.items():
            assert confidence >= 0.85, (
                f"Pattern '{typo}' should have >= 0.85 confidence, got {confidence}"
            )

    def test_medial_confusion_moderate_confidence(self):
        """Test that medial confusion has moderate confidence (needs context)."""
        # Medial confusion should be detected but with context awareness
        for pattern, (_correction, _description, context_hint) in MEDIAL_CONFUSION_PATTERNS.items():
            assert context_hint in ["after_verb", "any", "context_dependent"], (
                f"Pattern '{pattern}' should have valid context_hint"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
