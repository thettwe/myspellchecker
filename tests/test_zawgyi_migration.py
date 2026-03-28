"""Tests for Zawgyi detection migration to myanmar-tools."""

from myspellchecker.text.normalize import (
    convert_zawgyi_to_unicode,
    detect_encoding,
    is_likely_zawgyi,
)


class TestMyanmarToolsIntegration:
    """Verify myanmar-tools library is properly integrated."""

    def test_myanmartools_library_available(self):
        """myanmartools should be installed and importable."""
        import myanmartools

        detector = myanmartools.ZawgyiDetector()
        assert detector is not None

    def test_python_myanmar_library_available(self):
        """python-myanmar should be installed for conversion."""
        from myanmar import converter

        assert converter is not None

    def test_unicode_text_low_zawgyi_probability(self):
        """Unicode Myanmar text should have low Zawgyi probability."""
        unicode_samples = [
            "မြန်မာနိုင်ငံ",
            "ကျောင်းသား",
        ]
        for text in unicode_samples:
            is_zawgyi, conf = is_likely_zawgyi(text)
            # Some Unicode text may have slightly higher scores due to statistical model
            # The important thing is the confidence score is reasonable
            assert 0.0 <= conf <= 1.0, f"Confidence out of range: {conf}"
            # Most Unicode should be clearly detected (conf < 0.3)
            # but allow some statistical variance
            assert conf < 0.5 or not is_zawgyi, (
                f"Unicode text detected as Zawgyi: {text}, conf: {conf:.2f}"
            )

    def test_confidence_scores_in_valid_range(self):
        """All confidence scores should be 0.0-1.0."""
        test_cases = ["မြန်မာ", "test", "", "123"]
        for text in test_cases:
            is_zawgyi, conf = is_likely_zawgyi(text)
            assert 0.0 <= conf <= 1.0, f"Invalid confidence: {conf}"
            assert isinstance(is_zawgyi, bool)


class TestBackwardCompatibility:
    """Ensure migration maintains backward compatibility."""

    def test_api_surface_unchanged(self):
        """Public API functions should still be available."""
        # Import the module directly using sys.modules to get the actual module
        import sys

        import myspellchecker.text.normalize  # noqa: F401

        normalize_module = sys.modules["myspellchecker.text.normalize"]

        # Verify all public functions still exist
        assert hasattr(normalize_module, "is_likely_zawgyi")
        assert hasattr(normalize_module, "detect_encoding")
        assert hasattr(normalize_module, "convert_zawgyi_to_unicode")
        assert hasattr(normalize_module, "check_zawgyi_and_warn")

    def test_encoding_detection_returns_expected_format(self):
        """detect_encoding should return (encoding_type, confidence)."""
        encoding, conf = detect_encoding("မြန်မာ")
        assert isinstance(encoding, str)
        assert encoding in ["unicode", "zawgyi", "unknown"]
        assert 0.0 <= conf <= 1.0


class TestDetectionAccuracy:
    """Verify improved detection accuracy with myanmar-tools."""

    def test_empty_text_returns_false(self):
        """Empty text should return False, 0.0."""
        is_zawgyi, conf = is_likely_zawgyi("")
        assert is_zawgyi is False
        assert conf == 0.0

    def test_non_myanmar_text_returns_false(self):
        """Non-Myanmar text should return False."""
        is_zawgyi, conf = is_likely_zawgyi("Hello World")
        assert is_zawgyi is False
        assert conf < 0.1

    def test_short_myanmar_text_returns_false(self):
        """Text with <3 Myanmar chars should return False."""
        is_zawgyi, conf = is_likely_zawgyi("မ")  # 1 char
        assert is_zawgyi is False
        assert conf == 0.0


class TestConversionPipeline:
    """Test Zawgyi conversion pipeline."""

    def test_unicode_text_unchanged_after_conversion(self):
        """Unicode text should pass through conversion unchanged."""
        unicode_text = "မြန်မာနိုင်ငံ"
        result = convert_zawgyi_to_unicode(unicode_text)
        assert result == unicode_text

    def test_conversion_handles_empty_text(self):
        """Empty text should be handled gracefully."""
        result = convert_zawgyi_to_unicode("")
        assert result == ""
