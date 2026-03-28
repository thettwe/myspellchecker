"""Comprehensive tests for text/normalization_service.py.

Tests cover:
- NormalizationOptions dataclass and presets
- NormalizationService methods
- Convenience functions
- Zawgyi handling
"""

from unittest.mock import MagicMock, patch

import pytest

from myspellchecker.core.config.text_configs import ZawgyiConfig
from myspellchecker.text.normalization_service import (
    NormalizationOptions,
    NormalizationService,
    get_normalization_service,
    normalize_for_comparison,
    normalize_for_lookup,
    normalize_for_spell_checking,
)
from myspellchecker.text.zawgyi_support import (
    _convert_zawgyi_internal as _convert_zawgyi,
)
from myspellchecker.text.zawgyi_support import (
    get_zawgyi_detector as _get_zawgyi_detector,
)


class TestNormalizationOptions:
    """Tests for NormalizationOptions dataclass."""

    def test_default_values(self):
        """Test default option values."""
        opts = NormalizationOptions()
        assert opts.unicode_form == "NFC"
        assert opts.remove_zero_width is True
        assert opts.reorder_diacritics is True
        assert opts.convert_zawgyi is False
        assert opts.strip_whitespace is True
        assert opts.lowercase is False

    def test_custom_values(self):
        """Test custom option values."""
        opts = NormalizationOptions(
            unicode_form="NFD",
            remove_zero_width=False,
            reorder_diacritics=False,
            convert_zawgyi=True,
            strip_whitespace=False,
            lowercase=True,
        )
        assert opts.unicode_form == "NFD"
        assert opts.remove_zero_width is False
        assert opts.reorder_diacritics is False
        assert opts.convert_zawgyi is True
        assert opts.strip_whitespace is False
        assert opts.lowercase is True

    def test_frozen(self):
        """Test that options are immutable."""
        opts = NormalizationOptions()
        with pytest.raises(AttributeError):
            opts.unicode_form = "NFD"  # type: ignore


class TestNormalizationService:
    """Tests for NormalizationService class."""

    @pytest.fixture
    def service(self):
        """Create a NormalizationService instance."""
        return NormalizationService()

    def test_initialization_with_config(self):
        """Test service initialization with custom config."""
        config = ZawgyiConfig(conversion_threshold=0.8)
        service = NormalizationService(zawgyi_config=config)
        assert service.zawgyi_config.conversion_threshold == 0.8

    def test_normalize_empty_string(self, service):
        """Test normalizing empty string."""
        result = service.normalize("")
        assert result == ""

    def test_normalize_whitespace_only(self, service):
        """Test normalizing whitespace-only string."""
        result = service.normalize("   ")
        assert result == ""

    def test_normalize_basic(self, service):
        """Test basic normalization."""
        result = service.normalize("  ကာ  ")
        assert result == "ကာ"

    def test_normalize_removes_zero_width(self, service):
        """Test zero-width character removal."""
        text = "က\u200bာ"  # Zero-width space
        result = service.normalize(text)
        assert "\u200b" not in result

    def test_normalize_with_custom_options(self, service):
        """Test normalization with custom options."""
        opts = NormalizationOptions(
            strip_whitespace=False,
            remove_zero_width=False,
        )
        result = service.normalize("  ကာ  ", opts)
        # Whitespace preserved
        assert result.startswith(" ") or result.endswith(" ")

    def test_normalize_lowercase(self, service):
        """Test lowercase option."""
        opts = NormalizationOptions(lowercase=True)
        result = service.normalize("HELLO", opts)
        assert result == "hello"

    def test_for_display_preserves_whitespace(self, service):
        """Test for_display method preserves whitespace."""
        result = service.for_display("  ကာ  ")
        assert "ကာ" in result

    def test_is_myanmar_text_true(self, service):
        """Test is_myanmar_text returns True for Myanmar text."""
        assert service.is_myanmar_text("ကာ") is True
        assert service.is_myanmar_text("မြန်မာ") is True

    def test_is_myanmar_text_false(self, service):
        """Test is_myanmar_text returns False for non-Myanmar text."""
        assert service.is_myanmar_text("hello") is False
        assert service.is_myanmar_text("12345") is False

    def test_is_myanmar_text_empty(self, service):
        """Test is_myanmar_text returns False for empty string."""
        assert service.is_myanmar_text("") is False

    def test_idempotency(self, service):
        """Test that normalization is idempotent."""
        text = "  ကာ  "
        result1 = service.for_dictionary_lookup(text)
        result2 = service.for_dictionary_lookup(result1)
        assert result1 == result2


class TestZawgyiConversion:
    """Tests for Zawgyi detection and conversion."""

    @pytest.fixture
    def service(self):
        """Create a NormalizationService instance."""
        return NormalizationService()

    def test_convert_zawgyi_if_detected_empty(self, service):
        """Test Zawgyi conversion with empty string."""
        result = service._convert_zawgyi_if_detected("")
        assert result == ""

    @patch("myspellchecker.text.zawgyi_support.is_zawgyi_converter_available")
    def test_convert_zawgyi_converter_unavailable(self, mock_available, service):
        """Test when Zawgyi converter is not available."""
        mock_available.return_value = False
        result = service._convert_zawgyi_if_detected("test")
        assert result == "test"

    @patch("myspellchecker.text.zawgyi_support._convert_zawgyi_internal")
    @patch("myspellchecker.text.zawgyi_support.get_zawgyi_detector")
    @patch("myspellchecker.text.zawgyi_support.is_zawgyi_converter_available")
    def test_convert_zawgyi_below_threshold(
        self, mock_available, mock_detector, mock_convert, service
    ):
        """Test when Zawgyi probability is below threshold."""
        mock_available.return_value = True
        mock_detector_instance = MagicMock()
        mock_detector_instance.get_zawgyi_probability.return_value = 0.1
        mock_detector.return_value = mock_detector_instance

        result = service._convert_zawgyi_if_detected("test")
        assert result == "test"
        mock_convert.assert_not_called()

    @patch("myspellchecker.text.zawgyi_support._convert_zawgyi_internal")
    @patch("myspellchecker.text.zawgyi_support.get_zawgyi_detector")
    @patch("myspellchecker.text.zawgyi_support.is_zawgyi_converter_available")
    def test_convert_zawgyi_above_threshold(
        self, mock_available, mock_detector, mock_convert, service
    ):
        """Test when Zawgyi probability is above threshold."""
        mock_available.return_value = True
        mock_detector_instance = MagicMock()
        mock_detector_instance.get_zawgyi_probability.return_value = 0.99
        mock_detector.return_value = mock_detector_instance
        mock_convert.return_value = "converted"

        result = service._convert_zawgyi_if_detected("zawgyi_text")
        assert result == "converted"
        mock_convert.assert_called_once_with("zawgyi_text")

    @patch("myspellchecker.text.zawgyi_support.get_zawgyi_detector")
    @patch("myspellchecker.text.zawgyi_support.is_zawgyi_converter_available")
    def test_convert_zawgyi_exception_handling(self, mock_available, mock_detector, service):
        """Test exception handling during Zawgyi conversion."""
        mock_available.return_value = True
        mock_detector.side_effect = RuntimeError("Test error")

        # Should not raise, just return original text
        result = service._convert_zawgyi_if_detected("test")
        assert result == "test"


class TestZawgyiHelperFunctions:
    """Tests for Zawgyi helper functions."""

    def test_get_zawgyi_detector_cached(self):
        """Test that _get_zawgyi_detector is cached."""
        _get_zawgyi_detector.cache_clear()
        result1 = _get_zawgyi_detector()
        result2 = _get_zawgyi_detector()
        assert result1 is result2

    @patch("myanmar.converter")
    def test_convert_zawgyi_success(self, mock_converter):
        """Test successful Zawgyi conversion."""
        mock_converter.convert.return_value = "unicode_text"
        result = _convert_zawgyi("zawgyi_text")
        assert result == "unicode_text"

    @patch("myanmar.converter")
    def test_convert_zawgyi_failure(self, mock_converter):
        """Test Zawgyi conversion failure handling."""
        mock_converter.convert.side_effect = RuntimeError("Conversion error")
        result = _convert_zawgyi("zawgyi_text")
        assert result == "zawgyi_text"


class TestGetNormalizationService:
    """Tests for get_normalization_service function."""

    def test_get_service_cached(self):
        """Test that default service is cached."""
        service1 = get_normalization_service()
        service2 = get_normalization_service()
        assert service1 is service2

    def test_custom_config_creates_new_instance(self):
        """Test that custom config creates new instance."""
        service1 = get_normalization_service()
        config = ZawgyiConfig(conversion_threshold=0.5)
        service2 = get_normalization_service(zawgyi_config=config)
        assert service1 is not service2


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_normalize_for_spell_checking(self):
        """Test normalize_for_spell_checking function."""
        result = normalize_for_spell_checking("  ကာ  ")
        assert result == "ကာ"

    def test_normalize_for_lookup(self):
        """Test normalize_for_lookup function."""
        result = normalize_for_lookup("  ကာ  ")
        assert result == "ကာ"

    def test_normalize_for_comparison(self):
        """Test normalize_for_comparison function."""
        result = normalize_for_comparison("  ကာ  ")
        assert result == "ကာ"


class TestUnicodeNormalization:
    """Tests for Unicode normalization forms."""

    @pytest.fixture
    def service(self):
        """Create a NormalizationService instance."""
        return NormalizationService()

    def test_nfc_normalization(self, service):
        """Test NFC normalization."""
        opts = NormalizationOptions(unicode_form="NFC")
        result = service.normalize("café", opts)
        assert result == "café"

    def test_nfkc_normalization(self, service):
        """Test NFKC normalization decomposes ligatures."""
        opts = NormalizationOptions(unicode_form="NFKC")
        result = service.normalize("ﬁ", opts)  # fi ligature
        assert result == "fi"
