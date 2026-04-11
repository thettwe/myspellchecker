"""
Unified Normalization Service for Myanmar Text.

This module provides a centralized NormalizationService that consolidates
all text normalization logic into a single, consistent interface.

Architecture Role
=================
This is **Layer 3** of the normalization hierarchy - the service layer that
provides purpose-specific normalization through a clean, unified interface.

Normalization Hierarchy::

    Layer 3: NormalizationService (this module)
        Purpose-specific methods: for_spell_checking(), for_dictionary_lookup()
        Uses presets to configure normalization behavior
        Thread-safe singleton pattern for shared access
        ↓ calls
    Layer 2: normalize.py (Python wrappers)
        Composes Cython primitives into higher-level functions
        Handles Zawgyi detection/conversion, character variants, nasal endings
        ↓ calls
    Layer 1: normalize_c.pyx (Cython/C++)
        Performance-critical primitives (~20x faster than pure Python)
        remove_zero_width_chars, reorder_myanmar_diacritics, etc.

When to Use This Module
=======================
**Use NormalizationService when:**
- You need consistent normalization across components
- You want purpose-specific behavior (spell checking vs display)
- You're working with user-facing code that needs different modes
- You need thread-safe normalization access

**Use normalize.py directly when:**
- You need custom normalization combinations
- You're implementing new normalization logic
- You need access to Zawgyi detection/conversion utilities

**Use normalize_c directly when:**
- You need maximum performance in hot paths
- You're implementing new Cython-level features
- You need GIL-free operations for parallel processing

Available Methods
=================
- ``for_spell_checking()``: Fast normalization for validation pipeline
- ``for_dictionary_lookup()``: Full normalization with Zawgyi handling
- ``for_comparison()``: Aggressive normalization for text comparison
- ``for_display()``: Minimal normalization preserving user formatting
- ``for_ingestion()``: Full normalization for corpus data processing

Presets
=======
Pre-configured NormalizationOptions for common use cases:
- ``PRESET_SPELL_CHECK``: Fast, no Zawgyi conversion
- ``PRESET_DICTIONARY_LOOKUP``: Full with Zawgyi detection/conversion
- ``PRESET_COMPARISON``: Aggressive for matching
- ``PRESET_DISPLAY``: Minimal, preserves formatting
- ``PRESET_INGESTION``: Full for corpus processing

Example Usage
=============
>>> from myspellchecker.text import get_normalization_service
>>> service = get_normalization_service()
>>>
>>> # Standard spell checking
>>> normalized = service.for_spell_checking("မြန်မာ")
>>>
>>> # Dictionary lookup (with Zawgyi handling)
>>> normalized = service.for_dictionary_lookup(user_input)
>>>
>>> # Custom normalization with preset
>>> from myspellchecker.text import NormalizationOptions
>>> custom = NormalizationOptions(convert_zawgyi=True, lowercase=True)
>>> normalized = service.normalize(text, custom)

Thread Safety
=============
The module-level singleton is thread-safe via ThreadSafeSingleton.
Individual NormalizationService instances are also thread-safe as
they only use thread-safe Cython functions and stateless operations.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from typing import Literal

from myspellchecker.core.config.text_configs import ZawgyiConfig
from myspellchecker.text.normalize import (
    normalize_character_variants,
    normalize_e_vowel_tall_aa,
    normalize_tall_aa_after_wa,
    normalize_u_vowel_with_asat,
)
from myspellchecker.text.normalize_c import (
    get_myanmar_ratio as c_get_myanmar_ratio,
)
from myspellchecker.text.normalize_c import (
    remove_zero_width_chars as c_remove_zero_width,
)
from myspellchecker.text.normalize_c import (
    reorder_myanmar_diacritics as c_reorder_diacritics,
)
from myspellchecker.utils.logging_utils import get_logger
from myspellchecker.utils.singleton import ThreadSafeSingleton

logger = get_logger(__name__)

__all__ = [
    "NormalizationOptions",
    "NormalizationService",
    "get_normalization_service",
    "normalize_for_comparison",
    "normalize_for_lookup",
    "normalize_for_spell_checking",
]


@dataclass(frozen=True)
class NormalizationOptions:
    """Configuration options for normalization operations.

    Attributes:
        unicode_form: Unicode normalization form to apply.
        remove_zero_width: Whether to remove zero-width characters.
        reorder_diacritics: Whether to apply Myanmar-specific reordering.
        convert_zawgyi: Whether to detect and convert Zawgyi encoding.
        strip_whitespace: Whether to strip leading/trailing whitespace.
        lowercase: Whether to convert to lowercase (for non-Myanmar text).
        character_variants: Whether to normalize Myanmar character variants
            (e.g., Tall-AA → AA, zero → Wa). Uses normalize_zero_to_wa=False
            to preserve numeric data.
    """

    unicode_form: Literal["NFC", "NFD", "NFKC", "NFKD"] = "NFC"
    remove_zero_width: bool = True
    reorder_diacritics: bool = True
    convert_zawgyi: bool = False
    strip_whitespace: bool = True
    lowercase: bool = False
    character_variants: bool = False


# Pre-defined normalization presets
PRESET_SPELL_CHECK = NormalizationOptions(
    unicode_form="NFC",
    remove_zero_width=True,
    reorder_diacritics=True,
    convert_zawgyi=False,
    strip_whitespace=True,
    lowercase=False,
)

PRESET_DICTIONARY_LOOKUP = NormalizationOptions(
    unicode_form="NFC",
    remove_zero_width=True,
    reorder_diacritics=True,
    convert_zawgyi=True,
    strip_whitespace=True,
    lowercase=False,
    character_variants=True,
)

PRESET_COMPARISON = NormalizationOptions(
    unicode_form="NFC",
    remove_zero_width=True,
    reorder_diacritics=True,
    convert_zawgyi=True,
    strip_whitespace=True,
    lowercase=False,
)

PRESET_DISPLAY = NormalizationOptions(
    unicode_form="NFC",
    remove_zero_width=False,
    reorder_diacritics=True,
    convert_zawgyi=False,
    strip_whitespace=False,
    lowercase=False,
)

PRESET_INGESTION = NormalizationOptions(
    unicode_form="NFC",
    remove_zero_width=True,
    reorder_diacritics=True,
    convert_zawgyi=True,
    strip_whitespace=True,
    lowercase=False,
    character_variants=True,
)


@dataclass
class NormalizationService:
    """Unified normalization service for Myanmar text.

    This service provides a single source of truth for all text normalization
    operations. It ensures consistent normalization behavior across all
    components of the spell checker.

    Usage:
        >>> service = NormalizationService()
        >>>
        >>> # For spell checking (fast, no Zawgyi conversion)
        >>> normalized = service.for_spell_checking("မြန်မာ")
        >>>
        >>> # For dictionary lookups (includes Zawgyi detection)
        >>> normalized = service.for_dictionary_lookup(user_input)
        >>>
        >>> # For text comparison
        >>> normalized = service.for_comparison(text1)
        >>>
        >>> # Custom normalization
        >>> normalized = service.normalize(text, PRESET_INGESTION)

    Thread Safety:
        This service is thread-safe. The underlying Cython functions and
        Zawgyi detector use thread-safe caching.
    """

    zawgyi_config: ZawgyiConfig = field(default_factory=ZawgyiConfig)

    def normalize(
        self,
        text: str,
        options: NormalizationOptions | None = None,
    ) -> str:
        """Apply normalization with specified options.

        This is the core normalization method that other methods delegate to.
        It provides a consistent normalization pipeline with configurable steps.

        Args:
            text: Input text to normalize.
            options: Normalization options. Defaults to PRESET_SPELL_CHECK.

        Returns:
            Normalized text string.

        Edge Cases Handled:
            - **Empty/None-like strings**: Returns input unchanged (early exit)
            - **Whitespace-only**: Returns empty string (if strip_whitespace=True)
            - **Zawgyi conversion failure**: Logs debug message, continues
            - **Non-Myanmar text**: Passed through with Unicode normalization
            - **Mixed Myanmar/non-Myanmar**: Each portion handled appropriately

        Pipeline Steps:
            1. Strip whitespace (if enabled, early for efficiency)
            2. Zawgyi conversion (if enabled, before Unicode normalization)
            3. Unicode normalization (NFC/NFD/NFKC/NFKD)
            4. Zero-width character removal (ZWSP, ZWNJ, ZWJ, BOM)
            5. Myanmar diacritic reordering (UTN #11 compliant)
            6. Sequence-level normalizations (Tall-AA, E-vowel, U-vowel fixes)
            7. Character variant normalization (Extended-A mappings, etc.)
            8. Lowercase conversion (if enabled, for non-Myanmar)
            9. Final strip (catch edge cases from earlier steps)

        Note:
            The step order is important. Zawgyi conversion must happen before
            Unicode normalization, and diacritic reordering after zero-width
            removal for correct results.
        """
        if not text:
            return text

        opts = options or PRESET_SPELL_CHECK
        result = text

        # Step 1: Strip whitespace (if enabled, do early)
        if opts.strip_whitespace:
            result = result.strip()
            if not result:
                return result

        # Step 2: Zawgyi conversion (before Unicode normalization)
        if opts.convert_zawgyi:
            result = self._convert_zawgyi_if_detected(result)

        # Step 3: Unicode normalization
        result = unicodedata.normalize(opts.unicode_form, result)

        # Step 4: Remove zero-width characters
        if opts.remove_zero_width:
            result = c_remove_zero_width(result)

        # Step 5: Myanmar-specific diacritic reordering
        if opts.reorder_diacritics:
            result = c_reorder_diacritics(result)

        # Step 6: Sequence-level normalizations (fix invalid sequences unconditionally)
        # These correct genuinely wrong character sequences, not variant preferences.
        result = normalize_tall_aa_after_wa(result)
        result = normalize_e_vowel_tall_aa(result)
        result = normalize_u_vowel_with_asat(result)

        # Step 7: Character variant normalization (Extended-A mappings, etc.)
        if opts.character_variants:
            result = normalize_character_variants(result, normalize_zero_to_wa=False)

        # Step 8: Lowercase (for non-Myanmar text comparison)
        if opts.lowercase:
            result = result.lower()

        # Step 9: Final strip (catch any edge cases)
        if opts.strip_whitespace:
            result = result.strip()

        return result

    def for_spell_checking(self, text: str) -> str:
        """Normalize text for the spell checking validation pipeline.

        This is the standard normalization used during spell checking.
        It's fast and does NOT convert Zawgyi (Zawgyi detection is
        handled separately to provide user warnings).

        Pipeline:
            1. Strip whitespace
            2. Unicode NFC normalization
            3. Remove zero-width characters
            4. Myanmar diacritic reordering

        Args:
            text: Input text to normalize.

        Returns:
            Normalized text ready for spell checking.

        Example:
            >>> service = NormalizationService()
            >>> service.for_spell_checking("  မြန်မာ  ")
            'မြန်မာ'
        """
        return self.normalize(text, PRESET_SPELL_CHECK)

    def for_dictionary_lookup(self, text: str) -> str:
        """Normalize text for dictionary or index lookups.

        This is the complete normalization used before any dictionary
        lookup, index access, or database query. It includes Zawgyi
        detection and conversion to ensure consistent matching.

        Pipeline:
            1. Strip whitespace
            2. Zawgyi to Unicode conversion (if detected)
            3. Unicode NFC normalization
            4. Remove zero-width characters
            5. Myanmar diacritic reordering

        Args:
            text: Input text to normalize for lookup.

        Returns:
            Normalized text suitable for dictionary lookups.

        Example:
            >>> service = NormalizationService()
            >>> service.for_dictionary_lookup("မြန်မာ")
            'မြန်မာ'

        Note:
            This function is idempotent - calling it multiple times
            on the same text produces identical results.
        """
        return self.normalize(text, PRESET_DICTIONARY_LOOKUP)

    def for_comparison(self, text: str) -> str:
        """Normalize text for comparison operations.

        This provides aggressive normalization to maximize matching
        chances when comparing text. Use this when checking if two
        pieces of text are semantically equivalent.

        Pipeline:
            1. Strip whitespace
            2. Zawgyi to Unicode conversion (if detected)
            3. Unicode NFC normalization
            4. Remove zero-width characters
            5. Myanmar diacritic reordering

        Args:
            text: Input text to normalize for comparison.

        Returns:
            Normalized text for comparison.

        Example:
            >>> service = NormalizationService()
            >>> a = service.for_comparison(user_input)
            >>> b = service.for_comparison(dictionary_entry)
            >>> if a == b:
            ...     print("Match!")
        """
        return self.normalize(text, PRESET_COMPARISON)

    def for_display(self, text: str) -> str:
        """Normalize text for display to users.

        This provides minimal normalization that preserves user
        formatting while fixing character ordering issues.

        Pipeline:
            1. Unicode NFC normalization
            2. Myanmar diacritic reordering

        Args:
            text: Input text to normalize for display.

        Returns:
            Normalized text for display.

        Note:
            Does NOT strip whitespace or remove zero-width characters
            to preserve user formatting intentions.
        """
        return self.normalize(text, PRESET_DISPLAY)

    def for_ingestion(self, text: str) -> str:
        """Normalize text during corpus ingestion.

        This is used by the data pipeline when ingesting corpus files
        to build the dictionary database. It includes full normalization
        with Zawgyi conversion and character variant mapping.

        Pipeline:
            1. Strip whitespace
            2. Zawgyi to Unicode conversion (if detected)
            3. Unicode NFC normalization
            4. Remove zero-width characters
            5. Myanmar diacritic reordering
            6. Character variant normalization (Tall-AA, etc.)

        Args:
            text: Input text from corpus file.

        Returns:
            Normalized text for database storage.
        """
        return self.normalize(text, PRESET_INGESTION)

    def is_myanmar_text(self, text: str, *, allow_extended: bool = False) -> bool:
        """Check if text is primarily Myanmar script.

        Args:
            text: Text to check.
            allow_extended: If False (default), only core Burmese characters count.
                           If True, Extended Myanmar blocks count as Myanmar.
                           Default is Burmese-only.

        Returns:
            True if Myanmar character proportion >= threshold.
        """
        if not text:
            return False

        ratio = c_get_myanmar_ratio(text, allow_extended)
        return bool(ratio >= self.zawgyi_config.myanmar_text_threshold)

    def _convert_zawgyi_if_detected(self, text: str) -> str:
        """Convert Zawgyi to Unicode if detected.

        Internal method that handles Zawgyi detection and conversion.
        Uses lazy imports to avoid loading heavy dependencies until needed.
        """
        if not text:
            return text

        # Lazy imports to avoid circular dependency
        from myspellchecker.text.zawgyi_support import (
            _convert_zawgyi_internal as _convert_zawgyi,
        )
        from myspellchecker.text.zawgyi_support import (
            get_zawgyi_detector as _get_zawgyi_detector,
        )
        from myspellchecker.text.zawgyi_support import (
            is_zawgyi_converter_available as _is_zawgyi_converter_available,
        )

        # Check if conversion tools are available
        if not _is_zawgyi_converter_available():
            return text

        # Get detector
        try:
            detector = _get_zawgyi_detector()
            if detector is None:
                return text

            prob = detector.get_zawgyi_probability(text)
            if prob < self.zawgyi_config.conversion_threshold:
                return text

            # Convert
            return _convert_zawgyi(text)

        except (RuntimeError, UnicodeError, ValueError, KeyError) as e:
            logger.debug(f"Zawgyi conversion skipped: {e}")
            return text


# Module-level singleton for convenience (thread-safe)
_singleton: ThreadSafeSingleton[NormalizationService] = ThreadSafeSingleton()


def get_normalization_service(
    zawgyi_config: ZawgyiConfig | None = None,
) -> NormalizationService:
    """Get a NormalizationService instance.

    Returns a cached default instance, or creates a new instance
    with custom configuration if provided. Thread-safe for singleton creation.

    Args:
        zawgyi_config: Optional custom Zawgyi configuration.

    Returns:
        NormalizationService instance.

    Example:
        >>> service = get_normalization_service()
        >>> normalized = service.for_spell_checking(text)

    Note:
        Uses ThreadSafeSingleton for thread-safe singleton initialization.
    """
    if zawgyi_config is not None:
        # Custom config bypasses the singleton cache
        return NormalizationService(zawgyi_config=zawgyi_config)

    return _singleton.get(NormalizationService)


# ============================================================================
# Convenience Functions (backward compatible)
# ============================================================================


def normalize_for_spell_checking(text: str) -> str:
    """Convenience function for spell checking normalization.

    Equivalent to: get_normalization_service().for_spell_checking(text)
    """
    return get_normalization_service().for_spell_checking(text)


def normalize_for_lookup(text: str) -> str:
    """Convenience function for dictionary lookup normalization.

    Equivalent to: get_normalization_service().for_dictionary_lookup(text)
    """
    return get_normalization_service().for_dictionary_lookup(text)


def normalize_for_comparison(text: str) -> str:
    """Convenience function for comparison normalization.

    Equivalent to: get_normalization_service().for_comparison(text)
    """
    return get_normalization_service().for_comparison(text)
