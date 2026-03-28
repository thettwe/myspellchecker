"""
Response Builder.

This module provides utilities for building Response objects,
extracted from SpellChecker for better testability and reusability.
"""

from __future__ import annotations

from typing import Any

from myspellchecker.core.constants.core_constants import ET_SEMANTIC_ERROR
from myspellchecker.core.response import ContextError, Error, SyllableError, WordError

__all__ = [
    "build_response_metadata",
]


def build_response_metadata(
    errors: list[Error],
    layers_applied: list[str],
    processing_time: float,
    zawgyi_warning: Any | None = None,
) -> dict[str, Any]:
    """
    Build the response metadata dictionary.

    Args:
        errors: List of detected errors.
        layers_applied: List of validation layers that were applied.
        processing_time: Time spent processing (in seconds).
        zawgyi_warning: Optional Zawgyi detection warning.

    Returns:
        Metadata dictionary with error counts and processing info.

    Example:
        >>> from myspellchecker.core.response import SyllableError, WordError
        >>> errors = [
        ...     SyllableError(position=0, text="abc", suggestions=[]),
        ...     WordError(position=10, text="xyz", suggestions=[]),
        ... ]
        >>> metadata = build_response_metadata(errors, ["syllable", "word"], 0.05)
        >>> metadata["syllable_errors"]
        1
        >>> metadata["word_errors"]
        1
    """
    semantic_error_count = sum(
        1 for e in errors if isinstance(e, ContextError) and e.error_type == ET_SEMANTIC_ERROR
    )
    context_error_count = sum(
        1 for e in errors if isinstance(e, ContextError) and e.error_type != ET_SEMANTIC_ERROR
    )

    # Always include zawgyi_warning for consistent metadata structure
    # Note: processing_time is in seconds (from time.perf_counter())
    metadata: dict[str, Any] = {
        "total_errors": len(errors),
        "syllable_errors": sum(1 for e in errors if isinstance(e, SyllableError)),
        "word_errors": sum(1 for e in errors if isinstance(e, WordError)),
        "context_errors": context_error_count,
        "semantic_errors": semantic_error_count,
        "layers_applied": layers_applied,
        "processing_time": processing_time,  # Time in seconds
        "zawgyi_warning": (
            {
                "message": zawgyi_warning.message,
                "confidence": zawgyi_warning.confidence,
                "suggestion": zawgyi_warning.suggestion,
            }
            if zawgyi_warning and hasattr(zawgyi_warning, "message")
            else None
        ),
    }

    return metadata
