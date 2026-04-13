"""Loan word corrections YAML parser.

Parses loan_word_corrections.yaml into a dict mapping
incorrect forms to their correction entries for O(1) lookup.
"""

from __future__ import annotations

from typing import Any

from myspellchecker.utils.logging_utils import get_logger

logger = get_logger(__name__)

__all__ = [
    "parse_loan_word_corrections_config",
]


def parse_loan_word_corrections_config(
    config: dict[str, Any],
    *,
    loan_word_corrections: dict[str, dict[str, Any]],
) -> None:
    """Parse loan_word_corrections.yaml configuration.

    Builds a dict keyed by incorrect form for O(1) lookup during
    spell checking.  Each value contains the correct form, source
    language, source word, error type, and confidence.

    Mutates the provided container in-place.

    Args:
        config: Raw YAML dict loaded from loan_word_corrections.yaml.
        loan_word_corrections: Output dict to populate (incorrect -> info).
    """
    corrections = config.get("corrections", [])

    for item in corrections:
        try:
            incorrect = item.get("incorrect")
            correct = item.get("correct")
            if not incorrect or not correct:
                continue

            loan_word_corrections[incorrect] = {
                "correct": correct,
                "source_language": item.get("source_language", ""),
                "source_word": item.get("source_word", ""),
                "error_type": item.get("error_type", "loan_word_misspelling"),
                "confidence": item.get("confidence", 0.95),
            }
        except (KeyError, TypeError, ValueError, AttributeError) as e:
            logger.warning("Skipping bad loan word correction entry: %s", e)
