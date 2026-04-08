"""
Per-request overrides for SpellChecker.check().

Allows callers to override instance-level configuration on a per-call basis
without modifying the SpellChecker's config object.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["CheckOptions"]


@dataclass
class CheckOptions:
    """Per-request overrides for SpellChecker.check().

    Any field set to None falls through to instance defaults.

    Attributes:
        context_checking: If False, skip context validation (N-gram, semantic,
            grammar strategies). If None, uses the instance default.
        grammar_checking: If False, filter out grammar errors from the result.
            If None, uses the instance default.
        max_suggestions: Limit the number of suggestions returned per error.
            If None, uses the instance default.
        use_semantic: Override semantic checking. If None, uses the instance
            default or the explicit ``use_semantic`` parameter.

    Example:
        >>> from myspellchecker import SpellChecker, CheckOptions
        >>> checker = SpellChecker()
        >>> # Limit suggestions to 3 per error, disable grammar
        >>> opts = CheckOptions(max_suggestions=3, grammar_checking=False)
        >>> result = checker.check("မြန်မာနိုင်ငံ", options=opts)
    """

    context_checking: bool | None = None
    grammar_checking: bool | None = None
    max_suggestions: int | None = None
    use_semantic: bool | None = None
