"""Grammar validation package for Myanmar language rules."""

from __future__ import annotations

from typing import Any

# Eager imports for lightweight dataclasses and error types
from .checkers import (
    AspectError,
    AspectInfo,
    AspectPattern,
    ClassifierError,
    CompoundError,
    CompoundInfo,
    NegationError,
    NegationInfo,
    RegisterError,
    RegisterInfo,
)
from .config import GrammarRuleConfig
from .engine import SyntacticRuleChecker
from .patterns import (
    detect_sentence_type,
    get_aspiration_confusion,
    get_medial_confusion_correction,
    get_medial_order_correction,
    get_particle_typo_correction,
    is_question_particle,
    is_question_word,
)

# Lazy imports for heavy checker classes (deferred until first use)

__all__ = [
    "AspectChecker",
    "AspectError",
    "AspectInfo",
    "AspectPattern",
    "ClassifierChecker",
    "ClassifierError",
    "CompoundChecker",
    "CompoundError",
    "CompoundInfo",
    "GrammarRuleConfig",
    "NegationChecker",
    "NegationError",
    "NegationInfo",
    "RegisterChecker",
    "RegisterError",
    "RegisterInfo",
    "SyntacticRuleChecker",
    "detect_sentence_type",
    "get_aspiration_confusion",
    "get_medial_confusion_correction",
    "get_medial_order_correction",
    "get_particle_typo_correction",
    "is_question_particle",
    "is_question_word",
]


def __getattr__(name: str) -> Any:
    """Lazy import for heavy checker classes."""
    if name == "AspectChecker":
        from .checkers.aspect import AspectChecker

        return AspectChecker
    if name == "ClassifierChecker":
        from .checkers.classifier import ClassifierChecker

        return ClassifierChecker
    if name == "CompoundChecker":
        from .checkers.compound import CompoundChecker

        return CompoundChecker
    if name == "NegationChecker":
        from .checkers.negation import NegationChecker

        return NegationChecker
    if name == "RegisterChecker":
        from .checkers.register import RegisterChecker

        return RegisterChecker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
