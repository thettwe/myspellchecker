"""
Validation Strategy Pattern.

This module implements the Strategy pattern for context validation,
decomposing validation into focused, composable strategies.

Each strategy:
- Handles one specific validation concern
- Can be tested independently
- Has a priority for execution order
- Can be added/removed without affecting others

Default strategies (wired by SpellCheckerBuilder, in priority order):
1. ToneValidationStrategy (priority 10) - Tone mark disambiguation
2. OrthographyValidationStrategy (priority 15) - Medial order and compatibility
3. SyntacticValidationStrategy (priority 20) - Grammar rules
4. BrokenCompoundStrategy (priority 25) - Wrongly split compound words
5. POSSequenceValidationStrategy (priority 30) - POS patterns
6. QuestionStructureValidationStrategy (priority 40) - Question particles
7. HomophoneValidationStrategy (priority 45) - Homophone detection
8. ConfusableSemanticStrategy (priority 48) - MLM-enhanced confusable detection
9. NgramContextValidationStrategy (priority 50) - Bigram/trigram checks
10. SemanticValidationStrategy (priority 70) - AI-powered semantic checking
"""

from myspellchecker.core.validation_strategies.base import (
    ErrorCandidate,
    ValidationContext,
    ValidationStrategy,
)
from myspellchecker.core.validation_strategies.broken_compound_strategy import (
    BrokenCompoundStrategy,
)
from myspellchecker.core.validation_strategies.confusable_semantic_strategy import (
    ConfusableSemanticStrategy,
)
from myspellchecker.core.validation_strategies.homophone_strategy import HomophoneValidationStrategy
from myspellchecker.core.validation_strategies.ngram_strategy import NgramContextValidationStrategy
from myspellchecker.core.validation_strategies.orthography_strategy import (
    OrthographyValidationStrategy,
)
from myspellchecker.core.validation_strategies.pos_sequence_strategy import (
    POSSequenceValidationStrategy,
)
from myspellchecker.core.validation_strategies.question_strategy import (
    QuestionStructureValidationStrategy,
)
from myspellchecker.core.validation_strategies.semantic_strategy import SemanticValidationStrategy
from myspellchecker.core.validation_strategies.syntactic_strategy import SyntacticValidationStrategy
from myspellchecker.core.validation_strategies.tone_strategy import ToneValidationStrategy

__all__ = [
    "ErrorCandidate",
    "ValidationContext",
    "ValidationStrategy",
    "ToneValidationStrategy",
    "OrthographyValidationStrategy",
    "SyntacticValidationStrategy",
    "BrokenCompoundStrategy",
    "POSSequenceValidationStrategy",
    "QuestionStructureValidationStrategy",
    "HomophoneValidationStrategy",
    "ConfusableSemanticStrategy",
    "NgramContextValidationStrategy",
    "SemanticValidationStrategy",
]
