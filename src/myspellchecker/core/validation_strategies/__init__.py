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
4. StatisticalConfusableStrategy (priority 24) - Bigram-based confusable detection
5. BrokenCompoundStrategy (priority 25) - Wrongly split compound words
6. POSSequenceValidationStrategy (priority 30) - POS patterns
7. QuestionStructureValidationStrategy (priority 40) - Question particles
8. HomophoneValidationStrategy (priority 45) - Homophone detection
9. ConfusableCompoundClassifierStrategy (priority 47) - MLP-based compound detection
10. ConfusableSemanticStrategy (priority 48) - MLM-enhanced confusable detection
11. NgramContextValidationStrategy (priority 50) - Bigram/trigram checks
12. SemanticValidationStrategy (priority 70) - AI-powered semantic checking
"""

from myspellchecker.core.validation_strategies.base import (
    ErrorCandidate,
    ValidationContext,
    ValidationStrategy,
)
from myspellchecker.core.validation_strategies.broken_compound_strategy import (
    BrokenCompoundStrategy,
)
from myspellchecker.core.validation_strategies.compound_merge_probe_strategy import (
    CompoundMergeProbeStrategy,
)
from myspellchecker.core.validation_strategies.confusable_compound_classifier_strategy import (
    ConfusableCompoundClassifierStrategy,
)
from myspellchecker.core.validation_strategies.confusable_semantic_strategy import (
    ConfusableSemanticStrategy,
)
from myspellchecker.core.validation_strategies.cross_whitespace_probe_strategy import (
    CrossWhitespaceProbeStrategy,
)
from myspellchecker.core.validation_strategies.hidden_compound_strategy import (
    HiddenCompoundStrategy,
)
from myspellchecker.core.validation_strategies.homophone_strategy import HomophoneValidationStrategy
from myspellchecker.core.validation_strategies.loan_word_strategy import LoanWordValidationStrategy
from myspellchecker.core.validation_strategies.mlm_span_mask_candgen_strategy import (
    MLMSpanMaskCandGenStrategy,
)
from myspellchecker.core.validation_strategies.ngram_strategy import NgramContextValidationStrategy
from myspellchecker.core.validation_strategies.orthography_strategy import (
    OrthographyValidationStrategy,
)
from myspellchecker.core.validation_strategies.pos_sequence_strategy import (
    POSSequenceValidationStrategy,
)
from myspellchecker.core.validation_strategies.pre_segmenter_raw_probe_strategy import (
    PreSegmenterRawProbeStrategy,
)
from myspellchecker.core.validation_strategies.question_strategy import (
    QuestionStructureValidationStrategy,
)
from myspellchecker.core.validation_strategies.semantic_strategy import SemanticValidationStrategy
from myspellchecker.core.validation_strategies.statistical_confusable_strategy import (
    StatisticalConfusableStrategy,
)
from myspellchecker.core.validation_strategies.syllable_window_oov_strategy import (
    SyllableWindowOOVStrategy,
)
from myspellchecker.core.validation_strategies.syntactic_strategy import SyntacticValidationStrategy
from myspellchecker.core.validation_strategies.tone_safety_net_strategy import (
    ToneSafetyNetStrategy,
)
from myspellchecker.core.validation_strategies.tone_strategy import ToneValidationStrategy
from myspellchecker.core.validation_strategies.visarga_strategy import VisargaStrategy

__all__ = [
    "ErrorCandidate",
    "ValidationContext",
    "ValidationStrategy",
    "ToneValidationStrategy",
    "OrthographyValidationStrategy",
    "SyntacticValidationStrategy",
    "StatisticalConfusableStrategy",
    "SyllableWindowOOVStrategy",
    "ToneSafetyNetStrategy",
    "HiddenCompoundStrategy",
    "PreSegmenterRawProbeStrategy",
    "BrokenCompoundStrategy",
    "CompoundMergeProbeStrategy",
    "CrossWhitespaceProbeStrategy",
    "POSSequenceValidationStrategy",
    "QuestionStructureValidationStrategy",
    "HomophoneValidationStrategy",
    "LoanWordValidationStrategy",
    "VisargaStrategy",
    "ConfusableCompoundClassifierStrategy",
    "ConfusableSemanticStrategy",
    "MLMSpanMaskCandGenStrategy",
    "NgramContextValidationStrategy",
    "SemanticValidationStrategy",
]
