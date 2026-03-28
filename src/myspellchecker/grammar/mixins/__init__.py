"""Grammar engine mixins for SyntacticRuleChecker decomposition.

Each mixin provides a group of related methods extracted from
``engine.py`` to reduce file size while preserving the exact same
method signatures and behaviour.
"""

from myspellchecker.grammar.mixins.checker_delegation_mixin import CheckerDelegationMixin
from myspellchecker.grammar.mixins.config_grammar_mixin import ConfigGrammarMixin
from myspellchecker.grammar.mixins.pos_tag_mixin import POSTagMixin
from myspellchecker.grammar.mixins.sentence_structure_mixin import SentenceStructureMixin
from myspellchecker.grammar.mixins.word_rule_mixin import WordRuleMixin

__all__ = [
    "CheckerDelegationMixin",
    "ConfigGrammarMixin",
    "POSTagMixin",
    "SentenceStructureMixin",
    "WordRuleMixin",
]
