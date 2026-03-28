"""
Tests for dependency injection and factory-based initialization of SpellChecker.
"""

from unittest.mock import Mock

from myspellchecker.algorithms import NgramContextChecker, SymSpell
from myspellchecker.core.config import SpellCheckerConfig
from myspellchecker.core.spellchecker import SpellChecker
from myspellchecker.core.syllable_rules import SyllableRuleValidator
from myspellchecker.providers import DictionaryProvider, MemoryProvider
from myspellchecker.segmenters import DefaultSegmenter
from myspellchecker.utils import PhoneticHasher


def _create_mock_provider():
    """Create a mock provider with proper return types for all methods."""
    provider = Mock(spec=MemoryProvider)
    # Ensure POS probability methods return empty dicts, not Mock
    provider.get_pos_unigram_probabilities.return_value = {}
    provider.get_pos_bigram_probabilities.return_value = {}
    provider.get_pos_trigram_probabilities.return_value = {}
    # Ensure dictionary-related methods return proper types
    provider.get_all_syllables.return_value = []
    provider.get_all_words.return_value = []
    return provider


class TestSpellCheckerInitialization:
    """Test initialization modes of SpellChecker."""

    def test_default_initialization(self):
        """Test legacy default initialization (uses Factory implicitly)."""
        checker = SpellChecker()
        assert isinstance(checker.segmenter, DefaultSegmenter)
        # Provider might be SQLite or Memory depending on env
        assert isinstance(checker.provider, DictionaryProvider)
        assert isinstance(checker.symspell, SymSpell)
        assert isinstance(checker.context_checker, NgramContextChecker)
        assert isinstance(checker.syllable_rule_validator, SyllableRuleValidator)
        # Default phonetic is True
        assert isinstance(checker.phonetic_hasher, PhoneticHasher)

    def test_factory_initialization(self):
        """Test initialization via direct SpellChecker instantiation."""
        config = SpellCheckerConfig(use_context_checker=False, use_ner=False)
        checker = SpellChecker(config=config)

        assert isinstance(checker.segmenter, DefaultSegmenter)
        assert checker.context_checker is None
        assert checker.name_heuristic is None
        assert isinstance(checker.syllable_rule_validator, SyllableRuleValidator)

    def test_dependency_injection(self):
        """Test dependency injection via config and constructor."""
        mock_segmenter = Mock(spec=DefaultSegmenter)
        mock_provider = _create_mock_provider()

        # The new API injects segmenter and provider; symspell/context_checker
        # are now built internally by ComponentFactory
        checker = SpellChecker(
            segmenter=mock_segmenter,
            provider=mock_provider,
        )

        assert checker.segmenter is mock_segmenter
        assert checker.provider is mock_provider
        # symspell and context_checker are now built internally, not injected directly
        assert isinstance(checker.symspell, SymSpell)
        assert isinstance(checker.context_checker, NgramContextChecker)

        # These were not provided but should be auto-created based on default config
        assert isinstance(checker.syllable_rule_validator, SyllableRuleValidator)

    def test_partial_injection_fallback(self):
        """
        Test partial injection.

        If critical dependencies (symspell) are missing, the legacy init logic
        should kick in to build them, respecting the config.
        """
        mock_provider = _create_mock_provider()

        # Only provider injected. SymSpell must be built internally.
        checker = SpellChecker(provider=mock_provider)

        assert checker.provider is mock_provider
        assert isinstance(checker.symspell, SymSpell)
        assert checker.symspell.provider is mock_provider  # Should use the injected provider

    def test_factory_with_custom_config(self):
        """Test SpellChecker builds components respecting config."""
        from myspellchecker.core.config import SymSpellConfig

        config = SpellCheckerConfig(max_edit_distance=1, symspell=SymSpellConfig(prefix_length=5))

        checker = SpellChecker(config=config)

        assert checker.symspell.max_edit_distance == 1
        assert checker.symspell.prefix_length == 5
