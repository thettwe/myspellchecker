from unittest.mock import MagicMock, patch

from myspellchecker.core.config import SemanticConfig, SpellCheckerConfig
from myspellchecker.core.spellchecker import SpellChecker
from myspellchecker.providers import MemoryProvider


def _create_mock_provider():
    """Create a mock provider with proper return types for POS methods."""
    provider = MagicMock()
    # Ensure POS probability methods return empty dicts, not MagicMock
    provider.get_pos_unigram_probabilities.return_value = {}
    provider.get_pos_bigram_probabilities.return_value = {}
    provider.get_pos_trigram_probabilities.return_value = {}
    # Ensure dictionary-related methods return proper types
    provider.get_all_syllables.return_value = []
    provider.get_all_words.return_value = []
    return provider


class TestSpellCheckerFactoryCoverage:
    def test_create_default(self):
        mock_provider = _create_mock_provider()
        with patch(
            "myspellchecker.core.spellchecker.SQLiteProvider", return_value=mock_provider
        ) as MockSQLite:
            checker = SpellChecker()
            assert isinstance(checker, SpellChecker)
            MockSQLite.assert_called()

    def test_create_with_config(self):
        # Use a real MemoryProvider since config validates provider type
        provider = MemoryProvider()
        config = SpellCheckerConfig(provider=provider, use_phonetic=False)
        checker = SpellChecker(config=config)
        assert checker.provider == provider
        assert checker.phonetic_hasher is None

    def test_create_with_explicit_provider(self):
        """Test that explicit provider in config is used."""
        provider = MemoryProvider()
        config = SpellCheckerConfig(provider=provider)
        checker = SpellChecker(config=config)
        assert checker.provider is provider

    def test_create_semantic_checker_success(self):
        config = SpellCheckerConfig(
            semantic=SemanticConfig(model_path="dummy", tokenizer_path="dummy")
        )
        mock_provider = _create_mock_provider()
        with patch("myspellchecker.algorithms.semantic_checker.SemanticChecker") as MockSem:
            with patch(
                "myspellchecker.core.spellchecker.SQLiteProvider", return_value=mock_provider
            ):
                checker = SpellChecker(config=config)
                MockSem.assert_called()
                # semantic_checker is now stored directly on SpellChecker
                assert checker.semantic_checker is not None

    def test_create_semantic_checker_failure(self):
        config = SpellCheckerConfig(
            semantic=SemanticConfig(model_path="dummy", tokenizer_path="dummy")
        )
        mock_provider = _create_mock_provider()
        # Use RuntimeError which is one of the caught exception types
        with patch(
            "myspellchecker.algorithms.semantic_checker.SemanticChecker",
            side_effect=RuntimeError("Fail"),
        ):
            with patch(
                "myspellchecker.core.spellchecker.SQLiteProvider", return_value=mock_provider
            ):
                # Should log error but not raise
                checker = SpellChecker(config=config)
                # semantic_checker is now stored directly on SpellChecker
                assert checker.semantic_checker is None
