import asyncio
from unittest.mock import MagicMock, patch

import pytest

from myspellchecker import SpellChecker
from myspellchecker.core.config import SpellCheckerConfig
from myspellchecker.core.exceptions import ProcessingError
from myspellchecker.core.response import ContextError, Error, Response
from myspellchecker.segmenters import Segmenter


class TestSpellCheckerFullCoverage:
    @pytest.fixture
    def mock_provider(self):
        return MagicMock()

    @pytest.fixture
    def mock_segmenter(self):
        seg = MagicMock(spec=Segmenter)
        seg.segment_syllables.return_value = ["test"]
        return seg

    def test_init_all_params(self):
        """Test SpellChecker initialization with validator injection."""
        # Test validator injection - validators can be mocked
        syllable_validator = MagicMock()
        word_validator = MagicMock()
        context_validator = MagicMock()

        # Create checker with only the validators we want to test
        # Provider and segmenter will use defaults
        checker = SpellChecker(
            syllable_validator=syllable_validator,
            word_validator=word_validator,
            context_validator=context_validator,
        )

        assert checker.syllable_validator == syllable_validator
        assert checker.word_validator == word_validator
        assert checker.context_validator == context_validator

        # Test Properties - symspell and syllable_rule_validator are accessed through validators
        assert checker.symspell == syllable_validator.symspell
        assert checker.syllable_rule_validator == syllable_validator.syllable_rule_validator

        # Note: context_checker, name_heuristic, semantic_checker, phonetic_hasher
        # are now stored directly on SpellChecker (not accessed through context_validator)
        # These are set via ComponentFactory or direct initialization
        # When using mocked validators, these will have their default values from ComponentFactory
        assert hasattr(checker, "context_checker")
        assert hasattr(checker, "name_heuristic")
        assert hasattr(checker, "semantic_checker")
        assert hasattr(checker, "phonetic_hasher")

    def test_init_properties_none(self):
        """Test SpellChecker property fallbacks when validators are None."""
        # Initialize with None validators to test property fallbacks
        checker = SpellChecker(syllable_validator=None, word_validator=None, context_validator=None)
        # We need to manually set validators to None because init creates default ones
        checker.syllable_validator = None
        checker.word_validator = None
        checker.context_validator = None
        # Also clear the directly stored components
        checker._context_checker = None
        checker._name_heuristic = None
        checker._semantic_checker = None
        checker._phonetic_hasher = None

        assert checker.symspell is None
        assert checker.context_checker is None
        assert checker.syllable_rule_validator is None
        assert checker.name_heuristic is None
        assert checker.semantic_checker is None
        assert checker.phonetic_hasher is None

    def test_init_symspell_property_fallback(self):
        # If syllable_validator is None but word_validator is not
        word_validator = MagicMock()
        checker = SpellChecker(word_validator=word_validator)
        checker.syllable_validator = None

        assert checker.symspell == word_validator.symspell

    def test_check_async(self):
        checker = SpellChecker()
        result = asyncio.run(checker.check_async("test"))
        assert isinstance(result, Response)
        assert result.text == "test"

    def test_check_batch_input_validation(self):
        checker = SpellChecker()

        with pytest.raises(TypeError):
            checker.check_batch("not a list")

        with pytest.raises(ProcessingError):
            checker.check_batch([])  # Empty list

        with pytest.raises(ProcessingError):
            checker.check_batch(["valid", 123])  # Invalid item type

    def test_check_input_validation(self):
        checker = SpellChecker()
        with pytest.raises(TypeError):
            checker.check(123)

        # Test check normalization empty return
        with patch("myspellchecker.core.spellchecker.normalize") as mock_norm:
            mock_norm.return_value = ""
            result = checker.check("   ")
            assert result.has_errors is False
            assert result.corrected_text == "   "

    def test_generate_corrected_text_sorting(self):
        """Test generate_corrected_text sorts errors by position."""
        from myspellchecker.core.correction_utils import generate_corrected_text

        # Create errors in unsorted order
        error1 = Error(text="a", position=0, suggestions=["A"], error_type="test")
        error2 = Error(text="b", position=2, suggestions=["B"], error_type="test")

        # Text: "a b"
        # Corrected: "A B"

        corrected = generate_corrected_text("a b", [error2, error1])
        assert corrected == "A B"

    def test_generate_corrected_text_context_error(self):
        """Test generate_corrected_text handles ContextError with suggestions."""
        from myspellchecker.core.correction_utils import generate_corrected_text

        # ContextErrors with suggestions are corrected like other error types
        error = ContextError(text="bad", position=0, suggestions=["good"])
        corrected = generate_corrected_text("bad", [error])
        assert corrected == "good"  # ContextError with suggestions is applied

    def test_check_default_provider_fallback(self):
        # Test fallback to MemoryProvider if SQLite fails
        # Requires fallback_to_empty_provider=True (default is False)
        # _get_default_provider catches MissingDatabaseError, not FileNotFoundError
        from myspellchecker.core.exceptions import MissingDatabaseError

        with patch("myspellchecker.core.spellchecker.SQLiteProvider") as mock_sqlite:
            mock_sqlite.side_effect = MissingDatabaseError(message="test db not found")
            config = SpellCheckerConfig(fallback_to_empty_provider=True)
            with pytest.warns(RuntimeWarning):
                checker = SpellChecker(config=config)
                from myspellchecker.providers import MemoryProvider

                assert isinstance(checker.provider, MemoryProvider)

    def test_check_default_provider_raises_error_without_fallback(self):
        # Test that MissingDatabaseError is raised when fallback is disabled (default)
        from myspellchecker.core.exceptions import MissingDatabaseError

        with patch("myspellchecker.core.spellchecker.SQLiteProvider") as mock_sqlite:
            mock_sqlite.side_effect = MissingDatabaseError(message="db not found")
            config = SpellCheckerConfig(fallback_to_empty_provider=False)
            with pytest.raises(MissingDatabaseError) as exc_info:
                SpellChecker(config=config)
            assert "no database available" in str(exc_info.value).lower()

    def test_init_semantic_checker_from_config(self):
        from myspellchecker.core.config import SemanticConfig

        config = SpellCheckerConfig(
            semantic=SemanticConfig(model_path="dummy_path", tokenizer_path="dummy_path")
        )

        # SemanticChecker is lazily imported in component_factory.create_semantic_checker
        # Patch at the source module
        with patch("myspellchecker.algorithms.semantic_checker.SemanticChecker") as MockSem:
            checker = SpellChecker(config=config)
            MockSem.assert_called_once()
            assert checker.semantic_checker is not None

    def test_init_semantic_checker_failure(self):
        from myspellchecker.core.config import SemanticConfig

        config = SpellCheckerConfig(
            semantic=SemanticConfig(model_path="dummy_path", tokenizer_path="dummy_path")
        )

        with patch("myspellchecker.core.spellchecker.SemanticChecker") as MockSem:
            MockSem.side_effect = Exception("Fail")
            checker = SpellChecker(config=config)
            # Should not raise, just log error
            assert checker._semantic_checker is None

    def test_apply_semantic_reranking_injects_generated_neighbor_for_sparse_error(self):
        checker = SpellChecker.__new__(SpellChecker)
        checker.logger = MagicMock()

        semantic = MagicMock()
        semantic.score_candidates.return_value = [("ကျောင်း", 5.0)]
        checker._semantic_checker = semantic

        symspell = MagicMock()
        symspell.lookup.return_value = [MagicMock(term="ကျောင်း")]
        checker.syllable_validator = MagicMock()
        checker.syllable_validator.symspell = symspell
        checker.word_validator = None

        error = Error(
            text="ကွောင်း",
            position=0,
            suggestions=[],
            error_type="medial_confusion",
        )

        checker._apply_semantic_reranking("ကလေးကို ကွောင်းပို့ပေးမယ်", [error])

        assert error.suggestions
        assert error.suggestions[0] == "ကျောင်း"
        semantic.score_candidates.assert_called_once()
        _, kwargs = semantic.score_candidates.call_args
        assert "ကျောင်း" in kwargs["candidates"]

    def test_apply_semantic_reranking_skips_generation_for_non_target_error_type(self):
        checker = SpellChecker.__new__(SpellChecker)
        checker.logger = MagicMock()

        semantic = MagicMock()
        checker._semantic_checker = semantic

        symspell = MagicMock()
        symspell.lookup.return_value = [MagicMock(term="တယ်")]
        checker.syllable_validator = MagicMock()
        checker.syllable_validator.symspell = symspell
        checker.word_validator = None

        error = Error(
            text="မှတ်ချက်",
            position=0,
            suggestions=[],
            error_type="syntax_error",
        )

        checker._apply_semantic_reranking("မှတ်ချက် - စာကြောင်း", [error])

        semantic.score_candidates.assert_not_called()
        assert error.suggestions == []
