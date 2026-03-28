"""
Integration tests for POS tagger system.

Tests integration with Pipeline (build-time) and SpellChecker (runtime).
"""

import sys
from unittest.mock import MagicMock

import pytest

from myspellchecker.algorithms.pos_tagger_rule import RuleBasedPOSTagger
from myspellchecker.core.config import POSTaggerConfig, SpellCheckerConfig
from myspellchecker.data_pipeline.config import PipelineConfig
from myspellchecker.data_pipeline.pipeline import Pipeline
from myspellchecker.providers.sqlite import SQLiteProvider

# Check transformer availability - must check actual transformers library
# not just the wrapper class which uses lazy imports
try:
    import transformers  # noqa: F401

    from myspellchecker.algorithms.pos_tagger_transformer import TransformerPOSTagger

    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False
    TransformerPOSTagger = None  # type: ignore[misc,assignment]


# Check if transformers is mocked (from other test modules)
def _is_transformers_mocked():
    """Check if transformers module is mocked."""
    transformers_module = sys.modules.get("transformers")
    return isinstance(transformers_module, MagicMock)


@pytest.mark.slow
class TestPipelineIntegration:
    """Test POS tagger integration with data pipeline."""

    def test_pipeline_with_rule_based_tagger(self, tmp_path, mock_console):
        """Test pipeline with rule-based POS tagger."""
        # Create sample corpus
        corpus_file = tmp_path / "corpus.txt"
        corpus_file.write_text("မြန်မာနိုင်ငံ ကောင်းသည်။\n", encoding="utf-8")

        output_db = tmp_path / "test.db"

        # Create pipeline config with POS tagger and mock console
        config = PipelineConfig(
            pos_tagger=POSTaggerConfig(
                tagger_type="rule_based",
            ),
            keep_intermediate=False,
            console=mock_console,
        )

        # Create and run pipeline
        pipeline = Pipeline(config=config, work_dir=str(tmp_path / "work"))

        try:
            result_db = pipeline.build_database(
                input_files=[corpus_file],
                database_path=output_db,
                sample=True,  # Use sample mode for faster tests
            )

            assert result_db.exists()
            assert output_db.exists()

        except Exception as e:
            pytest.fail(f"Pipeline with rule-based tagger failed: {e}")

    @pytest.mark.skipif(not _HAS_TRANSFORMERS, reason="transformers not installed")
    def test_pipeline_with_transformer_tagger(self, tmp_path, mock_console):
        """Test pipeline with transformer POS tagger."""
        # Runtime skip if transformers is mocked (must be checked at runtime, not collection time)
        if _is_transformers_mocked():
            pytest.skip("transformers is mocked by another test module")

        # Create minimal corpus
        corpus_file = tmp_path / "corpus.txt"
        corpus_file.write_text("မြန်မာ\n", encoding="utf-8")

        output_db = tmp_path / "test.db"

        # Create pipeline config with transformer tagger and mock console
        config = PipelineConfig(
            pos_tagger=POSTaggerConfig(
                tagger_type="transformer",
                device=-1,  # CPU
                batch_size=8,
            ),
            keep_intermediate=False,
            console=mock_console,
        )

        pipeline = Pipeline(config=config, work_dir=str(tmp_path / "work"))

        try:
            result_db = pipeline.build_database(
                input_files=[corpus_file],
                database_path=output_db,
                sample=True,  # Use sample mode for faster tests
            )

            assert result_db.exists()

        except Exception as e:
            pytest.fail(f"Pipeline with transformer tagger failed: {e}")

    def test_pipeline_without_pos_tagger(self, tmp_path, mock_console):
        """Test pipeline works without POS tagger configuration."""
        corpus_file = tmp_path / "corpus.txt"
        corpus_file.write_text("မြန်မာ\n", encoding="utf-8")

        output_db = tmp_path / "test.db"

        # Pipeline without POS tagger config but with mock console
        config = PipelineConfig(console=mock_console)
        pipeline = Pipeline(config=config, work_dir=str(tmp_path / "work"))

        try:
            result_db = pipeline.build_database(
                input_files=[corpus_file],
                database_path=output_db,
                sample=True,  # Use sample mode for faster tests
            )

            assert result_db.exists()

        except Exception as e:
            pytest.fail(f"Pipeline without POS tagger failed: {e}")


@pytest.mark.slow
class TestSQLiteProviderIntegration:
    """Test POS tagger integration with SQLiteProvider."""

    @pytest.fixture
    def sample_db(self, tmp_path, mock_console):
        """Create a sample database for testing."""
        from myspellchecker.data_pipeline.config import PipelineConfig
        from myspellchecker.data_pipeline.pipeline import Pipeline

        corpus_file = tmp_path / "corpus.txt"
        corpus_file.write_text("မြန်မာနိုင်ငံ ကောင်းသည်။\nအလုပ်သမား အများကြီး။\n", encoding="utf-8")

        output_db = tmp_path / "test.db"

        config = PipelineConfig(console=mock_console)
        pipeline = Pipeline(config=config, work_dir=str(tmp_path / "work"))
        pipeline.build_database(
            input_files=[corpus_file],
            database_path=output_db,
            sample=False,
        )

        return output_db

    def test_provider_with_rule_based_tagger(self, sample_db):
        """Test SQLiteProvider with rule-based POS tagger."""
        tagger = RuleBasedPOSTagger()
        provider = SQLiteProvider(database_path=str(sample_db), pos_tagger=tagger)

        # Test getting POS for known Myanmar word
        pos = provider.get_word_pos("မြန်မာ")
        # May return POS from DB or from tagger fallback
        # If not in DB, tagger returns 'V' for this word
        assert pos is None or isinstance(pos, str)

        # Test getting POS for OOV non-Myanmar word
        # Tagger returns "UNK" for non-Myanmar text which is filtered out
        oov_pos = provider.get_word_pos("unknownword12345")
        # None is acceptable - tagger returns UNK which is filtered
        assert oov_pos is None or isinstance(oov_pos, str)

    def test_provider_without_tagger_uses_default(self, sample_db):
        """Test that provider uses default tagger when none provided."""
        provider = SQLiteProvider(database_path=str(sample_db))

        # Should use default RuleBasedPOSTagger
        assert provider.pos_tagger is not None
        assert isinstance(provider.pos_tagger, RuleBasedPOSTagger)

        # For non-Myanmar word "test", tagger returns UNK which is filtered
        pos = provider.get_word_pos("test")
        # None is acceptable since "test" is not Myanmar text
        assert pos is None or isinstance(pos, str)

    @pytest.mark.skipif(not _HAS_TRANSFORMERS, reason="transformers not installed")
    def test_provider_with_transformer_tagger(self, sample_db):
        """Test SQLiteProvider with transformer POS tagger."""
        # Runtime skip if transformers is mocked (must be checked at runtime, not collection time)
        if _is_transformers_mocked():
            pytest.skip("transformers is mocked by another test module")

        tagger = TransformerPOSTagger(device=-1)
        provider = SQLiteProvider(database_path=str(sample_db), pos_tagger=tagger)

        # Test getting POS for OOV word using transformer
        oov_pos = provider.get_word_pos("unknownword12345")
        assert oov_pos is not None
        assert isinstance(oov_pos, str)


@pytest.mark.slow
class TestSpellCheckerIntegration:
    """Test POS tagger integration with SpellChecker."""

    @pytest.fixture
    def sample_db(self, tmp_path, mock_console):
        """Create a sample database for spell checker testing."""
        from myspellchecker.data_pipeline.config import PipelineConfig
        from myspellchecker.data_pipeline.pipeline import Pipeline

        corpus_file = tmp_path / "corpus.txt"
        corpus_file.write_text("မြန်မာနိုင်ငံ ကောင်းလှပသည်။\n" * 10, encoding="utf-8")

        output_db = tmp_path / "test.db"

        config = PipelineConfig(console=mock_console)
        pipeline = Pipeline(config=config, work_dir=str(tmp_path / "work"))
        pipeline.build_database(
            input_files=[corpus_file],
            database_path=output_db,
            sample=False,
        )

        return output_db

    def test_spellchecker_with_rule_based_tagger(self, sample_db):
        """Test SpellChecker with rule-based POS tagger config."""
        from myspellchecker import SpellChecker
        from myspellchecker.providers.sqlite import SQLiteProvider

        # Create provider with POS tagger
        tagger = RuleBasedPOSTagger()
        provider = SQLiteProvider(database_path=str(sample_db), pos_tagger=tagger)

        config = SpellCheckerConfig(
            provider=provider,
            pos_tagger=POSTaggerConfig(
                tagger_type="rule_based",
            ),
        )

        checker = SpellChecker(config=config)

        # Verify tagger is configured
        assert checker.provider.pos_tagger is not None
        assert isinstance(checker.provider.pos_tagger, RuleBasedPOSTagger)

        # Test spell checking works
        result = checker.check("မြန်မာ")
        assert result is not None

    def test_spellchecker_without_pos_config_uses_default(self, sample_db):
        """Test that SpellChecker uses default tagger when not configured."""
        from myspellchecker import SpellChecker
        from myspellchecker.providers.sqlite import SQLiteProvider

        # Create provider without explicit POS tagger
        provider = SQLiteProvider(database_path=str(sample_db))

        config = SpellCheckerConfig(provider=provider)

        checker = SpellChecker(config=config)

        # Should have default tagger
        assert checker.provider.pos_tagger is not None

    @pytest.mark.skipif(not _HAS_TRANSFORMERS, reason="transformers not installed")
    def test_spellchecker_with_transformer_tagger(self, sample_db):
        """Test SpellChecker with transformer POS tagger config."""
        # Runtime skip if transformers is mocked
        if _is_transformers_mocked():
            pytest.skip("transformers is mocked by another test module")

        from myspellchecker import SpellChecker
        from myspellchecker.providers.sqlite import SQLiteProvider

        # Create provider with transformer POS tagger
        tagger = TransformerPOSTagger(device=-1)
        provider = SQLiteProvider(database_path=str(sample_db), pos_tagger=tagger)

        config = SpellCheckerConfig(
            provider=provider,
            pos_tagger=POSTaggerConfig(
                tagger_type="transformer",
                device=-1,
            ),
        )

        checker = SpellChecker(config=config)

        # Verify transformer tagger is configured
        assert checker.provider.pos_tagger is not None
        assert isinstance(checker.provider.pos_tagger, TransformerPOSTagger)

        # Test spell checking works
        result = checker.check("မြန်မာ")
        assert result is not None


@pytest.mark.slow
class TestEndToEndIntegration:
    """End-to-end integration tests."""

    def test_build_and_check_with_rule_based(self, tmp_path, mock_console):
        """Test complete flow: build with tagger → spell check."""
        from myspellchecker import SpellChecker
        from myspellchecker.data_pipeline.pipeline import Pipeline

        # Step 1: Build database with POS tagger
        corpus_file = tmp_path / "corpus.txt"
        corpus_file.write_text("မြန်မာနိုင်ငံ ကောင်းလှပသည်။\n" * 5, encoding="utf-8")

        output_db = tmp_path / "test.db"

        config = PipelineConfig(
            pos_tagger=POSTaggerConfig(tagger_type="rule_based"),
            keep_intermediate=False,
            console=mock_console,
        )

        pipeline = Pipeline(config=config, work_dir=str(tmp_path / "work"))
        pipeline.build_database(
            input_files=[corpus_file],
            database_path=output_db,
            sample=False,
        )

        # Step 2: Use database with spell checker
        from myspellchecker.providers.sqlite import SQLiteProvider

        # Create provider with database
        tagger = RuleBasedPOSTagger()
        provider = SQLiteProvider(database_path=str(output_db), pos_tagger=tagger)

        checker_config = SpellCheckerConfig(
            provider=provider,
            pos_tagger=POSTaggerConfig(tagger_type="rule_based"),
        )

        checker = SpellChecker(config=checker_config)

        # Step 3: Verify spell checking works
        result = checker.check("မြန်မာ ကောင်း")
        assert result is not None
        assert hasattr(result, "has_errors")

    def test_pos_tagger_fallback_chain(self, tmp_path, mock_console):
        """Test complete POS tagging fallback chain."""
        from myspellchecker import SpellChecker
        from myspellchecker.data_pipeline.pipeline import Pipeline

        # Build minimal database
        corpus_file = tmp_path / "corpus.txt"
        corpus_file.write_text("မြန်မာ\n", encoding="utf-8")

        output_db = tmp_path / "test.db"

        config = PipelineConfig(console=mock_console)
        pipeline = Pipeline(config=config, work_dir=str(tmp_path / "work"))
        pipeline.build_database(
            input_files=[corpus_file],
            database_path=output_db,
            sample=False,
        )

        # Create checker with tagger
        from myspellchecker.providers.sqlite import SQLiteProvider

        tagger = RuleBasedPOSTagger()
        provider = SQLiteProvider(database_path=str(output_db), pos_tagger=tagger)

        config = SpellCheckerConfig(
            provider=provider,
            pos_tagger=POSTaggerConfig(tagger_type="rule_based"),
        )

        checker = SpellChecker(config=config)
        provider = checker.provider

        # Test fallback chain for OOV word:
        # 1. Database (not found)
        # 2. Stemming (not found)
        # 3. POS Tagger (returns UNK for non-Myanmar text, filtered out)
        # 4. MorphologyAnalyzer (returns empty set)

        oov_pos = provider.get_word_pos("CompletelyUnknownWord123")
        # None is acceptable - non-Myanmar text returns UNK which is filtered
        assert oov_pos is None or isinstance(oov_pos, str)
