# Skipped / xfailed Test Registry
# =================================
# This registry documents every skipped and xfailed test, why it is skipped,
# and what would fix it. Organized by skip category.
#
# ── Category 1: Cython extension not compiled ──
# These tests require `python setup.py build_ext --inplace` to compile .pyx files.
# Remediation: Run `python setup.py build_ext --inplace` (macOS needs `brew install libomp`).
#
# test_ingester_c.py (8 tests) - Cython ingester_c not compiled
# test_mmap_reader.py (5 tests) - Cython mmap_reader not compiled
# test_repair_c.py (8 tests) - Cython repair_c not compiled
# test_tsv_reader_c.py (5 tests) - Cython tsv_reader_c not compiled
# test_syllable_rules_c.py (2 tests) - Cython syllable_rules_c not built
# test_syllable_rules_property.py::TestParityOnRandomInputs (1 test) - Cython not built
# test_medial_ya_ai_vowel.py::test_medial_ya_with_ai_vowel_cython - Cython extension not available
# test_python_cython_equivalence.py::cy_validator fixture (2 tests) - Cython not available
# test_python_cython_equivalence.py::cy_lenient fixture (2 tests) - Cython not available
# test_repair.py::TestCythonRepair (4 tests) - Cython repair module not available
# test_myanmar_constants.py (5 tests: batch_processor vowels/consonants/numerals/fragment,
#     repair_c consonants) - Cython batch_processor or repair_c not compiled
# test_viterbi_smoothing.py (3 tests) - Cython Viterbi module not available
# test_extended_myanmar_validation.py::test_cython_allow_extended_parameter - Cython rebuild needed
#     (run 'python setup.py build_ext --inplace')
#
# ── Category 2: Optional dependency not installed ──
# Remediation: Install the missing package.
#
# test_pos_integration.py (3 tests) - transformers not installed
#     Remediation: pip install 'myspellchecker[transformers]'
# test_grammar_rules_schema_validation.py::TestSchemaValidation - jsonschema not installed
#     Remediation: pip install jsonschema
# test_training_components.py (5 skipif + 5 runtime skips) - torch not installed
#     Remediation: pip install 'myspellchecker[train]'
# test_stress/test_batch_processing.py - pytest-benchmark not installed
#     Remediation: pip install pytest-benchmark
# test_default_segmenter.py::test_segment_benchmark - pytest-benchmark not installed
#     Remediation: pip install pytest-benchmark
# test_config_loader.py (2 tests) - PyYAML not available
#     Remediation: pip install pyyaml
# test_yaml_config_edge_cases.py (skip_if_no_yaml fixture) - PyYAML not available
#     Remediation: pip install pyyaml
# integration/test_config_error_handling.py::test_yaml_loading_without_pyyaml
#     - PyYAML IS installed (cannot test missing-import path)
#     Remediation: Only testable when PyYAML is absent; consider mocking sys.modules
#
# ── Category 3: transformers mocked by another test module ──
# Remediation: Run these tests in isolation or fix test ordering.
#
# test_pos_integration.py (3 runtime skips) - transformers is mocked by another test module
#     These tests check `_is_transformers_mocked()` at runtime because another test module
#     patches the transformers import at collection time.
#
# ── Category 4: SpellChecker not available (integration tests) ──
# These tests try `SpellChecker.create_default()` which needs a built DB or provider.
# Remediation: Build a test DB with `myspellchecker build --sample` or ensure
# the session-scoped `patch_default_db` fixture is active.
#
# integration/test_joint_segment_tagger_integration.py::spellchecker fixture
# integration/test_morphology_integration.py::spellchecker fixture
# integration/test_particle_typo_integration.py::spellchecker fixture
# test_pos_sequence_validation.py::spellchecker fixture
# test_question_detection.py::spellchecker fixture
#
# ── Category 5: Production DB not available ──
# Remediation: Place the production DB at data/mySpellChecker_production.db.
#
# test_syllable_rules_property.py::TestProductionDBOracle (2 tests)
#     - test_all_db_syllables_pass_lenient
#     - test_high_freq_syllables_pass_strict
#
# ── Category 6: xfail — known limitations ──
#
# test_pali_spellchecking.py (3 xfails):
#     - test_pali_orthography_correction - Requires Pali dictionary (DB lacks Pali)
#     - test_pali_voicing_disambiguation - Requires Pali dictionary
#     - test_pali_complex_words - Requires Pali dictionary
#     Remediation: Add Pali loanword entries to the production dictionary.
#
# ── Category 7: Data-dependent / conditional skips ──
#
# test_ngram_strategy.py::test_ngram_strategy_skipped_words
#     - Skips when SKIPPED_CONTEXT_WORDS is empty (currently empty by design)
#     Remediation: Will auto-enable when SKIPPED_CONTEXT_WORDS is populated.
#
# ── Category 8: OS-dependent ──
#
# test_path_validation_adversarial.py::test_symlink_loop_rejected
#     - Skips when OS prevents symlink loops (some OS configurations block this)
#     Remediation: None needed; this is an environment-specific guard.

import logging
import os
import shutil
import sqlite3
import tempfile
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(scope="session", autouse=True)
def mock_resource_downloads_session():
    """Session-scoped: prevent HuggingFace downloads, provide working WordTokenizer."""
    tmp_dir = Path(tempfile.mkdtemp())
    dummy_mmap = tmp_dir / "segmentation.mmap"
    dummy_mmap.write_bytes(b"\x00" * 100)
    dummy_crf = tmp_dir / "wordseg_c2_crf.crfsuite"
    dummy_crf.write_bytes(b"\x00" * 100)

    def mock_get_resource(name, cache_dir=None, force_download=False):
        if name == "segmentation":
            return dummy_mmap
        elif name == "crf":
            return dummy_crf
        raise ValueError(f"Unknown resource: {name}")

    def mock_init_myword(self):
        from myspellchecker.segmenters.regex import RegexSegmenter

        self._using_mmap = False
        self._using_cython = False
        _regex = RegexSegmenter()

        def simple_viterbi(text):
            return (0.0, _regex.segment_syllables(text) if text.strip() else [])

        self._viterbi_func = simple_viterbi

    def mock_init_crf(self):
        mock_tagger = MagicMock()
        mock_tagger.tag.return_value = ["B"] * 10
        self.tagger = mock_tagger

    from myspellchecker.tokenizers.word import WordTokenizer

    with (
        patch("myspellchecker.tokenizers.resource_loader.get_resource_path", mock_get_resource),
        patch.object(WordTokenizer, "_init_myword", mock_init_myword),
        patch.object(WordTokenizer, "_init_crf", mock_init_crf),
    ):
        yield

    shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.fixture
def mock_console():
    """Create a mock PipelineConsole to avoid Rich rendering issues in tests."""
    mock = MagicMock()
    mock.console = MagicMock()
    return mock


@pytest.fixture(autouse=True)
def reset_grammar_config_singleton():
    """Clear the GrammarRuleConfig singleton between tests.

    The singleton caches the config to avoid redundant YAML loading in production,
    but tests that mock GrammarRuleConfig can pollute the singleton state.
    """
    yield
    from myspellchecker.grammar.config import _grammar_config_singleton

    _grammar_config_singleton.clear()


@pytest.fixture(autouse=True)
def reset_rich_console():
    """Patch Rich console output to avoid rendering issues in tests.

    Rich console can cause state pollution between tests when it caches
    terminal capabilities and output buffers. This fixture patches the
    Rich Console to write to a StringIO instead of stdout.
    """
    # Create a null console that writes to StringIO
    null_output = StringIO()

    # Patch console creation to use null output
    with patch("myspellchecker.utils.console.sys.stdout", null_output):
        yield


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset all singletons between tests to prevent state pollution.

    Singletons can retain state from previous tests, causing test isolation
    issues. This fixture clears all singleton instances after each test.
    """
    yield

    # Clear all singletons after each test
    try:
        from myspellchecker.utils.singleton import clear_all_singletons

        clear_all_singletons()
    except ImportError:
        pass

    # Also clear individual module-level singletons
    try:
        from myspellchecker.grammar.checkers import compound, negation

        if hasattr(negation, "_singleton"):
            negation._singleton.clear()
        if hasattr(compound, "_singleton"):
            compound._singleton.clear()
    except ImportError:
        pass


@pytest.fixture(autouse=True)
def clear_lru_caches():
    """Clear lru_cache decorated functions to prevent state pollution.

    LRU caches can cause tests to see stale data from previous tests.
    This fixture clears known caches after each test.
    """
    yield

    # Clear known lru_cache decorated functions
    try:
        from myspellchecker.segmenters.default import _load_bundled_segmenter

        if hasattr(_load_bundled_segmenter, "cache_clear"):
            _load_bundled_segmenter.cache_clear()
    except (ImportError, AttributeError):
        pass

    try:
        from myspellchecker.utils.logging_utils import get_logger

        if hasattr(get_logger, "cache_clear"):
            get_logger.cache_clear()
    except (ImportError, AttributeError):
        pass


@pytest.fixture(autouse=True)
def reset_logging():
    """Reset logging configuration before each test to prevent pollution.

    This ensures that tests using caplog can properly capture logs even when
    the myspellchecker logger has propagate=False set by configure_logging().
    """
    # Store original handlers and propagation settings for myspellchecker loggers
    logger = logging.getLogger("myspellchecker")
    original_handlers = logger.handlers[:]
    original_propagate = logger.propagate
    original_level = logger.level

    yield

    # Restore original settings
    logger.handlers = original_handlers
    logger.propagate = original_propagate
    logger.level = original_level


@pytest.fixture(scope="session")
def test_database_path():
    """
    Create a temporary SQLite database for testing.
    Returns the path to the temporary database file.
    """
    # Create a temporary file
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    database_path = Path(path)

    # Initialize database with schema and sample data
    conn = sqlite3.connect(str(database_path))
    cursor = conn.cursor()

    # Create tables
    cursor.execute(
        """
    CREATE TABLE syllables (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        syllable TEXT UNIQUE NOT NULL,
        frequency INTEGER DEFAULT 0
    );
    """
    )
    cursor.execute("CREATE INDEX idx_syllables_text ON syllables (syllable);")

    cursor.execute(
        """
    CREATE TABLE words (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        word TEXT UNIQUE NOT NULL,
        syllable_count INTEGER,
        frequency INTEGER DEFAULT 0
    );
    """
    )
    cursor.execute("CREATE INDEX idx_words_text ON words (word);")

    cursor.execute(
        """
    CREATE TABLE bigrams (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        word1_id INTEGER,
        word2_id INTEGER,
        probability REAL DEFAULT 0.0,
        FOREIGN KEY(word1_id) REFERENCES words(id),
        FOREIGN KEY(word2_id) REFERENCES words(id),
        UNIQUE(word1_id, word2_id)
    );
    """
    )
    cursor.execute("CREATE INDEX idx_bigrams_w1_w2 ON bigrams (word1_id, word2_id);")

    # Insert sample syllables
    syllables = [
        ("မြန်", 1000),
        ("မာ", 1000),
        ("သည်", 5000),
        ("၏", 2000),
        ("။", 5000),
        ("ကျောင်း", 500),
        ("သား", 500),
        ("သွား", 800),
        ("စား", 800),
    ]
    cursor.executemany("INSERT INTO syllables (syllable, frequency) VALUES (?, ?)", syllables)

    # Insert sample words
    words = [
        ("မြန်မာ", 2, 1000),
        ("သည်", 1, 5000),
        ("၏", 1, 2000),
        ("။", 1, 5000),
        ("ကျောင်း", 1, 500),
        ("သား", 1, 500),
        ("ကျောင်းသား", 2, 300),
        ("သွား", 1, 800),
        ("စား", 1, 800),
        ("သူ", 1, 1500),
        ("သင်္ဘော", 2, 100),
        ("မေတ္တာ", 2, 150),
        ("ပုဂ္ဂိုလ်", 2, 120),
        ("ကတိ", 2, 200),
        ("ဂတိ", 2, 180),
    ]
    cursor.executemany(
        "INSERT INTO words (word, syllable_count, frequency) VALUES (?, ?, ?)", words
    )

    # Get word IDs for bigrams
    word_ids = {}
    cursor.execute("SELECT id, word FROM words")
    for row in cursor.fetchall():
        word_ids[row[1]] = row[0]

    # Insert sample bigrams
    bigrams = [
        ("သည်", "။", 0.8),
        ("မြန်", "မာ", 0.9),  # Note: usually handled as word, but good for testing
        ("သူ", "သွား", 0.3),
        ("သူ", "ကျောင်း", 0.005),  # Low probability example
    ]

    for w1, w2, prob in bigrams:
        if w1 in word_ids and w2 in word_ids:
            cursor.execute(
                "INSERT INTO bigrams (word1_id, word2_id, probability) VALUES (?, ?, ?)",
                (word_ids[w1], word_ids[w2], prob),
            )

    conn.commit()
    conn.close()

    yield database_path

    # Cleanup
    if database_path.exists():
        os.unlink(database_path)


@pytest.fixture(scope="session", autouse=True)
def patch_default_db(test_database_path):
    """
    Globally patch SQLiteProvider to use the test database when no path is given.
    Session-scoped to avoid 7,000+ redundant monkeypatch calls.

    Since the bundled default DB was removed, SQLiteProvider now raises
    MissingDatabaseError when database_path is None. This patch intercepts
    __init__ to inject the test database path in that case.
    """
    from unittest.mock import patch

    from myspellchecker.providers.sqlite import SQLiteProvider

    _original_init = SQLiteProvider.__init__

    def _patched_init(self, database_path=None, **kwargs):
        if database_path is None:
            database_path = str(test_database_path)
        _original_init(self, database_path=database_path, **kwargs)

    with patch.object(SQLiteProvider, "__init__", _patched_init):
        yield
