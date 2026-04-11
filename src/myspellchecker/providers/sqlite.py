"""
SQLite-based dictionary provider implementation.

This module provides a production-ready SQLite backend for dictionary data,
optimized for fast lookups with proper indexing and caching.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Generator

    from myspellchecker.algorithms.pos_tagger_base import POSTaggerBase
    from myspellchecker.utils.cache import CacheManager

from myspellchecker.algorithms.pos_tagger_rule import RuleBasedPOSTagger
from myspellchecker.core.config.validation_configs import ProviderConfig
from myspellchecker.core.constants import (
    STATS_KEY_BIGRAM_COUNT,
    STATS_KEY_FIVEGRAM_COUNT,
    STATS_KEY_FOURGRAM_COUNT,
    STATS_KEY_SYLLABLE_COUNT,
    STATS_KEY_TRIGRAM_COUNT,
    STATS_KEY_WORD_COUNT,
)
from myspellchecker.core.exceptions import MissingDatabaseError
from myspellchecker.providers._sqlite_cache import CacheMixin, create_caches
from myspellchecker.providers._sqlite_enrichment import EnrichmentMixin
from myspellchecker.providers._sqlite_ner import NEREntityMixin

# ---------------------------------------------------------------------------
# Re-export everything from helper modules for backward compatibility.
# Tests and _sqlite_bulk_ops.py import these names from this module.
# ---------------------------------------------------------------------------
from myspellchecker.providers._sqlite_schema import (  # noqa: F401
    _MISSING_COLUMN_PATTERNS,
    _MISSING_TABLE_PATTERNS,
    CURRENT_SCHEMA_VERSION,
    DEFAULT_PROVIDER_CACHE_SIZE,
    ITERATOR_FETCH_SIZE,
    MIN_COMPATIBLE_SCHEMA_VERSION,
    SQLITE_MAX_BATCH_SIZE,
    VALID_COLUMNS,
    VALID_DB_EXTENSIONS,
    VALID_TABLES,
    _check_schema_version,
    _is_missing_column_error,
    _is_missing_table_error,
    _validate_batch_items,
    _validate_database_path,
)
from myspellchecker.text.morphology import MorphologyAnalyzer
from myspellchecker.text.stemmer import Stemmer
from myspellchecker.utils.cache import LRUCache
from myspellchecker.utils.logging_utils import get_logger

from .base import DictionaryProvider
from .connection_pool import ConnectionPool

__all__ = [
    "SQLiteProvider",
]

# Default provider config instance — kept here for the one constant that is
# NOT derived in _sqlite_schema (it's only used inside __init__).
_DEFAULT_PROVIDER_CONFIG = ProviderConfig()


class SQLiteProvider(NEREntityMixin, EnrichmentMixin, CacheMixin, DictionaryProvider):
    """
    SQLite-based dictionary provider with caching and thread-safety.

    This provider reads dictionary data from a SQLite database created by
    the build pipeline (myspellchecker build command). It implements all required
    methods for syllable, word, and context-level spell checking.

    **Protocol Implementation**:
        Explicitly implements all provider protocol interfaces (PEP 544):
        - SyllableRepository: Syllable validation and frequency
        - WordRepository: Word validation and frequency
        - NgramRepository: N-gram probability lookups
        - POSRepository: POS tagging data access

        This allows validators and algorithms to depend on minimal interfaces
        instead of the full DictionaryProvider (Interface Segregation Principle).

    Features:
        - Fast lookups using database indexes
        - LRU caching for frequently accessed data
        - Thread-safe read operations
        - Lazy connection initialization
        - Automatic database path resolution

    Database Schema:
        - syllables(id, syllable, frequency)
        - words(id, word, syllable_count, frequency)
        - bigrams(id, word1_id, word2_id, probability)

    Thread Safety:
        This provider is thread-safe for read operations. Each thread gets
        its own database connection via thread-local storage.

    Example:
        >>> from myspellchecker.providers import SQLiteProvider
        >>>
        >>> # Use default database
        >>> provider = SQLiteProvider()
        >>>
        >>> # Check syllable validity
        >>> provider.is_valid_syllable("မြန်")
        True
        >>>
        >>> # Get frequency
        >>> freq = provider.get_syllable_frequency("မြန်")
        >>> print(f"Frequency: {freq}")
        >>>
        >>> # Use custom database
        >>> custom_provider = SQLiteProvider(database_path="/path/to/custom.db")
    """

    def __init__(
        self,
        database_path: str | None = None,
        cache_size: int = DEFAULT_PROVIDER_CACHE_SIZE,
        check_same_thread: bool = False,
        pos_tagger: POSTaggerBase | None = None,
        pool_min_size: int | None = None,
        pool_max_size: int | None = None,
        pool_timeout: float | None = None,
        pool_max_connection_age: float | None = None,
        sqlite_timeout: float | None = None,
        cache_manager: CacheManager | None = None,
        curated_min_frequency: int = 0,
    ):
        """
        Initialize SQLite provider.

        Args:
            database_path: Path to SQLite database file. Required — no bundled database
                    is included. Build one first with `myspellchecker build`.
            cache_size: Size of LRU cache for frequency lookups (default: 8192,
                matches AlgorithmCacheConfig.frequency_cache_size).
            check_same_thread: If False, allows sharing connection between threads
                              (default: False for better performance).
            pos_tagger: Optional POS tagger for OOV word tagging. If None, uses
                       RuleBasedPOSTagger with morphology fallback.
            pool_min_size: Minimum connections in pool (default: ConnectionPoolConfig.min_size).
            pool_max_size: Maximum connections in pool (default: ConnectionPoolConfig.max_size).
            pool_timeout: Connection checkout timeout in seconds
                (default: ConnectionPoolConfig.timeout).
            pool_max_connection_age: Max connection age before recreation
                (default: ConnectionPoolConfig.max_connection_age).
            sqlite_timeout: SQLite busy timeout in seconds
                (default: ConnectionPoolConfig.sqlite_timeout).
                Specifies how long to wait when the database is locked before
                raising an OperationalError. Prevents hangs in concurrent access.
            cache_manager: Optional CacheManager for dependency injection.
            curated_min_frequency: Minimum effective frequency for curated words
                (is_curated=1). Words below this floor are boosted to this value.
                Default 0 (disabled). Set to 50 to match SymSpell count_threshold.
                If provided, caches are created through the manager.
                Useful for testing and cache sharing across components.

        Raises:
            FileNotFoundError: If database file doesn't exist.
            sqlite3.DatabaseError: If database is corrupted or invalid.

        Example:
            >>> # Use custom database with larger cache
            >>> provider = SQLiteProvider(
            ...     database_path="/path/to/custom.db",
            ...     cache_size=2048
            ... )
            >>>
            >>> # Use custom pool configuration
            >>> provider = SQLiteProvider(
            ...     pool_min_size=2,
            ...     pool_max_size=10
            ... )
            >>>
            >>> # Use custom POS tagger
            >>> from myspellchecker.algorithms.pos_tagger_transformer import TransformerPOSTagger
            >>> tagger = TransformerPOSTagger()
            >>> provider = SQLiteProvider(pos_tagger=tagger)
            >>>
            >>> # Use shared CacheManager
            >>> from myspellchecker.utils.cache import CacheManager
            >>> manager = CacheManager(default_maxsize=2048)
            >>> provider = SQLiteProvider(cache_manager=manager)
        """
        self.logger = get_logger(__name__)

        # Resolve and validate database path (security: prevent path traversal)
        if database_path is None:
            raise MissingDatabaseError(
                message="No database path provided.",
                suggestion=(
                    "No bundled database is included. Build one first, then pass the path:\n"
                    "  1. myspellchecker build --sample  "
                    "# creates mySpellChecker-default.db\n"
                    "  2. SQLiteProvider(database_path='mySpellChecker-default.db')\n\n"
                    "Or build from your own corpus:\n"
                    "  myspellchecker build --input <corpus> --output my.db"
                ),
            )
        else:
            # Validate user-provided path for security
            self.database_path = _validate_database_path(Path(database_path))

        self.logger.info(f"Using SQLite database: {self.database_path}")

        # Validate database exists
        if not self.database_path.exists():
            raise MissingDatabaseError(
                message=f"Database not found: {self.database_path}",
                searched_paths=[str(self.database_path)],
                suggestion=(
                    "Build the database using the CLI:\n"
                    "  myspellchecker build --sample                    # Quick sample database\n"
                    "  myspellchecker build --input <corpus> --output <db>  "
                    "# Build from your corpus"
                ),
            )

        # Check schema version for compatibility
        self._schema_version = _check_schema_version(self.database_path, self.logger)

        # Connection settings
        self.check_same_thread = check_same_thread
        self.cache_size = cache_size

        # Initialize connection pool using ConnectionPoolConfig defaults for any
        # unspecified parameters (avoids duplicating default values here)
        from myspellchecker.core.config.validation_configs import ConnectionPoolConfig

        _defaults = ConnectionPoolConfig()
        pool_config = ConnectionPoolConfig(
            min_size=pool_min_size if pool_min_size is not None else _defaults.min_size,
            max_size=pool_max_size if pool_max_size is not None else _defaults.max_size,
            timeout=pool_timeout if pool_timeout is not None else _defaults.timeout,
            max_connection_age=(
                pool_max_connection_age
                if pool_max_connection_age is not None
                else _defaults.max_connection_age
            ),
            check_same_thread=check_same_thread,
            sqlite_timeout=(
                sqlite_timeout if sqlite_timeout is not None else _defaults.sqlite_timeout
            ),
            skip_health_check=True,  # Spell check DBs are read-only
        )
        self.logger.info(
            f"Initializing connection pool "
            f"(min={pool_config.min_size}, max={pool_config.max_size}, "
            f"timeout={pool_config.timeout}s)"
        )
        self._pool = ConnectionPool(database_path=self.database_path, pool_config=pool_config)

        # Cache management - use provided manager or create direct caches
        self._cache_manager = cache_manager
        caches = create_caches(cache_size, cache_manager, self.logger)
        self._word_id_cache = caches["word_id_cache"]
        self._syllable_freq_cache = caches["syllable_freq_cache"]
        self._word_freq_cache = caches["word_freq_cache"]
        self._valid_word_cache = caches["valid_word_cache"]
        self._valid_syllable_cache = caches["valid_syllable_cache"]
        self._bigram_prob_cache = caches["bigram_prob_cache"]
        self._trigram_prob_cache = caches["trigram_prob_cache"]
        self._fourgram_prob_cache = caches["fourgram_prob_cache"]
        self._fivegram_prob_cache = caches["fivegram_prob_cache"]

        # Initialize POS tagger for OOV word tagging
        self.pos_tagger: POSTaggerBase
        if pos_tagger is None:
            # Default to RuleBasedPOSTagger with morphology fallback
            self.pos_tagger = RuleBasedPOSTagger(use_morphology_fallback=True)
        else:
            self.pos_tagger = pos_tagger

        # MorphologyAnalyzer for OOV POS fallback
        self._morphology_analyzer = MorphologyAnalyzer()

        # Curated vocabulary frequency floor
        self._curated_min_frequency = curated_min_frequency

        # In-memory word set for O(1) membership testing.
        # Eliminates SQLite round-trips for is_valid_word() — the single
        # most called method in the pipeline (>1M calls per run).
        self._word_set: frozenset[str] | None = None
        self._word_freq_map: dict[str, int] | None = None

        # In-memory n-gram maps for O(1) probability lookups.
        # Eliminates ~47K SQLite queries (1.17s) per run.
        # Memory: ~80 MB bigrams + ~40 MB trigrams.
        self._bigram_map: dict[tuple[str, str], float] | None = None
        self._trigram_map: dict[tuple[str, str, str], float] | None = None

        # In-memory enrichment caches (loaded lazily on first access)
        self._confusable_map: dict[str, list[tuple[str, str, float, float, int]]] | None = None
        self._compound_map: dict[str, tuple[str, str, int, int, float]] | None = None
        self._collocation_map: dict[tuple[str, str], float] | None = None
        self._register_map: dict[str, str] | None = None
        self._enrichment_lock = threading.Lock()

        # NER entity cache (loaded lazily on first access)
        self._ner_entity_map: dict[str, set[str]] | None = None
        self._ner_lock = threading.Lock()

        # Initialize Stemmer for OOV root lookup
        self.stemmer = Stemmer()

        # Delegate POS resolution to extracted module
        from ._sqlite_pos_resolver import POSResolver

        self._pos_resolver = POSResolver(
            stemmer=self.stemmer,
            pos_tagger=self.pos_tagger,
            morphology_analyzer=self._morphology_analyzer,
        )

        # Delegate bulk operations to extracted module
        from ._sqlite_bulk_ops import BulkQueryExecutor

        self._bulk_ops = BulkQueryExecutor(
            execute_query_fn=self._execute_query,
            syllable_freq_cache=self._syllable_freq_cache,
            word_freq_cache=self._word_freq_cache,
            curated_min_frequency=self._curated_min_frequency,
            logger=self.logger,
        )

        # Pre-load word set LAST (after all other init is complete).
        self._preload_word_set()

    def _preload_word_set(self) -> None:
        """Pre-load words, frequencies, and n-gram probabilities into memory.

        Trades ~145 MB of memory for O(1) lookups across words, bigrams, and
        trigrams, eliminating millions of SQLite round-trips per run.
        """
        try:
            with self._execute_query() as conn:
                cursor = conn.cursor()

                # Load words + frequencies and build id-to-word map for n-gram resolution.
                # sqlite3.Row supports both index and key access; use index for speed.
                cursor.execute("SELECT id, word, frequency FROM words")
                word_freq: dict[str, int] = {}
                id_to_word: dict[int, str] = {}
                for row in cursor:
                    word_id, word, freq = row[0], row[1], row[2]
                    word_freq[word] = freq if freq else 0
                    id_to_word[word_id] = word

                self._word_set = frozenset(word_freq.keys())
                self._word_freq_map: dict[str, int] | None = word_freq
                self.logger.info(
                    f"Pre-loaded {len(self._word_set)} words + frequencies into memory"
                )

                # Load bigrams: (word1, word2) -> probability
                self._preload_bigrams(cursor, id_to_word)

                # Load trigrams: (word1, word2, word3) -> probability
                self._preload_trigrams(cursor, id_to_word)

        except (sqlite3.OperationalError, sqlite3.DatabaseError, OSError) as e:
            self.logger.warning(f"Failed to pre-load word data: {e}")
            self._word_set = None
            self._word_freq_map = None
            self._bigram_map = None
            self._trigram_map = None

    def _preload_bigrams(self, cursor: sqlite3.Cursor, id_to_word: dict[int, str]) -> None:
        """Pre-load bigram probabilities into an in-memory dict."""
        try:
            cursor.execute("SELECT word1_id, word2_id, probability FROM bigrams")
            bigram_map: dict[tuple[str, str], float] = {}
            for row in cursor:
                w1 = id_to_word.get(row[0])
                w2 = id_to_word.get(row[1])
                if w1 is not None and w2 is not None:
                    bigram_map[(w1, w2)] = float(row[2]) if row[2] else 0.0
            self._bigram_map = bigram_map
            self.logger.info(f"Pre-loaded {len(bigram_map):,} bigram probabilities into memory")
        except sqlite3.OperationalError as e:
            if not _is_missing_table_error(e):
                raise
            self.logger.debug("bigrams table not found; skipping bigram preload")
            self._bigram_map = None

    def _preload_trigrams(self, cursor: sqlite3.Cursor, id_to_word: dict[int, str]) -> None:
        """Pre-load trigram probabilities into an in-memory dict."""
        try:
            cursor.execute("SELECT word1_id, word2_id, word3_id, probability FROM trigrams")
            trigram_map: dict[tuple[str, str, str], float] = {}
            for row in cursor:
                w1 = id_to_word.get(row[0])
                w2 = id_to_word.get(row[1])
                w3 = id_to_word.get(row[2])
                if w1 is not None and w2 is not None and w3 is not None:
                    trigram_map[(w1, w2, w3)] = float(row[3]) if row[3] else 0.0
            self._trigram_map = trigram_map
            self.logger.info(f"Pre-loaded {len(trigram_map):,} trigram probabilities into memory")
        except sqlite3.OperationalError as e:
            if not _is_missing_table_error(e):
                raise
            self.logger.debug("trigrams table not found; skipping trigram preload")
            self._trigram_map = None

    @contextmanager
    def _execute_query(self) -> Generator[sqlite3.Connection, None, None]:
        """
        Context manager for database queries using connection pooling.

        This method checks out a connection from the pool, executes the query,
        and automatically returns the connection to the pool when done.

        Yields:
            SQLite connection for executing queries.

        Raises:
            TimeoutError: If connection pool is exhausted.

        Example:
            >>> with provider._execute_query() as conn:
            ...     cursor = conn.cursor()
            ...     cursor.execute("SELECT * FROM syllables WHERE syllable = ?", (text,))
            ...     result = cursor.fetchone()

        Performance:
            Fast connection reuse (~0.1ms overhead per checkout)
        """
        with self._pool.checkout() as conn:
            yield conn

    def _cached_frequency_query(
        self, cache: "LRUCache", key: str, table: str, key_column: str
    ) -> int:
        """
        Generic cache-aware frequency query helper.

        Args:
            cache: LRU cache instance to use.
            key: The lookup key (syllable or word).
            table: Database table name ('syllables' or 'words').
            key_column: Column name to query ('syllable' or 'word').

        Returns:
            Frequency count (0 if not found or on error).

        Security:
            Table and column names are validated against whitelist constants
            (VALID_TABLES, VALID_COLUMNS) to prevent SQL injection.
        """
        # Defense-in-depth: Validate table and column names against whitelist
        # This prevents SQL injection even if caller passes untrusted input
        # NOTE: Using explicit checks instead of assert because asserts can be
        # disabled with python -O, creating a security vulnerability
        if table not in VALID_TABLES:
            raise ValueError(f"Invalid table name: {table!r}. Must be one of {VALID_TABLES}")
        if key_column not in VALID_COLUMNS:
            raise ValueError(f"Invalid column name: {key_column!r}. Must be one of {VALID_COLUMNS}")

        # Check cache first
        cached = cache.get(key)
        if cached is not None:
            return cast(int, cached)

        # Query database with differentiated error handling
        try:
            with self._execute_query() as conn:
                cursor = conn.cursor()
                cursor.execute(f"SELECT frequency FROM {table} WHERE {key_column} = ?", (key,))
                result = cursor.fetchone()
                freq = result["frequency"] if result else 0
                # Cache successful lookups (including "not found" = 0)
                cache.set(key, freq)
                return freq
        except sqlite3.OperationalError as e:
            error_str = str(e).lower()
            # Only cache permanent schema errors (won't recover without rebuild)
            if any(
                pattern in error_str
                for pattern in _MISSING_COLUMN_PATTERNS + _MISSING_TABLE_PATTERNS
            ):
                self.logger.warning(f"Schema error for {key_column}={key!r}: {e} (caching as 0)")
                cache.set(key, 0)
                return 0
            # Don't cache transient errors (locks, I/O) - they may recover
            # Return 0 but allow retry on next call
            self.logger.warning(
                f"Transient database error for {key_column}={key!r}: {e} (not caching)"
            )
            return 0
        except sqlite3.DatabaseError as e:
            # Other database errors (IntegrityError, ProgrammingError, etc.)
            # These are unexpected and should be logged at error level
            self.logger.error(
                f"Unexpected database error for {key_column}={key!r}: {e} (not caching)"
            )
            # Don't cache - let caller handle or retry
            return 0

    def is_valid_syllable(self, syllable: str) -> bool:
        """
        Check if a syllable exists in the dictionary.

        This method provides fast syllable validation using an indexed
        database query. Results are not cached as SQLite's built-in
        page cache is sufficient for common queries.

        Args:
            syllable: Myanmar syllable (Unicode string) to validate.

        Returns:
            True if syllable exists in dictionary, False otherwise.

        Performance:
            ~0.1-0.5ms per query (with SQLite index)

        Example:
            >>> provider = SQLiteProvider()
            >>> provider.is_valid_syllable("မြန်")
            True
            >>> provider.is_valid_syllable("xyz")
            False
            >>> provider.is_valid_syllable("")
            False

        Notes:
            - Empty strings return False
            - Uses indexed lookup on syllables.syllable column
            - Thread-safe (uses connection pooling or thread-local connections)
        """
        if not syllable:
            return False

        cached = self._valid_syllable_cache.get(syllable)
        if cached is not None:
            return cast(bool, cached)

        with self._execute_query() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM syllables WHERE syllable = ? LIMIT 1", (syllable,))
            result = cursor.fetchone()
            is_valid = result is not None

        self._valid_syllable_cache.set(syllable, is_valid)
        return is_valid

    def is_valid_word(self, word: str) -> bool:
        """
        Check if a multi-syllable word exists in the dictionary.

        This method validates complete words for Layer 2 spell checking.
        A word may consist of valid syllables but still be invalid as
        a combination.

        Args:
            word: Myanmar word (Unicode string) to validate.

        Returns:
            True if word exists in dictionary, False otherwise.

        Performance:
            ~0.1-0.5ms per query (with SQLite index)

        Example:
            >>> provider = SQLiteProvider()
            >>> provider.is_valid_word("မြန်မာ")  # "Myanmar"
            True
            >>> provider.is_valid_word("မြန်စာ")  # Invalid combination
            False
            >>> provider.is_valid_word("")
            False

        Notes:
            - Empty strings return False
            - Uses indexed lookup on words.word column
            - Thread-safe (uses connection pooling or thread-local connections)
        """
        if not word:
            return False

        # Fast path: in-memory word set (O(1), no SQLite round-trip)
        if self._word_set is not None:
            return word in self._word_set

        cached = self._valid_word_cache.get(word)
        if cached is not None:
            return cast(bool, cached)

        with self._execute_query() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM words WHERE word = ? LIMIT 1", (word,))
            result = cursor.fetchone()
            is_valid = result is not None

        self._valid_word_cache.set(word, is_valid)
        return is_valid

    def is_valid_vocabulary(self, word: str) -> bool:
        """
        Check if a word is curated vocabulary (from POS seed).

        This method provides strict validation for words that are known
        to be legitimate vocabulary entries, as opposed to words that
        may have been extracted from corpus segmentation but aren't
        necessarily valid vocabulary.

        Args:
            word: Myanmar word (Unicode string) to validate.

        Returns:
            True if word is curated vocabulary (is_curated=1), False otherwise.

        Performance:
            ~0.1-0.5ms per query (with SQLite index)

        Example:
            >>> provider = SQLiteProvider()
            >>> provider.is_valid_vocabulary("မြန်မာ")  # Curated word
            True
            >>> provider.is_valid_vocabulary("segmentation_artifact")
            False

        Notes:
            - Empty strings return False
            - Only returns True for words with is_curated=1
            - Thread-safe
        """
        if not word:
            return False

        with self._execute_query() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    "SELECT 1 FROM words WHERE word = ? AND is_curated = 1 LIMIT 1",
                    (word,),
                )
                result = cursor.fetchone()
                return result is not None
            except sqlite3.OperationalError as e:
                # Only handle missing column errors (schema compatibility)
                # Re-raise critical errors (disk I/O, corruption, locks)
                if not _is_missing_column_error(e):
                    raise
                # Fallback if is_curated column doesn't exist (old database)
                self.logger.warning("is_curated column not found. Using is_valid_word fallback.")
                return self.is_valid_word(word)

    def get_syllable_frequency(self, syllable: str) -> int:
        """
        Get corpus frequency count for a syllable.

        Frequency data is used by SymSpell to rank correction suggestions.
        This method uses an LRU cache to avoid repeated database queries
        for common syllables.

        Args:
            syllable: Myanmar syllable (Unicode string).

        Returns:
            Integer frequency count (0 if not found).

        Performance:
            - Cache hit: ~0.001ms (dict lookup)
            - Cache miss: ~0.1-0.5ms (database query + cache store)

        Example:
            >>> provider = SQLiteProvider()
            >>> provider.get_syllable_frequency("မြန်")
            15432
            >>> provider.get_syllable_frequency("xyz")
            0

        Notes:
            - Returns 0 for unknown syllables
            - Uses LRU cache (size configurable in __init__)
            - LRU eviction when cache_size exceeded
            - Thread-safe
        """
        if not syllable:
            return 0
        return self._cached_frequency_query(
            self._syllable_freq_cache, syllable, "syllables", "syllable"
        )

    def get_word_frequency(self, word: str) -> int:
        """
        Get corpus frequency count for a word.

        Similar to get_syllable_frequency but for multi-syllable words.
        Used for word-level suggestion ranking.

        Args:
            word: Myanmar word (Unicode string).

        Returns:
            Integer frequency count (0 if not found).

        Performance:
            - Cache hit: ~0.001ms (dict lookup)
            - Cache miss: ~0.1-0.5ms (database query + cache store)

        Example:
            >>> provider = SQLiteProvider()
            >>> provider.get_word_frequency("မြန်မာ")  # "Myanmar"
            8752
            >>> provider.get_word_frequency("unknown")
            0

        Notes:
            - Returns 0 for unknown words
            - Uses LRU cache (size configurable in __init__)
            - LRU eviction when cache_size exceeded
            - Thread-safe
        """
        if not word:
            return 0

        # Fast path: in-memory frequency map (O(1), no SQLite round-trip)
        if self._word_freq_map is not None:
            raw_freq = self._word_freq_map.get(word, 0)
            if raw_freq < self._curated_min_frequency and self._curated_min_frequency > 0:
                if self.is_valid_vocabulary(word):
                    return self._curated_min_frequency
            return raw_freq

        raw_freq = self._cached_frequency_query(self._word_freq_cache, word, "words", "word")
        if raw_freq < self._curated_min_frequency and self._curated_min_frequency > 0:
            if self.is_valid_vocabulary(word):
                return self._curated_min_frequency
        return raw_freq

    def get_word_pos(self, word: str) -> str | None:
        """
        Get Part-of-Speech (POS) tag(s) for a word.

        Fallback chain:
        1. Direct dictionary lookup (pos_tag from transformer, then inferred_pos)
        2. Stemming + root lookup with suffix transformation rules
        3. POS tagger for OOV words
        4. MorphologyAnalyzer guess as last resort

        Args:
            word: Myanmar word (Unicode string).

        Returns:
            POS tag string or None if not found/unknown. Multi-POS words
            return pipe-separated tags ordered by corpus frequency, e.g.
            ``'N|V'``. Use ``tag in result.split('|')`` to check membership.

        Note:
            The pos_tag column may contain pipe-separated multi-POS tags from
            the HuggingFace transformer (e.g., ``N|V``, ``N|ADJ``). The
            inferred_pos column may contain more granular tags from rule-based
            inference (e.g., P_SUBJ, P_OBJ). When both exist, pos_tag is
            preferred as it comes from the neural model.
        """
        if not word:
            return None

        with self._execute_query() as conn:
            cursor = conn.cursor()

            try:
                return self._pos_resolver.resolve(cursor, word)
            except sqlite3.OperationalError as e:
                # Only handle missing column errors (schema compatibility)
                # Re-raise critical errors (disk I/O, corruption, locks)
                if not _is_missing_column_error(e):
                    raise
                self.logger.warning("pos_tag column missing in words table")
                return None

    def get_word_id(self, word: str) -> int | None:
        """
        Get database ID for a word (helper for bigram lookups).

        This is an internal helper method used by get_bigram_probability().
        Results are cached using an LRU cache to avoid repeated lookups
        while preventing unbounded memory growth.

        Args:
            word: Myanmar word (Unicode string).

        Returns:
            Integer word ID, or None if word not found.

        Notes:
            - Internal method, not part of public API
            - Results are cached with LRU eviction (size = cache_size)
            - Thread-safe
        """
        if not word:
            return None

        # Check LRU cache first (use sentinel to distinguish "not cached" from "cached None")
        _NOT_FOUND = -1  # Word IDs are always positive, so -1 is a safe sentinel
        cached = self._word_id_cache.get(word, _NOT_FOUND)
        if cached != _NOT_FOUND:
            return cached if cached is not None else None

        # Query database
        with self._execute_query() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM words WHERE word = ?", (word,))
            result = cursor.fetchone()

            if result:
                word_id = int(result["id"])
                # Cache the result (LRU cache is thread-safe)
                self._word_id_cache.set(word, word_id)
                return word_id

        # Cache the "not found" result to avoid repeated DB queries
        self._word_id_cache.set(word, None)
        return None

    def get_bigram_probability(self, prev_word: str, current_word: str) -> float:
        """
        Get conditional probability P(current_word | prev_word).

        This method supports Layer 3 (context-aware) spell checking by
        providing bigram probabilities for detecting unlikely word sequences.

        The query uses a JOIN to lookup word IDs and retrieve the probability.
        This is optimized with proper indexes on the bigrams table.

        Args:
            prev_word: Previous word in sequence (Unicode string).
            current_word: Current word in sequence (Unicode string).

        Returns:
            Probability as float in range [0.0, 1.0].
            Returns 0.0 if bigram not found.

        Performance:
            - ~1-5ms per query (with JOIN and indexes)
            - Faster if word IDs are cached

        Example:
            >>> provider = SQLiteProvider()
            >>> provider.get_bigram_probability("သူ", "သွား")
            0.234  # Common sequence
            >>> provider.get_bigram_probability("abc", "xyz")
            0.0  # Unknown words

        Notes:
            - Returns 0.0 for unseen bigrams
            - Uses indexed JOIN on bigrams table
            - Thread-safe
        """
        if not prev_word or not current_word:
            return 0.0

        # Fast path: in-memory bigram map (O(1), no SQLite round-trip)
        if self._bigram_map is not None:
            return self._bigram_map.get((prev_word, current_word), 0.0)

        cache_key = (prev_word, current_word)
        cached = self._bigram_prob_cache.get(cache_key)
        if cached is not None:
            return cast(float, cached)

        # Get word IDs (with caching)
        word1_id = self.get_word_id(prev_word)
        word2_id = self.get_word_id(current_word)

        if word1_id is None or word2_id is None:
            self._bigram_prob_cache.set(cache_key, 0.0)
            return 0.0

        # Query bigram probability
        with self._execute_query() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT probability FROM bigrams WHERE word1_id = ? AND word2_id = ?",
                (word1_id, word2_id),
            )
            result = cursor.fetchone()
            prob = float(result["probability"]) if result else 0.0

        self._bigram_prob_cache.set(cache_key, prob)
        return prob

    def get_trigram_probability(self, w1: str, w2: str, w3: str) -> float:
        """
        Get conditional probability P(w3 | w1, w2).

        Args:
            w1: First word.
            w2: Second word.
            w3: Third word (target).

        Returns:
            Probability as float in range [0.0, 1.0].
            Returns 0.0 if trigram not found.
        """
        if not w1 or not w2 or not w3:
            return 0.0

        # Fast path: in-memory trigram map (O(1), no SQLite round-trip)
        if self._trigram_map is not None:
            return self._trigram_map.get((w1, w2, w3), 0.0)

        cache_key = (w1, w2, w3)
        cached = self._trigram_prob_cache.get(cache_key)
        if cached is not None:
            return cast(float, cached)

        # Get word IDs (with caching)
        id1 = self.get_word_id(w1)
        id2 = self.get_word_id(w2)
        id3 = self.get_word_id(w3)

        if id1 is None or id2 is None or id3 is None:
            self._trigram_prob_cache.set(cache_key, 0.0)
            return 0.0

        with self._execute_query() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT probability FROM trigrams "
                "WHERE word1_id = ? AND word2_id = ? AND word3_id = ?",
                (id1, id2, id3),
            )
            result = cursor.fetchone()
            prob = float(result["probability"]) if result else 0.0

        self._trigram_prob_cache.set(cache_key, prob)
        return prob

    def get_fourgram_probability(self, word1: str, word2: str, word3: str, word4: str) -> float:
        """
        Get conditional probability P(word4 | word1, word2, word3).

        Args:
            word1: First word.
            word2: Second word.
            word3: Third word.
            word4: Fourth word (target).

        Returns:
            Probability as float in range [0.0, 1.0].
            Returns 0.0 if fourgram not found or table missing.
        """
        if not word1 or not word2 or not word3 or not word4:
            return 0.0

        cache_key = (word1, word2, word3, word4)
        cached = self._fourgram_prob_cache.get(cache_key)
        if cached is not None:
            return cast(float, cached)

        # Get word IDs (with caching)
        id1 = self.get_word_id(word1)
        id2 = self.get_word_id(word2)
        id3 = self.get_word_id(word3)
        id4 = self.get_word_id(word4)

        if id1 is None or id2 is None or id3 is None or id4 is None:
            self._fourgram_prob_cache.set(cache_key, 0.0)
            return 0.0

        with self._execute_query() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    "SELECT probability FROM fourgrams "
                    "WHERE word1_id = ? AND word2_id = ? AND word3_id = ? AND word4_id = ?",
                    (id1, id2, id3, id4),
                )
                result = cursor.fetchone()
                prob = float(result["probability"]) if result else 0.0
            except sqlite3.OperationalError as e:
                # Only handle missing table errors (schema compatibility)
                # Re-raise critical errors (disk I/O, corruption, locks)
                if not _is_missing_table_error(e):
                    raise
                self.logger.debug("fourgrams table not found. Returning 0.0.")
                prob = 0.0

        self._fourgram_prob_cache.set(cache_key, prob)
        return prob

    def get_fivegram_probability(
        self, word1: str, word2: str, word3: str, word4: str, word5: str
    ) -> float:
        """
        Get conditional probability P(word5 | word1, word2, word3, word4).

        Args:
            word1: First word.
            word2: Second word.
            word3: Third word.
            word4: Fourth word.
            word5: Fifth word (target).

        Returns:
            Probability as float in range [0.0, 1.0].
            Returns 0.0 if fivegram not found or table missing.
        """
        if not word1 or not word2 or not word3 or not word4 or not word5:
            return 0.0

        cache_key = (word1, word2, word3, word4, word5)
        cached = self._fivegram_prob_cache.get(cache_key)
        if cached is not None:
            return cast(float, cached)

        # Get word IDs (with caching)
        id1 = self.get_word_id(word1)
        id2 = self.get_word_id(word2)
        id3 = self.get_word_id(word3)
        id4 = self.get_word_id(word4)
        id5 = self.get_word_id(word5)

        if id1 is None or id2 is None or id3 is None or id4 is None or id5 is None:
            self._fivegram_prob_cache.set(cache_key, 0.0)
            return 0.0

        with self._execute_query() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    "SELECT probability FROM fivegrams "
                    "WHERE word1_id = ? AND word2_id = ? AND word3_id = ? "
                    "AND word4_id = ? AND word5_id = ?",
                    (id1, id2, id3, id4, id5),
                )
                result = cursor.fetchone()
                prob = float(result["probability"]) if result else 0.0
            except sqlite3.OperationalError as e:
                # Only handle missing table errors (schema compatibility)
                # Re-raise critical errors (disk I/O, corruption, locks)
                if not _is_missing_table_error(e):
                    raise
                self.logger.debug("fivegrams table not found. Returning 0.0.")
                prob = 0.0

        self._fivegram_prob_cache.set(cache_key, prob)
        return prob

    def get_pos_unigram_probabilities(self) -> dict[str, float]:
        """
        Get all POS unigram probabilities from the database.

        Returns:
            Dictionary mapping pos_tag (str) to probability (float).
        """
        with self._execute_query() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("SELECT pos, probability FROM pos_unigrams")
                return {row["pos"]: float(row["probability"]) for row in cursor.fetchall()}
            except sqlite3.OperationalError as e:
                # Only handle missing table errors (schema compatibility)
                # Re-raise critical errors (disk I/O, corruption, locks)
                if not _is_missing_table_error(e):
                    raise
                self.logger.warning("pos_unigrams table not found. Returning empty dict.")
                return {}

    def get_pos_bigram_probabilities(self) -> dict[tuple[str, str], float]:
        """
        Get all POS bigram probabilities from the database.

        Returns:
            Dictionary mapping (pos1, pos2) tuple to probability (float).
        """
        with self._execute_query() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("SELECT pos1, pos2, probability FROM pos_bigrams")
                return {
                    (row["pos1"], row["pos2"]): float(row["probability"])
                    for row in cursor.fetchall()
                }
            except sqlite3.OperationalError as e:
                # Only handle missing table errors (schema compatibility)
                # Re-raise critical errors (disk I/O, corruption, locks)
                if not _is_missing_table_error(e):
                    raise
                self.logger.warning("pos_bigrams table not found. Returning empty dict.")
                return {}

    def get_pos_trigram_probabilities(self) -> dict[tuple[str, str, str], float]:
        """
        Get all POS trigram probabilities from the database.

        Returns:
            Dictionary mapping (pos1, pos2, pos3) tuple to probability (float).
        """
        with self._execute_query() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("SELECT pos1, pos2, pos3, probability FROM pos_trigrams")
                return {
                    (row["pos1"], row["pos2"], row["pos3"]): float(row["probability"])
                    for row in cursor.fetchall()
                }
            except sqlite3.OperationalError as e:
                # Only handle missing table errors (schema compatibility)
                # Re-raise critical errors (disk I/O, corruption, locks)
                if not _is_missing_table_error(e):
                    raise
                self.logger.warning("pos_trigrams table not found. Returning empty dict.")
                return {}

    def get_top_continuations(self, prev_word: str, limit: int = 20) -> list[tuple[str, float]]:
        """
        Get the most likely words to follow a given word.

        This method retrieves the top N words that commonly follow prev_word
        based on bigram probabilities stored in the database.

        Args:
            prev_word: Previous word in sequence (Unicode string).
            limit: Maximum number of continuations to return (default: 20).

        Returns:
            List of (word, probability) tuples, sorted by probability (descending).

        Performance:
            - ~2-10ms per query (with JOIN and indexes)

        Example:
            >>> provider = SQLiteProvider()
            >>> continuations = provider.get_top_continuations("သူ", limit=5)
            >>> for word, prob in continuations:
            ...     print(f"{word}: {prob:.3f}")

        Notes:
            - Returns empty list for unknown words
            - Results are sorted by probability (highest first)
            - Uses indexed JOIN for performance
        """
        if not prev_word:
            return []

        # Get word ID for prev_word
        word_id = self.get_word_id(prev_word)
        if word_id is None:
            return []

        # Query top continuations
        with self._execute_query() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT w.word, b.probability
                FROM bigrams b
                JOIN words w ON b.word2_id = w.id
                WHERE b.word1_id = ?
                ORDER BY b.probability DESC
                LIMIT ?
                """,
                (word_id, limit),
            )
            results = cursor.fetchall()
            return [(row["word"], row["probability"]) for row in results]

    def get_all_syllables(self) -> Iterator[tuple[str, int]]:
        """
        Get iterator over all syllables.

        Uses batch fetching to prevent memory exhaustion for large dictionaries.
        Fetches ITERATOR_FETCH_SIZE rows at a time instead of loading all into memory.

        The iterator properly cleans up the database connection even if iteration
        is abandoned early (e.g., via break or exception).

        Warning:
            This method holds a connection for the entire iteration duration.
            Abandoned iteration (not fully consumed) keeps the connection checked
            out until the iterator is garbage collected, which can cause pool
            exhaustion in high-concurrency scenarios.

        Yields:
            Tuple of (syllable, frequency) for each syllable in the database.
        """
        # Fetch all rows with connection held, then yield outside the `with`
        # block so the connection is returned to the pool promptly even if
        # the caller abandons iteration early.
        rows: list[tuple[str, int]] = []
        with self._execute_query() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("SELECT syllable, frequency FROM syllables")
                rows = [(r["syllable"], r["frequency"]) for r in cursor.fetchall()]
            finally:
                cursor.close()
        yield from rows

    def get_all_words(self) -> Iterator[tuple[str, int]]:
        """
        Get iterator over all words.

        Uses batch fetching to prevent memory exhaustion for large dictionaries.
        Fetches ITERATOR_FETCH_SIZE rows at a time instead of loading all into memory.

        The iterator properly cleans up the database cursor even if iteration
        is abandoned early (e.g., via break or exception).

        Warning:
            This method holds a connection for the entire iteration duration.
            Abandoned iteration (not fully consumed) keeps the connection checked
            out until the iterator is garbage collected, which can cause pool
            exhaustion in high-concurrency scenarios.

        Yields:
            Tuple of (word, frequency) for each word in the database.
        """
        # Fetch all rows with connection held, then yield outside the `with`
        # block so the connection is returned to the pool promptly.
        result_rows: list[tuple[str, int]] = []
        with self._execute_query() as conn:
            cursor = conn.cursor()
            try:
                try:
                    cursor.execute("SELECT word, frequency, is_curated FROM words")
                except sqlite3.OperationalError as e:
                    if not _is_missing_column_error(e):
                        raise
                    cursor.execute("SELECT word, frequency FROM words")
                has_curated = cursor.description and len(cursor.description) > 2
                for row in cursor.fetchall():
                    freq = row["frequency"]
                    if (
                        has_curated
                        and self._curated_min_frequency > 0
                        and row["is_curated"]
                        and freq < self._curated_min_frequency
                    ):
                        freq = self._curated_min_frequency
                    result_rows.append((row["word"], freq))
            finally:
                cursor.close()
        yield from result_rows

    def get_statistics(self) -> dict:
        """
        Get database statistics (counts, sizes, etc.).

        This is a utility method for debugging and monitoring.

        Returns:
            Dictionary with database statistics:
                - syllable_count: Number of syllables in database
                - word_count: Number of words in database
                - bigram_count: Number of bigrams in database
                - db_size_bytes: Database file size in bytes
                - database_path: Path to database file

        Example:
            >>> provider = SQLiteProvider()
            >>> stats = provider.get_statistics()
            >>> print(f"Syllables: {stats['syllable_count']:,}")
            >>> print(f"Words: {stats['word_count']:,}")
            >>> print(f"DB size: {stats['db_size_bytes'] / 1024:.2f} KB")

        Notes:
            - This method is primarily for debugging and monitoring
            - Not used during normal spell checking operations
        """
        with self._execute_query() as conn:
            cursor = conn.cursor()

            # Count syllables
            cursor.execute("SELECT COUNT(*) as count FROM syllables")
            syllable_count = cursor.fetchone()["count"]

            # Count words
            cursor.execute("SELECT COUNT(*) as count FROM words")
            word_count = cursor.fetchone()["count"]

            # Count bigrams
            cursor.execute("SELECT COUNT(*) as count FROM bigrams")
            bigram_count = cursor.fetchone()["count"]

            # Count trigrams
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trigrams'")
            if cursor.fetchone():
                cursor.execute("SELECT COUNT(*) as count FROM trigrams")
                trigram_count = cursor.fetchone()["count"]
            else:
                trigram_count = 0

            # Count fourgrams
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='fourgrams'")
            if cursor.fetchone():
                cursor.execute("SELECT COUNT(*) as count FROM fourgrams")
                fourgram_count = cursor.fetchone()["count"]
            else:
                fourgram_count = 0

            # Count fivegrams
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='fivegrams'")
            if cursor.fetchone():
                cursor.execute("SELECT COUNT(*) as count FROM fivegrams")
                fivegram_count = cursor.fetchone()["count"]
            else:
                fivegram_count = 0

            # Count POS unigrams
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='pos_unigrams'"
            )
            if cursor.fetchone():
                cursor.execute("SELECT COUNT(*) as count FROM pos_unigrams")
                pos_unigram_count = cursor.fetchone()["count"]
            else:
                pos_unigram_count = 0

            # Count POS bigrams
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='pos_bigrams'"
            )
            if cursor.fetchone():
                cursor.execute("SELECT COUNT(*) as count FROM pos_bigrams")
                pos_bigram_count = cursor.fetchone()["count"]
            else:
                pos_bigram_count = 0

            # Count curated vocabulary (words with is_curated=1)
            try:
                cursor.execute("SELECT COUNT(*) as count FROM words WHERE is_curated = 1")
                curated_word_count = cursor.fetchone()["count"]
            except sqlite3.OperationalError as e:
                # Only handle missing column errors (schema compatibility)
                # Re-raise critical errors (disk I/O, corruption, locks)
                if not _is_missing_column_error(e):
                    raise
                # is_curated column doesn't exist in old databases
                self.logger.debug("is_curated column not found (old database format)")
                curated_word_count = 0

        # Get database size (after closing connection)
        db_size = self.database_path.stat().st_size

        return {
            STATS_KEY_SYLLABLE_COUNT: syllable_count,
            STATS_KEY_WORD_COUNT: word_count,
            STATS_KEY_BIGRAM_COUNT: bigram_count,
            STATS_KEY_TRIGRAM_COUNT: trigram_count,
            STATS_KEY_FOURGRAM_COUNT: fourgram_count,
            STATS_KEY_FIVEGRAM_COUNT: fivegram_count,
            "pos_unigram_count": pos_unigram_count,
            "pos_bigram_count": pos_bigram_count,
            "curated_word_count": curated_word_count,
            "db_size_bytes": db_size,
            "database_path": str(self.database_path),
        }

    def get_metadata(self, key: str) -> str | None:
        """Get a metadata value from the database.

        Args:
            key: Metadata key (e.g., 'total_word_count')

        Returns:
            Metadata value as string, or None if not found or table doesn't exist.
        """
        try:
            with self._execute_query() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT value FROM metadata WHERE key = ?", (key,))
                row = cursor.fetchone()
                return row[0] if row else None
        except (sqlite3.OperationalError, sqlite3.DatabaseError) as e:
            self.logger.debug("get_metadata('%s') failed: %s", key, e)
            return None

    def close(self) -> None:
        """
        Close all database connections in the pool.

        Typically not needed as connections are closed automatically,
        but useful for explicit cleanup in long-running applications.

        Example:
            >>> provider = SQLiteProvider()
            >>> # ... use provider ...
            >>> provider.close()

        Notes:
            - Connections are closed automatically when object is destroyed
            - Safe to call multiple times
            - Closes ALL pooled connections
        """
        if hasattr(self, "_pool") and self._pool is not None:
            self._pool.close_all()
            self.logger.debug("Connection pool closed")

    def __enter__(self) -> SQLiteProvider:
        """
        Context manager entry.

        Enables pythonic usage with 'with' statement for automatic resource cleanup.

        Returns:
            Self for use in the with statement body.

        Example:
            >>> with SQLiteProvider("mydict.db") as provider:
            ...     syllable_valid = provider.is_valid_syllable("မြန်")
            ...     # Automatic cleanup on exit
        """
        return self

    def __exit__(
        self,
        exc_type: type | None,
        exc_val: BaseException | None,
        exc_tb: object | None,
    ) -> None:
        """
        Context manager exit.

        Automatically closes all pooled connections when exiting the context.
        This ensures proper resource cleanup even if exceptions occur.

        Args:
            exc_type: Exception type if an exception was raised, None otherwise.
            exc_val: Exception value if an exception was raised, None otherwise.
            exc_tb: Exception traceback if an exception was raised, None otherwise.

        Example:
            >>> with SQLiteProvider("mydict.db") as provider:
            ...     provider.is_valid_syllable("မြန်")
            ...     # If exception occurs here, connections still closed
        """
        self.close()

    # ==========================================================================
    # Optimized Bulk Operations - delegated to BulkQueryExecutor
    # ==========================================================================

    def is_valid_syllables_bulk(self, syllables: list[str]) -> dict[str, bool]:
        """Check validity of multiple syllables using optimized batch query."""
        return self._bulk_ops.is_valid_syllables_bulk(syllables)

    def is_valid_words_bulk(self, words: list[str]) -> dict[str, bool]:
        """Check validity of multiple words using optimized batch query."""
        return self._bulk_ops.is_valid_words_bulk(words)

    def is_valid_vocabulary_bulk(self, words: list[str]) -> dict[str, bool]:
        """Check validity of multiple words as curated vocabulary."""
        return self._bulk_ops.is_valid_vocabulary_bulk(words)

    def get_syllable_frequencies_bulk(self, syllables: list[str]) -> dict[str, int]:
        """Get corpus frequencies for multiple syllables using batch query."""
        return self._bulk_ops.get_syllable_frequencies_bulk(syllables)

    def get_word_frequencies_bulk(self, words: list[str]) -> dict[str, int]:
        """Get corpus frequencies for multiple words using batch query."""
        return self._bulk_ops.get_word_frequencies_bulk(words)

    def get_word_pos_bulk(self, words: list[str]) -> dict[str, str | None]:
        """Get POS tags for multiple words using batch query."""
        return self._bulk_ops.get_word_pos_bulk(words)

    def __del__(self) -> None:
        """
        Cleanup: close connection when object is destroyed.

        Note: __del__ is not guaranteed to be called and should not be relied
        upon for cleanup. Always use the context manager or explicitly call
        close() when possible.

        Robustness improvements:
        - Guards against partially initialized objects (e.g., when __init__
          raises an exception before _pool is created)
        - Catches and suppresses all exceptions during cleanup to avoid
          raising from __del__ (which can cause interpreter issues)
        - Attempts to log errors but fails silently if logger is unavailable
          (common during interpreter shutdown)
        """
        # Guard against __del__ being called on partially initialized objects
        # This can happen if __init__ raises before _pool is assigned
        if not hasattr(self, "_pool"):
            return

        try:
            self.close()
        except (RuntimeError, sqlite3.Error, OSError, AttributeError) as e:
            # Destructor errors cannot be reliably logged as logger may be
            # torn down during interpreter shutdown. Best effort: try to log
            # but fail silently if logging is unavailable.
            try:
                if hasattr(self, "logger"):
                    self.logger.debug(f"Error during SQLiteProvider cleanup: {e}")
            except (RuntimeError, AttributeError):
                pass  # Logger unavailable during interpreter shutdown

    def __repr__(self) -> str:
        """String representation of provider."""
        return f"SQLiteProvider(database_path='{self.database_path}')"
