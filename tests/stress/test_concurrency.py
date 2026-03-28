"""
Stress tests for concurrency.
"""

import random
from concurrent.futures import ThreadPoolExecutor

import pytest

from myspellchecker import SpellChecker
from myspellchecker.providers import SQLiteProvider

pytestmark = pytest.mark.slow


@pytest.fixture
def spellchecker(test_database_path):
    """Create a SpellChecker instance with SQLiteProvider."""
    provider = SQLiteProvider(database_path=str(test_database_path), pool_max_size=15)
    checker = SpellChecker(provider=provider)
    yield checker
    provider.close()


def test_concurrent_requests(spellchecker):
    """
    Test concurrent requests to SpellChecker.
    """
    rng = random.Random(42)
    texts = ["မြန်မာ နိုင်ငံ", "သူ ကျောင်း သွား", "မင်္ဂလာ ပါ", "နေကောင်း လား", "ထမင်း စား"]

    errors = []

    def worker():
        try:
            # Randomly select text and check
            text = rng.choice(texts)
            result = spellchecker.check(text)

            # Basic validation
            assert result.text == text
            assert isinstance(result.has_errors, bool)

            # Randomly access provider directly
            if rng.random() > 0.5:
                spellchecker.provider.is_valid_word("မြန်မာ")

        except Exception as e:
            errors.append(e)

    # Run with 10 threads, 100 iterations each
    n_threads = 10
    n_iterations = 100

    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = [executor.submit(worker) for _ in range(n_threads * n_iterations)]

        # Wait for all to complete
        for future in futures:
            future.result()

    assert len(errors) == 0, f"Encountered errors during concurrent execution: {errors}"


def test_concurrent_provider_access(spellchecker):
    """
    Test concurrent access to the underlying provider.
    """
    provider = spellchecker.provider
    errors = []

    def worker():
        try:
            # Mix of read operations
            provider.is_valid_syllable("မြန်")
            provider.get_word_frequency("မြန်မာ")
            provider.get_bigram_probability("သူ", "သွား")
            provider.get_top_continuations("သူ")
        except Exception as e:
            errors.append(e)

    # Run with 20 threads
    n_threads = 20
    n_iterations = 50

    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = [executor.submit(worker) for _ in range(n_threads * n_iterations)]

        for future in futures:
            future.result()

    assert len(errors) == 0, f"Provider errors: {errors}"
