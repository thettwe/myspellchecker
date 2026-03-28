"""
Stress tests for batch processing.
"""

import time

import pytest

from myspellchecker import SpellChecker
from myspellchecker.providers import SQLiteProvider

pytestmark = pytest.mark.slow

# Check if pytest-benchmark is available
try:
    import pytest_benchmark  # noqa: F401

    HAS_BENCHMARK = True
except ImportError:
    HAS_BENCHMARK = False


@pytest.fixture
def spellchecker(test_database_path):
    """Create a SpellChecker instance with SQLiteProvider."""
    provider = SQLiteProvider(database_path=str(test_database_path))
    checker = SpellChecker(provider=provider)
    yield checker
    provider.close()


@pytest.mark.skipif(not HAS_BENCHMARK, reason="pytest-benchmark not installed")
def test_large_batch_processing(spellchecker, benchmark):
    """
    Test processing a large batch of texts.
    """
    # Generate 1000 sample texts
    texts = [
        "မြန်မာ နိုင်ငံ သည် လှပ သည်",
        "သူ ကျောင်း သွား သည်",
        "မင်္ဂလာ ပါ ခင်ဗျာ",
        "နေကောင်း လား",
        "ထမင်း စား ပြီး ပြီ လား",
    ] * 200  # 5 * 200 = 1000 texts

    def process_batch():
        results = []
        for text in texts:
            results.append(spellchecker.check(text))
        return results

    # Benchmark the processing
    results = benchmark(process_batch)

    assert len(results) == 1000
    assert all(r.text for r in results)


def test_batch_processing_performance_threshold(spellchecker):
    """
    Ensure batch processing meets performance threshold.
    Target: > 100 texts per second on standard hardware.
    """
    texts = ["မြန်မာ နိုင်ငံ သည် လှပ သည်", "သူ ကျောင်း သွား သည်"] * 500  # 1000 texts

    start_time = time.time()
    for text in texts:
        spellchecker.check(text)
    end_time = time.time()

    duration = end_time - start_time
    texts_per_second = len(texts) / duration

    # Log performance

    # Assert minimum performance (adjusted for CI/test environments)
    # This is a soft assertion, mainly for monitoring
    assert texts_per_second > 50
