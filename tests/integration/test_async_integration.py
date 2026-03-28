import asyncio

from myspellchecker import SpellChecker, ValidationLevel
from myspellchecker.core.config import SpellCheckerConfig
from myspellchecker.providers import MemoryProvider


# Since we don't have pytest-asyncio installed, we can run the event loop manually
def run_async(coro):
    return asyncio.run(coro)


def _create_test_checker():
    """Create a SpellChecker with sample Myanmar words for async tests."""
    provider = MemoryProvider()
    # Words that don't trigger grammar false positives
    # (avoid words with syllables that both start with "မ" which triggers "Double negation" rule)
    valid_words = ["ကျောင်းသား", "ကျောင်း", "သွားသည်"]
    syllables = ["ကျောင်း", "သား", "သွား", "သည်"]
    for w in valid_words:
        provider.add_word(w, frequency=100)
        provider.add_syllable(w, frequency=100)
    for s in syllables:
        provider.add_syllable(s, frequency=100)
        provider.add_word(s, frequency=100)
    config = SpellCheckerConfig(provider=provider, use_context_checker=False)
    return SpellChecker(config=config)


def test_check_async_basic():
    async def _test():
        with _create_test_checker() as checker:
            result = await checker.check_async("ကျောင်း", level=ValidationLevel.WORD)
            assert not result.has_errors
            assert result.text == "ကျောင်း"

    run_async(_test())


def test_check_async_with_error():
    async def _test():
        with SpellChecker() as checker:
            # "မြန်" is valid syllable but "မြန်စာ" invalid word
            result = await checker.check_async("မြန်စာ", level=ValidationLevel.WORD)
            assert result.has_errors
            assert len(result.errors) > 0

    run_async(_test())


def test_check_async_concurrency():
    async def _test():
        with _create_test_checker() as checker:
            # Valid words from provider + one with invalid syllable
            texts = ["ကျောင်းသား", "ကျောင်း", "သွားသည်", "ဆြ"]  # Last has invalid syllable

            tasks = [checker.check_async(text, level=ValidationLevel.WORD) for text in texts]
            results = await asyncio.gather(*tasks)

            assert len(results) == 4
            assert not results[0].has_errors
            assert not results[1].has_errors
            assert not results[2].has_errors
            assert results[3].has_errors

    run_async(_test())
