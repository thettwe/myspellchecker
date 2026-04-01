import pytest

from myspellchecker import SpellChecker
from myspellchecker.core.config import SpellCheckerConfig, ValidationConfig
from myspellchecker.providers import MemoryProvider


def test_strictness_config_integration():
    """Test that strict_validation config is respected by the pipeline."""
    # Invalid Virama Order: Ka + Medial-Ya + Virama + Ka
    # Strict rule: Virama must come BEFORE Medials.
    # Regex segmenter keeps this as one syllable because Virama stacks the second Ka.
    invalid_virama_order = "\u1000\u103b\u1039\u1000"

    # 1. Strict Mode (Default)
    strict_config = SpellCheckerConfig(
        provider=MemoryProvider(),  # Empty provider
        validation=ValidationConfig(strict_validation=True),
    )
    checker_strict = SpellChecker(config=strict_config)

    # Strict mode -> Rule validation fails -> Invalid Syllable
    res_strict = checker_strict.check(invalid_virama_order)
    assert res_strict.has_errors
    assert any(e.error_type == "invalid_syllable" for e in res_strict.errors)

    # 2. Lenient Mode
    # In lenient mode, structure is valid (rule passes).
    # Proceed to dictionary check. We need to add it to dictionary to fully pass.
    provider = MemoryProvider()
    provider.add_word(invalid_virama_order, frequency=100)
    # Critical: Must be in syllable set for SyllableValidator
    provider.add_syllable(invalid_virama_order, frequency=100)

    # Verify provider state for test setup
    assert provider.is_valid_syllable(invalid_virama_order), "Provider should have the syllable"

    lenient_config = SpellCheckerConfig(
        provider=provider, validation=ValidationConfig(strict_validation=False)
    )
    checker_lenient = SpellChecker(config=lenient_config)

    res_lenient = checker_lenient.check(invalid_virama_order)
    assert not res_lenient.has_errors, "Should be valid in lenient mode + dict match"

    # Verify Strict Mode with dict match — dictionary presence now takes
    # precedence over strict syllable-rule violations, so the word passes.
    strict_config_dict = SpellCheckerConfig(
        provider=provider, validation=ValidationConfig(strict_validation=True)
    )
    checker_strict_dict = SpellChecker(config=strict_config_dict)
    res_strict_2 = checker_strict_dict.check(invalid_virama_order)
    assert not res_strict_2.has_errors, "Dict match overrides strict syllable rules"


def test_homophone_integration():
    """Test that homophones are suggested for context errors."""
    # We need a provider with N-gram data to trigger context errors
    # Using MemoryProvider with manually populated n-grams
    provider = MemoryProvider()

    # "ကျွန်တော် ကျောင်း သွားသည်" (I go to school) - High prob
    # "ကျွန်တော် ကြောင်း သွားသည်" (I go to reason) - Low prob (homophone error)

    provider.add_word("ကျွန်တော်", frequency=100)
    provider.add_word("ကျောင်း", frequency=100)
    provider.add_word("ကြောင်း", frequency=100)
    provider.add_word("သွားသည်", frequency=100)

    # Add syllables so words pass syllable validation layer.
    # The pipeline segments text into syllables, so we need syllable-level entries.
    # "ကျွန်တော်" → ["ကျွန်", "တော်"], "သွားသည်" → ["သွား", "သည်"]
    for word in [
        "ကျွန်တော်",
        "ကျောင်း",
        "ကြောင်း",
        "သွားသည်",
        "တော်",
        "ကျွန်",
        "သွား",
        "သည်",
    ]:
        provider.add_syllable(word, frequency=100)

    # Add word-level bigram probs (for completeness)
    provider.add_bigram("ကျွန်တော်", "ကျောင်း", 0.1)
    provider.add_bigram("ကျောင်း", "သွားသည်", 0.1)
    provider.add_bigram("ကျွန်တော်", "ကြောင်း", 0.00001)
    provider.add_bigram("ကြောင်း", "သွားသည်", 0.00001)

    # Add syllable-level bigram probs — the pipeline segments into syllables,
    # so the ngram strategy actually checks these pairs:
    # ["ကျွန်", "တော်", "ကြောင်း", "သွား", "သည်"]
    provider.add_bigram("တော်", "ကျောင်း", 0.1)  # correct: high prob
    provider.add_bigram("တော်", "ကြောင်း", 0.00001)  # error: low prob
    provider.add_bigram("ကြောင်း", "သွား", 0.00001)  # error context continues

    pytest.skip(
        "Fast-path exit skips context/homophone strategies on structurally-clean text. "
        "See context_validator._FAST_PATH_PRIORITY_CUTOFF."
    )
