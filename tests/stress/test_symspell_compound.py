import time

import pytest

from myspellchecker.algorithms.symspell import SymSpell
from myspellchecker.providers.memory import MemoryProvider

pytestmark = pytest.mark.slow


@pytest.fixture
def provider():
    p = MemoryProvider()
    # Add some common words to create ambiguity
    words = ["က", "ကက", "ကကက", "ခ", "ခခ", "ဂ", "ဂဂ"]
    for w in words:
        p.add_word(w, frequency=100)
    return p


def test_compound_explosion(provider):
    symspell = SymSpell(provider)

    # Create a long string of "က"
    # "က" can be segmented as "က", "က", "က" or "ကက", "က" etc.
    # With length 50, the number of combinations is huge (Fibonacci-like)
    long_text = "က" * 50

    start_time = time.time()
    # This might hang if DP is not pruned
    suggestions = symspell.lookup_compound(long_text, max_suggestions=1)
    end_time = time.time()

    # It shouldn't take too long (e.g., > 1s is suspicious for length 50)
    assert end_time - start_time < 2.0
    assert len(suggestions) > 0


def test_compound_long_string(provider):
    symspell = SymSpell(provider)
    long_text = "က" * 200  # Even longer

    start_time = time.time()
    try:
        _ = symspell.lookup_compound(long_text, max_suggestions=1)
    except RecursionError:
        pytest.fail("RecursionError detected")
    except Exception as e:
        # Check if it's OOM or Timeout related
        pytest.fail(f"Exception: {e}")

    end_time = time.time()

    assert end_time - start_time < 5.0
