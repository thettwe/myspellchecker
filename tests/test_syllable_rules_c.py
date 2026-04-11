import time

import pytest

from myspellchecker.core.syllable_rules import SyllableRuleValidator as PyValidator

try:
    from myspellchecker.core.syllable_rules_c import SyllableRuleValidator as CValidator

    HAS_CYTHON = True
except ImportError:
    HAS_CYTHON = False


@pytest.mark.skipif(not HAS_CYTHON, reason="Cython module not built")
def test_cython_validator_correctness():
    py_val = PyValidator(strict=True)
    c_val = CValidator(strict=True)

    test_cases = [
        # Ming (Kinzi) - actually Ma+Nga+Asat+Virama+Ga?
        ("Valid", "\u1019\u1004\u103a\u1039\u1002"),
        # Mingalar is 3 syllables. "Ming" "Ga" "Lar".
        # Ma + i + Nga + Asat + Virama + Ga ...
        # Wait, Kinzi is Nga+Asat+Virama. It sits on the NEXT char.
        # So "Min" + "Ga" -> Ming-Ga.
        # Syllable 1: Ma + I + Nga(Kinzi)? No.
        # Let's use simple valid syllables.
        ("Valid_Ka", "\u1000"),
        ("Valid_Kyaung:", "\u1000\u103b\u1031\u102c\u1004\u103a\u1038"),  # Kyaung:
        ("Invalid_DoubleTone", "\u1000\u103c\u1037\u100a\u1037"),  # Double Dot
        ("Valid_Kyi_Exception", "\u1000\u103c\u100a\u1037"),  # Kyi (Look)
        ("Invalid_Scope", "\u101f\u1014\u103a\u1055"),  # Han + SS (1055)
        ("Invalid_Pos", "\u1000\u1037\u1014\u103a"),  # Ka + Dot + Na + Asat
        ("Valid_Kant", "\u1000\u1014\u103a\u1037"),  # Ka + Na + Asat + Dot
        ("Invalid_NonStd", "\u1014\u1021\u103a"),  # Na + Ah + Asat
        ("Invalid_Diacritic", "\u1000\u103b\u103b"),  # Ka + Ya + Ya
        ("Valid_Myan", "\u1019\u103c\u1014\u103a"),  # Myan
        ("Valid_Mar", "\u1019\u102c"),  # Mar
        ("Empty", ""),
    ]

    for name, syl in test_cases:
        py_res = py_val.validate(syl)
        c_res = c_val.validate(syl)
        assert py_res == c_res, f"Mismatch for '{name}': Py={py_res}, C={c_res}"


@pytest.mark.skipif(not HAS_CYTHON, reason="Cython module not built")
def test_cython_validator_benchmark():
    py_val = PyValidator(strict=True)
    c_val = CValidator(strict=True)

    # Generate large dataset
    base_syls = [
        "\u1000\u103b\u1031\u102c\u1004\u103a\u1038",  # Valid
        "\u1000\u103c\u100a\u1037",  # Valid
        "\u1000\u103c\u1037\u100a\u1037",  # Invalid
        "\u1019\u103c\u1014\u103a",  # Valid
        "\u101f\u1014\u103a\u1055",  # Invalid
    ]
    syllables = base_syls * 20000  # 100,000 syllables

    # Python
    start = time.time()
    for s in syllables:
        py_val.validate(s)
    py_time = time.time() - start

    # Cython
    start = time.time()
    for s in syllables:
        c_val.validate(s)
    c_time = time.time() - start

    # Cython should generally be faster, but don't hard-fail on timing
    # (CI machines and warm-up can cause variance)
    speedup = py_time / c_time if c_time > 0 else 0
    assert speedup > 0.5, (
        f"Cython ({c_time:.4f}s) unexpectedly much slower than Python ({py_time:.4f}s)"
    )


if __name__ == "__main__":
    if HAS_CYTHON:
        test_cython_validator_correctness()
        test_cython_validator_benchmark()
