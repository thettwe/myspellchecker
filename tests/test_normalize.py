from myspellchecker.text.normalize import normalize


def test_kinzi_ordering():
    # Kinzi: Nga + Asat + Virama (1004 + 103A + 1039)
    # Input: Nga + Virama + Asat (Wrong order)
    text = "\u1004\u1039\u103a"
    # Expected: Nga + Asat + Virama (Canonical for Kinzi)
    # Asat (31) < Virama (99/Implicit)
    normalized = normalize(text)
    assert normalized == "\u1004\u103a\u1039"


def test_virama_explicit_handling():
    # If we have Upper + Virama + Asat (Invalid? No, Kinzi is Asat+Virama)
    # But suppose we have Upper + Tone + Virama (Invalid structurally, but let's check sort)
    # Tone (33 Visarga) vs Virama
    text = "\u1000\u1039\u1038"  # Ka + Virama + Visarga
    # Visarga (33) < Virama (99)
    # Result: Ka + Visarga + Virama
    normalized = normalize(text)
    assert normalized == "\u1000\u1038\u1039"

    # Wait, is "Ka + Visarga + Virama" canonical?
    # Structurally, a stacked consonant usually doesn't have Visarga on the Upper.
    # But if it did, where does it go?
    # This is an edge case.


def test_medial_ordering():
    # Ka + Wa + Ya (Wrong: Ya < Wa)
    # Ka + 103D + 103B
    text = "\u1000\u103d\u103b"
    # Expected: Ka + Ya + Wa (1000 + 103B + 103D)
    normalized = normalize(text)
    assert normalized == "\u1000\u103b\u103d"


def test_vowel_ordering():
    # Ka + Aa (102C) + E (1031)
    # Visual: E-Ka-Aa (Aw vowel)
    # Storage: Ka + E + Aa
    # E (20) < Aa (23)
    text = "\u1000\u102c\u1031"
    normalized = normalize(text)
    assert normalized == "\u1000\u1031\u102c"
