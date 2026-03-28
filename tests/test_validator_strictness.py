from myspellchecker.core.syllable_rules import SyllableRuleValidator


def test_validator_strictness_tone_count():
    validator = SyllableRuleValidator(strict=True)

    # Double Dots (Detected in database as "ကြ့ည့်")
    assert not validator.validate("ကြ့ည့်"), "Should reject double tone marks"

    # Valid word "ကြည့်" (Ka+Ra+Nya+Dot)
    # Use explicit unicode to avoid normalization issues
    # \u1000 (Ka) \u103c (Ra) \u100a (Nya) \u1037 (Dot)
    assert validator.validate("\u1000\u103c\u100a\u1037"), "Should accept valid word with one tone"


def test_validator_strictness_tone_position():
    validator = SyllableRuleValidator(strict=True)

    # Tone before Final Consonant (e.g., Dot before Nya)
    # Construct synthetic bad example: "က" + "့" + "န်"
    assert not validator.validate("ကံ့န်"), "Should reject tone mark before final consonant"

    # Valid position: Ka(1000) Na(1014) Asat(103A) Dot(1037)
    assert validator.validate("\u1000\u1014\u103a\u1037"), (
        "Should accept tone mark after final consonant"
    )


def test_validator_strictness_char_scope():
    validator = SyllableRuleValidator(strict=True)

    # Rare extension character (U+1055 in "ဟန်ဵ")
    # "Ha" (101F) "Na" (1014) "Asat" (103A) "SS" (1055)
    assert not validator.validate("\u101f\u1014\u103a\u1055"), (
        "Should reject characters outside core range"
    )

    # Valid complex word "ကျောင်း" (Kyaung:)
    assert validator.validate("ကျောင်း"), "Should accept valid complex word"


def test_validator_strictness_non_standard_final():
    validator = SyllableRuleValidator(strict=True)

    # "နအ်" - Na + Ah + Asat.
    # Ah (Independent Vowel carrier) usually doesn't take Asat in standard spelling.
    assert not validator.validate("နအ်"), "Should reject Independent Vowel + Asat"


def test_validator_strictness_diacritic_spam():
    validator = SyllableRuleValidator(strict=True)

    # Two 'Ya-pins' (Detected by existing rules? Let's check strictness)
    assert not validator.validate("ကျျ"), "Should reject duplicate medials"

    # Two 'Aa' vowels
    assert not validator.validate("ကာာ"), "Should reject duplicate vowels"


if __name__ == "__main__":
    test_validator_strictness_tone_count()
    test_validator_strictness_tone_position()
    test_validator_strictness_char_scope()
    test_validator_strictness_diacritic_spam()
