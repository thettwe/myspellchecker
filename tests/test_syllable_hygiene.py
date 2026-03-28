from myspellchecker.core.syllable_rules import SyllableRuleValidator


def test_syllable_hygiene_one_final():
    validator = SyllableRuleValidator(strict=True)

    # Double Final: Anusvara + Consonant+Asat
    # "ကံန်း" (Ka + Anusvara + Na + Asat + Visarga)
    assert not validator.validate("ကံန်း"), "Should reject Anusvara + Final"

    # Valid: Just Anusvara
    assert validator.validate("ကံ"), "Should accept single Anusvara"

    # Valid: Just Final
    assert validator.validate("ကန်း"), "Should accept single Final"


if __name__ == "__main__":
    test_syllable_hygiene_one_final()
