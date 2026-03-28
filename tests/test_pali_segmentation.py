import pytest

from myspellchecker.core.syllable_rules import SyllableRuleValidator
from myspellchecker.segmenters.regex import RegexSegmenter


@pytest.fixture
def segmenter():
    return RegexSegmenter()


@pytest.fixture
def validator():
    return SyllableRuleValidator()


def test_pali_kinzi_segmentation(segmenter):
    # သင်္ဘော (Thin-Baw) - Kinzi case
    # Storage: သ + င + ် + ္ + ဘ + ေ + ာ
    # Kinzi coda (သင်္) is merged with next segment to produce a valid syllable
    text = "သင်္ဘော"
    syllables = segmenter.segment_syllables(text)
    assert syllables == ["သင်္ဘော"]


def test_pali_kinzi_multi_syllable(segmenter):
    # အင်္ဂလိပ် (English) - Kinzi + multiple syllables
    # Storage: အ + င + ် + ္ + ဂ + လ + ိ + ပ + ်
    # Kinzi coda (အင်္) merges with next segment (ဂ), then လိပ် remains separate
    text = "အင်္ဂလိပ်"
    syllables = segmenter.segment_syllables(text)
    assert syllables == ["အင်္ဂ", "လိပ်"]


def test_pali_kinzi_other_words(segmenter):
    # Common Kinzi words
    assert segmenter.segment_syllables("သင်္ကေတ") == ["သင်္ကေ", "တ"]
    assert segmenter.segment_syllables("ကင်္ဘာ") == ["ကင်္ဘာ"]


def test_pali_stacked_consonants(segmenter):
    # မေတ္တာ (Metta)
    text = "မေတ္တာ"
    syllables = segmenter.segment_syllables(text)
    assert syllables == ["မေ", "တ္တာ"]

    # သတ္တု (That-Tu)
    text = "သတ္တု"
    syllables = segmenter.segment_syllables(text)
    assert syllables == ["သ", "တ္တု"]

    # ဝိဇ္ဇာ (Pyin-Nya / Vijja)
    text = "ဝိဇ္ဇာ"
    syllables = segmenter.segment_syllables(text)
    assert syllables == ["ဝိ", "ဇ္ဇာ"]


def test_complex_medials(segmenter):
    # ကျွန်တော် (I)
    text = "ကျွန်တော်"
    syllables = segmenter.segment_syllables(text)
    assert syllables == ["ကျွန်", "တော်"]


def test_pali_loan_without_stacking(segmenter):
    # ဗုဒ္ဓ (Buddha)

    text = "ဗုဒ္ဓ"

    syllables = segmenter.segment_syllables(text)

    # Orthographic split: ဗု and ဒ္ဓ

    assert syllables == ["ဗု", "ဒ္ဓ"]


def test_multiple_syllables_pali(segmenter):
    # အာနန္ဒာ (Ananda)

    text = "အာနန္ဒာ"

    syllables = segmenter.segment_syllables(text)

    # Orthographic split: အာ, န, န္ဒာ

    assert syllables == ["အာ", "န", "န္ဒာ"]


# --- Kinzi validation tests (issues #1360, #1361, #1362) ---


class TestKinziValidation:
    """Merged Kinzi segments must pass SyllableRuleValidator."""

    @pytest.mark.parametrize(
        "word,expected_segments",
        [
            ("သင်္ဘော", ["သင်္ဘော"]),  # ship
            ("အင်္ဂလိပ်", ["အင်္ဂ", "လိပ်"]),  # English
            ("သင်္ကေတ", ["သင်္ကေ", "တ"]),  # symbol
            ("သင်္ချိုင်း", ["သင်္ချိုင်း"]),  # graveyard (11 codepoints)
            ("အင်္ဂါ", ["အင်္ဂါ"]),  # Tuesday
            ("သင်္ဂဏန်း", ["သင်္ဂ", "ဏန်း"]),  # mathematics
            ("အင်္ကျီ", ["အင်္ကျီ"]),  # shirt (Kinzi + medial)
            ("ကင်္ဘာ", ["ကင်္ဘာ"]),
            ("သင်္ဂြိုဟ်", ["သင်္ဂြိုဟ်"]),  # planet (Kinzi + medial)
        ],
    )
    def test_kinzi_segments_pass_validation(self, word, expected_segments, segmenter, validator):
        """Each segment of a Kinzi word must be individually valid."""
        syllables = segmenter.segment_syllables(word)
        assert syllables == expected_segments
        for seg in syllables:
            assert validator.validate(seg), f"Segment {seg!r} from {word!r} failed validation"

    def test_max_syllable_length_accommodates_kinzi(self, validator):
        """Default max_syllable_length must handle merged Kinzi segments."""
        # သင်္ချိုင်း = 11 codepoints, should not be rejected as corruption
        assert validator.validate("သင်္ချိုင်း")

    def test_kinzi_asat_not_rejected_before_medials(self, validator):
        """Asat in Kinzi sequence should not trigger pre-medial diacritic rejection."""
        # အင်္ကျီ has Asat at pos 2 (Kinzi) before Medial Ya at pos 5
        assert validator.validate("အင်္ကျီ")
        # သင်္ဂြိုဟ် has Asat at pos 2 (Kinzi) before Medial Ra at pos 5
        assert validator.validate("သင်္ဂြိုဟ်")

    def test_non_initial_kinzi_medial_base_resolution(self, validator):
        """Medial base must resolve to consonant after Kinzi, not first consonant."""
        # အင်္ကျီ: medial base should be က (Ka), not အ
        # If base were အ, Medial Ya would be incompatible
        assert validator.validate("အင်္ကျီ")

    def test_stacked_consonants_unaffected(self, validator):
        """Non-Kinzi stacked consonants must still validate correctly."""
        assert validator.validate("တ္တာ")
        assert validator.validate("က္က")
        assert validator.validate("ဒ္ဓ")
        assert validator.validate("န္ဒာ")
