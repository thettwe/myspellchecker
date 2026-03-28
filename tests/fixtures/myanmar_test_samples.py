"""
Test samples for Myanmar text segmentation and spell checking.

These samples include various Myanmar text examples with known syllable
and word boundaries for testing and validation.
"""

from typing import Dict, List, Tuple


class MyanmarTestSamples:
    """Collection of Myanmar text test samples with ground truth segmentations."""

    # Format: (text, expected_syllables, expected_words, description)
    SAMPLES: List[Tuple[str, List[str], List[str], str]] = [
        # Basic examples
        (
            "မြန်မာ",
            ["မြန်", "မာ"],
            ["မြန်မာ"],
            "Myanmar (country name) - common word",
        ),
        (
            "နိုင်ငံ",
            ["နိုင်", "ငံ"],
            ["နိုင်ငံ"],
            "Country - common word",
        ),
        (
            "မြန်မာနိုင်ငံ",
            ["မြန်", "မာ", "နိုင်", "ငံ"],
            ["မြန်မာ", "နိုင်ငံ"],
            "Myanmar country - compound phrase",
        ),
        # Sentence with punctuation
        (
            "သူသွားသည်။",
            ["သူ", "သွား", "သည်", "။"],
            ["သူ", "သွား", "သည်", "။"],
            "He goes - simple sentence with period",
        ),
        # Complex diacritics
        (
            "ကျောင်းသား",
            ["ကျောင်း", "သား"],
            ["ကျောင်းသား"],
            "Student - word with medial consonants",
        ),
        # Numbers and mixed content
        (
            "၁၂၃",
            ["၁", "၂", "၃"],
            ["၁၂၃"],
            "Numbers 123 in Myanmar numerals",
        ),
        # Longer sentence
        (
            "သူမနေကောင်းပါဘူး",
            ["သူ", "မ", "နေ", "ကောင်း", "ပါ", "ဘူး"],
            ["သူ", "မ", "နေကောင်း", "ပါ", "ဘူး"],
            "She is not well - negation sentence",
        ),
        # Kinzi and special characters
        (
            "င်္ဂလိပ်",
            ["င်္ဂ", "လိပ်"],
            ["င်္ဂလိပ်"],
            "English - word with kinzi",
        ),
        # Tone marks
        (
            "ထမင်း",
            ["ထ", "မင်း"],
            ["ထမင်း"],
            "Rice - common word with tone mark",
        ),
        # Short syllables
        (
            "အိမ်",
            ["အိမ်"],
            ["အိမ်"],
            "House - single syllable word",
        ),
        # Question marker
        (
            "ဘယ်လောက်လဲ",
            ["ဘယ်", "လောက်", "လဲ"],
            ["ဘယ်လောက်", "လဲ"],
            "How much? - question with marker",
        ),
        # Medial combinations
        (
            "ကျွန်တော်",
            ["ကျွန်", "တော်"],
            ["ကျွန်တော်"],
            "I (male polite) - pronoun with medials",
        ),
        # Stacked consonants
        (
            "စက်",
            ["စက်"],
            ["စက်"],
            "Machine - single syllable with asat",
        ),
    ]

    @classmethod
    def get_texts(cls) -> List[str]:
        """Return just the text strings."""
        return [sample[0] for sample in cls.SAMPLES]

    @classmethod
    def get_syllable_ground_truth(cls) -> Dict[str, List[str]]:
        """Return mapping of texts to expected syllables."""
        return {sample[0]: sample[1] for sample in cls.SAMPLES}
