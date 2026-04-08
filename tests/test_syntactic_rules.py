from myspellchecker.core.config import SpellCheckerConfig
from myspellchecker.core.config.validation_configs import ValidationConfig
from myspellchecker.core.constants import ValidationLevel
from myspellchecker.core.spellchecker import SpellChecker
from myspellchecker.grammar.engine import SyntacticRuleChecker
from myspellchecker.providers import DictionaryProvider


class MockProvider(DictionaryProvider):
    def __init__(self):
        self.pos_map = {}

    def is_valid_syllable(self, syllable: str) -> bool:
        return True

    def is_valid_word(self, word: str) -> bool:
        return True

    def get_syllable_frequency(self, syllable: str) -> int:
        return 100

    def get_word_frequency(self, word: str) -> int:
        return 100

    def get_bigram_probability(self, w1: str, w2: str) -> float:
        return 0.1

    def get_trigram_probability(self, w1: str, w2: str, w3: str) -> float:
        return 0.1

    def get_fourgram_probability(self, w1: str, w2: str, w3: str, w4: str) -> float:
        return 0.0

    def get_fivegram_probability(self, w1: str, w2: str, w3: str, w4: str, w5: str) -> float:
        return 0.0

    def get_top_continuations(self, w1: str, limit: int = 20) -> list:
        return []

    def get_all_syllables(self):
        yield from []

    def get_all_words(self):
        yield from []

    def get_word_pos(self, word: str):
        return self.pos_map.get(word)

    def set_pos(self, word, pos):
        self.pos_map[word] = pos

    def get_pos_unigram_probabilities(self):
        return {}

    def get_pos_bigram_probabilities(self):
        return {}

    def get_pos_trigram_probabilities(self):
        return {}


def test_syntactic_checker_init():
    provider = MockProvider()
    checker = SyntacticRuleChecker(provider)
    assert checker is not None


def test_verb_particle_agreement():
    provider = MockProvider()
    checker = SyntacticRuleChecker(provider)

    # Setup: "သွား" (Go - V), "ကျောင်း" (School - N)
    # Rule: Verb followed by "ကျောင်း" should suggest "ကြောင်း"
    provider.set_pos("သွား", "V")
    provider.set_pos("ကျောင်း", "N")

    # Correct sequence: "သွားကြောင်း" (That he went - correct particle usage)
    # Input: "သွား" + "ကျောင်း" (Go School? No, likely "Go That" -> "သွားကြောင်း")

    words = ["သူ", "သွား", "ကျောင်း"]
    corrections = checker.check_sequence(words)

    assert len(corrections) == 1
    idx, error_word, suggestion, _confidence = corrections[0]
    assert error_word == "ကျောင်း"
    assert suggestion == "ကြောင်း"
    # idx is 2 (0=သူ, 1=သွား, 2=ကျောင်း)
    assert idx == 2


def test_typo_particle_ma():
    provider = MockProvider()
    checker = SyntacticRuleChecker(provider)

    provider.set_pos("အိမ်", "N")

    # Input: "အိမ် မာ" (House Hard? -> House At)
    words = ["အိမ်", "မာ"]
    corrections = checker.check_sequence(words)

    assert len(corrections) == 1
    assert corrections[0][1] == "မာ"
    assert corrections[0][2] == "မှာ"


def test_typo_particle_ga():
    provider = MockProvider()
    checker = SyntacticRuleChecker(provider)

    provider.set_pos("အိမ်", "N")

    # Input: "အိမ် ဂ" (House Ga? -> House Ka/From)
    words = ["အိမ်", "ဂ"]
    corrections = checker.check_sequence(words)

    assert len(corrections) == 1
    assert corrections[0][1] == "ဂ"
    assert corrections[0][2] == "က"


def test_spellchecker_integration():
    # Integration test with SpellChecker
    provider = MockProvider()
    provider.set_pos("သွား", "V")
    provider.set_pos("ကျောင်း", "N")

    # Lower the syntax_error gate for this test (default is 1.0 = disabled
    # because syntax_error has zero TPs on the production benchmark).
    thresholds = ValidationConfig().output_confidence_thresholds.copy()
    thresholds["syntax_error"] = 0.0
    config = SpellCheckerConfig(
        provider=provider,
        use_context_checker=True,
        validation=ValidationConfig(
            output_confidence_thresholds=thresholds,
            use_candidate_fusion=False,
        ),
    )
    checker = SpellChecker(config=config)

    # Mock segmenter to ensure consistent word segmentation for this test
    # This avoids issues with the default segmenter's behavior in test environment
    from unittest.mock import MagicMock

    checker.segmenter.segment_words = MagicMock(return_value=["သူ", "သွား", "ကျောင်း"])
    checker.segmenter.segment_sentences = MagicMock(return_value=["သူ သွား ကျောင်း"])

    # Only "context" layer (WORD) runs syntactic checks
    # "သူ သွား ကျောင်း" -> "သူ သွား ကြောင်း"

    # Check segmentation first to ensure words are separated
    words = checker.segmenter.segment_words("သူ သွား ကျောင်း")
    # Clean up spaces/punctuation for verification
    clean_words = [w for w in words if w.strip()]

    # Ensure we have the sequence we expect
    assert "သွား" in clean_words
    assert "ကျောင်း" in clean_words

    # Let's verify simple case
    result = checker.check("သူ သွား ကျောင်း", level=ValidationLevel.WORD)

    # Should have at least one error
    assert result.has_errors

    # Find the context error
    errors = [e for e in result.errors if e.error_type == "syntax_error"]
    assert len(errors) > 0
    assert errors[0].text == "ကျောင်း"
    assert "ကြောင်း" in errors[0].suggestions
