from unittest.mock import MagicMock

import pytest

from myspellchecker import SpellChecker
from myspellchecker.core.config import SpellCheckerConfig
from myspellchecker.providers import DictionaryProvider


class TestSpellCheckerPosTagging:
    @pytest.fixture(autouse=True)
    def setup(self):
        # Mock the provider to return specific POS tags and bigram probabilities
        self.mock_provider = MagicMock(spec=DictionaryProvider)
        self.mock_provider.get_word_pos.side_effect = lambda word: {
            "ကျွန်တော်": "PRO",
            "ကျောင်း": "N",
            "သွား": "V",
            "သည်": "P",
            "စား": "V",
            "ငါး": "N",
            "နိုင်": "V",
            "သည့်": "P",
            "လုပ်": "V|N",  # Ambiguous word
            "လုပ်ကိုင်": "V",
        }.get(word, None)

        self.mock_provider.get_pos_unigram_probabilities.return_value = {
            "PRO": 0.1,
            "N": 0.2,
            "V": 0.3,
            "P": 0.15,
        }
        self.mock_provider.get_pos_bigram_probabilities.return_value = {
            ("PRO", "N"): 0.3,
            ("N", "V"): 0.4,
            ("V", "P"): 0.8,  # Strong transition for Verb ending sentence
            ("PRO", "V"): 0.7,
            ("V", "N"): 0.1,
            ("N", "N"): 0.2,
            ("N", "P"): 0.1,
        }
        # Ensure trigram probs returns proper dict (required by ViterbiTagger)
        self.mock_provider.get_pos_trigram_probabilities.return_value = {}
        # Ensure syllable/word methods return proper types for SymSpell
        self.mock_provider.get_all_syllables.return_value = []
        self.mock_provider.get_all_words.return_value = []

        # Use a minimal config for the SpellChecker
        from myspellchecker.core.config import POSTaggerConfig

        self.config = SpellCheckerConfig(
            provider=self.mock_provider,
            use_context_checker=True,  # Ensure context checker is enabled to load Viterbi
            pos_tagger=POSTaggerConfig(tagger_type="viterbi", unknown_tag="UNK"),
        )
        self.spell_checker = SpellChecker(config=self.config)

    def test_get_pos_tags_unambiguous_sentence(self):
        """Test POS tagging for an unambiguous sentence."""
        # Use pre-segmented words to avoid dependency on word segmenter
        words = ["ကျွန်တော်", "ကျောင်း", "သွား", "သည်"]
        expected_tags = ["PRO", "N", "V", "P"]
        tags = self.spell_checker.get_pos_tags(words=words)
        assert tags == expected_tags

    def test_get_pos_tags_ambiguous_word_resolution(self):
        """Test POS tagging for a sentence with an ambiguous word, resolved by context."""
        # "ကျွန်တော်" (PRO) "လုပ်" (N/V) "သည်" (P)
        # PRO -> V -> P is likely
        # PRO -> N -> P is less likely
        words = ["ကျွန်တော်", "လုပ်", "သည်"]
        expected_tags = ["PRO", "V", "P"]  # "လုပ်" should be V
        tags = self.spell_checker.get_pos_tags(words=words)
        assert tags == expected_tags

    def test_get_pos_tags_unknown_word(self):
        """Test POS tagging for a sentence containing known and unknown words."""
        # Use pre-segmented words to avoid dependency on word segmenter
        words = ["ကျွန်တော်", "မသိသော", "စကားလုံး"]
        # "ကျွန်တော်" -> PRO (from mock), "မသိသော" -> UNK (not in mock), "စကားလုံး" -> UNK
        tags = self.spell_checker.get_pos_tags(words=words)
        assert len(tags) == 3
        assert tags[0] == "PRO"  # Known word

    def test_get_pos_tags_empty_text(self):
        """Test POS tagging for empty text."""
        text = ""
        tags = self.spell_checker.get_pos_tags(text)
        assert tags == []
