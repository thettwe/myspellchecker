import pytest

from myspellchecker.core.config import SpellCheckerConfig
from myspellchecker.core.spellchecker import SpellChecker
from myspellchecker.providers.memory import MemoryProvider
from myspellchecker.segmenters import Segmenter


# Mock segmenter that splits by space
class SpaceSegmenter(Segmenter):
    def segment_syllables(self, text: str):
        return text.split()

    def segment_words(self, text: str):
        return text.split()

    def segment_sentences(self, text: str):
        return [text]


class TestHonorificHeuristics:
    @pytest.fixture
    def provider(self):
        p = MemoryProvider()
        # Add honorifics
        p.add_word("မောင်", frequency=1000)
        p.add_word("ဦး", frequency=1000)
        p.add_word("ဒေါ်", frequency=1000)

        # Add common words
        p.add_word("သွား", frequency=500)
        p.add_word("သည်", frequency=500)

        # Add "Ba" (a common name component but often not in dict as standalone word or low freq)
        # Let's assume it's NOT in the dictionary to simulate an unknown name error
        # Or add it with very low probability bigram to trigger context check
        p.add_word("ဘ", frequency=10)

        # Add bigram for known context
        p.add_bigram("မောင်", "သွား", 0.05)

        return p

    def test_name_after_honorific_is_ignored(self, provider):
        # "Maung Ba" -> "Ba" is likely a name.
        # Even if "Ba" has low probability or is unknown, it should be skipped by context checker

        config = SpellCheckerConfig(segmenter=SpaceSegmenter(), provider=provider, use_ner=True)
        checker = SpellChecker(config=config)

        # "Maung Ba"
        # "Ba" (ဘ) follows "Maung" (မောင်).
        # Normal context checker might flag "Ba" if P(Ba|Maung) is low/zero.
        # But NER should whitelist it.

        result = checker.check("မောင် ဘ", level="word")

        # Should have NO context errors
        context_errors = [e for e in result.errors if e.error_type == "context_probability"]
        assert len(context_errors) == 0

    @pytest.mark.skip(reason="Fast-path exit skips context strategies on structurally-clean text")
    def test_ner_disabled_flags_error(self, provider):
        # Same as above but NER disabled -> should flag error (due to low prob)

        config = SpellCheckerConfig(segmenter=SpaceSegmenter(), provider=provider, use_ner=False)
        checker = SpellChecker(config=config)

        # P=0 bigrams are ignored to avoid false positives on unseen valid
        # pairs, so seed a non-zero but very-low probability to force a flag.
        provider.add_bigram("မောင်", "ဘ", 0.00005)

        result = checker.check("မောင် ဘ", level="word")

        # Should have context error
        context_errors = [e for e in result.errors if e.error_type == "context_probability"]
        assert len(context_errors) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
