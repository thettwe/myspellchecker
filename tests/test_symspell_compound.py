from unittest.mock import MagicMock

import pytest

from myspellchecker.algorithms.symspell import Suggestion, SymSpell


class TestSymSpellCompoundCoverage:
    @pytest.fixture
    def symspell(self):
        provider = MagicMock()

        # Mock word validity and frequency
        def is_valid(w):
            return w in ["valid", "compound", "word"]

        def get_freq(w):
            return 100 if w in ["valid", "compound", "word"] else 0

        provider.is_valid_word.side_effect = is_valid
        provider.get_word_frequency.side_effect = get_freq

        return SymSpell(provider, max_word_length=10)

    def test_lookup_compound_empty(self, symspell):
        assert symspell.lookup_compound("") == []
        assert symspell.lookup_compound("   ") == []

    def test_lookup_compound_single_valid_word(self, symspell):
        # Should identify "valid" as valid word and return it
        res = symspell.lookup_compound("valid")
        assert len(res) > 0
        assert res[0][0] == ["valid"]
        assert res[0][1] == 0  # distance 0

    def test_lookup_compound_split_correct(self, symspell):
        # "validword" -> "valid", "word"
        res = symspell.lookup_compound("validword")
        # Expect top result to be split
        assert ["valid", "word"] in [r[0] for r in res]

    def test_lookup_compound_with_typo(self, symspell):
        # "validx" -> "valid" (distance 1)
        # Mock lookup to find "valid" for "validx"
        suggestion = Suggestion("valid", 1, 100)

        # We need to mock the internal lookup call which _segment_compound uses
        # But lookup calls self.provider...
        # Let's try to let it run naturally if possible, but we need "valid" to be
        # returned for "validx"
        # Since we didn't build index, regular lookup fails.
        # We should mock self.lookup

        symspell.lookup = MagicMock()

        def lookup_side_effect(term, **kwargs):
            if term == "validx":
                return [suggestion]
            return []

        symspell.lookup.side_effect = lookup_side_effect

        res = symspell.lookup_compound("validx")

        # Should find "valid"
        assert res[0][0] == ["valid"]
        assert res[0][1] == 1

    def test_segment_compound_beam_search_pruning(self, symspell):
        # Force beam width exceeded
        symspell.beam_width = 1
        # We need ambiguity. "compoundword" -> "compound", "word" OR "com", "pound", "word" etc.
        # If we mock is_valid_word to be True for many substrings, we generate many paths.

        symspell.provider.is_valid_word.return_value = True
        symspell.provider.get_word_frequency.return_value = 1

        res = symspell.lookup_compound("abc")
        assert len(res) > 0

    def test_lookup_compound_no_valid_segmentation(self, symspell):
        # Input that cannot be split into valid words and no corrections found
        symspell.provider.is_valid_word.return_value = False
        symspell.lookup = MagicMock(return_value=[])  # No corrections

        res = symspell.lookup_compound("unknown")
        # Should return original as fallback
        assert res[0][0] == ["unknown"]
        assert res[0][1] == 0  # Although logic might say distance 0, it's just a fallback.
