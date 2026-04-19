"""Unit tests for WordValidator._merge_probe_adjacent_pairs cascade behavior.

Regression guard for seg-left-cascade-fix (2026-04-20): the pre-fix implementation
cascaded merges only rightward, so a 3-fragment chunk like
``['စွမ်းဆောင်', 'ရ', 'ည']`` where only the right pair initially matched a probe
would produce ``['စွမ်းဆောင်', 'ရည']`` instead of the correct ``['စွမ်းဆောင်ရည']``.

Workstream: segmenter-post-merge-rescue.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from myspellchecker.algorithms import NgramContextChecker, SymSpell
from myspellchecker.core.config import SpellCheckerConfig
from myspellchecker.core.validators import WordValidator
from myspellchecker.segmenters import Segmenter


class TestMergeProbeCascade:
    """Pop-and-retry cascade in _merge_probe_adjacent_pairs."""

    @pytest.fixture
    def validator(self) -> WordValidator:
        config = SpellCheckerConfig()
        config.validation.use_segmenter_post_merge_rescue = True

        mock_segmenter = Mock(spec=Segmenter)
        mock_provider = Mock()
        mock_provider.get_word_frequency.return_value = 0
        mock_symspell = Mock(spec=SymSpell)
        mock_context_checker = Mock(spec=NgramContextChecker)

        return WordValidator(
            config,
            mock_segmenter,
            mock_provider,
            mock_provider,
            mock_symspell,
            mock_context_checker,
        )

    def test_three_fragment_left_cascade(self, validator: WordValidator) -> None:
        """Right pair merges first, then left neighbour is absorbed via Probe-3."""
        dict_words = {"ရည်", "စွမ်းဆောင်ရည်"}
        validator.word_repository.is_valid_word.side_effect = lambda w: w in dict_words

        result = validator._merge_probe_adjacent_pairs(["စွမ်းဆောင်", "ရ", "ည"])
        assert result == ["စွမ်းဆောင်ရည"]

    def test_does_not_merge_two_valid_independent_words(self, validator: WordValidator) -> None:
        """Probe-2 guard: both fragments dict-valid → no merge."""
        dict_words = {"ကောင်း", "တယ်", "ကောင်းတယ်"}
        validator.word_repository.is_valid_word.side_effect = lambda w: w in dict_words

        result = validator._merge_probe_adjacent_pairs(["ကောင်း", "တယ်"])
        assert result == ["ကောင်း", "တယ်"]

    def test_unrelated_adjacent_tokens_untouched(self, validator: WordValidator) -> None:
        """No probe fires → output equals input."""
        validator.word_repository.is_valid_word.return_value = False
        tokens = ["သံ", "ဂါ", "နေ"]
        result = validator._merge_probe_adjacent_pairs(tokens)
        assert result == tokens

    def test_two_fragment_asat_probe_still_fires(self, validator: WordValidator) -> None:
        """Baseline: Probe-3 (merged + asat in dict) on a 2-fragment input."""
        dict_words = {"ရည်"}
        validator.word_repository.is_valid_word.side_effect = lambda w: w in dict_words
        result = validator._merge_probe_adjacent_pairs(["ရ", "ည"])
        assert result == ["ရည"]

    def test_short_input_returned_unchanged(self, validator: WordValidator) -> None:
        """Lists of length < 2 bypass the probe."""
        assert validator._merge_probe_adjacent_pairs([]) == []
        assert validator._merge_probe_adjacent_pairs(["solo"]) == ["solo"]

    def test_empty_and_whitespace_tokens_skip_probe(self, validator: WordValidator) -> None:
        """Empty / whitespace tokens are not considered for merging."""
        validator.word_repository.is_valid_word.return_value = True
        # Even though is_valid_word is True for everything, empty token
        # should not participate in a merge.
        result = validator._merge_probe_adjacent_pairs(["foo", "", "bar"])
        assert result == ["foo", "", "bar"]
