"""Unit tests for Probe-5 (SymSpell near-match) in _probe_adjacent_merge.

Probe-5 is the seg-lever2-01 lever: when probes 1-4 fail, run SymSpell on the
merged string and accept the merge if a near-match (ed<=2, freq>=floor) exists.
Targets the 239 over-split FN per the 2026-04-19 candidate_not_generated audit
where merged differs from gold by ed<=2 (e.g. consonant substitution after
merge).

Workstream: segmenter-post-merge-rescue / task: seg-lever2-01.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from myspellchecker.algorithms import NgramContextChecker, SymSpell
from myspellchecker.algorithms.symspell import Suggestion
from myspellchecker.core.config import SpellCheckerConfig
from myspellchecker.core.validators import WordValidator
from myspellchecker.segmenters import Segmenter


class TestProbe5Symspell:
    """Probe-5: SymSpell on merged string with guards."""

    @pytest.fixture
    def validator(self) -> WordValidator:
        config = SpellCheckerConfig()
        config.validation.use_segmenter_post_merge_rescue = True
        config.validation.use_segmenter_merge_symspell_probe = True

        mock_segmenter = Mock(spec=Segmenter)
        mock_provider = Mock()
        mock_provider.get_word_frequency.return_value = 0
        mock_provider.is_valid_word.return_value = False
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

    def _make_suggestion(self, term: str, ed: int = 1, freq: int = 1000) -> Suggestion:
        return Suggestion(term=term, edit_distance=ed, frequency=freq)

    def test_probe5_fires_on_near_match(self, validator: WordValidator) -> None:
        """ed<=2 match above freq floor → merge accepted."""
        validator.symspell.lookup.return_value = [self._make_suggestion("ကွန်ပျူတာ", ed=1, freq=5000)]
        result = validator._merge_probe_adjacent_pairs(["ကွန်ပျု", "တာ"])
        assert result == ["ကွန်ပျုတာ"]

    def test_probe5_skipped_when_flag_off(self, validator: WordValidator) -> None:
        """Disabled by default — flag must be explicit."""
        validator.config.validation.use_segmenter_merge_symspell_probe = False
        validator.symspell.lookup.return_value = [self._make_suggestion("ကွန်ပျူတာ", ed=1, freq=5000)]
        result = validator._merge_probe_adjacent_pairs(["ကွန်ပျု", "တာ"])
        assert result == ["ကွန်ပျု", "တာ"]

    def test_probe5_rejects_low_frequency(self, validator: WordValidator) -> None:
        """Candidate below freq floor → no merge."""
        validator.symspell.lookup.return_value = [self._make_suggestion("ကွန်ပျူတာ", ed=1, freq=10)]
        result = validator._merge_probe_adjacent_pairs(["ကွန်ပျု", "တာ"])
        assert result == ["ကွန်ပျု", "တာ"]

    def test_probe5_rejects_high_edit_distance(self, validator: WordValidator) -> None:
        """ed > max_ed → no merge."""
        validator.symspell.lookup.return_value = [self._make_suggestion("ကွန်ပျူတာ", ed=3, freq=5000)]
        result = validator._merge_probe_adjacent_pairs(["ကွန်ပျု", "တာ"])
        assert result == ["ကွန်ပျု", "တာ"]

    def test_probe5_rejects_trivial_fragment_match(self, validator: WordValidator) -> None:
        """Top-1 equals a fragment → likely spurious, no merge."""
        validator.symspell.lookup.return_value = [self._make_suggestion("ကွန်ပျု", ed=0, freq=5000)]
        result = validator._merge_probe_adjacent_pairs(["ကွန်ပျု", "တာ"])
        assert result == ["ကွန်ပျု", "တာ"]

    def test_probe5_rejects_short_merged(self, validator: WordValidator) -> None:
        """Merged length below min_merged_len → no lookup."""
        validator.config.validation.segmenter_merge_symspell_min_merged_len = 10
        validator.symspell.lookup.return_value = [self._make_suggestion("ကွန်ပျူတာ", ed=1, freq=5000)]
        result = validator._merge_probe_adjacent_pairs(["ကွန်ပျု", "တာ"])
        assert result == ["ကွန်ပျု", "တာ"]

    def test_probe5_skipped_when_both_fragments_valid(self, validator: WordValidator) -> None:
        """Same OOV guard as Probe-2 — both valid → skip Probe-5."""
        dict_words = {"ကောင်း", "တယ်"}
        validator.word_repository.is_valid_word.side_effect = lambda w: w in dict_words
        validator.symspell.lookup.return_value = [
            self._make_suggestion("ကောင်းတယ်ပါ", ed=1, freq=5000)
        ]
        result = validator._merge_probe_adjacent_pairs(["ကောင်း", "တယ်"])
        assert result == ["ကောင်း", "တယ်"]

    def test_probe5_skipped_when_symspell_is_none(self, validator: WordValidator) -> None:
        """No SymSpell instance → Probe-5 silent-skips."""
        validator.symspell = None
        result = validator._merge_probe_adjacent_pairs(["ကွန်ပျု", "တာ"])
        assert result == ["ကွန်ပျု", "တာ"]

    def test_probe5_handles_lookup_exception(self, validator: WordValidator) -> None:
        """SymSpell errors must not crash the merge path."""
        validator.symspell.lookup.side_effect = RuntimeError("boom")
        result = validator._merge_probe_adjacent_pairs(["ကွန်ပျု", "တာ"])
        assert result == ["ကွန်ပျု", "တာ"]
