"""Tests for MergedWordChecker — detecting segmenter-merged particle+verb words."""

import pytest

from myspellchecker.core.config.grammar_configs import MergedWordCheckerConfig
from myspellchecker.core.constants import ErrorType
from myspellchecker.grammar.checkers.merged_word import (
    AMBIGUOUS_MERGES,
    CLAUSE_LINKING_MARKERS,
    NOUN_LIKE_TAGS,
    MergedWordChecker,
    MergedWordError,
)

# ===========================================================================
# MergedWordError dataclass
# ===========================================================================


class TestMergedWordError:
    """Tests for the MergedWordError dataclass."""

    def test_default_error_type(self):
        """Default error_type is 'merged_word'."""
        err = MergedWordError(
            text="ကစား",
            position=1,
            suggestions=["က စား"],
            confidence=0.8,
            reason="test",
        )
        assert err.error_type == ErrorType.MERGED_WORD.value
        assert err.error_type == "merged_word"

    def test_decomposition_default(self):
        """Default decomposition is empty tuple."""
        err = MergedWordError(
            text="ကစား",
            position=1,
            suggestions=["က စား"],
            confidence=0.8,
            reason="test",
        )
        assert err.decomposition == ("", "")

    def test_decomposition_set(self):
        """Decomposition can be explicitly set."""
        err = MergedWordError(
            text="ကစား",
            position=1,
            suggestions=["က စား"],
            confidence=0.8,
            reason="test",
            decomposition=("က", "စား"),
        )
        assert err.decomposition == ("က", "စား")

    def test_str_representation(self):
        """__str__ shows position and decomposition."""
        err = MergedWordError(
            text="ကစား",
            position=2,
            suggestions=["က စား"],
            confidence=0.8,
            reason="test",
            decomposition=("က", "စား"),
        )
        s = str(err)
        assert "pos=2" in s
        assert "ကစား" in s
        assert "က စား" in s


# ===========================================================================
# MergedWordChecker initialization
# ===========================================================================


class TestMergedWordCheckerInit:
    """Tests for MergedWordChecker initialization."""

    def test_default_confidence(self):
        """Default confidence comes from MergedWordCheckerConfig."""
        checker = MergedWordChecker()
        assert checker.confidence == 0.80

    def test_explicit_confidence(self):
        """Explicit confidence overrides config default."""
        checker = MergedWordChecker(confidence=0.9)
        assert checker.confidence == 0.9

    def test_config_override(self):
        """checker_config controls default when confidence is not passed."""
        config = MergedWordCheckerConfig(default_confidence=0.7)
        checker = MergedWordChecker(checker_config=config)
        assert checker.confidence == 0.7

    def test_explicit_confidence_beats_config(self):
        """Explicit confidence takes precedence over config."""
        config = MergedWordCheckerConfig(default_confidence=0.7)
        checker = MergedWordChecker(confidence=0.95, checker_config=config)
        assert checker.confidence == 0.95

    def test_merge_rules_loaded(self):
        """merge_rules is populated from AMBIGUOUS_MERGES."""
        checker = MergedWordChecker()
        assert checker.merge_rules is AMBIGUOUS_MERGES
        assert "ကစား" in checker.merge_rules


# ===========================================================================
# validate_sequence: empty / no-op cases
# ===========================================================================


class TestValidateSequenceEmpty:
    """Tests for validate_sequence returning empty results."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.checker = MergedWordChecker()

    def test_empty_words(self):
        """Empty word list returns no errors."""
        assert self.checker.validate_sequence([], []) == []

    def test_none_pos_tags(self):
        """None POS tags returns no errors."""
        assert self.checker.validate_sequence(["a", "b"], None) == []

    def test_mismatched_lengths(self):
        """Mismatched words and pos_tags returns no errors."""
        assert self.checker.validate_sequence(["a", "b"], ["N"]) == []

    def test_no_ambiguous_words(self):
        """Sentence without ambiguous merge words returns no errors."""
        words = ["သူ", "စာ", "ဖတ်", "တယ်"]
        pos = ["PRON", "N", "V", "PART"]
        assert self.checker.validate_sequence(words, pos) == []

    def test_ambiguous_word_at_position_zero(self):
        """Ambiguous word at position 0 has no preceding word, so no error."""
        words = ["ကစား", "သောကြောင့်"]
        pos = ["V", "PART"]
        assert self.checker.validate_sequence(words, pos) == []

    def test_ambiguous_word_at_end_no_following(self):
        """Ambiguous word at sentence end has no following word, so no error."""
        words = ["သူ", "ကစား"]
        pos = ["PRON", "V"]
        assert self.checker.validate_sequence(words, pos) == []

    def test_prev_not_noun_like(self):
        """Preceding word is not noun-like (e.g., verb) — no error."""
        words = ["သွား", "ကစား", "သောကြောင့်"]
        pos = ["V", "V", "PART"]
        assert self.checker.validate_sequence(words, pos) == []

    def test_next_not_clause_linking(self):
        """Following word is not a clause-linking marker — no error."""
        words = ["သူ", "ကစား", "တယ်"]
        pos = ["PRON", "V", "PART"]
        # "တယ်" is a sentence-final particle, not a clause-linking marker
        assert self.checker.validate_sequence(words, pos) == []

    def test_prev_pos_none(self):
        """Preceding word has None POS tag — no error."""
        words = ["သူ", "ကစား", "သောကြောင့်"]
        pos = [None, "V", "PART"]
        assert self.checker.validate_sequence(words, pos) == []


# ===========================================================================
# validate_sequence: detection cases
# ===========================================================================


class TestValidateSequenceDetection:
    """Tests for validate_sequence successfully detecting merged words."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.checker = MergedWordChecker()

    def test_classic_merge_detection(self):
        """
        Classic case: PRON + ကစား + clause-linker.

        သူ ကစား သောကြောင့် → should flag ကစား as possible "က" + "စား"
        """
        words = ["သူ", "ကစား", "သောကြောင့်"]
        pos = ["PRON", "V", "PART"]
        errors = self.checker.validate_sequence(words, pos)

        assert len(errors) == 1
        err = errors[0]
        assert isinstance(err, MergedWordError)
        assert err.text == "ကစား"
        assert err.position == 1
        assert err.decomposition == ("က", "စား")
        assert "က စား" in err.suggestions
        assert err.error_type == "merged_word"
        assert err.confidence == 0.80

    def test_noun_pos_triggers_detection(self):
        """N (noun) POS tag for preceding word also triggers detection."""
        words = ["ကလေး", "ကစား", "လို့"]
        pos = ["N", "V", "PART"]
        errors = self.checker.validate_sequence(words, pos)
        # "ကလေး" (child) is N, "ကစား" is ambiguous, "လို့" is clause-linking
        assert len(errors) == 1
        assert errors[0].text == "ကစား"

    def test_np_pos_triggers_detection(self):
        """NP (noun phrase) POS tag triggers detection."""
        words = ["something", "ကစား", "ကြောင့်"]
        pos = ["NP", "V", "PART"]
        errors = self.checker.validate_sequence(words, pos)
        assert len(errors) == 1

    def test_noun_tag_variant(self):
        """NOUN POS tag variant also triggers detection."""
        words = ["something", "ကစား", "ရင်"]
        pos = ["NOUN", "V", "PART"]
        errors = self.checker.validate_sequence(words, pos)
        assert len(errors) == 1

    def test_pipe_separated_pos_with_noun(self):
        """Pipe-separated POS like 'N|V' containing N triggers detection."""
        words = ["ambig", "ကစား", "သဖြင့်"]
        pos = ["N|V", "V", "PART"]
        errors = self.checker.validate_sequence(words, pos)
        assert len(errors) == 1

    def test_all_clause_linking_markers(self):
        """Each clause-linking marker triggers detection when other conditions hold."""
        for marker in CLAUSE_LINKING_MARKERS:
            words = ["သူ", "ကစား", marker]
            pos = ["PRON", "V", "PART"]
            errors = self.checker.validate_sequence(words, pos)
            assert len(errors) == 1, f"Expected error for marker: {marker}"
            assert errors[0].text == "ကစား"

    def test_error_reason_includes_context(self):
        """Error reason includes the preceding word and decomposition description."""
        words = ["သူ", "ကစား", "ကြောင့်"]
        pos = ["PRON", "V", "PART"]
        errors = self.checker.validate_sequence(words, pos)

        assert len(errors) == 1
        reason = errors[0].reason
        assert "သူ" in reason
        assert "ကစား" in reason
        assert "က" in reason
        assert "စား" in reason

    def test_custom_confidence(self):
        """Custom confidence is applied to detected errors."""
        checker = MergedWordChecker(confidence=0.60)
        words = ["သူ", "ကစား", "သောကြောင့်"]
        pos = ["PRON", "V", "PART"]
        errors = checker.validate_sequence(words, pos)

        assert len(errors) == 1
        assert errors[0].confidence == 0.60


# ===========================================================================
# _has_noun_like_tag
# ===========================================================================


class TestHasNounLikeTag:
    """Tests for the _has_noun_like_tag helper method."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.checker = MergedWordChecker()

    def test_single_noun_tag(self):
        assert self.checker._has_noun_like_tag("N") is True

    def test_single_pron_tag(self):
        assert self.checker._has_noun_like_tag("PRON") is True

    def test_single_np_tag(self):
        assert self.checker._has_noun_like_tag("NP") is True

    def test_single_noun_full_tag(self):
        assert self.checker._has_noun_like_tag("NOUN") is True

    def test_pipe_separated_with_noun(self):
        assert self.checker._has_noun_like_tag("N|V") is True

    def test_pipe_separated_without_noun(self):
        assert self.checker._has_noun_like_tag("V|ADJ") is False

    def test_verb_only(self):
        assert self.checker._has_noun_like_tag("V") is False

    def test_empty_string(self):
        assert self.checker._has_noun_like_tag("") is False


# ===========================================================================
# Constants validation
# ===========================================================================


class TestConstants:
    """Validate the module-level constants."""

    def test_ambiguous_merges_structure(self):
        """Each AMBIGUOUS_MERGES entry has (particle, verb, type, description)."""
        for word, rule in AMBIGUOUS_MERGES.items():
            assert isinstance(word, str)
            assert len(rule) == 4
            particle, verb, ptype, desc = rule
            assert isinstance(particle, str)
            assert isinstance(verb, str)
            assert isinstance(ptype, str)
            assert isinstance(desc, str)

    def test_noun_like_tags_is_frozenset(self):
        assert isinstance(NOUN_LIKE_TAGS, frozenset)
        assert "N" in NOUN_LIKE_TAGS
        assert "PRON" in NOUN_LIKE_TAGS

    def test_clause_linking_markers_is_frozenset(self):
        assert isinstance(CLAUSE_LINKING_MARKERS, frozenset)
        assert "သောကြောင့်" in CLAUSE_LINKING_MARKERS
        assert "လို့" in CLAUSE_LINKING_MARKERS

    def test_sentence_final_particles_excluded(self):
        """Sentence-final particles like တယ်, ပါတယ်, သည် should NOT be included."""
        assert "တယ်" not in CLAUSE_LINKING_MARKERS
        assert "ပါတယ်" not in CLAUSE_LINKING_MARKERS
        assert "သည်" not in CLAUSE_LINKING_MARKERS
