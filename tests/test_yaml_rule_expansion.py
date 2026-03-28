"""Consolidated tests for YAML rule expansion — classifier, grammar, sentence
segmentation, tone rules, and typo corrections."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from myspellchecker.grammar.checkers.classifier import ClassifierChecker
from myspellchecker.grammar.engine import SyntacticRuleChecker
from myspellchecker.segmenters.default import DefaultSegmenter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_word_pos(word: str) -> str:
    """Return generic POS tags for grammar rule tests."""
    if word in ["ရဲ့", "က", "ကို", "မှာ", "သည်", "တယ်", "မယ်", "ပြီ"]:
        return "P"
    if word in ["စာအုပ်", "လူ"]:
        return "N"
    if word in ["သွား", "စား"]:
        return "V"
    return "UNK"


# ---------------------------------------------------------------------------
# Classifier Agreement
# ---------------------------------------------------------------------------


class TestClassifierAgreementExpansion:
    """Verify expanded classifier agreement rules."""

    @pytest.fixture
    def checker(self):
        return ClassifierChecker()

    def test_vehicle_agreement_error(self, checker):
        error = checker.check_agreement("ယောက်", "ကား")
        assert error is not None, "Expected agreement error for 'ကား ... ယောက်'"
        assert error.suggestion in ["စင်း", "စီး"], (
            f"Expected suggestion 'စင်း' or 'စီး', got {error.suggestion}"
        )

    def test_correct_vehicle_usage(self, checker):
        error = checker.check_agreement("စီး", "ကား")
        assert error is None, "Should not flag correct usage 'ကား ... စီး'"

    def test_flat_object_agreement(self, checker):
        error = checker.check_agreement("လုံး", "စက္ကူ")
        assert error is not None
        assert error.suggestion in ["ချပ်", "ရွက်"]

    def test_leaf_classifier_valid(self, checker):
        error = checker.check_agreement("ရွက်", "စက္ကူ")
        assert error is None


# ---------------------------------------------------------------------------
# Grammar Rules
# ---------------------------------------------------------------------------


class TestGrammarRulesExpansion:
    """Verify grammar_rules.yaml expansion and engine rule loading."""

    @pytest.fixture
    def checker(self):
        mock_provider = MagicMock()
        mock_provider.get_word_pos.side_effect = _mock_word_pos
        return SyntacticRuleChecker(mock_provider)

    def test_particle_tags_loaded(self, checker):
        assert hasattr(checker.config, "particle_tags")
        assert checker.config.particle_tags.get("ရဲ့") == "P_POSS"
        assert checker.config.particle_tags.get("က") == "P_SUBJ"

    def test_poss_subj_invalid_sequence(self, checker):
        err = checker.config.get_invalid_sequence_error("P_POSS", "P_SUBJ")
        assert err is not None
        assert "Possessive particle" in err["message"]

    def test_subj_obj_warning_sequence(self, checker):
        err = checker.config.get_invalid_sequence_error("P_SUBJ", "P_OBJ")
        assert err is not None
        assert "adjacent" in err["message"]
        assert err["severity"] == "warning"

    def test_double_sentence_ending(self, checker):
        err = checker.config.get_invalid_sequence_error("P_SENT", "P_SENT")
        assert err is not None
        assert "Double sentence ending" in err["message"]


# ---------------------------------------------------------------------------
# Sentence Segmentation
# ---------------------------------------------------------------------------


class TestSentenceSegmentationHeuristics:
    """Test improved sentence segmentation using SFPs."""

    @pytest.fixture
    def segmenter(self):
        return DefaultSegmenter(word_engine="myword")

    def test_standard_separator(self, segmenter):
        text = "ဒါက ပထမ ဝါကျ။ ဒါက ဒုတိယ ဝါကျ။"
        sents = segmenter.segment_sentences(text)
        assert len(sents) == 2
        assert sents[0] == "ဒါက ပထမ ဝါကျ။"
        assert sents[1] == "ဒါက ဒုတိယ ဝါကျ။"

    def test_sfp_plus_space(self, segmenter):
        text = "သူ သွားမယ် သူ လာမယ်"
        sents = segmenter.segment_sentences(text)
        assert len(sents) == 2
        assert sents[0] == "သူ သွားမယ်"
        assert sents[1] == "သူ လာမယ်"

    def test_sfp_plus_newline(self, segmenter):
        text = "နေကောင်းလား\nကောင်းပါတယ်"
        sents = segmenter.segment_sentences(text)
        assert len(sents) == 2
        assert sents[0] == "နေကောင်းလား"
        assert sents[1] == "ကောင်းပါတယ်"

    def test_sfp_embedded_no_split(self, segmenter):
        text = "သွားမယ်လို့ပြောတယ်"
        sents = segmenter.segment_sentences(text)
        assert len(sents) == 1
        assert sents[0] == "သွားမယ်လို့ပြောတယ်"

    def test_mixed_separators(self, segmenter):
        text = "ဟုတ်တယ်။ ဒါပေမယ့် မသွားဘူး။"
        sents = segmenter.segment_sentences(text)
        assert len(sents) == 2
        assert sents[0] == "ဟုတ်တယ်။"
        assert sents[1] == "ဒါပေမယ့် မသွားဘူး။"


# ---------------------------------------------------------------------------
# Tone Rules
# ---------------------------------------------------------------------------


class TestToneRulesExpansion:
    """Verify tone_rules.yaml expansion and schema."""

    @pytest.fixture
    def tone_data(self):
        rules_path = Path("src/myspellchecker/rules/tone_rules.yaml")
        assert rules_path.exists(), "tone_rules.yaml not found"
        with open(rules_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def test_ambiguous_words_key_exists(self, tone_data):
        assert "ambiguous_words" in tone_data

    def test_minimum_entry_count(self, tone_data):
        count = len(tone_data["ambiguous_words"])
        assert count >= 40, f"Expected at least 40 entries, found {count}"

    def test_entry_schema(self, tone_data):
        for word, contexts in tone_data["ambiguous_words"].items():
            assert isinstance(word, str), f"Key {word} must be string"
            assert isinstance(contexts, dict), f"Value for {word} must be dict"
            for context_name, rule in contexts.items():
                assert "patterns" in rule, f"Rule {word}.{context_name} missing 'patterns'"
                assert isinstance(rule["patterns"], list), (
                    f"Rule {word}.{context_name}.patterns must be list"
                )
                assert "correct_form" in rule, f"Rule {word}.{context_name} missing 'correct_form'"
                assert "meaning" in rule, f"Rule {word}.{context_name} missing 'meaning'"

    @pytest.mark.parametrize(
        "word",
        ["မှ", "မှား", "စ", "စား", "ခ", "ခါ", "ခါး", "ပ", "ပါ", "သူ", "သူ့"],
    )
    def test_spot_check_entries(self, tone_data, word):
        assert word in tone_data["ambiguous_words"]


# ---------------------------------------------------------------------------
# Typo Corrections
# ---------------------------------------------------------------------------


class TestTypoExpansion:
    """Verify typo_corrections.yaml visual_errors are loaded and detected."""

    @pytest.fixture
    def checker(self):
        mock_provider = MagicMock()
        mock_provider.get_word_pos.return_value = "UNK"
        return SyntacticRuleChecker(mock_provider)

    def test_word_correction_loaded(self, checker):
        assert "ဥာဏ်" in checker.config.word_corrections
        assert checker.config.word_corrections["ဥာဏ်"]["correction"] == "ဉာဏ်"

    def test_correction_detected_in_sequence(self, checker):
        words = ["သူ", "ဥာဏ်", "ကောင်း", "တယ်"]
        corrections = checker.check_sequence(words)
        found = any(
            word == "ဥာဏ်" and suggestion == "ဉာဏ်" for _idx, word, suggestion, _conf in corrections
        )
        assert found, "Did not find correction for 'ဥာဏ်'"

    def test_particle_typo_loaded(self, checker):
        assert "ဂ" in checker.config.particle_typos
        assert checker.config.particle_typos["ဂ"]["correction"] == "က"
