"""Tests for loan word correction feature.

Covers the parser, config accessor, mixin check method, and engine
integration at priority 55.
"""

from unittest.mock import MagicMock

import pytest

from myspellchecker.grammar.config import GrammarRuleConfig, _grammar_config_singleton
from myspellchecker.grammar.engine import RulePriority, SyntacticRuleChecker
from myspellchecker.grammar.parsers.loan_word_parser import (
    parse_loan_word_corrections_config,
)

# --- Parser tests ---


class TestLoanWordParser:
    def test_basic_parsing(self):
        config = {
            "corrections": [
                {
                    "incorrect": "abc",
                    "correct": "def",
                    "source_language": "english",
                    "source_word": "test",
                    "error_type": "medial_confusion",
                    "confidence": 0.97,
                },
            ]
        }
        result: dict = {}
        parse_loan_word_corrections_config(config, loan_word_corrections=result)

        assert "abc" in result
        assert result["abc"]["correct"] == "def"
        assert result["abc"]["confidence"] == 0.97
        assert result["abc"]["source_language"] == "english"
        assert result["abc"]["source_word"] == "test"
        assert result["abc"]["error_type"] == "medial_confusion"

    def test_missing_incorrect_skipped(self):
        config = {
            "corrections": [
                {"correct": "def", "confidence": 0.9},
            ]
        }
        result: dict = {}
        parse_loan_word_corrections_config(config, loan_word_corrections=result)
        assert len(result) == 0

    def test_missing_correct_skipped(self):
        config = {
            "corrections": [
                {"incorrect": "abc", "confidence": 0.9},
            ]
        }
        result: dict = {}
        parse_loan_word_corrections_config(config, loan_word_corrections=result)
        assert len(result) == 0

    def test_empty_corrections(self):
        config = {"corrections": []}
        result: dict = {}
        parse_loan_word_corrections_config(config, loan_word_corrections=result)
        assert len(result) == 0

    def test_missing_corrections_key(self):
        config = {}
        result: dict = {}
        parse_loan_word_corrections_config(config, loan_word_corrections=result)
        assert len(result) == 0

    def test_defaults_applied(self):
        config = {
            "corrections": [
                {"incorrect": "abc", "correct": "def"},
            ]
        }
        result: dict = {}
        parse_loan_word_corrections_config(config, loan_word_corrections=result)

        assert result["abc"]["confidence"] == 0.95
        assert result["abc"]["error_type"] == "loan_word_misspelling"
        assert result["abc"]["source_language"] == ""
        assert result["abc"]["source_word"] == ""

    def test_malformed_entry_skipped(self):
        """Non-dict entry should be skipped gracefully."""
        config = {
            "corrections": [
                "not a dict",
                {"incorrect": "abc", "correct": "def"},
            ]
        }
        result: dict = {}
        parse_loan_word_corrections_config(config, loan_word_corrections=result)
        assert len(result) == 1
        assert "abc" in result

    def test_multiple_entries(self):
        config = {
            "corrections": [
                {"incorrect": "a", "correct": "b", "confidence": 0.9},
                {"incorrect": "c", "correct": "d", "confidence": 0.8},
            ]
        }
        result: dict = {}
        parse_loan_word_corrections_config(config, loan_word_corrections=result)
        assert len(result) == 2


# --- Config accessor tests ---


class TestConfigAccessor:
    def test_loan_word_corrections_loaded(self):
        _grammar_config_singleton._instances.clear()
        config = GrammarRuleConfig()
        assert len(config.loan_word_corrections) > 0

    def test_get_loan_word_correction_found(self):
        _grammar_config_singleton._instances.clear()
        config = GrammarRuleConfig()
        result = config.get_loan_word_correction("ကွန်ပြူတာ")
        assert result is not None
        assert result["correct"] == "ကွန်ပျူတာ"

    def test_get_loan_word_correction_not_found(self):
        _grammar_config_singleton._instances.clear()
        config = GrammarRuleConfig()
        result = config.get_loan_word_correction("nonexistent")
        assert result is None

    def test_correct_form_not_flagged(self):
        _grammar_config_singleton._instances.clear()
        config = GrammarRuleConfig()
        # The correct form should NOT be in the corrections dict
        assert config.get_loan_word_correction("ကွန်ပျူတာ") is None


# --- Mixin tests ---


class TestLoanWordMixin:
    def _make_checker(self):
        checker = SyntacticRuleChecker(MagicMock())
        return checker

    def test_check_loan_word_found(self):
        checker = self._make_checker()
        checker.config.get_loan_word_correction = MagicMock(
            return_value={
                "correct": "ကွန်ပျူတာ",
                "source_word": "computer",
                "source_language": "english",
                "confidence": 0.98,
            }
        )
        result = checker._check_loan_word_corrections("ကွန်ပြူတာ")
        assert result is not None
        assert result[0] == "ကွန်ပျူတာ"
        assert result[2] == 0.98
        assert "computer" in result[1]

    def test_check_loan_word_not_found(self):
        checker = self._make_checker()
        checker.config.get_loan_word_correction = MagicMock(return_value=None)
        result = checker._check_loan_word_corrections("valid_word")
        assert result is None

    def test_check_loan_word_no_source_word(self):
        checker = self._make_checker()
        checker.config.get_loan_word_correction = MagicMock(
            return_value={
                "correct": "abc",
                "source_word": "",
                "source_language": "",
                "confidence": 0.95,
            }
        )
        result = checker._check_loan_word_corrections("xyz")
        assert result is not None
        assert "Loan word correction" in result[1]


# --- Engine integration tests ---


class TestEngineIntegration:
    def test_loan_word_priority_constant(self):
        assert RulePriority.LOAN_WORD == 55

    def test_loan_word_priority_in_range(self):
        assert RulePriority.CONFIG_PATTERN < RulePriority.LOAN_WORD < RulePriority.POS_SEQUENCE

    def test_engine_detects_loan_word_error(self):
        """End-to-end: engine should detect and correct a loan word error."""
        provider = MagicMock()
        provider.get_word_pos = MagicMock(return_value=None)
        provider.get_word_frequency = MagicMock(return_value=0)

        checker = SyntacticRuleChecker(provider)
        corrections = checker.check_sequence(["ကွန်ပြူတာ"])
        # Should find the loan word correction
        loan_corrections = [c for c in corrections if c[2] == "ကွန်ပျူတာ"]
        assert len(loan_corrections) == 1
        assert loan_corrections[0][3] == pytest.approx(0.98)

    def test_engine_no_fp_on_correct_form(self):
        """Correct loan word form should not trigger any loan word correction."""
        provider = MagicMock()
        provider.get_word_pos = MagicMock(return_value=None)
        provider.get_word_frequency = MagicMock(return_value=0)

        checker = SyntacticRuleChecker(provider)
        corrections = checker.check_sequence(["ကွန်ပျူတာ"])
        loan_corrections = [c for c in corrections if c[2] == "ကွန်ပျူတာ"]
        assert len(loan_corrections) == 0
