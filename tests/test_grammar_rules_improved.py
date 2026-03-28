"""
Tests for improved Grammar Rules Validator.

Tests the YAML configuration system, verb-particle agreement,
sentence structure validation, and medial confusion detection.
"""

import pytest

from myspellchecker.grammar.config import GrammarRuleConfig
from myspellchecker.grammar.engine import SyntacticRuleChecker
from myspellchecker.providers.memory import MemoryProvider


class TestGrammarRuleConfig:
    """Test the YAML-based grammar rule configuration."""

    @pytest.fixture
    def config(self):
        """Load default grammar config."""
        return GrammarRuleConfig()

    def test_config_loads_particle_typos(self, config):
        """Test that particle typos are loaded from config."""
        assert len(config.particle_typos) > 0
        # Check for known particle typo
        assert "မာ" in config.particle_typos

    def test_config_loads_medial_confusions(self, config):
        """Test that medial confusions are loaded."""
        assert len(config.medial_confusions) > 0
        # Check for known medial confusion (dict keyed by pattern)
        assert "ကျောင်း" in config.medial_confusions

    def test_config_loads_verb_particles(self, config):
        """Test that verb particles are loaded."""
        assert len(config.verb_particles) > 0
        # Check for known verb particles
        assert "ခဲ့" in config.verb_particles
        assert "မယ်" in config.verb_particles
        assert "တယ်" in config.verb_particles

    def test_config_loads_noun_particles(self, config):
        """Test that noun particles are loaded."""
        assert len(config.noun_particles) > 0
        # Check for known noun particles
        assert "က" in config.noun_particles
        assert "ကို" in config.noun_particles
        assert "မှာ" in config.noun_particles

    def test_config_loads_sentence_final_particles(self, config):
        """Test that sentence-final particles are loaded."""
        assert len(config.sentence_final_particles) > 0
        # Check for known sentence-final particles
        assert "တယ်" in config.sentence_final_particles
        assert "ပါတယ်" in config.sentence_final_particles
        assert "လား" in config.sentence_final_particles

    def test_config_loads_invalid_pos_sequences(self, config):
        """Test that invalid POS sequences are loaded."""
        # Check for loaded rules
        sequences = [r["sequence"] for r in config.invalid_pos_sequences]
        assert "V-V" in sequences
        assert "P-P" in sequences
        assert "N-N" in sequences

        # Check P-P severity downgraded
        pp_rule = next(r for r in config.invalid_pos_sequences if r["sequence"] == "P-P")
        assert pp_rule["severity"] == "info"

        # Check for new fine-grained rule
        if "P_SENT-P_SENT" in sequences:
            sent_rule = next(
                r for r in config.invalid_pos_sequences if r["sequence"] == "P_SENT-P_SENT"
            )
            assert sent_rule["severity"] == "error"

    def test_get_particle_typo(self, config):
        """Test particle typo lookup."""
        info = config.get_particle_typo("မာ")
        assert info is not None
        assert info.get("correction") == "မှာ"

    def test_get_medial_confusion(self, config):
        """Test medial confusion lookup."""
        info = config.get_medial_confusion("ကျောင်း")
        assert info is not None
        assert info.get("correction") == "ကြောင်း"

    def test_is_verb_particle(self, config):
        """Test verb particle check."""
        assert config.is_verb_particle("ခဲ့") is True
        assert config.is_verb_particle("xyz") is False

    def test_is_noun_particle(self, config):
        """Test noun particle check."""
        assert config.is_noun_particle("က") is True
        assert config.is_noun_particle("xyz") is False

    def test_is_sentence_final(self, config):
        """Test sentence-final particle check."""
        assert config.is_sentence_final("တယ်") is True
        assert config.is_sentence_final("xyz") is False

    def test_get_invalid_sequence_error(self, config):
        """Test invalid sequence error lookup."""
        error = config.get_invalid_sequence_error("V", "V")
        assert error is not None
        assert error.get("severity") == "info"  # Changed from error to info for SVCs

        # Test non-existent sequence
        assert config.get_invalid_sequence_error("N", "V") is None


class TestMedialConfusionDetection:
    """Test medial confusion detection functionality."""

    @pytest.fixture
    def checker(self):
        """Create a checker with memory provider."""
        provider = MemoryProvider()
        # Add test words with frequencies
        provider.add_word("ငါ", 1000)
        provider.add_word("သွား", 1000)
        provider.add_word("ကျောင်း", 1000)
        provider.add_word("ကြောင်း", 1000)
        provider.add_word("ကျေးဇူး", 1000)
        # Add POS tags separately
        provider.add_word_pos("ငါ", "N")
        provider.add_word_pos("သွား", "V")
        provider.add_word_pos("ကျောင်း", "N")
        provider.add_word_pos("ကြောင်း", "P")
        provider.add_word_pos("ကျေးဇူး", "N")
        return SyntacticRuleChecker(provider)

    def test_medial_confusion_after_verb(self, checker):
        """Test ya-pin/ya-yit confusion after verb."""
        # ကျောင်း → ကြောင်း after verb
        errors = checker.check_sequence(["သွား", "ကျောင်း"])
        assert len(errors) == 1
        assert errors[0][2] == "ကြောင်း"

    def test_medial_confusion_after_noun_ok(self, checker):
        """Test that ကျောင်း (school) is not flagged after noun."""
        # ကျောင်း is valid after noun (meaning "school")
        errors = checker.check_sequence(["ငါ", "ကျောင်း"])
        # Should not flag as error after noun
        school_errors = [e for e in errors if e[1] == "ကျောင်း" and e[2] == "ကြောင်း"]
        assert len(school_errors) == 0


class TestWordCorrections:
    """Test common word corrections."""

    @pytest.fixture
    def checker(self):
        """Create a checker with memory provider."""
        provider = MemoryProvider()
        provider.add_word("ကြေးဇူး", 100)
        provider.add_word("ကျေးဇူး", 1000)
        provider.add_word_pos("ကြေးဇူး", "N")
        provider.add_word_pos("ကျေးဇူး", "N")
        return SyntacticRuleChecker(provider)

    def test_thanks_correction(self, checker):
        """Test ကြေးဇူး → ကျေးဇူး correction."""
        errors = checker.check_sequence(["ကြေးဇူး"])
        assert len(errors) == 1
        assert errors[0][2] == "ကျေးဇူး"


class TestMissingAsatCorrections:
    """Test missing asat/tone mark corrections."""

    @pytest.fixture
    def checker(self):
        """Create a checker with memory provider."""
        provider = MemoryProvider()
        provider.add_word("ပါတယ", 100)
        provider.add_word("ပါတယ်", 1000)
        provider.add_word("တယ", 100)
        provider.add_word("တယ်", 1000)
        provider.add_word("သည", 100)
        provider.add_word("သည်", 1000)
        provider.add_word("မယ", 100)
        provider.add_word("မယ်", 1000)
        provider.add_word_pos("ပါတယ", "P")
        provider.add_word_pos("ပါတယ်", "P")
        provider.add_word_pos("တယ", "P")
        provider.add_word_pos("တယ်", "P")
        provider.add_word_pos("သည", "P")
        provider.add_word_pos("သည်", "P")
        provider.add_word_pos("မယ", "P")
        provider.add_word_pos("မယ်", "P")
        return SyntacticRuleChecker(provider)

    def test_missing_asat_patae(self, checker):
        """Test ပါတယ → ပါတယ် correction."""
        errors = checker.check_sequence(["ပါတယ"])
        assert len(errors) == 1
        assert errors[0][2] == "ပါတယ်"

    def test_missing_asat_tae(self, checker):
        """Test တယ → တယ် correction."""
        errors = checker.check_sequence(["တယ"])
        assert len(errors) == 1
        assert errors[0][2] == "တယ်"

    def test_missing_asat_thii(self, checker):
        """Test သည → သည် correction."""
        errors = checker.check_sequence(["သည"])
        assert len(errors) == 1
        assert errors[0][2] == "သည်"

    def test_missing_asat_mae(self, checker):
        """Test မယ → မယ် correction."""
        errors = checker.check_sequence(["မယ"])
        assert len(errors) == 1
        assert errors[0][2] == "မယ်"


class TestConfigIntegration:
    """Test that config is properly integrated into the checker."""

    @pytest.fixture
    def checker(self):
        """Create a checker with memory provider."""
        provider = MemoryProvider()
        provider.add_word("မာ", 100)
        provider.add_word("မှာ", 1000)
        provider.add_word("စာ", 1000)
        provider.add_word_pos("မာ", "P")
        provider.add_word_pos("မှာ", "P")
        provider.add_word_pos("စာ", "N")
        return SyntacticRuleChecker(provider)

    def test_checker_uses_config_particle_typos(self, checker):
        """Test that checker uses config for particle typo detection."""
        # Verify config is loaded
        assert checker.config is not None
        assert len(checker.config.particle_typos) > 0

    def test_checker_uses_config_verb_particles(self, checker):
        """Test that checker uses config for verb particle validation."""
        assert len(checker.config.verb_particles) > 0
        assert "ခဲ့" in checker.config.verb_particles

    def test_checker_uses_config_noun_particles(self, checker):
        """Test that checker uses config for noun particle validation."""
        assert len(checker.config.noun_particles) > 0
        assert "က" in checker.config.noun_particles
