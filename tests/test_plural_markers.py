"""
Tests for plural markers များ (formal) and တွေ (colloquial)
in morphology configuration for morphological analysis.

Note: Plural markers are stored in src/myspellchecker/rules/morphology.yaml
after the Phase 6 configuration migration.
"""

from pathlib import Path

import pytest
import yaml

# Path to morphology YAML file
MORPHOLOGY_YAML_PATH = (
    Path(__file__).parent.parent / "src" / "myspellchecker" / "rules" / "morphology.yaml"
)


@pytest.fixture(scope="module")
def morphology_data():
    """Load morphology YAML data."""
    with open(MORPHOLOGY_YAML_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def noun_suffixes(morphology_data):
    """Extract noun suffixes from morphology data."""
    suffixes = morphology_data.get("suffixes", {})
    noun_suffixes_data = suffixes.get("noun_suffixes", [])
    return [entry["suffix"] for entry in noun_suffixes_data]


class TestPluralMarkersInSuffixes:
    """Test that plural markers are included in noun suffixes."""

    def test_plural_markers_in_noun_suffixes(self, noun_suffixes):
        """Both plural markers should be in noun suffixes."""
        assert "များ" in noun_suffixes, "များ should be recognized as noun suffix"
        assert "တွေ" in noun_suffixes, "တွေ should be recognized as noun suffix"

    def test_plural_suffix_pattern_matching(self, noun_suffixes):
        """Test that plural markers can be matched as suffixes in words."""
        test_words = ["စာအုပ်များ", "လူတွေ"]
        for word in test_words:
            found_suffix = any(word.endswith(suffix) for suffix in noun_suffixes)
            assert found_suffix, f"Should find suffix for {word}"


class TestSuffixSetIntegrity:
    """Test overall suffix set integrity."""

    def test_suffix_count_reasonable(self, noun_suffixes):
        """Should have at least 17 noun suffixes (including plurals)."""
        assert len(noun_suffixes) >= 17, (
            f"Should have at least 17 suffixes, got {len(noun_suffixes)}"
        )

    def test_no_duplicate_suffixes(self, noun_suffixes):
        """No duplicates should exist in suffix list."""
        assert len(noun_suffixes) == len(set(noun_suffixes)), "No duplicates in suffixes"


class TestGrammarPatternsNounSuffixes:
    """Test NOUN_SUFFIXES constant in grammar/patterns.py."""

    def test_plural_markers_in_grammar_patterns(self):
        """များ and တွေ should be in grammar.patterns.NOUN_SUFFIXES."""
        from myspellchecker.grammar.patterns import NOUN_SUFFIXES

        assert "များ" in NOUN_SUFFIXES
        assert "တွေ" in NOUN_SUFFIXES

    def test_core_constants_exports_noun_suffixes(self):
        """core.constants should export NOUN_SUFFIXES with plural markers."""
        from myspellchecker.core.constants import NOUN_SUFFIXES

        assert len(NOUN_SUFFIXES) > 0
        assert "များ" in NOUN_SUFFIXES
        assert "တွေ" in NOUN_SUFFIXES

    def test_noun_suffixes_consistency(self):
        """grammar.patterns and core.constants NOUN_SUFFIXES should have same plural markers."""
        from myspellchecker.core.constants import NOUN_SUFFIXES as CORE_SUFFIXES
        from myspellchecker.grammar.patterns import NOUN_SUFFIXES as GRAMMAR_SUFFIXES

        for marker in ["များ", "တွေ"]:
            assert marker in CORE_SUFFIXES, f"core.constants should have {marker}"
            assert marker in GRAMMAR_SUFFIXES, f"grammar.patterns should have {marker}"
