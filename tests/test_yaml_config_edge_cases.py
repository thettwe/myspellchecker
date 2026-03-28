"""
Additional edge case tests for YAML config file loading.

Tests covering:
- YAML with special characters and Unicode
- Empty YAML files
- YAML with nested configurations
- Invalid YAML syntax handling
- YAML with comments
- YAML anchors and references
- Large YAML files
"""

import json
import tempfile
from pathlib import Path

import pytest

from myspellchecker.core.config.loader import (
    ConfigLoader,
    is_yaml_available,
    load_config,
    load_config_from_file,
)
from myspellchecker.core.exceptions import InvalidConfigError


@pytest.fixture
def skip_if_no_yaml():
    """Skip test if PyYAML is not available."""
    if not is_yaml_available():
        pytest.skip("PyYAML not available")


class TestYamlSpecialContent:
    """Test YAML files with special content."""

    def test_yaml_with_unicode_myanmar_text(self, skip_if_no_yaml):
        """Test YAML file containing Myanmar Unicode text."""
        yaml_content = """
# Myanmar text in config
max_edit_distance: 2
database: "မြန်မာ_dictionary.db"
custom_words:
  - "ကောင်းပါတယ်"
  - "မင်္ဂလာပါ"
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write(yaml_content)
            path = f.name

        try:
            config = load_config_from_file(path)
            assert config["database"] == "မြန်မာ_dictionary.db"
            assert "ကောင်းပါတယ်" in config["custom_words"]
        finally:
            Path(path).unlink()

    def test_yaml_with_comments(self, skip_if_no_yaml):
        """Test YAML file with inline and block comments."""
        yaml_content = """
# Main configuration
max_edit_distance: 2  # Maximum distance
max_suggestions: 5    # Suggestion limit

# Feature flags
use_phonetic: true
# This line is commented out: use_ner: false
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            path = f.name

        try:
            config = load_config_from_file(path)
            assert config["max_edit_distance"] == 2
            assert config["use_phonetic"] is True
            # Commented out line should not appear
            assert "use_ner" not in config
        finally:
            Path(path).unlink()

    def test_yaml_empty_file(self, skip_if_no_yaml):
        """Test loading empty YAML file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            path = f.name

        try:
            config = load_config_from_file(path)
            assert config == {}
        finally:
            Path(path).unlink()

    def test_yaml_whitespace_only(self, skip_if_no_yaml):
        """Test YAML file with only spaces (not tabs, as tabs are invalid in YAML)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            # Only spaces, no tabs (tabs are invalid in YAML)
            f.write("   \n   \n   ")
            path = f.name

        try:
            config = load_config_from_file(path)
            assert config == {} or config is None
        finally:
            Path(path).unlink()

    def test_yaml_comments_only(self, skip_if_no_yaml):
        """Test YAML file with only comments."""
        yaml_content = """
# This is a comment
# Another comment
# Third comment
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            path = f.name

        try:
            config = load_config_from_file(path)
            assert config == {} or config is None
        finally:
            Path(path).unlink()


class TestYamlNestedConfigs:
    """Test YAML files with nested configurations."""

    def test_yaml_deeply_nested(self, skip_if_no_yaml):
        """Test YAML with deeply nested structures."""
        yaml_content = """
level1:
  level2:
    level3:
      level4:
        value: 42
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            path = f.name

        try:
            config = load_config_from_file(path)
            assert config["level1"]["level2"]["level3"]["level4"]["value"] == 42
        finally:
            Path(path).unlink()

    def test_yaml_mixed_types(self, skip_if_no_yaml):
        """Test YAML with mixed data types."""
        yaml_content = """
string_val: "hello"
int_val: 42
float_val: 3.14
bool_true: true
bool_false: false
null_val: null
list_val:
  - item1
  - item2
dict_val:
  key: value
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            path = f.name

        try:
            config = load_config_from_file(path)
            assert config["string_val"] == "hello"
            assert config["int_val"] == 42
            assert config["float_val"] == 3.14
            assert config["bool_true"] is True
            assert config["bool_false"] is False
            assert config["null_val"] is None
            assert config["list_val"] == ["item1", "item2"]
            assert config["dict_val"] == {"key": "value"}
        finally:
            Path(path).unlink()

    def test_yaml_with_anchors_and_aliases(self, skip_if_no_yaml):
        """Test YAML with anchors and aliases (references)."""
        yaml_content = """
defaults: &defaults
  max_edit_distance: 2
  max_suggestions: 5

production:
  <<: *defaults
  max_suggestions: 10
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            path = f.name

        try:
            config = load_config_from_file(path)
            # Verify anchor/alias resolution
            assert config["production"]["max_edit_distance"] == 2
            assert config["production"]["max_suggestions"] == 10
        finally:
            Path(path).unlink()


class TestYamlErrorHandling:
    """Test error handling for invalid YAML files."""

    def test_yaml_invalid_syntax(self, skip_if_no_yaml):
        """Test invalid YAML syntax raises appropriate error."""
        yaml_content = """
key: value
  bad_indent: this is wrong
    even_worse: indentation
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            path = f.name

        try:
            loader = ConfigLoader()
            with pytest.raises(InvalidConfigError, match="Invalid YAML"):
                loader.load(config_file=path)
        finally:
            Path(path).unlink()

    def test_yaml_not_dict_raises_error(self, skip_if_no_yaml):
        """Test YAML that's not a dict raises error."""
        yaml_content = "- item1\n- item2\n- item3"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            path = f.name

        try:
            loader = ConfigLoader()
            with pytest.raises(InvalidConfigError, match="must contain a dictionary"):
                loader.load(config_file=path)
        finally:
            Path(path).unlink()

    def test_yaml_scalar_raises_error(self, skip_if_no_yaml):
        """Test YAML that's just a scalar raises error."""
        yaml_content = "just a string value"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            path = f.name

        try:
            loader = ConfigLoader()
            with pytest.raises(InvalidConfigError, match="must contain a dictionary"):
                loader.load(config_file=path)
        finally:
            Path(path).unlink()


class TestYamlConfigIntegration:
    """Integration tests for YAML config loading with SpellCheckerConfig."""

    def test_yaml_valid_spellchecker_config(self, skip_if_no_yaml):
        """Test loading valid SpellCheckerConfig from YAML."""
        yaml_content = """
max_edit_distance: 2
max_suggestions: 8
use_phonetic: true
use_context_checker: false
use_ner: true
use_rule_based_validation: true
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            path = f.name

        try:
            config = load_config(config_file=path, use_env=False)
            assert config.max_edit_distance == 2
            assert config.max_suggestions == 8
            assert config.use_phonetic is True
            assert config.use_context_checker is False
            assert config.use_ner is True
            assert config.use_rule_based_validation is True
        finally:
            Path(path).unlink()

    def test_yaml_nested_spellchecker_config(self, skip_if_no_yaml):
        """Test loading nested config structures from YAML."""
        yaml_content = """
max_edit_distance: 2
symspell:
  prefix_length: 7
ngram_context:
  bigram_threshold: 0.001
  trigram_threshold: 0.0001
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            path = f.name

        try:
            config = load_config(config_file=path, use_env=False)
            assert config.max_edit_distance == 2
            assert config.symspell.prefix_length == 7
            assert config.ngram_context.bigram_threshold == 0.001
            assert config.ngram_context.trigram_threshold == 0.0001
        finally:
            Path(path).unlink()

    def test_yaml_with_profile_override(self, skip_if_no_yaml):
        """Test YAML config overriding profile defaults."""
        yaml_content = """
max_edit_distance: 3
max_suggestions: 15
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            path = f.name

        try:
            loader = ConfigLoader()
            # File should override profile
            config = loader.load(profile="fast", config_file=path, use_env=False)
            # Fast profile has max_edit_distance=1, file has 3
            assert config.max_edit_distance == 3
            assert config.max_suggestions == 15
        finally:
            Path(path).unlink()


class TestYmlExtension:
    """Test .yml extension handling."""

    def test_yml_extension_supported(self, skip_if_no_yaml):
        """Test that .yml extension works like .yaml."""
        yaml_content = """
max_edit_distance: 2
max_suggestions: 5
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write(yaml_content)
            path = f.name

        try:
            config = load_config_from_file(path)
            assert config["max_edit_distance"] == 2
        finally:
            Path(path).unlink()


class TestJsonComparison:
    """Compare YAML and JSON loading behavior."""

    def test_yaml_json_equivalence(self, skip_if_no_yaml):
        """Test that equivalent YAML and JSON produce same config."""
        config_data = {
            "max_edit_distance": 2,
            "max_suggestions": 5,
            "use_phonetic": True,
        }

        yaml_content = """
max_edit_distance: 2
max_suggestions: 5
use_phonetic: true
"""
        # Create YAML file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            yaml_path = f.name

        # Create JSON file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            json_path = f.name

        try:
            yaml_config = load_config_from_file(yaml_path)
            json_config = load_config_from_file(json_path)

            assert yaml_config == json_config
        finally:
            Path(yaml_path).unlink()
            Path(json_path).unlink()


class TestLargeYamlFiles:
    """Test handling of large YAML files."""

    def test_yaml_many_keys(self, skip_if_no_yaml):
        """Test YAML file with many keys."""
        # Generate YAML with 100 keys
        lines = ["max_edit_distance: 2"]
        for i in range(100):
            lines.append(f"custom_key_{i}: value_{i}")

        yaml_content = "\n".join(lines)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            path = f.name

        try:
            config = load_config_from_file(path)
            assert len(config) == 101  # 100 custom + 1 max_edit_distance
            assert config["custom_key_50"] == "value_50"
        finally:
            Path(path).unlink()

    def test_yaml_large_list(self, skip_if_no_yaml):
        """Test YAML with large list."""
        items = [f"item_{i}" for i in range(1000)]
        yaml_content = "items:\n" + "\n".join(f"  - {item}" for item in items)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            path = f.name

        try:
            config = load_config_from_file(path)
            assert len(config["items"]) == 1000
        finally:
            Path(path).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
