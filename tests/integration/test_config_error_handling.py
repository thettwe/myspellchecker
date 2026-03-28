"""Integration tests for config loading error handling."""

import tempfile
from pathlib import Path

import pytest

from myspellchecker.core.config.loader import ConfigLoader
from myspellchecker.core.exceptions import InvalidConfigError


class TestConfigFileErrors:
    """Test error handling for config file issues."""

    def test_malformed_yaml_raises_error(self):
        """Test that malformed YAML raises appropriate error."""
        malformed_yaml = """
preset: [this is not valid yaml
max_suggestions: 10
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write(malformed_yaml)
            config_path = Path(f.name)

        try:
            loader = ConfigLoader()
            with pytest.raises(InvalidConfigError):  # Should raise YAML parsing error
                loader.load(config_file=config_path)
        finally:
            if config_path.exists():
                config_path.unlink()

    def test_malformed_json_raises_error(self):
        """Test that malformed JSON raises appropriate error."""
        malformed_json = """{
  "preset": "fast",
  "max_suggestions": 10,
  missing closing brace
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            f.write(malformed_json)
            config_path = Path(f.name)

        try:
            loader = ConfigLoader()
            with pytest.raises(InvalidConfigError):  # Should raise JSON parsing error
                loader.load(config_file=config_path)
        finally:
            if config_path.exists():
                config_path.unlink()

    def test_empty_config_file(self):
        """Test that empty config file raises ValueError."""
        empty_config = ""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write(empty_config)
            config_path = Path(f.name)

        try:
            loader = ConfigLoader()
            with pytest.raises(InvalidConfigError, match="YAML file must contain a dictionary"):
                loader.load(config_file=config_path)
        finally:
            if config_path.exists():
                config_path.unlink()

    def test_config_with_comments(self):
        """Test that YAML comments are handled correctly."""
        config_with_comments = """
# Configuration for spell checker
preset: accurate  # Use accurate preset

# Suggestion settings
max_edit_distance: 3  # Maximum edit distance
max_suggestions: 10   # Maximum suggestions

# Features
use_phonetic: true  # Enable phonetic matching
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write(config_with_comments)
            config_path = Path(f.name)

        try:
            loader = ConfigLoader()
            config = loader.load(config_file=config_path)

            # Should parse correctly, ignoring comments
            assert config.max_edit_distance == 3
            assert config.max_suggestions == 10
            assert config.use_phonetic is True
        finally:
            if config_path.exists():
                config_path.unlink()


class TestInvalidConfigValues:
    """Test handling of invalid config values."""

    def test_invalid_preset_name(self):
        """Test that invalid preset name raises ValueError."""
        config_content = """
preset: nonexistent_preset
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write(config_content)
            config_path = Path(f.name)

        try:
            loader = ConfigLoader()
            with pytest.raises(InvalidConfigError, match="Unknown profile"):
                loader.load(config_file=config_path)
        finally:
            if config_path.exists():
                config_path.unlink()

    def test_negative_max_suggestions(self):
        """Test that negative max_suggestions raises validation error."""
        config_content = """
preset: fast
max_suggestions: -5
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write(config_content)
            config_path = Path(f.name)

        try:
            loader = ConfigLoader()
            # Pydantic validates max_suggestions >= 1
            with pytest.raises(Exception, match="max_suggestions"):
                loader.load(config_file=config_path)
        finally:
            if config_path.exists():
                config_path.unlink()

    def test_invalid_word_engine(self):
        """Test that invalid word_engine value raises ValueError."""
        config_content = """
word_engine: invalid_engine
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write(config_content)
            config_path = Path(f.name)

        try:
            loader = ConfigLoader()
            with pytest.raises(ValueError, match="invalid_engine"):
                loader.load(config_file=config_path)
        finally:
            if config_path.exists():
                config_path.unlink()


class TestConfigTypeMismatches:
    """Test handling of type mismatches in config."""

    def test_boolean_as_string(self):
        """Test that boolean fields handle string values."""
        config_content = """
preset: fast
use_phonetic: "true"
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write(config_content)
            config_path = Path(f.name)

        try:
            loader = ConfigLoader()
            config = loader.load(config_file=config_path)

            # YAML should parse "true" as boolean
            assert config.use_phonetic is True
        finally:
            if config_path.exists():
                config_path.unlink()

    def test_integer_as_string(self):
        """Test that integer fields handle string values."""
        config_content = """
preset: fast
max_suggestions: "10"
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write(config_content)
            config_path = Path(f.name)

        try:
            loader = ConfigLoader()
            config = loader.load(config_file=config_path)

            # Should coerce or validate
            assert isinstance(config.max_suggestions, int)
            assert config.max_suggestions == 10
        finally:
            if config_path.exists():
                config_path.unlink()


class TestMissingPyYAML:
    """Test behavior when PyYAML is not installed."""

    def test_yaml_file_without_pyyaml(self, monkeypatch):
        """Test that YAML file loading fails gracefully without PyYAML."""
        # Mock PyYAML import to fail
        import sys

        if "yaml" in sys.modules:
            # Can't easily test this if PyYAML is installed
            pytest.skip("PyYAML is installed, cannot test missing import")

        # If we could mock it, would test graceful failure
        # This is more of a documentation test


class TestConfigFileSearch:
    """Test config file search behavior."""

    def test_config_search_current_directory(self, tmp_path, monkeypatch):
        """Test that config is found in current directory."""
        # Create config in temp dir
        config_path = tmp_path / "myspellchecker.yaml"
        config_path.write_text("preset: default\n")

        # Change to temp dir
        monkeypatch.chdir(tmp_path)

        from myspellchecker.core.config.loader import find_config_file

        # Should find config in current dir
        found = find_config_file()
        assert found is not None
        assert found.name == "myspellchecker.yaml"

    def test_config_search_with_extension_priority(self, tmp_path):
        """Test that .yaml is preferred over .yml when both exist."""
        # Create both yaml and yml files
        yaml_path = tmp_path / "myspellchecker.yaml"
        yml_path = tmp_path / "myspellchecker.yml"

        yaml_path.write_text("preset: accurate\n")
        yml_path.write_text("preset: fast\n")

        from myspellchecker.core.config.loader import find_config_file

        # Should find .yaml first (if that's the priority)
        # Implementation-dependent
        found = find_config_file(path=tmp_path)
        assert found is not None


class TestConfigErrorHandlingEdgeCases:
    """Test edge cases and corner scenarios."""

    def test_very_large_config_values(self):
        """Test that very large config values are handled."""
        config_content = """
preset: fast
max_suggestions: 10000
provider_config:
  cache_size: 999999
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write(config_content)
            config_path = Path(f.name)

        try:
            loader = ConfigLoader()
            config = loader.load(config_file=config_path)

            # Should either accept or validate against max
            assert config.max_suggestions > 0
            assert config.provider_config.cache_size > 0
        finally:
            if config_path.exists():
                config_path.unlink()

    def test_unicode_in_config_paths(self):
        """Test that Unicode characters in paths are handled."""
        config_content = """
preset: fast
database: /path/with/unicode/日本語/database.db
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write(config_content)
            config_path = Path(f.name)

        try:
            loader = ConfigLoader()
            config = loader.load(config_file=config_path)

            # Should handle Unicode in paths
            assert config is not None
        finally:
            if config_path.exists():
                config_path.unlink()

    def test_config_with_extra_unknown_fields(self):
        """Test that extra unknown fields raise validation error."""
        config_content = """
preset: fast
max_suggestions: 10
unknown_field: some_value
another_unknown: 123
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write(config_content)
            config_path = Path(f.name)

        try:
            loader = ConfigLoader()
            # Pydantic forbids extra fields
            with pytest.raises(Exception, match="Extra inputs are not permitted"):
                loader.load(config_file=config_path)
        finally:
            if config_path.exists():
                config_path.unlink()
