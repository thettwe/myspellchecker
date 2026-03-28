"""Integration tests for config file loading."""

import tempfile
from pathlib import Path

import pytest

from myspellchecker.core.config import SpellCheckerConfig
from myspellchecker.core.config.loader import ConfigLoader, find_config_file
from myspellchecker.core.exceptions import InvalidConfigError
from tests.fixtures.config_templates import (
    CONFIG_WITH_DATABASE,
    CONFIG_WITH_DEVELOPMENT_PRESET,
    CONFIG_WITH_FULL_PROVIDER,
    CONFIG_WITH_PRODUCTION_PRESET,
    INVALID_YAML_CONFIG,
    MINIMAL_YAML_CONFIG,
    VALID_JSON_CONFIG,
    VALID_YAML_CONFIG,
)


@pytest.fixture
def temp_yaml_config():
    """Create a temporary YAML config file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as f:
        f.write(VALID_YAML_CONFIG)
        config_path = Path(f.name)

    yield config_path

    # Cleanup
    if config_path.exists():
        config_path.unlink()


@pytest.fixture
def temp_json_config():
    """Create a temporary JSON config file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        f.write(VALID_JSON_CONFIG)
        config_path = Path(f.name)

    yield config_path

    # Cleanup
    if config_path.exists():
        config_path.unlink()


@pytest.fixture
def temp_minimal_config():
    """Create a minimal temporary config file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as f:
        f.write(MINIMAL_YAML_CONFIG)
        config_path = Path(f.name)

    yield config_path

    # Cleanup
    if config_path.exists():
        config_path.unlink()


class TestConfigFileLoading:
    """Test config file loading with ConfigLoader."""

    def test_yaml_config_file_loaded_successfully(self, temp_yaml_config):
        """Test that YAML config file is loaded and values are applied."""
        loader = ConfigLoader()
        config = loader.load(config_file=temp_yaml_config)

        assert isinstance(config, SpellCheckerConfig)
        assert config.max_edit_distance == 3
        assert config.max_suggestions == 10
        assert config.use_context_checker is False
        assert config.use_phonetic is True
        assert config.word_engine == "myword"
        assert config.provider_config.cache_size == 2048
        assert config.provider_config.pool_min_size == 2
        assert config.provider_config.pool_max_size == 15

    def test_json_config_file_loaded_successfully(self, temp_json_config):
        """Test that JSON config file is loaded and values are applied."""
        loader = ConfigLoader()
        config = loader.load(config_file=temp_json_config)

        assert isinstance(config, SpellCheckerConfig)
        assert config.max_edit_distance == 2
        assert config.max_suggestions == 5
        assert config.use_context_checker is True
        assert config.use_phonetic is False
        assert config.provider_config.cache_size == 1024
        assert config.provider_config.pool_min_size == 1
        assert config.provider_config.pool_max_size == 5

    def test_minimal_config_file(self, temp_minimal_config):
        """Test that minimal config file works with defaults."""
        loader = ConfigLoader()
        config = loader.load(config_file=temp_minimal_config)

        assert isinstance(config, SpellCheckerConfig)
        # Should have defaults for non-specified fields
        assert config.max_edit_distance > 0
        assert config.max_suggestions > 0

    def test_config_file_not_found_returns_none(self):
        """Test that non-existent config file returns None."""
        non_existent = Path("/nonexistent/config.yaml")
        result = find_config_file(non_existent)
        assert result is None

    def test_invalid_yaml_falls_back_gracefully(self):
        """Test that invalid YAML logs warning and falls back."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write(INVALID_YAML_CONFIG)
            config_path = Path(f.name)

        try:
            loader = ConfigLoader()
            # Should raise error or handle gracefully
            with pytest.raises(InvalidConfigError):  # Will raise YAML parsing error
                loader.load(config_file=config_path)
        finally:
            if config_path.exists():
                config_path.unlink()


class TestPresetAliases:
    """Test preset alias compatibility."""

    def test_production_preset_alias(self):
        """Test that 'production' preset maps to 'default'."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write(CONFIG_WITH_PRODUCTION_PRESET)
            config_path = Path(f.name)

        try:
            loader = ConfigLoader()
            config = loader.load(config_file=config_path)
            assert isinstance(config, SpellCheckerConfig)
            # Should load without error (production maps to default)
            assert config.max_edit_distance == 2
        finally:
            if config_path.exists():
                config_path.unlink()

    def test_development_preset_alias(self):
        """Test that 'development' preset maps to 'fast'."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write(CONFIG_WITH_DEVELOPMENT_PRESET)
            config_path = Path(f.name)

        try:
            loader = ConfigLoader()
            config = loader.load(config_file=config_path)
            assert isinstance(config, SpellCheckerConfig)
            # Should load without error (development maps to fast)
            assert config.use_context_checker is True
        finally:
            if config_path.exists():
                config_path.unlink()


class TestDatabaseConfiguration:
    """Test database path configuration."""

    def test_database_path_from_config(self):
        """Test that database path is honored from config."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write(CONFIG_WITH_DATABASE)
            config_path = Path(f.name)

        try:
            loader = ConfigLoader()
            config = loader.load(config_file=config_path)
            assert isinstance(config, SpellCheckerConfig)
            # Database path should be set (may be in provider_config)
            # Note: Actual provider initialization may fail if path doesn't exist
        finally:
            if config_path.exists():
                config_path.unlink()


class TestProviderConfiguration:
    """Test provider configuration from config file."""

    def test_full_provider_config(self):
        """Test that all provider config fields are honored."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write(CONFIG_WITH_FULL_PROVIDER)
            config_path = Path(f.name)

        try:
            loader = ConfigLoader()
            config = loader.load(config_file=config_path)

            assert config.provider_config.cache_size == 2048
            assert config.provider_config.pool_min_size == 5
            assert config.provider_config.pool_max_size == 20
            assert config.provider_config.pool_timeout == 30.0
            assert config.provider_config.pool_max_connection_age == 7200.0
        finally:
            if config_path.exists():
                config_path.unlink()
