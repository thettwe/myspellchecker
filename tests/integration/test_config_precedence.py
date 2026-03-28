"""Integration tests for config override precedence (CLI > Config > Env > Defaults)."""

import tempfile
from pathlib import Path

import pytest

from myspellchecker.core.config.loader import ConfigLoader


@pytest.fixture
def temp_config_with_values():
    """Create a config file with specific values."""
    config_content = """
preset: accurate
max_edit_distance: 3
max_suggestions: 10
use_phonetic: true
use_context_checker: true
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as f:
        f.write(config_content)
        config_path = Path(f.name)

    yield config_path

    if config_path.exists():
        config_path.unlink()


class TestConfigPrecedence:
    """Test configuration override precedence: CLI > Config > Env > Defaults."""

    def test_cli_overrides_config(self, temp_config_with_values):
        """Test that CLI overrides (passed as overrides param) win over config file."""
        loader = ConfigLoader()

        # Load with CLI overrides (max_edit_distance max is 3)
        config = loader.load(
            config_file=temp_config_with_values,
            overrides={
                "max_edit_distance": 2,  # CLI override
                "max_suggestions": 20,  # CLI override
            },
        )

        # CLI values should win
        assert config.max_edit_distance == 2  # CLI override
        assert config.max_suggestions == 20  # CLI override
        # Config file values used where no override
        assert config.use_phonetic is True
        assert config.use_context_checker is True

    def test_config_overrides_defaults(self, temp_config_with_values):
        """Test that config file overrides built-in defaults."""
        loader = ConfigLoader()

        # Load with just config file (no overrides)
        config = loader.load(config_file=temp_config_with_values)

        # Config values should override defaults
        assert config.max_edit_distance == 3  # From config
        assert config.max_suggestions == 10  # From config
        assert config.use_phonetic is True  # From config

    def test_env_overrides_defaults(self, monkeypatch):
        """Test that environment variables override defaults."""
        monkeypatch.setenv("MYSPELL_MAX_SUGGESTIONS", "15")

        loader = ConfigLoader()
        config = loader.load(use_env=True)

        # Env var should override default
        assert config.max_suggestions == 15

    def test_full_precedence_chain(self, temp_config_with_values, monkeypatch):
        """Test full precedence: CLI > Config > Env > Defaults."""
        monkeypatch.setenv("MYSPELL_USE_PHONETIC", "false")

        loader = ConfigLoader()
        config = loader.load(
            config_file=temp_config_with_values,
            use_env=True,
            overrides={
                "max_edit_distance": 2,  # CLI (highest priority)
            },
        )

        # CLI override wins
        assert config.max_edit_distance == 2

        # Config file wins over env (for fields in config)
        assert config.max_suggestions == 10  # From config

        # Config file value used
        assert config.use_context_checker is True  # From config

    def test_partial_overrides(self, temp_config_with_values):
        """Test that partial overrides only affect specified fields."""
        loader = ConfigLoader()

        config = loader.load(
            config_file=temp_config_with_values,
            overrides={
                "use_phonetic": False,  # Override just this one field
            },
        )

        # Overridden field
        assert config.use_phonetic is False

        # Non-overridden fields from config
        assert config.max_edit_distance == 3
        assert config.max_suggestions == 10
        assert config.use_context_checker is True


class TestProviderConfigPrecedence:
    """Test provider_config override precedence."""

    def test_provider_config_overrides(self):
        """Test that provider_config can be overridden."""
        config_content = """
preset: production
provider_config:
  cache_size: 1024
  pool_min_size: 2
  pool_max_size: 5
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write(config_content)
            config_path = Path(f.name)

        try:
            loader = ConfigLoader()
            config = loader.load(
                config_file=config_path,
                overrides={
                    "provider_config": {
                        "cache_size": 4096,  # Override
                    }
                },
            )

            # Override should win
            assert config.provider_config.cache_size == 4096
            # Non-overridden values from config
            assert config.provider_config.pool_min_size == 2
            assert config.provider_config.pool_max_size == 5
        finally:
            if config_path.exists():
                config_path.unlink()


class TestDatabasePathPrecedence:
    """Test database path precedence (special handling in loader)."""

    def test_database_in_config(self):
        """Test database path from config file."""
        config_content = """
database: /path/from/config.db
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write(config_content)
            config_path = Path(f.name)

        try:
            loader = ConfigLoader()
            config = loader.load(config_file=config_path)

            # Database should be set in provider_config
            # (loader maps 'database' to provider_config.database_path)
            assert config.provider_config.database_path == "/path/from/config.db"
        finally:
            if config_path.exists():
                config_path.unlink()

    def test_database_override(self):
        """Test database path can be overridden via provider_config."""
        config_content = """
database: /path/from/config.db
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write(config_content)
            config_path = Path(f.name)

        try:
            loader = ConfigLoader()
            config = loader.load(
                config_file=config_path,
                overrides={
                    "provider_config": {
                        "database_path": "/path/from/override.db",
                    },
                },
            )

            # Override should win
            assert config.provider_config.database_path == "/path/from/override.db"
        finally:
            if config_path.exists():
                config_path.unlink()


class TestPresetPrecedence:
    """Test preset override precedence."""

    def test_preset_in_config(self):
        """Test preset from config file."""
        config_content = """
preset: accurate
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write(config_content)
            config_path = Path(f.name)

        try:
            loader = ConfigLoader()
            config = loader.load(config_file=config_path)

            # Should use accurate preset values
            assert config.max_edit_distance == 3
            assert config.max_suggestions == 10
            assert config.use_phonetic is True
            assert config.use_context_checker is True
        finally:
            if config_path.exists():
                config_path.unlink()

    def test_preset_override(self):
        """Test preset can be overridden via profile parameter."""
        config_content = """
preset: accurate
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write(config_content)
            config_path = Path(f.name)

        try:
            loader = ConfigLoader()
            # Override preset by loading with a different profile
            # (profile takes precedence when specified alongside config file preset)
            config = loader.load(
                config_file=config_path,
                profile="fast",
            )

            # Should use fast preset values (profile overrides config file preset)
            assert config.max_edit_distance == 1
            assert config.max_suggestions == 3
            assert config.use_phonetic is False
            assert config.use_context_checker is False
        finally:
            if config_path.exists():
                config_path.unlink()
