"""Extended tests for config/loader.py to boost coverage."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from myspellchecker.core.exceptions import InvalidConfigError


class TestConfigLoaderFileMethods:
    """Test ConfigLoader file loading methods."""

    def test_load_from_json_file(self):
        """Test loading config from JSON file."""
        from myspellchecker.core.config.loader import load_config_from_file

        config_data = {
            "max_edit_distance": 2,
            "max_suggestions": 5,
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            path = f.name

        try:
            config = load_config_from_file(path)
            assert config is not None
            assert config["max_edit_distance"] == 2
            assert config["max_suggestions"] == 5
        finally:
            Path(path).unlink()

    def test_load_from_yaml_file(self):
        """Test loading config from YAML file."""
        from myspellchecker.core.config.loader import is_yaml_available, load_config_from_file

        if not is_yaml_available():
            pytest.skip("PyYAML not available")

        yaml_content = """
max_edit_distance: 2
max_suggestions: 5
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            path = f.name

        try:
            config = load_config_from_file(path)
            assert config is not None
            assert config["max_edit_distance"] == 2
        finally:
            Path(path).unlink()

    def test_load_from_unsupported_format(self):
        """Test loading from unsupported file format raises error."""
        from myspellchecker.core.config.loader import load_config_from_file

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("some content")
            path = f.name

        try:
            with pytest.raises(InvalidConfigError, match="Unsupported config file format"):
                load_config_from_file(path)
        finally:
            Path(path).unlink()


class TestFindConfigFile:
    """Test find_config_file function."""

    def test_find_config_file_with_explicit_path(self):
        """Test finding config file with explicit path."""
        from myspellchecker.core.config.loader import find_config_file

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({}, f)
            path = f.name

        try:
            result = find_config_file(path)
            assert result is not None
            assert result == Path(path)
        finally:
            Path(path).unlink()

    def test_find_config_file_nonexistent(self):
        """Test finding nonexistent config file returns None."""
        from myspellchecker.core.config.loader import find_config_file

        result = find_config_file("/nonexistent/path/config.json")
        assert result is None

    def test_find_config_file_no_path(self):
        """Test finding config file without explicit path."""
        from myspellchecker.core.config.loader import find_config_file

        # Should search in standard locations and return None or Path
        result = find_config_file()
        assert result is None or isinstance(result, Path)


class TestInitConfigFile:
    """Test init_config_file function."""

    def test_init_config_file_json(self):
        """Test initializing a JSON config file."""
        from myspellchecker.core.config.loader import init_config_file

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "myconfig.json"
            result = init_config_file(path)

            assert result == path
            assert path.exists()

            # Verify content
            with open(path) as f:
                data = json.load(f)
            assert "preset" in data
            assert "max_edit_distance" in data

    def test_init_config_file_yaml(self):
        """Test initializing a YAML config file."""
        from myspellchecker.core.config.loader import init_config_file, is_yaml_available

        if not is_yaml_available():
            pytest.skip("PyYAML not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "myconfig.yaml"
            result = init_config_file(path)

            assert result == path
            assert path.exists()

    def test_init_config_file_exists_no_force(self):
        """Test init raises error when file exists without force."""
        from myspellchecker.core.config.loader import init_config_file

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "myconfig.json"
            # Create file first
            init_config_file(path)

            # Should raise FileExistsError
            with pytest.raises(FileExistsError):
                init_config_file(path)

    def test_init_config_file_exists_with_force(self):
        """Test init overwrites file when force=True."""
        from myspellchecker.core.config.loader import init_config_file

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "myconfig.json"
            # Create file first
            init_config_file(path)

            # Should succeed with force=True
            result = init_config_file(path, force=True)
            assert result == path


class TestConfigLoaderClass:
    """Test ConfigLoader class methods."""

    def test_loader_load_with_profile(self):
        """Test loading config with profile."""
        from myspellchecker.core.config.loader import ConfigLoader

        loader = ConfigLoader()
        config = loader.load(profile="fast")

        assert config is not None
        assert config.max_edit_distance == 1  # Fast profile uses max_edit_distance=1

    def test_loader_load_with_file(self):
        """Test loading config from file."""
        from myspellchecker.core.config.loader import ConfigLoader

        config_data = {
            "max_edit_distance": 3,
            "max_suggestions": 10,
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            path = f.name

        try:
            loader = ConfigLoader()
            config = loader.load(config_file=path)

            assert config.max_edit_distance == 3
            assert config.max_suggestions == 10
        finally:
            Path(path).unlink()

    def test_loader_load_with_overrides(self):
        """Test loading config with programmatic overrides."""
        from myspellchecker.core.config.loader import ConfigLoader

        loader = ConfigLoader()
        config = loader.load(
            profile="fast", overrides={"max_edit_distance": 3, "max_suggestions": 15}
        )

        # Overrides take precedence
        assert config.max_edit_distance == 3
        assert config.max_suggestions == 15

    def test_loader_load_with_env_vars(self):
        """Test loading config from environment variables."""
        from myspellchecker.core.config.loader import ConfigLoader

        with patch.dict(
            "os.environ",
            {
                "MYSPELL_MAX_EDIT_DISTANCE": "3",
                "MYSPELL_MAX_SUGGESTIONS": "10",
            },
        ):
            loader = ConfigLoader()
            config = loader.load(use_env=True)

            assert config.max_edit_distance == 3
            assert config.max_suggestions == 10

    def test_loader_load_without_env_vars(self):
        """Test loading config with env vars disabled."""
        from myspellchecker.core.config.loader import ConfigLoader

        with patch.dict(
            "os.environ",
            {
                "MYSPELL_MAX_EDIT_DISTANCE": "3",
            },
        ):
            loader = ConfigLoader()
            config = loader.load(use_env=False)

            # Should use default, not env var
            assert config.max_edit_distance == 2  # Default value

    def test_loader_custom_env_prefix(self):
        """Test loader with custom environment prefix."""
        from myspellchecker.core.config.loader import ConfigLoader

        with patch.dict(
            "os.environ",
            {
                "CUSTOM_MAX_EDIT_DISTANCE": "3",
            },
        ):
            loader = ConfigLoader(env_prefix="CUSTOM_")
            config = loader.load(use_env=True)

            assert config.max_edit_distance == 3

    def test_loader_parse_bool_true_values(self):
        """Test boolean parsing for true values."""
        from myspellchecker.core.config.loader import ConfigLoader

        loader = ConfigLoader()

        for value in ["true", "True", "TRUE", "1", "yes", "Yes", "on", "ON"]:
            assert loader._parse_bool(value) is True

    def test_loader_parse_bool_false_values(self):
        """Test boolean parsing for false values."""
        from myspellchecker.core.config.loader import ConfigLoader

        loader = ConfigLoader()

        for value in ["false", "False", "FALSE", "0", "no", "No", "off", "OFF"]:
            assert loader._parse_bool(value) is False

    def test_loader_parse_bool_invalid(self):
        """Test boolean parsing raises error for invalid values."""
        from myspellchecker.core.config.loader import ConfigLoader

        loader = ConfigLoader()

        with pytest.raises(InvalidConfigError, match="Cannot parse as boolean"):
            loader._parse_bool("maybe")

    def test_loader_set_nested_key(self):
        """Test setting nested configuration keys."""
        from myspellchecker.core.config.loader import ConfigLoader

        loader = ConfigLoader()
        config = {}

        loader._set_nested_key(config, "pos_tagger.tagger_type", "transformer")

        assert config == {"pos_tagger": {"tagger_type": "transformer"}}

    def test_loader_merge_dicts_simple(self):
        """Test merging simple dictionaries."""
        from myspellchecker.core.config.loader import ConfigLoader

        loader = ConfigLoader()
        base = {"a": 1, "b": 2}
        overlay = {"b": 3, "c": 4}

        result = loader._merge_dicts(base, overlay)

        assert result == {"a": 1, "b": 3, "c": 4}

    def test_loader_merge_dicts_nested(self):
        """Test merging nested dictionaries."""
        from myspellchecker.core.config.loader import ConfigLoader

        loader = ConfigLoader()
        base = {"a": {"x": 1, "y": 2}}
        overlay = {"a": {"y": 3, "z": 4}}

        result = loader._merge_dicts(base, overlay)

        assert result == {"a": {"x": 1, "y": 3, "z": 4}}

    def test_loader_load_json_invalid(self):
        """Test loading invalid JSON file raises error."""
        from myspellchecker.core.config.loader import ConfigLoader

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not valid json {")
            path = f.name

        try:
            loader = ConfigLoader()
            with pytest.raises(InvalidConfigError, match="Invalid JSON"):
                loader.load(config_file=path)
        finally:
            Path(path).unlink()

    def test_loader_load_json_not_dict(self):
        """Test loading JSON that's not a dict raises error."""
        from myspellchecker.core.config.loader import ConfigLoader

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(["list", "not", "dict"], f)
            path = f.name

        try:
            loader = ConfigLoader()
            with pytest.raises(InvalidConfigError, match="must contain an object"):
                loader.load(config_file=path)
        finally:
            Path(path).unlink()

    def test_loader_file_not_found(self):
        """Test loading from nonexistent file raises error."""
        from myspellchecker.core.config.loader import ConfigLoader

        loader = ConfigLoader()
        with pytest.raises(FileNotFoundError):
            loader.load(config_file="/nonexistent/path/config.json")

    def test_loader_unsupported_file_format(self):
        """Test loading unsupported file format raises error."""
        from myspellchecker.core.config.loader import ConfigLoader

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("content")
            path = f.name

        try:
            loader = ConfigLoader()
            with pytest.raises(InvalidConfigError, match="Cannot auto-detect format"):
                loader.load(config_file=path)
        finally:
            Path(path).unlink()


class TestLoadConfigFunction:
    """Test the convenience load_config function."""

    def test_load_config_default(self):
        """Test load_config with defaults."""
        from myspellchecker.core.config.loader import load_config

        config = load_config()
        assert config is not None

    def test_load_config_with_profile(self):
        """Test load_config with profile."""
        from myspellchecker.core.config.loader import load_config

        config = load_config(profile="fast")
        assert config.max_edit_distance == 1

    def test_load_config_with_overrides(self):
        """Test load_config with keyword overrides."""
        from myspellchecker.core.config.loader import load_config

        config = load_config(max_edit_distance=3, max_suggestions=15)
        assert config.max_edit_distance == 3
        assert config.max_suggestions == 15


class TestIsYamlAvailable:
    """Test is_yaml_available function."""

    def test_is_yaml_available(self):
        """Test is_yaml_available returns boolean."""
        from myspellchecker.core.config.loader import is_yaml_available

        result = is_yaml_available()
        assert isinstance(result, bool)


class TestConfigLoaderEnvVarMapping:
    """Test environment variable mapping."""

    def test_env_var_boolean_context_checker(self):
        """Test USE_CONTEXT_CHECKER env var."""
        from myspellchecker.core.config.loader import ConfigLoader

        with patch.dict("os.environ", {"MYSPELL_USE_CONTEXT_CHECKER": "true"}):
            loader = ConfigLoader()
            config = loader.load(use_env=True)
            assert config.use_context_checker is True

    def test_env_var_boolean_phonetic(self):
        """Test USE_PHONETIC env var."""
        from myspellchecker.core.config.loader import ConfigLoader

        with patch.dict("os.environ", {"MYSPELL_USE_PHONETIC": "false"}):
            loader = ConfigLoader()
            config = loader.load(use_env=True)
            assert config.use_phonetic is False

    def test_env_var_invalid_integer(self):
        """Test invalid integer env var raises error."""
        from myspellchecker.core.config.loader import ConfigLoader

        with patch.dict("os.environ", {"MYSPELL_MAX_EDIT_DISTANCE": "not_a_number"}):
            loader = ConfigLoader()
            with pytest.raises(
                InvalidConfigError, match="Invalid value for MYSPELL_MAX_EDIT_DISTANCE"
            ):
                loader.load(use_env=True)

    def test_env_var_targeted_rule_toggles(self):
        """Test targeted-rule toggle env vars."""
        from myspellchecker.core.config.loader import ConfigLoader

        with patch.dict(
            "os.environ",
            {
                "MYSPELL_RANKER_ENABLE_TARGETED_RERANK_HINTS": "false",
                "MYSPELL_RANKER_ENABLE_TARGETED_CANDIDATE_INJECTIONS": "false",
                "MYSPELL_RANKER_ENABLE_TARGETED_GRAMMAR_COMPLETION_TEMPLATES": "false",
            },
        ):
            loader = ConfigLoader()
            config = loader.load(use_env=True)
            assert config.ranker.enable_targeted_rerank_hints is False
            assert config.ranker.enable_targeted_candidate_injections is False
            assert config.ranker.enable_targeted_grammar_completion_templates is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
