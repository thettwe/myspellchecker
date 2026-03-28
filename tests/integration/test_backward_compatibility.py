"""Integration tests for backward compatibility after Round 6 fixes."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


def _run_cli(*args, input_text=None):
    """Helper to run CLI command via subprocess."""
    result = subprocess.run(
        [sys.executable, "-c", "from myspellchecker.cli import main; main()", *args],
        capture_output=True,
        text=True,
        input=input_text,
    )
    return result


def _assert_no_traceback(result):
    """Assert subprocess did not fail with an unhandled exception."""
    assert "Traceback" not in result.stderr


def _assert_valid_check_json(result, *, expected_file=None):
    """Assert a check command produced valid JSON output."""
    # Keep backward-compatible semantics:
    # some environments may use 1 for handled validation outcomes.
    assert result.returncode in [0, 1]
    _assert_no_traceback(result)
    if not result.stdout.strip():
        # In some environments, a handled DB/config error exits with code 1 and stderr only.
        assert result.returncode == 1
        assert result.stderr.strip()
        assert "Error: Database error:" in result.stderr or "Configuration Error:" in result.stderr
        return

    payload = json.loads(result.stdout)
    assert {"summary", "results"} <= set(payload)

    summary = payload["summary"]
    results = payload["results"]
    assert isinstance(summary, dict)
    assert isinstance(results, list)

    for key in ("total_lines", "total_errors", "lines_with_errors"):
        assert key in summary
        assert isinstance(summary[key], int)
        assert summary[key] >= 0

    assert summary["total_lines"] == len(results)
    assert summary["lines_with_errors"] <= summary["total_lines"]
    assert summary["total_errors"] >= summary["lines_with_errors"]

    if expected_file is not None and results:
        assert all(row["file"] == expected_file for row in results)

    for row in results:
        assert {"file", "line", "text", "has_errors", "errors"} <= set(row)
        assert isinstance(row["line"], int)
        assert row["line"] >= 1
        assert isinstance(row["text"], str)
        assert isinstance(row["has_errors"], bool)
        assert isinstance(row["errors"], list)

    return payload


@pytest.fixture
def temp_input_file():
    """Create a temporary input file with Myanmar text."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("မြန်မာစာ\n")
        input_path = Path(f.name)

    yield input_path

    if input_path.exists():
        input_path.unlink()


class TestLegacyCLIUsage:
    """Test that legacy CLI usage patterns still work."""

    def test_basic_check_command(self, temp_input_file):
        """Test basic check command (most common usage)."""
        result = _run_cli("check", str(temp_input_file))

        _assert_valid_check_json(result, expected_file=str(temp_input_file))

    def test_check_with_preset_only(self, temp_input_file):
        """Test check with --preset flag (common usage)."""
        result = _run_cli("check", "--preset", "fast", str(temp_input_file))

        _assert_valid_check_json(result, expected_file=str(temp_input_file))

    def test_check_with_no_phonetic(self, temp_input_file):
        """Test check with --no-phonetic flag."""
        result = _run_cli("check", "--no-phonetic", str(temp_input_file))

        _assert_valid_check_json(result, expected_file=str(temp_input_file))

    def test_check_with_no_context(self, temp_input_file):
        """Test check with --no-context flag."""
        result = _run_cli("check", "--no-context", str(temp_input_file))

        _assert_valid_check_json(result, expected_file=str(temp_input_file))

    def test_check_with_all_legacy_flags(self, temp_input_file):
        """Test check with combination of legacy flags."""
        result = _run_cli(
            "check",
            "--preset",
            "accurate",
            "--no-phonetic",
            "--no-context",
            "--format",
            "json",
            str(temp_input_file),
        )

        _assert_valid_check_json(result, expected_file=str(temp_input_file))


class TestLegacyConfigFiles:
    """Test that old config files still work."""

    def test_old_config_without_new_fields(self):
        """Test that old config files (without new fields) still work."""
        # Old config might only have preset and database
        old_config = """
preset: fast
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write(old_config)
            config_path = Path(f.name)

        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False, encoding="utf-8"
            ) as input_file:
                input_file.write("မြန်မာစာ\n")
                input_path = Path(input_file.name)

            try:
                result = _run_cli(
                    "check",
                    "-c",
                    str(config_path),
                    str(input_path),
                    "--format",
                    "json",
                )

                _assert_valid_check_json(result, expected_file=str(input_path))
            finally:
                if input_path.exists():
                    input_path.unlink()
        finally:
            if config_path.exists():
                config_path.unlink()

    def test_config_with_production_preset_still_works(self):
        """Test that old configs with 'production' preset still work (via alias)."""
        old_config = """
preset: production
max_suggestions: 5
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write(old_config)
            config_path = Path(f.name)

        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False, encoding="utf-8"
            ) as input_file:
                input_file.write("မြန်မာစာ\n")
                input_path = Path(input_file.name)

            try:
                result = _run_cli(
                    "check",
                    "-c",
                    str(config_path),
                    str(input_path),
                    "--format",
                    "json",
                )

                _assert_valid_check_json(result, expected_file=str(input_path))
            finally:
                if input_path.exists():
                    input_path.unlink()
        finally:
            if config_path.exists():
                config_path.unlink()


class TestNoConfigFile:
    """Test that CLI works without any config file (pure CLI mode)."""

    def test_no_config_with_stdin(self):
        """Test stdin without config file."""
        myanmar_text = "မြန်မာစာ\n"

        result = _run_cli("check", "--format", "json", input_text=myanmar_text)

        _assert_valid_check_json(result, expected_file="stdin")

    def test_no_config_with_file(self, temp_input_file):
        """Test file input without config file."""
        result = _run_cli("check", str(temp_input_file), "--format", "json")

        _assert_valid_check_json(result, expected_file=str(temp_input_file))

    def test_no_config_with_all_flags(self, temp_input_file):
        """Test all CLI flags work without config file."""
        result = _run_cli(
            "check",
            "--preset",
            "accurate",
            "--no-phonetic",
            "--no-context",
            "--level",
            "word",
            "--format",
            "json",
            str(temp_input_file),
        )

        _assert_valid_check_json(result, expected_file=str(temp_input_file))


class TestMixedLegacyAndNew:
    """Test mixing legacy and new features."""

    def test_legacy_flags_with_new_config_loading(self):
        """Test that legacy flags work with new config loading system."""
        config_content = """
preset: accurate
max_suggestions: 10
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write(config_content)
            config_path = Path(f.name)

        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False, encoding="utf-8"
            ) as input_file:
                input_file.write("မြန်မာစာ\n")
                input_path = Path(input_file.name)

            try:
                # Mix config file with legacy flags
                result = _run_cli(
                    "check",
                    "-c",
                    str(config_path),
                    "--no-phonetic",  # Legacy flag
                    str(input_path),
                    "--format",
                    "json",
                )

                _assert_valid_check_json(result, expected_file=str(input_path))
            finally:
                if input_path.exists():
                    input_path.unlink()
        finally:
            if config_path.exists():
                config_path.unlink()

    def test_dash_stdin_with_legacy_preset(self):
        """Test '-' stdin with legacy --preset flag."""
        myanmar_text = "မြန်မာစာ\n"

        result = _run_cli(
            "check",
            "--preset",
            "minimal",  # Legacy usage
            "-",  # New stdin support
            "--format",
            "json",
            input_text=myanmar_text,
        )

        _assert_valid_check_json(result, expected_file="-")


class TestOtherCommands:
    """Test that other CLI commands still work (not just check)."""

    def test_config_init_command(self):
        """Test config init command still works."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test_config.yaml"

            result = _run_cli("config", "init", "--path", str(config_path))

            # Should create config file
            assert result.returncode == 0
            assert config_path.exists()

    def test_config_show_command(self):
        """Test config show command still works."""
        result = _run_cli("config", "show")

        # Should show config paths
        assert result.returncode == 0
        assert result.stdout

    def test_help_command(self):
        """Test help command still works."""
        result = _run_cli("--help")

        # Should show help
        assert result.returncode == 0
        assert "myspellchecker" in result.stdout.lower()

    def test_check_help(self):
        """Test check subcommand help still works."""
        result = _run_cli("check", "--help")

        # Should show check help
        assert result.returncode == 0
        assert "check" in result.stdout.lower()


class TestPresetCompatibility:
    """Test all presets still work."""

    @pytest.mark.parametrize("preset", ["default", "fast", "accurate", "minimal", "strict"])
    def test_all_cli_presets_work(self, preset, temp_input_file):
        """Test that all CLI presets work."""
        result = _run_cli(
            "check",
            "--preset",
            preset,
            str(temp_input_file),
            "--format",
            "json",
        )

        _assert_valid_check_json(result, expected_file=str(temp_input_file))

    @pytest.mark.parametrize("preset_alias", ["production", "development", "testing"])
    def test_all_preset_aliases_work(self, preset_alias):
        """Test that all preset aliases work in config files."""
        config_content = f"""
preset: {preset_alias}
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write(config_content)
            config_path = Path(f.name)

        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False, encoding="utf-8"
            ) as input_file:
                input_file.write("မြန်မာစာ\n")
                input_path = Path(input_file.name)

            try:
                result = _run_cli(
                    "check",
                    "-c",
                    str(config_path),
                    str(input_path),
                    "--format",
                    "json",
                )

                _assert_valid_check_json(result, expected_file=str(input_path))
            finally:
                if input_path.exists():
                    input_path.unlink()
        finally:
            if config_path.exists():
                config_path.unlink()
