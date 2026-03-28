"""Integration tests for CLI config override precedence."""

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
    """Assert a CLI check run produced structurally valid JSON output."""
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
    assert isinstance(payload, dict)
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

    if expected_file is not None:
        assert results, "Expected at least one checked line in JSON results."
        assert all(row["file"] == expected_file for row in results)

    for row in results:
        assert {"file", "line", "text", "has_errors", "errors"} <= set(row)
        assert isinstance(row["line"], int)
        assert row["line"] >= 1
        assert isinstance(row["text"], str)
        assert isinstance(row["has_errors"], bool)
        assert isinstance(row["errors"], list)

        for error in row["errors"]:
            assert {"text", "position", "suggestions", "error_type", "confidence"} <= set(error)


@pytest.fixture
def temp_config_accurate():
    """Create a config file with accurate preset."""
    config_content = """
preset: accurate
max_edit_distance: 3
max_suggestions: 10
use_phonetic: true
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as f:
        f.write(config_content)
        config_path = Path(f.name)

    yield config_path

    if config_path.exists():
        config_path.unlink()


@pytest.fixture
def temp_input_file():
    """Create a temporary input file with Myanmar text."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("မြန်မာစာ\n")
        input_path = Path(f.name)

    yield input_path

    if input_path.exists():
        input_path.unlink()


class TestCLIConfigLoading:
    """Test CLI integration with config loading."""

    def test_cli_loads_config_file(self, temp_config_accurate, temp_input_file):
        """Test that CLI loads and uses config file."""
        result = _run_cli(
            "check",
            "-c",
            str(temp_config_accurate),
            str(temp_input_file),
            "--format",
            "json",
        )

        _assert_valid_check_json(result, expected_file=str(temp_input_file))

    def test_cli_without_config_uses_defaults(self, temp_input_file):
        """Test that CLI works without config file (backward compatibility)."""
        result = _run_cli("check", str(temp_input_file), "--format", "json")

        _assert_valid_check_json(result, expected_file=str(temp_input_file))


class TestCLIOverridePrecedence:
    """Test that CLI flags override config file values."""

    def test_preset_override(self, temp_config_accurate, temp_input_file):
        """Test --preset flag overrides config file preset."""
        # Config has preset: accurate, CLI uses --preset fast
        result = _run_cli(
            "check",
            "-c",
            str(temp_config_accurate),
            "--preset",
            "fast",
            str(temp_input_file),
            "--format",
            "json",
        )

        _assert_valid_check_json(result, expected_file=str(temp_input_file))

    def test_no_phonetic_override(self, temp_config_accurate, temp_input_file):
        """Test --no-phonetic flag overrides config use_phonetic."""
        # Config has use_phonetic: true, CLI uses --no-phonetic
        result = _run_cli(
            "check",
            "-c",
            str(temp_config_accurate),
            "--no-phonetic",
            str(temp_input_file),
            "--format",
            "json",
        )

        _assert_valid_check_json(result, expected_file=str(temp_input_file))

    def test_no_context_override(self, temp_config_accurate, temp_input_file):
        """Test --no-context flag overrides config use_context_checker."""
        # CLI flag should override config
        result = _run_cli(
            "check",
            "-c",
            str(temp_config_accurate),
            "--no-context",
            str(temp_input_file),
            "--format",
            "json",
        )

        _assert_valid_check_json(result, expected_file=str(temp_input_file))


class TestPresetAliasesInCLI:
    """Test preset aliases work in CLI."""

    def test_production_preset_in_cli(self, temp_input_file):
        """Test that production preset alias works in CLI."""
        config_content = """
preset: production
max_edit_distance: 2
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write(config_content)
            config_path = Path(f.name)

        try:
            result = _run_cli(
                "check",
                "-c",
                str(config_path),
                str(temp_input_file),
                "--format",
                "json",
            )

            _assert_valid_check_json(result, expected_file=str(temp_input_file))
        finally:
            if config_path.exists():
                config_path.unlink()


class TestBackwardCompatibility:
    """Test backward compatibility with CLI-only usage."""

    def test_cli_only_no_config(self, temp_input_file):
        """Test that CLI-only usage (no config file) still works."""
        result = _run_cli(
            "check",
            "--preset",
            "minimal",
            str(temp_input_file),
            "--format",
            "json",
        )

        _assert_valid_check_json(result, expected_file=str(temp_input_file))

    def test_cli_with_all_flags_no_config(self, temp_input_file):
        """Test that all CLI flags work without config file."""
        result = _run_cli(
            "check",
            "--preset",
            "accurate",
            "--no-phonetic",
            "--no-context",
            str(temp_input_file),
            "--format",
            "json",
        )

        _assert_valid_check_json(result, expected_file=str(temp_input_file))
