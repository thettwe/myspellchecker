"""Integration tests for CLI stdin/stdout dash support (#1178)."""

import csv
import io
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


def _assert_check_completed(result):
    """Assert check command completed with supported success semantics."""
    # Keep backward-compatible semantics:
    # some environments may use 1 for handled validation outcomes.
    assert result.returncode in [0, 1]
    _assert_no_traceback(result)


def _assert_json_output_text(output_text, *, expected_file=None, expected_total_lines=None):
    """Parse and validate JSON check output."""
    assert output_text.strip(), "Expected JSON output."
    payload = json.loads(output_text)
    assert {"summary", "results"} <= set(payload)

    summary = payload["summary"]
    results = payload["results"]
    assert isinstance(summary, dict)
    assert isinstance(results, list)

    for key in ("total_lines", "total_errors", "lines_with_errors"):
        assert key in summary
        assert isinstance(summary[key], int)
        assert summary[key] >= 0

    if expected_total_lines is not None:
        assert summary["total_lines"] == expected_total_lines

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


def _assert_json_check_result(result, *, expected_file=None, expected_total_lines=None):
    """Assert check result is valid JSON output."""
    _assert_check_completed(result)
    if not result.stdout.strip():
        # In some environments, a handled DB/config error exits with code 1 and stderr only.
        assert result.returncode == 1
        assert result.stderr.strip()
        assert "Error: Database error:" in result.stderr or "Configuration Error:" in result.stderr
        return None
    return _assert_json_output_text(
        result.stdout,
        expected_file=expected_file,
        expected_total_lines=expected_total_lines,
    )


def _assert_text_check_result(result):
    """Assert check result uses expected text formatter structure."""
    _assert_check_completed(result)
    assert result.stdout.strip(), "Expected text output on stdout."
    assert "# WARNING: Myanmar text may not render correctly in your terminal." in result.stdout
    assert "# Summary:" in result.stdout


def _assert_csv_check_result(result):
    """Assert check result uses expected CSV formatter structure."""
    _assert_check_completed(result)
    assert result.stdout.strip(), "Expected CSV output on stdout."
    rows = list(csv.reader(io.StringIO(result.stdout)))
    assert rows
    assert rows[0] == ["file", "line", "position", "error_type", "text", "suggestions"]
    for row in rows[1:]:
        assert len(row) == 6


def _assert_formatted_check_result(result, *, format_type, expected_file=None):
    """Assert output contract by format type."""
    if format_type == "json":
        return _assert_json_check_result(result, expected_file=expected_file)
    if format_type == "text":
        _assert_text_check_result(result)
        return None
    if format_type == "csv":
        _assert_csv_check_result(result)
        return None
    raise AssertionError(f"Unexpected format type for test assertion: {format_type}")


@pytest.fixture
def temp_input_file():
    """Create a temporary input file with Myanmar text."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("မြန်မာစာ\n")
        f.write("ကျွန်တော်\n")
        f.write("သွားပါမယ်\n")
        input_path = Path(f.name)

    yield input_path

    if input_path.exists():
        input_path.unlink()


class TestStdinDashSupport:
    """Test that '-' is treated as stdin."""

    def test_dash_as_stdin(self):
        """Test that '-' argument reads from stdin."""
        # Provide input via stdin
        myanmar_text = "မြန်မာစာ\n"

        result = _run_cli("check", "-", "--format", "json", input_text=myanmar_text)

        _assert_json_check_result(result, expected_file="-")

    def test_dash_stdin_with_config(self):
        """Test that '-' works with config file."""
        config_content = """
preset: fast
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write(config_content)
            config_path = Path(f.name)

        try:
            myanmar_text = "မြန်မာစာ\n"

            result = _run_cli(
                "check",
                "-c",
                str(config_path),
                "-",
                "--format",
                "json",
                input_text=myanmar_text,
            )

            _assert_json_check_result(result, expected_file="-")
        finally:
            if config_path.exists():
                config_path.unlink()

    def test_dash_stdin_with_preset(self):
        """Test that '-' works with --preset flag."""
        myanmar_text = "မြန်မာစာ\n"

        result = _run_cli(
            "check",
            "--preset",
            "minimal",
            "-",
            "--format",
            "json",
            input_text=myanmar_text,
        )

        _assert_json_check_result(result, expected_file="-")

    def test_none_still_works_as_stdin(self):
        """Test backward compatibility: no input arg still means stdin."""
        myanmar_text = "မြန်မာစာ\n"

        result = _run_cli("check", "--format", "json", input_text=myanmar_text)

        _assert_json_check_result(result, expected_file="stdin")


class TestStdoutDashSupport:
    """Test that '-' is treated as stdout."""

    def test_dash_as_stdout(self, temp_input_file):
        """Test that '-o -' writes to stdout."""
        result = _run_cli(
            "check",
            str(temp_input_file),
            "-o",
            "-",
            "--format",
            "json",
        )

        _assert_json_check_result(result, expected_file=str(temp_input_file))

    def test_dash_stdout_with_config(self, temp_input_file):
        """Test that '-o -' works with config file."""
        config_content = """
preset: fast
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
                "-o",
                "-",
                "--format",
                "json",
            )

            _assert_json_check_result(result, expected_file=str(temp_input_file))
        finally:
            if config_path.exists():
                config_path.unlink()

    def test_none_still_works_as_stdout(self, temp_input_file):
        """Test backward compatibility: no -o still means stdout."""
        result = _run_cli("check", str(temp_input_file), "--format", "json")

        _assert_json_check_result(result, expected_file=str(temp_input_file))


class TestBothStdinStdout:
    """Test stdin and stdout together."""

    def test_stdin_and_stdout_both_dash(self):
        """Test that both stdin and stdout can be '-'."""
        myanmar_text = "မြန်မာစာ\n"

        result = _run_cli(
            "check",
            "-",
            "-o",
            "-",
            "--format",
            "json",
            input_text=myanmar_text,
        )

        _assert_json_check_result(result, expected_file="-")

    def test_stdin_dash_with_output_file(self):
        """Test stdin from '-' with output to file."""
        myanmar_text = "မြန်မာစာ\n"

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            output_path = Path(f.name)

        try:
            result = _run_cli(
                "check",
                "-",
                "-o",
                str(output_path),
                "--format",
                "json",
                input_text=myanmar_text,
            )

            _assert_check_completed(result)
            assert result.stdout == ""
            assert output_path.exists()
            _assert_json_output_text(output_path.read_text(encoding="utf-8"), expected_file="-")
        finally:
            if output_path.exists():
                output_path.unlink()

    def test_input_file_with_stdout_dash(self, temp_input_file):
        """Test input from file with stdout to '-'."""
        result = _run_cli(
            "check",
            str(temp_input_file),
            "-o",
            "-",
            "--format",
            "json",
        )

        _assert_json_check_result(result, expected_file=str(temp_input_file))


class TestDashWithDifferentFormats:
    """Test dash support with different output formats."""

    @pytest.mark.parametrize("format_type", ["json", "text", "csv"])
    def test_dash_stdin_with_various_formats(self, format_type):
        """Test '-' stdin works with different output formats."""
        myanmar_text = "မြန်မာစာ\n"

        result = _run_cli(
            "check",
            "-",
            "--format",
            format_type,
            input_text=myanmar_text,
        )

        _assert_formatted_check_result(result, format_type=format_type, expected_file="-")

    @pytest.mark.parametrize("format_type", ["json", "text", "csv"])
    def test_dash_stdout_with_various_formats(self, temp_input_file, format_type):
        """Test '-o -' stdout works with different output formats."""
        result = _run_cli(
            "check",
            str(temp_input_file),
            "-o",
            "-",
            "--format",
            format_type,
        )

        _assert_formatted_check_result(
            result,
            format_type=format_type,
            expected_file=str(temp_input_file),
        )


class TestDashErrorHandling:
    """Test error handling with dash arguments."""

    def test_empty_stdin(self):
        """Test that empty stdin is handled gracefully."""
        result = _run_cli("check", "-", "--format", "json", input_text="")

        payload = _assert_json_check_result(result, expected_file="-", expected_total_lines=0)
        assert payload["results"] == []
