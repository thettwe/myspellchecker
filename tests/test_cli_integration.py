"""
CLI command integration tests.

Tests covering:
- CLI argument parsing
- Output formatters (JSON, Text, CSV)
- Input/output file handling
- Command invocations (check, build, completion, config)
- Error handling and exit codes
"""

import io
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from myspellchecker.cli import (
    CSVFormatter,
    Formatter,
    JSONFormatter,
    TextFormatter,
    confidence_type,
    generate_completion_script,
    get_checker,
    get_config_from_preset,
    open_input_file,
    open_output_file,
    process_stream,
    validate_build_inputs,
)
from myspellchecker.core.constants import ValidationLevel


class TestArgumentParsing:
    """Test CLI argument parsing and validation."""

    def test_confidence_type_valid_values(self):
        """Test confidence_type with valid float values."""
        assert confidence_type("0.0") == 0.0
        assert confidence_type("1.0") == 1.0
        assert confidence_type("0.5") == 0.5
        assert confidence_type("0.75") == 0.75

    def test_confidence_type_invalid_string(self):
        """Test confidence_type with non-numeric string."""
        import argparse

        with pytest.raises(argparse.ArgumentTypeError, match="Invalid float value"):
            confidence_type("not-a-number")

    def test_confidence_type_out_of_range(self):
        """Test confidence_type with values outside 0-1 range."""
        import argparse

        with pytest.raises(argparse.ArgumentTypeError, match="must be between"):
            confidence_type("1.5")

        with pytest.raises(argparse.ArgumentTypeError, match="must be between"):
            confidence_type("-0.1")


class TestFileHandling:
    """Test input/output file handling functions."""

    def test_open_input_file_none_returns_stdin(self):
        """Test open_input_file returns stdin when path is None."""
        result = open_input_file(None)
        assert result is sys.stdin

    def test_open_input_file_valid_file(self):
        """Test open_input_file opens valid file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("test content")
            path = f.name

        try:
            result = open_input_file(path)
            assert result is not None
            content = result.read()
            assert content == "test content"
            result.close()
        finally:
            Path(path).unlink()

    def test_open_input_file_nonexistent(self):
        """Test open_input_file with nonexistent file exits."""
        with pytest.raises(SystemExit) as exc_info:
            open_input_file("/nonexistent/path/file.txt")
        assert exc_info.value.code == 2

    def test_open_output_file_none_returns_stdout(self):
        """Test open_output_file returns stdout when path is None."""
        result = open_output_file(None)
        assert result is sys.stdout

    def test_open_output_file_valid_path(self):
        """Test open_output_file creates file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "output.txt"

            result = open_output_file(str(path))
            assert result is not None
            result.write("test")
            result.close()

            assert path.exists()
            assert path.read_text() == "test"


class TestOutputFormatters:
    """Test output formatter implementations."""

    def test_formatter_base_class(self):
        """Test Formatter base class methods."""
        formatter = Formatter()
        output = io.StringIO()

        # Base class methods should not raise
        formatter.begin(output)
        formatter.write_result(output, None, 1, "test.txt")
        formatter.end(output, {})

    def test_json_formatter_empty_results(self):
        """Test JSONFormatter with no results."""
        formatter = JSONFormatter()
        output = io.StringIO()

        formatter.end(output, {"total_lines": 0, "total_errors": 0})

        result = json.loads(output.getvalue())
        assert result["summary"]["total_lines"] == 0
        assert result["summary"]["total_errors"] == 0
        assert result["results"] == []

    def test_json_formatter_with_results(self):
        """Test JSONFormatter with results."""
        formatter = JSONFormatter()
        output = io.StringIO()

        # Mock result object
        mock_result = MagicMock()
        mock_result.text = "test text"
        mock_result.has_errors = True
        mock_error = MagicMock()
        mock_error.to_dict.return_value = {"type": "syllable", "text": "test"}
        mock_result.errors = [mock_error]

        formatter.write_result(output, mock_result, 1, "test.txt")
        formatter.end(output, {"total_lines": 1, "total_errors": 1})

        result = json.loads(output.getvalue())
        assert len(result["results"]) == 1
        assert result["results"][0]["file"] == "test.txt"
        assert result["results"][0]["line"] == 1
        assert result["results"][0]["has_errors"] is True

    def test_text_formatter_begin(self):
        """Test TextFormatter begin method outputs warning."""
        formatter = TextFormatter()
        output = io.StringIO()

        formatter.begin(output)

        content = output.getvalue()
        assert "WARNING" in content
        assert "Myanmar" in content

    def test_text_formatter_no_errors(self):
        """Test TextFormatter writes nothing for results without errors."""
        formatter = TextFormatter()
        output = io.StringIO()

        mock_result = MagicMock()
        mock_result.has_errors = False

        formatter.write_result(output, mock_result, 1, "test.txt")

        assert output.getvalue() == ""

    def test_text_formatter_with_errors(self):
        """Test TextFormatter writes errors in grep-like format."""
        formatter = TextFormatter()
        output = io.StringIO()

        mock_result = MagicMock()
        mock_result.has_errors = True
        mock_error = MagicMock()
        mock_error.position = 5
        mock_error.error_type = "syllable"
        mock_error.text = "test"
        mock_error.suggestions = ["suggestion1", "suggestion2"]
        mock_result.errors = [mock_error]

        formatter.write_result(output, mock_result, 1, "test.txt")

        content = output.getvalue()
        assert "test.txt:1:5" in content
        assert "syllable" in content
        assert "suggestion1" in content

    def test_text_formatter_end_summary(self):
        """Test TextFormatter end method outputs summary."""
        formatter = TextFormatter()
        output = io.StringIO()

        formatter.end(output, {"total_errors": 5, "total_lines": 10})

        content = output.getvalue()
        assert "Summary" in content
        assert "5 errors" in content
        assert "10 lines" in content

    def test_csv_formatter_begin(self):
        """Test CSVFormatter begin method outputs header row."""
        formatter = CSVFormatter()
        output = io.StringIO()

        formatter.begin(output)

        content = output.getvalue()
        assert "file,line,position,error_type,text,suggestions" in content

    def test_csv_formatter_sanitize_field(self):
        """Test CSVFormatter sanitizes fields to prevent formula injection."""
        formatter = CSVFormatter()

        # Fields starting with formula characters should be prefixed
        assert formatter._sanitize_csv_field("=SUM(A1)") == "'=SUM(A1)"
        assert formatter._sanitize_csv_field("+1234") == "'+1234"
        assert formatter._sanitize_csv_field("-test") == "'-test"
        assert formatter._sanitize_csv_field("@mention") == "'@mention"

        # Normal fields should be unchanged
        assert formatter._sanitize_csv_field("normal text") == "normal text"
        assert formatter._sanitize_csv_field("") == ""

    def test_csv_formatter_with_errors(self):
        """Test CSVFormatter writes CSV rows for errors."""
        formatter = CSVFormatter()
        output = io.StringIO()

        formatter.begin(output)

        mock_result = MagicMock()
        mock_result.has_errors = True
        mock_error = MagicMock()
        mock_error.position = 5
        mock_error.error_type = "syllable"
        mock_error.text = "test"
        mock_error.suggestions = ["s1", "s2"]
        mock_result.errors = [mock_error]

        formatter.write_result(output, mock_result, 1, "test.txt")

        content = output.getvalue()
        lines = content.strip().split("\n")
        assert len(lines) == 2  # header + data row
        assert "test.txt" in lines[1]


class TestConfigPresets:
    """Test configuration preset loading."""

    def test_get_config_from_preset_default(self):
        """Test loading default preset."""
        config = get_config_from_preset("default")
        assert config is not None

    def test_get_config_from_preset_fast(self):
        """Test loading fast preset."""
        config = get_config_from_preset("fast")
        assert config is not None
        # Fast preset disables phonetic and context
        assert config.use_phonetic is False
        assert config.use_context_checker is False

    def test_get_config_from_preset_accurate(self):
        """Test loading accurate preset."""
        config = get_config_from_preset("accurate")
        assert config is not None

    def test_get_config_from_preset_minimal(self):
        """Test loading minimal preset."""
        config = get_config_from_preset("minimal")
        assert config is not None

    def test_get_config_from_preset_strict(self):
        """Test loading strict preset."""
        config = get_config_from_preset("strict")
        assert config is not None

    def test_get_config_from_preset_invalid(self):
        """Test loading invalid preset raises error."""
        with pytest.raises(ValueError, match="Unknown preset"):
            get_config_from_preset("invalid_preset")


class TestCompletionScripts:
    """Test shell completion script generation."""

    def test_generate_bash_completion(self):
        """Test generating bash completion script."""
        script = generate_completion_script("bash")
        assert "complete -F _myspellchecker_completion myspellchecker" in script
        assert "COMPREPLY" in script
        assert "check" in script
        assert "build" in script

    def test_generate_zsh_completion(self):
        """Test generating zsh completion script."""
        script = generate_completion_script("zsh")
        assert "#compdef myspellchecker" in script
        assert "_myspellchecker" in script

    def test_generate_fish_completion(self):
        """Test generating fish completion script."""
        script = generate_completion_script("fish")
        assert "complete -c myspellchecker" in script
        assert "__fish_use_subcommand" in script

    def test_generate_completion_invalid_shell(self):
        """Test generating completion for unsupported shell."""
        with pytest.raises(ValueError, match="Unsupported shell"):
            generate_completion_script("powershell")


class TestBuildValidation:
    """Test build input validation."""

    def test_validate_build_inputs_no_files(self):
        """Test validation with no input files."""
        result = validate_build_inputs([], "output.db")
        assert result["valid"] is True  # Just warns, doesn't fail
        assert len(result["warnings"]) > 0

    def test_validate_build_inputs_missing_file(self):
        """Test validation with missing input file."""
        result = validate_build_inputs(
            ["/nonexistent/file.txt"],
            "output.db",
        )
        assert result["valid"] is False
        assert any("not found" in err for err in result["errors"])

    def test_validate_build_inputs_valid_file(self):
        """Test validation with valid input file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("test content")
            path = f.name

        try:
            result = validate_build_inputs([path], "output.db")
            assert result["valid"] is True
            assert result["stats"]["total_files"] == 1
        finally:
            Path(path).unlink()

    def test_validate_build_inputs_empty_file_warning(self):
        """Test validation warns about empty files."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            # Write nothing - empty file
            path = f.name

        try:
            result = validate_build_inputs([path], "output.db")
            assert any("Empty file" in warn for warn in result["warnings"])
        finally:
            Path(path).unlink()

    def test_validate_build_inputs_output_exists_warning(self):
        """Test validation warns when output file exists."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".db", delete=False) as f:
            output_path = f.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("content")
            input_path = f.name

        try:
            result = validate_build_inputs([input_path], output_path)
            assert any("will be overwritten" in warn for warn in result["warnings"])
        finally:
            Path(output_path).unlink()
            Path(input_path).unlink()

    def test_validate_build_inputs_invalid_output_dir(self):
        """Test validation fails for invalid output directory."""
        result = validate_build_inputs(
            [],
            "/nonexistent/directory/output.db",
        )
        assert result["valid"] is False
        assert any("directory" in err.lower() for err in result["errors"])


class TestGetChecker:
    """Test SpellChecker initialization through CLI."""

    def test_get_checker_default(self):
        """Test get_checker with default settings."""
        # May use MemoryProvider if no default database exists
        checker = get_checker()
        assert checker is not None
        checker.close()

    def test_get_checker_with_flags(self):
        """Test get_checker with feature flags disabled."""
        checker = get_checker(no_phonetic=True, no_context=True)
        assert checker is not None
        checker.close()

    def test_get_checker_with_preset(self):
        """Test get_checker with preset."""
        checker = get_checker(preset="fast")
        assert checker is not None
        checker.close()

    def test_get_checker_invalid_db_exits(self):
        """Test get_checker with invalid database path exits."""
        with pytest.raises(SystemExit) as exc_info:
            get_checker(database_path="/nonexistent/path/db.sqlite")
        assert exc_info.value.code == 1


class TestProcessStream:
    """Test stream processing function."""

    def test_process_stream_empty_input(self):
        """Test processing empty input stream."""
        input_stream = io.StringIO("")
        output_stream = io.StringIO()

        checker = get_checker(preset="fast")
        formatter = JSONFormatter()

        try:
            stats = process_stream(
                input_stream,
                "test.txt",
                checker,
                formatter,
                output_stream,
                ValidationLevel.SYLLABLE,
            )

            assert stats["total_lines"] == 0
            assert stats["total_errors"] == 0
        finally:
            checker.close()

    def test_process_stream_blank_lines_skipped(self):
        """Test that blank lines are skipped."""
        input_stream = io.StringIO("\n\n\n")
        output_stream = io.StringIO()

        checker = get_checker(preset="fast")
        formatter = JSONFormatter()

        try:
            stats = process_stream(
                input_stream,
                "test.txt",
                checker,
                formatter,
                output_stream,
                ValidationLevel.SYLLABLE,
            )

            assert stats["total_lines"] == 0
        finally:
            checker.close()

    def test_process_stream_with_content(self):
        """Test processing stream with content."""
        input_stream = io.StringIO("test line 1\ntest line 2")
        output_stream = io.StringIO()

        checker = get_checker(preset="fast")
        formatter = JSONFormatter()

        try:
            stats = process_stream(
                input_stream,
                "test.txt",
                checker,
                formatter,
                output_stream,
                ValidationLevel.SYLLABLE,
            )

            assert stats["total_lines"] == 2
        finally:
            checker.close()


class TestCLIMain:
    """Test CLI main function invocation."""

    def test_main_help(self):
        """Test main with --help doesn't crash."""
        from myspellchecker.cli import main

        with patch.object(sys, "argv", ["myspellchecker", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            # Help exits with code 0
            assert exc_info.value.code == 0

    def test_main_check_help(self):
        """Test main check --help doesn't crash."""
        from myspellchecker.cli import main

        with patch.object(sys, "argv", ["myspellchecker", "check", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

    def test_main_build_help(self):
        """Test main build --help doesn't crash."""
        from myspellchecker.cli import main

        with patch.object(sys, "argv", ["myspellchecker", "build", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

    def test_main_completion_bash(self):
        """Test main completion --shell bash."""
        from myspellchecker.cli import main

        with patch.object(sys, "argv", ["myspellchecker", "completion", "--shell", "bash"]):
            with patch("builtins.print") as mock_print:
                main()
                # Should print bash completion script
                output = mock_print.call_args[0][0]
                assert "complete" in output

    def test_main_completion_zsh(self):
        """Test main completion --shell zsh."""
        from myspellchecker.cli import main

        with patch.object(sys, "argv", ["myspellchecker", "completion", "--shell", "zsh"]):
            with patch("builtins.print") as mock_print:
                main()
                output = mock_print.call_args[0][0]
                assert "#compdef" in output

    def test_main_completion_fish(self):
        """Test main completion --shell fish."""
        from myspellchecker.cli import main

        with patch.object(sys, "argv", ["myspellchecker", "completion", "--shell", "fish"]):
            with patch("builtins.print") as mock_print:
                main()
                output = mock_print.call_args[0][0]
                assert "complete -c myspellchecker" in output

    def test_main_config_show(self):
        """Test main config show command."""
        from myspellchecker.cli import main

        with patch.object(sys, "argv", ["myspellchecker", "config", "show"]):
            with patch("builtins.print") as mock_print:
                main()
                # Should print configuration search paths
                calls = [str(call) for call in mock_print.call_args_list]
                call_text = " ".join(calls)
                assert "Configuration" in call_text or "Search" in call_text


class TestCLICheckCommand:
    """Test CLI check command with various options."""

    def test_check_with_file_json_output(self):
        """Test check command with file input and JSON output."""
        from myspellchecker.cli import main

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("test text")
            input_path = f.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            output_path = f.name

        try:
            with patch.object(
                sys,
                "argv",
                ["myspellchecker", "check", input_path, "-o", output_path, "-f", "json"],
            ):
                main()

            # Verify output file was created and contains valid JSON
            with open(output_path) as f:
                result = json.load(f)
                assert "summary" in result
                assert "results" in result
        finally:
            Path(input_path).unlink()
            Path(output_path).unlink()

    def test_check_with_preset_fast(self):
        """Test check command with fast preset."""
        from myspellchecker.cli import main

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("test")
            input_path = f.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            output_path = f.name

        try:
            with patch.object(
                sys,
                "argv",
                [
                    "myspellchecker",
                    "check",
                    input_path,
                    "-o",
                    output_path,
                    "-f",
                    "json",
                    "--preset",
                    "fast",
                ],
            ):
                main()

            assert Path(output_path).exists()
        finally:
            Path(input_path).unlink()
            Path(output_path).unlink()

    def test_check_with_text_format(self):
        """Test check command with text output format."""
        from myspellchecker.cli import main

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("test")
            input_path = f.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            output_path = f.name

        try:
            with patch.object(
                sys,
                "argv",
                ["myspellchecker", "check", input_path, "-o", output_path, "-f", "text"],
            ):
                main()

            content = Path(output_path).read_text()
            assert "WARNING" in content  # TextFormatter outputs warning
        finally:
            Path(input_path).unlink()
            Path(output_path).unlink()

    def test_check_with_csv_format(self):
        """Test check command with CSV output format."""
        from myspellchecker.cli import main

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("test")
            input_path = f.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            output_path = f.name

        try:
            with patch.object(
                sys,
                "argv",
                ["myspellchecker", "check", input_path, "-o", output_path, "-f", "csv"],
            ):
                main()

            content = Path(output_path).read_text()
            assert "file,line,position" in content  # CSV header
        finally:
            Path(input_path).unlink()
            Path(output_path).unlink()


class TestCLIBuildCommand:
    """Test CLI build command."""

    def test_build_validate_only(self):
        """Test build command with --validate flag."""
        from myspellchecker.cli import main

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("test content")
            input_path = f.name

        try:
            with patch.object(
                sys,
                "argv",
                ["myspellchecker", "build", "-i", input_path, "--validate"],
            ):
                with pytest.raises(SystemExit) as exc_info:
                    main()
                # Validation should pass and exit with 0
                assert exc_info.value.code == 0
        finally:
            Path(input_path).unlink()

    def test_build_validate_missing_file(self):
        """Test build validate with missing input file."""
        from myspellchecker.cli import main

        with patch.object(
            sys,
            "argv",
            ["myspellchecker", "build", "-i", "/nonexistent/file.txt", "--validate"],
        ):
            with pytest.raises(SystemExit) as exc_info:
                main()
            # Validation should fail
            assert exc_info.value.code == 1


class TestCLISegmentCommand:
    """Test CLI segment command."""

    def test_segment_help(self):
        """Test segment --help."""
        from myspellchecker.cli import main

        with patch.object(sys, "argv", ["myspellchecker", "segment", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0


class TestCLIConfigCommand:
    """Test CLI config command."""

    def test_config_init(self):
        """Test config init creates configuration file."""
        from myspellchecker.cli import main

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test_config.yml"

            with patch.object(
                sys,
                "argv",
                ["myspellchecker", "config", "init", "--path", str(config_path)],
            ):
                main()

            assert config_path.exists()

    def test_config_init_existing_no_force(self):
        """Test config init doesn't overwrite without --force."""
        from myspellchecker.cli import main

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write("existing content")
            config_path = f.name

        try:
            with patch.object(
                sys,
                "argv",
                ["myspellchecker", "config", "init", "--path", config_path],
            ):
                # init_config_file raises FileExistsError which isn't caught
                # as ConfigurationError in the CLI, so it propagates up
                with pytest.raises((SystemExit, FileExistsError)):
                    main()

            # File should still have original content
            content = Path(config_path).read_text()
            assert "existing content" in content
        finally:
            Path(config_path).unlink()

    def test_config_init_existing_with_force(self):
        """Test config init overwrites with --force."""
        from myspellchecker.cli import main

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write("existing content")
            config_path = f.name

        try:
            with patch.object(
                sys,
                "argv",
                ["myspellchecker", "config", "init", "--path", config_path, "--force"],
            ):
                main()

            # File should be overwritten
            content = Path(config_path).read_text()
            assert "existing content" not in content
        finally:
            Path(config_path).unlink()


class TestCLIErrorHandling:
    """Test CLI error handling and exit codes."""

    def test_invalid_input_file_exits_with_code_2(self):
        """Test that invalid input file exits with code 2."""
        with pytest.raises(SystemExit) as exc_info:
            open_input_file("/nonexistent/file.txt")
        assert exc_info.value.code == 2

    def test_invalid_db_exits_with_code_1(self):
        """Test that invalid database exits with code 1."""
        with pytest.raises(SystemExit) as exc_info:
            get_checker(database_path="/nonexistent/db.sqlite")
        assert exc_info.value.code == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
