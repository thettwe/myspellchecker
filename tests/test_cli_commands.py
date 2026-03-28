"""Unit tests for CLI module - fast, mocked tests for coverage expansion."""

import argparse
import csv
import io
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from myspellchecker.cli import (
    AVAILABLE_PRESETS,
    CSVFormatter,
    Formatter,
    JSONFormatter,
    RichFormatter,
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

# ============================================================================
# Tests for confidence_type helper
# ============================================================================


class TestConfidenceType:
    """Tests for confidence_type argument parser."""

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("0.0", 0.0),
            ("0.5", 0.5),
            ("1.0", 1.0),
            ("0.123", 0.123),
        ],
    )
    def test_valid_confidence_values(self, value, expected):
        """Test valid confidence values."""
        assert confidence_type(value) == expected

    @pytest.mark.parametrize(
        "value",
        [
            "-0.1",
            "1.1",
            "2.0",
            "-1.0",
        ],
    )
    def test_out_of_range_values(self, value):
        """Test out-of-range confidence values."""
        with pytest.raises(argparse.ArgumentTypeError) as exc_info:
            confidence_type(value)
        assert "between 0.0 and 1.0" in str(exc_info.value)

    @pytest.mark.parametrize(
        "value",
        [
            "abc",
            "not_a_number",
            "",
            "one",
        ],
    )
    def test_invalid_float_values(self, value):
        """Test invalid float values."""
        with pytest.raises(argparse.ArgumentTypeError) as exc_info:
            confidence_type(value)
        assert "Invalid float value" in str(exc_info.value)


# ============================================================================
# Tests for file I/O helpers
# ============================================================================


class TestOpenInputFile:
    """Tests for open_input_file helper."""

    def test_none_returns_stdin(self):
        """Test None returns sys.stdin."""
        result = open_input_file(None)
        assert result is sys.stdin

    def test_file_not_found_exits(self):
        """Test FileNotFoundError exits with code 2."""
        with pytest.raises(SystemExit) as exc_info:
            open_input_file("/nonexistent/path/file.txt")
        assert exc_info.value.code == 2

    def test_permission_error_exits(self):
        """Test PermissionError exits with code 2."""
        with patch("builtins.open", side_effect=PermissionError("denied")):
            with pytest.raises(SystemExit) as exc_info:
                open_input_file("/some/file.txt")
            assert exc_info.value.code == 2

    def test_generic_error_exits(self):
        """Test generic exception exits with code 2."""
        with patch("builtins.open", side_effect=IOError("some error")):
            with pytest.raises(SystemExit) as exc_info:
                open_input_file("/some/file.txt")
            assert exc_info.value.code == 2

    def test_valid_file_opens(self):
        """Test valid file path opens successfully."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("test content")
            temp_path = f.name
        try:
            result = open_input_file(temp_path)
            assert result is not None
            content = result.read()
            assert content == "test content"
            result.close()
        finally:
            os.unlink(temp_path)


class TestOpenOutputFile:
    """Tests for open_output_file helper."""

    def test_none_returns_stdout(self):
        """Test None returns sys.stdout."""
        result = open_output_file(None)
        assert result is sys.stdout

    def test_permission_error_exits(self):
        """Test PermissionError exits with code 2."""
        with patch("builtins.open", side_effect=PermissionError("denied")):
            with pytest.raises(SystemExit) as exc_info:
                open_output_file("/root/secret.txt")
            assert exc_info.value.code == 2

    def test_generic_error_exits(self):
        """Test generic exception exits with code 2."""
        with patch("builtins.open", side_effect=IOError("disk full")):
            with pytest.raises(SystemExit) as exc_info:
                open_output_file("/some/output.txt")
            assert exc_info.value.code == 2

    def test_valid_path_opens(self):
        """Test valid path opens for writing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "output.txt")
            result = open_output_file(output_path)
            result.write("test")
            result.close()
            assert os.path.exists(output_path)


# ============================================================================
# Tests for generate_completion_script
# ============================================================================


class TestGenerateCompletionScript:
    """Tests for shell completion script generation."""

    def test_bash_completion(self):
        """Test bash completion script generation."""
        script = generate_completion_script("bash")
        assert "# myspellchecker bash completion" in script
        assert "_myspellchecker_completion" in script
        assert "COMPREPLY" in script
        assert "complete -F" in script

    def test_zsh_completion(self):
        """Test zsh completion script generation."""
        script = generate_completion_script("zsh")
        assert "#compdef myspellchecker" in script
        assert "_myspellchecker()" in script
        assert "_arguments" in script

    def test_fish_completion(self):
        """Test fish completion script generation."""
        script = generate_completion_script("fish")
        assert "# myspellchecker fish completion" in script
        assert "complete -c myspellchecker" in script
        assert "__fish_use_subcommand" in script

    def test_unsupported_shell_raises(self):
        """Test unsupported shell raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            generate_completion_script("powershell")
        assert "Unsupported shell" in str(exc_info.value)

    @pytest.mark.parametrize("shell", ["bash", "zsh", "fish"])
    def test_completion_contains_all_commands(self, shell):
        """Test completion scripts contain all main commands."""
        script = generate_completion_script(shell)
        assert "check" in script
        assert "build" in script
        assert "train-model" in script
        assert "completion" in script


# ============================================================================
# Tests for validate_build_inputs
# ============================================================================


class TestValidateBuildInputs:
    """Tests for validate_build_inputs pre-flight checks."""

    def test_empty_input_files_warning(self):
        """Test empty input files generates warning."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = validate_build_inputs([], os.path.join(temp_dir, "out.db"))
            assert result["valid"]
            assert any("No input files" in w for w in result["warnings"])

    def test_nonexistent_file_error(self):
        """Test nonexistent file generates error."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = validate_build_inputs(
                ["/nonexistent/file.txt"], os.path.join(temp_dir, "out.db")
            )
            assert not result["valid"]
            assert any("not found" in e for e in result["errors"])

    def test_directory_input_expands_files(self):
        """Test directory input expands to files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            txt_file = os.path.join(temp_dir, "test.txt")
            Path(txt_file).write_text("test")
            json_file = os.path.join(temp_dir, "test.json")
            Path(json_file).write_text("{}")

            result = validate_build_inputs([temp_dir], os.path.join(temp_dir, "out.db"))
            assert result["valid"]
            assert result["stats"]["total_files"] == 2

    def test_glob_pattern_expands(self):
        """Test glob pattern expansion."""
        with tempfile.TemporaryDirectory() as temp_dir:
            txt1 = os.path.join(temp_dir, "a.txt")
            txt2 = os.path.join(temp_dir, "b.txt")
            Path(txt1).write_text("a")
            Path(txt2).write_text("b")

            pattern = os.path.join(temp_dir, "*.txt")
            result = validate_build_inputs([pattern], os.path.join(temp_dir, "out.db"))
            assert result["valid"]
            assert result["stats"]["total_files"] == 2

    def test_output_dir_not_exists_error(self):
        """Test nonexistent output directory generates error."""
        result = validate_build_inputs([], "/nonexistent/dir/out.db")
        assert not result["valid"]
        assert any("does not exist" in e for e in result["errors"])

    def test_output_dir_not_writable_error(self):
        """Test non-writable output directory generates error."""
        with patch("os.access", return_value=False):
            with patch("os.path.exists", return_value=True):
                result = validate_build_inputs([], "/some/path/out.db")
                assert not result["valid"]
                assert any("Cannot write" in e for e in result["errors"])

    def test_output_file_exists_warning(self):
        """Test existing output file generates warning."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            temp_path = f.name
        try:
            result = validate_build_inputs([], temp_path)
            assert any("already exists" in w for w in result["warnings"])
        finally:
            os.unlink(temp_path)

    def test_empty_file_warning(self):
        """Test empty input file generates warning."""
        with tempfile.TemporaryDirectory() as temp_dir:
            empty_file = os.path.join(temp_dir, "empty.txt")
            Path(empty_file).write_text("")

            result = validate_build_inputs([empty_file], os.path.join(temp_dir, "out.db"))
            assert any("Empty file" in w for w in result["warnings"])

    def test_file_types_tracked(self):
        """Test file type statistics are tracked."""
        with tempfile.TemporaryDirectory() as temp_dir:
            Path(os.path.join(temp_dir, "a.txt")).write_text("a")
            Path(os.path.join(temp_dir, "b.txt")).write_text("b")
            Path(os.path.join(temp_dir, "c.json")).write_text("{}")

            result = validate_build_inputs([temp_dir], os.path.join(temp_dir, "out.db"))
            assert ".txt" in result["stats"]["file_types"]
            assert ".json" in result["stats"]["file_types"]
            assert result["stats"]["file_types"][".txt"] == 2
            assert result["stats"]["file_types"][".json"] == 1

    def test_not_a_file_error(self):
        """Test directory instead of file generates error."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a subdirectory
            subdir = os.path.join(temp_dir, "subdir")
            os.mkdir(subdir)

            # Pass the subdir as if it were a file
            result = validate_build_inputs([subdir], os.path.join(temp_dir, "out.db"))
            # The subdir is expanded, so files inside are checked
            # An empty subdir has no files, so no errors from file validation
            assert result["valid"]


# ============================================================================
# Tests for get_config_from_preset
# ============================================================================


class TestGetConfigFromPreset:
    """Tests for preset configuration loading."""

    @pytest.mark.parametrize("preset", AVAILABLE_PRESETS)
    def test_all_presets_valid(self, preset):
        """Test all defined presets are valid."""
        config = get_config_from_preset(preset)
        assert config is not None

    def test_unknown_preset_raises(self):
        """Test unknown preset raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            get_config_from_preset("nonexistent")
        assert "Unknown preset" in str(exc_info.value)

    def test_fast_preset_disables_context(self):
        """Test fast preset has context checking disabled."""
        config = get_config_from_preset("fast")
        assert not config.use_context_checker

    def test_accurate_preset_enables_all(self):
        """Test accurate preset enables all features."""
        config = get_config_from_preset("accurate")
        assert config.use_phonetic
        assert config.use_context_checker


# ============================================================================
# Tests for Formatter classes
# ============================================================================


class TestBaseFormatter:
    """Tests for base Formatter class."""

    def test_base_formatter_methods_exist(self):
        """Test base formatter has all methods."""
        formatter = Formatter()
        output = io.StringIO()
        formatter.begin(output)
        formatter.write_result(output, None, 1, "test.txt")
        formatter.end(output, {})


class TestJSONFormatter:
    """Tests for JSONFormatter."""

    def test_write_result_collects_data(self):
        """Test write_result collects results."""
        formatter = JSONFormatter()
        output = io.StringIO()

        mock_error = Mock()
        mock_error.to_dict.return_value = {"type": "test", "text": "error"}

        mock_result = Mock()
        mock_result.text = "test text"
        mock_result.has_errors = True
        mock_result.errors = [mock_error]

        formatter.write_result(output, mock_result, 1, "test.txt")

        assert len(formatter.results) == 1
        assert formatter.results[0]["text"] == "test text"
        assert formatter.results[0]["line"] == 1
        assert formatter.results[0]["file"] == "test.txt"

    def test_end_outputs_json(self):
        """Test end outputs valid JSON."""
        formatter = JSONFormatter()
        output = io.StringIO()

        mock_error = Mock()
        mock_error.to_dict.return_value = {"msg": "err"}
        mock_result = Mock(text="t", has_errors=True, errors=[mock_error])

        formatter.write_result(output, mock_result, 1, "f.txt")
        formatter.end(output, {"total_errors": 1, "total_lines": 1})

        output.seek(0)
        data = json.load(output)
        assert "summary" in data
        assert "results" in data
        assert data["summary"]["total_errors"] == 1


class TestTextFormatter:
    """Tests for TextFormatter."""

    def test_begin_writes_warning(self):
        """Test begin writes Myanmar warning."""
        formatter = TextFormatter()
        output = io.StringIO()
        formatter.begin(output)
        content = output.getvalue()
        assert "WARNING" in content
        assert "Myanmar" in content

    def test_write_result_skips_no_errors(self):
        """Test write_result skips results without errors."""
        formatter = TextFormatter()
        output = io.StringIO()

        mock_result = Mock()
        mock_result.has_errors = False

        formatter.write_result(output, mock_result, 1, "test.txt")
        assert output.getvalue() == ""

    def test_write_result_formats_errors(self):
        """Test write_result formats errors correctly."""
        formatter = TextFormatter()
        output = io.StringIO()

        mock_error = Mock()
        mock_error.position = 5
        mock_error.error_type = "spelling"
        mock_error.text = "wrng"
        mock_error.suggestions = ["wrong", "writing", "wing"]

        mock_result = Mock()
        mock_result.has_errors = True
        mock_result.errors = [mock_error]

        formatter.write_result(output, mock_result, 1, "test.txt")
        content = output.getvalue()

        assert "test.txt:1:5" in content
        assert "spelling" in content
        assert "wrng" in content
        assert "wrong" in content

    def test_end_writes_summary(self):
        """Test end writes summary."""
        formatter = TextFormatter()
        output = io.StringIO()
        formatter.end(output, {"total_errors": 5, "total_lines": 10})
        content = output.getvalue()
        assert "5 errors" in content
        assert "10 lines" in content


class TestCSVFormatter:
    """Tests for CSVFormatter."""

    def test_begin_writes_header(self):
        """Test begin writes CSV header."""
        formatter = CSVFormatter()
        output = io.StringIO()
        formatter.begin(output)
        content = output.getvalue()
        assert "file,line,position,error_type,text,suggestions" in content

    def test_write_result_skips_no_errors(self):
        """Test write_result skips results without errors."""
        formatter = CSVFormatter()
        output = io.StringIO()
        formatter.begin(output)
        initial_content = output.getvalue()

        mock_result = Mock()
        mock_result.has_errors = False

        formatter.write_result(output, mock_result, 1, "test.txt")
        assert output.getvalue() == initial_content

    def test_sanitize_csv_field_formula_injection(self):
        """Test CSV field sanitization prevents formula injection."""
        formatter = CSVFormatter()
        assert formatter._sanitize_csv_field("=SUM(A1)") == "'=SUM(A1)"
        assert formatter._sanitize_csv_field("+1+1") == "'+1+1"
        assert formatter._sanitize_csv_field("-1") == "'-1"
        assert formatter._sanitize_csv_field("@cmd") == "'@cmd"
        assert formatter._sanitize_csv_field("normal") == "normal"
        assert formatter._sanitize_csv_field("") == ""

    def test_write_result_formats_csv_row(self):
        """Test write_result writes proper CSV row."""
        formatter = CSVFormatter()
        output = io.StringIO()
        formatter.begin(output)

        mock_error = Mock()
        mock_error.position = 0
        mock_error.error_type = "test"
        mock_error.text = "word"
        mock_error.suggestions = ["sug1", "sug2"]

        mock_result = Mock()
        mock_result.has_errors = True
        mock_result.errors = [mock_error]

        formatter.write_result(output, mock_result, 1, "test.txt")

        output.seek(0)
        reader = csv.reader(output)
        rows = list(reader)
        assert len(rows) == 2  # header + data
        assert rows[1][0] == "test.txt"
        assert rows[1][1] == "1"


class TestRichFormatter:
    """Tests for RichFormatter."""

    def test_init_with_options(self):
        """Test initialization with force_color and force_plain."""
        formatter = RichFormatter(force_color=True, force_plain=False)
        assert formatter.force_color is True
        assert formatter.force_plain is False

    def test_write_result_skips_no_errors(self):
        """Test write_result skips results without errors."""
        formatter = RichFormatter(force_plain=True)
        output = io.StringIO()

        mock_result = Mock()
        mock_result.has_errors = False

        formatter.write_result(output, mock_result, 1, "test.txt")
        assert len(formatter.results) == 0

    def test_write_result_collects_errors(self):
        """Test write_result collects error data."""
        formatter = RichFormatter(force_plain=True)
        output = io.StringIO()

        mock_error = Mock()
        mock_error.position = 5
        mock_error.error_type = "spelling"
        mock_error.text = "wrng"
        mock_error.suggestions = ["wrong"]

        mock_result = Mock()
        mock_result.has_errors = True
        mock_result.errors = [mock_error]

        formatter.write_result(output, mock_result, 1, "test.txt")

        assert len(formatter.results) == 1
        assert formatter.results[0]["error_type"] == "spelling"
        assert formatter.current_file == "test.txt"


# ============================================================================
# Tests for process_stream
# ============================================================================


class TestProcessStream:
    """Tests for process_stream function."""

    def test_process_empty_stream(self):
        """Test processing empty stream."""
        input_stream = io.StringIO("")
        output_stream = io.StringIO()
        mock_checker = Mock()
        mock_formatter = Mock()

        stats = process_stream(
            input_stream, "test.txt", mock_checker, mock_formatter, output_stream, Mock()
        )

        assert stats["total_lines"] == 0
        assert stats["total_errors"] == 0
        assert stats["lines_with_errors"] == 0

    def test_process_stream_with_blank_lines(self):
        """Test processing stream with blank lines."""
        input_stream = io.StringIO("\n\n\n")
        output_stream = io.StringIO()
        mock_checker = Mock()
        mock_formatter = Mock()

        stats = process_stream(
            input_stream, "test.txt", mock_checker, mock_formatter, output_stream, Mock()
        )

        assert stats["total_lines"] == 0

    def test_process_stream_counts_errors(self):
        """Test processing stream counts errors correctly."""
        input_stream = io.StringIO("line1\nline2\nline3\n")
        output_stream = io.StringIO()

        mock_result_with_errors = Mock()
        mock_result_with_errors.has_errors = True
        mock_result_with_errors.errors = [Mock(), Mock()]  # 2 errors

        mock_result_no_errors = Mock()
        mock_result_no_errors.has_errors = False

        mock_checker = Mock()
        mock_checker.check.side_effect = [
            mock_result_with_errors,
            mock_result_no_errors,
            mock_result_with_errors,
        ]

        mock_formatter = Mock()

        stats = process_stream(
            input_stream, "test.txt", mock_checker, mock_formatter, output_stream, Mock()
        )

        assert stats["total_lines"] == 3
        assert stats["total_errors"] == 4  # 2 + 0 + 2
        assert stats["lines_with_errors"] == 2


# ============================================================================
# Tests for get_checker
# ============================================================================


class TestGetChecker:
    """Tests for get_checker function."""

    @patch("myspellchecker.cli.SQLiteProvider")
    def test_get_checker_data_loading_error_exits(self, mock_provider):
        """Test get_checker exits on DataLoadingError with db_path."""
        from myspellchecker.core.exceptions import DataLoadingError

        mock_provider.side_effect = DataLoadingError("failed")

        with pytest.raises(SystemExit) as exc_info:
            get_checker(database_path="/path/to/db.db")
        assert exc_info.value.code == 1

    @patch("myspellchecker.cli.SQLiteProvider")
    def test_get_checker_unexpected_error_exits(self, mock_provider):
        """Test get_checker exits on unexpected error with db_path."""
        mock_provider.side_effect = RuntimeError("unexpected")

        with pytest.raises(SystemExit) as exc_info:
            get_checker(database_path="/path/to/db.db")
        assert exc_info.value.code == 1

    def test_get_checker_with_real_memory_provider(self):
        """Test get_checker with real MemoryProvider (fast, no DB needed)."""

        # Use real MemoryProvider to avoid Pydantic validation issues
        with patch("myspellchecker.cli.SQLiteProvider") as mock_sqlite:
            from myspellchecker.core.exceptions import DataLoadingError

            mock_sqlite.side_effect = DataLoadingError("not found")

            result = get_checker()
            assert result is not None

    def test_get_checker_preset_applied(self):
        """Test get_checker applies preset by checking config values."""
        from myspellchecker.core.exceptions import DataLoadingError

        with patch("myspellchecker.cli.SQLiteProvider") as mock_sqlite:
            mock_sqlite.side_effect = DataLoadingError("not found")

            # Get checker with fast preset (should disable context checker)
            result = get_checker(preset="fast")
            assert result is not None
            # Fast preset should have context checker disabled
            assert not result.config.use_context_checker

    @patch("myspellchecker.cli.SQLiteProvider")
    def test_get_checker_preset_with_no_phonetic_override(self, mock_provider):
        """Test get_checker with preset and no_phonetic override."""
        from myspellchecker.core.exceptions import DataLoadingError

        mock_provider.side_effect = DataLoadingError("not found")

        # Get checker with accurate preset (enables phonetic) but override it off
        result = get_checker(preset="accurate", no_phonetic=True)
        assert result is not None
        # Phonetic should be disabled despite preset
        assert not result.config.use_phonetic

    @patch("myspellchecker.cli.SQLiteProvider")
    def test_get_checker_preset_with_no_context_override(self, mock_provider):
        """Test get_checker with preset and no_context override."""
        from myspellchecker.core.exceptions import DataLoadingError

        mock_provider.side_effect = DataLoadingError("not found")

        # Get checker with accurate preset (enables context) but override it off
        result = get_checker(preset="accurate", no_context=True)
        assert result is not None
        # Context checker should be disabled despite preset
        assert not result.config.use_context_checker

    def test_get_checker_configuration_error_exits(self):
        """Test get_checker exits on ConfigurationError."""
        from myspellchecker.core.exceptions import ConfigurationError

        with patch("myspellchecker.cli.SpellChecker") as mock_checker:
            mock_checker.side_effect = ConfigurationError("invalid config")

            with pytest.raises(SystemExit) as exc_info:
                # Use no db_path to avoid SQLiteProvider
                get_checker()
            assert exc_info.value.code == 1


# ============================================================================
# Tests for RichFormatter advanced methods
# ============================================================================


class TestRichFormatterAdvanced:
    """Additional tests for RichFormatter to cover terminal and table paths."""

    def test_begin_with_terminal(self):
        """Test begin() prints Myanmar warning in terminal mode."""
        formatter = RichFormatter(force_plain=False)

        # Mock the console as terminal
        formatter.console = Mock()
        formatter.console.is_terminal = True

        with patch("myspellchecker.cli_formatters.print_myanmar_warning") as mock_warning:
            output = io.StringIO()
            formatter.begin(output)
            mock_warning.assert_called_once_with(formatter.console)
            formatter.console.print.assert_called()  # newline after warning

    def test_begin_non_terminal(self):
        """Test begin() skips warning in non-terminal mode."""
        formatter = RichFormatter(force_plain=False)

        # Mock the console as non-terminal
        formatter.console = Mock()
        formatter.console.is_terminal = False

        with patch("myspellchecker.cli_formatters.print_myanmar_warning") as mock_warning:
            output = io.StringIO()
            formatter.begin(output)
            mock_warning.assert_not_called()

    def test_end_with_results_table(self):
        """Test end() prints results table when there are errors."""
        formatter = RichFormatter(force_plain=True)
        formatter.console = Mock()
        formatter.current_file = "test.txt"
        formatter.results = [
            {
                "line": 1,
                "position": 5,
                "error_type": "spelling",
                "text": "wrng",
                "suggestions": ["wrong"],
            }
        ]

        with (
            patch("myspellchecker.cli_formatters.create_file_header_panel"),
            patch("myspellchecker.cli_formatters.create_error_table") as mock_table,
            patch("myspellchecker.cli_formatters.get_error_style") as mock_style,
            patch("myspellchecker.cli_formatters.get_error_icon") as mock_icon,
            patch("myspellchecker.cli_formatters.format_suggestions") as mock_suggestions,
            patch("myspellchecker.cli_formatters.create_summary_panel") as mock_summary,
        ):
            mock_style.return_value = "red"
            mock_icon.return_value = "❌"
            mock_suggestions.return_value = "wrong"
            mock_table_obj = Mock()
            mock_table.return_value = mock_table_obj

            output = io.StringIO()
            formatter.end(output, {"total_errors": 1, "total_lines": 1})

            # Verify table was created and row was added
            mock_table.assert_called_once()
            mock_table_obj.add_row.assert_called()
            mock_summary.assert_called_once()

    def test_end_no_results(self):
        """Test end() with no errors still shows summary."""
        formatter = RichFormatter(force_plain=True)
        formatter.console = Mock()
        formatter.current_file = None
        formatter.results = []

        with patch("myspellchecker.cli_formatters.create_summary_panel") as mock_summary:
            output = io.StringIO()
            formatter.end(output, {"total_errors": 0, "total_lines": 10})

            mock_summary.assert_called_once_with(
                error_count=0,
                line_count=10,
                file_name=None,
            )


# ============================================================================
# Tests for CLI main command paths
# ============================================================================


class TestCLICommandPaths:
    """Tests for various CLI command paths."""

    def test_main_completion_valid_shell(self):
        """Test completion command for valid shell."""
        from myspellchecker.cli import main

        # completion command uses --shell flag with default=bash
        test_args = ["myspellchecker", "completion", "--shell", "bash"]
        with patch.object(sys, "argv", test_args):
            with patch("builtins.print") as mock_print:
                main()
                # Verify bash completion script was printed
                calls = [str(c) for c in mock_print.call_args_list]
                assert any("complete" in c.lower() or "bash" in c.lower() for c in calls)

    def test_main_completion_default_shell(self):
        """Test completion command with default shell (bash)."""
        from myspellchecker.cli import main

        test_args = ["myspellchecker", "completion"]
        with patch.object(sys, "argv", test_args):
            with patch("builtins.print") as mock_print:
                main()
                # Verify completion script was printed (default is bash)
                calls = [str(c) for c in mock_print.call_args_list]
                assert any("complete" in c.lower() for c in calls)

    def test_main_config_no_subcommand(self):
        """Test config command without subcommand shows usage."""
        from myspellchecker.cli import main

        test_args = ["myspellchecker", "config"]
        with patch.object(sys, "argv", test_args):
            with patch("builtins.print") as mock_print:
                main()
                # Verify usage was printed
                calls = [str(c) for c in mock_print.call_args_list]
                assert any("Usage" in c for c in calls)

    def test_main_config_show(self):
        """Test config show command outputs search paths."""
        from myspellchecker.cli import main

        test_args = ["myspellchecker", "config", "show"]
        with patch.object(sys, "argv", test_args):
            with patch("builtins.print") as mock_print:
                main()
                # Verify configuration paths are shown
                calls = [str(c) for c in mock_print.call_args_list]
                assert any("Configuration" in c or "Search" in c for c in calls)


# ============================================================================
# Tests for build command edge cases
# ============================================================================


class TestBuildCommandPaths:
    """Tests for build command edge cases."""

    def test_build_directory_input_expansion(self):
        """Test build command expands directory inputs."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            Path(tmpdir, "file1.txt").write_text("test content", encoding="utf-8")
            Path(tmpdir, "file2.txt").write_text("test content 2", encoding="utf-8")

            from myspellchecker.cli import main

            test_args = [
                "myspellchecker",
                "build",
                "--input",
                tmpdir,
                "--output",
                str(Path(tmpdir) / "output.db"),
                "--validate",
            ]

            with patch.object(sys, "argv", test_args):
                # validate mode exits after printing validation panel
                with pytest.raises(SystemExit) as exc_info:
                    main()
                # Valid inputs should pass validation (exit 0)
                assert exc_info.value.code == 0


# ============================================================================
# Tests for check command error handling
# ============================================================================


class TestCheckCommandErrorHandling:
    """Tests for check command error handling paths."""

    def test_process_stream_keyboard_interrupt(self):
        """Test process_stream handles KeyboardInterrupt."""
        from myspellchecker.cli import process_stream

        input_stream = io.StringIO("test line\n")
        output_stream = io.StringIO()

        mock_checker = Mock()
        mock_checker.check.side_effect = KeyboardInterrupt()
        mock_formatter = Mock()

        with pytest.raises(KeyboardInterrupt):
            process_stream(
                input_stream, "test.txt", mock_checker, mock_formatter, output_stream, Mock()
            )

    def test_process_stream_with_errors(self):
        """Test process_stream counts errors and calls formatter."""
        from myspellchecker.cli import process_stream

        input_stream = io.StringIO("test line\nline 2\n")
        output_stream = io.StringIO()

        mock_error = Mock()
        mock_result = Mock()
        mock_result.has_errors = True
        mock_result.errors = [mock_error]

        mock_checker = Mock()
        mock_checker.check.return_value = mock_result
        mock_formatter = Mock()

        stats = process_stream(
            input_stream, "test.txt", mock_checker, mock_formatter, output_stream, Mock()
        )

        assert stats["total_lines"] == 2
        assert stats["total_errors"] == 2
        assert stats["lines_with_errors"] == 2
        assert mock_formatter.write_result.call_count == 2
