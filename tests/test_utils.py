"""Unit tests for utils module to improve coverage."""

import io
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest


class TestGetLogger:
    """Tests for get_logger function."""

    def test_get_logger_with_prefix(self):
        """Test get_logger adds myspellchecker prefix."""
        from myspellchecker.utils.logging_utils import LOGGER_PREFIX, get_logger

        logger = get_logger("test_module")

        assert logger.name.startswith(LOGGER_PREFIX)

    def test_get_logger_caching(self):
        """Test get_logger returns same instance for same name."""
        from myspellchecker.utils.logging_utils import get_logger

        logger1 = get_logger("test_cache")
        logger2 = get_logger("test_cache")

        assert logger1 is logger2


class TestConfigureLogging:
    """Tests for configure_logging function."""

    def test_configure_logging_with_string_level(self):
        """Test configure_logging with string level."""
        from myspellchecker.utils.logging_utils import LOGGER_PREFIX, configure_logging

        configure_logging(level="DEBUG")

        logger = logging.getLogger(LOGGER_PREFIX)
        assert logger.level == logging.DEBUG

    def test_configure_logging_with_int_level(self):
        """Test configure_logging with int level."""
        from myspellchecker.utils.logging_utils import LOGGER_PREFIX, configure_logging

        configure_logging(level=logging.WARNING)

        logger = logging.getLogger(LOGGER_PREFIX)
        assert logger.level == logging.WARNING

    def test_configure_logging_with_custom_format(self):
        """Test configure_logging with custom format string."""
        from myspellchecker.utils.logging_utils import LOGGER_PREFIX, configure_logging

        custom_format = "%(levelname)s - %(message)s"
        configure_logging(level="INFO", format_string=custom_format)

        logger = logging.getLogger(LOGGER_PREFIX)
        assert len(logger.handlers) == 1
        assert logger.handlers[0].formatter._fmt == custom_format

    def test_configure_logging_json_output(self):
        """Test configure_logging with JSON output."""
        from myspellchecker.utils.logging_utils import (
            LOGGER_PREFIX,
            configure_logging,
        )

        configure_logging(level="INFO", json_output=True)

        logger = logging.getLogger(LOGGER_PREFIX)
        fmt = logger.handlers[0].formatter._fmt
        assert "time" in fmt and "level" in fmt and "message" in fmt

    def test_configure_logging_debug_mode(self):
        """Test configure_logging with debug_mode."""
        from myspellchecker.utils.logging_utils import (
            LOGGER_PREFIX,
            configure_logging,
        )

        configure_logging(level="DEBUG", debug_mode=True)

        logger = logging.getLogger(LOGGER_PREFIX)
        assert "asctime" in logger.handlers[0].formatter._fmt
        assert "lineno" in logger.handlers[0].formatter._fmt

    def test_configure_logging_custom_stream(self):
        """Test configure_logging with custom stream."""
        from myspellchecker.utils.logging_utils import LOGGER_PREFIX, configure_logging

        custom_stream = io.StringIO()
        configure_logging(level="INFO", stream=custom_stream)

        logger = logging.getLogger(LOGGER_PREFIX)
        assert logger.handlers[0].stream is custom_stream

    def test_configure_logging_removes_existing_handlers(self):
        """Test configure_logging removes existing handlers."""
        from myspellchecker.utils.logging_utils import LOGGER_PREFIX, configure_logging

        configure_logging(level="INFO")
        configure_logging(level="DEBUG")
        configure_logging(level="WARNING")

        logger = logging.getLogger(LOGGER_PREFIX)
        assert len(logger.handlers) == 1


class TestCheckDiskSpace:
    """Tests for check_disk_space function."""

    def test_check_disk_space_passes_with_enough_space(self):
        """Test check_disk_space passes when enough space available."""
        from myspellchecker.utils.io_utils import check_disk_space

        with TemporaryDirectory() as tmpdir:
            check_disk_space(tmpdir, required_mb=1)

    def test_check_disk_space_insufficient_storage(self):
        """Test check_disk_space raises on insufficient space."""
        from myspellchecker.core.exceptions import InsufficientStorageError
        from myspellchecker.utils.io_utils import check_disk_space

        with TemporaryDirectory() as tmpdir:
            with pytest.raises(InsufficientStorageError):
                check_disk_space(tmpdir, required_mb=999_999_999)

    def test_check_disk_space_nonexistent_path(self):
        """Test check_disk_space handles non-existent paths by finding parent."""
        from myspellchecker.utils.io_utils import check_disk_space

        with TemporaryDirectory() as tmpdir:
            nonexistent = Path(tmpdir) / "does" / "not" / "exist"
            check_disk_space(nonexistent, required_mb=1)

    @patch("myspellchecker.utils.io_utils.shutil.disk_usage")
    def test_check_disk_space_file_not_found(self, mock_disk_usage):
        """Test check_disk_space handles FileNotFoundError."""
        from myspellchecker.utils.io_utils import check_disk_space

        mock_disk_usage.side_effect = FileNotFoundError("Path not found")
        check_disk_space("/some/path", required_mb=1)

    @patch("myspellchecker.utils.io_utils.shutil.disk_usage")
    def test_check_disk_space_os_error(self, mock_disk_usage):
        """Test check_disk_space handles OSError."""
        from myspellchecker.utils.io_utils import check_disk_space

        mock_disk_usage.side_effect = OSError("Permission denied")
        check_disk_space("/some/path", required_mb=1)


class TestGetConsole:
    """Tests for get_console function."""

    def test_get_console_force_plain(self):
        """Test get_console with force_plain disables colors."""
        from myspellchecker.utils.console import get_console

        console = get_console(force_plain=True)
        assert console.no_color is True

    def test_get_console_custom_file(self):
        """Test get_console with custom file."""
        from myspellchecker.utils.console import get_console

        output = io.StringIO()
        console = get_console(file=output)
        assert console.file is output


class TestConsoleOutputFormatters:
    """Tests for console output formatting functions."""

    def test_format_suggestions_with_items(self):
        """Test format_suggestions includes suggestion text."""
        from myspellchecker.utils.console import format_suggestions

        result = format_suggestions(["fix1", "fix2", "fix3"])
        assert "fix1" in result
        assert "fix2" in result

    def test_format_suggestions_empty(self):
        """Test format_suggestions with empty list shows 'No suggestions'."""
        from myspellchecker.utils.console import format_suggestions

        result = format_suggestions([])
        assert "No suggestions" in result


class TestPrintMyanmarWarning:
    """Tests for Myanmar text warning."""

    def test_print_myanmar_warning_in_terminal(self):
        """Test print_myanmar_warning outputs to console when in terminal mode."""
        from rich.console import Console

        from myspellchecker.utils.console import print_myanmar_warning

        output = io.StringIO()
        console = Console(file=output, force_terminal=True, no_color=True)

        print_myanmar_warning(console)

        content = output.getvalue()
        assert "Myanmar" in content

    def test_print_myanmar_warning_not_terminal(self):
        """Test print_myanmar_warning does nothing when not a terminal."""
        from rich.console import Console

        from myspellchecker.utils.console import print_myanmar_warning

        output = io.StringIO()
        console = Console(file=output, force_terminal=False, no_color=True)

        print_myanmar_warning(console)

        content = output.getvalue()
        assert content == ""


class TestTableCreation:
    """Tests for table creation functions."""

    def test_create_error_table_default_title(self):
        """Test create_error_table default title."""
        from myspellchecker.utils.console import create_error_table

        table = create_error_table()
        assert table.title == "Spell Check Results"

    def test_create_error_table_custom_title(self):
        """Test create_error_table with custom title."""
        from myspellchecker.utils.console import create_error_table

        table = create_error_table(title="Custom Title")
        assert table.title == "Custom Title"


class TestPanelCreation:
    """Tests for panel creation functions."""

    def test_create_summary_panel_no_errors(self):
        """Test create_summary_panel with no errors shows Summary title."""
        from myspellchecker.utils.console import create_summary_panel

        panel = create_summary_panel(error_count=0, line_count=100)
        assert panel.title == "Summary"

    def test_create_summary_panel_with_filename(self):
        """Test create_summary_panel with filename includes it in title."""
        from myspellchecker.utils.console import create_summary_panel

        panel = create_summary_panel(error_count=0, line_count=10, file_name="test.txt")
        assert "test.txt" in panel.title


class TestPipelineConsole:
    """Tests for PipelineConsole class."""

    def test_pipeline_console_table_verified(self):
        """Test PipelineConsole table_verified tracks tables."""
        from myspellchecker.utils.console import PipelineConsole

        pc = PipelineConsole()
        pc.table_verified("syllables")
        pc.table_verified("words")
        assert len(pc._verified_tables) == 2

    def test_pipeline_console_show_schema_summary_clears(self):
        """Test PipelineConsole show_schema_summary clears verified tables."""
        from myspellchecker.utils.console import PipelineConsole

        pc = PipelineConsole()
        pc.table_verified("syllables")
        pc.show_schema_summary()
        assert len(pc._verified_tables) == 0

    def test_pipeline_console_show_sample_data(self):
        """Test PipelineConsole show_sample_data with data."""
        from myspellchecker.utils.console import PipelineConsole

        pc = PipelineConsole()
        data = [{"word": "test", "freq": 10}, {"word": "another", "freq": 5}]
        pc.show_sample_data("Sample", data)

    def test_pipeline_console_show_sample_data_empty(self):
        """Test PipelineConsole show_sample_data with empty data does not raise."""
        from myspellchecker.utils.console import PipelineConsole

        pc = PipelineConsole()
        pc.show_sample_data("Empty", [])


class TestLoggingIntegration:
    """Integration tests for logging setup."""

    def test_logging_integration_with_configure(self):
        """Test logging works after configuration."""
        from myspellchecker.utils.logging_utils import configure_logging, get_logger

        output = io.StringIO()
        configure_logging(level="DEBUG", stream=output)

        logger = get_logger("test.integration")
        logger.info("Test message")

        content = output.getvalue()
        assert "Test message" in content

    def test_logging_propagation_disabled(self):
        """Test logging doesn't propagate to root logger."""
        from myspellchecker.utils.logging_utils import (
            LOGGER_PREFIX,
            configure_logging,
        )

        configure_logging(level="INFO")

        logger = logging.getLogger(LOGGER_PREFIX)
        assert logger.propagate is False


class TestQuietMode:
    """Tests for quiet mode logging."""

    def test_configure_quiet_mode(self):
        """Test configure_logging with high level filters lower-level messages."""
        from myspellchecker.utils.logging_utils import configure_logging, get_logger

        output = io.StringIO()
        configure_logging(level="ERROR", stream=output)

        logger = get_logger("test.quiet")
        logger.info("Should not appear")
        logger.error("Should appear")

        content = output.getvalue()
        assert "Should not appear" not in content
        assert "Should appear" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
