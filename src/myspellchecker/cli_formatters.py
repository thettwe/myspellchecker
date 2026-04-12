"""Output formatters for the mySpellChecker CLI."""

from __future__ import annotations

import csv
import json
from typing import Any, TextIO

from myspellchecker.utils.console import (
    create_error_table,
    create_file_header_panel,
    create_summary_panel,
    format_suggestions,
    get_console,
    get_error_icon,
    get_error_style,
    print_myanmar_warning,
)

# Available presets for --preset flag
AVAILABLE_PRESETS = ["default", "fast", "accurate", "minimal", "strict"]


class Formatter:
    """Base class for output formatters."""

    def begin(self, output: TextIO) -> None:
        """Initialize output stream before writing results."""

    def write_result(self, output: TextIO, result: Any, line_no: int, file_name: str) -> None:
        """Write a single spell check result to the output stream."""

    def end(self, output: TextIO, summary: dict[str, Any]) -> None:
        """Finalize output and write summary statistics."""


class JSONFormatter(Formatter):
    """JSON output formatter."""

    def __init__(self) -> None:
        self.results: list = []

    def write_result(self, output: TextIO, result: Any, line_no: int, file_name: str) -> None:
        """Collect spell check result for JSON serialization."""
        self.results.append(
            {
                "file": file_name,
                "line": line_no,
                "text": result.text,
                "has_errors": result.has_errors,
                "errors": [e.to_dict() for e in result.errors],
            }
        )

    def end(self, output: TextIO, summary: dict[str, Any]) -> None:
        """Write collected results and summary as JSON to output."""
        final_output = {"summary": summary, "results": self.results}
        json.dump(final_output, output, indent=2, ensure_ascii=False)
        output.write("\n")


class TextFormatter(Formatter):
    """Simple text formatter (grep-like)."""

    def begin(self, output: TextIO) -> None:
        """Write header with Myanmar text rendering warning."""
        output.write("# WARNING: Myanmar text may not render correctly in your terminal.\n")
        output.write("# Use a text editor with proper font support to view this output.\n\n")

    def write_result(self, output: TextIO, result: Any, line_no: int, file_name: str) -> None:
        """Write spell errors in grep-like format (file:line:pos: error)."""
        if not result.has_errors:
            return

        for error in result.errors:
            suggestions = ", ".join(error.suggestions[:3])
            output.write(
                f"{file_name}:{line_no}:{error.position}: {error.error_type} "
                f"'{error.text}' -> Try: [{suggestions}]\n"
            )

    def end(self, output: TextIO, summary: dict[str, Any]) -> None:
        """Write summary line with total error and line counts."""
        output.write(
            f"\n# Summary: {summary['total_errors']} errors found "
            f"in {summary['total_lines']} lines.\n"
        )


class CSVFormatter(Formatter):
    """CSV output formatter."""

    def begin(self, output: TextIO) -> None:
        """Write CSV header row with column names."""
        self.writer = csv.writer(output)
        self.writer.writerow(["file", "line", "position", "error_type", "text", "suggestions"])

    def _sanitize_csv_field(self, text: str) -> str:
        """Sanitize CSV fields to prevent formula injection."""
        if not text:
            return text
        if text.startswith(("=", "+", "-", "@")):
            return f"'{text}"
        return text

    def write_result(self, output: TextIO, result: Any, line_no: int, file_name: str) -> None:
        """Write spell errors as sanitized CSV rows."""
        if not result.has_errors:
            return

        for error in result.errors:
            suggestions = ", ".join(error.suggestions[:3])
            self.writer.writerow(
                [
                    file_name,
                    line_no,
                    error.position,
                    error.error_type,
                    self._sanitize_csv_field(error.text),
                    self._sanitize_csv_field(suggestions),
                ]
            )


class RichFormatter(Formatter):
    """Rich-formatted output with colors and tables.

    Uses the rich library for beautiful terminal output with:
    - Colored error type indicators
    - Table-based error display
    - Summary panel with statistics
    """

    def __init__(self, force_color: bool = False, force_plain: bool = False) -> None:
        self.force_color = force_color
        self.force_plain = force_plain
        self.console = get_console(force_plain=force_plain, force_color=force_color)
        self.results: list[dict[str, Any]] = []
        self.current_file: str | None = None
        self.table: Any | None = None

    def begin(self, output: TextIO) -> None:
        """Display Myanmar text rendering warning in terminal."""
        # Print Myanmar warning if terminal
        if self.console.is_terminal and not self.force_plain:
            print_myanmar_warning(self.console)
            self.console.print()

    def write_result(self, output: TextIO, result: Any, line_no: int, file_name: str) -> None:
        """Collect spell errors for rich table display."""
        if not result.has_errors:
            return

        # Store result for table display
        for error in result.errors:
            self.results.append(
                {
                    "file": file_name,
                    "line": line_no,
                    "position": error.position,
                    "error_type": error.error_type,
                    "text": error.text,
                    "suggestions": error.suggestions,
                }
            )
        self.current_file = file_name

    def end(self, output: TextIO, summary: dict[str, Any]) -> None:
        """Render rich table with collected errors and summary panel."""
        # Print file header panel
        if self.current_file:
            self.console.print(create_file_header_panel(self.current_file))
            self.console.print()

        # Print results table if there are errors
        if self.results:
            table = create_error_table()
            for r in self.results:
                style = get_error_style(r["error_type"])
                icon = get_error_icon(r["error_type"])
                error_type_display = f"{icon} {r['error_type']}"
                suggestions_display = format_suggestions(r["suggestions"])

                table.add_row(
                    str(r["line"]),
                    str(r["position"]),
                    f"[{style}]{error_type_display}[/]",
                    r["text"],
                    suggestions_display,
                )

            self.console.print(table)
            self.console.print()

        # Print summary panel
        self.console.print(
            create_summary_panel(
                error_count=summary.get("total_errors", 0),
                line_count=summary.get("total_lines", 0),
                file_name=self.current_file,
            )
        )
