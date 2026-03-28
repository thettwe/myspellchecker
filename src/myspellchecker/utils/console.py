"""Console output utilities with rich/fallback support.

This module provides a unified console interface that uses Rich for terminal
formatting when available and falls back to plain text output for pipes/CI.

Panel builders are in ``console_panels.py`` and table builders are in
``console_tables.py``.  Everything is re-exported here so existing
``from myspellchecker.utils.console import ...`` imports keep working.
"""

from __future__ import annotations

import sys
from typing import Any

from rich.console import Console
from rich.theme import Theme

from myspellchecker.core.constants.core_constants import ErrorType

# Re-export panel builders
from myspellchecker.utils.console_panels import (
    create_build_complete_panel,
    create_build_header_panel,
    create_file_header_panel,
    create_hydration_panel,
    create_pipeline_step_panel,
    create_summary_panel,
    create_validation_panel,
    create_word_mapping_panel,
)

# Re-export table builders
from myspellchecker.utils.console_tables import (
    create_error_table,
    create_sample_data_table,
    create_schema_table,
    create_stats_table,
)

__all__ = [
    "PipelineConsole",
    "create_build_complete_panel",
    "create_build_header_panel",
    "create_error_table",
    "create_hydration_panel",
    "create_pipeline_step_panel",
    "create_schema_table",
    "create_sample_data_table",
    "create_stats_table",
    "create_summary_panel",
    "create_validation_panel",
    "create_word_mapping_panel",
    "create_file_header_panel",
    "format_suggestions",
    "get_console",
    "get_error_icon",
    "get_error_style",
    "is_terminal",
    "print_myanmar_warning",
]

# ---------------------------------------------------------------------------
# Theme & style constants
# ---------------------------------------------------------------------------

# Custom theme for Myanmar spell checker
MYSPELL_THEME = Theme(
    {
        "error": "bold red",
        "warning": "bold yellow",
        "success": "bold green",
        "info": "bold blue",
        "syllable_error": "red",
        "word_error": "#ff8c00",  # Orange
        "context_error": "yellow",
        "tone_ambiguity": "cyan",
        "syntax_error": "magenta",
        "header": "bold cyan",
        "muted": "dim",
        "highlight": "bold white",
        "myanmar": "white",  # Myanmar text
    }
)

# Error type to style mapping
ERROR_STYLES = {
    "syllable_error": "syllable_error",
    ErrorType.SYLLABLE.value: "syllable_error",
    "word_error": "word_error",
    ErrorType.WORD.value: "word_error",
    "context_error": "context_error",
    ErrorType.CONTEXT_PROBABILITY.value: "context_error",
    ErrorType.TONE_AMBIGUITY.value: "tone_ambiguity",
    ErrorType.SYNTAX_ERROR.value: "syntax_error",
}

# Error type to icon mapping
ERROR_ICONS = {
    "syllable_error": "\U0001f524",  # letter symbols
    ErrorType.SYLLABLE.value: "\U0001f524",
    "word_error": "\U0001f4dd",  # memo
    ErrorType.WORD.value: "\U0001f4dd",
    "context_error": "\U0001f517",  # link
    ErrorType.CONTEXT_PROBABILITY.value: "\U0001f517",
    ErrorType.TONE_AMBIGUITY.value: "\U0001f3b5",  # musical note
    ErrorType.SYNTAX_ERROR.value: "\u26a0\ufe0f",  # warning
}


# ---------------------------------------------------------------------------
# Core console functions
# ---------------------------------------------------------------------------


def get_console(
    force_plain: bool = False,
    force_color: bool = False,
    file: Any = None,
) -> Console:
    """Get console instance with appropriate settings.

    Args:
        force_plain: Force plain text output (no colors/formatting)
        force_color: Force color output even when not a TTY
        file: Output file (defaults to sys.stdout)

    Returns:
        Configured Console instance
    """
    if file is None:
        file = sys.stdout

    # Determine if we should use colors
    if force_plain:
        force_terminal = False
        no_color = True
    elif force_color:
        force_terminal = True
        no_color = False
    else:
        # Auto-detect: use colors only for TTY
        force_terminal = None
        no_color = None

    return Console(
        file=file,
        theme=MYSPELL_THEME,
        force_terminal=force_terminal,
        no_color=no_color,
        highlight=False,  # Don't auto-highlight (preserves Myanmar text)
    )


def is_terminal() -> bool:
    """Check if stdout is a terminal (TTY)."""
    return sys.stdout.isatty()


def get_error_style(error_type: str) -> str:
    """Get the style for an error type.

    Args:
        error_type: Type of error

    Returns:
        Style name from theme
    """
    return ERROR_STYLES.get(error_type, "error")


def get_error_icon(error_type: str) -> str:
    """Get the icon for an error type.

    Args:
        error_type: Type of error

    Returns:
        Icon string
    """
    return ERROR_ICONS.get(error_type, "\u2717")


def format_suggestions(suggestions: list[str], max_display: int = 5) -> str:
    """Format suggestions for display.

    Args:
        suggestions: List of suggestion strings
        max_display: Maximum suggestions to show

    Returns:
        Formatted suggestion string
    """
    if not suggestions:
        return "[muted]No suggestions[/]"

    display = suggestions[:max_display]
    formatted = ", ".join(display)

    if len(suggestions) > max_display:
        formatted += f" [muted](+{len(suggestions) - max_display} more)[/]"

    return formatted


def print_myanmar_warning(console: Console) -> None:
    """Print a warning about Myanmar text rendering.

    Args:
        console: Console instance to print to
    """
    if console.is_terminal:
        console.print(
            "[warning]\u26a0[/] [muted]Myanmar text may not render correctly in some terminals.[/]",
            highlight=False,
        )


# ---------------------------------------------------------------------------
# PipelineConsole class
# ---------------------------------------------------------------------------


class PipelineConsole:
    """Console wrapper for pipeline output with rich formatting (no colors)."""

    def __init__(self):
        # Always use no_color=True for consistent pipeline output
        self.console = Console(
            theme=MYSPELL_THEME,
            no_color=True,
            highlight=False,
        )
        self._verified_tables: list[str] = []

    def step_start(self, step: int, total: int, title: str) -> None:
        """Print step start message."""
        self.console.print()
        self.console.print(create_pipeline_step_panel(step, total, title, "running"))

    def step_complete(self, step: int, total: int, title: str, duration: str = "") -> None:
        """Print step complete message."""
        msg = f"[success]\u2713[/] {title}"
        if duration:
            msg += f" [muted]({duration})[/]"
        self.console.print(msg)

    def step_skipped(self, step: int, total: int, title: str, reason: str = "") -> None:
        """Print step skipped message."""
        msg = f"[muted]\u21b7 {title} - SKIPPED[/]"
        if reason:
            msg += f" [muted]({reason})[/]"
        self.console.print(msg)

    def table_verified(self, table_name: str) -> None:
        """Record a verified table."""
        self._verified_tables.append(table_name)

    def show_schema_summary(self) -> None:
        """Show schema verification summary."""
        if self._verified_tables:
            self.console.print(create_schema_table(self._verified_tables))
            self._verified_tables = []

    def show_stats(self, stats: dict, title: str = "Database Statistics") -> None:
        """Show statistics table."""
        self.console.print()
        self.console.print(create_stats_table(stats, title))

    def show_sample_data(
        self,
        title: str,
        data: list[dict],
    ) -> None:
        """Show sample data table from list of dictionaries.

        Args:
            title: Table title
            data: List of dictionaries where keys are column names
        """
        if not data:
            return
        # Extract columns from first dict
        columns = list(data[0].keys())
        # Convert dicts to tuples
        tuples = [tuple(d.values()) for d in data]
        self.console.print()
        self.console.print(create_sample_data_table(tuples, columns, title))

    def show_hydration(self, syllables: int, words: int, files: int = 0) -> None:
        """Show hydration results."""
        self.console.print(create_hydration_panel(syllables, words, files))

    def show_word_mapping(self, count: int) -> None:
        """Show word mapping results."""
        self.console.print(create_word_mapping_panel(count))

    def info(self, message: str) -> None:
        """Print info message."""
        self.console.print(f"[info]\u2139[/] {message}")

    def success(self, message: str) -> None:
        """Print success message."""
        self.console.print(f"[success]\u2713[/] {message}")

    def warning(self, message: str) -> None:
        """Print warning message."""
        self.console.print(f"[warning]\u26a0[/] {message}")

    def error(self, message: str) -> None:
        """Print error message."""
        self.console.print(f"[error]\u2717[/] {message}")

    def step(self, message: str) -> None:
        """Print step message (indented, with arrow)."""
        self.console.print(f"  [info]\u27a4[/] {message}")
