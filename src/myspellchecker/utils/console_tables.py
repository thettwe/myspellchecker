"""Table builder functions for Rich console output.

This module contains all table-building functions used for displaying
errors, schema information, statistics, and sample data.
"""

from __future__ import annotations

from rich.table import Table


def create_error_table(title: str = "Spell Check Results") -> Table:
    """Create a styled table for error display.

    Args:
        title: Table title

    Returns:
        Configured Table instance
    """
    table = Table(
        title=title,
        title_style="header",
        show_header=True,
        header_style="bold",
        border_style="dim",
        row_styles=["", "dim"],
    )
    table.add_column("Line", style="muted", justify="right", width=6)
    table.add_column("Pos", style="muted", justify="right", width=5)
    table.add_column("Type", style="info", width=12)
    table.add_column("Error", style="myanmar", min_width=15)
    table.add_column("Suggestions", style="success", min_width=20)
    return table


def create_schema_table(tables: list[str]) -> Table:
    """Create a table showing verified schema tables.

    Args:
        tables: List of table names that were verified

    Returns:
        Configured Table instance
    """
    table = Table(
        title="\U0001f5c4\ufe0f Database Schema",
        title_style="header",
        show_header=True,
        header_style="bold",
        border_style="dim",
        box=None,
        padding=(0, 1),
    )
    table.add_column("Table", style="info")
    table.add_column("Status", style="success", justify="center")

    for tbl in tables:
        table.add_row(tbl, "\u2713 verified")

    return table


def create_stats_table(stats: dict, title: str = "Database Statistics") -> Table:
    """Create a statistics table.

    Args:
        stats: Dictionary of statistic name to value
        title: Table title

    Returns:
        Configured Table instance
    """
    table = Table(
        title=f"\U0001f4ca {title}",
        title_style="header",
        show_header=True,
        header_style="bold",
        border_style="dim",
    )
    table.add_column("Metric", style="info")
    table.add_column("Value", style="highlight", justify="right")

    for key, value in stats.items():
        if isinstance(value, int):
            table.add_row(key, f"{value:,}")
        elif isinstance(value, float):
            table.add_row(key, f"{value:.2f}")
        else:
            table.add_row(key, str(value))

    return table


def create_sample_data_table(
    data: list[tuple],
    columns: list[str],
    title: str = "Sample Data",
) -> Table:
    """Create a table showing sample data.

    Args:
        data: List of tuples containing row data
        columns: List of column names
        title: Table title

    Returns:
        Configured Table instance
    """
    table = Table(
        title=title,
        title_style="header",
        show_header=True,
        header_style="bold",
        border_style="dim",
    )

    myanmar_cols = ["Syllable", "Word", "Word 1", "Word 2"]
    for col in columns:
        style = "myanmar" if col in myanmar_cols else "muted"
        table.add_column(col, style=style)

    for row in data:
        table.add_row(*[str(v) if v is not None else "-" for v in row])

    return table
