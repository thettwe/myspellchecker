"""Panel builder functions for Rich console output.

This module contains all panel-building functions used for pipeline output,
build progress, validation results, and spell check summaries.
"""

from __future__ import annotations

from rich.panel import Panel


def create_summary_panel(
    error_count: int,
    line_count: int,
    file_name: str | None = None,
) -> Panel:
    """Create a summary panel for spell check results.

    Args:
        error_count: Number of errors found
        line_count: Total lines processed
        file_name: Optional file name

    Returns:
        Configured Panel instance
    """
    clean_lines = line_count - error_count if error_count < line_count else 0

    if error_count == 0:
        content = "[success]\u2713 No errors found[/]\n"
    else:
        content = f"[error]\u2717 {error_count} error{'s' if error_count != 1 else ''} found[/]\n"

    content += f"[success]\u2713 {clean_lines} line{'s' if clean_lines != 1 else ''} clean[/]\n"
    content += (
        f"[info]\U0001f4ca {line_count} total line{'s' if line_count != 1 else ''} processed[/]"
    )

    title = "Summary"
    if file_name:
        title = f"Summary: {file_name}"

    return Panel(
        content,
        title=title,
        title_align="left",
        border_style="success" if error_count == 0 else "error",
        padding=(0, 2),
    )


def create_file_header_panel(file_name: str) -> Panel:
    """Create a header panel for file being checked.

    Args:
        file_name: Name of the file

    Returns:
        Configured Panel instance
    """
    return Panel(
        f"[info]File:[/] {file_name}",
        title="Spell Check Results",
        title_align="left",
        border_style="header",
        padding=(0, 2),
    )


def create_build_header_panel(
    input_files: list[str],
    database_path: str,
    sample: bool = False,
) -> Panel:
    """Create a header panel for build operation.

    Args:
        input_files: List of input file paths
        database_path: Output database path
        sample: Whether sample mode is enabled

    Returns:
        Configured Panel instance
    """
    if sample:
        content = "[info]Mode:[/] Sample generation\n"
        content += f"[info]Output:[/] {database_path}"
    else:
        file_count = len(input_files) if input_files else 0
        content = f"[info]Input:[/] {file_count} file{'s' if file_count != 1 else ''}\n"
        content += f"[info]Output:[/] {database_path}"

    return Panel(
        content,
        title="\U0001f528 Building Dictionary",
        title_align="left",
        border_style="info",
        padding=(0, 2),
    )


def create_build_complete_panel(database_path: str, success: bool = True) -> Panel:
    """Create a completion panel for build operation.

    Args:
        database_path: Output database path
        success: Whether build was successful

    Returns:
        Configured Panel instance
    """
    if success:
        content = "[success]\u2713 Build completed successfully![/]\n"
        content += f"[info]Database:[/] {database_path}"
        border_style = "success"
        title = "\U0001f389 Build Complete"
    else:
        content = "[error]\u2717 Build failed[/]"
        border_style = "error"
        title = "\u26a0\ufe0f Build Failed"

    return Panel(
        content,
        title=title,
        title_align="left",
        border_style=border_style,
        padding=(0, 2),
    )


def create_validation_panel(
    errors: list[str],
    warnings: list[str],
    statistics: dict,
    passed: bool,
) -> Panel:
    """Create a validation results panel.

    Args:
        errors: List of error messages
        warnings: List of warning messages
        statistics: Dictionary of statistics
        passed: Whether validation passed

    Returns:
        Configured Panel instance
    """
    content_parts = []

    # Errors section
    if errors:
        content_parts.append("[error]Errors:[/]")
        for err in errors:
            content_parts.append(f"  [error]\u2717[/] {err}")
        content_parts.append("")

    # Warnings section
    if warnings:
        content_parts.append("[warning]Warnings:[/]")
        for warn in warnings:
            content_parts.append(f"  [warning]\u26a0[/] {warn}")
        content_parts.append("")

    # Statistics section
    if statistics:
        content_parts.append("[info]Statistics:[/]")
        for key, value in statistics.items():
            content_parts.append(f"  {key}: [highlight]{value}[/]")

    content = "\n".join(content_parts)

    return Panel(
        content,
        title="Pre-flight Validation",
        title_align="left",
        border_style="success" if passed else "error",
        padding=(1, 2),
    )


def create_pipeline_step_panel(
    step: int,
    total: int,
    title: str,
    status: str = "running",
) -> Panel:
    """Create a panel for pipeline step.

    Args:
        step: Current step number
        total: Total number of steps
        title: Step title
        status: Status ('running', 'complete', 'skipped')

    Returns:
        Configured Panel instance
    """
    if status == "complete":
        icon = "\u2713"
        style = "success"
    elif status == "skipped":
        icon = "\u21b7"
        style = "muted"
    else:
        icon = "\U0001f504"
        style = "info"

    return Panel(
        f"[{style}]{icon}[/] {title}",
        title=f"[{style}]Step {step}/{total}[/]",
        title_align="left",
        border_style=style,
        padding=(0, 1),
    )


def create_hydration_panel(syllables: int, words: int, files: int = 0) -> Panel:
    """Create a panel showing hydration results.

    Args:
        syllables: Number of syllables hydrated
        words: Number of words hydrated
        files: Number of previously processed files

    Returns:
        Configured Panel instance
    """
    content = (
        f"[success]\u2713[/] Hydrated [highlight]{syllables:,}[/] syllables, "
        f"[highlight]{words:,}[/] words"
    )
    if files > 0:
        plural = "s" if files != 1 else ""
        content += (
            f"\n[info]\U0001f4c1[/] Found [highlight]{files}[/] previously processed file{plural}"
        )

    return Panel(
        content,
        title="\U0001f4be Incremental Update",
        title_align="left",
        border_style="info",
        padding=(0, 2),
    )


def create_word_mapping_panel(count: int) -> Panel:
    """Create a panel showing word ID mapping results.

    Args:
        count: Number of words mapped

    Returns:
        Configured Panel instance
    """
    return Panel(
        f"[success]\u2713[/] Mapped [highlight]{count:,}[/] words to IDs",
        title="\U0001f517 Word ID Mapping",
        title_align="left",
        border_style="success",
        padding=(0, 2),
    )
