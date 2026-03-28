"""
Training pipeline progress reporting.

Provides beautiful console output for the training pipeline using Rich.
"""

from __future__ import annotations

import time

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.theme import Theme

from myspellchecker.utils.logging_utils import get_logger

# Custom theme for training pipeline
TRAINING_THEME = Theme(
    {
        "error": "bold red",
        "warning": "bold yellow",
        "success": "bold green",
        "info": "bold blue",
        "header": "bold cyan",
        "muted": "dim",
        "highlight": "bold white",
        "step": "bold magenta",
    }
)


class TrainingReporter:
    """
    Console-based training reporter with Rich formatting.

    Provides beautiful terminal output for the training pipeline,
    matching the style of the data pipeline reporter (monotone, no colors).
    """

    def __init__(self, force_plain: bool = False):
        """
        Initialize the training reporter.

        Args:
            force_plain: If True, disable colors (for pipes/CI).
        """
        self.console = Console(
            theme=TRAINING_THEME,
            no_color=force_plain,
            highlight=False,
        )
        self.logger = get_logger(__name__)
        self._step_times: dict[str, float] = {}

    def show_header(
        self,
        corpus_path: str,
        output_dir: str,
        architecture: str,
        epochs: int,
        vocab_size: int,
    ) -> None:
        """Display the training pipeline header panel."""
        header_content = (
            f"[info]Corpus:[/] {corpus_path}\n"
            f"[info]Output:[/] {output_dir}\n"
            f"[info]Architecture:[/] {architecture}\n"
            f"[info]Epochs:[/] {epochs}\n"
            f"[info]Vocab Size:[/] {vocab_size:,}"
        )
        self.console.print()
        self.console.print(
            Panel(
                header_content,
                title="\U0001f9e0 Starting Training Pipeline",
                title_align="left",
                border_style="info",
                padding=(0, 2),
            )
        )

    def step_start(self, step: int, total: int, title: str) -> None:
        """Report that a pipeline step is starting."""
        self.logger.info(f"Starting step {step}/{total}: {title}")
        self._step_times[title] = time.time()
        self.console.print()
        self.console.print(
            Panel(
                f"[info]\U0001f504[/] {title}",
                title=f"[step]Step {step}/{total}[/]",
                title_align="left",
                border_style="info",
                padding=(0, 1),
            )
        )

    def step_complete(self, step: int, total: int, title: str, details: str = "") -> None:
        """Report that a pipeline step completed successfully."""
        duration = ""
        if title in self._step_times:
            elapsed = time.time() - self._step_times[title]
            duration = self._format_duration(elapsed)

        self.logger.info(f"Completed step {step}/{total}: {title} ({duration})")

        msg = f"[success]\u2713[/] {title}"
        if duration:
            msg += f" [muted]({duration})[/]"
        if details:
            msg += f"\n  [muted]{details}[/]"
        self.console.print(msg)

    def step_skipped(self, step: int, total: int, title: str, reason: str = "") -> None:
        """Report that a pipeline step was skipped."""
        self.logger.info(f"Skipped step {step}/{total}: {title} - {reason}")
        msg = f"[muted]\u21b7 {title} - SKIPPED[/]"
        if reason:
            msg += f" [muted]({reason})[/]"
        self.console.print(msg)

    def info(self, message: str) -> None:
        """Print info message."""
        self.logger.info(message)
        self.console.print(f"  [info]\u2139[/] {message}")

    def success(self, message: str) -> None:
        """Print success message."""
        self.logger.info(f"Success: {message}")
        self.console.print(f"  [success]\u2713[/] {message}")

    def warning(self, message: str) -> None:
        """Print warning message."""
        self.logger.warning(message)
        self.console.print(f"  [warning]\u26a0[/] {message}")

    def error(self, message: str) -> None:
        """Print error message."""
        self.logger.error(message)
        self.console.print(f"  [error]\u2717[/] {message}")

    def progress(self, message: str) -> None:
        """Print progress message."""
        self.logger.debug(message)
        self.console.print(f"  [info]\u27a4[/] {message}")

    def show_training_progress(self) -> Progress:
        """Create a progress bar for training operations."""
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=self.console,
            transient=False,
        )

    def show_streaming_progress(self) -> Progress:
        """Create a spinner + counter for streaming operations (unknown total).

        Shows: spinner, description, lines processed, elapsed time.
        Suitable for Pass 1 JSONL generation where total is unknown upfront.
        Use with total=None for indeterminate progress (spinner instead of bar).
        """
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TextColumn("{task.completed:,} lines", style="highlight"),
            TimeElapsedColumn(),
            console=self.console,
            transient=False,
        )

    def show_summary(
        self,
        step_durations: dict[str, tuple],
        total_duration: float,
        output_dir: str,
    ) -> None:
        """Display the training pipeline completion summary."""
        summary_table = Table(
            title="\U0001f4ca Training Summary",
            title_style="header",
            show_header=True,
            header_style="bold",
            border_style="dim",
        )
        summary_table.add_column("Step", style="info")
        summary_table.add_column("Duration", justify="right", style="highlight")
        summary_table.add_column("Status", justify="center")

        for step_name, (status, duration) in step_durations.items():
            if status == "skipped":
                summary_table.add_row(step_name, "--", "[muted]SKIPPED[/]")
            else:
                summary_table.add_row(
                    step_name,
                    self._format_duration(duration),
                    "[success]\u2713[/]",
                )

        # Add total row
        summary_table.add_section()
        summary_table.add_row(
            "[bold]Total Duration[/]",
            f"[bold]{self._format_duration(total_duration)}[/]",
            "",
        )

        self.console.print()
        self.console.print(summary_table)

        # Show completion panel
        self.console.print()
        self.console.print(
            Panel(
                f"[success]\u2713[/] Training completed successfully!\n"
                f"[info]Model:[/] {output_dir}",
                title="\U0001f389 Training Complete",
                title_align="left",
                border_style="success",
                padding=(0, 2),
            )
        )

    def show_training_metrics(
        self,
        loss: float,
        learning_rate: float,
        epoch: float,
        samples_per_second: float | None = None,
    ) -> None:
        """Display training metrics in a compact format."""
        metrics = f"loss={loss:.4f} | lr={learning_rate:.2e} | epoch={epoch:.2f}"
        if samples_per_second:
            metrics += f" | {samples_per_second:.1f} samples/s"
        self.console.print(f"  [muted]{metrics}[/]")

    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable form."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes}m {secs:.0f}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"


try:
    from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False


class RichProgressCallback(TrainerCallback if _HAS_TRANSFORMERS else object):  # type: ignore[misc]
    """HuggingFace TrainerCallback that displays Rich progress bars and metrics.

    Wires ``TrainingReporter.show_training_progress()`` for determinate training
    and ``show_training_metrics()`` for per-log-step metric display.

    Usage::

        reporter = TrainingReporter()
        callback = RichProgressCallback(reporter)
        trainer = Trainer(..., callbacks=[callback])
    """

    def __init__(self, reporter: TrainingReporter) -> None:
        self.reporter = reporter
        self._progress: Progress | None = None
        self._task_id: TaskID | None = None

    def on_train_begin(
        self,
        args: "TrainingArguments",
        state: "TrainerState",
        control: "TrainerControl",
        **kwargs,
    ) -> None:
        """Start the Rich progress bar at training begin."""
        self._progress = self.reporter.show_training_progress()
        self._progress.start()
        total = state.max_steps if state.max_steps > 0 else None
        self._task_id = self._progress.add_task("Training", total=total)

    def on_log(
        self,
        args: "TrainingArguments",
        state: "TrainerState",
        control: "TrainerControl",
        logs=None,
        **kwargs,
    ) -> None:
        """Display metrics on each log step and advance the progress bar."""
        if logs is None:
            return

        loss = logs.get("loss", logs.get("train_loss"))
        lr = logs.get("learning_rate")
        epoch = logs.get("epoch", state.epoch)
        samples_per_sec = logs.get("train_samples_per_second")

        if loss is not None and lr is not None and epoch is not None:
            self.reporter.show_training_metrics(
                loss=loss,
                learning_rate=lr,
                epoch=epoch,
                samples_per_second=samples_per_sec,
            )

        if self._progress is not None and self._task_id is not None:
            self._progress.update(self._task_id, completed=state.global_step)

    def on_train_end(
        self,
        args: "TrainingArguments",
        state: "TrainerState",
        control: "TrainerControl",
        **kwargs,
    ) -> None:
        """Stop the progress bar at training end."""
        if self._progress is not None:
            if self._task_id is not None:
                self._progress.update(self._task_id, completed=state.global_step)
            self._progress.stop()
            self._progress = None
            self._task_id = None
