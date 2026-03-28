"""
Pipeline progress reporting abstraction.

This module provides a PipelineReporter class that encapsulates all pipeline
progress reporting, making it easier to test pipeline logic and customize output.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from ..utils.logging_utils import get_logger

if TYPE_CHECKING:
    from ..utils.console import PipelineConsole


__all__ = [
    "MockReporter",
    "PipelineReporter",
    "ReporterInterface",
]


class ReporterInterface(ABC):
    """
    Abstract interface for pipeline reporting.

    This interface defines the contract for pipeline progress reporting,
    allowing different implementations (console, file, mock, etc.).
    """

    @abstractmethod
    def report_step_start(self, step: int, total: int, title: str) -> None:
        """Report that a pipeline step is starting."""
        raise NotImplementedError

    @abstractmethod
    def report_step_complete(self, step: int, total: int, title: str, duration: str = "") -> None:
        """Report that a pipeline step completed successfully."""
        raise NotImplementedError

    @abstractmethod
    def report_step_skipped(self, step: int, total: int, title: str, reason: str = "") -> None:
        """Report that a pipeline step was skipped."""
        raise NotImplementedError

    @abstractmethod
    def report_progress(self, message: str) -> None:
        """Report a progress message within a step."""
        raise NotImplementedError

    @abstractmethod
    def report_info(self, message: str) -> None:
        """Report an informational message."""
        raise NotImplementedError

    @abstractmethod
    def report_warning(self, message: str) -> None:
        """Report a warning message."""
        raise NotImplementedError

    @abstractmethod
    def report_error(self, message: str) -> None:
        """Report an error message."""
        raise NotImplementedError

    @abstractmethod
    def report_success(self, message: str) -> None:
        """Report a success message."""
        raise NotImplementedError


class PipelineReporter(ReporterInterface):
    """
    Console-based pipeline reporter.

    This class provides a clean interface for pipeline progress reporting,
    delegating to PipelineConsole for actual output formatting.

    Example:
        >>> from myspellchecker.utils.console import PipelineConsole
        >>> from myspellchecker.data_pipeline.reporter import PipelineReporter
        >>>
        >>> console = PipelineConsole()
        >>> reporter = PipelineReporter(console)
        >>> reporter.report_step_start(1, 4, "Processing data")
        >>> reporter.report_info("Found 1000 records")
        >>> reporter.report_step_complete(1, 4, "Processing data", "2.5s")
    """

    def __init__(self, console: "PipelineConsole"):
        """
        Initialize the pipeline reporter.

        Args:
            console: PipelineConsole instance for output formatting.
        """
        self.console = console
        self.logger = get_logger(__name__)

    def report_step_start(self, step: int, total: int, title: str) -> None:
        """Report that a pipeline step is starting."""
        self.logger.info(f"Starting step {step}/{total}: {title}")
        self.console.step_start(step, total, title)

    def report_step_complete(self, step: int, total: int, title: str, duration: str = "") -> None:
        """Report that a pipeline step completed successfully."""
        self.logger.info(f"Completed step {step}/{total}: {title} ({duration})")
        self.console.step_complete(step, total, title, duration)

    def report_step_skipped(self, step: int, total: int, title: str, reason: str = "") -> None:
        """Report that a pipeline step was skipped."""
        self.logger.info(f"Skipped step {step}/{total}: {title} - {reason}")
        self.console.step_skipped(step, total, title, reason)

    def report_progress(self, message: str) -> None:
        """Report a progress message within a step."""
        self.logger.debug(f"Progress: {message}")
        self.console.step(message)

    def report_info(self, message: str) -> None:
        """Report an informational message."""
        self.logger.info(message)
        self.console.info(message)

    def report_warning(self, message: str) -> None:
        """Report a warning message."""
        self.logger.warning(message)
        self.console.warning(message)

    def report_error(self, message: str) -> None:
        """Report an error message."""
        self.logger.error(message)
        self.console.error(message)

    def report_success(self, message: str) -> None:
        """Report a success message."""
        self.logger.info(f"Success: {message}")
        self.console.success(message)

    def print_raw(self, *args: Any, **kwargs: Any) -> None:
        """Print raw content to the console."""
        self.console.console.print(*args, **kwargs)

    def print_newline(self) -> None:
        """Print a blank line."""
        self.console.console.print()


class MockReporter(ReporterInterface):
    """
    Mock reporter for testing pipeline logic without console output.

    Records all reported messages for assertion in tests.

    Example:
        >>> reporter = MockReporter()
        >>> reporter.report_step_start(1, 4, "Processing")
        >>> reporter.report_info("Found data")
        >>> assert reporter.step_starts == [(1, 4, "Processing")]
        >>> assert "Found data" in reporter.infos
    """

    def __init__(self) -> None:
        """Initialize the mock reporter with empty message lists."""
        self.step_starts: list[tuple] = []
        self.step_completes: list[tuple] = []
        self.step_skips: list[tuple] = []
        self.progress_messages: list[str] = []
        self.infos: list[str] = []
        self.warnings: list[str] = []
        self.errors: list[str] = []
        self.successes: list[str] = []

    def report_step_start(self, step: int, total: int, title: str) -> None:
        """Record step start."""
        self.step_starts.append((step, total, title))

    def report_step_complete(self, step: int, total: int, title: str, duration: str = "") -> None:
        """Record step completion."""
        self.step_completes.append((step, total, title, duration))

    def report_step_skipped(self, step: int, total: int, title: str, reason: str = "") -> None:
        """Record step skip."""
        self.step_skips.append((step, total, title, reason))

    def report_progress(self, message: str) -> None:
        """Record progress message."""
        self.progress_messages.append(message)

    def report_info(self, message: str) -> None:
        """Record info message."""
        self.infos.append(message)

    def report_warning(self, message: str) -> None:
        """Record warning message."""
        self.warnings.append(message)

    def report_error(self, message: str) -> None:
        """Record error message."""
        self.errors.append(message)

    def report_success(self, message: str) -> None:
        """Record success message."""
        self.successes.append(message)

    def reset(self) -> None:
        """Clear all recorded messages for a fresh test."""
        self.step_starts.clear()
        self.step_completes.clear()
        self.step_skips.clear()
        self.progress_messages.clear()
        self.infos.clear()
        self.warnings.clear()
        self.errors.clear()
        self.successes.clear()
