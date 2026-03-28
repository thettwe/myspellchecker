"""Comprehensive tests for training/reporter.py.

Tests cover:
- TrainingReporter initialization
- Step reporting methods
- Message methods (info, success, warning, error, progress)
- Training metrics display
- Summary display
- Duration formatting
"""

import time
from io import StringIO

from rich.console import Console

from myspellchecker.training.reporter import TRAINING_THEME, TrainingReporter


def _make_reporter() -> tuple[TrainingReporter, StringIO]:
    """Create a TrainingReporter that writes to a StringIO buffer.

    Returns the reporter and the buffer so tests can inspect output.
    """
    buf = StringIO()
    reporter = TrainingReporter()
    reporter.console = Console(
        file=buf, theme=TRAINING_THEME, no_color=True, highlight=False, width=120
    )
    return reporter, buf


class TestTrainingReporterInitialization:
    """Tests for TrainingReporter initialization."""

    def test_basic_initialization(self):
        """Test basic initialization."""
        reporter = TrainingReporter()
        assert reporter.console is not None
        assert reporter._step_times == {}

    def test_initialization_with_force_plain(self):
        """Test initialization with force_plain option."""
        reporter = TrainingReporter(force_plain=True)
        assert reporter.console is not None


class TestTrainingReporterShowHeader:
    """Tests for show_header method."""

    def test_show_header(self):
        """Test show_header displays panel with expected content."""
        reporter, buf = _make_reporter()
        reporter.show_header(
            corpus_path="/path/to/corpus.txt",
            output_dir="/path/to/output",
            architecture="transformer",
            epochs=3,
            vocab_size=30000,
        )
        output = buf.getvalue()
        assert "/path/to/corpus.txt" in output
        assert "/path/to/output" in output
        assert "transformer" in output
        assert "30,000" in output


class TestTrainingReporterStepMethods:
    """Tests for step reporting methods."""

    def test_step_start(self):
        """Test step_start records start time."""
        reporter = TrainingReporter()
        reporter.step_start(1, 4, "Training")

        assert "Training" in reporter._step_times
        assert reporter._step_times["Training"] > 0

    def test_step_complete_with_tracked_time(self):
        """Test step_complete with tracked start time."""
        reporter, buf = _make_reporter()
        reporter.step_start(1, 4, "Training")
        time.sleep(0.01)
        reporter.step_complete(1, 4, "Training", "Model saved")
        output = buf.getvalue()
        assert "Training" in output

    def test_step_complete_without_tracked_time(self):
        """Test step_complete without tracked start time."""
        reporter, buf = _make_reporter()
        reporter.step_complete(1, 4, "Training")
        output = buf.getvalue()
        assert "Training" in output

    def test_step_complete_with_details(self):
        """Test step_complete with details."""
        reporter, buf = _make_reporter()
        reporter.step_start(1, 4, "Training")
        reporter.step_complete(1, 4, "Training", details="Saved to /output/model")
        output = buf.getvalue()
        assert "Training" in output
        assert "Saved to /output/model" in output

    def test_step_skipped_with_reason(self):
        """Test step_skipped with reason."""
        reporter, buf = _make_reporter()
        reporter.step_skipped(2, 4, "Validation", "No validation data")
        output = buf.getvalue()
        assert "Validation" in output
        assert "SKIPPED" in output
        assert "No validation data" in output

    def test_step_skipped_without_reason(self):
        """Test step_skipped without reason."""
        reporter, buf = _make_reporter()
        reporter.step_skipped(2, 4, "Validation")
        output = buf.getvalue()
        assert "Validation" in output
        assert "SKIPPED" in output


class TestTrainingReporterMessageMethods:
    """Tests for message methods."""

    def test_info(self):
        """Test info method outputs the message."""
        reporter, buf = _make_reporter()
        reporter.info("Loading model")
        output = buf.getvalue()
        assert "Loading model" in output

    def test_success(self):
        """Test success method outputs the message."""
        reporter, buf = _make_reporter()
        reporter.success("Model saved successfully")
        output = buf.getvalue()
        assert "Model saved successfully" in output

    def test_warning(self):
        """Test warning method outputs the message."""
        reporter, buf = _make_reporter()
        reporter.warning("Low GPU memory")
        output = buf.getvalue()
        assert "Low GPU memory" in output

    def test_error(self):
        """Test error method outputs the message."""
        reporter, buf = _make_reporter()
        reporter.error("Failed to load checkpoint")
        output = buf.getvalue()
        assert "Failed to load checkpoint" in output

    def test_progress(self):
        """Test progress method outputs the message."""
        reporter, buf = _make_reporter()
        reporter.progress("Processing batch 50/100")
        output = buf.getvalue()
        assert "Processing batch 50/100" in output


class TestTrainingReporterProgressBar:
    """Tests for progress bar method."""

    def test_show_training_progress(self):
        """Test show_training_progress creates Progress instance."""
        reporter = TrainingReporter()
        progress = reporter.show_training_progress()

        # Should return a Progress instance
        assert progress is not None


class TestTrainingReporterShowSummary:
    """Tests for show_summary method."""

    def test_show_summary_with_completed_steps(self):
        """Test show_summary with completed steps."""
        reporter, buf = _make_reporter()
        step_durations = {
            "Data Loading": ("completed", 5.5),
            "Training": ("completed", 120.0),
            "Validation": ("completed", 30.0),
        }
        reporter.show_summary(
            step_durations=step_durations,
            total_duration=155.5,
            output_dir="/output/model",
        )
        output = buf.getvalue()
        assert "Data Loading" in output
        assert "Training" in output
        assert "Validation" in output
        assert "Training Complete" in output

    def test_show_summary_with_skipped_steps(self):
        """Test show_summary with skipped steps."""
        reporter, buf = _make_reporter()
        step_durations = {
            "Data Loading": ("completed", 5.5),
            "Training": ("completed", 120.0),
            "Validation": ("skipped", 0.0),
        }
        reporter.show_summary(
            step_durations=step_durations,
            total_duration=125.5,
            output_dir="/output/model",
        )
        output = buf.getvalue()
        assert "Validation" in output
        assert "SKIPPED" in output

    def test_show_summary_all_skipped(self):
        """Test show_summary with all steps skipped."""
        reporter, buf = _make_reporter()
        step_durations = {
            "Step 1": ("skipped", 0.0),
            "Step 2": ("skipped", 0.0),
        }
        reporter.show_summary(
            step_durations=step_durations,
            total_duration=0.0,
            output_dir="/output/model",
        )
        output = buf.getvalue()
        assert "Step 1" in output
        assert "Step 2" in output
        assert "SKIPPED" in output


class TestTrainingReporterShowMetrics:
    """Tests for show_training_metrics method."""

    def test_show_training_metrics_basic(self):
        """Test show_training_metrics with basic metrics."""
        reporter, buf = _make_reporter()
        reporter.show_training_metrics(
            loss=0.5432,
            learning_rate=1e-4,
            epoch=1.5,
        )
        output = buf.getvalue()
        assert "0.5432" in output
        assert "1.00e-04" in output
        assert "1.50" in output

    def test_show_training_metrics_with_samples_per_second(self):
        """Test show_training_metrics with samples_per_second."""
        reporter, buf = _make_reporter()
        reporter.show_training_metrics(
            loss=0.5432,
            learning_rate=1e-4,
            epoch=1.5,
            samples_per_second=123.45,
        )
        output = buf.getvalue()
        assert "samples/s" in output
        assert "123" in output

    def test_show_training_metrics_edge_values(self):
        """Test show_training_metrics with edge values."""
        reporter, buf = _make_reporter()
        reporter.show_training_metrics(
            loss=0.0,
            learning_rate=1e-10,
            epoch=0.0,
            samples_per_second=0.1,
        )
        output = buf.getvalue()
        assert "0.0000" in output
        assert "0.00" in output


class TestTrainingReporterFormatDuration:
    """Tests for _format_duration method."""

    def test_format_duration_seconds(self):
        """Test formatting duration in seconds."""
        reporter = TrainingReporter()
        result = reporter._format_duration(45.5)
        assert result == "45.5s"

    def test_format_duration_minutes(self):
        """Test formatting duration in minutes."""
        reporter = TrainingReporter()
        result = reporter._format_duration(125.0)
        assert result == "2m 5s"

    def test_format_duration_hours(self):
        """Test formatting duration in hours."""
        reporter = TrainingReporter()
        result = reporter._format_duration(3725.0)
        assert result == "1h 2m"

    def test_format_duration_zero(self):
        """Test formatting zero duration."""
        reporter = TrainingReporter()
        result = reporter._format_duration(0.0)
        assert result == "0.0s"

    def test_format_duration_large_hours(self):
        """Test formatting large hour values."""
        reporter = TrainingReporter()
        result = reporter._format_duration(36000.0)  # 10 hours
        assert result == "10h 0m"

    def test_format_duration_boundary_60s(self):
        """Test formatting exactly 60 seconds."""
        reporter = TrainingReporter()
        result = reporter._format_duration(60.0)
        assert result == "1m 0s"

    def test_format_duration_boundary_3600s(self):
        """Test formatting exactly 3600 seconds (1 hour)."""
        reporter = TrainingReporter()
        result = reporter._format_duration(3600.0)
        assert result == "1h 0m"


class TestTrainingReporterIntegration:
    """Integration tests for TrainingReporter."""

    def test_full_workflow(self):
        """Test a full training workflow."""
        reporter, buf = _make_reporter()

        # Show header
        reporter.show_header(
            corpus_path="data/corpus.txt",
            output_dir="output/model",
            architecture="transformer",
            epochs=5,
            vocab_size=32000,
        )

        # Step 1: Data loading
        reporter.step_start(1, 3, "Data Loading")
        reporter.info("Found 10000 samples")
        reporter.progress("Loading training data")
        reporter.step_complete(1, 3, "Data Loading")

        # Step 2: Training
        reporter.step_start(2, 3, "Training")
        reporter.show_training_metrics(loss=0.8, learning_rate=1e-4, epoch=0.5)
        reporter.show_training_metrics(
            loss=0.3, learning_rate=5e-5, epoch=1.0, samples_per_second=100.0
        )
        reporter.step_complete(2, 3, "Training")

        # Step 3: Validation (skipped)
        reporter.step_skipped(3, 3, "Validation", "No validation set")

        # Show summary
        reporter.show_summary(
            step_durations={
                "Data Loading": ("completed", 10.0),
                "Training": ("completed", 300.0),
                "Validation": ("skipped", 0.0),
            },
            total_duration=310.0,
            output_dir="output/model",
        )

        reporter.success("Training pipeline completed")

        output = buf.getvalue()
        assert "data/corpus.txt" in output
        assert "Data Loading" in output
        assert "Training" in output
        assert "Validation" in output
        assert "SKIPPED" in output
        assert "Training pipeline completed" in output

    def test_error_workflow(self):
        """Test workflow with errors."""
        reporter, buf = _make_reporter()

        reporter.step_start(1, 2, "Data Loading")
        reporter.warning("Some files missing")
        reporter.error("Failed to load checkpoint")
        reporter.step_complete(1, 2, "Data Loading")

        output = buf.getvalue()
        assert "Some files missing" in output
        assert "Failed to load checkpoint" in output


class TestTrainingReporterEdgeCases:
    """Test edge cases."""

    def test_unicode_content(self):
        """Test Unicode content handling."""
        reporter, buf = _make_reporter()
        reporter.info("Myanmar: \u1019\u103c\u1014\u103a\u1019\u102c")
        reporter.success("Chinese: \u4e2d\u6587")
        reporter.warning("Japanese: \u65e5\u672c\u8a9e")
        output = buf.getvalue()
        assert "\u1019\u103c\u1014\u103a\u1019\u102c" in output
        assert "\u4e2d\u6587" in output
        assert "\u65e5\u672c\u8a9e" in output

    def test_empty_strings(self):
        """Test empty string handling."""
        reporter, buf = _make_reporter()
        reporter.info("")
        reporter.step_complete(1, 1, "", "")
        # Should produce some output (at least the Rich markup symbols)
        output = buf.getvalue()
        assert len(output) > 0

    def test_special_characters(self):
        """Test special character handling."""
        reporter, buf = _make_reporter()
        reporter.info("Path: /path/to/file.txt")
        reporter.info("Percent: 50%")
        reporter.info("Newline: line1\nline2")
        output = buf.getvalue()
        assert "/path/to/file.txt" in output
        assert "50%" in output
