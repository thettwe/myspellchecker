"""
Tests for enhanced training features.

Tests ModelArchitecture enum, TrainingMetricsCallback, and TrainingConfig
with new parameters: architecture, resume_from_checkpoint, warmup_ratio,
weight_decay, save_metrics.
"""

import importlib
import sys

import pytest

import myspellchecker.training.trainer


def _restore_transformers_module():
    """
    Restore the real transformers module if it was mocked.

    test_training_paths.py does module-level mocking that can corrupt
    sys.modules['transformers'] with a MagicMock.
    """
    transformers_module = sys.modules.get("transformers")
    if transformers_module is not None:
        # Check if it's a mock (MagicMock has _mock_name attribute)
        if hasattr(transformers_module, "_mock_name"):
            # Remove the mock so the real module can be imported
            sys.modules.pop("transformers", None)
            sys.modules.pop("transformers.trainer_callback", None)


@pytest.fixture(scope="module", autouse=True)
def ensure_clean_trainer_module():
    """
    Ensure the trainer module is properly loaded.

    This is needed because test_training_paths.py does module-level mocking
    that can corrupt the trainer module if it runs before this test file.
    """
    # First, restore the real transformers module if it was mocked
    _restore_transformers_module()

    # Reload the trainer module to ensure we have the real classes
    importlib.reload(myspellchecker.training.trainer)
    yield


class TestModelArchitecture:
    """Tests for ModelArchitecture enum."""

    def test_model_architecture_roberta_value(self):
        """RoBERTa architecture has correct value."""
        from myspellchecker.training.trainer import ModelArchitecture

        assert ModelArchitecture.ROBERTA.value == "roberta"

    def test_model_architecture_bert_value(self):
        """BERT architecture has correct value."""
        from myspellchecker.training.trainer import ModelArchitecture

        assert ModelArchitecture.BERT.value == "bert"

    def test_model_architecture_from_string_roberta(self):
        """from_string creates RoBERTa from string."""
        from myspellchecker.training.trainer import ModelArchitecture

        result = ModelArchitecture.from_string("roberta")
        assert result == ModelArchitecture.ROBERTA

    def test_model_architecture_from_string_bert(self):
        """from_string creates BERT from string."""
        from myspellchecker.training.trainer import ModelArchitecture

        result = ModelArchitecture.from_string("bert")
        assert result == ModelArchitecture.BERT

    def test_model_architecture_from_string_case_insensitive(self):
        """from_string is case insensitive."""
        from myspellchecker.training.trainer import ModelArchitecture

        assert ModelArchitecture.from_string("ROBERTA") == ModelArchitecture.ROBERTA
        assert ModelArchitecture.from_string("RoBERTa") == ModelArchitecture.ROBERTA
        assert ModelArchitecture.from_string("BERT") == ModelArchitecture.BERT
        assert ModelArchitecture.from_string("Bert") == ModelArchitecture.BERT

    def test_model_architecture_from_string_invalid(self):
        """from_string raises ValueError for invalid architecture."""
        from myspellchecker.training.trainer import ModelArchitecture

        with pytest.raises(ValueError) as exc_info:
            ModelArchitecture.from_string("gpt")
        assert "Unknown architecture" in str(exc_info.value)
        assert "gpt" in str(exc_info.value)

    def test_model_architecture_is_string_subclass(self):
        """ModelArchitecture is a string subclass for JSON serialization."""
        from myspellchecker.training.trainer import ModelArchitecture

        assert isinstance(ModelArchitecture.ROBERTA, str)
        # As a str subclass, the value is the string
        assert ModelArchitecture.ROBERTA.value == "roberta"


class TestTrainingConfigEnhancements:
    """Tests for enhanced TrainingConfig parameters."""

    def test_training_config_default_architecture(self):
        """Default architecture is roberta."""
        from myspellchecker.training.pipeline import TrainingConfig

        config = TrainingConfig(input_file="test.txt", output_dir="./output")
        assert config.architecture == "roberta"

    def test_training_config_custom_architecture_roberta(self):
        """Can set architecture to roberta."""
        from myspellchecker.training.pipeline import TrainingConfig

        config = TrainingConfig(
            input_file="test.txt", output_dir="./output", architecture="roberta"
        )
        assert config.architecture == "roberta"

    def test_training_config_custom_architecture_bert(self):
        """Can set architecture to bert."""
        from myspellchecker.training.pipeline import TrainingConfig

        config = TrainingConfig(input_file="test.txt", output_dir="./output", architecture="bert")
        assert config.architecture == "bert"

    def test_training_config_invalid_architecture(self):
        """Invalid architecture raises ValueError."""
        from myspellchecker.training.pipeline import TrainingConfig

        with pytest.raises(ValueError) as exc_info:
            TrainingConfig(input_file="test.txt", output_dir="./output", architecture="gpt")
        assert "Invalid architecture" in str(exc_info.value)

    def test_training_config_default_resume_from_checkpoint(self):
        """Default resume_from_checkpoint is None."""
        from myspellchecker.training.pipeline import TrainingConfig

        config = TrainingConfig(input_file="test.txt", output_dir="./output")
        assert config.resume_from_checkpoint is None

    def test_training_config_custom_resume_from_checkpoint(self):
        """Can set resume_from_checkpoint."""
        from myspellchecker.training.pipeline import TrainingConfig

        config = TrainingConfig(
            input_file="test.txt",
            output_dir="./output",
            resume_from_checkpoint="./checkpoint-500",
        )
        assert config.resume_from_checkpoint == "./checkpoint-500"

    def test_training_config_default_warmup_ratio(self):
        """Default warmup_ratio is 0.1."""
        from myspellchecker.training.pipeline import TrainingConfig

        config = TrainingConfig(input_file="test.txt", output_dir="./output")
        assert config.warmup_ratio == 0.1

    def test_training_config_custom_warmup_ratio(self):
        """Can set custom warmup_ratio."""
        from myspellchecker.training.pipeline import TrainingConfig

        config = TrainingConfig(input_file="test.txt", output_dir="./output", warmup_ratio=0.2)
        assert config.warmup_ratio == 0.2

    def test_training_config_warmup_ratio_zero_valid(self):
        """warmup_ratio of 0 is valid (no warmup)."""
        from myspellchecker.training.pipeline import TrainingConfig

        config = TrainingConfig(input_file="test.txt", output_dir="./output", warmup_ratio=0.0)
        assert config.warmup_ratio == 0.0

    def test_training_config_warmup_ratio_invalid_negative(self):
        """Negative warmup_ratio raises ValueError."""
        from myspellchecker.training.pipeline import TrainingConfig

        with pytest.raises(ValueError) as exc_info:
            TrainingConfig(input_file="test.txt", output_dir="./output", warmup_ratio=-0.1)
        assert "warmup_ratio" in str(exc_info.value)

    def test_training_config_warmup_ratio_invalid_too_high(self):
        """warmup_ratio >= 1 raises ValueError."""
        from myspellchecker.training.pipeline import TrainingConfig

        with pytest.raises(ValueError) as exc_info:
            TrainingConfig(input_file="test.txt", output_dir="./output", warmup_ratio=1.0)
        assert "warmup_ratio" in str(exc_info.value)

    def test_training_config_default_weight_decay(self):
        """Default weight_decay is 0.01."""
        from myspellchecker.training.pipeline import TrainingConfig

        config = TrainingConfig(input_file="test.txt", output_dir="./output")
        assert config.weight_decay == 0.01

    def test_training_config_custom_weight_decay(self):
        """Can set custom weight_decay."""
        from myspellchecker.training.pipeline import TrainingConfig

        config = TrainingConfig(input_file="test.txt", output_dir="./output", weight_decay=0.05)
        assert config.weight_decay == 0.05

    def test_training_config_weight_decay_zero_valid(self):
        """weight_decay of 0 is valid (no weight decay)."""
        from myspellchecker.training.pipeline import TrainingConfig

        config = TrainingConfig(input_file="test.txt", output_dir="./output", weight_decay=0.0)
        assert config.weight_decay == 0.0

    def test_training_config_weight_decay_invalid_negative(self):
        """Negative weight_decay raises ValueError."""
        from myspellchecker.training.pipeline import TrainingConfig

        with pytest.raises(ValueError) as exc_info:
            TrainingConfig(input_file="test.txt", output_dir="./output", weight_decay=-0.01)
        assert "weight_decay" in str(exc_info.value)

    def test_training_config_default_save_metrics(self):
        """Default save_metrics is True."""
        from myspellchecker.training.pipeline import TrainingConfig

        config = TrainingConfig(input_file="test.txt", output_dir="./output")
        assert config.save_metrics is True

    def test_training_config_save_metrics_false(self):
        """Can set save_metrics to False."""
        from myspellchecker.training.pipeline import TrainingConfig

        config = TrainingConfig(input_file="test.txt", output_dir="./output", save_metrics=False)
        assert config.save_metrics is False


class TestTrainingMetricsCallback:
    """Tests for TrainingMetricsCallback."""

    def test_metrics_callback_import(self):
        """TrainingMetricsCallback can be imported."""
        from myspellchecker.training.trainer import TrainingMetricsCallback

        assert TrainingMetricsCallback is not None

    def test_metrics_callback_init(self, tmp_path):
        """TrainingMetricsCallback initializes correctly."""
        from myspellchecker.training.trainer import TrainingMetricsCallback

        out = str(tmp_path)
        callback = TrainingMetricsCallback(output_dir=out)
        assert callback.output_dir == out
        assert callback.metrics == []
        assert callback.metrics_file == f"{out}/training_metrics.json"

    def test_metrics_callback_export_from_module(self):
        """TrainingMetricsCallback exported from training module."""
        from myspellchecker.training import TrainingMetricsCallback

        assert TrainingMetricsCallback is not None


class TestModuleExports:
    """Tests for training module exports."""

    def test_model_architecture_exported(self):
        """ModelArchitecture exported from training module."""
        from myspellchecker.training import ModelArchitecture

        assert ModelArchitecture is not None
        assert hasattr(ModelArchitecture, "ROBERTA")
        assert hasattr(ModelArchitecture, "BERT")

    def test_training_metrics_callback_exported(self):
        """TrainingMetricsCallback exported from training module."""
        from myspellchecker.training import TrainingMetricsCallback

        assert TrainingMetricsCallback is not None

    def test_all_exports_include_new_classes(self):
        """__all__ includes new classes."""
        from myspellchecker.training import __all__

        assert "ModelArchitecture" in __all__
        assert "TrainingMetricsCallback" in __all__
