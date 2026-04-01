"""
Unit tests for the training module.

Tests cover:
- TrainingConfig dataclass validation
- ModelTrainer with mocked torch/transformers
- TrainingPipeline with mocked dependencies
- ONNXExporter with mocked onnx/torch
- TRAINING_AVAILABLE flag behavior

All tests use mocking to avoid heavy ML dependencies and ensure fast execution.
"""

import importlib
import os
import sys
import tempfile
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest


def _make_stub_module(name: str, **attrs):
    """Create a lightweight module stub with explicit attributes."""
    module = ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


class TestTrainingConfig:
    """Test TrainingConfig dataclass."""

    def test_training_config_creation_with_defaults(self):
        """Test TrainingConfig creation with required fields and check defaults."""
        from myspellchecker.training.pipeline import TrainingConfig

        config = TrainingConfig(input_file="/tmp/corpus.txt", output_dir="/tmp/output")

        assert config.input_file == "/tmp/corpus.txt"
        assert config.output_dir == "/tmp/output"
        assert config.vocab_size == 30_000
        assert config.epochs == 5
        assert config.batch_size == 16
        assert config.learning_rate == 5e-5
        assert config.keep_checkpoints is False
        assert config.min_frequency == 2
        assert config.hidden_size % config.num_heads == 0

    def test_training_config_with_custom_values(self):
        """Test TrainingConfig with custom values."""
        from myspellchecker.training.pipeline import TrainingConfig

        config = TrainingConfig(
            input_file="/tmp/corpus.txt",
            output_dir="/tmp/output",
            vocab_size=10000,
            epochs=10,
            batch_size=32,
            learning_rate=1e-4,
            hidden_size=512,
            num_layers=8,
            num_heads=8,
            max_length=256,
        )

        assert config.vocab_size == 10000
        assert config.epochs == 10
        assert config.batch_size == 32
        assert config.learning_rate == 1e-4
        assert config.hidden_size == 512
        assert config.num_layers == 8
        assert config.num_heads == 8
        assert config.max_length == 256

    def test_training_config_validates_epochs(self):
        """Test that epochs < 1 raises ValueError."""
        from myspellchecker.training.pipeline import TrainingConfig

        with pytest.raises(ValueError, match="epochs must be >= 1"):
            TrainingConfig(input_file="/tmp/corpus.txt", output_dir="/tmp/output", epochs=0)

    def test_training_config_validates_vocab_size(self):
        """Test that vocab_size < 100 raises ValueError."""
        from myspellchecker.training.pipeline import TrainingConfig

        with pytest.raises(ValueError, match="vocab_size must be >= 100"):
            TrainingConfig(input_file="/tmp/corpus.txt", output_dir="/tmp/output", vocab_size=50)

    def test_training_config_validates_batch_size(self):
        """Test that batch_size < 1 raises ValueError."""
        from myspellchecker.training.pipeline import TrainingConfig

        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            TrainingConfig(input_file="/tmp/corpus.txt", output_dir="/tmp/output", batch_size=0)

    def test_training_config_validates_num_layers(self):
        """Test that num_layers < 1 raises ValueError."""
        from myspellchecker.training.pipeline import TrainingConfig

        with pytest.raises(ValueError, match="num_layers must be >= 1"):
            TrainingConfig(input_file="/tmp/corpus.txt", output_dir="/tmp/output", num_layers=0)

    def test_training_config_validates_num_heads(self):
        """Test that num_heads < 1 raises ValueError."""
        from myspellchecker.training.pipeline import TrainingConfig

        with pytest.raises(ValueError, match="num_heads must be >= 1"):
            TrainingConfig(input_file="/tmp/corpus.txt", output_dir="/tmp/output", num_heads=0)

    def test_training_config_validates_max_length(self):
        """Test that max_length < 16 raises ValueError."""
        from myspellchecker.training.pipeline import TrainingConfig

        with pytest.raises(ValueError, match="max_length must be >= 16"):
            TrainingConfig(input_file="/tmp/corpus.txt", output_dir="/tmp/output", max_length=8)

    def test_training_config_validates_hidden_size(self):
        """Test that hidden_size < 1 raises ValueError."""
        from myspellchecker.training.pipeline import TrainingConfig

        with pytest.raises(ValueError, match="hidden_size must be >= 1"):
            TrainingConfig(input_file="/tmp/corpus.txt", output_dir="/tmp/output", hidden_size=0)

    def test_training_config_validates_learning_rate(self):
        """Test that learning_rate <= 0 raises ValueError."""
        from myspellchecker.training.pipeline import TrainingConfig

        with pytest.raises(ValueError, match="learning_rate must be > 0"):
            TrainingConfig(input_file="/tmp/corpus.txt", output_dir="/tmp/output", learning_rate=0)

        with pytest.raises(ValueError, match="learning_rate must be > 0"):
            TrainingConfig(
                input_file="/tmp/corpus.txt", output_dir="/tmp/output", learning_rate=-0.001
            )

    def test_training_config_validates_hidden_size_divisible_by_heads(self):
        """Test that hidden_size must be divisible by num_heads."""
        from myspellchecker.training.pipeline import TrainingConfig

        with pytest.raises(ValueError, match="hidden_size.*must be divisible by num_heads"):
            TrainingConfig(
                input_file="/tmp/corpus.txt", output_dir="/tmp/output", hidden_size=256, num_heads=3
            )


class TestModelTrainer:
    """Test ModelTrainer class with mocked dependencies."""

    def test_trainer_raises_import_error_without_torch(self):
        """Test ModelTrainer raises ImportError when torch is missing."""
        import myspellchecker.training.trainer as trainer_module

        original_torch = trainer_module.torch

        try:
            trainer_module.torch = None
            from myspellchecker.training.trainer import ModelTrainer

            with pytest.raises(ImportError, match="Training requires"):
                ModelTrainer()
        finally:
            trainer_module.torch = original_torch


class TestLineByLineDataset:
    """Test LineByLineDataset class."""

    @pytest.mark.skipif(
        not pytest.importorskip("torch", reason="torch not installed"), reason="torch not installed"
    )
    def test_dataset_file_not_found(self):
        """Test LineByLineDataset raises FileNotFoundError."""
        try:
            from myspellchecker.training.trainer import LineByLineDataset

            mock_tokenizer = MagicMock()

            with pytest.raises(FileNotFoundError, match="Corpus file not found"):
                LineByLineDataset(
                    tokenizer=mock_tokenizer, file_path="/nonexistent/path.txt", block_size=128
                )
        except ImportError:
            pytest.skip("Training dependencies not installed")

    @pytest.mark.skipif(
        not pytest.importorskip("torch", reason="torch not installed"), reason="torch not installed"
    )
    def test_dataset_empty_file_raises_error(self):
        """Test LineByLineDataset raises error on empty file."""
        try:
            from myspellchecker.training.trainer import LineByLineDataset

            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
                f.write("")
                temp_path = f.name

            try:
                mock_tokenizer = MagicMock()
                mock_tokenizer.return_value = {"input_ids": []}

                with pytest.raises(ValueError, match="No valid lines found"):
                    LineByLineDataset(tokenizer=mock_tokenizer, file_path=temp_path, block_size=128)
            finally:
                os.unlink(temp_path)
        except ImportError:
            pytest.skip("Training dependencies not installed")


class TestTrainingPipeline:
    """Test TrainingPipeline class with mocked dependencies."""

    def test_pipeline_raises_without_dependencies(self):
        """Test TrainingPipeline raises ImportError without training deps."""
        import myspellchecker.training.trainer as trainer_module

        original_torch = trainer_module.torch

        try:
            trainer_module.torch = None
            from myspellchecker.training.pipeline import TrainingPipeline

            with pytest.raises(ImportError):
                TrainingPipeline()
        finally:
            trainer_module.torch = original_torch

    @pytest.mark.skipif(
        not pytest.importorskip("torch", reason="torch not installed"), reason="torch not installed"
    )
    def test_pipeline_run_file_not_found(self):
        """Test pipeline run raises FileNotFoundError for missing input."""
        try:
            from myspellchecker.training.pipeline import TrainingConfig, TrainingPipeline

            pipeline = TrainingPipeline()
            config = TrainingConfig(input_file="/nonexistent/corpus.txt", output_dir="/tmp/output")

            with pytest.raises(FileNotFoundError, match="Input corpus not found"):
                pipeline.run(config)
        except ImportError:
            pytest.skip("Training dependencies not installed")


class TestONNXExporter:
    """Test ONNXExporter class with mocked dependencies."""

    def test_exporter_raises_import_error_without_torch(self):
        """Test ONNXExporter raises ImportError when dependencies missing."""
        import myspellchecker.training.exporter as exporter_module

        original_torch = exporter_module.torch

        try:
            exporter_module.torch = None
            from myspellchecker.training.exporter import ONNXExporter

            with pytest.raises(ImportError, match="Exporting requires"):
                ONNXExporter()
        finally:
            exporter_module.torch = original_torch

    def test_exporter_export_invalid_model_dir(self, monkeypatch):
        """Test exporter raises on invalid model directory deterministically."""
        pytest.importorskip("transformers", reason="transformers required for exporter test")
        import myspellchecker.training.exporter as exporter_module

        class _FailingTokenizerLoader:
            @staticmethod
            def from_pretrained(model_dir):
                raise OSError(f"missing tokenizer for {model_dir}")

        class _FailingModelLoader:
            @staticmethod
            def from_pretrained(model_dir):
                raise OSError(f"missing model for {model_dir}")

        monkeypatch.setattr(exporter_module, "torch", object())
        monkeypatch.setattr(exporter_module, "onnxruntime", object())
        monkeypatch.setattr(exporter_module, "PreTrainedTokenizerFast", _FailingTokenizerLoader)
        monkeypatch.setattr(exporter_module, "AutoModelForMaskedLM", _FailingModelLoader)

        exporter = exporter_module.ONNXExporter()

        with tempfile.TemporaryDirectory() as output_dir:
            with pytest.raises(ValueError, match="Failed to load model"):
                exporter.export(model_dir="/nonexistent/model", output_dir=output_dir)


class TestTrainingFlagConstants:
    """Test training availability flags."""

    def test_flags_reflect_import_status(self):
        """Test TRAINING_AVAILABLE by reloading trainer under controlled imports."""
        import myspellchecker.training.trainer as trainer_module

        # Force dependency import failure at module load.
        with patch.dict(sys.modules, {"transformers": _make_stub_module("transformers")}):
            trainer_module = importlib.reload(trainer_module)
            assert trainer_module.TRAINING_AVAILABLE is False

        # Force dependency import success with local stub modules.
        torch_data_stub = _make_stub_module(
            "torch.utils.data",
            Dataset=object,
            IterableDataset=object,
        )
        torch_utils_stub = _make_stub_module("torch.utils", data=torch_data_stub)
        torch_stub = _make_stub_module("torch", utils=torch_utils_stub)
        stub_modules = {
            "torch": torch_stub,
            "torch.utils": torch_utils_stub,
            "torch.utils.data": torch_data_stub,
            "tokenizers": _make_stub_module("tokenizers", ByteLevelBPETokenizer=object),
            "transformers": _make_stub_module(
                "transformers",
                BertConfig=object,
                BertForMaskedLM=object,
                DataCollatorForLanguageModeling=object,
                PreTrainedTokenizerFast=object,
                RobertaConfig=object,
                RobertaForMaskedLM=object,
                Trainer=object,
                TrainerCallback=object,
                TrainerControl=object,
                TrainerState=object,
                TrainingArguments=object,
            ),
        }
        with patch.dict(sys.modules, stub_modules):
            trainer_module = importlib.reload(trainer_module)
            assert trainer_module.TRAINING_AVAILABLE is True

        # Reload once more after patch contexts to restore normal module state.
        importlib.reload(trainer_module)
