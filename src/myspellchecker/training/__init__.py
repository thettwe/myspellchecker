"""
Training module for creating custom Semantic Models.

This module contains tools to train custom Tokenizers and Masked Language Models
(RoBERTa) from raw text corpora, and export them to ONNX for use with the
SemanticChecker.

This module requires the 'train' optional dependencies:
    pip install myspellchecker[train]
"""

# Define exports - classes and constants
__all__ = [
    "ModelTrainer",
    "ModelArchitecture",
    "TrainingMetricsCallback",
    "ONNXExporter",
    "TrainingPipeline",
    "TrainingConfig",
    # Pydantic config classes (training/config.py)
    "TrainerConfig",
    "ExporterConfig",
    "EmbeddingSurgeryConfig",
    "SyntheticErrorGeneratorConfig",
    # Synthetic error generation (for denoising/data augmentation)
    "SyntheticErrorGenerator",
    # Corpus preprocessing (no optional dependencies)
    "CorpusPreprocessor",
    "PreprocessingConfig",
    "PreprocessingReport",
    # Constants
    "SPECIAL_TOKENS",
    "SPECIAL_TOKENS_MAP",
    "DEFAULT_DUMMY_TEXT",
    "MIN_TRAINING_LINES",
    "LARGE_FILE_WARNING_MB",
    "DEFAULT_VOCAB_SIZE",
    "DEFAULT_MIN_FREQUENCY",
    "DEFAULT_OPSET_VERSION",
]


# Lazy imports to avoid crashing if dependencies are missing
def __getattr__(name):
    if name == "ModelTrainer":
        from myspellchecker.training.trainer import ModelTrainer

        return ModelTrainer
    if name == "ModelArchitecture":
        from myspellchecker.training.trainer import ModelArchitecture

        return ModelArchitecture
    if name == "TrainingMetricsCallback":
        from myspellchecker.training.trainer import TrainingMetricsCallback

        return TrainingMetricsCallback
    if name == "ONNXExporter":
        from myspellchecker.training.exporter import ONNXExporter

        return ONNXExporter
    if name == "TrainingPipeline":
        from myspellchecker.training.pipeline import TrainingPipeline

        return TrainingPipeline
    if name == "TrainingConfig":
        from myspellchecker.training.pipeline import TrainingConfig

        return TrainingConfig

    # Pydantic config classes (no optional dependencies required)
    _config_classes = {
        "TrainerConfig",
        "ExporterConfig",
        "EmbeddingSurgeryConfig",
        "SyntheticErrorGeneratorConfig",
    }
    if name in _config_classes:
        from myspellchecker.training import config as _training_config

        return getattr(_training_config, name)

    # Synthetic error generation
    if name == "SyntheticErrorGenerator":
        from myspellchecker.training.generator import SyntheticErrorGenerator

        return SyntheticErrorGenerator

    # Corpus preprocessing (no optional dependencies required)
    if name == "CorpusPreprocessor":
        from myspellchecker.training.corpus_preprocessor import CorpusPreprocessor

        return CorpusPreprocessor
    if name == "PreprocessingConfig":
        from myspellchecker.training.corpus_preprocessor import PreprocessingConfig

        return PreprocessingConfig
    if name == "PreprocessingReport":
        from myspellchecker.training.corpus_preprocessor import PreprocessingReport

        return PreprocessingReport

    # Constants (no optional dependencies required)
    constants_map = {
        "SPECIAL_TOKENS",
        "SPECIAL_TOKENS_MAP",
        "DEFAULT_DUMMY_TEXT",
        "MIN_TRAINING_LINES",
        "LARGE_FILE_WARNING_MB",
        "DEFAULT_VOCAB_SIZE",
        "DEFAULT_MIN_FREQUENCY",
        "DEFAULT_OPSET_VERSION",
    }
    if name in constants_map:
        from myspellchecker.training import constants

        return getattr(constants, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
