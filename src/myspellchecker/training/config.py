"""
Training Configuration Classes.

Pydantic-based configuration for training hyperparameters and pipeline parameters.
Defaults match the canonical values in ``training/constants.py`` exactly.

Classes:
    - TrainerConfig: Tokenizer and MLM training defaults
    - ExporterConfig: ONNX export parameters
    - EmbeddingSurgeryConfig: Embedding surgery warmup and learning rate
    - SyntheticErrorGeneratorConfig: Error generator length limits
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class TrainerConfig(BaseModel):
    """
    Configuration for MLM model training hyperparameters.

    These values mirror the constants in ``training/constants.py`` and serve as
    the programmatic entry point when callers want to override defaults.

    Attributes:
        vocab_size: BPE vocabulary size for tokenizer training (default: 30000).
        min_token_frequency: Minimum token frequency for BPE merges (default: 2).
        mlm_probability: Masked language model masking ratio (default: 0.15).
        warmup_ratio: Fraction of total steps used for LR warmup (default: 0.1).
        weight_decay: AdamW weight decay coefficient (default: 0.01).
        save_total_limit: Maximum number of checkpoints to keep (default: 3).
        max_workers_cap: Hard cap on DataLoader num_workers (default: 32).
        min_training_lines: Minimum corpus lines for meaningful training (default: 1000).
        large_file_warning_mb: Warn about memory usage above this size in MB (default: 100).
    """

    model_config = ConfigDict(
        validate_assignment=True,
        frozen=False,
        extra="forbid",
    )

    vocab_size: int = Field(
        default=30_000,
        ge=100,
        description="BPE vocabulary size for tokenizer training",
    )
    min_token_frequency: int = Field(
        default=2,
        ge=1,
        description="Minimum token frequency for BPE merges",
    )
    mlm_probability: float = Field(
        default=0.15,
        gt=0.0,
        lt=1.0,
        description="Masked language model masking ratio",
    )
    warmup_ratio: float = Field(
        default=0.1,
        ge=0.0,
        lt=1.0,
        description="Fraction of total steps used for learning rate warmup",
    )
    weight_decay: float = Field(
        default=0.01,
        ge=0.0,
        description="AdamW weight decay coefficient",
    )
    save_total_limit: int = Field(
        default=3,
        ge=1,
        description="Maximum number of checkpoints to keep on disk",
    )
    max_workers_cap: int = Field(
        default=32,
        ge=1,
        description="Hard cap on DataLoader num_workers to prevent over-subscription",
    )
    min_training_lines: int = Field(
        default=1000,
        ge=1,
        description="Minimum corpus lines required for meaningful model training",
    )
    large_file_warning_mb: int = Field(
        default=100,
        ge=1,
        description="Warn user about memory usage above this corpus size (MB)",
    )


class ExporterConfig(BaseModel):
    """
    Configuration for ONNX model export.

    Attributes:
        default_opset_version: ONNX opset version for export (default: 18).
            Use opset 18+ for compatibility with modern PyTorch/ONNX (torch>=2.0).
    """

    model_config = ConfigDict(
        validate_assignment=True,
        frozen=False,
        extra="forbid",
    )

    default_opset_version: int = Field(
        default=18,
        ge=9,
        description="ONNX opset version for export (18+ recommended for torch>=2.0)",
    )


class EmbeddingSurgeryConfig(BaseModel):
    """
    Configuration for embedding surgery during vocabulary expansion.

    Embedding surgery transfers knowledge from a pre-trained model's embedding
    layer to a new vocabulary, using a warmup phase to stabilize the new embeddings.

    Attributes:
        warmup_steps: Number of warmup steps for embedding surgery (default: 25000).
        learning_rate: Learning rate for embedding warmup phase (default: 1e-3).
    """

    model_config = ConfigDict(
        validate_assignment=True,
        frozen=False,
        extra="forbid",
    )

    warmup_steps: int = Field(
        default=25_000,
        ge=0,
        description="Number of warmup steps for embedding surgery",
    )
    learning_rate: float = Field(
        default=1e-3,
        gt=0.0,
        description="Learning rate for embedding warmup phase",
    )


class SyntheticErrorGeneratorConfig(BaseModel):
    """
    Configuration for the synthetic error generator used in MLM denoising training.

    Controls the maximum input length and syllable count for corrupted text generation.

    Attributes:
        max_length: Maximum BPE token length for generated text (default: 256).
        max_syllables: Maximum Myanmar syllables per sentence (default: 80).
            Derived from max_length assuming ~3 BPE tokens per syllable.
    """

    model_config = ConfigDict(
        validate_assignment=True,
        frozen=False,
        extra="forbid",
    )

    max_length: int = Field(
        default=256,
        ge=16,
        description="Maximum BPE token length for generated text",
    )
    max_syllables: int = Field(
        default=80,
        ge=1,
        description=(
            "Maximum Myanmar syllables per sentence. "
            "Derived from (max_length - 16) // 3 at default values."
        ),
    )
