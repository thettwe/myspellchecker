"""
Training module constants.

Centralizes configuration values used across trainer.py, exporter.py, and pipeline.py.

These module-level constants serve as canonical defaults.  When used
programmatically, prefer the Pydantic config classes in ``training.config``
(TrainerConfig, ExporterConfig, EmbeddingSurgeryConfig, SyntheticErrorGeneratorConfig)
which expose the same defaults with validation and override support.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Special tokens for tokenizer and model (RoBERTa convention)
SPECIAL_TOKENS: list[str] = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]

# Special tokens mapping for PreTrainedTokenizerFast
SPECIAL_TOKENS_MAP: dict[str, str] = {
    "bos_token": "<s>",
    "eos_token": "</s>",
    "unk_token": "<unk>",
    "sep_token": "</s>",
    "pad_token": "<pad>",
    "cls_token": "<s>",
    "mask_token": "<mask>",
}

# Default Myanmar text for ONNX export dummy input
# This sentence covers common syllable patterns for representative encoding
DEFAULT_DUMMY_TEXT: str = "မြန်မာစကားကို အခြခံပြီး တည်ဆောက်ထားသော မော်ဒယ် ဖြစ်သည်။"

# Training thresholds
MIN_TRAINING_LINES: int = 1000  # Minimum lines for meaningful model training
LARGE_FILE_WARNING_MB: int = 100  # Warn user about memory usage above this size

# Tokenizer defaults
DEFAULT_VOCAB_SIZE: int = 30_000
DEFAULT_MIN_FREQUENCY: int = 2

# ONNX export defaults
# Use opset 18+ for compatibility with modern PyTorch/ONNX (torch>=2.0 exports to opset 18)
DEFAULT_OPSET_VERSION: int = 18

# Shared training defaults
DEFAULT_MLM_PROBABILITY: float = 0.15
DEFAULT_SAVE_TOTAL_LIMIT: int = 3
DEFAULT_WARMUP_RATIO: float = 0.1
DEFAULT_WEIGHT_DECAY: float = 0.01
# Hard cap to prevent over-subscription on very large machines
_MAX_WORKERS_CAP: int = 32
# Label value for masked/ignored positions in loss computation (PyTorch convention)
IGNORE_LABEL_INDEX: int = -100

# ---------------------------------------------------------------------------
# Synthetic error generator constants (used for denoising in MLM training)
# ---------------------------------------------------------------------------
LABEL_CORRECT: int = 0
LABEL_ERROR: int = 1
DEFAULT_CORRUPTION_RATIO: float = 0.15

# Maximum syllables per sentence fed to the generator.
# Derived from DEFAULT_GENERATOR_MAX_LENGTH assuming ~3 BPE tokens per Myanmar
# syllable. Sentences longer than this risk having injected errors truncated.
DEFAULT_GENERATOR_MAX_LENGTH: int = 256
DEFAULT_GENERATOR_MAX_SYLLABLES: int = (DEFAULT_GENERATOR_MAX_LENGTH - 16) // 3  # = 80

# Hardcoded fallback — used when corruption_weights.yaml is missing.
_FALLBACK_CORRUPTION_WEIGHTS: dict[str, float] = {
    "homophone_swap": 0.25,
    "word_swap": 0.20,
    "particle_confusion": 0.15,
    "medial_confusion": 0.08,
    "char_deletion": 0.08,
    "aspirated_confusion": 0.07,
    "typo_pattern": 0.07,
    "similar_char_swap": 0.05,
    "char_insertion": 0.05,
}


def _load_corruption_weights() -> dict[str, float]:
    """Load corruption weights from YAML, falling back to hardcoded defaults."""
    from pathlib import Path

    yaml_path = Path(__file__).resolve().parent.parent / "rules" / "corruption_weights.yaml"
    try:
        import yaml

        with open(yaml_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        weights = data.get("weights", {})
        if weights and isinstance(weights, dict):
            return dict(weights)
    except Exception:
        logger.warning("Failed to load corruption weights from %s, using defaults", yaml_path)
    return dict(_FALLBACK_CORRUPTION_WEIGHTS)


DEFAULT_CORRUPTION_WEIGHTS: dict[str, float] = _load_corruption_weights()
# Save-steps scaling thresholds
SAVE_STEPS_DEFAULT: int = 500
SAVE_STEPS_LARGE_RUN_THRESHOLD: int = 10_000
SAVE_STEPS_NUM_SAVES: int = 20
LOGGING_STEPS_MIN: int = 50
LOGGING_STEPS_RATIO: int = 10

# Embedding surgery defaults (used by MLM pipeline and trainer)
DEFAULT_EMBEDDING_WARMUP_STEPS: int = 25_000
DEFAULT_EMBEDDING_LR: float = 1e-3


def compute_save_steps(
    max_steps: int | None,
    default: int = SAVE_STEPS_DEFAULT,
) -> tuple[int, int]:
    """Compute save_steps and logging_steps for a training run.

    Scales checkpointing for large runs to avoid I/O storms while
    keeping a reasonable ~20 checkpoints across the run.

    Returns:
        (save_steps, logging_steps) tuple.
    """
    if max_steps is not None and max_steps > SAVE_STEPS_LARGE_RUN_THRESHOLD:
        save_steps = max(default, max_steps // SAVE_STEPS_NUM_SAVES)
    else:
        save_steps = default
    logging_steps = max(LOGGING_STEPS_MIN, save_steps // LOGGING_STEPS_RATIO)
    return save_steps, logging_steps


def get_dataloader_workers(cuda_available: bool) -> int:
    """Compute DataLoader num_workers from detected hardware.

    Auto-scales based on CPU count and GPU count (via WORLD_SIZE under DDP).
    Each DDP process gets ``(cpu_count // world_size) - 1`` workers,
    reserving 1 CPU for the training loop itself.

    Examples (all automatic, no manual tuning):
        ml.g5.12xlarge  (48 CPUs, 4 GPUs DDP) → 11 workers/process
        ml.g5.2xlarge   ( 8 CPUs, 1 GPU)      →  7 workers
        CPU-only                               →  0 workers
    """
    import os

    if not cuda_available:
        return 0

    cpu_count = os.cpu_count() or 1
    # Under DDP (torchrun sets WORLD_SIZE), divide CPUs among processes
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    cpus_per_process = cpu_count // world_size
    # Reserve 1 CPU for the training loop itself
    available = max(1, cpus_per_process - 1)
    return min(_MAX_WORKERS_CAP, available)
