"""
Argument parser setup for the mySpellChecker CLI.

Extracted from cli.py to reduce the size of the main() function.
Contains build_parser() for creating the ArgumentParser and
parse_and_route() for handling default subcommand insertion.
"""

import argparse
import sys

from myspellchecker.cli_formatters import AVAILABLE_PRESETS
from myspellchecker.cli_utils import confidence_type
from myspellchecker.core.constants import DEFAULT_DB_NAME
from myspellchecker.data_pipeline.config import PipelineConfig
from myspellchecker.training.constants import DEFAULT_MIN_FREQUENCY, DEFAULT_VOCAB_SIZE
from myspellchecker.utils.console import is_terminal

AVAILABLE_COMMANDS = [
    "check",
    "build",
    "train-model",
    "segment",
    "infer-pos",
    "completion",
    "config",
]

PRESET_TO_PROFILE = {
    "default": "production",
    "fast": "fast",
    "accurate": "accurate",
    "minimal": "testing",
    "strict": "production",
}


def build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser for the mySpellChecker CLI."""
    # Enhanced help text with examples
    main_epilog = """
Examples:
  # Check a file for spelling errors
  myspellchecker check input.txt -f json -o errors.json

  # Check with fast preset for real-time use
  myspellchecker check input.txt --preset fast

  # Build dictionary from corpus files
  myspellchecker build -i corpus/*.txt -o my-dictionary.db

  # Validate build inputs before running
  myspellchecker build -i corpus/ --validate

  # Generate shell completion script
  myspellchecker completion --shell bash > ~/.bash_completion.d/myspellchecker

For more information, see: https://github.com/thettwe/my-spellchecker
"""

    parser = argparse.ArgumentParser(
        description="mySpellChecker CLI - Myanmar language spell checker",
        epilog=main_epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # 'build' command with enhanced help
    build_epilog = """
Examples:
  # Build from text files
  myspellchecker build -i corpus.txt -o dictionary.db

  # Build from multiple sources with glob patterns
  myspellchecker build -i "data/*.txt" "extra/*.json"

  # Build from directory (auto-detects txt, json, jsonl)
  myspellchecker build -i ./corpus/ -o dictionary.db

  # Validate inputs before building
  myspellchecker build -i corpus/ --validate

  # Incremental update on existing database
  myspellchecker build -i new_data.txt -o dictionary.db --incremental

  # Generate sample database for testing
  myspellchecker build --sample

  # Build with curated lexicon from local file (words marked as is_curated=1)
  myspellchecker build -i corpus.txt --curated-input data/curated_lexicon.csv -o dictionary.db

  # Build with official curated lexicon downloaded from HuggingFace (cached after first use)
  myspellchecker build -i corpus.txt --curated-lexicon-hf -o dictionary.db

  # Combine corpus with curated lexicon and transformer POS tagger
  myspellchecker build -i corpus.txt --curated-input data/curated_lexicon.csv \\
    --pos-tagger transformer --pos-device 0 -o dictionary.db
"""
    parser_build = subparsers.add_parser(
        "build",
        help="Build dictionary database from corpus",
        epilog=build_epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser_build.add_argument(
        "--input", "-i", nargs="+", help="Input corpus file(s) (UTF-8 encoded text, CSV, TSV, JSON)"
    )
    parser_build.add_argument(
        "--output",
        "-o",
        default=DEFAULT_DB_NAME,
        help=f"Output database path (default: {DEFAULT_DB_NAME})",
    )
    parser_build.add_argument(
        "--work-dir", help="Directory for intermediate files (default: temp_build)"
    )
    parser_build.add_argument(
        "--keep-intermediate", action="store_true", help="Keep intermediate files after build"
    )
    parser_build.add_argument(
        "--sample", action="store_true", help="Generate sample corpus for testing"
    )
    parser_build.add_argument(
        "--col", default="text", help="Column name/index for CSV/TSV files (default: text)"
    )
    parser_build.add_argument(
        "--json-key", default="text", help="Key name for JSON objects (default: text)"
    )
    parser_build.add_argument(
        "--pos-tagger",
        choices=["rule_based", "transformer", "viterbi"],
        help=(
            "POS tagger type for dynamic tagging during build (default: rule_based). "
            "'transformer' requires 'pip install myspellchecker[transformers]'"
        ),
    )
    parser_build.add_argument(
        "--pos-model",
        help=(
            "HuggingFace model ID or local path for transformer tagger "
            "(default: chuuhtetnaing/myanmar-pos-model)"
        ),
    )
    parser_build.add_argument(
        "--pos-device",
        type=int,
        default=-1,
        help="Device for transformer POS tagger: -1=CPU, 0+=GPU index (default: -1)",
    )
    parser_build.add_argument(
        "--incremental", action="store_true", help="Perform incremental update on existing database"
    )
    parser_build.add_argument(
        "--curated-input",
        type=str,
        help=(
            "Path to curated lexicon CSV file. Words from this file will be marked "
            "as 'is_curated=1' in the database. CSV must have a 'word' column header."
        ),
    )
    parser_build.add_argument(
        "--curated-lexicon-hf",
        action="store_true",
        help=(
            "Download and use the official curated lexicon from HuggingFace "
            "(thettwe/myspellchecker-resources). Cached after first download. "
            "Mutually exclusive with --curated-input."
        ),
    )
    parser_build.add_argument(
        "--word-engine",
        default=PipelineConfig.word_engine,
        choices=["crf", "myword", "transformer"],
        help=(
            f"Word segmentation engine to use (default: {PipelineConfig.word_engine}). "
            "'myword' is recommended for better particle handling. "
            "'transformer' uses a HuggingFace model (requires transformers package)."
        ),
    )
    parser_build.add_argument(
        "--seg-model",
        default=None,
        help=(
            "Custom HuggingFace model name/path for transformer word segmentation. "
            "Only used when --word-engine=transformer. "
            "Default: chuuhtetnaing/myanmar-text-segmentation-model"
        ),
    )
    parser_build.add_argument(
        "--seg-device",
        type=int,
        default=-1,
        help=(
            "Device for transformer word segmentation inference. "
            "-1 for CPU (default), 0+ for GPU index. "
            "Only used when --word-engine=transformer."
        ),
    )
    parser_build.add_argument(
        "--validate",
        action="store_true",
        help="Validate inputs only without building (pre-flight check)",
    )
    parser_build.add_argument(
        "--min-frequency",
        type=int,
        default=None,
        help="Minimum word frequency threshold (default: from config, typically 50)",
    )
    parser_build.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: auto-detect based on CPU cores)",
    )
    parser_build.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for processing (default: 10000). Reduce if running out of memory",
    )
    parser_build.add_argument(
        "--worker-timeout",
        type=int,
        default=PipelineConfig.worker_timeout,
        help=(
            "Worker timeout in seconds for parallel processing operations "
            f"(default: {PipelineConfig.worker_timeout})"
        ),
    )
    parser_build.add_argument(
        "--no-dedup",
        action="store_true",
        help="Disable line-level deduplication during ingestion",
    )
    parser_build.add_argument(
        "--no-desegment",
        action="store_true",
        help="Keep word segmentation markers (spaces/underscores between Myanmar chars)",
    )
    parser_build.add_argument(
        "--no-enrich",
        action="store_true",
        help="Skip Step 5 enrichment (confusable pairs, compounds, collocations, register tags)",
    )
    parser_build.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging with detailed timing breakdowns",
    )

    # 'train-model' command with enhanced help
    train_epilog = """
Examples:
  # Train with default settings (RoBERTa architecture)
  myspellchecker train-model -i corpus.txt -o ./models/

  # Train BERT model with more epochs
  myspellchecker train-model -i corpus.txt -o ./models/ --architecture bert --epochs 10

  # Train with custom hyperparameters
  myspellchecker train-model -i corpus.txt -o ./models/ \\
    --learning-rate 3e-5 --warmup-ratio 0.1 --weight-decay 0.01

  # Train larger model
  myspellchecker train-model -i corpus.txt -o ./models/ \\
    --hidden-size 512 --layers 6 --heads 8

  # Resume training from checkpoint
  myspellchecker train-model -i corpus.txt -o ./models/ \\
    --resume ./models/checkpoints/checkpoint-500

  # Keep checkpoints and disable metrics
  myspellchecker train-model -i corpus.txt -o ./models/ \\
    --keep-checkpoints --no-metrics

  # Streaming mode for large corpora (constant memory)
  myspellchecker train-model -i large_corpus.txt -o ./models/ --streaming

  # Streaming with checkpoint resume
  myspellchecker train-model -i large_corpus.txt -o ./models/ \\
    --streaming --checkpoint-dir /opt/ml/checkpoints

Architectures:
  roberta   RoBERTa (default) - Dynamic masking, no NSP
  bert      BERT - Static masking, with NSP capability
"""
    parser_train = subparsers.add_parser(
        "train-model",
        help="Train a custom semantic model",
        epilog=train_epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser_train.add_argument(
        "--input", "-i", required=True, help="Input corpus file (raw text, one sentence per line)"
    )
    parser_train.add_argument(
        "--output", "-o", required=True, help="Output directory for the model"
    )
    # Model architecture
    parser_train.add_argument(
        "--architecture",
        "-a",
        choices=["roberta", "bert"],
        default="roberta",
        help="Model architecture (default: roberta)",
    )
    # Training parameters
    parser_train.add_argument(
        "--epochs", type=int, default=5, help="Number of training epochs (default: 5)"
    )
    parser_train.add_argument(
        "--batch-size", type=int, default=16, help="Training batch size (default: 16)"
    )
    parser_train.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Peak learning rate (default: 5e-5)",
    )
    parser_train.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Ratio of steps for LR warmup (default: 0.1)",
    )
    parser_train.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay for optimizer (default: 0.01)",
    )
    # Model architecture parameters
    parser_train.add_argument(
        "--hidden-size",
        type=int,
        default=256,
        help="Size of hidden layers (default: 256)",
    )
    parser_train.add_argument(
        "--layers", type=int, default=4, help="Number of transformer layers (default: 4)"
    )
    parser_train.add_argument(
        "--heads", type=int, default=4, help="Number of attention heads (default: 4)"
    )
    parser_train.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Maximum sequence length (default: 128)",
    )
    # Tokenizer parameters
    parser_train.add_argument(
        "--vocab-size",
        type=int,
        default=DEFAULT_VOCAB_SIZE,
        help=f"Tokenizer vocabulary size (default: {DEFAULT_VOCAB_SIZE})",
    )
    parser_train.add_argument(
        "--min-frequency",
        type=int,
        default=DEFAULT_MIN_FREQUENCY,
        help=f"Minimum token frequency (default: {DEFAULT_MIN_FREQUENCY})",
    )
    # Checkpoint and metrics
    parser_train.add_argument(
        "--resume",
        metavar="CHECKPOINT",
        help="Resume training from checkpoint directory",
    )
    parser_train.add_argument(
        "--keep-checkpoints",
        action="store_true",
        help="Keep intermediate PyTorch checkpoints after export",
    )
    parser_train.add_argument(
        "--no-metrics",
        action="store_true",
        help="Disable saving training metrics to JSON",
    )
    # Streaming mode
    parser_train.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming mode for large corpora (constant memory usage)",
    )
    parser_train.add_argument(
        "--checkpoint-dir",
        metavar="DIR",
        help="Persistent checkpoint directory for job resume (e.g. /opt/ml/checkpoints)",
    )
    parser_train.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Cap total training steps (overrides epochs × steps_per_epoch)",
    )
    parser_train.add_argument(
        "--fp16",
        action="store_true",
        help="Enable mixed-precision (FP16) training for faster speed and lower memory",
    )
    parser_train.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Accumulate gradients over N steps (effective batch = batch-size × N, default: 1)",
    )

    # 'check' command (default) with enhanced help
    check_epilog = """
Examples:
  # Check a file and output JSON
  myspellchecker check input.txt -f json

  # Check with text output (grep-like format)
  myspellchecker check input.txt -f text

  # Check from stdin
  echo "ျမန်မာစာ" | myspellchecker check

  # Use fast preset for real-time applications
  myspellchecker check input.txt --preset fast

  # Use accurate preset for formal documents
  myspellchecker check input.txt --preset accurate

  # Use strict preset with more error detection
  myspellchecker check input.txt --preset strict

  # Disable phonetic matching for speed
  myspellchecker check input.txt --no-phonetic

  # Use custom database
  myspellchecker check input.txt --db custom.db

Presets:
  default   Balanced accuracy and performance (recommended)
  fast      Optimized for speed, disables context/phonetic
  accurate  Maximum accuracy, enables all features
  minimal   Basic validation only, minimal resources
  strict    Conservative thresholds, catches more errors
"""
    parser_check = subparsers.add_parser(
        "check",
        help="Check text for spelling errors",
        epilog=check_epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser_check.add_argument(
        "input",
        nargs="?",
        type=str,
        default=None,
        help="Input file path (or stdin if omitted)",
    )

    parser_check.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output file path (default: stdout)",
    )

    parser_check.add_argument(
        "-f",
        "--format",
        choices=["json", "text", "csv", "rich"],
        default="rich" if is_terminal() else "json",
        help="Output format (default: rich for TTY, json for pipes)",
    )
    color_group = parser_check.add_mutually_exclusive_group()
    color_group.add_argument(
        "--color",
        action="store_true",
        dest="force_color",
        default=None,
        help="Force color output even when not a TTY",
    )
    color_group.add_argument(
        "--no-color",
        action="store_true",
        dest="no_color",
        default=False,
        help="Disable color output",
    )

    parser_check.add_argument(
        "--level",
        choices=["syllable", "word"],
        default="syllable",
        help="Validation level (default: syllable)",
    )

    parser_check.add_argument("--db", help="Custom database path")
    parser_check.add_argument(
        "--no-phonetic", action="store_true", help="Disable phonetic matching"
    )
    parser_check.add_argument("--no-context", action="store_true", help="Disable context checking")
    parser_check.add_argument(
        "--no-ner", action="store_true", help="Disable Named Entity Recognition"
    )
    parser_check.add_argument(
        "--ner-model",
        help=(
            "HuggingFace model name for transformer NER "
            "(default: chuuhtetnaing/myanmar-ner-model). "
            "Requires 'pip install myspellchecker[transformers]'"
        ),
    )
    parser_check.add_argument(
        "--ner-device",
        type=int,
        default=-1,
        help="Device for NER inference: -1=CPU, 0+=GPU index (default: -1)",
    )
    parser_check.add_argument(
        "-p",
        "--preset",
        choices=AVAILABLE_PRESETS,
        help="Configuration preset (default, fast, accurate, minimal, strict)",
    )
    parser_check.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser_check.add_argument(
        "-c",
        "--config",
        metavar="FILE",
        help="Path to configuration file (YAML or JSON format)",
    )

    # 'segment' command for segmentation and POS tagging
    segment_epilog = """
Examples:
  # Segment text into words
  echo "မြန်မာနိုင်ငံ" | myspellchecker segment

  # Segment and tag with POS tags (joint mode)
  echo "မြန်မာနိုင်ငံ" | myspellchecker segment --tag

  # Segment from file with custom database
  myspellchecker segment input.txt --db custom.db --tag

  # Output as JSON
  myspellchecker segment input.txt --format json --tag
"""
    parser_segment = subparsers.add_parser(
        "segment",
        help="Segment text into words and optionally tag with POS",
        epilog=segment_epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser_segment.add_argument(
        "input",
        nargs="?",
        type=str,
        default=None,
        help="Input file path (or stdin if omitted)",
    )
    parser_segment.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output file path (default: stdout)",
    )
    parser_segment.add_argument(
        "-f",
        "--format",
        choices=["text", "json", "tsv"],
        default="text",
        help="Output format (default: text)",
    )
    parser_segment.add_argument(
        "--tag",
        action="store_true",
        help="Include POS tags (uses joint segmentation-tagging)",
    )
    parser_segment.add_argument("--db", help="Custom database path")
    parser_segment.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    # 'completion' command for shell completions
    completion_epilog = """
Examples:
  # Generate bash completion
  myspellchecker completion --shell bash > ~/.bash_completion.d/myspellchecker
  source ~/.bash_completion.d/myspellchecker

  # Generate zsh completion
  myspellchecker completion --shell zsh > ~/.zsh/completions/_myspellchecker

  # Generate fish completion
  myspellchecker completion --shell fish > ~/.config/fish/completions/myspellchecker.fish
"""
    parser_completion = subparsers.add_parser(
        "completion",
        help="Generate shell completion script",
        epilog=completion_epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser_completion.add_argument(
        "--shell",
        choices=["bash", "zsh", "fish"],
        default="bash",
        help="Shell type (default: bash)",
    )

    # 'infer-pos' command for POS tag inference
    infer_pos_epilog = """
Examples:
  # Infer POS for all untagged words in database
  myspellchecker infer-pos --db dictionary.db

  # Infer POS only for high-frequency words
  myspellchecker infer-pos --db dictionary.db --min-frequency 10

  # Set minimum confidence threshold
  myspellchecker infer-pos --db dictionary.db --min-confidence 0.7

  # Show statistics without modifying database
  myspellchecker infer-pos --db dictionary.db --dry-run

Inference Sources:
  numeral_detection    Myanmar numerals (၀-၉) and numeral words
  prefix_pattern       Words with prefix patterns (e.g., အ prefix → Noun)
  proper_noun_suffix   Proper noun suffixes (country, city names)
  ambiguous_registry   Known ambiguous words (multi-POS)
  morphological        Suffix-based morphological analysis
"""
    parser_infer_pos = subparsers.add_parser(
        "infer-pos",
        help="Infer POS tags for untagged words using rule-based engine",
        epilog=infer_pos_epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser_infer_pos.add_argument(
        "--db",
        required=True,
        help="Database path to update with inferred POS tags",
    )
    parser_infer_pos.add_argument(
        "--min-frequency",
        type=int,
        default=0,
        help="Minimum word frequency for inference (default: 0, infer all)",
    )
    parser_infer_pos.add_argument(
        "--min-confidence",
        type=confidence_type,
        default=0.0,
        help="Minimum confidence threshold (0.0-1.0, default: 0.0)",
    )
    parser_infer_pos.add_argument(
        "--include-tagged",
        action="store_true",
        help="Also infer for words that already have pos_tag (updates inferred_pos only)",
    )
    parser_infer_pos.add_argument(
        "--dry-run",
        action="store_true",
        help="Show statistics without modifying the database",
    )
    parser_infer_pos.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output with detailed statistics",
    )

    # 'config' command for configuration file management
    config_epilog = """
Examples:
  # Initialize configuration file (default: ~/.config/myspellchecker/myspellchecker.yaml)
  myspellchecker config init

  # Initialize with custom path
  myspellchecker config init --path ./myspellchecker.yaml

  # Overwrite existing configuration file
  myspellchecker config init --force

  # Show current configuration search paths
  myspellchecker config show

Configuration File Locations (searched in order):
  1. Path specified with --config flag
  2. Current directory: myspellchecker.yaml, myspellchecker.yml, or myspellchecker.json
  3. User config directory: ~/.config/myspellchecker/myspellchecker.{yaml,yml,json}
"""
    parser_config = subparsers.add_parser(
        "config",
        help="Manage configuration files",
        epilog=config_epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    config_subparsers = parser_config.add_subparsers(dest="config_command")

    # config init
    parser_config_init = config_subparsers.add_parser(
        "init",
        help="Create a new configuration file with defaults",
    )
    parser_config_init.add_argument(
        "--path",
        metavar="FILE",
        help="Path for configuration file (default: ~/.config/myspellchecker/myspellchecker.yaml)",
    )
    parser_config_init.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing configuration file",
    )

    # config show
    config_subparsers.add_parser(
        "show",
        help="Show configuration file search paths and current config",
    )

    return parser


def parse_and_route(parser: argparse.ArgumentParser) -> tuple[argparse.Namespace, str]:
    """Parse arguments with default 'check' command insertion.

    When the first argument is not a known command or help flag,
    inserts 'check' as the default subcommand.

    Args:
        parser: The argument parser from build_parser().

    Returns:
        Tuple of (parsed args namespace, command name string).
    """
    argv = list(sys.argv)  # Work on a copy to avoid mutating global sys.argv
    if len(argv) <= 1:
        # No arguments: default to check subparser so its defaults are applied
        argv.insert(1, "check")
    elif argv[1] not in [
        *AVAILABLE_COMMANDS,
        "-h",
        "--help",
    ]:
        argv.insert(1, "check")

    args = parser.parse_args(argv[1:])
    command = args.command or "check"

    return args, command
