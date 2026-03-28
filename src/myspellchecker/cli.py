#!/usr/bin/env python3
"""
mySpellChecker CLI Tool.

A command-line interface for the myspellchecker library, optimized for
automation pipelines, server-side validation, and batch processing.
"""

from __future__ import annotations

import logging
import os  # noqa: F401 - re-export (tests patch myspellchecker.cli.os)
import sqlite3
import sys
from typing import Any, TextIO

from myspellchecker import SpellChecker
from myspellchecker.cli_completions import generate_completion_script  # noqa: F401 - re-export
from myspellchecker.cli_formatters import (
    AVAILABLE_PRESETS,  # noqa: F401 - re-export for backward compatibility
    CSVFormatter,  # noqa: F401 - re-export for backward compatibility
    Formatter,  # noqa: F401 - re-export for backward compatibility
    JSONFormatter,  # noqa: F401 - re-export for backward compatibility
    RichFormatter,  # noqa: F401 - re-export for backward compatibility
    TextFormatter,  # noqa: F401 - re-export for backward compatibility
)
from myspellchecker.cli_parser import build_parser, parse_and_route
from myspellchecker.cli_utils import (
    confidence_type,  # noqa: F401 - re-export for backward compatibility
    open_input_file,  # noqa: F401 - re-export for backward compatibility
    open_output_file,  # noqa: F401 - re-export for backward compatibility
)
from myspellchecker.commands import (
    _cmd_build,
    _cmd_check,
    _cmd_completion,
    _cmd_config,
    _cmd_infer_pos,
    _cmd_segment,
    _cmd_train_model,
    validate_build_inputs,  # noqa: F401 - re-export for backward compatibility
)
from myspellchecker.core.config import SpellCheckerConfig
from myspellchecker.core.constants import ValidationLevel
from myspellchecker.core.exceptions import (
    ConfigurationError,
    DataLoadingError,
)

# Lazy: run_pipeline imported on demand in _cmd_build (requires pyarrow/duckdb)
from myspellchecker.providers import DictionaryProvider, MemoryProvider, SQLiteProvider
from myspellchecker.training import TrainingPipeline  # noqa: F401 - re-export (tests patch)
from myspellchecker.utils.console import get_console  # noqa: F401 - re-export (tests patch)
from myspellchecker.utils.logging_utils import configure_logging, get_logger

# Module logger
logger = get_logger(__name__)


def get_config_from_preset(preset: str) -> SpellCheckerConfig:
    """Get a SpellCheckerConfig from a preset name.

    Supports both CLI presets and config file profiles for compatibility.
    """
    from myspellchecker.core.builder import ConfigPresets

    preset_map = {
        "default": ConfigPresets.default,
        "fast": ConfigPresets.fast,
        "accurate": ConfigPresets.accurate,
        "minimal": ConfigPresets.minimal,
        "strict": ConfigPresets.strict,
        "production": ConfigPresets.default,
        "development": ConfigPresets.fast,
        "testing": ConfigPresets.minimal,
    }
    if preset not in preset_map:
        available = sorted(set(preset_map.keys()))
        raise ValueError(
            f"Unknown preset: {preset}. Available: {', '.join(available)}. "
            "Use 'myspellchecker check --help' for preset descriptions."
        )
    return preset_map[preset]()


def get_checker(
    database_path: str | None = None,
    no_phonetic: bool = False,
    no_context: bool = False,
    preset: str | None = None,
    joint_config: Any | None = None,
    no_ner: bool = False,
    ner_model: str | None = None,
    ner_device: int = -1,
) -> SpellChecker:
    """Initialize the spell checker."""
    from myspellchecker.core.config import JointConfig  # noqa: F811

    provider: DictionaryProvider
    if database_path:
        try:
            provider = SQLiteProvider(database_path=str(database_path))
        except DataLoadingError as e:
            print(f"Error loading database: {e}", file=sys.stderr)
            sys.exit(1)
        except (sqlite3.DatabaseError, OSError, RuntimeError) as e:
            print(f"Unexpected error loading database: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        try:
            provider = SQLiteProvider()
        except DataLoadingError:
            print("Warning: Default database not found.", file=sys.stderr)
            print(
                "  Run 'myspellchecker build --sample' to create a test dictionary.",
                file=sys.stderr,
            )
            print("  Or provide a custom DB with '--db <path>'.", file=sys.stderr)
            print(
                "  Using empty provider (dictionary lookups will return no results; "
                "only rule-based validation will apply).",
                file=sys.stderr,
            )
            provider = MemoryProvider()

    try:
        ner_config = None
        if ner_model:
            from myspellchecker.text.ner_model import NERConfig

            ner_config = NERConfig(
                model_type="transformer",
                model_name=ner_model,
                device=ner_device,
            )

        if preset:
            config = get_config_from_preset(preset)
            config.provider = provider
            if no_phonetic:
                config.use_phonetic = False
            if no_context:
                config.use_context_checker = False
            if no_ner:
                config.use_ner = False
            if ner_config:
                config.ner = ner_config
            if joint_config:
                config.joint = joint_config
        else:
            config = SpellCheckerConfig(
                provider=provider,
                use_phonetic=not no_phonetic,
                use_context_checker=not no_context,
                use_ner=not no_ner,
                ner=ner_config,
                joint=joint_config or JointConfig(),
            )
        return SpellChecker(config=config)
    except ConfigurationError as e:
        print(f"Configuration Error: {e}", file=sys.stderr)
        sys.exit(1)


def process_stream(
    input_stream: TextIO,
    file_name: str,
    checker: SpellChecker,
    formatter: Formatter,
    output_stream: TextIO,
    level: ValidationLevel,
    max_memory_mb: int = 100,
) -> dict[str, int]:
    """Process a single input stream using StreamingChecker."""
    from myspellchecker.core.streaming import StreamingChecker, StreamingConfig

    streaming_config = StreamingConfig(max_memory_mb=max_memory_mb)
    streaming = StreamingChecker(checker, config=streaming_config)

    stats = {"total_lines": 0, "total_errors": 0, "lines_with_errors": 0}

    for chunk_result in streaming.check_stream(input_stream, level=level):
        result = chunk_result.response
        line_no = chunk_result.line_number

        stats["total_lines"] += 1
        if result.has_errors:
            stats["total_errors"] += len(result.errors)
            stats["lines_with_errors"] += 1

        formatter.write_result(output_stream, result, line_no, file_name)

    return stats


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

# Command dispatch table
_COMMAND_HANDLERS = {
    "completion": _cmd_completion,
    "config": _cmd_config,
    "build": _cmd_build,
    "train-model": _cmd_train_model,
    "segment": _cmd_segment,
    "infer-pos": _cmd_infer_pos,
    "check": _cmd_check,
}


def main() -> None:
    """Command-line interface entry point for myspellchecker."""
    parser = build_parser()
    args, command = parse_and_route(parser)

    # Setup logging
    if getattr(args, "verbose", False):
        log_level = logging.DEBUG
        debug_mode = True
    elif command == "build":
        log_level = logging.INFO
        debug_mode = False
    else:
        log_level = logging.WARNING
        debug_mode = False

    build_format = None
    if command == "build":
        if getattr(args, "verbose", False):
            build_format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        else:
            build_format = "%(message)s"

    configure_logging(
        level=log_level,
        format_string=build_format if command == "build" else None,
        debug_mode=debug_mode,
    )

    handler = _COMMAND_HANDLERS.get(command)
    if handler:
        handler(args)
    else:
        print(f"Unknown command: {command}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
