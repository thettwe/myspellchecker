"""Handle the 'check' command."""

from __future__ import annotations

import sqlite3
import sys
from typing import Any

from myspellchecker.core.constants import ValidationLevel
from myspellchecker.core.exceptions import (
    ConfigurationError,
    DataLoadingError,
    MyanmarSpellcheckError,
)
from myspellchecker.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _cmd_check(args) -> None:
    """Handle the 'check' command."""
    # Lazy imports from myspellchecker.cli so that test patches on
    # myspellchecker.cli.{open_input_file,get_checker,...} are picked up.
    import myspellchecker.cli as _cli
    from myspellchecker import SpellChecker
    from myspellchecker.cli_formatters import (
        CSVFormatter,
        JSONFormatter,
        RichFormatter,
        TextFormatter,
    )
    from myspellchecker.cli_parser import PRESET_TO_PROFILE

    input_file = _cli.open_input_file(args.input)
    output_file = _cli.open_output_file(args.output)

    if input_file == sys.stdin and input_file.isatty():
        print("mySpellChecker CLI", file=sys.stderr)
        print("------------------", file=sys.stderr)
        print("Reading from stdin... (Press Ctrl+D to finish)", file=sys.stderr)

    # Load configuration from file if available
    config_file_path = getattr(args, "config", None)
    config = None
    found_config = None

    try:
        from myspellchecker.core.config.loader import (
            ConfigLoader,
            find_config_file,
            is_yaml_available,
        )

        if is_yaml_available():
            found_config = find_config_file(config_file_path)

            if found_config:
                logger.debug(f"Loading config from: {found_config}")

                cli_profile = None
                if args.preset:
                    cli_profile = PRESET_TO_PROFILE.get(args.preset)

                cli_overrides: dict[str, Any] = {}

                if args.db:
                    if "provider_config" not in cli_overrides:
                        cli_overrides["provider_config"] = {}
                    cli_overrides["provider_config"]["database_path"] = args.db

                if args.no_phonetic:
                    cli_overrides["use_phonetic"] = False
                if args.no_context:
                    cli_overrides["use_context_checker"] = False
                if args.no_ner:
                    cli_overrides["use_ner"] = False

                loader = ConfigLoader()
                config = loader.load(
                    profile=cli_profile,  # type: ignore[arg-type]
                    config_file=found_config,
                    use_env=True,
                    overrides=cli_overrides,
                )

                db_path = config.provider_config.database_path if config.provider_config else None
                logger.debug(
                    f"Loaded config: preset={getattr(config, 'preset', None)}, db={db_path}"
                )
            else:
                logger.debug("No config file found, using CLI parameters")

        elif config_file_path:
            print(
                f"Error: Config file '{config_file_path}' specified but "
                "PyYAML is not installed.\n"
                "Install with: pip install pyyaml",
                file=sys.stderr,
            )
            sys.exit(1)

    except ConfigurationError as e:
        logger.warning("Config file error: %s", e)
        config = None
    except (OSError, ValueError) as e:
        if found_config:
            logger.warning("Failed to load config from %s: %s", found_config, e)
        else:
            logger.warning("Failed to load config: %s", e)
        logger.info("Using default configuration")
        config = None

    # Initialize spell checker
    try:
        if config:
            checker = SpellChecker(config=config)
        else:
            preset = getattr(args, "preset", None)
            database_path = args.db
            checker = _cli.get_checker(
                database_path,
                args.no_phonetic,
                args.no_context,
                preset=preset,
                no_ner=getattr(args, "no_ner", False),
                ner_model=getattr(args, "ner_model", None),
                ner_device=getattr(args, "ner_device", -1),
            )
    except DataLoadingError as e:
        print(f"Error: Database error: {e}", file=sys.stderr)
        print(
            "  Run 'myspellchecker build --sample' to create a test dictionary.",
            file=sys.stderr,
        )
        print("  Or provide a custom DB with '--db <path>'.", file=sys.stderr)
        sys.exit(1)
    except (sqlite3.DatabaseError, OSError, RuntimeError) as e:
        print(f"Unexpected database error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)

    with checker:
        force_color = getattr(args, "force_color", False)
        no_color = getattr(args, "no_color", False)

        formatters = {
            "json": JSONFormatter(),
            "text": TextFormatter(),
            "csv": CSVFormatter(),
            "rich": RichFormatter(force_color=force_color, force_plain=no_color),
        }
        formatter = formatters[args.format]

        file_name = args.input if args.input else "stdin"
        validation_level = ValidationLevel(args.level)

        try:
            formatter.begin(output_file)
            stats = _cli.process_stream(
                input_file, file_name, checker, formatter, output_file, validation_level
            )
            formatter.end(output_file, stats)
        except KeyboardInterrupt:
            print("\nProcess interrupted.", file=sys.stderr)
            sys.exit(130)
        except MyanmarSpellcheckError as e:
            print(f"Error processing input: {e}", file=sys.stderr)
            sys.exit(1)
        except (RuntimeError, OSError) as e:
            print(f"Unexpected error: {e}", file=sys.stderr)
            sys.exit(1)
        finally:
            if input_file != sys.stdin:
                input_file.close()
            if output_file != sys.stdout:
                output_file.close()
