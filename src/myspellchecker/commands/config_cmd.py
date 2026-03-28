"""Handle the 'config' command."""

from __future__ import annotations

import sys
from pathlib import Path

from myspellchecker.core.exceptions import ConfigurationError


def _cmd_config(args) -> None:
    """Handle the 'config' command."""
    from myspellchecker.core.config.loader import (
        CONFIG_FILE_NAMES,
        USER_CONFIG_DIR,
        find_config_file,
        init_config_file,
        is_yaml_available,
    )

    config_cmd = getattr(args, "config_command", None)

    if config_cmd == "init":
        try:
            path = init_config_file(
                path=getattr(args, "path", None),
                force=getattr(args, "force", False),
            )
            print(f"Created configuration file: {path}")
            print("\nEdit this file to customize spell checker settings.")
            print("Use 'myspellchecker config show' to view search paths.")
        except ConfigurationError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    elif config_cmd == "show":
        print("Configuration File Search Paths:")
        print("=" * 50)

        yaml_available = is_yaml_available()
        if not yaml_available:
            print("\n[!] PyYAML not installed. Config file support disabled.")
            print("    Install with: pip install pyyaml")
            print()

        print("\n1. Current directory:")
        cwd = Path.cwd()
        for name in CONFIG_FILE_NAMES:
            path = cwd / name
            status = "[FOUND]" if path.exists() else "[not found]"
            print(f"   {path} {status}")

        print("\n2. User config directory:")
        for name in CONFIG_FILE_NAMES:
            path = USER_CONFIG_DIR / name
            status = "[FOUND]" if path.exists() else "[not found]"
            print(f"   {path} {status}")

        print("\nActive Configuration:")
        print("-" * 50)
        try:
            config_path = find_config_file()
            if config_path:
                print(f"Using: {config_path}")
                if yaml_available:
                    from myspellchecker.core.config.loader import load_config_from_file

                    data = load_config_from_file(config_path)
                    if data.get("preset"):
                        print(f"  Preset: {data['preset']}")
                    if data.get("database"):
                        print(f"  Database: {data['database']}")
            else:
                print("No configuration file found.")
                print("\nRun 'myspellchecker config init' to create one.")
        except ConfigurationError as e:
            print(f"Error reading config: {e}")

    else:
        print("Usage: myspellchecker config <command>")
        print("\nAvailable commands:")
        print("  init   Create a new configuration file")
        print("  show   Show configuration search paths")
        print("\nRun 'myspellchecker config --help' for more info.")
