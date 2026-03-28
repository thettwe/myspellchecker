"""Handle the 'completion' command."""

from __future__ import annotations

import sys


def _cmd_completion(args) -> None:
    """Handle the 'completion' command."""
    from myspellchecker.cli_completions import generate_completion_script

    try:
        script = generate_completion_script(args.shell)
        print(script)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
