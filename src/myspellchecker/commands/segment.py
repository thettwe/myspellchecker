"""Handle the 'segment' command."""

from __future__ import annotations

import sys


def _cmd_segment(args) -> None:
    """Handle the 'segment' command."""
    import json as json_module

    # Lazy import to avoid circular import and to allow test patches
    # on myspellchecker.cli.{open_input_file,get_checker,...}.
    import myspellchecker.cli as _cli

    input_file = _cli.open_input_file(args.input)
    output_file = _cli.open_output_file(args.output)

    if input_file == sys.stdin and input_file.isatty():
        print("mySpellChecker Segmenter", file=sys.stderr)
        print("------------------------", file=sys.stderr)
        print("Reading from stdin... (Press Ctrl+D to finish)", file=sys.stderr)

    try:
        from myspellchecker.core.config import JointConfig

        joint_config = JointConfig(enabled=args.tag) if args.tag else None

        with _cli.get_checker(args.db, preset="fast", joint_config=joint_config) as checker:
            for line in input_file:
                line = line.strip()
                if not line:
                    continue

                if args.tag:
                    words, tags = checker.segment_and_tag(line)
                else:
                    words = checker.segmenter.segment_words(line)
                    tags = None

                if args.format == "json":
                    result = {"words": words}
                    if tags:
                        result["tags"] = tags
                    output_file.write(json_module.dumps(result, ensure_ascii=False) + "\n")
                elif args.format == "tsv":
                    if tags:
                        for word, tag in zip(words, tags, strict=False):
                            output_file.write(f"{word}\t{tag}\n")
                        output_file.write("\n")
                    else:
                        output_file.write("\t".join(words) + "\n")
                else:  # text format
                    if tags:
                        tagged = [f"{w}/{t}" for w, t in zip(words, tags, strict=False)]
                        output_file.write(" ".join(tagged) + "\n")
                    else:
                        output_file.write(" ".join(words) + "\n")

    except KeyboardInterrupt:
        print("\nProcess interrupted.", file=sys.stderr)
        sys.exit(130)
    except (RuntimeError, OSError, MemoryError) as e:
        print(f"Segmentation failed: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        if input_file != sys.stdin:
            input_file.close()
        if output_file != sys.stdout:
            output_file.close()
