"""Handle the 'build' command."""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any

from myspellchecker.core.exceptions import MyanmarSpellcheckError


def validate_build_inputs(
    input_files: list[str],
    database_path: str,
) -> dict[str, Any]:
    """Validate build inputs before running the pipeline."""
    import glob
    import os

    results: dict[str, Any] = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "stats": {
            "total_files": 0,
            "total_size_bytes": 0,
            "file_types": {},
        },
    }

    if not input_files:
        results["warnings"].append("No input files specified. Use --sample to generate test data.")

    expanded_files = []
    for path_str in input_files or []:
        if os.path.isdir(path_str):
            for ext in ["*.txt", "*.json", "*.jsonl"]:
                expanded_files.extend(glob.glob(os.path.join(path_str, ext)))
        elif "*" in path_str or "?" in path_str:
            expanded_files.extend(glob.glob(path_str))
        else:
            expanded_files.append(path_str)

    for file_path in expanded_files:
        if not os.path.exists(file_path):
            results["errors"].append(f"Input file not found: {file_path}")
            results["valid"] = False
            continue

        if not os.path.isfile(file_path):
            results["errors"].append(f"Not a file: {file_path}")
            results["valid"] = False
            continue

        if not os.access(file_path, os.R_OK):
            results["errors"].append(f"Cannot read file: {file_path}")
            results["valid"] = False
            continue

        file_size = os.path.getsize(file_path)
        results["stats"]["total_files"] += 1
        results["stats"]["total_size_bytes"] += file_size

        ext = Path(file_path).suffix.lower()
        results["stats"]["file_types"][ext] = results["stats"]["file_types"].get(ext, 0) + 1

        if file_size == 0:
            results["warnings"].append(f"Empty file: {file_path}")

    output_dir = os.path.dirname(database_path) or "."
    if not os.path.exists(output_dir):
        results["errors"].append(f"Output directory does not exist: {output_dir}")
        results["valid"] = False
    elif not os.access(output_dir, os.W_OK):
        results["errors"].append(f"Cannot write to output directory: {output_dir}")
        results["valid"] = False

    if os.path.exists(database_path):
        results["warnings"].append(
            f"Output file already exists and will be overwritten: {database_path}"
        )

    return results


def _cmd_build(args) -> None:
    """Handle the 'build' command."""
    import glob as glob_module
    import os

    import myspellchecker.cli as _cli

    # Lazy import — data_pipeline requires pyarrow/duckdb (optional [build] deps)
    from myspellchecker.data_pipeline import run_pipeline as _run_pipeline

    expanded_inputs = []
    if args.input:
        for path_str in args.input:
            if os.path.isdir(path_str):
                for ext in ["*.txt", "*.json", "*.jsonl"]:
                    expanded_inputs.extend(glob_module.glob(os.path.join(path_str, ext)))
            elif "*" in path_str or "?" in path_str:
                expanded_inputs.extend(glob_module.glob(path_str))
            else:
                expanded_inputs.append(path_str)

    # Handle --validate flag (pre-flight check)
    if args.validate:
        console = _cli.get_console()
        console.print("[info]Validating build inputs...[/]")
        console.print()

        validation = _cli.validate_build_inputs(
            input_files=args.input or [],
            database_path=args.output,
        )

        stats = validation["stats"]
        statistics = {"Total files": str(stats["total_files"])}
        if stats["total_size_bytes"] > 0:
            size_mb = stats["total_size_bytes"] / (1024 * 1024)
            statistics["Total size"] = f"{size_mb:.2f} MB"
        if stats["file_types"]:
            file_types_str = ", ".join(
                f"{ext}: {count}" for ext, count in stats["file_types"].items()
            )
            statistics["File types"] = file_types_str

        from myspellchecker.utils.console import create_validation_panel

        panel = create_validation_panel(
            errors=validation["errors"],
            warnings=validation["warnings"],
            statistics=statistics,
            passed=validation["valid"],
        )
        console.print(panel)
        console.print()

        if validation["valid"]:
            console.print("[success]\u2713 Validation passed.[/] Ready to build.")
            sys.exit(0)
        else:
            console.print("[error]\u2717 Validation failed.[/] Please fix errors before building.")
            sys.exit(1)

    # Show build header panel (no colors)
    from myspellchecker.utils.console import create_build_header_panel, get_console

    console = get_console(force_plain=True)
    console.print(
        create_build_header_panel(
            input_files=expanded_inputs,
            database_path=args.output,
            sample=args.sample,
        )
    )
    console.print()

    # Create POS tagger configuration
    pos_tagger_config = None
    if hasattr(args, "pos_tagger") and args.pos_tagger:
        from myspellchecker.core.config import POSTaggerConfig

        pos_tagger_config = POSTaggerConfig(
            tagger_type=args.pos_tagger,
            model_name=getattr(args, "pos_model", None),
            device=getattr(args, "pos_device", -1),
        )

    # Parse curated input file if provided
    curated_words = None
    curated_input_path = getattr(args, "curated_input", None)
    use_curated_hf = getattr(args, "curated_lexicon_hf", False)

    if curated_input_path and use_curated_hf:
        print(
            "Error: --curated-input and --curated-lexicon-hf are mutually exclusive.",
            file=sys.stderr,
        )
        sys.exit(2)

    if use_curated_hf:
        from myspellchecker.tokenizers.resource_loader import get_curated_lexicon_path

        try:
            console.print("[info]Fetching curated lexicon from HuggingFace...[/]")
            curated_input_file = get_curated_lexicon_path()
            curated_input_path = str(curated_input_file)
            console.print(f"[info]Curated lexicon ready: {curated_input_path}[/]")
        except Exception as e:
            print(f"Error downloading curated lexicon from HuggingFace: {e}", file=sys.stderr)
            sys.exit(2)

    if curated_input_path:
        curated_input_file = Path(curated_input_path)
        if not curated_input_file.exists():
            print(f"Error: Curated input file not found: {curated_input_path}", file=sys.stderr)
            sys.exit(2)
        try:
            curated_words = {}
            with open(curated_input_file, encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if not header:
                    print(
                        f"Error: Curated input file is empty: {curated_input_path}",
                        file=sys.stderr,
                    )
                    sys.exit(2)
                word_col_idx = 0
                if "word" in header:
                    word_col_idx = header.index("word")
                pos_col_idx = None
                if "pos_tag" in header:
                    pos_col_idx = header.index("pos_tag")
                for row in reader:
                    if row and len(row) > word_col_idx:
                        word = row[word_col_idx].strip()
                        if word:
                            pos_tag = ""
                            if pos_col_idx is not None and len(row) > pos_col_idx:
                                pos_tag = row[pos_col_idx].strip()
                            curated_words[word] = pos_tag
            pos_count = sum(1 for p in curated_words.values() if p)
            console.print(
                f"[info]Loaded {len(curated_words):,} curated words "
                f"({pos_count:,} with POS tags) from {curated_input_path}[/]"
            )
        except Exception as e:
            print(f"Error reading curated input file: {e}", file=sys.stderr)
            sys.exit(2)

    try:
        _run_pipeline(
            input_files=expanded_inputs,
            database_path=args.output,
            work_dir=args.work_dir or "temp_build",
            keep_intermediate=args.keep_intermediate,
            sample=args.sample,
            text_col=args.col,
            json_key=args.json_key,
            pos_tagger_config=pos_tagger_config,
            incremental=args.incremental,
            word_engine=args.word_engine,
            seg_model=getattr(args, "seg_model", None),
            seg_device=getattr(args, "seg_device", -1),
            min_frequency=getattr(args, "min_frequency", None),
            worker_timeout=getattr(args, "worker_timeout", 300),
            num_workers=getattr(args, "num_workers", None),
            batch_size=getattr(args, "batch_size", None),
            curated_words=curated_words,
            remove_segmentation_markers=not getattr(args, "no_desegment", False),
            deduplicate_lines=not getattr(args, "no_dedup", False),
            enrich=not getattr(args, "no_enrich", False),
        )
    except MyanmarSpellcheckError as e:
        console.print()
        console.print(f"Error: {e}")
        sys.exit(1)
    except (RuntimeError, OSError, MemoryError) as e:
        console.print()
        console.print(f"Unexpected error: {e}")
        sys.exit(1)
