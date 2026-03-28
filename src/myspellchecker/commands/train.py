"""Handle the 'train-model' command."""

from __future__ import annotations

import sys


def _cmd_train_model(args) -> None:
    """Handle the 'train-model' command."""
    # Lazy import so that test patches on myspellchecker.cli.{os,TrainingPipeline} work.
    import myspellchecker.cli as _cli

    if not _cli.os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(2)

    if args.resume and not _cli.os.path.isdir(args.resume):
        print(f"Error: Checkpoint directory not found: {args.resume}", file=sys.stderr)
        sys.exit(2)

    if args.hidden_size % args.heads != 0:
        valid_heads = [h for h in [1, 2, 4, 8, 16] if args.hidden_size % h == 0]
        print(
            f"Error: hidden-size ({args.hidden_size}) must be divisible by heads ({args.heads}).",
            file=sys.stderr,
        )
        print(
            f"Valid heads for hidden-size={args.hidden_size}: {valid_heads}",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        from myspellchecker.training import TrainingConfig

        pipeline = _cli.TrainingPipeline()
        config = TrainingConfig(
            input_file=args.input,
            output_dir=args.output,
            architecture=args.architecture,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            warmup_ratio=args.warmup_ratio,
            weight_decay=args.weight_decay,
            hidden_size=args.hidden_size,
            num_layers=args.layers,
            num_heads=args.heads,
            max_length=args.max_length,
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
            resume_from_checkpoint=args.resume,
            keep_checkpoints=args.keep_checkpoints,
            save_metrics=not args.no_metrics,
            streaming=getattr(args, "streaming", False),
            checkpoint_dir=getattr(args, "checkpoint_dir", None),
            max_steps=getattr(args, "max_steps", None),
            fp16=getattr(args, "fp16", False),
            gradient_accumulation_steps=getattr(args, "gradient_accumulation_steps", 1),
        )
        if config.streaming:
            pipeline.run_streaming(config)
        else:
            pipeline.run(config)
    except ImportError as e:
        print(f"Dependency Error: {e}", file=sys.stderr)
        print(
            "Install training dependencies: pip install myspellchecker[train]",
            file=sys.stderr,
        )
        sys.exit(1)
    except ValueError as e:
        print(f"Configuration Error: {e}", file=sys.stderr)
        sys.exit(1)
    except (RuntimeError, OSError, MemoryError) as e:
        print(f"Training failed: {e}", file=sys.stderr)
        sys.exit(1)
