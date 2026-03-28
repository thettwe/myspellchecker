"""Handle the 'infer-pos' command."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path


def _cmd_infer_pos(args) -> None:
    """Handle the 'infer-pos' command."""
    from myspellchecker.algorithms.pos_inference import POSInferenceEngine
    from myspellchecker.data_pipeline.database_packager import DatabasePackager

    database_path = Path(args.db)
    if not database_path.exists():
        print(f"Error: Database not found: {database_path}", file=sys.stderr)
        sys.exit(1)

    packager = None
    try:
        packager = DatabasePackager.from_existing(database_path)

        before_stats = packager.get_pos_coverage_stats()
        total = before_stats["total_words"] or 1
        pos_pct = (before_stats["with_pos_tag"] / total) * 100
        inferred_pct = (before_stats["with_inferred_pos"] / total) * 100
        print("POS Coverage Before Inference:")
        print(f"  Total words: {before_stats['total_words']:,}")
        print(f"  With POS tag: {before_stats['with_pos_tag']:,}")
        print(f"  With inferred POS: {before_stats['with_inferred_pos']:,}")
        print(f"  Coverage: {pos_pct:.1f}% (+{inferred_pct:.1f}% inferred)")
        print()

        if args.dry_run:
            print("Dry run mode - no changes will be made")
            print()

            conn = sqlite3.connect(str(database_path))
            try:
                cursor = conn.cursor()

                query = """
                    SELECT word, frequency FROM words
                    WHERE (pos_tag IS NULL OR pos_tag = '')
                """
                params = []
                if not args.include_tagged:
                    query += " AND (inferred_pos IS NULL OR inferred_pos = '')"
                if args.min_frequency > 0:
                    query += " AND frequency >= ?"
                    params.append(args.min_frequency)
                query += " ORDER BY frequency DESC LIMIT 20"

                cursor.execute(query, params)
                sample_words = cursor.fetchall()
            finally:
                conn.close()

            if sample_words:
                engine = POSInferenceEngine()
                print("Sample inference results (top 20 by frequency):")
                print("-" * 70)
                print(f"{'Word':<20} {'Freq':>8} {'POS':<8} {'Conf':>6} {'Source':<20}")
                print("-" * 70)

                for word, freq in sample_words:
                    inference_result = engine.infer_pos(word)
                    has_pos = inference_result.inferred_pos
                    meets_confidence = inference_result.confidence >= args.min_confidence
                    if has_pos and meets_confidence:
                        pos = inference_result.inferred_pos
                        confidence = inference_result.confidence
                        src = inference_result.source.value
                        print(f"{word:<20} {freq:>8} {pos:<8} {confidence:>6.2f} {src:<20}")
                    elif args.verbose:
                        confidence = inference_result.confidence
                        src = inference_result.source.value
                        print(f"{word:<20} {freq:>8} {'?':<8} {confidence:>6.2f} {src:<20}")
                print("-" * 70)
            else:
                print("No untagged words found matching criteria.")
        else:
            print("Running POS inference...")
            stats = packager.apply_inferred_pos(
                min_frequency=args.min_frequency,
                min_confidence=args.min_confidence,
                skip_tagged=not args.include_tagged,
            )

            print()
            print("Inference Results:")
            print(f"  Words processed: {stats['total_words']:,}")
            print(f"  Successfully inferred: {stats['inferred']:,}")
            print(f"  Ambiguous words: {stats['ambiguous']:,}")
            unknown = (
                stats["total_words"]
                - stats["inferred"]
                - stats["skipped_tagged"]
                - stats["skipped_low_conf"]
            )
            print(f"  Unknown (no inference): {unknown:,}")
            if stats["skipped_tagged"] > 0:
                print(f"  Skipped (already tagged): {stats['skipped_tagged']:,}")
            if stats["skipped_low_conf"] > 0:
                print(f"  Skipped (low confidence): {stats['skipped_low_conf']:,}")

            if stats.get("by_source"):
                print()
                print("Inference by source:")
                for source, count in sorted(stats["by_source"].items()):
                    print(f"  {source}: {count:,}")

            print()
            after_stats = packager.get_pos_coverage_stats()
            after_total = after_stats["total_words"] or 1
            after_pos_pct = (after_stats["with_pos_tag"] / after_total) * 100
            after_inferred_pct = (after_stats["with_inferred_pos"] / after_total) * 100
            print("POS Coverage After Inference:")
            print(f"  Total words: {after_stats['total_words']:,}")
            print(f"  With POS tag: {after_stats['with_pos_tag']:,}")
            print(f"  With inferred POS: {after_stats['with_inferred_pos']:,}")
            print(f"  Coverage: {after_pos_pct:.1f}% (+{after_inferred_pct:.1f}% inferred)")

            before_combined = pos_pct + inferred_pct
            after_combined = after_pos_pct + after_inferred_pct
            improvement = after_combined - before_combined
            print(f"  Improvement: +{improvement:.1f}%")

    except (sqlite3.DatabaseError, RuntimeError, OSError) as e:
        print(f"Error during POS inference: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)
    finally:
        if packager is not None:
            packager.close()
