"""POS tagging processing mixin for FrequencyBuilder.

Handles POS tagging pass over corpus data, POS checkpoint save/load/recovery,
POS cache for skipping transformer calls, and POS probability calculations.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pyarrow as pa

from ..core.constants import INVALID_WORDS
from ..core.exceptions import PipelineError
from ..utils.logging_utils import get_logger

if TYPE_CHECKING:
    from rich.console import Console

logger = get_logger(__name__)


class POSProcessorMixin:
    """POS tagging and probability calculation methods for FrequencyBuilder.

    All methods assume the following attributes are set by
    ``FrequencyBuilder.__init__``:

    - ``self.logger``
    - ``self.pos_tagger``
    - ``self.word_pos_tags``
    - ``self.pos_unigram_counts``, ``self.pos_bigram_counts``,
      ``self.pos_trigram_counts``
    - ``self.pos_bigram_predecessor_counts``, ``self.pos_bigram_successor_counts``
    - ``self._pos_checkpoint_interval``, ``self._pos_batch_buffer_size``
    - ``self.sub_step_timings``
    """

    logger: Any
    pos_tagger: Any
    word_pos_tags: dict[str, Counter]
    pos_unigram_counts: Counter
    pos_bigram_counts: Counter
    pos_trigram_counts: Counter
    pos_bigram_predecessor_counts: Counter
    pos_bigram_successor_counts: Counter
    _pos_checkpoint_interval: int
    _pos_batch_buffer_size: int
    sub_step_timings: list[tuple[str, float]]

    # ------------------------------------------------------------------
    # POS checkpoint management
    # ------------------------------------------------------------------

    def _save_pos_checkpoint(
        self,
        duckdb_temp_dir: Path,
        row_groups_completed: int,
        total_row_groups: int,
        sentences_processed: int,
    ) -> None:
        """Save accumulated POS data to Parquet checkpoint files.

        Uses atomic write (write to .tmp then rename) to prevent corruption
        if a crash occurs mid-write.

        Checkpoint files:
            - checkpoint_pos_progress.json: row group progress marker
            - checkpoint_pos_word_tags.parquet: flattened (word, tag, count) rows
            - checkpoint_pos_counters.parquet: 5 counter types as (type, key1-3, count)
        """
        import json

        import pyarrow.parquet as pq

        # --- 1. Word-tag mapping (largest checkpoint, ~50M rows for big corpora) ---
        wt_words, wt_tags, wt_counts = [], [], []
        for word, tag_counter in self.word_pos_tags.items():
            for tag, count in tag_counter.items():
                wt_words.append(word)
                wt_tags.append(tag)
                wt_counts.append(count)

        wt_table = pa.table({"word": wt_words, "tag": wt_tags, "count": wt_counts})
        wt_path = duckdb_temp_dir / "checkpoint_pos_word_tags.parquet"
        wt_tmp = wt_path.with_suffix(".parquet.tmp")
        pq.write_table(wt_table, str(wt_tmp), compression="snappy")
        wt_tmp.rename(wt_path)

        # --- 2. POS counters (5 types combined into one table) ---
        ct_types, ct_k1, ct_k2, ct_k3, ct_counts = [], [], [], [], []

        # Unigrams: type="uni", key1=tag
        for tag, count in self.pos_unigram_counts.items():
            ct_types.append("uni")
            ct_k1.append(tag)
            ct_k2.append("")
            ct_k3.append("")
            ct_counts.append(count)

        # Bigrams: type="bi", key1=t1, key2=t2
        for (t1, t2), count in self.pos_bigram_counts.items():
            ct_types.append("bi")
            ct_k1.append(t1)
            ct_k2.append(t2)
            ct_k3.append("")
            ct_counts.append(count)

        # Trigrams: type="tri", key1=t1, key2=t2, key3=t3
        for (t1, t2, t3), count in self.pos_trigram_counts.items():
            ct_types.append("tri")
            ct_k1.append(t1)
            ct_k2.append(t2)
            ct_k3.append(t3)
            ct_counts.append(count)

        # Predecessor counts: type="pred", key1=t1
        for t1, count in self.pos_bigram_predecessor_counts.items():
            ct_types.append("pred")
            ct_k1.append(t1)
            ct_k2.append("")
            ct_k3.append("")
            ct_counts.append(count)

        # Successor counts: type="succ", key1=t1, key2=t2
        for (t1, t2), count in self.pos_bigram_successor_counts.items():
            ct_types.append("succ")
            ct_k1.append(t1)
            ct_k2.append(t2)
            ct_k3.append("")
            ct_counts.append(count)

        ct_table = pa.table(
            {
                "type": ct_types,
                "key1": ct_k1,
                "key2": ct_k2,
                "key3": ct_k3,
                "count": ct_counts,
            }
        )
        ct_path = duckdb_temp_dir / "checkpoint_pos_counters.parquet"
        ct_tmp = ct_path.with_suffix(".parquet.tmp")
        pq.write_table(ct_table, str(ct_tmp), compression="snappy")
        ct_tmp.rename(ct_path)

        # --- 3. Progress marker (written last — acts as commit marker) ---
        progress = {
            "row_groups_completed": row_groups_completed,
            "total_row_groups": total_row_groups,
            "sentences_processed": sentences_processed,
        }
        prog_path = duckdb_temp_dir / "checkpoint_pos_progress.json"
        prog_tmp = prog_path.with_suffix(".json.tmp")
        prog_tmp.write_text(json.dumps(progress))
        prog_tmp.rename(prog_path)

        self.logger.debug(
            "POS checkpoint saved: rg %d/%d, %d sentences",
            row_groups_completed,
            total_row_groups,
            sentences_processed,
        )

    def _load_pos_checkpoint(
        self,
        duckdb_temp_dir: Path,
        expected_total_row_groups: int,
    ) -> tuple[int, int] | None:
        """Load POS checkpoint and restore counters.

        Returns:
            Tuple of (row_group_to_resume_from, sentences_processed) or None
            if no valid checkpoint exists.
        """
        import json

        import pyarrow.parquet as pq

        prog_path = duckdb_temp_dir / "checkpoint_pos_progress.json"
        wt_path = duckdb_temp_dir / "checkpoint_pos_word_tags.parquet"
        ct_path = duckdb_temp_dir / "checkpoint_pos_counters.parquet"

        if not (prog_path.exists() and wt_path.exists() and ct_path.exists()):
            return None

        # Read progress marker
        progress = json.loads(prog_path.read_text())
        rg_completed = progress.get("row_groups_completed", 0)
        total_rg = progress.get("total_row_groups", 0)
        sentences_done = progress.get("sentences_processed", 0)

        # Validate: total row groups must match (file hasn't changed)
        if total_rg != expected_total_row_groups:
            self.logger.warning(
                "POS checkpoint total_row_groups mismatch (%d vs %d), discarding",
                total_rg,
                expected_total_row_groups,
            )
            self._cleanup_pos_checkpoint(duckdb_temp_dir)
            return None

        # Restore word-tag mapping
        wt_table = pq.read_table(str(wt_path))
        words = wt_table.column("word").to_pylist()
        tags = wt_table.column("tag").to_pylist()
        counts = wt_table.column("count").to_pylist()
        for word, tag, count in zip(words, tags, counts, strict=False):
            if word not in self.word_pos_tags:
                self.word_pos_tags[word] = Counter()
            self.word_pos_tags[word][tag] += count

        # Restore POS counters
        ct_table = pq.read_table(str(ct_path))
        ct_types = ct_table.column("type").to_pylist()
        ct_k1 = ct_table.column("key1").to_pylist()
        ct_k2 = ct_table.column("key2").to_pylist()
        ct_k3 = ct_table.column("key3").to_pylist()
        ct_counts = ct_table.column("count").to_pylist()

        for ctype, k1, k2, k3, count in zip(ct_types, ct_k1, ct_k2, ct_k3, ct_counts, strict=False):
            if ctype == "uni":
                self.pos_unigram_counts[k1] += count
            elif ctype == "bi":
                self.pos_bigram_counts[(k1, k2)] += count
            elif ctype == "tri":
                self.pos_trigram_counts[(k1, k2, k3)] += count
            elif ctype == "pred":
                self.pos_bigram_predecessor_counts[k1] += count
            elif ctype == "succ":
                self.pos_bigram_successor_counts[(k1, k2)] += count

        self.logger.info(
            "Restored POS checkpoint: resuming from row group %d/%d (%d sentences)",
            rg_completed,
            expected_total_row_groups,
            sentences_done,
        )
        return rg_completed, sentences_done

    def _cleanup_pos_checkpoint(self, duckdb_temp_dir: Path) -> None:
        """Remove POS checkpoint files."""
        for name in (
            "checkpoint_pos_progress.json",
            "checkpoint_pos_word_tags.parquet",
            "checkpoint_pos_counters.parquet",
        ):
            p = duckdb_temp_dir / name
            if p.exists():
                p.unlink()
        # Also clean up any leftover .tmp files
        for tmp in duckdb_temp_dir.glob("checkpoint_pos_*.tmp"):
            tmp.unlink()

    # ------------------------------------------------------------------
    # POS data loading and processing
    # ------------------------------------------------------------------

    @staticmethod
    def _prefetch_pos_row_group(
        pf: Any,
        rg_idx: int,
    ) -> list[list[str]]:
        """Read and filter a single Parquet row group for POS tagging.

        Designed to run in a background thread to overlap I/O with GPU compute.

        Args:
            pf: Open ParquetFile handle
            rg_idx: Row group index to read

        Returns:
            List of filtered sentence word lists
        """
        table = pf.read_row_group(rg_idx, columns=["words"])
        words_col = table.column("words").to_pylist()
        sentences = []
        for words in words_col:
            filtered = [w for w in words if w and w not in INVALID_WORDS]
            if filtered:
                sentences.append(filtered)
        return sentences

    def _duckdb_step_pos(
        self,
        console: "Console",
        temp_parquet: Path,
        total_rows: int,
        duckdb_temp_dir: Path,
    ) -> None:
        """POS tagging pass with Parquet row-group random access and checkpoint recovery."""
        import time
        from concurrent.futures import ThreadPoolExecutor

        import pyarrow.parquet as pq
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            SpinnerColumn,
            TaskProgressColumn,
            TextColumn,
            TimeElapsedColumn,
            TimeRemainingColumn,
        )

        console.print("[cyan]Running POS tagging pass...[/cyan]")
        pos_start = time.time()
        pos_sentences_processed = 0
        POS_BATCH_SIZE = self._pos_batch_buffer_size
        checkpoint_interval = self._pos_checkpoint_interval

        has_batch = hasattr(self.pos_tagger, "tag_sentences_batch")

        pos_cache: dict[str, str] = {}
        pos_cache_hits_total = 0
        POS_CACHE_WARMUP = 500_000

        pf = pq.ParquetFile(str(temp_parquet))
        total_rg = pf.metadata.num_row_groups

        if total_rg > 0:
            avg_rows_per_rg = total_rows / total_rg
            rg_checkpoint_interval = max(1, int(checkpoint_interval / max(avg_rows_per_rg, 1)))
        else:
            rg_checkpoint_interval = 1

        start_rg = 0
        checkpoint_result = self._load_pos_checkpoint(duckdb_temp_dir, total_rg)
        if checkpoint_result is not None:
            start_rg, pos_sentences_processed = checkpoint_result
            console.print(
                f"  [green]✓ POS checkpoint: resuming from row group "
                f"{start_rg}/{total_rg} ({pos_sentences_processed:,} sentences)[/green]"
            )

        sentences_since_checkpoint = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[cyan]POS tagging[/cyan]"),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TextColumn("ETA"),
            TimeRemainingColumn(),
            console=console,
        ) as pos_progress:
            pos_task = pos_progress.add_task(
                "POS tagging",
                total=total_rows,
                completed=pos_sentences_processed,
            )

            with ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="pos_prefetch"
            ) as prefetch_pool:
                prefetch_future = None
                if start_rg < total_rg:
                    prefetch_future = prefetch_pool.submit(
                        self._prefetch_pos_row_group, pf, start_rg
                    )

                for rg_idx in range(start_rg, total_rg):
                    if prefetch_future is None:
                        raise PipelineError(
                            f"Prefetch future is None at rg_idx={rg_idx} — "
                            f"corpus may have 0 row groups"
                        )
                    sentences_for_rg = prefetch_future.result()

                    if rg_idx + 1 < total_rg:
                        prefetch_future = prefetch_pool.submit(
                            self._prefetch_pos_row_group, pf, rg_idx + 1
                        )

                    for batch_start in range(0, len(sentences_for_rg), POS_BATCH_SIZE):
                        batch = sentences_for_rg[batch_start : batch_start + POS_BATCH_SIZE]
                        hits = self._process_pos_batch(
                            batch, has_batch, pos_cache=pos_cache or None
                        )
                        pos_cache_hits_total += hits
                        pos_sentences_processed += len(batch)
                        sentences_since_checkpoint += len(batch)
                        pos_progress.update(pos_task, advance=len(batch))

                    if (
                        sentences_since_checkpoint >= checkpoint_interval
                        or (rg_idx + 1) % rg_checkpoint_interval == 0
                    ):
                        self._save_pos_checkpoint(
                            duckdb_temp_dir,
                            rg_idx + 1,
                            total_rg,
                            pos_sentences_processed,
                        )
                        sentences_since_checkpoint = 0

                        if pos_sentences_processed >= POS_CACHE_WARMUP:
                            pos_cache = self._build_pos_cache()
                            self.logger.info(
                                "POS cache rebuilt: %d words cached, "
                                "%d/%d sentences resolved from cache (%.1f%%)",
                                len(pos_cache),
                                pos_cache_hits_total,
                                pos_sentences_processed,
                                100 * pos_cache_hits_total / max(pos_sentences_processed, 1),
                            )

            if total_rg > 0 and start_rg < total_rg:
                self._save_pos_checkpoint(
                    duckdb_temp_dir, total_rg, total_rg, pos_sentences_processed
                )

        pos_time = time.time() - pos_start
        rate = pos_sentences_processed / pos_time if pos_time > 0 else 0
        cache_pct = 100 * pos_cache_hits_total / max(pos_sentences_processed, 1)
        console.print(
            f"  ✓ POS tagged {pos_sentences_processed:,} sentences "
            f"({pos_time:.1f}s, {rate:.0f} sent/s)"
        )
        if pos_cache_hits_total > 0:
            console.print(
                f"  ✓ Cache: {pos_cache_hits_total:,} sentences skipped tagger "
                f"({cache_pct:.1f}%), {len(pos_cache):,} words cached"
            )
        self.sub_step_timings.append(("POS tagging", pos_time))

        self._cleanup_pos_checkpoint(duckdb_temp_dir)

    # ------------------------------------------------------------------
    # POS cache and batch processing
    # ------------------------------------------------------------------

    def _build_pos_cache(self, min_count: int = 50, dominance: float = 0.95) -> dict[str, str]:
        """Build a cache of words with a single dominant POS tag.

        After warmup, many high-frequency words have been tagged thousands of
        times and one tag dominates (>95%). These can be used directly without
        calling the transformer, saving significant GPU time.

        Args:
            min_count: Minimum total tag observations for a word to be cached.
            dominance: Fraction of observations the top tag must hold (0-1).

        Returns:
            Dict mapping word -> dominant POS tag.
        """
        cache: dict[str, str] = {}
        for word, tag_counter in self.word_pos_tags.items():
            total = sum(tag_counter.values())
            if total < min_count:
                continue
            top_tag, top_count = tag_counter.most_common(1)[0]
            if top_count / total >= dominance:
                cache[word] = top_tag
        return cache

    @staticmethod
    def _try_cached_pos(words: list[str], cache: dict[str, str]) -> list[str] | None:
        """Try to resolve all words in a sentence from the POS cache.

        Returns the full tag list if every word is cached, otherwise None
        (meaning the sentence must go through the transformer).
        """
        tags: list[str] = []
        for w in words:
            tag = cache.get(w)
            if tag is None:
                return None
            tags.append(tag)
        return tags

    def _process_pos_batch(
        self,
        sentences: list[list[str]],
        has_batch: bool,
        pos_cache: dict[str, str] | None = None,
    ) -> int:
        """Process a batch of sentences for POS tagging.

        When *pos_cache* is provided, sentences whose words are **all** present
        in the cache are resolved without calling the transformer, saving
        significant GPU time on large corpora (60-80% fewer tagger calls after
        warmup).

        Args:
            sentences: List of sentences (each sentence is a list of words)
            has_batch: Whether the tagger supports batch processing
            pos_cache: Optional word→tag cache from ``_build_pos_cache``.

        Returns:
            Number of sentences resolved from cache (0 if no cache).
        """
        # --- Split into cached vs. uncached ---------------------------------
        resolved: list[tuple] = []  # (words, tags) already resolved
        needs_tagging: list[list[str]] = []

        if pos_cache:
            for sent in sentences:
                cached_tags = self._try_cached_pos(sent, pos_cache)
                if cached_tags is not None:
                    resolved.append((sent, cached_tags))
                else:
                    needs_tagging.append(sent)
        else:
            needs_tagging = sentences

        cache_hits = len(resolved)

        # --- Tag uncached sentences via transformer / tagger ----------------
        if needs_tagging and self.pos_tagger is not None:
            if has_batch:
                try:
                    all_tags = self.pos_tagger.tag_sentences_batch(needs_tagging)
                except (RuntimeError, ValueError) as e:
                    self.logger.debug(f"Batch POS tagging failed: {e}, falling back")
                    all_tags = [self.pos_tagger.tag_sequence(s) for s in needs_tagging]
            else:
                all_tags = [self.pos_tagger.tag_sequence(s) for s in needs_tagging]

            for words, tags in zip(needs_tagging, all_tags, strict=False):
                resolved.append((words, tags))

        # --- Count n-grams for all sentences --------------------------------
        for sentence_words, pos_tags in resolved:
            pos_tags_sequence = [[tag] for tag in pos_tags]

            # Store word → POS mapping for later saving to database
            for word, tag in zip(sentence_words, pos_tags, strict=False):
                if word and tag and tag != "UNK":
                    if word not in self.word_pos_tags:
                        self.word_pos_tags[word] = Counter()
                    self.word_pos_tags[word][tag] += 1

            # Count POS unigrams
            for tags_list in pos_tags_sequence:
                for tag in tags_list:
                    self.pos_unigram_counts[tag] += 1

            # Count POS bigrams and trigrams
            seq_len = len(pos_tags_sequence)
            if seq_len >= 2:
                for k in range(seq_len - 1):
                    tags1_list = pos_tags_sequence[k]
                    tags2_list = pos_tags_sequence[k + 1]

                    for t1 in tags1_list:
                        for t2 in tags2_list:
                            self.pos_bigram_counts[(t1, t2)] += 1
                            self.pos_bigram_predecessor_counts[t1] += 1

                    # Trigrams
                    if k < seq_len - 2:
                        tags3_list = pos_tags_sequence[k + 2]
                        for t1 in tags1_list:
                            for t2 in tags2_list:
                                self.pos_bigram_successor_counts[(t1, t2)] += 1
                                for t3 in tags3_list:
                                    self.pos_trigram_counts[(t1, t2, t3)] += 1

        return cache_hits

    # ------------------------------------------------------------------
    # POS probability calculations
    # ------------------------------------------------------------------

    def calculate_pos_unigram_probabilities(self) -> dict[str, float]:
        """
        Calculate POS unigram probabilities.

        P(tag) = count(tag) / total_pos_unigrams_in_corpus (sum of all pos_unigram_counts)

        Returns:
            Dictionary mapping tag to probability
        """
        self.logger.info("\nCalculating POS unigram probabilities...")
        pos_unigram_probs = {}
        total_pos_unigrams = sum(self.pos_unigram_counts.values())

        if total_pos_unigrams == 0:
            self.logger.warning("  No POS unigrams found to calculate probabilities.")
            return {}

        for tag, count in self.pos_unigram_counts.items():
            pos_unigram_probs[tag] = count / total_pos_unigrams

        self.logger.info(f"  Calculated probabilities for {len(pos_unigram_probs):,} POS unigrams")
        return pos_unigram_probs

    def calculate_pos_bigram_probabilities(self) -> dict[tuple[str, str], float]:
        """
        Calculate POS bigram probabilities using Maximum Likelihood Estimation.

        P(tag2 | tag1) = count(tag1, tag2) / count(tag1)

        Returns:
            Dictionary mapping (tag1, tag2) to probability
        """
        self.logger.info("\nCalculating POS bigram probabilities...")
        pos_bigram_probs = {}

        # Get vocabulary size (V) for smoothing
        V = len(self.pos_unigram_counts)
        if V == 0:
            self.logger.warning(
                "  No POS unigrams found, cannot apply smoothing. Returning empty dict."
            )
            return {}

        for (
            tag1,
            _,
        ) in self.pos_unigram_counts.items():  # Iterate over all unique tags seen as unigrams
            tag1_total = self.pos_bigram_predecessor_counts.get(
                tag1, 0
            )  # Count of tag1 as a predecessor

            if tag1_total == 0:
                # If tag1 was never seen as a predecessor, all its bigrams (Tag1, X) have 0 count.
                # Assign uniform small probability 1/V for robustness
                for tag2 in self.pos_unigram_counts.keys():
                    pos_bigram_probs[(tag1, tag2)] = 1 / V
                continue

            for (
                tag2
            ) in self.pos_unigram_counts.keys():  # Iterate over all unique tags as possible_tag2
                observed_count = self.pos_bigram_counts.get((tag1, tag2), 0)

                # Laplace smoothing: P(tag2 | tag1) = (count(tag1, tag2) + 1) / (count(tag1) + V)
                probability = (observed_count + 1) / (tag1_total + V)
                pos_bigram_probs[(tag1, tag2)] = probability

        self.logger.info(
            f"  Calculated {len(pos_bigram_probs):,} POS bigram probabilities (Laplace smoothed)"
        )
        return pos_bigram_probs

    def calculate_pos_trigram_probabilities(self) -> dict[tuple[str, str, str], float]:
        """
        Calculate POS trigram probabilities using Laplace (Add-One) Smoothing.

        P(tag3 | tag1, tag2) = (count(tag1, tag2, tag3) + 1) / (count(tag1, tag2) + V)
        Where V is the vocabulary size of unique POS tags.

        Returns:
            Dictionary mapping (tag1, tag2, tag3) tuple to probability (float).
        """
        self.logger.info("\nCalculating POS trigram probabilities...")
        pos_trigram_probs = {}

        # Get vocabulary size (V) for smoothing
        V = len(self.pos_unigram_counts)
        if V == 0:
            self.logger.warning(
                "  No POS unigrams found, cannot apply smoothing. Returning empty dict."
            )
            return {}

        # Iterate over each unique (tag1, tag2) pair as a predecessor bigram
        for (tag1, tag2), predecessor_count in self.pos_bigram_successor_counts.items():
            # predecessor_count is effectively count(tag1, tag2) for this context

            if predecessor_count == 0:
                # This bigram was observed as a predecessor, but its count is zero?
                # This shouldn't happen if it's in pos_bigram_predecessor_counts
                continue

            # Iterate over all possible tag3 (next tag) from the vocabulary
            for tag3 in self.pos_unigram_counts.keys():
                # Get observed count(tag1, tag2, tag3), default to 0 if unseen
                observed_count = self.pos_trigram_counts.get((tag1, tag2, tag3), 0)

                # Laplace: P(tag3|tag1,tag2) = (count(t1,t2,t3)+1)/(count(t1,t2)+V)
                probability = (observed_count + 1) / (predecessor_count + V)
                pos_trigram_probs[(tag1, tag2, tag3)] = probability

        self.logger.info(
            f"  Calculated {len(pos_trigram_probs):,} POS trigram probabilities (Laplace smoothed)"
        )
        return pos_trigram_probs
