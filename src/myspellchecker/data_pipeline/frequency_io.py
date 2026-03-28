"""File I/O and reporting mixin for FrequencyBuilder.

Handles saving frequency tables and probabilities to TSV files,
hydrating builder state, and printing statistics.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from ..core.constants import (
    DATA_KEY_FREQUENCY,
    DATA_KEY_POS,
    DATA_KEY_PROBABILITY,
    DATA_KEY_SYLLABLE,
    DATA_KEY_SYLLABLE_COUNT,
    DATA_KEY_WORD,
    DEFAULT_FILE_ENCODING,
    DEFAULT_PIPELINE_BIGRAM_PROBS_FILE,
    DEFAULT_PIPELINE_POS_BIGRAM_PROBS_FILE,
    DEFAULT_PIPELINE_POS_TRIGRAM_PROBS_FILE,
    DEFAULT_PIPELINE_POS_UNIGRAM_PROBS_FILE,
    DEFAULT_PIPELINE_SYLLABLE_FREQS_FILE,
    DEFAULT_PIPELINE_TRIGRAM_PROBS_FILE,
    DEFAULT_PIPELINE_WORD_FREQS_FILE,
)
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class FrequencyIOMixin:
    """File I/O and reporting methods for FrequencyBuilder.

    All methods assume the following attributes are set by
    ``FrequencyBuilder.__init__``:

    - ``self.logger``
    - ``self.output_dir``
    - ``self.syllable_counts``, ``self.word_counts``
    - ``self.word_syllables``
    - ``self.word_pos_tags``
    - ``self.pos_tagger``
    - ``self._duckdb_ngram_store``
    - ``self.stats``
    """

    logger: Any
    output_dir: Path
    syllable_counts: Counter
    word_counts: Counter
    word_syllables: dict[str, int]
    word_pos_tags: dict[str, Counter]
    pos_tagger: Any
    _duckdb_ngram_store: Any
    stats: dict

    def hydrate(
        self,
        syllable_counts: Counter,
        word_counts: Counter,
        bigram_counts: Counter,
        trigram_counts: Counter,
        word_syllables: dict[str, int],
        pos_unigram_probs: dict[str, float],
        pos_bigram_probs: dict[tuple[str, str], float],
        pos_trigram_probs: dict[tuple[str, str, str], float],
    ) -> None:
        """
        Hydrate the builder with existing state for incremental updates.

        Only syllable/word unigrams and word_syllables are hydrated into Python
        counters.  Bigram/trigram counts are managed entirely in DuckDB and are
        rebuilt from the corpus during ``load_data()``.

        Args:
            syllable_counts: Existing syllable frequencies
            word_counts: Existing word frequencies
            bigram_counts: Accepted for API compat (unused — DuckDB rebuilds n-grams)
            trigram_counts: Accepted for API compat (unused — DuckDB rebuilds n-grams)
            word_syllables: Existing word syllable counts
            pos_unigram_probs: Accepted for API compat (recalculated from corpus)
            pos_bigram_probs: Accepted for API compat (recalculated from corpus)
            pos_trigram_probs: Accepted for API compat (recalculated from corpus)
        """
        self.logger.info("Hydrating FrequencyBuilder state...")
        self.syllable_counts.update(syllable_counts)
        self.word_counts.update(word_counts)
        self.word_syllables.update(word_syllables)
        self.logger.info("Hydration complete (n-grams managed by DuckDB).")

    def save_syllable_frequencies(
        self, filename: str = DEFAULT_PIPELINE_SYLLABLE_FREQS_FILE
    ) -> None:
        """
        Save syllable frequencies to TSV file.

        Format: syllable\tfrequency

        Args:
            filename: Output filename
        """
        output_path = self.output_dir / filename

        sorted_syllables = sorted(self.syllable_counts.items(), key=lambda x: x[1], reverse=True)

        with open(output_path, "w", encoding=DEFAULT_FILE_ENCODING) as f:
            f.write(f"{DATA_KEY_SYLLABLE}\t{DATA_KEY_FREQUENCY}\n")
            for syllable, count in sorted_syllables:
                f.write(f"{syllable}\t{count}\n")

        self.logger.info(f"Saved syllable frequencies to: {output_path}")
        self.logger.info(f"  Total entries: {len(sorted_syllables):,}")

    def save_word_frequencies(
        self,
        filename: str = DEFAULT_PIPELINE_WORD_FREQS_FILE,
        pos_min_ratio: float = 0.05,
    ) -> None:
        """
        Save word frequencies to TSV file.

        Format: word\tsyllable_count\tfrequency\tpos_tag

        The pos_tag column uses pipe-separated format with counts when a word
        has multiple POS tags observed in the corpus: ``N:85|V:15``

        Tags are ordered by frequency (most common first). Tags below
        *pos_min_ratio* of the total observations for that word are filtered
        out to reduce noise.

        Args:
            filename: Output filename
            pos_min_ratio: Minimum ratio (0.0-1.0) of observations for a POS
                tag to be included. Default 0.05 (5%). Set to 0 to keep all.
        """
        output_path = self.output_dir / filename

        sorted_words = sorted(self.word_counts.items(), key=lambda x: x[1], reverse=True)

        # Statistics
        words_with_pos = 0
        words_with_multi_pos = 0

        with open(output_path, "w", encoding=DEFAULT_FILE_ENCODING) as f:
            f.write(
                f"{DATA_KEY_WORD}\t{DATA_KEY_SYLLABLE_COUNT}\t{DATA_KEY_FREQUENCY}\t{DATA_KEY_POS}\n"
            )
            for word, count in sorted_words:
                syllable_count = self.word_syllables.get(word, 1)
                pos_tag = ""
                if word in self.word_pos_tags and self.word_pos_tags[word]:
                    tag_counter = self.word_pos_tags[word]
                    total_obs = sum(tag_counter.values())
                    # Filter tags below minimum ratio and sort by count desc
                    min_count = max(1, int(total_obs * pos_min_ratio))
                    filtered = [
                        (tag, cnt) for tag, cnt in tag_counter.most_common() if cnt >= min_count
                    ]
                    if filtered:
                        pos_tag = "|".join(f"{tag}:{cnt}" for tag, cnt in filtered)
                        words_with_pos += 1
                        if len(filtered) > 1:
                            words_with_multi_pos += 1
                f.write(f"{word}\t{syllable_count}\t{count}\t{pos_tag}\n")

        self.logger.info(f"Saved word frequencies to: {output_path}")
        self.logger.info(f"  Total entries: {len(sorted_words):,}")
        if self.pos_tagger and words_with_pos > 0:
            self.logger.info(f"  Words with POS tags: {words_with_pos:,}")
            self.logger.info(f"  Words with multi-POS: {words_with_multi_pos:,}")

    def save_bigram_probabilities(
        self,
        bigram_probs: None = None,
        filename: str = DEFAULT_PIPELINE_BIGRAM_PROBS_FILE,
    ):
        """
        Save bigram probabilities to TSV via DuckDB COPY TO.

        DuckDB calculates probabilities and exports in one SQL step.

        Args:
            bigram_probs: Unused (kept for API compat). Always None.
            filename: Output filename
        """
        if not self._duckdb_ngram_store:
            self.logger.warning("No DuckDB store — skipping bigram probability export")
            return
        output_path = self.output_dir / filename
        count = self._duckdb_ngram_store.save_bigram_probabilities(output_path)
        self.logger.info(f"Saved bigram probabilities to: {output_path}")
        self.logger.info(f"  Total entries: {count:,}")

    def save_trigram_probabilities(
        self,
        trigram_probs: None = None,
        filename: str = DEFAULT_PIPELINE_TRIGRAM_PROBS_FILE,
    ):
        """
        Save trigram probabilities to TSV via DuckDB COPY TO.

        DuckDB calculates probabilities and exports in one SQL step.

        Args:
            trigram_probs: Unused (kept for API compat). Always None.
            filename: Output filename
        """
        if not self._duckdb_ngram_store:
            self.logger.warning("No DuckDB store — skipping trigram probability export")
            return
        output_path = self.output_dir / filename
        count = self._duckdb_ngram_store.save_trigram_probabilities(output_path)
        self.logger.info(f"Saved trigram probabilities to: {output_path}")
        self.logger.info(f"  Total entries: {count:,}")

    def save_pos_unigram_probabilities(
        self,
        pos_unigram_probs: dict[str, float],
        filename: str = DEFAULT_PIPELINE_POS_UNIGRAM_PROBS_FILE,
    ):
        """
        Save POS unigram probabilities to TSV file.

        Format: pos\tprobability

        Args:
            pos_unigram_probs: Dictionary of POS unigram probabilities
            filename: Output filename
        """
        output_path = self.output_dir / filename

        sorted_probs = sorted(pos_unigram_probs.items(), key=lambda x: x[1], reverse=True)

        with open(output_path, "w", encoding=DEFAULT_FILE_ENCODING) as f:
            f.write(f"{DATA_KEY_POS}\t{DATA_KEY_PROBABILITY}\n")
            for pos, prob in sorted_probs:
                f.write(f"{pos}\t{prob:.6f}\n")

        self.logger.info(f"Saved POS unigram probabilities to: {output_path}")
        self.logger.info(f"  Total entries: {len(sorted_probs):,}")

    def save_pos_bigram_probabilities(
        self,
        pos_bigram_probs: dict[tuple[str, str], float],
        filename: str = DEFAULT_PIPELINE_POS_BIGRAM_PROBS_FILE,
    ):
        """
        Save POS bigram probabilities to TSV file.

        Format: pos1\tpos2\tprobability

        Args:
            pos_bigram_probs: Dictionary of POS bigram probabilities
            filename: Output filename
        """
        output_path = self.output_dir / filename

        sorted_probs = sorted(pos_bigram_probs.items(), key=lambda x: x[1], reverse=True)

        with open(output_path, "w", encoding=DEFAULT_FILE_ENCODING) as f:
            f.write(f"pos1\tpos2\t{DATA_KEY_PROBABILITY}\n")  # Using pos1/pos2 for clarity
            for (pos1, pos2), prob in sorted_probs:
                f.write(f"{pos1}\t{pos2}\t{prob:.6f}\n")

        self.logger.info(f"Saved POS bigram probabilities to: {output_path}")
        self.logger.info(f"  Total entries: {len(sorted_probs):,} (smoothed)")

    def save_pos_trigram_probabilities(
        self,
        pos_trigram_probs: dict[tuple[str, str, str], float],
        filename: str = DEFAULT_PIPELINE_POS_TRIGRAM_PROBS_FILE,
    ):
        """
        Save POS trigram probabilities to TSV file.

        Format: pos1\tpos2\tpos3\tprobability

        Args:
            pos_trigram_probs: Dictionary of POS trigram probabilities
            filename: Output filename
        """
        output_path = self.output_dir / filename

        sorted_probs = sorted(pos_trigram_probs.items(), key=lambda x: x[1], reverse=True)

        with open(output_path, "w", encoding=DEFAULT_FILE_ENCODING) as f:
            f.write(f"pos1\tpos2\tpos3\t{DATA_KEY_PROBABILITY}\n")
            for (pos1, pos2, pos3), prob in sorted_probs:
                f.write(f"{pos1}\t{pos2}\t{pos3}\t{prob:.6f}\n")

        self.logger.info(f"Saved POS trigram probabilities to: {output_path}")
        self.logger.info(f"  Total entries: {len(sorted_probs):,} (smoothed)")

    def print_stats(self) -> None:
        """Print frequency building statistics."""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("FREQUENCY BUILDING STATISTICS")
        self.logger.info("=" * 60)
        self.logger.info(f"Total syllables processed:    {self.stats['total_syllables']:,}")
        self.logger.info(
            f"Invalid syllables skipped:    {self.stats['invalid_syllables_skipped']:,}"
        )
        self.logger.info(f"Total words processed:        {self.stats['total_words']:,}")
        self.logger.info(f"Total bigrams processed:      {self.stats['total_bigrams']:,}")
        self.logger.info(f"Total trigrams processed:     {self.stats['total_trigrams']:,}")
        self.logger.info(f"Total POS unigrams processed: {self.stats['total_pos_unigrams']:,}")
        self.logger.info(f"Total POS bigrams processed:  {self.stats['total_pos_bigrams']:,}")
        self.logger.info("")
        self.logger.info(f"Unique syllables:             {self.stats['unique_syllables']:,}")
        self.logger.info(f"Unique words:                 {self.stats['unique_words']:,}")
        self.logger.info(f"Unique bigrams:               {self.stats['unique_bigrams']:,}")
        self.logger.info(f"Unique trigrams:              {self.stats['unique_trigrams']:,}")
        self.logger.info(f"Unique POS unigrams:          {self.stats['unique_pos_unigrams']:,}")
        self.logger.info(f"Unique POS bigrams:           {self.stats['unique_pos_bigrams']:,}")
        self.logger.info("")
        self.logger.info(f"Syllables after filtering:    {len(self.syllable_counts):,}")
        self.logger.info(f"Words after filtering:        {len(self.word_counts):,}")
        if self._duckdb_ngram_store:
            ngram_stats = self._duckdb_ngram_store.get_stats()
            self.logger.info(f"Bigrams after filtering:      {ngram_stats['unique_bigrams']:,}")
            self.logger.info(f"Trigrams after filtering:     {ngram_stats['unique_trigrams']:,}")
        self.logger.info("=" * 60)
