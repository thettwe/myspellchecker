"""Reranker training data generator.

Generates labeled JSONL training data for the neural suggestion reranker MLP
(trained in ``reranker_trainer.py``). This module is used exclusively during
the **offline training pipeline** -- it is NOT used at inference time.

Pipeline overview:
    1. Read clean sentences from a segmented Arrow IPC corpus (produced by the
       data pipeline's ``database_packager``).
    2. Corrupt a random word per sentence using ``SyntheticErrorGenerator``
       (from ``training/generator.py``).
    3. Run the full spell-checker pipeline on the corrupted sentence to collect
       candidate suggestions.
    4. Extract 19 features per candidate (edit distance, frequency, n-gram
       context, phonetic similarity, confusable status, edit type, etc.).
    5. Write labeled JSONL with the gold correction index for supervised training.

System integration:
    - **Upstream**: Requires a production SQLite database (built by
      ``data_pipeline/pipeline.py``) and a segmented Arrow corpus.
    - **Downstream**: Output JSONL is consumed by ``RerankerTrainer`` in
      ``reranker_trainer.py``, which trains a small MLP and exports to ONNX.
    - **At inference**: The trained ONNX model is loaded by
      ``algorithms/ranker.py`` to re-score spell-checker suggestions. This
      module itself is not involved at inference time.

Key classes and functions:
    - ``RerankerDataGenerator``: Main class. Lazily initializes SpellChecker
      and SyntheticErrorGenerator, then processes sentences one at a time.
    - ``RerankerExample``: Dataclass for a single training example (sentence,
      corrupted form, candidates, 19-feature vectors, gold index).
    - ``GenerationStats``: Tracks success/failure counts during generation.
    - ``generate_threaded()``: Preferred entry point for large-scale generation.
      Uses DuckDB for vectorized corpus sampling + ThreadPoolExecutor for
      processing (single-threaded due to SpellChecker thread-safety constraint).
    - ``generate_parallel()``: Legacy multiprocessing entry point (deprecated;
      duplicates ~500 MB SpellChecker per worker).
    - ``NUM_FEATURES`` / ``FEATURE_NAMES``: Constants defining the 19-feature
      schema (v2). Must stay in sync with ``_extract_features()`` and the trainer.

Usage (single-threaded):
    >>> from myspellchecker.training.reranker_data import RerankerDataGenerator
    >>> gen = RerankerDataGenerator(
    ...     db_path="data/mySpellChecker_production.db",
    ...     arrow_corpus_path="data/segmented_corpus.arrow",
    ... )
    >>> gen.generate(num_examples=100_000, output_path="data/reranker_training.jsonl")

Usage (threaded, recommended for large runs):
    >>> from myspellchecker.training.reranker_data import generate_threaded
    >>> stats = generate_threaded(
    ...     db_path="data/mySpellChecker_production.db",
    ...     arrow_corpus_path="data/segmented_corpus.arrow",
    ...     output_path="data/reranker_training_100k.jsonl",
    ...     num_examples=100_000,
    ... )
"""

from __future__ import annotations

import json
import math
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from myspellchecker.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Per-worker memory estimate in bytes.  Each worker loads:
# - SpellChecker + SQLiteProvider with in-memory preload (~300-500 MB)
# - SyntheticErrorGenerator + PhoneticHasher (~20 MB)
# - Python interpreter overhead (~50 MB)
# Conservative estimate to avoid OOM.
_ESTIMATED_WORKER_MEMORY_BYTES = 600 * 1024 * 1024  # 600 MB

# Maximum work items to buffer in memory at once during parallel generation.
# Each item is a (text, words) tuple — roughly ~1 KB per sentence.
_STREAM_CHUNK_SIZE = 10_000

# Reserve this fraction of system memory for the OS and main process.
_MEMORY_RESERVE_FRACTION = 0.25


def _get_available_memory_bytes() -> int:
    """Return available system memory in bytes.

    Uses platform-specific methods to get *available* (not total) memory.
    Falls back to total memory if available memory cannot be determined.
    """
    try:
        import psutil

        return psutil.virtual_memory().available
    except ImportError:
        pass

    # macOS fallback: vm_stat + sysctl
    import platform
    import subprocess

    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            total = int(result.stdout.strip())
            # Estimate 60% available (conservative without vm_stat parsing)
            return int(total * 0.6)
        except Exception:
            pass

    # Linux fallback: /proc/meminfo
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024  # KB → bytes
    except (FileNotFoundError, ValueError):
        pass

    # Last resort: assume 8 GB available
    logger.warning("Could not determine available memory; assuming 8 GB")
    return 8 * 1024 * 1024 * 1024


def _calculate_safe_workers(
    requested_workers: int,
    max_memory_pct: float = 0.75,
) -> int:
    """Calculate the number of workers that fit in available memory.

    Args:
        requested_workers: User-requested worker count (0 = auto).
        max_memory_pct: Maximum fraction of available memory to use
            for workers (default: 75%, rest reserved for OS + main process).

    Returns:
        Safe number of workers (>= 1).
    """
    cpu_count = os.cpu_count() or 4

    if requested_workers <= 0:
        requested_workers = max(1, cpu_count - 1)

    available = _get_available_memory_bytes()
    usable = int(available * max_memory_pct)
    memory_based_max = max(1, usable // _ESTIMATED_WORKER_MEMORY_BYTES)

    safe_workers = min(requested_workers, memory_based_max)

    if safe_workers < requested_workers:
        logger.warning(
            "Reducing workers from %d to %d based on available memory "
            "(%.1f GB available, ~%d MB per worker)",
            requested_workers,
            safe_workers,
            available / (1024**3),
            _ESTIMATED_WORKER_MEMORY_BYTES // (1024 * 1024),
        )
    else:
        logger.info(
            "Memory check OK: %d workers × %d MB = %.1f GB (%.1f GB available, %.0f%% budget)",
            safe_workers,
            _ESTIMATED_WORKER_MEMORY_BYTES // (1024 * 1024),
            safe_workers * _ESTIMATED_WORKER_MEMORY_BYTES / (1024**3),
            available / (1024**3),
            max_memory_pct * 100,
        )

    return safe_workers


# Number of features per candidate (keep in sync with FEATURE_NAMES)
NUM_FEATURES = 19

# Legacy constant for backward-compatible loading of v1 training data.
NUM_FEATURES_V1 = 20

FEATURE_NAMES: list[str] = [
    # --- Distance features (0-6) ---
    "edit_distance",  # 0: raw Damerau-Levenshtein distance
    "weighted_distance",  # 1: Myanmar-weighted edit distance
    "log_frequency",  # 2: log1p(word_frequency)
    "phonetic_score",  # 3: phonetic similarity [0, 1]
    "syllable_count_diff",  # 4: absolute syllable count difference
    "plausibility_ratio",  # 5: weighted_dist / raw_dist
    "span_length_ratio",  # 6: len(candidate) / len(error)
    # --- Context features (7-9) ---
    "mlm_logit",  # 7: MLM logit from semantic checker (wired in v2)
    "ngram_left_prob",  # 8: left context n-gram probability
    "ngram_right_prob",  # 9: right context n-gram probability
    # --- Classification features (10) ---
    "is_confusable",  # 10: 1.0 if Myanmar confusable variant
    # --- Frequency/structure features (11-14) ---
    "relative_log_freq",  # 11: log_freq / max(log_freq across candidates)
    "char_length_diff",  # 12: len(candidate) - len(error), signed
    "is_substring",  # 13: 1.0 if candidate contains error or vice versa
    "original_rank",  # 14: 1/(1+rank) — prior ranking signal
    # --- v2 features: edit type & similarity (15-18) ---
    "ngram_improvement_ratio",  # 15: log(P_cand_ctx / P_error_ctx) improvement
    "edit_type_subst",  # 16: 1.0 if primary edit is substitution
    "edit_type_delete",  # 17: 1.0 if primary edit is deletion/insertion
    "char_dice_coeff",  # 18: character bigram Dice coefficient
]

# Index of the original_rank feature (label leakage for MLP; safe for GBT).
ORIGINAL_RANK_INDEX = FEATURE_NAMES.index("original_rank")  # 14

# MLP cross-feature definitions: (name, left_index, right_index).
# These are computed from base features at training/inference time.
# GBT models learn interactions via tree splits and do NOT need these.
MLP_CROSS_FEATURES: list[tuple[str, int, int]] = [
    ("edit_dist_x_ngram_improv", 0, 15),  # edit_distance * ngram_improvement_ratio
    ("phonetic_x_confusable", 3, 10),  # phonetic_score * is_confusable
    ("freq_x_dice", 11, 18),  # relative_log_freq * char_dice_coeff
    ("mlm_x_ngram_sum", 7, -1),  # mlm_logit * (ngram_left + ngram_right) — special
    ("edit_dist_x_freq", 0, 11),  # edit_distance * relative_log_freq (cost vs likelihood)
]

MLP_CROSS_FEATURE_NAMES: list[str] = [name for name, _, _ in MLP_CROSS_FEATURES]


@dataclass
class RerankerExample:
    """A single reranker training example."""

    sentence: str
    corrupted_sentence: str
    error_word: str
    error_position: int
    gold_correction: str
    gold_index: int
    candidates: list[str]
    features: list[list[float]]
    error_type: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GenerationStats:
    """Statistics from a generation run."""

    total_sentences_sampled: int = 0
    corruption_failures: int = 0
    no_error_detected: int = 0
    gold_not_in_candidates: int = 0
    no_suggestions: int = 0
    examples_generated: int = 0
    pipeline_errors: int = 0
    elapsed_seconds: float = 0.0


class RerankerDataGenerator:
    """Generate training data for the neural suggestion reranker.

    Reads clean sentences from an Arrow IPC corpus, corrupts a random
    word per sentence using SyntheticErrorGenerator, runs the spell
    checker to collect candidate suggestions + features, and writes
    labeled JSONL where the gold correction index is known.

    Args:
        db_path: Path to the production SQLite database.
        arrow_corpus_path: Path to the segmented Arrow IPC corpus.
        seed: Random seed for reproducibility.
        min_words: Minimum word count per sentence.
        max_words: Maximum word count per sentence.
        max_suggestions: Maximum suggestions to keep per error.
        semantic_model_path: Optional path to semantic MLM model directory.
            When provided, real MLM logits are wired into feature 7
            (``mlm_logit``) instead of the placeholder ``0.0``.
    """

    def __init__(
        self,
        db_path: str,
        arrow_corpus_path: str,
        seed: int = 42,
        min_words: int = 3,
        max_words: int = 20,
        max_suggestions: int = 10,
        semantic_model_path: str | None = None,
    ):
        self.db_path = db_path
        self.arrow_corpus_path = arrow_corpus_path
        self.seed = seed
        self.min_words = min_words
        self.max_words = max_words
        self.max_suggestions = max_suggestions
        self.semantic_model_path = semantic_model_path
        self.rng = random.Random(seed)

        # Lazy-initialized components
        self._checker = None
        self._error_gen = None
        self._phonetic_hasher = None
        self._semantic_checker = None

    def _init_components(self):
        """Lazily initialize heavy components."""
        from myspellchecker.algorithms.distance.edit_distance import (
            damerau_levenshtein_distance,
            weighted_damerau_levenshtein_distance,
        )
        from myspellchecker.core.config import SpellCheckerConfig
        from myspellchecker.core.constants import ValidationLevel
        from myspellchecker.core.myanmar_confusables import (
            generate_confusable_variants,
        )
        from myspellchecker.core.spellchecker import SpellChecker
        from myspellchecker.providers import SQLiteProvider
        from myspellchecker.text.phonetic import PhoneticHasher
        from myspellchecker.training.generator import SyntheticErrorGenerator

        # Store imports for use in methods
        self._damerau_levenshtein_distance = damerau_levenshtein_distance
        self._weighted_damerau_levenshtein_distance = weighted_damerau_levenshtein_distance
        self._generate_confusable_variants = generate_confusable_variants
        self._ValidationLevel = ValidationLevel

        logger.info("Initializing SpellChecker with DB: %s", self.db_path)
        provider = SQLiteProvider(database_path=self.db_path)
        config = SpellCheckerConfig(
            max_suggestions=self.max_suggestions,
            use_phonetic=True,
            use_context_checker=True,
        )
        self._checker = SpellChecker(config=config, provider=provider)
        self._phonetic_hasher = self._checker._phonetic_hasher or PhoneticHasher()
        self._error_gen = SyntheticErrorGenerator(
            corruption_ratio=0.5,  # Unused; we call _apply_corruption directly
            seed=self.seed,
        )

        # Load semantic checker for MLM logit wiring (optional)
        if self.semantic_model_path:
            try:
                from myspellchecker.algorithms.semantic_checker import SemanticChecker

                sem_path = Path(self.semantic_model_path)
                if sem_path.is_dir():
                    # Directory: resolve model.onnx + tokenizer.json inside it
                    model_file = sem_path / "model.onnx"
                    tokenizer_file = sem_path / "tokenizer.json"
                    if not model_file.exists():
                        raise FileNotFoundError(f"No model.onnx in {sem_path}")
                    self._semantic_checker = SemanticChecker(
                        model_path=str(model_file),
                        tokenizer_path=str(tokenizer_file) if tokenizer_file.exists() else None,
                    )
                else:
                    # Direct file path (e.g., model.onnx)
                    self._semantic_checker = SemanticChecker(model_path=str(sem_path))
                logger.info("Semantic checker loaded from %s", self.semantic_model_path)
            except Exception as e:
                logger.warning("Failed to load semantic checker: %s (mlm_logit will be 0.0)", e)
                self._semantic_checker = None

        logger.info("Components initialized successfully")

    @property
    def checker(self):
        if self._checker is None:
            self._init_components()
        return self._checker

    @property
    def semantic_checker(self):
        if self._checker is None:
            self._init_components()
        return self._semantic_checker

    @property
    def error_gen(self):
        if self._error_gen is None:
            self._init_components()
        return self._error_gen

    @property
    def phonetic_hasher(self):
        if self._phonetic_hasher is None:
            self._init_components()
        return self._phonetic_hasher

    def generate(
        self,
        num_examples: int,
        output_path: str,
        batch_log_interval: int = 1000,
    ) -> GenerationStats:
        """Generate reranker training data and write to JSONL.

        Streams the Arrow corpus batch-by-batch, sampling sentences
        from each batch. Stops once num_examples have been collected.

        Args:
            num_examples: Target number of training examples.
            output_path: Path to output JSONL file.
            batch_log_interval: Log progress every N examples.

        Returns:
            GenerationStats with counts of successes and failures.
        """
        import pyarrow.ipc as ipc

        # Ensure components are initialized
        _ = self.checker

        stats = GenerationStats()
        t_start = time.time()

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Starting generation: target=%d examples, corpus=%s",
            num_examples,
            self.arrow_corpus_path,
        )

        with open(output, "w", encoding="utf-8") as out_f:
            with open(self.arrow_corpus_path, "rb") as arrow_f:
                reader = ipc.open_stream(arrow_f)

                batch_num = 0
                try:
                    while stats.examples_generated < num_examples:
                        try:
                            batch = reader.read_next_batch()
                        except StopIteration:
                            break

                        batch_num += 1
                        rows_in_batch = batch.num_rows
                        if rows_in_batch == 0:
                            continue

                        # Sample ~1% of rows from each batch for diversity,
                        # but ensure we sample at least 1 row.
                        remaining = num_examples - stats.examples_generated
                        sample_count = max(1, min(remaining, rows_in_batch // 100))
                        indices = self.rng.sample(
                            range(rows_in_batch),
                            min(sample_count, rows_in_batch),
                        )

                        words_col = batch.column("words")
                        text_col = batch.column("text")

                        for idx in indices:
                            if stats.examples_generated >= num_examples:
                                break

                            words = words_col[idx].as_py()
                            text = text_col[idx].as_py()

                            if not words or not text:
                                continue

                            # Filter by word count
                            if len(words) < self.min_words or len(words) > self.max_words:
                                continue

                            stats.total_sentences_sampled += 1
                            example = self._process_sentence(text, words, stats)

                            if example is not None:
                                line = json.dumps(example.to_dict(), ensure_ascii=False)
                                out_f.write(line + "\n")
                                stats.examples_generated += 1

                                if stats.examples_generated % batch_log_interval == 0:
                                    elapsed = time.time() - t_start
                                    rate = stats.examples_generated / elapsed
                                    logger.info(
                                        "Progress: %d/%d examples (%.1f ex/s, batch %d)",
                                        stats.examples_generated,
                                        num_examples,
                                        rate,
                                        batch_num,
                                    )

                except StopIteration:
                    pass

        stats.elapsed_seconds = time.time() - t_start
        logger.info(
            "Generation complete: %d examples in %.1fs "
            "(sampled=%d, no_error=%d, gold_miss=%d, "
            "no_sugg=%d, corrupt_fail=%d, pipe_err=%d)",
            stats.examples_generated,
            stats.elapsed_seconds,
            stats.total_sentences_sampled,
            stats.no_error_detected,
            stats.gold_not_in_candidates,
            stats.no_suggestions,
            stats.corruption_failures,
            stats.pipeline_errors,
        )
        return stats

    def _process_sentence(
        self,
        text: str,
        words: list[str],
        stats: GenerationStats,
        *,
        rng: random.Random | None = None,
        error_gen: Any | None = None,
    ) -> RerankerExample | None:
        """Process a single sentence into a reranker training example.

        1. Pick a random word to corrupt
        2. Corrupt it with SyntheticErrorGenerator
        3. Run spell checker on corrupted sentence
        4. Find the gold (original word) in suggestion list
        5. Extract 19 features (v2) for each candidate
        6. Return training example or None if gold not in candidates

        Args:
            text: Original clean sentence.
            words: Pre-segmented word list from the Arrow corpus.
            stats: Mutable stats object for tracking failures.
            rng: Optional per-thread RNG override (for thread safety).
            error_gen: Optional per-thread error generator override.

        Returns:
            RerankerExample or None if the example should be skipped.
        """
        _rng = rng if rng is not None else self.rng
        _error_gen = error_gen if error_gen is not None else self.error_gen

        # Pick a random word index to corrupt (skip punctuation / short words)
        eligible = [i for i, w in enumerate(words) if len(w) >= 2 and _is_myanmar_word(w)]
        if not eligible:
            stats.corruption_failures += 1
            return None

        target_idx = _rng.choice(eligible)
        original_word = words[target_idx]

        # Corrupt the word
        method_name = _error_gen._select_corruption_method()
        corrupted_word = _error_gen._apply_corruption(method_name, original_word)
        if corrupted_word == original_word:
            stats.corruption_failures += 1
            return None

        # Reconstruct sentence with the corrupted word
        corrupted_words = list(words)
        corrupted_words[target_idx] = corrupted_word
        corrupted_text = "".join(corrupted_words)

        # Compute character offset of the corrupted word
        error_position = sum(len(w) for w in words[:target_idx])

        # Run the spell checker
        try:
            result = self.checker.check(
                corrupted_text,
                level=self._ValidationLevel.WORD,
            )
        except Exception as e:
            logger.debug("Pipeline error on sentence: %s", e)
            stats.pipeline_errors += 1
            return None

        if not result.has_errors or not result.errors:
            stats.no_error_detected += 1
            return None

        # Find the error that overlaps with our corrupted position
        matched_error = None
        for err in result.errors:
            err_start = err.position
            err_end = err_start + len(err.text)
            # Check overlap with the corrupted word span
            corr_start = error_position
            corr_end = corr_start + len(corrupted_word)
            if err_start < corr_end and err_end > corr_start:
                matched_error = err
                break

        if matched_error is None:
            stats.no_error_detected += 1
            return None

        candidates = matched_error.suggestions
        if not candidates:
            stats.no_suggestions += 1
            return None

        # Check if the gold correction is in the candidate list
        gold_index = -1
        for i, cand in enumerate(candidates):
            if cand == original_word:
                gold_index = i
                break

        if gold_index < 0:
            stats.gold_not_in_candidates += 1
            return None

        # Compute MLM logits if semantic checker is available
        mlm_scores: dict[str, float] = {}
        if self.semantic_checker is not None:
            try:
                occurrence = corrupted_text[:error_position].count(corrupted_word)
                mlm_scores = self.semantic_checker.score_mask_candidates(
                    corrupted_text,
                    corrupted_word,
                    candidates,
                    occurrence=occurrence,
                )
            except (AttributeError, TypeError, ValueError):
                pass

        # Extract features for each candidate
        features = self._extract_features(
            corrupted_word,
            candidates,
            words,
            target_idx,
            mlm_scores=mlm_scores,
        )

        return RerankerExample(
            sentence=text,
            corrupted_sentence=corrupted_text,
            error_word=corrupted_word,
            error_position=error_position,
            gold_correction=original_word,
            gold_index=gold_index,
            candidates=candidates,
            features=features,
            error_type=method_name,
        )

    def _extract_features(
        self,
        error_word: str,
        candidates: list[str],
        context_words: list[str],
        target_idx: int,
        mlm_scores: dict[str, float] | None = None,
    ) -> list[list[float]]:
        """Extract NUM_FEATURES (19) features for each candidate.

        Features (v2, 19 total):
            0. edit_distance: raw Damerau-Levenshtein
            1. weighted_distance: Myanmar-weighted distance
            2. log_frequency: log1p(word_frequency)
            3. phonetic_score: phonetic similarity 0-1
            4. syllable_count_diff: syllable count difference
            5. plausibility_ratio: weighted_dist / raw_dist
            6. span_length_ratio: len(candidate) / len(error)
            7. mlm_logit: MLM logit from semantic checker (real when model loaded)
            8. ngram_left_prob: left context probability
            9. ngram_right_prob: right context probability
            10. is_confusable: 1.0 if Myanmar confusable
            11. relative_log_freq: normalized frequency within candidate list
            12. char_length_diff: signed character length difference
            13. is_substring: 1.0 if substring relationship exists
            14. original_rank: 1/(1+rank) prior ranking signal
            15. ngram_improvement_ratio: log(P_cand/P_error) context improvement
            16. edit_type_subst: 1.0 if primary edit is substitution
            17. edit_type_delete: 1.0 if primary edit is deletion/insertion
            18. char_dice_coeff: character bigram Dice coefficient

        Args:
            error_word: The corrupted word.
            candidates: List of candidate suggestions.
            context_words: Full word list of the sentence.
            target_idx: Index of the corrupted word in context_words.

        Returns:
            List of feature vectors, one per candidate.
        """
        from myspellchecker.segmenters.regex import RegexSegmenter

        provider = self.checker.provider
        ngram_checker = self.checker.context_checker
        hasher = self.phonetic_hasher

        # Precompute confusable variants for the error word
        confusable_set = self._generate_confusable_variants(error_word, hasher)

        # Build context windows
        prev_words = context_words[:target_idx]
        next_words = context_words[target_idx + 1 :]

        # Count error syllables once
        syllable_seg = RegexSegmenter()
        try:
            error_syllables = len(syllable_seg.segment_syllables(error_word))
        except Exception:
            error_syllables = 1

        error_len = len(error_word) or 1

        all_features: list[list[float]] = []

        # Pre-compute log frequencies for relative normalization
        log_freqs: list[float] = []
        for cand in candidates:
            freq = provider.get_word_frequency(cand) or 0
            if isinstance(freq, (int, float)):
                log_freqs.append(math.log1p(freq))
            else:
                log_freqs.append(0.0)
        max_log_freq = max(log_freqs) if log_freqs else 1.0

        for i, cand in enumerate(candidates):
            # 0. Edit distance
            edit_dist = float(self._damerau_levenshtein_distance(error_word, cand))

            # 1. Weighted distance
            weighted_dist = float(self._weighted_damerau_levenshtein_distance(error_word, cand))

            # 2. Log frequency (absolute)
            log_freq = log_freqs[i]

            # 3. Phonetic score
            try:
                phon_score = hasher.compute_phonetic_similarity(error_word, cand)
            except Exception:
                phon_score = 0.0

            # 4. Syllable count difference
            try:
                cand_syllables = len(syllable_seg.segment_syllables(cand))
            except Exception:
                cand_syllables = 1
            syl_diff = float(abs(cand_syllables - error_syllables))

            # 5. Plausibility ratio
            if edit_dist > 0:
                plausibility = weighted_dist / edit_dist
            else:
                plausibility = 1.0

            # 6. Span length ratio
            span_ratio = len(cand) / error_len

            # 7. MLM logit (real when semantic checker loaded, else 0.0)
            mlm_logit = (mlm_scores or {}).get(cand, 0.0)

            # 8. N-gram left probability
            if ngram_checker and prev_words:
                ngram_left = ngram_checker.get_best_left_probability(prev_words[-3:], cand)
            else:
                ngram_left = 0.0

            # 9. N-gram right probability
            if ngram_checker and next_words:
                ngram_right = ngram_checker.get_best_right_probability(cand, next_words[:3])
            else:
                ngram_right = 0.0

            # 10. Is confusable
            is_conf = 1.0 if cand in confusable_set else 0.0

            # 11. Relative log frequency (within this candidate list)
            relative_log_freq = log_freq / max_log_freq if max_log_freq > 0 else 0.0

            # 12. Character length difference (signed: positive = candidate longer)
            char_length_diff = float(len(cand) - len(error_word))

            # 13. Is substring (candidate contains error or vice versa)
            is_substr = 1.0 if (cand in error_word or error_word in cand) else 0.0

            # 14. Original rank signal: 1/(1+rank) so rank0=1.0, rank1=0.5, ...
            original_rank = 1.0 / (1.0 + i)

            # --- v2 features ---

            # 15. N-gram improvement ratio: how much context improves with candidate
            #     Uses best of left/right context.  Handles 0→nonzero transitions.
            ngram_improv = 0.0
            if ngram_checker:
                improvements: list[float] = []
                if prev_words:
                    err_l = ngram_checker.get_best_left_probability(prev_words[-3:], error_word)
                    if err_l > 0 and ngram_left > 0:
                        improvements.append(math.log(ngram_left / err_l))
                    elif err_l == 0.0 and ngram_left > 0:
                        improvements.append(5.0)  # impossible → likely
                if next_words:
                    err_r = ngram_checker.get_best_right_probability(error_word, next_words[:3])
                    if err_r > 0 and ngram_right > 0:
                        improvements.append(math.log(ngram_right / err_r))
                    elif err_r == 0.0 and ngram_right > 0:
                        improvements.append(5.0)
                if improvements:
                    ngram_improv = max(-5.0, min(5.0, max(improvements)))

            # 16-17. Edit type classification
            edit_type_subst = 0.0
            edit_type_delete = 0.0
            if len(cand) == len(error_word):
                edit_type_subst = 1.0  # Same length = likely substitution
            elif len(cand) != len(error_word):
                edit_type_delete = 1.0  # Different length = insertion/deletion

            # 18. Character bigram Dice coefficient
            char_dice = _char_bigram_dice(error_word, cand)

            feat_vec = [
                edit_dist,
                weighted_dist,
                log_freq,
                phon_score,
                syl_diff,
                plausibility,
                span_ratio,
                mlm_logit,
                ngram_left,
                ngram_right,
                is_conf,
                relative_log_freq,
                char_length_diff,
                is_substr,
                original_rank,
                ngram_improv,
                edit_type_subst,
                edit_type_delete,
                char_dice,
            ]
            all_features.append(feat_vec)

        return all_features


def _char_bigram_dice(a: str, b: str) -> float:
    """Compute character bigram Dice coefficient between two strings.

    Uses multiset intersection (Counter) to correctly handle repeated
    bigrams, which is important for Myanmar reduplication patterns.

    Returns 2*|intersection| / (|bigrams_a| + |bigrams_b|), a value in
    [0, 1] measuring character-level overlap.
    """
    from collections import Counter

    if len(a) < 2 and len(b) < 2:
        return 1.0 if a == b else 0.0
    bigrams_a = Counter(a[i : i + 2] for i in range(len(a) - 1)) if len(a) >= 2 else Counter()
    bigrams_b = Counter(b[i : i + 2] for i in range(len(b) - 1)) if len(b) >= 2 else Counter()
    total = sum(bigrams_a.values()) + sum(bigrams_b.values())
    if total == 0:
        return 0.0
    intersection = sum(min(bigrams_a[bg], bigrams_b[bg]) for bg in bigrams_a)
    return 2.0 * intersection / total


def _is_myanmar_word(word: str) -> bool:
    """Check if a word contains Myanmar characters (U+1000-U+109F)."""
    return any("\u1000" <= ch <= "\u109f" for ch in word)


# ---------------------------------------------------------------------------
# Resource monitoring
# ---------------------------------------------------------------------------


class _ResourceMonitor:
    """Monitor system memory and provide throttling signals.

    Checks memory usage every *check_interval* calls and sets a throttle
    flag when usage exceeds the configured threshold.
    """

    def __init__(
        self,
        max_memory_pct: float = 0.85,
        check_interval: int = 50,
    ):
        self.max_memory_pct = max_memory_pct
        self.check_interval = check_interval
        self._counter = 0
        self._throttled = False
        self._last_mem_pct = 0.0

    def check(self) -> bool:
        """Return True if OK to proceed, False if throttling is needed."""
        self._counter += 1
        if self._counter % self.check_interval != 0:
            return not self._throttled

        try:
            import psutil

            mem = psutil.virtual_memory()
            self._last_mem_pct = mem.percent
            if mem.percent > self.max_memory_pct * 100:
                if not self._throttled:
                    logger.warning(
                        "Memory pressure: %.1f%% used (threshold %.0f%%), throttling",
                        mem.percent,
                        self.max_memory_pct * 100,
                    )
                self._throttled = True
            else:
                if self._throttled:
                    logger.info("Memory pressure relieved: %.1f%% used", mem.percent)
                self._throttled = False
        except ImportError:
            pass

        return not self._throttled

    @property
    def memory_pct(self) -> float:
        return self._last_mem_pct


# ---------------------------------------------------------------------------
# DuckDB-based corpus sampling
# ---------------------------------------------------------------------------


def _presample_sentences_duckdb(
    arrow_corpus_path: str,
    min_words: int,
    max_words: int,
    target_count: int,
    seed: int,
    *,
    max_batches: int = 2000,
) -> list[tuple[str, list[str]]]:
    """Pre-sample sentences from Arrow corpus using DuckDB for filtering.

    Streams Arrow batches one at a time, uses DuckDB's vectorized engine
    to filter by word count, and performs reservoir sampling into a
    fixed-size buffer.  Memory usage is O(target_count) regardless of
    corpus size.

    Args:
        arrow_corpus_path: Path to Arrow IPC stream file.
        min_words: Minimum word count per sentence.
        max_words: Maximum word count per sentence.
        target_count: Number of sentences to sample.
        seed: Random seed for reproducibility.
        max_batches: Stop after this many batches (caps I/O time).

    Returns:
        List of (text, words) tuples, shuffled.
    """
    import duckdb
    import pyarrow as pa
    import pyarrow.ipc as ipc

    con = duckdb.connect()
    rng = random.Random(seed)

    reservoir_cap = target_count
    reservoir: list[tuple[str, list[str]]] = []
    seen = 0

    logger.info(
        "DuckDB pre-sampling: target=%d sentences from %s (max %d batches)",
        target_count,
        arrow_corpus_path,
        max_batches,
    )
    t0 = time.time()

    with open(arrow_corpus_path, "rb") as f:
        reader = ipc.open_stream(f)
        batch_num = 0

        for batch in reader:
            if batch.num_rows == 0:
                continue

            batch_num += 1
            if batch_num > max_batches:
                break

            # Use DuckDB for vectorized filtering (zero-copy Arrow scan)
            arrow_tbl = pa.Table.from_batches([batch])
            con.register("_batch", arrow_tbl)
            filtered = con.execute(
                "SELECT text, words FROM _batch WHERE word_count BETWEEN $1 AND $2",
                [min_words, max_words],
            ).fetchall()
            con.unregister("_batch")

            # Reservoir sampling: O(target_count) memory
            for text, words in filtered:
                seen += 1
                if len(reservoir) < reservoir_cap:
                    reservoir.append((text, words))
                else:
                    j = rng.randrange(seen)
                    if j < reservoir_cap:
                        reservoir[j] = (text, words)

            # Log progress every 200 batches
            if batch_num % 200 == 0:
                elapsed = time.time() - t0
                logger.info(
                    "Sampling: batch %d/%d, %d in reservoir (%d seen, %.1fs)",
                    batch_num,
                    max_batches,
                    len(reservoir),
                    seen,
                    elapsed,
                )

            # Early exit: reservoir saturated with good statistical coverage
            if len(reservoir) >= reservoir_cap and seen >= reservoir_cap * 10:
                logger.info(
                    "Early exit: reservoir full (%d) with %dx coverage after %d batches",
                    len(reservoir),
                    seen // reservoir_cap,
                    batch_num,
                )
                break

    con.close()
    rng.shuffle(reservoir)

    elapsed = time.time() - t0
    logger.info(
        "DuckDB pre-sampling complete: %d candidates from %d qualifying "
        "sentences in %d batches (%.1fs, %.0f rows/s)",
        len(reservoir),
        seen,
        batch_num,
        elapsed,
        seen / max(elapsed, 0.001),
    )
    return reservoir


# ---------------------------------------------------------------------------
# Threaded generation (DuckDB + ThreadPoolExecutor)
# ---------------------------------------------------------------------------

_thread_local = threading.local()
_thread_local_gen = 0  # Incremented per generate_threaded call to force re-init


def generate_threaded(
    db_path: str,
    arrow_corpus_path: str,
    output_path: str,
    num_examples: int,
    *,
    num_threads: int = 0,
    max_memory_pct: float = 0.85,
    min_words: int = 3,
    max_words: int = 20,
    max_suggestions: int = 10,
    seed: int = 42,
    log_interval: int = 1000,
    max_batches: int = 2000,
    semantic_model_path: str | None = None,
) -> GenerationStats:
    """Generate reranker training data using DuckDB + threading.

    Two-phase approach:

    **Phase 1** — Pre-sample candidate sentences from the Arrow corpus
    using DuckDB for vectorized filtering.  Memory: O(num_examples).

    **Phase 2** — Process candidates through a single shared
    SpellChecker via ``ThreadPoolExecutor``.  Cython extensions and
    ONNX Runtime release the GIL, enabling real parallelism without
    duplicating the ~500 MB SpellChecker per worker.

    Compared to ``generate_parallel`` (multiprocessing):
        - Memory: ~500 MB total vs. ~500 MB × N workers
        - CPU: capped at ``cpu_count // 2`` to prevent thermal throttle
        - Stability: no fork-based memory duplication or zombie processes

    Args:
        db_path: Path to the production SQLite database.
        arrow_corpus_path: Path to the segmented Arrow IPC corpus.
        output_path: Path to output JSONL file.
        num_examples: Target number of training examples.
        num_threads: Thread count (0 = auto: cpu_count // 2).
        max_memory_pct: Memory pressure threshold (0-1).
        min_words: Minimum word count per sentence.
        max_words: Maximum word count per sentence.
        max_suggestions: Max suggestions per error.
        seed: Random seed.
        log_interval: Log progress every N examples.
        max_batches: Max Arrow batches to read during sampling.
        semantic_model_path: Optional path to semantic MLM model for
            wiring real MLM logits into feature 7.

    Returns:
        GenerationStats with generation counts.
    """
    cpu_count = os.cpu_count() or 4

    if num_threads <= 0:
        # Use half the cores to prevent thermal throttle
        num_threads = max(1, cpu_count // 2)
    # Always leave at least 2 cores free for OS + I/O
    num_threads = max(1, min(num_threads, cpu_count - 2))

    logger.info(
        "Threaded generation: target=%d, threads=%d/%d cores, max_mem=%.0f%%",
        num_examples,
        num_threads,
        cpu_count,
        max_memory_pct * 100,
    )

    # --- Phase 1: Pre-sample sentences using DuckDB ---
    # Oversample 15x to account for corruption/detection yield losses.
    # Empirical yield is ~8-10% (corruption failures + gold misses + no-detect).
    oversample_factor = 15
    sentences = _presample_sentences_duckdb(
        arrow_corpus_path,
        min_words,
        max_words,
        target_count=num_examples * oversample_factor,
        seed=seed,
        max_batches=max_batches,
    )

    if not sentences:
        logger.error("No qualifying sentences found in corpus")
        return GenerationStats()

    logger.info(
        "Phase 1 complete: %d candidate sentences for %d target examples",
        len(sentences),
        num_examples,
    )

    # --- Phase 2: Process with SpellChecker + thread pool ---
    # SpellChecker has mutable per-check state that is not thread-safe.
    # Force single-threaded to avoid silent data corruption.
    if num_threads > 1:
        logger.warning(
            "num_threads=%d requested but SpellChecker is not thread-safe; "
            "forcing num_threads=1 for correctness",
            num_threads,
        )
        num_threads = 1
    generator = RerankerDataGenerator(
        db_path=db_path,
        arrow_corpus_path=arrow_corpus_path,
        seed=seed,
        min_words=min_words,
        max_words=max_words,
        max_suggestions=max_suggestions,
        semantic_model_path=semantic_model_path,
    )
    generator._init_components()

    global _thread_local_gen
    _thread_local_gen += 1
    current_gen = _thread_local_gen

    stats = GenerationStats()
    monitor = _ResourceMonitor(max_memory_pct=max_memory_pct)
    t_start = time.time()

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _process_item(text_words: tuple[str, list[str]]) -> tuple[dict | None, str]:
        """Thread worker: process one sentence, return (example_dict, status)."""
        text, words = text_words

        # Per-thread RNG + error generator, re-initialized when seed changes
        if getattr(_thread_local, "gen", None) != current_gen:
            _thread_local.gen = current_gen
            tid = threading.get_ident()
            _thread_local.rng = random.Random(seed ^ tid)
            from myspellchecker.training.generator import SyntheticErrorGenerator

            _thread_local.error_gen = SyntheticErrorGenerator(
                corruption_ratio=0.5,
                seed=seed ^ tid,
            )

        local_stats = GenerationStats()
        example = generator._process_sentence(
            text,
            words,
            local_stats,
            rng=_thread_local.rng,
            error_gen=_thread_local.error_gen,
        )

        if example is not None:
            return example.to_dict(), "ok"

        # Determine failure reason from local stats
        if local_stats.corruption_failures:
            return None, "corruption"
        if local_stats.no_error_detected:
            return None, "no_error"
        if local_stats.gold_not_in_candidates:
            return None, "gold_miss"
        if local_stats.no_suggestions:
            return None, "no_sugg"
        if local_stats.pipeline_errors:
            return None, "pipe_err"
        return None, "unknown"

    batch_size = num_threads * 8
    sentence_idx = 0
    resample_pass = 0
    max_resample_passes = 5

    with open(out_path, "w", encoding="utf-8") as out_f:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            while stats.examples_generated < num_examples:
                # Exhausted pre-sampled sentences — re-sample from corpus
                if sentence_idx >= len(sentences):
                    resample_pass += 1
                    if resample_pass > max_resample_passes:
                        logger.warning(
                            "Exhausted %d resample passes, stopping at %d/%d examples",
                            max_resample_passes,
                            stats.examples_generated,
                            num_examples,
                        )
                        break
                    remaining = num_examples - stats.examples_generated
                    logger.info(
                        "Resample pass %d: need %d more examples, re-sampling corpus",
                        resample_pass,
                        remaining,
                    )
                    sentences = _presample_sentences_duckdb(
                        arrow_corpus_path,
                        min_words,
                        max_words,
                        target_count=remaining * oversample_factor,
                        seed=seed + resample_pass * 1000,
                        max_batches=max_batches,
                    )
                    sentence_idx = 0
                    if not sentences:
                        logger.error("Re-sample returned no sentences")
                        break

                # Memory backpressure
                if not monitor.check():
                    logger.warning(
                        "Memory pressure (%.0f%%), pausing 3s and halving batch",
                        monitor.memory_pct,
                    )
                    time.sleep(3.0)
                    batch_size = max(num_threads, batch_size // 2)

                # Submit a batch of work items
                batch_end = min(sentence_idx + batch_size, len(sentences))
                batch_items = sentences[sentence_idx:batch_end]
                sentence_idx = batch_end

                futures = [executor.submit(_process_item, item) for item in batch_items]

                for future in as_completed(futures):
                    result_dict, status = future.result()
                    stats.total_sentences_sampled += 1

                    if status == "ok" and result_dict is not None:
                        out_f.write(json.dumps(result_dict, ensure_ascii=False) + "\n")
                        stats.examples_generated += 1

                        if stats.examples_generated % log_interval == 0:
                            elapsed = time.time() - t_start
                            rate = stats.examples_generated / elapsed
                            mem_info = ""
                            try:
                                import psutil

                                mem = psutil.virtual_memory()
                                mem_info = f", mem={mem.percent:.0f}%"
                            except ImportError:
                                pass
                            logger.info(
                                "Progress: %d/%d (%.1f ex/s%s)",
                                stats.examples_generated,
                                num_examples,
                                rate,
                                mem_info,
                            )

                        if stats.examples_generated >= num_examples:
                            # Cancel remaining futures
                            for f in futures:
                                f.cancel()
                            break
                    else:
                        if status == "corruption":
                            stats.corruption_failures += 1
                        elif status == "no_error":
                            stats.no_error_detected += 1
                        elif status == "gold_miss":
                            stats.gold_not_in_candidates += 1
                        elif status == "no_sugg":
                            stats.no_suggestions += 1
                        elif status == "pipe_err":
                            stats.pipeline_errors += 1

    stats.elapsed_seconds = time.time() - t_start
    logger.info(
        "Threaded generation complete: %d examples in %.1fs (%.1f ex/s)",
        stats.examples_generated,
        stats.elapsed_seconds,
        stats.examples_generated / max(stats.elapsed_seconds, 0.001),
    )
    return stats


# ---------------------------------------------------------------------------
# Legacy multiprocessing generation (kept for backward compat)
# ---------------------------------------------------------------------------

# Module-level worker state (initialized per process)
_worker_gen: RerankerDataGenerator | None = None


def _worker_init(db_path: str, seed: int, min_w: int, max_w: int, max_s: int) -> None:
    """Initialize a per-process RerankerDataGenerator."""
    global _worker_gen  # noqa: PLW0603
    _worker_gen = RerankerDataGenerator(
        db_path=db_path,
        arrow_corpus_path="",  # Not used in workers
        seed=seed + __import__("os").getpid(),
        min_words=min_w,
        max_words=max_w,
        max_suggestions=max_s,
    )
    _worker_gen._init_components()


def _worker_process(item: tuple[str, list[str]]) -> dict | None:
    """Process a single (text, words) pair in a worker."""
    text, words = item
    assert _worker_gen is not None
    stats = GenerationStats()
    example = _worker_gen._process_sentence(text, words, stats)
    if example is not None:
        return example.to_dict()
    return None


def _stream_corpus_chunks(
    arrow_corpus_path: str,
    rng: random.Random,
    min_words: int,
    max_words: int,
    chunk_size: int = _STREAM_CHUNK_SIZE,
):
    """Yield chunks of (text, words) tuples from the Arrow corpus.

    .. deprecated::
        Use :func:`generate_threaded` with DuckDB sampling instead.
        This multiprocessing approach duplicates ~500 MB per worker.
    """
    import pyarrow.ipc as ipc

    chunk: list[tuple[str, list[str]]] = []

    with open(arrow_corpus_path, "rb") as arrow_f:
        reader = ipc.open_stream(arrow_f)
        for batch in reader:
            rows = batch.num_rows
            if rows == 0:
                continue

            sample_n = max(1, rows // 50)
            indices = rng.sample(range(rows), min(sample_n, rows))

            words_col = batch.column("words")
            text_col = batch.column("text")

            for idx in indices:
                words = words_col[idx].as_py()
                text = text_col[idx].as_py()
                if words and text and min_words <= len(words) <= max_words:
                    chunk.append((text, words))
                    if len(chunk) >= chunk_size:
                        rng.shuffle(chunk)
                        yield chunk
                        chunk = []

    if chunk:
        rng.shuffle(chunk)
        yield chunk


def generate_parallel(
    db_path: str,
    arrow_corpus_path: str,
    output_path: str,
    num_examples: int,
    *,
    num_workers: int = 0,
    max_memory_pct: float = 0.75,
    min_words: int = 3,
    max_words: int = 20,
    max_suggestions: int = 10,
    seed: int = 42,
    log_interval: int = 1000,
) -> GenerationStats:
    """Generate reranker training data using multiple worker processes.

    .. deprecated::
        Use :func:`generate_threaded` instead.  This function duplicates
        the full SpellChecker (~500 MB) per worker process, which causes
        OOM on machines with limited memory.
    """
    import multiprocessing as mp

    logger.warning(
        "generate_parallel uses multiprocessing which duplicates ~500 MB "
        "per worker. Consider using generate_threaded() instead."
    )

    num_workers = _calculate_safe_workers(num_workers, max_memory_pct)

    rng = random.Random(seed)
    stats = GenerationStats()
    t_start = time.time()

    logger.info(
        "Parallel generation: target=%d, workers=%d, corpus=%s",
        num_examples,
        num_workers,
        arrow_corpus_path,
    )

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with mp.Pool(
        processes=num_workers,
        initializer=_worker_init,
        initargs=(db_path, seed, min_words, max_words, max_suggestions),
    ) as pool:
        with open(out_path, "w", encoding="utf-8") as out_f:
            for chunk in _stream_corpus_chunks(arrow_corpus_path, rng, min_words, max_words):
                if stats.examples_generated >= num_examples:
                    break

                for result in pool.imap_unordered(_worker_process, chunk, chunksize=32):
                    stats.total_sentences_sampled += 1
                    if result is not None:
                        out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                        stats.examples_generated += 1
                        if stats.examples_generated % log_interval == 0:
                            elapsed = time.time() - t_start
                            rate = stats.examples_generated / elapsed
                            logger.info(
                                "Progress: %d/%d (%.1f ex/s)",
                                stats.examples_generated,
                                num_examples,
                                rate,
                            )
                        if stats.examples_generated >= num_examples:
                            break

    stats.elapsed_seconds = time.time() - t_start
    logger.info(
        "Parallel generation complete: %d examples in %.1fs (%.1f ex/s)",
        stats.examples_generated,
        stats.elapsed_seconds,
        stats.examples_generated / max(stats.elapsed_seconds, 0.001),
    )
    return stats
