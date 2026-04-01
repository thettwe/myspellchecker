"""
Step 5: Database enrichment pipeline.

Mines confusable pairs, broken compounds, collocations, and register tags
from existing dictionary data (words + n-grams) and inserts them into
enrichment tables.  Runs as a post-packaging step after the main build.

This replaces hardcoded data constants (exempt pairs, confusion dicts,
frequency thresholds) with data-driven detection sourced from corpus
statistics.
"""

from __future__ import annotations

import math
import sqlite3
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Iterator

from ..core.myanmar_confusables import (
    _MEDIALS_INSERT_DELETE,
    _MEDIALS_INSERT_ONLY,
    ASPIRATION_PAIRS,
    MEDIAL_SWAP_PAIRS,
    NASAL_PAIRS,
    STOP_CODA_PAIRS,
    TONE_MARK_PAIRS,
    VOWEL_LENGTH_PAIRS,
    _insert_medial_after_consonants,
    _replace_per_position,
)
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ConfusablePair:
    """A mined confusable pair with metadata."""

    word1: str
    word2: str
    confusion_type: str  # aspiration, medial, nasal, stop_coda, tone, vowel_length
    context_overlap: float = 0.0
    freq_ratio: float = 1.0
    suppress: int = 0
    source: str = "mined"


@dataclass(frozen=True, slots=True)
class CompoundConfusion:
    """A mined broken compound pattern."""

    compound: str
    part1: str
    part2: str
    compound_freq: int = 0
    split_freq: int = 0
    pmi: float = 0.0


@dataclass(frozen=True, slots=True)
class Collocation:
    """A collocation with PMI score."""

    word1: str
    word2: str
    pmi: float
    npmi: float = 0.0
    count: int = 0


@dataclass(frozen=True, slots=True)
class RegisterTag:
    """A word's register classification."""

    word: str
    register: str  # formal, informal, neutral, literary
    confidence: float = 0.0
    formal_count: int = 0
    informal_count: int = 0


# ---------------------------------------------------------------------------
# Enrichment configuration
# ---------------------------------------------------------------------------


@dataclass
class EnrichmentConfig:
    """Configuration for the enrichment pipeline."""

    # Feature toggles
    enrich_confusables: bool = True
    enrich_compounds: bool = True
    enrich_collocations: bool = True
    enrich_register: bool = True

    # Confusable mining thresholds
    confusable_min_freq: int = 50
    confusable_max_freq_ratio: float = 1000.0

    # Compound mining thresholds
    compound_min_freq: int = 100
    compound_min_split_count: int = 10
    compound_min_pmi: float = 2.0

    # Collocation thresholds
    collocation_min_count: int = 20
    collocation_min_pmi: float = 3.0

    # Register thresholds
    register_min_total: int = 50
    register_threshold: float = 0.3

    # NER entity seeding
    seed_ner_entities: bool = True
    enrich_ner: bool = False  # Phase 3 stub — corpus mining not yet implemented


# ---------------------------------------------------------------------------
# Typed variant generation (wraps myanmar_confusables with type tags)
# ---------------------------------------------------------------------------


def generate_typed_variants(word: str) -> Iterator[tuple[str, str]]:
    """
    Generate Myanmar variants with their confusion type labels.

    Yields (variant, confusion_type) tuples for each valid variant.
    Unlike generate_myanmar_variants(), this preserves the type information
    needed for the confusable_pairs table.
    """
    from ..text.normalize import normalize

    normalized = unicodedata.normalize("NFC", word)

    # 1. Aspiration swaps
    for a, b in ASPIRATION_PAIRS:
        for i, char in enumerate(normalized):
            if char == a:
                v = normalize(normalized[:i] + b + normalized[i + 1 :])
                if v != word:
                    yield v, "aspiration"
            elif char == b:
                v = normalize(normalized[:i] + a + normalized[i + 1 :])
                if v != word:
                    yield v, "aspiration"

    # 2. Medial swaps (per-position, consistent with myanmar_confusables.py)
    for medial_a, medial_b in MEDIAL_SWAP_PAIRS:
        _medial_variants: set[str] = set()
        _replace_per_position(normalized, medial_a, medial_b, _medial_variants)
        _replace_per_position(normalized, medial_b, medial_a, _medial_variants)
        for v in _medial_variants:
            v = normalize(v)
            if v != word:
                yield v, "medial"

    # 2b. Medial insertion/deletion
    for medial in _MEDIALS_INSERT_DELETE:
        if medial in normalized:
            deleted = normalized.replace(medial, "")
            if len(deleted) > 1:
                v = normalize(deleted)
                if v != word:
                    yield v, "medial"
        variants_set: set[str] = set()
        _insert_medial_after_consonants(normalized, medial, variants_set)
        for var in variants_set:
            v = normalize(var)
            if v != word:
                yield v, "medial"

    for medial in _MEDIALS_INSERT_ONLY:
        variants_set = set()
        _insert_medial_after_consonants(normalized, medial, variants_set)
        for var in variants_set:
            v = normalize(var)
            if v != word:
                yield v, "medial"

    # 3. Nasal pairs
    for nasal_a, nasal_b in NASAL_PAIRS:
        if nasal_a in normalized:
            v = normalize(normalized.replace(nasal_a, nasal_b))
            if v != word:
                yield v, "nasal"
        if nasal_b in normalized:
            v = normalize(normalized.replace(nasal_b, nasal_a))
            if v != word:
                yield v, "nasal"

    # 4. Stop-coda pairs
    for stop_a, stop_b in STOP_CODA_PAIRS:
        start = 0
        while True:
            idx = normalized.find(stop_a, start)
            if idx < 0:
                break
            v = normalize(normalized[:idx] + stop_b + normalized[idx + len(stop_a) :])
            if v != word:
                yield v, "stop_coda"
            start = idx + 1
        start = 0
        while True:
            idx = normalized.find(stop_b, start)
            if idx < 0:
                break
            v = normalize(normalized[:idx] + stop_a + normalized[idx + len(stop_b) :])
            if v != word:
                yield v, "stop_coda"
            start = idx + 1

    # 5. Tone mark swaps
    for a, b in TONE_MARK_PAIRS:
        if a in normalized:
            v = normalize(normalized.replace(a, b))
            if v != word:
                yield v, "tone"
        if b in normalized:
            v = normalize(normalized.replace(b, a))
            if v != word:
                yield v, "tone"

    # 6. Vowel length swaps
    for a, b in VOWEL_LENGTH_PAIRS:
        if a in normalized:
            v = normalize(normalized.replace(a, b))
            if v != word:
                yield v, "vowel_length"
        if b in normalized:
            v = normalize(normalized.replace(b, a))
            if v != word:
                yield v, "vowel_length"


# ---------------------------------------------------------------------------
# Mining algorithms
# ---------------------------------------------------------------------------


def mine_confusable_pairs(
    conn: sqlite3.Connection,
    config: EnrichmentConfig | None = None,
) -> list[ConfusablePair]:
    """
    Mine confusable pairs by generating Myanmar variants for each word
    and checking which variants exist in the dictionary.

    Uses generate_typed_variants() to tag each pair with its confusion type.
    Computes context overlap from preloaded bigram distributions.
    """
    cfg = config or EnrichmentConfig()
    cursor = conn.cursor()

    # Build word→freq lookup from DB
    logger.info("Loading word frequencies for confusable mining...")
    cursor.execute(
        "SELECT word, frequency FROM words WHERE frequency >= ?", (cfg.confusable_min_freq,)
    )
    word_freq: dict[str, int] = {row[0]: row[1] for row in cursor.fetchall()}
    logger.info("Loaded %d words with freq >= %d", len(word_freq), cfg.confusable_min_freq)

    # Phase 1: Discover all candidate pairs (fast — no DB queries per pair)
    candidate_pairs: list[tuple[str, str, str, float]] = []
    seen: set[tuple[str, str, str]] = set()

    t0 = time.monotonic()
    processed = 0

    for word in word_freq:
        processed += 1
        if processed % 100000 == 0:
            elapsed = time.monotonic() - t0
            logger.info("  Variant scan: %d/%d words (%.1fs)", processed, len(word_freq), elapsed)

        for variant, confusion_type in generate_typed_variants(word):
            if variant not in word_freq:
                continue

            w1, w2 = (word, variant) if word <= variant else (variant, word)
            key = (w1, w2, confusion_type)
            if key in seen:
                continue
            seen.add(key)

            freq_w = word_freq[word]
            freq_v = word_freq[variant]
            ratio = max(freq_w, freq_v) / max(min(freq_w, freq_v), 1)

            if ratio > cfg.confusable_max_freq_ratio:
                continue

            candidate_pairs.append((w1, w2, confusion_type, ratio))

    elapsed = time.monotonic() - t0
    logger.info(
        "Found %d candidate pairs from %d words in %.1fs",
        len(candidate_pairs),
        len(word_freq),
        elapsed,
    )

    if not candidate_pairs:
        return []

    # Phase 2: Batch-compute context overlap using preloaded left-context vectors
    t1 = time.monotonic()

    # Collect all words that appear in pairs
    pair_words: set[str] = set()
    for w1, w2, _, _ in candidate_pairs:
        pair_words.add(w1)
        pair_words.add(w2)

    # Preload left-context vectors for pair words only
    left_contexts = _preload_left_contexts(conn, pair_words)
    logger.info(
        "Preloaded left contexts for %d words in %.1fs",
        len(left_contexts),
        time.monotonic() - t1,
    )

    # Compute overlap and build final pairs
    pairs: list[ConfusablePair] = []
    for w1, w2, confusion_type, ratio in candidate_pairs:
        ctx1 = left_contexts.get(w1, {})
        ctx2 = left_contexts.get(w2, {})
        context_overlap = _cosine_similarity(ctx1, ctx2) if ctx1 and ctx2 else 0.0

        pairs.append(
            ConfusablePair(
                word1=w1,
                word2=w2,
                confusion_type=confusion_type,
                context_overlap=context_overlap,
                freq_ratio=ratio,
            )
        )

    total_elapsed = time.monotonic() - t0
    logger.info(
        "Mined %d confusable pairs from %d words in %.1fs",
        len(pairs),
        len(word_freq),
        total_elapsed,
    )
    return pairs


def _preload_left_contexts(
    conn: sqlite3.Connection,
    target_words: set[str],
) -> dict[str, dict[int, int]]:
    """Preload left-context bigram vectors for a set of target words.

    Returns {word: {left_word_id: count}} for each target word.
    Much faster than per-pair queries when processing thousands of pairs.
    """
    cursor = conn.cursor()

    # Build word→id mapping (stream rows to avoid 600K-row fetchall() spike)
    cursor.execute("SELECT word, id FROM words")
    word_to_id: dict[str, int] = {}
    for row in cursor:
        word_to_id[row[0]] = row[1]

    target_ids: set[int] = set()
    id_to_target: dict[int, str] = {}
    for word in target_words:
        wid = word_to_id.get(word)
        if wid is not None:
            target_ids.add(wid)
            id_to_target[wid] = word

    if not target_ids:
        return {}

    # Load bigrams where word2_id is a target word (left context)
    # Filter in SQL to avoid loading all 2M+ bigrams into Python memory
    result: dict[str, dict[int, int]] = {}
    target_id_list = list(target_ids)
    batch_size = 900  # SQLite SQLITE_LIMIT_VARIABLE_NUMBER safe limit
    for batch_start in range(0, len(target_id_list), batch_size):
        batch = target_id_list[batch_start : batch_start + batch_size]
        placeholders = ",".join("?" for _ in batch)
        cursor.execute(
            f"SELECT word1_id, word2_id, count FROM bigrams "
            f"WHERE word2_id IN ({placeholders}) AND count > 0",
            batch,
        )
        for row in cursor:
            w2_id = row[1]
            word = id_to_target[w2_id]
            if word not in result:
                result[word] = {}
            result[word][row[0]] = row[2]

    return result


def _cosine_similarity(vec1: dict[int, int], vec2: dict[int, int]) -> float:
    """Cosine similarity between two sparse vectors."""
    common_keys = vec1.keys() & vec2.keys()
    if not common_keys:
        return 0.0

    dot = sum(vec1[k] * vec2[k] for k in common_keys)
    mag1 = math.sqrt(sum(v * v for v in vec1.values()))
    mag2 = math.sqrt(sum(v * v for v in vec2.values()))

    if mag1 == 0.0 or mag2 == 0.0:
        return 0.0

    return dot / (mag1 * mag2)


def mine_broken_compounds(
    conn: sqlite3.Connection,
    config: EnrichmentConfig | None = None,
) -> list[CompoundConfusion]:
    """
    Mine broken compound patterns by finding bigrams that concatenate
    to known dictionary words.
    """
    cfg = config or EnrichmentConfig()
    cursor = conn.cursor()

    # Build word→freq lookup
    cursor.execute("SELECT word, frequency FROM words")
    word_freq: dict[str, int] = {row[0]: row[1] for row in cursor.fetchall()}

    # Get all bigrams with sufficient count
    cursor.execute(
        """
        SELECT w1.word, w2.word, b.count
        FROM bigrams b
        JOIN words w1 ON b.word1_id = w1.id
        JOIN words w2 ON b.word2_id = w2.id
        WHERE b.count >= ?
    """,
        (cfg.compound_min_split_count,),
    )

    results: list[CompoundConfusion] = []
    seen: set[str] = set()
    # Using unigram total as denominator for both joint and marginal probabilities
    # is mathematically sound: in PMI = log2(P(x,y) / (P(x)*P(y))), the total T
    # cancels out: log2((count_xy/T) / ((count_x/T)*(count_y/T))) =
    # log2(count_xy * T / (count_x * count_y)).  The result is identical regardless
    # of whether T is the unigram total or the bigram total.
    total_words = sum(word_freq.values()) or 1

    for w1, w2, bigram_count in cursor.fetchall():
        compound = w1 + w2
        if compound in seen:
            continue

        compound_freq = word_freq.get(compound, 0)
        if compound_freq < cfg.compound_min_freq:
            continue

        # PMI: joint and marginal probabilities share the same denominator (total_words),
        # so the total cancels — see comment above.
        f1 = word_freq.get(w1, 0)
        f2 = word_freq.get(w2, 0)
        if f1 > 0 and f2 > 0 and bigram_count > 0:
            pmi = math.log2(
                (bigram_count / total_words) / ((f1 / total_words) * (f2 / total_words))
            )
        else:
            pmi = 0.0

        if pmi < cfg.compound_min_pmi:
            continue

        seen.add(compound)
        results.append(
            CompoundConfusion(
                compound=compound,
                part1=w1,
                part2=w2,
                compound_freq=compound_freq,
                split_freq=bigram_count,
                pmi=pmi,
            )
        )

    logger.info("Mined %d broken compound patterns", len(results))
    return results


def mine_collocations(
    conn: sqlite3.Connection,
    config: EnrichmentConfig | None = None,
) -> list[Collocation]:
    """
    Extract collocations with PMI/NPMI scores from bigram counts.
    """
    cfg = config or EnrichmentConfig()
    cursor = conn.cursor()

    # Get total word count for probability calculation.
    # Using unigram total for both joint and marginal probabilities is correct:
    # the denominator cancels in the PMI formula (see mine_broken_compounds).
    cursor.execute("SELECT SUM(frequency) FROM words")
    total_count = cursor.fetchone()[0] or 1

    # Get bigrams with sufficient count
    cursor.execute(
        """
        SELECT w1.word, w2.word, b.count
        FROM bigrams b
        JOIN words w1 ON b.word1_id = w1.id
        JOIN words w2 ON b.word2_id = w2.id
        WHERE b.count >= ?
    """,
        (cfg.collocation_min_count,),
    )

    # Build word freq lookup (needed for PMI)
    word_freq_cursor = conn.cursor()
    word_freq_cursor.execute("SELECT word, frequency FROM words")
    word_freq: dict[str, int] = {row[0]: row[1] for row in word_freq_cursor.fetchall()}

    results: list[Collocation] = []

    for w1, w2, count in cursor.fetchall():
        f1 = word_freq.get(w1, 0)
        f2 = word_freq.get(w2, 0)
        if f1 == 0 or f2 == 0:
            continue

        # PMI = log2(P(w1,w2) / (P(w1) * P(w2)))
        p_joint = count / total_count
        p_w1 = f1 / total_count
        p_w2 = f2 / total_count
        pmi = math.log2(p_joint / (p_w1 * p_w2))

        if pmi < cfg.collocation_min_pmi:
            continue

        # NPMI = PMI / -log2(P(w1,w2))  — normalizes to [-1, 1]
        npmi = pmi / (-math.log2(p_joint)) if 0 < p_joint < 1.0 else 0.0

        results.append(
            Collocation(
                word1=w1,
                word2=w2,
                pmi=pmi,
                npmi=npmi,
                count=count,
            )
        )

    logger.info("Mined %d collocations (PMI >= %.1f)", len(results), cfg.collocation_min_pmi)
    return results


# Register marker sets (closed-class, linguistically determined)
_FORMAL_MARKERS: frozenset[str] = frozenset(
    {
        "သည်",
        "၏",
        "သော",
        "နှင့်",
        "တွင်",
        "လျှင်",
        "မည်",
        "၍",
        "၌",
    }
)
_INFORMAL_MARKERS: frozenset[str] = frozenset(
    {
        "တယ်",
        "တဲ့",
        "နဲ့",
        "ရင်",
        "မယ်",
        "လို့",
        "တွေ",
        "ရဲ့",
    }
)


def mine_register_tags(
    conn: sqlite3.Connection,
    config: EnrichmentConfig | None = None,
) -> list[RegisterTag]:
    """
    Tag words by register based on co-occurrence with formal/informal markers.

    For each word, counts how often it appears in bigrams with known
    formal vs informal markers.  Words strongly biased toward one register
    get tagged.
    """
    cfg = config or EnrichmentConfig()
    cursor = conn.cursor()

    # Build word→id and id→word lookups
    cursor.execute("SELECT word, id FROM words")
    word_to_id: dict[str, int] = {}
    id_to_word: dict[int, str] = {}
    for word, wid in cursor.fetchall():
        word_to_id[word] = wid
        id_to_word[wid] = word

    # Get IDs for formal/informal markers
    formal_ids: set[int] = set()
    for marker in _FORMAL_MARKERS:
        if marker in word_to_id:
            formal_ids.add(word_to_id[marker])

    informal_ids: set[int] = set()
    for marker in _INFORMAL_MARKERS:
        if marker in word_to_id:
            informal_ids.add(word_to_id[marker])

    if not formal_ids or not informal_ids:
        logger.warning("Missing register markers in dictionary — skipping register mining")
        return []

    # Count co-occurrences: for each word, how often does it appear
    # adjacent to formal vs informal markers (in either direction)
    word_formal: dict[int, int] = {}
    word_informal: dict[int, int] = {}

    # Right context: word → marker (word appears before a marker)
    cursor.execute("SELECT word1_id, word2_id, count FROM bigrams WHERE count > 0")
    for w1_id, w2_id, count in cursor:
        if w2_id in formal_ids:
            word_formal[w1_id] = word_formal.get(w1_id, 0) + count
        elif w2_id in informal_ids:
            word_informal[w1_id] = word_informal.get(w1_id, 0) + count
        # Left context: marker → word (word appears after a marker)
        # Use separate if (not elif) so both directions are counted independently
        if w1_id in formal_ids:
            word_formal[w2_id] = word_formal.get(w2_id, 0) + count
        if w1_id in informal_ids:
            word_informal[w2_id] = word_informal.get(w2_id, 0) + count

    results: list[RegisterTag] = []

    for wid in set(word_formal.keys()) | set(word_informal.keys()):
        word = id_to_word.get(wid)
        if word is None:
            continue
        # Skip markers themselves
        if word in _FORMAL_MARKERS or word in _INFORMAL_MARKERS:
            continue

        fc = word_formal.get(wid, 0)
        ic = word_informal.get(wid, 0)
        total = fc + ic

        if total < cfg.register_min_total:
            continue

        score = (fc - ic) / (total + 1)

        if score > cfg.register_threshold:
            register = "formal"
        elif score < -cfg.register_threshold:
            register = "informal"
        else:
            register = "neutral"

        results.append(
            RegisterTag(
                word=word,
                register=register,
                confidence=abs(score),
                formal_count=fc,
                informal_count=ic,
            )
        )

    logger.info("Mined %d register tags", len(results))
    return results


# ---------------------------------------------------------------------------
# Insertion into DB
# ---------------------------------------------------------------------------


def insert_confusable_pairs(conn: sqlite3.Connection, pairs: list[ConfusablePair]) -> int:
    """Insert mined confusable pairs into the DB. Returns count inserted."""
    if not pairs:
        return 0
    cursor = conn.cursor()
    changes_before = conn.total_changes
    cursor.executemany(
        """INSERT OR IGNORE INTO confusable_pairs
           (word1, word2, confusion_type, context_overlap, freq_ratio, suppress, source)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        [
            (
                p.word1,
                p.word2,
                p.confusion_type,
                p.context_overlap,
                p.freq_ratio,
                p.suppress,
                p.source,
            )
            for p in pairs
        ],
    )
    conn.commit()
    inserted = conn.total_changes - changes_before
    logger.info("Inserted %d / %d confusable pairs", inserted, len(pairs))
    return inserted


def insert_compound_confusions(conn: sqlite3.Connection, compounds: list[CompoundConfusion]) -> int:
    """Insert mined compound confusions into the DB. Returns count inserted."""
    if not compounds:
        return 0
    cursor = conn.cursor()
    changes_before = conn.total_changes
    cursor.executemany(
        """INSERT OR IGNORE INTO compound_confusions
           (compound, part1, part2, compound_freq, split_freq, pmi)
           VALUES (?, ?, ?, ?, ?, ?)""",
        [(c.compound, c.part1, c.part2, c.compound_freq, c.split_freq, c.pmi) for c in compounds],
    )
    conn.commit()
    inserted = conn.total_changes - changes_before
    logger.info("Inserted %d / %d compound confusions", inserted, len(compounds))
    return inserted


def insert_collocations(conn: sqlite3.Connection, collocations: list[Collocation]) -> int:
    """Insert mined collocations into the DB. Returns count inserted."""
    if not collocations:
        return 0
    cursor = conn.cursor()
    changes_before = conn.total_changes
    cursor.executemany(
        """INSERT OR IGNORE INTO collocations
           (word1, word2, pmi, npmi, count)
           VALUES (?, ?, ?, ?, ?)""",
        [(c.word1, c.word2, c.pmi, c.npmi, c.count) for c in collocations],
    )
    conn.commit()
    inserted = conn.total_changes - changes_before
    logger.info("Inserted %d / %d collocations", inserted, len(collocations))
    return inserted


def insert_register_tags(conn: sqlite3.Connection, tags: list[RegisterTag]) -> int:
    """Insert mined register tags into the DB. Returns count inserted."""
    if not tags:
        return 0
    cursor = conn.cursor()
    cursor.executemany(
        """INSERT OR REPLACE INTO register_tags
           (word, register, confidence, formal_count, informal_count)
           VALUES (?, ?, ?, ?, ?)""",
        [(t.word, t.register, t.confidence, t.formal_count, t.informal_count) for t in tags],
    )
    conn.commit()
    # INSERT OR REPLACE counts as 2 changes (delete+insert) for replaced rows,
    # so use cursor.rowcount instead of total_changes delta for accurate count
    inserted = cursor.rowcount if cursor.rowcount >= 0 else len(tags)
    logger.info("Inserted %d register tags", inserted)
    return inserted


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


@dataclass
class EnrichmentReport:
    """Summary of enrichment results."""

    confusable_pairs: int = 0
    compound_confusions: int = 0
    collocations: int = 0
    register_tags: int = 0
    ner_entities: int = 0
    elapsed_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)


def run_enrichment(
    db_path: str,
    config: EnrichmentConfig | None = None,
) -> EnrichmentReport:
    """
    Run the full enrichment pipeline on a packaged database.

    This is Step 5 of the build pipeline — called after the DB is
    packaged with words, n-grams, and POS data.

    Args:
        db_path: Path to the SQLite database file.
        config: Enrichment configuration (uses defaults if None).

    Returns:
        EnrichmentReport with counts of enrichment data inserted.
    """
    cfg = config or EnrichmentConfig()
    report = EnrichmentReport()
    t0 = time.monotonic()

    logger.info("Starting enrichment pipeline on %s", db_path)

    conn = sqlite3.connect(db_path)
    try:
        # Ensure enrichment tables exist (idempotent)
        _ensure_enrichment_tables(conn)

        # 5a: Confusable pairs
        if cfg.enrich_confusables:
            try:
                pairs = mine_confusable_pairs(conn, cfg)
                report.confusable_pairs = insert_confusable_pairs(conn, pairs)
            except Exception as e:
                msg = f"Confusable mining failed: {e}"
                logger.error(msg)
                report.errors.append(msg)

        # 5b: Broken compounds
        if cfg.enrich_compounds:
            try:
                compounds = mine_broken_compounds(conn, cfg)
                report.compound_confusions = insert_compound_confusions(conn, compounds)
            except Exception as e:
                msg = f"Compound mining failed: {e}"
                logger.error(msg)
                report.errors.append(msg)

        # 5c: Collocations
        if cfg.enrich_collocations:
            try:
                collocs = mine_collocations(conn, cfg)
                report.collocations = insert_collocations(conn, collocs)
            except Exception as e:
                msg = f"Collocation mining failed: {e}"
                logger.error(msg)
                report.errors.append(msg)

        # 5d: Register tags
        if cfg.enrich_register:
            try:
                tags = mine_register_tags(conn, cfg)
                report.register_tags = insert_register_tags(conn, tags)
            except Exception as e:
                msg = f"Register mining failed: {e}"
                logger.error(msg)
                report.errors.append(msg)

        # 5e: Seed NER entities from YAML gazetteer
        if cfg.seed_ner_entities:
            try:
                report.ner_entities = seed_ner_from_gazetteer(conn)
            except Exception as e:
                msg = f"NER entity seeding failed: {e}"
                logger.error(msg)
                report.errors.append(msg)

    finally:
        conn.close()

    report.elapsed_seconds = time.monotonic() - t0
    logger.info(
        "Enrichment complete in %.1fs: %d confusables, %d compounds, "
        "%d collocations, %d register tags, %d NER entities",
        report.elapsed_seconds,
        report.confusable_pairs,
        report.compound_confusions,
        report.collocations,
        report.register_tags,
        report.ner_entities,
    )
    return report


def seed_ner_from_gazetteer(conn: sqlite3.Connection) -> int:
    """Seed ``ner_entities`` table from the YAML gazetteer.

    Reads :func:`~myspellchecker.text.ner.get_gazetteer_data` and inserts all
    entities with ``source='curated'``.  Uses ``INSERT OR IGNORE`` so it is
    safe to call repeatedly.

    Returns the number of entities inserted.
    """
    from myspellchecker.text.ner import get_gazetteer_data

    gaz = get_gazetteer_data()
    rows: list[tuple[str, str, str, float, int]] = []

    # Map GazetteerData fields to entity_type codes
    field_type_map = {
        "person_prefixes": "PER",
        "common_name_syllables": "PER",
        "organizations": "ORG",
        "townships": "LOC",
        "states_regions": "LOC",
        "major_cities": "LOC",
        "historical_places": "LOC",
        "international_places": "LOC",
        "countries": "LOC",
        "geographic_features": "LOC",
        "ethnic_groups": "ETHNICITY",
        "religious": "RELIGIOUS",
        "historical_figures": "HISTORICAL",
        "temporal": "TEMPORAL",
        "pali_sanskrit": "PALI",
    }

    for field_name, entity_type in field_type_map.items():
        entities = getattr(gaz, field_name, frozenset())
        for entity in entities:
            rows.append((entity, entity_type, "curated", 1.0, 0))

    if not rows:
        return 0

    cursor = conn.cursor()
    changes_before = conn.total_changes
    cursor.executemany(
        "INSERT OR IGNORE INTO ner_entities "
        "(entity, entity_type, source, confidence, frequency) "
        "VALUES (?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    inserted = conn.total_changes - changes_before
    logger.info("Seeded %d NER entities from gazetteer", inserted)
    return inserted


def _ensure_enrichment_tables(conn: sqlite3.Connection) -> None:
    """Create enrichment tables if they don't exist (for existing DBs)."""
    from .schema_manager import SchemaManager

    cursor = conn.cursor()

    # Check which tables already exist
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    existing = {row[0] for row in cursor.fetchall()}

    enrichment_tables = [
        "confusable_pairs",
        "compound_confusions",
        "collocations",
        "register_tags",
        "ner_entities",
    ]

    for table_name in enrichment_tables:
        if table_name not in existing:
            sql = SchemaManager.TABLES.get(table_name)
            if sql:
                cursor.execute(sql)
                logger.info("Created enrichment table: %s", table_name)

    # Create indexes
    enrichment_indexes = [
        "idx_confusable_word1",
        "idx_confusable_word2",
        "idx_compound_word",
        "idx_compound_parts",
        "idx_colloc_word1",
        "idx_colloc_word2",
        "idx_ner_entity",
        "idx_ner_entity_type",
    ]
    for idx_name in enrichment_indexes:
        sql = SchemaManager.INDEXES.get(idx_name)
        if sql:
            cursor.execute(sql)

    conn.commit()


# ---------------------------------------------------------------------------
# Phase 3 stubs: corpus-based NER entity mining (not yet implemented)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CorpusNEREntity:
    """A corpus-mined NER entity for the ``ner_entities`` table."""

    entity: str
    entity_type: str  # PER, LOC, ORG, etc.
    confidence: float = 0.0
    frequency: int = 0
    source: str = "corpus"


def insert_ner_entities(conn: sqlite3.Connection, entities: list[CorpusNEREntity]) -> int:
    """Insert corpus-mined NER entities into the ``ner_entities`` table.

    Uses ``INSERT OR IGNORE`` for deduplication.
    """
    if not entities:
        return 0
    cursor = conn.cursor()
    changes_before = conn.total_changes
    cursor.executemany(
        "INSERT OR IGNORE INTO ner_entities "
        "(entity, entity_type, source, confidence, frequency) "
        "VALUES (?, ?, ?, ?, ?)",
        [(e.entity, e.entity_type, e.source, e.confidence, e.frequency) for e in entities],
    )
    conn.commit()
    return conn.total_changes - changes_before
