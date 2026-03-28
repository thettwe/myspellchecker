"""Tests for the data pipeline enrichment module (Step 5)."""

from __future__ import annotations

import sqlite3

import pytest

from myspellchecker.data_pipeline.enrichment import (
    Collocation,
    ConfusablePair,
    EnrichmentConfig,
    EnrichmentReport,
    RegisterTag,
    _cosine_similarity,
    generate_typed_variants,
    insert_collocations,
    insert_confusable_pairs,
    insert_register_tags,
    mine_broken_compounds,
    mine_collocations,
    mine_confusable_pairs,
    mine_register_tags,
    run_enrichment,
)
from myspellchecker.data_pipeline.schema_manager import SchemaManager


@pytest.fixture
def enrichment_db(tmp_path):
    """Create a test database with words and bigrams for enrichment testing."""
    db_path = str(tmp_path / "test_enrichment.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Create full schema
    sm = SchemaManager(conn, cursor)
    sm.create_schema()

    # Insert test words (Myanmar aspiration pair: ခေါင်း/ကောင်း)
    test_words = [
        ("ခေါင်း", 5000),  # head (aspirated)
        ("ကောင်း", 8000),  # good (unaspirated)
        ("သူ", 20000),  # he/she
        ("မြန်မာ", 10000),  # Myanmar
        ("စာ", 15000),  # letter/study
        ("ကျောင်း", 12000),  # school
        ("သား", 7000),  # son
        ("ကျောင်းသား", 6000),  # student (compound)
        ("သည်", 50000),  # formal SFP
        ("တယ်", 45000),  # informal SFP
        ("ငါ", 3000),  # I (informal)
        ("ကျွန်တော်", 4000),  # I (formal)
    ]

    for word, freq in test_words:
        cursor.execute(
            "INSERT INTO words (word, frequency) VALUES (?, ?)",
            (word, freq),
        )

    # Build word ID lookup
    cursor.execute("SELECT id, word FROM words")
    word_ids = {row["word"]: row["id"] for row in cursor.fetchall()}

    # Insert bigrams
    test_bigrams = [
        ("သူ", "ကောင်း", 0.1, 500),  # "he" + "good" — common
        ("သူ", "ခေါင်း", 0.02, 100),  # "he" + "head" — less common
        ("စာ", "ကောင်း", 0.05, 200),  # "letter" + "good"
        ("ခေါင်း", "ကောင်း", 0.01, 30),  # "head" + "good" — rare
        ("ကျောင်း", "သား", 0.15, 800),  # school + son (compound parts)
        ("သူ", "သည်", 0.08, 400),  # formal context
        ("ငါ", "တယ်", 0.12, 350),  # informal context
        ("ကျွန်တော်", "သည်", 0.06, 250),  # formal pronoun + formal SFP
    ]

    for w1, w2, prob, count in test_bigrams:
        id1 = word_ids.get(w1)
        id2 = word_ids.get(w2)
        if id1 and id2:
            cursor.execute(
                "INSERT INTO bigrams (word1_id, word2_id, probability, count) VALUES (?, ?, ?, ?)",
                (id1, id2, prob, count),
            )

    conn.commit()
    conn.close()
    return db_path


class TestGenerateTypedVariants:
    """Test that typed variant generation produces correct type labels."""

    def test_aspiration_variant(self):
        """ခေါင်း should generate ကေါင်း as aspiration variant."""
        variants = list(generate_typed_variants("ခေါင်း"))
        aspiration_variants = [(v, t) for v, t in variants if t == "aspiration"]
        variant_words = [v for v, _ in aspiration_variants]
        # ခ→က aspiration swap
        assert any("က" in v for v in variant_words)

    def test_medial_variant(self):
        """ကျောင်း should generate ကြောင်း as medial variant."""
        variants = list(generate_typed_variants("ကျောင်း"))
        medial_variants = [(v, t) for v, t in variants if t == "medial"]
        assert len(medial_variants) > 0

    def test_tone_variant(self):
        """Word with visarga should generate dot-below variant."""
        variants = list(generate_typed_variants("ကောင်း"))
        tone_variants = [(v, t) for v, t in variants if t == "tone"]
        assert len(tone_variants) > 0

    def test_no_self_variant(self):
        """Should never yield the original word as a variant."""
        word = "ခေါင်း"
        for variant, _ in generate_typed_variants(word):
            assert variant != word


class TestCosineSimilarity:
    """Test cosine similarity computation."""

    def test_identical_vectors(self):
        vec = {1: 10, 2: 20, 3: 30}
        assert _cosine_similarity(vec, vec) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        vec1 = {1: 10}
        vec2 = {2: 20}
        assert _cosine_similarity(vec1, vec2) == pytest.approx(0.0)

    def test_empty_vectors(self):
        assert _cosine_similarity({}, {1: 10}) == pytest.approx(0.0)
        assert _cosine_similarity({}, {}) == pytest.approx(0.0)

    def test_partial_overlap(self):
        vec1 = {1: 3, 2: 4}
        vec2 = {1: 4, 3: 3}
        # dot = 3*4 = 12, mag1 = 5, mag2 = 5
        expected = 12 / (5 * 5)
        assert _cosine_similarity(vec1, vec2) == pytest.approx(expected)


class TestMineConfusablePairs:
    """Test confusable pair mining."""

    def test_finds_aspiration_pairs(self, enrichment_db):
        conn = sqlite3.connect(enrichment_db)
        conn.row_factory = sqlite3.Row
        config = EnrichmentConfig(confusable_min_freq=1)
        pairs = mine_confusable_pairs(conn, config)
        conn.close()

        # Should find at least some pairs
        assert len(pairs) > 0

        # Check that all pairs have valid confusion types
        valid_types = {"aspiration", "medial", "nasal", "stop_coda", "tone", "vowel_length"}
        for pair in pairs:
            assert pair.confusion_type in valid_types

    def test_respects_min_frequency(self, enrichment_db):
        conn = sqlite3.connect(enrichment_db)
        conn.row_factory = sqlite3.Row
        config = EnrichmentConfig(confusable_min_freq=100000)  # Very high
        pairs = mine_confusable_pairs(conn, config)
        conn.close()
        assert len(pairs) == 0  # No words meet threshold

    def test_deduplicates_pairs(self, enrichment_db):
        conn = sqlite3.connect(enrichment_db)
        conn.row_factory = sqlite3.Row
        config = EnrichmentConfig(confusable_min_freq=1)
        pairs = mine_confusable_pairs(conn, config)
        conn.close()

        # Check no duplicate (w1, w2, type) combos
        seen = set()
        for p in pairs:
            key = (p.word1, p.word2, p.confusion_type)
            assert key not in seen, f"Duplicate pair: {key}"
            seen.add(key)


class TestMineBrokenCompounds:
    """Test broken compound mining."""

    def test_finds_compounds(self, enrichment_db):
        conn = sqlite3.connect(enrichment_db)
        conn.row_factory = sqlite3.Row
        config = EnrichmentConfig(
            compound_min_freq=1,
            compound_min_split_count=1,
            compound_min_pmi=0.0,
        )
        compounds = mine_broken_compounds(conn, config)
        conn.close()

        # Should find ကျောင်းသား from bigram (ကျောင်း, သား)
        compound_words = [c.compound for c in compounds]
        assert "ကျောင်းသား" in compound_words


class TestMineCollocations:
    """Test collocation mining."""

    def test_finds_collocations(self, enrichment_db):
        conn = sqlite3.connect(enrichment_db)
        conn.row_factory = sqlite3.Row
        config = EnrichmentConfig(collocation_min_count=1, collocation_min_pmi=0.0)
        collocations = mine_collocations(conn, config)
        conn.close()

        assert len(collocations) > 0
        for c in collocations:
            assert c.pmi >= 0.0
            assert c.count > 0

    def test_respects_min_pmi(self, enrichment_db):
        conn = sqlite3.connect(enrichment_db)
        conn.row_factory = sqlite3.Row
        config = EnrichmentConfig(collocation_min_count=1, collocation_min_pmi=100.0)
        collocations = mine_collocations(conn, config)
        conn.close()
        assert len(collocations) == 0


class TestMineRegisterTags:
    """Test register tag mining."""

    def test_finds_register_tags(self, enrichment_db):
        conn = sqlite3.connect(enrichment_db)
        conn.row_factory = sqlite3.Row
        config = EnrichmentConfig(register_min_total=1, register_threshold=0.1)
        tags = mine_register_tags(conn, config)
        conn.close()

        assert len(tags) > 0
        valid_registers = {"formal", "informal", "neutral"}
        for tag in tags:
            assert tag.register in valid_registers


class TestInsertions:
    """Test DB insertion functions."""

    def test_insert_confusable_pairs(self, enrichment_db):
        conn = sqlite3.connect(enrichment_db)
        pairs = [
            ConfusablePair("word1", "word2", "aspiration", 0.5, 2.0),
            ConfusablePair("word3", "word4", "medial", 0.3, 1.5),
        ]
        count = insert_confusable_pairs(conn, pairs)
        assert count == 2

        # Verify data
        rows = conn.execute("SELECT * FROM confusable_pairs").fetchall()
        assert len(rows) == 2
        conn.close()

    def test_insert_collocations(self, enrichment_db):
        conn = sqlite3.connect(enrichment_db)
        collocations = [
            Collocation("w1", "w2", 5.0, 0.7, 100),
        ]
        count = insert_collocations(conn, collocations)
        assert count >= 1
        conn.close()

    def test_insert_register_tags(self, enrichment_db):
        conn = sqlite3.connect(enrichment_db)
        tags = [
            RegisterTag("word1", "formal", 0.8, 100, 10),
            RegisterTag("word2", "informal", 0.7, 10, 90),
        ]
        count = insert_register_tags(conn, tags)
        assert count >= 1
        conn.close()

    def test_insert_empty_list(self, enrichment_db):
        conn = sqlite3.connect(enrichment_db)
        assert insert_confusable_pairs(conn, []) == 0
        assert insert_collocations(conn, []) == 0
        assert insert_register_tags(conn, []) == 0
        conn.close()


class TestRunEnrichment:
    """Test the full enrichment orchestrator."""

    def test_full_enrichment(self, enrichment_db):
        config = EnrichmentConfig(
            confusable_min_freq=1,
            compound_min_freq=1,
            compound_min_split_count=1,
            compound_min_pmi=0.0,
            collocation_min_count=1,
            collocation_min_pmi=0.0,
            register_min_total=1,
            register_threshold=0.1,
        )
        report = run_enrichment(enrichment_db, config)

        assert isinstance(report, EnrichmentReport)
        assert report.confusable_pairs >= 0
        assert report.collocations >= 0
        assert report.elapsed_seconds > 0
        assert len(report.errors) == 0

    def test_enrichment_disabled(self, enrichment_db):
        config = EnrichmentConfig(
            enrich_confusables=False,
            enrich_compounds=False,
            enrich_collocations=False,
            enrich_register=False,
        )
        report = run_enrichment(enrichment_db, config)
        assert report.confusable_pairs == 0
        assert report.compound_confusions == 0
        assert report.collocations == 0
        assert report.register_tags == 0

    def test_enrichment_idempotent(self, enrichment_db):
        """Running enrichment twice should not duplicate data (INSERT OR IGNORE)."""
        config = EnrichmentConfig(
            confusable_min_freq=1,
            collocation_min_count=1,
            collocation_min_pmi=0.0,
        )
        run_enrichment(enrichment_db, config)  # First run populates
        report2 = run_enrichment(enrichment_db, config)

        # Second run should insert 0 (all already exist)
        assert report2.confusable_pairs == 0
        assert report2.collocations == 0
