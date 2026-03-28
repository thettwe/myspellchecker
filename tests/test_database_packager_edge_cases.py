import pytest

from myspellchecker.data_pipeline.database_packager import DatabasePackager


class TestDatabasePackagerExtra:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        # Fix: database_path must be a file, not a directory
        self.database_path = tmp_path / "test.db"
        self.data_dir = tmp_path / "data"
        self.data_dir.mkdir()

        # Create minimal valid TSV files
        (self.data_dir / "syllable_frequencies.tsv").write_text(
            "syllable\tfrequency\nက\t10", encoding="utf-8"
        )
        (self.data_dir / "word_frequencies.tsv").write_text(
            "word\tsyllable_count\tfrequency\nက\t1\t10", encoding="utf-8"
        )
        (self.data_dir / "bigram_probabilities.tsv").write_text(
            "word1\tword2\tprobability\tcount\nက\tခ\t0.5\t5", encoding="utf-8"
        )

        # Note: DatabasePackager(input_dir, output_db)
        self.packager = DatabasePackager(self.data_dir, self.database_path)

    def test_optimize_database(self):
        self.packager.connect()
        self.packager.create_schema()
        self.packager.optimize_database()

    def test_verify_database(self):
        self.packager.connect()
        self.packager.create_schema()
        # verify_database() doesn't return anything, just prints stats
        # It should not raise any errors
        self.packager.verify_database()

    def test_load_trigrams_missing(self):
        self.packager.connect()
        self.packager.create_schema()
        # File doesn't exist, should handle gracefully
        self.packager.load_trigrams()

    def test_print_stats(self):
        self.packager.print_stats()
