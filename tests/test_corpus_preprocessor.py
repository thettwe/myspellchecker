"""Tests for the corpus preprocessor module."""

from myspellchecker.training.corpus_preprocessor import (
    CorpusPreprocessor,
    PreprocessingConfig,
    PreprocessingReport,
    _get_chunk_boundaries,
)


class TestPreprocessingConfig:
    """Tests for PreprocessingConfig defaults and customization."""

    def test_defaults(self):
        config = PreprocessingConfig()
        assert config.zawgyi_threshold == 0.7
        assert config.zawgyi_ambiguous_low == 0.3
        assert config.min_myanmar_ratio == 0.6
        assert config.min_line_length == 10
        assert config.max_line_length == 1000
        assert config.enable_dedup is True

    def test_custom_values(self):
        config = PreprocessingConfig(
            zawgyi_threshold=0.8,
            min_myanmar_ratio=0.5,
            min_line_length=5,
            max_line_length=500,
            enable_dedup=False,
        )
        assert config.zawgyi_threshold == 0.8
        assert config.min_myanmar_ratio == 0.5
        assert config.min_line_length == 5
        assert config.max_line_length == 500
        assert config.enable_dedup is False


class TestPreprocessingReport:
    """Tests for PreprocessingReport."""

    def test_defaults(self):
        report = PreprocessingReport()
        assert report.total == 0
        assert report.kept == 0
        assert report.zawgyi_converted == 0
        assert report.duplicates == 0

    def test_summary_empty(self):
        report = PreprocessingReport()
        summary = report.summary()
        assert "0 lines" in summary

    def test_summary_with_data(self):
        report = PreprocessingReport(total=100, kept=80, duplicates=10, filtered_empty=5)
        summary = report.summary()
        assert "100" in summary
        assert "80" in summary
        assert "80.0%" in summary


class TestCorpusPreprocessor:
    """Tests for CorpusPreprocessor."""

    def test_empty_line_filtered(self):
        preprocessor = CorpusPreprocessor()
        report = PreprocessingReport()
        result = preprocessor.process_line("", report)
        assert result is None
        assert report.filtered_empty == 1

    def test_whitespace_only_filtered(self):
        preprocessor = CorpusPreprocessor()
        report = PreprocessingReport()
        result = preprocessor.process_line("   \t  \n", report)
        assert result is None
        assert report.filtered_empty == 1

    def test_short_line_filtered(self):
        config = PreprocessingConfig(min_line_length=10)
        preprocessor = CorpusPreprocessor(config)
        report = PreprocessingReport()
        # Short Myanmar text (< 10 chars)
        result = preprocessor.process_line("ကောင်း", report)
        assert result is None
        assert report.filtered_length == 1

    def test_long_line_filtered(self):
        config = PreprocessingConfig(max_line_length=20)
        preprocessor = CorpusPreprocessor(config)
        report = PreprocessingReport()
        # Long Myanmar text
        long_text = "မြန်မာနိုင်ငံသည် အရှေ့တောင်အာရှတွင် တည်ရှိသည်။"
        result = preprocessor.process_line(long_text, report)
        assert result is None
        assert report.filtered_length == 1

    def test_low_myanmar_ratio_filtered(self):
        config = PreprocessingConfig(min_myanmar_ratio=0.6, min_line_length=3)
        preprocessor = CorpusPreprocessor(config)
        report = PreprocessingReport()
        # Mostly English text
        result = preprocessor.process_line("Hello World this is English text with ကော", report)
        assert result is None
        assert report.filtered_myanmar_ratio == 1

    def test_valid_myanmar_line_kept(self):
        config = PreprocessingConfig(min_line_length=5, enable_dedup=False)
        preprocessor = CorpusPreprocessor(config)
        report = PreprocessingReport()
        text = "မြန်မာနိုင်ငံသည် အရှေ့တောင်အာရှတွင် တည်ရှိသည်။"
        result = preprocessor.process_line(text, report)
        assert result is not None
        assert len(result) > 0

    def test_dedup_filters_duplicates(self):
        config = PreprocessingConfig(min_line_length=5, enable_dedup=True)
        preprocessor = CorpusPreprocessor(config)
        report = PreprocessingReport()
        text = "မြန်မာနိုင်ငံသည် အရှေ့တောင်အာရှတွင် တည်ရှိသည်။"

        # First time: kept
        result1 = preprocessor.process_line(text, report)
        assert result1 is not None

        # Second time: duplicate
        result2 = preprocessor.process_line(text, report)
        assert result2 is None
        assert report.duplicates == 1

    def test_dedup_disabled(self):
        config = PreprocessingConfig(min_line_length=5, enable_dedup=False)
        preprocessor = CorpusPreprocessor(config)
        report = PreprocessingReport()
        text = "မြန်မာနိုင်ငံသည် အရှေ့တောင်အာရှတွင် တည်ရှိသည်။"

        result1 = preprocessor.process_line(text, report)
        result2 = preprocessor.process_line(text, report)
        assert result1 is not None
        assert result2 is not None
        assert report.duplicates == 0

    def test_formal_register_detected(self):
        config = PreprocessingConfig(min_line_length=5, enable_dedup=False)
        preprocessor = CorpusPreprocessor(config)
        report = PreprocessingReport()
        text = "မြန်မာနိုင်ငံသည် အရှေ့တောင်အာရှတွင် တည်ရှိပါသည်။"
        result = preprocessor.process_line(text, report)
        assert result is not None
        assert report.formal_count == 1

    def test_colloquial_register_detected(self):
        config = PreprocessingConfig(min_line_length=5, enable_dedup=False)
        preprocessor = CorpusPreprocessor(config)
        report = PreprocessingReport()
        text = "ဒီစာအုပ်က တကယ်ကောင်းတယ်။ ဖတ်ကြည့်ပါတယ်။"
        result = preprocessor.process_line(text, report)
        assert result is not None
        assert report.colloquial_count == 1

    def test_normalization_applied(self):
        """Verify normalization removes zero-width characters."""
        config = PreprocessingConfig(min_line_length=5, enable_dedup=False)
        preprocessor = CorpusPreprocessor(config)
        report = PreprocessingReport()
        # Text with zero-width space (U+200B) inserted
        text = "မြန်မာ\u200bနိုင်ငံသည် အရှေ့တောင်အာရှတွင် တည်ရှိသည်။"
        result = preprocessor.process_line(text, report)
        assert result is not None
        assert "\u200b" not in result

    def test_process_file(self, tmp_path):
        """Test full file processing."""
        input_file = tmp_path / "input.txt"
        output_file = tmp_path / "output.txt"

        lines = [
            "မြန်မာနိုင်ငံသည် အရှေ့တောင်အာရှတွင် တည်ရှိသည်။\n",
            "\n",  # Empty
            "မြန်မာနိုင်ငံသည် အရှေ့တောင်အာရှတွင် တည်ရှိသည်။\n",  # Duplicate
            "Hello World English only text here\n",  # Low Myanmar ratio
            "ရန်ကုန်မြို့သည် မြန်မာနိုင်ငံ၏ အကြီးဆုံးမြို့ ဖြစ်ပါသည်။\n",
        ]
        input_file.write_text("".join(lines), encoding="utf-8")

        config = PreprocessingConfig(min_line_length=5, enable_dedup=True)
        preprocessor = CorpusPreprocessor(config)
        report = preprocessor.process_file(input_file, output_file)

        assert report.total == 5
        assert report.kept == 2  # First Myanmar line + Yangon line
        assert report.filtered_empty == 1
        assert report.duplicates == 1

        # Verify output file
        output_lines = output_file.read_text(encoding="utf-8").strip().split("\n")
        assert len(output_lines) == 2

    def test_process_file_creates_output_dir(self, tmp_path):
        """Test that process_file creates output directory if needed."""
        input_file = tmp_path / "input.txt"
        output_file = tmp_path / "subdir" / "nested" / "output.txt"

        input_file.write_text(
            "မြန်မာနိုင်ငံသည် အရှေ့တောင်အာရှတွင် တည်ရှိသည်။\n",
            encoding="utf-8",
        )

        config = PreprocessingConfig(min_line_length=5)
        preprocessor = CorpusPreprocessor(config)
        report = preprocessor.process_file(input_file, output_file)

        assert output_file.exists()
        assert report.kept == 1

    def test_lazy_import_from_training(self):
        """Test that classes can be imported from training package."""
        from myspellchecker.training import (
            CorpusPreprocessor,
            PreprocessingConfig,
            PreprocessingReport,
        )

        assert CorpusPreprocessor is not None
        assert PreprocessingConfig is not None
        assert PreprocessingReport is not None


class TestPreprocessingReportMerge:
    """Tests for PreprocessingReport.merge()."""

    def test_merge_adds_counters(self):
        a = PreprocessingReport(total=100, kept=80, duplicates=10, filtered_empty=5)
        b = PreprocessingReport(total=50, kept=40, duplicates=3, filtered_empty=2)
        a.merge(b)
        assert a.total == 150
        assert a.kept == 120
        assert a.duplicates == 13
        assert a.filtered_empty == 7

    def test_merge_with_zero_report(self):
        a = PreprocessingReport(total=100, kept=80)
        b = PreprocessingReport()
        a.merge(b)
        assert a.total == 100
        assert a.kept == 80


class TestChunkBoundaries:
    """Tests for _get_chunk_boundaries()."""

    def test_single_chunk(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("line1\nline2\nline3\n")
        boundaries = _get_chunk_boundaries(f, 1)
        assert boundaries == [0, f.stat().st_size]

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("")
        boundaries = _get_chunk_boundaries(f, 4)
        assert boundaries == [0, 0]

    def test_boundaries_align_to_line_ends(self, tmp_path):
        """Each boundary must fall on a line boundary (after \\n)."""
        content = "line one\nline two\nline three\nline four\nline five\n"
        f = tmp_path / "test.txt"
        f.write_text(content)
        boundaries = _get_chunk_boundaries(f, 3)

        # All interior boundaries should be right after a newline
        content_bytes = content.encode("utf-8")
        for b in boundaries[1:-1]:
            # The byte just before the boundary should be \n
            assert content_bytes[b - 1 : b] == b"\n"

    def test_boundary_count(self, tmp_path):
        """Should produce at most num_chunks + 1 boundaries."""
        lines = [f"{'x' * 50}\n" for _ in range(100)]
        f = tmp_path / "test.txt"
        f.write_text("".join(lines))
        boundaries = _get_chunk_boundaries(f, 4)
        # num_chunks + 1 boundaries (at most), first=0, last=file_size
        assert boundaries[0] == 0
        assert boundaries[-1] == f.stat().st_size
        assert len(boundaries) <= 5

    def test_more_chunks_than_lines(self, tmp_path):
        """Requesting more chunks than lines should still work."""
        f = tmp_path / "small.txt"
        f.write_text("ab\ncd\n")
        boundaries = _get_chunk_boundaries(f, 10)
        assert boundaries[0] == 0
        assert boundaries[-1] == f.stat().st_size
        # Should gracefully reduce chunk count
        assert len(boundaries) >= 2


class TestParallelProcessing:
    """Tests for CorpusPreprocessor.process_file_parallel()."""

    def _make_corpus(self, tmp_path, num_lines=20):
        """Create a test corpus with Myanmar lines, empty, English, and duplicates."""
        myanmar_lines = [
            "မြန်မာနိုင်ငံသည် အရှေ့တောင်အာရှတွင် တည်ရှိသည်။",
            "ရန်ကုန်မြို့သည် မြန်မာနိုင်ငံ၏ အကြီးဆုံးမြို့ ဖြစ်ပါသည်။",
            "မန္တလေးမြို့သည် မြန်မာနိုင်ငံ၏ ဒုတိယအကြီးဆုံးမြို့ ဖြစ်သည်။",
            "ပညာရေးသည် တိုင်းပြည်ဖွံ့ဖြိုးတိုးတက်ရေးအတွက် အရေးကြီးပါသည်။",
            "ကျန်းမာရေးသည် လူတိုင်းအတွက် အဓိကကျပါသည်။",
        ]
        lines = []
        for i in range(num_lines):
            lines.append(myanmar_lines[i % len(myanmar_lines)])
        # Add some lines that should be filtered
        lines.append("")  # empty
        lines.append("Hello World this is purely English text nothing Myanmar here")
        lines.append(myanmar_lines[0])  # duplicate of first line

        input_file = tmp_path / "corpus.txt"
        input_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return input_file

    def test_parallel_matches_sequential(self, tmp_path):
        """Parallel and sequential should produce identical output."""
        input_file = self._make_corpus(tmp_path, num_lines=50)

        config = PreprocessingConfig(min_line_length=5, enable_dedup=True)

        # Sequential
        seq_output = tmp_path / "seq_output.txt"
        seq_preprocessor = CorpusPreprocessor(config)
        seq_report = seq_preprocessor.process_file(input_file, seq_output)

        # Parallel (force 2 workers, lower threshold)
        par_output = tmp_path / "par_output.txt"
        par_preprocessor = CorpusPreprocessor(config)
        par_report = par_preprocessor.process_file_parallel(input_file, par_output, num_workers=2)

        # Same total line count and kept count
        assert par_report.total == seq_report.total
        assert par_report.kept == seq_report.kept

        # Same output content (order preserved since chunks are sequential)
        seq_lines = set(seq_output.read_text(encoding="utf-8").strip().split("\n"))
        par_lines = set(par_output.read_text(encoding="utf-8").strip().split("\n"))
        assert seq_lines == par_lines

    def test_parallel_fallback_to_sequential_for_small_files(self, tmp_path):
        """Small files should fall back to process_file (no chunking)."""
        input_file = self._make_corpus(tmp_path, num_lines=5)

        config = PreprocessingConfig(min_line_length=5, enable_dedup=True)
        preprocessor = CorpusPreprocessor(config)
        report = preprocessor.process_file_parallel(
            input_file, tmp_path / "output.txt", num_workers=4
        )

        # Should still work — just used sequential path
        assert report.total > 0
        assert report.kept > 0

    def test_parallel_single_worker(self, tmp_path):
        """num_workers=1 should use sequential processing."""
        input_file = self._make_corpus(tmp_path, num_lines=10)

        config = PreprocessingConfig(min_line_length=5, enable_dedup=True)
        preprocessor = CorpusPreprocessor(config)
        report = preprocessor.process_file_parallel(
            input_file, tmp_path / "output.txt", num_workers=1
        )

        assert report.total > 0
        assert report.kept > 0

    def test_parallel_creates_output_dir(self, tmp_path):
        """Output directory should be created if it doesn't exist."""
        input_file = self._make_corpus(tmp_path, num_lines=10)
        output_file = tmp_path / "deep" / "nested" / "output.txt"

        config = PreprocessingConfig(min_line_length=5)
        preprocessor = CorpusPreprocessor(config)
        report = preprocessor.process_file_parallel(input_file, output_file, num_workers=2)

        assert output_file.exists()
        assert report.kept > 0

    def test_parallel_dedup_across_chunks(self, tmp_path):
        """Cross-chunk duplicates should be removed in the merge pass."""
        # Create a file where the same line appears in what will be different chunks
        line = "မြန်မာနိုင်ငံသည် အရှေ့တောင်အာရှတွင် တည်ရှိသည်။"
        # Repeat the same line many times so it spans multiple chunks
        content = "\n".join([line] * 100) + "\n"
        input_file = tmp_path / "dupes.txt"
        input_file.write_text(content, encoding="utf-8")

        config = PreprocessingConfig(min_line_length=5, enable_dedup=True)
        preprocessor = CorpusPreprocessor(config)
        report = preprocessor.process_file_parallel(
            input_file, tmp_path / "output.txt", num_workers=2
        )

        # Only 1 unique line should survive
        assert report.kept == 1
        assert report.duplicates == 99

    def test_parallel_no_dedup(self, tmp_path):
        """With dedup disabled, all valid lines should be kept."""
        line = "မြန်မာနိုင်ငံသည် အရှေ့တောင်အာရှတွင် တည်ရှိသည်။"
        content = "\n".join([line] * 20) + "\n"
        input_file = tmp_path / "no_dedup.txt"
        input_file.write_text(content, encoding="utf-8")

        config = PreprocessingConfig(min_line_length=5, enable_dedup=False)
        preprocessor = CorpusPreprocessor(config)
        report = preprocessor.process_file_parallel(
            input_file, tmp_path / "output.txt", num_workers=2
        )

        assert report.kept == 20
        assert report.duplicates == 0
