import json
import shutil

import pytest

# Subprocess-based tests cannot use in-process mocks for resource downloads.
# Skip them when HuggingFace segmentation resources are not available locally.
_RESOURCES_AVAILABLE = False
try:
    from myspellchecker.tokenizers.resource_loader import _get_bundled_resource_path

    _RESOURCES_AVAILABLE = _get_bundled_resource_path("segmentation") is not None
except Exception:
    pass

_skip_no_resources = pytest.mark.skipif(
    not _RESOURCES_AVAILABLE,
    reason="Requires locally bundled HuggingFace segmentation resources",
)

# Subprocess-based build tests cannot disable the 50GB disk space pre-flight
# check. Skip when insufficient disk space is available.
_disk = shutil.disk_usage("/")
_ENOUGH_DISK = (_disk.free / (1024 * 1024)) >= 51200

_skip_low_disk = pytest.mark.skipif(
    not _ENOUGH_DISK,
    reason=f"Requires >=50GB free disk (have {_disk.free / (1024**3):.1f}GB)",
)


def test_help(run_myspell):
    res = run_myspell(["--help"])
    assert res.returncode == 0
    assert "mySpellChecker CLI" in res.stdout


@_skip_no_resources
@_skip_low_disk
def test_build_and_check_custom_db(tmp_path, run_myspell):
    """Test the full loop: Build DB -> Check against DB."""
    # 1. Create corpus
    # "မြန်မာ" repeated 50 times (default min_frequency is 50)
    corpus_content = "မြန်မာ\n" * 50
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(corpus_content, encoding="utf-8")

    database_path = tmp_path / "test.db"

    # 2. Build (use tmp_path as work-dir to avoid stale intermediate files,
    # and --min-frequency 1 to capture all words from the tiny corpus)
    work_dir = tmp_path / "work"
    res_build = run_myspell(
        [
            "build",
            "--input",
            str(corpus),
            "--output",
            str(database_path),
            "--work-dir",
            str(work_dir),
            "--min-frequency",
            "1",
        ]
    )
    assert res_build.returncode == 0, f"Build failed: {res_build.stderr}"
    assert database_path.exists()

    # 3. Check using custom DB
    # "မြန်မာ" should be valid (in DB) — use word-level check since the tiny
    # corpus produces low-frequency syllables that trigger syllable-level rules.
    res_valid = run_myspell(
        ["check", "--db", str(database_path), "--level", "word"], input_text="မြန်မာ"
    )
    assert res_valid.returncode == 0
    data_valid = json.loads(res_valid.stdout)
    assert data_valid["summary"]["total_errors"] == 0

    # "ရန်ကုန်" (Yangon) is NOT in our tiny DB.
    # It should be flagged as 'unknown_word' if we check at word level?
    # Or just handled by syllable rules?
    # If we use "--level word", and provider has only "Myanmar", "Yangon" might be valid syllable
    # but unknown word.
    res_unknown = run_myspell(
        ["check", "--db", str(database_path), "--level", "word"],
        input_text="ရန်ကုန်",
    )

    # It might NOT be an error if syllable validity passes.
    # But if we expect word validity, it might fail if not in dictionary.
    # The current SpellChecker logic:
    # Check Syllable -> Check Word.
    # If Syllable Valid, check Word in DB. If not in DB -> Unknown Word Error?
    # Let's check the output to see what happens.
    # For now, just asserting the command ran successfully.
    assert res_unknown.returncode == 0


@_skip_no_resources
@_skip_low_disk
def test_check_text_format(tmp_path, run_myspell):
    # 1. Build a tiny dummy DB to avoid loading the 1.7GB default DB
    corpus = tmp_path / "dummy_corpus.txt"
    corpus.write_text("မြန်မာ\n" * 5, encoding="utf-8")
    database_path = tmp_path / "dummy.db"

    work_dir = tmp_path / "work"
    res_build = run_myspell(
        [
            "build",
            "--input",
            str(corpus),
            "--output",
            str(database_path),
            "--work-dir",
            str(work_dir),
            "--min-frequency",
            "1",
        ]
    )
    assert res_build.returncode == 0

    # 2. Run check with custom DB
    p = tmp_path / "test.txt"
    p.write_text("ကး်", encoding="utf-8")  # Invalid

    res = run_myspell(["check", "-f", "text", str(p), "--db", str(database_path)])
    assert res.returncode == 0
    assert "errors found" in res.stdout
    assert "invalid_syllable" in res.stdout


@_skip_no_resources
def test_missing_file(run_myspell, e2e_test_db):
    # Provide explicit DB to avoid loading default large DB
    res = run_myspell(["check", "non_existent_file.txt", "--db", e2e_test_db])
    # CLI manually handles file opening and exits with 2 for file not found
    assert res.returncode == 2
    assert "File not found" in res.stderr
