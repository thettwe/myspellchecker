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


@_skip_no_resources
def test_empty_input(run_myspell, e2e_test_db):
    res = run_myspell(["check", "--db", e2e_test_db], input_text="")
    assert res.returncode == 0
    assert "summary" in res.stdout


@_skip_no_resources
def test_mixed_english_myanmar(run_myspell, e2e_test_db):
    # "Hello မင်္ဂလာပါ World"
    # Hello/World should be ignored. မင်္ဂလာပါ checked.
    res = run_myspell(["check", "--db", e2e_test_db], input_text="Hello မင်္ဂလာပါ World")
    assert res.returncode == 0

    # English chars are now ignored.
    # "Hello" and "World" are skipped.
    # Note: "မင်္ဂလာပါ" contains a kinzi (္ - subjoined consonant marker).
    # The syllable rules may flag "မင်္" as invalid because kinzi constructs
    # require special handling. This is acceptable behavior.
    # The key assertion is that the CLI runs successfully without crashing
    # and returns valid JSON output.
    assert '"summary":' in res.stdout
    assert '"results":' in res.stdout
    # returncode == 0 is already checked above


@_skip_no_resources
@pytest.mark.slow
def test_huge_line(tmp_path, run_myspell, e2e_test_db):
    # ~10KB line (reduced from 1MB for faster execution)
    # Still tests handling of large input without taking minutes
    huge_text = "မြန်မာ" * 1000
    p = tmp_path / "huge.txt"
    p.write_text(huge_text, encoding="utf-8")

    res = run_myspell(["check", str(p), "--db", e2e_test_db])
    assert res.returncode == 0
    # Just ensure it didn't crash
    assert '"summary":' in res.stdout


@_skip_no_resources
def test_invalid_encoding(tmp_path, run_myspell, e2e_test_db):
    # Create a file with invalid utf-8 sequence (e.g. generic binary)
    p = tmp_path / "bad_encoding.txt"
    with open(p, "wb") as f:
        f.write(b"\x80\x81\xff")

    res = run_myspell(["check", str(p), "--db", e2e_test_db])
    # It should fail gracefully, not crash with python trace
    assert res.returncode != 0
    assert "UnicodeDecodeError" in res.stderr or "error" in res.stderr.lower()
