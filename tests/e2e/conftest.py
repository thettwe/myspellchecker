import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from myspellchecker.data_pipeline.config import PipelineConfig
from myspellchecker.data_pipeline.pipeline import Pipeline

_THIS_DIR = str(Path(__file__).parent)


def pytest_collection_modifyitems(items):
    """Auto-apply e2e marker to all tests in this directory."""
    for item in items:
        if str(item.fspath).startswith(_THIS_DIR):
            item.add_marker(pytest.mark.e2e)


# Locate myspell once
_MYSPELL_CMD = None


def get_myspell_cmd():
    global _MYSPELL_CMD
    if _MYSPELL_CMD:
        return _MYSPELL_CMD

    cmd = shutil.which("myspellchecker")
    if not cmd:
        possible_path = Path(sys.executable).parent / "myspellchecker"
        if possible_path.exists():
            cmd = [str(possible_path)]
        else:
            cmd = [sys.executable, "-m", "myspellchecker.cli"]
    else:
        cmd = [cmd]

    _MYSPELL_CMD = cmd
    return cmd


@pytest.fixture(scope="session")
def myspell_cmd():
    return get_myspell_cmd()


@pytest.fixture(scope="session")
def e2e_test_db(tmp_path_factory):
    """
    Creates a small, persistent dummy database for E2E CLI tests.
    Returns the path as a string.
    """
    work_dir = tmp_path_factory.mktemp("e2e_db_build")
    database_path = work_dir / "e2e_test.db"
    corpus_path = work_dir / "corpus.txt"

    # Create a tiny corpus
    # Repeated enough to pass default frequency filters if any
    content = "မြန်မာ\n" * 10 + "မင်္ဂလာပါ\n" * 10
    corpus_path.write_text(content, encoding="utf-8")

    # Build it (disable disk space check — test corpus is tiny)
    config = PipelineConfig(work_dir=str(work_dir), disk_space_check_mb=0)
    pipeline = Pipeline(config=config)
    pipeline.build_database(
        input_files=[corpus_path], database_path=database_path, min_frequency=1, sample=True
    )

    return str(database_path)


@pytest.fixture
def run_myspell(myspell_cmd):
    def _run(args, input_text=None):
        cmd = myspell_cmd + args
        # If input_text is None, use DEVNULL to prevent hanging on stdin read.
        # If input_text is provided, subprocess handles it via the 'input' arg.
        stdin_arg = subprocess.DEVNULL if input_text is None else None

        return subprocess.run(
            cmd, input=input_text, stdin=stdin_arg, capture_output=True, text=True, encoding="utf-8"
        )

    return _run
