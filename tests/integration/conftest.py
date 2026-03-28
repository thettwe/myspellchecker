"""Auto-apply integration marker to all tests in this directory."""

import pytest

_THIS_DIR = str(__import__("pathlib").Path(__file__).parent)


def pytest_collection_modifyitems(items):
    for item in items:
        if str(item.fspath).startswith(_THIS_DIR):
            item.add_marker(pytest.mark.integration)
