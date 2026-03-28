# Tests

## Directory Structure

```
tests/
├── conftest.py              # Root fixtures + skipped test registry
├── test_*.py                # Unit tests (~170 files)
├── integration/             # Cross-component integration tests (~25 files)
├── e2e/                     # End-to-end CLI and library usage tests
│   ├── conftest.py          # E2E fixtures (myspell_cmd, e2e_test_db, run_myspell)
│   └── test_*.py            # CLI E2E, library usage, custom providers, robustness
├── stress/                  # Concurrency, batch processing, compound stress tests
├── fixtures/                # Shared test data
│   ├── myanmar_test_samples.py  # Ground-truth segmentations for Myanmar text
│   ├── config_templates.py      # YAML/JSON config templates for config tests
│   └── benchmarks/              # POS gold standard data
└── benchmarks/              # Benchmark suite (separate from tests/; lives at repo root)
```

## Running Tests

```bash
# All tests
pytest tests/

# Single file
pytest tests/test_syllable_rules.py

# Single test method
pytest tests/test_syllable_rules.py::TestSyllableValidator::test_valid_syllable

# By marker
pytest tests/ -m unit
pytest tests/ -m integration
pytest tests/ -m "not slow"

# With coverage
pytest tests/ --cov=src/myspellchecker --cov-report=term-missing

# Pattern matching
pytest tests/ -k "context"

# Verbose
pytest tests/ -v
```

## Markers

Defined in `pyproject.toml` under `[tool.pytest.ini_options]`:

| Marker | Purpose | When to use |
|--------|---------|-------------|
| `unit` | Fast, isolated single-function tests | Default for most tests |
| `integration` | Cross-component interaction tests | Tests that wire multiple real components |
| `slow` | Tests that take >5s | Large corpus processing, full pipeline runs |
| `e2e` | End-to-end tests via CLI or public API | Tests that invoke `myspellchecker` as a user would |
| `benchmark` | Performance measurement tests | Requires `pytest-benchmark` installed |

Usage: `@pytest.mark.unit` on test class or method.

## Key Fixtures (conftest.py)

### Session-scoped (run once per test session)

| Fixture | Auto | Purpose |
|---------|------|---------|
| `mock_resource_downloads_session` | Yes | Prevents HuggingFace downloads; stubs `WordTokenizer` with `RegexSegmenter` fallback |
| `patch_default_db` | Yes | Patches `SQLiteProvider.__init__` to inject a test DB when no path is given |
| `test_database_path` | No | Creates a temporary SQLite DB with sample syllables, words, and bigrams |

### Per-test (run before each test)

| Fixture | Auto | Purpose |
|---------|------|---------|
| `reset_grammar_config_singleton` | Yes | Clears `GrammarRuleConfig` singleton to prevent cross-test pollution |
| `reset_rich_console` | Yes | Redirects Rich console output to `StringIO` |
| `reset_singletons` | Yes | Clears all singleton instances (`clear_all_singletons`) |
| `clear_lru_caches` | Yes | Clears known `@lru_cache` functions (segmenter, logger) |
| `reset_logging` | Yes | Restores `myspellchecker` logger handlers/propagation after each test |
| `mock_console` | No | Returns a `MagicMock` PipelineConsole (use explicitly when needed) |

### E2E fixtures (tests/e2e/conftest.py)

| Fixture | Scope | Purpose |
|---------|-------|---------|
| `myspell_cmd` | Session | Resolves the `myspellchecker` CLI command path |
| `e2e_test_db` | Session | Builds a small test DB from a tiny corpus via the pipeline |
| `run_myspell` | Function | Helper to invoke CLI with args and optional stdin input |

## Skipped Test Registry

The top of `conftest.py` documents every skipped and xfailed test, organized by category:

1. **Cython extension not compiled** -- Needs `python setup.py build_ext --inplace`
2. **Optional dependency not installed** -- `transformers`, `torch`, `jsonschema`, `pytest-benchmark`, `pyyaml`
3. **Transformers mocked by another module** -- Test ordering issue
4. **SpellChecker not available** -- Needs a built DB or active `patch_default_db`
5. **Production DB not available** -- Needs `data/mySpellChecker_production.db`
6. **xfail: known limitations** -- Pali dictionary entries not yet added
7. **Data-dependent** -- Conditional on runtime data (e.g., `SKIPPED_CONTEXT_WORDS`)
8. **OS-dependent** -- Environment-specific guards (e.g., symlink loops)

## Writing New Tests

### Naming

- File: `test_<module_or_feature>.py`
- Class: `Test<ComponentName>` (optional, group related tests)
- Method: `test_<behavior_under_test>`

### Markers

Always mark tests appropriately:

```python
import pytest

@pytest.mark.unit
class TestMyFeature:
    def test_basic_case(self):
        ...

    @pytest.mark.slow
    def test_large_corpus(self):
        ...
```

### Using fixtures

The auto-use fixtures handle singleton/cache cleanup automatically. For tests needing a database:

```python
def test_with_db(test_database_path):
    from myspellchecker.providers.sqlite import SQLiteProvider
    provider = SQLiteProvider(database_path=str(test_database_path))
    ...
```

### Parametrize patterns

Use `@pytest.mark.parametrize` for Myanmar text validation:

```python
@pytest.mark.parametrize("text,expected", [
    ("မြန်မာ", True),
    ("xyz", False),
])
def test_validation(text, expected):
    assert validate(text) == expected
```

### Mocking heavy dependencies

For ONNX/semantic tests, mock the inference backend:

```python
from unittest.mock import MagicMock, patch

def test_semantic_checker():
    mock_backend = MagicMock()
    mock_backend.predict.return_value = [0.95]
    with patch("myspellchecker.algorithms.semantic_checker.InferenceBackend", return_value=mock_backend):
        ...
```

### Coverage requirement

Maintain >=75% code coverage. Check with:

```bash
pytest tests/ --cov=src/myspellchecker --cov-fail-under=75
```
