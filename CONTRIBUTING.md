# Contributing to mySpellChecker

Thank you for your interest in contributing to mySpellChecker! We welcome contributions from the community to help improve Myanmar spell checking.

## Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/thettwe/myspellchecker.git
   cd myspellchecker
   ```

2. **Create a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   # Install in editable mode with dev dependencies
   pip install -e ".[dev]"
   ```

## Code Quality Standards

We enforce strict code quality standards to ensure reliability and maintainability. Please run the following commands before submitting a Pull Request.

### Formatting & Linting
We use `ruff` for both code formatting and linting, and `mypy` for static type checking.

```bash
# Format code
ruff format .

# Lint code
ruff check .

# Type check
mypy src/myspellchecker
```

### Testing
We use `pytest` for testing. All new features must include tests.

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=myspellchecker
```

## Dictionary Data

The dictionary database is managed via the `myspellchecker` CLI. The data pipeline logic resides in `src/myspellchecker/data_pipeline/`.

To build the dictionary from a corpus:

1. Prepare your corpus file (e.g., `corpus.txt`).
2. Run the build command:
   ```bash
   myspellchecker build --input corpus.txt --output mySpellChecker.db
   ```

For testing purposes, you can generate a sample dictionary:
```bash
myspellchecker build --sample
```

## Submitting a Pull Request

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/amazing-feature`).
3. Commit your changes (`git commit -m 'Add amazing feature'`).
4. Push to the branch (`git push origin feature/amazing-feature`).
5. Open a Pull Request.

Please ensure all tests pass and code quality checks are green!
