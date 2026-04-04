# mySpellChecker: Myanmar (Burmese) Text Intelligence Library

> **Myanmar (Burmese) text intelligence library — 12-strategy checking pipeline, dictionary building, and AI model training, from O(1) SymSpell lookups to ONNX-powered inference.**

[![PyPI](https://img.shields.io/pypi/v/myspellchecker)](https://pypi.org/project/myspellchecker/)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Coverage](https://img.shields.io/badge/coverage-74%25-green)](tests/)
[![Tests](https://img.shields.io/badge/tests-4%2C658_passed-brightgreen)](tests/)

## Overview

**mySpellChecker** is a comprehensive text intelligence library built specifically for the Myanmar language. It covers three domains: a **12-strategy checking pipeline** (from rule-based validation through grammar checking, N-gram context, confusable detection, homophone detection, to ONNX-powered AI inference), a **dictionary building pipeline** (corpus ingestion, segmentation, N-gram frequency, SQLite packaging), and **AI model training** (semantic MLM fine-tuning with ONNX export). Since Myanmar script is written as a continuous stream without spaces between words, the library uses a multi-layer validation approach — starting with fast syllable-level checks and progressively applying deeper analysis including POS tagging, 8 grammar checkers, and context-aware semantic validation.

## Key Features

> **Note**: v1.0 supports **Standard Burmese (Myanmar) only**. Other Myanmar-script languages (Shan, Karen, Mon, etc.) and extended Unicode ranges are planned for future releases.

### Checking Pipeline

*   **12-Strategy Validation Pipeline**: Composable strategies from fast rule checks (sub-10ms) to AI inference, each layer building on the previous.
*   **Syllable-First Architecture**: Validates most errors at the syllable level before assembling into words for deeper analysis.
*   **SymSpell Algorithm**: Custom O(1) symmetric delete implementation with Myanmar-specific variant generation for fast correction suggestions.
*   **N-gram Context Checking**: Bigram/Trigram probabilities detect real-word errors (correct spelling, wrong context).
*   **Homophone Detection**: Bidirectional N-gram analysis catches sound-alike word errors with frequency-aware guards.
*   **Confusable Detection**: Multi-layer valid-word confusion detection — statistical bigram, MLP classifier, and MLM semantic analysis.
*   **Grammar Checking**: 8 specialized checkers — Aspect, Classifier, Compound, MergedWord, Negation, Particle, TenseAgreement, Register.
*   **POS Tagging**: Pluggable backends — Rule-Based (fast), Viterbi HMM (balanced), Transformer (93% accuracy).
*   **Joint Segmentation**: Simultaneous word segmentation and POS tagging in a single pass.
*   **Compound & Morpheme Handling**: DP-based compound resolution, productive reduplication validation, and morpheme-level correction for OOV words.
*   **AI Semantic Checking (Optional)**: ONNX masked language model for context-aware validation.
*   **Named Entity Recognition**: Heuristic and Transformer-based NER to reduce false positives on names and places.

### Dictionary Building Pipeline

*   **Multi-Format Corpus Ingestion**: Build dictionaries from `.txt`, `.csv`, `.tsv`, `.json`, `.jsonl`, `.parquet` files.
*   **Incremental Builds**: Resume corpus processing without reprocessing completed files.
*   **Pluggable Storage**: SQLite (default, disk-based) or MemoryProvider (RAM-based) with thread-safe connection pooling.

### AI Model Training

*   **Semantic Model Training**: Train masked language models with word-boundary BPE, whole-word masking, and denoising objectives.
*   **ONNX Export & Quantization**: Convert trained models to ONNX with quantization for production deployment.

### Myanmar Language Support

*   **Text Normalization**: Unified service — zero-width character removal, NFC/NFD normalization, Zawgyi conversion.
*   **Zawgyi Detection**: Built-in detection and warning for legacy Zawgyi encoded text.
*   **Phonetic & Colloquial Handling**: Phonetic hashing, colloquial variant detection (e.g., ကျနော် → ကျွန်တော်), configurable strictness.
*   **Tone Processing**: Tone mark validation, disambiguation, and context-based correction.
*   **Bilingual Error Messages**: Error reporting in English and Myanmar (Burmese).

### Performance & Production

*   **Cython/C++ Extensions**: 11 performance-critical paths compiled to C++ with OpenMP parallelization.
*   **Streaming & Batch APIs**: Process large documents with streaming, batch (`check_batch`), and async (`check_async`) APIs.
*   **Configurable**: Pre-defined profiles (production, fast, accurate, development, testing), environment/file-based config loading, and DI container for advanced wiring.

## Documentation

Full documentation is available at **[docs.myspellchecker.com](https://docs.myspellchecker.com/)**.

### Getting Started
*   **[Introduction](https://docs.myspellchecker.com/introduction)**: Overview of the library and its architecture.
*   **[Installation](https://docs.myspellchecker.com/guides/installation)**: Installation options and system requirements.
*   **[Quick Start](https://docs.myspellchecker.com/guides/quickstart)**: Get up and running in 5 minutes.
*   **[Configuration Guide](https://docs.myspellchecker.com/guides/configuration)**: All configuration options and profiles.

### Text Checking
*   **[Overview](https://docs.myspellchecker.com/features/index)**: 12-strategy text checking pipeline.
*   **[Syllable Validation](https://docs.myspellchecker.com/features/syllable-validation)**: Core validation layer.
*   **[Word Validation](https://docs.myspellchecker.com/features/word-validation)**: Dictionary + SymSpell suggestions.
*   **[Context Checking](https://docs.myspellchecker.com/features/context-checking)**: N-gram probability analysis.
*   **[Confusable Detection](https://docs.myspellchecker.com/features/confusable-detection)**: Multi-layer confusable word detection.
*   **[Homophone Detection](https://docs.myspellchecker.com/features/homophones)**: Sound-alike error detection.

### Grammar & NER
*   **[Grammar Checking](https://docs.myspellchecker.com/features/grammar-checking)**: Syntactic validation.
*   **[Grammar Checkers](https://docs.myspellchecker.com/features/grammar-checkers)**: 8 specialized checkers.
*   **[Grammar Engine](https://docs.myspellchecker.com/features/grammar-engine)**: Rule engine internals.
*   **[Named Entity Recognition](https://docs.myspellchecker.com/features/ner)**: NER with 3 implementations + gazetteer.
*   **[Loan Word Variants](https://docs.myspellchecker.com/features/loan-words)**: Transliteration variant handling for English, Pali/Sanskrit loan words.

### Language Processing
*   **[POS Tagging](https://docs.myspellchecker.com/features/pos-tagging)**: Pluggable tagging (Rule-Based, Viterbi, Transformer).
*   **[Morphology Analysis](https://docs.myspellchecker.com/features/morphology)**: Word structure analysis.
*   **[Compound Resolution](https://docs.myspellchecker.com/features/compound-resolution)**: Compound word and reduplication validation.
*   **[Segmenters](https://docs.myspellchecker.com/features/segmenters)**: Word segmentation engines.

### AI-Powered Checking
*   **[Semantic Checking](https://docs.myspellchecker.com/features/semantic-checking)**: AI-powered MLM validation.
*   **[Validation Strategies](https://docs.myspellchecker.com/features/validation-strategies)**: 12 composable strategies.
*   **[Training Models](https://docs.myspellchecker.com/guides/training)**: Train custom semantic models.

### Text Utilities
*   **[Text Normalization](https://docs.myspellchecker.com/features/normalization)**: Unified normalization service.
*   **[Text Utilities](https://docs.myspellchecker.com/features/text-utilities)**: Stemmer, Phonetic, Tone, Zawgyi.
*   **[Text Validation](https://docs.myspellchecker.com/features/text-validation)**: Input text validation.

### Performance & Scale
*   **[Streaming](https://docs.myspellchecker.com/features/streaming)**: Large document processing.
*   **[Batch Processing](https://docs.myspellchecker.com/features/batch-processing)**: High-throughput parallel processing.
*   **[Async API](https://docs.myspellchecker.com/features/async-api)**: Non-blocking spell check operations.
*   **[Performance Tuning](https://docs.myspellchecker.com/guides/performance-tuning)**: Optimization strategies.
*   **[Connection Pooling](https://docs.myspellchecker.com/guides/connection-pool)**: Database connection management.

### Customization
*   **[Customization Guide](https://docs.myspellchecker.com/guides/customization)**: Extending and customizing behavior.
*   **[Custom Dictionaries](https://docs.myspellchecker.com/guides/custom-dictionaries)**: Build and customize dictionaries.
*   **[Custom Grammar Rules](https://docs.myspellchecker.com/guides/custom-grammar-rules)**: Write YAML grammar rules.
*   **[Caching](https://docs.myspellchecker.com/guides/caching)**: Algorithm and result caching.
*   **[Resource Caching](https://docs.myspellchecker.com/guides/resource-caching)**: Model and resource caching.
*   **[Logging](https://docs.myspellchecker.com/guides/logging)**: Centralized logging configuration.

### Integration & Deployment
*   **[Integration Guide](https://docs.myspellchecker.com/guides/integration)**: Integrate with web apps and APIs.
*   **[Docker](https://docs.myspellchecker.com/guides/docker)**: Container deployment guide.
*   **[Zawgyi Support](https://docs.myspellchecker.com/guides/zawgyi-support)**: Legacy encoding handling.

### Dictionary Building
*   **[Pipeline Overview](https://docs.myspellchecker.com/data-pipeline/index)**: Dictionary building pipeline.
*   **[Corpus Format](https://docs.myspellchecker.com/data-pipeline/corpus-format)**: Supported input formats.
*   **[Ingestion](https://docs.myspellchecker.com/data-pipeline/ingestion)**: Corpus ingestion details.
*   **[Building Dictionaries](https://docs.myspellchecker.com/data-pipeline/building)**: Step-by-step build guide.
*   **[Optimization](https://docs.myspellchecker.com/data-pipeline/optimization)**: Performance tuning for large corpora.

### API & CLI Reference
*   **[API Reference](https://docs.myspellchecker.com/api-reference/index)**: Full API documentation.
*   **[SpellChecker API](https://docs.myspellchecker.com/core/spellchecker)**: Main SpellChecker class reference.
*   **[Configuration API](https://docs.myspellchecker.com/core/configuration)**: Configuration class reference.
*   **[Provider Capabilities](https://docs.myspellchecker.com/api-reference/provider-capabilities)**: Dictionary provider interface.
*   **[Tokenizers](https://docs.myspellchecker.com/api-reference/tokenizers)**: Tokenizer API reference.
*   **[CLI Reference](https://docs.myspellchecker.com/cli/index)**: Command-line interface guide.

### Core Internals
*   **[Core Overview](https://docs.myspellchecker.com/core/index)**: Core package internals.
*   **[Syllable Validation](https://docs.myspellchecker.com/core/syllable-validation)**: Syllable validator internals.
*   **[Word Validation](https://docs.myspellchecker.com/core/word-validation)**: Word validator internals.
*   **[Training Internals](https://docs.myspellchecker.com/core/training)**: ML training pipeline internals.
*   **[Algorithm Factory](https://docs.myspellchecker.com/guides/algorithm-factory)**: Algorithm instantiation patterns.
*   **[I/O Utilities](https://docs.myspellchecker.com/guides/io-utilities)**: File I/O utilities reference.

### Algorithms
*   **[Algorithms Overview](https://docs.myspellchecker.com/algorithms/index)**: Algorithm catalog.
*   **[SymSpell](https://docs.myspellchecker.com/algorithms/symspell)**: O(1) suggestion algorithm.
*   **[Edit Distance](https://docs.myspellchecker.com/algorithms/edit-distance)**: Myanmar-aware Levenshtein distance.
*   **[Suggestion Ranking](https://docs.myspellchecker.com/algorithms/suggestion-ranking)**: Multi-signal ranking pipeline.
*   **[Neural Reranker](https://docs.myspellchecker.com/algorithms/neural-reranker)**: ONNX-based MLP/GBT suggestion reranker.
*   **[Suggestion Strategy](https://docs.myspellchecker.com/algorithms/suggestion-strategy)**: Strategy pattern for suggestions.
*   **[Morpheme Suggestions](https://docs.myspellchecker.com/algorithms/morpheme-suggestion)**: Morpheme-level and medial swap corrections.
*   **[N-gram Context](https://docs.myspellchecker.com/algorithms/ngram)**: Bigram/Trigram probability models.
*   **[Context-Aware Checking](https://docs.myspellchecker.com/algorithms/context-aware)**: N-gram and syntactic rules.
*   **[Semantic Algorithm](https://docs.myspellchecker.com/algorithms/semantic)**: AI/ML inference internals.
*   **[Grammar Rules Engine](https://docs.myspellchecker.com/algorithms/grammar-rules)**: Grammar rule processing.
*   **[Tone Disambiguation](https://docs.myspellchecker.com/algorithms/tone-disambiguation)**: Tone mark resolution.
*   **[NER Algorithm](https://docs.myspellchecker.com/algorithms/ner)**: NER implementation details.

### Segmentation & Tagging
*   **[Segmentation Overview](https://docs.myspellchecker.com/algorithms/segmentation)**: Segmentation algorithm catalog.
*   **[Syllable Segmentation](https://docs.myspellchecker.com/algorithms/syllable-segmentation)**: Syllable-level segmentation.
*   **[Normalization Algorithm](https://docs.myspellchecker.com/algorithms/normalization)**: Text normalization internals.
*   **[Phonetic Algorithm](https://docs.myspellchecker.com/algorithms/phonetic)**: Phonetic hashing and similarity.
*   **[Viterbi POS Tagger](https://docs.myspellchecker.com/algorithms/viterbi)**: HMM-based POS tagging.
*   **[POS Disambiguator](https://docs.myspellchecker.com/algorithms/pos-disambiguator)**: POS disambiguation logic.
*   **[Joint Segmentation](https://docs.myspellchecker.com/algorithms/joint-segment-tagger)**: Combined segmentation + tagging.

### Architecture
*   **[Architecture Overview](https://docs.myspellchecker.com/architecture/index)**: Multi-layer validation pipeline.
*   **[System Design](https://docs.myspellchecker.com/architecture/system-design)**: Component architecture.
*   **[Validation Pipeline](https://docs.myspellchecker.com/architecture/validation-pipeline)**: Pipeline execution flow.
*   **[Component Diagram](https://docs.myspellchecker.com/architecture/component-diagram)**: Visual component map.
*   **[Data Flow](https://docs.myspellchecker.com/architecture/data-flow)**: Data flow through the system.
*   **[Dependency Injection](https://docs.myspellchecker.com/architecture/dependency-injection)**: Component management system.
*   **[Extension Points](https://docs.myspellchecker.com/architecture/extension-points)**: How to extend the library.

### Error & Rules Reference
*   **[Reference Overview](https://docs.myspellchecker.com/reference/index)**: Technical reference index.
*   **[Error Types](https://docs.myspellchecker.com/reference/error-types)**: Error classification reference.
*   **[Error Codes](https://docs.myspellchecker.com/reference/error-codes)**: Complete error code listing.
*   **[Rules System](https://docs.myspellchecker.com/reference/rules-system)**: YAML configuration files.

### Data Reference
*   **[Constants](https://docs.myspellchecker.com/reference/constants)**: Myanmar Unicode constants and character sets.
*   **[Glossary](https://docs.myspellchecker.com/reference/glossary)**: Terms and definitions.
*   **[Phonetic Data](https://docs.myspellchecker.com/reference/phonetic-data)**: Phonetic groups and similarity mappings.

### Data Pipeline Internals
*   **[Pipeline Core](https://docs.myspellchecker.com/core/data-pipeline)**: Data pipeline core module.
*   **[Database Schema](https://docs.myspellchecker.com/data-pipeline/database-schema)**: SQLite schema reference.
*   **[Schema Management](https://docs.myspellchecker.com/data-pipeline/schema-management)**: Schema versioning and migrations.
*   **[Providers](https://docs.myspellchecker.com/data-pipeline/providers)**: Data source providers.
*   **[Processing](https://docs.myspellchecker.com/data-pipeline/processing)**: Text processing stages.
*   **[POS Inference](https://docs.myspellchecker.com/data-pipeline/pos-inference)**: POS tagging during build.
*   **[Segmentation Repair](https://docs.myspellchecker.com/data-pipeline/segmentation-repair)**: Segmentation error correction.
*   **[Pipeline Reporter](https://docs.myspellchecker.com/data-pipeline/pipeline-reporter)**: Build progress reporting.

### Help & FAQ
*   **[FAQ](https://docs.myspellchecker.com/reference/faq)**: Frequently asked questions.
*   **[Troubleshooting](https://docs.myspellchecker.com/reference/troubleshooting)**: Common issues and solutions.
*   **[Comparisons](https://docs.myspellchecker.com/reference/comparisons)**: How mySpellChecker compares to other tools.

### Development
*   **[Development Guide](https://docs.myspellchecker.com/development/index)**: Development overview.
*   **[Setup](https://docs.myspellchecker.com/development/setup)**: Development environment setup.
*   **[Contributing](https://docs.myspellchecker.com/development/contributing)**: Contribution guidelines.
*   **[Naming Conventions](https://docs.myspellchecker.com/development/naming-conventions)**: Code naming standards.
*   **[Testing](https://docs.myspellchecker.com/development/testing)**: Test suite and coverage.
*   **[Benchmarks](https://docs.myspellchecker.com/development/benchmarks)**: Benchmark suite and scoring methodology.
*   **[Cython Dev Guide](https://docs.myspellchecker.com/development/cython-guide)**: Working with Cython extensions.
*   **[Cython Reference](https://docs.myspellchecker.com/guides/cython)**: Cython patterns and optimization.
*   **[CLI Formatting](https://docs.myspellchecker.com/guides/cli-formatting)**: CLI output formatting internals.

## Quick Start

### 1. Installation

**Prerequisites:**
*   Python 3.10+
*   C++ Compiler (GCC/Clang/MSVC) for building Cython extensions.

**Standard (Recommended):**
```bash
pip install myspellchecker
```

**With Transformer POS Tagging (Optional):**
```bash
# Enables transformer-based POS tagging for 93% accuracy
pip install "myspellchecker[transformers]"
```

**Full (with all features):**
```bash
pip install "myspellchecker[ai,build,train,transformers]"
```

### 2. Build Dictionary

The library requires a dictionary database. You can build a sample one or use your own corpus.

```bash
# Install build dependencies (pyarrow, duckdb, etc.)
pip install "myspellchecker[build]"

# Build a sample database for testing
myspellchecker build --sample

# Build from your own text corpus
myspellchecker build --input corpus.txt --output mySpellChecker.db
```

### 3. Usage

**Python:**

```python
from myspellchecker.core import SpellCheckerBuilder, ConfigPresets, ValidationLevel

# 1. Initialize with Builder (Recommended)
checker = (
    SpellCheckerBuilder()
    .with_config(ConfigPresets.DEFAULT)
    .with_phonetic(True)
    .build()
)

# 2. Simple Syllable Check (Fastest)
text = "မြနမ်ာနိုင်ငံ"
result = checker.check(text)
print(f"Corrected: {result.corrected_text}")
# Output: မြန်မာနိုင်ငံ

# 3. Context-Aware Check (Slower, more accurate)
# Detects that 'နီ' (Red) is wrong in this context, suggests 'နေ' (Stay/Ing)
text = "မင်းဘာလုပ်နီလဲ"
result = checker.check(text, level=ValidationLevel.WORD)
print(f"Corrected: {result.corrected_text}")
# Output: မင်းဘာလုပ်နေလဲ
```

**CLI:**

See the [CLI Reference](https://docs.myspellchecker.com/cli/index) for full details.

```bash
# Check a string
echo "မင်္ဂလာပါ" | myspellchecker

# Check a file with rich output
myspellchecker check input.txt --format rich

# Segment text with POS tags
echo "မြန်မာနိုင်ငံ" | myspellchecker segment --tag

```

### 4. Configuration

```python
from myspellchecker import SpellChecker
from myspellchecker.core.config import SpellCheckerConfig
from myspellchecker.providers.sqlite import SQLiteProvider

# Configure with custom settings
config = SpellCheckerConfig(
    max_edit_distance=2,
    max_suggestions=5,
    use_context_checker=True,
    use_phonetic=True,
    use_ner=True
)
checker = SpellChecker(
    config=config,
    provider=SQLiteProvider(database_path="mySpellChecker.db")
)
```

See the [Configuration Guide](https://docs.myspellchecker.com/guides/configuration) for all options.

### 5. Logging

Configure logging globally at the start of your application:

```python
from myspellchecker.utils.logging_utils import configure_logging

# Enable verbose debug logging
configure_logging(level="DEBUG")

# Or use structured JSON logging for production
configure_logging(level="INFO", json_output=True)
```

See the [Logging Guide](https://docs.myspellchecker.com/guides/logging) for details.

## Advanced Features

### Grammar Checking

Eight specialized grammar checkers for Myanmar:

```python
from myspellchecker.grammar.checkers.register import RegisterChecker

checker = RegisterChecker()
errors = checker.validate_sequence(["သူ", "သည်", "စာအုပ်", "ဖတ်", "တယ်"])
# Detects mixed register (formal "သည်" + colloquial "တယ်")
```

See the [Grammar Checkers Guide](https://docs.myspellchecker.com/features/grammar-checkers) for details.

### Named Entity Recognition

Reduce false positives by identifying names and places:

```python
from myspellchecker.core.config import SpellCheckerConfig

config = SpellCheckerConfig(use_ner=True)
```

See the [NER Guide](https://docs.myspellchecker.com/features/ner) for details.

### POS Tagging & Joint Segmentation

**mySpellChecker** supports advanced linguistic analysis:

- **Pluggable POS Tagging**: Rule-Based (fastest), Viterbi (balanced), or Transformer (most accurate).
- **Joint Segmentation**: Combine word breaking and tagging in a single pass.

```python
from myspellchecker.core.config import SpellCheckerConfig, JointConfig

config = SpellCheckerConfig(
    joint=JointConfig(enabled=True)
)
checker = SpellChecker(config=config)
words, tags = checker.segment_and_tag("မြန်မာနိုင်ငံ")
```

See the [POS Tagging Guide](https://docs.myspellchecker.com/features/pos-tagging) for details.

### Validation Strategies

Composable validation pipeline with 12 strategies:

| Strategy | Priority | Purpose |
|----------|----------|---------|
| ToneValidation | 10 | Tone mark disambiguation |
| Orthography | 15 | Medial order and compatibility |
| SyntacticRule | 20 | Grammar rule checking |
| StatisticalConfusable | 24 | Bigram-based confusable detection |
| BrokenCompound | 25 | Broken compound word detection |
| POSSequence | 30 | POS sequence validation |
| Question | 40 | Question structure |
| Homophone | 45 | Sound-alike detection |
| ConfusableCompoundClassifier | 47 | MLP-based confusable/compound detection |
| ConfusableSemantic | 48 | MLM-enhanced confusable detection |
| NgramContext | 50 | N-gram probability |
| Semantic | 70 | AI-powered validation (ONNX) |

See the [Validation Strategies Guide](https://docs.myspellchecker.com/features/validation-strategies) for details.

## Benchmark Results

Tested on a 1,138-sentence benchmark suite (444 clean, 694 with errors, 564 error spans) covering 3 difficulty tiers and 6 domains. The dictionary database and semantic model are **not bundled** with the library — users build or provide their own.

**Test environment:**
- Dictionary: Production SQLite database (565 MB, 601K words, 2.2M bigrams, enrichment tables)
- Semantic model: Custom RoBERTa MLM (6L/768H, ONNX quantized, 71 MB)
- Hardware: Apple Silicon, Python 3.14
- Benchmark: [`benchmarks/myspellchecker_benchmark.yaml`](benchmarks/) (1,138 sentences)

### With Semantic Model

| Metric | Value |
|--------|-------|
| **F1 Score** | 98.3% |
| **Precision** | 97.1% |
| **Recall** | 99.6% |
| **False Positives** | 14 (0% on clean sentences) |
| **False Negatives** | 2 |
| **Top-1 Suggestion Accuracy** | 81.2% |
| **MRR** | 0.8395 |
| **Mean Latency** | 35.2 ms/sentence |
| **P50 Latency** | 32.1 ms |

### Without Semantic Model

| Metric | Value |
|--------|-------|
| **F1 Score** | 96.2% |
| **Precision** | 97.8% |
| **Recall** | 94.7% |
| **False Positives** | 10 (0% on clean sentences) |
| **False Negatives** | 25 |
| **Top-1 Suggestion Accuracy** | 85.2% |
| **MRR** | 0.8731 |
| **Mean Latency** | ~12 ms/sentence |
| **P50 Latency** | ~7 ms |

The semantic model adds ~23ms mean latency but boosts recall from 94.7% to 99.6% by catching 23 additional context-dependent errors that rule-based methods miss. Both modes maintain sub-35ms P50 latency suitable for interactive use.

## Development

### Setup
```bash
git clone https://github.com/thettwe/myspellchecker.git
cd myspellchecker
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

### Testing

The test suite has 4,658 tests across 213 files with 74% code coverage, organized into unit, integration, e2e, and stress tiers with auto-applied pytest markers.

```bash
# Run default test suite (~5 min, skips slow tests)
pytest tests/

# Run by category
pytest tests/ -m integration    # 307 integration tests
pytest tests/ -m e2e            # 10 end-to-end CLI tests
pytest tests/ -m slow           # 39 slow tests (property-based, stress, DB builds)

# Run with coverage
pytest tests/ --cov=src/myspellchecker --cov-fail-under=75

# Formatting and linting
ruff format .
ruff check .
mypy src/myspellchecker
```

See the [Development Guide](https://docs.myspellchecker.com/development/index) for contributing guidelines and the [Testing Guide](https://docs.myspellchecker.com/development/testing) for test suite details.

## Acknowledgments

mySpellChecker integrates tools and research from the Myanmar NLP community:

### Models & Resources

| Resource | Author | Description | Link |
|----------|--------|-------------|------|
| **Myanmar POS Model** | Chuu Htet Naing | XLM-RoBERTa-based POS tagger (93.37% accuracy) | [HuggingFace](https://huggingface.co/chuuhtetnaing/myanmar-pos-model) |
| **Myanmar NER Model** | Chuu Htet Naing | Transformer-based named entity recognition | [HuggingFace](https://huggingface.co/chuuhtetnaing/myanmar-ner-model) |
| **Myanmar Text Segmentation Model** | Chuu Htet Naing | Transformer-based word segmenter | [HuggingFace](https://huggingface.co/chuuhtetnaing/myanmar-text-segmentation-model) |
| **myWord Segmentation** | Ye Kyaw Thu | Viterbi-based Myanmar word segmentation | [GitHub](https://github.com/ye-kyaw-thu/myWord) |
| **myPOS** | Ye Kyaw Thu | POS corpus used for CRF training | [GitHub](https://github.com/ye-kyaw-thu/myPOS) |
| **myNER** | Ye Kyaw Thu et al. | NER corpus with 7-tag annotation scheme, joint POS training | [arXiv](https://arxiv.org/abs/2504.04038) |
| **myG2P** | Ye Kyaw Thu | Myanmar grapheme-to-phoneme conversion dictionary | [GitHub](https://github.com/ye-kyaw-thu/myG2P) |
| **CRF Word Segmenter** | Ye Kyaw Thu | CRF-based syllable-to-word segmentation model | [GitHub](https://github.com/ye-kyaw-thu) |
| **myanmartools** | Google | Zawgyi detection and conversion | [GitHub](https://github.com/google/myanmar-tools) |

### Key Dependencies

| Library | Purpose | License |
|---------|---------|---------|
| [pycrfsuite](https://github.com/scrapinghub/python-crfsuite) | CRF model inference | MIT |
| [transformers](https://github.com/huggingface/transformers) | Transformer model inference | Apache 2.0 |

### Algorithm References

| Algorithm | Author | Description | Link |
|-----------|--------|-------------|------|
| **SymSpell** | Wolf Garbe | Symmetric delete spelling correction algorithm. mySpellChecker includes a custom implementation with Myanmar-specific variant generation. | [GitHub](https://github.com/wolfgarbe/SymSpell) |
| **SymSpell4Burmese** | Hlaing Myat Nwe et al. | Foundational research on adapting SymSpell for Burmese | [IEEE](https://ieeexplore.ieee.org/document/9678171/) |

### Citations

If you use mySpellChecker in your research, please cite the relevant works:

```bibtex
@misc{chuuhtetnaing-myanmar-pos,
  author = {Chuu Htet Naing},
  title = {Myanmar POS Model},
  year = {2024},
  publisher = {Hugging Face},
  url = {https://huggingface.co/chuuhtetnaing/myanmar-pos-model}
}

@misc{yekyawthu-myword,
  author = {Ye Kyaw Thu},
  title = {myWord: Word Segmentation Tool for Burmese},
  year = {2017},
  publisher = {GitHub},
  url = {https://github.com/ye-kyaw-thu/myWord}
}

@misc{garbe-symspell,
  author = {Wolf Garbe},
  title = {SymSpell: Symmetric Delete Spelling Correction Algorithm},
  year = {2012},
  publisher = {GitHub},
  url = {https://github.com/wolfgarbe/SymSpell}
}

@inproceedings{symspell4burmese,
  title = {SymSpell4Burmese: Symmetric Delete Spelling Correction Algorithm for Burmese},
  author = {Hlaing Myat Nwe and others},
  year = {2021},
  booktitle = {IEEE Conference},
  url = {https://ieeexplore.ieee.org/document/9678171/}
}

@misc{yekyawthu-mypos,
  author = {Ye Kyaw Thu},
  title = {myPOS: POS Corpus for Myanmar Language},
  publisher = {GitHub},
  url = {https://github.com/ye-kyaw-thu/myPOS}
}

@misc{chuuhtetnaing-myanmar-segmentation,
  author = {Chuu Htet Naing},
  title = {Myanmar Text Segmentation Model},
  publisher = {Hugging Face},
  url = {https://huggingface.co/chuuhtetnaing/myanmar-text-segmentation-model}
}

@misc{chuuhtetnaing-myanmar-ner,
  author = {Chuu Htet Naing},
  title = {Myanmar NER Model},
  publisher = {Hugging Face},
  url = {https://huggingface.co/chuuhtetnaing/myanmar-ner-model}
}

@inproceedings{myner-2025,
  title = {myNER: Contextualized Burmese Named Entity Recognition with Bidirectional LSTM and fastText Embeddings via Joint Training with POS Tagging},
  author = {Kaung Lwin Thant and Kwankamol Nongpong and Ye Kyaw Thu and Thura Aung and Khaing Hsu Wai and Thazin Myint Oo},
  year = {2025},
  booktitle = {4th International Conference on Cybernetics and Innovations (ICCI 2025)},
  note = {Best Presentation Award},
  url = {https://arxiv.org/abs/2504.04038}
}

@misc{yekyawthu-myg2p,
  author = {Ye Kyaw Thu},
  title = {myG2P: Myanmar Grapheme to Phoneme Conversion Dictionary},
  publisher = {GitHub},
  url = {https://github.com/ye-kyaw-thu/myG2P}
}
```

Thanks to these researchers and developers for making their work publicly available, enabling high-quality Myanmar language processing.

## License

This project is licensed under the [MIT License](LICENSE).
