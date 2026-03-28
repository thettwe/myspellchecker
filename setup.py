import os
import platform
import sys
import warnings

from setuptools import Extension, setup

try:
    from Cython.Build import cythonize

    HAS_CYTHON = True
except ImportError:
    HAS_CYTHON = False


# Detect platform and configure OpenMP flags
def get_openmp_flags():
    """
    Get OpenMP compiler and linker flags for the current platform.

    Platform behavior:
    - macOS: Check for Homebrew libomp, warn if not found (requires: brew install libomp)
    - Windows: Disable OpenMP (MSVC /openmp requires proper Visual Studio setup)
    - Linux/Unix: Enable OpenMP with -fopenmp flags
    """
    if sys.platform == "darwin":  # macOS
        # Check for Homebrew libomp
        brew_prefix = os.environ.get("HOMEBREW_PREFIX", "/opt/homebrew")
        if platform.machine() == "x86_64":
            brew_prefix = "/usr/local"  # Intel Mac

        libomp_include = f"{brew_prefix}/opt/libomp/include"
        libomp_lib = f"{brew_prefix}/opt/libomp/lib"

        if os.path.exists(libomp_include):
            return {
                "extra_compile_args": ["-Xpreprocessor", "-fopenmp", f"-I{libomp_include}", "-O3"],
                "extra_link_args": ["-lomp", f"-L{libomp_lib}"],
            }
        else:
            # No OpenMP available, return empty flags
            warnings.warn(
                "libomp not found. OpenMP parallelization disabled. "
                "Install with: brew install libomp",
                UserWarning,
                stacklevel=2,
            )
            return {"extra_compile_args": ["-O3"], "extra_link_args": []}
    elif sys.platform == "win32":  # Windows
        # Windows MSVC doesn't support -fopenmp flag directly
        # OpenMP is enabled via /openmp flag, but requires proper MSVC setup
        # For now, disable OpenMP on Windows to ensure builds work out-of-the-box
        warnings.warn(
            "OpenMP parallelization disabled on Windows. "
            "For OpenMP support, ensure Visual Studio with OpenMP is installed.",
            UserWarning,
            stacklevel=2,
        )
        return {"extra_compile_args": ["/O2"], "extra_link_args": []}
    else:  # Linux and other Unix-like systems
        return {
            "extra_compile_args": ["-fopenmp", "-O3"],
            "extra_link_args": ["-fopenmp"],
        }


openmp_flags = get_openmp_flags()

# Define extensions
extensions = [
    # Tokenizers - Cython modules for word segmentation
    Extension(
        name="myspellchecker.tokenizers.cython.word_segment",
        sources=["src/myspellchecker/tokenizers/cython/word_segment.pyx"],
        language="c++",
    ),
    Extension(
        name="myspellchecker.tokenizers.cython.mmap_reader",
        sources=["src/myspellchecker/tokenizers/cython/mmap_reader.pyx"],
        language="c++",
    ),
    # Text processing
    Extension(
        name="myspellchecker.text.normalize_c",
        sources=["src/myspellchecker/text/normalize_c.pyx"],
        language="c++",
    ),
    # Distance metrics
    Extension(
        name="myspellchecker.algorithms.distance.edit_distance_c",
        sources=["src/myspellchecker/algorithms/distance/edit_distance_c.pyx"],
        language="c++",
    ),
    # Data Pipeline
    Extension(
        name="myspellchecker.data_pipeline.batch_processor",
        sources=["src/myspellchecker/data_pipeline/batch_processor.pyx"],
        language="c++",
        **openmp_flags,  # OpenMP for parallel batch processing
    ),
    Extension(
        name="myspellchecker.data_pipeline.frequency_counter",
        sources=["src/myspellchecker/data_pipeline/frequency_counter.pyx"],
        language="c++",
    ),
    Extension(
        name="myspellchecker.data_pipeline.repair_c",
        sources=["src/myspellchecker/data_pipeline/repair_c.pyx"],
        language="c++",
    ),
    Extension(
        name="myspellchecker.data_pipeline.ingester_c",
        sources=["src/myspellchecker/data_pipeline/ingester_c.pyx"],
        language="c++",
    ),
    Extension(
        name="myspellchecker.data_pipeline.tsv_reader_c",
        sources=["src/myspellchecker/data_pipeline/tsv_reader_c.pyx"],
        language="c++",
    ),
    # Core
    Extension(
        name="myspellchecker.core.syllable_rules_c",
        sources=["src/myspellchecker/core/syllable_rules_c.pyx"],
        language="c++",
    ),
    # Algorithms
    Extension(
        name="myspellchecker.algorithms.viterbi_c",  # Suffix with _c to avoid name collision
        sources=["src/myspellchecker/algorithms/viterbi.pyx"],
        language="c++",
    ),
]

# Minimal setup() call - most configuration is in pyproject.toml
if HAS_CYTHON:
    setup(
        ext_modules=cythonize(extensions, compiler_directives={"language_level": "3"}),
    )
else:
    # Pure Python fallback — Cython extensions are optional.
    # The library gracefully falls back to pure Python implementations.
    # Install Cython for compiled extensions: pip install Cython>=3.0.0
    warnings.warn(
        "Cython not found. Building without compiled extensions. "
        "The library will use pure Python fallbacks (slower). "
        "Install Cython for optimized builds: pip install 'Cython>=3.0.0'",
        UserWarning,
        stacklevel=2,
    )
    setup()
