"""
Resource loader for tokenization resources.

Downloads from HuggingFace Datasets on first use, caches locally.
This enables a lightweight package distribution while keeping large
binary files hosted externally.

Resources:
- segmentation.mmap: Word segmentation dictionary (memory-mapped)
- wordseg_c2_crf.crfsuite: CRF model for syllable tokenization

Environment Variables:
- MYSPELL_CACHE_DIR: Override default cache directory
- MYSPELL_OFFLINE: If set to "true" or "1", never download (raise error if missing)
"""

from __future__ import annotations

import os
from pathlib import Path

from myspellchecker.core.config.algorithm_configs import ResourceConfig
from myspellchecker.core.exceptions import TokenizationError
from myspellchecker.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Default resource config — single source of truth for resource defaults.
_DEFAULT_RESOURCE_CONFIG = ResourceConfig()

# Resource version (bump with releases for reproducibility)
# Use "main" until a versioned tag is created on HuggingFace
RESOURCE_VERSION: str = _DEFAULT_RESOURCE_CONFIG.resource_version

# HuggingFace Dataset repository for tokenization resources (pinned to version)
HF_REPO: str = _DEFAULT_RESOURCE_CONFIG.hf_repo_url

# Resource URLs (versioned for reproducibility)
RESOURCE_URLS = {
    "segmentation": f"{HF_REPO}/segmentation/segmentation.mmap",
    "crf": f"{HF_REPO}/models/wordseg_c2_crf.crfsuite",
    "curated_lexicon": (f"{HF_REPO}/curated_lexicon/curated_lexicon.csv"),
}

# Default cache directory
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "myspellchecker" / "resources"

# Resource filenames within package data directory
_RESOURCE_FILENAMES = {
    "segmentation": "segmentation.mmap",
    "crf": "wordseg_c2_crf.crfsuite",
    "curated_lexicon": "curated_lexicon.csv",
}


def _get_bundled_resource_path(name: str) -> Path | None:
    """
    Check for a package-bundled resource using importlib.resources.

    This is package-safe and works in both source trees and wheel installs.
    Returns None if the resource is not bundled in the package.

    Args:
        name: Resource name (key in _RESOURCE_FILENAMES)

    Returns:
        Path to the bundled resource, or None if not found
    """
    filename = _RESOURCE_FILENAMES.get(name)
    if filename is None:
        return None

    try:
        try:
            from importlib.resources import files  # Python 3.9+
        except ImportError:
            from importlib_resources import files  # type: ignore[no-redef]

        resource_dir = files("myspellchecker") / "data" / "models"
        resource_file = resource_dir / filename

        # as_posix() / str() works for both on-disk and zip-imported packages
        resource_path = Path(str(resource_file))
        if resource_path.is_file():
            return resource_path
    except (TypeError, FileNotFoundError, ModuleNotFoundError):
        pass

    return None


def get_resource_path(
    name: str,
    cache_dir: Path | None = None,
    force_download: bool = False,
) -> Path:
    """
    Get path to a tokenization resource, downloading if necessary.

    Checks in order:
    1. Package-bundled resource via importlib.resources (package-safe)
    2. Cache directory (previously downloaded)
    3. Downloads from HuggingFace (first time)

    Args:
        name: Resource name ("segmentation" or "crf")
        cache_dir: Custom cache directory (default: ~/.cache/myspellchecker/resources)
                  Can be overridden with MYSPELL_CACHE_DIR environment variable.
        force_download: Skip bundled resources. Note: Does NOT force re-download of
                       cached resources. Delete the cache directory manually to force.

    Returns:
        Path to the resource file

    Raises:
        ValueError: If resource name is unknown
        RuntimeError: If download fails or MYSPELL_OFFLINE=true and resource missing

    Environment Variables:
        MYSPELL_CACHE_DIR: Override default cache directory
        MYSPELL_OFFLINE: If "true" or "1", never download (raise error if resource missing)
    """
    if name not in RESOURCE_URLS:
        raise ValueError(f"Unknown resource: {name}. Available: {list(RESOURCE_URLS.keys())}")

    # Check if offline mode is enabled
    offline_mode = os.getenv("MYSPELL_OFFLINE", "").lower() in ("true", "1", "yes")

    # Check package-bundled resource first
    if not force_download:
        bundled = _get_bundled_resource_path(name)
        if bundled is not None:
            logger.debug(f"Using bundled resource: {bundled}")
            return bundled

    # Check/create cache directory (respect env var override)
    if cache_dir is None:
        env_cache_dir = os.getenv("MYSPELL_CACHE_DIR")
        cache_dir = Path(env_cache_dir) if env_cache_dir else DEFAULT_CACHE_DIR

    cache_dir.mkdir(parents=True, exist_ok=True)

    url = RESOURCE_URLS[name]

    # Try using cached_path library (preferred - handles caching/versioning)
    try:
        from cached_path import cached_path as download_cached
        from cached_path import find_latest_cached

        # Check if file is already cached
        # Use find_latest_cached which handles cached_path's complex naming (hash + etag)
        cached_file = find_latest_cached(url, cache_dir=str(cache_dir))

        # In offline mode, prevent network access
        if offline_mode:
            if cached_file is not None:
                logger.debug(f"Using cached resource (offline mode): {cached_file}")
                return Path(cached_file)
            else:
                # Not cached and offline - cannot proceed
                msg = (
                    f"Resource '{name}' not found in cache"
                    f" and MYSPELL_OFFLINE=true prevents download.\n"
                    f"Searched:\n"
                    f"  - Package-bundled resources\n"
                    f"  - Cache directory: {cache_dir}\n"
                    f"To fix:\n"
                    f"  1. Download first without MYSPELL_OFFLINE:\n"
                    f"     unset MYSPELL_OFFLINE\n"
                    f"     python -c 'from myspellchecker.tokenizers"
                    f".resource_loader import get_resource_path;"
                    f' get_resource_path("{name}")\'\n'
                    f"  2. Then retry with MYSPELL_OFFLINE=true"
                )
                raise TokenizationError(msg)

        # Normal mode: download if needed (cached_path handles caching)
        # cached_path will use cache if available, download if not
        if cached_file is None:
            logger.info(f"Downloading {name} resource (first time)...")
        else:
            logger.debug(f"Using cached {name} resource...")

        downloaded = download_cached(
            url,
            cache_dir=str(cache_dir),
            force_extract=False,
        )

        return Path(downloaded)

    except ImportError:
        # Fallback to urllib if cached_path not available
        import shutil
        import urllib.request

        # Simple filename-based caching for fallback mode
        filename = url.split("/")[-1]
        simple_cached_path = cache_dir / filename

        # Return cached if exists and not forcing download
        if simple_cached_path.exists() and not force_download:
            logger.debug(f"Using cached resource: {simple_cached_path}")
            return simple_cached_path

        # Check offline mode
        if offline_mode:
            raise TokenizationError(
                f"Resource '{name}' not found and"
                f" MYSPELL_OFFLINE=true prevents download.\n"
                f"Searched:\n"
                f"  - Package-bundled resources\n"
                f"  - Cached: {simple_cached_path}\n"
                f"Install cached-path library for"
                f" better caching: pip install cached-path"
            ) from None

        # Download
        logger.info(f"Downloading {name} resource...")
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                with open(simple_cached_path, "wb") as out_file:
                    shutil.copyfileobj(response, out_file)
        except (TimeoutError, OSError) as exc:
            raise TokenizationError(
                f"Failed to download '{name}' resource: {exc}\n"
                f"Set MYSPELL_OFFLINE=true to skip downloads,\n"
                f"or pre-cache the resource manually."
            ) from None

        logger.info(f"Download complete: {simple_cached_path}")
        return simple_cached_path


def get_segmentation_mmap_path(
    cache_dir: Path | None = None,
    force_download: bool = False,
) -> Path:
    """
    Get path to segmentation.mmap file.

    Convenience wrapper around get_resource_path("segmentation").
    """
    return get_resource_path("segmentation", cache_dir, force_download)


def get_crf_model_path(
    cache_dir: Path | None = None,
    force_download: bool = False,
) -> Path:
    """
    Get path to CRF model file.

    Convenience wrapper around get_resource_path("crf").
    """
    return get_resource_path("crf", cache_dir, force_download)


def get_curated_lexicon_path(
    cache_dir: Path | None = None,
    force_download: bool = False,
) -> Path:
    """
    Get path to the curated lexicon CSV file.

    Downloads from HuggingFace on first use, then serves from cache.
    Convenience wrapper around get_resource_path("curated_lexicon").
    """
    return get_resource_path("curated_lexicon", cache_dir, force_download)


def ensure_resources_available(
    cache_dir: Path | None = None,
    force_download: bool = False,
) -> dict[str, Path]:
    """Download all tokenization resources and return their paths.

    Useful for pre-downloading resources in a main process before
    spawning workers that will use them.

    Returns:
        Dict mapping resource name to its local file path.
    """
    paths: dict[str, Path] = {}
    for name in RESOURCE_URLS:
        paths[name] = get_resource_path(name, cache_dir, force_download)
    return paths
