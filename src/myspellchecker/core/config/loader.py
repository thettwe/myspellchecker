"""
Configuration Loader for mySpellChecker.

This module provides utilities to load SpellCheckerConfig from various sources:
- Profile presets (development, production, testing, fast, accurate)
- YAML/JSON configuration files
- Environment variables
- Programmatic overrides

Environment Variable Mapping:
    Core Settings:
        MYSPELL_MAX_EDIT_DISTANCE: Maximum edit distance for suggestions (1-3)
        MYSPELL_MAX_SUGGESTIONS: Maximum number of suggestions (>=1)
        MYSPELL_USE_CONTEXT_CHECKER: Enable context validation (true/false)
        MYSPELL_USE_PHONETIC: Enable phonetic matching (true/false)
        MYSPELL_USE_NER: Enable Named Entity Recognition (true/false)
        MYSPELL_USE_RULE_BASED_VALIDATION: Enable rule-based validation (true/false)
        MYSPELL_WORD_ENGINE: Word segmentation engine (myword, crf)
        MYSPELL_FALLBACK_TO_EMPTY_PROVIDER: Fall back to empty provider (true/false)

    POS Tagger Settings:
        MYSPELL_POS_TAGGER_TYPE: POS tagger type (rule_based, transformer, viterbi)
        MYSPELL_POS_TAGGER_BEAM_WIDTH: Viterbi beam width (integer)
        MYSPELL_POS_TAGGER_MODEL_NAME: Transformer model name (string)

    SymSpell Settings:
        MYSPELL_SYMSPELL_PREFIX_LENGTH: Prefix length for SymSpell (4-10)
        MYSPELL_SYMSPELL_BEAM_WIDTH: Beam width for SymSpell (integer)
        MYSPELL_SYMSPELL_USE_WEIGHTED_DISTANCE: Enable Myanmar-weighted distance (true/false)

    Context Checker Settings:
        MYSPELL_NGRAM_BIGRAM_THRESHOLD: Bigram probability threshold (float)
        MYSPELL_NGRAM_TRIGRAM_THRESHOLD: Trigram probability threshold (float)
        MYSPELL_NGRAM_RERANK_LEFT_WEIGHT: Left-context rerank weight (0.0-1.0)
        MYSPELL_NGRAM_RERANK_RIGHT_WEIGHT: Right-context rerank weight (0.0-1.0)

    Phonetic Settings:
        MYSPELL_PHONETIC_BYPASS_THRESHOLD: Phonetic similarity threshold (0.0-1.0)
        MYSPELL_PHONETIC_EXTRA_DISTANCE: Extra edit distance for phonetic bypass (0-3)

    Ranker Settings:
        MYSPELL_RANKER_UNIFIED_BASE_TYPE: Base ranker for UnifiedRanker (default, frequency_first,
            phonetic_first, edit_distance_only)
        MYSPELL_RANKER_ENABLE_TARGETED_RERANK_HINTS: Enable targeted rerank hints (true/false)
        MYSPELL_RANKER_ENABLE_TARGETED_CANDIDATE_INJECTIONS: Enable targeted candidate
            injections (true/false)
        MYSPELL_RANKER_ENABLE_TARGETED_GRAMMAR_COMPLETION_TEMPLATES: Enable targeted
            grammar completion templates (true/false)

    Provider Settings:
        MYSPELL_DATABASE_PATH: Path to SQLite database file
        MYSPELL_POOL_MIN_SIZE: Connection pool minimum size (integer)
        MYSPELL_POOL_MAX_SIZE: Connection pool maximum size (integer)

    Validation Settings:
        MYSPELL_ALLOW_EXTENDED_MYANMAR: Allow Extended Myanmar characters
            for Shan, Mon, etc. When enabled, accepts:
            - Extended Core (U+1050-U+109F)
            - Extended-A (U+AA60-U+AA7F)
            - Extended-B (U+A9E0-U+A9FF)
            - Non-standard core chars (U+1022, U+1028, U+1033-U+1035)
            (true/false, default: false)


Example:
    >>> from myspellchecker.core.config.loader import ConfigLoader
    >>>
    >>> # Load from profile
    >>> loader = ConfigLoader()
    >>> config = loader.load(profile="fast")
    >>>
    >>> # Load from file with env overrides
    >>> config = loader.load(config_file="myconfig.yaml", use_env=True)
    >>>
    >>> # Load with programmatic overrides
    >>> config = loader.load(
    ...     profile="production",
    ...     overrides={"max_edit_distance": 3, "max_suggestions": 10}
    ... )
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Literal, cast

from myspellchecker.core.config.main import SpellCheckerConfig
from myspellchecker.core.config.profiles import ProfileName, get_profile
from myspellchecker.core.exceptions import InvalidConfigError

# Type alias for supported file formats
ConfigFileFormat = Literal["yaml", "json", "auto"]

# Configuration file search paths
CONFIG_FILE_NAMES = ["myspellchecker.yaml", "myspellchecker.yml", "myspellchecker.json"]
USER_CONFIG_DIR = Path.home() / ".config" / "myspellchecker"


def is_yaml_available() -> bool:
    """Check if PyYAML is installed."""
    import importlib.util

    return importlib.util.find_spec("yaml") is not None


def find_config_file(path: str | Path | None = None) -> Path | None:
    """
    Find configuration file in standard locations.

    Args:
        path: Optional explicit path to config file

    Returns:
        Path to config file if found, None otherwise
    """
    if path:
        p = Path(path)
        return p if p.exists() else None

    # Search in current directory
    cwd = Path.cwd()
    for name in CONFIG_FILE_NAMES:
        config_path = cwd / name
        if config_path.exists():
            return config_path

    # Search in user config directory
    for name in CONFIG_FILE_NAMES:
        config_path = USER_CONFIG_DIR / name
        if config_path.exists():
            return config_path

    return None


def load_config_from_file(path: str | Path) -> dict[str, Any]:
    """
    Load raw configuration data from file.

    Args:
        path: Path to config file (YAML or JSON)

    Returns:
        Dictionary with raw config data
    """
    path = Path(path)

    data: Any
    if path.suffix in [".yaml", ".yml"]:
        if not is_yaml_available():
            raise ImportError("PyYAML is required to load YAML config files")
        import yaml  # type: ignore[import-untyped]

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    elif path.suffix == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    else:
        raise InvalidConfigError(f"Unsupported config file format: {path.suffix}")

    if not isinstance(data, dict):
        raise InvalidConfigError(
            f"Configuration file must contain a top-level object/dict, got {type(data).__name__}"
        )
    return cast(dict[str, Any], data)


def init_config_file(path: str | Path | None = None, *, force: bool = False) -> Path:
    """
    Initialize a new configuration file with default settings.

    Args:
        path: Optional path for config file. If not provided, creates in user config dir
        force: If True, overwrites existing file (keyword-only)

    Returns:
        Path to created config file

    Raises:
        FileExistsError: If file exists and force=False
    """
    if path:
        config_path = Path(path)
    else:
        USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        config_path = USER_CONFIG_DIR / "myspellchecker.yaml"

    if config_path.exists() and not force:
        raise FileExistsError(f"Config file already exists: {config_path}")

    # Create default config file with proper schema
    # Note: 'preset' and 'database' are special keys handled by the loader
    # CLI flags override these settings (e.g., --preset, --db, --no-phonetic, --no-context)
    default_config = {
        # Special loader keys (not SpellCheckerConfig fields)
        # Profile name (options: production, development, testing, fast, accurate)
        "preset": "production",
        # Maps to provider_config.database_path (alternative: set in provider_config below)
        "database": None,
        # SpellCheckerConfig fields
        "max_edit_distance": 2,
        "max_suggestions": 5,
        "use_context_checker": True,
        "use_phonetic": True,
        "word_engine": "myword",
        # ProviderConfig fields (nested under provider_config)
        "provider_config": {
            "database_path": None,  # Database path (alternative: use top-level 'database' key)
            "cache_size": 1024,
            "pool_min_size": 1,  # Minimum connections in pool
            "pool_max_size": 5,  # Maximum connections in pool
        },
    }

    if config_path.suffix in [".yaml", ".yml"]:
        if not is_yaml_available():
            raise ImportError("PyYAML is required to create YAML config files")
        import yaml  # type: ignore[import-untyped]

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(default_config, f, default_flow_style=False)
    else:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(default_config, f, indent=2)

    return config_path


class ConfigLoader:
    """
    Configuration loader with support for profiles, files, and environment variables.

    The loader follows this precedence (highest to lowest):
    1. Programmatic overrides (via overrides parameter)
    2. Environment variables (if use_env=True)
    3. Configuration file (if config_file specified)
    4. Profile defaults (if profile specified)
    5. Built-in defaults
    """

    def __init__(self, env_prefix: str = "MYSPELL_"):
        """
        Initialize the configuration loader.

        Args:
            env_prefix: Prefix for environment variables (default: "MYSPELL_")
        """
        self.env_prefix = env_prefix

    def load(
        self,
        profile: ProfileName | None = None,
        config_file: str | Path | None = None,
        file_format: ConfigFileFormat = "auto",
        use_env: bool = True,
        overrides: dict[str, Any] | None = None,
    ) -> SpellCheckerConfig:
        """
        Load configuration from multiple sources with precedence.

        Args:
            profile: Profile name to load (development, production, testing, fast, accurate)
            config_file: Path to YAML/JSON configuration file
            file_format: File format (yaml, json, auto). Auto-detects from extension.
            use_env: Whether to read environment variables
            overrides: Programmatic overrides (highest precedence)

        Returns:
            SpellCheckerConfig: Loaded and validated configuration

        Raises:
            FileNotFoundError: If config_file specified but not found
            ValueError: If invalid profile or configuration

        Example:
            >>> loader = ConfigLoader()
            >>>
            >>> # Load from profile
            >>> config = loader.load(profile="fast")
            >>>
            >>> # Load from file with env overrides
            >>> config = loader.load(config_file="myconfig.yaml", use_env=True)
            >>>
            >>> # Complex loading
            >>> config = loader.load(
            ...     profile="production",
            ...     config_file="custom.yaml",
            ...     use_env=True,
            ...     overrides={"max_edit_distance": 3}
            ... )
        """
        # Start with base config (either from profile or defaults)
        if profile:
            config_dict = self._profile_to_dict(profile)
        else:
            config_dict = {}

        # Layer 2: Configuration file (if specified)
        if config_file:
            file_config = self._load_from_file(config_file, file_format)
            # Handle special config file keys (preset, database)
            file_config = self._transform_file_config(file_config, profile)
            config_dict = self._merge_dicts(config_dict, file_config)

        # Layer 3: Environment variables (if enabled)
        if use_env:
            env_config = self._load_from_env()
            config_dict = self._merge_dicts(config_dict, env_config)

        # Layer 4: Programmatic overrides (highest precedence)
        if overrides:
            config_dict = self._merge_dicts(config_dict, overrides)

        # Create and validate config
        return SpellCheckerConfig(**config_dict)

    def _profile_to_dict(self, profile: ProfileName) -> dict[str, Any]:
        """
        Convert profile config to dictionary.

        Args:
            profile: Profile name

        Returns:
            Dictionary representation of profile config
        """
        config = get_profile(profile)
        return config.model_dump(exclude_none=True, exclude_unset=True)

    def _load_from_file(
        self,
        config_file: str | Path,
        file_format: ConfigFileFormat = "auto",
    ) -> dict[str, Any]:
        """
        Load configuration from YAML or JSON file.

        Args:
            config_file: Path to configuration file
            file_format: File format (yaml, json, auto)

        Returns:
            Configuration dictionary

        Raises:
            FileNotFoundError: If file not found
            ValueError: If unsupported format or invalid content
        """
        path = Path(config_file)

        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        # Auto-detect format from extension
        suffix = path.suffix.lower()
        if file_format == "auto":
            if suffix in (".yaml", ".yml"):
                file_format = "yaml"
            elif suffix == ".json":
                file_format = "json"
            else:
                raise InvalidConfigError(
                    f"Cannot auto-detect format from extension: {suffix}. "
                    "Supported: .yaml, .yml, .json"
                )
        else:
            # Warn if explicit format doesn't match extension
            expected_ext = {
                "yaml": (".yaml", ".yml"),
                "json": (".json",),
            }
            if suffix and suffix not in expected_ext.get(file_format, ()):
                import warnings

                warnings.warn(
                    f"File extension '{suffix}' does not match specified format "
                    f"'{file_format}'. This may cause parsing errors.",
                    UserWarning,
                    stacklevel=3,
                )

        # Load based on format
        if file_format == "yaml":
            return self._load_yaml(path)
        elif file_format == "json":
            return self._load_json(path)
        else:
            raise InvalidConfigError(f"Unsupported file format: {file_format}")

    def _load_yaml(self, path: Path) -> dict[str, Any]:
        """
        Load configuration from YAML file.

        Args:
            path: Path to YAML file

        Returns:
            Configuration dictionary

        Raises:
            ImportError: If PyYAML not installed
            ValueError: If invalid YAML content
        """
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as err:
            raise ImportError(
                "PyYAML is required to load YAML configuration files. "
                "Install it with: pip install pyyaml"
            ) from err

        try:
            with path.open("r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            if not isinstance(config, dict):
                raise InvalidConfigError(f"YAML file must contain a dictionary, got {type(config)}")

            return config
        except yaml.YAMLError as err:
            # Extract line/column info from YAML error if available
            location = ""
            if hasattr(err, "problem_mark") and err.problem_mark:
                mark = err.problem_mark
                location = f" at line {mark.line + 1}, column {mark.column + 1}"
            raise InvalidConfigError(f"Invalid YAML in {path}{location}: {err}") from err

    def _load_json(self, path: Path) -> dict[str, Any]:
        """
        Load configuration from JSON file.

        Args:
            path: Path to JSON file

        Returns:
            Configuration dictionary

        Raises:
            ValueError: If invalid JSON content
        """
        try:
            with path.open("r", encoding="utf-8") as f:
                config = json.load(f)

            if not isinstance(config, dict):
                raise InvalidConfigError(f"JSON file must contain an object, got {type(config)}")

            return config
        except json.JSONDecodeError as err:
            # Include line/column info from JSON error
            raise InvalidConfigError(
                f"Invalid JSON in {path} at line {err.lineno}, column {err.colno}: {err.msg}"
            ) from err

    def _load_from_env(self) -> dict[str, Any]:
        """
        Load configuration from environment variables.

        Returns:
            Configuration dictionary from environment variables

        Supports comprehensive environment variable mapping for all major
        configuration options. See module docstring for full list.
        """
        config: dict[str, Any] = {}

        # Map environment variables to config keys with validation
        # Format: "ENV_SUFFIX": ("config.key.path", converter_function)
        # Note: Validators enforce same constraints as Pydantic Field definitions
        env_mappings: dict[str, tuple[str, Callable[[str], Any]]] = {
            # Core settings (with range validation matching Pydantic constraints)
            "MAX_EDIT_DISTANCE": (
                "max_edit_distance",
                lambda v: self._parse_int_range(v, 1, 3, "max_edit_distance"),
            ),
            "MAX_SUGGESTIONS": (
                "max_suggestions",
                lambda v: self._parse_int_min(v, 1, "max_suggestions"),
            ),
            "USE_CONTEXT_CHECKER": ("use_context_checker", self._parse_bool),
            "USE_PHONETIC": ("use_phonetic", self._parse_bool),
            "USE_NER": ("use_ner", self._parse_bool),
            "USE_RULE_BASED_VALIDATION": ("use_rule_based_validation", self._parse_bool),
            "WORD_ENGINE": ("word_engine", str),
            "FALLBACK_TO_EMPTY_PROVIDER": ("fallback_to_empty_provider", self._parse_bool),
            # POS tagger settings
            "POS_TAGGER_TYPE": ("pos_tagger.tagger_type", str),
            "POS_TAGGER_BEAM_WIDTH": (
                "pos_tagger.viterbi_beam_width",
                lambda v: self._parse_int_min(v, 1, "viterbi_beam_width"),
            ),
            "POS_TAGGER_MODEL_NAME": ("pos_tagger.model_name", str),
            # SymSpell settings
            "SYMSPELL_PREFIX_LENGTH": (
                "symspell.prefix_length",
                lambda v: self._parse_int_range(v, 4, 10, "prefix_length"),
            ),
            "SYMSPELL_BEAM_WIDTH": (
                "symspell.beam_width",
                lambda v: self._parse_int_min(v, 1, "beam_width"),
            ),
            "SYMSPELL_USE_WEIGHTED_DISTANCE": (
                "symspell.use_weighted_distance",
                self._parse_bool,
            ),
            # N-gram context settings (thresholds are 0.0-1.0 probabilities)
            "NGRAM_BIGRAM_THRESHOLD": (
                "ngram_context.bigram_threshold",
                lambda v: self._parse_float_range(v, 0.0, 1.0, "bigram_threshold"),
            ),
            "NGRAM_TRIGRAM_THRESHOLD": (
                "ngram_context.trigram_threshold",
                lambda v: self._parse_float_range(v, 0.0, 1.0, "trigram_threshold"),
            ),
            "NGRAM_RERANK_LEFT_WEIGHT": (
                "ngram_context.rerank_left_weight",
                lambda v: self._parse_float_range(v, 0.0, 1.0, "rerank_left_weight"),
            ),
            "NGRAM_RERANK_RIGHT_WEIGHT": (
                "ngram_context.rerank_right_weight",
                lambda v: self._parse_float_range(v, 0.0, 1.0, "rerank_right_weight"),
            ),
            # Phonetic settings
            "PHONETIC_BYPASS_THRESHOLD": (
                "phonetic.phonetic_bypass_threshold",
                lambda v: self._parse_float_range(v, 0.0, 1.0, "phonetic_bypass_threshold"),
            ),
            "PHONETIC_EXTRA_DISTANCE": (
                "phonetic.phonetic_extra_distance",
                lambda v: self._parse_int_range(v, 0, 3, "phonetic_extra_distance"),
            ),
            # Ranker settings
            "RANKER_UNIFIED_BASE_TYPE": (
                "ranker.unified_base_ranker_type",
                str,
            ),
            "RANKER_ENABLE_TARGETED_RERANK_HINTS": (
                "ranker.enable_targeted_rerank_hints",
                self._parse_bool,
            ),
            "RANKER_ENABLE_TARGETED_CANDIDATE_INJECTIONS": (
                "ranker.enable_targeted_candidate_injections",
                self._parse_bool,
            ),
            "RANKER_ENABLE_TARGETED_GRAMMAR_COMPLETION_TEMPLATES": (
                "ranker.enable_targeted_grammar_completion_templates",
                self._parse_bool,
            ),
            # Provider settings
            "DATABASE_PATH": ("provider_config.database_path", str),
            "POOL_MIN_SIZE": (
                "provider_config.pool_min_size",
                lambda v: self._parse_int_min(v, 0, "pool_min_size"),
            ),
            "POOL_MAX_SIZE": (
                "provider_config.pool_max_size",
                lambda v: self._parse_int_min(v, 1, "pool_max_size"),
            ),
            # Validation settings
            "ALLOW_EXTENDED_MYANMAR": (
                "validation.allow_extended_myanmar",
                self._parse_bool,
            ),
        }

        for env_key, (config_key, converter) in env_mappings.items():
            env_var = f"{self.env_prefix}{env_key}"
            value = os.environ.get(env_var)

            if value is not None:
                try:
                    converted_value = converter(value)
                    self._set_nested_key(config, config_key, converted_value)
                except (ValueError, TypeError, InvalidConfigError) as err:
                    converter_name = getattr(converter, "__name__", str(converter))
                    raise InvalidConfigError(
                        f"Invalid value for {env_var}: '{value}'. "
                        f"Expected type: {converter_name}. Error: {err}"
                    ) from err

        return config

    def _parse_bool(self, value: str) -> bool:
        """
        Parse boolean from string.

        Args:
            value: String value (true, false, 1, 0, yes, no)

        Returns:
            Boolean value

        Raises:
            ValueError: If invalid boolean string
        """
        lower_value = value.lower().strip()
        if lower_value in ("true", "1", "yes", "on"):
            return True
        elif lower_value in ("false", "0", "no", "off"):
            return False
        else:
            raise InvalidConfigError(f"Cannot parse as boolean: {value}")

    def _parse_int_range(self, value: str, min_val: int, max_val: int, name: str) -> int:
        """
        Parse integer with range validation.

        Args:
            value: String value to parse
            min_val: Minimum allowed value (inclusive)
            max_val: Maximum allowed value (inclusive)
            name: Parameter name for error messages

        Returns:
            Validated integer value

        Raises:
            ValueError: If value is not an integer or out of range
        """
        try:
            int_val = int(value)
        except ValueError as err:
            raise InvalidConfigError(f"Expected integer for {name}, got: {value}") from err

        if int_val < min_val or int_val > max_val:
            raise InvalidConfigError(
                f"{name} must be between {min_val} and {max_val}, got: {int_val}"
            )
        return int_val

    def _parse_int_min(self, value: str, min_val: int, name: str) -> int:
        """
        Parse integer with minimum value validation.

        Args:
            value: String value to parse
            min_val: Minimum allowed value (inclusive)
            name: Parameter name for error messages

        Returns:
            Validated integer value

        Raises:
            ValueError: If value is not an integer or below minimum
        """
        try:
            int_val = int(value)
        except ValueError as err:
            raise InvalidConfigError(f"Expected integer for {name}, got: {value}") from err

        if int_val < min_val:
            raise InvalidConfigError(f"{name} must be >= {min_val}, got: {int_val}")
        return int_val

    def _parse_float_range(self, value: str, min_val: float, max_val: float, name: str) -> float:
        """
        Parse float with range validation.

        Args:
            value: String value to parse
            min_val: Minimum allowed value (inclusive)
            max_val: Maximum allowed value (inclusive)
            name: Parameter name for error messages

        Returns:
            Validated float value

        Raises:
            ValueError: If value is not a float or out of range
        """
        try:
            float_val = float(value)
        except ValueError as err:
            raise InvalidConfigError(f"Expected number for {name}, got: {value}") from err

        if float_val < min_val or float_val > max_val:
            raise InvalidConfigError(
                f"{name} must be between {min_val} and {max_val}, got: {float_val}"
            )
        return float_val

    def _set_nested_key(self, config: dict[str, Any], key: str, value: Any) -> None:
        """
        Set nested configuration key using dot notation.

        Args:
            config: Configuration dictionary
            key: Dot-separated key (e.g., "pos_tagger.tagger_type")
            value: Value to set

        Example:
            >>> config = {}
            >>> self._set_nested_key(config, "pos_tagger.tagger_type", "transformer")
            >>> config
            {"pos_tagger": {"tagger_type": "transformer"}}
        """
        parts = key.split(".")
        current = config

        # Navigate to parent
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        # Set final value
        current[parts[-1]] = value

    def _transform_file_config(
        self, file_config: dict[str, Any], current_profile: ProfileName | None
    ) -> dict[str, Any]:
        """
        Transform config file format to SpellCheckerConfig format.

        Handles special keys:
        - 'preset': Load profile as base (if not already using a profile)
        - 'database': Map to provider_config.database_path

        Args:
            file_config: Raw config from file
            current_profile: Current profile being used (if any)

        Returns:
            Transformed config dictionary compatible with SpellCheckerConfig
        """
        transformed = file_config.copy()

        # Handle 'preset' key - load profile if specified and not already using one
        if "preset" in transformed:
            preset_name = transformed.pop("preset")
            if preset_name and not current_profile:
                # Load profile as base, then merge file config on top
                profile_dict = self._profile_to_dict(preset_name)
                transformed = self._merge_dicts(profile_dict, transformed)

        # Handle 'database' key - map to provider_config.database_path
        if "database" in transformed:
            database_path = transformed.pop("database")
            if database_path:
                # Ensure provider_config exists
                if "provider_config" not in transformed:
                    transformed["provider_config"] = {}
                if not isinstance(transformed["provider_config"], dict):
                    transformed["provider_config"] = {}
                # Set database_path (file config takes precedence)
                if "database_path" not in transformed["provider_config"]:
                    transformed["provider_config"]["database_path"] = database_path

        return transformed

    def _merge_dicts(self, base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
        """
        Deep merge two dictionaries (overlay takes precedence).

        Args:
            base: Base dictionary
            overlay: Overlay dictionary (higher precedence)

        Returns:
            Merged dictionary
        """
        result = base.copy()

        for key, value in overlay.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # Recursively merge nested dictionaries
                result[key] = self._merge_dicts(result[key], value)
            else:
                # Override with overlay value
                result[key] = value

        return result


def load_config(
    profile: ProfileName | None = None,
    config_file: str | Path | None = None,
    use_env: bool = True,
    **overrides: Any,
) -> SpellCheckerConfig:
    """
    Convenience function to load configuration.

    Args:
        profile: Profile name (development, production, testing, fast, accurate)
        config_file: Path to YAML/JSON configuration file
        use_env: Whether to read environment variables (default: True)
        **overrides: Additional keyword arguments as overrides

    Returns:
        SpellCheckerConfig: Loaded configuration

    Example:
        >>> from myspellchecker.core.config.loader import load_config
        >>>
        >>> # Load from profile
        >>> config = load_config(profile="fast")
        >>>
        >>> # Load from file with overrides
        >>> config = load_config(
        ...     config_file="myconfig.yaml",
        ...     max_edit_distance=3,
        ...     max_suggestions=10
        ... )
    """
    loader = ConfigLoader()
    return loader.load(
        profile=profile,
        config_file=config_file,
        use_env=use_env,
        overrides=overrides if overrides else None,
    )
