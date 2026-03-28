"""
Comprehensive tests for Pydantic v2 configuration validation.

Tests all validation features introduced during the Pydantic migration:
- Field validation constraints (ge, le, gt, lt)
- Model validators (cross-field validation)
- XOR validation (mutually exclusive fields)
- Conditional validation
- Deprecation warnings
- ValidationError messaging
"""

import warnings

import pytest
from pydantic import ValidationError

from myspellchecker.core.config import (
    NgramContextConfig,
    POSTaggerConfig,
    ProviderConfig,
    SemanticConfig,
    SpellCheckerConfig,
    SymSpellConfig,
    ValidationConfig,
)
from myspellchecker.core.config.validation_configs import ConnectionPoolConfig


class TestSymSpellConfig:
    """Test SymSpellConfig field validation."""

    def test_valid_config(self):
        """Test creating config with valid values."""
        config = SymSpellConfig(prefix_length=7, beam_width=50)
        assert config.prefix_length == 7
        assert config.beam_width == 50

    def test_prefix_length_validation(self):
        """Test prefix_length must be >= 1."""
        with pytest.raises(ValidationError) as exc:
            SymSpellConfig(prefix_length=0)
        assert "greater than or equal to 1" in str(exc.value)

    def test_negative_prefix_length(self):
        """Test negative prefix_length is rejected."""
        with pytest.raises(ValidationError) as exc:
            SymSpellConfig(prefix_length=-1)
        assert "greater than or equal to 1" in str(exc.value)

    def test_count_threshold_validation(self):
        """Test count_threshold must be >= 0."""
        with pytest.raises(ValidationError) as exc:
            SymSpellConfig(count_threshold=-1)
        assert "greater than or equal to 0" in str(exc.value)

    def test_beam_width_validation(self):
        """Test beam_width must be >= 1."""
        with pytest.raises(ValidationError) as exc:
            SymSpellConfig(beam_width=0)
        assert "greater than or equal to 1" in str(exc.value)

    def test_max_word_length_validation(self):
        """Test max_word_length must be >= 1."""
        with pytest.raises(ValidationError) as exc:
            SymSpellConfig(max_word_length=0)
        assert "greater than or equal to 1" in str(exc.value)

    def test_damerau_cache_size_validation(self):
        """Test damerau_cache_size must be >= 0."""
        config = SymSpellConfig(damerau_cache_size=0)  # 0 is valid (disabled)
        assert config.damerau_cache_size == 0

        with pytest.raises(ValidationError):
            SymSpellConfig(damerau_cache_size=-1)

    def test_extra_fields_forbidden(self):
        """Test that extra fields are rejected."""
        with pytest.raises(ValidationError) as exc:
            SymSpellConfig(unknown_field="value")
        assert "extra inputs are not permitted" in str(exc.value).lower()

    def test_defaults(self):
        """Test default values."""
        config = SymSpellConfig()
        assert config.prefix_length == 10  # raised for Myanmar multisyllabic words
        assert config.count_threshold == 50
        assert config.max_word_length == 15
        assert config.beam_width == 25


class TestNgramContextConfig:
    """Test NgramContextConfig field and model validation."""

    def test_valid_config(self):
        """Test creating config with valid values."""
        config = NgramContextConfig(
            threshold=0.01,
            edit_distance_weight=0.6,
            probability_weight=0.4,
        )
        assert config.threshold == 0.01
        assert config.edit_distance_weight == 0.6
        assert config.probability_weight == 0.4

    def test_threshold_validation(self):
        """Test threshold must be > 0 (no upper bound)."""
        with pytest.raises(ValidationError):
            NgramContextConfig(threshold=0.0)  # Must be > 0

        with pytest.raises(ValidationError):
            NgramContextConfig(threshold=-0.1)

        # Should accept values > 0, including > 1
        config = NgramContextConfig(threshold=1.5)
        assert config.threshold == 1.5

    def test_weight_sum_warning(self):
        """Test warning when weights don't sum to 1.0."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = NgramContextConfig(
                edit_distance_weight=0.7,
                probability_weight=0.2,  # Sum = 0.9, not 1.0
            )
            assert len(w) == 1
            assert "should ideally sum to 1.0" in str(w[0].message)
            assert config.edit_distance_weight == 0.7

    def test_weight_sum_no_warning(self):
        """Test no warning when weights sum to 1.0."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = NgramContextConfig(
                edit_distance_weight=0.6,
                probability_weight=0.4,  # Sum = 1.0
            )
            assert len(w) == 0
            assert config.edit_distance_weight == 0.6

    def test_candidate_limit_validation(self):
        """Test candidate_limit must be >= 1."""
        with pytest.raises(ValidationError):
            NgramContextConfig(candidate_limit=0)

    def test_defaults(self):
        """Test default values."""
        config = NgramContextConfig()
        assert config.threshold == 0.01
        assert config.edit_distance_weight == 0.6
        assert config.probability_weight == 0.4


class TestSemanticConfig:
    """Test SemanticConfig XOR validation."""

    def test_path_based_config(self):
        """Test valid path-based configuration."""
        config = SemanticConfig(
            model_path="/path/to/model.onnx",
            tokenizer_path="/path/to/tokenizer.json",
        )
        assert config.model_path == "/path/to/model.onnx"
        assert config.tokenizer_path == "/path/to/tokenizer.json"

    def test_instance_based_config(self):
        """Test valid instance-based configuration."""
        # Note: We can't test with real instances easily, but we can test None
        config = SemanticConfig(model=None, tokenizer=None)
        assert config.model is None
        assert config.tokenizer is None

    def test_mixed_config_rejected(self):
        """Test that mixing path and instance config is rejected."""
        with pytest.raises(ValidationError) as exc:
            SemanticConfig(
                model_path="/path/to/model.onnx",
                model="dummy",  # Can't mix!
            )
        assert "Cannot mix path-based" in str(exc.value)

    def test_model_path_without_tokenizer_path(self):
        """Test that model_path requires tokenizer_path."""
        with pytest.raises(ValidationError) as exc:
            SemanticConfig(model_path="/path/to/model.onnx")
        assert "tokenizer_path must be provided" in str(exc.value)

    def test_tokenizer_path_without_model_path(self):
        """Test that tokenizer_path requires model_path."""
        with pytest.raises(ValidationError) as exc:
            SemanticConfig(tokenizer_path="/path/to/tokenizer.json")
        assert "model_path must be provided" in str(exc.value)

    def test_model_without_tokenizer(self):
        """Test that model instance requires tokenizer instance."""
        with pytest.raises(ValidationError) as exc:
            SemanticConfig(model="dummy")
        assert "tokenizer instance must be provided" in str(exc.value)

    def test_tokenizer_without_model(self):
        """Test that tokenizer instance requires model instance."""
        with pytest.raises(ValidationError) as exc:
            SemanticConfig(tokenizer="dummy")
        assert "model instance must be provided" in str(exc.value)

    def test_defaults(self):
        """Test default values."""
        config = SemanticConfig()
        assert config.model_path is None
        assert config.tokenizer_path is None


class TestPOSTaggerConfig:
    """Test POSTaggerConfig Literal types and conditional validation."""

    def test_valid_tagger_types(self):
        """Test all valid tagger types."""
        for tagger_type in ["rule_based", "transformer", "viterbi", "custom"]:
            config = POSTaggerConfig(tagger_type=tagger_type)
            assert config.tagger_type == tagger_type

    def test_invalid_tagger_type(self):
        """Test that invalid tagger types are rejected."""
        with pytest.raises(ValidationError) as exc:
            POSTaggerConfig(tagger_type="invalid")
        assert "Input should be" in str(exc.value)

    def test_transformer_config(self):
        """Test transformer-specific configuration."""
        config = POSTaggerConfig(
            tagger_type="transformer",
            model_name="custom/model",
            device=0,
        )
        assert config.model_name == "custom/model"
        assert config.device == 0

    def test_device_values(self):
        """Test device field accepts various values."""
        config_cpu = POSTaggerConfig(device=-1)  # CPU
        assert config_cpu.device == -1

        config_gpu0 = POSTaggerConfig(device=0)  # GPU 0
        assert config_gpu0.device == 0

        config_gpu1 = POSTaggerConfig(device=1)  # GPU 1
        assert config_gpu1.device == 1

    def test_viterbi_tagger_type(self):
        """Test selecting viterbi tagger type."""
        config = POSTaggerConfig(tagger_type="viterbi")
        assert config.tagger_type == "viterbi"
        # Viterbi-specific config (beam_width, emission_weight) is in POSTaggerConfig

    def test_defaults(self):
        """Test default values."""
        config = POSTaggerConfig()
        assert config.tagger_type == "rule_based"
        assert config.device == -1  # CPU


class TestValidationConfig:
    """Test ValidationConfig field validation."""

    def test_valid_config(self):
        """Test creating config with valid values."""
        config = ValidationConfig(max_syllable_length=15)
        assert config.max_syllable_length == 15

    def test_max_syllable_length_validation(self):
        """Test max_syllable_length must be >= 1."""
        with pytest.raises(ValidationError):
            ValidationConfig(max_syllable_length=0)

    def test_zawgyi_confidence_threshold_validation(self):
        """Test zawgyi_confidence_threshold must be between 0 and 1."""
        with pytest.raises(ValidationError):
            ValidationConfig(zawgyi_confidence_threshold=-0.1)

        with pytest.raises(ValidationError):
            ValidationConfig(zawgyi_confidence_threshold=1.5)

        # Valid boundaries
        config1 = ValidationConfig(zawgyi_confidence_threshold=0.0)
        assert config1.zawgyi_confidence_threshold == 0.0

        config2 = ValidationConfig(zawgyi_confidence_threshold=1.0)
        assert config2.zawgyi_confidence_threshold == 1.0

    def test_defaults(self):
        """Test default values."""
        config = ValidationConfig()
        assert config.max_syllable_length == 12
        assert config.use_zawgyi_detection is True
        assert config.zawgyi_confidence_threshold == 0.95


class TestProviderConfig:
    """Test ProviderConfig field validation."""

    def test_valid_config(self):
        """Test creating config with valid values."""
        config = ProviderConfig(cache_size=2048, pool_max_size=10)
        assert config.cache_size == 2048
        assert config.pool_max_size == 10

    def test_cache_size_validation(self):
        """Test cache_size must be >= 0."""
        config = ProviderConfig(cache_size=0)  # 0 is valid (disabled)
        assert config.cache_size == 0

        with pytest.raises(ValidationError):
            ProviderConfig(cache_size=-1)

    def test_defaults(self):
        """Test default values."""
        config = ProviderConfig()
        assert config.cache_size == 1024
        assert config.pool_min_size == 1
        assert config.pool_max_size == 5

    def test_pool_min_size_validation(self):
        """Test pool_min_size must be >= 0 and <= 100."""
        config = ProviderConfig(pool_min_size=0)  # 0 is valid
        assert config.pool_min_size == 0

        with pytest.raises(ValidationError):
            ProviderConfig(pool_min_size=-1)

        with pytest.raises(ValidationError):
            ProviderConfig(pool_min_size=101)

    def test_pool_max_size_validation(self):
        """Test pool_max_size must be >= 1 and <= 100."""
        config = ProviderConfig(pool_max_size=1)  # 1 is valid
        assert config.pool_max_size == 1

        with pytest.raises(ValidationError):
            ProviderConfig(pool_max_size=0)

        with pytest.raises(ValidationError):
            ProviderConfig(pool_max_size=101)

    def test_pool_max_size_must_be_gte_min_size(self):
        """Test pool_max_size >= pool_min_size cross-field validation."""
        # Valid: max >= min
        config = ProviderConfig(pool_min_size=2, pool_max_size=5)
        assert config.pool_min_size == 2
        assert config.pool_max_size == 5

        # Valid: max == min
        config = ProviderConfig(pool_min_size=3, pool_max_size=3)
        assert config.pool_min_size == config.pool_max_size

        # Invalid: max < min
        with pytest.raises(ValidationError) as exc:
            ProviderConfig(pool_min_size=10, pool_max_size=5)
        assert "pool_max_size" in str(exc.value)
        assert "pool_min_size" in str(exc.value)

    def test_pool_timeout_validation(self):
        """Test pool_timeout must be > 0 and <= 60."""
        config = ProviderConfig(pool_timeout=0.1)  # Valid minimum
        assert config.pool_timeout == 0.1

        config = ProviderConfig(pool_timeout=60.0)  # Valid maximum
        assert config.pool_timeout == 60.0

        with pytest.raises(ValidationError):
            ProviderConfig(pool_timeout=0.0)

        with pytest.raises(ValidationError):
            ProviderConfig(pool_timeout=61.0)


class TestConnectionPoolConfig:
    """Test ConnectionPoolConfig field validation."""

    def test_valid_config(self):
        """Test creating config with valid values."""
        config = ConnectionPoolConfig(min_size=2, max_size=10, timeout=5.0)
        assert config.min_size == 2
        assert config.max_size == 10
        assert config.timeout == 5.0

    def test_min_size_validation(self):
        """Test min_size must be >= 0 and <= 100."""
        config = ConnectionPoolConfig(min_size=0)  # 0 is valid
        assert config.min_size == 0

        with pytest.raises(ValidationError):
            ConnectionPoolConfig(min_size=-1)

        with pytest.raises(ValidationError):
            ConnectionPoolConfig(min_size=101)

    def test_max_size_validation(self):
        """Test max_size must be >= 1 and <= 100."""
        # Use min_size=1 to allow max_size=1
        config = ConnectionPoolConfig(min_size=1, max_size=1)
        assert config.max_size == 1

        with pytest.raises(ValidationError):
            ConnectionPoolConfig(min_size=0, max_size=0)

        with pytest.raises(ValidationError):
            ConnectionPoolConfig(max_size=101)

    def test_max_size_must_be_gte_min_size(self):
        """Test max_size >= min_size cross-field validation."""
        # Valid: max >= min
        config = ConnectionPoolConfig(min_size=2, max_size=5)
        assert config.min_size == 2
        assert config.max_size == 5

        # Valid: max == min
        config = ConnectionPoolConfig(min_size=3, max_size=3)
        assert config.min_size == config.max_size

        # Invalid: max < min
        with pytest.raises(ValidationError) as exc:
            ConnectionPoolConfig(min_size=10, max_size=5)
        assert "max_size" in str(exc.value)
        assert "min_size" in str(exc.value)

    def test_timeout_validation(self):
        """Test timeout must be > 0 and <= 60."""
        config = ConnectionPoolConfig(timeout=0.1)  # Valid minimum
        assert config.timeout == 0.1

        config = ConnectionPoolConfig(timeout=60.0)  # Valid maximum
        assert config.timeout == 60.0

        with pytest.raises(ValidationError):
            ConnectionPoolConfig(timeout=0.0)

        with pytest.raises(ValidationError):
            ConnectionPoolConfig(timeout=61.0)

    def test_max_connection_age_validation(self):
        """Test max_connection_age must be > 0."""
        config = ConnectionPoolConfig(max_connection_age=1.0)  # Valid
        assert config.max_connection_age == 1.0

        with pytest.raises(ValidationError):
            ConnectionPoolConfig(max_connection_age=0.0)

        with pytest.raises(ValidationError):
            ConnectionPoolConfig(max_connection_age=-1.0)

    def test_defaults(self):
        """Test default values."""
        config = ConnectionPoolConfig()
        assert config.min_size == 2
        assert config.max_size == 10
        assert config.timeout == 5.0
        assert config.max_connection_age == 3600.0
        assert config.check_same_thread is False

    def test_extra_fields_forbidden(self):
        """Test that extra fields are rejected."""
        with pytest.raises(ValidationError) as exc:
            ConnectionPoolConfig(unknown_field="value")
        assert "extra inputs are not permitted" in str(exc.value).lower()


class TestValidationConsistency:
    """Test validation behavior consistency across similar config classes."""

    def test_pool_validation_error_format_consistency(self):
        """Test that pool size validation errors have consistent format."""
        # ProviderConfig error format
        with pytest.raises(ValidationError) as exc_provider:
            ProviderConfig(pool_min_size=10, pool_max_size=5)

        provider_error = str(exc_provider.value)
        assert "pool_max_size" in provider_error
        assert "pool_min_size" in provider_error
        assert "must be >=" in provider_error

        # ConnectionPoolConfig error format
        with pytest.raises(ValidationError) as exc_pool:
            ConnectionPoolConfig(min_size=10, max_size=5)

        pool_error = str(exc_pool.value)
        assert "max_size" in pool_error
        assert "min_size" in pool_error
        assert "must be >=" in pool_error

    def test_all_configs_use_extra_forbid(self):
        """Test all config classes reject extra fields."""
        configs_to_test = [
            (SymSpellConfig, {"unknown": "value"}),
            (NgramContextConfig, {"unknown": "value"}),
            (ValidationConfig, {"unknown": "value"}),
            (ProviderConfig, {"unknown": "value"}),
            (ConnectionPoolConfig, {"unknown": "value"}),
            (POSTaggerConfig, {"unknown": "value"}),
        ]

        for config_cls, extra_data in configs_to_test:
            with pytest.raises(ValidationError) as exc:
                config_cls(**extra_data)
            assert "extra" in str(exc.value).lower(), (
                f"{config_cls.__name__} should forbid extra fields"
            )


class TestSpellCheckerConfig:
    """Test SpellCheckerConfig main configuration."""

    def test_valid_config(self):
        """Test creating config with valid values."""
        config = SpellCheckerConfig(
            max_edit_distance=3,
            max_suggestions=10,
            use_phonetic=False,
        )
        assert config.max_edit_distance == 3
        assert config.max_suggestions == 10
        assert config.use_phonetic is False

    def test_max_edit_distance_validation(self):
        """Test max_edit_distance must be between 1 and 3."""
        with pytest.raises(ValidationError):
            SpellCheckerConfig(max_edit_distance=0)

        with pytest.raises(ValidationError):
            SpellCheckerConfig(max_edit_distance=4)

        # Valid boundaries
        config1 = SpellCheckerConfig(max_edit_distance=1)
        assert config1.max_edit_distance == 1

        config3 = SpellCheckerConfig(max_edit_distance=3)
        assert config3.max_edit_distance == 3

    def test_max_suggestions_validation(self):
        """Test max_suggestions must be >= 1."""
        with pytest.raises(ValidationError):
            SpellCheckerConfig(max_suggestions=0)  # Must be at least 1

        with pytest.raises(ValidationError):
            SpellCheckerConfig(max_suggestions=-1)

        # Valid minimum
        config = SpellCheckerConfig(max_suggestions=1)
        assert config.max_suggestions == 1

    def test_word_engine_literal(self):
        """Test word_engine only accepts specific values."""
        for engine in ["myword", "crf"]:
            config = SpellCheckerConfig(word_engine=engine)
            assert config.word_engine == engine

        with pytest.raises(ValidationError) as exc:
            SpellCheckerConfig(word_engine="invalid")
        assert "Input should be" in str(exc.value)

    def test_nested_configs(self):
        """Test nested configuration objects."""
        config = SpellCheckerConfig(
            symspell=SymSpellConfig(prefix_length=8),
            ngram_context=NgramContextConfig(threshold=0.05),
        )
        assert config.symspell.prefix_length == 8
        assert config.ngram_context.threshold == 0.05

    def test_nested_config_default_factory(self):
        """Test that nested configs use default_factory to avoid sharing."""
        config1 = SpellCheckerConfig()
        config2 = SpellCheckerConfig()

        # Modify one config's nested object to a non-default value
        config1.symspell.prefix_length = 8

        # Should not affect the other config
        assert (
            config2.symspell.prefix_length == 10
        )  # default value (raised for Myanmar multisyllabic words)

    def test_model_validate(self):
        """Test model_validate method for dict input."""
        data = {
            "max_edit_distance": 2,
            "max_suggestions": 10,
            "symspell": {"prefix_length": 8},
        }
        config = SpellCheckerConfig.model_validate(data)
        assert config.max_edit_distance == 2
        assert config.symspell.prefix_length == 8

    def test_model_dump(self):
        """Test model_dump method for serialization."""
        config = SpellCheckerConfig(max_edit_distance=3, use_phonetic=False)
        data = config.model_dump()
        assert isinstance(data, dict)
        assert data["max_edit_distance"] == 3
        assert data["use_phonetic"] is False

    def test_defaults(self):
        """Test default values."""
        config = SpellCheckerConfig()
        assert config.max_edit_distance == 2
        assert config.max_suggestions == 5
        assert config.use_phonetic is True
        assert config.use_context_checker is True
        assert config.use_ner is True
        assert config.use_rule_based_validation is True
        assert config.word_engine == "myword"


class TestConfigInteroperability:
    """Test that configs work together correctly."""

    def test_full_config_from_dict(self):
        """Test creating full config from nested dictionary."""
        data = {
            "max_edit_distance": 3,
            "symspell": {
                "prefix_length": 8,
                "beam_width": 100,
            },
            "ngram_context": {
                "threshold": 0.05,
                "edit_distance_weight": 0.7,
                "probability_weight": 0.3,
            },
            "validation": {
                "max_syllable_length": 15,
            },
        }

        config = SpellCheckerConfig.model_validate(data)
        assert config.max_edit_distance == 3
        assert config.symspell.prefix_length == 8
        assert config.symspell.beam_width == 100
        assert config.ngram_context.threshold == 0.05
        assert config.validation.max_syllable_length == 15

    def test_partial_config_with_defaults(self):
        """Test that partial config uses defaults for missing values."""
        data = {
            "max_edit_distance": 1,
            "symspell": {"prefix_length": 5},
        }

        config = SpellCheckerConfig.model_validate(data)
        assert config.max_edit_distance == 1
        assert config.symspell.prefix_length == 5
        # Defaults should fill in
        assert config.max_suggestions == 5  # default
        assert config.symspell.beam_width == 25  # default

    def test_validation_error_has_context(self):
        """Test that validation errors provide helpful context."""
        data = {
            "max_edit_distance": 10,  # Invalid: > 3
            "symspell": {
                "prefix_length": -1,  # Invalid: < 1
            },
        }

        with pytest.raises(ValidationError) as exc:
            SpellCheckerConfig.model_validate(data)

        error_str = str(exc.value)
        # Should mention both errors
        assert "max_edit_distance" in error_str
        assert "prefix_length" in error_str
