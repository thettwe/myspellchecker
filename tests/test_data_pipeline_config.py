"""Extended tests for data_pipeline/config.py to boost coverage."""

import pytest


class TestPipelineConfigValidation:
    """Test PipelineConfig validation in __post_init__."""

    def test_batch_size_too_small(self):
        """Test that batch_size < 100 raises error."""
        from myspellchecker.core.exceptions import ConfigurationError
        from myspellchecker.data_pipeline.config import PipelineConfig

        with pytest.raises(ConfigurationError) as exc_info:
            PipelineConfig(batch_size=50)

        assert "batch_size" in str(exc_info.value)
        assert "100" in str(exc_info.value)

    def test_num_shards_zero(self):
        """Test that num_shards < 1 raises error."""
        from myspellchecker.core.exceptions import ConfigurationError
        from myspellchecker.data_pipeline.config import PipelineConfig

        with pytest.raises(ConfigurationError) as exc_info:
            PipelineConfig(num_shards=0)

        assert "num_shards" in str(exc_info.value)
        assert "1" in str(exc_info.value)

    def test_num_workers_zero(self):
        """Test that num_workers < 1 (but not None) raises error."""
        from myspellchecker.core.exceptions import ConfigurationError
        from myspellchecker.data_pipeline.config import PipelineConfig

        with pytest.raises(ConfigurationError) as exc_info:
            PipelineConfig(num_workers=0)

        assert "num_workers" in str(exc_info.value)
        assert "1" in str(exc_info.value)

    def test_num_workers_negative(self):
        """Test that negative num_workers raises error."""
        from myspellchecker.core.exceptions import ConfigurationError
        from myspellchecker.data_pipeline.config import PipelineConfig

        with pytest.raises(ConfigurationError) as exc_info:
            PipelineConfig(num_workers=-1)

        assert "num_workers" in str(exc_info.value)

    def test_num_workers_none_allowed(self):
        """Test that num_workers=None is allowed (auto-detect)."""
        from myspellchecker.data_pipeline.config import PipelineConfig

        config = PipelineConfig(num_workers=None)
        assert config.num_workers is None

    def test_min_frequency_zero(self):
        """Test that min_frequency < 1 raises error."""
        from myspellchecker.core.exceptions import ConfigurationError
        from myspellchecker.data_pipeline.config import PipelineConfig

        with pytest.raises(ConfigurationError) as exc_info:
            PipelineConfig(min_frequency=0)

        assert "min_frequency" in str(exc_info.value)
        assert "1" in str(exc_info.value)

    def test_invalid_word_engine(self):
        """Test that invalid word_engine raises error."""
        from myspellchecker.core.exceptions import ConfigurationError
        from myspellchecker.data_pipeline.config import PipelineConfig

        with pytest.raises(ConfigurationError) as exc_info:
            PipelineConfig(word_engine="invalid_engine")

        assert "word_engine" in str(exc_info.value)
        assert "crf" in str(exc_info.value)
        assert "myword" in str(exc_info.value)

    def test_negative_disk_space_check(self):
        """Test that negative disk_space_check_mb raises error."""
        from myspellchecker.core.exceptions import ConfigurationError
        from myspellchecker.data_pipeline.config import PipelineConfig

        with pytest.raises(ConfigurationError) as exc_info:
            PipelineConfig(disk_space_check_mb=-100)

        assert "disk_space_check_mb" in str(exc_info.value)
        assert "0" in str(exc_info.value)

    def test_invalid_pos_tagger_type(self):
        """Test that invalid pos_tagger type raises error."""
        from myspellchecker.core.exceptions import ConfigurationError
        from myspellchecker.data_pipeline.config import PipelineConfig

        with pytest.raises(ConfigurationError) as exc_info:
            PipelineConfig(pos_tagger="not_a_config")  # Should be POSTaggerConfig

        assert "pos_tagger" in str(exc_info.value)
        assert "POSTaggerConfig" in str(exc_info.value)

    def test_valid_pos_tagger_config(self):
        """Test that valid POSTaggerConfig is accepted."""
        from myspellchecker.core.config import POSTaggerConfig
        from myspellchecker.data_pipeline.config import PipelineConfig

        pos_config = POSTaggerConfig(tagger_type="rule_based")
        config = PipelineConfig(pos_tagger=pos_config)
        assert config.pos_tagger == pos_config


class TestPipelineConfigWithOverrides:
    """Test PipelineConfig.with_overrides method."""

    def test_with_overrides_single_field(self):
        """Test overriding a single field."""
        from myspellchecker.data_pipeline.config import PipelineConfig

        base = PipelineConfig(batch_size=10000)
        new_config = base.with_overrides(batch_size=50000)

        assert base.batch_size == 10000  # Original unchanged
        assert new_config.batch_size == 50000  # New config has override

    def test_with_overrides_multiple_fields(self):
        """Test overriding multiple fields."""
        from myspellchecker.data_pipeline.config import PipelineConfig

        base = PipelineConfig()
        new_config = base.with_overrides(batch_size=25000, num_shards=40, min_frequency=100)

        assert new_config.batch_size == 25000
        assert new_config.num_shards == 40
        assert new_config.min_frequency == 100

    def test_with_overrides_preserves_other_fields(self):
        """Test that non-overridden fields are preserved."""
        from myspellchecker.data_pipeline.config import PipelineConfig

        base = PipelineConfig(
            batch_size=10000,
            num_shards=10,
            word_engine="crf",
            keep_intermediate=True,
        )
        new_config = base.with_overrides(batch_size=20000)

        assert new_config.batch_size == 20000
        assert new_config.num_shards == 10  # Preserved
        assert new_config.word_engine == "crf"  # Preserved
        assert new_config.keep_intermediate is True  # Preserved


class TestOtherConfigs:
    """Test other config dataclasses."""

    def test_ingester_config_defaults(self):
        """Test IngesterConfig default values."""
        from myspellchecker.data_pipeline.config import IngesterConfig

        config = IngesterConfig()
        assert config.encoding == "utf-8"
        assert config.skip_empty_lines is True
        assert config.normalize_unicode is True

    def test_ingester_config_custom(self):
        """Test IngesterConfig with custom values."""
        from myspellchecker.data_pipeline.config import IngesterConfig

        config = IngesterConfig(
            batch_size=5000,
            encoding="utf-16",
            skip_empty_lines=False,
            normalize_unicode=False,
        )
        assert config.batch_size == 5000
        assert config.encoding == "utf-16"
        assert config.skip_empty_lines is False
        assert config.normalize_unicode is False

    def test_segmenter_config_defaults(self):
        """Test SegmenterConfig default values."""
        from myspellchecker.data_pipeline.config import SegmenterConfig

        config = SegmenterConfig()
        assert config.word_engine == "myword"
        assert config.num_workers is None
        assert config.enable_pos_tagging is True
        assert config.chunk_size == 50000

    def test_segmenter_config_custom(self):
        """Test SegmenterConfig with custom values."""
        from myspellchecker.data_pipeline.config import SegmenterConfig

        config = SegmenterConfig(
            batch_size=20000,
            word_engine="crf",
            num_workers=4,
            enable_pos_tagging=False,
            chunk_size=100000,
        )
        assert config.batch_size == 20000
        assert config.word_engine == "crf"
        assert config.num_workers == 4
        assert config.enable_pos_tagging is False
        assert config.chunk_size == 100000

    def test_frequency_builder_config_defaults(self):
        """Test FrequencyBuilderConfig default values."""
        from myspellchecker.data_pipeline.config import FrequencyBuilderConfig

        config = FrequencyBuilderConfig()
        assert config.min_syllable_frequency == 1
        assert config.min_word_frequency == 1
        assert config.min_bigram_count == 1
        assert config.min_trigram_count == 1
        assert config.smoothing_factor == 0.0

    def test_frequency_builder_config_custom(self):
        """Test FrequencyBuilderConfig with custom values."""
        from myspellchecker.data_pipeline.config import FrequencyBuilderConfig

        config = FrequencyBuilderConfig(
            min_syllable_frequency=5,
            min_word_frequency=10,
            min_bigram_count=3,
            min_trigram_count=2,
            smoothing_factor=0.01,
        )
        assert config.min_syllable_frequency == 5
        assert config.min_word_frequency == 10
        assert config.min_bigram_count == 3
        assert config.min_trigram_count == 2
        assert config.smoothing_factor == 0.01

    def test_packager_config_defaults(self):
        """Test PackagerConfig default values."""
        from myspellchecker.data_pipeline.config import PackagerConfig

        config = PackagerConfig()
        assert config.create_indexes is True
        assert config.vacuum_after_build is True
        assert config.enable_fts is False

    def test_packager_config_custom(self):
        """Test PackagerConfig with custom values."""
        from myspellchecker.data_pipeline.config import PackagerConfig

        config = PackagerConfig(
            batch_size=20000,
            create_indexes=False,
            vacuum_after_build=False,
            enable_fts=True,
        )
        assert config.batch_size == 20000
        assert config.create_indexes is False
        assert config.vacuum_after_build is False
        assert config.enable_fts is True


class TestPipelineConfigValidWordEngines:
    """Test valid word engine options."""

    def test_word_engine_crf(self):
        """Test crf word engine is valid."""
        from myspellchecker.data_pipeline.config import PipelineConfig

        config = PipelineConfig(word_engine="crf")
        assert config.word_engine == "crf"

    def test_word_engine_myword(self):
        """Test myword word engine is valid."""
        from myspellchecker.data_pipeline.config import PipelineConfig

        config = PipelineConfig(word_engine="myword")
        assert config.word_engine == "myword"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
