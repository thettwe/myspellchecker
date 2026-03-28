"""Tests for TransformerWordSegmenter.

Tests use mocking to avoid requiring the transformers package or
downloading the actual model. The pattern follows test_pos_tagger_transformer.py.
"""

from unittest.mock import MagicMock, patch

import pytest

from myspellchecker.core.exceptions import TokenizationError
from myspellchecker.tokenizers import transformer_word_segmenter


def test_init_error_without_transformers():
    """Test ImportError raised when transformers is not installed."""
    with patch.object(transformer_word_segmenter, "_HAS_TRANSFORMERS", False):
        from myspellchecker.tokenizers.transformer_word_segmenter import (
            TransformerWordSegmenter,
        )

        with pytest.raises(ImportError, match="transformers"):
            TransformerWordSegmenter()


def test_init_success():
    """Test successful initialization with mocked pipeline."""
    with patch.object(transformer_word_segmenter, "_HAS_TRANSFORMERS", True):
        with patch.object(transformer_word_segmenter, "hf_pipeline", create=True) as mock_pipeline:
            from myspellchecker.tokenizers.transformer_word_segmenter import (
                TransformerWordSegmenter,
            )

            segmenter = TransformerWordSegmenter()
            assert segmenter.model_name == TransformerWordSegmenter.DEFAULT_MODEL
            assert segmenter.device == -1
            assert segmenter.batch_size == 64  # Auto-tuned from 32 for CPU
            assert segmenter.max_length == 512
            mock_pipeline.assert_called_once()


def test_init_custom_params():
    """Test initialization with custom parameters."""
    with patch.object(transformer_word_segmenter, "_HAS_TRANSFORMERS", True):
        with patch.object(transformer_word_segmenter, "hf_pipeline", create=True) as mock_pipeline:
            from myspellchecker.tokenizers.transformer_word_segmenter import (
                TransformerWordSegmenter,
            )

            segmenter = TransformerWordSegmenter(
                model_name="custom/model",
                device=-1,
                batch_size=64,
                max_length=256,
            )
            assert segmenter.model_name == "custom/model"
            assert segmenter.batch_size == 64
            assert segmenter.max_length == 256
            mock_pipeline.assert_called_once()


def test_init_model_load_error():
    """Test ValueError raised when model fails to load."""
    with patch.object(transformer_word_segmenter, "_HAS_TRANSFORMERS", True):
        with patch.object(
            transformer_word_segmenter,
            "hf_pipeline",
            side_effect=OSError("Model not found"),
            create=True,
        ):
            from myspellchecker.tokenizers.transformer_word_segmenter import (
                TransformerWordSegmenter,
            )

            with pytest.raises(TokenizationError, match="Failed to load model"):
                TransformerWordSegmenter()


def test_init_gpu_fallback_no_cuda():
    """Test GPU fallback to CPU when CUDA not available."""
    with patch.object(transformer_word_segmenter, "_HAS_TRANSFORMERS", True):
        with patch.object(transformer_word_segmenter, "hf_pipeline", create=True):
            # Mock torch with no CUDA and no MPS
            mock_torch = MagicMock()
            mock_torch.cuda.is_available.return_value = False
            mock_torch.backends.mps.is_available.return_value = False

            with patch.dict("sys.modules", {"torch": mock_torch}):
                from myspellchecker.tokenizers.transformer_word_segmenter import (
                    TransformerWordSegmenter,
                )

                segmenter = TransformerWordSegmenter(device=0)
                # Should fall back to CPU
                assert segmenter.device == -1


def test_segment_basic():
    """Test basic single-text segmentation."""
    with patch.object(transformer_word_segmenter, "_HAS_TRANSFORMERS", True):
        with patch.object(transformer_word_segmenter, "hf_pipeline", create=True) as mock_pipeline:
            from myspellchecker.tokenizers.transformer_word_segmenter import (
                TransformerWordSegmenter,
            )

            mock_pipe_instance = MagicMock()
            mock_pipeline.return_value = mock_pipe_instance

            # Mock pipeline output: B/I labels
            mock_pipe_instance.return_value = [
                {"entity_group": "B", "word": "မြန်မာ", "start": 0, "end": 6, "score": 0.99},
                {"entity_group": "B", "word": "နိုင်ငံ", "start": 6, "end": 12, "score": 0.98},
                {"entity_group": "B", "word": "သည်", "start": 12, "end": 15, "score": 0.97},
            ]

            segmenter = TransformerWordSegmenter()
            segmenter._pipeline = mock_pipe_instance

            result = segmenter.segment("မြန်မာနိုင်ငံသည်")
            assert result == ["မြန်မာ", "နိုင်ငံ", "သည်"]


def test_segment_with_bi_merge():
    """Test B/I tag merging logic."""
    with patch.object(transformer_word_segmenter, "_HAS_TRANSFORMERS", True):
        with patch.object(transformer_word_segmenter, "hf_pipeline", create=True) as mock_pipeline:
            from myspellchecker.tokenizers.transformer_word_segmenter import (
                TransformerWordSegmenter,
            )

            mock_pipe_instance = MagicMock()
            mock_pipeline.return_value = mock_pipe_instance

            # B + I tokens should merge into one word
            mock_pipe_instance.return_value = [
                {"entity_group": "B", "word": "ကျွန်", "score": 0.99},
                {"entity_group": "I", "word": "တော်", "score": 0.98},
                {"entity_group": "B", "word": "သွား", "score": 0.97},
                {"entity_group": "I", "word": "ပါ", "score": 0.96},
                {"entity_group": "I", "word": "မယ်", "score": 0.95},
            ]

            segmenter = TransformerWordSegmenter()
            segmenter._pipeline = mock_pipe_instance

            result = segmenter.segment("ကျွန်တော်သွားပါမယ်")
            assert result == ["ကျွန်တော်", "သွားပါမယ်"]


def test_segment_empty_input():
    """Test segment with empty/whitespace input."""
    with patch.object(transformer_word_segmenter, "_HAS_TRANSFORMERS", True):
        with patch.object(transformer_word_segmenter, "hf_pipeline", create=True):
            from myspellchecker.tokenizers.transformer_word_segmenter import (
                TransformerWordSegmenter,
            )

            segmenter = TransformerWordSegmenter()

            assert segmenter.segment("") == []
            assert segmenter.segment("   ") == []


def test_segment_runtime_error_propagates():
    """Test that RuntimeError during inference propagates."""
    with patch.object(transformer_word_segmenter, "_HAS_TRANSFORMERS", True):
        with patch.object(transformer_word_segmenter, "hf_pipeline", create=True) as mock_pipeline:
            from myspellchecker.tokenizers.transformer_word_segmenter import (
                TransformerWordSegmenter,
            )

            mock_pipe_instance = MagicMock()
            mock_pipeline.return_value = mock_pipe_instance
            mock_pipe_instance.side_effect = RuntimeError("CUDA error")

            segmenter = TransformerWordSegmenter()
            segmenter._pipeline = mock_pipe_instance

            with pytest.raises(RuntimeError, match="CUDA error"):
                segmenter.segment("test")


def test_segment_batch_basic():
    """Test batch segmentation."""
    with patch.object(transformer_word_segmenter, "_HAS_TRANSFORMERS", True):
        with patch.object(transformer_word_segmenter, "hf_pipeline", create=True) as mock_pipeline:
            from myspellchecker.tokenizers.transformer_word_segmenter import (
                TransformerWordSegmenter,
            )

            mock_pipe_instance = MagicMock()
            mock_pipeline.return_value = mock_pipe_instance

            # Pipeline returns list of lists for batch
            mock_pipe_instance.return_value = [
                [
                    {"entity_group": "B", "word": "မြန်မာ", "score": 0.99},
                    {"entity_group": "B", "word": "နိုင်ငံ", "score": 0.98},
                ],
                [
                    {"entity_group": "B", "word": "စာ", "score": 0.97},
                    {"entity_group": "B", "word": "ရေး", "score": 0.96},
                ],
            ]

            segmenter = TransformerWordSegmenter()
            segmenter._pipeline = mock_pipe_instance

            result = segmenter.segment_batch(["မြန်မာနိုင်ငံ", "စာရေး"])
            assert result == [["မြန်မာ", "နိုင်ငံ"], ["စာ", "ရေး"]]


def test_segment_batch_empty():
    """Test batch segmentation with empty input."""
    with patch.object(transformer_word_segmenter, "_HAS_TRANSFORMERS", True):
        with patch.object(transformer_word_segmenter, "hf_pipeline", create=True):
            from myspellchecker.tokenizers.transformer_word_segmenter import (
                TransformerWordSegmenter,
            )

            segmenter = TransformerWordSegmenter()

            assert segmenter.segment_batch([]) == []
            assert segmenter.segment_batch(["", "  "]) == [[], []]


def test_segment_batch_with_empty_texts():
    """Test batch with mix of empty and non-empty texts."""
    with patch.object(transformer_word_segmenter, "_HAS_TRANSFORMERS", True):
        with patch.object(transformer_word_segmenter, "hf_pipeline", create=True) as mock_pipeline:
            from myspellchecker.tokenizers.transformer_word_segmenter import (
                TransformerWordSegmenter,
            )

            mock_pipe_instance = MagicMock()
            mock_pipeline.return_value = mock_pipe_instance

            # Only non-empty texts are passed to pipeline
            mock_pipe_instance.return_value = [
                [{"entity_group": "B", "word": "test", "score": 0.99}],
            ]

            segmenter = TransformerWordSegmenter()
            segmenter._pipeline = mock_pipe_instance

            result = segmenter.segment_batch(["", "test", ""])
            assert len(result) == 3
            assert result[0] == []  # Empty input
            assert result[1] == ["test"]  # Non-empty
            assert result[2] == []  # Empty input


def test_segment_batch_fallback_on_error():
    """Test batch falls back to individual processing on error."""
    with patch.object(transformer_word_segmenter, "_HAS_TRANSFORMERS", True):
        with patch.object(transformer_word_segmenter, "hf_pipeline", create=True) as mock_pipeline:
            from myspellchecker.tokenizers.transformer_word_segmenter import (
                TransformerWordSegmenter,
            )

            mock_pipe_instance = MagicMock()
            mock_pipeline.return_value = mock_pipe_instance

            # First call (batch) raises error
            mock_pipe_instance.side_effect = RuntimeError("Batch failed")

            segmenter = TransformerWordSegmenter()
            segmenter._pipeline = mock_pipe_instance

            # segment_batch falls back to individual segment() calls
            # which will also fail, producing empty lists
            result = segmenter.segment_batch(["text1", "text2"])
            assert result == [[], []]


def test_merge_bi_tags_empty():
    """Test _merge_bi_tags with empty results."""
    with patch.object(transformer_word_segmenter, "_HAS_TRANSFORMERS", True):
        with patch.object(transformer_word_segmenter, "hf_pipeline", create=True):
            from myspellchecker.tokenizers.transformer_word_segmenter import (
                TransformerWordSegmenter,
            )

            segmenter = TransformerWordSegmenter()
            assert segmenter._merge_bi_tags([]) == []


def test_merge_bi_tags_all_b():
    """Test _merge_bi_tags where all tokens are B-tagged (each is a word)."""
    with patch.object(transformer_word_segmenter, "_HAS_TRANSFORMERS", True):
        with patch.object(transformer_word_segmenter, "hf_pipeline", create=True):
            from myspellchecker.tokenizers.transformer_word_segmenter import (
                TransformerWordSegmenter,
            )

            segmenter = TransformerWordSegmenter()

            results = [
                {"entity_group": "B", "word": "word1"},
                {"entity_group": "B", "word": "word2"},
                {"entity_group": "B", "word": "word3"},
            ]
            assert segmenter._merge_bi_tags(results) == ["word1", "word2", "word3"]


def test_merge_bi_tags_b_followed_by_i():
    """Test _merge_bi_tags with B+I sequences."""
    with patch.object(transformer_word_segmenter, "_HAS_TRANSFORMERS", True):
        with patch.object(transformer_word_segmenter, "hf_pipeline", create=True):
            from myspellchecker.tokenizers.transformer_word_segmenter import (
                TransformerWordSegmenter,
            )

            segmenter = TransformerWordSegmenter()

            results = [
                {"entity_group": "B", "word": "ကျွန်"},
                {"entity_group": "I", "word": "တော်"},
                {"entity_group": "B", "word": "သွား"},
                {"entity_group": "I", "word": "ပါ"},
                {"entity_group": "I", "word": "မယ်"},
            ]
            assert segmenter._merge_bi_tags(results) == ["ကျွန်တော်", "သွားပါမယ်"]


def test_merge_bi_tags_i_without_preceding_b():
    """Test _merge_bi_tags when I appears without preceding B."""
    with patch.object(transformer_word_segmenter, "_HAS_TRANSFORMERS", True):
        with patch.object(transformer_word_segmenter, "hf_pipeline", create=True):
            from myspellchecker.tokenizers.transformer_word_segmenter import (
                TransformerWordSegmenter,
            )

            segmenter = TransformerWordSegmenter()

            # I without preceding B - should treat as new word start
            results = [
                {"entity_group": "I", "word": "orphan"},
                {"entity_group": "B", "word": "normal"},
            ]
            assert segmenter._merge_bi_tags(results) == ["orphan", "normal"]


def test_merge_bi_tags_unknown_label():
    """Test _merge_bi_tags with unknown entity_group label."""
    with patch.object(transformer_word_segmenter, "_HAS_TRANSFORMERS", True):
        with patch.object(transformer_word_segmenter, "hf_pipeline", create=True):
            from myspellchecker.tokenizers.transformer_word_segmenter import (
                TransformerWordSegmenter,
            )

            segmenter = TransformerWordSegmenter()

            results = [
                {"entity_group": "B", "word": "first"},
                {"entity_group": "X", "word": "unknown"},  # Unknown tag
                {"entity_group": "B", "word": "third"},
            ]
            # Unknown tag treated as B (new word)
            assert segmenter._merge_bi_tags(results) == ["first", "unknown", "third"]


def test_merge_bi_tags_empty_tokens_skipped():
    """Test _merge_bi_tags skips tokens with empty word."""
    with patch.object(transformer_word_segmenter, "_HAS_TRANSFORMERS", True):
        with patch.object(transformer_word_segmenter, "hf_pipeline", create=True):
            from myspellchecker.tokenizers.transformer_word_segmenter import (
                TransformerWordSegmenter,
            )

            segmenter = TransformerWordSegmenter()

            results = [
                {"entity_group": "B", "word": "word1"},
                {"entity_group": "B", "word": ""},  # Empty token
                {"entity_group": "B", "word": "  "},  # Whitespace-only
                {"entity_group": "B", "word": "word2"},
            ]
            assert segmenter._merge_bi_tags(results) == ["word1", "word2"]


def test_is_fork_safe_cpu():
    """Test is_fork_safe returns True for CPU mode."""
    with patch.object(transformer_word_segmenter, "_HAS_TRANSFORMERS", True):
        with patch.object(transformer_word_segmenter, "hf_pipeline", create=True):
            from myspellchecker.tokenizers.transformer_word_segmenter import (
                TransformerWordSegmenter,
            )

            segmenter = TransformerWordSegmenter(device=-1)
            assert segmenter.is_fork_safe is True


def test_is_fork_safe_gpu():
    """Test is_fork_safe returns False for GPU mode."""
    with patch.object(transformer_word_segmenter, "_HAS_TRANSFORMERS", True):
        with patch.object(transformer_word_segmenter, "hf_pipeline", create=True):
            from myspellchecker.tokenizers.transformer_word_segmenter import (
                TransformerWordSegmenter,
            )

            segmenter = TransformerWordSegmenter(device=-1)
            # Manually set device to simulate GPU
            segmenter.device = 0
            assert segmenter.is_fork_safe is False

            segmenter.device = 1
            assert segmenter.is_fork_safe is False


def test_default_model_constant():
    """Test DEFAULT_MODEL is the expected HuggingFace model ID."""
    with patch.object(transformer_word_segmenter, "_HAS_TRANSFORMERS", True):
        with patch.object(transformer_word_segmenter, "hf_pipeline", create=True):
            from myspellchecker.tokenizers.transformer_word_segmenter import (
                TransformerWordSegmenter,
            )

            assert (
                TransformerWordSegmenter.DEFAULT_MODEL
                == "chuuhtetnaing/myanmar-text-segmentation-model"
            )


class TestDefaultSegmenterTransformerIntegration:
    """Test transformer integration in DefaultSegmenter."""

    def test_engine_map_includes_transformer(self):
        """Test that DefaultSegmenter accepts 'transformer' engine."""
        from myspellchecker.core.constants import SEGMENTER_ENGINE_TRANSFORMER

        assert SEGMENTER_ENGINE_TRANSFORMER == "transformer"

    def test_default_segmenter_with_transformer_engine(self):
        """Test DefaultSegmenter initializes with transformer engine."""
        with patch.object(transformer_word_segmenter, "_HAS_TRANSFORMERS", True):
            with patch.object(transformer_word_segmenter, "hf_pipeline", create=True):
                from myspellchecker.segmenters.default import DefaultSegmenter

                # Should not raise
                segmenter = DefaultSegmenter(word_engine="transformer")
                assert segmenter.word_engine == "transformer"


class TestPipelineConfigTransformer:
    """Test PipelineConfig with transformer word_engine."""

    def test_config_accepts_transformer(self):
        """Test PipelineConfig validates 'transformer' as word_engine."""
        from myspellchecker.data_pipeline.config import PipelineConfig

        config = PipelineConfig(word_engine="transformer")
        assert config.word_engine == "transformer"

    def test_config_with_seg_model(self):
        """Test PipelineConfig accepts seg_model and seg_device."""
        from myspellchecker.data_pipeline.config import PipelineConfig

        config = PipelineConfig(
            word_engine="transformer",
            seg_model="custom/model",
            seg_device=0,
        )
        assert config.seg_model == "custom/model"
        assert config.seg_device == 0

    def test_config_defaults(self):
        """Test PipelineConfig defaults for seg_model and seg_device."""
        from myspellchecker.data_pipeline.config import PipelineConfig

        config = PipelineConfig()
        assert config.seg_model is None
        assert config.seg_device == -1
