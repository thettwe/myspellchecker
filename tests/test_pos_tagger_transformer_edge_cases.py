from unittest.mock import MagicMock, patch

import pytest

# We need to import the module to patch it
from myspellchecker.algorithms import pos_tagger_transformer


def test_transformer_tagger_init_error():
    """Test import error if transformers missing."""
    with patch.object(pos_tagger_transformer, "_HAS_TRANSFORMERS", False):
        from myspellchecker.algorithms.pos_tagger_transformer import TransformerPOSTagger

        with pytest.raises(ImportError):
            TransformerPOSTagger()


def test_transformer_tagger_init_success():
    """Test successful initialization."""
    with patch.object(pos_tagger_transformer, "_HAS_TRANSFORMERS", True):
        # We need to patch pipeline where it is looked up in the module
        # Use create=True since pipeline may not exist if transformers is not installed
        with patch.object(pos_tagger_transformer, "pipeline", create=True) as mock_pipeline:
            from myspellchecker.algorithms.pos_tagger_transformer import TransformerPOSTagger

            tagger = TransformerPOSTagger()
            assert tagger.model_name == TransformerPOSTagger.DEFAULT_MODEL
            mock_pipeline.assert_called()


def test_transformer_tagger_load_error():
    """Test model load failure."""
    with patch.object(pos_tagger_transformer, "_HAS_TRANSFORMERS", True):
        # Use OSError as that's what the code catches and re-raises as ValueError
        # Use create=True since pipeline may not exist if transformers is not installed
        with patch.object(
            pos_tagger_transformer, "pipeline", side_effect=OSError("Load failed"), create=True
        ):
            from myspellchecker.algorithms.pos_tagger_transformer import TransformerPOSTagger

            with pytest.raises(ValueError, match="Failed to load model"):
                TransformerPOSTagger()


def test_tag_word():
    """Test single word tagging."""
    with patch.object(pos_tagger_transformer, "_HAS_TRANSFORMERS", True):
        with patch.object(pos_tagger_transformer, "pipeline", create=True) as mock_pipeline:
            from myspellchecker.algorithms.pos_tagger_transformer import TransformerPOSTagger

            # Setup the mock pipeline instance
            mock_pipe_instance = MagicMock()
            mock_pipeline.return_value = mock_pipe_instance

            # pipeline("word") returns a list of dicts
            mock_pipe_instance.return_value = [{"entity_group": "NOUN"}]

            tagger = TransformerPOSTagger()
            tagger._pipeline = mock_pipe_instance  # Ensure the instance is attached

            tag = tagger.tag_word("test")
            assert tag == "NOUN"

            # Test empty result
            mock_pipe_instance.return_value = []
            assert tagger.tag_word("test") == "UNK"

            # Test empty input
            assert tagger.tag_word("") == "UNK"

            # Test exception fallback - use RuntimeError as that's what the code catches
            mock_pipe_instance.side_effect = RuntimeError("Fail")
            assert tagger.tag_word("test") == "UNK"


def test_tag_sequence():
    """Test sequence tagging."""
    with patch.object(pos_tagger_transformer, "_HAS_TRANSFORMERS", True):
        with patch.object(pos_tagger_transformer, "pipeline", create=True) as mock_pipeline:
            from myspellchecker.algorithms.pos_tagger_transformer import TransformerPOSTagger

            mock_pipe_instance = MagicMock()
            mock_pipeline.return_value = mock_pipe_instance

            # Setup mock results for "word1 word2"
            mock_pipe_instance.return_value = [
                {"word": "word1", "entity_group": "NOUN"},
                {"word": "word2", "entity_group": "VERB"},
            ]

            tagger = TransformerPOSTagger()
            tagger._pipeline = mock_pipe_instance

            tags = tagger.tag_sequence(["word1", "word2"])
            assert tags == ["NOUN", "VERB"]

            # Test empty input
            assert tagger.tag_sequence([]) == []

            # Test exception fallback - use RuntimeError as that's what the code catches
            # Reset side effect
            mock_pipe_instance.side_effect = RuntimeError("Fail")
            with patch.object(tagger, "tag_word", return_value="UNK") as mock_tag_word:
                tags = tagger.tag_sequence(["word1", "word2"])
                assert tags == ["UNK", "UNK"]
                assert mock_tag_word.call_count == 2


def test_map_results_to_words():
    """Test logic for aligning tokens to words."""
    with patch.object(pos_tagger_transformer, "_HAS_TRANSFORMERS", True):
        with patch.object(pos_tagger_transformer, "pipeline", create=True):
            from myspellchecker.algorithms.pos_tagger_transformer import TransformerPOSTagger

            tagger = TransformerPOSTagger()

            words = ["unmatched", "matched", "split"]
            results = [
                {"word": "other", "entity_group": "X"},  # Mismatch
                {"word": "matched", "entity_group": "V"},  # Match
                {"word": "sp", "entity_group": "N"},  # Split match
                {"word": "##lit", "entity_group": "N"},  # continuation
            ]

            tags = tagger._map_results_to_words(words, results)
            # Bug #1228 fix: "matched" no longer falsely matches "unmatched"
            # "unmatched" -> UNK (no match), "matched" -> V, "split" -> N (via "sp" BPE)
            assert tags == ["UNK", "V", "N"]


def test_tag_word_with_confidence():
    """Test tagging with confidence."""
    with patch.object(pos_tagger_transformer, "_HAS_TRANSFORMERS", True):
        with patch.object(pos_tagger_transformer, "pipeline", create=True) as mock_pipeline:
            from myspellchecker.algorithms.pos_tagger_transformer import TransformerPOSTagger

            mock_pipe_instance = MagicMock()
            mock_pipeline.return_value = mock_pipe_instance

            mock_pipe_instance.return_value = [{"entity_group": "NOUN", "score": 0.9}]

            tagger = TransformerPOSTagger()
            tagger._pipeline = mock_pipe_instance

            pred = tagger.tag_word_with_confidence("word")
            assert pred.tag == "NOUN"
            assert pred.confidence == 0.9

            # Exception - use RuntimeError as that's what the code catches
            mock_pipe_instance.side_effect = RuntimeError("Err")
            pred = tagger.tag_word_with_confidence("word")
            assert pred.tag == "UNK"
            assert pred.confidence == 0.0


def test_tag_sequence_with_confidence():
    """Test sequence tagging with confidence."""
    with patch.object(pos_tagger_transformer, "_HAS_TRANSFORMERS", True):
        with patch.object(pos_tagger_transformer, "pipeline", create=True) as mock_pipeline:
            from myspellchecker.algorithms.pos_tagger_transformer import TransformerPOSTagger

            mock_pipe_instance = MagicMock()
            mock_pipeline.return_value = mock_pipe_instance

            # Note: transformer pipeline output usually has 'score', 'entity_group', 'word'
            mock_pipe_instance.return_value = [{"entity_group": "N", "score": 0.9, "word": "w1"}]

            tagger = TransformerPOSTagger()
            tagger._pipeline = mock_pipe_instance

            preds = tagger.tag_sequence_with_confidence(["w1"])
            assert len(preds) == 1
            assert preds[0].tag == "N"
            assert preds[0].confidence == 0.9

            # Exception fallback - use RuntimeError as that's what the code catches
            mock_pipe_instance.side_effect = RuntimeError("Fail")
            with patch.object(tagger, "tag_word_with_confidence") as mock_single:
                tagger.tag_sequence_with_confidence(["w1"])
                mock_single.assert_called()


def test_properties():
    """Test properties."""
    with patch.object(pos_tagger_transformer, "_HAS_TRANSFORMERS", True):
        with patch.object(pos_tagger_transformer, "pipeline", create=True):
            from myspellchecker.algorithms.pos_tagger_transformer import TransformerPOSTagger

            tagger = TransformerPOSTagger()
            assert tagger.supports_batch is True
            assert tagger.is_fork_safe is True  # CPU mode (device=-1) is fork-safe
            # Compare values or names to avoid Enum identity issues across imports
            assert tagger.tagger_type.value == "transformer"
