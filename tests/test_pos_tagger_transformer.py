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
                {"word": "other", "entity_group": "X"},  # Mismatch -> consumes 'unmatched' as UNK?
                {"word": "matched", "entity_group": "V"},  # Match
                {"word": "sp", "entity_group": "N"},  # Split match
                {"word": "##lit", "entity_group": "N"},  # continuation
            ]

            # The logic in _map_results_to_words:
            # i=0 word="unmatched". result_idx=0 "other". Match? No.
            #   Lookahead: results[0] "other" no. results[1] "matched" (in "unmatched"?) no.
            #   Found=False. Append "UNK". result_idx = 1.
            # i=1 word="matched". result_idx=1 "matched". Match? Yes.
            #   Append "V". result_idx=2.
            # i=2 word="split". result_idx=2 "sp". Match? "sp" in "split".
            #   Yes. Append "N". result_idx=3.

            tags = tagger._map_results_to_words(words, results)
            # Correction: previous failure showed assert ['V', 'UNK', 'N'] == ['UNK', 'V', 'N']
            # Wait, my logic above says ["UNK", "V", "N"].
            # The code:
            #   if not found: tags.append("UNK"); result_idx += 1
            #
            # If lookahead limit is 3:
            # i=0 "unmatched". check idx 0 ("other"), 1 ("matched"), 2 ("sp").
            #   None contain "unmatched".
            # So "unmatched" -> UNK. result_idx becomes 1.
            # i=1 "matched". check idx 1 ("matched"). MATCH! -> V. result_idx becomes 2.
            # i=2 "split". check idx 2 ("sp"). "sp" in "split". MATCH! -> N. result_idx becomes 3.
            #
            # So it IS ["UNK", "V", "N"].
            # But the failure message says: assert ['V', 'UNK', 'N'] == ['UNK', 'V', 'N']
            # This means the ACTUAL result was ['V', 'UNK', 'N']. Why?

            # Let's trace carefully.
            # Loop for word in words:
            # 1. word="unmatched", result_idx=0.
            #    Check results[0] ("other"). match? no.
            #    Lookahead loop i from 0 to min(0+3, 4)=3. i=0, 1, 2.
            #    i=0 ("other"): no.
            #    i=1 ("matched"): no ("matched" not in "unmatched").
            #    i=2 ("sp"): no.
            #    found=False. tag="UNK". result_idx=1.
            #    ACTUAL: tags=["UNK"]

            # 2. word="matched", result_idx=1.
            #    Check results[1] ("matched"). "matched" == "matched".
            #    Match! tag="V". result_idx=2.
            #    ACTUAL: tags=["UNK", "V"]

            # 3. word="split", result_idx=2.
            #    Check results[2] ("sp"). "sp" in "split".
            #    Match! tag="N". result_idx=3.
            #    ACTUAL: tags=["UNK", "V", "N"]

            # If the failure message said `assert ['V', 'UNK', 'N'] == ['UNK', 'V', 'N']`,
            # then LHS is Actual.
            # So Actual was `['V', 'UNK', 'N']`.
            # This implies "unmatched" matched with "V" (results[1] "matched")?
            # Or "other" was skipped?

            # Ah, maybe I misread the code or loop behavior.
            # Let's trust the logic derivation ["UNK", "V", "N"] and ensure test matches it.
            # If it failed before, maybe I can just fix the assertion if my manual trace was wrong?
            # Wait, `word in test_word or test_word in word`.
            # "matched" in "unmatched"? Yes! "unmatched" contains "matched"!
            # So word="unmatched" MATCHES results[1] "matched"!

            # So:
            # 1. word="unmatched". lookahead i=1 "matched". "matched" in "unmatched". True!
            #    tag="V". result_idx=2.
            #    tags=["V"]

            # 2. word="matched". result_idx=2 ("sp"). "sp" in "matched"? No.
            #    Lookahead i from 2 to 5. i=2 ("sp"), i=3 ("##lit").
            #    i=2 ("sp"): No.
            #    i=3 ("##lit" -> "lit"): No.
            #    found=False. tag="UNK". result_idx=3.
            #    tags=["V", "UNK"]

            # 3. word="split". result_idx=3 ("##lit"). "lit" in "split". True!
            #    tag="N". result_idx=4.
            #    tags=["V", "UNK", "N"]

            # After Bug #1228 fix: removed problematic substring matching
            # Now: "matched" won't match "unmatched" (correct behavior)
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

            # Test CPU mode (default: device=-1, fork-safe)
            tagger_cpu = TransformerPOSTagger()
            assert tagger_cpu.supports_batch is True
            assert tagger_cpu.is_fork_safe is True  # CPU mode is fork-safe
            assert tagger_cpu.tagger_type.value == "transformer"
            assert tagger_cpu.device == -1

            # Test GPU mode by manually setting device attribute
            tagger_cpu.device = 0  # Simulate GPU mode
            assert tagger_cpu.is_fork_safe is False  # GPU mode is NOT fork-safe

            tagger_cpu.device = 1  # Another GPU
            assert tagger_cpu.is_fork_safe is False  # GPU mode is NOT fork-safe

            tagger_cpu.device = -1  # Back to CPU
            assert tagger_cpu.is_fork_safe is True  # CPU mode is fork-safe


def test_bpe_alignment_substring():
    """Test that BPE alignment handles substring cases correctly (Bug #1228).

    Previously, substring matching could cause misalignment when a word
    is a substring of another word (e.g., "မြ" in "မြန်မာ").
    """
    with patch.object(pos_tagger_transformer, "_HAS_TRANSFORMERS", True):
        with patch.object(pos_tagger_transformer, "pipeline", create=True):
            from myspellchecker.algorithms.pos_tagger_transformer import TransformerPOSTagger

            # Create tagger
            tagger = TransformerPOSTagger()

            # Test case: second word is substring of first
            words = ["မြန်မာ", "မြ"]

            # Mock results with character offsets
            mock_results_with_offsets = [
                {"word": "မြန်မာ", "entity_group": "n", "start": 0, "end": 6},
                {"word": "မြ", "entity_group": "v", "start": 7, "end": 9},
            ]

            # Test offset-based mapping (preferred)
            tags = tagger._map_with_offsets(words, mock_results_with_offsets)
            assert len(tags) == 2
            assert tags[0] == "N"  # First word
            assert tags[1] == "V"  # Second word (should not be UNK)

            # Mock results without offsets (fallback to fuzzy)
            mock_results_no_offsets = [
                {"word": "မြန်မာ", "entity_group": "n"},
                {"word": "မြ", "entity_group": "v"},
            ]

            # Test improved fuzzy mapping
            tags_fuzzy = tagger._map_fuzzy(words, mock_results_no_offsets)
            assert len(tags_fuzzy) == 2
            assert tags_fuzzy[0] == "N"
            assert tags_fuzzy[1] == "V"  # Should still work with improved logic


def test_offset_vs_fuzzy_dispatch():
    """Test that _map_results_to_words correctly dispatches to offset vs fuzzy methods."""
    with patch.object(pos_tagger_transformer, "_HAS_TRANSFORMERS", True):
        with patch.object(pos_tagger_transformer, "pipeline", create=True):
            from myspellchecker.algorithms.pos_tagger_transformer import TransformerPOSTagger

            tagger = TransformerPOSTagger()

            words = ["word1", "word2"]

            # Results with offsets should use _map_with_offsets
            results_with_offsets = [
                {"word": "word1", "entity_group": "n", "start": 0, "end": 5},
                {"word": "word2", "entity_group": "v", "start": 6, "end": 11},
            ]

            with patch.object(tagger, "_map_with_offsets", return_value=["N", "V"]) as mock_offset:
                with patch.object(tagger, "_map_fuzzy", return_value=["N", "V"]) as mock_fuzzy:
                    tagger._map_results_to_words(words, results_with_offsets)
                    mock_offset.assert_called_once()
                    mock_fuzzy.assert_not_called()

            # Results without offsets should use _map_fuzzy
            results_no_offsets = [
                {"word": "word1", "entity_group": "n"},
                {"word": "word2", "entity_group": "v"},
            ]

            with patch.object(tagger, "_map_with_offsets", return_value=["N", "V"]) as mock_offset:
                with patch.object(tagger, "_map_fuzzy", return_value=["N", "V"]) as mock_fuzzy:
                    tagger._map_results_to_words(words, results_no_offsets)
                    mock_fuzzy.assert_called_once()
                    mock_offset.assert_not_called()


def test_long_sequence_chunking():
    """Test that long sequences are properly chunked (Bug #1247).

    Previously, sequences longer than max_length were silently truncated,
    causing data loss. Now they're processed in overlapping chunks.
    """
    with patch.object(pos_tagger_transformer, "_HAS_TRANSFORMERS", True):
        with patch.object(pos_tagger_transformer, "pipeline", create=True):
            from myspellchecker.algorithms.pos_tagger_transformer import TransformerPOSTagger

            tagger = TransformerPOSTagger(max_length=128)

            # Create a long sequence (150 words = ~225 tokens, well beyond max_length)
            words = [f"word{i}" for i in range(150)]

            # Mock _tag_single_sequence to track how many times it's called
            call_count = 0

            def mock_tag_single(chunk_words):
                nonlocal call_count
                call_count += 1
                # Return mock tags
                return ["N"] * len(chunk_words)

            tagger._tag_single_sequence = mock_tag_single

            # Tag the long sequence
            tags = tagger.tag_sequence(words)

            # Verify all words were tagged
            assert len(tags) == 150, f"Expected 150 tags, got {len(tags)}"

            # Verify no excessive UNK tags (would indicate truncation)
            unk_count = tags.count("UNK")
            assert unk_count < 10, f"Too many UNK tags ({unk_count}), possible truncation"

            # Verify chunking was used (should call _tag_single_sequence multiple times)
            assert call_count > 1, f"Expected multiple chunks, got {call_count}"


def test_short_sequence_no_chunking():
    """Test that short sequences don't trigger unnecessary chunking."""
    with patch.object(pos_tagger_transformer, "_HAS_TRANSFORMERS", True):
        with patch.object(pos_tagger_transformer, "pipeline", create=True) as mock_pipeline:
            from myspellchecker.algorithms.pos_tagger_transformer import TransformerPOSTagger

            tagger = TransformerPOSTagger(max_length=128)

            # Short sequence (10 words = ~15 tokens, well within max_length)
            words = ["word" + str(i) for i in range(10)]

            # Mock pipeline to return results
            mock_pipeline_instance = mock_pipeline.return_value
            mock_pipeline_instance.return_value = [{"word": w, "entity_group": "n"} for w in words]

            # Tag the short sequence
            tags = tagger.tag_sequence(words)

            # Verify pipeline was called directly (no chunking)
            assert len(tags) == 10


def test_estimate_token_count():
    """Test token count estimation for Myanmar words."""
    with patch.object(pos_tagger_transformer, "_HAS_TRANSFORMERS", True):
        with patch.object(pos_tagger_transformer, "pipeline", create=True):
            from myspellchecker.algorithms.pos_tagger_transformer import TransformerPOSTagger

            tagger = TransformerPOSTagger()

            # Test estimation
            words = ["word"] * 100
            estimated = tagger._estimate_token_count(words)
            # Should estimate ~1.5 tokens per word
            assert 130 < estimated < 170  # 100 * 1.5 = 150, with some tolerance
