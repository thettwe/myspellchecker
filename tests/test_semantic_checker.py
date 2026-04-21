import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from myspellchecker.core.exceptions import ModelLoadError

# Mock imports
with patch.dict(
    sys.modules,
    {
        "onnxruntime": MagicMock(),
        "transformers": MagicMock(),
        "tokenizers": MagicMock(),
        "torch": MagicMock(),
    },
):
    from myspellchecker.algorithms.semantic_checker import (
        HFTokenizerWrapper,
        PyTorchInferenceSession,
        SemanticChecker,
    )


def test_semantic_checker_init_error():
    """Test initialization error when neither path nor object provided."""
    with pytest.raises(ModelLoadError, match="must be provided"):
        SemanticChecker()


def test_semantic_checker_init_objects():
    """Test initialization with provided objects."""
    model = MagicMock()
    tokenizer = MagicMock()

    # Mock get_inputs/outputs for init
    model.get_inputs.return_value = [MagicMock(name="input_ids")]
    model.get_outputs.return_value = [MagicMock(name="logits")]

    # Mock tokenizer.token_to_id
    tokenizer.token_to_id.return_value = 100

    with patch.object(SemanticChecker, "_validate_model_dimensions"):
        checker = SemanticChecker(model=model, tokenizer=tokenizer)
        assert checker.session == model
        assert checker.tokenizer == tokenizer


def test_hf_tokenizer_wrapper():
    """Test HFTokenizerWrapper methods."""
    mock_hf = MagicMock()
    # Mock encode output
    mock_hf.return_value = {"input_ids": [1, 2, 3], "offset_mapping": [(0, 1), (1, 2), (2, 3)]}
    mock_hf.convert_tokens_to_ids.return_value = 5
    mock_hf.decode.return_value = "decoded"
    mock_hf.unk_token_id = 999

    wrapper = HFTokenizerWrapper(mock_hf)

    # Test encode
    res = wrapper.encode("text")
    assert res.ids == [1, 2, 3]
    assert res.offsets == [(0, 1), (1, 2), (2, 3)]

    # Test token_to_id
    assert wrapper.token_to_id("token") == 5
    mock_hf.convert_tokens_to_ids.return_value = 999  # UNK
    assert wrapper.token_to_id("unk") is None

    # Test decode
    assert wrapper.decode([1, 2]) == "decoded"


def test_predict_mask_simple():
    """Test predict_mask with simple replacement."""
    model = MagicMock()
    # input/output names
    model.get_inputs.return_value = [type("I", (), {"name": "input_ids"})()]
    model.get_outputs.return_value = [type("O", (), {"name": "logits"})()]

    # Mock run output: [Batch, Seq, Vocab]
    # Assume seq len 3, mask at index 1
    # Vocab size 30000 (realistic size to pass validation)
    logits = np.zeros((1, 3, 30000), dtype=np.float32)
    # Set high score for token 5 at position 1
    logits[0, 1, 5] = 10.0
    model.run.return_value = [logits]

    tokenizer = MagicMock()
    tokenizer.token_to_id.return_value = 99  # MASK ID
    # Encode "A [MASK] B" -> [10, 99, 11]
    enc_res = MagicMock()
    enc_res.ids = [10, 99, 11]
    enc_res.attention_mask = [1, 1, 1]
    tokenizer.encode.return_value = enc_res
    tokenizer.decode.side_effect = lambda ids: f"word_{ids[0]}"

    with patch.object(SemanticChecker, "_validate_model_dimensions"):
        checker = SemanticChecker(model=model, tokenizer=tokenizer)

    # Mock np.where to return mask index
    with patch("numpy.where", return_value=(np.array([0]), np.array([1]))):
        preds = checker.predict_mask("A B C", "B", top_k=1)
        assert len(preds) == 1
        assert preds[0][0] == "word_5"
        assert preds[0][1] == 10.0


def test_predict_mask_myanmar_aligned():
    """Test predict_mask with Myanmar word alignment."""
    model = MagicMock()
    model.get_inputs.return_value = [type("I", (), {"name": "input_ids"})()]
    model.get_outputs.return_value = [type("O", (), {"name": "logits"})()]

    # Mock logits with realistic vocab size
    logits = np.zeros((1, 5, 30000), dtype=np.float32)
    logits[0, 1, 5] = 10.0  # Pos 1 -> ID 5
    logits[0, 2, 6] = 10.0  # Pos 2 -> ID 6
    model.run.return_value = [logits]

    tokenizer = MagicMock()
    tokenizer.token_to_id.return_value = 99

    # Sentence: "က ခ ဂ"
    # Target: "ခ" (assume it spans tokens at index 1 and 2)
    enc_res = MagicMock()
    enc_res.ids = [10, 20, 21, 11, 12]
    # Offsets: "က"=(0,1), "ခ"=(2,3) but split into two tokens?
    # Let's say "ခ" is at char 2-3.
    # Tokens: 0:(0,1)[က], 1:(2,3)[ခ part1], 2:(2,3)[ခ part2], ...
    enc_res.offsets = [(0, 1), (2, 3), (2, 3), (4, 5), (5, 6)]
    tokenizer.encode.return_value = enc_res

    # Mock decode
    def mock_decode(ids):
        if ids == [5]:
            return "part1"
        if ids == [6]:
            return "part2"
        if ids == [5, 6]:
            return "ခ"  # Combined
        return "unk"

    tokenizer.decode.side_effect = mock_decode

    with patch.object(SemanticChecker, "_validate_model_dimensions"):
        checker = SemanticChecker(model=model, tokenizer=tokenizer)

    # "က ခ ဂ", target="ခ"
    preds = checker.predict_mask("က ခ ဂ", "ခ", top_k=1)
    # Logic: alignment finds indices 1, 2.
    # Masks them. Runs model.
    # Decodes multi-token prediction.
    # Should combine top from 1 and 2.

    assert len(preds) == 1
    assert preds[0][0] == "ခ"


def test_scan_sentence():
    """Test scan_sentence method."""
    model = MagicMock()
    model.get_inputs.return_value = [type("I", (), {"name": "input_ids"})()]
    model.get_outputs.return_value = [type("O", (), {"name": "logits"})()]

    tokenizer = MagicMock()
    tokenizer.token_to_id.return_value = 99

    with patch.object(SemanticChecker, "_validate_model_dimensions"):
        checker = SemanticChecker(model=model, tokenizer=tokenizer)

    # Mock predict_mask to return suggestions
    # Word "error" -> suggestions ["correct"]
    def mock_predict(sent, word, top_k=5, occurrence=0):
        if word == "မှား":
            return [("မှန်", 10.0), ("ကောင်း", 5.0)]
        return [(word, 10.0)]  # Correct words predict themselves

    checker.predict_mask = mock_predict

    # "ဒါ မှား တယ်" -> "မှား" should be flagged
    errors = checker.scan_sentence("ဒါ မှား တယ်", ["ဒါ", "မှား", "တယ်"])

    assert len(errors) == 1
    idx, word, sugs, conf = errors[0]
    assert idx == 1
    assert word == "မှား"
    assert sugs[0] == "မှန်"
    assert conf > 0


def test_prefix_skip_uses_margin_not_binary_prefix_match():
    """Prefix evidence should only skip when score gap is small."""
    model = MagicMock()
    model.get_inputs.return_value = [type("I", (), {"name": "input_ids"})()]
    model.get_outputs.return_value = [type("O", (), {"name": "logits"})()]
    tokenizer = MagicMock()
    tokenizer.token_to_id.return_value = 99

    with patch.object(SemanticChecker, "_validate_model_dimensions"):
        checker = SemanticChecker(model=model, tokenizer=tokenizer)

    # Prefix candidate exists, but top candidate outranks by a large margin:
    # do not skip.
    assert not checker._should_skip_due_to_prefix_evidence(
        "ခွဲ",
        [("ကွဲ", 9.3), ("ခွဲခြမ်း", 7.8), ("ဖြေ", 6.5)],
        top_n=5,
    )

    # Prefix candidate remains near the top score: skip.
    assert checker._should_skip_due_to_prefix_evidence(
        "ခွဲ",
        [("ခွဲခြမ်း", 9.2), ("ကွဲ", 8.9), ("ဖြေ", 7.0)],
        top_n=5,
    )


def test_prefix_skip_ignores_morpheme_boundary_suffixes():
    """Predictions differing by visarga/asat/dot-below should not count as prefix evidence.

    When the MLM predicts a compound like "အတိုင်းအတာ" for target "အတိုင်",
    the suffix "းအတာ" starts with visarga — this signals a missing-visarga
    error, not a legitimate prefix relationship.
    """
    model = MagicMock()
    model.get_inputs.return_value = [type("I", (), {"name": "input_ids"})()]
    model.get_outputs.return_value = [type("O", (), {"name": "logits"})()]
    tokenizer = MagicMock()
    tokenizer.token_to_id.return_value = 99

    with patch.object(SemanticChecker, "_validate_model_dimensions"):
        checker = SemanticChecker(model=model, tokenizer=tokenizer)

    # Visarga suffix: "အတိုင်းအတာ" starts with "အတိုင်" but suffix is "းအတာ"
    # — should NOT skip (visarga boundary).
    assert not checker._should_skip_due_to_prefix_evidence(
        "အတိုင်",
        [("အတိုင်းအတာ", 9.5), ("ပြောင်း", 7.0)],
        top_n=5,
    )

    # Dot-below suffix: prediction differs by ့ — should NOT skip.
    assert not checker._should_skip_due_to_prefix_evidence(
        "ခွင်",
        [("ခွင့်ပြု", 9.5), ("ဖွင့်", 7.0)],
        top_n=5,
    )

    # Asat suffix: prediction differs by ် — should NOT skip.
    assert not checker._should_skip_due_to_prefix_evidence(
        "အကျိုး",
        [("အကျိုး်ပြု", 9.5), ("ဖြေ", 7.0)],
        top_n=5,
    )

    # Legitimate prefix (consonant suffix) should still skip when margin is close.
    assert checker._should_skip_due_to_prefix_evidence(
        "ကျွန်",
        [("ကျွန်တော်", 9.2), ("ကွဲ", 8.0)],
        top_n=5,
    )


def test_score_mask_candidates_scores_explicit_candidates():
    """Explicit candidate scoring should work even outside decode top-k."""
    model = MagicMock()
    model.get_inputs.return_value = [type("I", (), {"name": "input_ids"})()]
    model.get_outputs.return_value = [type("O", (), {"name": "logits"})()]
    tokenizer = MagicMock()
    tokenizer.token_to_id.return_value = 99

    with patch.object(SemanticChecker, "_validate_model_dimensions"):
        checker = SemanticChecker(model=model, tokenizer=tokenizer)

    checker._get_mask_logits = MagicMock(
        return_value=(np.array([[[0.1, 3.0, 8.5, 1.2]]], dtype=np.float32), [0])
    )
    checker._encode_candidate_token_ids = MagicMock(
        side_effect=lambda text: [1] if text == "orig" else [2]
    )

    scores = checker.score_mask_candidates(
        sentence="dummy sentence",
        target_word="orig",
        candidates=["orig", "cand"],
    )

    assert "orig" in scores
    assert "cand" in scores
    assert scores["cand"] > scores["orig"]


def test_is_hf_model_name():
    """Test helper for detecting HF model names."""
    # Since it's an instance method, we need an instance
    # But we can cheat or assume the logic is simple.
    # Actually, we mocked the class so we can't test private methods easily unless we instantiate.
    # Let's instantiate a dummy.

    model = MagicMock()
    model.get_inputs.return_value = [MagicMock()]
    model.get_outputs.return_value = [MagicMock()]
    tokenizer = MagicMock()
    tokenizer.token_to_id.return_value = 1

    with patch.object(SemanticChecker, "_validate_model_dimensions"):
        checker = SemanticChecker(model=model, tokenizer=tokenizer)
    assert checker._is_hf_model_name("org/model")
    assert checker._is_hf_model_name("bert-base-uncased")
    # Paths with slashes are considered potential HF names (if not found as files)
    assert checker._is_hf_model_name("local/path/model.onnx")
    # Simple name without keywords or slashes
    assert not checker._is_hf_model_name("my_custom_model")


def test_get_model_logit_scale():
    """Test logit scale logic."""
    model = MagicMock()
    model.get_inputs.return_value = [MagicMock()]
    model.get_outputs.return_value = [MagicMock()]
    tokenizer = MagicMock()
    tokenizer.token_to_id.return_value = 1

    with patch.object(SemanticChecker, "_validate_model_dimensions"):
        checker = SemanticChecker(model=model, tokenizer=tokenizer)

    checker._model_name = "xlm-roberta-base"
    assert checker._get_model_logit_scale() == 10.0

    checker._model_name = "bert-base"
    assert checker._get_model_logit_scale() == 50.0


def test_calibrate_confidence():
    """Test confidence calibration."""
    model = MagicMock()
    model.get_inputs.return_value = [MagicMock()]
    model.get_outputs.return_value = [MagicMock()]
    tokenizer = MagicMock()
    tokenizer.token_to_id.return_value = 1

    with patch.object(SemanticChecker, "_validate_model_dimensions"):
        checker = SemanticChecker(model=model, tokenizer=tokenizer)

    # Test low score
    assert checker._calibrate_confidence(-5.0, 10.0) == 0.1

    # Test high score
    # 20.0 / 10.0 = 2.0.  1.0 - (0.2 / 3.0) = 1 - 0.066 = 0.933
    conf = checker._calibrate_confidence(20.0, 10.0)
    assert 0.9 < conf < 1.0


def test_pytorch_inference_session():
    """Test PyTorch inference wrapper."""
    pytest.importorskip("torch", reason="PyTorch required for inference session test")
    with patch("myspellchecker.algorithms.semantic_checker.torch"):
        mock_model = MagicMock()
        mock_output = MagicMock()
        mock_output.logits = MagicMock()
        mock_output.logits.cpu().numpy.return_value = np.array([1, 2])
        mock_model.return_value = mock_output

        session = PyTorchInferenceSession(mock_model)

        inputs = {"input_ids": np.array([1, 2])}
        res = session.run([], inputs)

        assert len(res) == 1
        mock_model.assert_called()
        mock_model.eval.assert_called()


def test_duplicate_word_occurrence():
    """Test that scan_sentence correctly handles duplicate words with occurrence parameter.

    Bug #1232: Previously, scan_sentence always masked the first occurrence of duplicate words,
    causing incorrect semantic context for subsequent occurrences.
    """
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    # Mock model setup
    mock_model.get_inputs.return_value = [MagicMock(name="input_ids")]
    mock_model.get_outputs.return_value = [MagicMock(name="logits")]
    mock_tokenizer.token_to_id.return_value = 100

    with patch.object(SemanticChecker, "_validate_model_dimensions"):
        checker = SemanticChecker(model=mock_model, tokenizer=mock_tokenizer)

        # Mock the _get_word_token_alignment to track occurrences
        alignment_calls = []

        def mock_alignment(sentence, word, occurrence=0):
            alignment_calls.append((sentence, word, occurrence))
            # Return mock token alignment
            return ([1, 2, 3, 4, 5], 1, 2)

        checker._get_word_token_alignment = mock_alignment

        # Mock other dependencies
        checker._cached_encode = MagicMock(
            return_value=MagicMock(ids=[1, 2, 3, 4, 5], offsets=[(0, 3), (3, 6), (6, 9)])
        )
        mock_model.run.return_value = [
            np.array([[[0.1, 0.9, 0.1]]])  # Mock logits
        ]
        mock_tokenizer.decode.return_value = "word"

        # Test sentence with duplicate words
        sentence = "ကလေး ကလေး သွားတယ်"
        words = ["ကလေး", "ကလေး", "သွားတယ်"]

        # Call scan_sentence (return value unused; we verify via alignment_calls)
        checker.scan_sentence(sentence, words)

        # Verify that occurrences were tracked correctly
        # First "ကလေး" should use occurrence=0
        # Second "ကလေး" should use occurrence=1
        assert len(alignment_calls) >= 2
        assert alignment_calls[0][2] == 0  # First occurrence
        assert alignment_calls[1][2] == 1  # Second occurrence


def test_multi_token_beam_search():
    """Test beam search for multi-token predictions (Bug #1231).

    Previously, diagonal selection was used ([0][0], [1][1], [2][2]),
    which missed optimal combinations like ([0][0], [1][2], [2][0]).
    """
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    # Mock model setup
    mock_model.get_inputs.return_value = [MagicMock(name="input_ids")]
    mock_model.get_outputs.return_value = [MagicMock(name="logits")]
    mock_tokenizer.token_to_id.return_value = 100

    with patch.object(SemanticChecker, "_validate_model_dimensions"):
        checker = SemanticChecker(model=mock_model, tokenizer=mock_tokenizer)

        # Mock tokenizer.decode to return specific words
        decode_map = {
            (1, 10): "word1",  # Optimal: [1, 10, 20]
            (1, 11): "word2",
            (1, 12): "word3",
            (2, 10): "word4",
            (2, 11): "word5",  # Diagonal: [2, 11, 21]
            (2, 12): "word6",
        }

        def mock_decode(token_ids):
            key = tuple(token_ids) if len(token_ids) > 1 else token_ids[0]
            return decode_map.get(key, "unknown")

        mock_tokenizer.decode.side_effect = mock_decode

        # Simulate multi-token predictions
        # Position 0: [1=9.0, 2=7.0, 3=5.0]
        # Position 1: [10=8.0, 11=7.0, 12=6.0]
        # Position 2: [20=9.0, 21=7.0, 22=5.0]
        # Optimal: [1, 10, 20] = 9.0 + 8.0 + 9.0 = 26.0 (avg: 8.67)
        # Diagonal: [2, 11, 21] = 7.0 + 7.0 + 7.0 = 21.0 (avg: 7.0)
        all_position_preds = [
            [(1, 9.0), (2, 7.0), (3, 5.0)],  # Position 0
            [(10, 8.0), (11, 7.0), (12, 6.0)],  # Position 1
            [(20, 9.0), (21, 7.0), (22, 5.0)],  # Position 2
        ]

        # Mock decode for all combinations
        mock_tokenizer.decode.side_effect = lambda ids: f"word_{'_'.join(map(str, ids))}"

        # Run beam search
        results = checker._beam_search_multi_token(all_position_preds, top_k=5)

        # Verify results
        assert len(results) > 0

        # Best combination should be [1, 10, 20] with avg score 26.0/3 = 8.67
        best_word, best_score = results[0]
        assert best_word == "word_1_10_20"
        assert 8.5 < best_score < 8.8  # Approximately 8.67

        # Verify diagonal combination is NOT necessarily first
        # (it might appear later with lower score)
        diagonal_word = "word_2_11_21"
        diagonal_found = any(w == diagonal_word for w, _ in results)
        if diagonal_found:
            diagonal_idx = [w for w, _ in results].index(diagonal_word)
            # Diagonal should not be first (optimal should be first)
            assert diagonal_idx > 0


def test_beam_search_deduplication():
    """Test that beam search deduplicates words."""
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    mock_model.get_inputs.return_value = [MagicMock(name="input_ids")]
    mock_model.get_outputs.return_value = [MagicMock(name="logits")]
    mock_tokenizer.token_to_id.return_value = 100

    with patch.object(SemanticChecker, "_validate_model_dimensions"):
        checker = SemanticChecker(model=mock_model, tokenizer=mock_tokenizer)

        # Mock decode to return same word for different token combinations
        mock_tokenizer.decode.side_effect = lambda ids: "same_word"

        all_position_preds = [
            [(1, 9.0), (2, 8.0)],
            [(10, 9.0), (11, 8.0)],
        ]

        results = checker._beam_search_multi_token(all_position_preds, top_k=5)

        # Should only return one "same_word" despite multiple combinations
        assert len(results) == 1
        assert results[0][0] == "same_word"


def test_encoding_result_creation():
    """Test EncodingResult dataclass."""
    from myspellchecker.algorithms.semantic_checker import EncodingResult

    result = EncodingResult(ids=[1, 2, 3], offsets=[(0, 1), (1, 2), (2, 3)])
    assert result.ids == [1, 2, 3]
    assert result.offsets == [(0, 1), (1, 2), (2, 3)]


def test_hf_tokenizer_wrapper_encode_no_offsets():
    """Test HFTokenizerWrapper encode when offset_mapping is missing."""
    mock_hf = MagicMock()
    mock_hf.return_value = {"input_ids": [101, 999, 102]}
    wrapper = HFTokenizerWrapper(mock_hf)
    result = wrapper.encode("test")
    assert result.ids == [101, 999, 102]
    assert result.offsets == []


def test_extract_model_name():
    """Test _extract_model_name for various model path formats."""
    model = MagicMock()
    model.get_inputs.return_value = [MagicMock()]
    model.get_outputs.return_value = [MagicMock()]
    tokenizer = MagicMock()
    tokenizer.token_to_id.return_value = 1

    with patch.object(SemanticChecker, "_validate_model_dimensions"):
        checker = SemanticChecker(model=model, tokenizer=tokenizer)

    assert checker._extract_model_name("xlm-roberta-base") == "xlm-roberta-base"
    assert checker._extract_model_name("facebook/xlm-roberta-base") == "xlm-roberta-base"
    assert checker._extract_model_name("/models/xlm-roberta-base") == "xlm-roberta-base"
    assert checker._extract_model_name("/models/mymodel.onnx") == "mymodel.onnx"


def test_is_myanmar_text():
    """Test _is_myanmar_text detection."""
    model = MagicMock()
    model.get_inputs.return_value = [MagicMock()]
    model.get_outputs.return_value = [MagicMock()]
    tokenizer = MagicMock()
    tokenizer.token_to_id.return_value = 1

    with patch.object(SemanticChecker, "_validate_model_dimensions"):
        checker = SemanticChecker(model=model, tokenizer=tokenizer)

    assert checker._is_myanmar_text("မြန်မာ") is True
    assert checker._is_myanmar_text("သည်") is True
    assert checker._is_myanmar_text("hello") is False
    assert checker._is_myanmar_text("123") is False
    assert checker._is_myanmar_text("မြန်မာ hello") is True


def test_clear_inference_cache():
    """Test that clear_inference_cache() empties the logits LRU cache."""
    model = MagicMock()
    tokenizer = MagicMock()
    model.get_inputs.return_value = [type("I", (), {"name": "input_ids"})()]
    model.get_outputs.return_value = [type("O", (), {"name": "logits"})()]
    tokenizer.token_to_id.return_value = 100

    with patch.object(SemanticChecker, "_validate_model_dimensions"):
        checker = SemanticChecker(model=model, tokenizer=tokenizer)

    # Manually insert an entry into the cache via set()
    checker._logits_cache.set(("test", "word", 0), (None, [1]))
    assert len(checker._logits_cache) == 1

    checker.clear_inference_cache()
    assert len(checker._logits_cache) == 0
