"""Tests for inference backend abstractions."""

from unittest.mock import MagicMock, patch

import numpy as np

from myspellchecker.algorithms.inference_backends import (
    EncodingResult,
    HFTokenizerWrapper,
    PyTorchInferenceSession,
    RawTokenizersWrapper,
)


def test_encoding_result_stores_ids_and_offsets() -> None:
    ids = [1, 42, 99, 2]
    offsets = [(0, 0), (0, 3), (3, 6), (0, 0)]
    result = EncodingResult(ids=ids, offsets=offsets)
    assert result.ids == [1, 42, 99, 2]
    assert result.offsets == [(0, 0), (0, 3), (3, 6), (0, 0)]


def test_hf_tokenizer_wrapper_encode_returns_encoding_result() -> None:
    mock_tok = MagicMock()
    mock_tok.return_value = {
        "input_ids": [0, 123, 456, 2],
        "offset_mapping": [(0, 0), (0, 3), (3, 6), (0, 0)],
    }

    wrapper = HFTokenizerWrapper(mock_tok)
    result = wrapper.encode("ကျောင်း")

    assert result.ids == [0, 123, 456, 2]
    assert result.offsets == [(0, 0), (0, 3), (3, 6), (0, 0)]
    mock_tok.assert_called_once_with("ကျောင်း", return_offsets_mapping=True)


def test_hf_tokenizer_wrapper_encode_fallback_when_no_offset_mapping() -> None:
    mock_tok = MagicMock()
    mock_tok.return_value = {
        "input_ids": [0, 123, 2],
    }

    wrapper = HFTokenizerWrapper(mock_tok)
    result = wrapper.encode("စာ")

    assert result.ids == [0, 123, 2]
    assert result.offsets == []


def test_hf_tokenizer_wrapper_token_to_id_returns_none_for_unknown() -> None:
    mock_tok = MagicMock()
    mock_tok.unk_token_id = 3
    mock_tok.convert_tokens_to_ids.return_value = 3

    wrapper = HFTokenizerWrapper(mock_tok)
    assert wrapper.token_to_id("<unknown_token>") is None


def test_hf_tokenizer_wrapper_token_to_id_returns_id_for_known() -> None:
    mock_tok = MagicMock()
    mock_tok.unk_token_id = 3
    mock_tok.convert_tokens_to_ids.return_value = 42

    wrapper = HFTokenizerWrapper(mock_tok)
    assert wrapper.token_to_id("▁ကျောင်း") == 42


def test_hf_tokenizer_wrapper_decode_calls_underlying_tokenizer() -> None:
    mock_tok = MagicMock()
    mock_tok.decode.return_value = "ကျောင်းသား"

    wrapper = HFTokenizerWrapper(mock_tok)
    result = wrapper.decode([123, 456])
    assert result == "ကျောင်းသား"
    mock_tok.decode.assert_called_once_with([123, 456], skip_special_tokens=True)


def test_raw_tokenizers_wrapper_encode_returns_encoding_result() -> None:
    mock_encoding = MagicMock()
    mock_encoding.ids = [1, 55, 2]
    mock_encoding.offsets = [(0, 0), (0, 6), (0, 0)]

    mock_tok = MagicMock()
    mock_tok.encode.return_value = mock_encoding

    wrapper = RawTokenizersWrapper(mock_tok)
    result = wrapper.encode("စာအုပ်")

    assert result.ids == [1, 55, 2]
    assert result.offsets == [(0, 0), (0, 6), (0, 0)]


def test_pytorch_inference_session_run_with_mock_model() -> None:
    mock_model = MagicMock()
    mock_logits = MagicMock()
    mock_logits.cpu.return_value.numpy.return_value = np.array([[[0.1, 0.9]]])
    mock_output = MagicMock()
    mock_output.logits = mock_logits
    mock_model.return_value = mock_output

    with patch("myspellchecker.algorithms.inference_backends.torch") as mock_torch:
        mock_torch.no_grad.return_value.__enter__ = MagicMock()
        mock_torch.no_grad.return_value.__exit__ = MagicMock(return_value=False)
        mock_torch.tensor = MagicMock(return_value=MagicMock())
        mock_torch.long = "long"

        session = PyTorchInferenceSession(mock_model, device="cpu")
        result = session.run(
            output_names=["logits"],
            input_dict={"input_ids": np.array([[1, 2, 3]])},
        )

        assert len(result) == 1
        np.testing.assert_array_equal(result[0], np.array([[[0.1, 0.9]]]))


def test_pytorch_inference_session_exposes_input_output_specs() -> None:
    mock_model = MagicMock()

    with patch("myspellchecker.algorithms.inference_backends.torch") as mock_torch:
        mock_torch.__bool__ = MagicMock(return_value=True)
        session = PyTorchInferenceSession(mock_model, device="cpu")

        inputs = session.get_inputs()
        outputs = session.get_outputs()

        assert len(inputs) == 1
        assert inputs[0].name == "input_ids"
        assert len(outputs) == 1
        assert outputs[0].name == "logits"
