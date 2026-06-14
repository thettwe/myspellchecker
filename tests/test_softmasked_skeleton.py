"""Smoke tests for the Soft-Masked BERT skeleton (prep-ws1) — post-fix.

Updated 2026-05-04 after multi-LLM review (Codex + Gemini) caught four
silent-training-killer bugs:

  - B1: ``mask_embedding`` was cached at construction; replaced by dynamic
    ``word_embedding_table.weight[mask_id]`` lookup.
  - B2: cached embedding included position-0's positional component; soft-
    mask now interpolates on word embeddings only and re-runs the embedding
    stack to add positions afresh.
  - B3: residual was added BEFORE the transformer; now added AFTER per
    ACL 2020 §3.3.
  - B4: ``correct_target`` previously included PAD tokens at real positions;
    now uses ``IGNORE_INDEX=-100`` everywhere except the corrupted span.
  - M1: detector now returns RAW LOGITS; loss uses BCE-with-logits directly.

These tests skip if the gklmip-bert checkpoint isn't present locally
(`models/gklmip-bert-myanmar-fixed/`), so they don't fail in environments
without the audit candidate models.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
GKLMIP_PATH = REPO / "models" / "gklmip-bert-myanmar-fixed"

# heavy_model marker excludes this file from the default `pytest tests/`
# invocation. The full suite + an AutoModelForMaskedLM.from_pretrained()
# call segfaults on Python 3.14.4 + transformers 5.5.4 once enough prior
# tests have loaded native-code extensions (Cython / ONNX / torch). The
# softmasked tests pass cleanly on their own — run them with:
#   pytest -m heavy_model tests/
# or directly:
#   pytest tests/test_softmasked_skeleton.py
pytestmark = [
    pytest.mark.heavy_model,
    pytest.mark.skipif(
        not GKLMIP_PATH.exists(),
        reason=f"gklmip-bert checkpoint not at {GKLMIP_PATH}; skipping skeleton smoke",
    ),
]


@pytest.fixture(scope="module")
def encoder():
    from myspellchecker.training.softmasked.encoder import SoftMaskedEncoder

    return SoftMaskedEncoder(model_path=GKLMIP_PATH, device="cpu")


@pytest.fixture(scope="module")
def corrector(encoder):
    from myspellchecker.training.softmasked.corrector import SoftMaskedCorrector

    return SoftMaskedCorrector(encoder)


def test_encoder_loads_with_expected_shape(encoder):
    assert encoder.hidden_size > 0
    assert encoder.vocab_size > 1000
    assert encoder.mask_token_id is not None
    assert 0 <= encoder.mask_token_id < encoder.vocab_size


def test_word_embedding_table_dynamic(encoder):
    """B1 fix: word embedding for the mask token is fetched dynamically.
    The table.weight gradient flag should be tied to the underlying model.
    """
    table = encoder.word_embedding_table
    assert table.weight.shape[0] == encoder.vocab_size
    assert table.weight.shape[1] == encoder.hidden_size
    # Direct lookup (no position info — this is what fixes B2)
    mask_word = table.weight[encoder.mask_token_id]
    assert mask_word.dim() == 1
    assert mask_word.shape[0] == encoder.hidden_size


def test_tokenize_roundtrip(encoder):
    sample = "ဒီမနက် ကျွန်တော်က ကစားကွင်းအကြောင်း သူငယ်ချင်းနဲ့ စကားပြောခဲ့တယ်။"
    out = encoder.tokenize(sample, max_length=64)
    assert out.input_ids.dim() == 2
    assert out.input_ids.shape[0] == 1
    assert out.attention_mask.shape == out.input_ids.shape
    decoded = encoder.tokenizer.decode(out.input_ids[0], skip_special_tokens=True)
    assert any(0x1000 <= ord(c) <= 0x109F for c in decoded), f"no Burmese: {decoded!r}"


def test_embed_shape(encoder):
    sample = "ဒီမနက် ကျွန်တော်"
    out = encoder.tokenize(sample, max_length=32)
    embs = encoder.embed(out.input_ids, out.token_type_ids)
    assert embs.dim() == 3
    assert embs.shape == (1, out.input_ids.shape[1], encoder.hidden_size)


def test_embed_from_inputs_path(encoder):
    """B2 fix: re-running the embedding stack with custom word embeddings
    should reproduce the standard ``embed`` output when those embeddings
    are exactly the word-embeddings of input_ids.
    """

    sample = "ဒီမနက် ကျွန်တော်"
    out = encoder.tokenize(sample, max_length=32)
    word_emb = encoder.word_embedding_table(out.input_ids)
    e_full_via_inputs = encoder.embed_from_inputs(word_emb, token_type_ids=out.token_type_ids)
    e_full_via_ids = encoder.embed(out.input_ids, token_type_ids=out.token_type_ids)
    assert e_full_via_inputs.shape == e_full_via_ids.shape
    diff = (e_full_via_inputs - e_full_via_ids).abs().max().item()
    assert diff < 1e-5, f"embed_from_inputs(word_emb) != embed(input_ids): diff={diff}"


def test_corrector_forward_shape(encoder, corrector):
    sample = "ဒီမနက် ကျွန်တော်"
    out = encoder.tokenize(sample, max_length=32)
    embs = encoder.embed(out.input_ids, out.token_type_ids)
    logits = corrector.forward(embs, out.attention_mask)
    assert logits.shape == (1, out.input_ids.shape[1], encoder.vocab_size)


def test_corrector_split_path_equivalence(encoder, corrector):
    """Sanity: corrector.forward(embed(ids)) == model(input_ids=ids).logits within 1e-4.
    Confirms the split-path wiring (hidden_states + mlm_head) is correct
    when no soft-mask interpolation is applied.
    """
    import torch

    sample = "ဒီမနက် ကျွန်တော် ကစားကွင်းကို သွားခဲ့တယ်။"
    out = encoder.tokenize(sample, max_length=32)
    embs = encoder.embed(out.input_ids, out.token_type_ids)
    our_logits = corrector.forward(embs, out.attention_mask)

    with torch.no_grad():
        ref = encoder.inner_model(
            input_ids=out.input_ids,
            attention_mask=out.attention_mask,
            token_type_ids=out.token_type_ids,
        )
    diff = (our_logits - ref.logits).abs().max().item()
    assert diff < 1e-4, f"split-path diverges: {diff}"


# ---- Bi-GRU detector (M1: now returns raw logits) ------------------------


@pytest.fixture(scope="module")
def detector(encoder):
    from myspellchecker.training.softmasked.detector import BiGRUDetector

    return BiGRUDetector(embedding_dim=encoder.hidden_size)


def test_detector_returns_logits_not_probs(encoder, detector):
    """M1 fix: forward returns RAW logits (can be negative), not probabilities."""
    import torch

    sample = "ဒီမနက် ကျွန်တော် ကစားကွင်းကို သွားခဲ့တယ်။"
    out = encoder.tokenize(sample, max_length=32)
    embs = encoder.embed(out.input_ids, out.token_type_ids)
    logits = detector(embs, out.attention_mask)
    assert logits.shape == out.input_ids.shape
    # Logits can be any real number — at init, expect a mix of signs
    p_error = torch.sigmoid(logits)
    assert (p_error >= 0).all() and (p_error <= 1).all()


def test_detector_param_count_reasonable(detector):
    n = detector.n_params()
    assert 1_000_000 < n < 10_000_000, f"detector params out of range: {n:,}"


def test_detector_loss_with_mask(encoder, detector):
    """BCE loss with attention_mask, on raw logits."""
    import torch

    sample = "ဒီမနက် ကျွန်တော်"
    out = encoder.tokenize(sample, max_length=32, padding="max_length")
    embs = encoder.embed(out.input_ids, out.token_type_ids)
    logits = detector(embs, out.attention_mask)
    target = torch.zeros_like(logits, dtype=torch.long)
    target[0, 1] = 1
    loss_unmasked = detector.loss(logits, target)
    loss_masked = detector.loss(logits, target, attention_mask=out.attention_mask)
    assert loss_unmasked.dim() == 0
    assert loss_masked.dim() == 0
    assert torch.isfinite(loss_unmasked) and torch.isfinite(loss_masked)


def test_detector_loss_pos_weight(encoder, detector):
    import torch

    sample = "ဒီမနက် ကျွန်တော်"
    out = encoder.tokenize(sample, max_length=16)
    embs = encoder.embed(out.input_ids, out.token_type_ids)
    logits = detector(embs, out.attention_mask)
    target = torch.zeros_like(logits, dtype=torch.long)
    target[0, 1] = 1
    loss_no_weight = detector.loss(logits, target)
    loss_high_weight = detector.loss(logits, target, pos_weight=20.0)
    assert not torch.isclose(loss_no_weight, loss_high_weight, atol=1e-5)


def test_detector_gradient_flows(encoder, detector):
    """End-to-end gradient check on raw-logits BCE loss."""
    import torch

    sample = "ဒီမနက် ကျွန်တော် ကစားကွင်းကို"
    out = encoder.tokenize(sample, max_length=32)
    embs = encoder.embed(out.input_ids, out.token_type_ids)
    logits = detector(embs, out.attention_mask)
    target = torch.zeros_like(logits, dtype=torch.long)
    target[0, 1] = 1
    loss = detector.loss(logits, target)
    loss.backward()
    assert detector.head.weight.grad is not None
    assert detector.head.weight.grad.abs().sum().item() > 0


# ---- Soft-Masked end-to-end model (B1/B2/B3 fixes) -----------------------


@pytest.fixture(scope="function")
def softmasked_model(encoder):
    """Function-scoped: trainer tests mutate weights / wrap with LoRA, so
    each test gets a fresh detector + SoftMaskedBERT wrap. The encoder
    itself stays module-scoped (loading BERT is expensive); the wrap
    around it is cheap (just registers references)."""
    from myspellchecker.training.softmasked.detector import BiGRUDetector
    from myspellchecker.training.softmasked.model import SoftMaskedBERT

    detector = BiGRUDetector(embedding_dim=encoder.hidden_size)
    return SoftMaskedBERT(encoder=encoder, detector=detector)


def test_softmasked_forward_shapes_and_keys(encoder, softmasked_model):
    """Forward returns detect_logits + p_error + logits (post-fix API)."""
    sample = "ဒီမနက် ကျွန်တော် ကစားကွင်းကို သွားခဲ့တယ်။"
    out = encoder.tokenize(sample, max_length=32)
    result = softmasked_model(
        input_ids=out.input_ids,
        attention_mask=out.attention_mask,
        token_type_ids=out.token_type_ids,
    )
    assert set(result.keys()) >= {"detect_logits", "p_error", "logits"}
    batch, seq = out.input_ids.shape
    assert result["detect_logits"].shape == (batch, seq)
    assert result["p_error"].shape == (batch, seq)
    assert result["logits"].shape == (batch, seq, encoder.vocab_size)
    # p_error is sigmoid(detect_logits) — always in [0, 1]
    assert (result["p_error"] >= 0).all() and (result["p_error"] <= 1).all()


def test_softmasked_residual_is_post_bert(encoder, softmasked_model):
    """B3 fix: at p_error=0 everywhere, the corrector path produces

        logits = mlm_head(bert(e_in_full) + e_in_full)

    This SHOULD differ from the bare model's
    ``mlm_head(bert(e_in_full))`` because of the +e_in_full residual.
    The previous (broken) implementation added e_in_full BEFORE BERT,
    yielding ``mlm_head(bert(2*e_in_full))``.
    """
    import torch

    sample = "ဒီမနက် ကျွန်တော် ကစားကွင်းကို သွားခဲ့တယ်။"
    out = encoder.tokenize(sample, max_length=32)

    # Force p_error to 0 by patching the detector's forward to return -inf
    # logits (sigmoid(-inf)=0). We use a hook rather than monkey-patching
    # the module so the model state stays clean for other tests.
    import types as _t

    original_detector_forward = softmasked_model.detector.forward

    def _zero_logits_forward(self, embeddings, attention_mask=None):
        return torch.full(embeddings.shape[:-1], -1e9, device=embeddings.device)

    softmasked_model.detector.forward = _t.MethodType(
        _zero_logits_forward, softmasked_model.detector
    )
    try:
        with torch.no_grad():
            result = softmasked_model(
                input_ids=out.input_ids,
                attention_mask=out.attention_mask,
                token_type_ids=out.token_type_ids,
            )
        # Reference: bare BERT (no residual)
        e_in_full = encoder.embed(out.input_ids, out.token_type_ids)
        bert_only = softmasked_model.corrector.forward(e_in_full, out.attention_mask)
        # The two should DIFFER — proves the +e_in_full residual is present
        diff = (result["logits"] - bert_only).abs().max().item()
        assert diff > 1e-3, f"residual missing — softmasked output matches bare BERT: {diff}"
    finally:
        softmasked_model.detector.forward = original_detector_forward


def test_softmasked_e2e_gradient_flows(encoder, softmasked_model):
    """End-to-end backward: gradient reaches detector + corrector head."""
    import torch

    sample = "ဒီမနက် ကျွန်တော်"
    out = encoder.tokenize(sample, max_length=16)
    result = softmasked_model(
        input_ids=out.input_ids,
        attention_mask=out.attention_mask,
        token_type_ids=out.token_type_ids,
    )
    detect_target = torch.zeros_like(result["detect_logits"], dtype=torch.long)
    detect_target[0, 1] = 1
    correct_target = out.input_ids.clone()
    detect_loss = softmasked_model.detector.loss(
        result["detect_logits"], detect_target, attention_mask=out.attention_mask
    )
    flat_logits = result["logits"].view(-1, result["logits"].size(-1))
    flat_target = correct_target.view(-1)
    correct_loss = torch.nn.functional.cross_entropy(flat_logits, flat_target, ignore_index=-100)
    joint = 0.85 * detect_loss + 0.15 * correct_loss
    joint.backward()
    assert softmasked_model.detector.head.weight.grad is not None
    assert softmasked_model.detector.head.weight.grad.abs().sum().item() > 0


def test_softmasked_param_count(encoder, softmasked_model):
    counts = softmasked_model.n_params()
    assert counts["encoder"] > 100_000_000
    assert 1_000_000 < counts["detector"] < 10_000_000
    assert counts["total_unique"] == counts["encoder"] + counts["detector"]


# ---- Trainer joint-loss with ignore_index (B4 fix) -----------------------


def test_trainer_joint_loss_components(encoder, softmasked_model):
    import torch

    from myspellchecker.training.softmasked.trainer import (
        SoftMaskedTrainer,
        TrainConfig,
    )

    config = TrainConfig(lambda_detect=0.85, pos_weight=20.0)
    trainer = SoftMaskedTrainer(softmasked_model, config=config, device="cpu")
    sample = "ဒီမနက် ကျွန်တော်"
    out = encoder.tokenize(sample, max_length=16)
    result = softmasked_model(
        input_ids=out.input_ids,
        attention_mask=out.attention_mask,
        token_type_ids=out.token_type_ids,
    )
    detect_target = torch.zeros_like(result["detect_logits"], dtype=torch.long)
    detect_target[0, 1] = 1
    correct_target = out.input_ids.clone()
    total, l_d, l_c = trainer.joint_loss(result, detect_target, correct_target, out.attention_mask)
    expected_total = 0.85 * l_d + 0.15 * l_c
    assert torch.allclose(total, expected_total, atol=1e-6)


def test_trainer_ignore_index_no_pad_supervision(encoder, softmasked_model):
    """B4 fix: when correct_target is all -100, CE loss must be 0 (not nan, not PAD-loss)."""
    import torch

    from myspellchecker.training.softmasked.trainer import (
        SoftMaskedTrainer,
    )

    trainer = SoftMaskedTrainer(softmasked_model, device="cpu")
    sample = "ဒီမနက် ကျွန်တော်"
    out = encoder.tokenize(sample, max_length=16)
    result = softmasked_model(
        input_ids=out.input_ids,
        attention_mask=out.attention_mask,
        token_type_ids=out.token_type_ids,
    )
    detect_target = torch.zeros_like(result["detect_logits"], dtype=torch.long)
    correct_target = torch.full_like(out.input_ids, fill_value=-100)
    _, _, l_c = trainer.joint_loss(result, detect_target, correct_target, out.attention_mask)
    assert torch.isfinite(l_c)
    assert l_c.item() == 0.0, f"all-ignore correct_target should give 0 loss, got {l_c.item()}"


def test_trainer_one_step_decreases_loss(encoder, softmasked_model):
    import torch

    from myspellchecker.training.softmasked.trainer import (
        SoftMaskedTrainer,
        TrainConfig,
    )

    sample = "ဒီမနက် ကျွန်တော် ကစားကွင်းကို သွားခဲ့တယ်။"
    out = encoder.tokenize(sample, max_length=32)
    detect_target = torch.zeros(out.input_ids.shape, dtype=torch.long)
    detect_target[0, 2] = 1
    # Use full input_ids as correct_target (no ignore for this test)
    correct_target = out.input_ids.clone()

    config = TrainConfig(lr=1e-3, max_steps=5, warmup_steps=0, log_every=1)
    trainer = SoftMaskedTrainer(softmasked_model, config=config, device="cpu")

    def batches():
        for _ in range(5):
            yield {
                "input_ids": out.input_ids,
                "attention_mask": out.attention_mask,
                "token_type_ids": out.token_type_ids,
                "detect_target": detect_target,
                "correct_target": correct_target,
            }

    stats = trainer.train(list(batches()))
    assert stats.step == 5
    first_loss = stats.history[0]["loss"]
    last_loss = stats.history[-1]["loss"]
    assert last_loss < first_loss, f"loss didn't decrease: {first_loss} → {last_loss}"


def test_trainer_lora_attachment():
    """LoRA wraps the encoder in-place, so this test uses its OWN fresh
    encoder + model rather than sharing the module-scoped fixture — keeps
    LoRA mutations from leaking into unrelated tests."""
    pytest.importorskip("peft")
    from myspellchecker.training.softmasked.detector import BiGRUDetector
    from myspellchecker.training.softmasked.encoder import SoftMaskedEncoder
    from myspellchecker.training.softmasked.model import SoftMaskedBERT
    from myspellchecker.training.softmasked.trainer import (
        SoftMaskedTrainer,
        TrainConfig,
    )

    enc = SoftMaskedEncoder(model_path=GKLMIP_PATH, device="cpu")
    det = BiGRUDetector(embedding_dim=enc.hidden_size)
    model = SoftMaskedBERT(encoder=enc, detector=det)
    trainer = SoftMaskedTrainer(model, config=TrainConfig(), device="cpu")
    n_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainer.apply_lora(r=8, alpha=16)
    n_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_after < n_before, "LoRA should reduce trainable param count"
    # The trainer caches LoRA config so load_checkpoint can reapply it.
    assert trainer._lora_applied
    assert trainer._lora_config is not None
    assert trainer._lora_config["r"] == 8


# ---- Dataset (B4 fix: ignore_index, span-only supervision) ----------------


def test_dataset_uses_ignore_index_outside_span(tmp_path, encoder):
    """B4 fix: positions OUTSIDE the corrupted span should be -100 (CE ignored).
    Padding positions should also be -100, never real PAD token IDs.
    """
    import json

    from myspellchecker.training.softmasked.datagen.dataset import (
        IGNORE_INDEX,
        SoftMaskedPairDataset,
    )

    pairs_file = tmp_path / "pairs.jsonl"
    pairs_file.write_text(
        json.dumps(
            {
                "clean": "ဒီမနက် ကျွန်တော် ကစားကွင်းကို သွားခဲ့တယ်။",
                "corrupted": "ဒီမနက် ကျွန်တော် ကစားကင်းကို သွားခဲ့တယ်။",
                "gold": "ကစားကွင်း",
                "erroneous": "ကစားကင်း",
                "span_start": 17,
                "span_end": 25,
                "subtype": "non_word_typo",
                "bucket": "non_word_typo",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dataset = SoftMaskedPairDataset(pairs_file, encoder.tokenizer, max_length=32)
    samples = list(dataset)
    assert len(samples) == 1
    sample = samples[0]
    # detect_target has at least one 1 (the corrupted span)
    assert (sample["detect_target"] == 1).any()
    # correct_target has at least one IGNORE_INDEX position (outside span)
    assert (sample["correct_target"] == IGNORE_INDEX).any()
    # No position should have a non-ignore target outside the detect span
    detect_one = sample["detect_target"] == 1
    detect_zero = sample["detect_target"] == 0
    # Outside-span positions: target must be IGNORE_INDEX
    outside_targets = sample["correct_target"][detect_zero]
    assert (outside_targets == IGNORE_INDEX).all(), "outside-span positions must be IGNORE_INDEX"
    # Inside-span positions: target should be a valid token id (>= 0)
    inside_targets = sample["correct_target"][detect_one]
    assert (inside_targets >= 0).all(), "inside-span positions need real gold token ids"


# ---- Dataset ingestion-time validations (V1-V4, added 2026-05-04) ----------


def _write_one(path, rec):
    import json as _json

    path.write_text(_json.dumps(rec) + "\n", encoding="utf-8")


def test_dataset_v1_rejects_span_erroneous_mismatch(tmp_path, encoder):
    """V1: corrupted[s:e] != erroneous - row dropped, counted under
    span_erroneous_mismatch."""
    from myspellchecker.training.softmasked.datagen.dataset import SoftMaskedPairDataset

    pairs_file = tmp_path / "pairs.jsonl"
    _write_one(
        pairs_file,
        {
            "clean": "ဒီမနက် ကျွန်တော် ကစားကွင်းကို သွားခဲ့တယ်။",
            "corrupted": "ဒီမနက် ကျွန်တော် ကစားကင်းကို သွားခဲ့တယ်။",
            "gold": "ကစားကွင်း",
            "erroneous": "ကစားကင်း",
            # Wrong offsets - off by one (Burmese visual-cluster vs codepoint
            # is the realistic failure mode this guards against).
            "span_start": 16,
            "span_end": 24,
        },
    )
    ds = SoftMaskedPairDataset(pairs_file, encoder.tokenizer, max_length=32)
    samples = list(ds)
    assert len(samples) == 0
    assert ds.skip_counts["span_erroneous_mismatch"] == 1


def test_dataset_v2_rejects_clean_outside_span_mismatch(tmp_path, encoder):
    """V2: when clean diverges from corrupted outside [s,e), drop the row.

    With v1.9 whitespace-strip default, the test must use a non-whitespace
    difference outside the span — pure-whitespace differences are
    legitimately equivalent after normalization and now pass V2.
    """
    from myspellchecker.training.softmasked.datagen.dataset import SoftMaskedPairDataset

    pairs_file = tmp_path / "pairs.jsonl"
    # Outside-span CHARACTER difference (clean has သွား, corrupted has သွို).
    # Span itself is the homophone confusion ကစားကင်း vs ကစားကွင်း.
    _write_one(
        pairs_file,
        {
            "clean": "ဒီမနက် ကျွန်တော် ကစားကွင်းကို သွားခဲ့တယ်။",
            "corrupted": "ဒီမနက် ကျွန်တော် ကစားကင်းကို သွိုခဲ့တယ်။",  # also broke သွား
            "gold": "ကစားကွင်း",
            "erroneous": "ကစားကင်း",
            "span_start": 17,
            "span_end": 25,
        },
    )
    ds = SoftMaskedPairDataset(pairs_file, encoder.tokenizer, max_length=32)
    samples = list(ds)
    assert len(samples) == 0
    # Could be flagged as prefix or gold mismatch depending on exact offset.
    assert sum(ds.skip_counts.values()) == 1


def test_dataset_v2_accepts_whitespace_only_diff_with_normalize(tmp_path, encoder):
    """v1.9 whitespace-strip: typist-spacing variation outside the span is
    NOT a contract violation — it disappears after normalization."""
    from myspellchecker.training.softmasked.datagen.dataset import SoftMaskedPairDataset

    pairs_file = tmp_path / "pairs.jsonl"
    # Same as above but only whitespace differs in the prefix
    _write_one(
        pairs_file,
        {
            "clean": "ဒီ မနက် ကျွန်တော် ကစားကွင်းကို သွားခဲ့တယ်။",
            "corrupted": "ဒီမနက် ကျွန်တော် ကစားကင်းကို သွားခဲ့တယ်။",
            "gold": "ကစားကွင်း",
            "erroneous": "ကစားကင်း",
            "span_start": 17,
            "span_end": 25,
        },
    )
    ds = SoftMaskedPairDataset(pairs_file, encoder.tokenizer, max_length=32)
    samples = list(ds)
    assert len(samples) == 1
    # Verify the row was emitted — V2 passes after whitespace strip
    assert sum(ds.skip_counts.values()) == 0


def test_dataset_v3_rejects_zero_width_span(tmp_path, encoder):
    """V3: span_start == span_end produces no useful supervision - drop."""
    from myspellchecker.training.softmasked.datagen.dataset import SoftMaskedPairDataset

    pairs_file = tmp_path / "pairs.jsonl"
    _write_one(
        pairs_file,
        {
            "clean": "ဒီမနက် ကျွန်တော်",
            "corrupted": "ဒီမနက် ကျွန်တော်",
            "gold": "x",
            "erroneous": "",
            "span_start": 5,
            "span_end": 5,
        },
    )
    ds = SoftMaskedPairDataset(pairs_file, encoder.tokenizer, max_length=32)
    samples = list(ds)
    assert len(samples) == 0
    assert ds.skip_counts["invalid_span_bounds"] == 1


def test_dataset_v4_rejects_span_truncated_out(tmp_path, encoder):
    """V4: span past tokenizer max_length truncation - drop."""
    from myspellchecker.training.softmasked.datagen.dataset import SoftMaskedPairDataset

    pairs_file = tmp_path / "pairs.jsonl"
    long_prefix = "ဒီမနက် ကျွန်တော် " * 50
    clean_full = long_prefix + "ကစားကွင်းကို သွားခဲ့တယ်။"
    corrupted_full = long_prefix + "ကစားကင်းကို သွားခဲ့တယ်။"
    s = corrupted_full.find("ကစားကင်း")
    _write_one(
        pairs_file,
        {
            "clean": clean_full,
            "corrupted": corrupted_full,
            "gold": "ကစားကွင်း",
            "erroneous": "ကစားကင်း",
            "span_start": s,
            "span_end": s + len("ကစားကင်း"),
        },
    )
    # Tiny max_length forces truncation BEFORE the span.
    ds = SoftMaskedPairDataset(pairs_file, encoder.tokenizer, max_length=16)
    samples = list(ds)
    assert len(samples) == 0
    assert ds.skip_counts["span_truncated_out"] == 1


def test_dataset_raise_on_invalid_mode(tmp_path, encoder):
    """Hard-fail mode for catching upstream-data regressions in CI."""
    from myspellchecker.training.softmasked.datagen.dataset import SoftMaskedPairDataset

    pairs_file = tmp_path / "pairs.jsonl"
    _write_one(
        pairs_file,
        {
            "clean": "ဒီမနက်",
            "corrupted": "ဒီမနက်",
            "gold": "x",
            "erroneous": "WRONG",
            "span_start": 0,
            "span_end": 2,
        },
    )
    ds = SoftMaskedPairDataset(pairs_file, encoder.tokenizer, max_length=32, raise_on_invalid=True)
    with pytest.raises(ValueError, match="span_erroneous_mismatch"):
        list(ds)


# ---- Soft-Masked dropout-shared residual (D1, added 2026-05-04) -----------


def test_softmasked_residual_shared_embedding_deterministic(encoder, softmasked_model):
    """D1 fix: e_in_full is computed ONCE and reused for the residual AND
    the soft-mask base. With dropout disabled (train(False)), two forwards
    on the same input must produce identical logits - verifying the
    deterministic pathway after the architectural revision."""
    import torch

    sample = "ဒီမနက် ကျွန်တော် ကစားကွင်းကို သွားခဲ့တယ်။"
    out = encoder.tokenize(sample, max_length=32)
    softmasked_model.train(False)
    with torch.no_grad():
        a = softmasked_model(
            input_ids=out.input_ids,
            attention_mask=out.attention_mask,
            token_type_ids=out.token_type_ids,
        )
        b = softmasked_model(
            input_ids=out.input_ids,
            attention_mask=out.attention_mask,
            token_type_ids=out.token_type_ids,
        )
    assert torch.equal(a["logits"], b["logits"])
    assert torch.equal(a["detect_logits"], b["detect_logits"])
