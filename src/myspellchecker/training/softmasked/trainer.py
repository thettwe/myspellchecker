"""Joint trainer for Soft-Masked BERT.

Loss: ``L = λ × L_detect_BCE + (1 - λ) × L_correct_CE``

  - ``L_detect_BCE`` — binary cross-entropy on per-position p(error).
    Optional ``pos_weight`` for the sparse-positive prior.
  - ``L_correct_CE`` — cross-entropy of corrector logits against gold token
    IDs at every non-padding position. ALL positions contribute, not just
    the corrupted span — Soft-Masked supervises the corrector on the full
    sequence (the residual + soft-mask makes the corrector responsible for
    every position, even the un-flagged ones).

LoRA via ``peft``: applied to the BERT corrector body's attention projections
(``query``, ``key``, ``value``, ``output.dense``). Detector trains full-rank
because it's small (~5M params).

Usage::

    encoder = SoftMaskedEncoder(model_path)
    detector = BiGRUDetector(encoder.hidden_size)
    model = SoftMaskedBERT(encoder, detector)
    trainer = SoftMaskedTrainer(model, lambda_detect=0.85)
    trainer.apply_lora(r=8, alpha=16)  # optional
    trainer.train(dataloader, valid_dataloader, ...)
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .model import SoftMaskedBERT


@dataclass
class TrainConfig:
    lambda_detect: float = 0.85
    lr: float = 3e-4  # head LR; encoder LR scales down via param-group
    encoder_lr_scale: float = 0.1  # encoder gets 0.1× the head LR
    weight_decay: float = 0.01
    pos_weight: float | None = 20.0  # BCE class-prior weighting on errors
    warmup_steps: int = 200
    max_steps: int | None = None
    num_epochs: int = 3
    grad_clip: float = 1.0
    log_every: int = 50
    valid_every: int | None = 500
    save_every: int | None = None
    seed: int = 42


@dataclass
class TrainStats:
    step: int = 0
    epoch: int = 0
    loss: float = 0.0
    loss_detect: float = 0.0
    loss_correct: float = 0.0
    elapsed_s: float = 0.0
    history: list[dict[str, Any]] = field(default_factory=list)


class SoftMaskedTrainer:
    """Wraps a ``SoftMaskedBERT`` model with optimizer, LR schedule, joint loss,
    and a validation hook compatible with the prep-ws0 audit harness primitives.
    """

    def __init__(
        self,
        model: SoftMaskedBERT,
        config: TrainConfig | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        self.model = model
        self.config = config or TrainConfig()
        self.device = device or model.encoder.device
        self.model.to(self.device)
        # Keep the encoder's stored device attribute consistent with where
        # the parameters actually live — readers (loggers, downstream
        # tokenize() calls) rely on encoder.device.
        self.model.encoder.device = next(self.model.parameters()).device
        self._optimizer: torch.optim.Optimizer | None = None
        self._scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        self._lora_applied = False
        # Cache enough config to re-apply on load_checkpoint (D2 fix).
        self._lora_config: dict[str, Any] | None = None

    # ----- LoRA application -----------------------------------------------

    def apply_lora(
        self,
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.05,
        target_modules: list[str] | None = None,
    ) -> None:
        """Wrap the BERT corrector body with LoRA adapters.

        Detector + MLM head stay full-rank. Encoder body is reduced to LoRA-
        only updates.
        """
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as exc:
            raise ImportError(
                "peft is required for LoRA. Install with `pip install peft`."
            ) from exc
        if self._lora_applied:
            return
        if target_modules is None:
            # BERT attention modules
            target_modules = ["query", "key", "value", "dense"]
        cfg = LoraConfig(
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=target_modules,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        wrapped = get_peft_model(self.model.encoder.inner_model, cfg)
        # Update both the encoder's owned reference AND the SoftMaskedBERT's
        # registered submodule so parameter-iteration sees the LoRA wrap.
        self.model.encoder.model = wrapped
        self.model.bert = wrapped
        from .corrector import SoftMaskedCorrector

        self.model.corrector = SoftMaskedCorrector(self.model.encoder)
        self._lora_applied = True
        # Persist the args so load_checkpoint can reapply the same wrap on
        # a fresh trainer (D2 fix — saved state_dict has LoRA-tagged keys
        # that won't load into the unwrapped base model).
        self._lora_config = {
            "r": r,
            "alpha": alpha,
            "dropout": dropout,
            "target_modules": list(target_modules),
        }

    # ----- Optimizer + schedule -------------------------------------------

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Two param groups: detector at full LR, encoder body at scaled LR."""
        cfg = self.config
        encoder_params: list[nn.Parameter] = []
        detector_params: list[nn.Parameter] = []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("detector"):
                detector_params.append(p)
            else:
                encoder_params.append(p)
        return torch.optim.AdamW(
            [
                {"params": detector_params, "lr": cfg.lr},
                {"params": encoder_params, "lr": cfg.lr * cfg.encoder_lr_scale},
            ],
            weight_decay=cfg.weight_decay,
        )

    def _build_scheduler(self, optimizer: torch.optim.Optimizer, total_steps: int):
        """Linear warmup → linear decay to 0."""
        warmup = self.config.warmup_steps

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return float(step) / max(1, warmup)
            remaining = max(1, total_steps - warmup)
            progress = (step - warmup) / remaining
            return max(0.0, 1.0 - progress)

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ----- Loss components ------------------------------------------------

    def joint_loss(
        self,
        result: dict[str, torch.Tensor],
        detect_target: torch.Tensor,
        correct_target: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Joint loss returns (total, detect_loss, correct_loss).

        - detect loss is BCE-with-logits on the detector's raw output (M1 fix)
        - correct loss is CE with ``ignore_index=-100`` so positions outside
          the corrupted span — where dataset.py doesn't have ground truth —
          contribute zero to the loss instead of being supervised against
          PAD or arbitrary tokens (B4 fix).
        """
        cfg = self.config
        l_detect = self.model.detector.loss(
            result["detect_logits"],
            detect_target,
            attention_mask=attention_mask,
            pos_weight=cfg.pos_weight,
        )
        # Cross-entropy on logits vs gold token IDs, ignoring -100 positions.
        # ignore_index handles padding AND non-span positions in one mechanism.
        logits = result["logits"]  # [B, S, V]
        flat_logits = logits.view(-1, logits.size(-1))
        flat_target = correct_target.view(-1)
        l_correct = nn.functional.cross_entropy(
            flat_logits, flat_target, reduction="mean", ignore_index=-100
        )
        # If the entire batch has no supervised positions (all -100), CE
        # returns nan — clamp to 0 to avoid propagating nan into the joint
        # loss. (Defensive: shouldn't happen in practice since detect_target
        # marks at least the corrupted span at every row.)
        if torch.isnan(l_correct):
            l_correct = torch.zeros((), device=l_correct.device, dtype=l_correct.dtype)
        total = cfg.lambda_detect * l_detect + (1.0 - cfg.lambda_detect) * l_correct
        return total, l_detect, l_correct

    # ----- Training loop --------------------------------------------------

    def train(
        self,
        train_dataloader,
        valid_dataloader=None,
        *,
        save_dir: str | Path | None = None,
        on_validate=None,
    ) -> TrainStats:
        """Run training for ``num_epochs`` (or ``max_steps`` if set).

        Each batch yielded by ``train_dataloader`` should be a dict with keys:
          - ``input_ids`` : [B, S] tokenized corrupted sentence
          - ``attention_mask`` : [B, S]
          - ``token_type_ids`` : [B, S] or None
          - ``detect_target`` : [B, S] per-position 0/1 error labels
          - ``correct_target`` : [B, S] per-position GOLD token ids
        """
        cfg = self.config
        torch.manual_seed(cfg.seed)
        self.model.train()
        self._optimizer = self._build_optimizer()
        try:
            total_steps = (
                cfg.max_steps
                if cfg.max_steps is not None
                else cfg.num_epochs * len(train_dataloader)
            )
        except TypeError:
            total_steps = cfg.max_steps or 1000
        self._scheduler = self._build_scheduler(self._optimizer, max(1, total_steps))

        stats = TrainStats()
        t0 = time.time()
        ema_loss = None
        ema_alpha = 0.05

        for epoch in range(cfg.num_epochs):
            stats.epoch = epoch
            for batch in train_dataloader:
                if cfg.max_steps is not None and stats.step >= cfg.max_steps:
                    break
                batch = self._batch_to_device(batch)
                result = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    token_type_ids=batch.get("token_type_ids"),
                )
                total, l_d, l_c = self.joint_loss(
                    result,
                    batch["detect_target"],
                    batch["correct_target"],
                    batch["attention_mask"],
                )
                self._optimizer.zero_grad(set_to_none=True)
                total.backward()
                if cfg.grad_clip is not None and cfg.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                self._optimizer.step()
                self._scheduler.step()
                stats.step += 1
                stats.loss = total.item()
                stats.loss_detect = l_d.item()
                stats.loss_correct = l_c.item()
                stats.elapsed_s = time.time() - t0
                ema_loss = (
                    total.item()
                    if ema_loss is None
                    else (ema_alpha * total.item() + (1 - ema_alpha) * ema_loss)
                )

                if stats.step % cfg.log_every == 0:
                    stats.history.append(
                        {
                            "step": stats.step,
                            "epoch": epoch,
                            "loss": stats.loss,
                            "loss_detect": stats.loss_detect,
                            "loss_correct": stats.loss_correct,
                            "ema_loss": ema_loss,
                            "lr_detector": self._optimizer.param_groups[0]["lr"],
                            "lr_encoder": self._optimizer.param_groups[1]["lr"],
                            "elapsed_s": stats.elapsed_s,
                        }
                    )
                    print(
                        f"[train] step={stats.step:>5} epoch={epoch} "
                        f"loss={stats.loss:.4f} "
                        f"(detect={stats.loss_detect:.4f} correct={stats.loss_correct:.4f}) "
                        f"ema={ema_loss:.4f} elapsed={stats.elapsed_s:.0f}s",
                        flush=True,
                    )

                if (
                    valid_dataloader is not None
                    and cfg.valid_every is not None
                    and stats.step % cfg.valid_every == 0
                ):
                    metrics = self.validate(valid_dataloader)
                    print(f"[valid] step={stats.step} {metrics}", flush=True)
                    if on_validate is not None:
                        on_validate(stats, metrics)
                    self.model.train()

                if (
                    save_dir is not None
                    and cfg.save_every is not None
                    and stats.step % cfg.save_every == 0
                ):
                    self.save_checkpoint(save_dir, stats.step)

        if save_dir is not None:
            self.save_checkpoint(save_dir, stats.step)

        return stats

    @torch.no_grad()
    def validate(self, valid_dataloader) -> dict[str, float]:
        """Compute average detect/correct losses + detection F1 on a holdout."""
        self.model.train(False)
        n = 0
        l_d_sum = 0.0
        l_c_sum = 0.0
        tp = fp = fn = 0
        for batch in valid_dataloader:
            batch = self._batch_to_device(batch)
            result = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch.get("token_type_ids"),
            )
            _, l_d, l_c = self.joint_loss(
                result,
                batch["detect_target"],
                batch["correct_target"],
                batch["attention_mask"],
            )
            n += 1
            l_d_sum += l_d.item()
            l_c_sum += l_c.item()
            mask = batch["attention_mask"].bool()
            pred = (result["p_error"] >= 0.5) & mask
            target = (batch["detect_target"] >= 1) & mask
            tp += int((pred & target).sum().item())
            fp += int((pred & ~target).sum().item())
            fn += int((~pred & target).sum().item())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        return {
            "loss_detect": l_d_sum / max(n, 1),
            "loss_correct": l_c_sum / max(n, 1),
            "detect_precision": precision,
            "detect_recall": recall,
            "detect_f1": f1,
            "n_batches": n,
        }

    # ----- Persistence ----------------------------------------------------

    def save_checkpoint(self, save_dir: str | Path, step: int) -> Path:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = save_dir / f"step-{step}.pt"
        torch.save(
            {
                "step": step,
                "model_state": self.model.state_dict(),
                "optimizer_state": self._optimizer.state_dict() if self._optimizer else None,
                "config": asdict(self.config),
                "lora_applied": self._lora_applied,
                "lora_config": self._lora_config,
            },
            ckpt_path,
        )
        return ckpt_path

    def load_checkpoint(self, ckpt_path: str | Path) -> dict[str, Any]:
        """Load weights from a checkpoint, re-applying LoRA first if needed.

        Without re-applying LoRA, a saved wrapped state_dict has
        ``base_model.model.bert...`` style keys that won't match the
        unwrapped SoftMaskedBERT (D2 fix).
        """
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        if ckpt.get("lora_applied") and not self._lora_applied:
            lora_cfg = ckpt.get("lora_config") or {}
            self.apply_lora(
                r=int(lora_cfg.get("r", 8)),
                alpha=int(lora_cfg.get("alpha", 16)),
                dropout=float(lora_cfg.get("dropout", 0.05)),
                target_modules=lora_cfg.get("target_modules"),
            )
        self.model.load_state_dict(ckpt["model_state"])
        return ckpt

    # ----- Helpers --------------------------------------------------------

    def _batch_to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        out = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(self.device)
            else:
                out[k] = v
        return out
