"""Per-client local training (one FL round = one local pass over local data).

Design choices
--------------
- The server hands the client a *fresh copy* of the global model each round.
  The client constructs its own optimizer + scheduler over that copy, runs
  one (or `local_epochs`) pass(es), and returns its updated state_dict to
  the server for FedAvg.
- Because each round rebuilds the optimizer from scratch, AdamW's running
  moments do NOT persist across rounds. This is the standard vanilla FedAvg
  formulation; momentum-aware variants (FedOpt) would change this.
- Two parameter groups: backbone params (ResNet + BERT) at a smaller LR,
  everything else at a larger LR.
- Modality dropout is applied only on the multimodal client (controlled
  by `apply_modality_dropout`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import torch
from torch.utils.data import DataLoader

from model.model import MFLModel
from model.losses import CombinedLoss


@dataclass
class ClientConfig:
    client_id: str
    lr_backbone: float = 1e-5
    lr_head: float = 1e-4
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    local_epochs: int = 1
    apply_modality_dropout: bool = False  # True only for the multimodal client (c0)
    max_steps: Optional[int] = None       # cap steps for --dry-run; None = full epoch


def _build_optimizer(model: MFLModel, cfg: ClientConfig) -> torch.optim.Optimizer:
    backbone = list(model.backbone_params())
    head = list(model.head_params())
    return torch.optim.AdamW(
        [
            {"params": backbone, "lr": cfg.lr_backbone, "weight_decay": cfg.weight_decay},
            {"params": head,     "lr": cfg.lr_head,     "weight_decay": cfg.weight_decay},
        ],
        betas=(0.9, 0.999), eps=1e-8,
    )


def local_train(
    model: MFLModel,
    loader: DataLoader,
    loss_fn: CombinedLoss,
    cfg: ClientConfig,
    device: torch.device,
) -> Dict[str, float]:
    """Run `cfg.local_epochs` passes over `loader`, return training metrics.

    Mutates `model` in place (caller passes a per-round copy).

    Returns a dict:
        loss_total, loss_cls, loss_contrast : sample-weighted means
        n_steps, n_samples
        lr_backbone, lr_head                 : final LR values
    """
    model.train()
    optim = _build_optimizer(model, cfg)

    sums = {"loss_total": 0.0, "loss_cls": 0.0, "loss_contrast": 0.0}
    n_steps = 0
    n_samples = 0

    for _epoch in range(cfg.local_epochs):
        for batch in loader:
            if cfg.max_steps is not None and n_steps >= cfg.max_steps:
                break

            image = batch["image"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attn = batch["attention_mask"].to(device, non_blocking=True)
            has_text = batch["has_text"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            B = image.shape[0]

            outputs = model(
                image=image,
                input_ids=input_ids,
                attention_mask=attn,
                has_text=has_text,
                apply_modality_dropout=cfg.apply_modality_dropout,
            )
            losses = loss_fn(outputs, labels)
            loss = losses["loss_total"]

            optim.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip_norm is not None and cfg.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optim.step()

            # Sample-weighted accumulation
            sums["loss_total"]    += float(losses["loss_total"].detach().item()) * B
            sums["loss_cls"]      += float(losses["loss_cls"].detach().item())   * B
            sums["loss_contrast"] += float(losses["loss_contrast"].detach().item()) * B
            n_steps += 1
            n_samples += B

        if cfg.max_steps is not None and n_steps >= cfg.max_steps:
            break

    if n_samples == 0:
        means = {k: 0.0 for k in sums}
    else:
        means = {k: v / n_samples for k, v in sums.items()}

    means.update({
        "n_steps": n_steps,
        "n_samples": n_samples,
        "lr_backbone": optim.param_groups[0]["lr"],
        "lr_head":     optim.param_groups[1]["lr"],
    })
    return means
