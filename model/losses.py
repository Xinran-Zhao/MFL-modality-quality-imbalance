"""Multi-label BCE classification loss + symmetric InfoNCE contrastive loss + combo.

The contrastive term is computed only over the subset of samples that actually
have text. If a batch has fewer than 2 paired samples, the contrastive loss
is treated as zero (no negatives to contrast against).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- multi-label classification ----------

def multilabel_bce_loss(logits: torch.Tensor,
                        targets: torch.Tensor,
                        pos_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Mean BCE-with-logits over the (B, num_labels) tensor.

    Args:
        logits  : (B, num_labels)
        targets : (B, num_labels) float in {0, 1}
        pos_weight : (num_labels,) per-label positive weighting, or None.
    """
    return F.binary_cross_entropy_with_logits(
        logits, targets.float(), pos_weight=pos_weight, reduction="mean"
    )


# ---------- contrastive ----------

def info_nce_loss(z_img: torch.Tensor,
                  z_txt: torch.Tensor,
                  temperature: torch.Tensor,
                  has_text: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Symmetric CLIP-style InfoNCE loss.

    z_img, z_txt are L2-normalized (B, D_c) tensors. `temperature` is a scalar
    tensor (>0). `has_text`: (B,) bool — only rows where True participate.

    Returns 0 if fewer than 2 paired rows are present.
    """
    if has_text is not None:
        idx = torch.nonzero(has_text, as_tuple=False).squeeze(-1)
        if idx.numel() < 2:
            return z_img.sum() * 0.0    # keeps autograd graph but evaluates to 0
        z_img = z_img.index_select(0, idx)
        z_txt = z_txt.index_select(0, idx)
    elif z_img.shape[0] < 2:
        return z_img.sum() * 0.0

    # Cosine similarity matrix scaled by 1/temperature
    logits = (z_img @ z_txt.t()) / temperature.clamp_min(1e-6)
    targets = torch.arange(logits.shape[0], device=logits.device)
    loss_i2t = F.cross_entropy(logits, targets)
    loss_t2i = F.cross_entropy(logits.t(), targets)
    return 0.5 * (loss_i2t + loss_t2i)


# ---------- lambda warmup ----------

def linear_lambda_warmup(current_step: int,
                         warmup_steps: int,
                         lambda_final: float,
                         lambda_init: float = 0.0) -> float:
    """Linearly ramp lambda from `lambda_init` (at step 0) up to `lambda_final`
    by step `warmup_steps`; constant at `lambda_final` afterwards.

    `current_step` is whatever unit you're tracking (typically the FL round index).
    """
    if warmup_steps <= 0:
        return lambda_final
    t = max(0, min(current_step, warmup_steps))
    return lambda_init + (lambda_final - lambda_init) * (t / warmup_steps)


# ---------- combined loss ----------

class CombinedLoss(nn.Module):
    """L_total = L_cls + lambda(t) * L_contrast.

    Holds a per-label `pos_weight` buffer (optional) and exposes
    `set_lambda(value)` so the trainer can update lambda each round
    (e.g., via `linear_lambda_warmup`).
    """

    def __init__(self,
                 lambda_contrast: float = 0.5,
                 pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.lambda_contrast = float(lambda_contrast)
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight.float(), persistent=False)
        else:
            self.pos_weight = None  # type: ignore[assignment]

    def set_lambda(self, value: float) -> None:
        self.lambda_contrast = float(value)

    def forward(self,
                outputs: dict,
                targets: torch.Tensor) -> dict:
        """Compute losses given the dict returned by MFLModel.forward.

        Returns a dict {loss_total, loss_cls, loss_contrast, lambda} so the
        trainer can log each term individually.
        """
        cls_loss = multilabel_bce_loss(
            outputs["logits"], targets, pos_weight=self.pos_weight
        )
        nce = info_nce_loss(
            outputs["z_img"], outputs["z_txt"],
            outputs["temperature"], has_text=outputs.get("has_text"),
        )
        total = cls_loss + self.lambda_contrast * nce
        return {
            "loss_total": total,
            "loss_cls": cls_loss.detach(),
            "loss_contrast": nce.detach(),
            "lambda": torch.tensor(self.lambda_contrast, device=cls_loss.device),
        }
