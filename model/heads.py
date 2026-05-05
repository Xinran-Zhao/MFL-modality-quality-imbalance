"""Per-label classification heads (8 separate MLPs) and contrastive projector."""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class _BinaryHead(nn.Module):
    """Linear -> GELU -> Dropout -> Linear, returning a single logit."""

    def __init__(self, in_dim: int, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)              # (B,)


class MultiLabelHeads(nn.Module):
    """Bank of `num_labels` independent binary heads, each predicting one label.

    Outputs logits of shape (B, num_labels). Each head has its own parameters,
    matching the user's requirement of 8 separate classifiers.
    """

    def __init__(self, embed_dim: int = 256, num_labels: int = 8,
                 hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.num_labels = num_labels
        self.heads = nn.ModuleList([
            _BinaryHead(embed_dim, hidden=hidden, dropout=dropout)
            for _ in range(num_labels)
        ])

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        # Stack head outputs -> (B, num_labels)
        return torch.stack([h(fused) for h in self.heads], dim=-1)


class ContrastiveProjector(nn.Module):
    """Project the per-modality embeddings into a contrastive space and L2-normalize.

    Two separate linear projections (image, text) of shape (D -> D_c). L2-norm
    so dot products become cosine similarities, as in CLIP.
    """

    def __init__(self, embed_dim: int = 256, contrastive_dim: int = 128):
        super().__init__()
        self.image_proj = nn.Linear(embed_dim, contrastive_dim)
        self.text_proj  = nn.Linear(embed_dim, contrastive_dim)

    def forward(self, img_emb: torch.Tensor, txt_emb: torch.Tensor):
        z_img = F.normalize(self.image_proj(img_emb), dim=-1)
        z_txt = F.normalize(self.text_proj(txt_emb), dim=-1)
        return z_img, z_txt
