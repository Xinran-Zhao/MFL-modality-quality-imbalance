"""MLP fusion of image and text embeddings.

Missing-text handling (per design): substitute a literal zero vector for
the text embedding. The fusion MLP learns to interpret `concat(img, zeros)`
as the image-only mode.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class FusionMLP(nn.Module):
    """concat(img_emb, txt_emb) -> Linear(2D->2D) -> GELU -> Dropout -> Linear(2D->D) -> LN."""

    def __init__(self, embed_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, img_emb: torch.Tensor,
                txt_emb: Optional[torch.Tensor] = None,
                has_text: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Fuse image and text embeddings.

        Args:
            img_emb: (B, D)
            txt_emb: (B, D) or None. If None, treated as all-missing (zeros).
            has_text: (B,) bool tensor. For samples where False, the corresponding
                row of txt_emb is overwritten with zeros before fusion. Ignored
                when txt_emb is None (all rows treated as missing).

        Returns:
            fused: (B, D)
        """
        B, D = img_emb.shape
        if txt_emb is None:
            txt_emb = torch.zeros(B, D, device=img_emb.device, dtype=img_emb.dtype)
        elif has_text is not None:
            # Zero out rows where text is missing. has_text: (B,) bool -> (B,1) for broadcast.
            mask = has_text.to(dtype=img_emb.dtype, device=img_emb.device).unsqueeze(1)
            txt_emb = txt_emb * mask
        x = torch.cat([img_emb, txt_emb], dim=-1)   # (B, 2D)
        return self.mlp(x)                          # (B, D)
