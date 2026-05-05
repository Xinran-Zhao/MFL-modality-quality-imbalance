"""Top-level multimodal FL model composing the encoders, fusion, and heads.

Forward signature is uniform across multimodal and image-only modes:
the caller passes `has_text` (a (B,) bool tensor) and the model computes
text features only where they are present.

Modality dropout (training-time augmentation on the multimodal client c0)
is applied here when `apply_modality_dropout=True` is passed to forward —
the model will randomly mask `has_text` to False for some samples.
"""
from __future__ import annotations

import math
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from .encoders import ImageEncoder, TextEncoder
from .fusion import FusionMLP
from .heads import ContrastiveProjector, MultiLabelHeads


class MFLModel(nn.Module):
    """End-to-end model: encoders -> fusion -> classification + contrastive heads."""

    def __init__(
        self,
        embed_dim: int = 256,
        contrastive_dim: int = 128,
        num_labels: int = 8,
        head_hidden: int = 64,
        dropout: float = 0.1,
        modality_dropout_p: float = 0.15,
        image_backbone: str = "resnet50",
        text_backbone: str = "microsoft/BiomedVLP-CXR-BERT-specialized",
        text_trust_remote_code: bool = True,
        pretrained_image: bool = True,
        pretrained_text: bool = True,
        contrastive_temp_init: float = 0.07,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_labels = num_labels
        self.modality_dropout_p = modality_dropout_p

        self.image_encoder = ImageEncoder(
            embed_dim=embed_dim,
            backbone=image_backbone,
            pretrained=pretrained_image,
        )
        self.text_encoder = TextEncoder(
            embed_dim=embed_dim,
            backbone_name=text_backbone,
            trust_remote_code=text_trust_remote_code,
            pretrained=pretrained_text,
        )
        self.fusion = FusionMLP(embed_dim=embed_dim, dropout=dropout)
        self.classifier = MultiLabelHeads(
            embed_dim=embed_dim, num_labels=num_labels,
            hidden=head_hidden, dropout=dropout,
        )
        self.contrastive_projector = ContrastiveProjector(
            embed_dim=embed_dim, contrastive_dim=contrastive_dim,
        )
        # Learnable temperature (CLIP-style). We parameterize log(1/τ) for
        # numerical stability; effective τ = exp(-log_inv_temp).
        init = math.log(1.0 / contrastive_temp_init)
        self.log_inv_temp = nn.Parameter(torch.tensor(init, dtype=torch.float32))

    # ---- helpers --------------------------------------------------------

    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(-self.log_inv_temp)

    def backbone_params(self) -> Iterable[nn.Parameter]:
        """ResNet + BERT params (typically trained at a smaller LR)."""
        yield from self.image_encoder.backbone.parameters()
        yield from self.text_encoder.backbone.parameters()

    def head_params(self) -> Iterable[nn.Parameter]:
        """Everything else: projection heads, fusion, classifiers, contrastive
        projector, learnable temperature."""
        seen = set(id(p) for p in self.backbone_params())
        for p in self.parameters():
            if id(p) not in seen:
                yield p

    # ---- forward --------------------------------------------------------

    def _maybe_modality_dropout(
        self, has_text: torch.Tensor, apply: bool
    ) -> torch.Tensor:
        """Randomly switch some `has_text=True` rows to False during training."""
        if not (apply and self.training and self.modality_dropout_p > 0):
            return has_text
        # Only drop where text is actually present; leave already-missing alone.
        rand = torch.rand_like(has_text, dtype=torch.float32)
        keep = rand >= self.modality_dropout_p
        return has_text & keep

    def forward(
        self,
        image: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        has_text: Optional[torch.Tensor] = None,
        apply_modality_dropout: bool = False,
    ) -> dict:
        """Returns a dict with:
            logits        : (B, num_labels)  classification logits
            fused         : (B, D)
            img_emb       : (B, D)
            txt_emb       : (B, D)   zero-rows where has_text is False
            z_img, z_txt  : (B, D_c) contrastive projections (z_txt zero-rows where missing)
            has_text      : (B,) bool  effective mask after modality dropout
            temperature   : scalar tensor
        """
        B = image.shape[0]
        device = image.device

        # ---- has_text resolution ----
        if has_text is None:
            has_text = torch.tensor(
                [input_ids is not None] * B, device=device, dtype=torch.bool
            )
        else:
            has_text = has_text.to(device=device, dtype=torch.bool)
        has_text = self._maybe_modality_dropout(has_text, apply_modality_dropout)

        # ---- image branch (always) ----
        img_emb = self.image_encoder(image)                          # (B, D)

        # ---- text branch (only for samples with text) ----
        txt_emb = torch.zeros_like(img_emb)
        if input_ids is not None and has_text.any():
            idx = torch.nonzero(has_text, as_tuple=False).squeeze(-1)  # (K,)
            sub_ids = input_ids.index_select(0, idx)
            sub_mask = (
                attention_mask.index_select(0, idx)
                if attention_mask is not None else None
            )
            sub_emb = self.text_encoder(sub_ids, sub_mask)            # (K, D)
            txt_emb = txt_emb.index_copy(0, idx, sub_emb)

        # ---- fusion + classification ----
        fused = self.fusion(img_emb, txt_emb, has_text=has_text)     # (B, D)
        logits = self.classifier(fused)                              # (B, num_labels)

        # ---- contrastive projections (zero rows where text missing) ----
        z_img, z_txt = self.contrastive_projector(img_emb, txt_emb)
        if not has_text.all():
            mask = has_text.to(dtype=z_txt.dtype).unsqueeze(1)
            z_txt = z_txt * mask

        return {
            "logits": logits,
            "fused": fused,
            "img_emb": img_emb,
            "txt_emb": txt_emb,
            "z_img": z_img,
            "z_txt": z_txt,
            "has_text": has_text,
            "temperature": self.temperature,
        }
