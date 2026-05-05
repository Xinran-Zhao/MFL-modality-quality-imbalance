"""Image and text encoders with projection heads to a common embedding dim D."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torchvision import models as tv_models

# Backbone-feature dims for the supported torchvision ResNets.
_RESNET_FACTORY = {
    "resnet18":  (tv_models.resnet18,  tv_models.ResNet18_Weights.IMAGENET1K_V1, 512),
    "resnet34":  (tv_models.resnet34,  tv_models.ResNet34_Weights.IMAGENET1K_V1, 512),
    "resnet50":  (tv_models.resnet50,  tv_models.ResNet50_Weights.IMAGENET1K_V2, 2048),
    "resnet101": (tv_models.resnet101, tv_models.ResNet101_Weights.IMAGENET1K_V2, 2048),
}


class ImageEncoder(nn.Module):
    """ResNet backbone (ImageNet-pretrained) -> projection to common dim D.

    All parameters are trainable (full fine-tune). The final FC layer of the
    torchvision ResNet is replaced with Identity so we get the global-avg-pool
    feature directly.
    """

    def __init__(self, embed_dim: int = 256, backbone: str = "resnet50",
                 pretrained: bool = True):
        super().__init__()
        if backbone not in _RESNET_FACTORY:
            raise ValueError(f"unknown backbone {backbone!r}; "
                             f"supported: {list(_RESNET_FACTORY)}")
        ctor, weights, feat_dim = _RESNET_FACTORY[backbone]
        self.backbone_name = backbone
        self.feat_dim = feat_dim
        self.backbone = ctor(weights=weights if pretrained else None)
        # drop the classifier; downstream uses the avgpool output
        self.backbone.fc = nn.Identity()

        self.projector = nn.Sequential(
            nn.Linear(feat_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """image: (B, 3, H, W) -> (B, embed_dim)."""
        feats = self.backbone(image)        # (B, feat_dim)
        return self.projector(feats)        # (B, embed_dim)


class TextEncoder(nn.Module):
    """BERT-style backbone -> projection to common dim D.

    Uses the [CLS] token of the last hidden state as the pooled text feature.
    All parameters are trainable. By default uses
    `microsoft/BiomedVLP-CXR-BERT-specialized` (CXR-domain pretrained).
    """

    def __init__(self, embed_dim: int = 256,
                 backbone_name: str = "microsoft/BiomedVLP-CXR-BERT-specialized",
                 trust_remote_code: bool = True,
                 pretrained: bool = True):
        super().__init__()
        # Lazy import so that loading the package doesn't pull transformers
        # if the user only wants the image branch.
        from transformers import AutoModel, AutoConfig
        self.backbone_name = backbone_name
        if pretrained:
            self.backbone = AutoModel.from_pretrained(
                backbone_name, trust_remote_code=trust_remote_code)
        else:
            cfg = AutoConfig.from_pretrained(
                backbone_name, trust_remote_code=trust_remote_code)
            self.backbone = AutoModel.from_config(
                cfg, trust_remote_code=trust_remote_code)

        hidden = getattr(self.backbone.config, "hidden_size", None)
        if hidden is None:
            raise RuntimeError(
                f"could not infer hidden_size from {backbone_name}'s config")
        self.feat_dim = hidden

        self.projector = nn.Sequential(
            nn.Linear(hidden, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """input_ids: (B, L), attention_mask: (B, L) -> (B, embed_dim)."""
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        # last_hidden_state: (B, L, H); take [CLS] = position 0
        cls = out.last_hidden_state[:, 0, :]
        return self.projector(cls)
