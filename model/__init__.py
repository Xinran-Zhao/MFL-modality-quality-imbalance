"""MFLModel components for multimodal federated learning on chest X-rays."""
from .encoders import ImageEncoder, TextEncoder
from .fusion import FusionMLP
from .heads import MultiLabelHeads, ContrastiveProjector
from .model import MFLModel
from .losses import (
    multilabel_bce_loss,
    info_nce_loss,
    CombinedLoss,
    linear_lambda_warmup,
)

__all__ = [
    "ImageEncoder", "TextEncoder", "FusionMLP",
    "MultiLabelHeads", "ContrastiveProjector",
    "MFLModel",
    "multilabel_bce_loss", "info_nce_loss", "CombinedLoss",
    "linear_lambda_warmup",
]
