"""Image preprocessing and text tokenization utilities.

Image: ImageNet-style normalization for compatibility with the torchvision
ResNet backbones. Train transform applies a light random resized crop;
eval transform is a deterministic resize+center-crop.

Text: thin wrapper around the HuggingFace tokenizer matching the text
backbone in `model/encoders.py`. Default backbone is
`microsoft/BiomedVLP-CXR-BERT-specialized`.
"""
from __future__ import annotations

from typing import List, Sequence

from torchvision import transforms as T

# ImageNet stats (ResNet50 V2 weights are normalized this way)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def build_image_transform(image_size: int = 224, train: bool = False):
    """Return a torchvision transform pipeline taking PIL.Image -> Tensor.

    Args:
        image_size: final spatial size (H = W = image_size).
        train: if True, apply mild random augmentation (resized crop).
               Horizontal flip is intentionally NOT used for chest X-rays
               (cardiac silhouette is left-sided; flipping would teach
               incorrect anatomy).
    """
    if train:
        return T.Compose([
            T.Resize(int(image_size * 256 / 224)),         # 256 if image_size=224
            T.RandomResizedCrop(image_size, scale=(0.85, 1.0), ratio=(0.95, 1.05)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    return T.Compose([
        T.Resize(int(image_size * 256 / 224)),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_text_tokenizer(
    backbone_name: str = "microsoft/BiomedVLP-CXR-BERT-specialized",
    trust_remote_code: bool = True,
):
    """Return the matching HuggingFace tokenizer for the text backbone."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(
        backbone_name, trust_remote_code=trust_remote_code
    )


def tokenize_batch(tokenizer, texts: Sequence[str], max_length: int = 256):
    """Tokenize a list of strings to (input_ids, attention_mask) tensors.

    Pads to `max_length`, truncates beyond it.
    """
    enc = tokenizer(
        list(texts),
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return enc["input_ids"], enc["attention_mask"]
