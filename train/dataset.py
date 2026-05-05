"""Dataset utilities for MFL training.

`PairedCXRDataset` reads `prepared_data.csv` once, filters to a list of
uids, and returns one (image, input_ids, attention_mask, has_text,
labels, uid) sample per index.

`build_client_loaders` and `build_global_eval_loaders` wire up the
partition JSONs and modality config into PyTorch DataLoaders.

Modality handling
-----------------
The CSV `findings_text` is loaded for every uid. Whether a client
*sees* the text at training time is controlled by the modality config:

    client_0: {"image": True,  "text": True}   -> text is fed to the model
    client_1: {"image": True,  "text": False}  -> text is masked at the dataset level
                                                  (input_ids zeroed, has_text=False)

Eval loaders always carry the text along; the *server* decides at eval
time whether to forward it (multimodal protocol) or mask it (image-only
protocol).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from data.transforms import (
    build_image_transform,
    build_text_tokenizer,
    tokenize_batch,
)

LABEL_COLS = [
    "No_Finding", "Cardiomegaly", "Atelectasis", "Pleural_Effusion",
    "Opacity", "Calcinosis", "Calcified_Granuloma", "Airspace_Disease",
]


# ---------- core dataset ----------

class PairedCXRDataset(Dataset):
    """Image+text dataset over a fixed list of uids.

    Args:
        df_uid_indexed: prepared_data.csv frame indexed by uid (int).
        uids:           list of uids selected for this dataset.
        image_root:     directory containing the PNGs from `image_path`.
        image_tf:       torchvision transform (PIL.Image -> Tensor).
        keep_text:      if False, the text branch is masked off
                        (input_ids/attention_mask zeroed, has_text=False).
                        Used for the image-only client at training.
    """

    def __init__(
        self,
        df_uid_indexed: pd.DataFrame,
        uids: Sequence[int],
        image_root: str,
        image_tf,
        keep_text: bool = True,
    ):
        self.df = df_uid_indexed
        self.uids = [int(u) for u in uids]
        self.image_root = image_root
        self.image_tf = image_tf
        self.keep_text = bool(keep_text)

    def __len__(self) -> int:
        return len(self.uids)

    def get_text(self, idx: int) -> str:
        """Raw findings string for uid at position idx (used by tokenizer collator)."""
        uid = self.uids[idx]
        row = self.df.loc[uid]
        return str(row["findings_text"]) if self.keep_text else ""

    def __getitem__(self, idx: int) -> Dict:
        uid = self.uids[idx]
        row = self.df.loc[uid]

        # ---- image ----
        img_path = os.path.join(self.image_root, str(row["image_path"]))
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            image = self.image_tf(im)

        # ---- labels ----
        labels = torch.tensor(
            [int(row[c]) for c in LABEL_COLS], dtype=torch.float32
        )

        # ---- text-presence flag (tokens are added by the collator) ----
        has_text = bool(self.keep_text)

        return {
            "uid": uid,
            "image": image,
            "labels": labels,
            "has_text": has_text,
            "text": self.get_text(idx),
        }


# ---------- collator (tokenizes the batch in one call) ----------

class TokenizingCollator:
    """Stack images/labels and tokenize the per-sample text strings.

    Tokenizing once per batch (instead of per item in __getitem__) keeps
    DataLoader workers cheap and lets the tokenizer use its fast path.
    """

    def __init__(self, tokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        images = torch.stack([b["image"] for b in batch], dim=0)
        labels = torch.stack([b["labels"] for b in batch], dim=0)
        has_text = torch.tensor([b["has_text"] for b in batch], dtype=torch.bool)
        uids = torch.tensor([b["uid"] for b in batch], dtype=torch.long)

        texts = [b["text"] for b in batch]
        input_ids, attention_mask = tokenize_batch(
            self.tokenizer, texts, max_length=self.max_length
        )
        # Zero the rows that don't actually have text — keeps shape uniform
        # but guarantees the encoder sees nothing for those rows.
        if not has_text.all():
            mask_rows = (~has_text).nonzero(as_tuple=False).squeeze(-1)
            input_ids[mask_rows] = 0
            attention_mask[mask_rows] = 0

        return {
            "uids": uids,
            "image": images,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "has_text": has_text,
            "labels": labels,
        }


# ---------- partition loading ----------

@dataclass
class PartitionPaths:
    csv_path: str
    image_root: str
    partition_root: str          # e.g. data_partition/2clients
    scenario: str                # e.g. s1_33_67


def _load_json(p: str) -> dict:
    with open(p, "r") as f:
        return json.load(f)


def load_prepared_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df["has_text"] == 1].copy()
    df = df.drop_duplicates(subset=["uid"], keep="first")
    df["uid"] = df["uid"].astype(int)
    df = df.set_index("uid", drop=False)
    return df


def load_partition(paths: PartitionPaths):
    """Return (per-client uid lists, val_uids, test_uids, modality_config)."""
    scen_dir = os.path.join(paths.partition_root, paths.scenario)
    part = _load_json(os.path.join(scen_dir, "partition.json"))
    eval_split = _load_json(os.path.join(paths.partition_root, "_global_eval.json"))
    modality = _load_json(os.path.join(paths.partition_root, "_modality_config.json"))

    client_uids = {k: [int(u) for u in v] for k, v in part.items()}
    val_uids = [int(u) for u in eval_split["val"]]
    test_uids = [int(u) for u in eval_split["test"]]
    return client_uids, val_uids, test_uids, modality


# ---------- loader builders ----------

def build_client_loaders(
    paths: PartitionPaths,
    *,
    tokenizer,
    image_size: int = 224,
    train_batch_size: int = 16,
    num_workers: int = 4,
    pin_memory: bool = True,
    max_text_length: int = 256,
    seed: int = 42,
) -> Tuple[Dict[str, DataLoader], Dict[str, int], Dict[str, dict]]:
    """Build a training DataLoader per client (modality config respected).

    Returns:
        loaders        : {client_id: DataLoader}
        n_train        : {client_id: num_samples}  (for FedAvg weighting)
        modality       : modality config dict
    """
    df = load_prepared_csv(paths.csv_path)
    client_uids, _val, _test, modality = load_partition(paths)

    train_tf = build_image_transform(image_size=image_size, train=True)
    collate = TokenizingCollator(tokenizer, max_length=max_text_length)

    g = torch.Generator(); g.manual_seed(seed)

    loaders: Dict[str, DataLoader] = {}
    n_train: Dict[str, int] = {}
    for cid, uids in client_uids.items():
        keep_text = bool(modality.get(cid, {}).get("text", False))
        ds = PairedCXRDataset(
            df_uid_indexed=df, uids=uids,
            image_root=paths.image_root, image_tf=train_tf,
            keep_text=keep_text,
        )
        loaders[cid] = DataLoader(
            ds, batch_size=train_batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
            collate_fn=collate, drop_last=False, generator=g,
            persistent_workers=(num_workers > 0),
        )
        n_train[cid] = len(ds)
    return loaders, n_train, modality


def build_global_eval_loaders(
    paths: PartitionPaths,
    *,
    tokenizer,
    image_size: int = 224,
    eval_batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    max_text_length: int = 256,
) -> Tuple[DataLoader, DataLoader]:
    """Build (val_loader, test_loader). Text is always carried along; the
    server decides per-protocol whether to mask it at forward time."""
    df = load_prepared_csv(paths.csv_path)
    _client_uids, val_uids, test_uids, _modality = load_partition(paths)

    eval_tf = build_image_transform(image_size=image_size, train=False)
    collate = TokenizingCollator(tokenizer, max_length=max_text_length)

    def _mk(uids):
        ds = PairedCXRDataset(
            df_uid_indexed=df, uids=uids,
            image_root=paths.image_root, image_tf=eval_tf,
            keep_text=True,
        )
        return DataLoader(
            ds, batch_size=eval_batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
            collate_fn=collate, drop_last=False,
            persistent_workers=(num_workers > 0),
        )

    return _mk(val_uids), _mk(test_uids)


# ---------- helper for the tokenizer ----------

def make_tokenizer(text_backbone: str = "microsoft/BiomedVLP-CXR-BERT-specialized",
                   trust_remote_code: bool = True):
    return build_text_tokenizer(
        backbone_name=text_backbone, trust_remote_code=trust_remote_code
    )
