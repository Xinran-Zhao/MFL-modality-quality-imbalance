"""Pure functions for partitioning Indiana CXR paired samples into FL clients.

All public functions are deterministic given a seed and stateless w.r.t. disk I/O.
The CLI lives in `partition_2clients.py`.
"""
from __future__ import annotations

import hashlib
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

LABEL_COLS = [
    "No_Finding", "Cardiomegaly", "Atelectasis", "Pleural_Effusion",
    "Opacity", "Calcinosis", "Calcified_Granuloma", "Airspace_Disease",
]


# ---------- loading ----------

def load_paired_dedup(csv_path: str, paired_only: bool = True,
                      dedup: bool = True) -> pd.DataFrame:
    """Load prepared_data.csv, optionally keep only paired rows and dedupe by uid."""
    df = pd.read_csv(csv_path)
    if paired_only:
        df = df[df["has_text"] == 1]
    if dedup:
        # keep first frontal image per uid (rows are already frontal-only in prepared_data)
        df = df.drop_duplicates(subset=["uid"], keep="first")
    return df.reset_index(drop=True)


# ---------- stratified splitting ----------

def _proportional_split(items: Sequence, ratios: Sequence[float],
                        rng: np.random.Generator) -> List[list]:
    """Split a list of items into len(ratios) chunks via largest-remainder rounding."""
    items = list(items)
    rng.shuffle(items)
    n = len(items)
    raw = [n * r for r in ratios]
    sizes = [int(np.floor(x)) for x in raw]
    remainder = n - sum(sizes)
    fracs = sorted(
        ((raw[i] - sizes[i], i) for i in range(len(ratios))),
        key=lambda t: (-t[0], t[1]),
    )
    for k in range(remainder):
        sizes[fracs[k][1]] += 1
    out, pos = [], 0
    for s in sizes:
        out.append(items[pos:pos + s])
        pos += s
    return out


def stratified_split(df: pd.DataFrame, ratios: Sequence[float],
                     seed: int) -> List[List[int]]:
    """Stratify rows of df by full multi-label signature and split into len(ratios) chunks.

    Returns a list of uid-lists, one per ratio. Per-chunk label distributions are
    near-identical because each label-signature group is split proportionally.
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1.0, got {sum(ratios)}")
    rng = np.random.default_rng(seed)
    work = df.copy()
    work["_sig"] = list(map(tuple, work[LABEL_COLS].astype(int).values))
    chunks: List[list] = [[] for _ in ratios]
    for _, group in work.groupby("_sig", sort=True):
        uids = group["uid"].tolist()
        for i, part in enumerate(_proportional_split(uids, ratios, rng)):
            chunks[i].extend(part)
    return chunks


# ---------- pipeline pieces ----------

def carve_global_holdout(df: pd.DataFrame, val_frac: float, test_frac: float,
                         seed: int) -> Tuple[pd.DataFrame, List[int], List[int]]:
    """Stratified 3-way split into (training pool df, val uids, test uids)."""
    train_frac = 1.0 - val_frac - test_frac
    if train_frac <= 0:
        raise ValueError("val_frac + test_frac must be < 1.0")
    train_uids, val_uids, test_uids = stratified_split(
        df, (train_frac, val_frac, test_frac), seed=seed)
    train_pool = df[df["uid"].isin(train_uids)].reset_index(drop=True)
    return train_pool, val_uids, test_uids


def partition_clients(train_pool: pd.DataFrame, ratios: Sequence[float],
                      seed: int) -> List[List[int]]:
    """Stratified split of the training pool into one uid-list per client."""
    return stratified_split(train_pool, ratios, seed=seed)


def label_counts(df: pd.DataFrame, uids: Sequence[int]) -> dict:
    """Per-label positive counts over the rows of df with uid in uids."""
    sub = df[df["uid"].isin(uids)]
    return {c: int(sub[c].sum()) for c in LABEL_COLS}


# ---------- seed helpers ----------

def stable_offset(name: str, mod: int = 10000) -> int:
    """Stable, run-independent integer offset derived from a name (md5-based)."""
    return int(hashlib.md5(name.encode()).hexdigest(), 16) % mod
