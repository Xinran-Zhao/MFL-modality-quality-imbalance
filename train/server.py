"""Vanilla synchronous FedAvg server.

Per round:
    1. Broadcast the current global state_dict to every client (deep copy).
    2. Each client runs `local_train` and returns its updated state_dict.
    3. Server averages state_dicts weighted by num_train_samples.
    4. Server evaluates the new global model on the global val set under
       both protocols (multimodal and image-only).
    5. Server appends a row to metrics.jsonl, saves last.pt every round,
       and saves best.pt whenever val_macro_auroc_multimodal improves.

State_dict aggregation
----------------------
We average ALL parameters and float buffers (BatchNorm running stats,
the learnable `log_inv_temp`, etc.). Integer buffers (e.g.,
`num_batches_tracked`) are taken from client_0.
"""
from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from model.model import MFLModel
from model.losses import CombinedLoss, linear_lambda_warmup
from .client import ClientConfig, local_train


# ---------- FedAvg ----------

def fedavg_state_dicts(
    states: List[Dict[str, torch.Tensor]],
    weights: List[float],
) -> Dict[str, torch.Tensor]:
    """Weighted average of state_dicts.

    Float tensors are averaged; non-float tensors (e.g.
    `num_batches_tracked`) are copied from the first client.
    """
    assert len(states) == len(weights) and len(states) > 0
    total = float(sum(weights))
    assert total > 0, "FedAvg weights sum to zero"
    norm_w = [w / total for w in weights]

    out: Dict[str, torch.Tensor] = {}
    keys = list(states[0].keys())
    for k in keys:
        ref = states[0][k]
        if torch.is_floating_point(ref):
            acc = torch.zeros_like(ref, dtype=torch.float32)
            for s, w in zip(states, norm_w):
                acc += s[k].to(dtype=torch.float32) * w
            out[k] = acc.to(dtype=ref.dtype)
        else:
            out[k] = ref.clone()
    return out


# ---------- evaluation ----------

@torch.no_grad()
def evaluate(
    model: MFLModel,
    loader: DataLoader,
    device: torch.device,
    *,
    protocol: str,                # "multimodal" or "image_only"
    label_names: List[str],
) -> Dict[str, object]:
    """Compute per-label AUROC + macro AUROC + mean BCE on the loader.

    "multimodal" : pass text through (using batch's has_text flags).
    "image_only" : force has_text=False for the whole batch.
    """
    assert protocol in ("multimodal", "image_only")
    model.eval()

    all_logits: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []
    bce_sum = 0.0
    n = 0

    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attn = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        if protocol == "multimodal":
            has_text = batch["has_text"].to(device, non_blocking=True)
        else:
            has_text = torch.zeros(image.shape[0], dtype=torch.bool, device=device)

        outputs = model(
            image=image,
            input_ids=input_ids,
            attention_mask=attn,
            has_text=has_text,
            apply_modality_dropout=False,
        )
        logits = outputs["logits"]
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, labels, reduction="sum"
        )
        bce_sum += float(bce.item())
        n += int(labels.numel())

        all_logits.append(logits.detach().cpu())
        all_targets.append(labels.detach().cpu())

    logits = torch.cat(all_logits, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))

    per_label = _per_label_auroc(probs, targets)
    macro = float(np.nanmean([v for v in per_label.values() if not np.isnan(v)])) \
        if any(not np.isnan(v) for v in per_label.values()) else float("nan")

    return {
        "auroc_macro": macro,
        "auroc_per_label": {k: (None if np.isnan(v) else float(v))
                            for k, v in zip(label_names, per_label.values())},
        "bce": bce_sum / max(n, 1),
        "n_samples": int(targets.shape[0]),
    }


def _per_label_auroc(probs: np.ndarray, targets: np.ndarray) -> Dict[int, float]:
    """Per-label AUROC via the rank-sum formula (no sklearn dependency).

    Returns nan for labels that are all-positive or all-negative in the eval set.
    """
    out: Dict[int, float] = {}
    L = probs.shape[1]
    for j in range(L):
        y = targets[:, j].astype(np.int64)
        s = probs[:, j].astype(np.float64)
        n_pos = int(y.sum())
        n_neg = int(y.shape[0] - n_pos)
        if n_pos == 0 or n_neg == 0:
            out[j] = float("nan")
            continue
        # Rank-sum (handles ties via average rank)
        order = np.argsort(s, kind="mergesort")
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, len(s) + 1, dtype=np.float64)
        # Average ranks for ties
        s_sorted = s[order]
        i = 0
        while i < len(s_sorted):
            j2 = i
            while j2 + 1 < len(s_sorted) and s_sorted[j2 + 1] == s_sorted[i]:
                j2 += 1
            if j2 > i:
                avg = ranks[order[i:j2 + 1]].mean()
                ranks[order[i:j2 + 1]] = avg
            i = j2 + 1
        sum_ranks_pos = float(ranks[y == 1].sum())
        auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
        out[j] = float(auc)
    return out


# ---------- server ----------

@dataclass
class ServerConfig:
    rounds: int = 50
    lambda_final: float = 0.5
    lambda_warmup_rounds: int = 10
    label_names: Optional[List[str]] = None
    output_dir: str = "runs/run"


class FederatedServer:
    """Vanilla synchronous FedAvg orchestrator."""

    def __init__(
        self,
        global_model: MFLModel,
        loss_fn: CombinedLoss,
        client_configs: Dict[str, ClientConfig],
        client_loaders: Dict[str, DataLoader],
        client_weights: Dict[str, int],
        val_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device,
        cfg: ServerConfig,
    ):
        self.model = global_model.to(device)
        self.loss_fn = loss_fn.to(device)
        self.client_configs = client_configs
        self.client_loaders = client_loaders
        self.client_weights = client_weights
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.cfg = cfg

        self.label_names = cfg.label_names or [f"label_{i}" for i in range(8)]
        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.output_dir / "metrics.jsonl"
        self.best_path = self.output_dir / "best.pt"
        self.last_path = self.output_dir / "last.pt"

        self.best_macro = -float("inf")
        self.best_round = -1

    # ---- one round ----

    def _run_one_client(self, cid: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        """Deep-copy global model, run local_train, return (state_dict, metrics)."""
        local_model = copy.deepcopy(self.model).to(self.device)
        metrics = local_train(
            model=local_model,
            loader=self.client_loaders[cid],
            loss_fn=self.loss_fn,
            cfg=self.client_configs[cid],
            device=self.device,
        )
        state = {k: v.detach().cpu().clone() for k, v in local_model.state_dict().items()}
        del local_model
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return state, metrics

    def run_round(self, round_idx: int) -> Dict:
        # ---- update lambda for this round ----
        lam = linear_lambda_warmup(
            current_step=round_idx,
            warmup_steps=self.cfg.lambda_warmup_rounds,
            lambda_final=self.cfg.lambda_final,
            lambda_init=0.0,
        )
        self.loss_fn.set_lambda(lam)

        # ---- local training (sequential for simplicity; clients are independent) ----
        states: List[Dict[str, torch.Tensor]] = []
        weights: List[float] = []
        train_block: Dict[str, Dict[str, float]] = {}
        for cid in self.client_loaders.keys():
            s, m = self._run_one_client(cid)
            states.append(s)
            weights.append(float(self.client_weights[cid]))
            train_block[cid] = m

        # ---- aggregate ----
        new_state = fedavg_state_dicts(states, weights)
        self.model.load_state_dict(new_state, strict=True)

        # ---- evaluate ----
        val_mm = evaluate(self.model, self.val_loader, self.device,
                          protocol="multimodal", label_names=self.label_names)
        val_im = evaluate(self.model, self.val_loader, self.device,
                          protocol="image_only", label_names=self.label_names)

        # Use the multimodal protocol's macro AUROC for selection
        macro_mm = val_mm["auroc_macro"]
        improved = macro_mm > self.best_macro
        if improved:
            self.best_macro = macro_mm
            self.best_round = round_idx

        # Pull current LRs from any client metrics (they're rebuilt per round)
        any_cid = next(iter(train_block))
        record = {
            "round": round_idx,
            "lambda": float(lam),
            "temperature": float(self.model.temperature.detach().cpu().item()),
            "train": train_block,
            "val": {"multimodal": val_mm, "image_only": val_im},
            "lr": {
                "backbone": train_block[any_cid].get("lr_backbone"),
                "head":     train_block[any_cid].get("lr_head"),
            },
            "best_so_far": {"round": self.best_round, "val_auroc_macro_mm": self.best_macro},
        }
        self._append_metrics(record)
        self._save_ckpt(self.last_path, round_idx, record)
        if improved:
            self._save_ckpt(self.best_path, round_idx, record)
        return record

    # ---- final test ----

    def test_with_best(self) -> Dict:
        """Reload best.pt and evaluate on the global test set under both protocols."""
        if not self.best_path.exists():
            raise FileNotFoundError(f"no best ckpt at {self.best_path}")
        ckpt = torch.load(self.best_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"], strict=True)

        test_mm = evaluate(self.model, self.test_loader, self.device,
                           protocol="multimodal", label_names=self.label_names)
        test_im = evaluate(self.model, self.test_loader, self.device,
                           protocol="image_only", label_names=self.label_names)
        out = {
            "best_round": ckpt.get("round"),
            "test": {"multimodal": test_mm, "image_only": test_im},
        }
        with open(self.output_dir / "test_results.json", "w") as f:
            json.dump(out, f, indent=2)
        return out

    # ---- io helpers ----

    def _append_metrics(self, record: Dict) -> None:
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def _save_ckpt(self, path: Path, round_idx: int, record: Dict) -> None:
        torch.save({
            "round": round_idx,
            "model_state": {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
            "metrics": record,
        }, path)
