"""Command-line entry point for the MFL trainer.

Usage examples
--------------

# Default: run all 3 scenarios sequentially with default hyperparams
python -m train.cli --scenario all

# Single scenario, custom output dir
python -m train.cli --scenario s2_50_50 --output-dir runs/exp01_s2

# Quick smoke test on CPU (1 round, 4 batches per client, no checkpoint write)
python -m train.cli --scenario s1_33_67 --dry-run --device cpu --num-workers 0
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from data.transforms import IMAGENET_MEAN, IMAGENET_STD  # noqa: F401  (sanity import)
from model.model import MFLModel
from model.losses import CombinedLoss

from .client import ClientConfig
from .dataset import (
    LABEL_COLS,
    PartitionPaths,
    build_client_loaders,
    build_global_eval_loaders,
    make_tokenizer,
)
from .server import FederatedServer, ServerConfig


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARTITION_ROOT = str(REPO_ROOT / "data_partition" / "2clients")
DEFAULT_CSV = "/data/amciilab/xinran/indiana_cxr/prepared/prepared_data.csv"
DEFAULT_IMAGE_ROOT = "/data/amciilab/xinran/indiana_cxr/images/images_normalized"
DEFAULT_SCENARIOS = ["s1_33_67", "s2_50_50", "s3_67_33"]


# ---------- args ----------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MFL trainer (vanilla FedAvg, synchronous).")

    # ---- data ----
    p.add_argument("--csv-path", default=DEFAULT_CSV)
    p.add_argument("--image-root", default=DEFAULT_IMAGE_ROOT)
    p.add_argument("--partition-root", default=DEFAULT_PARTITION_ROOT)
    p.add_argument("--scenario", default="all",
                   help="s1_33_67 | s2_50_50 | s3_67_33 | all")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--max-text-length", type=int, default=256)

    # ---- model ----
    p.add_argument("--embed-dim", type=int, default=256)
    p.add_argument("--contrastive-dim", type=int, default=128)
    p.add_argument("--head-hidden", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--image-backbone", default="resnet50")
    p.add_argument("--text-backbone", default="microsoft/BiomedVLP-CXR-BERT-specialized")
    p.add_argument("--no-pretrained-image", action="store_true")
    p.add_argument("--no-pretrained-text", action="store_true")
    p.add_argument("--contrastive-temp-init", type=float, default=0.07)
    p.add_argument("--modality-dropout-p", type=float, default=0.15,
                   help="Modality dropout prob applied on the multimodal client only.")

    # ---- training ----
    p.add_argument("--rounds", type=int, default=50)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--train-batch-size", type=int, default=16)
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--lr-backbone", type=float, default=1e-5)
    p.add_argument("--lr-head", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--lambda-final", type=float, default=0.5)
    p.add_argument("--lambda-warmup-rounds", type=int, default=10)

    # ---- runtime ----
    p.add_argument("--device", default="auto", help="auto | cuda | cpu")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default=str(REPO_ROOT / "runs"))
    p.add_argument("--run-subdir", default=None,
                   help="Override the per-scenario output subdir under --output-dir. "
                        "Default is the scenario name (e.g., 's1_33_67'). For multi-seed "
                        "sweeps, set to e.g. 's1_33_67/seed_42'.")
    p.add_argument("--dry-run", action="store_true",
                   help="1 round, 4 batches/client, no test, no eval-loader workers.")
    return p


# ---------- helpers ----------

def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_scenarios(name: str) -> List[str]:
    if name == "all":
        return list(DEFAULT_SCENARIOS)
    return [name]


# ---------- single-scenario run ----------

def run_scenario(args: argparse.Namespace, scenario: str, device: torch.device) -> Dict:
    set_seed(args.seed)
    print(f"\n=== scenario {scenario} | device={device} ===", flush=True)

    paths = PartitionPaths(
        csv_path=args.csv_path,
        image_root=args.image_root,
        partition_root=args.partition_root,
        scenario=scenario,
    )
    subdir = args.run_subdir if args.run_subdir is not None else scenario
    out_dir = Path(args.output_dir) / subdir

    # ---- tokenizer + loaders ----
    tokenizer = make_tokenizer(text_backbone=args.text_backbone)
    nw_train = 0 if args.dry_run else args.num_workers
    nw_eval = 0 if args.dry_run else args.num_workers
    pin = (device.type == "cuda")

    client_loaders, n_train, modality = build_client_loaders(
        paths, tokenizer=tokenizer,
        image_size=args.image_size,
        train_batch_size=args.train_batch_size,
        num_workers=nw_train, pin_memory=pin,
        max_text_length=args.max_text_length, seed=args.seed,
    )
    val_loader, test_loader = build_global_eval_loaders(
        paths, tokenizer=tokenizer,
        image_size=args.image_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=nw_eval, pin_memory=pin,
        max_text_length=args.max_text_length,
    )

    print(f"  client sizes: { {k: n_train[k] for k in client_loaders} }", flush=True)
    print(f"  modality:     { modality }", flush=True)
    print(f"  val/test:     {len(val_loader.dataset)} / {len(test_loader.dataset)}",
          flush=True)

    # ---- model + loss ----
    model = MFLModel(
        embed_dim=args.embed_dim,
        contrastive_dim=args.contrastive_dim,
        num_labels=len(LABEL_COLS),
        head_hidden=args.head_hidden,
        dropout=args.dropout,
        modality_dropout_p=args.modality_dropout_p,
        image_backbone=args.image_backbone,
        text_backbone=args.text_backbone,
        text_trust_remote_code=True,
        pretrained_image=not args.no_pretrained_image,
        pretrained_text=not args.no_pretrained_text,
        contrastive_temp_init=args.contrastive_temp_init,
    )
    loss_fn = CombinedLoss(lambda_contrast=0.0, pos_weight=None)

    # ---- per-client configs (only c0 gets modality dropout) ----
    client_configs: Dict[str, ClientConfig] = {}
    for cid in client_loaders:
        client_configs[cid] = ClientConfig(
            client_id=cid,
            lr_backbone=args.lr_backbone,
            lr_head=args.lr_head,
            weight_decay=args.weight_decay,
            grad_clip_norm=args.grad_clip,
            local_epochs=args.local_epochs,
            apply_modality_dropout=bool(modality.get(cid, {}).get("text", False)),
            max_steps=4 if args.dry_run else None,
        )

    # ---- server ----
    server_cfg = ServerConfig(
        rounds=(1 if args.dry_run else args.rounds),
        lambda_final=args.lambda_final,
        lambda_warmup_rounds=args.lambda_warmup_rounds,
        label_names=LABEL_COLS,
        output_dir=str(out_dir),
    )
    server = FederatedServer(
        global_model=model,
        loss_fn=loss_fn,
        client_configs=client_configs,
        client_loaders=client_loaders,
        client_weights=n_train,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        cfg=server_cfg,
    )

    # ---- save run config for reproducibility ----
    (out_dir / "args.json").parent.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "args.json", "w") as f:
        json.dump(vars(args) | {"scenario": scenario,
                                "n_train": n_train,
                                "modality": modality}, f, indent=2)

    # ---- rounds ----
    t0 = time.time()
    for r in range(server_cfg.rounds):
        rt0 = time.time()
        rec = server.run_round(r)
        elapsed = time.time() - rt0
        print(
            f"  round {r:>3d}/{server_cfg.rounds-1} "
            f"| lam={rec['lambda']:.3f} "
            f"| val_mm_auroc={rec['val']['multimodal']['auroc_macro']:.4f} "
            f"| val_im_auroc={rec['val']['image_only']['auroc_macro']:.4f} "
            f"| best={server.best_macro:.4f}@{server.best_round} "
            f"| {elapsed:.1f}s",
            flush=True,
        )
    total = time.time() - t0
    print(f"  scenario {scenario} done in {total/60:.1f} min", flush=True)

    # ---- final test (skip in dry-run) ----
    if not args.dry_run:
        result = server.test_with_best()
        print(f"  TEST best@round={result['best_round']} "
              f"mm_auroc={result['test']['multimodal']['auroc_macro']:.4f} "
              f"im_auroc={result['test']['image_only']['auroc_macro']:.4f}",
              flush=True)
        return result
    return {"dry_run": True, "scenario": scenario}


# ---------- main ----------

def main(argv=None) -> int:
    args = build_argparser().parse_args(argv)
    device = resolve_device(args.device)
    set_seed(args.seed)

    scenarios = resolve_scenarios(args.scenario)
    if args.run_subdir is not None and len(scenarios) > 1:
        raise SystemExit(
            "--run-subdir is only valid with a single --scenario; got "
            f"{scenarios}. (When sweeping multiple scenarios in one process, "
            "let each scenario use its own default subdir.)"
        )
    print(f"Running scenarios: {scenarios}", flush=True)

    summary = {}
    for s in scenarios:
        summary[s] = run_scenario(args, s, device)

    summary_path = Path(args.output_dir) / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
