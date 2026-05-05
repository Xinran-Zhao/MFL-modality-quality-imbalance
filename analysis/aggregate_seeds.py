"""Aggregate multi-seed test results across scenarios.

Given a run-tag directory laid out as

    runs/<RUN_TAG>/
      s1_33_67/seed_42/test_results.json
      s1_33_67/seed_43/test_results.json
      ...
      s3_67_33/seed_46/test_results.json

this script reads every `test_results.json`, computes mean +/- std across
seeds for each (scenario, protocol) cell, and writes:

    runs/<RUN_TAG>/seed_summary.json   # full structured summary
    runs/<RUN_TAG>/seed_summary.csv    # one row per (scenario, protocol)

It also reports per-label AUROC mean/std so you can spot which labels
the contrastive signal is helping or hurting.

Usage:
    python -m analysis.aggregate_seeds runs/exp_12345
    python -m analysis.aggregate_seeds runs/exp_12345 --scenarios s1_33_67 s2_50_50
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional


PROTOCOLS = ("multimodal", "image_only")
DEFAULT_SCENARIOS = ("s1_33_67", "s2_50_50", "s3_67_33")


def _mean_std(xs: List[float]) -> Dict[str, float]:
    """Sample mean and std (ddof=1). Drops None/NaN entries."""
    vals = [float(x) for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    n = len(vals)
    if n == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    mean = sum(vals) / n
    if n == 1:
        return {"mean": mean, "std": float("nan"), "n": 1}
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    return {"mean": mean, "std": math.sqrt(var), "n": n}


def _collect_seed_files(run_dir: Path, scenario: str) -> List[Path]:
    return sorted((run_dir / scenario).glob("seed_*/test_results.json"))


def aggregate(run_dir: Path, scenarios: List[str]) -> Dict:
    summary: Dict[str, Dict] = {}

    for scen in scenarios:
        files = _collect_seed_files(run_dir, scen)
        if not files:
            print(f"[WARN] no seed runs found for {scen} under {run_dir}")
            continue

        per_seed = []
        for fp in files:
            with open(fp) as f:
                data = json.load(f)
            seed = fp.parent.name.replace("seed_", "")
            per_seed.append({
                "seed": seed,
                "best_round": data.get("best_round"),
                "test": data.get("test", {}),
                "path": str(fp),
            })

        scen_block = {
            "n_seeds": len(per_seed),
            "seeds": [s["seed"] for s in per_seed],
            "best_rounds": [s["best_round"] for s in per_seed],
            "protocols": {},
        }

        for proto in PROTOCOLS:
            macros = [s["test"].get(proto, {}).get("auroc_macro") for s in per_seed]
            bces   = [s["test"].get(proto, {}).get("bce")          for s in per_seed]

            # Per-label dict: collect across seeds, aggregate per label name
            per_label_lists: Dict[str, List[float]] = {}
            for s in per_seed:
                pl = s["test"].get(proto, {}).get("auroc_per_label", {}) or {}
                for lname, lval in pl.items():
                    per_label_lists.setdefault(lname, []).append(lval)

            scen_block["protocols"][proto] = {
                "auroc_macro": _mean_std(macros),
                "bce":         _mean_std(bces),
                "auroc_per_label": {
                    ln: _mean_std(vs) for ln, vs in per_label_lists.items()
                },
                "raw_macros": macros,
            }

        summary[scen] = scen_block

    return summary


def write_csv(summary: Dict, csv_path: Path) -> None:
    """Flat one-row-per-(scenario, protocol) summary, easy to paste into a paper."""
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "protocol", "n_seeds",
                    "auroc_macro_mean", "auroc_macro_std",
                    "bce_mean", "bce_std",
                    "raw_macros"])
        for scen, block in summary.items():
            for proto, pblock in block["protocols"].items():
                m = pblock["auroc_macro"]; b = pblock["bce"]
                w.writerow([
                    scen, proto, m["n"],
                    f"{m['mean']:.4f}", (f"{m['std']:.4f}" if m['n'] > 1 else ""),
                    f"{b['mean']:.4f}", (f"{b['std']:.4f}" if b['n'] > 1 else ""),
                    ";".join(f"{x:.4f}" if x is not None else "" for x in pblock["raw_macros"]),
                ])


def print_table(summary: Dict) -> None:
    print(f"\n{'scenario':12s} {'protocol':12s} {'n':>3s}  {'macro AUROC':>20s}  {'BCE':>16s}")
    print("-" * 70)
    for scen, block in summary.items():
        for proto, pblock in block["protocols"].items():
            m = pblock["auroc_macro"]; b = pblock["bce"]
            macro = (f"{m['mean']:.4f} +/- {m['std']:.4f}"
                     if m['n'] > 1 else f"{m['mean']:.4f}")
            bce_s = (f"{b['mean']:.4f}+/-{b['std']:.4f}"
                     if b['n'] > 1 else f"{b['mean']:.4f}")
            print(f"{scen:12s} {proto:12s} {m['n']:>3d}  {macro:>20s}  {bce_s:>16s}")
    print()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path,
                    help="Path to runs/<RUN_TAG>/ containing per-scenario subdirs.")
    ap.add_argument("--scenarios", nargs="+", default=list(DEFAULT_SCENARIOS))
    ap.add_argument("--out-prefix", default="seed_summary",
                    help="Output filename prefix written under run_dir.")
    args = ap.parse_args(argv)

    if not args.run_dir.exists():
        raise SystemExit(f"run_dir does not exist: {args.run_dir}")

    summary = aggregate(args.run_dir, args.scenarios)

    json_path = args.run_dir / f"{args.out_prefix}.json"
    csv_path  = args.run_dir / f"{args.out_prefix}.csv"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    write_csv(summary, csv_path)

    print_table(summary)
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
