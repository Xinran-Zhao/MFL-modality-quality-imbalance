"""CLI for partitioning Indiana CXR paired samples into FL clients.

Pipeline (Option C eval design):
  1. Load /data/.../prepared_data.csv, filter has_text==1, dedupe to 1 frontal image per uid.
  2. Carve a SHARED stratified global val/test holdout (default 15%/15% of paired pool).
     -> data_partition/2clients/_global_eval.json   (val + test uid lists)
     -> data_partition/2clients/_global_meta.json   (sizes, label counts, source paths)
     -> data_partition/2clients/_modality_config.json (per-client modality flags)
  3. For each scenario (client sample-size ratio), stratify-split the remaining
     training pool across clients.
     -> data_partition/2clients/<scenario>/partition.json       (per-client TRAIN uids)
     -> data_partition/2clients/<scenario>/partition_meta.json  (sizes + label counts)

Usage examples:
  # generate all 3 default scenarios in one go
  python partition_2clients.py --all-default

  # one explicit scenario (rerunning rewrites only its files + the shared global holdout)
  python partition_2clients.py --ratios 0.5,0.5 --scenario_name s2_50_50

Reproducibility:
  Default scenarios use seed = base_seed + scenario_index (1..N).
  Custom scenarios use seed = base_seed + 100 + stable_md5_offset(scenario_name).
  Both are deterministic given (base_seed, scenario_name).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# make sibling partition_lib.py importable regardless of cwd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from partition_lib import (  # noqa: E402
    LABEL_COLS,
    carve_global_holdout,
    label_counts,
    load_paired_dedup,
    partition_clients,
    stable_offset,
)

DEFAULT_DATA_CSV = "/data/amciilab/xinran/indiana_cxr/prepared/prepared_data.csv"
DEFAULT_IMAGE_ROOT = "/data/amciilab/xinran/indiana_cxr/images"

DEFAULT_SCENARIOS = {
    "s1_33_67": (1.0 / 3.0, 2.0 / 3.0),
    "s2_50_50": (0.5, 0.5),
    "s3_67_33": (2.0 / 3.0, 1.0 / 3.0),
}

DEFAULT_MODALITY = {
    "client_0": {"image": True,  "text": True},
    "client_1": {"image": True,  "text": False},
}


# ---------- CLI helpers ----------

def parse_ratios(s: str):
    parts = tuple(float(x) for x in s.split(","))
    if abs(sum(parts) - 1.0) > 1e-6:
        raise argparse.ArgumentTypeError(
            f"--ratios must sum to 1.0, got {parts} sum={sum(parts):.4f}")
    return parts


def scenario_seed(base_seed: int, name: str) -> int:
    """Deterministic seed per scenario name; matches all-default ordering when applicable."""
    if name in DEFAULT_SCENARIOS:
        return base_seed + list(DEFAULT_SCENARIOS).index(name) + 1
    return base_seed + 100 + stable_offset(name)


# ---------- writers ----------

def write_global(out_dir: Path, paired_df, val_uids, test_uids, args, n_modalities):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "_global_eval.json", "w") as f:
        json.dump({"val": val_uids, "test": test_uids}, f, indent=2)
    meta = {
        "source_csv": args.data_csv,
        "image_root": DEFAULT_IMAGE_ROOT,
        "paired_only": bool(args.paired_only),
        "dedup_by_uid": bool(args.dedup),
        "total_pool_after_filter": int(len(paired_df)),
        "val_frac": args.val_frac,
        "test_frac": args.test_frac,
        "seed": args.seed,
        "n_val": len(val_uids),
        "n_test": len(test_uids),
        "label_names": LABEL_COLS,
        "label_counts_val": label_counts(paired_df, val_uids),
        "label_counts_test": label_counts(paired_df, test_uids),
        "eval_protocol": (
            "Same val/test uids evaluated under TWO modality conditions: "
            "(a) multimodal (image+text), (b) image-only (text masked)."
        ),
        "num_clients": args.num_clients,
    }
    with open(out_dir / "_global_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    with open(out_dir / "_modality_config.json", "w") as f:
        json.dump(DEFAULT_MODALITY, f, indent=2)


def write_scenario(scen_dir: Path, paired_df, name, ratios, client_uids, seed):
    scen_dir.mkdir(parents=True, exist_ok=True)
    partition = {f"client_{i}": uids for i, uids in enumerate(client_uids)}
    with open(scen_dir / "partition.json", "w") as f:
        json.dump(partition, f, indent=2)
    meta = {
        "scenario": name,
        "client_ratios": list(ratios),
        "seed": seed,
        "num_clients": len(client_uids),
        "label_names": LABEL_COLS,
        "note": (
            "partition.json contains TRAIN-ONLY uids per client. "
            "Val/test live in ../_global_eval.json."
        ),
        "clients": {
            f"client_{i}": {
                "n_train": len(uids),
                "label_counts_train": label_counts(paired_df, uids),
            }
            for i, uids in enumerate(client_uids)
        },
    }
    with open(scen_dir / "partition_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


# ---------- main ----------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--data_csv", default=DEFAULT_DATA_CSV,
                    help="path to prepared_data.csv")
    ap.add_argument("--out_dir",
                    default=str(Path(__file__).resolve().parent / "2clients"),
                    help="output directory (per-scenario subfolders go inside)")
    ap.add_argument("--num_clients", type=int, default=2,
                    help="number of clients per scenario (length of --ratios)")
    ap.add_argument("--ratios", type=parse_ratios,
                    help='comma-separated client sample-size ratios summing to 1, e.g. "0.333,0.667"')
    ap.add_argument("--scenario_name", help='used together with --ratios, e.g. "s1_33_67"')
    ap.add_argument("--val_frac", type=float, default=0.15,
                    help="fraction of paired pool reserved for the global val set")
    ap.add_argument("--test_frac", type=float, default=0.15,
                    help="fraction of paired pool reserved for the global test set")
    ap.add_argument("--seed", type=int, default=42,
                    help="base RNG seed (global holdout uses --seed; scenarios derive from it)")
    ap.add_argument("--paired_only", action=argparse.BooleanOptionalAction, default=True,
                    help="keep only rows with has_text==1")
    ap.add_argument("--dedup", action=argparse.BooleanOptionalAction, default=True,
                    help="keep one frontal image per uid")
    ap.add_argument("--all-default", dest="all_default", action="store_true",
                    help="generate all 3 default scenarios (s1_33_67, s2_50_50, s3_67_33)")
    return ap


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)

    if args.all_default and (args.ratios or args.scenario_name):
        ap.error("--all-default cannot be combined with --ratios/--scenario_name")
    if not args.all_default:
        if args.ratios is None or args.scenario_name is None:
            ap.error("must provide both --ratios and --scenario_name (or use --all-default)")
        if len(args.ratios) != args.num_clients:
            ap.error(f"--ratios has {len(args.ratios)} values but --num_clients={args.num_clients}")

    out_dir = Path(args.out_dir)

    # 1) load + dedupe paired pool
    paired = load_paired_dedup(
        args.data_csv, paired_only=args.paired_only, dedup=args.dedup)
    print(f"loaded paired+deduped pool: {len(paired)} samples")

    # 2) shared global holdout
    train_pool, val_uids, test_uids = carve_global_holdout(
        paired, args.val_frac, args.test_frac, seed=args.seed)
    print(f"global holdout (seed={args.seed}): "
          f"train_pool={len(train_pool)}, val={len(val_uids)}, test={len(test_uids)}")
    write_global(out_dir, paired, val_uids, test_uids, args, args.num_clients)
    print(f"wrote shared global files -> {out_dir}/_global_eval.json,"
          f" _global_meta.json, _modality_config.json")

    # 3) per-scenario client splits over the training pool
    if args.all_default:
        scenarios = DEFAULT_SCENARIOS
    else:
        scenarios = {args.scenario_name: tuple(args.ratios)}

    for name, ratios in scenarios.items():
        seed = scenario_seed(args.seed, name)
        client_uids = partition_clients(train_pool, ratios, seed=seed)
        scen_dir = out_dir / name
        write_scenario(scen_dir, paired, name, ratios, client_uids, seed)
        print(f"\n[{name}]  seed={seed}  ratios={ratios}  -> {scen_dir}")
        for i, uids in enumerate(client_uids):
            modality = DEFAULT_MODALITY.get(f"client_{i}", {})
            mod_str = "+".join(k for k, v in modality.items() if v) or "?"
            print(f"  client_{i} [{mod_str:>11s}]: n_train={len(uids):4d}")
        # cross-client label-prevalence sanity print (clients only, not eval)
        if len(client_uids) == 2:
            c0 = label_counts(paired, client_uids[0])
            c1 = label_counts(paired, client_uids[1])
            n0, n1 = max(len(client_uids[0]), 1), max(len(client_uids[1]), 1)
            print("  per-label prevalence (client_0 vs client_1):")
            for lab in LABEL_COLS:
                print(f"    {lab:22s}  c0={c0[lab]/n0:.3f}   c1={c1[lab]/n1:.3f}")


if __name__ == "__main__":
    main()
