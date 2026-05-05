# SLURM job templates

## `train.sbatch` — multi-seed sweep (15 GPU jobs in parallel)

Submits **3 scenarios x 5 seeds = 15 GPU jobs** as a single SLURM job array.
All 15 share the same global val/test split, so cross-scenario AND cross-seed
comparisons are apples-to-apples.

### Submit

```bash
# from repo root
sbatch slurm/train.sbatch

# Tag the run (all 15 array tasks share the tag, write to runs/<tag>/...)
RUN_TAG=exp01_baseline sbatch slurm/train.sbatch

# Different env or repo path
CONDA_ENV=/path/to/env REPO_DIR=/path/to/repo sbatch slurm/train.sbatch
```

### Array index -> (scenario, seed) mapping

| task | scenario   | seed |  | task | scenario   | seed |  | task | scenario   | seed |
|-----:|------------|-----:|--|-----:|------------|-----:|--|-----:|------------|-----:|
|    1 | s1_33_67   |   42 |  |    6 | s2_50_50   |   42 |  |   11 | s3_67_33   |   42 |
|    2 | s1_33_67   |   43 |  |    7 | s2_50_50   |   43 |  |   12 | s3_67_33   |   43 |
|    3 | s1_33_67   |   44 |  |    8 | s2_50_50   |   44 |  |   13 | s3_67_33   |   44 |
|    4 | s1_33_67   |   45 |  |    9 | s2_50_50   |   45 |  |   14 | s3_67_33   |   45 |
|    5 | s1_33_67   |   46 |  |   10 | s2_50_50   |   46 |  |   15 | s3_67_33   |   46 |

### Output layout

```
slurm/logs/train_<jobid>_<task>.{out,err}        # 15 log pairs

runs/<RUN_TAG>/
  s1_33_67/seed_42/{metrics.jsonl, args.json, best.pt, last.pt, test_results.json}
  s1_33_67/seed_43/...
  ...
  s3_67_33/seed_46/...
  seed_summary.json     # written by analysis/aggregate_seeds.py (see below)
  seed_summary.csv
```

### Aggregate the 5 seeds per scenario

Once all 15 finish:

```bash
python -m analysis.aggregate_seeds runs/<RUN_TAG>
```

Prints a table and writes `seed_summary.{json,csv}` with **mean +/- std** per
(scenario, protocol) for both macro AUROC and per-label AUROC. Example output:

```
scenario     protocol     n   macro AUROC                BCE
----------------------------------------------------------------------
s1_33_67     multimodal   5   0.7842 +/- 0.0091    0.3215+/-0.0042
s1_33_67     image_only   5   0.7510 +/- 0.0107    0.3398+/-0.0051
s2_50_50     multimodal   5   ...
```

### Resource flags to tune

| Flag | Default | Notes |
|---|---|---|
| `--array=1-15` | all 15 at once | Append `%5` to limit to 5 concurrent: `--array=1-15%5` |
| `--gres=gpu:a100:1` | A100 only | Drop to `gpu:1` for any GPU if A100s queue long |
| `--time=0-02:00:00` | 2h per task | Bump if you raise `--rounds` or `--local-epochs` |
| `--mem=32G` | 32 GB | Comfortable; lower for shared partitions |
| `--cpus-per-task=4` | 4 | Matches `--num-workers 4` in the trainer |

See [Sol Hardware request guide](https://asurc.atlassian.net/wiki/spaces/RC/pages/1908998178/Sol+Hardware+-+How+to+Request)
for partition/QOS/GPU options.

### Monitoring

```bash
squeue -u $USER                              # all 15 array tasks
sacct -j <jobid> --format=JobID,Elapsed,MaxRSS,State
tail -f slurm/logs/train_<jobid>_1.out       # live log of task 1
```

### Cancelling

```bash
scancel <jobid>          # kill all 15
scancel <jobid>_7        # kill only task 7 (s2 seed=43)
scancel <jobid>_[1-5]    # kill all of s1
```

---

## `train_stripped.sbatch` — lexical-strip ablation (1 scenario x 5 seeds)

Runs the same trainer / partitions / seeds as `train.sbatch`, but with a
`prepared_data_stripped.csv` whose `findings_text` has had the 7 disease
label words (and clinical synonyms) replaced by `[DEIDENTIFIED]`. This
measures the **post-leakage multimodal AUROC ceiling** so cross-design
comparisons (lambda sweep, modality-dropout sweep) become interpretable.

### Generate the stripped CSV first

```bash
python -m data_partition.strip_label_words \
    --in  /data/amciilab/xinran/indiana_cxr/prepared/prepared_data.csv \
    --out /data/amciilab/xinran/indiana_cxr/prepared/prepared_data_stripped.csv
```

Verifies down to **0%** label-word presence in `findings_text` for all 7
positive subsets.

### Submit

```bash
# default: s2_50_50, 5 seeds
sbatch slurm/train_stripped.sbatch

# different scenario
SCENARIO=s3_67_33 sbatch slurm/train_stripped.sbatch

# tag the run
RUN_TAG=stripped_diag sbatch slurm/train_stripped.sbatch
```

### Compare

```bash
# original (leaky) and stripped runs in one table
python -m analysis.aggregate_seeds runs/baseline_20260505
python -m analysis.aggregate_seeds runs/stripped_<jobid>
```

If the stripped MM AUROC drops sharply (e.g. 0.98 -> 0.85) while IM AUROC
barely moves, the leakage explanation is confirmed and the stripped MM
AUROC becomes the meaningful baseline for the upcoming `lambda` and
`modality_dropout` sweeps.
