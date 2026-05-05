# SLURM job templates

## `train.sbatch`

Submits **3 GPU jobs in parallel** (one per scenario) as a SLURM job array.

### Usage

```bash
# from repo root
sbatch slurm/train.sbatch

# Optionally tag the run (all 3 array tasks share the tag, write to runs/<tag>/<scenario>/)
RUN_TAG=exp01_baseline sbatch slurm/train.sbatch

# Use a different env or repo path
CONDA_ENV=/path/to/env REPO_DIR=/path/to/repo sbatch slurm/train.sbatch
```

### What you get

```
slurm/logs/train_<jobid>_1.out   # scenario s1_33_67
slurm/logs/train_<jobid>_2.out   # scenario s2_50_50
slurm/logs/train_<jobid>_3.out   # scenario s3_67_33

runs/exp_<jobid>/
  s1_33_67/{metrics.jsonl, args.json, best.pt, last.pt, test_results.json}
  s2_50_50/...
  s3_67_33/...
  summary.json     # overall (written per-task; one file per scenario dir)
```

### Resource flags to tune

- `--gres=gpu:a100:1` — change to `gpu:1` for any GPU, or `gpu:v100:1` etc. See
  [Sol Hardware request guide](https://asurc.atlassian.net/wiki/spaces/RC/pages/1908998178/Sol+Hardware+-+How+to+Request).
- `--time=0-02:00:00` — wall-clock cap. R=50 with batch=16 finishes in <1h on A100.
- `--mem=32G`, `--cpus-per-task=4` — comfortable defaults; lower if you queue elsewhere.

### Monitoring

```bash
squeue -u $USER                          # see all 3 array tasks
sacct -j <jobid> --format=JobID,Elapsed,MaxRSS,State
tail -f slurm/logs/train_<jobid>_1.out   # live log
```

### Cancelling

```bash
scancel <jobid>          # kill all 3
scancel <jobid>_2        # kill only the s2 task
```
