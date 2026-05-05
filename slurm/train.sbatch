#!/bin/bash
# ============================================================================
# MFL trainer — parallel job array (3 scenarios x 5 seeds = 15 GPU jobs)
#
# Submit:    sbatch slurm/train.sbatch
# Status:    squeue -u $USER
# Cancel:    scancel <jobid>           (all 15)
#            scancel <jobid>_7         (just one task)
# Logs:      slurm/logs/train_<jobid>_<task>.{out,err}
#
# Array task -> (scenario, seed) mapping (5 seeds per scenario):
#       1.. 5  ->  s1_33_67   seeds 42, 43, 44, 45, 46
#       6..10  ->  s2_50_50   seeds 42, 43, 44, 45, 46
#      11..15  ->  s3_67_33   seeds 42, 43, 44, 45, 46
#
# All jobs share the same global val/test split, so cross-scenario AND
# cross-seed comparisons are apples-to-apples. Each task writes to its own
# subdirectory: runs/<RUN_TAG>/<scenario>/seed_<seed>/
#
# Use analysis/aggregate_seeds.py after the array finishes to get
# mean +/- std per scenario per protocol.
#
# Sol GPU request reference:
#   https://asurc.atlassian.net/wiki/spaces/RC/pages/1908998178/Sol+Hardware+-+How+to+Request
# ============================================================================

#SBATCH --job-name=mfl_fedavg
#SBATCH --array=1-15
#SBATCH --partition=general
#SBATCH --qos=public
#SBATCH --gres=gpu:a100:1            # change to gpu:1 for any-GPU, or gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-02:00:00            # per task; 2h is generous on A100
#SBATCH --output=slurm/logs/train_%A_%a.out
#SBATCH --error=slurm/logs/train_%A_%a.err
#SBATCH --mail-type=END,FAIL
# #SBATCH --mail-user=YOUR_ASURITE@asu.edu   # uncomment to receive emails
#
# To throttle concurrency (e.g., if Sol queues fill up), append %N to --array:
#     #SBATCH --array=1-15%5         # at most 5 running at a time
# Default above is no limit -> all 15 launch as soon as GPUs free up.

set -euo pipefail

# ---- (scenario, seed) mapping ----
SCENARIOS=(_ s1_33_67 s2_50_50 s3_67_33)        # idx 1..3
SEEDS=(_ 42 43 44 45 46)                        # idx 1..5
N_SEEDS=5

TASK="$SLURM_ARRAY_TASK_ID"
SCEN_IDX=$(( (TASK - 1) / N_SEEDS + 1 ))
SEED_IDX=$(( (TASK - 1) % N_SEEDS + 1 ))
SCENARIO="${SCENARIOS[$SCEN_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"
RUN_SUBDIR="${SCENARIO}/seed_${SEED}"

# ---- run tag (shared across all 15 tasks of one submission) ----
RUN_TAG="${RUN_TAG:-exp_${SLURM_ARRAY_JOB_ID}}"
OUTPUT_DIR="runs/${RUN_TAG}"

# ---- repo + env ----
REPO_DIR="${REPO_DIR:-$HOME/MFL-modality-quality-imbalance}"
CONDA_ENV="${CONDA_ENV:-/scratch/$USER/.conda/envs/multimodal-fl}"

cd "$REPO_DIR"
mkdir -p slurm/logs "$OUTPUT_DIR/$RUN_SUBDIR"

# Activate conda env (handles common Sol/conda layouts)
if   [ -f "/packages/apps/anaconda3/etc/profile.d/conda.sh" ]; then source /packages/apps/anaconda3/etc/profile.d/conda.sh
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ];          then source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi
conda activate "$CONDA_ENV"
export PYTHONNOUSERSITE=1                       # ignore stale ~/.local packages

# ---- diagnostics ----
echo "------------------------------------------------------------"
echo "Host:        $(hostname)"
echo "Job ID:      $SLURM_JOB_ID  (array $SLURM_ARRAY_JOB_ID task $TASK / 15)"
echo "Scenario:    $SCENARIO     (idx $SCEN_IDX)"
echo "Seed:        $SEED          (idx $SEED_IDX)"
echo "Run tag:     $RUN_TAG"
echo "Output:      $OUTPUT_DIR/$RUN_SUBDIR"
echo "Python:      $(which python)"
echo "GPU:"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
echo "------------------------------------------------------------"

# ---- launch ----
srun python -m train.cli \
    --scenario "$SCENARIO" \
    --seed "$SEED" \
    --device cuda \
    --num-workers 4 \
    --output-dir "$OUTPUT_DIR" \
    --run-subdir "$RUN_SUBDIR" \
    --rounds 50 \
    --local-epochs 1 \
    --train-batch-size 16 \
    --eval-batch-size 32 \
    --lr-backbone 1e-5 \
    --lr-head 1e-4 \
    --lambda-final 0.5 \
    --lambda-warmup-rounds 10 \
    --modality-dropout-p 0.15

echo "Task $TASK ($SCENARIO seed=$SEED) finished at $(date)"
