#!/bin/bash
# ============================================================================
# MFL trainer — parallel job array (one GPU job per scenario)
#
# Submit:    sbatch slurm/train.sbatch
# Status:    squeue -u $USER
# Cancel:    scancel <jobid>
# Logs:      slurm/logs/train_<jobid>_<task>.{out,err}
#
# The --array=1-3 line below launches THREE independent jobs in parallel,
# each on its own GPU, one per scenario:
#       task 1 -> s1_33_67
#       task 2 -> s2_50_50
#       task 3 -> s3_67_33
#
# All 3 share the same global val/test split, so their results are directly
# comparable. Each job writes to its own subdirectory under runs/${RUN_TAG}/.
#
# Sol GPU request reference:
#   https://asurc.atlassian.net/wiki/spaces/RC/pages/1908998178/Sol+Hardware+-+How+to+Request
# ============================================================================

#SBATCH --job-name=mfl_fedavg
#SBATCH --array=1-3
#SBATCH --partition=general
#SBATCH --qos=public
#SBATCH --gres=gpu:a100:1            # change to gpu:1 for any-GPU, or gpu:v100:1 etc.
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-02:00:00            # 2h is generous for R=50, batch=16 on A100
#SBATCH --output=slurm/logs/train_%A_%a.out
#SBATCH --error=slurm/logs/train_%A_%a.err
#SBATCH --mail-type=END,FAIL
# #SBATCH --mail-user=YOUR_ASURITE@asu.edu   # uncomment + set to receive emails

set -euo pipefail

# ---- map array index -> scenario ----
SCENARIOS=(_ s1_33_67 s2_50_50 s3_67_33)        # index 0 unused; 1..3 are the runs
SCENARIO="${SCENARIOS[$SLURM_ARRAY_TASK_ID]}"

# ---- run tag (shared by all 3 array tasks of the same submission) ----
RUN_TAG="${RUN_TAG:-exp_${SLURM_ARRAY_JOB_ID}}"
OUTPUT_DIR="runs/${RUN_TAG}"

# ---- repo + env ----
REPO_DIR="${REPO_DIR:-$HOME/MFL-modality-quality-imbalance}"
CONDA_ENV="${CONDA_ENV:-/scratch/$USER/.conda/envs/multimodal-fl}"

cd "$REPO_DIR"
mkdir -p slurm/logs "$OUTPUT_DIR"

# Activate conda env (works for both conda and mamba layouts on Sol)
if   [ -f "/packages/apps/anaconda3/etc/profile.d/conda.sh" ]; then source /packages/apps/anaconda3/etc/profile.d/conda.sh
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ];          then source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi
conda activate "$CONDA_ENV"
export PYTHONNOUSERSITE=1                       # ignore ~/.local stale packages

# ---- diagnostics ----
echo "------------------------------------------------------------"
echo "Host:        $(hostname)"
echo "Job ID:      $SLURM_JOB_ID  (array $SLURM_ARRAY_JOB_ID task $SLURM_ARRAY_TASK_ID)"
echo "Scenario:    $SCENARIO"
echo "Run tag:     $RUN_TAG"
echo "Output:      $OUTPUT_DIR"
echo "Python:      $(which python)"
echo "GPU:"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
echo "------------------------------------------------------------"

# ---- launch ----
srun python -m train.cli \
    --scenario "$SCENARIO" \
    --device cuda \
    --num-workers 4 \
    --output-dir "$OUTPUT_DIR" \
    --rounds 50 \
    --local-epochs 1 \
    --train-batch-size 16 \
    --eval-batch-size 32 \
    --lr-backbone 1e-5 \
    --lr-head 1e-4 \
    --lambda-final 0.5 \
    --lambda-warmup-rounds 10 \
    --modality-dropout-p 0.15 \
    --seed 42

echo "Task $SLURM_ARRAY_TASK_ID ($SCENARIO) finished at $(date)"
