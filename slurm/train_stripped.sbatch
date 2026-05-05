#!/bin/bash
# ============================================================================
# Lexical-strip ablation: same trainer, same partitions, same seeds, but the
# 7 disease label words (and clinical synonyms) have been masked out of
# findings_text via [DEIDENTIFIED]. See data_partition/strip_label_words.py
# for the exact patterns and rationale.
#
# This run answers: "what's the multimodal AUROC ceiling once label words
# are not in the input?" If MM AUROC drops sharply (e.g. 0.97 -> 0.85)
# while IM AUROC barely moves, the leakage explanation is confirmed.
#
# Default scope: 1 scenario x 5 seeds = 5 GPU jobs (cheap diagnostic).
# Bump --array=1-15 (and SCEN_PER_RUN logic below) for the full sweep.
#
# Submit:    sbatch slurm/train_stripped.sbatch
# Custom tag: RUN_TAG=stripped_diag sbatch slurm/train_stripped.sbatch
# ============================================================================

#SBATCH --job-name=mfl_strip
#SBATCH --array=1-5
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-02:00:00
#SBATCH --output=slurm/logs/strip_%A_%a.out
#SBATCH --error=slurm/logs/strip_%A_%a.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# ---- which scenario to ablate (default: s2_50_50, the canonical mid-point) ----
SCENARIO="${SCENARIO:-s2_50_50}"

# ---- 5-seed mapping (array task -> seed) ----
SEEDS=(_ 42 43 44 45 46)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
RUN_SUBDIR="${SCENARIO}/seed_${SEED}"

# ---- paths ----
RUN_TAG="${RUN_TAG:-stripped_${SLURM_ARRAY_JOB_ID}}"
OUTPUT_DIR="runs/${RUN_TAG}"
CSV_PATH="${CSV_PATH:-/data/amciilab/xinran/indiana_cxr/prepared/prepared_data_stripped.csv}"

REPO_DIR="${REPO_DIR:-$HOME/MFL-modality-quality-imbalance}"
CONDA_ENV="${CONDA_ENV:-/scratch/$USER/.conda/envs/multimodal-fl}"

cd "$REPO_DIR"
mkdir -p slurm/logs "$OUTPUT_DIR/$RUN_SUBDIR"

# ---- conda activation (same Sol pattern as train.sbatch) ----
CONDA_BASE="$(conda info --base 2>/dev/null || true)"
if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
elif [ -f "/packages/apps/mamba/2.0.8/etc/profile.d/conda.sh" ]; then
    source /packages/apps/mamba/2.0.8/etc/profile.d/conda.sh
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi
conda activate "$CONDA_ENV"
export PYTHONNOUSERSITE=1

# ---- diagnostics ----
echo "------------------------------------------------------------"
echo "Host:        $(hostname)"
echo "Job ID:      $SLURM_JOB_ID  (array $SLURM_ARRAY_JOB_ID task $SLURM_ARRAY_TASK_ID)"
echo "Scenario:    $SCENARIO"
echo "Seed:        $SEED"
echo "CSV:         $CSV_PATH         (lexical-strip ablation)"
echo "Run tag:     $RUN_TAG"
echo "Output:      $OUTPUT_DIR/$RUN_SUBDIR"
echo "Python:      $(which python)"
echo "GPU:"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
echo "------------------------------------------------------------"

# ---- launch ----
srun python -m train.cli \
    --csv-path "$CSV_PATH" \
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

echo "Task $SLURM_ARRAY_TASK_ID ($SCENARIO seed=$SEED, stripped) finished at $(date)"
