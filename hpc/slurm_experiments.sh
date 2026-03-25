#!/bin/bash
#SBATCH --job-name=adaguard-exp
#SBATCH --output=/scratch/projects/secure-distributed-ml/logs/exp-%A_%a.out
#SBATCH --error=/scratch/projects/secure-distributed-ml/logs/exp-%A_%a.err
#SBATCH --partition=general-long
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-11
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maguir@oakland.edu
# =============================================================
# AdaGuard — Multi-Seed Strategy Comparison (FAST)
#
# Uses 4 GPUs per job for client parallelism.
# Skips empirical attacks and GLMIP (run separately if needed).
#
# Array layout (12 jobs):
#   0-3:  seed=42   | none, fisher, maskcrypt, full
#   4-7:  seed=123  | none, fisher, maskcrypt, full
#   8-11: seed=456  | none, fisher, maskcrypt, full
#
# Submit: sbatch hpc/slurm_experiments.sh
# =============================================================

echo "Task $SLURM_ARRAY_TASK_ID of job $SLURM_ARRAY_JOB_ID on $(hostname) at $(date)"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ', ')"

module load Python/3.10.14
module load CUDA/12.4
source /projects/secure-distributed-ml/venv/bin/activate

export DATA_DIR="/scratch/projects/secure-distributed-ml/data"
export PYTHONUNBUFFERED=1
cd /projects/secure-distributed-ml/AdaGuard

# Map array index to seed and strategy
SEEDS=(42 42 42 42 123 123 123 123 456 456 456 456)
STRATEGIES=(none fisher maskcrypt full none fisher maskcrypt full none fisher maskcrypt full)

SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
STRATEGY="${STRATEGIES[$SLURM_ARRAY_TASK_ID]}"

RESULTS_DIR="/scratch/projects/secure-distributed-ml/results/experiment_${SLURM_ARRAY_JOB_ID}"
mkdir -p "$RESULTS_DIR"

OUTPUT="${RESULTS_DIR}/${STRATEGY}_seed${SEED}.json"

echo "Seed: $SEED | Strategy: $STRATEGY | Output: $OUTPUT"

stdbuf -oL -eL python -u run_headless.py \
    --config hpc/config_large.yaml \
    --strategy "$STRATEGY" \
    --seed "$SEED" \
    --output "$OUTPUT"

echo "Task $SLURM_ARRAY_TASK_ID ($STRATEGY, seed=$SEED) completed at $(date)"
