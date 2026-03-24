#!/bin/bash
#SBATCH --job-name=adaguard-exp
#SBATCH --output=/scratch/projects/secure-distributed-ml/logs/exp-%A_%a.out
#SBATCH --error=/scratch/projects/secure-distributed-ml/logs/exp-%A_%a.err
#SBATCH --partition=general-short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-11
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maguir@oakland.edu
# =============================================================
# AdaGuard — Multi-Seed Strategy Comparison
#
# Array index layout (12 jobs total):
#   0-3:  seed=42   strategies: none, fisher, maskcrypt, full
#   4-7:  seed=123  strategies: none, fisher, maskcrypt, full
#   8-11: seed=456  strategies: none, fisher, maskcrypt, full
#
# Submit: sbatch hpc/slurm_experiments.sh
# =============================================================

echo "Array task $SLURM_ARRAY_TASK_ID of job $SLURM_ARRAY_JOB_ID on $(hostname)"
echo "Started at $(date)"

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

echo "Seed: $SEED"
echo "Strategy: $STRATEGY"
echo "Output: $OUTPUT"

python -u run_headless.py \
    --config hpc/config_large.yaml \
    --strategy "$STRATEGY" \
    --seed "$SEED" \
    --output "$OUTPUT"

echo "Task $SLURM_ARRAY_TASK_ID ($STRATEGY, seed=$SEED) completed at $(date)"
