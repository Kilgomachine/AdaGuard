#!/bin/bash
#SBATCH --job-name=adaguard-sweep
#SBATCH --output=/scratch/projects/secure-distributed-ml/logs/sweep-%A_%a.out
#SBATCH --error=/scratch/projects/secure-distributed-ml/logs/sweep-%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-3
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maguir@oakland.edu
# =============================================================
# AdaGuard — Strategy Sweep (parallel array job)
# Runs all 4 strategies simultaneously on separate GPUs.
# Submit: sbatch hpc/slurm_sweep.sh
# =============================================================

echo "Array task $SLURM_ARRAY_TASK_ID of job $SLURM_ARRAY_JOB_ID"

module load Python/3.10.14
module load CUDA/12.4
source /projects/secure-distributed-ml/venv/bin/activate

export DATA_DIR="/scratch/projects/secure-distributed-ml/data"
cd /projects/secure-distributed-ml/AdaGuard

# Map array index to strategy
STRATEGIES=(none fisher maskcrypt full)
STRATEGY="${STRATEGIES[$SLURM_ARRAY_TASK_ID]}"

CONFIG="${CONFIG:-hpc/config_large.yaml}"
OUTPUT="/scratch/projects/secure-distributed-ml/results/sweep_${STRATEGY}_$(date +%Y%m%d_%H%M%S)_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json"

echo "Strategy: $STRATEGY"
echo "Output: $OUTPUT"

python run_headless.py \
    --config "$CONFIG" \
    --strategy "$STRATEGY" \
    --output "$OUTPUT"

echo "Task $SLURM_ARRAY_TASK_ID ($STRATEGY) completed at $(date)"
