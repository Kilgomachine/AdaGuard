#!/bin/bash
#SBATCH --job-name=adaguard-compare
#SBATCH --output=/scratch/projects/secure-distributed-ml/logs/compare-%j.out
#SBATCH --error=/scratch/projects/secure-distributed-ml/logs/compare-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maguir@oakland.edu
# =============================================================
# AdaGuard — Strategy Comparison (runs all 4 strategies)
# Submit: sbatch hpc/slurm_comparison.sh
# =============================================================

echo "Job $SLURM_JOB_ID started on $(hostname) at $(date)"

module load Python/3.10.14
module load CUDA/12.4
source /projects/secure-distributed-ml/venv/bin/activate

export DATA_DIR="/scratch/projects/secure-distributed-ml/data"
cd /projects/secure-distributed-ml/AdaGuard

CONFIG="${CONFIG:-hpc/config_large.yaml}"
OUTPUT="/scratch/projects/secure-distributed-ml/results/comparison_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID}.json"

python run_headless.py \
    --config "$CONFIG" \
    --experiment comparison \
    --output "$OUTPUT"

echo "Job completed at $(date)"
