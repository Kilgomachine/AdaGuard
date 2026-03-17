#!/bin/bash
#SBATCH --job-name=adaguard-fl
#SBATCH --output=/scratch/projects/secure-distributed-ml/logs/adaguard-%j.out
#SBATCH --error=/scratch/projects/secure-distributed-ml/logs/adaguard-%j.err
#SBATCH --partition=general-short
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maguir@oakland.edu
# =============================================================
# AdaGuard — Single GPU Job (100 clients, 50 rounds)
# Submit: sbatch hpc/slurm_single_gpu.sh
# =============================================================

echo "Job $SLURM_JOB_ID started on $(hostname) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Load modules
module load Python/3.10.14
module load CUDA/12.4

# Activate venv
source /projects/secure-distributed-ml/venv/bin/activate

# Set data directory to scratch (faster I/O)
export DATA_DIR="/scratch/projects/secure-distributed-ml/data"

# Navigate to project
cd /projects/secure-distributed-ml/AdaGuard

# Run experiment
STRATEGY="${STRATEGY:-fisher}"
CONFIG="${CONFIG:-hpc/config_large.yaml}"
OUTPUT="/scratch/projects/secure-distributed-ml/results/fl_${STRATEGY}_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID}.json"

echo "Strategy: $STRATEGY"
echo "Config: $CONFIG"
echo "Output: $OUTPUT"

export PYTHONUNBUFFERED=1

python -u run_headless.py \
    --config "$CONFIG" \
    --strategy "$STRATEGY" \
    --output "$OUTPUT"

echo "Job completed at $(date)"
