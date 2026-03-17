#!/bin/bash
#SBATCH --job-name=adaguard-1k
#SBATCH --output=/scratch/projects/secure-distributed-ml/logs/1k-%j.out
#SBATCH --error=/scratch/projects/secure-distributed-ml/logs/1k-%j.err
#SBATCH --partition=general-long
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maguir@oakland.edu
# =============================================================
# AdaGuard — 1000-Client Large-Scale Run
# Submit: sbatch hpc/slurm_1k_clients.sh
# =============================================================

echo "Job $SLURM_JOB_ID started on $(hostname) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

module load Python/3.10.14
module load CUDA/12.4
source /projects/secure-distributed-ml/venv/bin/activate

export DATA_DIR="/scratch/projects/secure-distributed-ml/data"
cd /projects/secure-distributed-ml/AdaGuard

STRATEGY="${STRATEGY:-fisher}"
OUTPUT="/scratch/projects/secure-distributed-ml/results/1k_${STRATEGY}_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID}.json"

echo "1000-client run, strategy: $STRATEGY"
echo "Output: $OUTPUT"

export PYTHONUNBUFFERED=1

python -u run_headless.py \
    --config hpc/config_1k.yaml \
    --strategy "$STRATEGY" \
    --output "$OUTPUT"

echo "Job completed at $(date)"
