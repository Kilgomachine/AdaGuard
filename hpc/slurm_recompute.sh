#!/bin/bash
#SBATCH --job-name=adaguard-recompute
#SBATCH --output=/scratch/projects/secure-distributed-ml/logs/recompute-%j.out
#SBATCH --error=/scratch/projects/secure-distributed-ml/logs/recompute-%j.err
#SBATCH --partition=general-long
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maguir@oakland.edu
# =============================================================
# AdaGuard Phase 2.5 — offline metric recomputation
#
# Recomputes PSNR/SSIM/LPIPS/MSE/etc. from the saved .pt files
# with consistent [0,1] normalization. Required because the live
# sweep saves originals in CIFAR-normalized form but recons in
# [0,1], producing garbage metrics in the per-scenario JSONs.
#
# Usage:
#   # Wait for the main sweep (array job 181097) to finish, then run:
#   sbatch --dependency=afterany:181097 hpc/slurm_recompute.sh
#
#   # Or run immediately against partial results:
#   sbatch hpc/slurm_recompute.sh
#
#   # Override the study job id:
#   STUDY_JOB=181097 sbatch hpc/slurm_recompute.sh
# =============================================================

echo "Recompute job $SLURM_JOB_ID on $(hostname) at $(date)"

module load Python/3.10.14
module load CUDA/12.4
source /projects/secure-distributed-ml/venv/bin/activate

STUDY_JOB="${STUDY_JOB:-181097}"
RESULTS_DIR="/scratch/projects/secure-distributed-ml/results/study_${STUDY_JOB}"
OUTPUT="${RESULTS_DIR}/corrected_metrics.json"

echo "Results dir: $RESULTS_DIR"
echo "Output:      $OUTPUT"

cd /projects/secure-distributed-ml/AdaGuard
python recompute_metrics.py \
    --results-dir "$RESULTS_DIR" \
    --output "$OUTPUT"

echo "Recompute finished at $(date)"
