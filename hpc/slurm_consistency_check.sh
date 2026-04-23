#!/bin/bash
#SBATCH --job-name=adaguard-consistency
#SBATCH --output=/scratch/projects/secure-distributed-ml/logs/consistency-%j.out
#SBATCH --error=/scratch/projects/secure-distributed-ml/logs/consistency-%j.err
#SBATCH --partition=general-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maguir@oakland.edu
# =============================================================
# AdaGuard Phase-1 artifact self-consistency check.
#
# Reconstructs the local model via (global - weight_delta),
# recomputes the gradient at the full saved batch, and compares
# layer-by-layer against the saved gradient_dict. If they match
# (max_rel_diff <= 1e-3), the artifact is internally consistent
# and the B != saved_B subsample attacks are attacking the real
# local gradient — i.e., the B=4 collapse is research, not bug.
#
# If the diff is large, Phase-1 saved something inconsistent
# with the triple (global, weight_delta, images, labels) and the
# batch-size override attacks are attacking a phantom. At that
# point we debug the simulator, not the attacks.
#
# Usage:
#   sbatch hpc/slurm_consistency_check.sh
#   ROUND=50 sbatch hpc/slurm_consistency_check.sh
# =============================================================

echo "Consistency job $SLURM_JOB_ID on $(hostname) at $(date)"

module load Python/3.10.14
module load CUDA/12.4
source /projects/secure-distributed-ml/venv/bin/activate

ARTIFACT_DIR="${ARTIFACT_DIR:-/scratch/projects/secure-distributed-ml/results/1k_experiment/artifacts_none_seed42_300clients}"
ROUND="${ROUND:-249}"
OUTDIR="/scratch/projects/secure-distributed-ml/results/consistency_${SLURM_JOB_ID}"
mkdir -p "$OUTDIR"

cd /projects/secure-distributed-ml/AdaGuard

echo "Artifact dir : $ARTIFACT_DIR"
echo "Round        : $ROUND"
echo ""

stdbuf -oL -eL python -u tests/attack_sanity_check.py \
    --artifact-dir "$ARTIFACT_DIR" \
    --round "$ROUND" \
    --attack gradinversion \
    --consistency-only \
    --out "$OUTDIR/consistency.json"

echo ""
echo "Consistency check finished at $(date)"
echo "Report in $OUTDIR/consistency.json"
