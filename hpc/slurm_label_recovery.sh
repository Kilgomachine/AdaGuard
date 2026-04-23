#!/bin/bash
#SBATCH --job-name=adaguard-llg
#SBATCH --output=/scratch/projects/secure-distributed-ml/logs/llg-%j.out
#SBATCH --error=/scratch/projects/secure-distributed-ml/logs/llg-%j.err
#SBATCH --partition=general
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maguir@oakland.edu
# =============================================================
# AdaGuard label-recovery (iDLG / LLG+) sweep across V1/V2/V4/V6.
#
# Pixel PSNR saturates at the ~9 dB noise floor at B=16 Fisher-non-IID
# for every defence (Attack Restructure 1). The paper's ASR headline
# is almost certainly label-recovery accuracy — this script computes
# it analytically from each client's fc.* gradient, no optimisation
# loop. Whole 4-defence sweep completes in seconds on CPU.
#
# Usage:
#   sbatch hpc/slurm_label_recovery.sh
#   ROUND=50 sbatch hpc/slurm_label_recovery.sh   # earlier snapshot
# =============================================================

echo "LLG sweep job $SLURM_JOB_ID on $(hostname) at $(date)"

module load Python/3.10.14
source /projects/secure-distributed-ml/venv/bin/activate

ROUND="${ROUND:-249}"
RESULTS_ROOT="/scratch/projects/secure-distributed-ml/results/1k_experiment"
OUTDIR="/scratch/projects/secure-distributed-ml/results/llg_${SLURM_JOB_ID}"
mkdir -p "$OUTDIR"

cd /projects/secure-distributed-ml/AdaGuard

stdbuf -oL -eL python -u tests/label_recovery_sweep.py \
    --round "$ROUND" \
    --num-classes 10 \
    --dir "V1=$RESULTS_ROOT/artifacts_none_seed42_300clients" \
    --dir "V2=$RESULTS_ROOT/artifacts_full_seed42_300clients" \
    --dir "V4=$RESULTS_ROOT/artifacts_maskcrypt_seed42_300clients" \
    --dir "V6=$RESULTS_ROOT/artifacts_fisher_seed42_300clients" \
    --out "$OUTDIR/label_recovery_round${ROUND}.json"

echo "LLG sweep finished at $(date)"
echo "Artifacts in $OUTDIR"
