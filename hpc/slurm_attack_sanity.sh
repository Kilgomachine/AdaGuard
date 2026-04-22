#!/bin/bash
#SBATCH --job-name=adaguard-sanity
#SBATCH --output=/scratch/projects/secure-distributed-ml/logs/sanity-%j.out
#SBATCH --error=/scratch/projects/secure-distributed-ml/logs/sanity-%j.err
#SBATCH --partition=general-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=4:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maguir@oakland.edu
# =============================================================
# AdaGuard attack-sanity check (V1 undefended).
#
# Runs each of the three paper-faithful attacks against a single
# Phase-1 client artifact from the 300-client seed-42 run and prints
# PSNR / SSIM / LPIPS vs. the paper-reported targets. Purpose: confirm
# the re-implementations actually land before spending another
# Phase-2 sweep on them.
#
# Usage:
#   # Default: run all three attacks at full paper budget.
#   sbatch hpc/slurm_attack_sanity.sh
#
#   # Run a specific attack only:
#   ATTACK=gradinversion sbatch hpc/slurm_attack_sanity.sh
#   ATTACK=gi_nas        sbatch hpc/slurm_attack_sanity.sh
#   ATTACK=ggcdm         sbatch hpc/slurm_attack_sanity.sh
#
#   # Smoke-test at reduced budget (seconds instead of hours):
#   SMOKE=1 sbatch hpc/slurm_attack_sanity.sh
# =============================================================

echo "Sanity job $SLURM_JOB_ID on $(hostname) at $(date)"

module load Python/3.10.14
module load CUDA/12.4
source /projects/secure-distributed-ml/venv/bin/activate

ARTIFACT_DIR="${ARTIFACT_DIR:-/scratch/projects/secure-distributed-ml/results/1k_experiment/artifacts_fisher_seed42_300clients}"
ROUND="${ROUND:-249}"
OUTDIR="/scratch/projects/secure-distributed-ml/results/sanity_${SLURM_JOB_ID}"
mkdir -p "$OUTDIR"

if [[ -n "$SMOKE" ]]; then
    GI_ITERS=500
    GINAS_ITERS=200
    GGCDM_ITERS=25
else
    GI_ITERS=20000
    GINAS_ITERS=2000
    GGCDM_ITERS=100
fi

cd /projects/secure-distributed-ml/AdaGuard

run_one () {
    local name="$1"; local iters="$2"
    echo "============================================================"
    echo "  ATTACK=$name iters=$iters round=$ROUND"
    echo "============================================================"
    stdbuf -oL -eL python -u tests/attack_sanity_check.py \
        --artifact-dir "$ARTIFACT_DIR" \
        --round "$ROUND" \
        --attack "$name" \
        --n-iter "$iters" \
        --variant paper \
        --out "$OUTDIR/${name}.json"
}

if [[ -n "$ATTACK" ]]; then
    case "$ATTACK" in
        gradinversion) run_one gradinversion "$GI_ITERS" ;;
        gi_nas)        run_one gi_nas        "$GINAS_ITERS" ;;
        ggcdm)         run_one ggcdm         "$GGCDM_ITERS" ;;
        *) echo "Unknown ATTACK=$ATTACK"; exit 1 ;;
    esac
else
    run_one gradinversion "$GI_ITERS"
    run_one gi_nas        "$GINAS_ITERS"
    run_one ggcdm         "$GGCDM_ITERS"
fi

echo "Sanity finished at $(date)"
echo "Artifacts in $OUTDIR"
