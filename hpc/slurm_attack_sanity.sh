#!/bin/bash
#SBATCH --job-name=adaguard-sanity
#SBATCH --output=/scratch/projects/secure-distributed-ml/logs/sanity-%j.out
#SBATCH --error=/scratch/projects/secure-distributed-ml/logs/sanity-%j.err
#SBATCH --partition=general-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=8:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maguir@oakland.edu
# =============================================================
# AdaGuard attack-sanity check at paper-matched batch sizes.
#
# Restructure 1 and our early Restructure 2 runs missed that the three
# target papers evaluate at small batch sizes (GI-NAS: B=4 on CIFAR-10;
# GradInversion: B=8-48 on ImageNet; GGCDM: B=1 on CelebA-256). Running
# at B=16 on CIFAR gives a ~9 dB noise-floor result for every defence
# because the inversion is underdetermined at that batch size, not
# because the attacks are broken. This script sweeps batch sizes to
# probe whether the attacks actually land their paper numbers at
# paper-matched protocols on our undefended (V1) artifacts.
#
# Default sweep:
#   GradInversion @ B=1  (iDLG-style single sample, Geiping recipe)
#   GradInversion @ B=4  (paper-matched small batch)
#   GradInversion @ B=16 (stress test, historical comparison)
#   GI-NAS        @ B=4  (Yu 2025 Table III: 35.99 dB target)
#   GI-NAS        @ B=1  (pure DIP baseline)
#
# GGCDM is omitted: it's a diffusion attack tuned for 256x256 faces,
# running it at B=1 on 32x32 CIFAR is interesting but noisy; we keep
# the CIFAR number we already have and can enable via ATTACK=ggcdm.
#
# Usage:
#   sbatch hpc/slurm_attack_sanity.sh
#   SMOKE=1 sbatch hpc/slurm_attack_sanity.sh        # fast, 500 iters
#   ATTACK=gi_nas BATCH=4 sbatch hpc/slurm_attack_sanity.sh
#   ARTIFACT_DIR=/scratch/.../artifacts_fisher_... sbatch ...
# =============================================================

echo "Sanity job $SLURM_JOB_ID on $(hostname) at $(date)"

module load Python/3.10.14
module load CUDA/12.4
source /projects/secure-distributed-ml/venv/bin/activate

ARTIFACT_DIR="${ARTIFACT_DIR:-/scratch/projects/secure-distributed-ml/results/1k_experiment/artifacts_none_seed42_300clients}"
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

echo "Raw artifact dir : $ARTIFACT_DIR"
echo "Round            : $ROUND"
echo ""

run_one () {
    local name="$1"; local iters="$2"; local bs="$3"
    local tag="${name}_b${bs}"
    echo "============================================================"
    echo "  ATTACK=$name  B=$bs  iters=$iters  round=$ROUND"
    echo "============================================================"
    stdbuf -oL -eL python -u tests/attack_sanity_check.py \
        --artifact-dir "$ARTIFACT_DIR" \
        --round "$ROUND" \
        --attack "$name" \
        --n-iter "$iters" \
        --batch-size "$bs" \
        --variant paper \
        --out "$OUTDIR/${tag}.json"
}

# Single-attack override: ATTACK=gi_nas BATCH=4 sbatch ...
if [[ -n "$ATTACK" ]]; then
    BS="${BATCH:-4}"
    case "$ATTACK" in
        gradinversion) run_one gradinversion "$GI_ITERS"    "$BS" ;;
        gi_nas)        run_one gi_nas        "$GINAS_ITERS" "$BS" ;;
        ggcdm)         run_one ggcdm         "$GGCDM_ITERS" "$BS" ;;
        *) echo "Unknown ATTACK=$ATTACK"; exit 1 ;;
    esac
else
    # Full paper-matched sweep on V1 undefended.
    run_one gradinversion "$GI_ITERS"    1
    run_one gradinversion "$GI_ITERS"    4
    run_one gradinversion "$GI_ITERS"    16
    run_one gi_nas        "$GINAS_ITERS" 1
    run_one gi_nas        "$GINAS_ITERS" 4
fi

echo "Sanity finished at $(date)"
echo "Artifacts in $OUTDIR"
