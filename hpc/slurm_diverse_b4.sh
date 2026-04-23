#!/bin/bash
#SBATCH --job-name=adaguard-diverse-b4
#SBATCH --output=/scratch/projects/secure-distributed-ml/logs/diverse-%j.out
#SBATCH --error=/scratch/projects/secure-distributed-ml/logs/diverse-%j.err
#SBATCH --partition=general-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=1:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maguir@oakland.edu
# =============================================================
# AdaGuard B=4 label-diversity experiment.
#
# Client 106's first 4 labels are [6,1,1,1] -- only 3 unique in
# 16 samples total. B=4 collapsed to 8.79 dB (GradInv) / 9.11 dB
# (GI-NAS) because duplicate labels make per-sample fc gradients
# parallel and the inversion underdetermined.
#
# This job:
#   1. Lists every client's label diversity for round 249 so we
#      know the distribution.
#   2. Auto-picks a client with >= 4 unique labels.
#   3. Runs GradInv @ B=4 and GI-NAS @ B=4 on a label-diverse
#      4-sample subset.
#
# Success criterion: GradInv B=4 >= 12 dB, GI-NAS B=4 >= 15 dB.
# If these land, the previous B=4 collapse was label-collinearity,
# not a fundamental attack failure -- the paper can frame the
# non-IID collinearity as a "free" privacy amplifier.
# If they still collapse, batch averaging alone is the killer and
# CIFAR/32x32/B=4 is just information-limited regardless.
#
# Usage:
#   sbatch hpc/slurm_diverse_b4.sh
# =============================================================

echo "Diverse-B4 job $SLURM_JOB_ID on $(hostname) at $(date)"

module load Python/3.10.14
module load CUDA/12.4
source /projects/secure-distributed-ml/venv/bin/activate

ARTIFACT_DIR="${ARTIFACT_DIR:-/scratch/projects/secure-distributed-ml/results/1k_experiment/artifacts_none_seed42_300clients}"
ROUND="${ROUND:-249}"
OUTDIR="/scratch/projects/secure-distributed-ml/results/diverse_${SLURM_JOB_ID}"
mkdir -p "$OUTDIR"

cd /projects/secure-distributed-ml/AdaGuard

echo "Artifact dir : $ARTIFACT_DIR"
echo "Round        : $ROUND"
echo ""

echo "============================================================"
echo "  Step 1: Listing per-client label diversity for round $ROUND"
echo "============================================================"
stdbuf -oL -eL python -u tests/attack_sanity_check.py \
    --artifact-dir "$ARTIFACT_DIR" \
    --round "$ROUND" \
    --attack gradinversion \
    --list-clients | tee "$OUTDIR/client_diversity.txt" | head -40

echo ""
echo "============================================================"
echo "  Step 2: GradInv @ B=4 on auto-picked diverse client"
echo "============================================================"
stdbuf -oL -eL python -u tests/attack_sanity_check.py \
    --artifact-dir "$ARTIFACT_DIR" \
    --round "$ROUND" \
    --auto-pick-diverse 4 \
    --diverse-subset \
    --attack gradinversion \
    --n-iter 20000 \
    --batch-size 4 \
    --variant paper \
    --out "$OUTDIR/gradinversion_b4_diverse.json"

echo ""
echo "============================================================"
echo "  Step 3: GI-NAS @ B=4 on auto-picked diverse client"
echo "============================================================"
stdbuf -oL -eL python -u tests/attack_sanity_check.py \
    --artifact-dir "$ARTIFACT_DIR" \
    --round "$ROUND" \
    --auto-pick-diverse 4 \
    --diverse-subset \
    --attack gi_nas \
    --n-iter 2000 \
    --batch-size 4 \
    --variant paper \
    --out "$OUTDIR/gi_nas_b4_diverse.json"

echo ""
echo "Diverse-B4 finished at $(date)"
echo "Artifacts in $OUTDIR"
