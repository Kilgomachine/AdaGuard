#!/bin/bash
#SBATCH --job-name=adaguard-breaching-validate
#SBATCH --output=/scratch/projects/secure-distributed-ml/logs/breaching-validate-%j.out
#SBATCH --error=/scratch/projects/secure-distributed-ml/logs/breaching-validate-%j.err
#SBATCH --partition=general-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=1:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maguir@oakland.edu
# =============================================================
# Validate the breaching-backed Yin 2021 GradInversion on one
# undefended B=1 cell before rerunning the full sweep.
#
# Target: PSNR within +/- 2 dB of our Geiping-recipe baseline
# (18.32 dB on V1 round 249 client 106 B=1). Breaching uses
# Yin's BN-matching (DeepInversion hooks), which at B=1 should
# match or slightly exceed Geiping-only.
#
# Also does a sanity run at B=4 to see whether BN matching
# rescues any of the B=4 collapse (Geiping recipe gave 8.79 dB
# at B=4; if Yin gives >= 12 dB the "non-IID amplifier" framing
# softens and we get a stronger attack story for the paper).
#
# Usage:
#   sbatch hpc/slurm_breaching_validate.sh
# =============================================================

echo "Breaching-validate job $SLURM_JOB_ID on $(hostname) at $(date)"

module load Python/3.10.14
module load CUDA/12.4
source /projects/secure-distributed-ml/venv/bin/activate

# Install breaching if not already present. pip-check then install.
python -c "import breaching" 2>/dev/null || {
    echo "[setup] breaching not found, installing..."
    pip install --no-cache-dir "breaching @ git+https://github.com/JonasGeiping/breaching.git"
    python -c "import breaching; print(f'breaching {breaching.__file__}')"
}
python -c "import breaching, hydra, omegaconf; print('breaching OK')"

ARTIFACT_DIR="${ARTIFACT_DIR:-/scratch/projects/secure-distributed-ml/results/1k_experiment/artifacts_none_seed42_300clients}"
ROUND="${ROUND:-249}"
OUTDIR="/scratch/projects/secure-distributed-ml/results/breaching_validate_${SLURM_JOB_ID}"
mkdir -p "$OUTDIR"

cd /projects/secure-distributed-ml/AdaGuard

echo "Artifact dir : $ARTIFACT_DIR"
echo "Round        : $ROUND"
echo ""

echo "============================================================"
echo "  Step 1: breaching Yin on V1 undefended, B=1 (validation)"
echo "  Expect PSNR ~ 18 dB (baseline Geiping recipe: 18.32 dB)"
echo "============================================================"
stdbuf -oL -eL python -u tests/attack_sanity_check.py \
    --artifact-dir "$ARTIFACT_DIR" \
    --round "$ROUND" \
    --attack gradinversion_breaching \
    --n-iter 20000 \
    --batch-size 1 \
    --defence none \
    --variant paper \
    --out "$OUTDIR/breaching_b1_undef.json"

echo ""
echo "============================================================"
echo "  Step 2: breaching Yin on V1 undefended, B=4 (BN-match test)"
echo "  Geiping-recipe baseline: 8.79 dB at B=4"
echo "  If Yin BN-match gives >= 12 dB, rewrite the non-IID"
echo "  amplifier subsection to say BN matching partially rescues."
echo "============================================================"
stdbuf -oL -eL python -u tests/attack_sanity_check.py \
    --artifact-dir "$ARTIFACT_DIR" \
    --round "$ROUND" \
    --attack gradinversion_breaching \
    --n-iter 20000 \
    --batch-size 4 \
    --defence none \
    --variant paper \
    --out "$OUTDIR/breaching_b4_undef.json"

echo ""
echo "Breaching-validate finished at $(date)"
echo "Artifacts in $OUTDIR"
echo ""
echo "Summary:"
for j in "$OUTDIR"/*.json; do
    python -c "
import json, os
d = json.load(open(r'$j'))
print(f\"{os.path.basename(r'$j'):40s}  PSNR={d.get('psnr', float('nan')):7.2f} dB  LPIPS={d.get('lpips', 0):.4f}  cos={d.get('gradient_score', 0):.3f}\")
"
done
