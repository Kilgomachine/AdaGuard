#!/bin/bash
#SBATCH --job-name=adaguard-ss-v5
#SBATCH --output=/scratch/projects/secure-distributed-ml/logs/v5-ss-%j.out
#SBATCH --error=/scratch/projects/secure-distributed-ml/logs/v5-ss-%j.err
#SBATCH --partition=general-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=2:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maguir@oakland.edu
# =============================================================
# AdaGuard V5 SelectiveShield in-family comparison.
#
# Runs 3 attacks (GradInversion, GGCDM, GI-NAS) at B=1 on round 249
# with --defence selectiveshield (Fisher-targeted SHE on top-K +
# Gaussian DP noise on the non-encrypted slice). Same encryption
# budget (rho=0.10) as V4 (Fisher); same classifier-head guarantee
# (head substring whitelist). Only mechanistic difference vs V4 is
# the DP noise on the non-encrypted slice.
#
# This is the in-family Fisher-vs-Fisher+DP comparison the professor
# flagged in pre-submission review. Pairs with the V4 (Fisher) cells
# already in data/paper_data/defence/seed{42,123,456}/fisher_*.json
# for the SelectiveShield-vs-AdaGuard delta.
#
# Run this 3 times -- once per seed -- by overriding ARTIFACT_DIR:
#   SEED=42  ARTIFACT_DIR=/scratch/.../artifacts_none_seed42_300clients  sbatch hpc/slurm_selectiveshield_v5.sh
#   SEED=123 ARTIFACT_DIR=/scratch/.../artifacts_none_seed123_300clients sbatch hpc/slurm_selectiveshield_v5.sh
#   SEED=456 ARTIFACT_DIR=/scratch/.../artifacts_none_seed456_300clients sbatch hpc/slurm_selectiveshield_v5.sh
#
# Expected runtime per seed: ~8 min on a Tesla V100 (5 min GradInv +
# 2 min GGCDM + 1 min GI-NAS). 2h walltime budgeted for safety.
# =============================================================

echo "V5 SelectiveShield job $SLURM_JOB_ID on $(hostname) at $(date)"

module load Python/3.10.14
module load CUDA/12.4
source /projects/secure-distributed-ml/venv/bin/activate

SEED="${SEED:-42}"
ARTIFACT_DIR="${ARTIFACT_DIR:-/scratch/projects/secure-distributed-ml/results/1k_experiment/artifacts_none_seed${SEED}_300clients}"
ROUND="${ROUND:-249}"
DEFENCE_PCT="${DEFENCE_PCT:-0.1}"
SS_DP_EPSILON="${SS_DP_EPSILON:-50.0}"
SS_DP_DELTA="${SS_DP_DELTA:-1e-5}"
SS_DP_CLIP_NORM="${SS_DP_CLIP_NORM:-1.0}"
OUTDIR="/scratch/projects/secure-distributed-ml/results/v5_selectiveshield_seed${SEED}_${SLURM_JOB_ID}"
mkdir -p "$OUTDIR"

cd /projects/secure-distributed-ml/AdaGuard

echo "Artifact dir   : $ARTIFACT_DIR"
echo "Round          : $ROUND"
echo "Phase-1 seed   : $SEED"
echo "Defence pct    : $DEFENCE_PCT"
echo "DP epsilon     : $SS_DP_EPSILON"
echo "DP delta       : $SS_DP_DELTA"
echo "DP clip norm   : $SS_DP_CLIP_NORM"
echo "Output dir     : $OUTDIR"
echo ""

run_one () {
    local attack="$1"; local iters="$2"
    local tag="selectiveshield_${attack}_b1_seed${SEED}"
    echo "============================================================"
    echo "  ATTACK=$attack  DEFENCE=selectiveshield  SEED=$SEED  B=1  iters=$iters"
    echo "============================================================"
    stdbuf -oL -eL python -u tests/attack_sanity_check.py \
        --artifact-dir "$ARTIFACT_DIR" \
        --round "$ROUND" \
        --attack "$attack" \
        --n-iter "$iters" \
        --batch-size 1 \
        --defence selectiveshield \
        --defence-pct "$DEFENCE_PCT" \
        --ss-dp-epsilon "$SS_DP_EPSILON" \
        --ss-dp-delta "$SS_DP_DELTA" \
        --ss-dp-clip-norm "$SS_DP_CLIP_NORM" \
        --variant paper \
        --out "$OUTDIR/${tag}.json"
    echo ""
}

run_one gradinversion 20000
run_one ggcdm         100
run_one gi_nas        2000

echo "V5 SelectiveShield finished at $(date)"
echo "Artifacts in $OUTDIR"
echo ""
echo "Summary table:"
for j in "$OUTDIR"/*.json; do
    python -c "
import json, os
d = json.load(open(r'$j'))
psnr = d.get('psnr', float('nan'))
lpips = d.get('lpips') or 0.0
asr = (d.get('label_recovery') or {}).get('asr', 0.0)
dp = d.get('defence_meta') or {}
sigma = dp.get('dp_sigma', 0.0)
print(f\"{os.path.basename(r'$j'):46s}  PSNR={psnr:7.2f} dB  LPIPS={lpips:.4f}  labelASR={asr:.3f}  dp_sigma={sigma:.4f}\")
"
done
