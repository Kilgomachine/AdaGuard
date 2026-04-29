#!/bin/bash
#SBATCH --job-name=adaguard-fisher-random-v13
#SBATCH --output=/scratch/projects/secure-distributed-ml/logs/v13-%j.out
#SBATCH --error=/scratch/projects/secure-distributed-ml/logs/v13-%j.err
#SBATCH --partition=general-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=2:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maguir@oakland.edu
# =============================================================
# AdaGuard V13 — Fisher-vs-random ablation at fixed rho=10%.
#
# Runs 3 attacks (GradInv/GGCDM/GI-NAS) at B=1 on round 249 with:
#   - --defence fisher
#   - --fisher-mask-mode random
#   - --fisher-random-seed 42 (reproducibility)
#   - --defence-pct 0.10 (matched to V4 / V6 in the registry)
#
# The classifier-head guarantee fires identically in random mode
# (it is orthogonal to the ranking principle), so this isolates
# the contribution of vulnerability-ranked targeting from the
# encryption budget itself.
#
# This is the Fisher-vs-random ablation that all four reviewers
# (Claude, Grok, ChatGPT, Deepseek) flagged as the single
# highest-value experimental extension. See CHANGELOG_REVIEW.md
# entry "NEW ABLATION: Fisher-vs-random".
#
# Run this 3 times — once per seed — by overriding ARTIFACT_DIR:
#   SEED=42  ARTIFACT_DIR=/scratch/.../artifacts_none_seed42_300clients  sbatch hpc/slurm_fisher_random_v13.sh
#   SEED=123 ARTIFACT_DIR=/scratch/.../artifacts_none_seed123_300clients sbatch hpc/slurm_fisher_random_v13.sh
#   SEED=456 ARTIFACT_DIR=/scratch/.../artifacts_none_seed456_300clients sbatch hpc/slurm_fisher_random_v13.sh
#
# Compare against V4 (V6 in registry numbering) — Fisher-targeted
# AdaGuard at the same rho=10% — already in
# data/paper_data/defence/seed{42,123,456}/fisher_*.json.
#
# Expected runtime per seed:
#   - GradInv 20k iters at B=1 ~= 5 min on V100
#   - GGCDM 100 iters at B=1 ~= 2 min
#   - GI-NAS 2k iters at B=1 ~= 1 min
#   - Total ~= 8 min per seed; 24 min wall-clock for all 3.
#   Budget 2h for safety margin.
# =============================================================

echo "V13 Fisher-random ablation job $SLURM_JOB_ID on $(hostname) at $(date)"

module load Python/3.10.14
module load CUDA/12.4
source /projects/secure-distributed-ml/venv/bin/activate

SEED="${SEED:-42}"
ARTIFACT_DIR="${ARTIFACT_DIR:-/scratch/projects/secure-distributed-ml/results/1k_experiment/artifacts_none_seed${SEED}_300clients}"
ROUND="${ROUND:-249}"
DEFENCE_PCT="${DEFENCE_PCT:-0.1}"
RANDOM_SEED="${RANDOM_SEED:-42}"  # for the within-replay random mask
OUTDIR="/scratch/projects/secure-distributed-ml/results/v13_fisher_random_seed${SEED}_${SLURM_JOB_ID}"
mkdir -p "$OUTDIR"

cd /projects/secure-distributed-ml/AdaGuard

echo "Artifact dir   : $ARTIFACT_DIR"
echo "Round          : $ROUND"
echo "Phase-1 seed   : $SEED"
echo "Defence pct    : $DEFENCE_PCT"
echo "Random seed    : $RANDOM_SEED (within-replay random-mask draw)"
echo "Output dir     : $OUTDIR"
echo ""

run_one () {
    local attack="$1"; local iters="$2"
    local tag="fisher_random_${attack}_b1_seed${SEED}"
    echo "============================================================"
    echo "  ATTACK=$attack  DEFENCE=fisher  MASK=random  SEED=$SEED  B=1  iters=$iters"
    echo "============================================================"
    stdbuf -oL -eL python -u tests/attack_sanity_check.py \
        --artifact-dir "$ARTIFACT_DIR" \
        --round "$ROUND" \
        --attack "$attack" \
        --n-iter "$iters" \
        --batch-size 1 \
        --defence fisher \
        --fisher-mask-mode random \
        --fisher-random-seed "$RANDOM_SEED" \
        --defence-pct "$DEFENCE_PCT" \
        --variant paper \
        --out "$OUTDIR/${tag}.json"
    echo ""
}

run_one gradinversion 20000
run_one ggcdm         100
run_one gi_nas        2000

echo "V13 Fisher-random ablation finished at $(date)"
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
mode = (d.get('defence_meta') or {}).get('mask_mode', '?')
print(f\"{os.path.basename(r'$j'):44s}  mask={mode:6s}  PSNR={psnr:7.2f} dB  LPIPS={lpips:.4f}  labelASR={asr:.3f}\")
"
done
