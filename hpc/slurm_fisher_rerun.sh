#!/bin/bash
#SBATCH --job-name=adaguard-fisher-rerun
#SBATCH --output=/scratch/projects/secure-distributed-ml/logs/fisher-rerun-%j.out
#SBATCH --error=/scratch/projects/secure-distributed-ml/logs/fisher-rerun-%j.err
#SBATCH --partition=general-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=2:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maguir@oakland.edu
# =============================================================
# AdaGuard-Fisher classifier-head guarantee re-run.
#
# Background:
#   The Phase-4 multi-seed sweep (jobs 181367/181577/181578)
#   exposed a failure mode in pure top-K-by-Fisher selection:
#   on seed 456 round 249, the 10-element fc.bias fell below
#   the global top-K threshold and the LLG+ label-decoding
#   primitive recovered the label with ASR=1.0 across all
#   three attack families (GradInversion, GGCDM, GI-NAS).
#
# Fix (committed to fisher.py module docstring + tests):
#   Force-include classifier-head parameters in the encrypted
#   set regardless of Fisher score. Default behaviour now
#   includes fc.bias / fc.weight / classifier.{bias,weight}
#   / head.{bias,weight}. Cost: ~5,140 params on ResNet-18,
#   +0.046% of total. Reproduce broken behaviour via
#   --no-classifier-head-guarantee.
#
# This job re-runs ONLY the Fisher cells (3 attacks) for the
# specified seed, with the post-fix selector. None / FHE /
# MaskCrypt rows are unchanged across the fix and need no
# re-run.
#
# Expected runtime per seed:
#   - GradInv 20k iters at B=1 ~= 5 min
#   - GGCDM   100  iters at B=1 ~= 2 min
#   - GI-NAS  2k   iters at B=1 ~= 1 min
#   = ~8-10 min total. 2h budget for safety margin.
#
# Usage (run once per seed; OUTDIR auto-namespaced by job id):
#   ARTIFACT_DIR=/scratch/.../artifacts_none_seed42_300clients  \
#       sbatch hpc/slurm_fisher_rerun.sh
#   ARTIFACT_DIR=/scratch/.../artifacts_none_seed123_300clients \
#       sbatch hpc/slurm_fisher_rerun.sh
#   ARTIFACT_DIR=/scratch/.../artifacts_none_seed456_300clients \
#       sbatch hpc/slurm_fisher_rerun.sh
# =============================================================

echo "Fisher-rerun job $SLURM_JOB_ID on $(hostname) at $(date)"

module load Python/3.10.14
module load CUDA/12.4
source /projects/secure-distributed-ml/venv/bin/activate

ARTIFACT_DIR="${ARTIFACT_DIR:-/scratch/projects/secure-distributed-ml/results/1k_experiment/artifacts_none_seed42_300clients}"
ROUND="${ROUND:-249}"
DEFENCE_PCT="${DEFENCE_PCT:-0.1}"
OUTDIR="/scratch/projects/secure-distributed-ml/results/fisher_rerun_${SLURM_JOB_ID}"
mkdir -p "$OUTDIR"

# Run unit tests first so we fail fast if the patch regressed.
cd /projects/secure-distributed-ml/AdaGuard
echo ""
echo "=== Running classifier-head guarantee unit tests ==="
python tests/test_fisher_classifier_head_guarantee.py
TEST_RC=$?
if [ "$TEST_RC" -ne 0 ]; then
    echo "[FATAL] Unit tests failed (rc=$TEST_RC). Aborting sweep."
    exit "$TEST_RC"
fi
echo "All guarantee tests passed."
echo ""

echo "Artifact dir : $ARTIFACT_DIR"
echo "Round        : $ROUND"
echo "Defence pct  : $DEFENCE_PCT"
echo "Output dir   : $OUTDIR"
echo ""

run_fisher () {
    local attack="$1"; local iters="$2"
    local tag="fisher_${attack}_b1"
    echo "============================================================"
    echo "  ATTACK=$attack  DEFENCE=fisher (post-fix)  B=1  iters=$iters"
    echo "============================================================"
    stdbuf -oL -eL python -u tests/attack_sanity_check.py \
        --artifact-dir "$ARTIFACT_DIR" \
        --round "$ROUND" \
        --attack "$attack" \
        --n-iter "$iters" \
        --batch-size 1 \
        --defence fisher \
        --defence-pct "$DEFENCE_PCT" \
        --variant paper \
        --out "$OUTDIR/${tag}.json"
    echo ""
}

run_fisher gradinversion 20000
run_fisher ggcdm         100
run_fisher gi_nas        2000

echo "Fisher-rerun finished at $(date)"
echo "Artifacts in $OUTDIR"
echo ""
echo "Summary (compare ASR against pre-fix; expect 0.000 across the board):"
for j in "$OUTDIR"/*.json; do
    python -c "
import json, os
d = json.load(open(r'$j'))
dm = d.get('defence_meta', {})
print(f\"{os.path.basename(r'$j'):40s}  PSNR={d.get('psnr', float('nan')):7.2f} dB  LPIPS={d.get('lpips', 0):.4f}  labelASR={d.get('label_recovery',{}).get('asr',0):.3f}  forced={dm.get('classifier_head_forced_count', 0)} extra params  guarantee={dm.get('classifier_head_guarantee', '?')}\")
"
done
