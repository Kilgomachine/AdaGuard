#!/bin/bash
#SBATCH --job-name=adaguard-defence-sweep-v2
#SBATCH --output=/scratch/projects/secure-distributed-ml/logs/defence-v2-%j.out
#SBATCH --error=/scratch/projects/secure-distributed-ml/logs/defence-v2-%j.err
#SBATCH --partition=general-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=3:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maguir@oakland.edu
# =============================================================
# AdaGuard defence sweep v2 — Yin 2021 GradInversion via
# breaching.seethroughgradients (Geiping's lab's reference
# implementation). Replaces job 181367's GradInv rows with
# the BN-matching Yin recipe.
#
# Run AFTER slurm_breaching_validate.sh has confirmed the
# breaching-backed undefended B=1 PSNR is within +/- 2 dB of
# our prior 18.32 dB baseline.
#
# GGCDM and GI-NAS rows are NOT re-run — their 181367 results
# stand (no trusted third-party implementations exist for
# either; we ship our paper-faithful versions with the DPS-
# proxy caveat for GGCDM in sec VI.E).
#
# Expected runtime:
#   - 4 defences x 1 attack (gradinversion_breaching) x 20k
#     iters at B=1. Yin's BN hooks add ~30% overhead vs.
#     Geiping, so ~7 min/run = ~28 min total. 3h budget.
#
# Usage:
#   sbatch hpc/slurm_defence_sweep_v2.sh
# =============================================================

echo "Defence-sweep-v2 job $SLURM_JOB_ID on $(hostname) at $(date)"

module load Python/3.10.14
module load CUDA/12.4
source /projects/secure-distributed-ml/venv/bin/activate

python -c "import breaching" 2>/dev/null || {
    echo "[setup] breaching not found, installing..."
    pip install --no-cache-dir "breaching @ git+https://github.com/JonasGeiping/breaching.git"
}
python -c "import breaching; print(f'breaching OK: {breaching.__file__}')"

ARTIFACT_DIR="${ARTIFACT_DIR:-/scratch/projects/secure-distributed-ml/results/1k_experiment/artifacts_none_seed42_300clients}"
ROUND="${ROUND:-249}"
DEFENCE_PCT="${DEFENCE_PCT:-0.1}"
OUTDIR="/scratch/projects/secure-distributed-ml/results/defence_v2_${SLURM_JOB_ID}"
mkdir -p "$OUTDIR"

cd /projects/secure-distributed-ml/AdaGuard

echo "Artifact dir : $ARTIFACT_DIR"
echo "Round        : $ROUND"
echo "Defence pct  : $DEFENCE_PCT (MaskCrypt + Fisher)"
echo ""

run_one () {
    local attack="$1"; local iters="$2"; local defence="$3"
    local tag="${defence}_${attack}_b1"
    echo "============================================================"
    echo "  ATTACK=$attack  DEFENCE=$defence  B=1  iters=$iters"
    echo "============================================================"
    stdbuf -oL -eL python -u tests/attack_sanity_check.py \
        --artifact-dir "$ARTIFACT_DIR" \
        --round "$ROUND" \
        --attack "$attack" \
        --n-iter "$iters" \
        --batch-size 1 \
        --defence "$defence" \
        --defence-pct "$DEFENCE_PCT" \
        --variant paper \
        --out "$OUTDIR/${tag}.json"
    echo ""
}

# Only re-run GradInversion (breaching-backed) against each defence.
# GGCDM/GI-NAS rows from job 181367 are reused (no upstream reference
# implementations exist to validate against).
for defence in none fhe maskcrypt fisher; do
    run_one gradinversion_breaching 20000 "$defence"
done

echo "Defence sweep v2 finished at $(date)"
echo "Artifacts in $OUTDIR"
echo ""
echo "Summary (compare PSNR against defence_181367 GradInv numbers:"
echo "  none=18.32  fhe=9.20  maskcrypt=10.43  fisher=9.14)"
for j in "$OUTDIR"/*.json; do
    python -c "
import json, os
d = json.load(open(r'$j'))
print(f\"{os.path.basename(r'$j'):50s}  PSNR={d.get('psnr', float('nan')):7.2f} dB  LPIPS={d.get('lpips', 0):.4f}  labelASR={d.get('label_recovery',{}).get('asr',0):.3f}\")
"
done
