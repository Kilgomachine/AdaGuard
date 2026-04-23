#!/bin/bash
#SBATCH --job-name=adaguard-defence-sweep
#SBATCH --output=/scratch/projects/secure-distributed-ml/logs/defence-%j.out
#SBATCH --error=/scratch/projects/secure-distributed-ml/logs/defence-%j.err
#SBATCH --partition=general-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=6:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maguir@oakland.edu
# =============================================================
# AdaGuard Phase-4 defence sweep at B=1 (iDLG regime).
#
# Prior sanity work established:
#   - Attacks land at B=1 on V1: GradInv 18.29 dB, GI-NAS 19.69 dB.
#   - B=4/16 is at the noise floor even undefended due to Fisher
#     non-IID forcing <=3 unique labels per client batch.
#   - Artifact self-consistency verified (max_rel_diff 1.67e-3).
#
# Therefore defence evaluation is only meaningful at B=1. This
# job runs the 4x3 matrix (V1/V2/V4/V6 x GradInv/GGCDM/GI-NAS)
# on a single V1 artifact with sweep-time defence re-application
# (simulator.py:455 saves raw pre-defence gradient, so we apply
# the defence inside attack_sanity_check.py via --defence).
#
# The three attacks are the paper's "Shadow Attacks" framework
# (sec IV.E): GradInversion covers batch resilience, GGCDM covers
# noise resilience, GI-NAS covers prior/heterogeneity resilience.
# Our GGCDM is a DPS proxy (Meng doesn't evaluate on CIFAR), this
# caveat goes in sec VI.E protocol-fidelity limitations.
#
# Expected runtime:
#   - 12 runs. GradInv 20k iters at B=1 ~= 5 min each on V100.
#   - GGCDM 100 iters at B=1 ~= 2 min each.
#   - GI-NAS 2k iters at B=1 ~= 1 min each.
#   - Total ~= 4 x (5+2+1) = 32 min. Budget 6h for safety margin.
#
# Usage:
#   sbatch hpc/slurm_defence_sweep.sh
#   DEFENCE_PCT=0.2 sbatch hpc/slurm_defence_sweep.sh
# =============================================================

echo "Defence-sweep job $SLURM_JOB_ID on $(hostname) at $(date)"

module load Python/3.10.14
module load CUDA/12.4
source /projects/secure-distributed-ml/venv/bin/activate

ARTIFACT_DIR="${ARTIFACT_DIR:-/scratch/projects/secure-distributed-ml/results/1k_experiment/artifacts_none_seed42_300clients}"
ROUND="${ROUND:-249}"
DEFENCE_PCT="${DEFENCE_PCT:-0.1}"
OUTDIR="/scratch/projects/secure-distributed-ml/results/defence_${SLURM_JOB_ID}"
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

# V1: undefended; V2: full FHE (all grads zeroed); V4: MaskCrypt;
# V6: AdaGuard core (Fisher encryption, top defence-pct by Fisher score).
for defence in none fhe maskcrypt fisher; do
    run_one gradinversion 20000 "$defence"
    run_one ggcdm         100   "$defence"
    run_one gi_nas        2000  "$defence"
done

echo "Defence sweep finished at $(date)"
echo "Artifacts in $OUTDIR"
echo ""
echo "Summary table:"
for j in "$OUTDIR"/*.json; do
    python -c "
import json, os
d = json.load(open('$j' if False else r'$j'))
print(f\"{os.path.basename(r'$j'):40s}  PSNR={d.get('psnr', float('nan')):7.2f} dB  LPIPS={d.get('lpips', 0):.4f}  labelASR={d.get('label_recovery',{}).get('asr',0):.3f}\")
"
done
