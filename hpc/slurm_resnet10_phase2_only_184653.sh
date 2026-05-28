#!/bin/bash
#SBATCH --job-name=adaguard-resnet10-p2
#SBATCH --output=/scratch/projects/secure-distributed-ml/logs/resnet10-p2-%j.out
#SBATCH --error=/scratch/projects/secure-distributed-ml/logs/resnet10-p2-%j.err
#SBATCH --partition=general-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=4:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maguir@oakland.edu
# =============================================================
# Section 2 recovery: re-run Phase-2 attacks against the ALREADY-
# trained ResNet-10 artefacts from job 184653. The original Phase-1
# completed successfully (128 saved rounds, training.json present);
# all 36 Phase-2 attacks failed because attack_sanity_check.py
# defaults --model to resnet18 and tried to load ResNet-10 weights
# into a ResNet-18 model. Fixed by passing --model resnet10.
#
# Re-uses existing artefacts at:
#   /scratch/.../resnet10_seed42_184653/artifacts/
# Writes new JSONs to:
#   /scratch/.../resnet10_seed42_184653/attacks/  (currently empty)
#
# Expected wall-clock: ~3h (just the 36 attack runs, no Phase-1).
# =============================================================

SEED=42
ORIG_JOB=184653

echo "ResNet-10 Phase-2 recovery job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "  source artefacts: job $ORIG_JOB, seed=$SEED"

module load Python/3.10.14
module load CUDA/12.4
source /projects/secure-distributed-ml/venv/bin/activate

export DATA_DIR=/scratch/projects/secure-distributed-ml/data

OUTROOT="/scratch/projects/secure-distributed-ml/results/resnet10_seed${SEED}_${ORIG_JOB}"
ARTIFACT_DIR="$OUTROOT/artifacts"
ATTACK_DIR="$OUTROOT/attacks"
mkdir -p "$ATTACK_DIR"

if [ ! -d "$ARTIFACT_DIR" ]; then
    echo "ERROR: artefact dir $ARTIFACT_DIR not found. Did 184653 get cleaned up?"
    exit 1
fi

cd /projects/secure-distributed-ml/AdaGuard

ROUNDS=(75 150 249)
RHOS=(0.05 0.10 0.15 0.20)

run_attack () {
    local attack="$1"; local iters="$2"; local rho="$3"; local round="$4"
    local rho_tag="${rho//./p}"
    local tag="fisher_${attack}_b1_rho${rho_tag}_round${round}_seed${SEED}"
    echo "  ATTACK=$attack RHO=$rho ROUND=$round SEED=$SEED"
    stdbuf -oL -eL python -u tests/attack_sanity_check.py \
        --artifact-dir "$ARTIFACT_DIR" \
        --round "$round" \
        --attack "$attack" \
        --n-iter "$iters" \
        --batch-size 1 \
        --defence fisher \
        --defence-pct "$rho" \
        --variant paper \
        --model resnet10 \
        --out "$ATTACK_DIR/${tag}.json"
}

for ROUND in "${ROUNDS[@]}"; do
    TARGET_ROUND="$ROUND"
    if [ ! -d "$ARTIFACT_DIR/round_${TARGET_ROUND}" ]; then
        AVAILABLE=$(ls -d "$ARTIFACT_DIR"/round_* 2>/dev/null | sed 's/.*round_//' | sort -n)
        NEAREST=$(echo "$AVAILABLE" | awk -v t="$TARGET_ROUND" \
            'BEGIN{best=-1; gap=99999} {g=$1-t; if(g<0)g=-g; if(g<gap){gap=g; best=$1}} END{print best}')
        echo "  NOTE: round_${TARGET_ROUND} missing, using nearest: round_${NEAREST}"
        TARGET_ROUND="$NEAREST"
    fi

    for RHO in "${RHOS[@]}"; do
        run_attack gradinversion 20000 "$RHO" "$TARGET_ROUND"
        run_attack ggcdm         100   "$RHO" "$TARGET_ROUND"
        run_attack gi_nas        2000  "$RHO" "$TARGET_ROUND"
    done
done

# Optional cleanup -- delete artefacts after Phase-2 lands (~150 GB freed).
# Commented out so you can verify JSONs first; uncomment if you want
# automatic cleanup on success.
# echo ""
# echo "=== Phase-3 cleanup: removing $ARTIFACT_DIR ==="
# rm -rf "$ARTIFACT_DIR"

echo ""
echo "ResNet-10 Phase-2 recovery finished at $(date)"
echo "JSONs in $ATTACK_DIR"
ls "$ATTACK_DIR" | wc -l
echo "(expecting 36)"
