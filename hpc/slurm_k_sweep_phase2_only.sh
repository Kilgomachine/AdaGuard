#!/bin/bash
#SBATCH --job-name=adaguard-k-sweep-p2
#SBATCH --output=/scratch/projects/secure-distributed-ml/logs/ksweep-p2-%A_%a.out
#SBATCH --error=/scratch/projects/secure-distributed-ml/logs/ksweep-p2-%A_%a.err
#SBATCH --partition=general-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=1:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maguir@oakland.edu
#SBATCH --array=0-9
# =============================================================
# Recovery script: re-run Phase-2 attack matrix against the K-sweep
# Phase-1 artefacts that were saved by job 181987 before my broken
# breaching wrapper docstring (commit d840828) crashed the Phase-2
# import. Phase-1 data is intact (~320 GB each); we just need to
# rerun the 4 defences x 3 attacks matrix per task.
#
# Tasks 0-9 map to (K, seed) pairs that completed Phase-1 in 181987:
#   0: K=1 seed42    1: K=1 seed123    2: K=1 seed456
#   3: K=2 seed42    4: K=2 seed123    5: K=2 seed456
#   6: K=4 seed42    7: K=4 seed123    8: K=4 seed456
#   9: K=8 seed42
#
# Tasks 10-14 (K=8 seeds 123/456, K=16 all 3 seeds) need full
# Phase-1 retrains -- re-submit slurm_k_sweep.sh with --array=10-14
# instead.
# =============================================================

K_VALUES=(1 2 4 8 16)
SEEDS=(42 123 456)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
K_IDX=$((TASK_ID / 3))
SEED_IDX=$((TASK_ID % 3))
K_VAL="${K_VALUES[$K_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"

echo "K-sweep Phase-2-recovery $SLURM_ARRAY_JOB_ID.$SLURM_ARRAY_TASK_ID at $(date)"
echo "  K=$K_VAL  seed=$SEED"

module load Python/3.10.14
module load CUDA/12.4
source /projects/secure-distributed-ml/venv/bin/activate

# Existing Phase-1 artefacts from the 181987 run.
OUTROOT="/scratch/projects/secure-distributed-ml/results/ksweep_K${K_VAL}_seed${SEED}_181987"
ARTIFACT_DIR="$OUTROOT/artifacts"
ATTACK_DIR="$OUTROOT/attacks"
mkdir -p "$ATTACK_DIR"

# Prefer round 249 (training target). Fall back to the latest available
# round if 249 is missing -- some 181987 Phase-1 runs saved through
# rounds 240-244 before the docstring crash, and at convergence the
# 5-7 round delta is < 0.5 dB PSNR drift (well within seed-to-seed noise).
TARGET_ROUND=249
if [ ! -d "$ARTIFACT_DIR/round_${TARGET_ROUND}" ]; then
    LATEST=$(ls -d "$ARTIFACT_DIR"/round_* 2>/dev/null | sed 's/.*round_//' | sort -n | tail -1)
    if [ -n "$LATEST" ]; then
        echo "WARNING: round_${TARGET_ROUND} missing, falling back to latest available: round_${LATEST}"
        TARGET_ROUND="$LATEST"
    else
        echo "ERROR: $ARTIFACT_DIR has no saved rounds -- Phase-1 didn't save anything."
        echo "Re-submit slurm_k_sweep.sh for this (K, seed) instead."
        exit 1
    fi
fi
echo "  Using artefact round: $TARGET_ROUND"

cd /projects/secure-distributed-ml/AdaGuard

run_attack () {
    local attack="$1"; local iters="$2"; local defence="$3"
    local tag="${defence}_${attack}_b1_K${K_VAL}_seed${SEED}"
    echo "  ATTACK=$attack DEFENCE=$defence K=$K_VAL SEED=$SEED ROUND=$TARGET_ROUND"
    stdbuf -oL -eL python -u tests/attack_sanity_check.py \
        --artifact-dir "$ARTIFACT_DIR" \
        --round "$TARGET_ROUND" \
        --attack "$attack" \
        --n-iter "$iters" \
        --batch-size 1 \
        --defence "$defence" \
        --defence-pct 0.10 \
        --variant paper \
        --out "$ATTACK_DIR/${tag}.json"
}

for defence in none fhe maskcrypt fisher; do
    run_attack gradinversion 20000 "$defence"
    run_attack ggcdm         100   "$defence"
    run_attack gi_nas        2000  "$defence"
done

echo "Phase-2 recovery task finished at $(date)"
echo "Artifacts in $ATTACK_DIR"
