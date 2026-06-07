#!/bin/bash
#SBATCH --job-name=adaguard-r50-p2
#SBATCH --output=/scratch/projects/secure-distributed-ml/logs/r50-p2-%j.out
#SBATCH --error=/scratch/projects/secure-distributed-ml/logs/r50-p2-%j.err
#SBATCH --partition=general-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maguir@oakland.edu
# =============================================================
# ResNet-50 extended rho-sweep, PHASE-2 ONLY against the
# artefacts from job 186093 (which TIMEOUT'd at the 36h wall
# after Phase-1 reached round 240 -- 9 rounds shy of the
# target 249, but still a usable late-training checkpoint).
#
# Strategy: skip Phase-1 entirely, point at the existing
# artifacts dir, run the same 11-rho x 3-attack sweep that the
# original script intended. The nearest-round logic below will
# auto-resolve target 249 -> round 240 since 240 is the highest
# saved.
#
# Time budget: 11 rhos x 3 attacks = 33 cells. At ~5-7 min each
# on a V100 (GradInv 20k iters, GGCDM 100 iters, GI-NAS 2k iters),
# headline ~3-4h. 6h limit gives buffer for slow attack convergence
# and any single-cell GPU-OOM retry.
#
# Output goes to a NEW dir so we don't clobber the 186093 logs.
# The source artefacts are READ-ONLY for this job.
#
# Forked from slurm_resnet50_extended_rho.sh; differs only in
# (i) Phase-1 skipped, (ii) ARTIFACT_DIR points at 186093,
# (iii) time limit reduced from 36h to 6h.
# =============================================================

SEED=42
SOURCE_JOB=186093

echo "ResNet-50 extended-rho Phase-2-only job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "  seed=$SEED"
echo "  source artefacts from job $SOURCE_JOB"

module load Python/3.10.14
module load CUDA/12.4
source /projects/secure-distributed-ml/venv/bin/activate

export DATA_DIR=/scratch/projects/secure-distributed-ml/data

# Source artefacts (from job 186093 TIMEOUT'd Phase-1).
SOURCE_OUTROOT="/scratch/projects/secure-distributed-ml/results/resnet50_extended_seed${SEED}_${SOURCE_JOB}"
ARTIFACT_DIR="$SOURCE_OUTROOT/artifacts"

# This job's output (Phase-2 attack JSONs only -- no Phase-1).
OUTROOT="/scratch/projects/secure-distributed-ml/results/resnet50_extended_phase2_seed${SEED}_${SLURM_JOB_ID}"
ATTACK_DIR="$OUTROOT/attacks"
mkdir -p "$ATTACK_DIR"

if [ ! -d "$ARTIFACT_DIR" ]; then
    echo "ERROR: source artefacts not at $ARTIFACT_DIR"
    echo "       Did job $SOURCE_JOB get cleaned up?"
    exit 1
fi

NUM_ROUNDS=$(ls -d "$ARTIFACT_DIR"/round_* 2>/dev/null | wc -l)
HIGHEST=$(ls -d "$ARTIFACT_DIR"/round_* 2>/dev/null | sed 's/.*round_//' | sort -n | tail -1)
echo "  artefact inventory: $NUM_ROUNDS saved rounds, highest = round_$HIGHEST"

cd /projects/secure-distributed-ml/AdaGuard

# Target round 249 (matches paper headline); nearest-round logic
# will resolve to round 240 since that is the highest saved.
ROUNDS=(249)
RHOS=(0.05 0.10 0.15 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90)

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
        --model resnet50 \
        --out "$ATTACK_DIR/${tag}.json"
}

for ROUND in "${ROUNDS[@]}"; do
    TARGET_ROUND="$ROUND"
    if [ ! -d "$ARTIFACT_DIR/round_${TARGET_ROUND}" ]; then
        AVAILABLE=$(ls -d "$ARTIFACT_DIR"/round_* 2>/dev/null | sed 's/.*round_//' | sort -n)
        if [ -z "$AVAILABLE" ]; then
            echo "ERROR: $ARTIFACT_DIR has no saved rounds."
            exit 1
        fi
        NEAREST=$(echo "$AVAILABLE" | awk -v t="$TARGET_ROUND" \
            'BEGIN{best=-1; gap=99999} {g=$1-t; if(g<0)g=-g; if(g<gap){gap=g; best=$1}} END{print best}')
        echo "  NOTE: round_${TARGET_ROUND} missing, using nearest available: round_${NEAREST}"
        TARGET_ROUND="$NEAREST"
    fi

    for RHO in "${RHOS[@]}"; do
        run_attack gradinversion 20000 "$RHO" "$TARGET_ROUND"
        run_attack ggcdm         100   "$RHO" "$TARGET_ROUND"
        run_attack gi_nas        2000  "$RHO" "$TARGET_ROUND"
    done
done

echo ""
echo "ResNet-50 extended-rho Phase-2-only job finished at $(date)"
echo "Output in $OUTROOT"
echo "  Attack JSONs: $ATTACK_DIR"
ls "$ATTACK_DIR" 2>/dev/null | wc -l
echo "(expecting 33 = 11 rhos x 3 attacks)"

# Reminder: source artefacts at $ARTIFACT_DIR (~643 GB) remain
# untouched and can be reused for further Phase-2 sweeps if needed.
