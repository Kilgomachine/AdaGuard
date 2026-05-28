#!/bin/bash
#SBATCH --job-name=adaguard-trajectory-p2
#SBATCH --output=/scratch/projects/secure-distributed-ml/logs/trajectory-p2-%A_%a.out
#SBATCH --error=/scratch/projects/secure-distributed-ml/logs/trajectory-p2-%A_%a.err
#SBATCH --partition=general-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=3:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maguir@oakland.edu
#SBATCH --array=0-2
# =============================================================
# Section 1 of the Fisher-sensitivity follow-up:
# Replay V4 (AdaGuard-Fisher) at 4 encryption fractions x 3 attacks
# at each of 3 training-stage checkpoints (rounds 75, 150, 249) on
# the existing ResNet-18 1k_experiment seed-42 Phase-1 artefacts.
# Single seed (matches K-sweep / T1xrho / B-sweep methodology).
#
# V1/V2/V3 baselines are DELIBERATELY OMITTED per the
# structured-plan deferral of the headline-matrix replication.
# This script measures how the V4 defence response (Fisher targeting)
# varies with encryption budget rho across training rounds -- the
# core question of the Fisher-sensitivity campaign.
#
# Tasks map to round indices:
#   0: round 75   (early stage)
#   1: round 150  (mid stage)
#   2: round 249  (late stage; matches the existing headline cell)
#
# Per task: 4 rho values x 3 attacks = 12 attack runs.
# Total across the array: 36 attack JSONs.
#
# This is a Phase-2-only replay script. It expects the per-round
# artefacts to already exist on /scratch under the canonical
# 1k_experiment layout. If they were cleaned by the 45-day expiry,
# use slurm_trajectory_full.sh instead (Phase-1 retrain + replay).
#
# Forked from slurm_k_sweep_phase2_only.sh (commit 8aff4fa).
# =============================================================

ROUNDS=(75 150 249)
RHOS=(0.05 0.10 0.15 0.20)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
ROUND="${ROUNDS[$TASK_ID]}"
SEED=42  # single seed per Section 1 design; see EXPERIMENTAL_PLAN.md

echo "Trajectory Phase-2-replay $SLURM_ARRAY_JOB_ID.$SLURM_ARRAY_TASK_ID at $(date)"
echo "  ROUND=$ROUND  SEED=$SEED  RHOS=${RHOS[*]}"

module load Python/3.10.14
module load CUDA/12.4
source /projects/secure-distributed-ml/venv/bin/activate

# Existing ResNet-18 1k_experiment artefacts. The 1k_experiment ran at
# 300 clients despite the directory name (paper §V-B); see HANDOFF §7.
OUTROOT="/scratch/projects/secure-distributed-ml/results/1k_experiment"
ARTIFACT_DIR="$OUTROOT/artifacts_fisher_seed${SEED}_300clients"

# Stage Section 1 outputs under their own subtree so they don't collide
# with the existing 1k_experiment attack JSONs.
ATTACK_DIR="$OUTROOT/trajectory_section1/round${ROUND}_seed${SEED}"
mkdir -p "$ATTACK_DIR"

# Sanity-check: the requested round must exist in the saved artefacts.
# Fall back to nearest available if missing (per slurm_k_sweep_phase2_only's
# rescue-script convention; the actual round used gets baked into the
# output filename).
TARGET_ROUND="$ROUND"
if [ ! -d "$ARTIFACT_DIR/round_${TARGET_ROUND}" ]; then
    AVAILABLE=$(ls -d "$ARTIFACT_DIR"/round_* 2>/dev/null | sed 's/.*round_//' | sort -n)
    if [ -z "$AVAILABLE" ]; then
        echo "ERROR: $ARTIFACT_DIR has no saved rounds. Phase-1 didn't save anything,"
        echo "       or the artefacts were cleaned by 45-day expiry."
        echo "       Use slurm_trajectory_full.sh to retrain Phase-1 with checkpoint saves."
        exit 1
    fi
    NEAREST=$(echo "$AVAILABLE" | awk -v target="$TARGET_ROUND" \
        'BEGIN{best=-1; gap=99999} {g=$1-target; if(g<0)g=-g; if(g<gap){gap=g; best=$1}} END{print best}')
    echo "WARNING: round_${TARGET_ROUND} missing, using nearest available: round_${NEAREST}"
    TARGET_ROUND="$NEAREST"
fi
echo "  Using artefact round: $TARGET_ROUND"

cd /projects/secure-distributed-ml/AdaGuard

run_attack () {
    local attack="$1"; local iters="$2"; local rho="$3"
    # Filename: fisher_{attack}_b1_rho{R}_round{R}_seed{S}.json
    # Pattern matches a new loader (load_trajectory_rhosweep) we'll add to
    # scripts/paper/_data.py so the paper-build pipeline can consume it.
    # Substitute "." -> "p" in rho for safe filenames (e.g. 0.10 -> 0p10).
    local rho_tag="${rho//./p}"
    local tag="fisher_${attack}_b1_rho${rho_tag}_round${TARGET_ROUND}_seed${SEED}"
    echo "  ATTACK=$attack RHO=$rho ROUND=$TARGET_ROUND SEED=$SEED"
    stdbuf -oL -eL python -u tests/attack_sanity_check.py \
        --artifact-dir "$ARTIFACT_DIR" \
        --round "$TARGET_ROUND" \
        --attack "$attack" \
        --n-iter "$iters" \
        --batch-size 1 \
        --defence fisher \
        --defence-pct "$rho" \
        --variant paper \
        --out "$ATTACK_DIR/${tag}.json"
}

# V4 (Fisher) only, 4 rho values x 3 attacks per stage = 12 attack cells.
# Iteration budgets match the existing slurm_k_sweep_phase2_only.sh.
for rho in "${RHOS[@]}"; do
    run_attack gradinversion 20000 "$rho"
    run_attack ggcdm         100   "$rho"
    run_attack gi_nas        2000  "$rho"
done

echo "Trajectory Phase-2-replay task finished at $(date)"
echo "JSONs in $ATTACK_DIR"
