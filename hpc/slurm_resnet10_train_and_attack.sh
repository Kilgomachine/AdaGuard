#!/bin/bash
#SBATCH --job-name=adaguard-resnet10
#SBATCH --output=/scratch/projects/secure-distributed-ml/logs/resnet10-%j.out
#SBATCH --error=/scratch/projects/secure-distributed-ml/logs/resnet10-%j.err
#SBATCH --partition=general-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maguir@oakland.edu
# =============================================================
# Section 2 of the Fisher-sensitivity follow-up:
# Train ResNet-10 Phase-1 (single seed=42, 250 rounds, save_every=2),
# then replay V4 (Fisher) at 4 encryption fractions x 3 attacks at
# each of 3 training-stage checkpoints (rounds 75, 150, 249).
#
# Architecture: ResNet-10 (~5M params, BasicBlock [1,1,1,1]).
# Adds the new model entry in adaguard/models/resnet.py + registry
# (commits to master before this script runs).
#
# Single-task job (no array). Expected wall-clock:
#   Phase-1 retrain     ~4-5h (ResNet-10 ~50% faster per round vs ResNet-18)
#   Phase-2 36 attacks  ~3h
#   Total               ~7-8h, well within the 12h limit.
#
# Forked from slurm_k_sweep.sh (commit 8aff4fa).
# =============================================================

SEED=42

echo "Section-2 ResNet-10 job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "  seed=$SEED"

module load Python/3.10.14
module load CUDA/12.4
source /projects/secure-distributed-ml/venv/bin/activate

# CIFAR-10 cache lives on scratch -- prevents torchvision re-download
# (see slurm_k_sweep.sh comment for the 2026-04-29 root cause).
export DATA_DIR=/scratch/projects/secure-distributed-ml/data

OUTROOT="/scratch/projects/secure-distributed-ml/results/resnet10_seed${SEED}_${SLURM_JOB_ID}"
ARTIFACT_DIR="$OUTROOT/artifacts"
ATTACK_DIR="$OUTROOT/attacks"
mkdir -p "$ARTIFACT_DIR" "$ATTACK_DIR"

cd /projects/secure-distributed-ml/AdaGuard

# Step 1: Phase-1 retrain.
# We use the dedicated config_resnet10_t2_low.yaml verbatim --
# it already has model: resnet10 and the right focus_layers /
# save_every_n_rounds. No per-task YAML override needed.
echo ""
echo "=== Phase-1 retrain (ResNet-10) ==="
stdbuf -oL -eL python -u run_headless.py \
    --config hpc/config_resnet10_t2_low.yaml \
    --output "$OUTROOT/training.json" \
    --tb-dir "$OUTROOT/tb_logs" \
    --artifacts-dir "$ARTIFACT_DIR" \
    --seed "$SEED"

# Step 2: Phase-2 V4 rho-sweep at 3 stages x 3 attacks.
# V4 (fisher) only -- V1/V2/V3 omitted per the deferral.
echo ""
echo "=== Phase-2 V4 rho-sweep at 3 stages ==="
ROUNDS=(75 150 249)
RHOS=(0.05 0.10 0.15 0.20)

run_attack () {
    local attack="$1"; local iters="$2"; local rho="$3"; local round="$4"
    # Filename: fisher_{attack}_b1_rho{R}_round{R}_seed{S}.json
    # Same naming convention as slurm_trajectory_phase2_only.sh so the
    # same loader (load_trajectory_rhosweep) handles both.
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
        --out "$ATTACK_DIR/${tag}.json"
}

for ROUND in "${ROUNDS[@]}"; do
    # Same nearest-available fallback as slurm_trajectory_phase2_only.sh
    # (save_every_n_rounds=2 means odd rounds aren't saved; round 75 -> 74).
    TARGET_ROUND="$ROUND"
    if [ ! -d "$ARTIFACT_DIR/round_${TARGET_ROUND}" ]; then
        AVAILABLE=$(ls -d "$ARTIFACT_DIR"/round_* 2>/dev/null | sed 's/.*round_//' | sort -n)
        if [ -z "$AVAILABLE" ]; then
            echo "ERROR: $ARTIFACT_DIR has no saved rounds. Phase-1 must have failed."
            exit 1
        fi
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

echo ""
echo "Section-2 ResNet-10 job finished at $(date)"
echo "Artifacts in $OUTROOT"
echo "  Phase-1: $ARTIFACT_DIR (~5-7 GB, ~125 rounds x 30 clients)"
echo "  Phase-2: $ATTACK_DIR (36 JSONs)"
