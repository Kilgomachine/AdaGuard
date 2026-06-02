#!/bin/bash
#SBATCH --job-name=adaguard-resnet50
#SBATCH --output=/scratch/projects/secure-distributed-ml/logs/resnet50-%j.out
#SBATCH --error=/scratch/projects/secure-distributed-ml/logs/resnet50-%j.err
#SBATCH --partition=general-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=30:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maguir@oakland.edu
# =============================================================
# Section 3 of the Fisher-sensitivity follow-up:
# Train ResNet-50 Phase-1 (single seed=42, 250 rounds, save_every=2),
# then replay V4 (Fisher) at 4 encryption fractions x 3 attacks at
# each of 3 training-stage checkpoints (rounds 75, 150, 249).
#
# Architecture: ResNet-50 (~23.5M params, Bottleneck [3,4,6,3]).
# Already in adaguard/models/registry; no model code change needed.
#
# Single-task job (no array). Expected wall-clock:
#   Phase-1 retrain     ~10-13h (ResNet-50 ~2-3x ResNet-18 per round)
#   Phase-2 36 attacks  ~3-4h
#   Total               ~14-17h, fits inside 20h limit.
#
# Resource bumps vs ResNet-10 script: mem 24G -> 32G (Bottleneck
# block activation footprint), time 12h -> 20h (slower per round).
#
# Forked from slurm_resnet10_train_and_attack.sh.
# =============================================================

SEED=42

echo "Section-3 ResNet-50 job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "  seed=$SEED"

module load Python/3.10.14
module load CUDA/12.4
source /projects/secure-distributed-ml/venv/bin/activate

# CIFAR-10 cache (prevents torchvision re-download; see slurm_k_sweep.sh)
export DATA_DIR=/scratch/projects/secure-distributed-ml/data

OUTROOT="/scratch/projects/secure-distributed-ml/results/resnet50_seed${SEED}_${SLURM_JOB_ID}"
ARTIFACT_DIR="$OUTROOT/artifacts"
ATTACK_DIR="$OUTROOT/attacks"
mkdir -p "$ARTIFACT_DIR" "$ATTACK_DIR"

cd /projects/secure-distributed-ml/AdaGuard

# Step 1: Phase-1 retrain.
echo ""
echo "=== Phase-1 retrain (ResNet-50) ==="
stdbuf -oL -eL python -u run_headless.py \
    --config hpc/config_resnet50_t2_low.yaml \
    --output "$OUTROOT/training.json" \
    --tb-dir "$OUTROOT/tb_logs" \
    --artifacts-dir "$ARTIFACT_DIR" \
    --seed "$SEED"

# Step 2: Phase-2 V4 rho-sweep at 3 stages x 3 attacks.
echo ""
echo "=== Phase-2 V4 rho-sweep at 3 stages ==="
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
        --model resnet50 \
        --out "$ATTACK_DIR/${tag}.json"
}

for ROUND in "${ROUNDS[@]}"; do
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
echo "=== Phase-3 cleanup: removing heavy gradient artefacts ==="
# Per user preference for "live compute" semantics: keep attack JSONs
# (the actual paper data) + training.json + tb_logs, but wipe the heavy
# per-round gradient artefacts after Phase-2 has finished consuming them.
# Critical for ResNet-50 because the per-round artefact footprint is
# ~700 GB-1 TB; preserving it would eat ~10-15% of the user's 10 TB
# quota per architecture. Permanent footprint drops from ~700 GB to
# ~50 MB.
PRE_CLEANUP_SIZE=$(du -sh "$ARTIFACT_DIR" 2>/dev/null | awk '{print $1}')
rm -rf "$ARTIFACT_DIR"
POST_CLEANUP_SIZE=$(du -sh "$OUTROOT" 2>/dev/null | awk '{print $1}')
echo "  Pre-cleanup artefact dir size: $PRE_CLEANUP_SIZE"
echo "  Post-cleanup OUTROOT size: $POST_CLEANUP_SIZE (just JSONs + training.json + tb_logs)"

echo ""
echo "Section-3 ResNet-50 job finished at $(date)"
echo "Output in $OUTROOT"
echo "  Phase-2 attack JSONs: $ATTACK_DIR (36 JSONs)"
echo "  Training trajectory: $OUTROOT/training.json"
echo "  Phase-1 artefacts: CLEANED UP (see Phase-3 cleanup step above)"
