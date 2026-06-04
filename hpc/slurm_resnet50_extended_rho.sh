#!/bin/bash
#SBATCH --job-name=adaguard-r50-ext
#SBATCH --output=/scratch/projects/secure-distributed-ml/logs/r50-ext-%j.out
#SBATCH --error=/scratch/projects/secure-distributed-ml/logs/r50-ext-%j.err
#SBATCH --partition=general-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=36:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maguir@oakland.edu
# =============================================================
# ResNet-50 extended rho-sweep: fresh Phase-1 retrain (~25h),
# then Phase-2 at 8 encryption fractions x 3 attacks at round 249
# only (~3-4h).
#
# Purpose: test whether ResNet-50's observed PSNR floor (~12 dB
# across rho 0.05-0.20 from job 185522) breaks at higher rho, or
# whether it represents a saturation regime bounded by non-gradient
# leakage (BN running stats, label-conditional priors, model state).
#
# Rho values: {0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70,
# 0.80, 0.90}. The 0.05-0.20 cells reproduce job 185522's measurements
# as a sanity check; 0.30-0.90 are the new floor-probing values.
# Sweeping all the way to 0.90 lets us test whether the observed
# ~12 dB floor persists even when only 10% of the gradient remains
# unencrypted -- if the floor holds at rho=0.90, encryption alone
# cannot break it (the residual reconstruction is bounded by
# non-gradient leakage: BN running stats, label-conditional priors,
# model state). If the floor breaks somewhere in {0.50-0.90}, we
# have a quantitative scaling-law data point for the paper.
#
# Round 249 only -- single late-stage measurement at each rho,
# matching the existing paper's headline-matrix methodology and
# keeping Phase-2 wall-clock to ~3-4h.
#
# PHASE-3 CLEANUP DISABLED: per-round artefacts are preserved
# (~700 GB) so we can run additional Phase-2-only sweeps (e.g.
# multi-stage extended rho, or alternative attack variants)
# without retraining. The cleanup decision moved to a manual
# follow-up.
#
# Forked from slurm_resnet50_train_and_attack.sh (commit history
# shows the quantile fix, Gini sort to CPU, clients_per_gpu=1).
# =============================================================

SEED=42

echo "Section-3-extended ResNet-50 job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "  seed=$SEED"

module load Python/3.10.14
module load CUDA/12.4
source /projects/secure-distributed-ml/venv/bin/activate

export DATA_DIR=/scratch/projects/secure-distributed-ml/data

OUTROOT="/scratch/projects/secure-distributed-ml/results/resnet50_extended_seed${SEED}_${SLURM_JOB_ID}"
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

# Step 2: Phase-2 -- extended rho sweep at round 249 only.
echo ""
echo "=== Phase-2 V4 extended rho-sweep at round 249 ==="
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

# NOTE: Phase-3 cleanup is INTENTIONALLY DISABLED for this job so the
# per-round artefacts (~700 GB) survive for follow-up Phase-2-only
# sweeps. Manual cleanup post-analysis:
#   rm -rf $ARTIFACT_DIR

echo ""
echo "Section-3-extended ResNet-50 job finished at $(date)"
echo "Output in $OUTROOT"
echo "  Phase-1 artefacts: $ARTIFACT_DIR (~700 GB, PRESERVED for follow-up sweeps)"
echo "  Phase-2 attack JSONs: $ATTACK_DIR (expect 33)"
echo "  Training trajectory: $OUTROOT/training.json"
ls "$ATTACK_DIR" 2>/dev/null | wc -l
echo "(expecting 33 = 11 rhos x 3 attacks)"
