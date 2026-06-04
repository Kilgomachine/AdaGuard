#!/bin/bash
#SBATCH --job-name=adaguard-r18-traj
#SBATCH --output=/scratch/projects/secure-distributed-ml/logs/r18-traj-%j.out
#SBATCH --error=/scratch/projects/secure-distributed-ml/logs/r18-traj-%j.err
#SBATCH --partition=general-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=6:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maguir@oakland.edu
# =============================================================
# ResNet-18 trajectory redo: replay V4 (Fisher) at 4 encryption
# fractions x 3 attacks at each of 3 training-stage checkpoints
# (rounds 75, 150, 249) against the EXISTING 1k_experiment ResNet-18
# seed-42 Phase-1 artefacts.
#
# Why redo: original Section 1 trajectory runs (jobs 184645, 184648)
# predated three fixes that affect the JSONs:
#   - LLG false-positive at signal=0 (corrected to predicted=[-1])
#   - Fisher tie-at-zero in selection (corrected to non-zero-only)
#   - Both above are now in master
#
# Recomputing gives us one self-consistent dataset across ResNet-10,
# ResNet-18, and ResNet-50 for the architecture-scaling story.
#
# Round 249 PSNRs should match the existing paper §VI-F numbers within
# noise (the fixes don't affect PSNR mechanics, only ASR reporting and
# early-training corner-case encryption mask).
#
# Single-task, ~3-4h wall-clock (36 attack cells at standard iter counts).
# Forked from slurm_resnet10_phase2_only_184653.sh.
# =============================================================

SEED=42

echo "ResNet-18 trajectory redo job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "  source artefacts: 1k_experiment ResNet-18 fisher seed=$SEED"

module load Python/3.10.14
module load CUDA/12.4
source /projects/secure-distributed-ml/venv/bin/activate

export DATA_DIR=/scratch/projects/secure-distributed-ml/data

# Existing ResNet-18 1k_experiment artefacts (verified in Section 0
# inventory: 128 saved rounds, 326 GB total).
OUTROOT="/scratch/projects/secure-distributed-ml/results/resnet18_trajectory_redo_seed${SEED}_${SLURM_JOB_ID}"
ARTIFACT_DIR="/scratch/projects/secure-distributed-ml/results/1k_experiment/artifacts_fisher_seed${SEED}_300clients"
ATTACK_DIR="$OUTROOT/attacks"
mkdir -p "$ATTACK_DIR"

if [ ! -d "$ARTIFACT_DIR" ]; then
    echo "ERROR: ResNet-18 1k_experiment artefacts not at $ARTIFACT_DIR"
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
        --model resnet18 \
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

echo ""
echo "ResNet-18 trajectory redo finished at $(date)"
echo "JSONs in $ATTACK_DIR"
ls "$ATTACK_DIR" | wc -l
echo "(expecting 36)"
