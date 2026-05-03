#!/bin/bash
#SBATCH --job-name=adaguard-k-sweep
#SBATCH --output=/scratch/projects/secure-distributed-ml/logs/ksweep-%A_%a.out
#SBATCH --error=/scratch/projects/secure-distributed-ml/logs/ksweep-%A_%a.err
#SBATCH --partition=general-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maguir@oakland.edu
#SBATCH --array=0-14
# =============================================================
# RQ5 expansion: gradient-accumulation K sweep, multi-seed.
#
# Per professor's pre-submission ask: "You have one K value (K=4)
# tested, on one seed. You don't know if K=4 is actually a good
# default or just lucky. Need to test K = 1, 2, 4, 8, 16 with 3
# seeds each."
#
# Slurm array layout: 5 K values x 3 seeds = 15 Phase-1 retrains.
# Each retrain is ~6h on a single V100 (250 rounds x 30 active
# clients x ~5-10s gradient compute, with K-pass accumulation
# adding linearly to client compute when accum fires).
#
# Each task overrides the baseline config_t2_low.yaml with the
# specific K value (which keeps grad_accum_enabled=True and
# T2=0.3 so accumulation fires every round). This means the K
# sweep tests the FORCED-accumulation regime, not the default
# T2=0.7 dormant regime.
#
# After each Phase-1 retrain completes, the script also runs the
# 4x3 defence x attack matrix on the resulting round-249 artefact
# (just like the existing slurm_defence_sweep.sh) so each task
# produces both the training trajectory AND the attack outcomes.
#
# IMPORTANT: this is a heavy job. 15 retrains x ~6h each = ~90
# GPU-hours. Submit selectively:
#   sbatch hpc/slurm_k_sweep.sh                   # all 15
#   sbatch --array=0-4 hpc/slurm_k_sweep.sh       # K-sweep at seed 42 only
#   sbatch --array=0,3,6,9,12 hpc/slurm_k_sweep.sh  # K=1 across all 3 seeds
#
# Output: per task, one Phase-1 artefact dir + 12 attack JSONs at
#   /scratch/.../ksweep_K${K}_seed${SEED}_${SLURM_ARRAY_JOB_ID}/
# =============================================================

K_VALUES=(1 2 4 8 16)
SEEDS=(42 123 456)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
K_IDX=$((TASK_ID / 3))
SEED_IDX=$((TASK_ID % 3))
K_VAL="${K_VALUES[$K_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"

echo "K-sweep task $SLURM_ARRAY_JOB_ID.$SLURM_ARRAY_TASK_ID on $(hostname) at $(date)"
echo "  K     : $K_VAL  (idx=$K_IDX of 5)"
echo "  seed  : $SEED   (idx=$SEED_IDX of 3)"

module load Python/3.10.14
module load CUDA/12.4
source /projects/secure-distributed-ml/venv/bin/activate

# CIFAR-10 lives on /scratch (not in the project tree). Without this
# torchvision tries to re-download from its canonical URL on every
# task and fails with HTTP 503 -- root cause of the silent
# Phase-1-never-trains failure observed on 2026-04-29 (see
# CHANGELOG_REVIEW.md Round 4 and the smoke test in 2026-05-03).
export DATA_DIR=/scratch/projects/secure-distributed-ml/data

OUTROOT="/scratch/projects/secure-distributed-ml/results/ksweep_K${K_VAL}_seed${SEED}_${SLURM_ARRAY_JOB_ID}"
ARTIFACT_DIR="$OUTROOT/artifacts"
ATTACK_DIR="$OUTROOT/attacks"
mkdir -p "$ARTIFACT_DIR" "$ATTACK_DIR"

cd /projects/secure-distributed-ml/AdaGuard

# Step 1: Phase-1 retrain with the configured K value.
# Build a per-task YAML override on the fly so we don't accumulate
# 15 hand-edited config files in hpc/.
TMP_CONFIG="$OUTROOT/config_K${K_VAL}_seed${SEED}.yaml"
python -c "
import yaml
with open('hpc/config_t2_low.yaml') as f:
    cfg = yaml.safe_load(f)
cfg['grad_accum_K'] = $K_VAL
cfg['seed'] = $SEED
cfg['save_artifacts'] = True
with open('$TMP_CONFIG', 'w') as f:
    yaml.safe_dump(cfg, f)
print('wrote', '$TMP_CONFIG')
"

echo ""
echo "=== Phase-1 retrain ==="
echo "Config: $TMP_CONFIG"
stdbuf -oL -eL python -u run_headless.py \
    --config "$TMP_CONFIG" \
    --output "$OUTROOT/training.json" \
    --tb-dir "$OUTROOT/tb_logs" \
    --artifacts-dir "$ARTIFACT_DIR" \
    --seed "$SEED"

# Step 2: 4x3 defence x attack matrix on round 249 of the retrained
# trajectory. Mirrors slurm_defence_sweep.sh.
echo ""
echo "=== Phase-2 attack matrix on round 249 ==="
run_attack () {
    local attack="$1"; local iters="$2"; local defence="$3"
    local tag="${defence}_${attack}_b1_K${K_VAL}_seed${SEED}"
    echo "  ATTACK=$attack DEFENCE=$defence K=$K_VAL SEED=$SEED"
    stdbuf -oL -eL python -u tests/attack_sanity_check.py \
        --artifact-dir "$ARTIFACT_DIR" \
        --round 249 \
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

echo "K-sweep task finished at $(date)"
echo "Artifacts in $OUTROOT"
