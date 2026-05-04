#!/bin/bash
#SBATCH --job-name=adaguard-b-sweep
#SBATCH --output=/scratch/projects/secure-distributed-ml/logs/bsweep-%A_%a.out
#SBATCH --error=/scratch/projects/secure-distributed-ml/logs/bsweep-%A_%a.err
#SBATCH --partition=general-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maguir@oakland.edu
#SBATCH --array=0-14
# =============================================================
# RQ5 expansion: client batch-size B sweep, multi-seed.
#
# Per professor's pre-submission ask: "Batch size is fixed at B=16.
# You don't know if AdaGuard still helps at larger batches (B=64,
# 128) where natural averaging is already strong. Need to test B =
# 8, 16, 32, 64, 128."
#
# Slurm array layout: 5 B values x 3 seeds = 15 Phase-1 retrains.
# Each retrain is ~6h on a single V100; total ~90 GPU-hours.
#
# Each task overrides the baseline config_1k.yaml with the specific
# client_batch_size value; everything else (T1, T2, encryption,
# focus_layers, ...) inherits from the baseline. After each
# Phase-1 retrain, the 4x3 defence x attack matrix is run on the
# round-249 artefact, mirroring slurm_k_sweep.sh.
#
# IMPORTANT: heavy job, like K sweep. Submit selectively:
#   sbatch hpc/slurm_b_sweep.sh                   # all 15
#   sbatch --array=0-4 hpc/slurm_b_sweep.sh       # B-sweep at seed 42 only
# =============================================================

B_VALUES=(8 16 32 64 128)
SEEDS=(42 123 456)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
B_IDX=$((TASK_ID / 3))
SEED_IDX=$((TASK_ID % 3))
B_VAL="${B_VALUES[$B_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"

echo "B-sweep task $SLURM_ARRAY_JOB_ID.$SLURM_ARRAY_TASK_ID on $(hostname) at $(date)"
echo "  B     : $B_VAL  (idx=$B_IDX of 5)"
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

OUTROOT="/scratch/projects/secure-distributed-ml/results/bsweep_B${B_VAL}_seed${SEED}_${SLURM_ARRAY_JOB_ID}"
ARTIFACT_DIR="$OUTROOT/artifacts"
ATTACK_DIR="$OUTROOT/attacks"
mkdir -p "$ARTIFACT_DIR" "$ATTACK_DIR"

cd /projects/secure-distributed-ml/AdaGuard

# Phase-1 retrain with overridden client_batch_size. Use config_t2_low.yaml
# (300 clients) instead of config_1k.yaml (1000 clients) -- the 1000-client
# config triggers a silent run_headless exit after round 1 (observed in
# 182448 array tasks 0-7, where Phase-1 aggregated round 1 then exited
# cleanly without raising; Phase-2 then crashed on missing round_249).
# config_t2_low.yaml matches what K-sweep, 1k_experiment, and the paper
# main results all use, so the per-task wall and disk budget are also
# proven within the 12h slurm allocation.
TMP_CONFIG="$OUTROOT/config_B${B_VAL}_seed${SEED}.yaml"
python -c "
import yaml
with open('hpc/config_t2_low.yaml') as f:
    cfg = yaml.safe_load(f)
cfg['client_batch_size'] = $B_VAL
cfg['seed'] = $SEED
cfg['save_artifacts'] = True
with open('$TMP_CONFIG', 'w') as f:
    yaml.safe_dump(cfg, f)
print('wrote', '$TMP_CONFIG')
"

echo ""
echo "=== Phase-1 retrain (B=$B_VAL) ==="
echo "Config: $TMP_CONFIG"
stdbuf -oL -eL python -u run_headless.py \
    --config "$TMP_CONFIG" \
    --output "$OUTROOT/training.json" \
    --tb-dir "$OUTROOT/tb_logs" \
    --artifacts-dir "$ARTIFACT_DIR" \
    --seed "$SEED"

# Phase-2 attack matrix on round 249.
# IMPORTANT: at larger B (e.g. 64, 128) the attack iteration counts
# may need adjustment - GradInversion at B=128 is mathematically
# harder and the same 20k iterations may not converge to the
# baseline-matching number. We log results regardless; the paper text
# can note any cells that need follow-up.
echo ""
echo "=== Phase-2 attack matrix on round 249 ==="
# Use --batch-size matching the actual saved-batch B for fidelity to
# the trained regime. attack_sanity_check.py will recompute the
# gradient at the requested B from the saved local model.
ATTACK_B=$B_VAL
run_attack () {
    local attack="$1"; local iters="$2"; local defence="$3"
    local tag="${defence}_${attack}_b${ATTACK_B}_B${B_VAL}_seed${SEED}"
    echo "  ATTACK=$attack DEFENCE=$defence B=$B_VAL SEED=$SEED attack_B=$ATTACK_B"
    stdbuf -oL -eL python -u tests/attack_sanity_check.py \
        --artifact-dir "$ARTIFACT_DIR" \
        --round 249 \
        --attack "$attack" \
        --n-iter "$iters" \
        --batch-size "$ATTACK_B" \
        --diverse-subset \
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

echo "B-sweep task finished at $(date)"
echo "Artifacts in $OUTROOT"
