#!/bin/bash
#SBATCH --job-name=adaguard-ls-pred
#SBATCH --output=/scratch/projects/secure-distributed-ml/logs/lspred-%A_%a.out
#SBATCH --error=/scratch/projects/secure-distributed-ml/logs/lspred-%A_%a.err
#SBATCH --partition=general-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=4:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maguir@oakland.edu
#SBATCH --array=0-14
# =============================================================
# LeakScore predictive validation against attack outcomes.
#
# Per professor's pre-submission ask: "You build LeakScore as a
# vulnerability indicator but never validate that high LeakScore
# correlates with successful inversion."
#
# This script runs all 3 attack families on the UNDEFENDED V1
# Phase-1 artefacts at multiple round indices and across multiple
# clients spanning the diversity histogram. Each cell produces an
# attack JSON whose PSNR / LPIPS / label-recovery ASR can then be
# correlated against the per-cell LeakScore the simulator already
# saved during Phase-1 training.
#
# Slurm array layout: 5 rounds x 3 seeds = 15 array tasks.
# Each task picks one (round, seed) pair and runs all 3 attacks
# x 5 clients = 15 attack runs internally (~1 h per task).
#
# Output: ~75 JSONs per seed (5 rounds x 5 clients x 3 attacks)
# under /scratch/.../lspred_seed${SEED}_round${ROUND}_${SLURM_ARRAY_JOB_ID}/
#
# Once results land, the integration script (TODO: write
# scripts/paper/build_leakscore_predictive_correlation.py) reads
# the per-cell attack outcomes, joins on the per-cell LeakScore from
# the matching Phase-1 trajectory JSONs, and produces:
#   - Pearson / Spearman correlation
#   - scatter plot (LeakScore on x, PSNR on y, color by client)
#   - paragraph + table for sec:limit-leakscore-validation
# =============================================================

ROUNDS=(50 100 150 200 249)
SEEDS=(42 123 456)
# Client IDs spanning the diversity histogram. EDIT THESE based on
# --list-clients output for the round 249 artefacts. Defaults are
# placeholders; verify against your scratch.
CLIENT_IDS=(106 50 100 150 200)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
ROUND_IDX=$((TASK_ID / 3))
SEED_IDX=$((TASK_ID % 3))
ROUND="${ROUNDS[$ROUND_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"

echo "LeakScore predictive task $SLURM_ARRAY_JOB_ID.$SLURM_ARRAY_TASK_ID on $(hostname) at $(date)"
echo "  round : $ROUND  (idx=$ROUND_IDX of 5)"
echo "  seed  : $SEED   (idx=$SEED_IDX of 3)"

module load Python/3.10.14
module load CUDA/12.4
source /projects/secure-distributed-ml/venv/bin/activate

ARTIFACT_DIR="${ARTIFACT_DIR:-/scratch/projects/secure-distributed-ml/results/1k_experiment/artifacts_none_seed${SEED}_300clients}"
OUTDIR="/scratch/projects/secure-distributed-ml/results/lspred_seed${SEED}_round${ROUND}_${SLURM_ARRAY_JOB_ID}"
mkdir -p "$OUTDIR"

cd /projects/secure-distributed-ml/AdaGuard

run_one () {
    local client_id="$1"; local attack="$2"; local iters="$3"
    local tag="lspred_${attack}_seed${SEED}_round${ROUND}_client${client_id}"
    echo "============================================================"
    echo "  ATTACK=$attack  ROUND=$ROUND  SEED=$SEED  CLIENT=$client_id  B=1  (V1 undefended)"
    echo "============================================================"
    stdbuf -oL -eL python -u tests/attack_sanity_check.py \
        --artifact-dir "$ARTIFACT_DIR" \
        --round "$ROUND" \
        --client-id "$client_id" \
        --attack "$attack" \
        --n-iter "$iters" \
        --batch-size 1 \
        --defence none \
        --variant paper \
        --out "$OUTDIR/${tag}.json"
    echo ""
}

for client_id in "${CLIENT_IDS[@]}"; do
    run_one "$client_id" gradinversion 20000
    run_one "$client_id" ggcdm         100
    run_one "$client_id" gi_nas        2000
done

echo "LeakScore predictive task finished at $(date)"
echo "Artifacts in $OUTDIR (5 clients x 3 attacks = 15 cells)"
