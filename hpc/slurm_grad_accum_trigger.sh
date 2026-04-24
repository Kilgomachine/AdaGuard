#!/bin/bash
#SBATCH --job-name=adaguard-gradaccum
#SBATCH --output=/scratch/projects/secure-distributed-ml/logs/gradaccum-%j.out
#SBATCH --error=/scratch/projects/secure-distributed-ml/logs/gradaccum-%j.err
#SBATCH --partition=general-long
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maguir@oakland.edu
# =============================================================
# AdaGuard — Gradient-Accumulation Trigger Experiment
#
# Re-runs Phase-1 training with T2 lowered from 0.7 to 0.3 so the
# gradient-accumulation pathway actually engages during training.
# Pairs with the existing T2=0.7 fisher seed=42 baseline so we can
# report a V6+accum row alongside the V6 row in the paper's
# defence matrix.
#
# Single-seed (seed=42) only — that is enough to characterise
# the accumulation-vs-no-accumulation delta; if the delta is
# meaningful we'll spend the multi-seed budget afterwards.
#
# Submit:  sbatch hpc/slurm_grad_accum_trigger.sh
# =============================================================

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ', ')"

module load Python/3.10.14
module load CUDA/12.4
source /projects/secure-distributed-ml/venv/bin/activate

export DATA_DIR="/scratch/projects/secure-distributed-ml/data"
export PYTHONUNBUFFERED=1
cd /projects/secure-distributed-ml/AdaGuard

SEED=42
STRATEGY=fisher

RESULTS_DIR="/scratch/projects/secure-distributed-ml/results/gradaccum_${SLURM_JOB_ID}"
ARTIFACTS_DIR="${RESULTS_DIR}/artifacts_${STRATEGY}_seed${SEED}_300clients_T2lo"
mkdir -p "$RESULTS_DIR"

OUTPUT="${RESULTS_DIR}/${STRATEGY}_seed${SEED}_T2lo.json"

echo "Seed:       $SEED"
echo "Strategy:   $STRATEGY (T2=0.3, accumulation will fire)"
echo "Artifacts:  $ARTIFACTS_DIR"
echo "Output:     $OUTPUT"

stdbuf -oL -eL python -u run_headless.py \
    --config hpc/config_t2_low.yaml \
    --strategy "$STRATEGY" \
    --seed "$SEED" \
    --output "$OUTPUT" \
    --artifacts-dir "$ARTIFACTS_DIR"

echo "Job completed at $(date)"
echo "Next: replay this artifact under V6 to get the accum-on numbers,"
echo "      then diff against the existing V6 baseline (T2=0.7)."
