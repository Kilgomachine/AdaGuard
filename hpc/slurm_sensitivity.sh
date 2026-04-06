#!/bin/bash
#SBATCH --job-name=adaguard-study
#SBATCH --output=/scratch/projects/secure-distributed-ml/logs/study-%A_%a.out
#SBATCH --error=/scratch/projects/secure-distributed-ml/logs/study-%A_%a.err
#SBATCH --partition=general-long
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --array=0-71
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maguir@oakland.edu
# =============================================================
# AdaGuard Sensitivity & Viability Study
#
# 72 scenarios total:
#   0-59:  Case 1 — Sensitivity (S1.1 through S11.3)
#   60-71: Case 2 — Viability  (V1 through V12)
#
# Each array task runs one scenario via run_sensitivity.py --slurm-index.
#
# Prerequisites: Phase 1 training must be complete with --artifacts-dir
#
# Usage:
#   # Run all 72 scenarios
#   sbatch hpc/slurm_sensitivity.sh
#
#   # Run only sensitivity (first 60)
#   sbatch --array=0-59 hpc/slurm_sensitivity.sh
#
#   # Run only viability (last 12)
#   sbatch --array=60-71 hpc/slurm_sensitivity.sh
#
#   # Run specific scenarios
#   sbatch --array=0,5,10,60,64 hpc/slurm_sensitivity.sh
#
#   # Override artifacts dir
#   ARTIFACTS_DIR=/path/to/artifacts sbatch hpc/slurm_sensitivity.sh
# =============================================================

echo "Task $SLURM_ARRAY_TASK_ID of job $SLURM_ARRAY_JOB_ID on $(hostname) at $(date)"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ', ')"

module load Python/3.10.14
module load CUDA/12.4
source /projects/secure-distributed-ml/venv/bin/activate

export DATA_DIR="/scratch/projects/secure-distributed-ml/data"
export PYTHONUNBUFFERED=1
cd /projects/secure-distributed-ml/AdaGuard

# Configurable via environment variables
STRATEGY="${STRATEGY:-fisher}"
SEED="${SEED:-42}"
NC="${NUM_CLIENTS:-1000}"
ARTIFACTS="${ARTIFACTS_DIR:-/scratch/projects/secure-distributed-ml/results/1k_experiment/artifacts_${STRATEGY}_seed${SEED}_${NC}clients}"

RESULTS_DIR="/scratch/projects/secure-distributed-ml/results/study_${SLURM_ARRAY_JOB_ID}"
TB_DIR="/scratch/projects/secure-distributed-ml/results/tb_logs/study_${SLURM_ARRAY_JOB_ID}"
mkdir -p "$RESULTS_DIR" "$TB_DIR"

echo "Slurm index: $SLURM_ARRAY_TASK_ID"
echo "Artifacts:   $ARTIFACTS"
echo "Results:     $RESULTS_DIR"
echo "TB logs:     $TB_DIR"

# Build rounds argument if specified
ROUNDS_ARG=""
if [ -n "$ROUNDS" ]; then
    ROUNDS_ARG="--rounds $ROUNDS"
fi

# Build attacks flag
ATTACKS_ARG=""
if [ "$NO_ATTACKS" = "1" ]; then
    ATTACKS_ARG="--no-attacks"
fi

stdbuf -oL -eL python -u run_sensitivity.py \
    --slurm-index "$SLURM_ARRAY_TASK_ID" \
    --artifacts-dir "$ARTIFACTS" \
    --config hpc/config_scenario.yaml \
    --output-dir "$RESULTS_DIR" \
    --tb-dir "$TB_DIR" \
    $ROUNDS_ARG \
    $ATTACKS_ARG

echo "Task $SLURM_ARRAY_TASK_ID completed at $(date)"
