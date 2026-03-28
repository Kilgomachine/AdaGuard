#!/bin/bash
#SBATCH --job-name=adaguard-scenario
#SBATCH --output=/scratch/projects/secure-distributed-ml/logs/scenario-%A_%a.out
#SBATCH --error=/scratch/projects/secure-distributed-ml/logs/scenario-%A_%a.err
#SBATCH --partition=general-long
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --array=0-10
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maguir@oakland.edu
# =============================================================
# AdaGuard Phase 2: Post-Training Scenario Evaluation
#
# Array layout (11 jobs, one per scenario):
#   0: no_encryption       6: label_only
#   1: full_he             7: entropy_only
#   2: maskcrypt           8: empirical_only
#   3: adaguard_high       9: 2label_1entropy
#   4: adaguard_mid       10: 1label_2entropy
#   5: adaguard_low
#
# Prerequisites: Phase 1 training must be complete with --artifacts-dir
#
# Usage:
#   # Run all 11 scenarios (default: fisher strategy, seed 42)
#   sbatch hpc/slurm_scenarios.sh
#
#   # Override artifacts dir and strategy
#   ARTIFACTS_DIR=/path/to/artifacts STRATEGY=fisher SEED=42 sbatch hpc/slurm_scenarios.sh
#
#   # Run specific rounds only
#   ROUNDS="0,50,100,200,249" sbatch hpc/slurm_scenarios.sh
# =============================================================

echo "Task $SLURM_ARRAY_TASK_ID of job $SLURM_ARRAY_JOB_ID on $(hostname) at $(date)"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ', ')"

module load Python/3.10.14
module load CUDA/12.4
source /projects/secure-distributed-ml/venv/bin/activate

export DATA_DIR="/scratch/projects/secure-distributed-ml/data"
export PYTHONUNBUFFERED=1
cd /projects/secure-distributed-ml/AdaGuard

# Map array index to scenario name
SCENARIOS=(
    no_encryption
    full_he
    maskcrypt
    adaguard_high
    adaguard_mid
    adaguard_low
    label_only
    entropy_only
    empirical_only
    2label_1entropy
    1label_2entropy
)

SCENARIO="${SCENARIOS[$SLURM_ARRAY_TASK_ID]}"

# Configurable via environment variables
STRATEGY="${STRATEGY:-fisher}"
SEED="${SEED:-42}"
NC="${NUM_CLIENTS:-1000}"
ARTIFACTS="${ARTIFACTS_DIR:-/scratch/projects/secure-distributed-ml/results/1k_experiment/artifacts_${STRATEGY}_seed${SEED}_${NC}clients}"

RESULTS_DIR="/scratch/projects/secure-distributed-ml/results/scenarios"
LOGS_DIR="/scratch/projects/secure-distributed-ml/logs"
TB_DIR="/scratch/projects/secure-distributed-ml/results/tb_logs"
mkdir -p "$RESULTS_DIR" "$LOGS_DIR" "$TB_DIR"

OUTPUT="${RESULTS_DIR}/${SCENARIO}_${STRATEGY}_seed${SEED}.json"
LOGFILE="${LOGS_DIR}/scenario_${SCENARIO}_${STRATEGY}_seed${SEED}.log"

echo "Scenario:  $SCENARIO"
echo "Strategy:  $STRATEGY"
echo "Seed:      $SEED"
echo "Artifacts: $ARTIFACTS"
echo "Output:    $OUTPUT"
echo "Log:       $LOGFILE"

# Build rounds argument if specified
ROUNDS_ARG=""
if [ -n "$ROUNDS" ]; then
    ROUNDS_ARG="--rounds $ROUNDS"
fi

stdbuf -oL -eL python -u run_scenario.py \
    --artifacts-dir "$ARTIFACTS" \
    --scenario "$SCENARIO" \
    --config hpc/config_scenario.yaml \
    --tb-dir "$TB_DIR" \
    --log "$LOGFILE" \
    --output "$OUTPUT" \
    $ROUNDS_ARG

echo "Scenario $SCENARIO completed at $(date)"
