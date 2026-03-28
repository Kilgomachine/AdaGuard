#!/bin/bash
#SBATCH --job-name=adaguard-1k
#SBATCH --output=/scratch/projects/secure-distributed-ml/logs/1k-%A_%a.out
#SBATCH --error=/scratch/projects/secure-distributed-ml/logs/1k-%A_%a.err
#SBATCH --partition=general-long
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --array=0-11
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maguir@oakland.edu
# =============================================================
# AdaGuard — 1000-Client Strategy Comparison (4 strategies x 3 seeds)
#
# Array layout (12 jobs):
#   0-3:  seed=42   | none, fisher, maskcrypt, full
#   4-7:  seed=123  | none, fisher, maskcrypt, full
#   8-11: seed=456  | none, fisher, maskcrypt, full
#
# Override client count:
#   NUM_CLIENTS=2000 CLIENTS_PER_ROUND=100 sbatch hpc/slurm_1k_clients.sh
#
# Submit: sbatch hpc/slurm_1k_clients.sh
# =============================================================

echo "Task $SLURM_ARRAY_TASK_ID of job $SLURM_ARRAY_JOB_ID on $(hostname) at $(date)"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ', ')"

module load Python/3.10.14
module load CUDA/12.4
source /projects/secure-distributed-ml/venv/bin/activate

export DATA_DIR="/scratch/projects/secure-distributed-ml/data"
export PYTHONUNBUFFERED=1
cd /projects/secure-distributed-ml/AdaGuard

# Map array index to seed and strategy
SEEDS=(42 42 42 42 123 123 123 123 456 456 456 456)
STRATEGIES=(none fisher maskcrypt full none fisher maskcrypt full none fisher maskcrypt full)

SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
STRATEGY="${STRATEGIES[$SLURM_ARRAY_TASK_ID]}"

# Allow environment variable overrides for client count
NC="${NUM_CLIENTS:-1000}"
CPR="${CLIENTS_PER_ROUND:-100}"
NR="${NUM_ROUNDS:-250}"

# Stable directories — do NOT use $SLURM_ARRAY_JOB_ID here, so checkpoints
# persist across resubmissions and training can resume after job restarts.
RESULTS_DIR="/scratch/projects/secure-distributed-ml/results/1k_experiment"
LOGS_DIR="/scratch/projects/secure-distributed-ml/logs"
mkdir -p "$RESULTS_DIR" "$LOGS_DIR"

OUTPUT="${RESULTS_DIR}/${STRATEGY}_seed${SEED}_${NC}clients.json"
LOGFILE="${LOGS_DIR}/adaguard_${STRATEGY}_seed${SEED}.log"

echo "Seed: $SEED | Strategy: $STRATEGY | Clients: $NC | Per-round: $CPR | Rounds: $NR"
echo "Output: $OUTPUT"
echo "Log: $LOGFILE"
echo "Job ID: $SLURM_ARRAY_JOB_ID, Task: $SLURM_ARRAY_TASK_ID"

CKPT_DIR="${RESULTS_DIR}/checkpoints_${STRATEGY}_seed${SEED}_${NC}clients"
TB_DIR="/scratch/projects/secure-distributed-ml/results/tb_logs"
ARTIFACTS_DIR="${RESULTS_DIR}/artifacts_${STRATEGY}_seed${SEED}_${NC}clients"

# --defer-attacks: train fast, save client data, run attacks later via run_scenario.py
# --log: persistent log file (survives SSH disconnects — just cat $LOGFILE)
# --resume-dir: auto-checkpoint after each round; auto-resumes if job is restarted
# --artifacts-dir: save per-client artifacts for Phase 2 scenario evaluation
stdbuf -oL -eL python -u run_headless.py \
    --config hpc/config_1k.yaml \
    --strategy "$STRATEGY" \
    --seed "$SEED" \
    --num-clients "$NC" \
    --clients-per-round "$CPR" \
    --num-rounds "$NR" \
    --pretrain-epochs 0 \
    --defer-attacks \
    --resume-dir "$CKPT_DIR" \
    --tb-dir "$TB_DIR" \
    --artifacts-dir "$ARTIFACTS_DIR" \
    --log "$LOGFILE" \
    --output "$OUTPUT"

echo "Task $SLURM_ARRAY_TASK_ID ($STRATEGY, seed=$SEED, ${NC} clients) completed at $(date)"
echo ""
echo "To run Phase 2 scenarios on saved artifacts:"
echo "  python run_scenario.py --artifacts-dir $ARTIFACTS_DIR --scenario adaguard_high --config hpc/config_scenario.yaml --output results.json"
echo "  sbatch hpc/slurm_scenarios.sh   # Run all 11 scenarios"
echo ""
echo "To reuse this training without retraining:"
echo "  python run_headless.py --load-training ${RESULTS_DIR}/snapshot_${STRATEGY}_seed${SEED}_${NC}clients/training_snapshot.pt --config hpc/config_1k.yaml --output NEW_OUTPUT.json"
