#!/bin/bash
#SBATCH --job-name=adaguard-llg
#SBATCH --output=/scratch/projects/secure-distributed-ml/logs/llg-%j.out
#SBATCH --error=/scratch/projects/secure-distributed-ml/logs/llg-%j.err
#SBATCH --partition=general-long
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maguir@oakland.edu
# =============================================================
# AdaGuard label-recovery (iDLG / LLG+) sweep across V1/V2/V4/V6.
#
# The Phase-1 pipeline writes the *raw* pre-defence gradient to every
# client artifact regardless of encryption strategy (see
# adaguard/federation/simulator.py:455 — defences only affect what the
# SERVER aggregates, not what gets saved). So we pull one raw artifact
# dir and re-apply each defence at sweep-time to simulate the
# attacker's view.
#
# V1 = no defence (raw gradient).
# V2 = full HE (every gradient zeroed).
# V4 = MaskCrypt top-k by |grad * (old - new)|, k = 10% of params.
# V6 = AdaGuard Fisher top-k by g^2, k = 10% of params.
#
# Usage:
#   sbatch hpc/slurm_label_recovery.sh
#   ROUND=50 sbatch hpc/slurm_label_recovery.sh     # earlier snapshot
#   ENC_PCT=0.20 sbatch hpc/slurm_label_recovery.sh # 20% encryption budget
# =============================================================

echo "LLG sweep job $SLURM_JOB_ID on $(hostname) at $(date)"

module load Python/3.10.14
source /projects/secure-distributed-ml/venv/bin/activate

ROUND="${ROUND:-249}"
ENC_PCT="${ENC_PCT:-0.10}"
RESULTS_ROOT="/scratch/projects/secure-distributed-ml/results/1k_experiment"
ARTIFACT_DIR="${ARTIFACT_DIR:-$RESULTS_ROOT/artifacts_none_seed42_300clients}"
OUTDIR="/scratch/projects/secure-distributed-ml/results/llg_${SLURM_JOB_ID}"
mkdir -p "$OUTDIR"

cd /projects/secure-distributed-ml/AdaGuard

echo "Raw artifact dir : $ARTIFACT_DIR"
echo "Round            : $ROUND"
echo "Encryption budget: $ENC_PCT"
echo ""

stdbuf -oL -eL python -u tests/label_recovery_sweep.py \
    --round "$ROUND" \
    --num-classes 10 \
    --artifact-dir "$ARTIFACT_DIR" \
    --scenario "V1=none" \
    --scenario "V2=full" \
    --scenario "V4=maskcrypt:$ENC_PCT" \
    --scenario "V6=fisher:$ENC_PCT" \
    --out "$OUTDIR/label_recovery_round${ROUND}_pct${ENC_PCT}.json"

echo "LLG sweep finished at $(date)"
echo "Artifacts in $OUTDIR"
