#!/bin/bash
#SBATCH --job-name=adaguard-t1-rho
#SBATCH --output=/scratch/projects/secure-distributed-ml/logs/t1rho-%j.out
#SBATCH --error=/scratch/projects/secure-distributed-ml/logs/t1rho-%j.err
#SBATCH --partition=general-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=8:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maguir@oakland.edu
# =============================================================
# RQ5 expansion: joint T1 x rho heatmap.
#
# The professor flagged that our existing sensitivity sweep moves one
# parameter at a time and so cannot detect interaction effects between
# T1 (controller threshold for encryption-fires) and rho (encryption
# fraction). This script runs a 5x5 grid:
#   T1  in {0.10, 0.20, 0.30, 0.40, 0.50}
#   rho in {0.05, 0.10, 0.15, 0.20, 0.30}
# x 3 attacks (GradInv, GGCDM, GI-NAS) = 75 attack runs per seed.
#
# CONTROLLER LOGIC NOTE
# ---------------------
# In the lightweight operating mode that the headline matrix
# evaluates, the controller is bypassed and rho=0.10 is used directly.
# To exercise the (T1, rho) interaction, this sweep manually computes
# the controller's output given the cached round-249 LeakScore
# (~0.388 multi-seed) and applies the resulting effective encryption
# fraction:
#   effective_pct = rho * (LS - T1) / (T2 - T1)  if T1 <= LS < T2
#   effective_pct = 0                            if LS < T1
#   effective_pct = rho + (1 - rho) * (LS - T2) / (1 - T2)  if LS >= T2
# T2 fixed at 0.7 (default). The sweep therefore reports the
# (T1, rho) -> effective_pct -> attack_outcome chain.
#
# Single-seed by default to keep the matrix tractable; submit with
# different SEED env vars to extend to multi-seed.
#
# Expected runtime: 75 attack runs at ~5 min average -> ~6 hours.
# 8h walltime budgeted for safety.
#
# Output: 75 JSONs per seed under
#   /scratch/.../t1rho_heatmap_seed${SEED}_${SLURM_JOB_ID}/
#   t1{T1}_rho{rho}_{attack}_seed${SEED}.json
# =============================================================

echo "T1 x rho heatmap job $SLURM_JOB_ID on $(hostname) at $(date)"

module load Python/3.10.14
module load CUDA/12.4
source /projects/secure-distributed-ml/venv/bin/activate

SEED="${SEED:-42}"
ARTIFACT_DIR="${ARTIFACT_DIR:-/scratch/projects/secure-distributed-ml/results/1k_experiment/artifacts_none_seed${SEED}_300clients}"
ROUND="${ROUND:-249}"
T2="${T2:-0.7}"
# Cached round-249 LeakScore (multi-seed mean ~0.388; single-seed
# values 0.3850 / 0.3899 / 0.3885 across seeds 42/123/456 - see
# Fig. fig:leakscore-trajectory). Override per-seed if desired.
LEAKSCORE="${LEAKSCORE:-0.388}"
OUTDIR="/scratch/projects/secure-distributed-ml/results/t1rho_heatmap_seed${SEED}_${SLURM_JOB_ID}"
mkdir -p "$OUTDIR"

cd /projects/secure-distributed-ml/AdaGuard

T1_VALUES=(0.10 0.20 0.30 0.40 0.50)
RHO_VALUES=(0.05 0.10 0.15 0.20 0.30)

# Compute effective_pct given (T1, rho, T2, LeakScore) using the
# AdaptiveEncryptionController formula. This calls the actual code
# path so the heatmap reflects the real controller, not a manual
# transcription.
compute_effective_pct () {
    local t1="$1"; local rho="$2"
    python -c "
import sys
sys.path.insert(0, '/projects/secure-distributed-ml/AdaGuard')
from adaguard.encryption.controller import AdaptiveEncryptionController
ctrl = AdaptiveEncryptionController(T1=$t1, T2=$T2, base_encrypt_pct=$rho)
policy = ctrl.decide($LEAKSCORE)
print(f'{policy.encrypt_pct:.6f}')
"
}

run_one () {
    local t1="$1"; local rho="$2"; local attack="$3"; local iters="$4"
    local effective_pct
    effective_pct=$(compute_effective_pct "$t1" "$rho")
    local tag="t1${t1}_rho${rho}_${attack}_seed${SEED}"
    echo "============================================================"
    echo "  T1=$t1 RHO=$rho LS=$LEAKSCORE -> effective_pct=$effective_pct"
    echo "  ATTACK=$attack  SEED=$SEED  B=1"
    echo "============================================================"
    if [ "$effective_pct" = "0.000000" ] || [ "$effective_pct" = "0.0" ]; then
        # Encryption did not fire at this (T1, rho). Record a
        # "no encryption" cell so the heatmap has the entry, then skip
        # the attack (it would just rerun V1 baseline).
        cat > "$OUTDIR/${tag}.json" <<EOF
{
  "attack": "$attack",
  "n_iter": 0,
  "skipped": true,
  "skip_reason": "controller produced encrypt_pct=0 at T1=$t1 rho=$rho LS=$LEAKSCORE",
  "t1": $t1,
  "rho": $rho,
  "effective_pct": 0.0,
  "leakscore": $LEAKSCORE
}
EOF
        echo "  (skipped: encryption did not fire)"
        return
    fi
    stdbuf -oL -eL python -u tests/attack_sanity_check.py \
        --artifact-dir "$ARTIFACT_DIR" \
        --round "$ROUND" \
        --attack "$attack" \
        --n-iter "$iters" \
        --batch-size 1 \
        --defence fisher \
        --defence-pct "$effective_pct" \
        --variant paper \
        --out "$OUTDIR/${tag}.json"
    # Annotate the JSON with the t1/rho/leakscore that produced it.
    python -c "
import json
d = json.load(open(r'$OUTDIR/${tag}.json'))
d['t1'] = $t1
d['rho'] = $rho
d['effective_pct'] = $effective_pct
d['leakscore'] = $LEAKSCORE
json.dump(d, open(r'$OUTDIR/${tag}.json', 'w'), indent=2)
"
    echo ""
}

for t1 in "${T1_VALUES[@]}"; do
    for rho in "${RHO_VALUES[@]}"; do
        run_one "$t1" "$rho" gradinversion 20000
        run_one "$t1" "$rho" ggcdm         100
        run_one "$t1" "$rho" gi_nas        2000
    done
done

echo "T1 x rho heatmap finished at $(date)"
echo "Artifacts in $OUTDIR (75 cells: 5 T1 x 5 rho x 3 attacks)"
