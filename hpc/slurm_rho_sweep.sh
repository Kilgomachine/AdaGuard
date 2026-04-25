#!/bin/bash
#SBATCH --job-name=adaguard-rho-sweep
#SBATCH --output=/scratch/projects/secure-distributed-ml/logs/rho-sweep-%j.out
#SBATCH --error=/scratch/projects/secure-distributed-ml/logs/rho-sweep-%j.err
#SBATCH --partition=general-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=2:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maguir@oakland.edu
# =============================================================
# Encryption-budget sensitivity sweep for V6 (AdaGuard-Fisher).
#
# Paper fixes rho = 10% throughout. This sweep characterises the
# privacy/utility response surface across {5, 10, 15, 20, 30}%
# so we can answer the obvious reviewer ask: "why 10%?"
#
# Single-seed (seed 42 baseline artifact), V6 only — V1 ignores
# rho, V2 is fixed at 100%, and V4's rho-curve is in MaskCrypt's
# own paper. Adds one new "S_rho" group to Table VII (sensitivity).
#
# 5 budgets x 3 attacks = 15 runs. GradInv 20k iters dominates
# (~5 min each); GGCDM 100 iters ~2 min; GI-NAS 2k iters ~1 min.
# Per-budget cost ~8 min wall, total ~40 min. Budget 2h for safety.
#
# Submit:  sbatch hpc/slurm_rho_sweep.sh
# =============================================================

echo "Rho-sweep job $SLURM_JOB_ID on $(hostname) at $(date)"

module load Python/3.10.14
module load CUDA/12.4
source /projects/secure-distributed-ml/venv/bin/activate

ARTIFACT_DIR="${ARTIFACT_DIR:-/scratch/projects/secure-distributed-ml/results/1k_experiment/artifacts_none_seed42_300clients}"
ROUND="${ROUND:-249}"
OUTDIR="/scratch/projects/secure-distributed-ml/results/rho_sweep_${SLURM_JOB_ID}"
mkdir -p "$OUTDIR"

cd /projects/secure-distributed-ml/AdaGuard

echo "Artifact dir : $ARTIFACT_DIR"
echo "Round        : $ROUND"
echo "Defence      : V6 (Fisher) only"
echo "Rho values   : 0.05, 0.10, 0.15, 0.20, 0.30"
echo ""

run_one () {
    local rho="$1"; local attack="$2"; local iters="$3"
    local rho_pct="$(python -c "print(int(${rho}*100))")"
    local tag="fisher_rho${rho_pct}_${attack}_b1"
    echo "============================================================"
    echo "  RHO=$rho  ATTACK=$attack  iters=$iters"
    echo "============================================================"
    stdbuf -oL -eL python -u tests/attack_sanity_check.py \
        --artifact-dir "$ARTIFACT_DIR" \
        --round "$ROUND" \
        --attack "$attack" \
        --n-iter "$iters" \
        --batch-size 1 \
        --defence fisher \
        --defence-pct "$rho" \
        --variant paper \
        --out "$OUTDIR/${tag}.json"
    echo ""
}

for rho in 0.05 0.10 0.15 0.20 0.30; do
    run_one "$rho" gradinversion 20000
    run_one "$rho" ggcdm         100
    run_one "$rho" gi_nas        2000
done

echo "Rho sweep finished at $(date)"
echo "Artifacts in $OUTDIR"
echo ""
echo "Summary table (rho vs PSNR per attack):"
for j in "$OUTDIR"/*.json; do
    python -c "
import json, os
d = json.load(open(r'$j'))
print(f\"{os.path.basename(r'$j'):45s}  PSNR={d.get('psnr', float('nan')):7.2f} dB  LPIPS={d.get('lpips', 0):.4f}  encPct={d.get('defence_meta',{}).get('pct_encrypted', 0)*100:5.1f}%\")
"
done
echo ""
echo "Next: integrate into Table VII as 'S_rho' group; update RQ5 takeaways."
