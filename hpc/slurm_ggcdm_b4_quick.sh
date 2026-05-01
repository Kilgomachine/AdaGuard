#!/bin/bash
#SBATCH --job-name=adaguard-ggcdm-b4
#SBATCH --output=/scratch/projects/secure-distributed-ml/logs/ggcdm-b4-%j.out
#SBATCH --error=/scratch/projects/secure-distributed-ml/logs/ggcdm-b4-%j.err
#SBATCH --partition=general-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=2:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maguir@oakland.edu
# =============================================================
# RQ4 expansion (1/2): GGCDM at B=4 on the most-diverse client
# (client_106) across all 4 defences (V1 None, V2 FHE, V3 MaskCrypt,
# V4 AdaGuard-Fisher). Run for one seed at a time -- submit 3x
# (SEED=42, 123, 456) to complete the multi-seed coverage.
#
# Why this script exists: the existing Phase-2 B=4 measurements only
# cover GradInversion + GI-NAS, only on V1 (undefended). The
# diffusion-prior attack family (GGCDM) at B=4 is the gap the
# professor flagged in pre-submission review. This is the cheap
# fast-result complement to the broader B=4 multi-client expansion
# (slurm_b4_multiclient_expansion.sh).
#
# Usage:
#   SEED=42  ARTIFACT_DIR=/scratch/.../artifacts_none_seed42_300clients  sbatch hpc/slurm_ggcdm_b4_quick.sh
#   SEED=123 ARTIFACT_DIR=/scratch/.../artifacts_none_seed123_300clients sbatch hpc/slurm_ggcdm_b4_quick.sh
#   SEED=456 ARTIFACT_DIR=/scratch/.../artifacts_none_seed456_300clients sbatch hpc/slurm_ggcdm_b4_quick.sh
#
# Expected runtime per seed: ~30 min (4 defences x ~7 min/defence;
# GGCDM at B=4 is the same 100 iterations as B=1 since the diffusion
# prior dominates compute, just one more sample).
#
# Output: 4 JSONs per seed at
#   /scratch/.../v_ggcdm_b4_seed${SEED}_${SLURM_JOB_ID}/
#   ggcdm_b4_{none,fhe,maskcrypt,fisher}_seed${SEED}.json
# =============================================================

echo "GGCDM B=4 single-client job $SLURM_JOB_ID on $(hostname) at $(date)"

module load Python/3.10.14
module load CUDA/12.4
source /projects/secure-distributed-ml/venv/bin/activate

SEED="${SEED:-42}"
ARTIFACT_DIR="${ARTIFACT_DIR:-/scratch/projects/secure-distributed-ml/results/1k_experiment/artifacts_none_seed${SEED}_300clients}"
ROUND="${ROUND:-249}"
DEFENCE_PCT="${DEFENCE_PCT:-0.1}"
# Use auto-pick to get the most-diverse client of round 249. Defaults
# to client_106 in our existing measurements but auto-picks robust
# across seeds in case of artefact differences.
DIVERSE_MIN_UNIQUE="${DIVERSE_MIN_UNIQUE:-3}"
OUTDIR="/scratch/projects/secure-distributed-ml/results/v_ggcdm_b4_seed${SEED}_${SLURM_JOB_ID}"
mkdir -p "$OUTDIR"

cd /projects/secure-distributed-ml/AdaGuard

echo "Artifact dir         : $ARTIFACT_DIR"
echo "Round                : $ROUND"
echo "Phase-1 seed         : $SEED"
echo "Diverse min unique   : $DIVERSE_MIN_UNIQUE"
echo "Defence pct (HE/SHE) : $DEFENCE_PCT"
echo "Output dir           : $OUTDIR"
echo ""

run_one () {
    local defence="$1"
    local tag="ggcdm_b4_${defence}_seed${SEED}"
    echo "============================================================"
    echo "  ATTACK=ggcdm  DEFENCE=$defence  SEED=$SEED  B=4  diverse"
    echo "============================================================"
    stdbuf -oL -eL python -u tests/attack_sanity_check.py \
        --artifact-dir "$ARTIFACT_DIR" \
        --round "$ROUND" \
        --auto-pick-diverse "$DIVERSE_MIN_UNIQUE" \
        --attack ggcdm \
        --n-iter 100 \
        --batch-size 4 \
        --diverse-subset \
        --defence "$defence" \
        --defence-pct "$DEFENCE_PCT" \
        --variant paper \
        --out "$OUTDIR/${tag}.json"
    echo ""
}

for defence in none fhe maskcrypt fisher; do
    run_one "$defence"
done

echo "GGCDM B=4 single-client finished at $(date)"
echo "Artifacts in $OUTDIR"
echo ""
echo "Summary table:"
for j in "$OUTDIR"/*.json; do
    python -c "
import json, os
d = json.load(open(r'$j'))
psnr = d.get('psnr', float('nan'))
lpips = d.get('lpips') or 0.0
asr = (d.get('label_recovery') or {}).get('asr', 0.0)
print(f\"{os.path.basename(r'$j'):44s}  PSNR={psnr:7.2f} dB  LPIPS={lpips:.4f}  labelASR={asr:.3f}\")
"
done
