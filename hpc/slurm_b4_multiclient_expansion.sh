#!/bin/bash
#SBATCH --job-name=adaguard-b4-multi
#SBATCH --output=/scratch/projects/secure-distributed-ml/logs/b4multi-%A_%a.out
#SBATCH --error=/scratch/projects/secure-distributed-ml/logs/b4multi-%A_%a.err
#SBATCH --partition=general-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=4:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maguir@oakland.edu
#SBATCH --array=0-14
# =============================================================
# RQ4 expansion (2/2): multi-client B=4 sweep.
#
# Tests AdaGuard's defence positioning at B=4 across multiple clients
# spanning the diversity histogram (1-3 unique labels per batch).
# Complements the single-client GGCDM B=4 sweep
# (slurm_ggcdm_b4_quick.sh) by adding (a) GradInversion + GI-NAS at
# B=4 across all 4 defences, (b) all three attacks across additional
# clients beyond client_106.
#
# Slurm array layout: 5 client_ids x 3 seeds = 15 array tasks.
# Each task runs 3 attacks x 4 defences = 12 attack runs internally
# (~1.5 h per task). Total wall-clock: gated by Matilda's array-task
# concurrency. With 8 concurrent slots, ~3 hours; serial, ~22 hours.
#
# CLIENT_IDS to use: tweak this list to span the diversity histogram
# (1-3 unique labels per batch). Default below picks 5 client_id
# slots; verify they exist in YOUR Phase-1 artefacts before running:
#   python tests/attack_sanity_check.py --list-clients --artifact-dir <dir> --round 249
# Then edit CLIENT_IDS below to the 5 you want to evaluate.
#
# Usage:
#   sbatch hpc/slurm_b4_multiclient_expansion.sh
#   (each array task picks one (client, seed) pair from the list)
# =============================================================

# Per-seed client_id lists picked to span class-combination diversity
# within the 3-unique-label stratum at round 249. Round 4 used a
# single CLIENT_IDS list (chosen from seed 42's saved active set),
# which produced empty output for seeds 123/456 because their
# saved-active client samples differ from seed 42's --- per-seed
# Phase-1 random sampling means the same global client id is NOT
# necessarily in every seed's saved round-249 active set.
#
# Per-seed lists below were picked via --list-clients on each seed's
# artefact dir at round 249, then chosen for maximum class-triple
# spread. Combined across seeds we cover all 10 CIFAR-10 classes.
#
#   seed 42:  client_106 {1,5,6}, client_13 {0,1,9}, client_185 {2,5,7},
#             client_173 {3,6,8}, client_222 {0,3,7}
#             (covers classes {0,1,2,3,5,6,7,9})
#   seed 123: client_107 {5,8,9}, client_114 {1,2,6}, client_125 {4,5,9},
#             client_158 {1,3,8}, client_159 {0,8,9}
#             (covers classes {0,1,2,3,4,5,6,8,9})
#   seed 456: client_0 {0,1,8}, client_108 {2,5,9}, client_112 {2,7,8},
#             client_113 {2,6,9}, client_120 {5,6,9}
#             (covers classes {0,1,2,5,6,7,8,9})
CLIENT_IDS_42=(106 13 185 173 222)
CLIENT_IDS_123=(107 114 125 158 159)
CLIENT_IDS_456=(0 108 112 113 120)
SEEDS=(42 123 456)

# Decode SLURM_ARRAY_TASK_ID into (client_idx, seed_idx) and pick
# the client_id from the seed-specific list (5 clients per seed).
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
N_CLIENTS=5
CLIENT_IDX=$((TASK_ID / 3))
SEED_IDX=$((TASK_ID % 3))
SEED="${SEEDS[$SEED_IDX]}"
case "$SEED" in
    42)  CLIENT_ID="${CLIENT_IDS_42[$CLIENT_IDX]}"  ;;
    123) CLIENT_ID="${CLIENT_IDS_123[$CLIENT_IDX]}" ;;
    456) CLIENT_ID="${CLIENT_IDS_456[$CLIENT_IDX]}" ;;
    *)   echo "ERROR: no CLIENT_IDS list for seed $SEED" >&2; exit 1 ;;
esac

echo "B=4 multi-client task $SLURM_ARRAY_JOB_ID.$SLURM_ARRAY_TASK_ID on $(hostname) at $(date)"
echo "  client_id : $CLIENT_ID  (idx=$CLIENT_IDX of $N_CLIENTS)"
echo "  seed      : $SEED       (idx=$SEED_IDX of 3)"

module load Python/3.10.14
module load CUDA/12.4
source /projects/secure-distributed-ml/venv/bin/activate

ARTIFACT_DIR="${ARTIFACT_DIR:-/scratch/projects/secure-distributed-ml/results/1k_experiment/artifacts_none_seed${SEED}_300clients}"
ROUND="${ROUND:-249}"
DEFENCE_PCT="${DEFENCE_PCT:-0.1}"
OUTDIR="/scratch/projects/secure-distributed-ml/results/b4_multi_seed${SEED}_client${CLIENT_ID}_${SLURM_ARRAY_JOB_ID}"
mkdir -p "$OUTDIR"

cd /projects/secure-distributed-ml/AdaGuard

run_one () {
    local attack="$1"; local iters="$2"; local defence="$3"
    local tag="${attack}_b4_${defence}_seed${SEED}_client${CLIENT_ID}"
    echo "============================================================"
    echo "  ATTACK=$attack  DEFENCE=$defence  SEED=$SEED  CLIENT=$CLIENT_ID  B=4"
    echo "============================================================"
    stdbuf -oL -eL python -u tests/attack_sanity_check.py \
        --artifact-dir "$ARTIFACT_DIR" \
        --round "$ROUND" \
        --client-id "$CLIENT_ID" \
        --attack "$attack" \
        --n-iter "$iters" \
        --batch-size 4 \
        --diverse-subset \
        --defence "$defence" \
        --defence-pct "$DEFENCE_PCT" \
        --variant paper \
        --out "$OUTDIR/${tag}.json"
    echo ""
}

for defence in none fhe maskcrypt fisher; do
    run_one gradinversion 20000 "$defence"
    run_one ggcdm         100   "$defence"
    run_one gi_nas        2000  "$defence"
done

echo "B=4 multi-client task finished at $(date)"
echo "Artifacts in $OUTDIR"
