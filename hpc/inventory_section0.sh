#!/usr/bin/env bash
# Section 0 inventory — Matilda scratch + project state for the
# Fisher-sensitivity follow-up campaign.
#
# Run on Matilda from anywhere (no cwd assumption):
#   ssh maguir@hpc-login.oakland.edu
#   bash /projects/secure-distributed-ml/AdaGuard/hpc/inventory_section0.sh \
#     2>&1 | tee /tmp/section0_inventory.txt
#
# Then paste the file contents back to the planning conversation.
#
# Read-only: does not modify anything. Tolerant of missing directories.

set +e  # don't bail on individual command failures

RESULTS_ROOT=/scratch/projects/secure-distributed-ml/results
EXP_DIR="$RESULTS_ROOT/1k_experiment"
PROJECT_DIR=/projects/secure-distributed-ml
SCRATCH_DIR=/scratch/projects/secure-distributed-ml

echo "================================================================"
echo "Section 0 — Matilda inventory"
echo "Host:  $(hostname)"
echo "Date:  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "User:  $USER"
echo "================================================================"

# ----------------------------------------------------------------------
# 1. Free space — scratch (where artefacts live) and project (where the
#    repo + venv live). Lustre quotas lag per HANDOFF; df is authoritative.
# ----------------------------------------------------------------------
echo
echo "=== [1] Free space ==="
echo
echo "--- df on results scratch ---"
df -h "$RESULTS_ROOT" 2>&1 | head -5
echo
echo "--- df on project ---"
df -h "$PROJECT_DIR" 2>&1 | head -5
echo
echo "--- Lustre quota (may lag after deletes) ---"
lfs quota -h -u "$USER" "$SCRATCH_DIR" 2>&1 || echo "  (lfs quota not applicable on this filesystem)"
lfs quota -h -u "$USER" "$PROJECT_DIR" 2>&1 || true

# ----------------------------------------------------------------------
# 2. What lives under results/1k_experiment/ — the headline-matrix dir
# ----------------------------------------------------------------------
echo
echo "=== [2] /scratch/.../1k_experiment/ contents ==="
echo
if [ ! -d "$EXP_DIR" ]; then
    echo "  MISSING: $EXP_DIR does not exist."
    echo "  Either the canonical headline-matrix artefacts were cleaned,"
    echo "  or the path is different on this Matilda account. Check"
    echo "  /scratch/projects/secure-distributed-ml/results/ for siblings."
    ls -la "$RESULTS_ROOT" 2>&1 | head -30
else
    ls -la "$EXP_DIR" 2>&1 | head -40
fi

# ----------------------------------------------------------------------
# 3. Per-defence per-seed artefact presence + saved-round counts
#    (HANDOFF §7: dirs named artifacts_<defence>_seed<seed>_300clients/)
# ----------------------------------------------------------------------
echo
echo "=== [3] Per-(defence, seed) artefact inventory ==="
echo
echo "  Expected layout: $EXP_DIR/artifacts_<defence>_seed<seed>_300clients/round_<N>/"
echo
for defence in none fhe maskcrypt fisher; do
    for seed in 42 123 456; do
        D="$EXP_DIR/artifacts_${defence}_seed${seed}_300clients"
        if [ ! -d "$D" ]; then
            printf "  [MISSING] %-50s\n" "${defence}/seed${seed}"
            continue
        fi
        # Count saved rounds and find min/max
        ROUND_LIST=$(ls -d "$D"/round_* 2>/dev/null | sed 's/.*round_//' | sort -n)
        if [ -z "$ROUND_LIST" ]; then
            printf "  [EMPTY ] %-50s (dir exists, no round_N/ subdirs)\n" "${defence}/seed${seed}"
            continue
        fi
        N_ROUNDS=$(echo "$ROUND_LIST" | wc -l)
        MIN_R=$(echo "$ROUND_LIST" | head -1)
        MAX_R=$(echo "$ROUND_LIST" | tail -1)
        SIZE=$(du -sh "$D" 2>/dev/null | awk '{print $1}')
        printf "  [PRESENT] %-50s rounds=%-4d min=%-4s max=%-4s size=%s\n" \
            "${defence}/seed${seed}" "$N_ROUNDS" "$MIN_R" "$MAX_R" "$SIZE"
    done
done

# ----------------------------------------------------------------------
# 4. Section-1-specific check: rounds 75, 150, 249 on fisher/seed42
#    This is what slurm_trajectory_phase2_only.sh actually needs.
# ----------------------------------------------------------------------
echo
echo "=== [4] Section 1 target rounds on fisher/seed42 ==="
echo
D="$EXP_DIR/artifacts_fisher_seed42_300clients"
for r in 75 150 249; do
    if [ -d "$D/round_$r" ]; then
        N_CLIENTS=$(ls "$D/round_$r"/client_*.pt 2>/dev/null | wc -l)
        SIZE=$(du -sh "$D/round_$r" 2>/dev/null | awk '{print $1}')
        echo "  [OK] round_$r  ($N_CLIENTS client artefacts, $SIZE)"
    else
        # find nearest available
        NEAREST=$(ls -d "$D"/round_* 2>/dev/null | sed 's/.*round_//' | sort -n | \
            awk -v target="$r" 'BEGIN{best=-1; gap=99999} {g=$1-target; if(g<0)g=-g; if(g<gap){gap=g; best=$1}} END{print best}')
        if [ "$NEAREST" = "-1" ] || [ -z "$NEAREST" ]; then
            echo "  [MISSING] round_$r  (and no neighbouring rounds either)"
        else
            echo "  [MISSING] round_$r  -> nearest available: round_$NEAREST"
        fi
    fi
done

# ----------------------------------------------------------------------
# 5. Confirm the project-side repo state and venv presence
# ----------------------------------------------------------------------
echo
echo "=== [5] Project-side state ==="
echo
REPO="$PROJECT_DIR/AdaGuard"
if [ -d "$REPO" ]; then
    echo "  Repo at: $REPO"
    git -C "$REPO" log -1 --oneline 2>&1 | head -1
    git -C "$REPO" rev-parse --abbrev-ref HEAD 2>&1 | head -1
else
    echo "  WARNING: $REPO does not exist."
fi
if [ -f "$PROJECT_DIR/venv/bin/activate" ]; then
    echo "  venv: $PROJECT_DIR/venv (activate script present)"
else
    echo "  WARNING: $PROJECT_DIR/venv/bin/activate not found."
fi

# ----------------------------------------------------------------------
# 6. Decision summary
# ----------------------------------------------------------------------
echo
echo "=== [6] Section 1 decision summary ==="
echo
if [ -d "$EXP_DIR/artifacts_fisher_seed42_300clients/round_249" ]; then
    echo "  fisher/seed42/round_249 exists -> Section 1 can run REPLAY-ONLY"
    echo "  (slurm_trajectory_phase2_only.sh). No new save space needed."
    if [ -d "$EXP_DIR/artifacts_fisher_seed42_300clients/round_75" ] && \
       [ -d "$EXP_DIR/artifacts_fisher_seed42_300clients/round_150" ]; then
        echo "  All three target rounds present -> no fallback rounds needed."
    else
        echo "  NOTE: not all of {75, 150, 249} are present. The Slurm script"
        echo "        falls back to nearest-available; check section [4] above."
    fi
else
    echo "  fisher/seed42/round_249 MISSING -> Section 1 needs a Phase-1 RETRAIN."
    echo "  (We'll need slurm_trajectory_full.sh -- not yet written.)"
fi
echo
echo "================================================================"
echo "Inventory complete. Paste this entire file back to planning."
echo "================================================================"
