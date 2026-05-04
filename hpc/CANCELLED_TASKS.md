# Cancelled HPC tasks awaiting re-run

Tracking what was cancelled or otherwise lost during the K-sweep +
B-sweep + B4-multi launch on 2026-05-04, and what still needs to be
re-submitted to complete the paper's RQ4/RQ5 evidence.

## Why we trimmed

Two orthogonal pressures drove the cancellation:

1. **Quota**: Project share is 10 TB hard cap. Initial launch projected
   ~13-15 TB end-state, so the 4 highest-numbered B-sweep array tasks
   were cancelled pre-emptively to land safely. After per-K-sweep
   `artifacts/` cleanup (kept JSON/checkpoints/snapshots), used
   dropped from 8.36 TB to 6.5 TB.
2. **B-sweep bug**: Job 182448 array tasks 0-9 all hit a silent
   Phase-1 exit after round 1 of training (config_1k.yaml's
   1000-client setup triggers a clean exit in run_headless after
   round-1 aggregation -- no exception, no traceback). Phase-2 then
   crashed on `FileNotFoundError: round_249`. All 10 attempted
   B-sweep tasks were lost.

## Fix applied (commit pending)

`hpc/slurm_b_sweep.sh` switched from `config_1k.yaml` (1000 clients,
buggy) to `config_t2_low.yaml` (300 clients, proven by K-sweep and
1k_experiment). One-line change in the tmp-config writer.

## Scope decision: "balanced" sweep

Original plan: 5 B values x 3 seeds = 15 tasks (--array=0-14).

Trimmed to **9 tasks**: B in {32, 64, 128} x 3 seeds = --array=6-14.

Rationale:
- B=4 already covered by B4-multi (job 182449, 15 client/seed cells).
- B=16 implicit from 1k_experiment main results (3 seeds x 4 defences).
- B=8 dropped: low-batch regime is well-covered between B=4 (B4-multi)
  and B=16 (main results); the marginal info from B=8 doesn't justify
  6 more GPU-hours.
- B in {32, 64, 128} is the actual professor RQ5 ask: "does AdaGuard
  still help at large batches where natural averaging is already strong?"
- 3 seeds gives error bars at each B.

Estimated cost: 9 tasks x ~6h Phase-1 + Phase-2 = ~54 GPU-hours,
~7-8h wall at 8-way concurrency.

## Resubmission command (after pull)

```bash
git pull
sbatch --array=6-14 hpc/slurm_b_sweep.sh
squeue -u maguir -j <new_job_id>
```

## What's missing from the paper's grid

After the balanced re-run completes, the B-coverage will be:

| B    | source                        | seeds | defences |
|------|-------------------------------|-------|----------|
| 1    | main 1k_experiment            | 3     | 4        |
| 4    | B4-multi (182449)             | 3 (5 clients each) | 4 |
| 32   | B-sweep balanced              | 3     | 4        |
| 64   | B-sweep balanced              | 3     | 4        |
| 128  | B-sweep balanced              | 3     | 4        |

B=2, B=8, B=16 are not directly evaluated in the new sweeps but the
paper can:
- Note B=16 was the default in the main-table experiment (so V1/V2/
  V3/V4 numbers at B=16 are the headline numbers).
- Drop B=2, B=8 from the discussion, OR add a one-paragraph note that
  the small-batch regime is dominated by the per-image gradient and
  is the easiest case for any defence (so omitted for compute budget).

## What's permanently lost (not worth re-running)

- Job 182448 array 0-14 (all B-sweep attempts before the fix): no
  usable Phase-1 artefacts (rounds 0-2 only out of 250). All disk
  was from these 15 partial tasks; they should be deleted to free
  ~150 GB if quota becomes tight again.
- Job 182467 (5 redundant K-sweep retrains): obviated by the
  `slurm_k_sweep_phase2_only.sh` round-242-244 fallback (commit
  e0c28cb). No re-run needed.

## Caveats for the paper text

If we ship without the balanced B-sweep:

- §VI.D B-sweep paragraph: rewrite to "B in {1, 4 (5 clients), 16 (3
  seeds)}", and soften the "across batch sizes" claim.
- §VII reviewer-extension subsection: explicitly defer the B=64, 128
  evidence to "extended evaluation; preliminary B=32 result available
  for seed 42" if even the balanced run doesn't finish.

If we re-run before submission, this note can be deleted.
