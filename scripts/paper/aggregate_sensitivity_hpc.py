#!/usr/bin/env python3
"""Aggregate Phase-2 sensitivity sweep results into a single small JSON.

Reads /scratch/projects/secure-distributed-ml/results/study_<JOBID>/S*.json
files (each containing a sampled-round Phase-2 run) and produces a small
JSON with the fields needed for the paper's sensitivity tables: the swept
parameter value, accuracy/LeakScore/encrypted% trajectory across the
sampled rounds, and a final-round summary.

Per-attack reconstruction metrics (PSNR/LPIPS/SSIM) are intentionally
NOT extracted — Phase 2.5's metric-recompute fix was applied to the
defence sweep but not the sensitivity sweep, so per-round PSNR values
in this data are pre-normalisation and not directly comparable. The
LeakScore + accuracy + encrypted% trajectory is what the paper needs.

Usage on HPC:
    python aggregate_sensitivity_hpc.py \\
        --study-dir /scratch/projects/secure-distributed-ml/results/study_181097 \\
        --out /scratch/projects/secure-distributed-ml/results/sensitivity_aggregate.json
"""

import argparse
import json
import re
from pathlib import Path

SCENARIO_RX = re.compile(r"(S\d+(?:\.\d+)?)")


def find_scenario_files(study_dir: Path):
    out = {}
    for p in study_dir.rglob("*.json"):
        m = SCENARIO_RX.search(p.name)
        if not m:
            continue
        sid = m.group(1)
        out.setdefault(sid, []).append(p)
    return out


def summarise_run(j: dict):
    rounds = j.get("rounds", []) or []
    if not rounds:
        return None

    overrides = j.get("config_overrides", {}) or {}
    variable = j.get("variable")
    value = j.get("value")

    acc_series = [r.get("accuracy") for r in rounds]
    leak_series = [r.get("avg_leakscore") for r in rounds]
    enc_series = [r.get("avg_pct_encrypted") for r in rounds]
    round_idx = [r.get("round") for r in rounds]

    last = rounds[-1]
    summary = {
        "scenario_id": j.get("scenario_id"),
        "case": j.get("case"),
        "group": j.get("group"),
        "description": j.get("description"),
        "variable": variable,
        "value": value,
        "config_overrides": overrides,
        "n_sampled_rounds": len(rounds),
        "sampled_round_indices": round_idx,
        "trajectory": {
            "accuracy": acc_series,
            "avg_leakscore": leak_series,
            "avg_pct_encrypted": enc_series,
        },
        "last_round": {
            "round": last.get("round"),
            "accuracy": last.get("accuracy"),
            "test_loss": last.get("test_loss"),
            "avg_leakscore": last.get("avg_leakscore"),
            "avg_pct_encrypted": last.get("avg_pct_encrypted"),
            "encryption": last.get("encryption"),
            "encryption_time_s": last.get("encryption_time_s"),
            "num_clients": last.get("num_clients"),
        },
        "total_time_s": j.get("total_time_s"),
    }
    valid_leak = [v for v in leak_series if isinstance(v, (int, float))]
    if valid_leak:
        summary["leakscore_summary"] = {
            "n": len(valid_leak),
            "min": min(valid_leak),
            "max": max(valid_leak),
            "mean": sum(valid_leak) / len(valid_leak),
        }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    files = find_scenario_files(args.study_dir)
    print(f"Found {len(files)} scenario IDs across "
          f"{sum(len(v) for v in files.values())} JSON files")

    aggregate = {}
    for sid in sorted(files):
        runs = []
        for p in files[sid]:
            try:
                with open(p) as f:
                    j = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                print(f"  skip {p.name}: {e}")
                continue
            s = summarise_run(j)
            if s:
                s["source_file"] = p.name
                runs.append(s)
        if runs:
            aggregate[sid] = runs

    with open(args.out, "w") as f:
        json.dump(aggregate, f, indent=2, sort_keys=True)
    print(f"Wrote {args.out} ({args.out.stat().st_size / 1024:.1f} KB) "
          f"with {len(aggregate)} scenario groups")


if __name__ == "__main__":
    main()
