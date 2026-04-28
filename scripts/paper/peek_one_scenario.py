#!/usr/bin/env python3
"""Quick structural peek at one Phase-2 sensitivity JSON.

Usage on HPC:
    python scripts/paper/peek_one_scenario.py \\
        --study-dir /scratch/projects/secure-distributed-ml/results/study_181097
"""

import argparse
import json
import re
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study-dir", required=True, type=Path)
    args = ap.parse_args()

    candidates = list(args.study_dir.rglob("*S5.5*.json"))
    if not candidates:
        candidates = list(args.study_dir.rglob("*S5*.json"))
    if not candidates:
        candidates = list(args.study_dir.rglob("*.json"))
    if not candidates:
        print("No JSON files found.")
        return

    p = candidates[0]
    print(f"=== Structure of {p.name} ===")
    with open(p) as f:
        d = json.load(f)

    print("Top-level keys:", list(d.keys()))
    for k, v in d.items():
        if isinstance(v, dict):
            print(f"  {k}: dict with keys = {list(v.keys())[:20]}")
        elif isinstance(v, list):
            print(f"  {k}: list of len {len(v)}")
            if v and isinstance(v[0], dict):
                print(f"    [0] keys: {sorted(v[0].keys())}")
        else:
            print(f"  {k}: {type(v).__name__} = {repr(v)[:80]}")

    # Dump first round in full so I can see all available fields
    if isinstance(d.get("rounds"), list) and d["rounds"]:
        print("\n=== First round full dump ===")
        print(json.dumps(d["rounds"][0], indent=2, default=str)[:2000])
        print("\n=== Last round full dump ===")
        print(json.dumps(d["rounds"][-1], indent=2, default=str)[:2000])


if __name__ == "__main__":
    main()
