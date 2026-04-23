#!/usr/bin/env python3
"""Run iDLG / LLG+ label recovery across multiple defence artifact dirs.

Pixel PSNR collapses to noise floor at B=16 Fisher-non-IID for every
defence (see Attack Restructure 1), so the paper's ASR headline is
almost certainly label-recovery accuracy. This script loads each
defence's Phase-1 artifacts at a given round, runs the analytical LLG+
recovery on each client, and prints a per-defence mean ASR. No
optimisation loop — milliseconds per client.

Example
-------
python tests/label_recovery_sweep.py \
    --round 249 --num-classes 10 \
    --dir V1=/scratch/.../artifacts_none_seed42_300clients \
    --dir V2=/scratch/.../artifacts_full_seed42_300clients \
    --dir V4=/scratch/.../artifacts_maskcrypt_seed42_300clients \
    --dir V6=/scratch/.../artifacts_fisher_seed42_300clients \
    --out results/label_recovery_sweep.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adaguard.attacks.label_recovery import label_recovery_summary  # noqa: E402


def _iter_clients(round_dir: Path, max_clients: int = None):
    clients = sorted(
        f for f in os.listdir(round_dir)
        if f.startswith('client_') and f.endswith('.pt')
    )
    if max_clients is not None:
        clients = clients[:max_clients]
    return clients


def _run_one_dir(label: str, artifact_dir: Path, rnd: int,
                 num_classes: int, max_clients: int = None):
    round_dir = artifact_dir / f'round_{rnd}'
    if not round_dir.exists():
        print(f"[{label}] MISSING {round_dir}")
        return None

    clients = _iter_clients(round_dir, max_clients=max_clients)
    if not clients:
        print(f"[{label}] no client_*.pt in {round_dir}")
        return None

    per_client = []
    for fname in clients:
        art = torch.load(round_dir / fname, map_location='cpu', weights_only=False)
        gd = art.get('gradient_dict')
        labels = art.get('labels')
        if gd is None or labels is None:
            continue
        res = label_recovery_summary(gd, labels, num_classes=num_classes)
        per_client.append({
            'client_file': fname,
            'asr': res['asr'],
            'source': res['source'],
            'key_used': res['key_used'],
            'signal': res['signal'],
            'batch_size': len(res['truth']),
        })

    if not per_client:
        print(f"[{label}] no usable clients")
        return None

    asrs = [c['asr'] for c in per_client]
    signals = [c['signal'] for c in per_client]
    sources = {c['source'] for c in per_client}
    keys = {c['key_used'] for c in per_client if c['key_used']}
    mean = sum(asrs) / len(asrs)
    mn = min(asrs)
    mx = max(asrs)
    print(
        f"[{label:>4}] clients={len(per_client):3d}  "
        f"mean ASR={mean:.4f}  min={mn:.4f}  max={mx:.4f}  "
        f"mean signal={sum(signals) / len(signals):.4f}  "
        f"src={'/'.join(sorted(sources))}  keys={'/'.join(sorted(keys))}"
    )

    return {
        'label': label,
        'artifact_dir': str(artifact_dir),
        'round': rnd,
        'num_clients': len(per_client),
        'mean_asr': mean,
        'min_asr': mn,
        'max_asr': mx,
        'mean_signal': sum(signals) / len(signals),
        'sources': sorted(sources),
        'keys': sorted(keys),
        'per_client': per_client,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        '--dir', action='append', default=[],
        help="label=path. Repeat for each defence. "
             "Example: --dir V1=/scratch/.../artifacts_none_seed42_300clients",
    )
    ap.add_argument('--round', type=int, default=249)
    ap.add_argument('--num-classes', type=int, default=10)
    ap.add_argument('--max-clients', type=int, default=None,
                    help="Optional cap on #clients per defence (default: all).")
    ap.add_argument('--out', default=None,
                    help="Optional JSON path for the full report.")
    args = ap.parse_args()

    if not args.dir:
        ap.error("Pass at least one --dir label=path entry.")

    print(f"Label recovery sweep: round={args.round} "
          f"num_classes={args.num_classes}")
    print("=" * 80)

    report = {
        'round': args.round,
        'num_classes': args.num_classes,
        'defences': [],
    }
    for entry in args.dir:
        if '=' not in entry:
            print(f"[WARN] --dir '{entry}' has no label=path form, skipping")
            continue
        label, path = entry.split('=', 1)
        label = label.strip()
        d = Path(path.strip())
        out = _run_one_dir(label, d, args.round,
                           num_classes=args.num_classes,
                           max_clients=args.max_clients)
        if out is not None:
            report['defences'].append(out)

    print("=" * 80)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Wrote full report -> {args.out}")


if __name__ == '__main__':
    main()
