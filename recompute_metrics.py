"""Post-process saved .pt reconstruction payloads and recompute metrics
with consistent [0,1] normalization.

Context: during the Phase-2 sweep, attack reconstructions are clamped to [0,1]
but originals were stored in CIFAR-10-normalized form
(mean=(0.4914,0.4822,0.4465), std=(0.2470,0.2435,0.2616)).
The per-scenario JSONs therefore contain PSNR/SSIM/LPIPS/MSE computed over
mismatched ranges. The saved tensors themselves are intact, so this script
rebuilds the metrics offline.

Usage:
    python recompute_metrics.py \\
        --results-dir /scratch/projects/secure-distributed-ml/results/study_181097 \\
        --output corrected_metrics.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from adaguard.utils.reconstruction import compute_all_reconstruction_metrics

CIFAR_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
CIFAR_STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)


def denormalize(x):
    """Undo CIFAR-10 mean/std normalization and clamp to [0,1]."""
    if x.ndim == 3:
        x = x.unsqueeze(0)
    return (x * CIFAR_STD + CIFAR_MEAN).clamp(0, 1)


def process_pt(path):
    """Load a single .pt payload and recompute metrics in [0,1] space."""
    payload = torch.load(path, map_location='cpu', weights_only=False)
    recon = payload.get('reconstructed')
    orig = payload.get('original')
    if recon is None or orig is None:
        return None

    if recon.ndim == 3:
        recon = recon.unsqueeze(0)
    if orig.ndim == 3:
        orig = orig.unsqueeze(0)

    recon_01 = recon.clamp(0, 1).float()
    orig_01 = denormalize(orig.float())

    n = min(recon_01.shape[0], orig_01.shape[0])
    recon_01 = recon_01[:n]
    orig_01 = orig_01[:n]

    labels = payload.get('labels')
    labels_np = labels.cpu().numpy() if labels is not None else None

    metrics = compute_all_reconstruction_metrics(
        orig_01, recon_01, labels_true=labels_np, use_lpips=True,
    )
    metrics['gradient_score'] = payload.get('metrics', {}).get('gradient_score', 0.0)
    return metrics


def aggregate(per_client):
    """Mean each metric across clients, per attack."""
    attacks = ['gradinversion', 'gi_nas', 'ggcdm']
    metric_names = ['mse', 'psnr', 'ssim', 'lpips', 'fid', 'asr',
                    'cosine_similarity', 'knn_distance', 'ddcs', 'avd',
                    'mcc', 'gmean', 'gradient_score']
    summary = {}
    for a in attacks:
        agg = {}
        for m in metric_names:
            vals = [c[a][m] for c in per_client.values()
                    if a in c and isinstance(c[a].get(m), (int, float))]
            if vals:
                agg[m] = float(np.mean(vals))
        summary[a] = agg
    for m in metric_names:
        vals = [summary[a].get(m) for a in attacks if m in summary.get(a, {})]
        vals = [v for v in vals if v is not None]
        if vals:
            summary[f'mean_{m}'] = float(np.mean(vals))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-dir', required=True,
                    help='Path to study_<JOB> directory containing reconstructions/')
    ap.add_argument('--output', default='corrected_metrics.json')
    args = ap.parse_args()

    root = Path(args.results_dir) / 'reconstructions'
    if not root.exists():
        raise SystemExit(f"No reconstructions/ under {args.results_dir}")

    # reconstructions/<scenario>/round_N/client_X/<attack>.pt
    out = {}
    pt_files = sorted(root.glob('*/round_*/client_*/*.pt'))
    print(f"Found {len(pt_files)} .pt files under {root}")

    per_scenario = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    # per_scenario[scenario][round_N][client_X][attack] = metrics

    for i, p in enumerate(pt_files):
        scenario = p.parts[-4]
        round_key = p.parts[-3]
        client_key = p.parts[-2]
        attack = p.stem
        try:
            metrics = process_pt(p)
        except Exception as e:
            print(f"  [{i+1}/{len(pt_files)}] {p}: FAILED — {e}")
            continue
        if metrics is None:
            continue
        per_scenario[scenario][round_key][client_key][attack] = metrics
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(pt_files)}]  last={scenario}/{round_key}/{client_key}/{attack}")

    # Build output: per-scenario rounds with per-client + per-round summary
    for scenario, rounds in per_scenario.items():
        out[scenario] = {'rounds': {}, 'summary': {}}
        all_clients_flat = {}  # for cross-round summary
        for rk, clients in rounds.items():
            out[scenario]['rounds'][rk] = {
                'clients': dict(clients),
                'summary': aggregate(clients),
            }
            for ck, ametrics in clients.items():
                all_clients_flat[f'{rk}/{ck}'] = ametrics
        out[scenario]['summary'] = aggregate(all_clients_flat)

    out_path = Path(args.output)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nWrote {out_path} — {len(out)} scenarios")

    # Compact console summary
    print("\nPer-scenario cross-attack mean PSNR / LPIPS / SSIM:")
    for s in sorted(out.keys()):
        sm = out[s]['summary']
        psnr = sm.get('mean_psnr', float('nan'))
        lpips_v = sm.get('mean_lpips', float('nan'))
        ssim = sm.get('mean_ssim', float('nan'))
        print(f"  {s:20s}  PSNR={psnr:7.3f}  LPIPS={lpips_v:6.4f}  SSIM={ssim:7.4f}")


if __name__ == '__main__':
    main()
