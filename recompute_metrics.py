"""Post-process saved .pt reconstruction payloads and recompute metrics
with consistent [0,1] normalization.

Context: during the Phase-2 sweep, attack reconstructions are clamped to [0,1]
but originals were stored in CIFAR-10-normalized form
(mean=(0.4914,0.4822,0.4465), std=(0.2470,0.2435,0.2616)).
The per-scenario JSONs therefore contain PSNR/SSIM/LPIPS/MSE computed over
mismatched ranges. The saved tensors themselves are intact, so this script
rebuilds the metrics offline.

IMPORTANT: run on a compute node, not the login node. Example:
    srun --partition=general-long --cpus-per-task=4 --mem=16G --time=2:00:00 --pty bash
    source /projects/secure-distributed-ml/venv/bin/activate
    cd /projects/secure-distributed-ml/AdaGuard
    python recompute_metrics.py --results-dir <study_JOB> --output <out>.json
"""

import argparse
import gc
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from adaguard.utils.reconstruction import (
    compute_mse, compute_psnr, compute_ssim, compute_asr,
    compute_cosine_similarity_images, compute_avd, compute_ddcs,
    compute_knn_distance,
)

CIFAR_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
CIFAR_STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)

# Device picked once at module load. LPIPS and tensors ride the GPU
# if available; denorm/MSE/PSNR/SSIM/etc run on CPU via numpy.
_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Module-level LPIPS model cache. Building AlexNet is ~200 MB and ~1s; doing
# it per-file is what OOM'd the login node. Build once, reuse everywhere.
_LPIPS_MODEL = None


def get_lpips_model():
    """Return a cached LPIPS(net='alex') model, or None if unavailable."""
    global _LPIPS_MODEL
    if _LPIPS_MODEL is not None:
        return _LPIPS_MODEL
    try:
        import lpips
        _LPIPS_MODEL = lpips.LPIPS(net='alex', verbose=False).eval().to(_DEVICE)
        for p in _LPIPS_MODEL.parameters():
            p.requires_grad_(False)
        print(f"[lpips] AlexNet loaded once on {_DEVICE}; will be reused.", flush=True)
    except Exception as e:
        print(f"[lpips] unavailable — LPIPS will be reported as NaN: {e}", flush=True)
        _LPIPS_MODEL = False
    return _LPIPS_MODEL


def compute_lpips_cached(orig_01, recon_01):
    """LPIPS using the cached model. Inputs must be in [0,1]."""
    model = get_lpips_model()
    if not model:
        return float('nan')
    with torch.no_grad():
        o = (orig_01 * 2 - 1).to(_DEVICE)
        r = (recon_01 * 2 - 1).to(_DEVICE)
        d = model(o, r)
    return float(d.mean().item())


def denormalize(x):
    """Undo CIFAR-10 mean/std normalization and clamp to [0,1]."""
    if x.ndim == 3:
        x = x.unsqueeze(0)
    return (x * CIFAR_STD + CIFAR_MEAN).clamp(0, 1)


def compute_metrics(orig_01, recon_01, labels=None):
    """All reconstruction metrics with both inputs already in [0,1].

    Note: FID is intentionally not computed per-file. It requires pooled
    distributions (not batch-of-16 samples), and its per-file cost via
    np.linalg.eigvals on a 3072x3072 covariance matrix dominated the
    runtime of the first batch job (TIMEOUT at 4h). FID should be computed
    once per scenario downstream if needed.
    """
    m = {}
    m['mse'] = compute_mse(orig_01, recon_01)
    m['psnr'] = compute_psnr(orig_01, recon_01)
    m['ssim'] = compute_ssim(orig_01, recon_01)
    m['asr'] = compute_asr(orig_01, recon_01)
    m['cosine_similarity'] = compute_cosine_similarity_images(orig_01, recon_01)
    m['avd'] = compute_avd(orig_01, recon_01)
    m['ddcs'] = compute_ddcs(orig_01, recon_01)
    if orig_01.ndim == 4:
        m['knn_distance'] = compute_knn_distance(orig_01, recon_01)
    else:
        m['knn_distance'] = 0.0
    m['fid'] = None  # deferred to per-scenario aggregation
    m['lpips'] = compute_lpips_cached(orig_01, recon_01)
    return m


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

    metrics = compute_metrics(orig_01, recon_01)
    metrics['gradient_score'] = payload.get('metrics', {}).get('gradient_score', 0.0)

    del payload, recon, orig, recon_01, orig_01
    return metrics


def aggregate(per_client):
    """Mean each metric across clients, per attack."""
    attacks = ['gradinversion', 'gi_nas', 'ggcdm']
    metric_names = ['mse', 'psnr', 'ssim', 'lpips', 'fid', 'asr',
                    'cosine_similarity', 'knn_distance', 'ddcs', 'avd',
                    'gradient_score']
    summary = {}
    for a in attacks:
        agg = {}
        for m in metric_names:
            vals = []
            for c in per_client.values():
                v = c.get(a, {}).get(m)
                if isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v)):
                    vals.append(v)
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
    ap.add_argument('--progress-every', type=int, default=25,
                    help='Print progress every N files')
    args = ap.parse_args()

    root = Path(args.results_dir) / 'reconstructions'
    if not root.exists():
        raise SystemExit(f"No reconstructions/ under {args.results_dir}")

    pt_files = sorted(root.glob('*/round_*/client_*/*.pt'))
    print(f"Found {len(pt_files)} .pt files under {root}", flush=True)
    print(f"Device: {_DEVICE}", flush=True)

    # Warm up LPIPS model once so we fail fast if it's broken.
    get_lpips_model()

    per_scenario = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    fail_count = 0
    for i, p in enumerate(pt_files):
        scenario = p.parts[-4]
        round_key = p.parts[-3]
        client_key = p.parts[-2]
        attack = p.stem
        try:
            metrics = process_pt(p)
        except Exception as e:
            fail_count += 1
            print(f"  [{i+1}/{len(pt_files)}] {scenario}/{round_key}/{client_key}/{attack}: FAILED — {e}")
            continue
        if metrics is None:
            continue
        per_scenario[scenario][round_key][client_key][attack] = metrics

        if (i + 1) % args.progress_every == 0:
            print(f"  [{i+1}/{len(pt_files)}]  {scenario}/{round_key}/{client_key}/{attack}  "
                  f"psnr={metrics['psnr']:.2f} lpips={metrics['lpips']:.3f}",
                  flush=True)
            gc.collect()

    out = {}
    for scenario, rounds in per_scenario.items():
        out[scenario] = {'rounds': {}, 'summary': {}}
        all_clients_flat = {}
        for rk, clients in rounds.items():
            out[scenario]['rounds'][rk] = {
                'clients': dict(clients),
                'summary': aggregate(clients),
            }
            for ck, ametrics in clients.items():
                all_clients_flat[f'{rk}/{ck}'] = ametrics
        out[scenario]['summary'] = aggregate(all_clients_flat)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nWrote {out_path} — {len(out)} scenarios, {fail_count} failed files")

    print("\nPer-scenario cross-attack mean PSNR / LPIPS / SSIM:")
    for s in sorted(out.keys()):
        sm = out[s]['summary']
        psnr = sm.get('mean_psnr', float('nan'))
        lpips_v = sm.get('mean_lpips', float('nan'))
        ssim = sm.get('mean_ssim', float('nan'))
        print(f"  {s:24s}  PSNR={psnr:7.3f}  LPIPS={lpips_v:6.4f}  SSIM={ssim:7.4f}")


if __name__ == '__main__':
    main()
