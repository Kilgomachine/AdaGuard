#!/usr/bin/env python3
"""Sanity-check runner for the paper-faithful gradient-inversion attacks.

Loads a single Phase-1 client artifact + its round's global model, runs ONE
attack against the undefended gradient, and prints PSNR / SSIM / LPIPS
against the paper's reported target. Intended to be run on HPC after any
attack re-implementation to verify the numbers actually land.

Examples
--------
# GradInversion (paper-faithful, 20k iters) on round 249 of the seed-42
# Phase-1 artifacts:
python tests/attack_sanity_check.py \
    --artifact-dir /scratch/projects/secure-distributed-ml/results/1k_experiment/artifacts_fisher_seed42_300clients \
    --round 249 \
    --attack gradinversion \
    --n-iter 20000

# Quick smoke (500 iters) to catch wiring bugs before burning a 20k run:
python tests/attack_sanity_check.py --artifact-dir ... --round 249 \
    --attack gradinversion --n-iter 500
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adaguard.models import create_model  # noqa: E402
from adaguard.utils.reconstruction import (  # noqa: E402
    compute_mse, compute_psnr, compute_ssim, compute_lpips,
)

# CIFAR-10 normalization used by Phase-1 training data loader.
CIFAR_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
CIFAR_STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)

# Paper PSNR targets on CIFAR-10 (undefended, B=16 unless noted).
# GradInversion: Yin et al 2021 — reports ~13 dB ImageNet full recipe;
#   no CIFAR number in paper, so we use the "functional" target of PSNR > 15 dB
#   (well above noise-floor ~7 dB).
# GI-NAS: Yu et al 2025 — Table IV, CIFAR-10 B=16 = 30.5 dB.
# GGCDM: Meng et al 2025 — undefended CelebA ~41 dB; DP sigma^2=0.01 ~15 dB.
PAPER_TARGETS = {
    'gradinversion': {'psnr': 15.0, 'label': '> 15 dB (well above noise floor)'},
    'gi_nas': {'psnr': 30.5, 'label': '30.5 dB (Yu 2025 Table IV CIFAR B=16)'},
    'ggcdm': {'psnr': 41.0, 'label': '~41 dB undefended (Meng 2025 CelebA)'},
}


def _denormalize(x):
    """Undo CIFAR normalisation. Phase-1 saves post-Normalize tensors."""
    mean = CIFAR_MEAN.to(x.device)
    std = CIFAR_STD.to(x.device)
    return (x * std + mean).clamp(0, 1)


def _load_artifact(artifact_dir: Path, rnd: int, client_id=None):
    """Load one client's gradient_dict + original images + labels."""
    round_dir = artifact_dir / f'round_{rnd}'
    if not round_dir.exists():
        raise FileNotFoundError(f"No round dir at {round_dir}")

    global_path = round_dir / 'global_model.pt'
    global_state = torch.load(global_path, map_location='cpu', weights_only=False)

    clients = sorted(
        f for f in os.listdir(round_dir)
        if f.startswith('client_') and f.endswith('.pt')
    )
    if not clients:
        raise FileNotFoundError(f"No client_*.pt in {round_dir}")

    if client_id is not None:
        fname = f'client_{client_id}.pt'
        if fname not in clients:
            raise FileNotFoundError(f"{fname} not in {round_dir}")
        pick = fname
    else:
        pick = clients[0]

    art = torch.load(round_dir / pick, map_location='cpu', weights_only=False)

    # Different pipelines have used 'images' vs 'original_images'. Accept both.
    originals = art.get('images', art.get('original_images'))
    labels = art.get('labels')
    gd = art.get('gradient_dict')

    if gd is None:
        raise RuntimeError(f"{pick} has no gradient_dict")

    return {
        'global_state': global_state,
        'gradient_dict': gd,
        'originals': originals,
        'labels': labels,
        'client_file': pick,
    }


def _build_attack(name: str, model, criterion, device, n_iter: int,
                  n_candidates: int = 1, variant: str = 'paper'):
    """Construct an attack by name.

    ``variant`` selects the implementation family:
      * ``paper`` (default) — paper-faithful attacks (GradInversionFull with
        full Yin recipe, GINASPaper, GGCDMPaper). Use for Phase 2 viability
        ASR claims.
      * ``full``  — pre-2026-04-21 heuristic versions (GradInversionFull,
        GINASFull, GGCDMFull). Use only to reproduce the old numbers.
    """
    from adaguard.attacks import (
        GradInversionFull, GINASFull, GINASPaper, GGCDMFull, GGCDMPaper,
    )

    if name == 'gradinversion':
        # GradInversionFull was upgraded in-place (Langevin + warmup +
        # registration-based group consistency), so 'paper' and 'full' map
        # to the same class — just with different defaults.
        kwargs = dict(
            n_iter=n_iter, lr=0.1,
            tv_lambda=1e-4, l2_lambda=1e-6, bn_lambda=0.1,
            n_candidates=n_candidates,
            use_l2_grad=True,
        )
        if variant == 'paper':
            kwargs['warmup_iters'] = 50
            kwargs['langevin_sigma'] = 1e-3
        else:
            kwargs['warmup_iters'] = 0
            kwargs['langevin_sigma'] = 0.0
        return GradInversionFull(model, criterion, device, **kwargs)

    if name == 'gi_nas':
        if variant == 'paper':
            return GINASPaper(
                model, criterion, device,
                n_iter=n_iter, lr=1e-3, n_candidates=max(1, n_candidates),
            )
        return GINASFull(
            model, criterion, device,
            n_iter=n_iter, lr=1e-3, n_restarts=max(1, n_candidates),
        )

    if name == 'ggcdm':
        if variant == 'paper':
            return GGCDMPaper(
                model, criterion, device,
                n_iter=n_iter, guidance_rate=0.5, guidance_scale=1.0,
            )
        return GGCDMFull(
            model, criterion, device,
            n_iter=n_iter, guidance_rate=0.20,
        )

    raise ValueError(f"Unknown attack '{name}'")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--artifact-dir', required=True,
                    help='Phase-1 artifacts dir (contains round_*/)')
    ap.add_argument('--round', type=int, default=249,
                    help='Which round to pull a client from (default 249)')
    ap.add_argument('--client-id', type=int, default=None,
                    help='Specific client id; defaults to the first found')
    ap.add_argument('--attack', required=True,
                    choices=['gradinversion', 'gi_nas', 'ggcdm'])
    ap.add_argument('--n-iter', type=int, default=20000,
                    help='Override iteration count (default: paper budget)')
    ap.add_argument('--n-candidates', type=int, default=None,
                    help='Group-consistency seeds (GradInversion) or NAS '
                         'candidates (GI-NAS). Default: GradInversion=4, '
                         'GI-NAS=5, GGCDM ignored. Pass 1 for fast smoke tests.')
    ap.add_argument('--model', default='resnet18')
    ap.add_argument('--num-classes', type=int, default=10)
    ap.add_argument('--variant', default='paper', choices=['paper', 'full'],
                    help="'paper' uses *Paper classes (default); 'full' falls "
                         "back to *Full for reproducing pre-fix numbers.")
    ap.add_argument('--out', default=None,
                    help='Optional JSON path to dump the metrics dict')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU:    {torch.cuda.get_device_name(0)}")

    art_dir = Path(args.artifact_dir)
    print(f"\nLoading artifact from {art_dir} round {args.round}...")
    data = _load_artifact(art_dir, args.round, args.client_id)
    print(f"  client: {data['client_file']}")

    orig = data['originals']
    labels = data['labels']
    gd = data['gradient_dict']

    if orig is None:
        print("  [WARN] no original images in artifact — PSNR/SSIM will be skipped")
    else:
        print(f"  originals: {tuple(orig.shape)} dtype={orig.dtype}")
    if labels is not None:
        print(f"  labels:    {labels.tolist() if labels.numel() <= 16 else labels.shape}")

    # Move everything to GPU.
    model = create_model(args.model, num_classes=args.num_classes).to(device)
    model.load_state_dict({k: v.to(device) for k, v in data['global_state'].items()})
    model.eval()
    criterion = nn.CrossEntropyLoss()

    gd_gpu = {k: v.to(device) for k, v in gd.items()}
    flat = torch.cat([g.flatten() for g in gd_gpu.values()])
    orig_gpu = orig.to(device) if orig is not None else None
    labels_gpu = labels.to(device) if labels is not None else None
    bs = orig.shape[0] if orig is not None else 1

    # Per-attack default for n_candidates — NAS and group-consistency only
    # work for n > 1.
    if args.n_candidates is None:
        n_cand = {'gradinversion': 4, 'gi_nas': 5, 'ggcdm': 1}[args.attack]
    else:
        n_cand = args.n_candidates

    print(f"\nBuilding attack={args.attack} variant={args.variant} "
          f"n_iter={args.n_iter} n_candidates={n_cand}")
    attack = _build_attack(
        args.attack, model, criterion, device,
        n_iter=args.n_iter, n_candidates=n_cand,
        variant=args.variant,
    )

    print("Running attack...")
    t0 = time.time()
    result = attack.attack(
        gd_gpu, flat, batch_size=bs,
        labels=labels_gpu, original_images=orig_gpu,
    )
    dt = time.time() - t0
    print(f"Attack finished in {dt:.1f}s")

    recon = result.get('reconstructed_images')
    if recon is None:
        print("[ERROR] attack returned no reconstructed_images")
        sys.exit(2)

    metrics = {
        'attack': args.attack,
        'n_iter': args.n_iter,
        'n_candidates': args.n_candidates,
        'time_s': dt,
        'gradient_score': float(result.get('score', 0.0)),
    }

    if orig is not None:
        # Phase-1 saves normalised tensors; denormalise before metric comparison.
        orig_01 = _denormalize(orig_gpu).clamp(0, 1)
        recon_01 = recon.clamp(0, 1)

        mse = compute_mse(orig_01, recon_01)
        psnr = compute_psnr(orig_01, recon_01)
        ssim = compute_ssim(orig_01, recon_01)
        try:
            lp = compute_lpips(orig_01, recon_01)
        except Exception as e:
            print(f"  LPIPS unavailable: {e}")
            lp = None
        metrics.update({'mse': mse, 'psnr': psnr, 'ssim': ssim, 'lpips': lp})

        target = PAPER_TARGETS.get(args.attack, {})
        tgt_psnr = target.get('psnr', float('nan'))
        tgt_label = target.get('label', '?')

        print("\n" + "=" * 60)
        print(f"  SANITY CHECK RESULTS ({args.attack})")
        print("=" * 60)
        print(f"  PSNR        : {psnr:7.2f} dB   (target: {tgt_label})")
        print(f"  SSIM        : {ssim:7.4f}")
        if lp is not None:
            print(f"  LPIPS       : {lp:7.4f}")
        print(f"  MSE         : {mse:.6f}")
        print(f"  grad score  : {metrics['gradient_score']:.4f}")
        print(f"  wallclock   : {dt:.1f}s")

        if not (psnr != psnr):  # not NaN
            delta = psnr - tgt_psnr
            verdict = ("WITHIN PAPER RANGE" if delta >= -2.0 else
                       "BELOW PAPER BY {:.1f} dB — investigate".format(-delta))
            print(f"  verdict     : {verdict}")
        print("=" * 60)

    if args.out:
        import json
        with open(args.out, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Wrote metrics -> {args.out}")


if __name__ == '__main__':
    main()
