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
from adaguard.attacks.label_recovery import label_recovery_summary  # noqa: E402

# CIFAR-10 normalization used by Phase-1 training data loader.
CIFAR_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
CIFAR_STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)

# Paper PSNR targets on CIFAR-10 (undefended). Verified against the actual
# papers on 2026-04-22 (see docs/attack_restructure_2.pdf §2). None of the
# three papers report "ASR" as a headline — all three use pixel PSNR /
# LPIPS / SSIM, at batch sizes the papers themselves specify:
#
# GradInversion (Yin 2021): CIFAR-10 is NOT in the main tables. Headline
#   numbers are on ImageNet 224x224 at B=8-48 (post-registration PSNR /
#   LPIPS / IIP). Label accuracy is a *sub-component* (99.56% at B=8),
#   not the main claim. Our functional target on CIFAR is "well above
#   noise floor" = 15 dB. Attack uses the Geiping 2020 recipe.
# GI-NAS (Yu 2025, arXiv 2405.20725 v4): **35.99 dB PSNR / 0.998 SSIM /
#   0.0009 LPIPS on CIFAR-10 at B=4** (Table III). The earlier 30.53
#   figure was from an older arXiv version at B=16. Our DIP proxy (no
#   per-batch NAS search) is expected to land below 36 dB; we'll
#   benchmark it at B=4 to match the paper's protocol.
# GGCDM (Meng 2025): CIFAR is not evaluated. Headline 41.12 dB is CelebA
#   256x256 B=1 with an FFHQ prior. Running our CIFAR-pretrained DDPM
#   version at B=1 is the closest protocol-match; the 41 dB number is
#   not directly applicable to CIFAR.
PAPER_TARGETS = {
    'gradinversion': {
        'psnr': 15.0,
        'label': '> 15 dB on CIFAR-10 (Yin 2021 headline is ImageNet-224 '
                 'post-registration PSNR; no CIFAR-10 B=16 target in paper)',
    },
    'gi_nas': {
        'psnr': 35.99,
        'label': '35.99 dB at B=4 (Yu 2025 Table III, arXiv v4). DIP '
                 'proxy expected below this; NAS search omitted.',
    },
    'ggcdm': {
        'psnr': 15.0,
        'label': 'CIFAR not evaluated by Meng 2025 (paper reports '
                 '41.12 dB on CelebA-256 B=1). Functional target 15 dB.',
    },
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
    weight_delta = art.get('weight_delta')

    if gd is None:
        raise RuntimeError(f"{pick} has no gradient_dict")

    return {
        'global_state': global_state,
        'gradient_dict': gd,
        'weight_delta': weight_delta,
        'originals': originals,
        'labels': labels,
        'client_file': pick,
    }


def _recompute_gradient(model, images, labels, criterion, device):
    """Recompute the client gradient at an arbitrary batch size.

    Matches the Phase-1 client-side protocol (see
    adaguard/federation/client.py lines 121-131): forward on the images,
    cross-entropy against the labels, backward, snapshot grads. Caller
    is responsible for having loaded the LOCAL model state (i.e.,
    global_state - weight_delta) into `model` beforehand.
    """
    model.zero_grad()
    imgs = images.to(device)
    lbls = labels.to(device)
    out = model(imgs)
    loss = criterion(out, lbls)
    loss.backward()
    gd = {
        name: p.grad.clone().detach()
        for name, p in model.named_parameters()
        if p.grad is not None
    }
    return gd


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
        GradInversionFull, GradInversionGeiping,
        GINASFull, GINASPaper,
        GGCDMFull, GGCDMPaper,
    )

    if name == 'gradinversion':
        # Attack Restructure 1 (2026-04-22): variant='paper' routes to the
        # Geiping 2020 Inverting-Gradients recipe (cosine + signed + TV).
        # Our Yin-faithful extensions (BN match, Langevin, group
        # consistency) were net-negative in ablations. variant='full'
        # keeps the Yin-extended GradInversionFull for reproducing the
        # pre-restructure numbers.
        if variant == 'paper':
            return GradInversionGeiping(
                model, criterion, device,
                n_iter=n_iter, lr=0.1, tv_lambda=1e-4,
            )
        kwargs = dict(
            n_iter=n_iter, lr=0.1,
            tv_lambda=1e-4, l2_lambda=1e-6, bn_lambda=0.1,
            n_candidates=n_candidates,
            use_l2_grad=True,
            warmup_iters=0,
            langevin_sigma=0.0,
        )
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
            # Restructure 1: mr=0.20 (Meng Table 6), guidance_scale=1.0,
            # DPS sqrt(L) scaling baked into GGCDMPaper.attack().
            return GGCDMPaper(
                model, criterion, device,
                n_iter=n_iter, guidance_rate=0.20, guidance_scale=1.0,
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
    ap.add_argument('--batch-size', type=int, default=None,
                    help='Override the client batch size. If set and '
                         '< saved batch, the first N images/labels are '
                         'kept and the gradient is recomputed against '
                         'the saved local model (global - weight_delta). '
                         'Use to reproduce paper-matched protocols: '
                         'GI-NAS B=4, GradInversion B=1/8, GGCDM B=1.')
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

    # Build the model. If --batch-size is set we reload with the LOCAL
    # weights (= global - weight_delta) so the recomputed gradient matches
    # Phase-1's post-training protocol; otherwise we just use global_state
    # (the saved full-batch gradient_dict was already computed against the
    # local model, so global_state is only used as a scaffold for the
    # attack forward pass).
    model = create_model(args.model, num_classes=args.num_classes).to(device)
    global_state_gpu = {k: v.to(device) for k, v in data['global_state'].items()}
    model.load_state_dict(global_state_gpu)
    model.eval()
    criterion = nn.CrossEntropyLoss()

    # Batch-size override: subsample and recompute the gradient against
    # the local model. This lets us probe paper-matched protocols
    # (GI-NAS B=4, GradInversion B=1/8) using the existing Phase-1
    # artifacts — no rerun required.
    if args.batch_size is not None and orig is not None:
        saved_bs = orig.shape[0]
        if args.batch_size > saved_bs:
            raise ValueError(
                f"--batch-size {args.batch_size} > saved batch {saved_bs}; "
                "cannot upsample."
            )
        if args.batch_size < saved_bs:
            wd = data.get('weight_delta')
            if wd is None:
                raise RuntimeError(
                    "Artifact has no weight_delta — cannot reconstruct "
                    "local model for gradient recompute."
                )
            local_state = {
                k: (global_state_gpu[k] - wd[k].to(device))
                if k in wd else global_state_gpu[k]
                for k in global_state_gpu
            }
            model.load_state_dict(local_state)
            model.eval()
            orig = orig[: args.batch_size]
            labels = labels[: args.batch_size]
            print(
                f"\n[batch-size override] saved B={saved_bs} -> "
                f"using first {args.batch_size} samples; "
                f"recomputing gradient against local model."
            )
            gd = _recompute_gradient(model, orig, labels, criterion, device)
            # Put model back into eval so the attack's forward pass is
            # deterministic w.r.t. BN.
            model.eval()

    gd_gpu = {k: v.to(device) for k, v in gd.items()}
    flat = torch.cat([g.flatten() for g in gd_gpu.values()])
    orig_gpu = orig.to(device) if orig is not None else None
    labels_gpu = labels.to(device) if labels is not None else None
    bs = orig.shape[0] if orig is not None else 1

    # Analytical label-recovery (iDLG / LLG+). Diagnostic side-channel —
    # not the paper's headline metric (that's pixel PSNR/LPIPS/SSIM), but
    # a useful signal because it's free off the same gradient_dict and
    # tells us whether the classifier gradient is leaking.
    label_rec = None
    if labels is not None:
        label_rec = label_recovery_summary(
            gd_gpu, labels_gpu, num_classes=args.num_classes,
        )
        print(
            f"\nLabel recovery (LLG+ side-channel): "
            f"asr={label_rec['asr']:.3f}  "
            f"source={label_rec['source']}  "
            f"key={label_rec['key_used']}  "
            f"signal={label_rec['signal']:.4f}"
        )

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

    if label_rec is not None:
        metrics['label_recovery'] = label_rec

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
        if label_rec is not None:
            print(
                f"  label ASR   : {label_rec['asr']:7.4f}   "
                f"(source={label_rec['source']}, "
                f"signal={label_rec['signal']:.4f})"
            )
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
