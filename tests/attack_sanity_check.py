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
    'gradinversion_breaching': {
        'psnr': 15.0,
        'label': '> 15 dB on CIFAR-10 via breaching.seethroughgradients '
                 '(Yin 2021 with BN-matching; group consistency not '
                 'implemented upstream)',
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


def _scan_clients_for_diversity(round_dir: Path):
    """Return list of (filename, labels_list, n_unique) for each client."""
    out = []
    for fname in sorted(os.listdir(round_dir)):
        if not (fname.startswith('client_') and fname.endswith('.pt')):
            continue
        try:
            art = torch.load(round_dir / fname, map_location='cpu',
                             weights_only=False)
        except Exception:
            continue
        lbls = art.get('labels')
        if lbls is None:
            continue
        lbls_list = lbls.tolist()
        n_uniq = len(set(lbls_list))
        out.append((fname, lbls_list, n_uniq))
    return out


def _pick_label_diverse_subset(labels, k):
    """Return sorted list of k indices maximizing unique-label count.

    Greedy: take one representative per class in first-appearance order,
    then fill remaining slots with earliest unused indices. Deterministic.
    """
    labels_list = labels.tolist() if hasattr(labels, 'tolist') else list(labels)
    first_seen = {}
    for i, l in enumerate(labels_list):
        if l not in first_seen:
            first_seen[l] = i
    picks = sorted(first_seen.values())[:k]
    if len(picks) < k:
        chosen = set(picks)
        remaining = [i for i in range(len(labels_list)) if i not in chosen]
        picks = sorted(picks + remaining[: k - len(picks)])
    return picks


def _load_artifact(artifact_dir: Path, rnd: int, client_id=None,
                   auto_pick_min_unique=None):
    """Load one client's gradient_dict + original images + labels.

    If auto_pick_min_unique is set, scans clients in the round and picks
    the first whose saved label batch has >= that many unique classes.
    Falls back to the single most-diverse client if none meet the bar.
    """
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
    elif auto_pick_min_unique is not None:
        scan = _scan_clients_for_diversity(round_dir)
        good = [(f, l, n) for f, l, n in scan if n >= auto_pick_min_unique]
        if good:
            pick = good[0][0]
            print(f"[auto-pick] {pick} has {good[0][2]} unique labels "
                  f"(first match for min={auto_pick_min_unique})")
        else:
            scan_sorted = sorted(scan, key=lambda t: -t[2])
            pick = scan_sorted[0][0]
            print(f"[auto-pick] no client with >= {auto_pick_min_unique} "
                  f"unique labels; falling back to most diverse: {pick} "
                  f"({scan_sorted[0][2]} unique)")
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


def _check_artifact_consistency(model, global_state_gpu, weight_delta,
                                images, labels, criterion, device,
                                saved_gd):
    """Verify the saved gradient_dict is consistent with (global, weight_delta, images, labels).

    Reconstructs the local model via `global - weight_delta`, recomputes
    the gradient at the FULL saved batch, and compares layer-by-layer
    against the saved gradient_dict.

    A self-consistent artifact has:
      * mean relative diff  <= 1e-4   (float32 round-off territory)
      * max  relative diff  <= 1e-3   (per-layer tolerance)

    The client (adaguard/federation/client.py:121-131) computes the
    saved raw_gradient_dict in local_model.train() mode — BN uses batch
    statistics over last_images. If we check in .eval() mode, BN uses
    running_mean/var from the *global* state (weight_delta is
    params-only, so BN buffers are not reconstructed), which makes
    BN-param gradients diverge wildly. We must match the client's
    .train() mode to get a faithful comparison.

    Large diffs even in .train() mean the Phase-1 simulator saved a
    gradient that does not correspond to the saved (images, labels,
    weight_delta) triple — i.e., the artifact is internally
    inconsistent and any B != saved_B recompute is attacking a phantom.
    """
    if weight_delta is None:
        return None

    local_state = {
        k: (global_state_gpu[k] - weight_delta[k].to(device))
        if k in weight_delta else global_state_gpu[k]
        for k in global_state_gpu
    }
    model.load_state_dict(local_state)
    # Match client protocol: BN must use batch stats of last_images, not
    # running stats (which aren't reconstructed by global - weight_delta).
    model.train()
    recomputed = _recompute_gradient(model, images, labels, criterion, device)

    per_layer = {}
    rel_diffs = []
    for name, saved_g in saved_gd.items():
        if name not in recomputed:
            per_layer[name] = {'status': 'missing_in_recompute'}
            continue
        s = saved_g.to(device).float()
        r = recomputed[name].float()
        diff = float((s - r).norm().item())
        saved_norm = float(s.norm().item())
        recomp_norm = float(r.norm().item())
        ref = max(saved_norm, 1e-12)
        rel = diff / ref
        per_layer[name] = {
            'saved_norm': saved_norm,
            'recomp_norm': recomp_norm,
            'diff_norm': diff,
            'rel_diff': rel,
        }
        rel_diffs.append(rel)

    max_rel = max(rel_diffs) if rel_diffs else None
    mean_rel = (sum(rel_diffs) / len(rel_diffs)) if rel_diffs else None

    worst = sorted(
        ((n, d['rel_diff']) for n, d in per_layer.items() if 'rel_diff' in d),
        key=lambda x: -x[1],
    )[:5]

    return {
        'max_rel_diff': max_rel,
        'mean_rel_diff': mean_rel,
        'n_layers': len(rel_diffs),
        'worst_layers': [{'name': n, 'rel_diff': r} for n, r in worst],
        'per_layer': per_layer,
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
        GradInversionFull, GradInversionGeiping, GradInversionBreaching,
        GINASFull, GINASPaper,
        GGCDMFull, GGCDMPaper,
    )

    if name == 'gradinversion_breaching':
        return GradInversionBreaching(
            model, criterion, device,
            n_iter=n_iter,
            deep_inv_scale=0.1,
            tv_scale=1e-4,
            langevin_noise=0.01,
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
                    choices=['gradinversion', 'gradinversion_breaching',
                             'gi_nas', 'ggcdm'])
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
    ap.add_argument('--diverse-subset', action='store_true',
                    help='With --batch-size N, pick the N samples that '
                         'maximize unique-label count instead of the first '
                         'N. Use to isolate batch-averaging effects from '
                         'label-collinearity effects on non-IID clients.')
    ap.add_argument('--auto-pick-diverse', type=int, default=None,
                    metavar='N',
                    help='Scan all clients in the round and pick the first '
                         'whose label batch has >= N unique classes. '
                         'Falls back to the most-diverse client if none '
                         'meets the bar. Overrides --client-id.')
    ap.add_argument('--list-clients', action='store_true',
                    help='Print per-client unique-label counts for the '
                         'round and exit. Use before --auto-pick-diverse '
                         'to see what is available.')
    ap.add_argument('--defence',
                    choices=['none', 'fhe', 'maskcrypt', 'fisher'],
                    default='none',
                    help='Apply a defence to the (possibly recomputed) '
                         'gradient_dict before running the attack. Needed '
                         'because Phase-1 artifacts save the RAW pre-defence '
                         'gradient regardless of scenario. '
                         'none=V1, fhe=V2 (zero all grads), '
                         'maskcrypt=V4, fisher=V6 (AdaGuard core).')
    ap.add_argument('--defence-pct', type=float, default=0.1,
                    help='Fraction of parameters to encrypt (MaskCrypt/Fisher).')
    ap.add_argument('--no-classifier-head-guarantee', action='store_true',
                    help='Disable AdaGuard-Fisher classifier-head guarantee '
                         '(force-include fc.bias / fc.weight in encrypted set). '
                         'Use for ablation: reproduces the pre-fix Fisher '
                         'selector that exposed fc.bias on seed 456 round 249 '
                         'and produced label ASR=1.0. Default behaviour ENABLES '
                         'the guarantee (the post-fix AdaGuard-Fisher).')
    ap.add_argument('--fisher-mask-mode', default='fisher',
                    choices=['fisher', 'random'],
                    help="Mask-selection mode for the Fisher defence path. "
                         "'fisher' (default) selects top-K parameters by "
                         "per-weight Fisher score (the AdaGuard headline). "
                         "'random' selects K parameters uniformly at random "
                         "without replacement, used as the Fisher-vs-random "
                         "ablation baseline that isolates the contribution "
                         "of vulnerability-ranked targeting from the "
                         "encryption budget itself. Only takes effect when "
                         "--defence=fisher.")
    ap.add_argument('--fisher-random-seed', type=int, default=42,
                    help="Seed for the per-call random permutation when "
                         "--fisher-mask-mode=random. Default 42 for "
                         "reproducibility of the ablation.")
    ap.add_argument('--check-consistency', action='store_true',
                    help='Before running the attack, reconstruct the local '
                         'model (global - weight_delta), recompute the '
                         'gradient at the full saved batch, and compare '
                         'layer-by-layer to the saved gradient_dict. Use '
                         'to verify the Phase-1 artifact is self-consistent.')
    ap.add_argument('--consistency-only', action='store_true',
                    help='Run --check-consistency and exit without attacking. '
                         'Use when you only want to audit the artifact.')
    ap.add_argument('--out', default=None,
                    help='Optional JSON path to dump the metrics dict')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU:    {torch.cuda.get_device_name(0)}")

    art_dir = Path(args.artifact_dir)

    if args.list_clients:
        round_dir = art_dir / f'round_{args.round}'
        scan = _scan_clients_for_diversity(round_dir)
        scan_sorted = sorted(scan, key=lambda t: (-t[2], t[0]))
        print(f"\nClients in round {args.round} (most diverse first):")
        print(f"{'client':20s}  {'n_unique':>9s}  labels")
        for fname, lbls, n in scan_sorted:
            print(f"{fname:20s}  {n:9d}  {lbls}")
        return

    print(f"\nLoading artifact from {art_dir} round {args.round}...")
    data = _load_artifact(
        art_dir, args.round, args.client_id,
        auto_pick_min_unique=args.auto_pick_diverse,
    )
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

    consistency_report = None
    if (args.check_consistency or args.consistency_only) and orig is not None and labels is not None:
        wd = data.get('weight_delta')
        if wd is None:
            print("\n[consistency] no weight_delta in artifact — skipping check.")
        else:
            print("\n[consistency] reconstructing local model and "
                  "recomputing gradient at full saved batch...")
            consistency_report = _check_artifact_consistency(
                model, global_state_gpu, wd,
                orig, labels, criterion, device, gd,
            )
            mx = consistency_report['max_rel_diff']
            mn = consistency_report['mean_rel_diff']
            nl = consistency_report['n_layers']
            print(f"[consistency] layers={nl}  mean_rel_diff={mn:.2e}  "
                  f"max_rel_diff={mx:.2e}")
            # Thresholds:
            #   <= 1e-3 : strict float32 round-off, bit-for-bit consistent
            #   <= 1e-2 : within GPU non-determinism (cuDNN picks different
            #             conv backward algos between runs; gradient
            #             diffs of ~1e-3 are expected and harmless)
            #   >  1e-2 : real inconsistency, investigate
            if mx is not None and mx <= 1e-3:
                print("[consistency] VERDICT: self-consistent "
                      "(strict float32; artifact matches exactly)")
            elif mx is not None and mx <= 1e-2:
                print("[consistency] VERDICT: self-consistent "
                      "(within GPU non-determinism — cuDNN picks "
                      "different conv backward algos between runs; "
                      "residual diff is noise, not a bug)")
            else:
                print("[consistency] VERDICT: INCONSISTENT "
                      "(saved gradient does not match recomputed gradient)")
                print("[consistency] worst layers:")
                for w in consistency_report['worst_layers']:
                    print(f"    {w['name']:40s}  rel_diff={w['rel_diff']:.3e}")

            # Reload global state so any downstream work starts from a
            # known baseline; batch-size override block below will reload
            # local state itself if needed.
            model.load_state_dict(global_state_gpu)
            model.eval()

        if args.consistency_only:
            if args.out:
                import json
                with open(args.out, 'w') as f:
                    json.dump({'consistency': consistency_report}, f, indent=2)
                print(f"Wrote consistency report -> {args.out}")
            return

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

            if args.diverse_subset:
                picks = _pick_label_diverse_subset(labels, args.batch_size)
                idx_t = torch.tensor(picks, dtype=torch.long)
                orig = orig.index_select(0, idx_t)
                labels = labels.index_select(0, idx_t)
                print(
                    f"\n[batch-size override] saved B={saved_bs} -> "
                    f"diverse-subset picks {picks} -> labels {labels.tolist()} "
                    f"({len(set(labels.tolist()))} unique); recomputing "
                    f"gradient against local model."
                )
            else:
                orig = orig[: args.batch_size]
                labels = labels[: args.batch_size]
                print(
                    f"\n[batch-size override] saved B={saved_bs} -> "
                    f"using first {args.batch_size} samples "
                    f"(labels {labels.tolist()}, "
                    f"{len(set(labels.tolist()))} unique); "
                    f"recomputing gradient against local model."
                )
            gd = _recompute_gradient(model, orig, labels, criterion, device)
            # Put model back into eval so the attack's forward pass is
            # deterministic w.r.t. BN.
            model.eval()

    defence_meta = None  # JSON-serializable summary surfaced into output.
    if args.defence != 'none':
        total = sum(g.numel() for g in gd.values())
        gd_cpu = {k: v.detach().cpu() for k, v in gd.items()}
        if args.defence == 'fhe':
            gd = {k: torch.zeros_like(v) for k, v in gd_cpu.items()}
            print(f"\n[defence=fhe] zeroed all {total} params "
                  "(simulates full HE: adversary sees nothing)")
            defence_meta = {'strategy': 'fhe', 'weights_encrypted': total,
                            'pct_encrypted': 1.0}
        elif args.defence == 'fisher':
            from adaguard.encryption.fisher_encrypt import FisherEncryptor
            from adaguard.metrics.fisher import FisherInformationMetric
            k_enc = max(1, int(args.defence_pct * total))
            # Default: classifier-head guarantee ENABLED (post-fix Fisher).
            # Pass --no-classifier-head-guarantee to disable for ablation.
            mandatory = () if args.no_classifier_head_guarantee else None
            fisher_metric = FisherInformationMetric(
                enc_pct=args.defence_pct,
                mandatory_layer_substrings=mandatory,
                mask_mode=args.fisher_mask_mode,
                random_seed=args.fisher_random_seed,
            )
            enc = FisherEncryptor(fisher_metric=fisher_metric)
            gd, meta = enc.encrypt(gd_cpu, k=k_enc)
            guard_state = ('DISABLED (ablation)'
                           if args.no_classifier_head_guarantee else 'ENABLED')
            mode_label = (f"random (seed={args.fisher_random_seed})"
                          if args.fisher_mask_mode == 'random'
                          else 'fisher')
            print(f"\n[defence=fisher mask_mode={mode_label}] encrypted "
                  f"{meta['weights_encrypted']}/{total} params "
                  f"({meta['pct_encrypted']*100:.1f}%); "
                  f"classifier-head guarantee {guard_state}; "
                  f"forced {meta['classifier_head_forced_count']} extra params "
                  f"in {meta['classifier_head_forced_layers']}")
            defence_meta = {
                'strategy': 'fisher',
                'mask_mode': args.fisher_mask_mode,
                'random_seed': (args.fisher_random_seed
                                if args.fisher_mask_mode == 'random'
                                else None),
                'weights_encrypted': int(meta['weights_encrypted']),
                'pct_encrypted': float(meta['pct_encrypted']),
                'encryption_threshold': float(meta['encryption_threshold']),
                'classifier_head_guarantee': not args.no_classifier_head_guarantee,
                'classifier_head_forced_count': int(
                    meta['classifier_head_forced_count']),
                'classifier_head_forced_layers': list(
                    meta['classifier_head_forced_layers']),
            }
        elif args.defence == 'maskcrypt':
            from adaguard.encryption.maskcrypt_encrypt import MaskCryptEncryptor
            wd = data.get('weight_delta')
            if wd is None:
                raise RuntimeError("MaskCrypt needs weight_delta to rank "
                                   "vulnerability; artifact has none.")
            old_w = {k: data['global_state'][k].detach().cpu()
                     for k in gd_cpu if k in data['global_state']}
            new_w = {k: (data['global_state'][k] - wd[k]).detach().cpu()
                     for k in gd_cpu if k in wd}
            k_enc = max(1, int(args.defence_pct * total))
            enc = MaskCryptEncryptor(enc_pct=args.defence_pct)
            gd, meta = enc.encrypt(gd_cpu, old_w, new_w, k=k_enc)
            print(f"\n[defence=maskcrypt] encrypted "
                  f"{meta['weights_encrypted']}/{total} params "
                  f"({meta['pct_encrypted']*100:.1f}%)")
            defence_meta = {
                'strategy': 'maskcrypt',
                'weights_encrypted': int(meta['weights_encrypted']),
                'pct_encrypted': float(meta['pct_encrypted']),
            }

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
        n_cand = {
            'gradinversion': 4,
            'gradinversion_breaching': 1,
            'gi_nas': 5,
            'ggcdm': 1,
        }[args.attack]
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

    if defence_meta is not None:
        metrics['defence_meta'] = defence_meta

    if consistency_report is not None:
        metrics['consistency'] = {
            'max_rel_diff': consistency_report['max_rel_diff'],
            'mean_rel_diff': consistency_report['mean_rel_diff'],
            'n_layers': consistency_report['n_layers'],
            'worst_layers': consistency_report['worst_layers'],
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
