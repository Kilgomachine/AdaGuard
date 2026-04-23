#!/usr/bin/env python3
"""Run iDLG / LLG+ label recovery across defence scenarios.

The Phase-1 pipeline (see adaguard/federation/simulator.py) always saves
the RAW pre-defence gradient to each client_*.pt artifact — defences
only affect what gets aggregated server-side, never what gets written
to disk. So reading gradient_dict directly would give identical LLG+
ASR across artifacts_{none,full,maskcrypt,fisher}.

To simulate what the attacker actually sees under each defence, this
script re-applies the encryption strategy at sweep time:

  none    : identity (raw gradient; V1 baseline).
  full    : zero every gradient tensor (V2 full HE).
  fisher  : AdaGuard-style top-k by g^2 zeroed out (V6).
  maskcrypt: top-k by |grad * (old - new)| zeroed (V4; needs
             global_model.pt to reconstruct old/new weights).

All four scenarios can be computed off the SAME raw artifacts_none_*
directory since the underlying gradient is identical — the defence
just changes the attacker-visible subset.

Example
-------
# Compare V1 / V2 / V6 (skip V4 MaskCrypt for speed):
python tests/label_recovery_sweep.py \\
    --round 249 --num-classes 10 \\
    --artifact-dir /scratch/.../artifacts_none_seed42_300clients \\
    --scenario V1=none \\
    --scenario V2=full \\
    --scenario V6=fisher:0.10 \\
    --out results/label_recovery_sweep.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adaguard.attacks.label_recovery import label_recovery_summary  # noqa: E402
from adaguard.encryption.fisher_encrypt import FisherEncryptor  # noqa: E402
from adaguard.encryption.maskcrypt_encrypt import MaskCryptEncryptor  # noqa: E402
from adaguard.metrics.fisher import FisherInformationMetric  # noqa: E402
from adaguard.metrics.maskcrypt import MaskCryptMetric  # noqa: E402


def _parse_scenario(entry: str):
    """Parse one --scenario spec.

    Format: 'label=strategy' or 'label=strategy:pct'
      label      = display label (V1, V2, V6, ...)
      strategy   = none | full | fisher | maskcrypt
      pct        = fraction in (0, 1]; required for fisher/maskcrypt
    """
    if '=' not in entry:
        raise ValueError(f"scenario '{entry}' needs label=strategy form")
    label, rest = entry.split('=', 1)
    label = label.strip()
    rest = rest.strip()
    if ':' in rest:
        strategy, pct_s = rest.split(':', 1)
        pct = float(pct_s)
    else:
        strategy, pct = rest, None
    strategy = strategy.lower()
    if strategy not in ('none', 'full', 'fisher', 'maskcrypt'):
        raise ValueError(f"unknown strategy '{strategy}' in scenario '{entry}'")
    if strategy in ('fisher', 'maskcrypt') and pct is None:
        pct = 0.10
    return {'label': label, 'strategy': strategy, 'pct': pct}


def _total_params(gd: Dict[str, torch.Tensor]) -> int:
    return sum(g.numel() for g in gd.values())


def _apply_defence(
    gd: Dict[str, torch.Tensor],
    scenario: dict,
    *,
    fisher_enc: FisherEncryptor,
    mc_enc: MaskCryptEncryptor,
    old_weights: Optional[Dict[str, torch.Tensor]] = None,
    new_weights: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, torch.Tensor]:
    """Return the attacker-visible gradient dict under the given scenario."""
    strat = scenario['strategy']
    if strat == 'none':
        return {k: v.clone() for k, v in gd.items()}
    if strat == 'full':
        return {k: torch.zeros_like(v) for k, v in gd.items()}
    if strat == 'fisher':
        k_int = max(1, int(round(scenario['pct'] * _total_params(gd))))
        protected, _ = fisher_enc.encrypt(gd, k=k_int)
        return protected
    if strat == 'maskcrypt':
        if old_weights is None or new_weights is None:
            # Can't run MaskCrypt without weights — fall through to 'none'
            # with a warning.
            print(f"  [WARN] maskcrypt needs old/new weights; skipping for this client")
            return {k: v.clone() for k, v in gd.items()}
        k_int = max(1, int(round(scenario['pct'] * _total_params(gd))))
        protected, _ = mc_enc.encrypt(gd, old_weights, new_weights, k=k_int)
        return protected
    raise ValueError(f"unhandled strategy '{strat}'")


def _iter_clients(round_dir: Path, max_clients: Optional[int] = None):
    clients = sorted(
        f for f in os.listdir(round_dir)
        if f.startswith('client_') and f.endswith('.pt')
    )
    if max_clients is not None:
        clients = clients[:max_clients]
    return clients


def _load_global(round_dir: Path) -> Optional[Dict[str, torch.Tensor]]:
    p = round_dir / 'global_model.pt'
    if not p.exists():
        return None
    return torch.load(p, map_location='cpu', weights_only=False)


def _run_scenario(scenario: dict, artifact_dir: Path, rnd: int,
                  num_classes: int, max_clients: Optional[int],
                  fisher_enc: FisherEncryptor,
                  mc_enc: MaskCryptEncryptor):
    round_dir = artifact_dir / f'round_{rnd}'
    if not round_dir.exists():
        print(f"[{scenario['label']}] MISSING {round_dir}")
        return None

    global_state = _load_global(round_dir)
    clients = _iter_clients(round_dir, max_clients=max_clients)
    if not clients:
        print(f"[{scenario['label']}] no client_*.pt in {round_dir}")
        return None

    per_client = []
    for fname in clients:
        art = torch.load(round_dir / fname, map_location='cpu', weights_only=False)
        gd_raw = art.get('gradient_dict')
        labels = art.get('labels')
        if gd_raw is None or labels is None:
            continue

        # MaskCrypt needs old/new weights. old = global_state, new = local.
        # weight_delta = global - local => local = global - weight_delta.
        old_w = new_w = None
        if scenario['strategy'] == 'maskcrypt' and global_state is not None:
            weight_delta = art.get('weight_delta')
            if weight_delta is not None:
                old_w = {k: v for k, v in global_state.items() if k in gd_raw}
                new_w = {
                    k: (old_w[k] - weight_delta[k]) if k in weight_delta else old_w[k]
                    for k in old_w
                }

        gd_attacker = _apply_defence(
            gd_raw, scenario,
            fisher_enc=fisher_enc, mc_enc=mc_enc,
            old_weights=old_w, new_weights=new_w,
        )

        res = label_recovery_summary(gd_attacker, labels, num_classes=num_classes)
        per_client.append({
            'client_file': fname,
            'asr': res['asr'],
            'source': res['source'],
            'key_used': res['key_used'],
            'signal': res['signal'],
            'batch_size': len(res['truth']),
        })

    if not per_client:
        print(f"[{scenario['label']}] no usable clients")
        return None

    asrs = [c['asr'] for c in per_client]
    signals = [c['signal'] for c in per_client]
    sources = {c['source'] for c in per_client}
    keys = {c['key_used'] for c in per_client if c['key_used']}
    mean = sum(asrs) / len(asrs)
    mn = min(asrs)
    mx = max(asrs)
    pct_desc = f":{scenario['pct']:.2f}" if scenario['pct'] is not None else ""
    print(
        f"[{scenario['label']:>4}] strategy={scenario['strategy']}{pct_desc:>5}  "
        f"clients={len(per_client):3d}  "
        f"mean ASR={mean:.4f}  min={mn:.4f}  max={mx:.4f}  "
        f"signal={sum(signals) / len(signals):.4f}  "
        f"src={'/'.join(sorted(sources))}  keys={'/'.join(sorted(keys))}"
    )

    return {
        'label': scenario['label'],
        'strategy': scenario['strategy'],
        'pct': scenario['pct'],
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
        '--artifact-dir', required=True,
        help="Raw-gradient source dir (typically artifacts_none_*). "
             "All scenarios read from this same dir — the defence is "
             "applied at sweep-time to simulate the attacker view.",
    )
    ap.add_argument(
        '--scenario', action='append', default=[],
        help="Repeatable. Format: 'label=strategy' or 'label=strategy:pct'. "
             "Strategies: none, full, fisher, maskcrypt. "
             "Example: --scenario V6=fisher:0.10",
    )
    ap.add_argument('--round', type=int, default=249)
    ap.add_argument('--num-classes', type=int, default=10)
    ap.add_argument('--max-clients', type=int, default=None,
                    help="Optional cap on #clients per scenario (default: all).")
    ap.add_argument('--out', default=None,
                    help="Optional JSON path for the full report.")
    args = ap.parse_args()

    if not args.scenario:
        ap.error("Pass at least one --scenario label=strategy[:pct] entry.")

    scenarios = [_parse_scenario(s) for s in args.scenario]

    # Shared encryptor instances — re-used across every client.
    fisher_enc = FisherEncryptor(fisher_metric=FisherInformationMetric())
    mc_enc = MaskCryptEncryptor(maskcrypt_metric=MaskCryptMetric())

    artifact_dir = Path(args.artifact_dir)
    print(f"Label recovery sweep: round={args.round}  "
          f"num_classes={args.num_classes}  dir={artifact_dir}")
    print("=" * 80)

    report = {
        'round': args.round,
        'num_classes': args.num_classes,
        'artifact_dir': str(artifact_dir),
        'scenarios': [],
    }
    for sc in scenarios:
        out = _run_scenario(
            sc, artifact_dir, args.round,
            num_classes=args.num_classes,
            max_clients=args.max_clients,
            fisher_enc=fisher_enc, mc_enc=mc_enc,
        )
        if out is not None:
            report['scenarios'].append(out)

    print("=" * 80)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Wrote full report -> {args.out}")


if __name__ == '__main__':
    main()
