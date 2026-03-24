#!/usr/bin/env python3
"""AdaGuard Headless Runner — for HPC batch jobs.

Runs the full FL simulation without Streamlit UI and saves results to JSON.

Usage:
    python run_headless.py --config hpc/config_large.yaml --output /scratch/projects/secure-distributed-ml/results/run1.json
    python run_headless.py --config hpc/config_1k.yaml --strategy maskcrypt --output results.json
    python run_headless.py --config hpc/config_large.yaml --experiment comparison --output comparison.json
"""

import argparse
import json
import time
import sys
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import torch


def make_serializable(obj):
    """Convert numpy/torch types to JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def run_fl_experiment(config_path, strategy, skip_glmip, skip_empirical, output_path,
                      pretrain_override=None, seed_override=None):
    """Run FL rounds and save results."""
    from adaguard.config import load_config, set_seed, get_device
    from adaguard.models import create_model
    from adaguard.data.cifar10 import load_cifar10, partition_data_non_iid
    from adaguard.federation.simulator import FederatedSimulator

    config = load_config(config_path)
    if pretrain_override is not None:
        config['pretrain_epochs'] = pretrain_override
    if seed_override is not None:
        config['seed'] = seed_override
    set_seed(config['seed'])
    device = get_device()

    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print(f"\nConfig: {config_path}")
    print(f"Clients: {config['num_clients']}, per round: {config['clients_per_round']}")
    print(f"Rounds: {config['num_rounds']}, Strategy: {strategy}")
    print(f"Pretrain epochs: {config['pretrain_epochs']}")
    print(f"Skip GLMIP: {skip_glmip}, Skip Empirical: {skip_empirical}")

    # Load data
    train_ds, test_ds = load_cifar10(data_root=os.environ.get('DATA_DIR', './data'))

    # Partition data
    client_data_map = partition_data_non_iid(train_ds, config['num_clients'])

    # Create model
    model = create_model(config.get('model', 'smallcnn'), num_classes=config['num_classes']).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params:,} params")

    # Create simulator
    sim = FederatedSimulator(
        model, train_ds, test_ds, client_data_map, config, device,
    )

    # Pre-train
    print("\n=== Pre-training ===")
    t0 = time.time()
    pretrain_history = sim.pretrain()
    pretrain_time = time.time() - t0
    print(f"Pre-training took {pretrain_time:.1f}s")

    # Run FL rounds
    print(f"\n=== Federated Learning ({config['num_rounds']} rounds) ===")
    round_results = []
    total_start = time.time()

    for rnd in range(config['num_rounds']):
        rnd_start = time.time()
        summary = sim.run_round(
            rnd,
            encryption_strategy=strategy,
            skip_glmip=skip_glmip,
            skip_empirical=skip_empirical,
        )
        rnd_time = time.time() - rnd_start

        # Strip client_details from saved JSON (has tensor refs, too large)
        summary_clean = {k: v for k, v in summary.items() if k != 'client_details'}
        summary_clean['round_time_s'] = rnd_time
        # Save per-client leakscores as a compact list
        clients = summary.get('client_details', [])
        if clients:
            summary_clean['per_client'] = [
                {
                    'id': cd.get('client_id'),
                    'loss': cd.get('metrics', {}).get('loss', 0),
                    'leakscore': cd.get('metrics', {}).get('combined_leakscore', 0),
                    'encrypted_pct': cd.get('metrics', {}).get('actual_pct_encrypted', 0),
                }
                for cd in clients
            ]
        round_results.append(summary_clean)

        acc = summary.get('accuracy', 0) * 100
        leak = summary.get('combined_leakscore', 0)
        enc = summary.get('actual_pct_encrypted', 0) * 100

        # ── Detailed round log ──
        print(f"\n{'─'*70}")
        print(f"  Round {rnd+1:3d}/{config['num_rounds']}  |  Time: {rnd_time:.1f}s")
        print(f"{'─'*70}")

        # Model performance
        loss_val = summary.get('loss', 0)
        print(f"  Model      │ Accuracy: {acc:5.1f}%  Loss: {loss_val:.4f}")

        # LeakScore breakdown
        ent = summary.get('entropy_avg', 0)
        lab = summary.get('label_avg', 0)
        emp = summary.get('empirical_avg', 0)
        print(f"  LeakScore  │ Combined: {leak:.4f}  "
              f"(Entropy: {ent:.3f}  Label: {lab:.3f}  Empirical: {emp:.3f})")

        # Entropy sub-metrics
        sh = summary.get('shannon_leak_score', 0)
        re = summary.get('renyi_leak_score', 0)
        me = summary.get('min_entropy_leak_score', 0)
        print(f"    Entropy  │ Shannon: {sh:.3f}  Renyi: {re:.3f}  MinEnt: {me:.3f}")

        # Label sub-metrics
        gl = summary.get('glmip_score', 0)
        cg = summary.get('confidence_gap', 0)
        cs = summary.get('cosine_leak_score', 0)
        print(f"    Label    │ GLMIP: {gl:.3f}  ConfGap: {cg:.3f}  Cosine: {cs:.3f}")

        # Empirical sub-metrics
        gi = summary.get('empirical_gradinversion', 0)
        gn = summary.get('empirical_ginas', 0)
        gc = summary.get('empirical_ggcdm', 0)
        print(f"    Empirical│ GradInv: {gi:.3f}  GI-NAS: {gn:.3f}  GGCDM: {gc:.3f}")

        # Encryption metrics
        fc = summary.get('fisher_concentration', 0)
        fn = summary.get('fisher_round_norm', 0)
        mv = summary.get('maskcrypt_vulnerability', 0)
        mg = summary.get('magnitude_score', 0)
        print(f"  Fisher     │ Concentration: {fc:.4f}  Normalized: {fn:.4f}")
        print(f"  MaskCrypt  │ Vulnerability: {mv:.4f}")
        print(f"  Magnitude  │ Score: {mg:.4f}")

        # Encryption decision
        level = summary.get('encryption_level', 'N/A')
        pct = summary.get('encryption_pct', 0)
        k = summary.get('params_encrypted', 0)
        print(f"  Encryption │ Level: {level}  Target: {pct*100:.1f}%  "
              f"Actual: {enc:.1f}%  Params: {int(k):,}")

        # Timing breakdown
        t_ent = summary.get('entropy_compute_time', 0)
        t_fish = summary.get('fisher_compute_time', 0)
        t_mc = summary.get('maskcrypt_compute_time', 0)
        t_gl = summary.get('glmip_compute_time', 0)
        t_emp = summary.get('empirical_compute_time', 0)
        print(f"  Timing     │ Entropy: {t_ent:.2f}s  Fisher: {t_fish:.2f}s  "
              f"MaskCrypt: {t_mc:.2f}s  GLMIP: {t_gl:.2f}s  Empirical: {t_emp:.2f}s")

        # Per-weight stats
        fw_mean = summary.get('fisher_per_weight_mean', 0)
        fw_p95 = summary.get('fisher_per_weight_p95', 0)
        mw_mean = summary.get('maskcrypt_per_weight_mean', 0)
        mw_p95 = summary.get('maskcrypt_per_weight_p95', 0)
        if fw_mean > 0 or mw_mean > 0:
            print(f"  Weights    │ Fisher mean: {fw_mean:.6f} p95: {fw_p95:.6f}  "
                  f"MaskCrypt mean: {mw_mean:.6f} p95: {mw_p95:.6f}")

        # Reconstruction quality (if empirical enabled)
        r_mse = summary.get('recon_mse', 0)
        r_psnr = summary.get('recon_psnr', 0)
        r_ssim = summary.get('recon_ssim', 0)
        if r_psnr > 0:
            print(f"  Recon      │ MSE: {r_mse:.4f}  PSNR: {r_psnr:.2f}dB  SSIM: {r_ssim:.4f}")

        # Per-client summary
        clients = summary.get('client_details', [])
        if clients:
            print(f"  Clients    │ {len(clients)} processed")
            for ci, cd in enumerate(clients[:3]):  # show first 3 clients
                cm = cd.get('metrics', {})
                print(f"    Client {cd.get('client_id', ci):3d} │ "
                      f"Loss: {cm.get('loss', 0):.4f}  "
                      f"LeakScore: {cm.get('combined_leakscore', 0):.4f}  "
                      f"Encrypted: {cm.get('actual_pct_encrypted', 0)*100:.1f}%")
            if len(clients) > 3:
                print(f"    ... +{len(clients)-3} more clients")

        # Running stats
        elapsed = time.time() - total_start
        avg_per_round = elapsed / (rnd + 1)
        remaining = avg_per_round * (config['num_rounds'] - rnd - 1)
        print(f"  Progress   │ Elapsed: {elapsed/60:.1f}min  "
              f"Avg/round: {avg_per_round:.1f}s  "
              f"ETA: {remaining/60:.1f}min")

    total_time = time.time() - total_start

    # Assemble output
    output = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'config_path': str(config_path),
            'strategy': strategy,
            'device': str(device),
            'gpu': torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
            'total_params': total_params,
            'seed': config['seed'],
            'skip_glmip': skip_glmip,
            'skip_empirical': skip_empirical,
            'total_time_s': total_time,
            'pretrain_time_s': pretrain_time,
        },
        'config': config,
        'pretrain_history': pretrain_history,
        'rounds': make_serializable(round_results),
    }

    # Save
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n=== Done ===")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"Results saved to: {out_path}")
    return output


def run_comparison(config_path, skip_glmip, skip_empirical, output_path,
                   pretrain_override=None, seed_override=None):
    """Run all 4 strategies and save comparison results."""
    from adaguard.config import load_config, set_seed, get_device
    from adaguard.models import create_model
    from adaguard.data.cifar10 import load_cifar10, partition_data_non_iid
    from adaguard.federation.simulator import FederatedSimulator

    config = load_config(config_path)
    if pretrain_override is not None:
        config['pretrain_epochs'] = pretrain_override
    if seed_override is not None:
        config['seed'] = seed_override
    set_seed(config['seed'])
    device = get_device()

    train_ds, test_ds = load_cifar10(data_root=os.environ.get('DATA_DIR', './data'))
    client_data_map = partition_data_non_iid(train_ds, config['num_clients'])

    base_model = create_model(config.get('model', 'smallcnn'), num_classes=config['num_classes']).to(device)
    base_state = base_model.state_dict()

    strategies = ['none', 'fisher', 'maskcrypt', 'full']
    all_results = {}

    for strategy in strategies:
        print(f"\n{'#'*60}")
        print(f"  Strategy: {strategy.upper()}")
        print(f"{'#'*60}")

        model = create_model(config.get('model', 'smallcnn'), num_classes=config['num_classes']).to(device)
        model.load_state_dict(base_state)

        sim = FederatedSimulator(
            model, train_ds, test_ds, client_data_map, config, device,
        )
        sim.pretrain()

        rounds = []
        for rnd in range(config['num_rounds']):
            summary = sim.run_round(
                rnd, encryption_strategy=strategy,
                skip_glmip=skip_glmip, skip_empirical=skip_empirical,
            )
            clean = {k: v for k, v in summary.items() if k != 'client_details'}
            rounds.append(clean)

            acc = summary.get('accuracy', 0) * 100
            leak = summary.get('combined_leakscore', 0)
            print(f"  Round {rnd+1:3d} — Acc: {acc:5.1f}%, LeakScore: {leak:.4f}")

        all_results[strategy] = rounds

    output = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'config_path': str(config_path),
            'experiment': 'comparison',
            'device': str(device),
        },
        'config': config,
        'strategies': make_serializable(all_results),
    }

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nComparison saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description='AdaGuard Headless Runner for HPC',
    )
    parser.add_argument('--config', '-c', type=str, default=None,
                        help='Path to YAML config file')
    parser.add_argument('--experiment', '-e',
                        choices=['fl_rounds', 'comparison'],
                        default='fl_rounds',
                        help='Experiment type (default: fl_rounds)')
    parser.add_argument('--strategy', '-s',
                        choices=['fisher', 'maskcrypt', 'full', 'none'],
                        default='fisher',
                        help='Encryption strategy (default: fisher)')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output JSON file path')
    parser.add_argument('--skip-glmip', action='store_true',
                        help='Skip GLMIP computation')
    parser.add_argument('--skip-empirical', action='store_true',
                        help='Skip empirical attack scoring')
    parser.add_argument('--pretrain-epochs', type=int, default=None,
                        help='Override pretrain epochs (0 to skip)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Override random seed')

    args = parser.parse_args()

    # Allow CLI override of pretrain epochs
    pretrain_override = args.pretrain_epochs

    if args.experiment == 'comparison':
        run_comparison(args.config, args.skip_glmip, args.skip_empirical,
                       args.output, pretrain_override=pretrain_override,
                       seed_override=args.seed)
    else:
        run_fl_experiment(args.config, args.strategy, args.skip_glmip,
                          args.skip_empirical, args.output,
                          pretrain_override=pretrain_override,
                          seed_override=args.seed)


if __name__ == '__main__':
    main()
