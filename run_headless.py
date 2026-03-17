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


def run_fl_experiment(config_path, strategy, skip_glmip, skip_empirical, output_path):
    """Run FL rounds and save results."""
    from adaguard.config import load_config, set_seed, get_device
    from adaguard.models.cnn import SmallCNN
    from adaguard.data.cifar10 import load_cifar10, partition_data_non_iid
    from adaguard.federation.simulator import FederatedSimulator

    config = load_config(config_path)
    set_seed(config['seed'])
    device = get_device()

    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print(f"\nConfig: {config_path}")
    print(f"Clients: {config['num_clients']}, per round: {config['clients_per_round']}")
    print(f"Rounds: {config['num_rounds']}, Strategy: {strategy}")
    print(f"Skip GLMIP: {skip_glmip}, Skip Empirical: {skip_empirical}")

    # Load data
    train_ds, test_ds = load_cifar10(data_root=os.environ.get('DATA_DIR', './data'))

    # Partition data
    client_data_map = partition_data_non_iid(train_ds, config['num_clients'])

    # Create model
    model = SmallCNN(num_classes=config['num_classes']).to(device)
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

        # Strip client_details to save space (has tensor refs)
        summary_clean = {k: v for k, v in summary.items() if k != 'client_details'}
        summary_clean['round_time_s'] = rnd_time
        round_results.append(summary_clean)

        acc = summary.get('accuracy', 0) * 100
        leak = summary.get('combined_leakscore', 0)
        enc = summary.get('actual_pct_encrypted', 0) * 100
        print(f"  Round {rnd+1:3d}/{config['num_rounds']} — "
              f"Acc: {acc:5.1f}%, LeakScore: {leak:.4f}, "
              f"Encrypted: {enc:5.1f}%, Time: {rnd_time:.1f}s")

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


def run_comparison(config_path, skip_glmip, skip_empirical, output_path):
    """Run all 4 strategies and save comparison results."""
    from adaguard.config import load_config, set_seed, get_device
    from adaguard.models.cnn import SmallCNN
    from adaguard.data.cifar10 import load_cifar10, partition_data_non_iid
    from adaguard.federation.simulator import FederatedSimulator

    config = load_config(config_path)
    set_seed(config['seed'])
    device = get_device()

    train_ds, test_ds = load_cifar10(data_root=os.environ.get('DATA_DIR', './data'))
    client_data_map = partition_data_non_iid(train_ds, config['num_clients'])

    base_model = SmallCNN(num_classes=config['num_classes']).to(device)
    base_state = base_model.state_dict()

    strategies = ['none', 'fisher', 'maskcrypt', 'full']
    all_results = {}

    for strategy in strategies:
        print(f"\n{'#'*60}")
        print(f"  Strategy: {strategy.upper()}")
        print(f"{'#'*60}")

        model = SmallCNN(num_classes=config['num_classes']).to(device)
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

    args = parser.parse_args()

    if args.experiment == 'comparison':
        run_comparison(args.config, args.skip_glmip, args.skip_empirical, args.output)
    else:
        run_fl_experiment(args.config, args.strategy, args.skip_glmip,
                          args.skip_empirical, args.output)


if __name__ == '__main__':
    main()
