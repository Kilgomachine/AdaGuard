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

# Force line-buffered stdout so `tail -f` sees output immediately
if not os.environ.get('PYTHONUNBUFFERED'):
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
    sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)


class TeeLogger:
    """Write to both stdout and a log file. Survives SSH disconnects."""
    def __init__(self, log_path):
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(log_path) or '.', exist_ok=True)
        self.log = open(log_path, 'a', buffering=1)  # line-buffered
        self.log.write(f"\n{'='*70}\n")
        self.log.write(f"  Session started: {datetime.now().isoformat()}\n")
        self.log.write(f"{'='*70}\n")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def fileno(self):
        return self.terminal.fileno()

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
                      pretrain_override=None, seed_override=None,
                      num_clients_override=None, clients_per_round_override=None,
                      num_rounds_override=None, client_lr_override=None,
                      defer_attacks=False):
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
    if num_clients_override is not None:
        config['num_clients'] = num_clients_override
    if clients_per_round_override is not None:
        config['clients_per_round'] = clients_per_round_override
    if num_rounds_override is not None:
        config['num_rounds'] = num_rounds_override
    if client_lr_override is not None:
        config['client_lr'] = client_lr_override
    set_seed(config['seed'])
    device = get_device()

    gpu_name = ''
    if device.type == 'cuda':
        n_gpu = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_mem / 1e9 if hasattr(torch.cuda.get_device_properties(0), 'total_mem') else torch.cuda.get_device_properties(0).total_memory / 1e9
    else:
        n_gpu = 0
        vram = 0

    print(f"\n{'='*70}")
    print(f"  AdaGuard Federated Learning Experiment")
    print(f"{'='*70}")
    print(f"  Device       {device} ({n_gpu}x {gpu_name}, {vram:.0f}GB)" if n_gpu else f"  Device       {device}")
    print(f"  Strategy     {strategy}")
    print(f"  Clients      {config['num_clients']} total, {config['clients_per_round']}/round")
    print(f"  Rounds       {config['num_rounds']}")
    print(f"  Client LR    {config.get('client_lr', config.get('fl_lr', 0.01))}")
    print(f"  Local steps  {config.get('client_local_steps', 1)}")
    print(f"  Seed         {config['seed']}")
    flags = []
    if skip_glmip:
        flags.append("skip GLMIP")
    if skip_empirical:
        flags.append("skip Empirical")
    if defer_attacks:
        flags.append("DEFER ATTACKS (train only, run attacks later)")
    if flags:
        print(f"  Flags      {', '.join(flags)}")
    print(f"{'='*70}\n")

    # Load data
    train_ds, test_ds = load_cifar10(data_root=os.environ.get('DATA_DIR', './data'))

    # Partition data
    client_data_map = partition_data_non_iid(
        train_ds, config['num_clients'],
        classes_per_client=config.get('classes_per_client', 2),
        min_samples=config.get('min_samples_per_client', 20),
    )

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
    # If deferring attacks, save client data next to output for later processing
    attack_save_dir = None
    if defer_attacks:
        attack_save_dir = str(Path(output_path).parent / f'client_data_{Path(output_path).stem}')
        print(f"  Attack data will be saved to: {attack_save_dir}")
        skip_glmip = True
        skip_empirical = True

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
            save_dir=attack_save_dir,
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
        loss_val = summary.get('loss', 0)

        # Timing
        elapsed = time.time() - total_start
        avg_per_round = elapsed / (rnd + 1)
        remaining = avg_per_round * (config['num_rounds'] - rnd - 1)

        # ── Clean round summary ──
        n = config['num_rounds']
        bar_len = 20
        bar_fill = int(bar_len * (rnd + 1) / n)
        bar = '#' * bar_fill + '-' * (bar_len - bar_fill)

        print(f"\n{'='*70}", flush=True)
        print(f"  ROUND {rnd+1}/{n}  [{bar}]  {rnd_time:.0f}s  "
              f"(elapsed {elapsed/60:.1f}m, ETA {remaining/60:.1f}m)", flush=True)
        print(f"{'='*70}", flush=True)

        # Global model — the main thing
        print(f"  GLOBAL MODEL   Accuracy: {acc:5.1f}%    Loss: {loss_val:.4f}", flush=True)

        # LeakScore
        ent = summary.get('entropy_avg', 0)
        lab = summary.get('label_avg', 0)
        emp = summary.get('empirical_avg', 0)
        print(f"  LEAKSCORE      Combined: {leak:.4f}   "
              f"[Entropy {ent:.3f}  Label {lab:.3f}  Empirical {emp:.3f}]", flush=True)

        # Encryption
        print(f"  ENCRYPTION     Strategy: {strategy}   Encrypted: {enc:.1f}%", flush=True)

        # Privacy metrics (one compact line)
        fc = summary.get('fisher_concentration', 0)
        mv = summary.get('maskcrypt_vulnerability', 0)
        mg = summary.get('magnitude_score', 0)
        print(f"  PRIVACY        Fisher: {fc:.4f}   MaskCrypt: {mv:.4f}   Magnitude: {mg:.4f}", flush=True)

        print(f"{'='*70}\n", flush=True)

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
    client_data_map = partition_data_non_iid(
        train_ds, config['num_clients'],
        classes_per_client=config.get('classes_per_client', 2),
        min_samples=config.get('min_samples_per_client', 20),
    )

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
    parser.add_argument('--num-clients', type=int, default=None,
                        help='Override number of FL clients')
    parser.add_argument('--clients-per-round', type=int, default=None,
                        help='Override clients selected per round')
    parser.add_argument('--num-rounds', type=int, default=None,
                        help='Override number of FL rounds')
    parser.add_argument('--client-lr', type=float, default=None,
                        help='Override client local learning rate')
    parser.add_argument('--defer-attacks', action='store_true',
                        help='Defer GLMIP/Empirical attacks — train fast, run attacks later via run_attacks.py')
    parser.add_argument('--log', type=str, default=None,
                        help='Path to persistent log file (survives SSH disconnects)')

    args = parser.parse_args()

    # Set up persistent logging if requested
    if args.log:
        sys.stdout = TeeLogger(args.log)
        sys.stderr = sys.stdout

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
                          seed_override=args.seed,
                          num_clients_override=args.num_clients,
                          clients_per_round_override=args.clients_per_round,
                          num_rounds_override=args.num_rounds,
                          client_lr_override=args.client_lr,
                          defer_attacks=args.defer_attacks)


if __name__ == '__main__':
    main()
