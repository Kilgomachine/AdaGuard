#!/usr/bin/env python3
"""AdaGuard Post-Training Attack Runner.

Runs GLMIP and Empirical attacks on saved client data from deferred training.
This allows training to run fast, then attacks can be run separately (even later).

Usage:
    # Run attacks on saved client data:
    python run_attacks.py --data-dir /scratch/.../client_data_fisher_seed42 \
                          --config hpc/config_1k.yaml \
                          --results /scratch/.../results/fisher_seed42.json \
                          --output /scratch/.../results/fisher_seed42_with_attacks.json

    # Run only on specific rounds:
    python run_attacks.py --data-dir ... --config ... --rounds 0,1,2,3,4
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime

if not os.environ.get('PYTHONUNBUFFERED'):
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

import numpy as np
import torch
import torch.nn as nn


def run_attacks_on_saved_data(data_dir, config_path, results_path=None,
                               output_path=None, rounds=None):
    """Run GLMIP and Empirical attacks on saved client gradient data."""
    from adaguard.config import load_config, set_seed, get_device
    from adaguard.models import create_model
    from adaguard.data.cifar10 import load_cifar10
    from adaguard.metrics.label_leakscore import GLMIPMetric, CosineSimilarityMetric
    from adaguard.metrics.empirical_leakscore import EmpiricalLeakScoreMetric

    config = load_config(config_path)
    set_seed(config['seed'])
    device = get_device()

    print(f"\n{'='*70}")
    print(f"  AdaGuard Attack Runner (Post-Training)")
    print(f"{'='*70}")
    print(f"  Data dir     {data_dir}")
    print(f"  Config       {config_path}")
    print(f"  Device       {device}")
    if device.type == 'cuda':
        print(f"  GPU          {torch.cuda.get_device_name(0)}")
    print(f"{'='*70}\n")

    # Load training data (needed for GLMIP)
    train_ds, _ = load_cifar10(data_root=os.environ.get('DATA_DIR', './data'))

    # Load existing results if provided
    existing_results = None
    if results_path and os.path.exists(results_path):
        with open(results_path) as f:
            existing_results = json.load(f)
        print(f"  Loaded existing results: {results_path}")

    # Find all round directories
    round_dirs = sorted([
        d for d in os.listdir(data_dir)
        if d.startswith('round_') and os.path.isdir(os.path.join(data_dir, d))
    ], key=lambda x: int(x.split('_')[1]))

    if rounds is not None:
        round_dirs = [d for d in round_dirs if int(d.split('_')[1]) in rounds]

    print(f"  Found {len(round_dirs)} rounds to process\n")

    attack_results = {}

    for round_dir_name in round_dirs:
        rnd = int(round_dir_name.split('_')[1])
        round_path = os.path.join(data_dir, round_dir_name)
        t_round_start = time.time()

        # Load global model state
        model_path = os.path.join(round_path, 'global_model.pt')
        if not os.path.exists(model_path):
            print(f"  [SKIP] Round {rnd}: no global_model.pt")
            continue

        global_model_state = torch.load(model_path, map_location='cpu')

        # Find client files
        client_files = sorted([
            f for f in os.listdir(round_path)
            if f.startswith('client_') and f.endswith('.pt')
        ])

        print(f"  Round {rnd}: {len(client_files)} clients", flush=True)
        round_attack_data = {}

        for cf in client_files:
            cid = int(cf.replace('client_', '').replace('.pt', ''))
            client_data = torch.load(os.path.join(round_path, cf), map_location=device)

            gd = {k: v.to(device) for k, v in client_data['gradient_dict'].items()}
            flat = client_data['flat_gradient'].to(device)
            original_images = client_data.get('original_images')
            if original_images is not None:
                original_images = original_images.to(device)

            client_attacks = {}

            # GLMIP
            t0 = time.time()
            glmip_model = create_model(
                config.get('model', 'smallcnn'), num_classes=config['num_classes'],
            ).to(device)
            glmip_model.load_state_dict(global_model_state)
            glmip_metric = GLMIPMetric()
            criterion = nn.CrossEntropyLoss()
            gl = glmip_metric.compute(
                glmip_model, train_ds, device, criterion,
                config['num_classes'], config['mi_samples_per_class'],
                focus_layers=config['focus_layers'],
            )
            client_attacks['glmip_score'] = gl['glmip_score']
            client_attacks['glmip_time'] = time.time() - t0

            class_means = gl.get('class_means', {})
            cos_metric = CosineSimilarityMetric()
            cos_r = cos_metric.compute(class_means)
            client_attacks.update(cos_r)
            del glmip_model

            # Empirical
            t0 = time.time()
            emp_model = create_model(
                config.get('model', 'smallcnn'), num_classes=config['num_classes'],
            ).to(device)
            emp_model.load_state_dict(global_model_state)
            criterion = nn.CrossEntropyLoss()
            emp_metric = EmpiricalLeakScoreMetric(
                emp_model, criterion, device,
                n_iter=config['empirical_iterations'],
                lr=config['empirical_lr'],
                focus_layers=config['focus_layers'],
            )
            emp_r = emp_metric.compute(gd, flat, original_images=original_images)
            client_attacks.update(emp_r)
            client_attacks['empirical_time'] = time.time() - t0
            del emp_model

            round_attack_data[cid] = client_attacks
            print(f"    C{cid:<4d}  GLMIP={gl['glmip_score']:.3f}  "
                  f"Emp={emp_r.get('empirical_mean', 0):.3f}  "
                  f"({client_attacks['glmip_time']:.0f}s + {client_attacks['empirical_time']:.0f}s)",
                  flush=True)

            # Free GPU memory
            del gd, flat, original_images, client_data
            torch.cuda.empty_cache()

        attack_results[rnd] = round_attack_data
        print(f"  Round {rnd} done in {time.time() - t_round_start:.0f}s\n", flush=True)

    # Merge with existing results if available
    if existing_results and 'rounds' in existing_results:
        for rnd_idx, rnd_data in enumerate(existing_results['rounds']):
            rnd_num = rnd_data.get('round', rnd_idx)
            if rnd_num in attack_results:
                # Update per-client data
                per_client = rnd_data.get('per_client', [])
                for pc in per_client:
                    cid = pc.get('id')
                    if cid in attack_results[rnd_num]:
                        pc.update(attack_results[rnd_num][cid])

                # Update round-level averages
                all_glmip = [v['glmip_score'] for v in attack_results[rnd_num].values()]
                all_emp = [v.get('empirical_mean', 0) for v in attack_results[rnd_num].values()]
                rnd_data['glmip_score'] = float(np.mean(all_glmip))
                rnd_data['empirical_avg'] = float(np.mean(all_emp))

        existing_results['metadata']['attacks_added'] = datetime.now().isoformat()

    # Save
    out = output_path or (results_path.replace('.json', '_attacks.json') if results_path else
                           os.path.join(data_dir, 'attack_results.json'))
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    save_data = existing_results if existing_results else {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'data_dir': data_dir,
            'config': config_path,
        },
        'attack_results': {str(k): v for k, v in attack_results.items()},
    }

    with open(out, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)

    print(f"Results saved to: {out}")


def main():
    parser = argparse.ArgumentParser(description='Run attacks on saved client data')
    parser.add_argument('--data-dir', required=True,
                        help='Directory with saved client data (from --defer-attacks)')
    parser.add_argument('--config', required=True,
                        help='Config YAML used during training')
    parser.add_argument('--results', default=None,
                        help='Existing results JSON to merge attack scores into')
    parser.add_argument('--output', default=None,
                        help='Output path (default: <results>_attacks.json)')
    parser.add_argument('--rounds', default=None,
                        help='Comma-separated round numbers to process (default: all)')

    args = parser.parse_args()
    rounds = None
    if args.rounds:
        rounds = [int(r) for r in args.rounds.split(',')]

    run_attacks_on_saved_data(args.data_dir, args.config, args.results,
                               args.output, rounds)


if __name__ == '__main__':
    main()
