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
import random
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


# ─── Tensorboard helper ───────────────────────────────────────────────
class TBLogger:
    """Tensorboard logger for FL training. Handles live logging + snapshot replay."""

    def __init__(self, log_dir, config, strategy, run_name=None):
        from torch.utils.tensorboard import SummaryWriter

        if run_name is None:
            seed = config.get('seed', 0)
            nc = config.get('num_clients', 0)
            run_name = f"{strategy}_seed{seed}_{nc}clients"

        self.run_dir = Path(log_dir) / run_name
        self.writer = SummaryWriter(log_dir=str(self.run_dir))

        # Log config as text
        cfg_lines = [f"**{k}**: {v}" for k, v in sorted(config.items())
                     if not isinstance(v, (list, dict))]
        self.writer.add_text('config/settings', '  \n'.join(cfg_lines), 0)
        self.writer.add_text('config/strategy', strategy, 0)
        self.writer.add_text('config/run_name', run_name, 0)

    def log_round(self, rnd, summary, round_time=None):
        """Log all metrics for a single round."""
        w = self.writer
        step = rnd + 1  # 1-indexed for display

        # Global model
        acc = summary.get('accuracy', 0)
        test_loss = summary.get('test_loss', 0)
        client_loss = summary.get('loss', 0)
        w.add_scalar('Global/Accuracy_%', acc * 100, step)
        w.add_scalar('Global/Test_Loss', test_loss, step)
        w.add_scalar('Global/Avg_Client_Loss', client_loss, step)

        # LeakScore — combined + averages
        leak = summary.get('combined_leakscore', 0)
        ent = summary.get('entropy_avg', 0)
        lab = summary.get('label_avg', 0)
        emp = summary.get('empirical_avg', 0)
        w.add_scalar('LeakScore/Combined', leak, step)
        w.add_scalar('LeakScore/Entropy_Avg', ent, step)
        w.add_scalar('LeakScore/Label_Avg', lab, step)
        w.add_scalar('LeakScore/Empirical_Avg', emp, step)

        # Individual label metrics (3 components)
        w.add_scalar('LeakScore_Label/GLMIP', summary.get('glmip_score', 0), step)
        w.add_scalar('LeakScore_Label/ConfidenceGap', summary.get('confidence_gap', 0), step)
        w.add_scalar('LeakScore_Label/CosineSimilarity', summary.get('cosine_leak_score', 0), step)

        # Individual entropy metrics (3 components)
        w.add_scalar('LeakScore_Entropy/Shannon', summary.get('shannon_leak_score', 0), step)
        w.add_scalar('LeakScore_Entropy/Renyi', summary.get('renyi_leak_score', 0), step)
        w.add_scalar('LeakScore_Entropy/MinEntropy', summary.get('min_entropy_leak_score', 0), step)

        # Privacy metrics
        fc = summary.get('fisher_concentration', 0)
        mv = summary.get('maskcrypt_vulnerability', 0)
        mg = summary.get('magnitude_score', 0)
        w.add_scalar('Privacy/Fisher_Concentration', fc, step)
        w.add_scalar('Privacy/MaskCrypt_Vulnerability', mv, step)
        w.add_scalar('Privacy/Magnitude_Score', mg, step)

        # Encryption
        enc = summary.get('actual_pct_encrypted', 0)
        w.add_scalar('Encryption/Pct_Encrypted', enc * 100, step)

        # Gradient Accumulation
        accum_pct = summary.get('grad_accum_activated', 0)
        w.add_scalar('GradAccum/Activated_Pct', accum_pct * 100, step)
        w.add_scalar('GradAccum/Effective_BatchSize', summary.get('grad_accum_effective_bs', 0), step)

        # Timing
        if round_time is not None:
            w.add_scalar('Timing/Round_Seconds', round_time, step)

        # Per-client loss distribution (histogram)
        per_client = summary.get('per_client', [])
        if per_client:
            losses = [c.get('loss', 0) for c in per_client if c.get('loss') is not None]
            if losses:
                w.add_histogram('Clients/Loss_Distribution', np.array(losses), step)
            leaks = [c.get('leakscore', 0) for c in per_client if c.get('leakscore') is not None]
            if leaks:
                w.add_histogram('Clients/LeakScore_Distribution', np.array(leaks), step)

        w.flush()

    def log_checkpoint_status(self, rnd, total_rounds, elapsed_s, status='running'):
        """Log training progress status."""
        pct = (rnd + 1) / total_rounds * 100 if total_rounds > 0 else 0
        self.writer.add_text('status/progress',
                             f"Round **{rnd+1}/{total_rounds}** ({pct:.0f}%) | "
                             f"Elapsed: **{elapsed_s/60:.1f}m** | Status: **{status}**",
                             rnd + 1)
        self.writer.flush()

    def log_resume(self, resumed_round, total_rounds):
        """Log that training was resumed from a checkpoint."""
        self.writer.add_text('status/events',
                             f"**RESUMED** from round {resumed_round}/{total_rounds}",
                             resumed_round)
        self.writer.flush()

    def log_completed(self, total_rounds, total_time, final_acc):
        """Log training completion."""
        self.writer.add_text('status/events',
                             f"**COMPLETED** {total_rounds} rounds in "
                             f"{total_time/60:.1f}m — Final Acc: {final_acc:.1f}%",
                             total_rounds)
        self.writer.flush()

    def replay_from_results(self, round_results):
        """Replay all round results into tensorboard (for snapshots/past runs)."""
        for i, r in enumerate(round_results):
            rnd_time = r.get('round_time_s', None)
            self.log_round(i, r, round_time=rnd_time)
        if round_results:
            last = round_results[-1]
            acc = last.get('accuracy', 0) * 100
            self.log_completed(len(round_results), 0, acc)
        print(f"  Replayed {len(round_results)} rounds to Tensorboard: {self.run_dir}")

    def close(self):
        self.writer.close()


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


def save_checkpoint(checkpoint_dir, rnd, sim, round_results, config, strategy,
                    pretrain_history, total_start, client_data_map_seed_info=None):
    """Save a checkpoint after each round so training can resume if interrupted."""
    ckpt_path = Path(checkpoint_dir) / 'checkpoint.pt'
    ckpt_tmp = Path(checkpoint_dir) / 'checkpoint.pt.tmp'
    ckpt_data = {
        'round': rnd,
        'global_model_state': sim.global_model.state_dict(),
        'weights_before_round': sim.weights_before_round,
        'weights_after_round': sim.weights_after_round,
        'weights_previous_round': sim.weights_previous_round,
        'round_history': sim.round_history,
        'round_results': round_results,
        'config': config,
        'strategy': strategy,
        'pretrain_history': pretrain_history,
        'elapsed_before_resume': time.time() - total_start,
        'rng_state_torch': torch.random.get_rng_state(),
        'rng_state_numpy': np.random.get_state(),
        'rng_state_python': random.getstate(),
    }
    if torch.cuda.is_available():
        ckpt_data['rng_state_cuda'] = [torch.cuda.get_rng_state(i)
                                        for i in range(torch.cuda.device_count())]
    # Atomic write: save to tmp then rename (avoids corrupt checkpoint on kill)
    torch.save(ckpt_data, ckpt_tmp)
    ckpt_tmp.rename(ckpt_path)


def load_checkpoint(checkpoint_dir):
    """Load checkpoint. Returns dict or None if no checkpoint exists."""
    ckpt_path = Path(checkpoint_dir) / 'checkpoint.pt'
    if not ckpt_path.exists():
        return None
    print(f"  Found checkpoint: {ckpt_path}")
    return torch.load(ckpt_path, map_location='cpu', weights_only=False)


def save_training_snapshot(snapshot_dir, sim, config, strategy, round_results,
                           pretrain_history, client_data_map):
    """Save everything needed to run analysis without retraining.

    Saves: final model, per-round model snapshots, config, round results,
    client data partition, and all round history for offline metrics/attacks.
    """
    snap_dir = Path(snapshot_dir)
    snap_dir.mkdir(parents=True, exist_ok=True)

    snapshot = {
        'final_model_state': sim.global_model.state_dict(),
        'config': config,
        'strategy': strategy,
        'round_results': round_results,
        'round_history': sim.round_history,
        'pretrain_history': pretrain_history,
        'client_data_map': {k: v for k, v in client_data_map.items()},
        'weights_before_last_round': sim.weights_before_round,
        'weights_after_last_round': sim.weights_after_round,
    }
    snap_path = snap_dir / 'training_snapshot.pt'
    torch.save(snapshot, snap_path)
    print(f"  Training snapshot saved to: {snap_path}")
    print(f"  To skip training next time, use: --load-training {snap_path}")
    return snap_path


def run_fl_experiment(config_path, strategy, skip_glmip, skip_empirical, output_path,
                      pretrain_override=None, seed_override=None,
                      num_clients_override=None, clients_per_round_override=None,
                      num_rounds_override=None, client_lr_override=None,
                      defer_attacks=False, resume_dir=None, load_training=None,
                      tb_dir=None, artifacts_dir=None):
    """Run FL rounds and save results.

    Args:
        resume_dir: directory for checkpointing (auto-resumes if checkpoint exists)
        load_training: path to training_snapshot.pt to skip training entirely
        tb_dir: tensorboard log directory (None = no tensorboard)
        artifacts_dir: save per-client artifacts for Phase 2 scenario replay
    """
    import random as _random
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

    from adaguard.utils.gpu import get_optimal_config, print_gpu_summary
    gpu_cfg = get_optimal_config(config, verbose=False)
    device = gpu_cfg['device']
    n_gpu = gpu_cfg['num_gpus']

    print(f"\n{'='*70}")
    print(f"  AdaGuard Federated Learning Experiment")
    print(f"{'='*70}")
    print_gpu_summary() if n_gpu else print(f"  Device       CPU")
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
    if resume_dir:
        flags.append(f"CHECKPOINT DIR: {resume_dir}")
    if load_training:
        flags.append(f"LOAD TRAINING: {load_training}")
    if artifacts_dir:
        flags.append(f"ARTIFACTS: {artifacts_dir}")
    if flags:
        print(f"  Flags      {', '.join(flags)}")
    print(f"{'='*70}\n")

    # ── Load training snapshot (skip all training) ──
    if load_training:
        print(f"\n=== Loading completed training from {load_training} ===")
        snapshot = torch.load(load_training, map_location='cpu', weights_only=False)
        config_snap = snapshot['config']
        # Use snapshot's model, results, history — no training needed
        model = create_model(config_snap.get('model', 'smallcnn'),
                             num_classes=config_snap['num_classes']).to(device)
        model.load_state_dict(snapshot['final_model_state'])
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Model loaded: {total_params:,} params")

        round_results = snapshot['round_results']
        pretrain_history = snapshot.get('pretrain_history', [])
        pretrain_time = 0.0
        total_time = 0.0

        # Print final accuracy from loaded training
        if round_results:
            last = round_results[-1]
            acc = last.get('accuracy', 0) * 100
            test_loss = last.get('test_loss', 0)
            print(f"  Loaded {len(round_results)} rounds — Final Acc: {acc:.1f}%, Loss: {test_loss:.4f}")

        # Replay into Tensorboard so past training is visible
        if tb_dir:
            tb = TBLogger(tb_dir, config_snap, snapshot.get('strategy', strategy),
                          run_name=f"snapshot_{strategy}_seed{config['seed']}")
            tb.replay_from_results(round_results)
            tb.close()

        print(f"  Training skipped — using saved model and results")
        print(f"  You can now run analysis/attacks on this data without retraining\n")

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
                'loaded_from': str(load_training),
            },
            'config': config_snap,
            'pretrain_history': pretrain_history,
            'rounds': make_serializable(round_results),
        }

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)

        print(f"Results saved to: {out_path}")
        return output

    # ── Normal training path ──
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

    # ── Check for checkpoint to resume from ──
    start_round = 0
    round_results = []
    elapsed_before = 0.0

    # Set up checkpoint directory
    checkpoint_dir = resume_dir
    if checkpoint_dir is None:
        # Default: next to output file
        checkpoint_dir = str(Path(output_path).parent / f'checkpoints_{Path(output_path).stem}')
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # ── Initialize Tensorboard ──
    tb = None
    if tb_dir:
        tb = TBLogger(tb_dir, config, strategy)
        print(f"  Tensorboard: {tb.run_dir}")

    ckpt = load_checkpoint(checkpoint_dir)
    if ckpt is not None:
        start_round = ckpt['round'] + 1
        round_results = ckpt['round_results']
        elapsed_before = ckpt.get('elapsed_before_resume', 0.0)

        # Restore model state
        sim.global_model.load_state_dict(ckpt['global_model_state'])
        sim.weights_before_round = ckpt.get('weights_before_round')
        sim.weights_after_round = ckpt.get('weights_after_round')
        sim.weights_previous_round = ckpt.get('weights_previous_round')
        sim.round_history = ckpt.get('round_history', [])

        # Restore RNG states for reproducibility
        torch.random.set_rng_state(ckpt['rng_state_torch'])
        np.random.set_state(ckpt['rng_state_numpy'])
        _random.setstate(ckpt['rng_state_python'])
        if torch.cuda.is_available() and 'rng_state_cuda' in ckpt:
            for i, state in enumerate(ckpt['rng_state_cuda']):
                torch.cuda.set_rng_state(state, i)

        last_acc = round_results[-1].get('accuracy', 0) * 100 if round_results else 0
        print(f"\n  RESUMED from round {start_round} (acc={last_acc:.1f}%, "
              f"prev elapsed={elapsed_before/60:.1f}m)")
        print(f"  Checkpoint: {checkpoint_dir}")

        # Replay past rounds into Tensorboard so the full history is visible
        if tb:
            tb.replay_from_results(round_results)
            tb.log_resume(start_round, config['num_rounds'])
    else:
        print(f"  Checkpoint dir: {checkpoint_dir} (saving after each round)")

    # ── Save artifact metadata for Phase 2 ──
    if artifacts_dir:
        Path(artifacts_dir).mkdir(parents=True, exist_ok=True)
        metadata = {
            'config': config,
            'strategy': strategy,
            'seed': config['seed'],
            'num_clients': config['num_clients'],
            'num_rounds': config['num_rounds'],
            'model': config.get('model', 'smallcnn'),
            'num_classes': config['num_classes'],
        }
        import json as _json
        with open(Path(artifacts_dir) / 'metadata.json', 'w') as f:
            _json.dump(metadata, f, indent=2, default=str)
        print(f"  Artifacts dir: {artifacts_dir}")

    # Run FL rounds
    # If deferring attacks, save client data next to output for later processing
    attack_save_dir = None
    if defer_attacks:
        attack_save_dir = str(Path(output_path).parent / f'client_data_{Path(output_path).stem}')
        print(f"  Attack data will be saved to: {attack_save_dir}")
        skip_glmip = True
        skip_empirical = True

    remaining_rounds = config['num_rounds'] - start_round
    if remaining_rounds <= 0:
        print(f"\n=== Training already complete ({config['num_rounds']} rounds) ===")
    else:
        print(f"\n=== Federated Learning ({config['num_rounds']} rounds, starting from {start_round + 1}) ===")

    total_start = time.time()

    for rnd in range(start_round, config['num_rounds']):
        rnd_start = time.time()
        summary = sim.run_round(
            rnd,
            encryption_strategy=strategy,
            skip_glmip=skip_glmip,
            skip_empirical=skip_empirical,
            save_dir=attack_save_dir,
            artifacts_dir=artifacts_dir,
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
        elapsed = time.time() - total_start + elapsed_before
        rounds_done = rnd + 1
        avg_per_round = elapsed / rounds_done
        remaining = avg_per_round * (config['num_rounds'] - rounds_done)

        # ── Clean round summary ──
        n = config['num_rounds']
        bar_len = 20
        bar_fill = int(bar_len * rounds_done / n)
        bar = '#' * bar_fill + '-' * (bar_len - bar_fill)

        print(f"\n{'='*70}", flush=True)
        print(f"  ROUND {rounds_done}/{n}  [{bar}]  {rnd_time:.0f}s  "
              f"(elapsed {elapsed/60:.1f}m, ETA {remaining/60:.1f}m)", flush=True)
        print(f"{'='*70}", flush=True)

        # Global model — the main thing
        test_loss = summary.get('test_loss', 0)
        print(f"  GLOBAL MODEL   Accuracy: {acc:5.1f}%    Test Loss: {test_loss:.4f}    Avg Client Loss: {loss_val:.4f}", flush=True)

        # LeakScore
        ent = summary.get('entropy_avg', 0)
        lab = summary.get('label_avg', 0)
        emp = summary.get('empirical_avg', 0)
        print(f"  LEAKSCORE      Combined: {leak:.4f}   "
              f"[Entropy {ent:.3f}  Label {lab:.3f}  Empirical {emp:.3f}]", flush=True)

        # Encryption
        print(f"  ENCRYPTION     Strategy: {strategy}   Encrypted: {enc:.1f}%", flush=True)

        # Gradient Accumulation
        accum_active = summary.get('grad_accum_activated', 0)
        accum_bs = summary.get('grad_accum_effective_bs', 0)
        if accum_active > 0:
            print(f"  GRAD ACCUM     Active: {accum_active*100:.0f}% of clients   Avg B_eff: {accum_bs:.0f}", flush=True)

        # Privacy metrics (one compact line)
        fc = summary.get('fisher_concentration', 0)
        mv = summary.get('maskcrypt_vulnerability', 0)
        mg = summary.get('magnitude_score', 0)
        print(f"  PRIVACY        Fisher: {fc:.4f}   MaskCrypt: {mv:.4f}   Magnitude: {mg:.4f}", flush=True)

        print(f"{'='*70}\n", flush=True)

        # ── Save checkpoint after each round ──
        save_checkpoint(checkpoint_dir, rnd, sim, round_results, config, strategy,
                        pretrain_history, total_start)

        # ── Log to Tensorboard ──
        if tb:
            tb.log_round(rnd, summary_clean, round_time=rnd_time)
            tb.log_checkpoint_status(rnd, config['num_rounds'], elapsed)

    total_time = time.time() - total_start + elapsed_before

    # ── Log training completion to Tensorboard ──
    if tb:
        final_acc = round_results[-1].get('accuracy', 0) * 100 if round_results else 0
        tb.log_completed(config['num_rounds'], total_time, final_acc)
        tb.close()

    # ── Save training snapshot for future reuse ──
    snapshot_dir = str(Path(output_path).parent / f'snapshot_{Path(output_path).stem}')
    save_training_snapshot(snapshot_dir, sim, config, strategy, round_results,
                           pretrain_history, client_data_map)

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
    parser.add_argument('--resume-dir', type=str, default=None,
                        help='Checkpoint directory for resume-on-interrupt (auto-saves after each round)')
    parser.add_argument('--load-training', type=str, default=None,
                        help='Path to training_snapshot.pt — skip training, load saved model+results')
    parser.add_argument('--tb-dir', type=str, default=None,
                        help='Tensorboard log directory (view with: tensorboard --logdir TB_DIR)')
    parser.add_argument('--artifacts-dir', type=str, default=None,
                        help='Save per-client artifacts for Phase 2 scenario replay (~1TB for 250 rounds)')

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
                          defer_attacks=args.defer_attacks,
                          resume_dir=args.resume_dir,
                          load_training=args.load_training,
                          tb_dir=args.tb_dir,
                          artifacts_dir=args.artifacts_dir)


if __name__ == '__main__':
    main()
