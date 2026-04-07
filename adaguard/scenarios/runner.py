"""Scenario Runner: replay saved training artifacts under different encryption policies.

Loads per-round, per-client artifacts from Phase 1 training, applies the scenario's
encryption strategy, re-aggregates via FedAvg, runs full paper-matched attacks,
and computes all reconstruction metrics.

Each scenario is independent and can be run/resumed separately.
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

from ..config import load_config, get_device, DEFAULT_CONFIG
from ..metrics.combined import CombinedLeakScore
from ..encryption import (AdaptiveEncryptionController, DPNoiseEncryptor,
                           FisherEncryptor, MaskCryptEncryptor)
from ..metrics.fisher import FisherInformationMetric
from ..metrics.maskcrypt import MaskCryptMetric
from ..federation.aggregator import fedavg_aggregate_state_dicts, fedavg_aggregate, apply_gradient_update
from ..models import create_model
from ..utils.reconstruction import compute_all_reconstruction_metrics


def _resolve_scenario_config(scenario):
    """Merge scenario config_overrides onto DEFAULT_CONFIG to get effective config."""
    import copy
    effective = copy.deepcopy(DEFAULT_CONFIG)
    overrides = scenario.get('config_overrides', {})
    effective.update(overrides)
    return effective


class ScenarioRunner:
    """Replay training artifacts under a specific encryption scenario."""

    def __init__(self, artifacts_dir, scenario_config, attack_config=None, device=None):
        """
        Args:
            artifacts_dir: path to saved training artifacts from Phase 1
            scenario_config: dict from registry (with 'config_overrides' key)
            attack_config: dict of attack hyperparameters (gi_iters, etc.)
            device: torch device (auto-detected if None)
        """
        self.artifacts_dir = Path(artifacts_dir)
        self.scenario = scenario_config
        self.attack_config = attack_config or {}
        self.device = device or get_device()

        # Load metadata
        meta_path = self.artifacts_dir / 'metadata.json'
        if not meta_path.exists():
            raise FileNotFoundError(f"No metadata.json in {artifacts_dir}. Run training with --artifacts-dir first.")
        with open(meta_path) as f:
            self.metadata = json.load(f)

        self.training_config = self.metadata['config']

        # Resolve effective config: defaults + scenario overrides
        self.config = _resolve_scenario_config(scenario_config)

        # Set up encryption for this scenario
        enc_type = self.config.get('encryption', 'fisher')
        T1 = self.config.get('T1', 0.3)
        T2 = self.config.get('T2', 0.7)
        enc_pct = self.config.get('encryption_top_percent', 0.1)

        self.controller = AdaptiveEncryptionController(T1=T1, T2=T2, base_encrypt_pct=enc_pct)
        self.fisher_encryptor = FisherEncryptor(
            fisher_metric=FisherInformationMetric(enc_pct=enc_pct)
        )

        # MaskCrypt with mode support (gradient_guided or random)
        mask_mode = self.config.get('maskcrypt_mask_mode', 'gradient_guided')
        self.maskcrypt_metric = MaskCryptMetric(enc_pct=enc_pct)
        self.maskcrypt_encryptor = MaskCryptEncryptor(
            maskcrypt_metric=self.maskcrypt_metric, enc_pct=enc_pct, mask_mode=mask_mode,
        )

        # DP noise encryptor
        self.dp_encryptor = DPNoiseEncryptor(
            epsilon=self.config.get('dp_epsilon', 50.0),
            delta=self.config.get('dp_delta', 1e-5),
            clip_norm=self.config.get('dp_clip_norm', 1.0),
        )

        # LeakScore with scenario weights
        self.combined_metric = CombinedLeakScore(
            label_weight=self.config.get('label_weight', 1.0),
            entropy_weight=self.config.get('entropy_weight', 1.0),
            empirical_weight=self.config.get('empirical_weight', 0.0),
        )

        # Detect GPUs with adaptive allocation
        from ..utils.gpu import get_optimal_config
        gpu_cfg = get_optimal_config(self.config, verbose=True)
        self.gpu_devices = gpu_cfg['devices']

    def _needs_metric_recompute(self):
        """Check if this scenario changes metric computation parameters.

        If the scenario overrides entropy_bins, mi_samples_per_class, or
        focus_layers, the saved metrics from Phase 1 are no longer valid
        and must be recomputed from the raw artifacts.

        Returns:
            set of metric categories to recompute: 'entropy', 'label', or both
        """
        overrides = self.scenario.get('config_overrides', {})
        recompute = set()

        # Entropy bins affects Shannon, Renyi, MinEntropy
        if 'entropy_bins' in overrides:
            training_bins = self.training_config.get('entropy_bins', 50)
            if overrides['entropy_bins'] != training_bins:
                recompute.add('entropy')

        # GLMIP samples_per_class or focus_layers affect label metrics
        if 'mi_samples_per_class' in overrides:
            training_spc = self.training_config.get('mi_samples_per_class', 20)
            if overrides['mi_samples_per_class'] != training_spc:
                recompute.add('label')
        if 'focus_layers' in overrides:
            training_fl = self.training_config.get('focus_layers')
            if overrides['focus_layers'] != training_fl:
                recompute.add('label')

        return recompute

    def _recompute_entropy_metrics(self, art, global_state):
        """Recompute entropy metrics with scenario's entropy_bins setting."""
        from ..metrics.entropy_leakscore import (
            ShannonEntropyMetric, RenyiEntropyMetric, MinEntropyMetric
        )
        bins = self.config.get('entropy_bins', 50)
        flat = art['flat_gradient']

        shannon = ShannonEntropyMetric(n_bins=bins).compute(flat)
        renyi = RenyiEntropyMetric(n_bins=bins).compute(flat)
        min_ent = MinEntropyMetric(n_bins=bins).compute(flat)

        entropy_avg = np.mean([
            shannon.get('shannon_score', 0.0),
            renyi.get('renyi_score', 0.0),
            min_ent.get('min_entropy_score', 0.0),
        ])
        return entropy_avg

    def _recompute_label_metrics(self, art, global_state):
        """Recompute label metrics with scenario's focus_layers/samples settings."""
        from ..metrics.label_leakscore import (
            GLMIPMetric, ConfidenceGapMetric, CosineSimilarityMetric
        )
        focus_layers = self.config.get('focus_layers')
        spc = self.config.get('mi_samples_per_class', 20)
        num_classes = self.config.get('num_classes', 10)

        scores = []

        # ConfidenceGap uses saved outputs + labels (no model needed)
        outputs = art.get('outputs')
        labels = art.get('labels')
        if outputs is not None:
            cg = ConfidenceGapMetric()
            cg_result = cg.compute(outputs, labels=labels)
            scores.append(cg_result.get('confidence_gap', 0.0))

        # CosineSimilarity uses saved gradient
        gd = art['gradient_dict']
        cos_metric = CosineSimilarityMetric()
        cos_result = cos_metric.compute(gd, focus_layers=focus_layers)
        scores.append(cos_result.get('cosine_similarity_score', 0.0))

        # GLMIP requires model + dataset — skip if not available on CPU
        # (it's expensive; use saved value as fallback)
        saved_glmip = art.get('metrics', {}).get('glmip_score', 0.0)
        scores.append(saved_glmip)

        return float(np.mean(scores)) if scores else 0.0

    def get_saved_rounds(self):
        """Return sorted list of round indices that have saved artifacts."""
        rounds = []
        for d in self.artifacts_dir.iterdir():
            if d.is_dir() and d.name.startswith('round_'):
                try:
                    rnd = int(d.name.split('_')[1])
                    rounds.append(rnd)
                except (ValueError, IndexError):
                    pass
        return sorted(rounds)

    def run_round(self, rnd, run_attacks=True, tb_logger=None):
        """Run scenario evaluation for a single round.

        Args:
            rnd: round index
            run_attacks: if True, run full paper-matched GI attacks
            tb_logger: optional TBLogger for Tensorboard output

        Returns:
            dict with all metrics for this round
        """
        round_dir = self.artifacts_dir / f'round_{rnd}'
        if not round_dir.exists():
            raise FileNotFoundError(f"No artifacts for round {rnd} at {round_dir}")

        enc_type = self.config.get('encryption', 'fisher')

        print(f"\n{'='*70}")
        print(f"  SCENARIO: {self.scenario.get('description', '?')}")
        print(f"  Round {rnd} — encryption={enc_type}")
        print(f"{'='*70}")

        # Load global model state for this round
        global_model_path = round_dir / 'global_model.pt'
        global_state = torch.load(global_model_path, map_location='cpu', weights_only=False)

        # Load client list
        clients_path = round_dir / 'selected_clients.json'
        with open(clients_path) as f:
            selected_clients = json.load(f)

        # Load client artifacts
        client_artifacts = {}
        for cid in selected_clients:
            art_path = round_dir / f'client_{cid}.pt'
            if art_path.exists():
                client_artifacts[cid] = torch.load(art_path, map_location='cpu', weights_only=False)

        if not client_artifacts:
            print(f"  No client artifacts found for round {rnd}")
            return {}

        print(f"  Loaded {len(client_artifacts)} client artifacts")

        # Check if this scenario requires recomputing metrics from raw data
        recompute_categories = self._needs_metric_recompute()
        if recompute_categories:
            print(f"  Recomputing metrics: {recompute_categories} "
                  f"(scenario overrides training-time parameters)")

        # ── Phase 1: Apply encryption to each client's gradients ──
        protected_state_dicts = []
        protected_gradients = []
        round_metrics = {
            'round': rnd,
            'num_clients': len(client_artifacts),
            'encryption': enc_type,
        }

        client_results = []
        t0 = time.time()

        for cid, art in client_artifacts.items():
            metrics = art.get('metrics', {})

            # Start from saved metrics, then recompute what the scenario changes
            entropy_avg = metrics.get('entropy_avg', 0.0)
            label_avg = metrics.get('label_avg', 0.0)
            empirical_avg = metrics.get('empirical_avg', 0.0)

            # Recompute entropy if bins changed (S1)
            if 'entropy' in recompute_categories:
                entropy_avg = self._recompute_entropy_metrics(art, global_state)

            # Recompute label if focus_layers or samples_per_class changed (S9, S11)
            if 'label' in recompute_categories:
                label_avg = self._recompute_label_metrics(art, global_state)

            # If this scenario uses empirical_weight > 0 and we need fresh empirical scores,
            # run lightweight attacks here (20 iterations, not full 20K)
            if self.config.get('empirical_weight', 0.0) > 0 and empirical_avg == 0.0:
                empirical_avg = self._run_lightweight_attacks(art, global_state)

            combined = self.combined_metric.compute(entropy_avg, label_avg, empirical_avg)

            # Apply encryption
            gd_cpu = art['gradient_dict']
            fisher_r = art.get('fisher_result', {})
            total_params = sum(g.numel() for g in gd_cpu.values())

            if enc_type == 'none':
                protected_gd = gd_cpu
                pct_encrypted = 0.0
            elif enc_type == 'full':
                protected_gd = {n: torch.zeros_like(g) for n, g in gd_cpu.items()}
                pct_encrypted = 1.0
            elif enc_type == 'dp':
                protected_gd, dp_meta = self.dp_encryptor.encrypt(gd_cpu)
                pct_encrypted = 0.0  # DP adds noise, doesn't "encrypt" specific params
            elif enc_type == 'maskcrypt':
                exposed_w = {k: v for k, v in global_state.items() if k in gd_cpu}
                local_w = art.get('local_weights', {})
                mc_pct = self.config.get('encryption_top_percent', 0.1)
                # Use pre-configured encryptor (handles gradient_guided vs random)
                protected_gd, enc_meta = self.maskcrypt_encryptor.encrypt(
                    gd_cpu, exposed_w, local_w,
                )
                pct_encrypted = enc_meta.get('pct_encrypted', mc_pct)
            else:  # 'fisher'
                policy = self.controller.decide(combined)
                k = self.controller.compute_k(combined, total_params)
                if policy.level == 'none':
                    protected_gd = gd_cpu
                    pct_encrypted = 0.0
                else:
                    fisher_r_cpu = {k_: (v.cpu() if isinstance(v, torch.Tensor) else v)
                                    for k_, v in fisher_r.items()}
                    protected_gd, enc_meta = self.fisher_encryptor.encrypt(
                        gd_cpu, k=k, fisher_result=fisher_r_cpu,
                    )
                    pct_encrypted = enc_meta.get('pct_encrypted', 0.0)

            # Collect for aggregation
            protected_gradients.append(protected_gd)
            local_sd = art.get('local_state_dict', {})
            if local_sd:
                protected_state_dicts.append(local_sd)

            client_results.append({
                'client_id': cid,
                'combined_leakscore': combined,
                'pct_encrypted': pct_encrypted,
                'entropy_avg': entropy_avg,
                'label_avg': label_avg,
                'empirical_avg': empirical_avg,
            })

        encryption_time = time.time() - t0
        print(f"  Encryption applied in {encryption_time:.1f}s")

        # ── Phase 2: Re-aggregate and evaluate accuracy ──
        model = create_model(
            self.config.get('model', 'smallcnn'),
            num_classes=self.config['num_classes'],
        ).to(self.device)
        model.load_state_dict({k: v.to(self.device) for k, v in global_state.items()})

        if protected_state_dicts:
            avg_state = fedavg_aggregate_state_dicts(protected_state_dicts)
            avg_state = {k: v.to(self.device) for k, v in avg_state.items()}
            model.load_state_dict(avg_state)
        elif protected_gradients:
            avg_grads = fedavg_aggregate(protected_gradients)
            avg_grads = {k: v.to(self.device) for k, v in avg_grads.items()}
            apply_gradient_update(model, avg_grads, self.config.get('fl_lr', 1.0))

        # Evaluate on test set
        from ..data.cifar10 import load_cifar10
        _, test_ds = load_cifar10(data_root=os.environ.get('DATA_DIR', './data'))
        accuracy, test_loss = self._evaluate(model, test_ds)

        avg_leakscore = np.mean([c['combined_leakscore'] for c in client_results])
        avg_encrypted = np.mean([c['pct_encrypted'] for c in client_results])

        round_metrics.update({
            'accuracy': accuracy,
            'test_loss': test_loss,
            'avg_leakscore': avg_leakscore,
            'avg_pct_encrypted': avg_encrypted,
            'encryption_time_s': encryption_time,
        })

        print(f"  Accuracy: {accuracy*100:.1f}%  |  Loss: {test_loss:.4f}  |  "
              f"Encrypted: {avg_encrypted*100:.1f}%")

        # ── Phase 3: Run full attacks on encrypted gradients ──
        attack_results = {}
        if run_attacks:
            print(f"  Running full attacks...")
            attack_results = self._run_full_attacks(
                client_artifacts, protected_gradients, global_state, selected_clients
            )
            round_metrics['attacks'] = attack_results

        # ── Log to Tensorboard ──
        if tb_logger is not None:
            scenario_name = self.scenario.get('description', 'unknown')
            self._log_to_tb(tb_logger, rnd, round_metrics, attack_results, client_artifacts)

        return round_metrics

    def _run_lightweight_attacks(self, artifact, global_state):
        """Run 20-iteration lightweight attacks for empirical leakscore."""
        from ..metrics.empirical_leakscore import EmpiricalLeakScoreMetric

        model = create_model(
            self.config.get('model', 'smallcnn'),
            num_classes=self.config['num_classes'],
        ).to(self.device)
        model.load_state_dict({k: v.to(self.device) for k, v in global_state.items()})

        criterion = torch.nn.CrossEntropyLoss()
        emp_metric = EmpiricalLeakScoreMetric(
            model, criterion, self.device,
            n_iter=self.config.get('empirical_iterations', 20),
            lr=self.config.get('empirical_lr', 0.1),
            focus_layers=self.config.get('focus_layers', []),
        )
        gd = {k: v.to(self.device) for k, v in artifact['gradient_dict'].items()}
        flat = artifact['flat_gradient'].to(self.device)
        images = artifact.get('images')
        if images is not None:
            images = images.to(self.device)

        result = emp_metric.compute(gd, flat, original_images=images)
        return result.get('empirical_mean', 0.0)

    def _run_full_attacks(self, client_artifacts, protected_gradients, global_state, selected_clients):
        """Run full paper-matched attacks on encrypted gradients.

        Distributes attacks across available GPUs.
        """
        from ..attacks.grad_inversion import GradInversionFull
        from ..attacks.gi_nas import GINASFull
        from ..attacks.ggcdm import GGCDMFull

        attack_configs = {
            'gradinversion': {
                'n_iter': self.attack_config.get('attack_gi_iters', 20000),
                'lr': self.attack_config.get('attack_gi_lr', 0.1),
                'tv_lambda': self.attack_config.get('attack_gi_tv_lambda', 1e-4),
                'l2_lambda': self.attack_config.get('attack_gi_l2_lambda', 1e-6),
                'bn_lambda': self.attack_config.get('attack_gi_bn_lambda', 0.1),
            },
            'gi_nas': {
                'n_iter': self.attack_config.get('attack_ginas_iters', 500),
                'lr': self.attack_config.get('attack_ginas_lr', 1e-3),
                'n_restarts': self.attack_config.get('attack_ginas_restarts', 8),
            },
            'ggcdm': {
                'n_iter': self.attack_config.get('attack_ggcdm_diffusion_steps', 1000),
                'guidance_rate': self.attack_config.get('attack_ggcdm_guidance_rate', 0.20),
            },
        }

        all_attack_results = {}
        # Run attacks on a subset of clients (first 5 or configurable)
        max_attack_clients = self.attack_config.get('max_attack_clients', 5)
        attack_client_ids = selected_clients[:max_attack_clients]

        for i, cid in enumerate(attack_client_ids):
            if cid not in client_artifacts:
                continue

            art = client_artifacts[cid]
            gpu_dev = self.gpu_devices[i % len(self.gpu_devices)]

            print(f"    Attacking client {cid} on {gpu_dev}...")
            t0 = time.time()

            # Get the encrypted gradient for this client
            enc_gd = protected_gradients[selected_clients.index(cid)] if cid in selected_clients else art['gradient_dict']
            original_images = art.get('images')
            labels = art.get('labels')

            # Create model copy on target GPU
            model = create_model(
                self.config.get('model', 'smallcnn'),
                num_classes=self.config['num_classes'],
            ).to(gpu_dev)
            model.load_state_dict({k: v.to(gpu_dev) for k, v in global_state.items()})
            criterion = torch.nn.CrossEntropyLoss()

            # Move encrypted gradients to GPU
            enc_gd_gpu = {k: v.to(gpu_dev) for k, v in enc_gd.items()}
            enc_flat = torch.cat([g.flatten() for g in enc_gd_gpu.values()])

            client_attack_results = {}

            # Run each attack
            for attack_name, acfg in attack_configs.items():
                try:
                    if attack_name == 'gradinversion':
                        attack = GradInversionFull(
                            model, criterion, gpu_dev,
                            n_iter=acfg['n_iter'],
                            lr=acfg['lr'],
                            tv_lambda=acfg.get('tv_lambda', 1e-4),
                            l2_lambda=acfg.get('l2_lambda', 1e-6),
                            bn_lambda=acfg.get('bn_lambda', 0.1),
                        )
                    elif attack_name == 'gi_nas':
                        attack = GINASFull(
                            model, criterion, gpu_dev,
                            n_iter=acfg['n_iter'],
                            lr=acfg['lr'],
                            n_restarts=acfg.get('n_restarts', 8),
                        )
                    else:  # ggcdm
                        attack = GGCDMFull(
                            model, criterion, gpu_dev,
                            n_iter=acfg['n_iter'],
                            guidance_rate=acfg.get('guidance_rate', 0.20),
                        )

                    result = attack.attack(enc_gd_gpu, enc_flat)
                    recon_images = result.get('reconstructed_images')

                    # Compute reconstruction metrics
                    metrics = {}
                    if recon_images is not None and original_images is not None:
                        orig = original_images.to(gpu_dev)
                        metrics = compute_all_reconstruction_metrics(
                            orig, recon_images,
                            labels_true=labels.cpu().numpy() if labels is not None else None,
                            use_lpips=True,
                        )
                    metrics['gradient_score'] = result.get('score', 0.0)
                    client_attack_results[attack_name] = metrics

                except Exception as e:
                    print(f"      {attack_name} failed: {e}")
                    client_attack_results[attack_name] = {'error': str(e)}

            attack_time = time.time() - t0
            client_attack_results['attack_time_s'] = attack_time
            all_attack_results[f'client_{cid}'] = client_attack_results

            print(f"    Client {cid} attacks done in {attack_time:.1f}s")
            del model
            torch.cuda.empty_cache()

        # Aggregate attack metrics across clients
        summary = self._aggregate_attack_metrics(all_attack_results)
        all_attack_results['summary'] = summary
        return all_attack_results

    def _aggregate_attack_metrics(self, attack_results):
        """Average attack metrics across clients."""
        attack_names = ['gradinversion', 'gi_nas', 'ggcdm']
        metric_names = ['mse', 'psnr', 'ssim', 'lpips', 'fid', 'asr',
                        'cosine_similarity', 'knn_distance', 'ddcs', 'avd',
                        'mcc', 'gmean', 'gradient_score']
        summary = {}

        for aname in attack_names:
            agg = {}
            for mname in metric_names:
                vals = []
                for ckey, cresults in attack_results.items():
                    if not ckey.startswith('client_'):
                        continue
                    if aname in cresults and mname in cresults[aname]:
                        v = cresults[aname][mname]
                        if isinstance(v, (int, float)):
                            vals.append(v)
                if vals:
                    agg[mname] = float(np.mean(vals))
            summary[aname] = agg

        # Cross-attack averages
        for mname in metric_names:
            vals = [summary[a].get(mname, 0) for a in attack_names if mname in summary.get(a, {})]
            if vals:
                summary[f'mean_{mname}'] = float(np.mean(vals))

        return summary

    def _evaluate(self, model, test_dataset):
        """Evaluate model on test set."""
        from torch.utils.data import DataLoader
        model.eval()
        correct, total = 0, 0
        total_loss = 0.0
        n_batches = 0
        criterion = torch.nn.CrossEntropyLoss()
        loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

        with torch.no_grad():
            for imgs, lbls in loader:
                imgs, lbls = imgs.to(self.device), lbls.to(self.device)
                out = model(imgs)
                total_loss += criterion(out, lbls).item()
                n_batches += 1
                preds = out.argmax(1)
                correct += preds.eq(lbls).sum().item()
                total += lbls.size(0)

        model.train()
        acc = correct / max(total, 1)
        test_loss = total_loss / max(n_batches, 1)
        return acc, test_loss

    def _log_to_tb(self, tb_logger, rnd, round_metrics, attack_results, client_artifacts):
        """Log scenario results to Tensorboard."""
        w = tb_logger.writer
        step = rnd + 1
        scenario_name = list(self.scenario.keys())[0] if isinstance(self.scenario, dict) else 'unknown'

        # Global metrics
        w.add_scalar('Global/Accuracy_%', round_metrics.get('accuracy', 0) * 100, step)
        w.add_scalar('Global/Test_Loss', round_metrics.get('test_loss', 0), step)
        w.add_scalar('LeakScore/Combined', round_metrics.get('avg_leakscore', 0), step)
        w.add_scalar('Encryption/Pct_Encrypted', round_metrics.get('avg_pct_encrypted', 0) * 100, step)

        # Attack metrics
        summary = attack_results.get('summary', {})
        for attack_name in ['gradinversion', 'gi_nas', 'ggcdm']:
            if attack_name in summary:
                for metric_name, value in summary[attack_name].items():
                    if isinstance(value, (int, float)):
                        tag = f'Attacks/{attack_name}/{metric_name}'
                        w.add_scalar(tag, value, step)

        # Cross-attack averages
        for key, value in summary.items():
            if key.startswith('mean_') and isinstance(value, (int, float)):
                w.add_scalar(f'Attacks_Summary/{key}', value, step)

        # Log reconstructed images (from first client with results)
        for ckey, cresults in attack_results.items():
            if not ckey.startswith('client_'):
                continue
            cid = int(ckey.split('_')[1])
            if cid in client_artifacts:
                orig_images = client_artifacts[cid].get('images')
                if orig_images is not None:
                    from torchvision.utils import make_grid
                    grid = make_grid(orig_images[:8].cpu(), nrow=4, normalize=True)
                    w.add_image('Images/Original', grid, step)
                break

        w.flush()
