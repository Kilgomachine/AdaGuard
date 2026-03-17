"""Federated Learning Simulator with AdaGuard integration.

Orchestrates FL rounds with leakage detection and adaptive encryption.
"""

import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..metrics import (
    EntropyLeakScoreMetric,
    GLMIPMetric, ConfidenceGapMetric, CosineSimilarityMetric,
    FisherInformationMetric, MaskCryptMetric,
    CombinedLeakScore,
    EmpiricalLeakScoreMetric,
    ALL_LEAK_METRICS,
)
from ..metrics.magnitude import GradientMagnitudeMetric
from ..encryption import AdaptiveEncryptionController, FisherEncryptor, MaskCryptEncryptor
from .client import FLClient
from .aggregator import fedavg_aggregate, apply_gradient_update


class FederatedSimulator:
    """Full FL simulation with AdaGuard leakage detection and adaptive encryption."""

    def __init__(self, model, train_dataset, test_dataset, client_data_map,
                 config, device):
        self.global_model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.client_data_map = client_data_map
        self.config = config
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

        # Create clients
        self.clients = {
            cid: FLClient(cid, train_dataset, indices, config, device)
            for cid, indices in client_data_map.items()
        }

        # Initialize metrics
        self.entropy_metric = EntropyLeakScoreMetric(
            num_bins=config['entropy_bins'],
        )
        self.glmip_metric = GLMIPMetric()
        self.conf_gap_metric = ConfidenceGapMetric()
        self.cosine_metric = CosineSimilarityMetric()
        self.fisher_metric = FisherInformationMetric(
            topk=config['fisher_topk'],
            enc_pct=config['encryption_top_percent'],
        )
        self.maskcrypt_metric = MaskCryptMetric(
            enc_pct=config['encryption_top_percent'],
        )
        self.combined_metric = CombinedLeakScore(
            alpha=config['alpha'],
            beta=config['beta'],
            gamma=config['gamma'],
        )
        self.empirical_metric = EmpiricalLeakScoreMetric(
            model, self.criterion, device,
            n_iter=config['empirical_iterations'],
            lr=config['empirical_lr'],
            focus_layers=config['focus_layers'],
        )
        self.magnitude_metric = GradientMagnitudeMetric()

        # Initialize encryption
        self.controller = AdaptiveEncryptionController(
            T1=config['T1'],
            T2=config['T2'],
            base_encrypt_pct=config['encryption_top_percent'],
        )
        self.fisher_encryptor = FisherEncryptor(
            fisher_metric=self.fisher_metric,
        )
        self.maskcrypt_encryptor = MaskCryptEncryptor(
            maskcrypt_metric=self.maskcrypt_metric,
        )

        # History
        self.round_history = []
        self.weights_before_round = None
        self.weights_after_round = None
        # MaskCrypt paper: track exposed weights from previous round
        self.weights_previous_round = None

    def run_round(self, rnd, encryption_strategy='fisher', num_clients=None,
                  batch_size=None, skip_glmip=False, skip_empirical=False):
        """Run one FL round with full AdaGuard pipeline."""
        nc = num_clients or self.config['clients_per_round']
        avail = list(self.clients.keys())
        selected = random.sample(avail, min(nc, len(avail)))

        # Track previous round's exposed weights for MaskCrypt paper formula
        if self.weights_before_round is not None:
            self.weights_previous_round = self.weights_before_round

        # Save weights before round
        self.weights_before_round = {
            n: p.clone().detach()
            for n, p in self.global_model.named_parameters()
        }

        global_state = self.global_model.state_dict()
        client_results = []
        all_gradients = []

        for cid in selected:
            result = self.clients[cid].train_step(global_state, batch_size)
            if result is None:
                continue

            gd = result['gradient_dict']
            flat = result['flat_gradient']
            outputs = result['outputs']
            local_weights = result.get('local_weights', {})
            original_images = result.get('images', None)

            # --- Compute LeakScore components ---
            metrics = {}

            # Entropy LeakScore (with timing)
            t0 = time.perf_counter()
            entropy_r = self.entropy_metric.compute(
                flat, gradient_dict=gd,
                focus_layers=self.config['focus_layers'],
            )
            metrics['entropy_compute_time'] = time.perf_counter() - t0
            metrics.update(entropy_r)

            # Label LeakScore (GLMIP + ConfGap + Cosine)
            if not skip_glmip:
                t0 = time.perf_counter()
                gl = self.glmip_metric.compute(
                    self.global_model, self.train_dataset, self.device,
                    self.criterion, self.config['num_classes'],
                    self.config['mi_samples_per_class'],
                    focus_layers=self.config['focus_layers'],
                )
                metrics['glmip_compute_time'] = time.perf_counter() - t0
                metrics['glmip_score'] = gl['glmip_score']
                class_means = gl.get('class_means', {})
            else:
                metrics['glmip_score'] = 0.0
                metrics['glmip_compute_time'] = 0.0
                class_means = {}

            cg = self.conf_gap_metric.compute(outputs)
            metrics.update(cg)
            cos_r = self.cosine_metric.compute(class_means)
            metrics.update(cos_r)

            # Empirical LeakScore (with timing + reconstruction quality)
            if not skip_empirical:
                t0 = time.perf_counter()
                emp_r = self.empirical_metric.compute(
                    gd, flat, original_images=original_images,
                )
                metrics['empirical_compute_time'] = time.perf_counter() - t0
                metrics.update(emp_r)
            else:
                metrics['empirical_gradinversion'] = 0.0
                metrics['empirical_ginas'] = 0.0
                metrics['empirical_ggcdm'] = 0.0
                metrics['empirical_mean'] = 0.0
                metrics['empirical_compute_time'] = 0.0
                metrics['recon_mse'] = 0.0
                metrics['recon_psnr'] = 0.0
                metrics['recon_ssim'] = 0.0

            # Fisher Information (with timing)
            t0 = time.perf_counter()
            fisher_r = self.fisher_metric.compute(gd)
            metrics['fisher_compute_time'] = time.perf_counter() - t0
            metrics.update(fisher_r)

            # Pre-compute Fisher per-weight histogram for visualization
            if fisher_r.get('fisher_per_weight_norm') is not None and fisher_r['fisher_per_weight_norm'].numel() > 0:
                fw = fisher_r['fisher_per_weight_norm']
                metrics['fisher_hist'] = torch.histc(fw.float(), bins=100, min=0, max=max(fw.max().item(), 1e-12)).tolist()
                metrics['fisher_per_weight_mean'] = fw.mean().item()
                metrics['fisher_per_weight_std'] = fw.std().item()
                metrics['fisher_per_weight_median'] = fw.median().item()
                metrics['fisher_per_weight_p95'] = torch.quantile(fw.float(), 0.95).item()

            # MaskCrypt (with timing) — using paper-correct formula
            t0 = time.perf_counter()
            # Paper formula: v[i] = g[i] * (w_exposed_prev - w_trained_current)
            exposed_w = self.weights_previous_round or self.weights_before_round
            mc_r = self.maskcrypt_metric.compute(gd, exposed_w, local_weights)
            metrics['maskcrypt_compute_time'] = time.perf_counter() - t0
            metrics.update(mc_r)

            # Pre-compute MaskCrypt per-weight histogram for visualization
            if mc_r.get('maskcrypt_per_weight_abs') is not None and mc_r['maskcrypt_per_weight_abs'].numel() > 0:
                mw = mc_r['maskcrypt_per_weight_abs']
                max_mw = max(mw.max().item(), 1e-12)
                metrics['maskcrypt_hist'] = torch.histc(mw.float(), bins=100, min=0, max=max_mw).tolist()
                metrics['maskcrypt_hist_max'] = max_mw
                metrics['maskcrypt_per_weight_mean'] = mw.mean().item()
                metrics['maskcrypt_per_weight_std'] = mw.std().item()
                metrics['maskcrypt_per_weight_median'] = mw.median().item()
                metrics['maskcrypt_per_weight_p95'] = torch.quantile(mw.float(), 0.95).item()

            # Gradient Magnitude Score
            mag_r = self.magnitude_metric.compute(gd)
            metrics.update(mag_r)

            # Per-parameter timing
            total_params = sum(g.numel() for g in gd.values())
            metrics['total_params'] = total_params
            if total_params > 0:
                metrics['fisher_per_param_us'] = metrics['fisher_compute_time'] * 1e6 / total_params
                metrics['maskcrypt_per_param_us'] = metrics['maskcrypt_compute_time'] * 1e6 / total_params

            # --- Combined LeakScore ---
            entropy_avg = np.mean([
                metrics['shannon_leak_score'],
                metrics['renyi_leak_score'],
                metrics['min_entropy_leak_score'],
            ])
            label_avg = np.mean([
                metrics['glmip_score'],
                metrics['confidence_gap'],
                metrics.get('cosine_leak_score', 0.0),
            ])
            empirical_avg = np.mean([
                metrics['empirical_gradinversion'],
                metrics['empirical_ginas'],
                metrics['empirical_ggcdm'],
            ])

            combined = self.combined_metric.compute(
                entropy_avg, label_avg, empirical_avg,
            )
            metrics['combined_leakscore'] = combined
            metrics['entropy_avg'] = float(entropy_avg)
            metrics['label_avg'] = float(label_avg)
            metrics['empirical_avg'] = float(empirical_avg)

            # --- Adaptive Encryption ---
            policy = self.controller.decide(combined)
            k = self.controller.compute_k(combined, total_params)

            metrics['encryption_level'] = policy.level
            metrics['encryption_pct'] = policy.encrypt_pct
            metrics['params_encrypted'] = k

            # Apply encryption
            if encryption_strategy == 'none' or policy.level == 'none':
                protected_gd = gd
                metrics['actual_pct_encrypted'] = 0.0
            elif encryption_strategy == 'full':
                protected_gd = {n: torch.zeros_like(g) for n, g in gd.items()}
                metrics['actual_pct_encrypted'] = 1.0
            elif encryption_strategy == 'maskcrypt':
                protected_gd, enc_meta = self.maskcrypt_encryptor.encrypt(
                    gd, exposed_w, local_weights, k=k, maskcrypt_result=mc_r,
                )
                metrics['actual_pct_encrypted'] = enc_meta['pct_encrypted']
            else:  # 'fisher' (default)
                protected_gd, enc_meta = self.fisher_encryptor.encrypt(
                    gd, k=k, fisher_result=fisher_r,
                )
                metrics['actual_pct_encrypted'] = enc_meta['pct_encrypted']

            metrics['loss'] = result['loss']
            all_gradients.append(protected_gd)
            client_results.append({
                'client_id': cid,
                'metrics': metrics,
            })

        # --- Server Aggregation ---
        if all_gradients:
            avg_grads = fedavg_aggregate(all_gradients)
            apply_gradient_update(
                self.global_model, avg_grads, self.config['fl_lr'],
            )

        self.weights_after_round = {
            n: p.clone().detach()
            for n, p in self.global_model.named_parameters()
        }

        # --- Evaluate ---
        accuracy = self._evaluate()

        # --- Aggregate round metrics ---
        round_summary = self._summarize_round(rnd, client_results, accuracy)
        self.round_history.append(round_summary)

        return round_summary

    def _evaluate(self):
        """Evaluate global model on test set."""
        self.global_model.eval()
        correct, total = 0, 0
        loader = DataLoader(self.test_dataset, batch_size=128, shuffle=False)

        with torch.no_grad():
            for imgs, lbls in loader:
                imgs, lbls = imgs.to(self.device), lbls.to(self.device)
                preds = self.global_model(imgs).argmax(1)
                correct += preds.eq(lbls).sum().item()
                total += lbls.size(0)

        self.global_model.train()
        return correct / max(total, 1)

    def _summarize_round(self, rnd, client_results, accuracy):
        """Average metrics across clients for round summary."""
        summary = {'round': rnd + 1, 'accuracy': accuracy}

        if not client_results:
            return summary

        # Average all scalar metrics
        all_keys = set()
        for cr in client_results:
            for k, v in cr['metrics'].items():
                if isinstance(v, (int, float)):
                    all_keys.add(k)

        for k in all_keys:
            vals = [
                cr['metrics'][k]
                for cr in client_results
                if k in cr['metrics'] and isinstance(cr['metrics'][k], (int, float))
            ]
            if vals:
                summary[k] = float(np.mean(vals))

        # Preserve list-type data (histograms) from first client
        list_keys = ['fisher_hist', 'maskcrypt_hist']
        for lk in list_keys:
            for cr in client_results:
                if lk in cr['metrics'] and isinstance(cr['metrics'][lk], list):
                    summary[lk] = cr['metrics'][lk]
                    break

        summary['client_details'] = client_results
        return summary

    def pretrain(self, progress_callback=None):
        """Pre-train the global model on full training set.

        Args:
            progress_callback: optional callable(epoch, total, acc, loss) for UI updates

        Returns:
            list of {epoch, loss, accuracy} dicts for plotting
        """
        import torch.optim as optim

        self.global_model.train()
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['pretrain_batch_size'],
            shuffle=True, num_workers=0,
        )
        optimizer = optim.SGD(
            self.global_model.parameters(),
            lr=self.config['pretrain_lr'], momentum=0.9,
        )

        history = []
        for epoch in range(self.config['pretrain_epochs']):
            running_loss, correct, total = 0.0, 0, 0
            for bi, (imgs, lbls) in enumerate(loader):
                imgs, lbls = imgs.to(self.device), lbls.to(self.device)
                optimizer.zero_grad()
                out = self.global_model(imgs)
                loss = self.criterion(out, lbls)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                _, pred = out.max(1)
                total += lbls.size(0)
                correct += pred.eq(lbls).sum().item()

            acc = 100.0 * correct / total
            avg_loss = running_loss / len(loader)
            history.append({'epoch': epoch + 1, 'loss': avg_loss, 'accuracy': acc})
            print(f"  Pretrain Epoch {epoch+1}/{self.config['pretrain_epochs']} "
                  f"— Loss: {avg_loss:.4f}, Acc: {acc:.1f}%")

            if progress_callback:
                progress_callback(epoch, self.config['pretrain_epochs'], acc, avg_loss)

        print("Pre-training complete.")
        return history
