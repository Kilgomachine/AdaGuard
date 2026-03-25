"""Federated Learning Simulator with AdaGuard integration.

Orchestrates FL rounds with leakage detection and adaptive encryption.
Supports multi-GPU parallelism for client processing.
"""

import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..models import create_model
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
from .aggregator import fedavg_aggregate, fedavg_aggregate_state_dicts, apply_gradient_update


def _get_gpu_devices():
    """Return list of available CUDA devices, or [cpu] if none."""
    if torch.cuda.is_available():
        return [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return [torch.device('cpu')]


def _process_client(cid, client, global_state, batch_size, gpu_device, config,
                    global_model_state, exposed_w, skip_glmip, skip_empirical,
                    train_dataset):
    """Process a single client on a specific GPU. Returns (cid, metrics, protected_gd) or None.

    This function is designed to run in a thread with its own GPU assignment.
    It creates its own metric instances to avoid cross-thread state issues.
    """
    import sys
    t_start = time.perf_counter()
    print(f"    [Client {cid:3d}] Starting on {gpu_device} ...", flush=True)

    # Train the client on assigned GPU
    # Temporarily override client device
    orig_device = client.device
    client.device = gpu_device
    result = client.train_step(global_state, batch_size)
    client.device = orig_device

    if result is None:
        print(f"    [Client {cid:3d}] No data, skipped", flush=True)
        return None

    t_train = time.perf_counter() - t_start
    print(f"    [Client {cid:3d}] Training done ({t_train:.1f}s), computing metrics...", flush=True)

    gd = result['gradient_dict']           # raw gradient for LeakScore
    weight_delta = result.get('weight_delta', gd)  # weight delta for FedAvg
    flat = result['flat_gradient']
    outputs = result['outputs']
    local_weights = result.get('local_weights', {})
    original_images = result.get('images', None)

    # Move tensors to assigned GPU
    gd = {k: v.to(gpu_device) for k, v in gd.items()}
    weight_delta_cpu = {k: v.cpu() for k, v in weight_delta.items()}  # keep on CPU for aggregation
    flat = flat.to(gpu_device)
    outputs = outputs.to(gpu_device)
    local_weights = {k: v.to(gpu_device) for k, v in local_weights.items()}
    if original_images is not None:
        original_images = original_images.to(gpu_device)
    exposed_w_gpu = {k: v.to(gpu_device) for k, v in exposed_w.items()}

    # Create per-thread metric instances on this GPU
    entropy_metric = EntropyLeakScoreMetric(num_bins=config['entropy_bins'])
    fisher_metric = FisherInformationMetric(
        topk=config['fisher_topk'], enc_pct=config['encryption_top_percent'],
    )
    maskcrypt_metric = MaskCryptMetric(enc_pct=config['encryption_top_percent'])
    magnitude_metric = GradientMagnitudeMetric()
    conf_gap_metric = ConfidenceGapMetric()
    cosine_metric = CosineSimilarityMetric()
    combined_metric = CombinedLeakScore(
        alpha=config['alpha'], beta=config['beta'], gamma=config['gamma'],
    )

    metrics = {}

    # Entropy LeakScore
    t0 = time.perf_counter()
    entropy_r = entropy_metric.compute(
        flat, gradient_dict=gd, focus_layers=config['focus_layers'],
    )
    t_entropy = time.perf_counter() - t0
    metrics['entropy_compute_time'] = t_entropy
    metrics.update(entropy_r)
    print(f"    [Client {cid:3d}]   Entropy:    {t_entropy:5.1f}s  "
          f"(Sh={entropy_r.get('shannon_leak_score', 0):.3f} "
          f"Re={entropy_r.get('renyi_leak_score', 0):.3f} "
          f"Min={entropy_r.get('min_entropy_leak_score', 0):.3f})", flush=True)

    # Label LeakScore (GLMIP + ConfGap + Cosine)
    if not skip_glmip:
        t0 = time.perf_counter()
        # Create a model copy on this GPU for GLMIP
        glmip_model = create_model(
            config.get('model', 'smallcnn'), num_classes=config['num_classes'],
        ).to(gpu_device)
        glmip_model.load_state_dict(global_model_state)
        glmip_metric = GLMIPMetric()
        criterion = nn.CrossEntropyLoss()
        gl = glmip_metric.compute(
            glmip_model, train_dataset, gpu_device,
            criterion, config['num_classes'],
            config['mi_samples_per_class'],
            focus_layers=config['focus_layers'],
        )
        t_glmip = time.perf_counter() - t0
        metrics['glmip_compute_time'] = t_glmip
        metrics['glmip_score'] = gl['glmip_score']
        class_means = gl.get('class_means', {})
        del glmip_model
        print(f"    [Client {cid:3d}]   GLMIP:      {t_glmip:5.1f}s  (score={gl['glmip_score']:.3f})", flush=True)
    else:
        metrics['glmip_score'] = 0.0
        metrics['glmip_compute_time'] = 0.0
        class_means = {}

    cg = conf_gap_metric.compute(outputs)
    metrics.update(cg)
    cos_r = cosine_metric.compute(class_means)
    metrics.update(cos_r)
    print(f"    [Client {cid:3d}]   Label:             "
          f"(ConfGap={cg.get('confidence_gap', 0):.3f} "
          f"Cosine={cos_r.get('cosine_leak_score', 0):.3f})", flush=True)

    # Empirical LeakScore
    if not skip_empirical:
        t0 = time.perf_counter()
        emp_model = create_model(
            config.get('model', 'smallcnn'), num_classes=config['num_classes'],
        ).to(gpu_device)
        emp_model.load_state_dict(global_model_state)
        criterion = nn.CrossEntropyLoss()
        emp_metric = EmpiricalLeakScoreMetric(
            emp_model, criterion, gpu_device,
            n_iter=config['empirical_iterations'],
            lr=config['empirical_lr'],
            focus_layers=config['focus_layers'],
        )
        emp_r = emp_metric.compute(gd, flat, original_images=original_images)
        t_emp = time.perf_counter() - t0
        metrics['empirical_compute_time'] = t_emp
        metrics.update(emp_r)
        del emp_model
        print(f"    [Client {cid:3d}]   Empirical:  {t_emp:5.1f}s  "
              f"(GI={emp_r.get('empirical_gradinversion', 0):.3f} "
              f"NAS={emp_r.get('empirical_ginas', 0):.3f} "
              f"GGCDM={emp_r.get('empirical_ggcdm', 0):.3f})", flush=True)
    else:
        metrics['empirical_gradinversion'] = 0.0
        metrics['empirical_ginas'] = 0.0
        metrics['empirical_ggcdm'] = 0.0
        metrics['empirical_mean'] = 0.0
        metrics['empirical_compute_time'] = 0.0
        metrics['recon_mse'] = 0.0
        metrics['recon_psnr'] = 0.0
        metrics['recon_ssim'] = 0.0

    # Fisher Information
    t0 = time.perf_counter()
    fisher_r = fisher_metric.compute(gd)
    t_fisher = time.perf_counter() - t0
    metrics['fisher_compute_time'] = t_fisher
    metrics.update(fisher_r)
    print(f"    [Client {cid:3d}]   Fisher:     {t_fisher:5.1f}s  "
          f"(conc={fisher_r.get('fisher_concentration', 0):.4f} "
          f"norm={fisher_r.get('fisher_round_norm', 0):.4f})", flush=True)

    # Fisher per-weight histogram
    if fisher_r.get('fisher_per_weight_norm') is not None and fisher_r['fisher_per_weight_norm'].numel() > 0:
        fw = fisher_r['fisher_per_weight_norm']
        metrics['fisher_hist'] = torch.histc(fw.float(), bins=100, min=0, max=max(fw.max().item(), 1e-12)).tolist()
        metrics['fisher_per_weight_mean'] = fw.mean().item()
        metrics['fisher_per_weight_std'] = fw.std().item()
        metrics['fisher_per_weight_median'] = fw.median().item()
        metrics['fisher_per_weight_p95'] = torch.quantile(fw.float(), 0.95).item()

    # MaskCrypt — paper-correct formula
    t0 = time.perf_counter()
    mc_r = maskcrypt_metric.compute(gd, exposed_w_gpu, local_weights)
    t_mc = time.perf_counter() - t0
    metrics['maskcrypt_compute_time'] = t_mc
    metrics.update(mc_r)
    print(f"    [Client {cid:3d}]   MaskCrypt:  {t_mc:5.1f}s  "
          f"(vuln={mc_r.get('maskcrypt_vulnerability', 0):.4f})", flush=True)

    # MaskCrypt per-weight histogram
    if mc_r.get('maskcrypt_per_weight_abs') is not None and mc_r['maskcrypt_per_weight_abs'].numel() > 0:
        mw = mc_r['maskcrypt_per_weight_abs']
        max_mw = max(mw.max().item(), 1e-12)
        metrics['maskcrypt_hist'] = torch.histc(mw.float(), bins=100, min=0, max=max_mw).tolist()
        metrics['maskcrypt_hist_max'] = max_mw
        metrics['maskcrypt_per_weight_mean'] = mw.mean().item()
        metrics['maskcrypt_per_weight_std'] = mw.std().item()
        metrics['maskcrypt_per_weight_median'] = mw.median().item()
        metrics['maskcrypt_per_weight_p95'] = torch.quantile(mw.float(), 0.95).item()

    # Gradient Magnitude
    mag_r = magnitude_metric.compute(gd)
    metrics.update(mag_r)

    # Per-parameter timing
    total_params = sum(g.numel() for g in gd.values())
    metrics['total_params'] = total_params
    if total_params > 0:
        metrics['fisher_per_param_us'] = metrics['fisher_compute_time'] * 1e6 / total_params
        metrics['maskcrypt_per_param_us'] = metrics['maskcrypt_compute_time'] * 1e6 / total_params

    # Combined LeakScore
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

    combined = combined_metric.compute(entropy_avg, label_avg, empirical_avg)
    metrics['combined_leakscore'] = combined
    metrics['entropy_avg'] = float(entropy_avg)
    metrics['label_avg'] = float(label_avg)
    metrics['empirical_avg'] = float(empirical_avg)

    metrics['loss'] = result['loss']

    t_total = time.perf_counter() - t_start
    print(f"    [Client {cid:3d}] ✓ Done in {t_total:.1f}s  |  "
          f"LeakScore={combined:.4f}  Loss={result['loss']:.4f}  "
          f"Magnitude={mag_r.get('magnitude_score', 0):.4f}", flush=True)

    local_state_cpu = {k: v.cpu() for k, v in result.get('local_state_dict', {}).items()}

    return cid, metrics, weight_delta_cpu, fisher_r, mc_r, exposed_w_gpu, local_weights, local_state_cpu


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

        # Detect available GPUs
        self.gpu_devices = _get_gpu_devices()
        self.num_gpus = len(self.gpu_devices)
        print(f"Multi-GPU: {self.num_gpus} device(s) available: {self.gpu_devices}")

        # Create clients (on primary device initially)
        self.clients = {
            cid: FLClient(cid, train_dataset, indices, config, device)
            for cid, indices in client_data_map.items()
        }

        # Initialize encryption (runs on primary device after client results return)
        self.controller = AdaptiveEncryptionController(
            T1=config['T1'], T2=config['T2'],
            base_encrypt_pct=config['encryption_top_percent'],
        )
        self.fisher_metric = FisherInformationMetric(
            topk=config['fisher_topk'], enc_pct=config['encryption_top_percent'],
        )
        self.maskcrypt_metric = MaskCryptMetric(
            enc_pct=config['encryption_top_percent'],
        )
        self.fisher_encryptor = FisherEncryptor(fisher_metric=self.fisher_metric)
        self.maskcrypt_encryptor = MaskCryptEncryptor(maskcrypt_metric=self.maskcrypt_metric)

        # History
        self.round_history = []
        self.weights_before_round = None
        self.weights_after_round = None
        self.weights_previous_round = None

    def run_round(self, rnd, encryption_strategy='fisher', num_clients=None,
                  batch_size=None, skip_glmip=False, skip_empirical=False):
        """Run one FL round with full AdaGuard pipeline.

        Clients are processed in parallel across available GPUs.
        """
        import sys
        nc = num_clients or self.config['clients_per_round']
        avail = list(self.clients.keys())
        selected = random.sample(avail, min(nc, len(avail)))

        print(f"\n{'='*70}", flush=True)
        print(f"  ROUND {rnd+1}  |  {len(selected)} clients  |  strategy={encryption_strategy}  "
              f"|  GPUs={self.num_gpus}", flush=True)
        print(f"{'='*70}", flush=True)

        # Track previous round's exposed weights for MaskCrypt paper formula
        if self.weights_before_round is not None:
            self.weights_previous_round = self.weights_before_round

        # Save weights before round
        self.weights_before_round = {
            n: p.clone().detach().cpu()
            for n, p in self.global_model.named_parameters()
        }

        global_state = {k: v.cpu() for k, v in self.global_model.state_dict().items()}
        global_model_state = global_state  # for creating model copies on other GPUs

        exposed_w = self.weights_previous_round or self.weights_before_round

        client_results = []
        all_gradients = []
        all_state_dicts = []  # full state dicts for BN buffer averaging

        # Assign clients to GPUs round-robin
        gpu_assignments = {cid: self.gpu_devices[i % self.num_gpus]
                           for i, cid in enumerate(selected)}

        # Process clients in parallel across GPUs
        cpg = self.config.get('clients_per_gpu', 3)
        if cpg <= 0:
            # Auto: estimate from available GPU memory
            cpg = max(1, len(selected) // self.num_gpus)
        clients_per_gpu = cpg
        max_workers = min(len(selected), self.num_gpus * clients_per_gpu)
        print(f"  [Parallel] {max_workers} workers across {self.num_gpus} GPUs "
              f"({clients_per_gpu} clients/GPU)", flush=True)

        if max_workers > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {}
                for cid in selected:
                    gpu_dev = gpu_assignments[cid]
                    future = executor.submit(
                        _process_client, cid, self.clients[cid],
                        global_state, batch_size, gpu_dev, self.config,
                        global_model_state, exposed_w,
                        skip_glmip, skip_empirical, self.train_dataset,
                    )
                    futures[future] = cid

                for future in as_completed(futures):
                    cid = futures[future]
                    try:
                        result = future.result()
                    except Exception as e:
                        print(f"  [ERROR] Client {cid}: {e}", flush=True)
                        continue

                    if result is None:
                        continue

                    cid, metrics, gd_cpu, fisher_r, mc_r, exposed_w_gpu, local_weights, local_state_cpu = result
                    self._apply_encryption(
                        metrics, gd_cpu, encryption_strategy,
                        fisher_r, mc_r, exposed_w_gpu, local_weights,
                        all_gradients, client_results, cid,
                        local_state_cpu, all_state_dicts,
                    )
        else:
            # Single GPU — run sequentially (same as before, avoids thread overhead)
            for cid in selected:
                gpu_dev = self.gpu_devices[0]
                result = _process_client(
                    cid, self.clients[cid], global_state, batch_size,
                    gpu_dev, self.config, global_model_state, exposed_w,
                    skip_glmip, skip_empirical, self.train_dataset,
                )
                if result is None:
                    continue

                cid, metrics, gd_cpu, fisher_r, mc_r, exposed_w_gpu, local_weights, local_state_cpu = result
                self._apply_encryption(
                    metrics, gd_cpu, encryption_strategy,
                    fisher_r, mc_r, exposed_w_gpu, local_weights,
                    all_gradients, client_results, cid,
                    local_state_cpu, all_state_dicts,
                )

        # --- Server Aggregation ---
        print(f"\n  [Server] Aggregating {len(all_gradients)} client updates...", flush=True)
        sys.stdout.flush()
        if all_state_dicts:
            # Proper FedAvg: average full state dicts (params + BN buffers)
            avg_state = fedavg_aggregate_state_dicts(all_state_dicts)
            avg_state = {k: v.to(self.device) for k, v in avg_state.items()}
            self.global_model.load_state_dict(avg_state)
            print(f"  [Server] FedAvg applied (full state dict with BN stats)", flush=True)
        elif all_gradients:
            # Fallback: param-only update (no BN averaging)
            avg_grads = fedavg_aggregate(all_gradients)
            avg_grads = {k: v.to(self.device) for k, v in avg_grads.items()}
            apply_gradient_update(
                self.global_model, avg_grads, self.config['fl_lr'],
            )

        self.weights_after_round = {
            n: p.clone().detach()
            for n, p in self.global_model.named_parameters()
        }

        # --- Evaluate ---
        print(f"  [Server] Evaluating global model...", flush=True)
        accuracy = self._evaluate()
        print(f"  [Server] Round {rnd+1} accuracy: {accuracy*100:.1f}%", flush=True)
        sys.stdout.flush()

        # --- Aggregate round metrics ---
        round_summary = self._summarize_round(rnd, client_results, accuracy)
        self.round_history.append(round_summary)

        return round_summary

    def _apply_encryption(self, metrics, gd_cpu, encryption_strategy,
                          fisher_r, mc_r, exposed_w_gpu, local_weights,
                          all_gradients, client_results, cid,
                          local_state_cpu=None, all_state_dicts=None):
        """Apply encryption policy and add to results."""
        total_params = metrics.get('total_params', 0)
        combined = metrics.get('combined_leakscore', 0)

        policy = self.controller.decide(combined)
        k = self.controller.compute_k(combined, total_params)

        metrics['encryption_level'] = policy.level
        metrics['encryption_pct'] = policy.encrypt_pct
        metrics['params_encrypted'] = k

        if encryption_strategy == 'none' or policy.level == 'none':
            protected_gd = gd_cpu
            metrics['actual_pct_encrypted'] = 0.0
        elif encryption_strategy == 'full':
            protected_gd = {n: torch.zeros_like(g) for n, g in gd_cpu.items()}
            metrics['actual_pct_encrypted'] = 1.0
        elif encryption_strategy == 'maskcrypt':
            # Move to CPU for encryption
            mc_r_cpu = {k: (v.cpu() if isinstance(v, torch.Tensor) else v)
                        for k, v in mc_r.items()}
            ew_cpu = {k: v.cpu() for k, v in exposed_w_gpu.items()}
            lw_cpu = {k: v.cpu() for k, v in local_weights.items()}
            protected_gd, enc_meta = self.maskcrypt_encryptor.encrypt(
                gd_cpu, ew_cpu, lw_cpu, k=k, maskcrypt_result=mc_r_cpu,
            )
            metrics['actual_pct_encrypted'] = enc_meta['pct_encrypted']
        else:  # 'fisher'
            fisher_r_cpu = {k: (v.cpu() if isinstance(v, torch.Tensor) else v)
                           for k, v in fisher_r.items()}
            protected_gd, enc_meta = self.fisher_encryptor.encrypt(
                gd_cpu, k=k, fisher_result=fisher_r_cpu,
            )
            metrics['actual_pct_encrypted'] = enc_meta['pct_encrypted']

        all_gradients.append(protected_gd)
        if all_state_dicts is not None and local_state_cpu:
            all_state_dicts.append(local_state_cpu)
        client_results.append({'client_id': cid, 'metrics': metrics})

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

        epochs = self.config['pretrain_epochs']
        if epochs <= 0:
            print("Pre-training skipped (pretrain_epochs=0).")
            return []

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
