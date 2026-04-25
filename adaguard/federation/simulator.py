"""Federated Learning Simulator with AdaGuard integration.

Orchestrates FL rounds with leakage detection and adaptive encryption.
Supports multi-GPU parallelism for client processing.
"""

import os
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



# Thread-safe progress tracking
import threading

_progress_lock = threading.Lock()
_progress_count = 0
_progress_total = 0


def _robust_save(obj, path, max_attempts=4, initial_delay=2.0):
    """torch.save with atomic tmp+rename and exponential-backoff retry.

    Lustre (and other parallel filesystems) can transiently reject large
    concurrent writes with
        PytorchStreamWriter failed writing file data/N: file write failed
    which leaves a corrupt half-written file at the target path. This
    helper writes to ``{path}.tmp`` first, renames on success, and retries
    up to ``max_attempts`` times with doubling backoff. Any corrupt tmp
    from a failed attempt is deleted before the next retry.
    """
    tmp_path = str(path) + '.tmp'
    delay = initial_delay
    last_err = None
    for attempt in range(max_attempts):
        try:
            torch.save(obj, tmp_path)
            os.replace(tmp_path, path)
            return
        except (RuntimeError, OSError) as e:
            last_err = e
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass
            if attempt < max_attempts - 1:
                print(f"    [warn] torch.save({path}) failed (attempt "
                      f"{attempt+1}/{max_attempts}): {e} — retrying in "
                      f"{delay:.0f}s", flush=True)
                time.sleep(delay)
                delay *= 2
    # All retries exhausted — re-raise so caller's try/except can log
    raise last_err

# Worker ID pool: each thread grabs a worker ID from its GPU's pool
_worker_ids = {}        # thread_id -> worker_id
_gpu_worker_pools = {}  # gpu_str -> list of available worker IDs
_worker_gpu_map = {}    # worker_id -> gpu_str (for display)


def _acquire_worker(gpu_device):
    """Grab a worker ID for this thread on the given GPU."""
    gpu_str = str(gpu_device)
    with _progress_lock:
        pool = _gpu_worker_pools.get(gpu_str, [])
        if pool:
            wid = pool.pop(0)
        else:
            wid = 0
        _worker_ids[threading.get_ident()] = wid
        return wid


def _release_worker(gpu_device, wid):
    """Return worker ID to the pool."""
    gpu_str = str(gpu_device)
    with _progress_lock:
        if gpu_str in _gpu_worker_pools:
            _gpu_worker_pools[gpu_str].append(wid)
        if threading.get_ident() in _worker_ids:
            del _worker_ids[threading.get_ident()]


def _process_client(cid, client, global_state, batch_size, gpu_device, config,
                    global_model_state, exposed_w_gpu, skip_glmip, skip_empirical,
                    train_dataset, save_dir=None, artifacts_dir=None):
    """Process a single client: train + fast metrics. Attacks are deferred to Phase 2.

    Fast metrics (run inline): Entropy, Fisher, Magnitude, ConfGap, Cosine, GLMIP
    Slow attacks (Phase 2): Empirical GI attacks — run offline via run_scenario.py.

    If artifacts_dir is set, saves comprehensive per-client data for Phase 2 replay.
    """
    global _progress_count
    t_start = time.perf_counter()

    # Acquire a worker ID for this thread
    wid = _acquire_worker(gpu_device)

    # ── Phase 1: Local Training ──
    t_phase = time.perf_counter()
    orig_device = client.device
    client.device = gpu_device
    result = client.train_step(global_state, batch_size)
    client.device = orig_device
    t_train = time.perf_counter() - t_phase

    if result is None:
        _release_worker(gpu_device, wid)
        return None

    gd = result['gradient_dict']
    weight_delta = result.get('weight_delta', gd)
    flat = result['flat_gradient']
    outputs = result['outputs']
    labels = result.get('labels', None)
    local_weights = result.get('local_weights', {})
    original_images = result.get('images', None)

    # Move tensors to assigned GPU
    gd = {k: v.to(gpu_device) for k, v in gd.items()}
    weight_delta_cpu = {k: v.cpu() for k, v in weight_delta.items()}
    flat = flat.to(gpu_device)
    outputs = outputs.to(gpu_device)
    local_weights = {k: v.to(gpu_device) for k, v in local_weights.items()}
    if original_images is not None:
        original_images = original_images.to(gpu_device)

    # ── Fast metrics (all <1s each) ──
    metrics = {}
    timing = {}

    # Entropy
    t0 = time.perf_counter()
    entropy_metric = EntropyLeakScoreMetric(num_bins=config['entropy_bins'])
    entropy_r = entropy_metric.compute(
        flat, gradient_dict=gd, focus_layers=config['focus_layers'],
    )
    timing['entropy'] = time.perf_counter() - t0
    metrics['entropy_compute_time'] = timing['entropy']
    metrics.update(entropy_r)

    # Confidence Gap (Eq. 4: gap over logit gradients ∂L/∂z)
    conf_gap_metric = ConfidenceGapMetric()
    cg = conf_gap_metric.compute(outputs, labels=labels)
    metrics.update(cg)

    # Fisher
    t0 = time.perf_counter()
    fisher_metric = FisherInformationMetric(
        topk=config['fisher_topk'], enc_pct=config['encryption_top_percent'],
    )
    fisher_r = fisher_metric.compute(gd)
    timing['fisher'] = time.perf_counter() - t0
    metrics['fisher_compute_time'] = timing['fisher']
    metrics.update(fisher_r)

    if fisher_r.get('fisher_per_weight_norm') is not None and fisher_r['fisher_per_weight_norm'].numel() > 0:
        fw = fisher_r['fisher_per_weight_norm']
        metrics['fisher_hist'] = torch.histc(fw.float(), bins=100, min=0, max=max(fw.max().item(), 1e-12)).tolist()
        metrics['fisher_per_weight_mean'] = fw.mean().item()
        metrics['fisher_per_weight_std'] = fw.std().item()
        metrics['fisher_per_weight_median'] = fw.median().item()
        metrics['fisher_per_weight_p95'] = torch.quantile(fw.float(), 0.95).item()

    # MaskCrypt — only computed on-demand when encryption_strategy='maskcrypt'
    # (not part of AdaGuard leakscore; used as baseline comparison only)
    mc_r = {}
    metrics['maskcrypt_compute_time'] = 0.0

    # Gradient Magnitude
    magnitude_metric = GradientMagnitudeMetric()
    mag_r = magnitude_metric.compute(gd)
    metrics.update(mag_r)

    # Per-parameter timing
    total_params = sum(g.numel() for g in gd.values())
    metrics['total_params'] = total_params
    if total_params > 0:
        metrics['fisher_per_param_us'] = metrics['fisher_compute_time'] * 1e6 / total_params

    # ── GLMIP + Empirical: run inline OR defer ──
    # BUG FIX: in deferred mode we intentionally do NOT populate glmip_score /
    # cosine_leak_score / empirical_* so that the label_avg/empirical_avg
    # computation below skips them rather than diluting the true risk with 0.0.
    # Compute-time / reconstruction placeholders are still set for logging.
    #
    # Deferred marker: trigger whenever either save_dir or artifacts_dir is
    # provided (i.e., --defer-attacks was set on the CLI). Previously this
    # checked only save_dir, which broke when we stopped writing the
    # redundant client_data dir to stay under disk quota.
    deferred_attacks = (save_dir is not None) or (artifacts_dir is not None)
    if deferred_attacks:
        # DEFERRED MODE: save data for later attack computation
        # (scores will be filled in by run_attacks.py on the saved artifacts)
        metrics['glmip_compute_time'] = 0.0
        metrics['mean_cosine_similarity'] = 0.0
        metrics['empirical_mean'] = 0.0
        metrics['empirical_compute_time'] = 0.0
        metrics['recon_mse'] = 0.0
        metrics['recon_psnr'] = 0.0
        metrics['recon_ssim'] = 0.0

        # Only write the separate client_data file when save_dir is set AND
        # we don't have a slim artifacts_dir (which already holds attack data).
        if save_dir is not None:
            client_save = {
                'gradient_dict': {k: v.cpu() for k, v in gd.items()},
                'flat_gradient': flat.cpu(),
                'outputs': outputs.cpu(),
            }
            if original_images is not None:
                client_save['original_images'] = original_images.cpu()

            import os
            os.makedirs(save_dir, exist_ok=True)
            _robust_save(client_save, os.path.join(save_dir, f'client_{cid}.pt'))
        timing['glmip'] = 0.0
        timing['empirical'] = 0.0
    else:
        # INLINE MODE: run attacks now (slower but complete)
        if not skip_glmip:
            t0 = time.perf_counter()
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
            timing['glmip'] = time.perf_counter() - t0
            metrics['glmip_compute_time'] = timing['glmip']
            metrics['glmip_score'] = gl['glmip_score']
            class_means = gl.get('class_means', {})
            del glmip_model
            cosine_metric = CosineSimilarityMetric()
            cos_r = cosine_metric.compute(class_means)
            metrics.update(cos_r)
        else:
            metrics['glmip_score'] = 0.0
            metrics['glmip_compute_time'] = 0.0
            metrics['cosine_leak_score'] = 0.0
            metrics['mean_cosine_similarity'] = 0.0
            timing['glmip'] = 0.0

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
            timing['empirical'] = time.perf_counter() - t0
            metrics['empirical_compute_time'] = timing['empirical']
            metrics.update(emp_r)
            del emp_model
        else:
            metrics['empirical_gradinversion'] = 0.0
            metrics['empirical_ginas'] = 0.0
            metrics['empirical_ggcdm'] = 0.0
            metrics['empirical_mean'] = 0.0
            metrics['empirical_compute_time'] = 0.0
            metrics['recon_mse'] = 0.0
            metrics['recon_psnr'] = 0.0
            metrics['recon_ssim'] = 0.0
            timing['empirical'] = 0.0

    # Combined LeakScore
    combined_metric = CombinedLeakScore(
        label_weight=config.get('label_weight', config.get('beta', 1.0)),
        entropy_weight=config.get('entropy_weight', config.get('alpha', 1.0)),
        empirical_weight=config.get('empirical_weight', config.get('gamma', 0.0)),
    )
    # Entropy components are always computed — safe to average directly.
    entropy_avg = np.mean([
        metrics['shannon_leak_score'],
        metrics['renyi_leak_score'],
        metrics['min_entropy_leak_score'],
    ])

    # BUG FIX: only average the label components that were actually computed.
    # ConfidenceGap is always available; GLMIP and CosineSimilarity are skipped
    # when skip_glmip is set (or in --defer-attacks mode). Previously the code
    # plugged 0.0 into np.mean() for the skipped components, artificially
    # dividing the true LeakScore by 3 and hiding the threat from the adaptive
    # encryption controller.
    label_scores = [metrics['confidence_gap']]
    if not skip_glmip and 'glmip_score' in metrics:
        label_scores.append(metrics['glmip_score'])
    if not skip_glmip and 'cosine_leak_score' in metrics:
        label_scores.append(metrics['cosine_leak_score'])
    label_avg = float(np.mean(label_scores))

    # BUG FIX: same dilution issue for empirical metrics. Only average the
    # attacks that actually ran. If empirical is skipped entirely we report
    # 0.0 (genuinely not measured — the combined weight should also be 0).
    #
    # For the adaptive controller we prefer the pessimistic bound: if any
    # single attack found a strong leak, we treat the client as vulnerable
    # even if the other attacks were weaker or crashed. The mean is kept for
    # logging/analysis only.
    if not skip_empirical and 'empirical_gradinversion' in metrics:
        empirical_components = [
            metrics['empirical_gradinversion'],
            metrics['empirical_ginas'],
            metrics['empirical_ggcdm'],
        ]
        # Drop any None markers from failed attacks (see empirical_leakscore.py).
        empirical_components = [c for c in empirical_components if c is not None]
        if empirical_components:
            empirical_avg = float(np.mean(empirical_components))
            # Pessimistic bound — fed into the adaptive controller below.
            empirical_risk = float(np.max(empirical_components))
        else:
            empirical_avg = 0.0
            empirical_risk = 0.0
    else:
        empirical_avg = 0.0
        empirical_risk = 0.0

    # Feed the pessimistic empirical_risk (max across attacks) into the
    # combined leakscore that drives the adaptive controller, not the mean.
    combined = combined_metric.compute(entropy_avg, label_avg, empirical_risk)
    metrics['combined_leakscore'] = combined
    metrics['entropy_avg'] = float(entropy_avg)
    metrics['label_avg'] = float(label_avg)
    metrics['empirical_avg'] = float(empirical_avg)
    metrics['empirical_risk'] = float(empirical_risk)

    # ── Gradient Accumulation (Paper Section IV.G.2, Eq. 15) ──
    # When LeakScore >= threshold ("Strong Risk"), accumulate gradients over
    # K forward-backward passes to inflate effective batch size B_eff = K*B,
    # making gradient inversion exponentially harder.
    metrics['grad_accum_activated'] = 0
    metrics['grad_accum_effective_bs'] = batch_size
    accum_threshold = config.get('grad_accum_threshold')
    if accum_threshold is None:
        accum_threshold = config.get('T2', 0.7)
    if (config.get('grad_accum_enabled', True)
            and combined >= accum_threshold):
        K = config.get('grad_accum_K', 4)
        t0_accum = time.perf_counter()
        model_state = result.get('_local_model_state')
        if model_state is not None:
            model_state_gpu = {k: v.to(gpu_device) for k, v in model_state.items()}
            orig_dev = client.device
            client.device = gpu_device
            accum_result = client.compute_accumulated_gradient(
                model_state_gpu, K, batch_size=batch_size,
            )
            client.device = orig_dev
            if accum_result is not None:
                # Replace the raw gradient with the accumulated version
                gd = {k: v.to(gpu_device) for k, v in accum_result['gradient_dict'].items()}
                flat = accum_result['flat_gradient'].to(gpu_device)
                metrics['grad_accum_activated'] = 1
                metrics['grad_accum_effective_bs'] = accum_result['effective_batch_size']
                metrics['grad_accum_passes'] = accum_result['accumulation_passes']
        timing['grad_accum'] = time.perf_counter() - t0_accum

    metrics['loss'] = result['loss']
    t_total = time.perf_counter() - t_start
    metrics['client_time'] = t_total

    # ── Print completion ──
    with _progress_lock:
        _progress_count += 1
        done = _progress_count
        total = _progress_total

    accum_str = f" ACCUM(K={config.get('grad_accum_K', 4)},B_eff={metrics['grad_accum_effective_bs']})" if metrics['grad_accum_activated'] else ""
    parts = [f"train={t_train:.0f}s"]
    for phase in ['entropy', 'glmip', 'empirical', 'fisher', 'grad_accum']:
        t = timing.get(phase, 0)
        if t > 0.1:
            parts.append(f"{phase}={t:.0f}s")
    time_str = " ".join(parts)

    gpu_short = str(gpu_device).replace('cuda:', 'GPU')
    print(f"  W{wid:<2d} [{done:3d}/{total}] C{cid:<4d}  "
          f"loss={metrics['loss']:.4f}  leak={combined:.4f}{accum_str}  "
          f"| {t_total:.0f}s ({time_str})", flush=True)

    _release_worker(gpu_device, wid)

    local_state_cpu = {k: v.cpu() for k, v in result.get('local_state_dict', {}).items()}

    # ── Save slim artifact for Phase 2 scenario replay ──
    # Drop redundant fields to fit the 10 TB project quota:
    #   flat_gradient   → recompute from gradient_dict (microseconds)
    #   fisher_result   → recompute as g^2 from gradient_dict
    #   outputs         → recompute from images + per-round global_model.pt
    #   local_state_dict→ not needed for privacy scenarios (only accuracy)
    #   local_weights   → derivable as global_state - weight_delta
    # We keep weight_delta because MaskCrypt scenarios need the
    # (old_weights, new_weights) pair and this is the compact form.
    # Per-client size: ~290 MB → ~90 MB. Total for 250 rounds × 12
    # experiments ≈ 8 TB (fits under 10 TB quota).
    if artifacts_dir is not None:
        # Strip large tensors from metrics before saving — they're derivable
        # from gradient_dict in Phase 2 and would balloon each artifact by
        # ~100 MB (measured: full metrics dict = 100.6 MB per client, of
        # which ~99 MB is fisher_per_weight + fisher_per_weight_norm +
        # encrypt_mask + param_names). Keep scalars, histograms, timings.
        _METRICS_DROP = (
            'fisher_per_weight',
            'fisher_per_weight_norm',
            'encrypt_mask',
            'param_names',
            'topk_fisher_values',
            'topk_fisher_indices',
        )
        slim_metrics = {k: v for k, v in metrics.items() if k not in _METRICS_DROP}

        artifact = {
            'client_id': cid,
            'gradient_dict': {k: v.cpu() for k, v in gd.items()},
            'weight_delta': weight_delta_cpu,
            'metrics': slim_metrics,
            'labels': result.get('labels', torch.tensor([])).cpu(),
        }
        if original_images is not None:
            artifact['images'] = original_images.cpu()

        import os
        os.makedirs(artifacts_dir, exist_ok=True)
        _robust_save(artifact, os.path.join(artifacts_dir, f'client_{cid}.pt'))

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

        # Detect available GPUs with adaptive allocation
        from ..utils.gpu import get_optimal_config
        gpu_cfg = get_optimal_config(config, verbose=True)
        self.gpu_devices = gpu_cfg['devices']
        self.num_gpus = gpu_cfg['num_gpus'] or 1  # at least 1 (CPU)
        self._auto_clients_per_gpu = gpu_cfg['clients_per_gpu']

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
                  batch_size=None, skip_glmip=False, skip_empirical=False,
                  save_dir=None, artifacts_dir=None):
        """Run one FL round with full AdaGuard pipeline.

        If save_dir is set, GLMIP and Empirical attacks are deferred —
        client gradient data is saved to save_dir/round_N/ for later processing
        via run_attacks.py. Training + fast metrics still run inline.

        Clients are processed in parallel across available GPUs.
        """
        import sys
        global _progress_count, _progress_total
        nc = num_clients or self.config['clients_per_round']
        avail = list(self.clients.keys())
        selected = random.sample(avail, min(nc, len(avail)))

        # Reset progress counter for this round
        with _progress_lock:
            _progress_count = 0
            _progress_total = len(selected)

        print(f"\n{'='*70}", flush=True)
        print(f"  ROUND {rnd+1}  |  {len(selected)} clients  |  strategy={encryption_strategy}  "
              f"|  {self.num_gpus} GPUs", flush=True)
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

        exposed_w_cpu = self.weights_previous_round or self.weights_before_round

        client_results = []
        all_gradients = []
        all_state_dicts = []  # full state dicts for BN buffer averaging

        # Pre-copy exposed_w to each GPU ONCE (not per-client)
        exposed_w_per_gpu = {}
        for gpu_dev in self.gpu_devices:
            exposed_w_per_gpu[str(gpu_dev)] = {
                k: v.to(gpu_dev) for k, v in exposed_w_cpu.items()
            }

        # Assign clients to GPUs round-robin
        # Use auto-detected value unless config explicitly sets clients_per_gpu > 0
        cpg = self.config.get('clients_per_gpu', 0)
        if cpg <= 0:
            cpg = self._auto_clients_per_gpu
        clients_per_gpu = cpg
        max_workers = min(len(selected), self.num_gpus * clients_per_gpu)

        gpu_assignments = {}
        gpu_client_groups = {str(g): [] for g in self.gpu_devices}
        for i, cid in enumerate(selected):
            gpu_dev = self.gpu_devices[i % self.num_gpus]
            gpu_assignments[cid] = gpu_dev
            gpu_client_groups[str(gpu_dev)].append(cid)

        # Set up worker ID pools: W1-W6 on cuda:0, W7-W12 on cuda:1, etc.
        with _progress_lock:
            _gpu_worker_pools.clear()
            _worker_gpu_map.clear()
            _worker_ids.clear()
            wid = 1
            for gpu_dev in self.gpu_devices:
                gpu_str = str(gpu_dev)
                _gpu_worker_pools[gpu_str] = []
                for _ in range(clients_per_gpu):
                    _gpu_worker_pools[gpu_str].append(wid)
                    _worker_gpu_map[wid] = gpu_str
                    wid += 1

        # Print worker-GPU mapping with client assignments
        print(f"  {max_workers} workers ({clients_per_gpu}/GPU × {self.num_gpus} GPUs)", flush=True)
        print(f"  {'─'*66}", flush=True)
        for gpu_dev in self.gpu_devices:
            gpu_str = str(gpu_dev)
            workers = _gpu_worker_pools[gpu_str]
            clients_on_gpu = gpu_client_groups[gpu_str]
            w_str = ", ".join(f"W{w}" for w in workers)
            c_str = ", ".join(f"C{c}" for c in clients_on_gpu)
            print(f"  {gpu_str:7s} │ Workers: {w_str}", flush=True)
            print(f"          │ Clients: {c_str}", flush=True)
        print(f"  {'─'*66}", flush=True)

        # Per-round save directory for deferred attacks (legacy)
        round_save_dir = None
        if save_dir is not None:
            round_save_dir = os.path.join(save_dir, f'round_{rnd}')
            os.makedirs(round_save_dir, exist_ok=True)
            _robust_save(global_model_state, os.path.join(round_save_dir, 'global_model.pt'))

        # Per-round artifacts directory for Phase 2 scenario replay
        round_artifacts_dir = None
        if artifacts_dir is not None:
            save_every = self.config.get('save_every_n_rounds', 1)
            total_rounds = self.config.get('num_rounds', 250)
            # Save this round if: every N rounds, or last 5 rounds
            should_save = (rnd % save_every == 0) or (rnd >= total_rounds - 5)
            if should_save:
                round_artifacts_dir = os.path.join(artifacts_dir, f'round_{rnd}')
                os.makedirs(round_artifacts_dir, exist_ok=True)
                # Save global model state before this round
                _robust_save(global_model_state, os.path.join(round_artifacts_dir, 'global_model.pt'))
                # Save selected client IDs
                import json
                with open(os.path.join(round_artifacts_dir, 'selected_clients.json'), 'w') as f:
                    json.dump(selected, f)

        if max_workers > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {}
                for cid in selected:
                    gpu_dev = gpu_assignments[cid]
                    ew_gpu = exposed_w_per_gpu[str(gpu_dev)]
                    future = executor.submit(
                        _process_client, cid, self.clients[cid],
                        global_state, batch_size, gpu_dev, self.config,
                        global_model_state, ew_gpu,
                        skip_glmip, skip_empirical, self.train_dataset,
                        round_save_dir, round_artifacts_dir,
                    )
                    futures[future] = cid

                for future in as_completed(futures):
                    cid = futures[future]
                    try:
                        result = future.result()
                    except Exception as e:
                        print(f"  [ERROR] Client {cid}: {e}", flush=True)
                        import traceback
                        traceback.print_exc()
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
            # Single GPU — run sequentially
            for cid in selected:
                gpu_dev = self.gpu_devices[0]
                ew_gpu = exposed_w_per_gpu[str(gpu_dev)]
                result = _process_client(
                    cid, self.clients[cid], global_state, batch_size,
                    gpu_dev, self.config, global_model_state, ew_gpu,
                    skip_glmip, skip_empirical, self.train_dataset,
                    round_save_dir, round_artifacts_dir,
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
        print(f"  Aggregating ...", flush=True)
        if all_state_dicts:
            avg_state = fedavg_aggregate_state_dicts(all_state_dicts)
            avg_state = {k: v.to(self.device) for k, v in avg_state.items()}
            self.global_model.load_state_dict(avg_state)
        elif all_gradients:
            avg_grads = fedavg_aggregate(all_gradients)
            avg_grads = {k: v.to(self.device) for k, v in avg_grads.items()}
            apply_gradient_update(
                self.global_model, avg_grads, self.config['fl_lr'],
            )

        # Free large per-round data to prevent OOM over many rounds
        del all_state_dicts, all_gradients, exposed_w_per_gpu
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.weights_after_round = {
            n: p.clone().detach()
            for n, p in self.global_model.named_parameters()
        }

        # --- Evaluate ---
        accuracy, test_loss = self._evaluate()
        print(f"  Global model — Test Acc: {accuracy*100:.1f}%  Test Loss: {test_loss:.4f}", flush=True)
        sys.stdout.flush()

        # --- Aggregate round metrics ---
        round_summary = self._summarize_round(rnd, client_results, accuracy)
        round_summary['test_loss'] = test_loss
        # Store lightweight copy in history (client_details can be huge over 50 rounds)
        history_copy = {k: v for k, v in round_summary.items() if k != 'client_details'}
        self.round_history.append(history_copy)

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
            # Compute MaskCrypt on-demand (baseline comparison only, not part of leakscore)
            ew_cpu = {k: v.cpu() for k, v in exposed_w_gpu.items()}
            lw_cpu = {k: v.cpu() for k, v in local_weights.items()}
            mc_result = self.maskcrypt_metric.compute(gd_cpu, ew_cpu, lw_cpu)
            protected_gd, enc_meta = self.maskcrypt_encryptor.encrypt(
                gd_cpu, ew_cpu, lw_cpu, k=k, maskcrypt_result=mc_result,
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
        """Evaluate global model on test set. Returns (accuracy, test_loss)."""
        self.global_model.eval()
        correct, total = 0, 0
        total_loss = 0.0
        n_batches = 0
        loader = DataLoader(self.test_dataset, batch_size=128, shuffle=False)

        with torch.no_grad():
            for imgs, lbls in loader:
                imgs, lbls = imgs.to(self.device), lbls.to(self.device)
                out = self.global_model(imgs)
                total_loss += self.criterion(out, lbls).item()
                n_batches += 1
                preds = out.argmax(1)
                correct += preds.eq(lbls).sum().item()
                total += lbls.size(0)

        self.global_model.train()
        acc = correct / max(total, 1)
        test_loss = total_loss / max(n_batches, 1)
        return acc, test_loss

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
