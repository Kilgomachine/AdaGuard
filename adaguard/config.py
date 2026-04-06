"""Central configuration for AdaGuard simulator."""

import copy
import random
import numpy as np
import torch
import yaml
from pathlib import Path


DEFAULT_CONFIG = {
    # Model
    'model': 'smallcnn',  # smallcnn, resnet18, resnet34, resnet50
    'num_classes': 10,
    'image_size': 32,
    'pretrain_epochs': 0,  # 0 to skip pretrain entirely
    'pretrain_lr': 0.01,
    'pretrain_batch_size': 64,

    # Federated Learning
    'num_clients': 10,
    'clients_per_round': 3,
    'num_rounds': 5,
    'client_batch_size': 4,
    'client_local_steps': 1,
    'client_lr': 0.01,   # local SGD learning rate
    'fl_lr': 1.0,        # server-side lr (1.0 = standard FedAvg)

    # LeakScore weights
    # label_weight: GLMIP + ConfidenceGap + CosineSimilarity
    # entropy_weight: Shannon + Renyi + MinEntropy
    # empirical_weight: lightweight GI attacks (Phase 2 only, 0 during training)
    'label_weight': 1.0,
    'entropy_weight': 1.0,
    'empirical_weight': 0.0,  # 0 during training; Phase 2 scenarios override this
    # Backward compat aliases (used by app.py)
    'alpha': 1.0,
    'beta': 1.0,
    'gamma': 0.0,

    # Encryption thresholds
    'T1': 0.3,
    'T2': 0.7,

    # Gradient Accumulation (Section IV.G.2)
    # Activated when LeakScore >= grad_accum_threshold (typically T2 = "Strong Risk")
    # B_eff = K * B where K = grad_accum_K
    'grad_accum_enabled': True,
    'grad_accum_K': 4,                # number of accumulation passes
    'grad_accum_threshold': 0.7,      # LeakScore threshold to trigger (default = T2)

    # Entropy metric
    'entropy_bins': 50,

    # Empirical attack settings (lightweight, for leakscore)
    'empirical_iterations': 20,
    'empirical_lr': 0.1,

    # Full paper-matched attack settings (Phase 2)
    'attack_gi_iters': 20000,       # GradInversion (Yin et al. 2021)
    'attack_gi_lr': 0.1,
    'attack_gi_tv_lambda': 1e-4,
    'attack_gi_l2_lambda': 1e-6,
    'attack_gi_bn_lambda': 0.1,
    'attack_ginas_candidates': 5000, # GI-NAS (Yu et al. 2025)
    'attack_ginas_lr': 1e-3,
    'attack_ggcdm_guidance_rate': 0.20, # GGCDM (Meng et al. 2025)
    'attack_ggcdm_diffusion_steps': 1000,

    # Full GI attack (validation/legacy)
    'gi_iters': 300,
    'gi_lr': 0.1,

    # Artifact saving (Phase 1 → Phase 2 bridge)
    'save_artifacts': True,       # save per-client artifacts for Phase 2
    'save_every_n_rounds': 1,     # save artifacts every N rounds (1 = all)
    'save_attack_clients': 100,   # how many clients per round to save (100 = all)

    # Label metric
    'mi_samples_per_class': 20,

    # Fisher / MaskCrypt
    'fisher_topk': 50,
    'encryption_top_percent': 0.1,
    'maskcrypt_mask_mode': 'gradient_guided',  # 'gradient_guided' or 'random'

    # Differential Privacy baseline
    'dp_epsilon': 50.0,
    'dp_delta': 1e-5,
    'dp_clip_norm': 1.0,

    # Focus layers (where label info concentrates per iDLG theorem)
    'focus_layers': ['fc2.weight', 'fc2.bias'],

    # Parallelism
    'clients_per_gpu': 3,  # max concurrent clients per GPU (0 = auto)

    # Experiment sweeps
    'batch_sizes_experiment': [1, 4, 8, 16, 32, 64],
    'noise_levels': [0, 0.001, 0.01, 0.05, 0.1],
    'client_counts_experiment': [1, 2, 5, 10],

    # Seed
    'seed': 42,
}


def load_config(path=None):
    """Load config from YAML file, falling back to defaults."""
    config = copy.deepcopy(DEFAULT_CONFIG)
    if path is not None:
        p = Path(path)
        if p.exists():
            with open(p) as f:
                overrides = yaml.safe_load(f)
            if overrides:
                config.update(overrides)
    return config


def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Get best available device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
