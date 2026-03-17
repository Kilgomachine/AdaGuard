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
    'fl_lr': 0.01,

    # LeakScore weights (alpha, beta, gamma)
    'alpha': 1.0,
    'beta': 1.0,
    'gamma': 1.0,

    # Encryption thresholds
    'T1': 0.3,
    'T2': 0.7,

    # Entropy metric
    'entropy_bins': 50,

    # Empirical attack settings
    'empirical_iterations': 20,
    'empirical_lr': 0.1,

    # Full GI attack (validation)
    'gi_iters': 300,
    'gi_lr': 0.1,

    # Label metric
    'mi_samples_per_class': 20,

    # Fisher / MaskCrypt
    'fisher_topk': 50,
    'encryption_top_percent': 0.1,

    # Focus layers (where label info concentrates per iDLG theorem)
    'focus_layers': ['fc2.weight', 'fc2.bias'],

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
