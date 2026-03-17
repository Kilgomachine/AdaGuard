"""Experiment orchestrator for AdaGuard research simulations."""

import copy

import numpy as np
import torch

from ..config import load_config, set_seed, get_device
from ..models.cnn import SmallCNN
from ..data.cifar10 import load_cifar10, partition_data_non_iid
from ..federation.simulator import FederatedSimulator
from ..logging_utils.logger import RoundLogger
from ..visualization.plots import AdaGuardPlotter


class ExperimentRunner:
    """Runs predefined experiments and generates results.

    Supports:
    - FL per-round evaluation with all metrics
    - Batch size ablation
    - Noise level ablation
    - Client count ablation
    - Alpha/Beta/Gamma weight study
    - Encryption strategy comparison
    """

    def __init__(self, config=None, config_path=None):
        self.config = config or load_config(config_path)
        set_seed(self.config['seed'])
        self.device = get_device()
        self.logger = RoundLogger()
        self.plotter = AdaGuardPlotter()

        # Load data
        self.train_dataset, self.test_dataset = load_cifar10()
        self.client_data_map = partition_data_non_iid(
            self.train_dataset, self.config['num_clients'],
        )

        # Create model
        self.model = SmallCNN(
            num_classes=self.config['num_classes'],
        ).to(self.device)

        print(f"Model: {sum(p.numel() for p in self.model.parameters()):,} params")
        print(f"Device: {self.device}")

    def _create_simulator(self, model=None):
        """Create a fresh simulator (optionally with a specific model)."""
        m = model or self.model
        return FederatedSimulator(
            m, self.train_dataset, self.test_dataset,
            self.client_data_map, self.config, self.device,
        )

    def run_fl_rounds(self, encryption_strategy='fisher',
                      skip_glmip=False, skip_empirical=False):
        """Run full FL simulation with per-round metrics.

        Args:
            encryption_strategy: 'fisher', 'maskcrypt', 'full', or 'none'
            skip_glmip: skip expensive GLMIP computation
            skip_empirical: skip empirical attack scoring

        Returns:
            list of round summary dicts
        """
        sim = self._create_simulator()
        print("Pre-training model...")
        sim.pretrain()

        self.logger.reset()
        results = []

        for rnd in range(self.config['num_rounds']):
            print(f"\n{'='*60}")
            print(f"  FL Round {rnd+1}/{self.config['num_rounds']}")
            print(f"{'='*60}")

            summary = sim.run_round(
                rnd,
                encryption_strategy=encryption_strategy,
                skip_glmip=skip_glmip,
                skip_empirical=skip_empirical,
            )
            self.logger.log_round(summary)
            results.append(summary)

            # Print summary
            print(f"  Accuracy: {summary.get('accuracy', 0)*100:.1f}%")
            print(f"  Combined LeakScore: {summary.get('combined_leakscore', 0):.4f}")
            print(f"  Encryption: {summary.get('encryption_level', 'n/a')} "
                  f"({summary.get('actual_pct_encrypted', 0)*100:.1f}%)")

        self.logger.save(f'fl_rounds_{encryption_strategy}')
        return results

    def run_encryption_comparison(self):
        """Run FL with all 4 encryption strategies for comparison.

        Returns:
            dict mapping strategy name → list of round results
        """
        strategies = ['none', 'fisher', 'maskcrypt', 'full']
        all_results = {}

        for strategy in strategies:
            print(f"\n{'#'*60}")
            print(f"  Strategy: {strategy.upper()}")
            print(f"{'#'*60}")

            # Fresh model for each strategy
            model_copy = SmallCNN(
                num_classes=self.config['num_classes'],
            ).to(self.device)
            model_copy.load_state_dict(self.model.state_dict())

            sim = FederatedSimulator(
                model_copy, self.train_dataset, self.test_dataset,
                self.client_data_map, self.config, self.device,
            )
            sim.pretrain()

            self.logger.reset()
            results = []
            for rnd in range(self.config['num_rounds']):
                summary = sim.run_round(rnd, encryption_strategy=strategy)
                self.logger.log_round(summary)
                results.append(summary)

            self.logger.save(f'comparison_{strategy}')
            all_results[strategy] = results

        return all_results

    def run_weight_study(self, weight_configs=None):
        """Study effect of alpha, beta, gamma weights on encryption decisions.

        Args:
            weight_configs: list of (alpha, beta, gamma) tuples

        Returns:
            dict mapping weight tuple → list of round results
        """
        if weight_configs is None:
            weight_configs = [
                (1, 0, 0), (0, 1, 0), (0, 0, 1),
                (1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2),
            ]

        all_results = {}

        for alpha, beta, gamma in weight_configs:
            print(f"\n  Weights: alpha={alpha}, beta={beta}, gamma={gamma}")

            config_copy = copy.deepcopy(self.config)
            config_copy['alpha'] = alpha
            config_copy['beta'] = beta
            config_copy['gamma'] = gamma

            model_copy = SmallCNN(
                num_classes=self.config['num_classes'],
            ).to(self.device)
            model_copy.load_state_dict(self.model.state_dict())

            sim = FederatedSimulator(
                model_copy, self.train_dataset, self.test_dataset,
                self.client_data_map, config_copy, self.device,
            )
            sim.pretrain()

            self.logger.reset()
            results = []
            for rnd in range(config_copy['num_rounds']):
                summary = sim.run_round(rnd, encryption_strategy='fisher')
                self.logger.log_round(summary)
                results.append(summary)

            key = f"a{alpha}_b{beta}_g{gamma}"
            self.logger.save(f'weight_study_{key}')
            all_results[(alpha, beta, gamma)] = results

        return all_results

    def run_batch_size_experiment(self):
        """Study effect of batch size on leakage scores.

        Returns:
            dict mapping batch_size → metrics dict
        """
        from ..utils.gradients import extract_gradients
        from ..metrics import (
            EntropyLeakScoreMetric, FisherInformationMetric, MaskCryptMetric,
        )
        from ..metrics.empirical_leakscore import EmpiricalLeakScoreMetric

        criterion = torch.nn.CrossEntropyLoss()
        entropy_m = EntropyLeakScoreMetric(num_bins=self.config['entropy_bins'])
        fisher_m = FisherInformationMetric(
            topk=self.config['fisher_topk'],
            enc_pct=self.config['encryption_top_percent'],
        )
        maskcrypt_m = MaskCryptMetric(enc_pct=self.config['encryption_top_percent'])
        empirical_m = EmpiricalLeakScoreMetric(
            self.model, criterion, self.device,
            n_iter=self.config['empirical_iterations'],
            lr=self.config['empirical_lr'],
            focus_layers=self.config['focus_layers'],
        )

        results = {}
        for bs in self.config['batch_sizes_experiment']:
            print(f"\n  Batch size: {bs}")

            # Get a batch of data
            from torch.utils.data import DataLoader
            loader = DataLoader(self.train_dataset, batch_size=bs, shuffle=True)
            imgs, lbls = next(iter(loader))

            gd, flat, loss_val, outputs = extract_gradients(
                self.model, imgs, lbls, criterion, self.device,
            )

            metrics = {}
            metrics.update(entropy_m.compute(
                flat, gradient_dict=gd,
                focus_layers=self.config['focus_layers'],
            ))
            fi = fisher_m.compute(gd)
            metrics.update(fi)

            old_w, new_w = MaskCryptMetric.simulate_weight_delta(
                self.model, gd, self.config['fl_lr'],
            )
            mc = maskcrypt_m.compute(gd, old_w, new_w)
            metrics.update(mc)

            emp = empirical_m.compute(gd, flat, batch_size=bs)
            metrics.update(emp)

            metrics['loss'] = loss_val
            results[bs] = metrics
            print(f"    Shannon: {metrics['shannon_leak_score']:.4f}, "
                  f"Fisher conc: {metrics['fisher_concentration']:.4f}")

        return results

    def run_noise_experiment(self):
        """Study effect of noise levels on leakage scores.

        Returns:
            dict mapping noise_sigma → metrics dict
        """
        from ..utils.gradients import extract_gradients, add_noise_to_gradients
        from ..metrics import EntropyLeakScoreMetric, FisherInformationMetric, MaskCryptMetric
        from ..metrics.empirical_leakscore import EmpiricalLeakScoreMetric

        criterion = torch.nn.CrossEntropyLoss()
        entropy_m = EntropyLeakScoreMetric(num_bins=self.config['entropy_bins'])
        fisher_m = FisherInformationMetric(
            topk=self.config['fisher_topk'],
            enc_pct=self.config['encryption_top_percent'],
        )
        maskcrypt_m = MaskCryptMetric(enc_pct=self.config['encryption_top_percent'])
        empirical_m = EmpiricalLeakScoreMetric(
            self.model, criterion, self.device,
            n_iter=self.config['empirical_iterations'],
            lr=self.config['empirical_lr'],
            focus_layers=self.config['focus_layers'],
        )

        # Get base gradients
        from torch.utils.data import DataLoader
        loader = DataLoader(self.train_dataset, batch_size=4, shuffle=True)
        imgs, lbls = next(iter(loader))
        base_gd, base_flat, _, outputs = extract_gradients(
            self.model, imgs, lbls, criterion, self.device,
        )

        results = {}
        for sigma in self.config['noise_levels']:
            print(f"\n  Noise sigma: {sigma}")

            if sigma == 0:
                gd, flat = base_gd, base_flat
            else:
                gd, flat = add_noise_to_gradients(base_gd, sigma)

            metrics = {}
            metrics.update(entropy_m.compute(
                flat, gradient_dict=gd,
                focus_layers=self.config['focus_layers'],
            ))
            fi = fisher_m.compute(gd)
            metrics.update(fi)

            old_w, new_w = MaskCryptMetric.simulate_weight_delta(
                self.model, gd, self.config['fl_lr'],
            )
            mc = maskcrypt_m.compute(gd, old_w, new_w)
            metrics.update(mc)

            emp = empirical_m.compute(gd, flat)
            metrics.update(emp)

            results[sigma] = metrics

        return results
