"""Differential Privacy baseline — Gaussian mechanism.

Clips per-client gradients to max L2 norm S, then adds calibrated
Gaussian noise N(0, sigma^2) where sigma = sqrt(2 * ln(1.25/delta)) * S / epsilon.

Matches MaskCrypt paper's DP baseline (epsilon=50, delta=1e-5).
"""

import math
import torch


class DPNoiseEncryptor:
    """Apply differential privacy noise to gradient updates."""

    def __init__(self, epsilon=50.0, delta=1e-5, clip_norm=1.0):
        self.epsilon = epsilon
        self.delta = delta
        self.clip_norm = clip_norm
        self.sigma = self._calibrate_sigma()

    def _calibrate_sigma(self):
        """Calibrate Gaussian noise scale from (epsilon, delta) and clip norm."""
        return math.sqrt(2.0 * math.log(1.25 / self.delta)) * self.clip_norm / self.epsilon

    def encrypt(self, gradient_dict):
        """Clip gradients and add calibrated Gaussian noise.

        Args:
            gradient_dict: per-layer gradient dict

        Returns:
            noisy_dict: gradient dict with DP noise applied
            metadata: noise parameters and stats
        """
        # Clip: compute global L2 norm across all layers
        flat = torch.cat([g.flatten() for g in gradient_dict.values()])
        global_norm = torch.norm(flat, 2).item()
        clip_factor = min(1.0, self.clip_norm / max(global_norm, 1e-12))

        noisy_dict = {}
        total_noise_norm = 0.0
        for name, grad in gradient_dict.items():
            clipped = grad * clip_factor
            noise = torch.randn_like(clipped) * self.sigma
            noisy_dict[name] = clipped + noise
            total_noise_norm += torch.norm(noise, 2).item() ** 2

        total_noise_norm = math.sqrt(total_noise_norm)

        metadata = {
            'strategy': 'dp',
            'epsilon': self.epsilon,
            'delta': self.delta,
            'clip_norm': self.clip_norm,
            'sigma': self.sigma,
            'original_grad_norm': global_norm,
            'clip_factor': clip_factor,
            'noise_norm': total_noise_norm,
            'pct_encrypted': 0.0,  # DP doesn't "encrypt" parameters
        }

        return noisy_dict, metadata
