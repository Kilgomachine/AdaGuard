"""Gradient Magnitude metric — overall gradient signal strength.

Computes the L2 norm of the full gradient vector and per-layer norms.
Higher magnitude gradients may indicate more informative (and leakable) updates.
"""

import torch


class GradientMagnitudeMetric:
    """Gradient magnitude as a leakage risk indicator.

    magnitude_score = l2 / (l2 + 1.0)  — smooth [0, inf) -> [0, 1)
    """

    def compute(self, gradient_dict):
        if not gradient_dict:
            return self._empty()

        per_layer = {}
        all_norms_sq = 0.0

        for name, grad in gradient_dict.items():
            layer_norm = torch.norm(grad.float(), 2).item()
            per_layer[name] = layer_norm
            all_norms_sq += layer_norm ** 2

        total_l2 = all_norms_sq ** 0.5
        # Smooth normalization to [0, 1)
        score = min(1.0, total_l2 / (total_l2 + 1.0)) if total_l2 > 0 else 0.0

        return {
            'magnitude_score': score,
            'magnitude_raw_l2': total_l2,
            'magnitude_per_layer': per_layer,
        }

    def _empty(self):
        return {
            'magnitude_score': 0.0,
            'magnitude_raw_l2': 0.0,
            'magnitude_per_layer': {},
        }
