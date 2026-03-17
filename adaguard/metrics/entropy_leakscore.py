"""Entropy-based LeakScore: Shannon, Renyi (alpha=2), Min-entropy.

Pipeline per gradient vector:
1. Flatten → L2 normalize
2. Standardize (z-score)
3. Histogram binning → probability distribution
4. Compute entropy variants
5. Normalize to [0,1] and invert (low entropy = high leakage risk)
"""

import numpy as np
import torch


class EntropyLeakScoreMetric:
    """Computes entropy-based gradient leakage scores.

    Supports dual-mode: focus-layer (primary, more sensitive) and
    full-gradient (reference).
    """

    def __init__(self, num_bins=50):
        self.num_bins = num_bins

    def _compute_one(self, g):
        """Compute Shannon, Renyi, Min-entropy for a single gradient vector."""
        if len(g) == 0:
            return {'shannon': 0.5, 'renyi': 0.5, 'min_ent': 0.5}

        mu, sigma = np.mean(g), np.std(g)
        if sigma < 1e-12:
            return {'shannon': 0.5, 'renyi': 0.5, 'min_ent': 0.5}

        # Standardize
        g_std = (g - mu) / sigma

        # Histogram → probabilities
        counts, _ = np.histogram(g_std, bins=self.num_bins)
        total = counts.sum()
        if total < 1:
            return {'shannon': 0.5, 'renyi': 0.5, 'min_ent': 0.5}

        p = counts.astype(np.float64) / total
        pnz = p[p > 0]
        lB = np.log(self.num_bins)

        # Shannon: H = -sum(p * log(p))
        H_sh = -np.sum(pnz * np.log(pnz))

        # Renyi (alpha=2): H_alpha = -log(sum(p^alpha))
        H_re = -np.log(np.sum(pnz ** 2))

        # Min-entropy: H_inf = -log(max(p))
        H_min = -np.log(np.max(pnz))

        # Normalize by log(B) and invert: 0=safe, 1=risky
        return {
            'shannon': max(0.0, min(1.0, 1.0 - H_sh / lB)),
            'renyi': max(0.0, min(1.0, 1.0 - H_re / lB)),
            'min_ent': max(0.0, min(1.0, 1.0 - H_min / lB)),
            'raw_sh': H_sh,
            'raw_re': H_re,
            'raw_min': H_min,
            'mean': float(mu),
            'std': float(sigma),
        }

    def compute(self, flat_gradient, gradient_dict=None, focus_layers=None):
        """Compute entropy leakage scores.

        Args:
            flat_gradient: full flattened gradient vector (tensor or ndarray)
            gradient_dict: per-layer gradient dict (optional, for focus-layer)
            focus_layers: list of layer names to focus on (optional)

        Returns:
            dict with shannon/renyi/min_entropy scores (focus + full)
        """
        # Full gradient entropy
        if isinstance(flat_gradient, torch.Tensor):
            g = flat_gradient.detach().cpu().numpy().astype(np.float64)
        else:
            g = np.array(flat_gradient, dtype=np.float64)
        full = self._compute_one(g)

        # Focus-layer entropy (more sensitive to batch-size effects)
        if gradient_dict and focus_layers:
            parts = []
            for name in focus_layers:
                if name in gradient_dict:
                    parts.append(
                        gradient_dict[name].detach().cpu().numpy().flatten().astype(np.float64)
                    )
            if parts:
                g_focus = np.concatenate(parts)
                focus = self._compute_one(g_focus)
            else:
                focus = full
        else:
            focus = full

        return {
            # Primary scores use focus-layer (more sensitive)
            'shannon_leak_score': focus['shannon'],
            'renyi_leak_score': focus['renyi'],
            'min_entropy_leak_score': focus['min_ent'],
            # Full-gradient scores for reference
            'shannon_full': full['shannon'],
            'renyi_full': full['renyi'],
            'min_entropy_full': full['min_ent'],
            'gradient_mean': full.get('mean', 0.0),
            'gradient_std': full.get('std', 0.0),
        }
