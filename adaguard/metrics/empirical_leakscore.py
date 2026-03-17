"""Empirical Minimised LeakScore — wraps GI attacks for leakage scoring.

Runs lightweight (20-iteration) versions of three GI attacks:
- GradInversion
- GI-NAS
- GGCDM

For each, score = |1 - ||g_real - g_reconstructed||_2 / ||g_real||_2|
EmpiricalLeakScore = mean of all attack scores (normalized to [0,1]).
Also computes image-space reconstruction quality (MSE, PSNR, SSIM) when
original images are provided.
"""

import numpy as np

from ..attacks import GradInversionAttack, GINASAttack, GGCDMAttack


class EmpiricalLeakScoreMetric:
    """Wrapper that runs lightweight GI attacks and averages their scores."""

    def __init__(self, model, criterion, device, n_iter=20, lr=0.1,
                 focus_layers=None):
        self.gi_attack = GradInversionAttack(
            model, criterion, device, n_iter=n_iter, lr=lr,
            focus_layers=focus_layers,
        )
        self.ginas_attack = GINASAttack(
            model, criterion, device, n_iter=n_iter, lr=lr,
            focus_layers=focus_layers,
        )
        self.ggcdm_attack = GGCDMAttack(
            model, criterion, device, n_iter=n_iter, lr=lr,
            focus_layers=focus_layers,
        )

    def compute(self, gradient_dict, flat_gradient, batch_size=1, labels=None,
                original_images=None):
        """Run all three attacks and compute empirical leakage scores.

        Args:
            original_images: if provided, compute MSE/PSNR/SSIM for reconstruction quality

        Returns:
            dict with individual attack scores + mean empirical score + reconstruction metrics
        """
        results = {}
        scores = []
        recon_mses = []
        recon_psnrs = []
        recon_ssims = []

        for attack, key in [
            (self.gi_attack, 'empirical_gradinversion'),
            (self.ginas_attack, 'empirical_ginas'),
            (self.ggcdm_attack, 'empirical_ggcdm'),
        ]:
            try:
                r = attack.attack(gradient_dict, flat_gradient, batch_size, labels,
                                  original_images=original_images)
                results.update(r)
                scores.append(r[key])

                # Collect reconstruction metrics if available
                prefix = 'gi_' + key.replace('empirical_', '')
                if f'{prefix}_mse' in r:
                    recon_mses.append(r[f'{prefix}_mse'])
                    recon_psnrs.append(r[f'{prefix}_psnr'])
                    recon_ssims.append(r[f'{prefix}_ssim'])
            except Exception as e:
                print(f"  [WARN] {key}: {e}")
                results[key] = 0.0
                scores.append(0.0)

        results['empirical_mean'] = float(np.mean(scores))

        # Aggregate reconstruction quality across attacks
        if recon_mses:
            results['recon_mse'] = float(np.mean(recon_mses))
            results['recon_psnr'] = float(np.mean(recon_psnrs))
            results['recon_ssim'] = float(np.mean(recon_ssims))
        else:
            results['recon_mse'] = 0.0
            results['recon_psnr'] = 0.0
            results['recon_ssim'] = 0.0

        return results
