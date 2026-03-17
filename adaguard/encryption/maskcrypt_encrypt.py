"""MaskCrypt-based parameter-level encryption.

Encrypts top-k parameters ranked by MaskCrypt vulnerability score:
v[i] = gradient[i] * (old_exposed_weight[i] - new_trained_weight[i])

Parameters with high |v[i]| are most vulnerable to gradient inversion.
"""

import torch

from ..metrics.maskcrypt import MaskCryptMetric


class MaskCryptEncryptor:
    """Encrypts gradient parameters based on MaskCrypt vulnerability ranking.

    Simulates selective homomorphic encryption by masking the most
    vulnerable parameters from the adversary's view.
    """

    def __init__(self, maskcrypt_metric=None, enc_pct=0.1):
        self.maskcrypt_metric = maskcrypt_metric or MaskCryptMetric(enc_pct=enc_pct)

    def encrypt(self, gradient_dict, old_weights, new_weights, k=None,
                maskcrypt_result=None):
        """Encrypt top-k parameters by MaskCrypt vulnerability.

        Args:
            gradient_dict: per-layer gradient dict
            old_weights: weights before local training
            new_weights: weights after local training
            k: number of parameters to encrypt (overrides metric default)
            maskcrypt_result: pre-computed result (optional)

        Returns:
            protected_dict: gradient dict with encrypted params zeroed
            metadata: encryption details
        """
        if maskcrypt_result is None:
            if k is not None:
                total_params = sum(g.numel() for g in gradient_dict.values())
                enc_pct = k / max(total_params, 1)
                maskcrypt_result = self.maskcrypt_metric.compute_with_dynamic_k(
                    gradient_dict, old_weights, new_weights, enc_pct,
                )
            else:
                maskcrypt_result = self.maskcrypt_metric.compute(
                    gradient_dict, old_weights, new_weights,
                )

        encrypt_mask = maskcrypt_result['maskcrypt_encrypt_mask']

        # Apply encryption: zero out encrypted parameters
        protected_dict = {}
        idx = 0
        for name, grad in gradient_dict.items():
            n = grad.numel()
            if idx + n <= len(encrypt_mask):
                layer_mask = encrypt_mask[idx:idx + n].view(grad.shape).to(grad.device)
                protected_dict[name] = grad * (~layer_mask).float()
            else:
                protected_dict[name] = grad.clone()
            idx += n

        metadata = {
            'strategy': 'maskcrypt',
            'weights_encrypted': maskcrypt_result['maskcrypt_weights_to_protect'],
            'pct_encrypted': (
                maskcrypt_result['maskcrypt_weights_to_protect']
                / max(sum(g.numel() for g in gradient_dict.values()), 1)
            ),
            'maskcrypt_threshold': maskcrypt_result['maskcrypt_threshold'],
            'maskcrypt_vuln_score': maskcrypt_result['maskcrypt_vuln_score'],
            'encrypt_mask': encrypt_mask,
        }

        return protected_dict, metadata

    def compute_overhead(self, metadata):
        """Estimate communication overhead."""
        pct = metadata['pct_encrypted']
        he_expansion = 50.0
        return 1.0 + pct * (he_expansion - 1.0)
