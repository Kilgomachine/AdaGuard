"""Fisher-based parameter-level encryption.

Encrypts top-k parameters ranked by Fisher Information (F_i = g_i^2).
High Fisher → parameter tightly coupled to specific data → encrypt it.

Encryption is simulated by zeroing out the encrypted gradient components
(representing that an adversary cannot observe them).
"""

import torch

from ..metrics.fisher import FisherInformationMetric


class FisherEncryptor:
    """Encrypts gradient parameters based on Fisher Information ranking.

    In a real system, 'encryption' would use homomorphic encryption
    or secure aggregation. Here we simulate by masking (zeroing)
    the encrypted parameters from the adversary's view.
    """

    def __init__(self, fisher_metric=None, topk=50):
        self.fisher_metric = fisher_metric or FisherInformationMetric(topk=topk)

    def encrypt(self, gradient_dict, k=None, fisher_result=None):
        """Encrypt top-k parameters by Fisher score.

        Args:
            gradient_dict: per-layer gradient dict
            k: number of parameters to encrypt (overrides metric default)
            fisher_result: pre-computed Fisher result (optional, avoids recompute)

        Returns:
            protected_dict: gradient dict with encrypted params zeroed
            metadata: encryption details (mask, count, etc.)
        """
        if fisher_result is None:
            if k is not None:
                total_params = sum(g.numel() for g in gradient_dict.values())
                enc_pct = k / max(total_params, 1)
                fisher_result = self.fisher_metric.compute_with_dynamic_k(
                    gradient_dict, enc_pct,
                )
            else:
                fisher_result = self.fisher_metric.compute(gradient_dict)

        encrypt_mask = fisher_result['encrypt_mask']

        # Apply encryption: zero out encrypted parameters (simulate HE)
        protected_dict = {}
        idx = 0
        for name, grad in gradient_dict.items():
            n = grad.numel()
            layer_mask = encrypt_mask[idx:idx + n].view(grad.shape).to(grad.device)
            # Encrypted params become zero (adversary can't see them)
            protected_dict[name] = grad * (~layer_mask).float()
            idx += n

        metadata = {
            'strategy': 'fisher',
            'weights_encrypted': fisher_result['weights_to_encrypt'],
            'pct_encrypted': fisher_result['pct_encrypted'],
            'encryption_threshold': fisher_result['encryption_threshold'],
            'encrypt_mask': encrypt_mask,
            # Classifier-head guarantee diagnostics (fisher.py module
            # docstring explains the failure mode this fixes).
            'classifier_head_forced_count': fisher_result.get(
                'classifier_head_forced_count', 0),
            'classifier_head_forced_layers': fisher_result.get(
                'classifier_head_forced_layers', []),
        }

        return protected_dict, metadata

    def compute_overhead(self, metadata):
        """Estimate communication overhead from encryption.

        Returns fraction of additional communication cost.
        In practice, HE ciphertext is ~50-100x larger than plaintext.
        """
        pct = metadata['pct_encrypted']
        # HE expansion factor (typical for BFV/CKKS schemes)
        he_expansion = 50.0
        return 1.0 + pct * (he_expansion - 1.0)
