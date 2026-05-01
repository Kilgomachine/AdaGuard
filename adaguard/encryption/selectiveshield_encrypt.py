"""SelectiveShield-style hybrid encryption: Fisher-targeted SHE + DP noise.

Implements the encryption side of SelectiveShield (Li et al. 2025,
arXiv:2508.04265). For each client gradient:

  1. Rank parameters by per-weight Fisher score F_i = g_i^2.
  2. ZERO out (simulate HE) the top-K by Fisher (the "encrypted set").
  3. Add calibrated Gaussian DP noise to the REMAINING (non-encrypted)
     parameters.

This is the in-family comparison flagged in pre-submission review.
The contrast with the existing defences is:

  - AdaGuard V4 (Fisher-targeted, ours): step (1) + (2) only; no DP
    on the non-encrypted set.
  - DP baseline V3 (we don't run it in headline): step (3) on the
    full gradient, no targeted encryption.
  - SelectiveShield V5 (this class): all three steps, so it sits
    between V3 and V4 on the privacy/utility tradeoff.

Single-client simplification
----------------------------
The published SelectiveShield protocol uses a mask-consensus step
(majority vote across clients) before encryption. Our replay harness
operates on one client at a time (the round-249 artefact), so we
implement a single-client variant that uses each client's local
Fisher ranking directly. This omits the consensus communication
overhead that SelectiveShield reports but is faithful to the
encryption + noise composition.

Classifier-head guarantee
-------------------------
The default ``FisherInformationMetric`` enables the classifier-head
guarantee, so it fires here too. This keeps the AdaGuard-vs-
SelectiveShield comparison clean: the only mechanistic difference
between V4 and V5 is the DP noise on the non-encrypted slice. To
disable the head guarantee for an ablation, pass a metric with
``mandatory_layer_substrings=()``.
"""

import torch

from ..metrics.fisher import FisherInformationMetric
from .dp_noise import DPNoiseEncryptor


class SelectiveShieldEncryptor:
    """Fisher-targeted SHE + DP noise on the non-encrypted set.

    Parameters
    ----------
    fisher_metric : FisherInformationMetric | None
        Use ``None`` to construct a default (which enables the
        classifier-head guarantee). Sharing the metric class with
        :class:`adaguard.encryption.fisher_encrypt.FisherEncryptor`
        means the only mechanistic difference between V4 (Fisher) and
        V5 (SelectiveShield) is the DP noise on the non-encrypted
        slice.
    enc_pct : float
        Fraction of parameters to encrypt (top-K by Fisher).
    dp_epsilon : float
        DP epsilon for the noise on the non-encrypted set.
    dp_delta : float
        DP delta.
    dp_clip_norm : float
        L2 clipping norm for the non-encrypted gradient slice.
    """

    def __init__(self, fisher_metric=None, enc_pct=0.10,
                 dp_epsilon=50.0, dp_delta=1e-5, dp_clip_norm=1.0):
        self.fisher_metric = fisher_metric or FisherInformationMetric(
            enc_pct=enc_pct,
        )
        self.dp_epsilon = dp_epsilon
        self.dp_delta = dp_delta
        self.dp_clip_norm = dp_clip_norm
        self.dp_encryptor = DPNoiseEncryptor(
            epsilon=dp_epsilon, delta=dp_delta, clip_norm=dp_clip_norm,
        )

    def encrypt(self, gradient_dict, k=None):
        """Fisher-encrypt top-K, DP-noise the rest, return composed gradient.

        Returns
        -------
        protected_dict : dict[str, Tensor]
            Gradients with encrypted parameters zeroed and the
            non-encrypted slice perturbed with DP noise.
        metadata : dict
            Combined Fisher + DP metadata for the JSON sweep summary.
        """
        # 1. Fisher mask (with classifier-head guarantee unless the
        #    caller supplied a metric without it).
        if k is not None:
            total_params = sum(g.numel() for g in gradient_dict.values())
            enc_pct = k / max(total_params, 1)
            fisher_result = self.fisher_metric.compute_with_dynamic_k(
                gradient_dict, enc_pct,
            )
        else:
            fisher_result = self.fisher_metric.compute(gradient_dict)
        encrypt_mask = fisher_result['encrypt_mask']

        # 2. Build the non-encrypted slice as its own dict so the DP
        #    mechanism's global L2 clip operates on just that slice
        #    rather than on the (mostly zeroed) full gradient.
        non_enc_dict = {}
        idx = 0
        for name, grad in gradient_dict.items():
            n = grad.numel()
            layer_mask = encrypt_mask[idx:idx + n].view(grad.shape).to(grad.device)
            # Where mask is True (encrypt), the entry will be zeroed
            # in the final composition; pre-zero here so the DP clip
            # doesn't see the encrypted-side magnitudes.
            non_enc_dict[name] = grad * (~layer_mask).float()
            idx += n

        # 3. DP noise on the non-encrypted slice.
        noisy_dict, dp_meta = self.dp_encryptor.encrypt(non_enc_dict)

        # 4. Final composition: encrypted positions stay zero (the
        #    server sees the HE-aggregated value, not the plaintext);
        #    the rest are the noisy version.
        protected_dict = {}
        idx = 0
        for name, grad in gradient_dict.items():
            n = grad.numel()
            layer_mask = encrypt_mask[idx:idx + n].view(grad.shape).to(grad.device)
            zero = torch.zeros_like(grad)
            protected_dict[name] = torch.where(layer_mask, zero, noisy_dict[name])
            idx += n

        metadata = {
            'strategy': 'selectiveshield',
            # Fisher side (mirrors FisherEncryptor metadata so downstream
            # JSON parsers don't need a separate code path).
            'mask_mode': fisher_result.get('mask_mode', 'fisher'),
            'weights_encrypted': fisher_result['weights_to_encrypt'],
            'pct_encrypted': fisher_result['pct_encrypted'],
            'encryption_threshold': fisher_result['encryption_threshold'],
            'encrypt_mask': encrypt_mask,
            'classifier_head_forced_count': fisher_result.get(
                'classifier_head_forced_count', 0),
            'classifier_head_forced_layers': fisher_result.get(
                'classifier_head_forced_layers', []),
            # DP side
            'dp_epsilon': dp_meta['epsilon'],
            'dp_delta': dp_meta['delta'],
            'dp_clip_norm': dp_meta['clip_norm'],
            'dp_sigma': dp_meta['sigma'],
            'dp_original_grad_norm': dp_meta['original_grad_norm'],
            'dp_clip_factor': dp_meta['clip_factor'],
            'dp_noise_norm': dp_meta['noise_norm'],
        }

        return protected_dict, metadata

    def compute_overhead(self, metadata):
        """Estimate communication overhead.

        Same HE-expansion model as MaskCrypt/Fisher (encrypted slice
        gets the ciphertext factor; non-encrypted slice stays
        plaintext-sized — DP noise does not change the wire format).
        """
        pct = metadata['pct_encrypted']
        he_expansion = 50.0  # Cheon et al. CKKS, characteristic
        return 1.0 + pct * (he_expansion - 1.0)
