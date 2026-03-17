"""Fisher Information metric — per-weight sensitivity analysis.

F_i = g_i^2 (empirical Fisher approximation).
Used to identify which parameters leak the most information
and should be prioritized for encryption.
"""

import numpy as np
import torch


class FisherInformationMetric:
    """Per-weight Fisher Information for gradient sensitivity ranking.

    F_i = g_i^2 per weight (empirical Fisher).
    F_round = (1/L) * sum of all F_i across layers.
    Concentration measured via Gini coefficient.
    """

    def __init__(self, topk=50, enc_pct=0.1):
        self.topk = topk
        self.enc_pct = enc_pct

    def compute(self, gradient_dict):
        per_layer = {}
        all_fi = []
        param_names = []

        for name, grad in gradient_dict.items():
            fi = (grad ** 2).flatten()  # F_i = g_i^2
            all_fi.append(fi)
            per_layer[name] = {
                'mean_fi': fi.mean().item(),
                'max_fi': fi.max().item(),
                'sum_fi': fi.sum().item(),
                'num_params': fi.numel(),
            }
            param_names.extend([name] * fi.numel())

        if not all_fi:
            return self._empty()

        fi_all = torch.cat(all_fi)
        total = fi_all.sum().item()
        n = fi_all.numel()
        L = len(gradient_dict)

        # Normalized: F_tilde_i = F_i / sum(F_j)
        fi_norm = fi_all / total if total > 1e-12 else torch.zeros_like(fi_all)

        # F_round = average Fisher per parameter
        f_round = total / max(n, 1)
        # Normalized to [0,1] via sigmoid-style mapping
        f_round_norm = min(1.0, f_round / (f_round + 1.0)) if f_round > 0 else 0.0

        # Concentration: Gini coefficient on fi_norm
        s = torch.sort(fi_norm)[0].cpu().numpy()
        idx = np.arange(1, n + 1)
        gini = (
            (2.0 * np.sum(idx * s) / (n * np.sum(s)) - (n + 1) / n)
            if np.sum(s) > 1e-12
            else 0.0
        )

        # Top-k weights
        topk_vals, topk_idx = torch.topk(fi_all, min(self.topk, n))

        # Encryption threshold: top enc_pct% of weights
        k_enc = max(1, int(self.enc_pct * n))
        threshold = torch.topk(fi_all, k_enc)[0][-1].item()
        mask_encrypt = fi_all >= threshold

        fi_cpu = fi_all.detach().cpu()
        fi_norm_cpu = fi_norm.detach().cpu()

        return {
            'fisher_concentration': max(0.0, min(1.0, gini)),
            'fisher_round': f_round,
            'fisher_round_norm': f_round_norm,
            'fisher_total': total,
            'fisher_per_layer': per_layer,
            'fisher_per_weight': fi_cpu,
            'fisher_per_weight_norm': fi_norm_cpu,
            'encryption_threshold': threshold,
            'encrypt_mask': mask_encrypt.cpu(),
            'weights_to_encrypt': int(mask_encrypt.sum().item()),
            'pct_encrypted': mask_encrypt.float().mean().item(),
            'topk_fisher_values': topk_vals.detach().cpu(),
            'topk_fisher_indices': topk_idx.detach().cpu(),
            'param_names': param_names,
        }

    def compute_with_dynamic_k(self, gradient_dict, encrypt_pct):
        """Compute Fisher with a dynamically determined encryption percentage."""
        old_pct = self.enc_pct
        self.enc_pct = encrypt_pct
        result = self.compute(gradient_dict)
        self.enc_pct = old_pct
        return result

    def _empty(self):
        return {
            'fisher_concentration': 0.0,
            'fisher_round': 0.0,
            'fisher_round_norm': 0.0,
            'fisher_total': 0.0,
            'fisher_per_layer': {},
            'fisher_per_weight': torch.tensor([]),
            'fisher_per_weight_norm': torch.tensor([]),
            'encryption_threshold': 0.0,
            'encrypt_mask': torch.tensor([]),
            'weights_to_encrypt': 0,
            'pct_encrypted': 0.0,
            'topk_fisher_values': torch.tensor([]),
            'topk_fisher_indices': torch.tensor([]),
            'param_names': [],
        }
