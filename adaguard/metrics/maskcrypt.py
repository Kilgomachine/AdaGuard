"""MaskCrypt metric — per-weight vulnerability scoring.

v[i] = gradient[i] * (old_exposed_weight[i] - new_trained_weight[i])
Identifies which weight updates reveal the most about the training data.
"""

import math

import torch


class MaskCryptMetric:
    """MaskCrypt-style selective encryption scoring.

    Computes vulnerability score v[i] = grad[i] * (old_w[i] - new_w[i])
    for each weight. Ranks by |v[i]| to identify encryption targets.
    """

    def __init__(self, enc_pct=0.1):
        self.enc_pct = enc_pct

    def compute(self, gradient_dict, old_weights, new_weights):
        all_v = []
        per_layer_v = {}

        for name in gradient_dict:
            if name not in old_weights or name not in new_weights:
                continue
            grad = gradient_dict[name].flatten()
            old_w = old_weights[name].flatten().to(grad.device)
            new_w = new_weights[name].flatten().to(grad.device)
            v = grad * (old_w - new_w)  # v[i] = grad[i] * (old[i] - new[i])
            all_v.append(v)
            per_layer_v[name] = {
                'mean_abs_v': v.abs().mean().item(),
                'max_abs_v': v.abs().max().item(),
                'sum_abs_v': v.abs().sum().item(),
            }

        if not all_v:
            return self._empty()

        v_all = torch.cat(all_v)
        v_abs = v_all.abs()
        n = v_abs.numel()
        total_v = v_abs.sum().item()

        if total_v < 1e-20:
            return self._empty()

        # Score 1: fraction of total |v| in top enc_pct weights
        k = max(1, int(self.enc_pct * n))
        topk_v, topk_idx = torch.topk(v_abs, k)
        top_fraction = topk_v.sum().item() / total_v

        # Score 2: normalized L2 magnitude
        v_l2 = torch.norm(v_abs, 2).item()
        v_l2_norm = (
            v_l2 / (math.sqrt(n) * v_abs.max().item())
            if v_abs.max().item() > 1e-20
            else 0.0
        )

        # Encryption mask
        threshold = topk_v[-1].item()
        encrypt_mask = v_abs >= threshold

        return {
            'maskcrypt_vuln_score': max(0.0, min(1.0, top_fraction)),
            'maskcrypt_l2_score': max(0.0, min(1.0, v_l2_norm)),
            'maskcrypt_per_weight': v_all.detach().cpu(),
            'maskcrypt_per_weight_abs': v_abs.detach().cpu(),
            'maskcrypt_per_layer': per_layer_v,
            'maskcrypt_total_v': total_v,
            'maskcrypt_encrypt_mask': encrypt_mask.cpu(),
            'maskcrypt_weights_to_protect': int(encrypt_mask.sum().item()),
            'maskcrypt_threshold': threshold,
            'maskcrypt_topk_values': topk_v.detach().cpu(),
            'maskcrypt_topk_indices': topk_idx.detach().cpu(),
        }

    def compute_with_dynamic_k(self, gradient_dict, old_weights, new_weights, encrypt_pct):
        """Compute MaskCrypt with a dynamically determined encryption percentage."""
        old_pct = self.enc_pct
        self.enc_pct = encrypt_pct
        result = self.compute(gradient_dict, old_weights, new_weights)
        self.enc_pct = old_pct
        return result

    def _empty(self):
        return {
            'maskcrypt_vuln_score': 0.0,
            'maskcrypt_l2_score': 0.0,
            'maskcrypt_per_weight': torch.tensor([]),
            'maskcrypt_per_weight_abs': torch.tensor([]),
            'maskcrypt_per_layer': {},
            'maskcrypt_total_v': 0.0,
            'maskcrypt_encrypt_mask': torch.tensor([]),
            'maskcrypt_weights_to_protect': 0,
            'maskcrypt_threshold': 0.0,
            'maskcrypt_topk_values': torch.tensor([]),
            'maskcrypt_topk_indices': torch.tensor([]),
        }

    @staticmethod
    def simulate_weight_delta(model, gradient_dict, lr):
        """Simulate new_weight = old_weight - lr * gradient."""
        old_w = {n: p.clone().detach() for n, p in model.named_parameters()}
        new_w = {}
        for name, p in model.named_parameters():
            if name in gradient_dict:
                new_w[name] = p.detach() - lr * gradient_dict[name]
            else:
                new_w[name] = p.detach().clone()
        return old_w, new_w
