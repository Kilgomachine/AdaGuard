"""Fisher Information metric — per-weight sensitivity analysis.

F_i = g_i^2 (empirical Fisher approximation).
Used to identify which parameters leak the most information
and should be prioritized for encryption.

Classifier-head guarantee
-------------------------
A pure global top-K-by-Fisher selector can omit small classifier-head
tensors (e.g. ResNet's 10-element ``fc.bias``) on certain initialisations
because their *aggregate* Fisher mass is small relative to a single conv
layer's worth of parameters, even when their per-parameter scores are
high. When ``fc.bias`` is exposed, the LLG+/iDLG label-decoding primitive
walks straight through and label-recovery ASR jumps from 0 to 1.

We observed this failure on seed 456's round-249 model (ASR=1.0 across
GradInversion, GGCDM, and GI-NAS) while seeds 42 and 123 happened to
include ``fc.bias`` in the top-K and saw ASR=0. The fix is structural:
classifier-head parameters are force-included in the encrypted set
regardless of Fisher score. The cost is negligible (~5,140 params on
ResNet-18/CIFAR-10, +0.046% of total) and removes the entire class of
label-leak failures.

To reproduce the *broken* (pre-fix) behaviour for ablation, pass an
empty tuple: ``mandatory_layer_substrings=()``.
"""

import numpy as np
import torch


# Default classifier-head substrings. Match common naming in torchvision
# (``fc.``), HuggingFace transformers (``classifier.``), and timm
# (``head.``) so the guarantee fires across the architectures we use
# without per-model configuration.
DEFAULT_MANDATORY_LAYER_SUBSTRINGS = (
    "fc.bias", "fc.weight",
    "classifier.bias", "classifier.weight",
    "head.bias", "head.weight",
)


class FisherInformationMetric:
    """Per-weight Fisher Information for gradient sensitivity ranking.

    F_i = g_i^2 per weight (empirical Fisher).
    F_round = (1/L) * sum of all F_i across layers.
    Concentration measured via Gini coefficient.

    Parameters
    ----------
    topk : int
        Number of (rank, value) pairs to return for diagnostics.
    enc_pct : float
        Fraction of parameters to encrypt.
    mandatory_layer_substrings : tuple[str, ...] | None
        Layers whose name contains any of these substrings are
        force-included in the encrypted set regardless of Fisher score.
        Pass ``()`` to disable (reproduces pre-fix behaviour). Pass
        ``None`` to use :data:`DEFAULT_MANDATORY_LAYER_SUBSTRINGS`.
    mask_mode : {'fisher', 'random'}
        How to select the encrypted set at the configured ``enc_pct`` budget.
        ``'fisher'`` (default) selects the top-K parameters by per-weight
        Fisher score. ``'random'`` selects K parameters uniformly at random
        without replacement; used as the ablation baseline that isolates the
        contribution of vulnerability-ranked targeting from the encryption
        budget itself. The classifier-head guarantee fires identically in
        both modes — it is orthogonal to the ranking principle.
    random_seed : int | None
        Seed for the per-call random permutation when ``mask_mode='random'``.
        Set explicitly when running the Fisher-vs-random ablation so the
        ablation is reproducible. Has no effect when ``mask_mode='fisher'``.
    """

    def __init__(self, topk=50, enc_pct=0.1, mandatory_layer_substrings=None,
                 mask_mode='fisher', random_seed=None):
        self.topk = topk
        self.enc_pct = enc_pct
        self.mandatory_layer_substrings = (
            DEFAULT_MANDATORY_LAYER_SUBSTRINGS
            if mandatory_layer_substrings is None
            else tuple(mandatory_layer_substrings)
        )
        if mask_mode not in ('fisher', 'random'):
            raise ValueError(
                f"mask_mode must be 'fisher' or 'random', got {mask_mode!r}"
            )
        self.mask_mode = mask_mode
        self.random_seed = random_seed

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

        # Encryption mask. The Fisher-vs-random ablation switches between
        # vulnerability-ranked selection and a budget-matched uniform-random
        # mask without touching anything else in this method, so the
        # diagnostic dict (Gini, top-K stats, fisher_total, ...) stays
        # interpretable in both modes.
        k_enc = max(1, int(self.enc_pct * n))
        if self.mask_mode == 'fisher':
            threshold = torch.topk(fi_all, k_enc)[0][-1].item()
            mask_encrypt = fi_all >= threshold
        else:  # 'random' — validated in __init__
            gen = torch.Generator()
            if self.random_seed is not None:
                gen.manual_seed(int(self.random_seed))
            perm = torch.randperm(n, generator=gen)
            mask_encrypt = torch.zeros(n, dtype=torch.bool)
            mask_encrypt[perm[:k_enc]] = True
            threshold = 0.0  # not meaningful in random mode

        # Classifier-head guarantee. Force-include any parameter whose
        # owning layer name matches a mandatory substring (e.g. fc.bias).
        # This is the structural fix for the seed-456 failure mode where
        # a 10-element fc.bias fell below the global top-K threshold and
        # exposed the LLG+ label-decoding signal — see module docstring.
        n_forced = 0
        forced_layers = []
        if self.mandatory_layer_substrings:
            cursor = 0
            for name, grad in gradient_dict.items():
                n_layer = grad.numel()
                if any(s in name for s in self.mandatory_layer_substrings):
                    already_in = int(
                        mask_encrypt[cursor:cursor + n_layer].sum().item()
                    )
                    n_forced += n_layer - already_in
                    if already_in < n_layer:
                        forced_layers.append(name)
                    mask_encrypt[cursor:cursor + n_layer] = True
                cursor += n_layer

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
            # Recompute from the merged mask so callers see the actual
            # encrypted set size (top-K plus mandatory layers).
            'weights_to_encrypt': int(mask_encrypt.sum().item()),
            'pct_encrypted': mask_encrypt.float().mean().item(),
            'topk_fisher_values': topk_vals.detach().cpu(),
            'topk_fisher_indices': topk_idx.detach().cpu(),
            'param_names': param_names,
            # Classifier-head guarantee diagnostics — let the sweep tool
            # report whether the guarantee actually fired for this round.
            'classifier_head_forced_count': n_forced,
            'classifier_head_forced_layers': forced_layers,
            # Mask-mode diagnostic so downstream JSONs can distinguish the
            # Fisher and random ablation cells without re-deriving from
            # scenario metadata.
            'mask_mode': self.mask_mode,
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
            'classifier_head_forced_count': 0,
            'classifier_head_forced_layers': [],
            'mask_mode': self.mask_mode,
        }
