"""Yin 2021 GradInversion via the breaching library.

Reference implementation from Geiping's lab:
  https://github.com/JonasGeiping/breaching
Preset: ``seethroughgradients`` (Yin 2021, cosine + Langevin + BN-matching
via DeepInversion forward hooks + total-variation).

Group-consistency regularization is not implemented upstream in breaching
either — this matches the state of the art for reproducible Yin evaluation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def _reset_hydra_singleton():
    try:
        from hydra.core.global_hydra import GlobalHydra
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
    except Exception:
        pass


class GradInversionBreaching:
    """Yin 2021 GradInversion wrapper around breaching.seethroughgradients.

    Matches the attack-class interface used elsewhere in tests/attack_sanity_check.py:
      attack(gradient_dict, flat, batch_size, labels, original_images)
        -> {'reconstructed_images': Tensor[B,3,32,32] in [0,1] DENORMALIZED
              pixel space (matches Geiping/GGCDM/GI-NAS attack outputs;
              attack_sanity_check.py's clamp(0,1) is a no-op on this output),
            'score': float cosine similarity to real gradient,
            'breaching_stats': dict}

    Wrapper denormalisation note
    -----------------------------
    breaching's seethroughgradients preset optimises in normalised
    (zero-mean, unit-variance) image space, so the raw rec["data"]
    tensor has values in roughly [-2, 2]. Without denormalisation
    here, the downstream PSNR comparison in tests/attack_sanity_check.py
    runs clamp(0, 1) on a normalised tensor and destroys ~70% of the
    signal -- root cause of the 6.47-vs-18 dB gap reported in
    sec VI.H of the paper. Fixed 2026-05-03.

    The model is expected to be positioned on ``device`` with its state_dict
    already loaded (typically from the artifact's ``global_state`` plus BN
    buffers). Gradients are passed as a name-keyed dict and converted to
    breaching's positional list form keyed by ``model.parameters()`` order.
    """

    def __init__(self, model, criterion, device,
                 n_iter=20000,
                 deep_inv_scale=0.1,
                 tv_scale=1e-4,
                 langevin_noise=0.01,
                 **_ignored):
        import breaching  # raises ImportError early with a clear stack
        self._breaching = breaching
        self.model = model
        self.criterion = criterion
        self.device = device
        self.n_iter = int(n_iter)
        self.deep_inv_scale = float(deep_inv_scale)
        self.tv_scale = float(tv_scale)
        self.langevin_noise = float(langevin_noise)

    def _build_cfg(self):
        _reset_hydra_singleton()
        cfg = self._breaching.get_attack_config("seethroughgradients")
        cfg.optim.max_iterations = self.n_iter
        cfg.optim.langevin_noise = self.langevin_noise
        cfg.regularization.deep_inversion.scale = self.deep_inv_scale
        cfg.regularization.total_variation.scale = self.tv_scale
        cfg.label_strategy = None
        return cfg

    def _align_grads(self, gradient_dict):
        param_names = [n for n, _ in self.model.named_parameters()]
        missing = set(param_names) - set(gradient_dict.keys())
        extra = set(gradient_dict.keys()) - set(param_names)
        if missing or extra:
            raise ValueError(
                f"gradient_dict does not align with model.named_parameters() — "
                f"missing={sorted(missing)[:3]} extra={sorted(extra)[:3]}"
            )
        return [gradient_dict[n].detach().to(self.device).float()
                for n in param_names]

    def attack(self, gradient_dict, flat, batch_size, labels,
               original_images=None):
        from omegaconf import DictConfig

        device = self.device
        grad_list = self._align_grads(gradient_dict)

        data_cfg = DictConfig(dict(
            modality="vision", task="classification",
            size=50_000, classes=10, shape=[3, 32, 32], normalize=True,
            mean=list(CIFAR_MEAN), std=list(CIFAR_STD),
        ))

        params_list = [p.detach().to(device) for p in self.model.parameters()]
        buffers_list = [b.detach().clone().to(device)
                        for b in self.model.buffers()]

        labels_t = torch.as_tensor(
            labels, device=device, dtype=torch.long,
        ).flatten()

        server_payload = [dict(
            parameters=params_list,
            buffers=buffers_list,
            metadata=data_cfg,
        )]
        shared_data = [dict(
            gradients=grad_list,
            buffers=buffers_list,
            metadata=dict(
                num_data_points=int(batch_size),
                labels=labels_t,
                local_hyperparams=None,
            ),
        )]

        cfg_attack = self._build_cfg()
        setup = dict(device=device, dtype=torch.float)
        attacker = self._breaching.attacks.prepare_attack(
            self.model, self.criterion, cfg_attack, setup,
        )

        rec, stats = attacker.reconstruct(
            server_payload, shared_data, {}, dryrun=False,
        )

        recon = rec["data"].detach()

        # Diagnostic: log the raw recon range so we can verify the
        # pre-denormalisation convention. Expected on a working
        # seethroughgradients run: roughly [-2, 2] (normalised CIFAR).
        try:
            print(f"[GradInversionBreaching] raw recon range: "
                  f"min={float(recon.min().item()):+.3f} "
                  f"max={float(recon.max().item()):+.3f} "
                  f"mean={float(recon.mean().item()):+.3f}")
        except Exception:
            pass

        # Denormalise to [0, 1] pixel space so downstream PSNR/SSIM/LPIPS
        # comparisons in tests/attack_sanity_check.py operate on the same
        # convention as the original ground-truth tensor (which gets
        # denormalised by _denormalize() before the comparison). Without
        # this, attack_sanity_check.py's clamp(0, 1) on a normalised
        # tensor destroys ~70% of the signal -- root cause of the
        # 6.47-vs-18 dB gap documented in sec VI.H. See module docstring.
        mean_t = torch.tensor(
            CIFAR_MEAN, device=recon.device, dtype=recon.dtype,
        ).view(1, 3, 1, 1)
        std_t = torch.tensor(
            CIFAR_STD, device=recon.device, dtype=recon.dtype,
        ).view(1, 3, 1, 1)
        recon = (recon * std_t + mean_t).clamp(0.0, 1.0)

        score = float('nan')
        try:
            was_training = self.model.training
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad_(True)
            self.model.zero_grad()
            logits = self.model(recon)
            ce = self.criterion(logits, labels_t)
            fake_grads = torch.autograd.grad(
                ce, self.model.parameters(), create_graph=False,
            )
            fake_flat = torch.cat([g.flatten() for g in fake_grads])
            real_flat = flat.flatten().to(device) if flat is not None else \
                torch.cat([g.flatten() for g in grad_list])
            score = float(F.cosine_similarity(
                fake_flat.unsqueeze(0), real_flat.unsqueeze(0),
            ).item())
            if was_training:
                self.model.train()
        except Exception as e:
            print(f"[GradInversionBreaching] score pass failed: {e}")

        stats_dict = {}
        if hasattr(stats, 'items'):
            for k in ('opt_value', 'num_iterations'):
                if k in stats:
                    stats_dict[k] = float(stats[k]) if isinstance(
                        stats[k], (int, float)
                    ) else str(stats[k])

        return {
            'reconstructed_images': recon,
            'score': score,
            'breaching_stats': stats_dict,
        }
