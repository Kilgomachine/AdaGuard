"""GGCDM attack — Gradient-Guided Conditional Diffusion Model inspired inversion.

Based on: "Enhanced Privacy Leakage from Noise-Perturbed Gradients via
Gradient-Guided Conditional Diffusion Models" (Meng et al.).
Uses cosine similarity + L2 hybrid loss with annealed noise injection.
"""

import torch
import torch.nn.functional as F
import torch.optim as optim

from ..utils.reconstruction import compute_mse, compute_psnr, compute_ssim


class GGCDMAttack:
    """Diffusion-inspired gradient inversion with annealed noise."""

    def __init__(self, model, criterion, device, n_iter=20, lr=0.1,
                 focus_layers=None):
        self.model = model
        self.criterion = criterion
        self.device = device
        self.n_iter = n_iter
        self.lr = lr
        self.focus_layers = focus_layers or []

    def _extract_focus(self, grads_tuple):
        if not self.focus_layers:
            return torch.cat([g.flatten() for g in grads_tuple])
        parts = []
        for (name, _), g in zip(self.model.named_parameters(), grads_tuple):
            if name in self.focus_layers:
                parts.append(g.flatten())
        return torch.cat(parts) if parts else torch.cat([g.flatten() for g in grads_tuple])

    def _get_real_focus(self, real_grad_dict):
        if not self.focus_layers:
            return torch.cat([g.flatten() for g in real_grad_dict.values()])
        parts = [
            real_grad_dict[n].flatten()
            for n in self.focus_layers if n in real_grad_dict
        ]
        return torch.cat(parts) if parts else torch.cat(
            [g.flatten() for g in real_grad_dict.values()]
        )

    def attack(self, real_grad_dict, real_flat_grad, batch_size=1, labels=None,
               original_images=None):
        """Run GGCDM-style gradient inversion attack."""
        if labels is None:
            if 'fc2.weight' in real_grad_dict and batch_size == 1:
                labels = torch.tensor(
                    [real_grad_dict['fc2.weight'].cpu().sum(1).argmin().item()],
                    dtype=torch.long, device=self.device,
                )
            else:
                labels = torch.randint(0, 10, (batch_size,), device=self.device)

        rf = self._get_real_focus(real_grad_dict).detach()
        x = torch.randn(batch_size, 3, 32, 32, device=self.device, requires_grad=True)
        opt = optim.Adam([x], lr=self.lr)

        for step in range(self.n_iter):
            ns = 1.0 - step / self.n_iter
            opt.zero_grad()
            self.model.zero_grad()

            if step < self.n_iter - 1:
                with torch.no_grad():
                    x.data += ns * 0.01 * torch.randn_like(x)

            dg = torch.autograd.grad(
                self.criterion(self.model(x), labels),
                self.model.parameters(),
                create_graph=True,
            )
            df = self._extract_focus(dg)

            loss = (
                1.0 - F.cosine_similarity(df.unsqueeze(0), rf.unsqueeze(0))
                + 0.01 * torch.norm(df - rf, 2) ** 2
            )
            loss.backward()
            opt.step()

        # Final reconstruction quality
        dg_f = torch.autograd.grad(
            self.criterion(self.model(x), labels),
            self.model.parameters(),
        )
        recon = self._extract_focus(dg_f).detach()
        ratio = torch.norm(rf - recon, 2).item() / max(torch.norm(rf, 2).item(), 1e-12)
        score = max(0.0, 1.0 - ratio)

        result = {
            'empirical_ggcdm': max(0.0, min(1.0, score)),
            'score': max(0.0, min(1.0, score)),
            'reconstructed_images': x.detach().clamp(0, 1),
        }

        if original_images is not None:
            try:
                recon_img = x.detach().clamp(0, 1)
                orig_img = original_images[:batch_size].detach().clamp(0, 1)
                result['gi_ggcdm_mse'] = compute_mse(orig_img, recon_img)
                result['gi_ggcdm_psnr'] = compute_psnr(orig_img, recon_img)
                result['gi_ggcdm_ssim'] = compute_ssim(orig_img, recon_img)
            except Exception:
                pass

        return result


class GGCDMFull(GGCDMAttack):
    """Paper-matched GGCDM attack (Meng et al. 2025).

    Key differences from lightweight version:
    - Many more iterations (1000+)
    - Gradient-guided sampling with configurable guidance rate
    - Proper annealed noise schedule (cosine)
    - Better loss weighting
    """

    def __init__(self, model, criterion, device, n_iter=1000, lr=0.1,
                 guidance_rate=0.20, focus_layers=None):
        super().__init__(model, criterion, device, n_iter=n_iter, lr=lr,
                         focus_layers=focus_layers)
        self.guidance_rate = guidance_rate

    def attack(self, real_grad_dict, real_flat_grad, batch_size=1, labels=None,
               original_images=None):
        """Run full GGCDM-style attack with gradient guidance."""
        if labels is None:
            if 'fc2.weight' in real_grad_dict and batch_size == 1:
                labels = torch.tensor(
                    [real_grad_dict['fc2.weight'].cpu().sum(1).argmin().item()],
                    dtype=torch.long, device=self.device,
                )
            else:
                labels = torch.randint(0, 10, (batch_size,), device=self.device)

        rf = self._get_real_focus(real_grad_dict).detach()
        x = torch.randn(batch_size, 3, 32, 32, device=self.device, requires_grad=True)
        opt = optim.Adam([x], lr=self.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.n_iter, eta_min=1e-5)

        mr = self.guidance_rate
        best_loss = float('inf')
        best_x = x.clone().detach()

        for step in range(self.n_iter):
            # Annealed noise schedule (cosine)
            t = step / self.n_iter
            noise_scale = 0.5 * (1 + torch.cos(torch.tensor(t * 3.14159)))

            opt.zero_grad()
            self.model.zero_grad()

            # Add annealed noise (diffusion-inspired)
            if step < self.n_iter - 1:
                with torch.no_grad():
                    x.data += noise_scale * 0.01 * torch.randn_like(x)

            dg = torch.autograd.grad(
                self.criterion(self.model(x), labels),
                self.model.parameters(),
                create_graph=True,
            )
            df = self._extract_focus(dg)

            # Gradient-guided loss with configurable mixing rate
            cos_loss = 1.0 - F.cosine_similarity(df.unsqueeze(0), rf.unsqueeze(0))
            l2_loss = torch.norm(df - rf, 2) ** 2

            # Blend unconditional (smooth) and conditional (accurate) via mr
            loss = (1 - mr) * cos_loss + mr * l2_loss * 0.01

            loss.backward()
            opt.step()
            scheduler.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_x = x.clone().detach()

            if (step + 1) % 200 == 0:
                print(f"        GGCDM [{step+1}/{self.n_iter}] loss={loss.item():.4f}", flush=True)

        # Final score
        dg_f = torch.autograd.grad(
            self.criterion(self.model(best_x.requires_grad_(True)), labels),
            self.model.parameters(),
        )
        recon = self._extract_focus(dg_f).detach()
        ratio = torch.norm(rf - recon, 2).item() / max(torch.norm(rf, 2).item(), 1e-12)
        score = max(0.0, 1.0 - ratio)

        result = {
            'empirical_ggcdm': max(0.0, min(1.0, score)),
            'score': max(0.0, min(1.0, score)),
            'reconstructed_images': best_x.clamp(0, 1),
        }

        if original_images is not None:
            try:
                recon_img = best_x.clamp(0, 1)
                orig_img = original_images[:batch_size].detach().clamp(0, 1)
                result['gi_ggcdm_mse'] = compute_mse(orig_img, recon_img)
                result['gi_ggcdm_psnr'] = compute_psnr(orig_img, recon_img)
                result['gi_ggcdm_ssim'] = compute_ssim(orig_img, recon_img)
            except Exception:
                pass

        return result
