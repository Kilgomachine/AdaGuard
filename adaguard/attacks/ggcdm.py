"""GGCDM attack — Gradient-Guided Conditional Diffusion Model inversion.

Based on: "Enhanced Privacy Leakage from Noise-Perturbed Gradients via
Gradient-Guided Conditional Diffusion Models" (Meng et al. 2025).

Three classes are provided:
  * :class:`GGCDMAttack`   — pixel-space cosine+L2 baseline (lightweight,
    used for LeakScore-style Shadow Attacks).
  * :class:`GGCDMFull`     — same algorithm, longer run + cosine LR; kept
    for backwards compatibility with pre-2026-04-21 results.
  * :class:`GGCDMPaper`    — faithful implementation of Meng et al.'s
    algorithm: pre-trained unconditional DDPM prior + DPS-style gradient
    guidance via Eq. 17 blending of unconditional and conditional sampling
    directions. This is the class that should be used for any paper
    claim about gradient inversion under diffusion priors.
"""

import torch
import torch.nn.functional as F
import torch.optim as optim

from ..utils.reconstruction import compute_mse, compute_psnr, compute_ssim

# CIFAR-10 normalisation constants — needed to bridge diffusion-model space
# (outputs in [-1, 1]) back to the classifier input space expected by the
# target network (mean/std-normalised [0, 1]).
_CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR_STD = (0.2470, 0.2435, 0.2616)


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


class GGCDMPaper:
    """Paper-faithful GGCDM (Meng et al. 2025).

    Uses a pretrained unconditional DDPM as the image prior and injects the
    gradient-matching objective as DPS-style guidance during reverse
    sampling (Chung et al. 2023, "Diffusion Posterior Sampling").

    Sampling update (Meng 2025, Eq. 17 adapted to unconditional prior):
        x_{t-1} = (1 - mr) · d_sample + mr · d_star
    where
        d_sample = DDIM(x_t, t)                              # unconditional direction
        d_star   = DDIM(x_t, t) - eta · ∇_{x_t} L_grad(x̂0)   # gradient-guided direction
        L_grad   = || ∇_θ CE(f_θ(x̂0), y) - g_real ||_2^2
        x̂0(x_t)  = (x_t - sqrt(1 - ᾱ_t) ϵ_θ(x_t, t)) / sqrt(ᾱ_t)

    The DDPM operates in [-1, 1] image space while the target classifier
    f_θ expects CIFAR-normalised inputs — we bridge the two at each step
    (scale + per-channel normalise) so the gradient-match loss is comparable
    to the true client gradient.

    Default checkpoint: ``google/ddpm-cifar10-32`` (Hugging Face).
    Override with ``diffusion_model_name`` for a local path in air-gapped runs.
    """

    def __init__(self, model, criterion, device, n_iter=100,
                 guidance_rate=0.5, guidance_scale=1.0,
                 diffusion_model_name='google/ddpm-cifar10-32',
                 focus_layers=None):
        self.model = model
        self.criterion = criterion
        self.device = device
        self.n_iter = int(n_iter)
        self.guidance_rate = float(guidance_rate)
        self.guidance_scale = float(guidance_scale)
        self.diffusion_model_name = diffusion_model_name
        self.focus_layers = focus_layers or []
        self._unet = None
        self._scheduler = None
        self._mean = torch.tensor(_CIFAR_MEAN, device=device).view(1, 3, 1, 1)
        self._std = torch.tensor(_CIFAR_STD, device=device).view(1, 3, 1, 1)

    def _load_diffusion(self):
        """Lazy-load diffusers U-Net and a DDIM scheduler.

        DDIM is used at inference to allow 50-200 step sampling instead of
        the full 1000 DDPM steps; the scheduler config is copied from the
        pretrained pipeline so α_t / ᾱ_t match the training distribution.
        """
        try:
            from diffusers import DDPMPipeline, DDIMScheduler
        except ImportError as e:
            raise RuntimeError(
                "GGCDMPaper requires the 'diffusers' package. "
                "Install with: pip install diffusers accelerate"
            ) from e

        pipe = DDPMPipeline.from_pretrained(self.diffusion_model_name)
        self._unet = pipe.unet.to(self.device).eval()
        for p in self._unet.parameters():
            p.requires_grad_(False)
        self._scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        self._scheduler.set_timesteps(self.n_iter, device=self.device)

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

    def _diffusion_to_classifier_space(self, x_diff):
        """Map DDPM output [-1, 1] -> CIFAR-normalised classifier input."""
        x_01 = (x_diff + 1.0) * 0.5
        return (x_01 - self._mean) / self._std

    def attack(self, real_grad_dict, real_flat_grad, batch_size=1, labels=None,
               original_images=None):
        if self._unet is None:
            self._load_diffusion()

        if labels is None:
            if 'fc2.weight' in real_grad_dict and batch_size == 1:
                labels = torch.tensor(
                    [real_grad_dict['fc2.weight'].cpu().sum(1).argmin().item()],
                    dtype=torch.long, device=self.device,
                )
            else:
                labels = torch.randint(0, 10, (batch_size,), device=self.device)

        rf = self._get_real_focus(real_grad_dict).detach()

        # Initialise at t = T (pure Gaussian in [-1, 1] space).
        x_t = torch.randn(batch_size, 3, 32, 32, device=self.device)
        mr = self.guidance_rate
        eta = self.guidance_scale

        timesteps = self._scheduler.timesteps.to(self.device)

        best_loss = float('inf')
        best_x01 = None

        for step_idx, t in enumerate(timesteps):
            t_scalar = t.long()

            # Branch A — unconditional ϵ_θ(x_t, t) (no grad, for DDIM step).
            with torch.no_grad():
                noise_uncond = self._unet(x_t, t_scalar).sample

            step_out = self._scheduler.step(noise_uncond, t_scalar, x_t)
            x_t_minus_1_uncond = step_out.prev_sample

            # Branch B — gradient-guided direction. Enable autograd through the
            # U-Net and classifier so we can differentiate L_grad w.r.t. x_t.
            x_t_g = x_t.detach().clone().requires_grad_(True)
            noise_g = self._unet(x_t_g, t_scalar).sample
            alpha_bar_t = self._scheduler.alphas_cumprod[t_scalar].to(self.device)
            x0_hat = (x_t_g - (1 - alpha_bar_t).sqrt() * noise_g) / alpha_bar_t.sqrt()
            x0_hat_clamped = x0_hat.clamp(-1.0, 1.0)

            x_cls = self._diffusion_to_classifier_space(x0_hat_clamped)
            logits = self.model(x_cls)
            ce = self.criterion(logits, labels)
            dg = torch.autograd.grad(
                ce, self.model.parameters(),
                create_graph=True, retain_graph=True,
            )
            df = self._extract_focus(dg)
            loss_grad = ((df - rf) ** 2).sum()

            grad_xt, = torch.autograd.grad(loss_grad, x_t_g)

            # Adaptive norm scaling so the guidance magnitude is comparable to
            # the unconditional step — otherwise the huge ||∇_{x_t} L_grad||
            # (parameter-space second-order) destroys the trajectory.
            denom = grad_xt.flatten(1).norm(dim=1).view(-1, 1, 1, 1) + 1e-12
            guidance = eta * grad_xt / denom

            # Eq. 17 blend of d_sample and d_star.
            d_sample = x_t_minus_1_uncond
            d_star = x_t_minus_1_uncond - guidance
            x_t = ((1 - mr) * d_sample + mr * d_star).detach()

            cur = loss_grad.item()
            if cur < best_loss:
                best_loss = cur
                # Store the current x̂0 prediction (in [0, 1]) as the best guess.
                best_x01 = ((x0_hat_clamped.detach() + 1) * 0.5).clamp(0, 1)

            if (step_idx + 1) % 20 == 0:
                print(
                    f"        GGCDM [{step_idx+1}/{len(timesteps)}] "
                    f"t={int(t_scalar.item())} "
                    f"loss_grad={cur:.4f}",
                    flush=True,
                )

        # Final x̂0 from last x_t (deterministic, no guidance).
        with torch.no_grad():
            noise_final = self._unet(x_t, timesteps[-1].long()).sample
            alpha_bar_final = self._scheduler.alphas_cumprod[timesteps[-1].long()].to(self.device)
            x0_final = (x_t - (1 - alpha_bar_final).sqrt() * noise_final) / alpha_bar_final.sqrt()
            x0_final_01 = ((x0_final.clamp(-1, 1) + 1) * 0.5).clamp(0, 1)

        recon = best_x01 if best_x01 is not None else x0_final_01

        # Gradient-space match score for LeakScore compatibility.
        x_cls_final = (recon - self._mean) / self._std
        dg_f = torch.autograd.grad(
            self.criterion(self.model(x_cls_final.detach().requires_grad_(True)), labels),
            self.model.parameters(),
        )
        recon_g = self._extract_focus(dg_f).detach()
        ratio = torch.norm(rf - recon_g, 2).item() / max(torch.norm(rf, 2).item(), 1e-12)
        score = max(0.0, 1.0 - ratio)

        result = {
            'empirical_ggcdm': max(0.0, min(1.0, score)),
            'score': max(0.0, min(1.0, score)),
            'reconstructed_images': recon,
        }

        if original_images is not None:
            try:
                orig_img = original_images[:batch_size].detach().clamp(0, 1)
                result['gi_ggcdm_mse'] = compute_mse(orig_img, recon)
                result['gi_ggcdm_psnr'] = compute_psnr(orig_img, recon)
                result['gi_ggcdm_ssim'] = compute_ssim(orig_img, recon)
            except Exception:
                pass

        return result
