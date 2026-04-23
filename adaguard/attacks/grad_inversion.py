"""GradInversion attack — focus-layer gradient matching.

Based on: "See through Gradients: Image Batch Recovery via GradInversion"
(Yin et al., NVIDIA). Uses cosine similarity loss + TV regularization.
"""

import torch
import torch.nn.functional as F
import torch.optim as optim

from ..utils.reconstruction import compute_mse, compute_psnr, compute_ssim


class GradInversionAttack:
    """Focus-layer gradient inversion attack."""

    def __init__(self, model, criterion, device, n_iter=20, lr=0.1,
                 tv_lambda=1e-4, focus_layers=None):
        self.model = model
        self.criterion = criterion
        self.device = device
        self.n_iter = n_iter
        self.lr = lr
        self.tv_lambda = tv_lambda
        self.focus_layers = focus_layers or []

    def _tv(self, x):
        return (
            torch.sum(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
            + torch.sum(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
        )

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
        """Run gradient inversion attack.

        Args:
            original_images: if provided, compute MSE/PSNR/SSIM against reconstructed images

        Returns:
            dict with 'empirical_gradinversion' score and optional reconstruction metrics
        """
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

        for _ in range(self.n_iter):
            opt.zero_grad()
            self.model.zero_grad()
            dg = torch.autograd.grad(
                self.criterion(self.model(x), labels),
                self.model.parameters(),
                create_graph=True,
            )
            df = self._extract_focus(dg)
            loss = (
                1.0 - F.cosine_similarity(df.unsqueeze(0), rf.unsqueeze(0))
                + self.tv_lambda * self._tv(x)
            )
            loss.backward()
            opt.step()

        # Final reconstruction quality (gradient space)
        dg_f = torch.autograd.grad(
            self.criterion(self.model(x), labels),
            self.model.parameters(),
        )
        recon = self._extract_focus(dg_f).detach()
        ratio = torch.norm(rf - recon, 2).item() / max(torch.norm(rf, 2).item(), 1e-12)
        score = max(0.0, 1.0 - ratio)

        result = {
            'empirical_gradinversion': max(0.0, min(1.0, score)),
            'score': max(0.0, min(1.0, score)),
            'reconstructed_images': x.detach().clamp(0, 1),
        }

        # Image-space reconstruction quality
        if original_images is not None:
            try:
                recon_img = x.detach().clamp(0, 1)
                orig_img = original_images[:batch_size].detach().clamp(0, 1)
                result['gi_gradinversion_mse'] = compute_mse(orig_img, recon_img)
                result['gi_gradinversion_psnr'] = compute_psnr(orig_img, recon_img)
                result['gi_gradinversion_ssim'] = compute_ssim(orig_img, recon_img)
            except Exception:
                pass

        return result


class GradInversionFull(GradInversionAttack):
    """Paper-matched GradInversion (Yin et al. 2021, NVIDIA).

    Implements the full Eq. 6/10 objective:
        L = R_grad + alpha_tv * R_TV + alpha_l2 * R_l2 + alpha_bn * R_BN
    where
        R_grad(x_hat) = || grad_W( L_ce(model(x_hat), y) ) - grad_W_real ||_2^2   (Eq. 5; L2, not cosine)
        R_BN(x_hat)  = sum_l  || mu_l(x_hat) - mu_l^BN ||_2 + || sig2_l(x_hat) - sig2_l^BN ||_2  (Eq. 10)

    R_BN is realised via forward-pre-hooks on every BN module: the hook captures
    the input activation, and _bn_loss() compares its per-channel batch stats to
    module.running_mean / module.running_var. This is the DeepInversion-style
    feature-distribution prior that was missing in the original implementation.

    Training-recipe additions relative to the lightweight baseline:
      * Warmup (Sec. 3.6): first ``warmup_iters`` steps use grad-match only so
        the image is not pinned to the BN/TV/L2 prior before it has a chance to
        approach a gradient-matching basin.
      * Langevin noise (Sec. 3.5): isotropic Gaussian injected into x after each
        step, variance annealed from ``langevin_sigma`` → 0, to escape local
        minima of the non-convex inversion objective.
      * Group consistency (Sec. 3.4, Eq. 12): when ``n_candidates > 1``, the
        final reconstruction is the *registered mean* of the N seed
        reconstructions — each is translated to align with seed 0 via 2D
        phase-correlation, then per-pixel averaged. This is the post-hoc
        approximation of the in-loop group consistency regulariser.

    Joint optimisation: Adam(lr=0.1), CosineAnnealingLR -> 0, 20000 iters.
    """

    def __init__(self, model, criterion, device, n_iter=20000, lr=0.1,
                 tv_lambda=1e-4, l2_lambda=1e-6, bn_lambda=0.1,
                 n_candidates=1, focus_layers=None, use_l2_grad=True,
                 warmup_iters=50, langevin_sigma=1e-3):
        super().__init__(model, criterion, device, n_iter=n_iter, lr=lr,
                         tv_lambda=tv_lambda, focus_layers=focus_layers)
        self.l2_lambda = l2_lambda
        self.bn_lambda = bn_lambda
        self.n_candidates = max(1, int(n_candidates))
        self.use_l2_grad = use_l2_grad
        self.warmup_iters = max(0, int(warmup_iters))
        self.langevin_sigma = float(langevin_sigma)
        self._bn_modules = []
        self._bn_inputs = {}
        self._bn_handles = []

    def _register_bn_hooks(self):
        self._bn_modules = []
        self._bn_inputs = {}
        self._bn_handles = []
        for module in self.model.modules():
            if isinstance(module, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
                if module.running_mean is not None and module.running_var is not None:
                    self._bn_modules.append(module)

                    def make_hook(m):
                        def hook(mod, inp):
                            self._bn_inputs[id(m)] = inp[0]
                        return hook

                    self._bn_handles.append(module.register_forward_pre_hook(make_hook(module)))

    def _remove_bn_hooks(self):
        for h in self._bn_handles:
            h.remove()
        self._bn_handles = []
        self._bn_inputs = {}
        self._bn_modules = []

    def _bn_loss(self):
        total = torch.tensor(0.0, device=self.device)
        for module in self._bn_modules:
            inp = self._bn_inputs.get(id(module))
            if inp is None:
                continue
            if inp.dim() == 4:
                batch_mean = inp.mean(dim=(0, 2, 3))
                batch_var = inp.var(dim=(0, 2, 3), unbiased=False)
            else:
                batch_mean = inp.mean(dim=0)
                batch_var = inp.var(dim=0, unbiased=False)
            total = total + torch.norm(batch_mean - module.running_mean.detach(), 2)
            total = total + torch.norm(batch_var - module.running_var.detach(), 2)
        return total

    def _optimize_one(self, rf, labels, batch_size, seed):
        gen = torch.Generator(device=self.device).manual_seed(int(seed))
        x = torch.randn(batch_size, 3, 32, 32, generator=gen,
                        device=self.device, requires_grad=True)
        opt = optim.Adam([x], lr=self.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.n_iter, eta_min=0.0)

        best_loss = float('inf')
        best_x = x.clone().detach()

        for step in range(self.n_iter):
            opt.zero_grad()
            self.model.zero_grad()
            self._bn_inputs.clear()

            logits = self.model(x)
            ce = self.criterion(logits, labels)
            dg = torch.autograd.grad(ce, self.model.parameters(), create_graph=True)
            df = self._extract_focus(dg)

            if self.use_l2_grad:
                grad_loss = ((df - rf) ** 2).sum()
            else:
                grad_loss = 1.0 - F.cosine_similarity(df.unsqueeze(0), rf.unsqueeze(0)).squeeze()

            # Warmup (Sec. 3.6): suppress priors until x has begun matching
            # gradients. Otherwise BN/TV/L2 dominate before grad_loss carries
            # any real signal and x snaps to a smooth but wrong minimum.
            in_warmup = step < self.warmup_iters
            if in_warmup:
                tv_loss = torch.tensor(0.0, device=self.device)
                l2_loss = torch.tensor(0.0, device=self.device)
                bn_loss = torch.tensor(0.0, device=self.device)
            else:
                tv_loss = self.tv_lambda * self._tv(x)
                l2_loss = self.l2_lambda * (x ** 2).sum()
                bn_loss = self.bn_lambda * self._bn_loss()

            loss = grad_loss + tv_loss + l2_loss + bn_loss
            loss.backward()
            opt.step()
            scheduler.step()

            # Langevin noise (Sec. 3.5) — annealed linearly to 0 at n_iter.
            if self.langevin_sigma > 0 and not in_warmup:
                t_remaining = max(0.0, 1.0 - step / max(1, self.n_iter))
                sigma_t = self.langevin_sigma * t_remaining
                if sigma_t > 0:
                    with torch.no_grad():
                        x.add_(sigma_t * torch.randn_like(x))

            cur = loss.item()
            if cur < best_loss:
                best_loss = cur
                best_x = x.detach().clone()

            if (step + 1) % 2000 == 0:
                print(
                    f"        GradInversion seed={seed} "
                    f"[{step+1}/{self.n_iter}] loss={cur:.4f} "
                    f"(grad={grad_loss.item():.4f} bn={bn_loss.item():.4f})",
                    flush=True,
                )

        return best_x, best_loss

    @staticmethod
    def _phase_correlate(ref, tgt):
        """Integer-pixel (dy, dx) that maximises cross-correlation of tgt→ref.

        Both inputs are (C, H, W) single-image tensors on the same device.
        Returns python ints.
        """
        g_ref = ref.mean(dim=0)
        g_tgt = tgt.mean(dim=0)
        F_ref = torch.fft.fft2(g_ref)
        F_tgt = torch.fft.fft2(g_tgt)
        cross = F_ref * F_tgt.conj()
        cross = cross / (cross.abs() + 1e-12)
        r = torch.fft.ifft2(cross).real
        idx = int(torch.argmax(r.flatten()).item())
        H, W = r.shape
        dy, dx = idx // W, idx % W
        if dy >= H // 2:
            dy -= H
        if dx >= W // 2:
            dx -= W
        return dy, dx

    def _register_group_mean(self, candidates):
        """Per-batch-position registration of N reconstructions via phase
        correlation, then per-pixel mean. Implements the Sec. 3.4 Eq. 12
        consistency operator as a post-hoc consensus."""
        if len(candidates) == 1:
            return candidates[0].detach()

        stacked = torch.stack([c.detach() for c in candidates], dim=0)  # (N,B,C,H,W)
        N, B, C, H, W = stacked.shape
        ref = stacked[0]  # (B, C, H, W)
        aligned = [ref]
        for i in range(1, N):
            shifted = torch.empty_like(stacked[i])
            for b in range(B):
                dy, dx = self._phase_correlate(ref[b], stacked[i, b])
                shifted[b] = torch.roll(stacked[i, b], shifts=(dy, dx), dims=(-2, -1))
            aligned.append(shifted)
        return torch.stack(aligned, dim=0).mean(dim=0)

    def attack(self, real_grad_dict, real_flat_grad, batch_size=1, labels=None,
               original_images=None):
        if labels is None:
            if 'fc2.weight' in real_grad_dict and batch_size == 1:
                labels = torch.tensor(
                    [real_grad_dict['fc2.weight'].cpu().sum(1).argmin().item()],
                    dtype=torch.long, device=self.device,
                )
            else:
                labels = torch.randint(0, 10, (batch_size,), device=self.device)

        rf = self._get_real_focus(real_grad_dict).detach()

        self._register_bn_hooks()
        try:
            candidates = []
            for i in range(self.n_candidates):
                x_i, l_i = self._optimize_one(rf, labels, batch_size, seed=42 + i)
                candidates.append((x_i, l_i))
        finally:
            self._remove_bn_hooks()

        # Group-consistency reconstruction: align all N seeds to seed-0 via
        # phase-correlation, then per-pixel average. Falls back to best-by-loss
        # when n_candidates == 1.
        if self.n_candidates > 1:
            seed_xs = [c[0] for c in candidates]
            best_x = self._register_group_mean(seed_xs)
        else:
            best_x = candidates[0][0]

        # Final gradient-space match score
        dg_f = torch.autograd.grad(
            self.criterion(self.model(best_x.detach().requires_grad_(True)), labels),
            self.model.parameters(),
        )
        recon = self._extract_focus(dg_f).detach()
        ratio = torch.norm(rf - recon, 2).item() / max(torch.norm(rf, 2).item(), 1e-12)
        score = max(0.0, 1.0 - ratio)

        result = {
            'empirical_gradinversion': max(0.0, min(1.0, score)),
            'score': max(0.0, min(1.0, score)),
            'reconstructed_images': best_x.detach().clamp(0, 1),
        }

        if original_images is not None:
            try:
                recon_img = best_x.detach().clamp(0, 1)
                orig_img = original_images[:batch_size].detach().clamp(0, 1)
                result['gi_gradinversion_mse'] = compute_mse(orig_img, recon_img)
                result['gi_gradinversion_psnr'] = compute_psnr(orig_img, recon_img)
                result['gi_gradinversion_ssim'] = compute_ssim(orig_img, recon_img)
            except Exception:
                pass

        return result


# ─── Geiping Inverting-Gradients recipe ─────────────────────────────

_CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR_STD = (0.2470, 0.2435, 0.2616)


class GradInversionGeiping:
    """Clean Inverting-Gradients recipe (Geiping et al. 2020).

    This is the optimisation core that Yin's GradInversion builds on top of.
    Our own Yin extensions (BN-feature matching, Langevin, group
    consistency) were net-negative in ablations (Attack Restructure 1,
    2026-04-22) — Geiping's cosine + signed-gradient + TV recipe recovered
    8-10 dB PSNR on V1 while the Yin extensions stalled at 5-7 dB.

    We therefore use Geiping's recipe as the GradInversion implementation
    for Phase 4 paper claims, documenting it as "GradInversion core
    (Geiping recipe, no BN / group-consistency extensions)". The
    breaching library also implements the Yin recipe with the same
    group-consistency gap — see breaching.seethroughgradients.

    Recipe:
      * Optimise a normalised image x in [-3, 3] (mean=0, std=1 prior).
      * Loss = 1 - cos_sim(grad_W(f(x), y), grad_W_real) + TV(x) * tv_lambda.
      * Signed gradient step on x before Adam update (Geiping Sec 3.2).
      * Adam(lr=0.1) + cosine schedule to zero over n_iter.
      * Box constraints every step: x ∈ [-3, 3] (roughly [0, 1] image
        range after de-normalisation).
    """

    def __init__(self, model, criterion, device, n_iter=20000, lr=0.1,
                 tv_lambda=1e-4, focus_layers=None):
        self.model = model
        self.criterion = criterion
        self.device = device
        self.n_iter = int(n_iter)
        self.lr = float(lr)
        self.tv_lambda = float(tv_lambda)
        self.focus_layers = focus_layers or []
        self._mean = torch.tensor(_CIFAR_MEAN, device=device).view(1, 3, 1, 1)
        self._std = torch.tensor(_CIFAR_STD, device=device).view(1, 3, 1, 1)

    def _tv(self, x):
        return (
            torch.sum(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
            + torch.sum(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
        )

    def _extract_focus(self, grads_tuple):
        if not self.focus_layers:
            return torch.cat([g.flatten() for g in grads_tuple])
        parts = []
        for (name, _), g in zip(self.model.named_parameters(), grads_tuple):
            if name in self.focus_layers:
                parts.append(g.flatten())
        return torch.cat(parts) if parts else torch.cat(
            [g.flatten() for g in grads_tuple]
        )

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
        if labels is None:
            if 'fc2.weight' in real_grad_dict and batch_size == 1:
                labels = torch.tensor(
                    [real_grad_dict['fc2.weight'].cpu().sum(1).argmin().item()],
                    dtype=torch.long, device=self.device,
                )
            else:
                labels = torch.randint(0, 10, (batch_size,), device=self.device)

        rf = self._get_real_focus(real_grad_dict).detach()

        x = torch.randn(batch_size, 3, 32, 32, device=self.device,
                        requires_grad=True)
        opt = optim.Adam([x], lr=self.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.n_iter, eta_min=0.0,
        )

        best_loss = float('inf')
        best_x = x.detach().clone()

        for step in range(self.n_iter):
            opt.zero_grad()
            self.model.zero_grad()

            logits = self.model(x)
            ce = self.criterion(logits, labels)
            dg = torch.autograd.grad(ce, self.model.parameters(),
                                     create_graph=True)
            df = self._extract_focus(dg)

            cos_loss = 1.0 - F.cosine_similarity(
                df.unsqueeze(0), rf.unsqueeze(0),
            ).squeeze()
            tv = self._tv(x) * self.tv_lambda
            loss = cos_loss + tv
            loss.backward()

            # Signed gradient step (Geiping Sec 3.2).
            with torch.no_grad():
                if x.grad is not None:
                    x.grad.sign_()

            opt.step()
            scheduler.step()

            with torch.no_grad():
                x.data.clamp_(-3.0, 3.0)

            cur = loss.item()
            if cur < best_loss:
                best_loss = cur
                best_x = x.detach().clone()

            if (step + 1) % 2000 == 0:
                print(
                    f"        GradInv-Geiping [{step+1}/{self.n_iter}] "
                    f"loss={cur:.4f} cos={cos_loss.item():.4f}",
                    flush=True,
                )

        # Recon in [0, 1] for metrics.
        with torch.no_grad():
            recon_01 = (best_x * self._std + self._mean).clamp(0, 1)

        # Gradient-space score for LeakScore compatibility.
        x_for_score = best_x.detach().clone().requires_grad_(True)
        dg_f = torch.autograd.grad(
            self.criterion(self.model(x_for_score), labels),
            self.model.parameters(),
        )
        recon_g = self._extract_focus(dg_f).detach()
        ratio = torch.norm(rf - recon_g, 2).item() / max(
            torch.norm(rf, 2).item(), 1e-12,
        )
        score = max(0.0, 1.0 - ratio)

        result = {
            'empirical_gradinversion': max(0.0, min(1.0, score)),
            'score': max(0.0, min(1.0, score)),
            'reconstructed_images': recon_01,
        }

        if original_images is not None:
            try:
                orig_img = original_images[:batch_size].detach().clamp(0, 1)
                result['gi_gradinversion_mse'] = compute_mse(orig_img, recon_01)
                result['gi_gradinversion_psnr'] = compute_psnr(orig_img, recon_01)
                result['gi_gradinversion_ssim'] = compute_ssim(orig_img, recon_01)
            except Exception:
                pass

        return result
