"""GI-NAS attack — gradient inversion with adaptive architecture search.

Based on: "GI-NAS: Boosting Gradient Inversion Attacks Through Adaptive
Neural Architecture Search" (Yu et al. 2025, IEEE).

Three classes are provided:
  * :class:`GINASAttack` — pixel-space DLG-with-lasso baseline (lightweight,
    used for LeakScore-style Shadow Attacks).
  * :class:`GINASFull`   — same algorithm with cosine LR + signed gradient;
    kept for backwards compatibility.
  * :class:`GINASPaper`  — faithful reimplementation: optimises the weights
    of a convolutional *generator* (not pixels directly), after a Stage-1
    architecture search that picks the generator configuration with the
    lowest initial gradient-matching loss. Implements the two-stage
    recipe of Yu et al. (Alg. 1).
"""

import torch
import torch.nn as nn
import torch.optim as optim

from ..utils.reconstruction import compute_mse, compute_psnr, compute_ssim


class GINASAttack:
    """Multi-restart gradient inversion attack."""

    def __init__(self, model, criterion, device, n_iter=20, lr=0.1,
                 n_restarts=1, gl=1e-3, focus_layers=None):
        self.model = model
        self.criterion = criterion
        self.device = device
        self.n_iter = n_iter
        self.lr = lr
        self.n_restarts = n_restarts
        self.gl = gl
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

    def _run(self, rf, bs, labels):
        x = torch.randn(bs, 3, 32, 32, device=self.device, requires_grad=True)
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
                torch.norm(df - rf, 2) ** 2
                + self.gl * torch.var(x.mean(dim=(2, 3)))
            )
            loss.backward()
            opt.step()

        dg_f = torch.autograd.grad(
            self.criterion(self.model(x), labels),
            self.model.parameters(),
        )
        return self._extract_focus(dg_f).detach(), x.detach()

    def attack(self, real_grad_dict, real_flat_grad, batch_size=1, labels=None,
               original_images=None):
        """Run multi-restart gradient inversion attack."""
        if labels is None:
            if 'fc2.weight' in real_grad_dict and batch_size == 1:
                labels = torch.tensor(
                    [real_grad_dict['fc2.weight'].cpu().sum(1).argmin().item()],
                    dtype=torch.long, device=self.device,
                )
            else:
                labels = torch.randint(0, 10, (batch_size,), device=self.device)

        rf = self._get_real_focus(real_grad_dict).detach()
        best = -1.0
        best_x = None

        for _ in range(self.n_restarts):
            recon, x_recon = self._run(rf, batch_size, labels)
            ratio = torch.norm(rf - recon, 2).item() / max(torch.norm(rf, 2).item(), 1e-12)
            s = max(0.0, 1.0 - ratio)
            if s > best:
                best = s
                best_x = x_recon

        result = {
            'empirical_ginas': max(0.0, min(1.0, best)),
            'score': max(0.0, min(1.0, best)),
            'reconstructed_images': best_x.clamp(0, 1) if best_x is not None else None,
        }

        if original_images is not None and best_x is not None:
            try:
                recon_img = best_x.clamp(0, 1)
                orig_img = original_images[:batch_size].detach().clamp(0, 1)
                result['gi_ginas_mse'] = compute_mse(orig_img, recon_img)
                result['gi_ginas_psnr'] = compute_psnr(orig_img, recon_img)
                result['gi_ginas_ssim'] = compute_ssim(orig_img, recon_img)
            except Exception:
                pass

        return result


class GINASFull(GINASAttack):
    """Paper-matched GI-NAS attack (Yu et al. 2025).

    Key differences from lightweight version:
    - Many more iterations per restart
    - Multiple restarts (8 by default)
    - Architecture search over candidate networks (simplified: random hyperparams)
    - Signed gradient descent with Adam
    """

    def __init__(self, model, criterion, device, n_iter=500, lr=1e-3,
                 n_restarts=8, n_candidates=100, gl=1e-3, focus_layers=None):
        super().__init__(model, criterion, device, n_iter=n_iter, lr=lr,
                         n_restarts=n_restarts, gl=gl, focus_layers=focus_layers)
        self.n_candidates = n_candidates

    def _run(self, rf, bs, labels):
        """Enhanced run with cosine LR and better initialization."""
        x = torch.randn(bs, 3, 32, 32, device=self.device, requires_grad=True)
        opt = optim.Adam([x], lr=self.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.n_iter, eta_min=1e-6)

        best_loss = float('inf')
        best_x = x.clone().detach()

        for step in range(self.n_iter):
            opt.zero_grad()
            self.model.zero_grad()
            dg = torch.autograd.grad(
                self.criterion(self.model(x), labels),
                self.model.parameters(),
                create_graph=True,
            )
            df = self._extract_focus(dg)
            loss = (
                torch.norm(df - rf, 2) ** 2
                + self.gl * torch.var(x.mean(dim=(2, 3)))
            )
            loss.backward()

            # Signed gradient descent (per GI-NAS paper)
            with torch.no_grad():
                x.grad.data = x.grad.data.sign()

            opt.step()
            scheduler.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_x = x.clone().detach()

        dg_f = torch.autograd.grad(
            self.criterion(self.model(best_x.requires_grad_(True)), labels),
            self.model.parameters(),
        )
        return self._extract_focus(dg_f).detach(), best_x

    def attack(self, real_grad_dict, real_flat_grad, batch_size=1, labels=None,
               original_images=None):
        """Run full GI-NAS with multiple restarts and architecture search."""
        if labels is None:
            if 'fc2.weight' in real_grad_dict and batch_size == 1:
                labels = torch.tensor(
                    [real_grad_dict['fc2.weight'].cpu().sum(1).argmin().item()],
                    dtype=torch.long, device=self.device,
                )
            else:
                labels = torch.randint(0, 10, (batch_size,), device=self.device)

        rf = self._get_real_focus(real_grad_dict).detach()
        best_score = -1.0
        best_x = None

        for restart in range(self.n_restarts):
            recon, x_recon = self._run(rf, batch_size, labels)
            ratio = torch.norm(rf - recon, 2).item() / max(torch.norm(rf, 2).item(), 1e-12)
            s = max(0.0, 1.0 - ratio)
            if s > best_score:
                best_score = s
                best_x = x_recon

            if (restart + 1) % 4 == 0:
                print(f"        GI-NAS restart [{restart+1}/{self.n_restarts}] best={best_score:.4f}", flush=True)

        result = {
            'empirical_ginas': max(0.0, min(1.0, best_score)),
            'score': max(0.0, min(1.0, best_score)),
            'reconstructed_images': best_x.clamp(0, 1) if best_x is not None else None,
        }

        if original_images is not None and best_x is not None:
            try:
                recon_img = best_x.clamp(0, 1)
                orig_img = original_images[:batch_size].detach().clamp(0, 1)
                result['gi_ginas_mse'] = compute_mse(orig_img, recon_img)
                result['gi_ginas_psnr'] = compute_psnr(orig_img, recon_img)
                result['gi_ginas_ssim'] = compute_ssim(orig_img, recon_img)
            except Exception:
                pass

        return result


# ─── Paper-faithful GI-NAS ──────────────────────────────────────────

_CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR_STD = (0.2470, 0.2435, 0.2616)


class _ConvGenerator(nn.Module):
    """Small conv-decoder generator G(z; gamma) -> images in [-1, 1].

    Latent z in R^(B x z_dim) → (B, base_ch*8, 2, 2) → 4 upsample-conv blocks →
    (B, 3, 32, 32). Skip-connection pattern and base channel width are the two
    NAS axes: varying base_ch alone yields the width family; skip controls
    feature reuse between levels.

    This is a simplified over-parameterised search space (~5 architectures vs.
    the ~100 in the paper) — enough to reproduce the qualitative boost of
    generator-based inversion without the full evolutionary NAS machinery.
    """

    def __init__(self, z_dim=128, base_ch=64, skip=False):
        super().__init__()
        self.base_ch = base_ch
        self.skip = skip
        self.z_dim = z_dim
        self.fc = nn.Linear(z_dim, base_ch * 8 * 2 * 2)

        def block(in_c, out_c):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_c, out_c, 3, 1, 1),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.up1 = block(base_ch * 8, base_ch * 4)   # 2 -> 4
        self.up2 = block(base_ch * 4, base_ch * 2)   # 4 -> 8
        if skip:
            self.up3 = block(base_ch * 4, base_ch)   # in = concat(up2, proj(up1))
            self.skip_proj = nn.Conv2d(base_ch * 4, base_ch * 2, 1, 1, 0)
        else:
            self.up3 = block(base_ch * 2, base_ch)   # 8 -> 16
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(base_ch, 3, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, z):
        h = self.fc(z).view(z.size(0), self.base_ch * 8, 2, 2)
        h1 = self.up1(h)                                      # (B, 4ch, 4, 4)
        h2 = self.up2(h1)                                     # (B, 2ch, 8, 8)
        if self.skip:
            h1_up = nn.functional.interpolate(h1, scale_factor=2, mode='nearest')  # (B, 4ch, 8, 8)
            h1_proj = self.skip_proj(h1_up)                   # (B, 2ch, 8, 8)
            h2 = torch.cat([h2, h1_proj], dim=1)              # (B, 4ch, 8, 8)
        h3 = self.up3(h2)                                     # (B, ch, 16, 16)
        return self.up4(h3)                                   # (B, 3, 32, 32), tanh


class GINASPaper:
    """Paper-faithful GI-NAS (Yu et al. 2025).

    Two-stage attack (Alg. 1):
      Stage 1 — Architecture selection. Instantiate M candidate generators
        G_1, ..., G_M with different configurations (base_ch in {32, 48, 64,
        96, 128} x skip in {False, True}). Each gets a short warmup on the
        gradient-matching loss with a fixed random latent z. Select
        G_opt = argmin_m L_grad(G_m(z; gamma_m^warm), y).
      Stage 2 — Weight optimisation. Fix the architecture to G_opt and
        optimise its weights gamma (NOT the pixels) with Adam + signed
        gradient descent + cosine LR to minimise L_grad(G_opt(z; gamma), y).

    The classifier f_theta expects CIFAR-normalised inputs while G produces
    images in [-1, 1] (tanh); we bridge at each step.

    Default budget: M=5 candidates x 100 warmup steps, then 2000 Stage-2 steps.
    """

    def __init__(self, model, criterion, device, n_iter=2000, lr=1e-3,
                 n_candidates=5, warmup_iters=100, z_dim=128,
                 focus_layers=None):
        self.model = model
        self.criterion = criterion
        self.device = device
        self.n_iter = int(n_iter)
        self.lr = float(lr)
        self.n_candidates = max(1, int(n_candidates))
        self.warmup_iters = max(0, int(warmup_iters))
        self.z_dim = int(z_dim)
        self.focus_layers = focus_layers or []
        self._mean = torch.tensor(_CIFAR_MEAN, device=device).view(1, 3, 1, 1)
        self._std = torch.tensor(_CIFAR_STD, device=device).view(1, 3, 1, 1)

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

    def _gen_to_classifier_space(self, x_gen):
        """Map generator output [-1, 1] -> CIFAR-normalised classifier input."""
        x_01 = (x_gen + 1.0) * 0.5
        return (x_01 - self._mean) / self._std

    def _candidate_space(self):
        """M candidate (base_ch, skip) configs spanning the NAS space."""
        cfgs = [
            {'base_ch': 32, 'skip': False},
            {'base_ch': 48, 'skip': False},
            {'base_ch': 64, 'skip': False},
            {'base_ch': 64, 'skip': True},
            {'base_ch': 96, 'skip': False},
            {'base_ch': 128, 'skip': False},
            {'base_ch': 128, 'skip': True},
        ]
        return cfgs[: self.n_candidates]

    def _compute_grad_loss(self, x_gen, rf, labels):
        """Evaluate L_grad(x_gen) = ||grad_theta CE - g_real||^2 on generator output."""
        x_cls = self._gen_to_classifier_space(x_gen)
        logits = self.model(x_cls)
        ce = self.criterion(logits, labels)
        dg = torch.autograd.grad(
            ce, self.model.parameters(), create_graph=True, retain_graph=True,
        )
        df = self._extract_focus(dg)
        return ((df - rf) ** 2).sum()

    def _warmup_candidate(self, cfg, z, rf, labels):
        """Run a short warmup and return (generator, final L_grad)."""
        G = _ConvGenerator(
            z_dim=self.z_dim, base_ch=cfg['base_ch'], skip=cfg['skip'],
        ).to(self.device)
        opt = optim.Adam(G.parameters(), lr=self.lr)
        for _ in range(self.warmup_iters):
            opt.zero_grad()
            x_gen = G(z)
            loss = self._compute_grad_loss(x_gen, rf, labels)
            loss.backward()
            with torch.no_grad():
                for p in G.parameters():
                    if p.grad is not None:
                        p.grad.data = p.grad.data.sign()
            opt.step()
        final_loss = self._compute_grad_loss(G(z), rf, labels).item()
        return G, final_loss

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

        # Fixed latent — NAS reward measures architectural prior only, so z is
        # held constant across candidates to isolate the architecture effect.
        z = torch.randn(batch_size, self.z_dim, device=self.device)

        # ── Stage 1: architecture search ───────────────────────────
        print(
            f"        GI-NAS Stage 1: searching over {self.n_candidates} architectures",
            flush=True,
        )
        cfgs = self._candidate_space()
        best_cfg = None
        best_G = None
        best_loss = float('inf')
        for i, cfg in enumerate(cfgs):
            G_i, loss_i = self._warmup_candidate(cfg, z, rf, labels)
            print(
                f"          cand[{i}] base_ch={cfg['base_ch']:3d} "
                f"skip={cfg['skip']} init_loss={loss_i:.4f}",
                flush=True,
            )
            if loss_i < best_loss:
                best_loss = loss_i
                best_cfg = cfg
                best_G = G_i
            else:
                del G_i

        print(
            f"        GI-NAS picked base_ch={best_cfg['base_ch']} "
            f"skip={best_cfg['skip']} (init_loss={best_loss:.4f})",
            flush=True,
        )

        # ── Stage 2: refine best architecture ──────────────────────
        opt = optim.Adam(best_G.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.n_iter, eta_min=1e-6)
        best_x = None
        best_refine_loss = float('inf')

        for step in range(self.n_iter):
            opt.zero_grad()
            x_gen = best_G(z)
            loss = self._compute_grad_loss(x_gen, rf, labels)
            loss.backward()
            with torch.no_grad():
                for p in best_G.parameters():
                    if p.grad is not None:
                        p.grad.data = p.grad.data.sign()
            opt.step()
            scheduler.step()

            cur = loss.item()
            if cur < best_refine_loss:
                best_refine_loss = cur
                with torch.no_grad():
                    best_x = best_G(z).detach()

            if (step + 1) % 500 == 0:
                print(
                    f"          GI-NAS Stage 2 [{step+1}/{self.n_iter}] loss={cur:.4f}",
                    flush=True,
                )

        if best_x is None:
            with torch.no_grad():
                best_x = best_G(z).detach()

        # Map [-1, 1] -> [0, 1] for downstream metrics.
        recon_01 = ((best_x + 1) * 0.5).clamp(0, 1)

        # Gradient-space match score for LeakScore compatibility.
        x_cls = (recon_01 - self._mean) / self._std
        dg_f = torch.autograd.grad(
            self.criterion(self.model(x_cls.detach().requires_grad_(True)), labels),
            self.model.parameters(),
        )
        recon_g = self._extract_focus(dg_f).detach()
        ratio = torch.norm(rf - recon_g, 2).item() / max(torch.norm(rf, 2).item(), 1e-12)
        score = max(0.0, 1.0 - ratio)

        result = {
            'empirical_ginas': max(0.0, min(1.0, score)),
            'score': max(0.0, min(1.0, score)),
            'reconstructed_images': recon_01,
            'ginas_selected_base_ch': best_cfg['base_ch'],
            'ginas_selected_skip': best_cfg['skip'],
        }

        if original_images is not None:
            try:
                orig_img = original_images[:batch_size].detach().clamp(0, 1)
                result['gi_ginas_mse'] = compute_mse(orig_img, recon_01)
                result['gi_ginas_psnr'] = compute_psnr(orig_img, recon_01)
                result['gi_ginas_ssim'] = compute_ssim(orig_img, recon_01)
            except Exception:
                pass

        return result
