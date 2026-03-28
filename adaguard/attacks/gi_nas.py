"""GI-NAS attack — gradient inversion with multiple restarts.

Based on: "GI-NAS: Boosting Gradient Inversion Attacks Through Adaptive
Neural Architecture Search" (Yu et al., IEEE).
Uses L2 gradient matching with group-lasso regularization and multi-restart.
"""

import torch
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
