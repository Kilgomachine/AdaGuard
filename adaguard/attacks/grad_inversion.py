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

        result = {'empirical_gradinversion': max(0.0, min(1.0, score))}

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
