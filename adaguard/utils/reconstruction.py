"""Reconstruction quality metrics: MSE, PSNR, SSIM."""

import numpy as np
import torch


def compute_mse(original, reconstructed):
    """Compute Mean Squared Error between two tensors."""
    if isinstance(original, torch.Tensor):
        original = original.detach().cpu().numpy()
    if isinstance(reconstructed, torch.Tensor):
        reconstructed = reconstructed.detach().cpu().numpy()
    return float(np.mean((original - reconstructed) ** 2))


def compute_psnr(original, reconstructed, max_val=1.0):
    """Compute Peak Signal-to-Noise Ratio."""
    mse = compute_mse(original, reconstructed)
    if mse < 1e-12:
        return 100.0  # Perfect reconstruction
    return float(10.0 * np.log10(max_val ** 2 / mse))


def compute_ssim(original, reconstructed):
    """Compute Structural Similarity Index (simplified single-channel).

    For multi-channel images, computes per-channel SSIM and averages.
    """
    if isinstance(original, torch.Tensor):
        original = original.detach().cpu().numpy()
    if isinstance(reconstructed, torch.Tensor):
        reconstructed = reconstructed.detach().cpu().numpy()

    # Constants for numerical stability
    C1 = (0.01) ** 2
    C2 = (0.03) ** 2

    def _ssim_single(img1, img2):
        mu1 = np.mean(img1)
        mu2 = np.mean(img2)
        sigma1_sq = np.var(img1)
        sigma2_sq = np.var(img2)
        sigma12 = np.mean((img1 - mu1) * (img2 - mu2))

        numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)
        return float(numerator / denominator)

    if original.ndim == 4:  # batch of images
        scores = []
        for i in range(original.shape[0]):
            for c in range(original.shape[1]):
                scores.append(_ssim_single(original[i, c], reconstructed[i, c]))
        return float(np.mean(scores))
    elif original.ndim == 3:  # single image (C, H, W)
        scores = [_ssim_single(original[c], reconstructed[c]) for c in range(original.shape[0])]
        return float(np.mean(scores))
    else:
        return _ssim_single(original, reconstructed)
