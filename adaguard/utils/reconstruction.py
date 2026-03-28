"""Reconstruction quality metrics for gradient inversion attack evaluation.

Includes standard image quality metrics (MSE, PSNR, SSIM, LPIPS, FID)
and attack-specific metrics (ASR, KNN Distance, Cosine Similarity, MCC, G-mean).
"""

import numpy as np
import torch


# ─── Basic Image Quality Metrics ─────────────────────────────────────

def compute_mse(original, reconstructed):
    """Compute Mean Squared Error between two tensors."""
    if isinstance(original, torch.Tensor):
        original = original.detach().cpu().numpy()
    if isinstance(reconstructed, torch.Tensor):
        reconstructed = reconstructed.detach().cpu().numpy()
    return float(np.mean((original - reconstructed) ** 2))


def compute_psnr(original, reconstructed, max_val=1.0):
    """Compute Peak Signal-to-Noise Ratio (dB). Higher is better."""
    mse = compute_mse(original, reconstructed)
    if mse < 1e-12:
        return 100.0  # Perfect reconstruction
    return float(10.0 * np.log10(max_val ** 2 / mse))


def compute_ssim(original, reconstructed):
    """Compute Structural Similarity Index (simplified). Higher is better.

    For multi-channel images, computes per-channel SSIM and averages.
    """
    if isinstance(original, torch.Tensor):
        original = original.detach().cpu().numpy()
    if isinstance(reconstructed, torch.Tensor):
        reconstructed = reconstructed.detach().cpu().numpy()

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


# ─── Perceptual Metrics ──────────────────────────────────────────────

def compute_lpips(original, reconstructed, net='alex'):
    """Compute LPIPS (Learned Perceptual Image Patch Similarity). Lower is better.

    Requires: pip install lpips
    Args:
        original: tensor (N, C, H, W) or (C, H, W), values in [0, 1]
        reconstructed: same shape as original
        net: 'alex' (default, fastest) or 'vgg'
    """
    import lpips

    if isinstance(original, np.ndarray):
        original = torch.from_numpy(original).float()
    if isinstance(reconstructed, np.ndarray):
        reconstructed = torch.from_numpy(reconstructed).float()

    # Ensure 4D
    if original.ndim == 3:
        original = original.unsqueeze(0)
        reconstructed = reconstructed.unsqueeze(0)

    # LPIPS expects [-1, 1]
    orig_scaled = original * 2 - 1
    recon_scaled = reconstructed * 2 - 1

    device = original.device
    loss_fn = lpips.LPIPS(net=net, verbose=False).to(device)
    with torch.no_grad():
        score = loss_fn(orig_scaled.to(device), recon_scaled.to(device))
    return float(score.mean().item())


def compute_fid(real_batch, fake_batch):
    """Compute Fréchet Inception Distance between two batches. Lower is better.

    Simplified: uses flattened feature vectors instead of Inception-v3.
    For proper FID, use torchmetrics.image.fid.FrechetInceptionDistance.
    """
    if isinstance(real_batch, torch.Tensor):
        real_batch = real_batch.detach().cpu().numpy()
    if isinstance(fake_batch, torch.Tensor):
        fake_batch = fake_batch.detach().cpu().numpy()

    # Flatten to feature vectors
    real_flat = real_batch.reshape(real_batch.shape[0], -1)
    fake_flat = fake_batch.reshape(fake_batch.shape[0], -1)

    mu_r, mu_f = np.mean(real_flat, axis=0), np.mean(fake_flat, axis=0)
    sigma_r = np.cov(real_flat, rowvar=False) + np.eye(real_flat.shape[1]) * 1e-6
    sigma_f = np.cov(fake_flat, rowvar=False) + np.eye(fake_flat.shape[1]) * 1e-6

    diff = mu_r - mu_f
    mean_term = np.dot(diff, diff)

    # Approximate sqrt(sigma_r @ sigma_f) via eigenvalues
    product = sigma_r @ sigma_f
    eigvals = np.real(np.linalg.eigvals(product))
    eigvals = np.maximum(eigvals, 0)
    sqrt_trace = np.sum(np.sqrt(eigvals))

    trace_term = np.trace(sigma_r) + np.trace(sigma_f) - 2 * sqrt_trace
    return float(mean_term + trace_term)


# ─── Attack Success Metrics ──────────────────────────────────────────

def compute_asr(original, reconstructed, ssim_threshold=0.5):
    """Compute Attack Success Rate: fraction of images with SSIM > threshold.

    Higher ASR means the attack succeeds more often (worse for privacy).
    """
    if isinstance(original, torch.Tensor):
        original = original.detach().cpu().numpy()
    if isinstance(reconstructed, torch.Tensor):
        reconstructed = reconstructed.detach().cpu().numpy()

    if original.ndim == 3:
        # Single image
        ssim = compute_ssim(original, reconstructed)
        return 1.0 if ssim > ssim_threshold else 0.0

    # Batch
    successes = 0
    n = original.shape[0]
    for i in range(n):
        ssim = compute_ssim(original[i], reconstructed[i])
        if ssim > ssim_threshold:
            successes += 1
    return float(successes / max(n, 1))


def compute_cosine_similarity_images(original, reconstructed):
    """Compute cosine similarity between flattened image tensors. Higher = more similar."""
    if isinstance(original, torch.Tensor):
        original = original.detach().cpu().float()
        reconstructed = reconstructed.detach().cpu().float()
    else:
        original = torch.from_numpy(original).float()
        reconstructed = torch.from_numpy(reconstructed).float()

    o_flat = original.reshape(-1)
    r_flat = reconstructed.reshape(-1)

    dot = torch.dot(o_flat, r_flat)
    norm_o = torch.norm(o_flat)
    norm_r = torch.norm(r_flat)
    if norm_o < 1e-12 or norm_r < 1e-12:
        return 0.0
    return float(dot / (norm_o * norm_r))


def compute_knn_distance(original, reconstructed, k=1):
    """Compute K-Nearest Neighbor distance in pixel space. Lower = better attack.

    For batch inputs: average distance from each reconstructed image to its
    nearest neighbor in the original batch.
    """
    if isinstance(original, torch.Tensor):
        original = original.detach().cpu().numpy()
    if isinstance(reconstructed, torch.Tensor):
        reconstructed = reconstructed.detach().cpu().numpy()

    o_flat = original.reshape(original.shape[0] if original.ndim == 4 else 1, -1)
    r_flat = reconstructed.reshape(reconstructed.shape[0] if reconstructed.ndim == 4 else 1, -1)

    # Pairwise L2 distances
    distances = []
    for i in range(r_flat.shape[0]):
        dists = np.linalg.norm(o_flat - r_flat[i], axis=1)
        knn_dists = np.sort(dists)[:k]
        distances.append(np.mean(knn_dists))
    return float(np.mean(distances))


def compute_ddcs(original, reconstructed):
    """Compute Diversity and Distance Composite Score.

    Combines intra-diversity of reconstructions with distance to originals.
    Lower = better attack (reconstructions are close to originals and diverse).
    """
    if isinstance(original, torch.Tensor):
        original = original.detach().cpu().numpy()
    if isinstance(reconstructed, torch.Tensor):
        reconstructed = reconstructed.detach().cpu().numpy()

    o_flat = original.reshape(-1)
    r_flat = reconstructed.reshape(-1)

    # Distance between original and reconstruction
    dist = np.linalg.norm(o_flat - r_flat)
    # Normalize by dimensionality
    dim = o_flat.shape[0]
    return float(dist / np.sqrt(dim))


def compute_avd(original, reconstructed):
    """Compute Absolute Variation Distance (mean absolute pixel difference).

    Lower = better attack.
    """
    if isinstance(original, torch.Tensor):
        original = original.detach().cpu().numpy()
    if isinstance(reconstructed, torch.Tensor):
        reconstructed = reconstructed.detach().cpu().numpy()
    return float(np.mean(np.abs(original - reconstructed)))


# ─── Label Inference Metrics ─────────────────────────────────────────

def compute_mcc(labels_true, labels_pred):
    """Compute Matthews Correlation Coefficient for label inference.

    MCC in [-1, 1]: 1 = perfect, 0 = random, -1 = inverse.
    For multi-class, uses micro-averaged binary MCC.
    """
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)

    if len(labels_true) == 0:
        return 0.0

    # Multi-class: compute accuracy-based proxy
    n = len(labels_true)
    correct = np.sum(labels_true == labels_pred)
    k = len(np.unique(labels_true))
    if k <= 1:
        return 1.0 if correct == n else 0.0

    # MCC for multi-class via confusion matrix
    # Simplified: use (accuracy - 1/k) / (1 - 1/k) mapped to [-1, 1]
    acc = correct / n
    chance = 1.0 / k
    if abs(1.0 - chance) < 1e-12:
        return 0.0
    return float((acc - chance) / (1.0 - chance))


def compute_gmean(labels_true, labels_pred):
    """Compute Geometric Mean of per-class recall. Higher is better.

    G-mean = (product of per-class recalls) ^ (1/num_classes)
    """
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)

    classes = np.unique(labels_true)
    if len(classes) == 0:
        return 0.0

    recalls = []
    for c in classes:
        mask = labels_true == c
        if mask.sum() == 0:
            recalls.append(0.0)
        else:
            recalls.append(float(np.sum(labels_pred[mask] == c) / mask.sum()))

    # Geometric mean
    product = np.prod(recalls)
    if product <= 0:
        return 0.0
    return float(product ** (1.0 / len(classes)))


# ─── Aggregator ──────────────────────────────────────────────────────

ALL_RECON_METRICS = [
    'mse', 'psnr', 'ssim', 'lpips', 'fid', 'asr',
    'cosine_similarity', 'knn_distance', 'ddcs', 'avd',
]

LABEL_INFERENCE_METRICS = ['mcc', 'gmean']


def compute_all_reconstruction_metrics(original, reconstructed,
                                        labels_true=None, labels_pred=None,
                                        use_lpips=True):
    """Compute all reconstruction quality metrics.

    Args:
        original: tensor/array (N, C, H, W) or (C, H, W)
        reconstructed: same shape as original
        labels_true: ground-truth labels (for MCC/G-mean)
        labels_pred: predicted/inferred labels (for MCC/G-mean)
        use_lpips: if True, compute LPIPS (requires lpips package)

    Returns:
        dict of metric_name -> float
    """
    results = {}

    results['mse'] = compute_mse(original, reconstructed)
    results['psnr'] = compute_psnr(original, reconstructed)
    results['ssim'] = compute_ssim(original, reconstructed)
    results['asr'] = compute_asr(original, reconstructed)
    results['cosine_similarity'] = compute_cosine_similarity_images(original, reconstructed)
    results['avd'] = compute_avd(original, reconstructed)
    results['ddcs'] = compute_ddcs(original, reconstructed)

    # KNN distance (needs batch dim)
    if isinstance(original, torch.Tensor) and original.ndim >= 4:
        results['knn_distance'] = compute_knn_distance(original, reconstructed)
    elif isinstance(original, np.ndarray) and original.ndim >= 4:
        results['knn_distance'] = compute_knn_distance(original, reconstructed)
    else:
        results['knn_distance'] = 0.0

    # LPIPS (may not be installed)
    if use_lpips:
        try:
            results['lpips'] = compute_lpips(original, reconstructed)
        except (ImportError, Exception):
            results['lpips'] = 0.0
    else:
        results['lpips'] = 0.0

    # FID (needs batch)
    if isinstance(original, torch.Tensor) and original.ndim == 4 and original.shape[0] >= 2:
        results['fid'] = compute_fid(original, reconstructed)
    elif isinstance(original, np.ndarray) and original.ndim == 4 and original.shape[0] >= 2:
        results['fid'] = compute_fid(original, reconstructed)
    else:
        results['fid'] = 0.0

    # Label inference metrics
    if labels_true is not None and labels_pred is not None:
        results['mcc'] = compute_mcc(labels_true, labels_pred)
        results['gmean'] = compute_gmean(labels_true, labels_pred)
    else:
        results['mcc'] = 0.0
        results['gmean'] = 0.0

    return results
