"""Comparison metrics for image quality and representation similarity."""

import torch
import numpy as np


def mse(img1, img2):
    """Mean squared error between two images.

    Args:
        img1, img2: Tensors of shape (H, W, 3) in [0, 1].

    Returns:
        Scalar MSE value.
    """
    return torch.mean((img1 - img2) ** 2).item()


def ssim(img1, img2, window_size=11, C1=0.01**2, C2=0.03**2):
    """Structural Similarity Index between two images.

    Simplified single-scale SSIM without gaussian weighting.

    Args:
        img1, img2: Tensors of shape (H, W, 3) in [0, 1].
        window_size: Size of the sliding window.
        C1, C2: Stability constants.

    Returns:
        Scalar SSIM value in [-1, 1] (1 = identical).
    """
    # Convert to (1, 3, H, W) for unfold
    img1_4d = img1.permute(2, 0, 1).unsqueeze(0)
    img2_4d = img2.permute(2, 0, 1).unsqueeze(0)

    # Average over channels
    mu1 = torch.nn.functional.avg_pool2d(img1_4d, window_size, stride=1, padding=window_size//2)
    mu2 = torch.nn.functional.avg_pool2d(img2_4d, window_size, stride=1, padding=window_size//2)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = torch.nn.functional.avg_pool2d(img1_4d ** 2, window_size, stride=1, padding=window_size//2) - mu1_sq
    sigma2_sq = torch.nn.functional.avg_pool2d(img2_4d ** 2, window_size, stride=1, padding=window_size//2) - mu2_sq
    sigma12 = torch.nn.functional.avg_pool2d(img1_4d * img2_4d, window_size, stride=1, padding=window_size//2) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean().item()


def feature_cosine_similarity(features1, features2):
    """Cosine similarity between feature maps from two models.

    Compares internal representations layer by layer.

    Args:
        features1, features2: Lists of tensors from generate_image(return_features=True).
            Each tensor has shape (H, W, D).

    Returns:
        List of per-layer cosine similarities.
    """
    similarities = []
    n_layers = min(len(features1), len(features2))
    for i in range(n_layers):
        f1 = features1[i].reshape(-1).float()
        f2 = features2[i].reshape(-1).float()
        if f1.shape != f2.shape:
            # Skip layers with different sizes
            similarities.append(float('nan'))
            continue
        cos_sim = torch.nn.functional.cosine_similarity(f1.unsqueeze(0), f2.unsqueeze(0)).item()
        similarities.append(cos_sim)
    return similarities
