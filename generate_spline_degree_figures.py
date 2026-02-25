"""Generate figures 13-18 comparing B-spline degrees 1-4.

Trains KAN+SGD on skull genome with degrees 1, 2, 3, 4 at 64x64 resolution,
500 iterations, 2 seeds. Produces:
  13_degree_convergence.png    — loss curves per degree
  14_degree_quality_bars.png   — final MSE/SSIM bars
  15_degree_wall_time.png      — training time comparison
  16_degree_spline_shapes.png  — same edge's learned spline across degrees
  17_degree_signal_propagation.png — layer-wise activation std per degree
  18_degree_image_comparison.png   — generated images side-by-side
"""

import os
import sys
import time

import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

from analysis.benchmark import benchmark_kan_sgd_degree, load_target_image, GENOME_CONFIGS
from analysis.spline_inspector import extract_spline_curve

# Match existing benchmark params
GENOME = 'skull'
IMG_SIZE = 64
N_ITERS = 500
SEEDS = [0, 1]
DEGREES = [1, 2, 3, 4]
FIG_DIR = 'figures'

os.makedirs(FIG_DIR, exist_ok=True)


def compute_ssim_simple(img1, img2):
    """Simple SSIM approximation (structural similarity).

    Uses the mean/variance/covariance formulation without windowing.
    Good enough for comparative benchmarks.
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu1 = img1.mean()
    mu2 = img2.mean()
    sigma1_sq = ((img1 - mu1) ** 2).mean()
    sigma2_sq = ((img2 - mu2) ** 2).mean()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()

    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
           ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim.item()


def run_experiments():
    """Run all degree experiments and return results."""
    print("Loading target image...")
    target_img = load_target_image(GENOME, img_size=256)

    results = {d: [] for d in DEGREES}

    for seed in SEEDS:
        for degree in DEGREES:
            print(f"  Seed {seed}, Degree {degree}...")
            result = benchmark_kan_sgd_degree(
                GENOME, target_img, spline_degree=degree,
                n_iters=N_ITERS, seed=seed, img_size=IMG_SIZE
            )
            # Compute metrics against target at IMG_SIZE
            import torch.nn.functional as F
            target_resized = target_img.permute(2, 0, 1).unsqueeze(0)
            target_resized = F.interpolate(target_resized, size=(IMG_SIZE, IMG_SIZE),
                                           mode='bilinear', align_corners=False)
            target_resized = target_resized.squeeze(0).permute(1, 2, 0)

            with torch.no_grad():
                final_mse = torch.mean((result['final_img'] - target_resized) ** 2).item()
                ssim = compute_ssim_simple(result['final_img'], target_resized)

            result['final_mse'] = final_mse
            result['ssim'] = ssim
            result['target_resized'] = target_resized
            results[degree].append(result)

    return results


def fig13_convergence(results):
    """Loss curves per degree."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i, degree in enumerate(DEGREES):
        all_losses = [r['losses'] for r in results[degree]]
        # Average across seeds
        min_len = min(len(l) for l in all_losses)
        losses_arr = np.array([l[:min_len] for l in all_losses])
        mean_loss = losses_arr.mean(axis=0)
        std_loss = losses_arr.std(axis=0)

        iters = np.arange(1, min_len + 1)
        ax.plot(iters, mean_loss, color=colors[i], label=f'Degree {degree}', linewidth=2)
        ax.fill_between(iters, mean_loss - std_loss, mean_loss + std_loss,
                        color=colors[i], alpha=0.15)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Convergence by B-Spline Degree (Skull, 64x64)')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, '13_degree_convergence.png'), dpi=150)
    plt.close(fig)
    print("  Saved 13_degree_convergence.png")


def fig14_quality_bars(results):
    """Final MSE and SSIM bars."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    mse_means, mse_stds = [], []
    ssim_means, ssim_stds = [], []

    for degree in DEGREES:
        mses = [r['final_mse'] for r in results[degree]]
        ssims = [r['ssim'] for r in results[degree]]
        mse_means.append(np.mean(mses))
        mse_stds.append(np.std(mses))
        ssim_means.append(np.mean(ssims))
        ssim_stds.append(np.std(ssims))

    x = np.arange(len(DEGREES))
    labels = [f'Deg {d}' for d in DEGREES]

    ax1.bar(x, mse_means, yerr=mse_stds, color=colors, capsize=5, edgecolor='black', linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel('Final MSE')
    ax1.set_title('Final MSE by Degree')
    ax1.grid(True, alpha=0.3, axis='y')

    ax2.bar(x, ssim_means, yerr=ssim_stds, color=colors, capsize=5, edgecolor='black', linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel('SSIM')
    ax2.set_title('SSIM by Degree')
    ax2.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Final Quality: B-Spline Degree Comparison (Skull)', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, '14_degree_quality_bars.png'), dpi=150)
    plt.close(fig)
    print("  Saved 14_degree_quality_bars.png")


def fig15_wall_time(results):
    """Training time comparison."""
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    times_mean, times_std = [], []
    for degree in DEGREES:
        times = [r['wall_time'] for r in results[degree]]
        times_mean.append(np.mean(times))
        times_std.append(np.std(times))

    x = np.arange(len(DEGREES))
    labels = [f'Deg {d}' for d in DEGREES]

    ax.bar(x, times_mean, yerr=times_std, color=colors, capsize=5, edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Wall Time (seconds)')
    ax.set_title('Training Time by B-Spline Degree (500 iters, 64x64)')
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, '15_degree_wall_time.png'), dpi=150)
    plt.close(fig)
    print("  Saved 15_degree_wall_time.png")


def fig16_spline_shapes(results):
    """Same edge's learned spline across degrees."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i, degree in enumerate(DEGREES):
        model = results[degree][0]['model']  # First seed
        layer = model.layers[0]  # First layer
        raw_inputs, spline_values, normalized = extract_spline_curve(layer, 0, 0, n_points=200)

        axes[i].plot(raw_inputs, spline_values, color=colors[i], linewidth=2)
        axes[i].set_title(f'Degree {degree}')
        axes[i].set_xlabel('Raw Input')
        axes[i].grid(True, alpha=0.3)

    axes[0].set_ylabel('Spline Output')
    fig.suptitle('Learned Spline Shape: Layer 0, Edge (0,0)', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, '16_degree_spline_shapes.png'), dpi=150)
    plt.close(fig)
    print("  Saved 16_degree_spline_shapes.png")


def fig17_signal_propagation(results):
    """Layer-wise activation std per degree."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i, degree in enumerate(DEGREES):
        model = results[degree][0]['model']
        model.eval()

        # Generate activations
        with torch.no_grad():
            _, features = model.generate_image(img_size=IMG_SIZE, return_features=True)

        stds = [feat.std().item() for feat in features]
        layers = list(range(len(stds)))
        ax.plot(layers, stds, 'o-', color=colors[i], label=f'Degree {degree}',
                linewidth=2, markersize=4)

    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Activation Std')
    ax.set_title('Signal Propagation by B-Spline Degree (Skull)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, '17_degree_signal_propagation.png'), dpi=150)
    plt.close(fig)
    print("  Saved 17_degree_signal_propagation.png")


def fig18_image_comparison(results):
    """Generated images side-by-side."""
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    # Target
    target = results[DEGREES[0]][0]['target_resized'].detach().cpu().numpy()
    axes[0].imshow(np.clip(target, 0, 1))
    axes[0].set_title('Target')
    axes[0].axis('off')

    colors_label = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for i, degree in enumerate(DEGREES):
        img = results[degree][0]['final_img'].detach().cpu().numpy()
        axes[i + 1].imshow(np.clip(img, 0, 1))
        mse = results[degree][0]['final_mse']
        axes[i + 1].set_title(f'Degree {degree}\nMSE: {mse:.5f}')
        axes[i + 1].axis('off')

    fig.suptitle('Generated Images by B-Spline Degree (Skull, 64x64, 500 iters)', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, '18_degree_image_comparison.png'), dpi=150)
    plt.close(fig)
    print("  Saved 18_degree_image_comparison.png")


if __name__ == '__main__':
    print("=== B-Spline Degree Comparison ===")
    print(f"Genome: {GENOME}, Image size: {IMG_SIZE}, Iterations: {N_ITERS}, Seeds: {SEEDS}")
    print()

    results = run_experiments()

    print("\nGenerating figures...")
    fig13_convergence(results)
    fig14_quality_bars(results)
    fig15_wall_time(results)
    fig16_spline_shapes(results)
    fig17_signal_propagation(results)
    fig18_image_comparison(results)

    print("\nDone! All figures saved to figures/")
