"""Generate just the remaining figures (09-12) that failed."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from analysis.benchmark import load_target_image, GENOME_CONFIGS
from src.kan import KAN_CPPN
from src.train import train_sgd

GENOME = 'skull'
IMG_SIZE = 64
FIG_DIR = os.path.join(os.path.dirname(__file__), 'figures')

plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 11


def savefig(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: figures/{name}")


def fig_09():
    print("Generating 09_algorithm_viz.png...")
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # --- PSO ---
    ax = axes[0]
    np.random.seed(42)

    positions = np.random.randn(5, 2) * 0.5
    velocities = np.zeros_like(positions)
    personal_best = positions.copy()
    personal_best_scores = np.full(5, np.inf)
    global_best = positions[0].copy()
    global_best_score = np.inf

    def loss_2d(pos):
        return (pos[0] - 0.3)**2 + (pos[1] + 0.2)**2

    xx, yy = np.meshgrid(np.linspace(-1.5, 2, 100), np.linspace(-1.5, 1.5, 100))
    zz = (xx - 0.3)**2 + (yy + 0.2)**2
    ax.contour(xx, yy, zz, levels=15, alpha=0.3, cmap='gray')

    trajectories = [[] for _ in range(5)]
    for step in range(20):
        for i in range(5):
            trajectories[i].append(positions[i].copy())
            score = loss_2d(positions[i])
            if score < personal_best_scores[i]:
                personal_best_scores[i] = score
                personal_best[i] = positions[i].copy()
            if score < global_best_score:
                global_best_score = score
                global_best = positions[i].copy()
        r1 = np.random.rand(5, 2)
        r2 = np.random.rand(5, 2)
        velocities = 0.7 * velocities + 1.5 * r1 * (personal_best - positions) + 1.5 * r2 * (global_best - positions)
        positions += velocities

    particle_colors = plt.cm.Set1(np.linspace(0, 1, 5))
    for i in range(5):
        traj = np.array(trajectories[i])
        ax.plot(traj[:, 0], traj[:, 1], '-o', color=particle_colors[i], markersize=3,
                alpha=0.6, label=f'Particle {i}', linewidth=1.5)
        ax.plot(traj[0, 0], traj[0, 1], 's', color=particle_colors[i], markersize=10)
        ax.plot(traj[-1, 0], traj[-1, 1], '*', color=particle_colors[i], markersize=14)

    ax.plot(0.3, -0.2, 'kx', markersize=18, markeredgewidth=3, label='Optimum', zorder=10)
    ax.set_title('SwarmKAN: PSO on Spline Coefficients\n'
                 '(5 particles, inertia=0.7, cognitive=social=1.5)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.set_xlabel('Coefficient dim 1', fontsize=11)
    ax.set_ylabel('Coefficient dim 2', fontsize=11)
    ax.grid(True, alpha=0.3)

    # --- NES ---
    ax = axes[1]
    center = np.array([0.0, 0.0])
    ax.contour(xx, yy, zz, levels=15, alpha=0.3, cmap='gray')

    np.random.seed(123)
    gen_colors = ['#4292c6', '#2171b5', '#084594']
    for gen in range(3):
        for i in range(8):
            eps = np.random.randn(2) * 0.3
            pos = center + eps
            neg = center - eps
            ax.plot([neg[0], pos[0]], [neg[1], pos[1]], '-', color=gen_colors[gen], alpha=0.25, linewidth=1)
            ax.plot(pos[0], pos[1], 'o', color='#2ca02c', markersize=5, alpha=0.5)
            ax.plot(neg[0], neg[1], 'o', color='#d62728', markersize=5, alpha=0.5)

        ax.plot(center[0], center[1], 'D', color=gen_colors[gen], markersize=12,
                label=f'Center (gen {gen})', zorder=5, markeredgecolor='black', markeredgewidth=0.5)
        center = center + np.array([0.1, -0.07]) * (gen + 1) * 0.5

    ax.plot(0.3, -0.2, 'kx', markersize=18, markeredgewidth=3, label='Optimum', zorder=10)
    ax.set_title('MemeticKAN: NES Gradient Estimation\n'
                 '(antithetic pairs: green=+eps, red=-eps)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.set_xlabel('Parameter dim 1', fontsize=11)
    ax.set_ylabel('Parameter dim 2', fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.suptitle('Algorithm Visualization (2D Projection of Coefficient Space)', fontsize=15)
    plt.tight_layout()
    savefig(fig, '09_algorithm_viz.png')


def fig_10_11_12():
    print("Running ablation studies...")
    cfg = GENOME_CONFIGS[GENOME]
    target_img = load_target_image(GENOME, img_size=IMG_SIZE)
    N_ITERS = 500

    torch.manual_seed(42)
    kan_ortho = KAN_CPPN(n_layers=cfg['n_layers'], hidden_size=cfg['hidden_size'])
    print("  Training orthogonal...", end=" ", flush=True)
    losses_ortho, _ = train_sgd(kan_ortho, target_img, lr=3e-3, n_iters=N_ITERS, log_interval=0)
    print(f"MSE={losses_ortho[-1]:.6f}")

    torch.manual_seed(42)
    kan_kaiming = KAN_CPPN(n_layers=cfg['n_layers'], hidden_size=cfg['hidden_size'])
    for layer in kan_kaiming.layers:
        torch.nn.init.kaiming_uniform_(layer.base_weight)
    print("  Training kaiming...", end=" ", flush=True)
    losses_kaiming, _ = train_sgd(kan_kaiming, target_img, lr=3e-3, n_iters=N_ITERS, log_interval=0)
    print(f"MSE={losses_kaiming[-1]:.6f}")

    torch.manual_seed(42)
    kan_no_res = KAN_CPPN(n_layers=cfg['n_layers'], hidden_size=cfg['hidden_size'])
    with torch.no_grad():
        for layer in kan_no_res.layers:
            layer.base_weight.fill_(0)
            layer.base_weight.requires_grad_(False)
    print("  Training no-residual...", end=" ", flush=True)
    losses_no_res, _ = train_sgd(kan_no_res, target_img, lr=3e-3, n_iters=N_ITERS, log_interval=0)
    print(f"MSE={losses_no_res[-1]:.6f}")

    # Signal propagation
    torch.manual_seed(42)
    kan_test = KAN_CPPN(n_layers=cfg['n_layers'], hidden_size=cfg['hidden_size'])
    test_input = torch.randn(100, 4)

    with torch.no_grad():
        x = test_input
        stds_ortho = [x.std().item()]
        for layer in kan_test.layers:
            x = layer(x)
            stds_ortho.append(x.std().item())

    for layer in kan_test.layers:
        torch.nn.init.kaiming_uniform_(layer.base_weight)

    with torch.no_grad():
        x = test_input
        stds_kaiming = [x.std().item()]
        for layer in kan_test.layers:
            x = layer(x)
            stds_kaiming.append(x.std().item())

    # --- Fig 10: Ablation convergence ---
    print("Generating 10_ablation_convergence.png...")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(losses_ortho, label=f'Orthogonal init (final: {losses_ortho[-1]:.4f})', linewidth=2, color='#2ca02c')
    ax.plot(losses_kaiming, label=f'Kaiming init (final: {losses_kaiming[-1]:.4f})', linewidth=2, color='#d62728')
    ax.plot(losses_no_res, label=f'No residual path (final: {losses_no_res[-1]:.4f})', linewidth=2, color='#9467bd')
    ax.set_yscale('log')
    ax.set_xlabel('Iteration', fontsize=13)
    ax.set_ylabel('MSE (log scale)', fontsize=13)
    ax.set_title(f'Ablation Study: Init Strategy & Residual Path — {GENOME.title()}', fontsize=15)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    savefig(fig, '10_ablation_convergence.png')

    # --- Fig 11: Signal propagation ---
    print("Generating 11_signal_propagation.png...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    ax.plot(stds_ortho, 'o-', label='Orthogonal', linewidth=2, markersize=6, color='#2ca02c')
    ax.plot(stds_kaiming, 's-', label='Kaiming', linewidth=2, markersize=6, color='#d62728')
    ax.set_xlabel('Layer', fontsize=13)
    ax.set_ylabel('Activation Std', fontsize=13)
    ax.set_title('Signal Propagation (Linear Scale)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(stds_ortho, 'o-', label='Orthogonal', linewidth=2, markersize=6, color='#2ca02c')
    ax.plot(stds_kaiming, 's-', label='Kaiming', linewidth=2, markersize=6, color='#d62728')
    ax.set_yscale('log')
    ax.set_xlabel('Layer', fontsize=13)
    ax.set_ylabel('Activation Std (log)', fontsize=13)
    ax.set_title('Signal Propagation (Log Scale)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Signal Propagation Through {len(stds_ortho)-1} Layers — {GENOME.title()} KAN', fontsize=15)
    plt.tight_layout()
    savefig(fig, '11_signal_propagation.png')

    # --- Fig 12: Design insights ---
    print("Generating 12_design_insights.png...")
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.axis('off')

    table_data = [
        ['Design Choice', 'Effect', 'When to Use'],
        ['Orthogonal base_weight', 'Preserves signal norm\nthrough deep nets', 'Deep networks (>10 layers)\nwith residual paths'],
        ['Residual base + spline', 'Prevents signal collapse,\nspline adds flexibility', 'Any learnable-activation\narchitecture'],
        ['Sigmoid grid normalization', 'Maps inputs to valid\ngrid domain [0,1]', 'Spline-based networks\nwith fixed grids'],
        ['Gradient normalization\n(grad / ||grad||)', 'Decouples lr from\ngradient magnitude', 'Networks with varying\ngradient scales'],
        ['Exclude base_weight\nfrom ES perturbation', 'Preserves orthogonality\nduring evolution', 'Hybrid evolutionary +\ngradient methods'],
        ['Antithetic sampling\n(+eps and -eps)', 'Halves ES gradient\nvariance', 'Any evolution strategy\nestimator'],
        ['Fresh optimizer\nper ES generation', 'Prevents stale Adam\nmomentum/variance', 'Memetic algorithms\nwith optimizer resets'],
        ['Piecewise linear splines\n(20 knots)', 'Simple, fast, fully\ndifferentiable', 'When spline complexity\nis not the bottleneck'],
    ]

    table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                      loc='center', cellLoc='left')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.2)

    for j in range(3):
        cell = table[0, j]
        cell.set_facecolor('#2c3e50')
        cell.set_text_props(color='white', fontweight='bold')

    for i in range(1, len(table_data)):
        color = '#f8f9fa' if i % 2 == 0 else '#ffffff'
        for j in range(3):
            table[i, j].set_facecolor(color)

    ax.set_title('Transferable Design Insights from KAN-CPPN', fontsize=16, fontweight='bold', pad=30)
    savefig(fig, '12_design_insights.png')


if __name__ == '__main__':
    fig_09()
    fig_10_11_12()
    print("\nAll remaining figures generated!")
