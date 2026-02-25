"""Generate all analysis figures and save to figures/.

Produces:
  01_convergence_curves.png    - Loss curves for all 4 methods
  02_final_mse_bars.png        - Final MSE bar chart with references
  03_wall_clock_time.png       - Training time comparison
  04_image_comparison.png      - Side-by-side output images with MSE/SSIM
  05_spline_vs_known.png       - Top 12 learned splines overlaid with best-fit known functions
  06_function_heatmap.png      - Which function each layer learned (edge count heatmap)
  07_spline_sweeps.png         - Most impactful splines: shape, fit+residual, image sweep
  08_1d_signals.png            - 1D sequence prototype (sine, square, sawtooth, etc.)
  09_algorithm_viz.png         - PSO particle trajectories + NES antithetic sampling
  10_ablation_convergence.png  - Orthogonal vs kaiming vs no-residual training curves
  11_signal_propagation.png    - Activation std through layers
  12_design_insights.png       - Summary table of transferable insights
"""

import sys
import os
import time
import traceback
import warnings

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(__file__))

from analysis.benchmark import (
    load_target_image, GENOME_CONFIGS,
    benchmark_mlp_sgd, benchmark_kan_sgd, benchmark_swarm_kan,
    benchmark_memetic_kan, get_pretrained_reference,
)
from analysis.comparison import mse, ssim
from analysis.spline_inspector import extract_spline_curve, fit_known_function, analyze_all_edges
from analysis.text_prototype import SequenceKAN, train_sequence_kan, make_test_signals
from src.kan import KAN_CPPN, FlattenKANParameters
from src.train import train_sgd
from src.visualize import discover_interesting_kan_sweeps, sweep_weight, get_kan_param_info
from collections import Counter

warnings.filterwarnings('ignore', category=UserWarning)

GENOME = 'skull'
IMG_SIZE = 64
FIG_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 11


def savefig(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: figures/{name}")


# =====================================================================
# SECTION 1: BENCHMARK
# =====================================================================
def run_benchmark():
    print("\n[1/6] Running benchmark (4 methods x 2 seeds)...")

    N_ITERS = 1000
    N_SEEDS = 2
    N_GENS = 20
    SGD_PER_GEN = max(1, N_ITERS // N_GENS)

    target_img = load_target_image(GENOME, img_size=IMG_SIZE)
    target_256 = load_target_image(GENOME, img_size=256)

    results = {
        'mlp_sgd': [], 'kan_sgd': [], 'swarm_kan': [], 'memetic_kan': [],
    }

    for seed in range(N_SEEDS):
        print(f"  Seed {seed}:", end=" ", flush=True)

        print("MLP", end=" ", flush=True)
        results['mlp_sgd'].append(
            benchmark_mlp_sgd(GENOME, target_img, n_iters=N_ITERS, seed=seed))
        print("KAN", end=" ", flush=True)
        results['kan_sgd'].append(
            benchmark_kan_sgd(GENOME, target_img, n_iters=N_ITERS, seed=seed))
        print("Swarm", end=" ", flush=True)
        results['swarm_kan'].append(
            benchmark_swarm_kan(GENOME, target_img, n_iters=N_ITERS, seed=seed))
        print("Memetic", end=" ", flush=True)
        results['memetic_kan'].append(
            benchmark_memetic_kan(GENOME, target_img,
                                   n_generations=N_GENS,
                                   sgd_steps_per_gen=SGD_PER_GEN, seed=seed))
        print("done")

    results['picbreeder'] = [get_pretrained_reference(GENOME, 'picbreeder')]
    results['sgd_pretrained'] = [get_pretrained_reference(GENOME, 'sgd')]

    return results, target_img, target_256


def fig_01_convergence(results):
    """Convergence curves with confidence bands."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {'mlp_sgd': '#1f77b4', 'kan_sgd': '#ff7f0e', 'swarm_kan': '#2ca02c', 'memetic_kan': '#d62728'}
    labels = {'mlp_sgd': 'MLP+SGD', 'kan_sgd': 'KAN+SGD', 'swarm_kan': 'SwarmKAN', 'memetic_kan': 'MemeticKAN'}

    for method, runs in results.items():
        if method in ('picbreeder', 'sgd_pretrained'):
            continue
        all_losses = np.array([r['losses'] for r in runs])
        mean_loss = all_losses.mean(axis=0)
        std_loss = all_losses.std(axis=0)

        if method == 'memetic_kan':
            total_iters = runs[0]['total_iters']
            x = np.linspace(0, total_iters, len(mean_loss))
        else:
            x = np.arange(len(mean_loss))

        ax.plot(x, mean_loss, color=colors[method], label=labels[method], linewidth=2)
        ax.fill_between(x, mean_loss - std_loss, mean_loss + std_loss, alpha=0.15, color=colors[method])

    ax.set_yscale('log')
    ax.set_xlabel('Iteration', fontsize=13)
    ax.set_ylabel('MSE (log scale)', fontsize=13)
    ax.set_title(f'Convergence Curves — {GENOME.title()} (64x64)', fontsize=15)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    savefig(fig, '01_convergence_curves.png')


def fig_02_final_mse(results, target_img, target_256):
    """Final MSE bar chart with pre-trained references."""
    fig, ax = plt.subplots(figsize=(10, 6))
    labels_map = {'mlp_sgd': 'MLP+SGD', 'kan_sgd': 'KAN+SGD', 'swarm_kan': 'SwarmKAN', 'memetic_kan': 'MemeticKAN'}
    method_names = []
    final_mses = []
    final_stds = []
    bar_colors = []

    for method in ['mlp_sgd', 'kan_sgd', 'swarm_kan', 'memetic_kan']:
        method_names.append(labels_map[method])
        finals = [r['losses'][-1] for r in results[method]]
        final_mses.append(np.mean(finals))
        final_stds.append(np.std(finals))

    bar_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for ref_name, ref_label in [('picbreeder', 'Picbreeder'), ('sgd_pretrained', 'SGD-pretrained')]:
        ref_img = results[ref_name][0]['final_img']
        ref_mse = mse(ref_img, target_256)
        method_names.append(ref_label)
        final_mses.append(ref_mse)
        final_stds.append(0)
        bar_colors.append('#9467bd' if ref_name == 'picbreeder' else '#8c564b')

    bars = ax.bar(method_names, final_mses, yerr=final_stds, capsize=5, color=bar_colors, edgecolor='black', linewidth=0.5)

    # Add value labels on bars
    for bar, val in zip(bars, final_mses):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:.4f}',
                ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Final MSE', fontsize=13)
    ax.set_title(f'Final Image Quality — {GENOME.title()}', fontsize=15)
    ax.tick_params(axis='x', rotation=20)
    ax.grid(axis='y', alpha=0.3)
    savefig(fig, '02_final_mse_bars.png')


def fig_03_wall_clock(results):
    """Wall-clock training time."""
    fig, ax = plt.subplots(figsize=(8, 5))
    labels_map = {'mlp_sgd': 'MLP+SGD', 'kan_sgd': 'KAN+SGD', 'swarm_kan': 'SwarmKAN', 'memetic_kan': 'MemeticKAN'}
    names = []
    times = []
    stds = []
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for method in ['mlp_sgd', 'kan_sgd', 'swarm_kan', 'memetic_kan']:
        names.append(labels_map[method])
        ts = [r['wall_time'] for r in results[method]]
        times.append(np.mean(ts))
        stds.append(np.std(ts))

    bars = ax.bar(names, times, yerr=stds, capsize=5, color=colors, edgecolor='black', linewidth=0.5)
    for bar, val in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:.0f}s',
                ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('Wall-clock Time (seconds)', fontsize=13)
    ax.set_title(f'Training Time — {GENOME.title()} (1000 iters, 64x64)', fontsize=15)
    ax.grid(axis='y', alpha=0.3)
    savefig(fig, '03_wall_clock_time.png')


def fig_04_image_comparison(results, target_img, target_256):
    """Side-by-side image comparison of all methods."""
    # Generate 256px images from trained models for visual comparison
    imgs = [('Target', target_256)]
    for method, label in [('mlp_sgd', 'MLP+SGD'), ('kan_sgd', 'KAN+SGD'),
                           ('swarm_kan', 'SwarmKAN'), ('memetic_kan', 'MemeticKAN'),
                           ('picbreeder', 'Picbreeder')]:
        model = results[method][0]['model']
        if method in ('picbreeder', 'sgd_pretrained'):
            img = results[method][0]['final_img']
        else:
            with torch.no_grad():
                img = model.generate_image(img_size=256)
        imgs.append((label, img))

    fig, axes = plt.subplots(1, len(imgs), figsize=(4 * len(imgs), 5))
    for i, (label, img) in enumerate(imgs):
        ax = axes[i]
        img_np = img.detach().cpu().numpy() if isinstance(img, torch.Tensor) else img
        ax.imshow(img_np.clip(0, 1))
        ax.set_title(label, fontsize=12, fontweight='bold')
        if i > 0:
            s = ssim(target_256, img)
            m = mse(target_256, img)
            ax.set_xlabel(f'MSE={m:.4f}  SSIM={s:.3f}', fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    plt.suptitle(f'Image Comparison — {GENOME.title()} (256x256 render)', fontsize=16)
    plt.tight_layout()
    savefig(fig, '04_image_comparison.png')


# =====================================================================
# SECTION 2: SPLINE FUNCTION ANALYSIS
# =====================================================================
def fig_05_spline_vs_known(results):
    """Top 12 learned splines overlaid with best-fit known functions."""
    print("\n[2/6] Analyzing spline functions...")
    kan_model = results['kan_sgd'][0]['model']
    top_edges = analyze_all_edges(kan_model, top_k=20)

    n_show = min(12, len(top_edges))
    fig, axes = plt.subplots(3, 4, figsize=(18, 13))

    for i, edge in enumerate(top_edges[:n_show]):
        ax = axes[i // 4, i % 4]
        ax.plot(edge['raw_inputs'], edge['spline_values'], 'b-', linewidth=2.5, label='Learned spline')

        match = edge['best_match']
        if match['fitted_curve'] is not None:
            ax.plot(edge['raw_inputs'], match['fitted_curve'], 'r--', linewidth=1.5,
                    label=f"Best fit: {match['name']}")

        ax.set_title(f"L{edge['layer_idx']} [{edge['in_idx']},{edge['out_idx']}]\n"
                     f"~= {match['name']} (L2={match['l2_distance']:.3f})",
                     fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-3, 3)

    plt.suptitle(f'Learned Spline Activations vs. Known Functions — {GENOME.title()} KAN', fontsize=15)
    plt.tight_layout()
    savefig(fig, '05_spline_vs_known.png')
    return kan_model, top_edges


def fig_06_function_heatmap(kan_model):
    """Heatmap of which function type each layer learned."""
    print("  Building function heatmap (analyzing all edges)...")
    all_edges = analyze_all_edges(kan_model, top_k=999999)

    function_counts = {}
    for edge in all_edges:
        layer = edge['layer_idx']
        fn_name = edge['best_match']['name']
        key = (layer, fn_name)
        function_counts[key] = function_counts.get(key, 0) + 1

    n_layers_total = len(kan_model.layers)
    fn_names = sorted(set(e['best_match']['name'] for e in all_edges))

    heatmap = np.zeros((n_layers_total, len(fn_names)))
    for (layer, fn), count in function_counts.items():
        col = fn_names.index(fn)
        heatmap[layer, col] = count

    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(heatmap, aspect='auto', cmap='YlOrRd')
    ax.set_xticks(range(len(fn_names)))
    ax.set_xticklabels(fn_names, rotation=45, ha='right', fontsize=11)
    ax.set_ylabel('Layer', fontsize=13)
    ax.set_xlabel('Best-fit Function', fontsize=13)
    ax.set_title(f'What Each Layer Learned — {GENOME.title()} KAN\n'
                 f'(count of edges best-fit to each function type, {len(all_edges)} total edges)',
                 fontsize=14)
    plt.colorbar(im, label='Edge count')

    # Add text annotations
    for i in range(heatmap.shape[0]):
        for j in range(heatmap.shape[1]):
            val = int(heatmap[i, j])
            if val > 0:
                color = 'white' if val > heatmap.max() * 0.6 else 'black'
                ax.text(j, i, str(val), ha='center', va='center', fontsize=7, color=color)

    plt.tight_layout()
    savefig(fig, '06_function_heatmap.png')

    # Print distribution
    fn_dist = Counter(e['best_match']['name'] for e in all_edges)
    print(f"  Function distribution ({len(all_edges)} edges):")
    for fn, count in fn_dist.most_common():
        print(f"    {fn:12s}: {count:4d} ({100*count/len(all_edges):.1f}%)")

    return all_edges


# =====================================================================
# SECTION 3: SPLINE VISUAL ANALYSIS WITH IMAGE SWEEPS
# =====================================================================
def fig_07_spline_sweeps(kan_model, target_img):
    """Most impactful splines: shape, fit+residual, image sweep."""
    print("\n[3/6] Finding most impactful spline edges + generating sweeps...")
    kan_flat = FlattenKANParameters(kan_model)
    interesting = discover_interesting_kan_sweeps(kan_model, kan_flat, target_img,
                                                  n_candidates_per_group=8, top_k=6)

    n_rows = len(interesting)
    fig = plt.figure(figsize=(22, 4.5 * n_rows))
    gs = gridspec.GridSpec(n_rows, 3, width_ratios=[1, 1, 3], hspace=0.35, wspace=0.25)

    params = kan_flat.flatten()

    for row, entry in enumerate(interesting):
        flat_idx = entry['flat_idx']
        desc = entry['description']
        info = get_kan_param_info(kan_model, flat_idx)
        layer_idx = info['layer_idx']
        layer = kan_model.layers[layer_idx]
        indices = info['local_shape_indices']

        if info['param_type'] == 'coeffs' and len(indices) >= 2:
            out_idx, in_idx = indices[0], indices[1]
            raw_inputs, spline_values, _ = extract_spline_curve(layer, in_idx, out_idx)
            match = fit_known_function(raw_inputs, spline_values)

            # Panel 1: Spline shape
            ax1 = fig.add_subplot(gs[row, 0])
            ax1.plot(raw_inputs, spline_values, 'b-', linewidth=2.5)
            ax1.set_title(f'{desc}\nSpline Shape', fontsize=10, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim(-3, 3)

            # Panel 2: Best fit + residual
            ax2 = fig.add_subplot(gs[row, 1])
            if match['fitted_curve'] is not None:
                ax2.plot(raw_inputs, match['fitted_curve'], 'r-', label=f"{match['name']}", linewidth=2)
                residual = spline_values - match['fitted_curve']
                ax2.plot(raw_inputs, residual, 'g--', label='Residual', linewidth=1.5, alpha=0.7)
            ax2.set_title(f"~= {match['name']} (L2={match['l2_distance']:.3f})", fontsize=10, fontweight='bold')
            ax2.legend(fontsize=9)
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim(-3, 3)
        else:
            ax1 = fig.add_subplot(gs[row, 0])
            ax1.text(0.5, 0.5, f'{desc}\n(base_weight or\nscaling weight)', ha='center', va='center',
                     fontsize=10, transform=ax1.transAxes)
            ax1.set_title(desc, fontsize=10, fontweight='bold')
            ax2 = fig.add_subplot(gs[row, 1])
            ax2.text(0.5, 0.5, 'N/A (not a spline)', ha='center', va='center',
                     fontsize=10, transform=ax2.transAxes)

        # Panel 3: Image sweep
        ax3 = fig.add_subplot(gs[row, 2])
        with torch.no_grad():
            sweep_imgs = sweep_weight(params, flat_idx, kan_flat, img_size=128, n=7)
        sweep_np = sweep_imgs.detach().cpu().numpy()
        strip = np.concatenate(sweep_np, axis=1)
        ax3.imshow(strip.clip(0, 1))
        ax3.set_title(f'Sweeping {desc}  (w-1 ... w ... w+1)', fontsize=10)
        ax3.set_xticks([]); ax3.set_yticks([])

    plt.suptitle(f'Spline Analysis + Image Sweeps — {GENOME.title()} KAN', fontsize=16, y=1.01)
    savefig(fig, '07_spline_sweeps.png')


# =====================================================================
# SECTION 4: TEXT SCALABILITY PROTOTYPE
# =====================================================================
def fig_08_1d_signals():
    """1D sequence approximation with KAN-CPPN."""
    print("\n[4/6] Training 1D signal prototypes...")
    test_signals = make_test_signals(seq_len=200)

    n_signals = len(test_signals)
    fig, axes = plt.subplots(n_signals, 2, figsize=(16, 3.5 * n_signals))

    for i, (name, (positions, target)) in enumerate(test_signals.items()):
        model = SequenceKAN(n_layers=4, hidden_size=16, output_size=1)
        losses = train_sequence_kan(model, target, positions, n_iters=3000, lr=3e-3)
        _, learned = model.generate_signal(seq_len=200)

        ax = axes[i, 0]
        ax.plot(positions.numpy(), target[:, 0].numpy(), 'b-', label='Target', linewidth=2.5)
        ax.plot(positions.numpy(), learned[:, 0].detach().numpy(), 'r--', label='KAN output', linewidth=2)
        ax.set_title(f'{name} — Final MSE: {losses[-1]:.6f}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        ax = axes[i, 1]
        ax.plot(losses, linewidth=1.5, color='#d62728')
        ax.set_yscale('log')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('MSE')
        ax.set_title(f'{name} — Loss Curve', fontsize=11)
        ax.grid(True, alpha=0.3)

        print(f"  {name}: MSE={losses[-1]:.6f}")

    plt.suptitle('1D Signal Approximation with KAN-CPPN\n'
                 '(Can CPPNs generalize from 2D images to 1D sequences?)', fontsize=15)
    plt.tight_layout()
    savefig(fig, '08_1d_signals.png')


# =====================================================================
# SECTION 5: ALGORITHM VISUALIZATION
# =====================================================================
def fig_09_algorithm_viz():
    """PSO and NES algorithm visualizations."""
    print("\n[5/6] Generating algorithm visualizations...")
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # --- PSO ---
    ax = axes[0]
    np.random.seed(42)
    n_steps = 20
    n_particles = 5

    positions = np.random.randn(n_particles, 2) * 0.5
    velocities = np.zeros_like(positions)
    personal_best = positions.copy()
    personal_best_scores = np.full(n_particles, np.inf)
    global_best = positions[0].copy()
    global_best_score = np.inf

    def loss_2d(pos):
        return (pos[0] - 0.3)**2 + (pos[1] + 0.2)**2

    trajectories = [[] for _ in range(n_particles)]
    for step in range(n_steps):
        for i in range(n_particles):
            trajectories[i].append(positions[i].copy())
            score = loss_2d(positions[i])
            if score < personal_best_scores[i]:
                personal_best_scores[i] = score
                personal_best[i] = positions[i].copy()
            if score < global_best_score:
                global_best_score = score
                global_best = positions[i].copy()
        r1 = np.random.rand(n_particles, 2)
        r2 = np.random.rand(n_particles, 2)
        velocities = 0.7 * velocities + 1.5 * r1 * (personal_best - positions) + 1.5 * r2 * (global_best - positions)
        positions += velocities

    # Draw contours of loss landscape
    xx, yy = np.meshgrid(np.linspace(-1.5, 2, 100), np.linspace(-1.5, 1.5, 100))
    zz = (xx - 0.3)**2 + (yy + 0.2)**2
    ax.contour(xx, yy, zz, levels=15, alpha=0.3, cmap='gray')

    particle_colors = plt.cm.Set1(np.linspace(0, 1, n_particles))
    for i in range(n_particles):
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
    sigma = 0.3
    n_perturbations = 8

    # Draw contours
    ax.contour(xx, yy, zz, levels=15, alpha=0.3, cmap='gray')

    np.random.seed(123)
    gen_colors = ['#4292c6', '#2171b5', '#084594']
    for gen in range(3):
        for i in range(n_perturbations):
            eps = np.random.randn(2) * sigma
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


# =====================================================================
# SECTION 6: DESIGN CHOICE ABLATIONS
# =====================================================================
def run_ablations(target_img):
    print("\n[6/6] Running ablation studies...")
    cfg = GENOME_CONFIGS[GENOME]
    N_ITERS = 500

    # Orthogonal
    torch.manual_seed(42)
    kan_ortho = KAN_CPPN(n_layers=cfg['n_layers'], hidden_size=cfg['hidden_size'])
    print("  Training orthogonal...", end=" ", flush=True)
    losses_ortho, _ = train_sgd(kan_ortho, target_img, lr=3e-3, n_iters=N_ITERS, log_interval=0)
    print(f"MSE={losses_ortho[-1]:.6f}")

    # Kaiming
    torch.manual_seed(42)
    kan_kaiming = KAN_CPPN(n_layers=cfg['n_layers'], hidden_size=cfg['hidden_size'])
    for layer in kan_kaiming.layers:
        torch.nn.init.kaiming_uniform_(layer.base_weight)
    print("  Training kaiming...", end=" ", flush=True)
    losses_kaiming, _ = train_sgd(kan_kaiming, target_img, lr=3e-3, n_iters=N_ITERS, log_interval=0)
    print(f"MSE={losses_kaiming[-1]:.6f}")

    # No residual
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
        stds_ortho_sig = [x.std().item()]
        for layer in kan_test.layers:
            x = layer(x)
            stds_ortho_sig.append(x.std().item())

    for layer in kan_test.layers:
        torch.nn.init.kaiming_uniform_(layer.base_weight)

    with torch.no_grad():
        x = test_input
        stds_kaiming_sig = [x.std().item()]
        for layer in kan_test.layers:
            x = layer(x)
            stds_kaiming_sig.append(x.std().item())

    return losses_ortho, losses_kaiming, losses_no_res, stds_ortho_sig, stds_kaiming_sig


def fig_10_ablation_convergence(losses_ortho, losses_kaiming, losses_no_res):
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


def fig_11_signal_propagation(stds_ortho, stds_kaiming):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Linear scale
    ax = axes[0]
    ax.plot(stds_ortho, 'o-', label='Orthogonal', linewidth=2, markersize=6, color='#2ca02c')
    ax.plot(stds_kaiming, 's-', label='Kaiming', linewidth=2, markersize=6, color='#d62728')
    ax.set_xlabel('Layer', fontsize=13)
    ax.set_ylabel('Activation Std', fontsize=13)
    ax.set_title('Signal Propagation (Linear Scale)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    # Log scale
    ax = axes[1]
    ax.plot(stds_ortho, 'o-', label='Orthogonal', linewidth=2, markersize=6, color='#2ca02c')
    ax.plot(stds_kaiming, 's-', label='Kaiming', linewidth=2, markersize=6, color='#d62728')
    ax.set_yscale('log')
    ax.set_xlabel('Layer', fontsize=13)
    ax.set_ylabel('Activation Std (log)', fontsize=13)
    ax.set_title('Signal Propagation (Log Scale)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    # Add annotations
    axes[0].annotate(f'Ortho final: {stds_ortho[-1]:.3f}',
                     xy=(len(stds_ortho)-1, stds_ortho[-1]),
                     xytext=(-80, 30), textcoords='offset points',
                     arrowprops=dict(arrowstyle='->', color='#2ca02c'), fontsize=10, color='#2ca02c')
    axes[0].annotate(f'Kaiming final: {stds_kaiming[-1]:.1f}',
                     xy=(len(stds_kaiming)-1, stds_kaiming[-1]),
                     xytext=(-100, -30), textcoords='offset points',
                     arrowprops=dict(arrowstyle='->', color='#d62728'), fontsize=10, color='#d62728')

    plt.suptitle(f'Signal Propagation Through {len(stds_ortho)-1} Layers — {GENOME.title()} KAN', fontsize=15)
    plt.tight_layout()
    savefig(fig, '11_signal_propagation.png')


def fig_12_design_insights():
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

    # Style header
    for j in range(3):
        cell = table[0, j]
        cell.set_facecolor('#2c3e50')
        cell.set_text_props(color='white', fontweight='bold')

    # Alternate row colors
    for i in range(1, len(table_data)):
        color = '#f8f9fa' if i % 2 == 0 else '#ffffff'
        for j in range(3):
            table[i, j].set_facecolor(color)

    ax.set_title('Transferable Design Insights from KAN-CPPN', fontsize=16, fontweight='bold', pad=30)
    savefig(fig, '12_design_insights.png')


# =====================================================================
# MAIN
# =====================================================================
def main():
    start = time.perf_counter()

    try:
        # Section 1: Benchmark
        results, target_img, target_256 = run_benchmark()
        fig_01_convergence(results)
        fig_02_final_mse(results, target_img, target_256)
        fig_03_wall_clock(results)
        fig_04_image_comparison(results, target_img, target_256)

        # Section 2: Spline function analysis
        kan_model, top_edges = fig_05_spline_vs_known(results)
        all_edges = fig_06_function_heatmap(kan_model)

        # Section 3: Spline visual sweeps
        fig_07_spline_sweeps(kan_model, target_img)

        # Section 4: Text scalability
        fig_08_1d_signals()

        # Section 5: Algorithm visualization
        fig_09_algorithm_viz()

        # Section 6: Ablations
        l_o, l_k, l_nr, s_o, s_k = run_ablations(target_img)
        fig_10_ablation_convergence(l_o, l_k, l_nr)
        fig_11_signal_propagation(s_o, s_k)
        fig_12_design_insights()

        total = time.perf_counter() - start
        print(f"\n{'='*60}")
        print(f"ALL DONE in {total:.0f}s ({total/60:.1f}m)")
        print(f"12 figures saved to figures/")
        print(f"{'='*60}")

    except Exception as e:
        print(f"\n!!! ERROR: {e}")
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
