"""
Phase 3: Train SwarmKAN-CPPNs and compare with vanilla KAN-CPPNs.

Trains both a SwarmKAN_CPPN and a vanilla KAN_CPPN on the same target image,
then compares loss curves, feature maps, and weight sweeps.

Usage:
    python experiments/phase3_swarm_kan.py [--genome skull|butterfly|apple] [--n_iters 10000] [--output_dir output/phase3]
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src import CPPN, FlattenCPPNParameters, KAN_CPPN, FlattenKANParameters
from src import SwarmKAN_CPPN
from src import load_genome, train_sgd, train_swarm
from src import viz_feature_maps, sweep_weight, plot_sweep_grid
from src import discover_interesting_kan_sweeps

# Architecture configs
GENOME_CONFIGS = {
    'skull':     {'n_layers': 12, 'hidden_size': 22},
    'butterfly': {'n_layers': 16, 'hidden_size': 23},
    'apple':     {'n_layers': 33, 'hidden_size': 38},
}


def run_genome(genome, n_iters, lr, grid_size, n_particles, swarm_interval,
               img_size, output_dir, device='cpu'):
    """Train SwarmKAN vs vanilla KAN on one genome."""
    genome_dir = os.path.join(output_dir, genome)
    os.makedirs(genome_dir, exist_ok=True)

    config = GENOME_CONFIGS[genome]
    print(f"\n{'='*60}")
    print(f"  Phase 3: SwarmKAN vs KAN for {genome}")
    print(f"  n_layers={config['n_layers']}, hidden_size={config['hidden_size']}")
    print(f"  n_particles={n_particles}, swarm_interval={swarm_interval}")
    print(f"{'='*60}")

    # --- Load target image ---
    print(f"  Device: {device}")
    print("  Loading picbreeder target...")
    arch, params = load_genome('picbreeder', genome)
    cppn = CPPN(arch)
    cppn_flat = FlattenCPPNParameters(cppn)
    cppn_flat.load_jax_flat_params(params)
    target_img = cppn.generate_image(img_size=img_size)
    print(f"  Target image shape: {target_img.shape}")

    # --- Train vanilla KAN-CPPN ---
    print(f"\n  Training vanilla KAN-CPPN ({n_iters} iters)...")
    kan = KAN_CPPN(
        n_layers=config['n_layers'],
        hidden_size=config['hidden_size'],
        n_inputs=4,
        grid_size=grid_size,
    ).to(device)
    losses_kan, kan = train_sgd(
        kan, target_img.detach(), lr=lr, n_iters=n_iters,
        log_interval=max(1, n_iters // 10),
    )

    # --- Train SwarmKAN-CPPN ---
    print(f"\n  Training SwarmKAN-CPPN ({n_iters} iters)...")
    swarm_kan = SwarmKAN_CPPN(
        n_layers=config['n_layers'],
        hidden_size=config['hidden_size'],
        n_inputs=4,
        grid_size=grid_size,
        n_particles=n_particles,
    ).to(device)
    losses_swarm, swarm_kan = train_swarm(
        swarm_kan, target_img.detach(), lr=lr, n_iters=n_iters,
        swarm_interval=swarm_interval,
        log_interval=max(1, n_iters // 10),
    )

    # --- Generate images ---
    print("  Generating trained images...")
    with torch.no_grad():
        kan_img, features_kan = kan.generate_image(img_size=img_size, return_features=True)
        swarm_img, features_swarm = swarm_kan.generate_image(img_size=img_size, return_features=True)

    # --- Loss curves comparison ---
    print("  Plotting loss curves...")
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    ax.plot(losses_kan, label=f"Vanilla KAN (final={losses_kan[-1]:.6f})", linewidth=0.5, alpha=0.8)
    ax.plot(losses_swarm, label=f"SwarmKAN (final={losses_swarm[-1]:.6f})", linewidth=0.5, alpha=0.8)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("MSE Loss")
    ax.set_title(f"Training Loss Comparison ({genome})")
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(genome_dir, "loss_comparison.png"), bbox_inches='tight')
    plt.close(fig)

    # --- Side-by-side comparison ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=150)
    axes[0].imshow(target_img.cpu().numpy())
    axes[0].set_title("Picbreeder (target)", fontsize=14)
    axes[0].axis('off')

    axes[1].imshow(kan_img.cpu().numpy())
    axes[1].set_title(f"Vanilla KAN (MSE={losses_kan[-1]:.6f})", fontsize=14)
    axes[1].axis('off')

    axes[2].imshow(swarm_img.cpu().numpy())
    axes[2].set_title(f"SwarmKAN (MSE={losses_swarm[-1]:.6f})", fontsize=14)
    axes[2].axis('off')

    fig.suptitle(f"{genome} Comparison", fontsize=16)
    fig.savefig(os.path.join(genome_dir, "comparison.png"), bbox_inches='tight')
    plt.close(fig)

    # --- Feature maps ---
    print("  Generating feature maps...")
    fig = viz_feature_maps(features_kan, title=f"Vanilla KAN {genome}")
    fig.savefig(os.path.join(genome_dir, "feature_maps_kan.png"), bbox_inches='tight')
    plt.close(fig)

    fig = viz_feature_maps(features_swarm, title=f"SwarmKAN {genome}")
    fig.savefig(os.path.join(genome_dir, "feature_maps_swarm.png"), bbox_inches='tight')
    plt.close(fig)

    # --- Weight sweeps ---
    print("  Discovering interesting weight sweeps...")
    for model_name, model in [("kan", kan), ("swarm", swarm_kan)]:
        flat = FlattenKANParameters(model)
        params_flat = flat.flatten()

        interesting = discover_interesting_kan_sweeps(
            model, flat, target_img.detach(), img_size=64, top_k=4,
        )
        for entry in interesting:
            print(f"    {model_name}: {entry['description']} (impact={entry['visual_impact']:.4f})")

        sweep_data = []
        for entry in interesting:
            try:
                imgs = sweep_weight(
                    params_flat, weight_id=entry['flat_idx'], cppn_flat=flat,
                    img_size=img_size, r=1, n=5,
                )
                sweep_data.append({
                    'imgs': imgs,
                    'weight_id': entry['flat_idx'],
                    'description': entry['description'],
                })
            except Exception as e:
                print(f"    Warning: {model_name} sweep {entry['description']} failed: {e}")

        if sweep_data:
            fig = plot_sweep_grid(sweep_data, title=f"{model_name.upper()} {genome} Weight Sweeps")
            fig.savefig(os.path.join(genome_dir, f"sweep_grid_{model_name}.png"), bbox_inches='tight')
            plt.close(fig)

    # Save loss data
    np.save(os.path.join(genome_dir, "losses_kan.npy"), np.array(losses_kan))
    np.save(os.path.join(genome_dir, "losses_swarm.npy"), np.array(losses_swarm))

    print(f"  Done with {genome}.")
    print(f"    Vanilla KAN final MSE: {losses_kan[-1]:.6f}")
    print(f"    SwarmKAN final MSE:     {losses_swarm[-1]:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Phase 3: SwarmKAN vs KAN")
    parser.add_argument('--genome', type=str, default=None,
                        choices=['skull', 'butterfly', 'apple'],
                        help="Run only this genome (default: all)")
    parser.add_argument('--n_iters', type=int, default=10000,
                        help="Training iterations (default: 10000)")
    parser.add_argument('--lr', type=float, default=3e-3,
                        help="Learning rate (default: 3e-3)")
    parser.add_argument('--grid_size', type=int, default=20,
                        help="KAN grid size (default: 20)")
    parser.add_argument('--n_particles', type=int, default=5,
                        help="PSO particles (default: 5)")
    parser.add_argument('--swarm_interval', type=int, default=5,
                        help="SGD steps between swarm updates (default: 5)")
    parser.add_argument('--img_size', type=int, default=128,
                        help="Training image size (default: 128)")
    parser.add_argument('--output_dir', type=str, default='output/phase3',
                        help="Output directory (default: output/phase3)")
    parser.add_argument('--device', type=str, default=None,
                        help="Device to use (default: auto-detect cuda/cpu)")
    args = parser.parse_args()

    # Auto-detect device
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    os.makedirs(args.output_dir, exist_ok=True)

    genomes = [args.genome] if args.genome else ['skull', 'butterfly', 'apple']

    print("Phase 3: SwarmKAN vs KAN Comparison")
    print(f"Genomes: {genomes}")
    print(f"Iterations: {args.n_iters}, LR: {args.lr}")
    print(f"Particles: {args.n_particles}, Swarm interval: {args.swarm_interval}")
    print(f"Device: {device}")
    print(f"Output: {args.output_dir}")

    for genome in genomes:
        try:
            run_genome(genome, args.n_iters, args.lr, args.grid_size,
                       args.n_particles, args.swarm_interval,
                       args.img_size, args.output_dir, device=device)
        except Exception as e:
            print(f"\nERROR processing {genome}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nPhase 3 complete. Results saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
