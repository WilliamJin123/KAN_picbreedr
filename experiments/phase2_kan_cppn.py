"""
Phase 2: Train KAN-CPPNs on picbreeder images and analyze.

Trains a KAN_CPPN to reproduce a picbreeder image using SGD, then analyzes
the result with loss curves, feature maps, weight sweeps, and side-by-side
comparison with the original picbreeder and SGD feature maps.

Usage:
    python experiments/phase2_kan_cppn.py [--genome skull|butterfly|apple] [--n_iters 10000] [--output_dir output/phase2]
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
from src import load_genome, train_sgd
from src import viz_feature_maps, sweep_weight, plot_sweep_strip, plot_sweep_grid
from src import discover_interesting_kan_sweeps
from src import sweep_all_edges, save_sweep_pages
from src import build_kan_graph_data, render_pruned_graph, render_full_graph_by_layer

# Architecture configs matching the reference genomes
GENOME_CONFIGS = {
    'skull':     {'n_layers': 12, 'hidden_size': 22},
    'butterfly': {'n_layers': 16, 'hidden_size': 23},
    'apple':     {'n_layers': 33, 'hidden_size': 38},
}


def run_genome(genome, n_iters, lr, grid_size, img_size, output_dir,
               checkpoint_interval=100, resume_from=None, spline_degree=1):
    """Train a KAN-CPPN on one genome and analyze."""
    genome_dir = os.path.join(output_dir, genome)
    os.makedirs(genome_dir, exist_ok=True)

    config = GENOME_CONFIGS[genome]
    print(f"\n{'='*60}")
    print(f"  Phase 2: KAN-CPPN for {genome}")
    print(f"  n_layers={config['n_layers']}, hidden_size={config['hidden_size']}")
    print(f"  grid_size={grid_size}, n_iters={n_iters}, lr={lr}")
    print(f"{'='*60}")

    # --- Load target (picbreeder) image ---
    print("  Loading picbreeder target...")
    arch_pb, params_pb = load_genome('picbreeder', genome)
    cppn_pb = CPPN(arch_pb)
    cppn_flat_pb = FlattenCPPNParameters(cppn_pb)
    cppn_flat_pb.load_jax_flat_params(params_pb)
    target_img, features_pb = cppn_pb.generate_image(img_size=img_size, return_features=True)
    print(f"  Target image shape: {target_img.shape}")

    # Save target image
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    ax.imshow(target_img.detach().cpu().numpy())
    ax.set_title(f"Picbreeder {genome} (target)", fontsize=14)
    ax.axis('off')
    fig.savefig(os.path.join(genome_dir, "target.png"), bbox_inches='tight')
    plt.close(fig)

    # --- Load SGD reference for comparison ---
    print("  Loading SGD reference...")
    arch_sgd, params_sgd = load_genome('sgd', genome)
    cppn_sgd = CPPN(arch_sgd)
    cppn_flat_sgd = FlattenCPPNParameters(cppn_sgd)
    cppn_flat_sgd.load_jax_flat_params(params_sgd)
    _, features_sgd = cppn_sgd.generate_image(img_size=img_size, return_features=True)

    # --- Create and train KAN-CPPN ---
    print("  Creating KAN-CPPN...")
    kan = KAN_CPPN(
        n_layers=config['n_layers'],
        hidden_size=config['hidden_size'],
        n_inputs=4,
        grid_size=grid_size,
        spline_degree=spline_degree,
    )
    kan_flat = FlattenKANParameters(kan)
    print(f"  KAN parameters: {kan_flat.n_params}")

    ckpt_dir = os.path.join(genome_dir, "checkpoints")
    print(f"  Training KAN-CPPN for {n_iters} iterations...")
    losses, kan = train_sgd(kan, target_img.detach(), lr=lr, n_iters=n_iters,
                            log_interval=max(1, n_iters // 10),
                            checkpoint_dir=ckpt_dir,
                            checkpoint_interval=checkpoint_interval,
                            resume_from=resume_from)

    # --- Generate trained image ---
    print("  Generating trained image...")
    with torch.no_grad():
        kan_img, features_kan = kan.generate_image(img_size=img_size, return_features=True)

    # Save trained image
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    ax.imshow(kan_img.detach().cpu().numpy())
    final_mse = losses[-1]
    ax.set_title(f"KAN-CPPN {genome} (MSE={final_mse:.6f})", fontsize=14)
    ax.axis('off')
    fig.savefig(os.path.join(genome_dir, "kan_trained.png"), bbox_inches='tight')
    plt.close(fig)

    # --- Side-by-side comparison ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=150)
    axes[0].imshow(target_img.detach().cpu().numpy())
    axes[0].set_title("Picbreeder (target)", fontsize=14)
    axes[0].axis('off')

    sgd_img = cppn_sgd.generate_image(img_size=img_size)
    axes[1].imshow(sgd_img.detach().cpu().numpy())
    axes[1].set_title("SGD MLP-CPPN", fontsize=14)
    axes[1].axis('off')

    axes[2].imshow(kan_img.detach().cpu().numpy())
    axes[2].set_title(f"KAN-CPPN (MSE={final_mse:.6f})", fontsize=14)
    axes[2].axis('off')

    fig.suptitle(f"{genome} Comparison", fontsize=16)
    fig.savefig(os.path.join(genome_dir, "comparison.png"), bbox_inches='tight')
    plt.close(fig)

    # --- Loss curve ---
    print("  Plotting loss curve...")
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    ax.plot(losses, linewidth=0.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("MSE Loss")
    ax.set_title(f"KAN-CPPN Training Loss ({genome})")
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(genome_dir, "loss_curve.png"), bbox_inches='tight')
    plt.close(fig)

    # --- Feature maps ---
    print("  Generating feature maps...")
    # KAN-CPPN feature maps
    fig = viz_feature_maps(features_kan, title=f"KAN-CPPN {genome} Feature Maps")
    fig.savefig(os.path.join(genome_dir, "feature_maps_kan.png"), bbox_inches='tight')
    plt.close(fig)

    # Picbreeder feature maps (for comparison)
    fig = viz_feature_maps(features_pb, title=f"Picbreeder {genome} Feature Maps")
    fig.savefig(os.path.join(genome_dir, "feature_maps_picbreeder.png"), bbox_inches='tight')
    plt.close(fig)

    # SGD feature maps (for comparison)
    fig = viz_feature_maps(features_sgd, title=f"SGD {genome} Feature Maps")
    fig.savefig(os.path.join(genome_dir, "feature_maps_sgd.png"), bbox_inches='tight')
    plt.close(fig)

    # --- Weight sweeps on trained KAN ---
    print("  Discovering interesting KAN weight sweeps...")
    kan_params = kan_flat.flatten()

    interesting = discover_interesting_kan_sweeps(
        kan, kan_flat, target_img.detach(), img_size=64, top_k=8,
    )
    for entry in interesting:
        print(f"    {entry['description']} (impact={entry['visual_impact']:.4f})")

    sweep_data = []
    for entry in interesting:
        try:
            imgs = sweep_weight(
                kan_params, weight_id=entry['flat_idx'], cppn_flat=kan_flat,
                img_size=img_size, r=1, n=5,
            )
            sweep_data.append({
                'imgs': imgs.detach().cpu(),
                'weight_id': entry['flat_idx'],
                'description': entry['description'],
            })
        except Exception as e:
            print(f"    Warning: sweep {entry['description']} failed: {e}")

    if sweep_data:
        fig = plot_sweep_grid(sweep_data, title=f"KAN-CPPN {genome} Weight Sweeps")
        fig.savefig(os.path.join(genome_dir, "sweep_grid_kan.png"), bbox_inches='tight')
        plt.close(fig)

    # --- Exhaustive per-edge weight sweeps ---
    print("  Generating exhaustive edge sweeps...")
    sweep_dir = os.path.join(genome_dir, "sweeps")
    sweep_results = sweep_all_edges(kan, kan_flat, img_size=64, n_sweep=5)
    save_sweep_pages(sweep_results, sweep_dir, title_prefix=f"KAN {genome}")

    # --- Architecture graph ---
    print("  Generating architecture graph...")
    graph_dir = os.path.join(genome_dir, "graph")
    os.makedirs(graph_dir, exist_ok=True)
    graph_data = build_kan_graph_data(kan)
    render_pruned_graph(graph_data, os.path.join(graph_dir, "pruned.png"),
                       title=f"KAN-CPPN {genome} (pruned)")
    render_full_graph_by_layer(graph_data, graph_dir, title_prefix=f"KAN {genome}")

    # Save loss data
    np.save(os.path.join(genome_dir, "losses.npy"), np.array(losses))

    print(f"  Done with {genome}. Final MSE: {final_mse:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Phase 2: KAN-CPPN Training")
    parser.add_argument('--genome', type=str, default=None,
                        choices=['skull', 'butterfly', 'apple'],
                        help="Run only this genome (default: all)")
    parser.add_argument('--n_iters', type=int, default=10000,
                        help="Number of training iterations (default: 10000)")
    parser.add_argument('--lr', type=float, default=3e-3,
                        help="Learning rate (default: 3e-3)")
    parser.add_argument('--grid_size', type=int, default=20,
                        help="KAN grid size (default: 20)")
    parser.add_argument('--img_size', type=int, default=128,
                        help="Training image size (default: 128)")
    parser.add_argument('--output_dir', type=str, default='output/phase2',
                        help="Output directory (default: output/phase2)")
    parser.add_argument('--checkpoint_interval', type=int, default=100,
                        help="Checkpoint every N iterations (default: 100)")
    parser.add_argument('--resume_from', type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument('--spline_degree', type=int, default=1,
                        help="B-spline degree (1-4, default: 1)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    genomes = [args.genome] if args.genome else ['skull', 'butterfly', 'apple']

    print("Phase 2: KAN-CPPN Training and Analysis")
    print(f"Genomes: {genomes}")
    print(f"Iterations: {args.n_iters}, LR: {args.lr}, Grid: {args.grid_size}")
    print(f"Image size: {args.img_size}")
    print(f"Output: {args.output_dir}")

    for genome in genomes:
        try:
            run_genome(genome, args.n_iters, args.lr, args.grid_size,
                       args.img_size, args.output_dir,
                       checkpoint_interval=args.checkpoint_interval,
                       resume_from=args.resume_from,
                       spline_degree=args.spline_degree)
        except Exception as e:
            print(f"\nERROR processing {genome}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nPhase 2 complete. Results saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
