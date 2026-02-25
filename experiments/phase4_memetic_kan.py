"""
Phase 4 + 4.1: NES-Memetic KAN evolution and weight reset experiment.

Phase 4: Optimizes a KAN-CPPN with NES (Natural Evolution Strategy) + SGD
to reproduce a picbreeder image. Uses antithetic sampling for variance-reduced
gradient estimation, combined with local SGD refinement.

Phase 4.1: Creates "mutated" target images by sweeping picbreeder CPPN weights,
trains a MemeticKAN on the mutated image, then resets KAN weights to default
to see if the original image can be recovered from the learned spline shapes.

Usage:
    python experiments/phase4_memetic_kan.py [--genome skull] [--n_generations 50] [--pop_size 20] [--output_dir output/phase4]
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
from src import MemeticKAN_CPPN
from src import load_genome, train_memetic
from src import viz_feature_maps, sweep_weight, plot_sweep_grid
from src import discover_interesting_kan_sweeps
from src import sweep_all_edges, save_sweep_pages
from src import build_kan_graph_data, render_pruned_graph, render_full_graph_by_layer

# Architecture configs
GENOME_CONFIGS = {
    'skull':     {'n_layers': 12, 'hidden_size': 22},
    'butterfly': {'n_layers': 16, 'hidden_size': 23},
    'apple':     {'n_layers': 33, 'hidden_size': 38},
}

# Weight IDs for creating mutated images (Phase 4.1)
MUTATION_WEIGHTS = {
    'skull': {
        'weight_id': 37,
        'description': "Jaw Width",
        'delta': 0.8,  # Amount to shift weight
    },
    'butterfly': {
        'weight_id': 1949,
        'description': "Wing Area",
        'delta': 0.8,
    },
    'apple': {
        'weight_id': 4140,
        'description': "Apple Size",
        'delta': 0.8,
    },
}


def run_phase4(genome, n_generations, pop_size, sgd_steps, lr, grid_size,
               img_size, sigma, lr_es, output_dir,
               checkpoint_interval=100, resume_from=None):
    """Phase 4: NES-Memetic KAN training."""
    genome_dir = os.path.join(output_dir, genome)
    os.makedirs(genome_dir, exist_ok=True)

    config = GENOME_CONFIGS[genome]
    print(f"\n{'='*60}")
    print(f"  Phase 4: NES-Memetic KAN for {genome}")
    print(f"  pop_size={pop_size}, generations={n_generations}")
    print(f"  sgd_steps/gen={sgd_steps}, lr={lr}")
    print(f"  sigma={sigma}, lr_es={lr_es}")
    print(f"{'='*60}")

    # --- Load target image ---
    print("  Loading picbreeder target...")
    arch, params = load_genome('picbreeder', genome)
    cppn = CPPN(arch)
    cppn_flat = FlattenCPPNParameters(cppn)
    cppn_flat.load_jax_flat_params(params)
    target_img = cppn.generate_image(img_size=img_size)

    # Save target
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    ax.imshow(target_img.detach().cpu().numpy())
    ax.set_title(f"Picbreeder {genome} (target)", fontsize=14)
    ax.axis('off')
    fig.savefig(os.path.join(genome_dir, "target.png"), bbox_inches='tight')
    plt.close(fig)

    # --- Create and train NES-MemeticKAN ---
    print("  Creating NES-MemeticKAN...")
    memetic = MemeticKAN_CPPN(
        pop_size=pop_size,
        n_layers=config['n_layers'],
        hidden_size=config['hidden_size'],
        n_inputs=4,
        grid_size=grid_size,
        sigma=sigma,
        lr_es=lr_es,
    )

    ckpt_dir = os.path.join(genome_dir, "checkpoints")
    print(f"  Evolving for {n_generations} generations...")
    fitness_history, best = train_memetic(
        memetic, target_img.detach(),
        n_generations=n_generations,
        sgd_steps_per_gen=sgd_steps,
        lr=lr,
        log_interval=max(1, n_generations // 10),
        checkpoint_dir=ckpt_dir,
        checkpoint_interval=checkpoint_interval,
        resume_from=resume_from,
    )

    # --- Generate best individual's image ---
    print("  Generating best individual's image...")
    with torch.no_grad():
        best_img, features_best = best.generate_image(img_size=img_size, return_features=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=150)
    axes[0].imshow(target_img.detach().cpu().numpy())
    axes[0].set_title("Picbreeder (target)", fontsize=14)
    axes[0].axis('off')

    axes[1].imshow(best_img.detach().cpu().numpy())
    final_fitness = fitness_history[-1]
    axes[1].set_title(f"MemeticKAN (MSE={final_fitness:.6f})", fontsize=14)
    axes[1].axis('off')

    fig.suptitle(f"Memetic KAN {genome}", fontsize=16)
    fig.savefig(os.path.join(genome_dir, "comparison.png"), bbox_inches='tight')
    plt.close(fig)

    # --- Fitness curve ---
    print("  Plotting fitness curve...")
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    ax.plot(fitness_history, linewidth=1)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Fitness (MSE)")
    ax.set_title(f"Memetic KAN Fitness ({genome})")
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(genome_dir, "fitness_curve.png"), bbox_inches='tight')
    plt.close(fig)

    # --- Feature maps ---
    print("  Generating feature maps...")
    fig = viz_feature_maps(features_best, title=f"MemeticKAN {genome}")
    fig.savefig(os.path.join(genome_dir, "feature_maps.png"), bbox_inches='tight')
    plt.close(fig)

    # --- Weight sweeps ---
    print("  Discovering interesting weight sweeps...")
    best_flat = FlattenKANParameters(best)
    best_params = best_flat.flatten()

    interesting = discover_interesting_kan_sweeps(
        best, best_flat, target_img.detach(), img_size=64, top_k=4,
    )
    for entry in interesting:
        print(f"    {entry['description']} (impact={entry['visual_impact']:.4f})")

    sweep_data = []
    for entry in interesting:
        try:
            imgs = sweep_weight(
                best_params, weight_id=entry['flat_idx'], cppn_flat=best_flat,
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
        fig = plot_sweep_grid(sweep_data, title=f"MemeticKAN {genome} Weight Sweeps")
        fig.savefig(os.path.join(genome_dir, "sweep_grid.png"), bbox_inches='tight')
        plt.close(fig)

    # --- Exhaustive per-edge weight sweeps ---
    print("  Generating exhaustive edge sweeps...")
    sweep_dir = os.path.join(genome_dir, "sweeps")
    sweep_results = sweep_all_edges(best, best_flat, img_size=64, n_sweep=5)
    save_sweep_pages(sweep_results, sweep_dir, title_prefix=f"Memetic {genome}")

    # --- Architecture graph ---
    print("  Generating architecture graph...")
    graph_dir = os.path.join(genome_dir, "graph")
    os.makedirs(graph_dir, exist_ok=True)
    graph_data = build_kan_graph_data(best)
    render_pruned_graph(graph_data, os.path.join(graph_dir, "pruned.png"),
                       title=f"MemeticKAN {genome} (pruned)")
    render_full_graph_by_layer(graph_data, graph_dir, title_prefix=f"Memetic {genome}")

    # Save fitness data
    np.save(os.path.join(genome_dir, "fitness_history.npy"), np.array(fitness_history))

    print(f"  Phase 4 done. Final MSE: {final_fitness:.6f}")
    return memetic, best


def run_phase4_1(genome, memetic, n_generations, sgd_steps, lr, grid_size,
                 img_size, sigma, lr_es, output_dir):
    """Phase 4.1: Weight reset experiment.

    Creates a mutated target by sweeping a picbreeder weight, trains a memetic
    KAN on it, then resets the weights to see if the original can be recovered.
    """
    genome_dir = os.path.join(output_dir, genome, "phase4_1")
    os.makedirs(genome_dir, exist_ok=True)

    config = GENOME_CONFIGS[genome]
    mutation_config = MUTATION_WEIGHTS[genome]

    print(f"\n{'='*60}")
    print(f"  Phase 4.1: Weight Reset Experiment for {genome}")
    print(f"  Mutation weight: {mutation_config['weight_id']} ({mutation_config['description']})")
    print(f"  Delta: {mutation_config['delta']}")
    print(f"{'='*60}")

    # --- Load original picbreeder image ---
    arch, params = load_genome('picbreeder', genome)
    cppn = CPPN(arch)
    cppn_flat = FlattenCPPNParameters(cppn)
    cppn_flat.load_jax_flat_params(params)
    original_img = cppn.generate_image(img_size=img_size)

    # --- Create mutated image by sweeping weight ---
    print("  Creating mutated target image...")
    w_id = mutation_config['weight_id']
    delta = mutation_config['delta']
    mutated_params = params.clone()
    mutated_params[w_id] = params[w_id] + delta
    mutated_img = cppn_flat.generate_image(mutated_params, img_size=img_size)

    # --- Train MemeticKAN on mutated image ---
    print(f"  Training MemeticKAN on mutated image ({n_generations} gens)...")
    memetic_mut = MemeticKAN_CPPN(
        pop_size=memetic.pop_size,
        n_layers=config['n_layers'],
        hidden_size=config['hidden_size'],
        n_inputs=4,
        grid_size=grid_size,
        sigma=sigma,
        lr_es=lr_es,
    )

    fitness_history, best_mut = train_memetic(
        memetic_mut, mutated_img.detach(),
        n_generations=n_generations,
        sgd_steps_per_gen=sgd_steps,
        lr=lr,
        log_interval=max(1, n_generations // 10),
    )

    # Generate reconstructed mutated image
    with torch.no_grad():
        reconstructed_img = best_mut.generate_image(img_size=img_size)

    # --- Reset weights and see what comes out ---
    print("  Resetting weights to default...")
    reset_individual = memetic_mut.reset_weights(best_mut, default_value=1.0)

    with torch.no_grad():
        reset_img = reset_individual.generate_image(img_size=img_size)

    # --- Compute MSEs ---
    mse_reconstruction = torch.mean((reconstructed_img - mutated_img) ** 2).item()
    mse_reset_vs_original = torch.mean((reset_img - original_img) ** 2).item()
    mse_reset_vs_mutated = torch.mean((reset_img - mutated_img) ** 2).item()

    # --- Plot comparison ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 12), dpi=150)

    axes[0, 0].imshow(original_img.detach().cpu().numpy())
    axes[0, 0].set_title("Original Picbreeder", fontsize=14)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(mutated_img.detach().cpu().numpy())
    axes[0, 1].set_title(f"Mutated (w{w_id} + {delta})", fontsize=14)
    axes[0, 1].axis('off')

    axes[1, 0].imshow(reconstructed_img.detach().cpu().numpy())
    axes[1, 0].set_title(f"KAN Reconstruction\n(MSE={mse_reconstruction:.6f})", fontsize=14)
    axes[1, 0].axis('off')

    axes[1, 1].imshow(reset_img.detach().cpu().numpy())
    axes[1, 1].set_title(f"KAN Reset (weights=1.0)\nvs orig MSE={mse_reset_vs_original:.4f}", fontsize=14)
    axes[1, 1].axis('off')

    fig.suptitle(
        f"Phase 4.1: Weight Reset ({genome}, {mutation_config['description']})",
        fontsize=16,
    )
    fig.savefig(os.path.join(genome_dir, "weight_reset_comparison.png"), bbox_inches='tight')
    plt.close(fig)

    # --- Fitness curve ---
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    ax.plot(fitness_history, linewidth=1)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Fitness (MSE)")
    ax.set_title(f"Phase 4.1 Fitness ({genome} - mutated target)")
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(genome_dir, "fitness_curve.png"), bbox_inches='tight')
    plt.close(fig)

    print(f"  Phase 4.1 done.")
    print(f"    Reconstruction MSE: {mse_reconstruction:.6f}")
    print(f"    Reset vs original MSE: {mse_reset_vs_original:.6f}")
    print(f"    Reset vs mutated MSE: {mse_reset_vs_mutated:.6f}")

    # Save metrics
    np.save(os.path.join(genome_dir, "fitness_history.npy"), np.array(fitness_history))
    metrics = {
        'mse_reconstruction': mse_reconstruction,
        'mse_reset_vs_original': mse_reset_vs_original,
        'mse_reset_vs_mutated': mse_reset_vs_mutated,
        'mutation_weight_id': w_id,
        'mutation_delta': delta,
    }
    np.save(os.path.join(genome_dir, "metrics.npy"), metrics)


def main():
    parser = argparse.ArgumentParser(description="Phase 4: NES-Memetic KAN + Weight Reset")
    parser.add_argument('--genome', type=str, default='skull',
                        choices=['skull', 'butterfly', 'apple'],
                        help="Genome to use (default: skull)")
    parser.add_argument('--n_generations', type=int, default=50,
                        help="Evolutionary generations (default: 50)")
    parser.add_argument('--pop_size', type=int, default=20,
                        help="Perturbation pairs per generation (default: 20)")
    parser.add_argument('--sgd_steps', type=int, default=50,
                        help="SGD steps per generation (default: 50)")
    parser.add_argument('--lr', type=float, default=3e-3,
                        help="Learning rate for SGD (default: 3e-3)")
    parser.add_argument('--sigma', type=float, default=0.02,
                        help="ES perturbation scale (default: 0.02)")
    parser.add_argument('--lr_es', type=float, default=0.01,
                        help="ES gradient learning rate (default: 0.01)")
    parser.add_argument('--grid_size', type=int, default=20,
                        help="KAN grid size (default: 20)")
    parser.add_argument('--img_size', type=int, default=128,
                        help="Image size (default: 128)")
    parser.add_argument('--output_dir', type=str, default='output/phase4',
                        help="Output directory (default: output/phase4)")
    parser.add_argument('--skip_4_1', action='store_true',
                        help="Skip Phase 4.1 weight reset experiment")
    parser.add_argument('--checkpoint_interval', type=int, default=100,
                        help="Checkpoint every N iterations (default: 100)")
    parser.add_argument('--resume_from', type=str, default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Phase 4: NES-Memetic KAN Evolution")
    print(f"Genome: {args.genome}")
    print(f"Pop size: {args.pop_size}, Generations: {args.n_generations}")
    print(f"SGD steps/gen: {args.sgd_steps}, LR: {args.lr}")
    print(f"Sigma: {args.sigma}, LR_ES: {args.lr_es}")
    print(f"Output: {args.output_dir}")

    try:
        memetic, best = run_phase4(
            args.genome, args.n_generations, args.pop_size,
            args.sgd_steps, args.lr, args.grid_size,
            args.img_size, args.sigma, args.lr_es, args.output_dir,
            checkpoint_interval=args.checkpoint_interval,
            resume_from=args.resume_from,
        )

        if not args.skip_4_1:
            run_phase4_1(
                args.genome, memetic, args.n_generations,
                args.sgd_steps, args.lr, args.grid_size,
                args.img_size, args.sigma, args.lr_es, args.output_dir,
            )
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

    print(f"\nPhase 4 complete. Results saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
