"""
Phase 1: Reproduce FER paper visualizations in PyTorch.

Loads all 6 pre-trained CPPN genomes (3 picbreeder + 3 SGD), generates images,
feature maps, weight sweeps (known IDs from paper), and random direction sweeps
sorted by variance. All figures saved to output/phase1/.

Usage:
    python experiments/phase1_cppn_visualization.py [--genome skull|butterfly|apple] [--img_size 256] [--output_dir output/phase1]
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

from src import CPPN, FlattenCPPNParameters, load_genome
from src import viz_feature_maps, sweep_weight, sweep_weight_random_direction
from src import plot_sweep_strip, plot_sweep_grid

# Known interesting weight IDs from the paper
PAPER_WEIGHTS = {
    'picbreeder': {
        'skull': [
            (4371, "Mouth Opening"),
            (5009, "Eye Winking"),
            (5097, "Eye Width"),
            (37, "Jaw Width"),
        ],
        'butterfly': [
            (1949, "Wing Area"),
            (3702, "Color"),
            (17, "Butterfly to Fly"),
            (133, "Vertical Shape"),
        ],
        'apple': [
            (42178, "Stem Angle"),
            (4140, "Apple Size"),
            (34459, "Cleans Background"),
            (17131, "Removes Stem"),
        ],
    },
    'sgd': {
        'skull': [
            (1587, "SGD Weight 1587"),
            (81, "SGD Weight 81"),
            (46, "SGD Weight 46"),
            (3185, "SGD Weight 3185"),
        ],
        'butterfly': [
            (5603, "SGD Weight 5603"),
            (3796, "SGD Weight 3796"),
            (3848, "SGD Weight 3848"),
            (78, "SGD Weight 78"),
        ],
        'apple': [
            (37753, "SGD Weight 37753"),
            (135, "SGD Weight 135"),
            (37721, "SGD Weight 37721"),
            (37809, "SGD Weight 37809"),
        ],
    },
}

# Special sweep parameters for apple weight 42178
SPECIAL_SWEEP_PARAMS = {
    ('picbreeder', 'apple', 42178): {'center_weight': -1.0, 'r': 2.0},
}


def run_genome(source, genome, img_size, output_dir, n_random_sweeps=200, top_k=60):
    """Run all Phase 1 experiments for a single genome.

    Args:
        source: 'picbreeder' or 'sgd'.
        genome: 'skull', 'butterfly', or 'apple'.
        img_size: Resolution for generated images.
        output_dir: Directory to save outputs.
        n_random_sweeps: Number of random direction sweeps to compute.
        top_k: Number of top-variance sweeps to plot.
    """
    prefix = f"{source}_{genome}"
    genome_dir = os.path.join(output_dir, prefix)
    os.makedirs(genome_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  {prefix}")
    print(f"{'='*60}")

    # Load genome
    print(f"  Loading genome...")
    arch, params = load_genome(source, genome)
    cppn = CPPN(arch)
    cppn_flat = FlattenCPPNParameters(cppn)
    cppn_flat.load_jax_flat_params(params)
    print(f"  Architecture: {arch}")
    print(f"  Parameters: {cppn_flat.n_params}")

    # 1. Generate image
    print(f"  Generating image ({img_size}x{img_size})...")
    img, features = cppn.generate_image(img_size=img_size, return_features=True)
    img_np = img.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    ax.imshow(img_np)
    ax.set_title(f"{prefix}", fontsize=14)
    ax.axis('off')
    fig.savefig(os.path.join(genome_dir, f"image.png"), bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved image.png")

    # 2. Feature maps
    print(f"  Generating feature maps...")
    fig = viz_feature_maps(features, title=f"{prefix} Feature Maps")
    fig.savefig(os.path.join(genome_dir, f"feature_maps.png"), bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved feature_maps.png")

    # 3. Paper weight sweeps
    weight_entries = PAPER_WEIGHTS.get(source, {}).get(genome, [])
    if weight_entries:
        print(f"  Running {len(weight_entries)} paper weight sweeps...")
        sweep_data = []
        for weight_id, description in weight_entries:
            sweep_kwargs = SPECIAL_SWEEP_PARAMS.get((source, genome, weight_id), {})
            r = sweep_kwargs.get('r', 1.0)
            center_weight = sweep_kwargs.get('center_weight', None)

            imgs = sweep_weight(
                params, weight_id=weight_id, cppn_flat=cppn_flat,
                img_size=img_size, center_weight=center_weight, r=r, n=5,
            )
            sweep_data.append({
                'imgs': imgs,
                'weight_id': weight_id,
                'description': description,
            })

            # Also save individual strip
            fig = plot_sweep_strip(imgs, title=f"{prefix} - {description} (w{weight_id})")
            fig.savefig(os.path.join(genome_dir, f"sweep_w{weight_id}.png"), bbox_inches='tight')
            plt.close(fig)

        # Grid of all paper sweeps
        fig = plot_sweep_grid(sweep_data, title=f"{prefix} Paper Weight Sweeps")
        fig.savefig(os.path.join(genome_dir, f"sweep_grid_paper.png"), bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved sweep grid and {len(weight_entries)} individual strips")

    # 4. Random direction sweeps
    print(f"  Running {n_random_sweeps} random direction sweeps...")
    random_sweeps = []
    for seed in range(n_random_sweeps):
        try:
            imgs = sweep_weight_random_direction(
                params, seed=seed, cppn_flat=cppn_flat,
                img_size=img_size, r=1, n=5,
            )
            # Compute variance across the sweep (a measure of how much this direction matters)
            variance = imgs.var(dim=0).mean().item()
            random_sweeps.append((seed, variance, imgs))
        except Exception as e:
            print(f"    Warning: seed {seed} failed: {e}")

    # Sort by variance (most interesting first)
    random_sweeps.sort(key=lambda x: x[1], reverse=True)
    actual_top_k = min(top_k, len(random_sweeps))

    print(f"  Plotting top-{actual_top_k} random sweeps by variance...")
    top_sweep_data = []
    for seed, variance, imgs in random_sweeps[:actual_top_k]:
        top_sweep_data.append({
            'imgs': imgs,
            'weight_id': f"seed={seed}",
            'description': f"var={variance:.4f}",
        })

    # Plot in batches of 10 for readability
    batch_size = 10
    for batch_idx in range(0, actual_top_k, batch_size):
        batch = top_sweep_data[batch_idx:batch_idx + batch_size]
        rank_start = batch_idx + 1
        rank_end = batch_idx + len(batch)
        fig = plot_sweep_grid(
            batch,
            title=f"{prefix} Random Sweeps (rank {rank_start}-{rank_end})",
        )
        fig.savefig(
            os.path.join(genome_dir, f"sweep_random_rank{rank_start:03d}-{rank_end:03d}.png"),
            bbox_inches='tight',
        )
        plt.close(fig)

    print(f"  Saved {actual_top_k} random sweep plots in batches of {batch_size}")
    print(f"  Done with {prefix}")


def main():
    parser = argparse.ArgumentParser(description="Phase 1: CPPN Visualization")
    parser.add_argument('--genome', type=str, default=None,
                        choices=['skull', 'butterfly', 'apple'],
                        help="Run only this genome (default: all)")
    parser.add_argument('--img_size', type=int, default=256,
                        help="Image resolution (default: 256)")
    parser.add_argument('--output_dir', type=str, default='output/phase1',
                        help="Output directory (default: output/phase1)")
    parser.add_argument('--n_random_sweeps', type=int, default=200,
                        help="Number of random direction sweeps (default: 200)")
    parser.add_argument('--top_k', type=int, default=60,
                        help="Number of top sweeps to plot (default: 60)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    genomes = [args.genome] if args.genome else ['skull', 'butterfly', 'apple']
    sources = ['picbreeder', 'sgd']

    print("Phase 1: CPPN Visualization")
    print(f"Genomes: {genomes}")
    print(f"Image size: {args.img_size}")
    print(f"Output: {args.output_dir}")

    for source in sources:
        for genome in genomes:
            try:
                run_genome(
                    source, genome, args.img_size, args.output_dir,
                    n_random_sweeps=args.n_random_sweeps,
                    top_k=args.top_k,
                )
            except Exception as e:
                print(f"\nERROR processing {source}_{genome}: {e}")
                import traceback
                traceback.print_exc()

    print(f"\nPhase 1 complete. Results saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
