"""Benchmarking harness for comparing training methods.

Trains all 5 methods (MLP+SGD, KAN+SGD, SwarmKAN, MemeticKAN, pre-trained)
on the same target images with identical conditions. Collects timing,
loss curves, and final metrics.
"""

import time
import torch
import numpy as np

from src.cppn import CPPN, FlattenCPPNParameters
from src.kan import KAN_CPPN, FlattenKANParameters
from src.swarm_kan import SwarmKAN_CPPN
from src.memetic_kan import MemeticKAN_CPPN
from src.train import train_sgd, train_swarm, train_memetic
from src.data import load_genome


# Genome configs: (n_layers, hidden_size) matching architecture strings
GENOME_CONFIGS = {
    'skull': {'n_layers': 12, 'hidden_size': 22, 'arch': "12;cache:15,gaussian:4,identity:2,sin:1"},
    'butterfly': {'n_layers': 16, 'hidden_size': 23, 'arch': "16;cache:17,gaussian:4,sigmoid:1,sin:1"},
    'apple': {'n_layers': 33, 'hidden_size': 38, 'arch': "33;cache:28,gaussian:3,sigmoid:5,sin:2"},
}


def load_target_image(genome_name, img_size=256):
    """Load the picbreeder target image for a genome.

    Args:
        genome_name: 'skull', 'butterfly', or 'apple'
        img_size: Image resolution.

    Returns:
        target_img: Tensor (img_size, img_size, 3)
    """
    arch, params = load_genome('picbreeder', genome_name)
    cppn = CPPN(arch=arch)
    flat = FlattenCPPNParameters(cppn)
    flat.load_jax_flat_params(params)
    with torch.no_grad():
        return cppn.generate_image(img_size=img_size)


def benchmark_mlp_sgd(genome_name, target_img, n_iters=5000, lr=3e-3, seed=0):
    """Train a fresh MLP-CPPN with SGD.

    Returns:
        dict with keys: losses, wall_time, final_img, model
    """
    torch.manual_seed(seed)
    cfg = GENOME_CONFIGS[genome_name]
    cppn = CPPN(arch=cfg['arch'])

    # Reinitialize weights randomly (fresh start)
    for layer in cppn.layers:
        torch.nn.init.xavier_uniform_(layer.weight)

    t0 = time.perf_counter()
    losses, cppn = train_sgd(cppn, target_img, lr=lr, n_iters=n_iters, log_interval=0)
    wall_time = time.perf_counter() - t0

    with torch.no_grad():
        final_img = cppn.generate_image(img_size=target_img.shape[0])

    return {'losses': losses, 'wall_time': wall_time, 'final_img': final_img, 'model': cppn}


def benchmark_kan_sgd(genome_name, target_img, n_iters=5000, lr=3e-3, seed=0):
    """Train a KAN-CPPN with SGD.

    Returns:
        dict with keys: losses, wall_time, final_img, model
    """
    torch.manual_seed(seed)
    cfg = GENOME_CONFIGS[genome_name]
    kan = KAN_CPPN(n_layers=cfg['n_layers'], hidden_size=cfg['hidden_size'])

    t0 = time.perf_counter()
    losses, kan = train_sgd(kan, target_img, lr=lr, n_iters=n_iters, log_interval=0)
    wall_time = time.perf_counter() - t0

    with torch.no_grad():
        final_img = kan.generate_image(img_size=target_img.shape[0])

    return {'losses': losses, 'wall_time': wall_time, 'final_img': final_img, 'model': kan}


def benchmark_swarm_kan(genome_name, target_img, n_iters=5000, lr=3e-3,
                         swarm_interval=5, seed=0):
    """Train a SwarmKAN-CPPN with PSO + SGD.

    Returns:
        dict with keys: losses, wall_time, final_img, model
    """
    torch.manual_seed(seed)
    cfg = GENOME_CONFIGS[genome_name]
    swarm = SwarmKAN_CPPN(n_layers=cfg['n_layers'], hidden_size=cfg['hidden_size'])

    t0 = time.perf_counter()
    losses, swarm = train_swarm(swarm, target_img, lr=lr, n_iters=n_iters,
                                 swarm_interval=swarm_interval, log_interval=0)
    wall_time = time.perf_counter() - t0

    with torch.no_grad():
        final_img = swarm.generate_image(img_size=target_img.shape[0])

    return {'losses': losses, 'wall_time': wall_time, 'final_img': final_img, 'model': swarm}


def benchmark_memetic_kan(genome_name, target_img, n_generations=50,
                           sgd_steps_per_gen=50, lr=3e-3, seed=0):
    """Train a MemeticKAN-CPPN with NES + SGD.

    Note: total effective iterations = n_generations * sgd_steps_per_gen.

    Returns:
        dict with keys: losses, wall_time, final_img, model, total_iters
    """
    torch.manual_seed(seed)
    cfg = GENOME_CONFIGS[genome_name]
    memetic = MemeticKAN_CPPN(
        n_layers=cfg['n_layers'],
        hidden_size=cfg['hidden_size'],
    )

    t0 = time.perf_counter()
    fitness_history, best = train_memetic(
        memetic, target_img,
        n_generations=n_generations,
        sgd_steps_per_gen=sgd_steps_per_gen,
        lr=lr,
        log_interval=0,
    )
    wall_time = time.perf_counter() - t0

    with torch.no_grad():
        final_img = best.generate_image(img_size=target_img.shape[0])

    return {
        'losses': fitness_history,
        'wall_time': wall_time,
        'final_img': final_img,
        'model': best,
        'total_iters': n_generations * sgd_steps_per_gen,
    }


def get_pretrained_reference(genome_name, source='picbreeder'):
    """Load pre-trained genome and compute its MSE against picbreeder target.

    Returns:
        dict with keys: final_img, model, flat_params
    """
    arch, params = load_genome(source, genome_name)
    cppn = CPPN(arch=arch)
    flat = FlattenCPPNParameters(cppn)
    flat.load_jax_flat_params(params)

    with torch.no_grad():
        img_size = 256
        final_img = cppn.generate_image(img_size=img_size)

    return {'final_img': final_img, 'model': cppn, 'flat_params': params}


def benchmark_kan_sgd_degree(genome_name, target_img, spline_degree=1,
                              n_iters=1000, lr=3e-3, seed=0, img_size=64):
    """Train a KAN-CPPN with SGD at a specific spline degree.

    Used for comparing convergence across B-spline degrees.

    Args:
        genome_name: Genome config key.
        target_img: Target image tensor.
        spline_degree: B-spline degree (1-4).
        n_iters: Number of training iterations.
        lr: Learning rate.
        seed: Random seed.
        img_size: Training image resolution.

    Returns:
        dict with keys: losses, wall_time, final_img, model, spline_degree
    """
    torch.manual_seed(seed)
    cfg = GENOME_CONFIGS[genome_name]
    kan = KAN_CPPN(n_layers=cfg['n_layers'], hidden_size=cfg['hidden_size'],
                   spline_degree=spline_degree)

    # Resize target if needed
    if target_img.shape[0] != img_size:
        import torch.nn.functional as F
        target_resized = target_img.permute(2, 0, 1).unsqueeze(0)
        target_resized = F.interpolate(target_resized, size=(img_size, img_size), mode='bilinear',
                                       align_corners=False)
        target_resized = target_resized.squeeze(0).permute(1, 2, 0)
    else:
        target_resized = target_img

    t0 = time.perf_counter()
    losses, kan = train_sgd(kan, target_resized, lr=lr, n_iters=n_iters, log_interval=0)
    wall_time = time.perf_counter() - t0

    with torch.no_grad():
        final_img = kan.generate_image(img_size=img_size)

    return {
        'losses': losses,
        'wall_time': wall_time,
        'final_img': final_img,
        'model': kan,
        'spline_degree': spline_degree,
    }


def run_full_benchmark(genome_name, n_iters=5000, n_seeds=3, n_generations=50,
                        sgd_steps_per_gen=None):
    """Run all methods on a genome with multiple seeds.

    Args:
        genome_name: 'skull', 'butterfly', or 'apple'
        n_iters: SGD iterations for MLP, KAN, SwarmKAN
        n_seeds: Number of random seeds for variance estimation
        n_generations: Generations for memetic
        sgd_steps_per_gen: SGD steps per memetic generation.
            If None, set to n_iters // n_generations so total work is comparable.

    Returns:
        dict mapping method_name -> list of result dicts (one per seed)
    """
    if sgd_steps_per_gen is None:
        sgd_steps_per_gen = max(1, n_iters // n_generations)

    target_img = load_target_image(genome_name)

    results = {
        'mlp_sgd': [],
        'kan_sgd': [],
        'swarm_kan': [],
        'memetic_kan': [],
    }

    for seed in range(n_seeds):
        print(f"\n=== Seed {seed} ===")

        print(f"  MLP+SGD...")
        results['mlp_sgd'].append(
            benchmark_mlp_sgd(genome_name, target_img, n_iters=n_iters, seed=seed))

        print(f"  KAN+SGD...")
        results['kan_sgd'].append(
            benchmark_kan_sgd(genome_name, target_img, n_iters=n_iters, seed=seed))

        print(f"  SwarmKAN...")
        results['swarm_kan'].append(
            benchmark_swarm_kan(genome_name, target_img, n_iters=n_iters, seed=seed))

        print(f"  MemeticKAN...")
        results['memetic_kan'].append(
            benchmark_memetic_kan(genome_name, target_img,
                                   n_generations=n_generations,
                                   sgd_steps_per_gen=sgd_steps_per_gen, seed=seed))

    # Add pre-trained references (no seed variance)
    results['picbreeder'] = [get_pretrained_reference(genome_name, 'picbreeder')]
    results['sgd_pretrained'] = [get_pretrained_reference(genome_name, 'sgd')]

    return results, target_img
