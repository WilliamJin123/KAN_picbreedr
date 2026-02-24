# KAN Picbreeder Analysis Suite — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a comprehensive analysis notebook + HTML report answering 6 research questions about KAN variants vs. standard NNs for CPPN image generation.

**Architecture:** Modular `analysis/` package with benchmark, spline inspection, comparison metrics, and text prototype modules. Single Jupyter notebook (`notebooks/kan_analysis.ipynb`) imports these modules and presents results as a research paper. HTML export via nbconvert.

**Tech Stack:** PyTorch, matplotlib, numpy, scipy (for curve fitting), jupyter, nbconvert. All existing `src/` modules used as-is.

---

### Task 1: Create analysis package scaffolding

**Files:**
- Create: `analysis/__init__.py`
- Create: `analysis/benchmark.py`
- Create: `analysis/spline_inspector.py`
- Create: `analysis/comparison.py`
- Create: `analysis/text_prototype.py`

**Step 1: Create analysis/__init__.py**

```python
"""Analysis suite for KAN Picbreeder project."""
```

**Step 2: Create analysis/benchmark.py skeleton**

```python
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
        final_img = cppn.generate_image(img_size=256)

    return {'final_img': final_img, 'model': cppn, 'flat_params': params}


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
```

**Step 3: Create analysis/comparison.py**

```python
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
```

**Step 4: Create analysis/spline_inspector.py**

```python
"""Inspect and analyze learned spline activation functions in KAN layers.

Extracts the effective spline curve for each (layer, in, out) edge,
compares against known activation functions, and identifies what
the KAN learned.
"""

import torch
import numpy as np
from scipy.optimize import curve_fit


# Library of known activation functions (evaluated on raw input domain)
# The KAN applies sigmoid first, so we define these on [0, 1] (normalized domain)
# AND on the raw domain [-3, 3] for comparison.
KNOWN_FUNCTIONS = {
    'identity': lambda x: x,
    'sin': lambda x: np.sin(x),
    'cos': lambda x: np.cos(x),
    'tanh': lambda x: np.tanh(x),
    'sigmoid': lambda x: 1 / (1 + np.exp(-x)) * 2 - 1,
    'gaussian': lambda x: np.exp(-x**2) * 2 - 1,
    'relu': lambda x: np.maximum(0, x),
    'quadratic': lambda x: x**2,
    'abs': lambda x: np.abs(x),
    'constant': lambda x: np.zeros_like(x),
}

# Parameterized versions for curve fitting
def _scaled_shifted(func):
    """Wrap a base function with learnable scale and shift: a * f(b*x + c) + d"""
    def parameterized(x, a, b, c, d):
        return a * func(b * x + c) + d
    return parameterized

FITTABLE_FUNCTIONS = {
    name: _scaled_shifted(fn) for name, fn in KNOWN_FUNCTIONS.items()
    if name != 'constant'
}


def extract_spline_curve(layer, in_idx, out_idx, n_points=1000):
    """Extract the effective spline curve for one (in, out) edge.

    Evaluates the spline at n_points uniformly spaced inputs by:
    1. Creating inputs in the raw domain [-3, 3]
    2. Applying sigmoid to map to [0, 1] (the grid domain)
    3. Interpolating on the spline grid
    4. Multiplying by the edge weight

    Args:
        layer: A KANCPPNLayer instance.
        in_idx: Input feature index.
        out_idx: Output feature index.
        n_points: Number of evaluation points.

    Returns:
        raw_inputs: numpy array of shape (n_points,) in [-3, 3]
        spline_values: numpy array of shape (n_points,) — the spline output
        normalized_inputs: numpy array in [0, 1] (after sigmoid)
    """
    raw_inputs = np.linspace(-3, 3, n_points)
    normalized = 1 / (1 + np.exp(-raw_inputs))  # sigmoid

    # Map to grid indices
    grid_size = layer.grid_size
    scaled = normalized * (grid_size - 1)
    idx = np.clip(scaled.astype(int), 0, grid_size - 2)
    frac = scaled - idx

    # Get coefficients for this edge
    coeffs = layer.coeffs[out_idx, in_idx].detach().cpu().numpy()  # (grid_size,)
    weight = layer.weights[out_idx, in_idx].detach().cpu().item()

    # Linear interpolation
    left = coeffs[idx]
    right = coeffs[idx + 1]
    spline_values = (left + frac * (right - left)) * weight

    return raw_inputs, spline_values, normalized


def fit_known_function(raw_inputs, spline_values):
    """Find the best-matching known activation function for a spline curve.

    Tries fitting each known function with scale/shift parameters.

    Args:
        raw_inputs: numpy array of input values.
        spline_values: numpy array of spline output values.

    Returns:
        best_match: dict with keys:
            name: str — name of best-matching function
            l2_distance: float — L2 error of the best fit
            params: tuple — (a, b, c, d) fitted parameters
            fitted_curve: numpy array — the fitted function evaluated at raw_inputs
            all_fits: dict mapping name -> (l2_distance, fitted_curve)
    """
    all_fits = {}

    for name, func in FITTABLE_FUNCTIONS.items():
        try:
            popt, _ = curve_fit(func, raw_inputs, spline_values,
                                p0=[1.0, 1.0, 0.0, 0.0],
                                maxfev=5000)
            fitted = func(raw_inputs, *popt)
            l2 = np.sqrt(np.mean((spline_values - fitted) ** 2))
            all_fits[name] = (l2, fitted, popt)
        except (RuntimeError, ValueError):
            # curve_fit failed — skip this function
            all_fits[name] = (float('inf'), None, None)

    # Also try constant (just the mean)
    mean_val = np.mean(spline_values)
    const_curve = np.full_like(spline_values, mean_val)
    const_l2 = np.sqrt(np.mean((spline_values - const_curve) ** 2))
    all_fits['constant'] = (const_l2, const_curve, (mean_val,))

    # Find best match
    best_name = min(all_fits, key=lambda k: all_fits[k][0])
    best_l2, best_curve, best_params = all_fits[best_name]

    return {
        'name': best_name,
        'l2_distance': best_l2,
        'params': best_params,
        'fitted_curve': best_curve,
        'all_fits': {k: (v[0], v[1]) for k, v in all_fits.items()},
    }


def analyze_all_edges(kan_cppn, top_k=20):
    """Analyze all spline edges in a KAN-CPPN and find what they learned.

    Args:
        kan_cppn: A KAN_CPPN instance.
        top_k: Return top_k edges by signal magnitude.

    Returns:
        List of dicts sorted by signal magnitude (descending), each with:
            layer_idx: int
            in_idx: int
            out_idx: int
            raw_inputs: numpy array
            spline_values: numpy array
            best_match: result from fit_known_function
            signal_magnitude: float — RMS of spline values
    """
    edges = []

    for layer_idx, layer in enumerate(kan_cppn.layers):
        for out_idx in range(layer.out_features):
            for in_idx in range(layer.in_features):
                raw_inputs, spline_values, _ = extract_spline_curve(layer, in_idx, out_idx)
                signal_mag = np.sqrt(np.mean(spline_values ** 2))

                best_match = fit_known_function(raw_inputs, spline_values)

                edges.append({
                    'layer_idx': layer_idx,
                    'in_idx': in_idx,
                    'out_idx': out_idx,
                    'raw_inputs': raw_inputs,
                    'spline_values': spline_values,
                    'best_match': best_match,
                    'signal_magnitude': signal_mag,
                })

    # Sort by signal magnitude (most active edges first)
    edges.sort(key=lambda e: e['signal_magnitude'], reverse=True)
    return edges[:top_k]
```

**Step 5: Create analysis/text_prototype.py**

```python
"""Toy prototype: KAN-CPPN for 1D sequence generation.

Explores whether the CPPN coordinate-based architecture can generalize
from 2D spatial signals (images) to 1D sequential signals (text-like).

Instead of (y, x, d, bias) -> (H, S, V), we use:
  (position, bias) -> (char_logits) over a sequence
"""

import torch
import torch.nn as nn
import numpy as np

from src.kan import KANCPPNLayer


class SequenceKAN(nn.Module):
    """A 1D KAN-CPPN that maps position -> signal value.

    Uses the same KAN spline layers but with 1D positional input
    instead of 2D spatial coordinates.

    Args:
        n_layers: Number of hidden layers.
        hidden_size: Neurons per hidden layer.
        output_size: Number of output channels.
        grid_size: Spline grid size.
    """

    def __init__(self, n_layers=4, hidden_size=16, output_size=1, grid_size=20):
        super().__init__()
        # Inputs: (position, bias) = 2 features
        layers = [KANCPPNLayer(2, hidden_size, grid_size)]
        for _ in range(n_layers - 1):
            layers.append(KANCPPNLayer(hidden_size, hidden_size, grid_size))
        layers.append(KANCPPNLayer(hidden_size, output_size, grid_size))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def generate_signal(self, seq_len=100):
        """Generate a 1D signal from positional coordinates.

        Args:
            seq_len: Length of the output sequence.

        Returns:
            positions: (seq_len,) tensor of input positions
            signal: (seq_len, output_size) tensor of output values
        """
        positions = torch.linspace(-1, 1, seq_len)
        bias = torch.ones(seq_len)
        inputs = torch.stack([positions, bias], dim=-1)  # (seq_len, 2)
        with torch.no_grad():
            signal = self.forward(inputs)
        return positions, signal


def train_sequence_kan(model, target_signal, positions, n_iters=2000, lr=3e-3):
    """Train a SequenceKAN to reproduce a target 1D signal.

    Args:
        model: SequenceKAN instance.
        target_signal: (seq_len, output_size) tensor.
        positions: (seq_len,) tensor of positions.
        n_iters: Training iterations.
        lr: Learning rate.

    Returns:
        losses: list of MSE values per iteration.
    """
    bias = torch.ones_like(positions)
    inputs = torch.stack([positions, bias], dim=-1)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    for i in range(n_iters):
        optimizer.zero_grad()
        output = model(inputs)
        loss = torch.mean((output - target_signal) ** 2)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return losses


def make_test_signals(seq_len=200):
    """Create a set of test target signals for the prototype.

    Returns:
        dict mapping signal_name -> (positions, target_signal) tuples
    """
    positions = torch.linspace(-1, 1, seq_len)

    signals = {
        'sine': torch.sin(2 * np.pi * positions).unsqueeze(-1),
        'square_wave': torch.sign(torch.sin(2 * np.pi * positions)).unsqueeze(-1),
        'sawtooth': (positions % 0.5).unsqueeze(-1) * 4 - 1,
        'gaussian_bump': torch.exp(-8 * positions**2).unsqueeze(-1),
        'multi_frequency': (
            0.5 * torch.sin(2 * np.pi * positions)
            + 0.3 * torch.sin(6 * np.pi * positions)
            + 0.2 * torch.cos(10 * np.pi * positions)
        ).unsqueeze(-1),
    }

    return {name: (positions, sig) for name, sig in signals.items()}
```

**Step 6: Commit scaffolding**

```bash
git add analysis/__init__.py analysis/benchmark.py analysis/comparison.py analysis/spline_inspector.py analysis/text_prototype.py
git commit -m "feat: add analysis package scaffolding (benchmark, spline inspector, comparison, text prototype)"
```

---

### Task 2: Create the Jupyter notebook structure

**Files:**
- Create: `notebooks/kan_analysis.ipynb`
- Create: `notebooks/export_html.py`

**Step 1: Create notebooks/export_html.py**

```python
"""Export the analysis notebook to HTML for sharing."""

import subprocess
import sys

def export():
    subprocess.run([
        sys.executable, '-m', 'jupyter', 'nbconvert',
        '--to', 'html',
        '--no-input',  # hide code cells in output
        'notebooks/kan_analysis.ipynb',
        '--output-dir', 'docs/',
        '--output', 'kan_analysis_report.html',
    ], check=True)
    print("Exported to docs/kan_analysis_report.html")

if __name__ == '__main__':
    export()
```

**Step 2: Create the notebook with all 6 sections as markdown + code cell pairs**

The notebook should have these cells in order:

1. **Title cell (markdown):**
```markdown
# KAN Picbreeder: Comprehensive Analysis

Comparing KAN variants (spline KAN, SwarmKAN, MemeticKAN) against standard MLPs for CPPN-based image generation. This notebook answers six research questions about performance, learned representations, scalability, and transferable design insights.
```

2. **Setup cell (code):**
```python
import sys
sys.path.insert(0, '..')

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from analysis.benchmark import (
    run_full_benchmark, load_target_image, GENOME_CONFIGS,
    get_pretrained_reference
)
from analysis.comparison import mse, ssim, feature_cosine_similarity
from analysis.spline_inspector import extract_spline_curve, fit_known_function, analyze_all_edges
from analysis.text_prototype import SequenceKAN, train_sequence_kan, make_test_signals

plt.rcParams['figure.dpi'] = 120
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 11

%matplotlib inline

GENOME = 'skull'  # Change to 'butterfly' or 'apple' to analyze other genomes
```

3. **Section 1 markdown:**
```markdown
## 1. Method Comparison: Who Wins?

We train 4 methods from scratch on the same target image with 3 random seeds, plus compare against pre-trained picbreeder and SGD genomes as reference bounds.

**Methods:**
- **MLP+SGD**: Standard CPPN with fixed activations (identity, gaussian, sin, sigmoid), trained with normalized-gradient SGD
- **KAN+SGD**: Same architecture but with learnable spline activations, trained with SGD
- **SwarmKAN**: KAN + Particle Swarm Optimization on spline coefficients (PSO every 5 SGD steps)
- **MemeticKAN**: KAN + Natural Evolution Strategy (antithetic sampling) + SGD local refinement
- **Picbreeder reference**: Pre-evolved via interactive evolution (upper bound for interpretability)
- **SGD pre-trained**: Pre-trained with standard SGD from the FER paper
```

4. **Section 1 benchmark code cell:**
```python
# Run full benchmark (this takes a few minutes)
results, target_img = run_full_benchmark(GENOME, n_iters=5000, n_seeds=3, n_generations=50)
```

5. **Section 1 visualization code cell (convergence curves):**
```python
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# --- Convergence curves ---
ax = axes[0]
colors = {'mlp_sgd': '#1f77b4', 'kan_sgd': '#ff7f0e', 'swarm_kan': '#2ca02c', 'memetic_kan': '#d62728'}
labels = {'mlp_sgd': 'MLP+SGD', 'kan_sgd': 'KAN+SGD', 'swarm_kan': 'SwarmKAN', 'memetic_kan': 'MemeticKAN'}

for method, runs in results.items():
    if method in ('picbreeder', 'sgd_pretrained'):
        continue
    all_losses = np.array([r['losses'] for r in runs])
    mean_loss = all_losses.mean(axis=0)
    std_loss = all_losses.std(axis=0)

    if method == 'memetic_kan':
        # Memetic losses are per-generation, stretch to match iteration scale
        total_iters = runs[0]['total_iters']
        x = np.linspace(0, total_iters, len(mean_loss))
    else:
        x = np.arange(len(mean_loss))

    ax.plot(x, mean_loss, color=colors[method], label=labels[method], linewidth=1.5)
    ax.fill_between(x, mean_loss - std_loss, mean_loss + std_loss, alpha=0.2, color=colors[method])

ax.set_yscale('log')
ax.set_xlabel('Iteration')
ax.set_ylabel('MSE (log scale)')
ax.set_title(f'Convergence Curves — {GENOME.title()}')
ax.legend()
ax.grid(True, alpha=0.3)

# --- Final MSE bar chart ---
ax = axes[1]
method_names = []
final_mses = []
final_stds = []

for method in ['mlp_sgd', 'kan_sgd', 'swarm_kan', 'memetic_kan']:
    method_names.append(labels[method])
    finals = [r['losses'][-1] for r in results[method]]
    final_mses.append(np.mean(finals))
    final_stds.append(np.std(finals))

# Add pre-trained references
for ref_name, ref_label in [('picbreeder', 'Picbreeder'), ('sgd_pretrained', 'SGD-pretrained')]:
    ref_img = results[ref_name][0]['final_img']
    ref_mse = mse(ref_img, target_img)
    method_names.append(ref_label)
    final_mses.append(ref_mse)
    final_stds.append(0)

bars = ax.bar(method_names, final_mses, yerr=final_stds, capsize=4,
              color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
ax.set_ylabel('Final MSE')
ax.set_title(f'Final Image Quality — {GENOME.title()}')
ax.tick_params(axis='x', rotation=30)

# --- Wall-clock time ---
ax = axes[2]
time_names = []
times = []
time_stds = []

for method in ['mlp_sgd', 'kan_sgd', 'swarm_kan', 'memetic_kan']:
    time_names.append(labels[method])
    ts = [r['wall_time'] for r in results[method]]
    times.append(np.mean(ts))
    time_stds.append(np.std(ts))

ax.bar(time_names, times, yerr=time_stds, capsize=4,
       color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
ax.set_ylabel('Wall-clock Time (seconds)')
ax.set_title(f'Training Time — {GENOME.title()}')
ax.tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.show()
```

6. **Section 1 SSIM + image comparison cell:**
```python
# SSIM comparison
fig, axes = plt.subplots(1, 6, figsize=(20, 4))

imgs_to_show = [
    ('Target', target_img),
]
for method, label in [('mlp_sgd', 'MLP+SGD'), ('kan_sgd', 'KAN+SGD'),
                       ('swarm_kan', 'SwarmKAN'), ('memetic_kan', 'MemeticKAN'),
                       ('picbreeder', 'Picbreeder')]:
    imgs_to_show.append((label, results[method][0]['final_img']))

for i, (label, img) in enumerate(imgs_to_show):
    ax = axes[i]
    img_np = img.detach().cpu().numpy() if isinstance(img, torch.Tensor) else img
    ax.imshow(img_np.clip(0, 1))
    ax.set_title(label, fontsize=10)
    if i > 0:
        s = ssim(target_img, img)
        m = mse(target_img, img)
        ax.set_xlabel(f'MSE={m:.4f}\nSSIM={s:.3f}', fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])

plt.suptitle(f'Image Comparison — {GENOME.title()}', fontsize=14)
plt.tight_layout()
plt.show()
```

7. **Section 2 markdown:**
```markdown
## 2. How Close Are the Learned Functions?

Each KAN edge has a learnable spline activation. We extract these curves and compare them against the CPPN's actual activation functions (gaussian, sin, sigmoid, identity, tanh).

**Method:** For each edge, we:
1. Evaluate the spline at 1000 points in [-3, 3]
2. Fit `a * f(b*x + c) + d` for each known function f
3. Pick the best fit by L2 distance
4. Build a heatmap: what did each edge learn?
```

8. **Section 2 analysis code cell:**
```python
# Get the best KAN model from seed 0
kan_model = results['kan_sgd'][0]['model']

# Analyze top-20 most active edges
top_edges = analyze_all_edges(kan_model, top_k=20)

# --- Grid of learned spline vs. best match ---
n_show = min(12, len(top_edges))
fig, axes = plt.subplots(3, 4, figsize=(16, 12))

for i, edge in enumerate(top_edges[:n_show]):
    ax = axes[i // 4, i % 4]

    ax.plot(edge['raw_inputs'], edge['spline_values'], 'b-', linewidth=2, label='Learned spline')

    match = edge['best_match']
    if match['fitted_curve'] is not None:
        ax.plot(edge['raw_inputs'], match['fitted_curve'], 'r--', linewidth=1.5,
                label=f"Best fit: {match['name']}")

    ax.set_title(f"L{edge['layer_idx']} [{edge['in_idx']},{edge['out_idx']}]\n"
                 f"≈ {match['name']} (L2={match['l2_distance']:.3f})",
                 fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-3, 3)

plt.suptitle(f'Learned Spline Activations vs. Known Functions — {GENOME.title()} KAN', fontsize=14)
plt.tight_layout()
plt.show()
```

9. **Section 2 heatmap code cell:**
```python
# Build function-type heatmap across all layers
function_counts = {}
all_edges_full = analyze_all_edges(kan_model, top_k=999999)  # Get all edges

for edge in all_edges_full:
    layer = edge['layer_idx']
    fn_name = edge['best_match']['name']
    key = (layer, fn_name)
    function_counts[key] = function_counts.get(key, 0) + 1

# Create heatmap matrix
n_layers_total = len(kan_model.layers)
fn_names = sorted(set(e['best_match']['name'] for e in all_edges_full))

heatmap = np.zeros((n_layers_total, len(fn_names)))
for (layer, fn), count in function_counts.items():
    col = fn_names.index(fn)
    heatmap[layer, col] = count

fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(heatmap, aspect='auto', cmap='YlOrRd')
ax.set_xticks(range(len(fn_names)))
ax.set_xticklabels(fn_names, rotation=45, ha='right')
ax.set_ylabel('Layer')
ax.set_xlabel('Best-fit Function')
ax.set_title(f'What Each Layer Learned — {GENOME.title()} KAN\n'
             '(count of edges best-fit to each function type)')
plt.colorbar(im, label='Edge count')
plt.tight_layout()
plt.show()

# Summary statistics
print(f"\nFunction distribution across {len(all_edges_full)} edges:")
from collections import Counter
fn_dist = Counter(e['best_match']['name'] for e in all_edges_full)
for fn, count in fn_dist.most_common():
    pct = 100 * count / len(all_edges_full)
    print(f"  {fn:12s}: {count:4d} ({pct:.1f}%)")
```

10. **Section 3 markdown:**
```markdown
## 3. What Did the Splines Learn? (Visual + Numerical)

For the most visually impactful spline edges, we show:
- The learned spline shape
- Its closest known function + residual
- How sweeping this parameter changes the output image
```

11. **Section 3 code cell:**
```python
from src.kan import FlattenKANParameters
from src.visualize import discover_interesting_kan_sweeps, sweep_weight, plot_sweep_grid

kan_flat = FlattenKANParameters(kan_model)

# Find most impactful parameters
interesting = discover_interesting_kan_sweeps(kan_model, kan_flat, target_img,
                                              n_candidates_per_group=8, top_k=6)

# Multi-panel figure: spline shape | best fit + residual | image sweep
fig = plt.figure(figsize=(20, 4 * len(interesting)))
gs = gridspec.GridSpec(len(interesting), 3, width_ratios=[1, 1, 3])

params = kan_flat.flatten()

for row, entry in enumerate(interesting):
    flat_idx = entry['flat_idx']
    desc = entry['description']

    # Parse layer/in/out from description like "Layer 5 coeffs[3,7,12]"
    # Find the corresponding edge info
    from src.visualize import get_kan_param_info
    info = get_kan_param_info(kan_model, flat_idx)
    layer_idx = info['layer_idx']
    layer = kan_model.layers[layer_idx]

    # Extract spline for this edge (use first two indices for out, in)
    indices = info['local_shape_indices']
    if info['param_type'] == 'coeffs' and len(indices) >= 2:
        out_idx, in_idx = indices[0], indices[1]
        raw_inputs, spline_values, _ = extract_spline_curve(layer, in_idx, out_idx)
        match = fit_known_function(raw_inputs, spline_values)

        # Panel 1: Spline shape
        ax1 = fig.add_subplot(gs[row, 0])
        ax1.plot(raw_inputs, spline_values, 'b-', linewidth=2)
        ax1.set_title(f'{desc}\nSpline Shape', fontsize=9)
        ax1.grid(True, alpha=0.3)

        # Panel 2: Best fit + residual
        ax2 = fig.add_subplot(gs[row, 1])
        if match['fitted_curve'] is not None:
            ax2.plot(raw_inputs, match['fitted_curve'], 'r-', label=f"{match['name']}", linewidth=1.5)
            residual = spline_values - match['fitted_curve']
            ax2.plot(raw_inputs, residual, 'g--', label='Residual', linewidth=1, alpha=0.7)
        ax2.set_title(f"≈ {match['name']} (L2={match['l2_distance']:.3f})", fontsize=9)
        ax2.legend(fontsize=7)
        ax2.grid(True, alpha=0.3)
    else:
        ax1 = fig.add_subplot(gs[row, 0])
        ax1.text(0.5, 0.5, f'{desc}\n(not a spline coeff)', ha='center', va='center')
        ax2 = fig.add_subplot(gs[row, 1])

    # Panel 3: Image sweep strip
    ax3 = fig.add_subplot(gs[row, 2])
    sweep_imgs = sweep_weight(params, flat_idx, kan_flat, img_size=128, n=7)
    sweep_np = sweep_imgs.detach().cpu().numpy()
    strip = np.concatenate(sweep_np, axis=1)
    ax3.imshow(strip.clip(0, 1))
    ax3.set_title(f'Weight Sweep: {desc}', fontsize=9)
    ax3.set_xticks([]); ax3.set_yticks([])

plt.suptitle(f'Spline Analysis — {GENOME.title()} KAN', fontsize=14)
plt.tight_layout()
plt.show()
```

12. **Section 4 markdown:**
```markdown
## 4. Can This Scale to Text?

### Theoretical Analysis

CPPNs are fundamentally **coordinate-based function approximators**: they map spatial coordinates (x, y) to output values (color). This is naturally suited to images because pixels have spatial structure — nearby pixels should have similar values.

**Key constraints for text:**
- Text is sequential, not spatial. There's no 2D grid.
- Text has discrete tokens, not continuous color values.
- Long-range dependencies in text span the entire sequence (not just local neighborhoods).

**What would need to change:**
1. **Input encoding**: Replace (x, y, d, bias) with (position, bias) or learned positional embeddings
2. **Output head**: Replace HSV → RGB with logits → softmax for token prediction
3. **Architecture depth**: Text likely needs deeper networks or attention for long-range deps
4. **Training signal**: Cross-entropy loss instead of MSE on pixel values

**Why it might partially work:**
- KAN's learnable activation functions could capture complex position→token mappings
- The CPPN's ability to generate patterns from coordinates is analogous to positional encoding
- Evolutionary search (memetic) could explore discrete loss landscapes better than pure SGD

**Why it probably won't scale well:**
- CPPNs have no mechanism for attending to other positions (no self-attention)
- Each position is processed independently — there's no information flow between positions
- Language requires modeling dependencies between tokens, not just position → token mapping

### Toy Prototype

Below we test whether a KAN-CPPN can learn simple 1D signals (sine, square wave, etc.) as a sanity check. If it can't even learn periodic 1D functions, text is out of the question.
```

13. **Section 4 code cell:**
```python
test_signals = make_test_signals(seq_len=200)

fig, axes = plt.subplots(len(test_signals), 2, figsize=(14, 3 * len(test_signals)))

for i, (name, (positions, target)) in enumerate(test_signals.items()):
    model = SequenceKAN(n_layers=4, hidden_size=16, output_size=1)
    losses = train_sequence_kan(model, target, positions, n_iters=3000, lr=3e-3)

    # Generate learned signal
    _, learned = model.generate_signal(seq_len=200)

    # Plot signal comparison
    ax = axes[i, 0]
    ax.plot(positions.numpy(), target[:, 0].numpy(), 'b-', label='Target', linewidth=2)
    ax.plot(positions.numpy(), learned[:, 0].detach().numpy(), 'r--', label='KAN output', linewidth=1.5)
    ax.set_title(f'{name} — Final MSE: {losses[-1]:.6f}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot loss curve
    ax = axes[i, 1]
    ax.plot(losses)
    ax.set_yscale('log')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('MSE')
    ax.set_title(f'{name} — Loss Curve')
    ax.grid(True, alpha=0.3)

plt.suptitle('1D Signal Approximation with KAN-CPPN', fontsize=14)
plt.tight_layout()
plt.show()

print("\nConclusion: KAN-CPPNs can/cannot learn 1D periodic signals,")
print("suggesting they are/aren't suitable building blocks for sequential tasks.")
print("The fundamental limitation is the lack of inter-position information flow.")
```

14. **Section 5 markdown:**
```markdown
## 5. How Swarm & Memetic KAN Actually Work

### Spline Type: Linear Interpolation (Piecewise Linear)

The KAN layers use **piecewise linear splines** — NOT B-splines, quadratic, or cubic.

Each edge (input_i → output_j) has:
- A **fixed grid** of 20 knots at `linspace(0, 1, 20)`
- **20 learnable coefficients** (one per knot)
- **Linear interpolation** between adjacent knots

The input is mapped to [0, 1] via sigmoid, then the two nearest knots are found and linearly interpolated. This is the simplest possible spline — computationally cheap and fully differentiable.

**Full edge computation:**
```
spline_ij(x) = weight_ij * lerp(coeffs[k], coeffs[k+1], frac)
```
where `k = floor(sigmoid(x) * 19)` and `frac` is the fractional part.

### SwarmKAN: PSO + SGD Hybrid

**Particle Swarm Optimization** on spline coefficients, interleaved with gradient descent.

**Algorithm per training step:**
1. Standard SGD step (Adam optimizer, normalized gradients)
2. Every 5 SGD steps, perform PSO update:
   - Each layer maintains 5 "particles" (alternative coefficient vectors)
   - Particle velocities updated via classic PSO formula:
     `v = 0.7*v + 1.5*r1*(personal_best - pos) + 1.5*r2*(global_best - pos)`
   - Particle positions updated: `pos += v`
   - Active coefficients blended with particle 0: `coeffs = 0.9*coeffs + 0.1*particle_0`

**Why this design:**
- SGD handles local optimization (fast convergence in the loss basin)
- PSO explores the activation function landscape (particles try different spline shapes)
- Soft blending (10%) prevents PSO from destroying SGD's progress
- Each layer has independent particle populations (no cross-layer interference)

### MemeticKAN: NES + SGD Hybrid

**Natural Evolution Strategy** (OpenAI-ES style) for global search, plus SGD for local refinement.

**Algorithm per generation:**
1. **ES gradient estimation:**
   - Take current spline params as "center" (base_weight is frozen/excluded)
   - Sample 20 random perturbation vectors ε
   - For each ε, evaluate fitness at (center + σε) and (center - σε) [antithetic pairs]
   - ES gradient = Σ [(f+ - f-) / (|f+| + |f-| + ε)] * ε / (2 * pop_size * σ)
   - Apply ES gradient to center params (only if it improves fitness)

2. **SGD local refinement:**
   - Create fresh Adam optimizer (old one's state is invalid after ES jump)
   - Run 50-100 SGD steps with normalized gradients
   - This refines the ES's coarse global move

**Key difference from SwarmKAN:**
- SwarmKAN perturbs per-layer independently; MemeticKAN perturbs globally
- SwarmKAN maintains persistent particle population; MemeticKAN samples fresh each generation
- MemeticKAN uses antithetic sampling for variance reduction (stronger signal)
- MemeticKAN gates ES updates (only accepts improvements)
```

15. **Section 5 algorithm visualization code cell:**
```python
# Visualize the PSO and NES algorithms step by step

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# --- PSO Visualization ---
ax = axes[0]
ax.set_title('SwarmKAN: PSO on Spline Coefficients', fontsize=12)

# Show a 2D slice of the coefficient space with particle trajectories
np.random.seed(42)
n_steps = 20
n_particles = 5

# Simulate PSO in 2D for illustration
positions = np.random.randn(n_particles, 2) * 0.5
velocities = np.zeros_like(positions)
personal_best = positions.copy()
personal_best_scores = np.full(n_particles, np.inf)
global_best = positions[0].copy()
global_best_score = np.inf

# Simple 2D loss landscape: (x-0.3)^2 + (y+0.2)^2
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

particle_colors = plt.cm.Set1(np.linspace(0, 1, n_particles))
for i in range(n_particles):
    traj = np.array(trajectories[i])
    ax.plot(traj[:, 0], traj[:, 1], '-o', color=particle_colors[i], markersize=3,
            alpha=0.6, label=f'Particle {i}')
    ax.plot(traj[0, 0], traj[0, 1], 's', color=particle_colors[i], markersize=8)
    ax.plot(traj[-1, 0], traj[-1, 1], '*', color=particle_colors[i], markersize=12)

ax.plot(0.3, -0.2, 'kx', markersize=15, markeredgewidth=3, label='Optimum')
ax.legend(fontsize=7, loc='upper right')
ax.set_xlabel('Coefficient dimension 1')
ax.set_ylabel('Coefficient dimension 2')
ax.grid(True, alpha=0.3)

# --- NES Visualization ---
ax = axes[1]
ax.set_title('MemeticKAN: NES Gradient Estimation', fontsize=12)

center = np.array([0.0, 0.0])
sigma = 0.3
n_perturbations = 8

np.random.seed(123)
for gen in range(3):
    color = plt.cm.Blues(0.4 + gen * 0.2)
    for i in range(n_perturbations):
        eps = np.random.randn(2) * sigma
        pos = center + eps
        neg = center - eps
        ax.plot([neg[0], pos[0]], [neg[1], pos[1]], '-', color=color, alpha=0.3, linewidth=1)
        ax.plot(pos[0], pos[1], 'o', color='green', markersize=4, alpha=0.5)
        ax.plot(neg[0], neg[1], 'o', color='red', markersize=4, alpha=0.5)

    ax.plot(center[0], center[1], 'D', color=color, markersize=10,
            label=f'Center (gen {gen})')
    # Simulate ES gradient move
    center = center + np.array([0.1, -0.07]) * (gen + 1) * 0.5

ax.plot(0.3, -0.2, 'kx', markersize=15, markeredgewidth=3, label='Optimum')
ax.legend(fontsize=8)
ax.set_xlabel('Parameter dimension 1')
ax.set_ylabel('Parameter dimension 2')
ax.grid(True, alpha=0.3)

plt.suptitle('Algorithm Visualization (2D Projection)', fontsize=14)
plt.tight_layout()
plt.show()
```

16. **Section 6 markdown:**
```markdown
## 6. Design Choices That Mattered (Transferable Insights)

These are the inductive biases and engineering decisions that made the KAN-CPPN work. Each one is a transferable "skill" for future deep learning projects.
```

17. **Section 6 ablation code cell:**
```python
# Ablation study: test key design choices

from src.kan import KAN_CPPN, KANCPPNLayer

cfg = GENOME_CONFIGS[GENOME]

def train_and_measure(model, target, n_iters=2000):
    """Quick training run, return final loss."""
    from src.train import train_sgd
    losses, _ = train_sgd(model, target, lr=3e-3, n_iters=n_iters, log_interval=0)
    return losses

# --- Ablation 1: Orthogonal vs. Kaiming init ---
torch.manual_seed(42)
kan_ortho = KAN_CPPN(n_layers=cfg['n_layers'], hidden_size=cfg['hidden_size'])
# Default is orthogonal

torch.manual_seed(42)
kan_kaiming = KAN_CPPN(n_layers=cfg['n_layers'], hidden_size=cfg['hidden_size'])
# Override with kaiming
for layer in kan_kaiming.layers:
    torch.nn.init.kaiming_uniform_(layer.base_weight)

losses_ortho = train_and_measure(kan_ortho, target_img)
losses_kaiming = train_and_measure(kan_kaiming, target_img)

# --- Ablation 2: With vs. without residual base path ---
# (We can't easily remove base_weight without modifying the layer,
#  so we zero it out instead)
torch.manual_seed(42)
kan_no_residual = KAN_CPPN(n_layers=cfg['n_layers'], hidden_size=cfg['hidden_size'])
with torch.no_grad():
    for layer in kan_no_residual.layers:
        layer.base_weight.fill_(0)
        layer.base_weight.requires_grad_(False)

losses_no_residual = train_and_measure(kan_no_residual, target_img)

# --- Signal propagation test ---
torch.manual_seed(42)
kan_test = KAN_CPPN(n_layers=cfg['n_layers'], hidden_size=cfg['hidden_size'])
test_input = torch.randn(100, 4)  # Random batch

with torch.no_grad():
    x = test_input
    stds_ortho = [x.std().item()]
    for layer in kan_test.layers:
        x = layer(x)
        stds_ortho.append(x.std().item())

# Reset with kaiming
for layer in kan_test.layers:
    torch.nn.init.kaiming_uniform_(layer.base_weight)

with torch.no_grad():
    x = test_input
    stds_kaiming = [x.std().item()]
    for layer in kan_test.layers:
        x = layer(x)
        stds_kaiming.append(x.std().item())

# --- Plot ablation results ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Convergence comparison
ax = axes[0]
ax.plot(losses_ortho, label='Orthogonal init (default)', linewidth=1.5)
ax.plot(losses_kaiming, label='Kaiming init', linewidth=1.5)
ax.plot(losses_no_residual, label='No residual path', linewidth=1.5)
ax.set_yscale('log')
ax.set_xlabel('Iteration')
ax.set_ylabel('MSE')
ax.set_title('Ablation: Init & Residual Path')
ax.legend()
ax.grid(True, alpha=0.3)

# Signal propagation
ax = axes[1]
ax.plot(stds_ortho, 'o-', label='Orthogonal', linewidth=1.5)
ax.plot(stds_kaiming, 's-', label='Kaiming', linewidth=1.5)
ax.set_xlabel('Layer')
ax.set_ylabel('Activation Std')
ax.set_title('Signal Propagation Through Layers')
ax.legend()
ax.grid(True, alpha=0.3)

# Summary table
ax = axes[2]
ax.axis('off')
table_data = [
    ['Design Choice', 'Effect', 'When to Use'],
    ['Orthogonal base_weight', 'Preserves signal norm\nthrough deep nets', 'Deep networks (>10 layers)\nwith residual paths'],
    ['Residual base + spline', 'Prevents signal collapse,\nspline adds flexibility', 'Any learnable-activation\narchitecture'],
    ['Sigmoid grid normalization', 'Maps inputs to valid\ngrid domain [0,1]', 'Spline-based networks\nwith fixed grids'],
    ['Gradient normalization', 'Decouples lr from\ngradient magnitude', 'Networks with varying\ngradient scales'],
    ['Exclude base_weight\nfrom ES', 'Preserves orthogonality\nduring evolution', 'Hybrid evolutionary +\ngradient methods'],
    ['Antithetic sampling', 'Halves ES gradient\nvariance', 'Any evolution strategy\nestimator'],
    ['Fresh optimizer\nper generation', 'Prevents stale Adam\nstate after ES jump', 'Memetic algorithms\nwith optimizer resets'],
]

table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                  loc='center', cellLoc='left')
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 1.8)
ax.set_title('Transferable Design Insights', fontsize=12, pad=20)

plt.tight_layout()
plt.show()
```

**Step 3: Commit notebook**

```bash
git add notebooks/kan_analysis.ipynb notebooks/export_html.py
git commit -m "feat: add comprehensive analysis notebook with 6 research sections"
```

---

### Task 3: Test the analysis modules

**Files:**
- Create: `tests/test_analysis.py`

**Step 1: Write tests**

```python
"""Tests for analysis modules."""

import torch
import numpy as np
import pytest


def test_mse():
    from analysis.comparison import mse
    img1 = torch.zeros(4, 4, 3)
    img2 = torch.ones(4, 4, 3)
    assert abs(mse(img1, img2) - 1.0) < 1e-6

    # Identical images
    assert mse(img1, img1) == 0.0


def test_ssim_identical():
    from analysis.comparison import ssim
    img = torch.rand(32, 32, 3)
    s = ssim(img, img)
    assert s > 0.99  # Should be ~1.0 for identical images


def test_ssim_different():
    from analysis.comparison import ssim
    img1 = torch.zeros(32, 32, 3)
    img2 = torch.ones(32, 32, 3)
    s = ssim(img1, img2)
    assert s < 0.1  # Very different images


def test_extract_spline_curve():
    from src.kan import KANCPPNLayer
    from analysis.spline_inspector import extract_spline_curve

    layer = KANCPPNLayer(4, 8, grid_size=20)
    raw, vals, normed = extract_spline_curve(layer, in_idx=0, out_idx=0)

    assert raw.shape == (1000,)
    assert vals.shape == (1000,)
    assert normed.shape == (1000,)
    assert raw[0] == -3.0
    assert raw[-1] == 3.0
    assert np.all(normed >= 0) and np.all(normed <= 1)


def test_fit_known_function():
    from analysis.spline_inspector import fit_known_function

    # Create a sine-like curve
    x = np.linspace(-3, 3, 1000)
    y = np.sin(x)

    result = fit_known_function(x, y)
    assert result['name'] == 'sin'
    assert result['l2_distance'] < 0.1
    assert result['fitted_curve'] is not None


def test_sequence_kan():
    from analysis.text_prototype import SequenceKAN

    model = SequenceKAN(n_layers=2, hidden_size=8, output_size=1)
    positions, signal = model.generate_signal(seq_len=50)

    assert positions.shape == (50,)
    assert signal.shape == (50, 1)


def test_benchmark_loads_target():
    from analysis.benchmark import load_target_image

    img = load_target_image('skull', img_size=64)
    assert img.shape == (64, 64, 3)
    assert img.min() >= 0 and img.max() <= 1


def test_feature_cosine_similarity():
    from analysis.comparison import feature_cosine_similarity

    f1 = [torch.randn(8, 8, 4), torch.randn(8, 8, 4)]
    f2 = [torch.randn(8, 8, 4), torch.randn(8, 8, 4)]

    sims = feature_cosine_similarity(f1, f2)
    assert len(sims) == 2
    assert all(-1 <= s <= 1 for s in sims)

    # Identical features should have similarity ~1.0
    sims_same = feature_cosine_similarity(f1, f1)
    assert all(s > 0.99 for s in sims_same)
```

**Step 2: Run tests**

```bash
source .venv/Scripts/activate && python -m pytest tests/test_analysis.py -v
```

Expected: All 8 tests PASS.

**Step 3: Commit tests**

```bash
git add tests/test_analysis.py
git commit -m "test: add analysis module tests (comparison, spline inspector, text prototype, benchmark)"
```

---

### Task 4: Run the notebook and verify all cells execute

**Step 1: Install jupyter if needed**

```bash
source .venv/Scripts/activate && uv pip install jupyter nbconvert
```

**Step 2: Run notebook cells via command line**

```bash
source .venv/Scripts/activate && python -m jupyter nbconvert --to notebook --execute notebooks/kan_analysis.ipynb --output kan_analysis_executed.ipynb --ExecutePreprocessor.timeout=600
```

**Step 3: Export to HTML**

```bash
source .venv/Scripts/activate && python notebooks/export_html.py
```

**Step 4: Commit executed notebook and report**

```bash
git add notebooks/kan_analysis_executed.ipynb docs/kan_analysis_report.html
git commit -m "feat: execute analysis notebook, generate HTML report"
```

---

### Task 5: Final review and cleanup

**Step 1: Run all tests (existing + new)**

```bash
source .venv/Scripts/activate && python -m pytest tests/ -v
```

Expected: All tests pass (7 existing + 8 new = 15 total).

**Step 2: Verify notebook outputs look correct**

Open `docs/kan_analysis_report.html` in a browser and verify:
- Section 1: Convergence curves show, bar charts render, images display
- Section 2: Spline vs. known function grid shows meaningful curves
- Section 3: Multi-panel spline + sweep figures render
- Section 4: 1D signals are learned (or not), theoretical text is present
- Section 5: Algorithm visualizations show particle trajectories and perturbation pairs
- Section 6: Ablation plots show differences, summary table is readable

**Step 3: Final commit**

```bash
git add -A
git commit -m "chore: final cleanup of analysis suite"
```
