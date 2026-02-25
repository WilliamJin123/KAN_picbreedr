# Design: Checkpointing, All-Edge Sweeps, KAN Architecture Graphs

**Date**: 2026-02-25
**Status**: Approved

## Features

### 1. Checkpointing System

**New module**: `src/checkpoint.py`

Provides `save_checkpoint()` and `load_checkpoint()` that serialize full training state:
- Model `state_dict()` (covers all param types: base_weight, spline coeffs, spline weights, SwarmKAN PSO buffers)
- Optimizer `state_dict()` (Adam momentum/variance)
- Loss history list
- Current iteration/generation number
- Training config dict (lr, n_iters, architecture, spline_degree, etc.)
- RNG states (torch, numpy)
- Model class name (for validation on load)

**Training loop changes** (`src/train.py`):

All three functions (`train_sgd`, `train_swarm`, `train_memetic`) gain:
- `checkpoint_dir: str = None` - directory for checkpoint files (None = no checkpointing)
- `checkpoint_interval: int = 100` - save every N iterations/generations
- `resume_from: str = None` - path to checkpoint file to resume from

Resume behavior:
1. Load checkpoint, restore model + optimizer + RNG states
2. Pre-populate losses list from checkpoint
3. Start loop from `iteration + 1`
4. Continue saving checkpoints at the configured interval

Directory layout:
```
output/<phase>/checkpoints/
    iter_0100.pt
    iter_0200.pt
    ...
    latest.pt
```

**Applies to**: KAN (all B-spline degrees 1-4), SwarmKAN, MemeticKAN.

### 2. All-Edge Weight Sweeps

**New function in `src/visualize.py`**: `sweep_all_edges()`

For each layer in the KAN, for each (input_neuron, output_neuron) edge:
- Perturb that edge's combined spline coefficients
- Generate 5 images at perturbation scales [-1, -0.5, 0, +0.5, +1]
- Arrange as a horizontal strip

Output: one tall PNG per layer (`sweeps/layer_NN.png`), with all edges stacked vertically.
- Row labels: `(in -> out)` index
- Image resolution: 64x64 per frame
- Layer 0: 4*22 = 88 rows. Middle layers: 22*22 = 484 rows. Output layer: 22*3 = 66 rows.

**Applies to**: all KAN variants, all B-spline degrees.

### 3. KAN Architecture Directed Graph

**New module**: `src/graph_viz.py`

Two versions generated for each trained KAN:

**Pruned version** (readable overview):
- Graphviz dot layout for node positions
- Only edges above median visual impact shown
- Each visible edge gets a mini spline curve inset (40x40px) + best-fit function name label
- Edge thickness proportional to impact
- Output: ~2000x4000px PNG

**Full version** (complete, split by layer pair):
- One bipartite sub-graph per `(layer_i -> layer_i+1)` transition
- All edges shown with mini spline insets and function labels
- Each sub-graph is reasonably sized since it only covers one layer transition
- Output: directory of PNGs, one per layer pair

Spline extraction via `analysis/spline_inspector.py`. Function fitting to known functions
(identity, sin, cos, tanh, sigmoid, gaussian, relu, quadratic, abs, constant).

**Applies to**: all KAN variants, all B-spline degrees.

## Files

### New
- `src/checkpoint.py`
- `src/graph_viz.py`

### Modified
- `src/train.py` - checkpoint params on all 3 training functions
- `src/visualize.py` - `sweep_all_edges()` function
- `experiments/phase2_kan_cppn.py` - wire checkpointing + new viz
- `experiments/phase3_swarm_kan.py` - wire checkpointing + new viz
- `experiments/phase4_memetic_kan.py` - wire checkpointing + new viz
