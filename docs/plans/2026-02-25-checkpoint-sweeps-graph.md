# Checkpointing, All-Edge Sweeps, KAN Architecture Graphs

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add checkpointing with resume capability, exhaustive per-edge weight sweep visualization, and directed architecture graphs with spline insets for all KAN variants (all degrees, swarm, memetic).

**Architecture:** New `src/checkpoint.py` module for serialization, new `sweep_all_edges()` in `src/visualize.py` for exhaustive sweeps, new `src/graph_viz.py` for directed architecture graphs. Training loops in `src/train.py` gain checkpoint/resume params. Experiment scripts wire everything together.

**Tech Stack:** PyTorch (torch.save/load), matplotlib, graphviz (Python bindings), scipy (for spline inspector reuse), numpy.

---

## Task 1: Checkpoint Module

**Files:**
- Create: `src/checkpoint.py`
- Create: `tests/test_checkpoint.py`

**Step 1: Write the failing test**

```python
# tests/test_checkpoint.py
"""Tests for checkpoint save/load/resume."""
import os
import tempfile
import torch
import numpy as np
import pytest

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.kan import KAN_CPPN
from src.swarm_kan import SwarmKAN_CPPN
from src.checkpoint import save_checkpoint, load_checkpoint


class TestCheckpointRoundtrip:
    """Verify save -> load preserves all state."""

    def test_kan_roundtrip(self):
        """KAN model state survives checkpoint roundtrip."""
        model = KAN_CPPN(n_layers=3, hidden_size=8, grid_size=10)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
        # Do a dummy forward/backward to populate optimizer state
        img = model.generate_image(img_size=16)
        loss = img.sum()
        loss.backward()
        optimizer.step()

        losses = [0.5, 0.4, 0.3]
        config = {'lr': 3e-3, 'n_iters': 1000, 'spline_degree': 1}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.pt')
            save_checkpoint(path, model, optimizer, losses, iteration=300, config=config)

            # Create fresh model and optimizer
            model2 = KAN_CPPN(n_layers=3, hidden_size=8, grid_size=10)
            optimizer2 = torch.optim.Adam(model2.parameters(), lr=3e-3)

            restored_losses, restored_iter, restored_config = load_checkpoint(
                path, model2, optimizer2
            )

            # Verify model params match
            for p1, p2 in zip(model.parameters(), model2.parameters()):
                assert torch.allclose(p1, p2), "Model params don't match after load"

            # Verify optimizer state has momentum buffers
            for key in optimizer.state_dict()['state']:
                s1 = optimizer.state_dict()['state'][key]
                s2 = optimizer2.state_dict()['state'][key]
                for k in s1:
                    if isinstance(s1[k], torch.Tensor):
                        assert torch.allclose(s1[k], s2[k]), f"Optimizer state {k} mismatch"

            assert restored_losses == losses
            assert restored_iter == 300
            assert restored_config == config

    def test_swarm_kan_roundtrip(self):
        """SwarmKAN PSO buffers survive checkpoint roundtrip."""
        model = SwarmKAN_CPPN(n_layers=3, hidden_size=8, grid_size=10, n_particles=3)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

        # Modify PSO state so it's non-default
        model.layers[0].global_best_score.fill_(0.42)
        model.layers[0].velocities += 0.1

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'swarm.pt')
            save_checkpoint(path, model, optimizer, [0.5], iteration=100,
                          config={'n_particles': 3})

            model2 = SwarmKAN_CPPN(n_layers=3, hidden_size=8, grid_size=10, n_particles=3)
            optimizer2 = torch.optim.Adam(model2.parameters(), lr=3e-3)
            load_checkpoint(path, model2, optimizer2)

            # Check PSO buffers restored
            assert torch.allclose(
                model.layers[0].global_best_score,
                model2.layers[0].global_best_score
            )
            assert torch.allclose(
                model.layers[0].velocities,
                model2.layers[0].velocities
            )

    def test_higher_degree_roundtrip(self):
        """B-spline degree 3 KAN survives checkpoint roundtrip."""
        model = KAN_CPPN(n_layers=3, hidden_size=8, grid_size=10, spline_degree=3)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'deg3.pt')
            save_checkpoint(path, model, optimizer, [0.1], iteration=50,
                          config={'spline_degree': 3})

            model2 = KAN_CPPN(n_layers=3, hidden_size=8, grid_size=10, spline_degree=3)
            optimizer2 = torch.optim.Adam(model2.parameters(), lr=3e-3)
            _, _, cfg = load_checkpoint(path, model2, optimizer2)

            assert cfg['spline_degree'] == 3
            for p1, p2 in zip(model.parameters(), model2.parameters()):
                assert torch.allclose(p1, p2)

    def test_latest_symlink(self):
        """save_checkpoint creates a latest.pt copy."""
        model = KAN_CPPN(n_layers=3, hidden_size=8, grid_size=10)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'iter_0100.pt')
            save_checkpoint(path, model, optimizer, [0.5], iteration=100, config={})

            latest_path = os.path.join(tmpdir, 'latest.pt')
            assert os.path.exists(latest_path), "latest.pt not created"

            # Load from latest should work
            model2 = KAN_CPPN(n_layers=3, hidden_size=8, grid_size=10)
            optimizer2 = torch.optim.Adam(model2.parameters(), lr=3e-3)
            _, restored_iter, _ = load_checkpoint(latest_path, model2, optimizer2)
            assert restored_iter == 100
```

**Step 2: Run test to verify it fails**

Run: `source .venv/Scripts/activate && python -m pytest tests/test_checkpoint.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.checkpoint'`

**Step 3: Write the implementation**

```python
# src/checkpoint.py
"""Checkpoint save/load for all KAN-CPPN variants.

Supports KAN_CPPN (all B-spline degrees), SwarmKAN_CPPN (preserves PSO
buffers), and MemeticKAN_CPPN (center network). Saves full training state
including optimizer, loss history, RNG states, and config.
"""

import os
import shutil
import torch
import numpy as np


def save_checkpoint(path, model, optimizer, losses, iteration, config):
    """Save a full training checkpoint.

    Args:
        path: File path for the checkpoint (e.g., 'checkpoints/iter_0100.pt').
        model: The nn.Module (KAN_CPPN, SwarmKAN_CPPN, or the center network
               of MemeticKAN_CPPN). state_dict() captures all params + buffers.
        optimizer: torch.optim.Optimizer instance.
        losses: List of loss values (full history up to this point).
        iteration: Current iteration or generation number (int).
        config: Dict of training hyperparameters for validation on resume.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': losses,
        'iteration': iteration,
        'config': config,
        'torch_rng_state': torch.get_rng_state(),
        'numpy_rng_state': np.random.get_state(),
        'model_class': type(model).__name__,
    }

    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    torch.save(checkpoint, path)

    # Copy to latest.pt in the same directory
    latest_path = os.path.join(os.path.dirname(path), 'latest.pt')
    shutil.copy2(path, latest_path)


def load_checkpoint(path, model, optimizer=None):
    """Load a training checkpoint and restore all state.

    Args:
        path: Path to the checkpoint file.
        model: An nn.Module of the same architecture as was saved.
               Its state_dict will be overwritten.
        optimizer: Optional optimizer to restore. If None, optimizer state
                   is skipped (useful for inference-only loading).

    Returns:
        Tuple of (losses, iteration, config).
    """
    checkpoint = torch.load(path, weights_only=False)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Restore RNG states for reproducibility
    if 'torch_rng_state' in checkpoint:
        torch.set_rng_state(checkpoint['torch_rng_state'])
    if 'numpy_rng_state' in checkpoint:
        np.random.set_state(checkpoint['numpy_rng_state'])

    return (
        checkpoint.get('losses', []),
        checkpoint.get('iteration', 0),
        checkpoint.get('config', {}),
    )
```

**Step 4: Run test to verify it passes**

Run: `source .venv/Scripts/activate && python -m pytest tests/test_checkpoint.py -v`
Expected: All 4 tests PASS

**Step 5: Export from `src/__init__.py`**

Add to `src/__init__.py`:
```python
from .checkpoint import save_checkpoint, load_checkpoint
```

**Step 6: Commit**

```bash
git add src/checkpoint.py tests/test_checkpoint.py src/__init__.py
git commit -m "feat: add checkpoint save/load for all KAN variants"
```

---

## Task 2: Wire Checkpointing Into Training Loops

**Files:**
- Modify: `src/train.py`
- Create: `tests/test_train_checkpoint.py`

**Step 1: Write the failing test**

```python
# tests/test_train_checkpoint.py
"""Test that training loops checkpoint and resume correctly."""
import os
import tempfile
import torch
import pytest

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.kan import KAN_CPPN
from src.swarm_kan import SwarmKAN_CPPN
from src.train import train_sgd, train_swarm


class TestTrainCheckpoint:

    def _make_target(self, img_size=16):
        """Create a simple target image."""
        return torch.rand(img_size, img_size, 3)

    def test_sgd_checkpoint_creates_files(self):
        """train_sgd with checkpoint_dir creates checkpoint files."""
        model = KAN_CPPN(n_layers=3, hidden_size=8, grid_size=10)
        target = self._make_target()

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = os.path.join(tmpdir, 'checkpoints')
            losses, _ = train_sgd(
                model, target, lr=3e-3, n_iters=250,
                log_interval=0,
                checkpoint_dir=ckpt_dir,
                checkpoint_interval=100,
            )
            assert len(losses) == 250
            assert os.path.exists(os.path.join(ckpt_dir, 'iter_0100.pt'))
            assert os.path.exists(os.path.join(ckpt_dir, 'iter_0200.pt'))
            assert os.path.exists(os.path.join(ckpt_dir, 'latest.pt'))

    def test_sgd_resume_continues_training(self):
        """train_sgd resumes from checkpoint and continues loss history."""
        target = self._make_target()

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = os.path.join(tmpdir, 'checkpoints')

            # Train for 200 iters with checkpoint at 100
            model1 = KAN_CPPN(n_layers=3, hidden_size=8, grid_size=10)
            torch.manual_seed(42)
            losses1, _ = train_sgd(
                model1, target, lr=3e-3, n_iters=200,
                log_interval=0,
                checkpoint_dir=ckpt_dir,
                checkpoint_interval=100,
            )

            # Resume from iter 100 checkpoint, train 100 more
            model2 = KAN_CPPN(n_layers=3, hidden_size=8, grid_size=10)
            losses2, _ = train_sgd(
                model2, target, lr=3e-3, n_iters=200,
                log_interval=0,
                checkpoint_dir=ckpt_dir,
                checkpoint_interval=100,
                resume_from=os.path.join(ckpt_dir, 'iter_0100.pt'),
            )
            # Should have full 200 losses (100 restored + 100 new)
            assert len(losses2) == 200

    def test_swarm_checkpoint_creates_files(self):
        """train_swarm with checkpoint_dir creates checkpoint files."""
        model = SwarmKAN_CPPN(n_layers=3, hidden_size=8, grid_size=10, n_particles=3)
        target = self._make_target()

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = os.path.join(tmpdir, 'checkpoints')
            losses, _ = train_swarm(
                model, target, lr=3e-3, n_iters=150,
                log_interval=0,
                checkpoint_dir=ckpt_dir,
                checkpoint_interval=100,
            )
            assert os.path.exists(os.path.join(ckpt_dir, 'iter_0100.pt'))
            assert os.path.exists(os.path.join(ckpt_dir, 'latest.pt'))
```

**Step 2: Run test to verify it fails**

Run: `source .venv/Scripts/activate && python -m pytest tests/test_train_checkpoint.py -v`
Expected: FAIL with `TypeError: train_sgd() got an unexpected keyword argument 'checkpoint_dir'`

**Step 3: Modify `src/train.py`**

Add checkpoint params to `train_sgd()` and `train_swarm()`. The key changes:

For `train_sgd()` — add params `checkpoint_dir=None, checkpoint_interval=100, resume_from=None`:
```python
def train_sgd(cppn, target_img, lr=3e-3, n_iters=10000, log_interval=100,
              checkpoint_dir=None, checkpoint_interval=100, resume_from=None):
```

Inside the function, after creating the optimizer, add resume logic:
```python
    losses = []
    start_iter = 0

    if resume_from is not None:
        from .checkpoint import save_checkpoint, load_checkpoint
        restored_losses, restored_iter, _ = load_checkpoint(resume_from, cppn, optimizer)
        losses = list(restored_losses)
        start_iter = restored_iter + 1

    cppn.train()
    for i in range(start_iter, n_iters):
        # ... existing training loop body (unchanged) ...

        losses.append(loss.item())

        # Checkpoint
        if checkpoint_dir is not None and (i + 1) % checkpoint_interval == 0:
            from .checkpoint import save_checkpoint
            ckpt_path = os.path.join(checkpoint_dir, f'iter_{i + 1:04d}.pt')
            save_checkpoint(ckpt_path, cppn, optimizer, losses, iteration=i,
                          config={'lr': lr, 'n_iters': n_iters})

        if log_interval and (i + 1) % log_interval == 0:
            print(f"Iter {i + 1}/{n_iters}, Loss: {loss.item():.6f}")

    return losses, cppn
```

Same pattern for `train_swarm()`.

For `train_memetic()` — add checkpoint support that saves between generations:
```python
def train_memetic(memetic, target_img, n_generations=100, sgd_steps_per_gen=50,
                  lr=3e-3, log_interval=10,
                  checkpoint_dir=None, checkpoint_interval=100, resume_from=None):
```

MemeticKAN is special: the `memetic` object owns the center network. We checkpoint `memetic.center` and restore to `memetic.center`. The `evolve()` method needs to be modified to accept `start_generation` and `initial_fitness_history`.

Modify `src/memetic_kan.py` `evolve()` to accept `start_generation=0` and `initial_fitness_history=None` params, then start the loop from `start_generation` and pre-populate fitness_history.

For `train_memetic`, resume loads into `memetic.center`:
```python
    start_gen = 0
    initial_history = None

    if resume_from is not None:
        from .checkpoint import load_checkpoint
        restored_losses, restored_iter, _ = load_checkpoint(
            resume_from, memetic.center,
        )
        start_gen = restored_iter + 1
        initial_history = list(restored_losses)
        memetic.best_fitness = min(restored_losses) if restored_losses else float('inf')

    # Wrap evolve to add checkpointing
    # ... (see implementation below)
```

The cleanest approach: add a `checkpoint_callback` to `evolve()` that gets called each generation.

**Step 4: Full implementation of modified `src/train.py`**

See the complete code in Task 2 Step 3 above. The changes are:
1. Add `import os` at top
2. Add 3 new params to each function signature
3. Add resume logic before the loop
4. Add checkpoint save inside the loop
5. Change loop range from `range(n_iters)` to `range(start_iter, n_iters)`

**Step 5: Modify `src/memetic_kan.py` evolve() to support resume**

Add `start_generation=0` and `initial_fitness_history=None` to `evolve()` signature. Change:
```python
for gen in range(n_generations):
```
to:
```python
fitness_history = list(initial_fitness_history) if initial_fitness_history else []
for gen in range(start_generation, n_generations):
```

Add optional `checkpoint_callback` param:
```python
def evolve(self, target_img, n_generations=100, sgd_steps_per_gen=50,
           lr=3e-3, log_interval=10, start_generation=0,
           initial_fitness_history=None, checkpoint_callback=None):
```

After each generation's fitness tracking, call:
```python
if checkpoint_callback is not None:
    checkpoint_callback(gen, fitness_history)
```

**Step 6: Run tests**

Run: `source .venv/Scripts/activate && python -m pytest tests/test_train_checkpoint.py -v`
Expected: All 3 tests PASS

**Step 7: Run existing tests to verify no regressions**

Run: `source .venv/Scripts/activate && python -m pytest tests/ -v`
Expected: All existing tests still PASS

**Step 8: Commit**

```bash
git add src/train.py src/memetic_kan.py tests/test_train_checkpoint.py
git commit -m "feat: add checkpoint/resume to all training loops"
```

---

## Task 3: Exhaustive Per-Edge Weight Sweeps

**Files:**
- Modify: `src/visualize.py`
- Create: `tests/test_sweep_all_edges.py`

**Step 1: Write the failing test**

```python
# tests/test_sweep_all_edges.py
"""Test exhaustive per-edge sweep visualization."""
import os
import tempfile
import torch
import pytest

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.kan import KAN_CPPN, FlattenKANParameters
from src.visualize import sweep_all_edges


class TestSweepAllEdges:

    def test_returns_correct_layer_count(self):
        """sweep_all_edges returns one entry per layer."""
        model = KAN_CPPN(n_layers=3, hidden_size=8, grid_size=10)
        flat = FlattenKANParameters(model)
        result = sweep_all_edges(model, flat, img_size=16, n_sweep=3)
        # 3 hidden layers + 1 output = 4 layers total (layers list has n_layers+1)
        assert len(result) == 4  # input->hidden, hidden->hidden x2, hidden->output

    def test_edge_count_per_layer(self):
        """Each layer entry has correct number of edge sweeps."""
        model = KAN_CPPN(n_layers=2, hidden_size=6, grid_size=10, n_inputs=4)
        flat = FlattenKANParameters(model)
        result = sweep_all_edges(model, flat, img_size=16, n_sweep=3)
        # Layer 0: 4 in * 6 out = 24 edges
        assert len(result[0]['edges']) == 24
        # Layer 1: 6 in * 6 out = 36 edges
        assert len(result[1]['edges']) == 36
        # Layer 2: 6 in * 3 out = 18 edges
        assert len(result[2]['edges']) == 18

    def test_sweep_images_shape(self):
        """Each edge sweep contains the right number of images with correct shape."""
        model = KAN_CPPN(n_layers=2, hidden_size=4, grid_size=10, n_inputs=4)
        flat = FlattenKANParameters(model)
        result = sweep_all_edges(model, flat, img_size=16, n_sweep=5)
        edge = result[0]['edges'][0]
        assert edge['imgs'].shape == (5, 16, 16, 3)

    def test_save_sweep_pages(self):
        """sweep_all_edges can save per-layer PNGs to a directory."""
        model = KAN_CPPN(n_layers=2, hidden_size=4, grid_size=10)
        flat = FlattenKANParameters(model)

        with tempfile.TemporaryDirectory() as tmpdir:
            from src.visualize import save_sweep_pages
            result = sweep_all_edges(model, flat, img_size=16, n_sweep=3)
            save_sweep_pages(result, tmpdir, title_prefix="test")

            assert os.path.exists(os.path.join(tmpdir, 'layer_00.png'))
            assert os.path.exists(os.path.join(tmpdir, 'layer_01.png'))
            assert os.path.exists(os.path.join(tmpdir, 'layer_02.png'))
```

**Step 2: Run test to verify it fails**

Run: `source .venv/Scripts/activate && python -m pytest tests/test_sweep_all_edges.py -v`
Expected: FAIL with `ImportError: cannot import name 'sweep_all_edges'`

**Step 3: Implement `sweep_all_edges()` and `save_sweep_pages()` in `src/visualize.py`**

Add these functions at the end of `src/visualize.py`:

```python
def sweep_all_edges(kan_cppn, kan_flat, img_size=64, n_sweep=5, r=1.0):
    """Generate weight sweep images for every edge in every layer.

    For each (layer, in_neuron, out_neuron) edge, perturbs the combined
    spline coefficients and generates n_sweep images.

    Args:
        kan_cppn: A KAN_CPPN (or SwarmKAN_CPPN) instance.
        kan_flat: A FlattenKANParameters instance wrapping the model.
        img_size: Resolution of sweep images (smaller = faster).
        n_sweep: Number of images per sweep.
        r: Perturbation radius.

    Returns:
        List of dicts (one per layer), each with:
            layer_idx: int
            in_features: int
            out_features: int
            edges: list of dicts, each with:
                in_idx: int
                out_idx: int
                imgs: tensor (n_sweep, img_size, img_size, 3)
    """
    results = []
    params = kan_flat.flatten()
    sweep_scales = torch.linspace(-r, r, n_sweep)

    with torch.no_grad():
        for layer_idx, layer in enumerate(kan_cppn.layers):
            layer_result = {
                'layer_idx': layer_idx,
                'in_features': layer.in_features,
                'out_features': layer.out_features,
                'edges': [],
            }

            for out_idx in range(layer.out_features):
                for in_idx in range(layer.in_features):
                    # Get the flat indices for this edge's coefficients
                    edge_coeffs = layer.coeffs[out_idx, in_idx]
                    n_basis = edge_coeffs.numel()

                    # Find flat offset of this specific coeffs slice
                    offset = _find_coeffs_offset(kan_cppn, layer_idx, out_idx, in_idx)

                    # Direction: perturb all coefficients for this edge equally
                    direction = torch.zeros_like(params)
                    direction[offset:offset + n_basis] = 1.0 / (n_basis ** 0.5)

                    imgs = []
                    for scale in sweep_scales:
                        p = params.clone()
                        p += scale * direction
                        img = kan_flat.generate_image(p, img_size=img_size)
                        imgs.append(img)

                    layer_result['edges'].append({
                        'in_idx': in_idx,
                        'out_idx': out_idx,
                        'imgs': torch.stack(imgs),
                    })

            results.append(layer_result)

    # Restore original params
    kan_flat.unflatten(params)
    return results


def _find_coeffs_offset(kan_cppn, target_layer_idx, out_idx, in_idx):
    """Find the flat parameter offset for a specific edge's coefficients.

    Args:
        kan_cppn: KAN_CPPN model.
        target_layer_idx: Which layer.
        out_idx: Output neuron index.
        in_idx: Input neuron index.

    Returns:
        Integer offset into the flattened parameter vector.
    """
    offset = 0
    for name, param in kan_cppn.named_parameters():
        parts = name.split('.')
        layer_idx = int(parts[1])
        param_type = parts[2]

        if layer_idx == target_layer_idx and param_type == 'coeffs':
            # coeffs shape: (out_features, in_features, n_basis)
            n_basis = param.shape[2]
            in_features = param.shape[1]
            # Flat index within this tensor: out_idx * (in_features * n_basis) + in_idx * n_basis
            local_offset = out_idx * (in_features * n_basis) + in_idx * n_basis
            return offset + local_offset

        offset += param.numel()

    raise ValueError(f"Could not find coeffs for layer {target_layer_idx}")


def save_sweep_pages(sweep_results, output_dir, title_prefix="", img_size_display=None):
    """Save per-layer sweep grids as PNG files.

    Args:
        sweep_results: Output from sweep_all_edges().
        output_dir: Directory to save PNG files.
        title_prefix: Prefix for figure titles.
        img_size_display: Display size per sweep image (None = auto).
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    for layer_data in sweep_results:
        layer_idx = layer_data['layer_idx']
        edges = layer_data['edges']
        n_edges = len(edges)
        n_sweep = edges[0]['imgs'].shape[0] if edges else 0

        if n_edges == 0:
            continue

        # Create a figure with one row per edge
        cell_size = 0.6  # inches per image cell
        fig_width = n_sweep * cell_size + 2  # extra for labels
        fig_height = n_edges * cell_size + 1.5  # extra for title

        fig, axs = plt.subplots(
            n_edges, n_sweep,
            figsize=(fig_width, fig_height),
            dpi=100,
            squeeze=False,
        )

        for row, edge in enumerate(edges):
            imgs = edge['imgs']
            if isinstance(imgs, torch.Tensor):
                imgs = imgs.detach().cpu().numpy()
            for col in range(n_sweep):
                ax = axs[row, col]
                ax.imshow(imgs[col])
                ax.set_xticks([])
                ax.set_yticks([])
                if col == 0:
                    ax.set_ylabel(f"({edge['in_idx']}->{edge['out_idx']})",
                                  fontsize=6, rotation=0, labelpad=30, va='center')

        prefix = f"{title_prefix} " if title_prefix else ""
        fig.suptitle(
            f"{prefix}Layer {layer_idx} "
            f"({layer_data['in_features']} in x {layer_data['out_features']} out = {n_edges} edges)",
            fontsize=10,
        )
        fig.tight_layout()
        fig.savefig(
            os.path.join(output_dir, f'layer_{layer_idx:02d}.png'),
            bbox_inches='tight',
        )
        plt.close(fig)
```

**Step 4: Export the new functions in `src/__init__.py`**

Add `sweep_all_edges, save_sweep_pages` to the visualize import line.

**Step 5: Run tests**

Run: `source .venv/Scripts/activate && python -m pytest tests/test_sweep_all_edges.py -v`
Expected: All 4 tests PASS

**Step 6: Run existing tests for regressions**

Run: `source .venv/Scripts/activate && python -m pytest tests/ -v`
Expected: All tests PASS

**Step 7: Commit**

```bash
git add src/visualize.py src/__init__.py tests/test_sweep_all_edges.py
git commit -m "feat: add exhaustive per-edge weight sweep visualization"
```

---

## Task 4: KAN Architecture Directed Graph with Spline Insets

**Files:**
- Create: `src/graph_viz.py`
- Create: `tests/test_graph_viz.py`

**Step 1: Write the failing test**

```python
# tests/test_graph_viz.py
"""Test KAN architecture graph visualization."""
import os
import tempfile
import torch
import pytest

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.kan import KAN_CPPN
from src.graph_viz import (
    build_kan_graph_data,
    render_pruned_graph,
    render_full_graph_by_layer,
)


class TestGraphData:

    def test_build_graph_data_structure(self):
        """build_kan_graph_data returns nodes and edges with spline info."""
        model = KAN_CPPN(n_layers=2, hidden_size=4, grid_size=10, n_inputs=4)
        data = build_kan_graph_data(model)

        assert 'nodes' in data
        assert 'edges' in data
        # 3 layers of nodes: input(4), hidden(4), hidden(4), output(3)
        assert len(data['nodes']) == 4 + 4 + 4 + 3

        # Edges: layer0(4*4=16) + layer1(4*4=16) + layer2(4*3=12) = 44
        assert len(data['edges']) == 44

        # Each edge should have spline curve and best-fit info
        edge = data['edges'][0]
        assert 'raw_inputs' in edge
        assert 'spline_values' in edge
        assert 'best_fit_name' in edge
        assert 'best_fit_score' in edge
        assert 'visual_impact' in edge


class TestRenderPruned:

    def test_renders_without_error(self):
        """Pruned graph renders to a file without crashing."""
        model = KAN_CPPN(n_layers=2, hidden_size=4, grid_size=10)
        data = build_kan_graph_data(model)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'pruned.png')
            render_pruned_graph(data, path, title="Test Pruned")
            assert os.path.exists(path)
            assert os.path.getsize(path) > 1000  # not empty


class TestRenderFull:

    def test_renders_per_layer_files(self):
        """Full graph renders one PNG per layer pair."""
        model = KAN_CPPN(n_layers=2, hidden_size=4, grid_size=10)
        data = build_kan_graph_data(model)

        with tempfile.TemporaryDirectory() as tmpdir:
            render_full_graph_by_layer(data, tmpdir, title_prefix="Test")
            # 3 KAN layers = 3 layer-pair files
            assert os.path.exists(os.path.join(tmpdir, 'layer_pair_00_01.png'))
            assert os.path.exists(os.path.join(tmpdir, 'layer_pair_01_02.png'))
            assert os.path.exists(os.path.join(tmpdir, 'layer_pair_02_03.png'))
```

**Step 2: Run test to verify it fails**

Run: `source .venv/Scripts/activate && python -m pytest tests/test_graph_viz.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.graph_viz'`

**Step 3: Implement `src/graph_viz.py`**

```python
# src/graph_viz.py
"""Directed graph visualization of KAN architectures with spline insets.

Generates two types of architecture visualizations:
1. Pruned graph: shows only high-impact edges with mini spline plots
2. Full graph by layer: one bipartite graph per layer transition with all edges

Uses matplotlib for rendering (no external graphviz dependency required).
"""

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from analysis.spline_inspector import extract_spline_curve, fit_known_function


def build_kan_graph_data(kan_cppn, impact_img_size=32):
    """Extract graph structure + spline data from a trained KAN-CPPN.

    Args:
        kan_cppn: A KAN_CPPN (or SwarmKAN_CPPN) instance.
        impact_img_size: Image size for visual impact computation.

    Returns:
        Dict with:
            nodes: list of dicts (layer_idx, neuron_idx, label)
            edges: list of dicts (src_layer, src_neuron, dst_layer, dst_neuron,
                   raw_inputs, spline_values, best_fit_name, best_fit_score,
                   visual_impact)
            n_layers: int (number of node layers = KAN layers + 1)
    """
    layers = kan_cppn.layers
    n_kan_layers = len(layers)

    # Build nodes: one column per "node layer" (inputs + one per KAN layer output)
    nodes = []
    # Input nodes
    input_labels = ['y', 'x', 'd', 'b']
    for i in range(layers[0].in_features):
        label = input_labels[i] if i < len(input_labels) else f'in_{i}'
        nodes.append({'layer_idx': 0, 'neuron_idx': i, 'label': label})
    # Hidden + output nodes
    for l_idx, layer in enumerate(layers):
        node_layer = l_idx + 1
        for n_idx in range(layer.out_features):
            if l_idx == n_kan_layers - 1:
                out_labels = ['h', 's', 'v']
                label = out_labels[n_idx] if n_idx < len(out_labels) else f'out_{n_idx}'
            else:
                label = f'h{node_layer}_{n_idx}'
            nodes.append({'layer_idx': node_layer, 'neuron_idx': n_idx, 'label': label})

    # Build edges with spline data
    edges = []
    for l_idx, layer in enumerate(layers):
        for out_idx in range(layer.out_features):
            for in_idx in range(layer.in_features):
                raw_inputs, spline_values, _ = extract_spline_curve(layer, in_idx, out_idx, n_points=200)
                best_match = fit_known_function(raw_inputs, spline_values)

                # Visual impact: RMS of spline values
                impact = float(np.sqrt(np.mean(spline_values ** 2)))

                edges.append({
                    'src_layer': l_idx,
                    'src_neuron': in_idx,
                    'dst_layer': l_idx + 1,
                    'dst_neuron': out_idx,
                    'kan_layer_idx': l_idx,
                    'raw_inputs': raw_inputs,
                    'spline_values': spline_values,
                    'best_fit_name': best_match['name'],
                    'best_fit_score': best_match['l2_distance'],
                    'fitted_curve': best_match['fitted_curve'],
                    'visual_impact': impact,
                })

    return {
        'nodes': nodes,
        'edges': edges,
        'n_node_layers': n_kan_layers + 1,
    }


def render_pruned_graph(graph_data, output_path, title="KAN Architecture",
                        percentile_threshold=50):
    """Render a pruned architecture graph with only high-impact edges.

    Only edges above the given percentile of visual impact are shown.
    Each shown edge has a mini spline plot and function name label.

    Args:
        graph_data: Output from build_kan_graph_data().
        output_path: Path to save the PNG.
        title: Figure title.
        percentile_threshold: Only show edges above this percentile of impact.
    """
    edges = graph_data['edges']
    nodes = graph_data['nodes']
    n_node_layers = graph_data['n_node_layers']

    # Compute impact threshold
    impacts = [e['visual_impact'] for e in edges]
    threshold = np.percentile(impacts, percentile_threshold)
    max_impact = max(impacts) if impacts else 1.0

    # Filter edges
    visible_edges = [e for e in edges if e['visual_impact'] >= threshold]

    # Layout: x = layer, y = neuron (centered)
    layer_sizes = {}
    for node in nodes:
        l = node['layer_idx']
        layer_sizes[l] = layer_sizes.get(l, 0) + 1

    max_layer_size = max(layer_sizes.values())

    def node_pos(layer_idx, neuron_idx):
        size = layer_sizes[layer_idx]
        y = (neuron_idx - (size - 1) / 2) * 1.0
        x = layer_idx * 3.0
        return x, y

    fig_width = n_node_layers * 3 + 2
    fig_height = max_layer_size * 1.0 + 2
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=150)

    # Draw edges
    for edge in visible_edges:
        x0, y0 = node_pos(edge['src_layer'], edge['src_neuron'])
        x1, y1 = node_pos(edge['dst_layer'], edge['dst_neuron'])

        # Edge thickness proportional to impact
        lw = 0.3 + 2.0 * (edge['visual_impact'] / max_impact)
        alpha = 0.3 + 0.7 * (edge['visual_impact'] / max_impact)

        ax.plot([x0, x1], [y0, y1], '-', color='steelblue', linewidth=lw, alpha=alpha)

        # Mini spline inset at edge midpoint
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        inset_size = 0.3
        inset_ax = ax.inset_axes(
            [mx - inset_size/2, my - inset_size/2, inset_size, inset_size],
            transform=ax.transData,
        )
        raw = edge['raw_inputs']
        vals = edge['spline_values']
        inset_ax.plot(raw, vals, 'b-', linewidth=0.5)
        if edge['fitted_curve'] is not None:
            inset_ax.plot(raw, edge['fitted_curve'], 'r--', linewidth=0.3, alpha=0.5)
        inset_ax.set_xticks([])
        inset_ax.set_yticks([])
        inset_ax.patch.set_alpha(0.8)
        inset_ax.patch.set_facecolor('white')
        for spine in inset_ax.spines.values():
            spine.set_linewidth(0.3)

        # Function name label
        label = f"{edge['best_fit_name']}"
        ax.text(mx, my - inset_size/2 - 0.05, label,
                ha='center', va='top', fontsize=3, color='dimgray')

    # Draw nodes
    for node in nodes:
        x, y = node_pos(node['layer_idx'], node['neuron_idx'])
        circle = plt.Circle((x, y), 0.15, facecolor='lightcoral', edgecolor='black',
                           linewidth=0.5, zorder=5)
        ax.add_patch(circle)
        ax.text(x, y, node['label'], ha='center', va='center', fontsize=4, zorder=6)

    ax.set_xlim(-1, n_node_layers * 3)
    ax.set_ylim(-max_layer_size / 2 - 1, max_layer_size / 2 + 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=12)

    fig.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close(fig)


def render_full_graph_by_layer(graph_data, output_dir, title_prefix=""):
    """Render one bipartite graph per layer transition with ALL edges.

    Each sub-figure shows the connections between two adjacent node layers,
    with a mini spline plot on every edge.

    Args:
        graph_data: Output from build_kan_graph_data().
        output_dir: Directory to save PNG files.
        title_prefix: Prefix for figure titles.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    edges = graph_data['edges']
    nodes = graph_data['nodes']

    # Group edges by KAN layer
    edges_by_layer = {}
    for edge in edges:
        l = edge['kan_layer_idx']
        edges_by_layer.setdefault(l, []).append(edge)

    # Group nodes by layer
    nodes_by_layer = {}
    for node in nodes:
        l = node['layer_idx']
        nodes_by_layer.setdefault(l, []).append(node)

    for kan_layer_idx, layer_edges in sorted(edges_by_layer.items()):
        src_layer = kan_layer_idx
        dst_layer = kan_layer_idx + 1
        src_nodes = nodes_by_layer.get(src_layer, [])
        dst_nodes = nodes_by_layer.get(dst_layer, [])

        n_src = len(src_nodes)
        n_dst = len(dst_nodes)
        n_edges = len(layer_edges)

        max_impact = max(e['visual_impact'] for e in layer_edges) if layer_edges else 1.0

        # Layout: two columns
        spacing = 1.2
        col_gap = 4.0

        fig_height = max(n_src, n_dst) * spacing + 2
        fig_width = col_gap + 4
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=150)

        def src_pos(i):
            return 0, (i - (n_src - 1) / 2) * spacing

        def dst_pos(i):
            return col_gap, (i - (n_dst - 1) / 2) * spacing

        # Draw edges with spline insets
        for edge in layer_edges:
            x0, y0 = src_pos(edge['src_neuron'])
            x1, y1 = dst_pos(edge['dst_neuron'])

            lw = 0.2 + 1.5 * (edge['visual_impact'] / max_impact)
            alpha = 0.2 + 0.6 * (edge['visual_impact'] / max_impact)
            ax.plot([x0, x1], [y0, y1], '-', color='steelblue', linewidth=lw, alpha=alpha)

            # Mini spline at midpoint
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            inset_size = 0.4
            inset_ax = ax.inset_axes(
                [mx - inset_size/2, my - inset_size/2, inset_size, inset_size],
                transform=ax.transData,
            )
            inset_ax.plot(edge['raw_inputs'], edge['spline_values'], 'b-', linewidth=0.4)
            if edge['fitted_curve'] is not None:
                inset_ax.plot(edge['raw_inputs'], edge['fitted_curve'], 'r--', linewidth=0.2, alpha=0.5)
            inset_ax.set_xticks([])
            inset_ax.set_yticks([])
            inset_ax.patch.set_alpha(0.85)
            inset_ax.patch.set_facecolor('white')
            for spine in inset_ax.spines.values():
                spine.set_linewidth(0.2)

            # Label
            ax.text(mx, my - inset_size/2 - 0.03,
                    f"{edge['best_fit_name']}",
                    ha='center', va='top', fontsize=3, color='dimgray')

        # Draw source nodes
        for i, node in enumerate(src_nodes):
            x, y = src_pos(i)
            circle = plt.Circle((x, y), 0.2, facecolor='lightcoral', edgecolor='black',
                               linewidth=0.5, zorder=5)
            ax.add_patch(circle)
            ax.text(x, y, node['label'], ha='center', va='center', fontsize=5, zorder=6)

        # Draw destination nodes
        for i, node in enumerate(dst_nodes):
            x, y = dst_pos(i)
            circle = plt.Circle((x, y), 0.2, facecolor='lightskyblue', edgecolor='black',
                               linewidth=0.5, zorder=5)
            ax.add_patch(circle)
            ax.text(x, y, node['label'], ha='center', va='center', fontsize=5, zorder=6)

        ax.set_xlim(-1, col_gap + 1)
        ax.set_ylim(-max(n_src, n_dst) * spacing / 2 - 1,
                     max(n_src, n_dst) * spacing / 2 + 1)
        ax.set_aspect('equal')
        ax.axis('off')

        prefix = f"{title_prefix} " if title_prefix else ""
        ax.set_title(f"{prefix}Layer {src_layer} -> {dst_layer} ({n_edges} edges)", fontsize=10)

        filename = f'layer_pair_{src_layer:02d}_{dst_layer:02d}.png'
        fig.savefig(os.path.join(output_dir, filename), bbox_inches='tight', dpi=150)
        plt.close(fig)
```

**Step 4: Export from `src/__init__.py`**

Add to `src/__init__.py`:
```python
from .graph_viz import build_kan_graph_data, render_pruned_graph, render_full_graph_by_layer
```

**Step 5: Run tests**

Run: `source .venv/Scripts/activate && python -m pytest tests/test_graph_viz.py -v`
Expected: All 3 tests PASS

**Step 6: Run all tests for regressions**

Run: `source .venv/Scripts/activate && python -m pytest tests/ -v`
Expected: All tests PASS

**Step 7: Commit**

```bash
git add src/graph_viz.py src/__init__.py tests/test_graph_viz.py
git commit -m "feat: add KAN directed graph visualization with spline insets"
```

---

## Task 5: Wire Everything Into Experiment Scripts

**Files:**
- Modify: `experiments/phase2_kan_cppn.py`
- Modify: `experiments/phase3_swarm_kan.py`
- Modify: `experiments/phase4_memetic_kan.py`

**Step 1: Modify `experiments/phase2_kan_cppn.py`**

Add to argparse:
```python
parser.add_argument('--checkpoint_interval', type=int, default=100,
                    help="Checkpoint every N iterations (default: 100)")
parser.add_argument('--resume_from', type=str, default=None,
                    help="Path to checkpoint to resume from")
parser.add_argument('--spline_degree', type=int, default=1,
                    help="B-spline degree (1-4, default: 1)")
```

Modify `run_genome()`:
- Add `checkpoint_interval`, `resume_from`, `spline_degree` params
- Pass `spline_degree` to `KAN_CPPN()`
- Pass `checkpoint_dir`, `checkpoint_interval`, `resume_from` to `train_sgd()`
- After training, call `sweep_all_edges()` + `save_sweep_pages()` to generate exhaustive sweeps
- After training, call `build_kan_graph_data()` + `render_pruned_graph()` + `render_full_graph_by_layer()`

Add these blocks after the existing weight sweep section:

```python
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
```

**Step 2: Modify `experiments/phase3_swarm_kan.py`**

Same pattern: add `--checkpoint_interval`, `--resume_from`, `--spline_degree` args.
Pass checkpoint params to `train_sgd()` and `train_swarm()`.
Add exhaustive sweeps + graph viz for both vanilla KAN and SwarmKAN models.

**Step 3: Modify `experiments/phase4_memetic_kan.py`**

Same pattern for memetic. Pass checkpoint params to `train_memetic()`.
Add exhaustive sweeps + graph viz for the best memetic individual.

**Step 4: Update imports in all experiment files**

Add to each experiment's imports:
```python
from src import sweep_all_edges, save_sweep_pages
from src import build_kan_graph_data, render_pruned_graph, render_full_graph_by_layer
```

**Step 5: Test experiment scripts run without error**

Run: `source .venv/Scripts/activate && python experiments/phase2_kan_cppn.py --genome skull --n_iters 50 --img_size 32 --checkpoint_interval 25 --output_dir output/test_phase2`
Expected: Completes without error, creates checkpoint files, sweep PNGs, and graph PNGs.

**Step 6: Clean up test output**

```bash
rm -rf output/test_phase2
```

**Step 7: Run all tests one final time**

Run: `source .venv/Scripts/activate && python -m pytest tests/ -v`
Expected: All tests PASS

**Step 8: Commit**

```bash
git add experiments/phase2_kan_cppn.py experiments/phase3_swarm_kan.py experiments/phase4_memetic_kan.py
git commit -m "feat: wire checkpointing, all-edge sweeps, and graphs into experiments"
```

---

## Dependency Graph

```
Task 1 (checkpoint module)
    └──> Task 2 (wire into training loops)
             └──> Task 5 (wire into experiments)

Task 3 (sweep_all_edges) ──> Task 5
Task 4 (graph_viz) ──> Task 5
```

Tasks 1, 3, and 4 are independent and can be done in parallel.
Task 2 depends on Task 1.
Task 5 depends on Tasks 2, 3, and 4.
