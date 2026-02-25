# src/checkpoint.py
"""Checkpoint save/load for all KAN-CPPN variants."""

import os
import shutil
import torch
import numpy as np


def save_checkpoint(path, model, optimizer, losses, iteration, config):
    """Save a full training checkpoint."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
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
    """Load a training checkpoint and restore all state."""
    checkpoint = torch.load(path, weights_only=False)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if 'torch_rng_state' in checkpoint:
        torch.set_rng_state(checkpoint['torch_rng_state'])
    if 'numpy_rng_state' in checkpoint:
        np.random.set_state(checkpoint['numpy_rng_state'])

    return (
        checkpoint.get('losses', []),
        checkpoint.get('iteration', 0),
        checkpoint.get('config', {}),
    )
