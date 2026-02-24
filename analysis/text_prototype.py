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
