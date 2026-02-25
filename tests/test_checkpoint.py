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
        img = model.generate_image(img_size=16)
        loss = img.sum()
        loss.backward()
        optimizer.step()

        losses = [0.5, 0.4, 0.3]
        config = {'lr': 3e-3, 'n_iters': 1000, 'spline_degree': 1}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.pt')
            save_checkpoint(path, model, optimizer, losses, iteration=300, config=config)

            model2 = KAN_CPPN(n_layers=3, hidden_size=8, grid_size=10)
            optimizer2 = torch.optim.Adam(model2.parameters(), lr=3e-3)

            restored_losses, restored_iter, restored_config = load_checkpoint(
                path, model2, optimizer2
            )

            for p1, p2 in zip(model.parameters(), model2.parameters()):
                assert torch.allclose(p1, p2), "Model params don't match after load"

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

        model.layers[0].global_best_score.fill_(0.42)
        model.layers[0].velocities += 0.1

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'swarm.pt')
            save_checkpoint(path, model, optimizer, [0.5], iteration=100,
                          config={'n_particles': 3})

            model2 = SwarmKAN_CPPN(n_layers=3, hidden_size=8, grid_size=10, n_particles=3)
            optimizer2 = torch.optim.Adam(model2.parameters(), lr=3e-3)
            load_checkpoint(path, model2, optimizer2)

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

            model2 = KAN_CPPN(n_layers=3, hidden_size=8, grid_size=10)
            optimizer2 = torch.optim.Adam(model2.parameters(), lr=3e-3)
            _, restored_iter, _ = load_checkpoint(latest_path, model2, optimizer2)
            assert restored_iter == 100
