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

            model1 = KAN_CPPN(n_layers=3, hidden_size=8, grid_size=10)
            torch.manual_seed(42)
            losses1, _ = train_sgd(
                model1, target, lr=3e-3, n_iters=200,
                log_interval=0,
                checkpoint_dir=ckpt_dir,
                checkpoint_interval=100,
            )

            model2 = KAN_CPPN(n_layers=3, hidden_size=8, grid_size=10)
            losses2, _ = train_sgd(
                model2, target, lr=3e-3, n_iters=200,
                log_interval=0,
                checkpoint_dir=ckpt_dir,
                checkpoint_interval=100,
                resume_from=os.path.join(ckpt_dir, 'iter_0100.pt'),
            )
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
