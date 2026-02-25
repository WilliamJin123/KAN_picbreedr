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
        assert len(result) == 4  # input->hidden, hidden->hidden x2, hidden->output

    def test_edge_count_per_layer(self):
        """Each layer entry has correct number of edge sweeps."""
        model = KAN_CPPN(n_layers=2, hidden_size=6, grid_size=10, n_inputs=4)
        flat = FlattenKANParameters(model)
        result = sweep_all_edges(model, flat, img_size=16, n_sweep=3)
        assert len(result[0]['edges']) == 24  # 4*6
        assert len(result[1]['edges']) == 36  # 6*6
        assert len(result[2]['edges']) == 18  # 6*3

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
