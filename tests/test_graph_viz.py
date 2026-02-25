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
        assert len(data['nodes']) == 4 + 4 + 4 + 3  # input(4) + hidden(4) + hidden(4) + output(3)
        assert len(data['edges']) == 44  # 4*4 + 4*4 + 4*3

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
            assert os.path.getsize(path) > 1000


class TestRenderFull:

    def test_renders_per_layer_files(self):
        """Full graph renders one PNG per layer pair."""
        model = KAN_CPPN(n_layers=2, hidden_size=4, grid_size=10)
        data = build_kan_graph_data(model)

        with tempfile.TemporaryDirectory() as tmpdir:
            render_full_graph_by_layer(data, tmpdir, title_prefix="Test")
            assert os.path.exists(os.path.join(tmpdir, 'layer_pair_00_01.png'))
            assert os.path.exists(os.path.join(tmpdir, 'layer_pair_01_02.png'))
            assert os.path.exists(os.path.join(tmpdir, 'layer_pair_02_03.png'))
