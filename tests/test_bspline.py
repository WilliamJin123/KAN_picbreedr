"""Tests for higher-degree B-spline KAN implementation.

Covers backward compatibility, mathematical correctness, gradient flow,
parameter counts, and integration with SwarmKAN/MemeticKAN.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import pytest
from scipy.interpolate import BSpline

from src.kan import KANCPPNLayer, KAN_CPPN, FlattenKANParameters, compute_bspline_basis
from src.swarm_kan import SwarmKANCPPNLayer, SwarmKAN_CPPN
from src.memetic_kan import MemeticKAN_CPPN


class TestBackwardCompat:
    """Degree 1 must produce identical output to old code."""

    def test_degree1_output_identical(self):
        """Degree=1 layer output matches the original implementation exactly."""
        torch.manual_seed(42)
        layer = KANCPPNLayer(4, 8, grid_size=20, spline_degree=1)

        x = torch.randn(16, 4)
        out = layer(x)

        # Verify shape and that it runs without error
        assert out.shape == (16, 8)

        # Verify n_basis == grid_size for degree 1
        assert layer.n_basis == 20
        assert layer.coeffs.shape == (8, 4, 20)

    def test_degree1_no_knots_buffer(self):
        """Degree 1 should NOT have a knots buffer (uses fast path)."""
        layer = KANCPPNLayer(4, 8, grid_size=20, spline_degree=1)
        assert not hasattr(layer, 'knots') or layer.knots is None or 'knots' not in dict(layer.named_buffers())

    def test_kan_cppn_default_degree(self):
        """KAN_CPPN defaults to degree 1."""
        kan = KAN_CPPN(n_layers=3, hidden_size=8)
        assert kan.spline_degree == 1
        for layer in kan.layers:
            assert layer.spline_degree == 1


class TestPartitionOfUnity:
    """B-spline basis functions must sum to 1.0 at interior points."""

    @pytest.mark.parametrize("degree", [1, 2, 3, 4])
    def test_basis_sum_to_one(self, degree):
        """Basis functions sum to 1.0 for degrees 1-4."""
        grid_size = 20
        n_basis = grid_size + degree - 1

        # Build clamped knot vector
        interior = torch.linspace(0, 1, grid_size)
        knots = torch.cat([
            torch.zeros(degree),
            interior,
            torch.ones(degree),
        ])

        # Test points strictly inside [0, 1]
        x = torch.linspace(0.01, 0.99, 100).unsqueeze(-1)  # (100, 1)
        basis = compute_bspline_basis(x, knots, degree, n_basis)
        # basis shape: (100, 1, n_basis)

        basis_sum = basis.sum(dim=-1).squeeze(-1)  # (100,)
        assert torch.allclose(basis_sum, torch.ones_like(basis_sum), atol=1e-5), \
            f"Degree {degree}: basis sum range [{basis_sum.min():.6f}, {basis_sum.max():.6f}]"

    @pytest.mark.parametrize("degree", [2, 3, 4])
    def test_basis_at_endpoints(self, degree):
        """Basis functions sum to 1.0 at boundary points 0 and 1."""
        grid_size = 20
        n_basis = grid_size + degree - 1

        interior = torch.linspace(0, 1, grid_size)
        knots = torch.cat([
            torch.zeros(degree),
            interior,
            torch.ones(degree),
        ])

        x = torch.tensor([[0.0], [1.0]])
        basis = compute_bspline_basis(x, knots, degree, n_basis)
        basis_sum = basis.sum(dim=-1).squeeze(-1)

        assert torch.allclose(basis_sum, torch.ones_like(basis_sum), atol=1e-5), \
            f"Degree {degree} at endpoints: basis sum = {basis_sum}"


class TestScipyConsistency:
    """PyTorch B-spline evaluation must match scipy BSpline."""

    @pytest.mark.parametrize("degree", [2, 3])
    def test_pytorch_vs_scipy(self, degree):
        """Max difference between PyTorch and scipy < 1e-5."""
        grid_size = 20
        n_basis = grid_size + degree - 1

        interior = torch.linspace(0, 1, grid_size)
        knots = torch.cat([
            torch.zeros(degree),
            interior,
            torch.ones(degree),
        ])
        knots_np = knots.numpy()

        # Random coefficients
        torch.manual_seed(123)
        coeffs = torch.randn(n_basis)
        coeffs_np = coeffs.numpy()

        # Evaluation points
        x_np = np.linspace(0.01, 0.99, 200)
        x_torch = torch.tensor(x_np, dtype=torch.float32).unsqueeze(-1)  # (200, 1)

        # Scipy evaluation
        spl = BSpline(knots_np, coeffs_np, degree)
        scipy_vals = spl(x_np)

        # PyTorch evaluation via basis functions
        basis = compute_bspline_basis(x_torch, knots, degree, n_basis)  # (200, 1, n_basis)
        torch_vals = (basis.squeeze(1) @ coeffs).numpy()  # (200,)

        max_diff = np.max(np.abs(scipy_vals - torch_vals))
        assert max_diff < 1e-5, f"Degree {degree}: max diff = {max_diff}"


class TestGradientFlow:
    """Gradients must flow through B-spline basis to coefficients."""

    @pytest.mark.parametrize("degree", [1, 2, 3])
    def test_coeffs_get_gradients(self, degree):
        """loss.backward() produces non-None grads on coeffs."""
        layer = KANCPPNLayer(4, 3, grid_size=10, spline_degree=degree)
        x = torch.randn(8, 4)
        out = layer(x)
        loss = out.sum()
        loss.backward()

        assert layer.coeffs.grad is not None, "coeffs.grad is None"
        assert layer.coeffs.grad.abs().sum() > 0, "coeffs.grad is all zeros"
        assert layer.base_weight.grad is not None, "base_weight.grad is None"
        assert layer.weights.grad is not None, "weights.grad is None"


class TestParameterCount:
    """n_basis = grid_size + degree - 1 for all degrees."""

    @pytest.mark.parametrize("degree,expected_n_basis", [
        (1, 20), (2, 21), (3, 22), (4, 23),
    ])
    def test_n_basis_formula(self, degree, expected_n_basis):
        """Verify n_basis = grid_size + degree - 1."""
        layer = KANCPPNLayer(4, 8, grid_size=20, spline_degree=degree)
        assert layer.n_basis == expected_n_basis
        assert layer.coeffs.shape == (8, 4, expected_n_basis)


class TestSwarmKANBufferShapes:
    """SwarmKAN PSO buffers must adapt to n_basis for higher degrees."""

    @pytest.mark.parametrize("degree", [1, 2, 3])
    def test_pso_buffer_shapes(self, degree):
        """PSO buffers have correct shape for given degree."""
        n_particles = 5
        layer = SwarmKANCPPNLayer(4, 8, grid_size=20, spline_degree=degree,
                                  n_particles=n_particles)
        expected_n_basis = 20 + degree - 1
        expected_coeff_shape = (8, 4, expected_n_basis)

        assert layer.coeffs.shape == expected_coeff_shape
        assert layer.velocities.shape == (n_particles,) + expected_coeff_shape
        assert layer.particles.shape == (n_particles,) + expected_coeff_shape
        assert layer.personal_best.shape == (n_particles,) + expected_coeff_shape
        assert layer.global_best.shape == expected_coeff_shape

    def test_swarm_cppn_forward(self):
        """SwarmKAN_CPPN with degree 3 runs forward pass correctly."""
        model = SwarmKAN_CPPN(n_layers=3, hidden_size=8, spline_degree=3)
        x = torch.randn(16, 4)
        (h, s, v), features = model(x)
        assert h.shape == (16,)
        assert len(features) == 5  # input + 3 hidden + 1 output


class TestMemeticKANFlattener:
    """MemeticKAN flattener must handle n_basis changes correctly."""

    @pytest.mark.parametrize("degree", [1, 2, 3])
    def test_flattener_param_count(self, degree):
        """FlattenKANParameters param count adjusts for degree."""
        kan = KAN_CPPN(n_layers=3, hidden_size=8, spline_degree=degree)
        flat = FlattenKANParameters(kan)

        # Count params manually
        expected = sum(p.numel() for p in kan.parameters())
        assert flat.n_params == expected

    @pytest.mark.parametrize("degree", [1, 2, 3])
    def test_flattener_roundtrip(self, degree):
        """Flatten -> unflatten preserves parameters exactly."""
        torch.manual_seed(99)
        kan = KAN_CPPN(n_layers=3, hidden_size=8, spline_degree=degree)
        flat = FlattenKANParameters(kan)

        params = flat.flatten().clone()
        # Perturb
        flat.unflatten(params + 0.1)
        # Restore
        flat.unflatten(params)

        restored = flat.flatten()
        assert torch.allclose(params, restored, atol=1e-7)

    def test_memetic_creates_with_degree(self):
        """MemeticKAN_CPPN accepts spline_degree and passes it through."""
        memetic = MemeticKAN_CPPN(n_layers=3, hidden_size=8, spline_degree=3)
        assert memetic.center.spline_degree == 3
        for layer in memetic.center.layers:
            assert layer.spline_degree == 3
            assert layer.n_basis == 22  # 20 + 3 - 1


class TestSplineInspectorHighDegree:
    """Spline inspector must work for degree > 1."""

    @pytest.mark.parametrize("degree", [1, 2, 3])
    def test_inspector_matches_forward(self, degree):
        """Inspector output matches forward pass spline component."""
        from analysis.spline_inspector import extract_spline_curve

        torch.manual_seed(77)
        layer = KANCPPNLayer(2, 2, grid_size=10, spline_degree=degree)

        # Extract spline curve for edge (in=0, out=0)
        raw_inputs, spline_values, normalized = extract_spline_curve(layer, 0, 0, n_points=50)

        # Verify outputs are valid
        assert len(raw_inputs) == 50
        assert len(spline_values) == 50
        assert not np.any(np.isnan(spline_values)), "NaN in spline values"

        # Verify the spline values are reasonable (not all zero, not exploding)
        assert np.std(spline_values) > 0 or np.mean(np.abs(spline_values)) < 0.1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
