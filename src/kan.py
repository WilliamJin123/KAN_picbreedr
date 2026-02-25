"""
KAN-CPPN layers for image generation.

Adapts Kolmogorov-Arnold Network layers for use in Compositional Pattern
Producing Networks (CPPNs). Instead of fixed activation functions, each
edge has a learnable spline-based activation.
"""

import math

import torch
import torch.nn as nn
import numpy as np

from .color import hsv2rgb


def compute_bspline_basis(x, knots, degree, n_basis):
    """Compute B-spline basis functions using span-based local evaluation.

    Uses the local support property: at most (degree+1) basis functions are
    non-zero for any point. Finds each point's knot span via searchsorted,
    then computes only the non-zero values using a compact recursion.

    Args:
        x: Tensor of shape (..., in_features) with values in [0, 1].
        knots: 1D tensor of knot positions (length n_basis + degree + 1).
        degree: Spline degree (1=linear, 2=quadratic, 3=cubic, etc.).
        n_basis: Number of basis functions.

    Returns:
        Tensor of shape (..., in_features, n_basis) with basis values.
    """
    orig_shape = x.shape  # (..., in_features)
    x_flat = x.reshape(-1)  # (N,)
    N = x_flat.shape[0]

    span = torch.searchsorted(knots, x_flat, right=False) - 1
    span = span.clamp(degree, n_basis - 1)

    N_local = torch.zeros(N, degree + 1, device=x.device, dtype=x.dtype)
    N_local[:, 0] = 1.0

    for k in range(1, degree + 1):
        saved = torch.zeros(N, device=x.device, dtype=x.dtype)
        for r in range(k):
            left_knot_idx = span - k + 1 + r
            right_knot_idx = span + 1 + r
            left_knot = knots[left_knot_idx.clamp(0, len(knots) - 1)]
            right_knot = knots[right_knot_idx.clamp(0, len(knots) - 1)]
            denom = right_knot - left_knot
            safe_denom = torch.where(denom.abs() > 1e-10, denom, torch.ones_like(denom))
            temp = torch.where(denom.abs() > 1e-10,
                               N_local[:, r] / safe_denom,
                               torch.zeros_like(N_local[:, r]))
            N_local[:, r] = saved + (right_knot - x_flat) * temp
            saved = (x_flat - left_knot) * temp
        N_local[:, k] = saved

    offsets = torch.arange(degree + 1, device=x.device).unsqueeze(0)
    basis_indices = (span.unsqueeze(1) - degree + offsets).clamp(0, n_basis - 1)

    basis_flat = torch.zeros(N, n_basis, device=x.device, dtype=x.dtype)
    basis_flat.scatter_add_(1, basis_indices, N_local)

    return basis_flat.reshape(orig_shape + (n_basis,))


def eval_bspline_direct(x_norm, knots, degree, n_basis, coeffs, weights):
    """Evaluate B-spline directly using de Boor's algorithm.

    Skips basis function computation — goes directly from coefficients to
    spline values. No in-place ops (autograd-safe).

    Args:
        x_norm: Tensor (batch, in_features) with values in [0, 1].
        knots: 1D tensor of knot positions.
        degree: Spline degree.
        n_basis: Number of basis functions.
        coeffs: Tensor (out_features, in_features, n_basis).
        weights: Tensor (out_features, in_features).

    Returns:
        spline_output: Tensor (batch, out_features).
    """
    batch, in_feat = x_norm.shape
    out_feat = coeffs.shape[0]

    # Find span for each (batch, in_features) point
    x_flat = x_norm.reshape(-1)
    span = torch.searchsorted(knots, x_flat, right=False) - 1
    span = span.clamp(degree, n_basis - 1).reshape(batch, in_feat)

    # Gather local coefficients: (degree+1) per (batch, out, in)
    offsets = torch.arange(degree + 1, device=x_norm.device)
    coeff_idx = (span.unsqueeze(-1) - degree + offsets).clamp(0, n_basis - 1)
    idx_expanded = coeff_idx.unsqueeze(1).expand(batch, out_feat, in_feat, degree + 1)
    coeffs_expanded = coeffs.unsqueeze(0).expand(batch, out_feat, in_feat, n_basis)
    d = torch.gather(coeffs_expanded, 3, idx_expanded)  # (batch, out, in, degree+1)

    # Store columns as a list to avoid in-place modification (autograd-safe)
    cols = [d[..., r] for r in range(degree + 1)]

    for k in range(1, degree + 1):
        new_cols = list(cols)  # shallow copy of references
        for r in range(degree, k - 1, -1):
            left_knot_idx = (span - degree + r).clamp(0, len(knots) - 1)
            right_knot_idx = (span - degree + r + k).clamp(0, len(knots) - 1)
            t_left = knots[left_knot_idx]
            t_right = knots[right_knot_idx]

            denom = t_right - t_left
            safe_denom = torch.where(denom.abs() > 1e-10, denom, torch.ones_like(denom))
            alpha = (x_norm - t_left) / safe_denom
            alpha = torch.where(denom.abs() > 1e-10, alpha, torch.zeros_like(alpha))
            # Expand alpha: (batch, in_feat) -> (batch, 1, in_feat)
            alpha = alpha.unsqueeze(1)

            new_cols[r] = (1 - alpha) * cols[r - 1] + alpha * cols[r]
        cols = new_cols

    # Result is cols[degree]: (batch, out, in)
    spline_vals = cols[degree]

    # Apply weights and sum over input features
    weighted = spline_vals * weights.unsqueeze(0)
    return weighted.sum(dim=2)


class KANCPPNLayer(nn.Module):
    """A KAN layer designed for CPPN use.

    Uses vectorized spline interpolation with sigmoid normalization mapping
    inputs to [0, 1] for grid lookup. No bias, matching the original CPPN
    design. Grid spans [0, 1] to match sigmoid output range.

    Supports higher-degree B-splines via the spline_degree parameter.
    Degree 1 uses the original fast linear interpolation path (bit-identical).
    Degree >= 2 uses iterative de Boor B-spline basis evaluation.

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        grid_size: Number of grid points for spline interpolation.
        spline_degree: B-spline degree (1=linear, 2=quadratic, 3=cubic).
    """

    def __init__(self, in_features, out_features, grid_size=20, spline_degree=1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_degree = spline_degree

        # Number of basis functions (and coefficients) depends on degree
        # For degree 1 with grid_size knots: n_basis = grid_size (same as before)
        # For degree k: n_basis = grid_size + degree - 1
        self.n_basis = grid_size + spline_degree - 1

        # Base linear weight (residual path) — prevents signal collapse in deep networks
        # Orthogonal init preserves signal norm exactly through deep layers
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.orthogonal_(self.base_weight)

        # Spline coefficients for each (output, input) pair
        self.coeffs = nn.Parameter(
            torch.randn(out_features, in_features, self.n_basis) * 0.01
        )
        # Weights for scaling spline outputs per (output, input) pair
        self.weights = nn.Parameter(
            torch.ones(out_features, in_features) * 0.1
        )

        # Grid spans [0, 1] to match sigmoid normalization
        self.register_buffer('grid', torch.linspace(0, 1, grid_size))

        # For degree >= 2, register clamped knot vector
        if spline_degree >= 2:
            interior = torch.linspace(0, 1, grid_size)
            knots = torch.cat([
                torch.zeros(spline_degree),
                interior,
                torch.ones(spline_degree),
            ])
            self.register_buffer('knots', knots)

    def forward(self, x):
        """Forward pass with vectorized spline interpolation.

        Args:
            x: Input tensor of shape (batch_size, in_features).

        Returns:
            Output tensor of shape (batch_size, out_features).
        """
        # Base linear path (residual) — keeps signal alive through deep networks
        base = x @ self.base_weight.T  # (batch_size, out_features)

        # Normalize to [0, 1] with sigmoid
        x_norm = torch.sigmoid(x)  # (batch_size, in_features)

        if self.spline_degree == 1:
            # === Fast path: original linear interpolation (bit-identical) ===
            scaled = x_norm * (self.grid_size - 1)  # (batch_size, in_features)

            idx = scaled.long().clamp(0, self.grid_size - 2)  # (batch_size, in_features)
            frac = scaled - idx.float()  # (batch_size, in_features)

            # Spline evaluation via gather on (in, out, grid) layout.
            coeffs_iog = self.coeffs.permute(1, 0, 2)  # (in, out, grid)

            idx_gather = idx.T.unsqueeze(1).expand(-1, self.out_features, -1)  # (in, out, batch)
            left = coeffs_iog.gather(2, idx_gather)   # (in, out, batch)
            right = coeffs_iog.gather(2, idx_gather + 1)  # (in, out, batch)

            frac_t = frac.T.unsqueeze(1)  # (in, 1, batch)
            interpolated = left + frac_t * (right - left)  # (in, out, batch)

            weighted = interpolated * self.weights.T.unsqueeze(2)  # (in, out, batch)
            spline_output = weighted.sum(dim=0).T  # (batch, out)
        else:
            # === B-spline path: degree >= 2 ===
            # Direct de Boor evaluation: coefficients -> spline values
            # Skips basis matrix construction for much better performance
            spline_output = eval_bspline_direct(
                x_norm, self.knots, self.spline_degree, self.n_basis,
                self.coeffs, self.weights
            )

        return base + spline_output


class KAN_CPPN(nn.Module):
    """Full CPPN using KAN layers.

    Generates images by processing coordinate inputs through a stack of
    KAN layers, outputting HSV color values that are converted to RGB.

    Args:
        n_layers: Number of hidden layers.
        hidden_size: Neurons per hidden layer.
        n_inputs: Number of coordinate inputs (default 4: y, x, d, b).
        grid_size: Grid points for spline interpolation in each KAN layer.
        spline_degree: B-spline degree (1=linear, 2=quadratic, 3=cubic).
    """

    def __init__(self, n_layers, hidden_size, n_inputs=4, grid_size=20, spline_degree=1):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.n_inputs = n_inputs
        self.grid_size = grid_size
        self.spline_degree = spline_degree

        layers = []
        # Input layer: n_inputs -> hidden_size
        layers.append(KANCPPNLayer(n_inputs, hidden_size, grid_size, spline_degree))
        # Hidden layers: hidden_size -> hidden_size
        for _ in range(n_layers - 1):
            layers.append(KANCPPNLayer(hidden_size, hidden_size, grid_size, spline_degree))
        # Output layer: hidden_size -> 3 (h, s, v)
        layers.append(KANCPPNLayer(hidden_size, 3, grid_size, spline_degree))

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        """Process coordinate inputs through the KAN-CPPN.

        Args:
            x: Input tensor of shape (batch_size, n_inputs) where inputs
               are (y, x, d, b).

        Returns:
            Tuple of ((h, s, v), features_list) where h/s/v are each
            (batch_size,) tensors and features_list contains intermediate
            activations.
        """
        features = [x]
        for layer in self.layers:
            x = layer(x)
            features.append(x)

        # Split output into h, s, v channels
        h, s, v = x[:, 0], x[:, 1], x[:, 2]
        return (h, s, v), features

    def generate_image(self, img_size=256, return_features=False):
        """Generate an image from the CPPN at the given resolution.

        Creates a coordinate grid matching the reference CPPN format,
        processes all pixels in a batch, and converts HSV to RGB.

        Args:
            img_size: Width and height of the output image.
            return_features: If True, also return intermediate activations.

        Returns:
            RGB image as tensor of shape (img_size, img_size, 3) in [0, 1].
            If return_features is True, also returns the features list.
        """
        device = next(self.parameters()).device

        # Create coordinate grid matching reference: y,x in [-1,1]
        coords = torch.linspace(-1, 1, img_size, device=device)
        # indexing='ij' matches JAX meshgrid behavior
        grid_x, grid_y = torch.meshgrid(coords, coords, indexing='ij')
        d = torch.sqrt(grid_x ** 2 + grid_y ** 2) * 1.4
        b = torch.ones_like(grid_x)

        # Stack inputs: (y, x, d, b) -> (img_size*img_size, 4)
        inputs = torch.stack([grid_y, grid_x, d, b], dim=-1)
        inputs_flat = inputs.reshape(-1, self.n_inputs)

        # Forward pass
        (h, s, v), features = self.forward(inputs_flat)

        # HSV to RGB conversion (same as reference)
        h_img = (h + 1) % 1  # shift and wrap to [0, 1]
        s_img = s.clamp(0, 1)
        v_img = v.abs().clamp(0, 1)

        r, g, b_ch = hsv2rgb(h_img, s_img, v_img)
        rgb = torch.stack([r, g, b_ch], dim=-1)
        rgb = rgb.reshape(img_size, img_size, 3)

        if return_features:
            # Reshape features to spatial form
            spatial_features = []
            for feat in features:
                spatial_features.append(feat.reshape(img_size, img_size, -1))
            return rgb, spatial_features
        return rgb


class FlattenKANParameters:
    """Flatten and unflatten KAN-CPPN parameters for evolutionary algorithms.

    Provides a 1D vector interface over learnable parameters of a KAN_CPPN,
    matching the FlattenCPPNParameters interface from the reference implementation.

    Args:
        cppn: A KAN_CPPN instance.
        exclude_base_weight: If True, exclude base_weight parameters from the
            flat vector. Used by ES to avoid corrupting the orthogonal residual
            path that prevents signal collapse in deep networks.
    """

    def __init__(self, cppn, exclude_base_weight=False):
        self.cppn = cppn
        self.exclude_base_weight = exclude_base_weight
        self._param_shapes = []
        self._param_names = []
        total = 0
        for name, param in cppn.named_parameters():
            if exclude_base_weight and 'base_weight' in name:
                continue
            self._param_names.append(name)
            self._param_shapes.append(param.shape)
            total += param.numel()
        self._n_params = total

    @property
    def n_params(self):
        """Total number of learnable parameters."""
        return self._n_params

    def flatten(self):
        """Flatten learnable parameters to a 1D vector.

        Returns:
            1D tensor of all included parameters concatenated.
        """
        params = []
        for name, param in self.cppn.named_parameters():
            if self.exclude_base_weight and 'base_weight' in name:
                continue
            params.append(param.data.reshape(-1))
        return torch.cat(params)

    def unflatten(self, flat_params):
        """Load parameters from a 1D vector back into the model.

        Args:
            flat_params: 1D tensor with n_params elements.
        """
        offset = 0
        for name, param in self.cppn.named_parameters():
            if self.exclude_base_weight and 'base_weight' in name:
                continue
            n = param.numel()
            param.data.copy_(flat_params[offset:offset + n].reshape(param.shape))
            offset += n

    def generate_image(self, flat_params=None, img_size=256, return_features=False):
        """Generate an image, optionally loading parameters first.

        Args:
            flat_params: If provided, unflatten into the model first.
            img_size: Resolution of the output image.
            return_features: Whether to return intermediate features.

        Returns:
            RGB image tensor (and optionally features).
        """
        if flat_params is not None:
            self.unflatten(flat_params)
        return self.cppn.generate_image(img_size=img_size, return_features=return_features)
