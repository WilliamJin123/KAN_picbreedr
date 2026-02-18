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


class KANCPPNLayer(nn.Module):
    """A KAN layer designed for CPPN use.

    Uses vectorized spline interpolation with sigmoid normalization mapping
    inputs to [0, 1] for grid lookup. No bias, matching the original CPPN
    design. Grid spans [0, 1] to match sigmoid output range.

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        grid_size: Number of grid points for spline interpolation.
    """

    def __init__(self, in_features, out_features, grid_size=20):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size

        # Base linear weight (residual path) — prevents signal collapse in deep networks
        # Orthogonal init preserves signal norm exactly through deep layers
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.orthogonal_(self.base_weight)

        # Spline coefficients for each (output, input) pair
        self.coeffs = nn.Parameter(
            torch.randn(out_features, in_features, grid_size) * 0.01
        )
        # Weights for scaling spline outputs per (output, input) pair
        self.weights = nn.Parameter(
            torch.ones(out_features, in_features) * 0.1
        )

        # Grid spans [0, 1] to match sigmoid normalization
        self.register_buffer('grid', torch.linspace(0, 1, grid_size))

    def forward(self, x):
        """Forward pass with vectorized spline interpolation.

        Args:
            x: Input tensor of shape (batch_size, in_features).

        Returns:
            Output tensor of shape (batch_size, out_features).
        """
        # Base linear path (residual) — keeps signal alive through deep networks
        base = x @ self.base_weight.T  # (batch_size, out_features)

        # Normalize to [0, 1] with sigmoid, then map to grid index range [0, grid_size-1]
        x_norm = torch.sigmoid(x)  # (batch_size, in_features)
        scaled = x_norm * (self.grid_size - 1)  # (batch_size, in_features)

        idx = scaled.long().clamp(0, self.grid_size - 2)  # (batch_size, in_features)
        frac = scaled - idx.float()  # (batch_size, in_features)

        # Spline evaluation via gather on (in, out, grid) layout.
        # coeffs: (out, in, grid) -> permute to (in, out, grid) so gather is along grid dim.
        coeffs_iog = self.coeffs.permute(1, 0, 2)  # (in, out, grid)

        # idx: (batch, in) -> (in, batch) -> (in, out, batch) for gathering from (in, out, grid)
        idx_gather = idx.T.unsqueeze(1).expand(-1, self.out_features, -1)  # (in, out, batch)
        left = coeffs_iog.gather(2, idx_gather)   # (in, out, batch)
        right = coeffs_iog.gather(2, idx_gather + 1)  # (in, out, batch)

        # Linear interpolation: frac (batch, in) -> (in, 1, batch) for broadcasting
        frac_t = frac.T.unsqueeze(1)  # (in, 1, batch)
        interpolated = left + frac_t * (right - left)  # (in, out, batch)

        # Apply weights and sum over input features
        # weights: (out, in) -> (in, out, 1) for broadcasting
        weighted = interpolated * self.weights.T.unsqueeze(2)  # (in, out, batch)
        spline_output = weighted.sum(dim=0).T  # (batch, out)

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
    """

    def __init__(self, n_layers, hidden_size, n_inputs=4, grid_size=20):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.n_inputs = n_inputs
        self.grid_size = grid_size

        layers = []
        # Input layer: n_inputs -> hidden_size
        layers.append(KANCPPNLayer(n_inputs, hidden_size, grid_size))
        # Hidden layers: hidden_size -> hidden_size
        for _ in range(n_layers - 1):
            layers.append(KANCPPNLayer(hidden_size, hidden_size, grid_size))
        # Output layer: hidden_size -> 3 (h, s, v)
        layers.append(KANCPPNLayer(hidden_size, 3, grid_size))

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
