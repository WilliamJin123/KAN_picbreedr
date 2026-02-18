import numpy as np
import torch
import torch.nn as nn

from .color import hsv2rgb

# Activation functions matching the JAX reference
cache = lambda x: x
identity = lambda x: x
activation_fn_map = {
    'cache': cache,
    'identity': identity,
    'cos': torch.cos,
    'sin': torch.sin,
    'tanh': torch.tanh,
    'sigmoid': lambda x: torch.sigmoid(x) * 2.0 - 1.0,
    'gaussian': lambda x: torch.exp(-x ** 2) * 2.0 - 1.0,
    'relu': torch.relu,
}


class CPPN(nn.Module):
    """PyTorch port of the Flax CPPN model.

    Args:
        arch: Architecture string, e.g. "12;cache:15,gaussian:4,identity:2,sin:1"
              Format: "{n_layers};{act1}:{n1},{act2}:{n2},..."
        inputs: Comma-separated input names, e.g. "y,x,d,b"
    """

    def __init__(self, arch="12;cache:15,gaussian:4,identity:2,sin:1", inputs="y,x,d,b"):
        super().__init__()
        self.arch = arch
        self.inputs_str = inputs

        n_layers_str, activation_neurons = arch.split(";")
        self.n_layers = int(n_layers_str)
        self.activations = [i.split(":")[0] for i in activation_neurons.split(",")]
        self.d_hidden = [int(i.split(":")[-1]) for i in activation_neurons.split(",")]
        self.d_hidden_total = sum(self.d_hidden)
        self.dh_cumsum = list(np.cumsum(self.d_hidden))

        d_in = len(inputs.split(","))

        # Create layers: n_layers hidden layers + 1 output layer
        layers = []
        layers.append(nn.Linear(d_in, self.d_hidden_total, bias=False))
        for _ in range(self.n_layers - 1):
            layers.append(nn.Linear(self.d_hidden_total, self.d_hidden_total, bias=False))
        layers.append(nn.Linear(self.d_hidden_total, 3, bias=False))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, n_inputs)

        Returns:
            (h, s, v): Tuple of tensors each of shape (batch_size,)
            features: List of intermediate activation tensors
        """
        features = [x]
        for i_layer in range(self.n_layers):
            x = self.layers[i_layer](x)
            # Split along last dim, apply per-group activations, concatenate
            parts = torch.split(x, self.d_hidden, dim=-1)
            parts = [activation_fn_map[act](p) for p, act in zip(parts, self.activations)]
            x = torch.cat(parts, dim=-1)
            features.append(x)

        x = self.layers[self.n_layers](x)  # final layer
        features.append(x)
        h, s, v = x[..., 0], x[..., 1], x[..., 2]
        return (h, s, v), features

    @torch.no_grad()
    def generate_image(self, img_size=256, return_features=False):
        """Generate an image from the CPPN at the given resolution.

        Args:
            img_size: Width and height of the output image.
            return_features: If True, also return intermediate features.

        Returns:
            rgb: Tensor of shape (img_size, img_size, 3) with values in [0, 1].
            features: (optional) List of per-layer feature tensors.
        """
        device = next(self.parameters()).device

        coord = torch.linspace(-1, 1, img_size, device=device)
        grid_x, grid_y = torch.meshgrid(coord, coord, indexing='ij')

        input_map = {
            'x': grid_x,
            'y': grid_y,
            'd': torch.sqrt(grid_x ** 2 + grid_y ** 2) * 1.4,
            'b': torch.ones_like(grid_x),
            'xabs': torch.abs(grid_x),
            'yabs': torch.abs(grid_y),
        }

        input_names = self.inputs_str.split(",")
        inputs = torch.stack([input_map[name] for name in input_names], dim=-1)
        # Flatten spatial dims for batch processing: (img_size*img_size, n_inputs)
        flat_inputs = inputs.reshape(-1, len(input_names))

        (h, s, v), features = self.forward(flat_inputs)

        # Reshape back to spatial
        h = h.reshape(img_size, img_size)
        s = s.reshape(img_size, img_size)
        v = v.reshape(img_size, img_size)

        r, g, b = hsv2rgb((h + 1) % 1, s.clip(0, 1), torch.abs(v).clip(0, 1))
        rgb = torch.stack([r, g, b], dim=-1)

        if return_features:
            spatial_features = []
            for feat in features:
                if feat.dim() == 2:
                    spatial_features.append(feat.reshape(img_size, img_size, -1))
                else:
                    spatial_features.append(feat)
            return rgb, spatial_features
        return rgb


class FlattenCPPNParameters:
    """Manages flattening/unflattening CPPN parameters to/from a 1D vector.

    Handles loading parameters saved by the JAX/Flax reference implementation,
    accounting for the differences in parameter ordering and layout:
    - JAX sorts dict keys lexicographically (Dense_0, Dense_1, Dense_10, ...)
    - Flax Dense kernel shape is (in_features, out_features) (row-major)
    - PyTorch Linear weight shape is (out_features, in_features)
    """

    def __init__(self, cppn):
        self.cppn = cppn
        self.n_params = sum(p.numel() for p in cppn.parameters())

        # Precompute the mapping from JAX sorted order to PyTorch sequential order.
        # In PyTorch, self.cppn.layers is [layer_0, layer_1, ..., layer_N].
        # In JAX/Flax, the params dict keys are Dense_0, Dense_1, ..., Dense_N,
        # and jax.tree_util sorts them lexicographically.
        n_total_layers = len(self.cppn.layers)
        jax_names = sorted([f'Dense_{i}' for i in range(n_total_layers)])
        # jax_sorted_indices[k] = which PyTorch layer index corresponds to the k-th
        # entry in JAX's sorted order
        self.jax_sorted_indices = [int(name.split('_')[1]) for name in jax_names]

    def load_jax_flat_params(self, flat_params):
        """Load a flat parameter vector saved by the JAX/evosax pipeline.

        The flat vector was created by evosax.ParameterReshaper which flattens
        JAX pytree leaves in sorted-key order. Each leaf is a Flax Dense kernel
        with shape (in_features, out_features), stored row-major (C order).

        Args:
            flat_params: 1D tensor of length self.n_params
        """
        if isinstance(flat_params, np.ndarray):
            flat_params = torch.tensor(flat_params, dtype=torch.float32)

        # Compute the size of each kernel in JAX sorted order
        n_total_layers = len(self.cppn.layers)
        jax_kernel_sizes = []
        for jax_idx in range(n_total_layers):
            pt_layer_idx = self.jax_sorted_indices[jax_idx]
            layer = self.cppn.layers[pt_layer_idx]
            # Flax kernel shape = (in_features, out_features)
            jax_kernel_sizes.append(layer.in_features * layer.out_features)

        # Split flat vector according to JAX sorted order
        kernels_flat = torch.split(flat_params, jax_kernel_sizes)

        # Assign each kernel to the correct PyTorch layer
        for jax_idx, kernel_flat in enumerate(kernels_flat):
            pt_layer_idx = self.jax_sorted_indices[jax_idx]
            layer = self.cppn.layers[pt_layer_idx]
            in_f, out_f = layer.in_features, layer.out_features
            # Flax kernel: (in_features, out_features), row-major
            # PyTorch weight: (out_features, in_features)
            kernel = kernel_flat.reshape(in_f, out_f).T
            layer.weight.data.copy_(kernel)

    def flatten(self):
        """Flatten current PyTorch parameters to a 1D vector in JAX order.

        Returns:
            1D tensor of length self.n_params, in JAX sorted-key order with
            Flax kernel layout (in_features, out_features) row-major.
        """
        parts = []
        for jax_idx in range(len(self.cppn.layers)):
            pt_layer_idx = self.jax_sorted_indices[jax_idx]
            layer = self.cppn.layers[pt_layer_idx]
            # PyTorch weight: (out_features, in_features) -> Flax: (in_features, out_features)
            kernel = layer.weight.data.T.contiguous()
            parts.append(kernel.reshape(-1))
        return torch.cat(parts)

    def generate_image(self, flat_params, img_size=256, return_features=False):
        """Load flat params and generate an image.

        Args:
            flat_params: 1D parameter tensor (in JAX order).
            img_size: Output image resolution.
            return_features: If True, also return intermediate features.

        Returns:
            Same as CPPN.generate_image().
        """
        self.load_jax_flat_params(flat_params)
        return self.cppn.generate_image(img_size=img_size, return_features=return_features)
