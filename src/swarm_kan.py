"""
Swarm-optimized KAN-CPPN for image generation.

Combines particle swarm optimization (PSO) with gradient descent to
optimize spline coefficients in KAN layers. Each KAN layer maintains
a population of coefficient "particles" that share information via
PSO update rules, enabling exploration of the activation function
landscape beyond what SGD alone can reach.
"""

import torch
import torch.nn as nn
import copy

from .kan import KANCPPNLayer, KAN_CPPN
from .color import hsv2rgb


class SwarmKANCPPNLayer(KANCPPNLayer):
    """A KAN-CPPN layer with PSO-based swarm optimization on spline coefficients.

    Maintains particle swarm state (velocity, personal best, global best) for
    the spline coefficients, enabling swarm-based exploration of activation
    functions alongside gradient descent.

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        grid_size: Number of grid points for spline interpolation.
        spline_degree: B-spline degree (1=linear, 2=quadratic, 3=cubic).
        n_particles: Number of particles in the swarm (including this one).
        inertia: PSO inertia weight (w).
        cognitive: PSO cognitive coefficient (c1 - pull toward personal best).
        social: PSO social coefficient (c2 - pull toward global best).
    """

    def __init__(self, in_features, out_features, grid_size=20, spline_degree=1,
                 n_particles=5, inertia=0.7, cognitive=1.5, social=1.5):
        super().__init__(in_features, out_features, grid_size, spline_degree)
        self.n_particles = n_particles
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social

        # PSO state for coeffs
        coeff_shape = (n_particles,) + self.coeffs.shape
        self.register_buffer('velocities', torch.zeros(coeff_shape))
        self.register_buffer('particles', torch.randn(coeff_shape) * 0.01)
        self.register_buffer('personal_best', self.particles.clone())
        self.register_buffer('personal_best_scores', torch.full((n_particles,), float('inf')))
        self.register_buffer('global_best', self.coeffs.data.clone())
        self.register_buffer('global_best_score', torch.tensor(float('inf')))

        # Set particle 0 to match current coefficients
        self.particles[0] = self.coeffs.data.clone()
        self.personal_best[0] = self.coeffs.data.clone()

    def swarm_step(self, current_loss=None):
        """Perform one PSO update step on the spline coefficients.

        Args:
            current_loss: The current loss value (used to update personal/global bests).
        """
        # Update personal and global best for particle 0 (the active one)
        if current_loss is not None:
            self.particles[0] = self.coeffs.data.clone()
            if current_loss < self.personal_best_scores[0]:
                self.personal_best_scores[0] = current_loss
                self.personal_best[0] = self.coeffs.data.clone()
            if current_loss < self.global_best_score:
                self.global_best_score.fill_(current_loss)
                self.global_best.copy_(self.coeffs.data)

        # PSO velocity update for all particles
        r1 = torch.rand_like(self.velocities)
        r2 = torch.rand_like(self.velocities)

        cognitive_component = self.cognitive * r1 * (self.personal_best - self.particles)
        social_component = self.social * r2 * (self.global_best.unsqueeze(0) - self.particles)

        self.velocities = (
            self.inertia * self.velocities
            + cognitive_component
            + social_component
        )

        # Update particle positions
        self.particles += self.velocities

        # Blend particle 0 back into the active coefficients
        # Use a soft blend so SGD gradients aren't completely overwritten
        blend_factor = 0.1
        self.coeffs.data = (1 - blend_factor) * self.coeffs.data + blend_factor * self.particles[0]


class SwarmKAN_CPPN(nn.Module):
    """Full CPPN using swarm-optimized KAN layers.

    Each KAN layer has PSO-based swarm optimization on its spline coefficients.
    Training alternates between SGD steps and swarm update steps.

    Args:
        n_layers: Number of hidden layers.
        hidden_size: Neurons per hidden layer.
        n_inputs: Number of coordinate inputs (default 4: y, x, d, b).
        grid_size: Grid points for spline interpolation.
        spline_degree: B-spline degree (1=linear, 2=quadratic, 3=cubic).
        n_particles: Number of PSO particles per layer.
        inertia: PSO inertia weight.
        cognitive: PSO cognitive coefficient.
        social: PSO social coefficient.
    """

    def __init__(self, n_layers, hidden_size, n_inputs=4, grid_size=20, spline_degree=1,
                 n_particles=5, inertia=0.7, cognitive=1.5, social=1.5):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.n_inputs = n_inputs
        self.grid_size = grid_size
        self.spline_degree = spline_degree

        layers = []
        # Input layer
        layers.append(SwarmKANCPPNLayer(
            n_inputs, hidden_size, grid_size, spline_degree, n_particles, inertia, cognitive, social
        ))
        # Hidden layers
        for _ in range(n_layers - 1):
            layers.append(SwarmKANCPPNLayer(
                hidden_size, hidden_size, grid_size, spline_degree, n_particles, inertia, cognitive, social
            ))
        # Output layer
        layers.append(SwarmKANCPPNLayer(
            hidden_size, 3, grid_size, spline_degree, n_particles, inertia, cognitive, social
        ))

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        """Process coordinate inputs through the swarm KAN-CPPN.

        Args:
            x: Input tensor of shape (batch_size, n_inputs).

        Returns:
            Tuple of ((h, s, v), features_list).
        """
        features = [x]
        for layer in self.layers:
            x = layer(x)
            features.append(x)

        h, s, v = x[:, 0], x[:, 1], x[:, 2]
        return (h, s, v), features

    def swarm_step(self, current_loss=None):
        """Perform one PSO update across all layers.

        Args:
            current_loss: Current loss value for personal/global best tracking.
        """
        loss_val = current_loss.item() if isinstance(current_loss, torch.Tensor) else current_loss
        for layer in self.layers:
            layer.swarm_step(current_loss=loss_val)

    def generate_image(self, img_size=256, return_features=False):
        """Generate an image from the swarm KAN-CPPN.

        Args:
            img_size: Resolution of the output image.
            return_features: Whether to return intermediate features.

        Returns:
            RGB image tensor of shape (img_size, img_size, 3).
        """
        device = next(self.parameters()).device

        coords = torch.linspace(-1, 1, img_size, device=device)
        grid_x, grid_y = torch.meshgrid(coords, coords, indexing='ij')
        d = torch.sqrt(grid_x ** 2 + grid_y ** 2) * 1.4
        b = torch.ones_like(grid_x)

        inputs = torch.stack([grid_y, grid_x, d, b], dim=-1)
        inputs_flat = inputs.reshape(-1, self.n_inputs)

        (h, s, v), features = self.forward(inputs_flat)

        h_img = (h + 1) % 1
        s_img = s.clamp(0, 1)
        v_img = v.abs().clamp(0, 1)

        r, g, b_ch = hsv2rgb(h_img, s_img, v_img)
        rgb = torch.stack([r, g, b_ch], dim=-1)
        rgb = rgb.reshape(img_size, img_size, 3)

        if return_features:
            spatial_features = []
            for feat in features:
                spatial_features.append(feat.reshape(img_size, img_size, -1))
            return rgb, spatial_features
        return rgb
