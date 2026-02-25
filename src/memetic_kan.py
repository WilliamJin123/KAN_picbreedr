"""
Memetic (NES + SGD) KAN-CPPN for image generation.

Uses Natural Evolution Strategy (OpenAI-ES style) with antithetic sampling
for global search, combined with local SGD refinement. This replaces the
previous tournament/crossover/mutation approach which was destructive for
KAN parameters due to crossover breaking co-adapted base_weight/coeffs.

Core algorithm per generation:
1. Maintain a single "center" network (the current best)
2. Sample pop_size random perturbation vectors with antithetic pairs
   (only perturbing spline coeffs and weights, NOT base_weight)
3. Estimate gradient from normalized fitness differences
4. Update spline parameters using ES gradient (if it improves fitness)
5. Run K steps of SGD on center for local refinement
"""

import math
import torch
import torch.nn as nn

from .kan import KAN_CPPN, FlattenKANParameters
from .color import hsv2rgb


class MemeticKAN_CPPN:
    """NES-Memetic optimizer for KAN-CPPNs.

    Maintains a single center network and uses Natural Evolution Strategy
    (antithetic sampling) for global search, with SGD for local refinement.
    ES only perturbs spline parameters (coeffs + weights), preserving the
    orthogonal base_weight that prevents signal collapse in deep networks.

    Args:
        pop_size: Number of perturbation pairs per generation.
        n_layers: Number of hidden layers.
        hidden_size: Neurons per hidden layer.
        n_inputs: Number of coordinate inputs.
        grid_size: Grid points for spline interpolation.
        spline_degree: B-spline degree (1=linear, 2=quadratic, 3=cubic).
        sigma: Perturbation scale for ES.
        lr_es: Learning rate for ES gradient updates.
        device: Torch device for computation.
    """

    def __init__(self, pop_size=20, n_layers=12, hidden_size=22, n_inputs=4,
                 grid_size=20, spline_degree=1, sigma=0.02, lr_es=0.01, device=None):
        self.pop_size = pop_size
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.n_inputs = n_inputs
        self.grid_size = grid_size
        self.spline_degree = spline_degree
        self.sigma = sigma
        self.lr_es = lr_es
        self.device = device or torch.device('cpu')

        # Single center network
        self.center = KAN_CPPN(n_layers, hidden_size, n_inputs, grid_size, spline_degree).to(self.device)

        # Bug 1 fix: ES flattener excludes base_weight to preserve orthogonality
        self.es_flattener = FlattenKANParameters(self.center, exclude_base_weight=True)
        # Full flattener for external use (weight sweeps, etc.)
        self.flattener = FlattenKANParameters(self.center)

        # Best fitness tracking
        self.best_fitness = float('inf')

    def evolve(self, target_img, n_generations=100, sgd_steps_per_gen=50,
               lr=3e-3, log_interval=10):
        """Main NES-Memetic training loop.

        Each generation:
        1. Sample perturbation vectors and evaluate antithetic pairs
           (only perturbing spline coeffs + weights, not base_weight).
        2. Compute ES gradient from normalized fitness differences.
        3. Update spline parameters with ES gradient (only if it improves fitness).
        4. Reset Adam state and run K steps of SGD for local refinement.

        Args:
            target_img: Target image tensor of shape (H, W, 3).
            n_generations: Number of evolutionary generations.
            sgd_steps_per_gen: SGD steps per generation for local refinement.
            lr: Learning rate for local SGD refinement.
            log_interval: Print progress every N generations.

        Returns:
            Tuple of (fitness_history, best_individual).
        """
        target_img = target_img.to(self.device)
        img_size = target_img.shape[0]
        fitness_history = []

        for gen in range(n_generations):
            # === Phase 1: ES gradient estimation (spline params only) ===
            center_params = self.es_flattener.flatten().detach()
            n_params = center_params.numel()

            epsilons = []
            fitness_diffs = []

            with torch.no_grad():
                # Evaluate pre-ES fitness for gating (Bug 4 fix)
                pre_es_img = self.center.generate_image(img_size=img_size)
                pre_es_fitness = torch.mean((pre_es_img - target_img) ** 2).item()

                for i in range(self.pop_size):
                    eps = torch.randn(n_params, device=self.device)
                    epsilons.append(eps)

                    # Positive perturbation
                    self.es_flattener.unflatten(center_params + self.sigma * eps)
                    img_pos = self.center.generate_image(img_size=img_size)
                    f_pos = torch.mean((img_pos - target_img) ** 2).item()

                    # Negative perturbation (antithetic)
                    self.es_flattener.unflatten(center_params - self.sigma * eps)
                    img_neg = self.center.generate_image(img_size=img_size)
                    f_neg = torch.mean((img_neg - target_img) ** 2).item()

                    # Bug 5 fix: normalize fitness differences for stability
                    denom = abs(f_pos) + abs(f_neg) + 1e-8
                    fitness_diffs.append((f_pos - f_neg) / denom)

            # Compute ES gradient
            es_grad = torch.zeros(n_params, device=self.device)
            for fd, eps in zip(fitness_diffs, epsilons):
                es_grad += fd * eps
            es_grad /= (2 * self.pop_size * self.sigma)

            # Apply ES update to candidate
            candidate_params = center_params - self.lr_es * es_grad

            # Bug 4 fix: only accept ES update if it improves fitness
            with torch.no_grad():
                self.es_flattener.unflatten(candidate_params)
                post_es_img = self.center.generate_image(img_size=img_size)
                post_es_fitness = torch.mean((post_es_img - target_img) ** 2).item()

                if post_es_fitness >= pre_es_fitness:
                    # ES made things worse — revert
                    self.es_flattener.unflatten(center_params)

            # === Phase 2: SGD local refinement ===
            # Bug 2 fix: fresh optimizer each generation so state matches
            # current parameters (ES may have just moved them)
            sgd_optimizer = torch.optim.Adam(self.center.parameters(), lr=lr)

            self.center.train()
            for step in range(sgd_steps_per_gen):
                sgd_optimizer.zero_grad()
                generated = self.center.generate_image(img_size=img_size)
                loss = torch.mean((generated - target_img) ** 2)
                loss.backward()

                # Normalize gradients (matching reference)
                total_norm = 0.0
                for p in self.center.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2
                total_norm = total_norm ** 0.5
                if total_norm > 0:
                    for p in self.center.parameters():
                        if p.grad is not None:
                            p.grad.data /= total_norm

                sgd_optimizer.step()

            # === Track fitness ===
            with torch.no_grad():
                generated = self.center.generate_image(img_size=img_size)
                fitness = torch.mean((generated - target_img) ** 2).item()

            self.best_fitness = min(self.best_fitness, fitness)
            fitness_history.append(self.best_fitness)

            if log_interval and (gen + 1) % log_interval == 0:
                print(f"Generation {gen+1}/{n_generations}, "
                      f"Fitness: {fitness:.6f}, Best: {self.best_fitness:.6f}")

        return fitness_history, self.center

    def get_best(self):
        """Return the center network (the current best).

        Returns:
            The center KAN_CPPN.
        """
        return self.center

    def reset_weights(self, individual=None, default_value=1.0):
        """Reset all weights to a default value while keeping learned spline coefficients.

        This is for Phase 4.1 - investigating whether the memetic KAN can
        recover original images when weights are reset but spline shapes
        are preserved.

        Args:
            individual: A KAN_CPPN to reset. If None, resets the center network.
            default_value: Value to set all weight parameters to.

        Returns:
            The reset individual.
        """
        if individual is None:
            individual = self.get_best()

        with torch.no_grad():
            for layer in individual.layers:
                # Reset spline scaling weights but keep coeffs (spline shapes)
                layer.weights.fill_(default_value)
                # Reset base linear weights to fresh orthogonal init
                nn.init.orthogonal_(layer.base_weight)

        return individual
