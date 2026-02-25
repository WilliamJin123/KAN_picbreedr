"""
Training loops for CPPN variants.

Provides training functions for standard SGD, swarm-optimized, and
memetic evolutionary approaches. All functions follow the same pattern:
take a CPPN model and target image, return loss history and trained model.
"""

import os

import torch
import torch.nn as nn


def train_sgd(cppn, target_img, lr=3e-3, n_iters=10000, log_interval=100,
              checkpoint_dir=None, checkpoint_interval=100, resume_from=None):
    """Train any CPPN variant with SGD.

    Works with MLP CPPN, KAN_CPPN, or any model that has a generate_image()
    method. Uses normalized gradients like the reference implementation
    (grad / norm(grad)).

    Args:
        cppn: A CPPN model with a generate_image(img_size) method.
        target_img: Target image tensor of shape (H, W, 3) in [0, 1].
        lr: Learning rate for Adam optimizer.
        n_iters: Total number of training iterations.
        log_interval: Print loss every N iterations. Set to 0 to disable.

    Returns:
        Tuple of (losses_list, trained_model).
    """
    device = next(cppn.parameters()).device
    target_img = target_img.to(device)
    img_size = target_img.shape[0]

    optimizer = torch.optim.Adam(cppn.parameters(), lr=lr)
    losses = []
    start_iter = 0

    if resume_from is not None:
        from .checkpoint import save_checkpoint, load_checkpoint
        restored_losses, restored_iter, _ = load_checkpoint(resume_from, cppn, optimizer)
        losses = list(restored_losses)
        start_iter = restored_iter + 1

    cppn.train()
    for i in range(start_iter, n_iters):
        optimizer.zero_grad()

        generated = cppn.generate_image(img_size=img_size)
        loss = torch.mean((generated - target_img) ** 2)

        loss.backward()

        # Normalize gradients (matching reference implementation)
        total_norm = 0.0
        for p in cppn.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        if total_norm > 0:
            for p in cppn.parameters():
                if p.grad is not None:
                    p.grad.data /= total_norm

        optimizer.step()

        losses.append(loss.item())

        # Checkpoint
        if checkpoint_dir is not None and (i + 1) % checkpoint_interval == 0:
            from .checkpoint import save_checkpoint
            ckpt_path = os.path.join(checkpoint_dir, f'iter_{i + 1:04d}.pt')
            save_checkpoint(ckpt_path, cppn, optimizer, losses, iteration=i,
                          config={'lr': lr, 'n_iters': n_iters})

        if log_interval and (i + 1) % log_interval == 0:
            print(f"Iter {i + 1}/{n_iters}, Loss: {loss.item():.6f}")

    return losses, cppn


def train_swarm(swarm_cppn, target_img, lr=3e-3, n_iters=10000,
                swarm_interval=5, log_interval=100,
                checkpoint_dir=None, checkpoint_interval=100, resume_from=None):
    """Train SwarmKAN_CPPN with alternating SGD and swarm steps.

    Performs standard SGD optimization with periodic PSO swarm updates
    that blend spline coefficients from the particle population.

    Args:
        swarm_cppn: A SwarmKAN_CPPN model.
        target_img: Target image tensor of shape (H, W, 3) in [0, 1].
        lr: Learning rate for Adam optimizer.
        n_iters: Total number of training iterations.
        swarm_interval: Perform a swarm step every N SGD steps.
        log_interval: Print loss every N iterations. Set to 0 to disable.

    Returns:
        Tuple of (losses_list, trained_model).
    """
    device = next(swarm_cppn.parameters()).device
    target_img = target_img.to(device)
    img_size = target_img.shape[0]

    optimizer = torch.optim.Adam(swarm_cppn.parameters(), lr=lr)
    losses = []
    start_iter = 0

    if resume_from is not None:
        from .checkpoint import save_checkpoint, load_checkpoint
        restored_losses, restored_iter, _ = load_checkpoint(resume_from, swarm_cppn, optimizer)
        losses = list(restored_losses)
        start_iter = restored_iter + 1

    swarm_cppn.train()
    for i in range(start_iter, n_iters):
        optimizer.zero_grad()

        generated = swarm_cppn.generate_image(img_size=img_size)
        loss = torch.mean((generated - target_img) ** 2)

        loss.backward()

        # Normalize gradients
        total_norm = 0.0
        for p in swarm_cppn.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        if total_norm > 0:
            for p in swarm_cppn.parameters():
                if p.grad is not None:
                    p.grad.data /= total_norm

        optimizer.step()

        losses.append(loss.item())

        # Periodic swarm update
        if (i + 1) % swarm_interval == 0:
            swarm_cppn.swarm_step(current_loss=loss.detach())

        # Checkpoint
        if checkpoint_dir is not None and (i + 1) % checkpoint_interval == 0:
            from .checkpoint import save_checkpoint
            ckpt_path = os.path.join(checkpoint_dir, f'iter_{i + 1:04d}.pt')
            save_checkpoint(ckpt_path, swarm_cppn, optimizer, losses, iteration=i,
                          config={'lr': lr, 'n_iters': n_iters})

        if log_interval and (i + 1) % log_interval == 0:
            print(f"Iter {i + 1}/{n_iters}, Loss: {loss.item():.6f}")

    return losses, swarm_cppn


def train_memetic(memetic, target_img, n_generations=100, sgd_steps_per_gen=50,
                  lr=3e-3, log_interval=10,
                  checkpoint_dir=None, checkpoint_interval=100, resume_from=None):
    """Train MemeticKAN_CPPN with NES-Memetic loop.

    Delegates to the MemeticKAN_CPPN.evolve() method which handles the
    full NES + SGD loop: ES gradient estimation via antithetic sampling,
    parameter update, and local SGD refinement.

    Args:
        memetic: A MemeticKAN_CPPN instance.
        target_img: Target image tensor of shape (H, W, 3) in [0, 1].
        n_generations: Number of evolutionary generations.
        sgd_steps_per_gen: SGD steps per generation for local refinement.
        lr: Learning rate for local SGD refinement.
        log_interval: Print progress every N generations. Set to 0 to disable.

    Returns:
        Tuple of (fitness_history, best_individual).
    """
    start_gen = 0
    initial_history = None

    if resume_from is not None:
        from .checkpoint import load_checkpoint
        restored_losses, restored_iter, _ = load_checkpoint(
            resume_from, memetic.center,
        )
        start_gen = restored_iter + 1
        initial_history = list(restored_losses)
        memetic.best_fitness = min(restored_losses) if restored_losses else float('inf')

    # Define checkpoint callback
    checkpoint_callback = None
    if checkpoint_dir is not None:
        from .checkpoint import save_checkpoint as _save_ckpt
        def checkpoint_callback(gen, fitness_history):
            if (gen + 1) % checkpoint_interval == 0:
                ckpt_path = os.path.join(checkpoint_dir, f'iter_{gen + 1:04d}.pt')
                _save_ckpt(ckpt_path, memetic.center, None, fitness_history,
                          iteration=gen, config={'n_generations': n_generations, 'lr': lr})

    return memetic.evolve(
        target_img,
        n_generations=n_generations,
        sgd_steps_per_gen=sgd_steps_per_gen,
        lr=lr,
        log_interval=log_interval,
        start_generation=start_gen,
        initial_fitness_history=initial_history,
        checkpoint_callback=checkpoint_callback,
    )
