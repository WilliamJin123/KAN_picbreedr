---
date: 2026-02-24
summary: "Particle swarm optimization layered onto KAN spline coefficients — abstract PSO concepts, full implementation walkthrough, and parameter tuning guide"
---

# Tutorial 04: Swarm KAN — Particle Swarm Optimization for KAN-CPPNs

## What We're Building (Big Picture)

In Tutorial 03, we built KAN layers where every edge has a learnable spline activation. Training used gradient descent (Adam) to adjust the spline coefficients. The notebook's `SwarmKAN` added a second optimization channel — a simple cohesion-based neighbor blending that periodically nudges coefficients toward random neighbors.

Now we're going further. Instead of that simple cohesion heuristic, we're attaching a **real particle swarm optimizer (PSO)** to the spline coefficients of each KAN layer. Each layer maintains a population of "particles" — candidate coefficient vectors — that explore the activation function landscape using PSO's velocity/position/memory dynamics.

**Key insight:** The spline coefficients define the *shape* of each activation function. Gradient descent finds the nearest local minimum for these shapes. PSO explores multiple candidate shapes simultaneously, sharing information about which shapes produce good loss values. The combination lets us escape local minima that gradient descent alone would get stuck in.

Why does this matter for CPPNs specifically? CPPN art relies on interesting, diverse activation functions. If gradient descent converges every spline to the same boring near-linear shape, the images are dull. PSO maintains diversity — different particles represent different activation shapes, and the swarm dynamics balance exploitation (converging toward good shapes) with exploration (maintaining variety).

```
Tutorial 03 KAN:       coeffs ← Adam(∇loss)
Notebook SwarmKAN:     coeffs ← Adam(∇loss) + periodic cohesion blend
This Swarm KAN-CPPN:   coeffs ← Adam(∇loss) + PSO(velocity, personal_best, global_best)
```

The jump from "blend toward random neighbors" to "full PSO with velocity and memory" gives us three things the notebook version lacked:
1. **Memory** — each particle remembers its personal best position
2. **Momentum** — velocities carry particles past local minima
3. **Directed search** — particles pull toward known-good regions, not random ones

---

## The Abstract Algorithm: Particle Swarm Optimization

You already know PSO from evolutionary computation, so this is about nailing the specifics of how the three forces interact — because the parameter choices we make later are all about tuning this balance.

### The Three Forces

Every particle has a **position** (a candidate coefficient vector) and a **velocity** (how it's currently moving through coefficient space). Each step, the velocity updates based on three pulls:

```
v_new = w * v_old          ← inertia:   "keep going the direction you were going"
      + c₁ * r₁ * (p_best - pos)   ← cognitive: "pull toward YOUR best-ever position"
      + c₂ * r₂ * (g_best - pos)   ← social:    "pull toward the SWARM's best-ever position"

pos_new = pos + v_new
```

Think of it as a three-way tug-of-war:

| Force | Controlled by | What it does | Too high | Too low |
|---|---|---|---|---|
| **Inertia** | `w` (weight on old velocity) | Keeps particles moving in their current direction | Particles fly past good regions, never settling | Particles stop moving, lose exploration ability |
| **Cognitive** | `c₁` (personal pull) | Each particle remembers and returns to its own best | Particles fixate on their first lucky find | Particles ignore their own experience |
| **Social** | `c₂` (global pull) | All particles converge toward the best-known position | Premature convergence — everyone rushes to one spot | No information sharing — particles wander independently |

### Why the Randomness Matters

`r₁` and `r₂` are fresh random numbers (uniform [0, 1]) at every step, for every particle, for every dimension. This isn't just noise — it's structural:

- Without randomness: every particle with the same distance to `p_best` gets the exact same cognitive pull. They'd move in lockstep and converge identically.
- With randomness: each particle gets a different-strength pull on each dimension. One particle might overshoot the global best in dimension 5 but undershoot in dimension 12. This **dimensional decorrelation** is what gives PSO its exploration power in high-dimensional spaces.

### Convergence vs Diversity

The fundamental tradeoff: a swarm that converges fast finds good solutions quickly but might miss better ones. A swarm that stays diverse explores more but takes longer to refine.

For our use case (spline coefficients in a CPPN), we lean toward **more diversity** because:
- The loss landscape for activation shapes is highly multimodal (many different spline shapes can produce interesting images)
- We're combining PSO with gradient descent, so SGD handles the precision — PSO just needs to find the right basin

---

## From Concept to Architecture: SwarmKANCPPNLayer

The implementation inherits from `KANCPPNLayer` (Tutorial 03's production KAN layer) and adds PSO state on top. Let's walk through the constructor:

```python
# CONCEPT: Inherit from KANCPPNLayer to get the full spline machinery
# WHY: We don't rewrite the forward pass — spline interpolation, base_weight residual,
#      sigmoid normalization are all inherited. We only ADD the swarm optimization.
class SwarmKANCPPNLayer(KANCPPNLayer):

    def __init__(self, in_features, out_features, grid_size=20,
                 n_particles=5, inertia=0.7, cognitive=1.5, social=1.5):
        # CONCEPT: super().__init__() runs KANCPPNLayer's constructor
        # This creates: self.coeffs (nn.Parameter), self.base_weight, self.weights, self.grid
        super().__init__(in_features, out_features, grid_size)
        self.n_particles = n_particles
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
```

Now the PSO state. This is where it gets interesting — **all PSO state is stored as buffers, not parameters:**

```python
        # CONCEPT: PSO state dimensions = (n_particles,) + coeffs.shape
        # coeffs.shape is (out_features, in_features, grid_size)
        # So each particle is a full copy of the coefficient tensor
        coeff_shape = (n_particles,) + self.coeffs.shape

        # CONCEPT: register_buffer for PSO state — NOT nn.Parameter
        # WHY: Velocities, particle positions, and bests are NOT trained by gradient descent.
        #      They're manipulated directly by the PSO algorithm.
        #      register_buffer means:
        #        ✓ Moves to GPU with model.to(device)
        #        ✓ Saved/loaded with state_dict
        #        ✗ NOT in model.parameters() — optimizer ignores them
        #        ✗ No gradient computation
        self.register_buffer('velocities', torch.zeros(coeff_shape))
        self.register_buffer('particles', torch.randn(coeff_shape) * 0.01)
        self.register_buffer('personal_best', self.particles.clone())
        self.register_buffer('personal_best_scores',
                             torch.full((n_particles,), float('inf')))
        self.register_buffer('global_best', self.coeffs.data.clone())
        self.register_buffer('global_best_score', torch.tensor(float('inf')))
```

### Why `register_buffer` and Not Plain Tensors?

| Storage method | `model.to('cuda')` | `state_dict` | `model.parameters()` |
|---|---|---|---|
| `self.x = torch.zeros(...)` | Won't move | Not saved | Not included |
| `nn.Parameter(...)` | Moves | Saved | **Included** (optimizer trains it) |
| `register_buffer(...)` | Moves | Saved | Not included |

Plain tensors would silently stay on CPU when you call `model.to('cuda')`, causing device mismatch errors during `swarm_step()`. And they wouldn't survive `torch.save()`/`torch.load()`. Buffers solve both problems without polluting the optimizer's parameter list.

### Particle 0: The Active Particle

```python
        # CONCEPT: Particle 0 is synced with the actual coefficients
        # WHY: self.coeffs is the nn.Parameter that gradient descent trains.
        #      Particle 0 is the PSO's view of "where the active network is right now."
        #      Before any swarm step, particle 0 gets copied FROM self.coeffs.
        #      After the swarm step, particle 0 gets blended BACK INTO self.coeffs.
        #      The other particles (1..n-1) are pure PSO explorers.
        self.particles[0] = self.coeffs.data.clone()
        self.personal_best[0] = self.coeffs.data.clone()
```

This is the bridge between PSO and SGD. Gradient descent only touches `self.coeffs`. PSO only touches the `particles` buffer. Particle 0 is the translator between these two worlds.

---

## The PSO Update Step

Here's the full `swarm_step()` method, annotated:

```python
def swarm_step(self, current_loss=None):
    # === Step 1: Sync particle 0 with the gradient-trained coefficients ===
    if current_loss is not None:
        # CONCEPT: Copy current coeffs into particle 0's position
        # WHY: Gradient descent may have moved self.coeffs since the last swarm step.
        #      Particle 0 needs to "catch up" to where SGD put the coefficients.
        self.particles[0] = self.coeffs.data.clone()

        # CONCEPT: Update personal best for particle 0
        # WHY: If the current loss (after SGD) is better than particle 0's
        #      previous best, remember this position.
        if current_loss < self.personal_best_scores[0]:
            self.personal_best_scores[0] = current_loss
            self.personal_best[0] = self.coeffs.data.clone()

        # CONCEPT: Update global best across all particles
        # WHY: The global best is the single best position ANY particle has ever found.
        #      All particles pull toward it (the social force).
        if current_loss < self.global_best_score:
            self.global_best_score.fill_(current_loss)
            self.global_best.copy_(self.coeffs.data)
```

Note: only particle 0 gets evaluated through the actual network. Particles 1-4 are "shadow explorers" — they update their positions via PSO dynamics but never get a real fitness evaluation. Their personal bests stay at `inf` unless you implement full evaluation (which would be 5x the compute). This is a deliberate tradeoff: cheap exploration at the cost of less informed particle movement.

```python
    # === Step 2: PSO velocity update for ALL particles ===
    # CONCEPT: Fresh random vectors for stochastic exploration
    # WHY: r1, r2 are per-particle, per-dimension random numbers.
    #      This decorrelates the cognitive and social pulls across dimensions.
    r1 = torch.rand_like(self.velocities)
    r2 = torch.rand_like(self.velocities)

    # CONCEPT: The three PSO forces
    cognitive_component = self.cognitive * r1 * (self.personal_best - self.particles)
    social_component = self.social * r2 * (self.global_best.unsqueeze(0) - self.particles)
    #                                       ↑ unsqueeze(0) broadcasts global_best
    #                                         across all particles

    self.velocities = (
        self.inertia * self.velocities    # momentum from previous step
        + cognitive_component              # pull toward personal best
        + social_component                 # pull toward global best
    )

    # === Step 3: Update particle positions ===
    self.particles += self.velocities
```

Up to here, this is textbook PSO. The next part is the key design decision:

```python
    # === Step 4: Blend particle 0 back into the active coefficients ===
    # CONCEPT: Soft blend, not hard replacement
    # WHY: self.coeffs is being trained by gradient descent simultaneously.
    #      If we did coeffs.data = particles[0], we'd completely overwrite
    #      whatever gradient descent learned since the last swarm step.
    #      A 0.1 blend means: "90% of what SGD says + 10% of what PSO says"
    blend_factor = 0.1
    self.coeffs.data = (
        (1 - blend_factor) * self.coeffs.data
        + blend_factor * self.particles[0]
    )
```

**Key insight:** The blend factor is the **authority split** between gradient descent and PSO. It answers: "when SGD and PSO disagree about where the coefficients should be, who wins?"

---

## Design Choices & Parameter Knobs

Here's the full parameter table with behavioral implications:

| Parameter | Default | What it controls | High value (e.g., 2x) | Low value (e.g., 0.5x) |
|---|---|---|---|---|
| `inertia` | 0.7 | Velocity momentum | Particles fly far, slow to redirect — more global exploration but overshooting | Particles stop quickly, search locally — faster convergence but may get stuck |
| `cognitive` | 1.5 | Pull toward personal best | Each particle stubbornly returns to its own best — independent exploration | Particles ignore their history — more influenced by the swarm |
| `social` | 1.5 | Pull toward global best | Whole swarm converges toward one point — fast exploitation, low diversity | Particles mostly ignore the best solution — high diversity, slow convergence |
| `n_particles` | 5 | Swarm population size | More diverse exploration, but particles 1-N aren't fitness-evaluated (wasted memory) | Minimal overhead but fewer PSO dynamics to draw from |
| `blend_factor` | 0.1 | SGD vs PSO authority | PSO dominates — large perturbations to coeffs each step, potentially destabilizing SGD | PSO barely matters — almost pure gradient descent |
| `grid_size` | 20 | Spline resolution | Higher-dimensional PSO search space (more coefficients per spline) — harder for PSO to navigate | Fewer coefficients — PSO can explore more efficiently but splines are less expressive |

### The Cognitive-Social Balance

The `cognitive = social = 1.5` default is the "balanced" configuration from PSO literature. But you can shift the personality of the swarm:

```
cognitive >> social  →  "individualist" swarm
    Each particle trusts its own experience more than the group.
    Good when: the loss landscape has many equally-good basins
    Bad when: you need fast convergence to a single solution

cognitive << social  →  "conformist" swarm
    Particles rush toward wherever the global best is.
    Good when: there's one clear best basin and you want to find it fast
    Bad when: the global best is a local minimum and you're now stuck
```

For CPPN art, the "individualist" setting makes more sense — we want diverse activation shapes, not everyone converging to the same spline.

### Why `inertia = 0.7` and not 1.0?

With `inertia = 1.0`, velocities never decay. Particles accelerate indefinitely, bouncing wildly through coefficient space. This is the PSO equivalent of gradient descent with no learning rate decay — it diverges.

With `inertia = 0.7`, each step retains 70% of the previous velocity. After 10 steps with no new pulls, the velocity decays to `0.7^10 ≈ 0.03` of its original magnitude. This natural deceleration lets particles settle into promising regions.

Common PSO practice uses a **linearly decaying inertia** (start at 0.9, end at 0.4) so early generations explore and late generations exploit. Our implementation uses a fixed 0.7, which is the midpoint of that range — a reasonable compromise.

---

## Full Network: SwarmKAN_CPPN

The full network stacks `SwarmKANCPPNLayer`s and adds image generation:

```python
class SwarmKAN_CPPN(nn.Module):
    def __init__(self, n_layers, hidden_size, n_inputs=4, grid_size=20,
                 n_particles=5, inertia=0.7, cognitive=1.5, social=1.5):
        super().__init__()
        # CONCEPT: Same layer stacking as KAN_CPPN but with swarm layers
        layers = []
        layers.append(SwarmKANCPPNLayer(
            n_inputs, hidden_size, grid_size, n_particles, inertia, cognitive, social
        ))
        for _ in range(n_layers - 1):
            layers.append(SwarmKANCPPNLayer(
                hidden_size, hidden_size, grid_size,
                n_particles, inertia, cognitive, social
            ))
        # Output: hidden_size → 3 (h, s, v channels)
        layers.append(SwarmKANCPPNLayer(
            hidden_size, 3, grid_size, n_particles, inertia, cognitive, social
        ))
        self.layers = nn.ModuleList(layers)
```

The `swarm_step()` method propagates the loss to every layer:

```python
    def swarm_step(self, current_loss=None):
        # CONCEPT: Convert tensor to Python float for buffer operations
        # WHY: .item() extracts the scalar from a 0-dim tensor.
        #      Buffer comparisons (current_loss < self.global_best_score)
        #      work more reliably with plain floats than tensor scalars.
        loss_val = (current_loss.item()
                    if isinstance(current_loss, torch.Tensor)
                    else current_loss)
        for layer in self.layers:
            layer.swarm_step(current_loss=loss_val)
```

**Note:** every layer receives the *same* global loss value. This means every layer's PSO is optimizing the same objective (end-to-end image reconstruction loss). An alternative design would give each layer a layer-local loss, but that's much harder to define for intermediate layers.

### Image Generation

The `generate_image()` method is identical to `KAN_CPPN.generate_image()` — coordinate grid creation, forward pass, HSV→RGB conversion. The swarm state doesn't affect the forward pass at all. It only matters during the optimization step between forward passes.

---

## The Training Loop Pattern

Here's how SGD and PSO interleave during training:

```python
# CONCEPT: The hybrid training loop
# Each iteration: 1 SGD step + 1 PSO step
optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

for step in range(n_steps):
    # --- Phase 1: Standard gradient descent ---
    optimizer.zero_grad()
    generated = model.generate_image(img_size=128)
    loss = torch.mean((generated - target) ** 2)
    loss.backward()
    optimizer.step()

    # --- Phase 2: Swarm update ---
    # CONCEPT: Pass the current loss so PSO can update personal/global bests
    # WHY: Without the loss, the swarm has no fitness signal —
    #      particles would move based on momentum and stale bests only.
    model.swarm_step(current_loss=loss)
```

The flow within each iteration:

```
1. SGD moves self.coeffs based on gradients        (local, precise)
2. swarm_step() syncs particle 0 ← self.coeffs     (PSO sees where SGD went)
3. swarm_step() updates personal/global bests       (PSO remembers good spots)
4. swarm_step() updates all particle velocities     (PSO plans next moves)
5. swarm_step() moves all particle positions        (PSO explores)
6. swarm_step() blends particle 0 → self.coeffs    (PSO nudges SGD's position)
7. Next iteration: SGD starts from the blended position
```

**Key insight:** SGD and PSO are playing a cooperative game. SGD does precise local refinement, then PSO gives it a small kick in a direction informed by the swarm's collective memory. If the kick was bad, SGD quickly corrects on the next step (because it has exact gradients). If the kick was good, SGD exploits the new position. Over time, the swarm's memory accumulates information about which regions of coefficient space produce good loss values, guiding the kicks toward increasingly useful directions.

---

## Connections: Notebook SwarmKAN vs This SwarmKAN_CPPN

| Aspect | Tutorial 03 SwarmKAN (notebook) | This SwarmKAN_CPPN |
|---|---|---|
| Optimization strategy | Cohesion-based neighbor blending | Full PSO (velocity + memory) |
| Neighbor selection | Random 3 neighbors from same layer | No neighbors — particles are independent with global best |
| Memory | None (stateless blending) | Personal best + global best per layer |
| Momentum | None | Velocity with inertia decay |
| What's optimized | Individual spline activations (per-edge) | Full coefficient tensor (all edges in layer jointly) |
| Fitness signal | None — blind blending | Loss-based personal/global best tracking |
| Architecture | Separate `SwarmKANActivation` per edge | Inherits `KANCPPNLayer`, adds PSO buffers |
| SGD interaction | Independent (both modify coeffs) | Coordinated (particle 0 syncs with coeffs) |

The notebook version is more like a **regularizer** — it smooths spline shapes toward their neighbors. This version is a genuine **second optimizer** running in parallel with SGD.

---

## Gotchas & Common Mistakes

### 1. Buffer-parameter device mismatch

If you create a `SwarmKANCPPNLayer` and then move it to GPU with `.to('cuda')`, the buffers (`velocities`, `particles`, etc.) move automatically because they're registered with `register_buffer`. But if you accidentally stored PSO state as plain `self.x = torch.zeros(...)`, those tensors stay on CPU. The next `swarm_step()` would crash with a device mismatch when multiplying CPU buffers with GPU coefficients.

### 2. Blend factor sensitivity at the extremes

```python
blend_factor = 0.0   # PSO has zero effect — pure SGD. Why have the swarm at all?
blend_factor = 1.0   # Particle 0 completely overwrites SGD's work every step.
                      # SGD computes gradients, steps, then its work is thrown away.
blend_factor = 0.5   # SGD and PSO fight — neither can make consistent progress.
                      # SGD takes a step, then half of it gets replaced by PSO.
```

The default `0.1` is conservative — PSO provides a 10% nudge. If you increase it, also decrease the SGD learning rate so the two optimizers don't fight.

### 3. Unevaluated particles (1 through N-1)

Only particle 0 gets a real fitness evaluation (via the actual forward pass and loss computation). Particles 1-4 move according to PSO dynamics but their `personal_best_scores` stay at `float('inf')` forever. This means:
- Their cognitive pull is always toward their random initial positions
- Only the social pull (toward `global_best`) provides useful signal

This is a compute-quality tradeoff. Fully evaluating all particles would require N forward passes per swarm step instead of 1, which is 5x slower for `n_particles=5`.

### 4. Velocity explosion without bounds

The PSO implementation doesn't clamp velocities. If the cognitive and social pulls are large (particles far from bests) and inertia is high, velocities can grow unboundedly. The particles fly to extreme coefficient values, producing `NaN` or `inf` in the spline outputs. If you see `NaN` loss after enabling swarm steps, try:
- Reducing `inertia` (e.g., 0.5)
- Adding velocity clamping: `self.velocities.clamp_(-1.0, 1.0)`
- Reducing `cognitive` and `social` coefficients

### 5. The swarm step is NOT differentiable

Everything inside `swarm_step()` operates on `.data` (explicit or implicit through buffer operations). No gradients flow through the PSO update. This is correct — PSO is a gradient-free optimizer. But it means you cannot backpropagate "through" the swarm step. The training loop must be structured as: forward → loss → backward → SGD step → swarm step (in that order, never interleaved).

---

## What's Next

Tutorial 05 takes the hybrid optimization idea much further with the **Memetic KAN** — replacing PSO's random-walk exploration with a **Natural Evolution Strategy (NES)** that estimates gradients through perturbation. Where PSO has particles wandering through coefficient space guided by memory, NES systematically probes directions in parameter space to build a gradient estimate, then combines that with SGD for local refinement. It also introduces a critical architectural insight: which parameters should evolution touch, and which should it leave alone.
