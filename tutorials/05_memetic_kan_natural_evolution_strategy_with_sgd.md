---
date: 2026-02-24
summary: "NES + SGD memetic optimizer for KAN-CPPNs — why crossover fails for KANs, antithetic gradient estimation, selective parameter perturbation, and the full evolve loop"
---

# Tutorial 05: Memetic KAN — Natural Evolution Strategy with SGD

## What We're Building (Big Picture)

Tutorial 04's Swarm KAN bolted PSO onto spline coefficients — particles wandering through coefficient space, guided by memory and momentum. It works, but PSO's random-walk exploration scales poorly as the parameter count grows. With 12 layers × 22 hidden neurons × 20 grid points, we have thousands of spline coefficients. PSO particles in a thousand-dimensional space are mostly wandering blindly.

The Memetic KAN takes a fundamentally different approach. Instead of maintaining a population of independent parameter vectors, it maintains a **single center network** and uses **Natural Evolution Strategy (NES)** to estimate which direction in parameter space improves fitness — then takes a step in that direction. Think of it as gradient descent, but the gradient is estimated by poking the network with random noise instead of backpropagation.

Then, critically, it follows each ES step with **local SGD refinement** — standard gradient descent to polish whatever region the ES step landed in. This two-phase loop is what makes it "memetic":

```
Memetic = evolutionary global search + local learning

Each generation:
  Phase 1 (ES):   "Where in parameter space should I look?"     ← global, gradient-free
  Phase 2 (SGD):  "Now that I'm here, optimize locally."        ← local, gradient-based
```

**The name "memetic"** comes from Richard Dawkins' concept of memes — units of cultural transmission. In memetic algorithms, "cultural learning" (SGD refinement) is passed to the next generation alongside "genetic" information (ES-selected parameters). The ES step is the genetic part (exploration), the SGD step is the cultural part (exploitation).

### Why Not Tournament Selection + Crossover?

The earlier version of this optimizer used a traditional evolutionary approach: maintain a population, evaluate fitness, select parents via tournament, crossover their parameters, mutate. This was **destructive for KANs** because of one specific problem:

**Crossover breaks co-adapted parameters.** Each KAN layer has two parameter groups that work together:
- `base_weight` — the residual linear path (orthogonally initialized to preserve signal norm)
- `coeffs` + `weights` — the spline path (learned activation shapes and their scaling)

These are co-adapted. The spline coefficients learn shapes that *complement* what the base_weight already provides. If you crossover parent A's base_weight with parent B's coefficients, the resulting network has a base_weight tuned for different spline shapes than the ones it's paired with. The careful balance is destroyed.

NES avoids this entirely because it doesn't recombine individuals. It perturbs the *same* center network in random directions, measures which directions improved fitness, and moves the center. No crossover, no recombination, no broken co-adaptation.

---

## The Abstract Algorithm: OpenAI-ES with Antithetic Sampling

### Gradient Estimation Without Backprop

The core insight of NES: you can estimate a gradient using only function evaluations (no need for the chain rule or computational graphs). Here's the idea:

To find which direction improves fitness `f(θ)` at parameters `θ`:
1. Pick a random direction `ε` (sampled from a normal distribution)
2. Evaluate fitness at `θ + σε` and `θ - σε` (perturbed in both directions)
3. The fitness difference `f(θ + σε) - f(θ - σε)` tells you: "was moving in direction `ε` good or bad?"
4. Repeat with many random directions, average the results
5. The average is an estimate of the gradient

Mathematically:

```
∇_θ f(θ) ≈ (1 / 2Nσ) Σᵢ [f(θ + σεᵢ) - f(θ - σεᵢ)] · εᵢ
```

This is a **finite-difference gradient estimate**. It doesn't require `f` to be differentiable — you only need to evaluate it. This is why ES works for problems where backprop is hard or impossible (robotics, game playing, etc.). For our case, we *could* use backprop everywhere, but ES gives us something backprop can't: it searches over a different subspace of parameters than SGD does (more on this in the design choices section).

### Why Antithetic Pairs?

Instead of sampling `N` independent random perturbations, we sample `N/2` and use each one twice: once as `+ε` and once as `-ε`. This is called **antithetic sampling**.

```
Without antithetic:  sample ε₁, ε₂, ..., εₙ independently
    Gradient uses: f(θ + σε₁), f(θ + σε₂), ..., f(θ + σεₙ)

With antithetic:     sample ε₁, ε₂, ..., εₙ/₂
    Gradient uses: f(θ + σε₁), f(θ - σε₁), f(θ + σε₂), f(θ - σε₂), ...
```

**Key insight:** Antithetic sampling halves the variance of the gradient estimate for free. Here's why: if `ε` happens to point in a direction where both `+ε` and `-ε` increase the loss (a curved valley), the independent estimate would count both as "this direction is good." The antithetic pair correctly cancels out: `f(θ+σε) - f(θ-σε)` isolates the *slope* along `ε`, not the curvature. This gives a cleaner gradient signal from the same number of function evaluations.

### Why This Isn't "Real" Evolution

Traditional evolution: maintain a population, select the fittest, reproduce with variation.

NES: maintain a single center, probe random directions, move the center toward better directions.

NES looks like evolution (random perturbations, fitness evaluation), but it's actually closer to **stochastic gradient descent** — it's estimating and following a gradient. The "population" (the perturbation pairs) exists only for one generation and is discarded. There's no survival, no reproduction, no genetic inheritance. It's gradient descent with a very noisy gradient estimator.

---

## The Critical Design Decision: Excluding base_weight from ES

This is the most important architectural choice in the entire implementation. Let's understand why.

### The Problem: Signal Collapse in Deep Networks

From Tutorial 03, each KAN layer computes:

```
output = base(x) + spline(x)
       = x @ base_weight.T + spline_interpolation(coeffs, weights, x)
```

The `base_weight` is initialized with **orthogonal initialization**, which has a special property: it preserves the norm of the input vector exactly. If `||x|| = 1.0`, then `||x @ W_orth.T|| = 1.0`. Through 12 layers:

```
Orthogonal base_weight:  signal std stays ~0.42 through all 12 layers  ✓
Random perturbation:     signal std → 0.004 by layer 12                ✗ (collapsed)
```

ES perturbation adds random noise to parameters: `θ_new = θ + σε`. When applied to `base_weight`, this destroys the orthogonality that keeps signals alive. The perturbed network produces near-zero outputs, gets a terrible fitness score, and the ES gradient estimate is dominated by noise rather than useful signal.

### The Solution: Two Flatteners

```python
# CONCEPT: ES flattener excludes base_weight to preserve orthogonality
# WHY: Only perturb the "soft" parameters (spline coeffs and weights).
#      The "structural" parameter (base_weight) is sacred — SGD can
#      fine-tune it via gradients, but ES's random noise would destroy it.
self.es_flattener = FlattenKANParameters(self.center, exclude_base_weight=True)

# Full flattener for external use (weight sweeps, visualization, etc.)
self.flattener = FlattenKANParameters(self.center)
```

`FlattenKANParameters` (from `kan.py`) converts a KAN-CPPN's parameters into a 1D vector and back. The `exclude_base_weight=True` flag skips all `base_weight` parameters, so ES only perturbs spline coefficients and spline weights.

This creates a clear division of labor:

| Parameter | Who optimizes it | Why |
|---|---|---|
| `base_weight` | SGD only | Orthogonal structure must be preserved. Gradients are small, precise adjustments that maintain near-orthogonality. |
| `coeffs` | ES + SGD | Spline shapes benefit from global search (ES) to find good basins, then local refinement (SGD) to optimize within the basin. |
| `weights` | ES + SGD | Spline scaling is low-dimensional per layer, easy for both optimizers. |

**Key insight:** This is an **inductive bias** — we're encoding our belief that "the residual linear path is structurally important and shouldn't be randomly perturbed." This belief comes from empirical evidence: without it, deep KAN networks produce zero-signal outputs.

---

## Phase 1: ES Gradient Estimation

Let's walk through the ES phase of the `evolve()` method:

```python
def evolve(self, target_img, n_generations=100, sgd_steps_per_gen=50,
           lr=3e-3, log_interval=10):
    target_img = target_img.to(self.device)
    img_size = target_img.shape[0]
    fitness_history = []

    for gen in range(n_generations):
        # === Phase 1: ES gradient estimation (spline params only) ===

        # CONCEPT: Flatten current spline params to a 1D vector
        # WHY: ES works in flat parameter space. We need a single vector
        #      to add perturbations to, not a tree of named tensors.
        center_params = self.es_flattener.flatten().detach()
        n_params = center_params.numel()
```

The `.detach()` is important — we don't want autograd tracking the ES operations. ES is gradient-free by design.

### The Perturbation Loop

```python
        epsilons = []
        fitness_diffs = []

        with torch.no_grad():
            # CONCEPT: Pre-ES fitness baseline for gating
            # WHY: We'll compare post-ES fitness against this baseline
            #      to decide whether the ES step actually helped.
            pre_es_img = self.center.generate_image(img_size=img_size)
            pre_es_fitness = torch.mean((pre_es_img - target_img) ** 2).item()

            for i in range(self.pop_size):
                # CONCEPT: Random perturbation direction
                # WHY: Each ε is a random vector in parameter space.
                #      It defines a direction to probe.
                eps = torch.randn(n_params, device=self.device)
                epsilons.append(eps)

                # CONCEPT: Positive perturbation — move center in direction +ε
                self.es_flattener.unflatten(center_params + self.sigma * eps)
                img_pos = self.center.generate_image(img_size=img_size)
                f_pos = torch.mean((img_pos - target_img) ** 2).item()

                # CONCEPT: Negative perturbation — move center in direction -ε
                # WHY: The antithetic pair. Same direction, opposite sign.
                self.es_flattener.unflatten(center_params - self.sigma * eps)
                img_neg = self.center.generate_image(img_size=img_size)
                f_neg = torch.mean((img_neg - target_img) ** 2).item()

                # CONCEPT: Normalized fitness difference
                # WHY: Raw (f_pos - f_neg) can vary wildly in magnitude
                #      depending on how close we are to the target.
                #      Dividing by (|f_pos| + |f_neg| + ε) normalizes to
                #      roughly [-1, 1], making the gradient estimate stable
                #      across different stages of training.
                denom = abs(f_pos) + abs(f_neg) + 1e-8
                fitness_diffs.append((f_pos - f_neg) / denom)
```

Each iteration of this loop:
1. Picks a random direction `ε`
2. Evaluates the network perturbed in both `+ε` and `-ε` directions
3. Records how much better/worse the `+` direction was compared to `-`

After the loop, we have `pop_size` direction/fitness-difference pairs. Time to compute the gradient:

```python
        # CONCEPT: ES gradient = weighted sum of perturbation directions
        # WHY: If moving in direction εᵢ improved fitness (f_pos < f_neg,
        #      so fitness_diff < 0), then εᵢ contributes to "move this way."
        #      Directions that hurt fitness contribute "move away from this."
        es_grad = torch.zeros(n_params, device=self.device)
        for fd, eps in zip(fitness_diffs, epsilons):
            es_grad += fd * eps
        es_grad /= (2 * self.pop_size * self.sigma)
```

The `/ (2 * pop_size * sigma)` normalization ensures the gradient magnitude doesn't depend on the number of samples or the perturbation scale — it's a proper average, scaled to the parameter space.

---

## Phase 2: ES Update with Fitness Gating

The ES gradient tells us "move parameters in this direction to reduce loss." But ES gradient estimates are **noisy**, especially with small `pop_size`. Sometimes the estimated direction is wrong. The fitness gate protects against this:

```python
        # CONCEPT: Candidate update — move center in the estimated gradient direction
        candidate_params = center_params - self.lr_es * es_grad

        # CONCEPT: Fitness gating — only accept if it actually improved
        # WHY: The ES gradient is a noisy estimate. With pop_size=20, the
        #      estimate can be substantially wrong, especially early in training
        #      when fitness differences are large and noisy.
        #      This "elitist" gate says: "I won't move unless I'm sure it's better."
        with torch.no_grad():
            self.es_flattener.unflatten(candidate_params)
            post_es_img = self.center.generate_image(img_size=img_size)
            post_es_fitness = torch.mean((post_es_img - target_img) ** 2).item()

            if post_es_fitness >= pre_es_fitness:
                # ES made things worse — revert to pre-ES parameters
                self.es_flattener.unflatten(center_params)
```

This is a **conservative/elitist** strategy. In traditional ES, you always follow the gradient estimate. Here, we check: "did the step actually improve things?" If not, we undo it entirely. This means:

- **Early training** (large gradients, noisy estimates): many ES steps get reverted. SGD does most of the work.
- **Mid training** (ES estimates improve): ES starts finding useful directions, more steps are accepted.
- **Late training** (small improvements): ES steps are often marginal, some accepted, some not.

The downside: this gate can be overly conservative. A step that makes things temporarily worse might lead to a better basin. By reverting, we lose that exploratory move. But in practice, this conservatism prevents the catastrophic regressions that uncontrolled ES can cause.

---

## Phase 3: SGD Local Refinement

After ES (possibly) moves the parameters to a new region, SGD polishes within that region:

```python
        # === Phase 2: SGD local refinement ===

        # CONCEPT: Fresh optimizer each generation
        # WHY: Adam maintains per-parameter momentum and variance estimates.
        #      If ES just moved the parameters to a completely different region,
        #      Adam's cached momentum is WRONG — it points in directions that
        #      were good for the OLD parameter values, not the new ones.
        #      A fresh Adam starts with no momentum, re-learning the local
        #      gradient landscape from scratch.
        sgd_optimizer = torch.optim.Adam(self.center.parameters(), lr=lr)

        self.center.train()
        for step in range(sgd_steps_per_gen):
            sgd_optimizer.zero_grad()
            generated = self.center.generate_image(img_size=img_size)
            loss = torch.mean((generated - target_img) ** 2)
            loss.backward()

            # CONCEPT: Gradient normalization — scale all gradients to unit norm
            # WHY: Different layers can have wildly different gradient magnitudes.
            #      Without normalization, layers with large gradients dominate
            #      the update, and layers with small gradients barely move.
            #      Normalizing to unit total norm gives every parameter an
            #      equal "vote" in the update direction.
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
```

**Key insight about fresh Adam:** This is not obvious and is a common bug. If you reuse the same Adam optimizer across ES generations, Adam's internal state (running mean and variance of gradients) reflects the *old* parameter region. When ES jumps to a new region, these statistics are stale. The effect: Adam takes oversized steps in the wrong direction for the first few iterations, potentially undoing the ES improvement. Resetting Adam each generation costs a few iterations of warm-up but avoids this poisoning.

### Gradient Normalization

The gradient normalization deserves attention. It computes the L2 norm across all parameter gradients, then divides every gradient by that norm:

```
Before normalization:  ∇base_weight ≈ 0.001,  ∇coeffs ≈ 5.0
After normalization:   ∇base_weight ≈ 0.0002, ∇coeffs ≈ 0.999

→ Adam with lr=3e-3 moves all parameters by similar amounts
```

Without this, the spline coefficients (which have large gradients because small changes in spline shape create big changes in output) would dominate the update, while the base_weight (which has small gradients because it's orthogonal and near-norm-preserving) would barely move. Normalization equalizes their learning rates.

---

## Design Choices & Parameter Knobs

| Parameter | Default | What it controls | Too high | Too low |
|---|---|---|---|---|
| `sigma` | 0.02 | Perturbation scale for ES probes | Large perturbations scramble the network — fitness of perturbed networks is random noise, gradient estimate is garbage | Tiny perturbations barely change fitness — finite-difference signal is drowned by numerical noise, gradient estimate is also garbage |
| `lr_es` | 0.01 | Step size for ES gradient update | Overshoots — jumps past the basin ES found | Undershoots — barely moves, ES discoveries aren't exploited |
| `pop_size` | 20 | Number of perturbation pairs | Better gradient estimate (lower variance) but 2× more forward passes per pair | Noisier gradient estimate, more ES steps get reverted by the fitness gate |
| `sgd_steps_per_gen` | 50 | Local refinement budget per generation | More polish per ES step — good if ES finds the right basin, wasteful if ES step was reverted | Less polish — each generation's SGD barely moves, ES does more of the work |
| `lr` (SGD) | 3e-3 | Adam learning rate for local refinement | Fast convergence but may overshoot, especially after an ES jump to unfamiliar territory | Slow, careful refinement — good stability but needs more steps per generation |
| `n_generations` | 100 | Total ES+SGD cycles | More time to explore and refine (total forward passes = generations × (2 × pop_size + sgd_steps)) | Less exploration, may not find good basins |

### The sigma-lr_es Interaction

These two parameters are tightly coupled:

```
sigma controls how FAR the probes reach from the center.
lr_es controls how FAR the center moves based on the probe results.
```

If `sigma` is large (0.1) and `lr_es` is also large (0.1), the center takes huge jumps based on information from far away — unstable, erratic behavior.

If `sigma` is small (0.001) and `lr_es` is small (0.001), the center barely moves and barely explores — essentially just expensive SGD with extra steps.

The default `sigma=0.02, lr_es=0.01` is calibrated so that:
- Probes reach far enough to detect different basins (σ × √n_params ≈ a few units in parameter space)
- But the center moves conservatively (lr_es << σ, so each step is smaller than the probe radius)

### Compute Budget Analysis

Per generation, the total forward passes are:

```
ES probes:          2 × pop_size = 40 forward passes  (+ and - perturbation)
ES fitness gate:    2 forward passes (pre and post)
SGD refinement:     sgd_steps_per_gen = 50 forward passes (each with backward too)

Total per gen:      ~92 forward passes, ~50 backward passes
Over 100 gens:      ~9,200 forward passes, ~5,000 backward passes
```

Compare to pure SGD: 5,000 forward+backward passes gets you 5,000 gradient steps. The memetic approach trades half its compute budget for ES exploration. Whether this tradeoff pays off depends on how multimodal the loss landscape is — if there's one smooth basin, pure SGD wins. If there are many basins, ES helps find the better ones.

---

## The Inductive Biases

Every algorithm encodes beliefs about the problem. Here's what the Memetic KAN "believes":

### 1. base_weight is structural, spline params are functional

The ES exclusion of `base_weight` encodes: "The residual linear path is infrastructure — it keeps signals flowing. Don't mess with it randomly. The spline shapes are the creative part — explore those freely."

This is analogous to: in a building, you don't randomly rearrange the load-bearing walls (base_weight). You redecorate the rooms (spline coefficients).

### 2. SGD is better at local optimization than ES

The algorithm allocates 50 SGD steps per 1 ES step. This encodes: "Once we're in the right neighborhood, exact gradients are better than noisy estimates." This is almost always true — SGD has vastly more information per step (exact gradient vs. noisy estimate from 20 samples).

### 3. Only accept improvements (conservatism)

The fitness gate encodes: "It's better to not move than to move in the wrong direction." This is a risk-averse stance. An alternative "optimistic" approach would always follow the ES gradient (trusting the estimate). The conservative approach is better when:
- The fitness landscape has cliffs (large regions where small parameter changes cause catastrophic loss increases)
- Training time is limited (can't afford to waste generations recovering from bad ES steps)

### 4. Fresh starts are better than stale momentum

Resetting Adam each generation encodes: "The local gradient landscape changes after ES moves us. Past momentum information is misleading." This is the memetic philosophy — each "generation" starts with a clean learning state, inheriting only the parameters (not the optimizer state) from the previous generation.

---

## reset_weights: Probing Where Knowledge Lives

The `reset_weights` method is an experimental tool for understanding what the network learned:

```python
def reset_weights(self, individual=None, default_value=1.0):
    # CONCEPT: Reset weights but keep spline coefficients
    # WHY: If we reset all weights to 1.0 and re-initialize base_weight,
    #      the only "knowledge" left is in the spline coefficient shapes.
    #      If the network can still produce recognizable images → the spline
    #      shapes ARE the learned representation.
    #      If it produces garbage → the weight values were essential too.
    if individual is None:
        individual = self.get_best()

    with torch.no_grad():
        for layer in individual.layers:
            layer.weights.fill_(default_value)       # reset spline scaling
            nn.init.orthogonal_(layer.base_weight)   # fresh orthogonal init
    return individual
```

This creates a **controlled experiment**:

| What's preserved | What's reset | Expected result |
|---|---|---|
| Spline coefficients (learned shapes) | base_weight + spline weights | If shapes carry the pattern → image is distorted but recognizable |
| Nothing | Everything | Random image (control) |

This connects to a deeper question about KAN representations: **is the knowledge in the spline shapes or in the weight magnitudes?** In standard MLPs, the knowledge is entirely in the weight values. In KANs, there are two places knowledge can live:

1. **Spline shapes** (what function each edge computes) — captured by `coeffs`
2. **Weight scaling** (how much each edge's output matters) — captured by `weights` and `base_weight`

The `reset_weights` experiment isolates these. If spline shapes alone can reconstruct the image (even imperfectly), it means KANs learn differently from MLPs — the *type* of function matters, not just the magnitude.

---

## Connections: SwarmKAN vs MemeticKAN

| Aspect | Swarm KAN (Tutorial 04) | Memetic KAN (this tutorial) |
|---|---|---|
| Global search method | PSO (velocity + memory) | NES (perturbation-based gradient) |
| Search space | Full coefficient tensor | Spline params only (excludes base_weight) |
| Population purpose | Persistent particles that accumulate experience | Ephemeral probes, discarded each generation |
| Fitness evaluation | Only particle 0 (cheap but limited) | All perturbation pairs (expensive but informative) |
| SGD integration | Soft blend (10%) each step | Full SGD phase (50 steps) each generation |
| Memory across steps | Particle positions, velocities, personal bests | Only the center network persists |
| Failure mode | Velocity explosion, particle divergence | Bad sigma: scrambled probes or no signal |
| Compute cost per step | 1 forward pass + cheap buffer ops | 2 × pop_size + sgd_steps forward passes |

### Why NES Won Over PSO for This Use Case

PSO's strength is maintaining a diverse population of solutions. But in high-dimensional spaces (thousands of spline coefficients), a handful of particles can't meaningfully cover the space. Each PSO step has only 1 evaluated particle out of 5 — the other 4 are moving blind.

NES uses 20 pairs (40 probe evaluations) to estimate one gradient vector. Every probe contributes information. The gradient estimate is imperfect but *informed* — it captures the local curvature of the fitness landscape, not just random wandering.

The tradeoff: NES is 40x more expensive per step than PSO. But each step is incomparably more useful. And the fitness gating ensures that bad steps are reverted, so the worst case is "wasted compute" rather than "corrupted parameters."

---

## Gotchas & Common Mistakes

### 1. Stale optimizer state across ES jumps

```python
# WRONG: Reuse optimizer across generations
optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
for gen in range(n_generations):
    # ES moves parameters to a new region...
    for step in range(sgd_steps):
        optimizer.step()  # Adam's momentum is from the OLD region!

# CORRECT: Fresh optimizer each generation
for gen in range(n_generations):
    # ES moves parameters...
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)  # reset
    for step in range(sgd_steps):
        optimizer.step()  # starts fresh, no stale momentum
```

Symptom of the bug: loss spikes at the start of each SGD phase, then recovers. The spike is Adam overshooting with stale momentum. With a fresh optimizer, the SGD phase monotonically decreases loss from the start.

### 2. ES perturbing base_weight causes signal collapse

```python
# WRONG: Flatten ALL parameters for ES
flattener = FlattenKANParameters(model)  # includes base_weight

# After perturbation: base_weight is no longer orthogonal
# Layer 1: signal std 1.0 → 0.82
# Layer 6: signal std 1.0 → 0.12
# Layer 12: signal std 1.0 → 0.004  ← effectively zero
# Result: all perturbed networks produce near-identical zero-ish images
# Fitness differences ≈ 0 → ES gradient ≈ 0 → no learning from ES

# CORRECT: Exclude base_weight
flattener = FlattenKANParameters(model, exclude_base_weight=True)
```

This is the most impactful bug in the implementation history. Without the exclusion, ES contributes nothing — it's 40 wasted forward passes per generation because all probes produce nearly identical (collapsed) outputs.

### 3. Sigma tuning: the "Goldilocks" problem

```
sigma = 0.001:  perturbations are tiny
    f(θ + σε) ≈ f(θ - σε) ≈ f(θ)
    fitness_diff ≈ 0 for all directions
    ES gradient ≈ 0 → no movement
    Symptom: ES steps are always reverted (they don't change anything)

sigma = 0.5:    perturbations are huge
    f(θ + σε) and f(θ - σε) are both random garbage
    fitness_diff is random noise
    ES gradient is random → moves center in random direction
    Symptom: ES steps are always reverted (they make things much worse)

sigma = 0.02:   perturbations are meaningful but not destructive
    f(θ + σε) is slightly different from f(θ - σε)
    fitness_diff captures real gradient information
    ES gradient points toward improvement
```

A good diagnostic: print the acceptance rate of ES steps (what fraction aren't reverted). If it's near 0%, sigma is too high or too low. If it's near 100%, lr_es might be too small (ES isn't moving far enough to matter). The sweet spot is typically 30-70% acceptance.

### 4. Fitness normalization prevents gradient scale drift

```python
# WITHOUT normalization:
fitness_diff = f_pos - f_neg
# Early training: f_pos ≈ 0.5, f_neg ≈ 0.5 → diff ≈ ±0.01
# Late training:  f_pos ≈ 0.001, f_neg ≈ 0.001 → diff ≈ ±0.00001
# The ES gradient shrinks 1000x as training progresses → ES stops contributing

# WITH normalization:
denom = abs(f_pos) + abs(f_neg) + 1e-8
fitness_diff = (f_pos - f_neg) / denom
# Early: (0.51 - 0.49) / (0.51 + 0.49) ≈ 0.02
# Late:  (0.00101 - 0.00099) / (0.00101 + 0.00099) ≈ 0.01
# Gradient magnitude stays in a consistent range throughout training
```

Without this normalization, the ES learning rate `lr_es` would need to increase as training progresses (to compensate for shrinking gradients). The normalization makes `lr_es` effective at all stages.

### 5. The `.item()` trap with tensor comparisons

```python
# In swarm_step (Tutorial 04):
if current_loss < self.global_best_score:  # tensor < tensor → tensor (bool)

# In evolve():
loss_val = current_loss.item()  # extract Python float first
# Now: float < float → Python bool
```

When comparing a Python float against a buffer tensor, the result is usually correct but occasionally surprising — especially with `float('inf')` initial values. The `.item()` conversion in `evolve()` ensures clean Python float comparisons throughout the ES phase.

---

## What's Next

With Tutorials 04 and 05, we've covered both hybrid optimization strategies for KAN-CPPNs — PSO-based swarm exploration and NES-based memetic optimization. The key takeaways:

- **PSO (Tutorial 04)** is cheap per step but blind in high dimensions. Best when you want a persistent exploration mechanism running alongside SGD with minimal compute overhead.
- **NES-Memetic (Tutorial 05)** is expensive per step but informed. Best when you can afford the compute budget and want directed exploration that respects the network's architectural constraints (base_weight exclusion).

Both encode the same fundamental insight: **gradient descent alone isn't enough for KAN-CPPNs.** The spline coefficient landscape is too multimodal, and the co-adaptation between base_weight and spline parameters creates structure that must be respected. Hybrid optimization — whether through swarm dynamics or evolution strategy — gives us the exploration needed to discover interesting activation shapes, while SGD provides the precision to refine them into compelling images.
