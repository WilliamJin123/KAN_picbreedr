---
date: 2026-02-15
summary: "Line-by-line PyTorch walkthrough of KAN layers — from naive spline activations through vectorized implementations to swarm-based coefficient optimization"
---

# Tutorial 03: Kolmogorov-Arnold Networks (KAN) in PyTorch

## What We're Building (Big Picture)

In Tutorial 02, we built a CPPN where every layer does this:

```
x → Dense multiply → split into groups → apply fixed activation (sin, gaussian, etc.) → concatenate
```

The activation functions are **fixed** — sin is always sin, gaussian is always gaussian. The only learnable part is the weight matrix `W`.

A KAN flips this. Instead of `learnable weights + fixed activations`, KAN does `learnable activations + summation`:

```
MLP layer:   y_j = activation( Σ_i  w_ji * x_i )     ← fixed activation, learnable weights
KAN layer:   y_j = Σ_i  φ_ji(x_i)                     ← learnable activation per edge, no weights
```

Each `φ_ji` is a **learnable univariate function** — specifically, a spline defined by a set of control points. The spline *is* the edge. There's no separate weight matrix.

**Why this matters for our project:** CPPNs produce interesting patterns because they use diverse activations (sin, gaussian, etc.). KAN takes this idea to the extreme — *every single connection* gets its own learned activation function. Instead of the architect choosing "15 cache, 4 gaussian, 2 identity, 1 sin" per layer, the network discovers what functions it needs.

You already know the **Kolmogorov-Arnold theorem**: any multivariate continuous function `f(x₁,...,xₙ)` can be written as sums and compositions of univariate functions. KAN is the direct neural network implementation of that theorem — each layer applies univariate functions (splines) and sums them, exactly mirroring the theorem's structure.

---

## The Core Idea: Spline Interpolation as a Learnable Function

### What's a Spline? (The 30-Second Version)

You know this already from KAN theory, but let's nail the implementation detail:

A **piecewise-linear spline** is the simplest kind. You have:
- A **grid** of fixed x-positions: `[0.0, 0.1, 0.2, ..., 1.0]`
- A **coefficient** (y-value) at each grid point: `[c₀, c₁, c₂, ..., cₙ]`

To evaluate the spline at some input `x`:
1. Find which two grid points `x` falls between
2. Linear interpolate between their coefficients

### Math vs PyTorch

| Math notation | What it means | PyTorch |
|---|---|---|
| Grid `G = {g₀, g₁, ..., g_k}` | Fixed knot points | `torch.linspace(0, 1, grid_size)` |
| Coefficients `C = {c₀, c₁, ..., c_k}` | Learnable y-values at knots | `nn.Parameter(torch.randn(grid_size))` |
| Find interval: `gᵢ ≤ x < gᵢ₊₁` | Which cell is x in? | `idx = (x_norm * (grid_size-1)).long().clamp(0, grid_size-2)` |
| Fraction: `t = (x - gᵢ)/(gᵢ₊₁ - gᵢ)` | How far between the two knots | `frac = (x_norm * (grid_size-1)) - idx.float()` |
| Interpolate: `(1-t)·cᵢ + t·cᵢ₊₁` | Blend the two coefficients | `(1 - frac) * coeffs[idx] + frac * coeffs[idx + 1]` |

### Minimal Isolated Example

```python
import torch
import torch.nn as nn

# A single learnable spline with 10 control points
grid_size = 10
grid = torch.linspace(0, 1, grid_size)        # fixed x-positions
coeffs = torch.randn(grid_size)               # learnable y-values

# Evaluate at x = 0.35
x = torch.tensor(0.35)
scaled = x * (grid_size - 1)     # 0.35 * 9 = 3.15
idx = int(scaled)                # 3  (left neighbor)
frac = scaled - idx              # 0.15  (how far past left neighbor)

result = (1 - frac) * coeffs[idx] + frac * coeffs[idx + 1]
# Blends 85% of coeffs[3] with 15% of coeffs[4]
```

**Key insight:** The spline is just a lookup table with linear blending. The "learning" happens because the coefficients (`c₀, c₁, ...`) are `nn.Parameter`s — gradient descent adjusts them to make the spline approximate whatever univariate function is needed.

---

## New PyTorch Concepts

### Concept 1: `torch.sigmoid` for Input Normalization

**The problem:** Our spline grid lives in `[0, 1]`, but network inputs can be any real number.

**The solution:** Push inputs through sigmoid first: `x_norm = torch.sigmoid(x)`.

```python
import torch

x = torch.tensor([-100.0, -1.0, 0.0, 1.0, 100.0])
print(torch.sigmoid(x))
# tensor([0.0000, 0.2689, 0.5000, 0.7312, 1.0000])
```

| Math | What it does | Range |
|---|---|---|
| `sigmoid(x) = 1/(1+e⁻ˣ)` | Squashes any real number to (0, 1) | (0, 1) |
| `tanh(x)` | Squashes to (-1, 1) | (-1, 1) |

**Key insight:** The notebook uses sigmoid for most layers (grid on `[0,1]`) and tanh for the `EnhancedPureKANLayer` (grid on `[-1,1]`). The choice determines what range of inputs the spline can "see" — sigmoid compresses everything to a narrow band, while tanh preserves more dynamic range.

### Concept 2: Integer Indexing with `.long()` and `.clamp()`

**What you already know:** Array indexing in NumPy requires integers.

**PyTorch equivalent:** `.long()` converts a float tensor to int64 (truncating, not rounding). `.clamp(min, max)` clips values to a range.

```python
scaled = torch.tensor([3.7, 8.2, -0.5, 9.9])

idx = scaled.long()           # tensor([3, 8, 0, 9]) — truncates toward zero
idx = idx.clamp(0, 7)         # tensor([3, 7, 0, 7]) — clips to valid range [0, grid_size-2]
```

**Why `clamp(0, grid_size - 2)`?** Because we access `coeffs[idx + 1]`. If `idx` were `grid_size - 1` (the last index), then `idx + 1` would be out of bounds. Clamping to `grid_size - 2` ensures `idx + 1` is always valid.

**NumPy equivalent:** `np.clip(x.astype(int), 0, n-2)` — same idea.

### Concept 3: Fractional Interpolation

The `(1 - frac) * left + frac * right` pattern is linear interpolation. This works element-wise on tensors:

```python
# For a batch of 1000 inputs, all interpolated simultaneously
frac = torch.rand(1000)                  # random fractions in [0, 1)
left = torch.randn(1000)                 # left coefficients
right = torch.randn(1000)                # right coefficients
result = (1 - frac) * left + frac * right  # shape: (1000,)
```

**Key insight:** No loops needed. PyTorch broadcasts the arithmetic across the entire batch. This is why KAN code looks mathematical — `(1-frac)*left + frac*right` IS the math, applied to tensors instead of scalars.

### Concept 4: `torch.gather` for Batched Index Lookups

**The problem:** We have a coefficient tensor of shape `(out_features, in_features, grid_size)` and we need to look up different grid indices for each sample in the batch.

**The simple way (Python loops):**
```python
# Slow — Python loop over every output and input feature
for o in range(out_features):
    for i in range(in_features):
        left_coeff = coeffs[o, i, idx[:, i]]     # idx[:, i] is the index per batch sample
```

**The fast way (`torch.gather`):**
```python
# gather(input, dim, index) — picks elements along 'dim' using 'index'
# Think of it as: output[b][o][i] = input[b][o][i][ index[b][o][i] ]

coeffs_expanded = coeffs.unsqueeze(0).expand(batch_size, -1, -1, -1)
# shape: (batch_size, out_features, in_features, grid_size)

idx_expanded = idx.unsqueeze(3)
# shape: (batch_size, out_features, in_features, 1)

gathered = torch.gather(coeffs_expanded, dim=3, index=idx_expanded).squeeze(3)
# shape: (batch_size, out_features, in_features)
```

**NumPy equivalent:** There's no direct equivalent — NumPy would use fancy indexing like `coeffs[np.arange(batch)[:, None, None], out_idx[None, :, None], in_idx[None, None, :], idx]`, which is arguably even harder to read. `torch.gather` is PyTorch's structured way of saying "use this tensor of indices to pick from that tensor."

**Key insight:** `gather` eliminates Python `for` loops over features. The speed difference is dramatic — Python loops are ~100x slower than vectorized operations on GPU.

### Concept 5: `.unsqueeze()` and `.expand()` — Broadcasting Dimensions

**The problem:** You have tensors of different shapes and need them to align for element-wise operations.

**`.unsqueeze(dim)`** adds a dimension of size 1 at position `dim`:
```python
x = torch.randn(32, 8)            # (batch, in_features)
x = x.unsqueeze(1)                # (batch, 1, in_features) — added dim at position 1
```

**`.expand(sizes)`** repeats a size-1 dimension without copying data:
```python
x = torch.randn(32, 1, 8)         # (batch, 1, in_features)
x = x.expand(-1, 16, -1)          # (batch, 16, in_features) — -1 means "keep this size"
# No memory copy! PyTorch uses stride tricks (like NumPy broadcasting)
```

**NumPy equivalent:** `np.expand_dims(x, axis=1)` for unsqueeze, `np.broadcast_to(x, shape)` for expand.

**When you see this pattern in KAN code:**
```python
x_norm = torch.sigmoid(x).unsqueeze(1)  # (batch, 1, in_features)
# The '1' dimension will broadcast across out_features
```

**Key insight:** `unsqueeze` + `expand` is how you do "apply this operation across a new dimension" without Python loops. If you see `unsqueeze(1)` in KAN code, it's preparing to broadcast across output features.

### Concept 6: Nested `nn.ModuleList`

The naive KAN needs one spline activation per (output, input) pair. That's `out_features × in_features` splines:

```python
# nn.ModuleList of nn.ModuleLists — a 2D grid of modules
self.activations = nn.ModuleList([
    nn.ModuleList([
        UnivariateSplineActivation(grid_size)
        for _ in range(in_features)      # one per input
    ]) for _ in range(out_features)      # one row per output
])

# Access: self.activations[out_idx][in_idx]
```

**Why not a plain nested list?** Same reason as Tutorial 02 — PyTorch only auto-discovers modules inside `nn.ModuleList`. A plain list of lists would be invisible to `model.parameters()`.

**Key insight:** The nested `nn.ModuleList` is the *naive* approach. The `VectorizedKANLayer` replaces this entire 2D grid of Module objects with a single `nn.Parameter` tensor of shape `(out_features, in_features, grid_size)`. Same math, way less overhead.

---

## Line-by-Line Walkthrough: The Evolution

The reference notebook builds KAN layers in four progressively optimized versions. We'll walk through each one, showing what changed and why.

### Version 1: The Naive Implementation (Cell 2)

This is the clearest version — one Python object per spline, explicit loops over every connection. Slow, but easy to understand.

#### `UnivariateSplineActivation` — One Learnable Spline

```python
# CONCEPT: nn.Module because it has learnable parameters (the spline coefficients)
# WHY: Each spline is a separate little "model" that PyTorch needs to track
class UnivariateSplineActivation(nn.Module):

    # CONCEPT: grid_size = number of control points in the spline
    # spline_degree is declared but not actually used (linear interp only)
    def __init__(self, grid_size=10, spline_degree=3):
        super().__init__()
        self.grid_size = grid_size
        self.spline_degree = spline_degree

        # CONCEPT: nn.Parameter with requires_grad=False — a constant stored in the module
        # WHY: The grid positions are fixed (evenly spaced 0 to 1), not learned.
        #      But they're stored as a Parameter so they appear in state_dict.
        # NOTE: A better approach is register_buffer (used in later versions)
        self.grid = nn.Parameter(torch.linspace(0, 1, grid_size), requires_grad=False)

        # CONCEPT: nn.Parameter (default requires_grad=True) — these ARE learned
        # WHY: The y-values at each grid point define the spline's shape.
        #      Gradient descent will adjust these to approximate whatever function is needed.
        self.coeffs = nn.Parameter(torch.randn(grid_size))

    def forward(self, x):
        # CONCEPT: sigmoid squashes any real input to (0, 1) — our grid domain
        # WHY: Without this, inputs outside [0,1] would index beyond the grid
        x_norm = torch.sigmoid(x)

        # CONCEPT: Scale normalized input to grid index space
        # If grid_size=10, this maps [0,1] → [0,9]
        # .long() truncates to integer (floor), giving the LEFT grid point index
        # .clamp(0, grid_size-2) ensures idx+1 is always valid
        idx = (x_norm * (self.grid_size - 1)).long().clamp(0, self.grid_size - 2)

        # CONCEPT: frac = how far between the two neighboring grid points
        # If scaled = 3.7, then idx = 3 and frac = 0.7
        frac = (x_norm * (self.grid_size - 1)) - idx.float()

        # CONCEPT: Linear interpolation between neighboring coefficients
        # (1 - 0.7) * coeffs[3] + 0.7 * coeffs[4]
        # This IS the spline evaluation — blend two neighbors by distance
        val = self.coeffs[idx] * (1 - frac) + self.coeffs[idx + 1] * frac
        return val
```

**What `forward` does geometrically:** Given an input scalar `x`, it finds the two nearest grid points, reads their learned coefficients, and returns a weighted blend. The result is a piecewise-linear curve whose shape is entirely determined by the learnable `coeffs`.

#### `KANLayer` — One Layer of the KAN

```python
# CONCEPT: A KAN layer has one spline activation per (output, input) pair
# WHY: The Kolmogorov-Arnold theorem says: y_j = Σ_i φ_ji(x_i)
#      Each φ_ji is a separate univariate function (spline) on one input dimension
class KANLayer(nn.Module):
    def __init__(self, in_features, out_features, grid_size=10, spline_degree=3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # CONCEPT: Nested nn.ModuleList — a 2D grid of spline activations
        # WHY: We need out_features × in_features separate splines.
        #      Outer list = one row per output neuron
        #      Inner list = one spline per input dimension for that output
        # NOTE: This creates out*in separate Module objects — heavy on memory/overhead
        self.activations = nn.ModuleList([
            nn.ModuleList([
                UnivariateSplineActivation(grid_size, spline_degree)
                for _ in range(in_features)
            ]) for _ in range(out_features)
        ])

    def forward(self, x):
        # x: [batch_size, in_features]
        outputs = []

        # CONCEPT: Python loop over every output neuron — THIS IS THE SLOW PART
        for out_idx in range(self.out_features):
            summed = 0

            # For each output, sum the spline-transformed inputs
            # This is the Σ_i φ_ji(x_i) from the KA theorem
            for in_idx in range(self.in_features):
                act = self.activations[out_idx][in_idx]

                # CONCEPT: x[:, in_idx] slices one feature across all batch samples
                # Shape: (batch_size,)
                summed += act(x[:, in_idx])

            # CONCEPT: unsqueeze(1) adds a dimension → (batch_size, 1)
            # WHY: So we can concatenate along dim=1 later to form (batch_size, out_features)
            outputs.append(summed.unsqueeze(1))

        # CONCEPT: torch.cat joins along dim=1 → (batch_size, out_features)
        return torch.cat(outputs, dim=1)
```

**Performance note:** For a layer with 32 inputs and 32 outputs, this creates 1,024 separate Module objects and runs 1,024 Python loop iterations per forward pass. That's why versions 2-4 exist.

#### `KANNetwork` — Stacking Layers

```python
# CONCEPT: The full KAN is just layers stacked sequentially
# WHY: Same principle as stacking nn.Linear layers in an MLP
class KANNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dims=[32, 32], output_dim=1, grid_size=10, spline_degree=3):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims

        # CONCEPT: Build KAN layers for each pair of consecutive dimensions
        # [input_dim, 32, 32] → KANLayer(input_dim→32), KANLayer(32→32)
        for i in range(len(dims) - 1):
            layers.append(KANLayer(dims[i], dims[i+1], grid_size, spline_degree))
        self.kan_layers = nn.ModuleList(layers)

        # CONCEPT: Final output is a plain nn.Linear, NOT a KAN layer
        # WHY: The output is just a linear projection (weighted sum).
        #      No need for a learnable spline on the final transformation —
        #      the KAN layers already provide all the nonlinearity needed.
        self.output_layer = nn.Linear(dims[-1], output_dim)

    def forward(self, x):
        for layer in self.kan_layers:
            x = layer(x)
        return self.output_layer(x)
```

---

### Version 2: Vectorized KAN Layer (Cell 4)

The big idea: replace the 2D grid of Module objects with a **single coefficient tensor** and do all the interpolation in one shot using broadcasting.

```python
class VectorizedKANLayer(nn.Module):
    def __init__(self, in_features, out_features, grid_size=10):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size

        # CONCEPT: One 3D tensor replaces out*in separate Module objects
        # Shape: (out_features, in_features, grid_size)
        # coeffs[o, i, :] = the spline coefficients for edge (i → o)
        # WHY: A single tensor can be indexed with broadcasting, eliminating Python loops
        self.coeffs = nn.Parameter(torch.randn(out_features, in_features, grid_size))

        # CONCEPT: register_buffer instead of nn.Parameter(requires_grad=False)
        # WHY: Cleaner — a buffer is explicitly "not a parameter."
        #      It still moves to GPU with model.to(device) and gets saved in state_dict.
        self.register_buffer('grid', torch.linspace(0, 1, grid_size))

        # CONCEPT: Separate weight matrix that SCALES the spline outputs
        # WHY: This adds expressiveness — the spline defines the function shape,
        #      and the weight controls the magnitude. Like having both "what function"
        #      and "how much of it" per connection.
        # NOTE: This departs from the pure KA theorem (which has no weights).
        #       Version 3 ("PureKAN") removes this.
        self.weights = nn.Parameter(torch.randn(out_features, in_features))

    def forward(self, x):
        # x shape: [batch_size, in_features]
        batch_size = x.size(0)

        # CONCEPT: unsqueeze(1) adds dimension for broadcasting across out_features
        # (batch, in_features) → (batch, 1, in_features)
        # The '1' will broadcast to match out_features later
        x_norm = torch.sigmoid(x).unsqueeze(1)

        # Same index + frac calculation as Version 1, but on 3D tensors now
        scaled = x_norm * (self.grid_size - 1)
        idx = scaled.long().clamp(0, self.grid_size - 2)   # (batch, 1, in_features)
        frac = scaled - idx.float()                         # (batch, 1, in_features)

        # CONCEPT: Build index tensors for advanced indexing
        # WHY: We need to pick coeffs[out, in, grid_idx] for every
        #      (batch_sample, output_neuron, input_neuron) combination at once
        batch_indices = torch.arange(batch_size, device=x.device).view(-1, 1, 1)
        out_indices = torch.arange(self.out_features, device=x.device).view(1, -1, 1)
        in_indices = torch.arange(self.in_features, device=x.device).view(1, 1, -1)

        # CONCEPT: .expand() broadcasts idx from (batch, 1, in) to (batch, out, in)
        # WHY: Every output neuron uses the same grid indices (same input normalization),
        #      but reads different coefficients (different spline per output)
        idx_expanded = idx.expand(-1, self.out_features, -1)

        # CONCEPT: Advanced indexing to grab left and right coefficients for ALL edges at once
        # coeffs shape: (out, in, grid_size)
        # We're indexing: coeffs[out_indices, in_indices, idx_expanded]
        # → reads one coefficient per (out, in) pair per batch sample
        coeffs_left = self.coeffs[out_indices, in_indices, idx_expanded]
        coeffs_right = self.coeffs[out_indices, in_indices, idx_expanded + 1]

        # Same linear interpolation as before, but fully batched
        # All shapes: (batch, out_features, in_features)
        interpolated = (1 - frac) * coeffs_left + frac * coeffs_right

        # CONCEPT: Element-wise multiply by weights, then sum across inputs
        # WHY: weighted[b, o, i] = weight[o, i] * spline_output[b, o, i]
        #      Then Σ_i gives us the output for neuron o
        # unsqueeze(0) on weights adds batch dimension for broadcasting
        weighted = interpolated * self.weights.unsqueeze(0)
        output = weighted.sum(dim=2)   # (batch, out_features)

        return output
```

**What changed from Version 1:**

| Version 1 (Naive) | Version 2 (Vectorized) |
|---|---|
| `out*in` separate `nn.Module` objects | One `nn.Parameter` tensor `(out, in, grid_size)` |
| Python `for o in for i in` loops | Broadcasting + advanced indexing |
| Each spline evaluated independently | All splines evaluated in one tensor operation |
| `register_buffer` not used for grid | `register_buffer` used (cleaner) |
| No weight scaling | Adds `self.weights` per-edge scaling |

---

### Version 3: Pure KAN Layer (Cell 5)

"Pure" means closer to the Kolmogorov-Arnold theorem — **no weight matrix**. The spline IS the entire edge function. There's also a dedicated bias mechanism.

```python
class PureKANLayer(nn.Module):
    def __init__(self, in_features, out_features, grid_size=10):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size

        # Same coefficient tensor as VectorizedKANLayer
        self.coeffs = nn.Parameter(torch.randn(out_features, in_features, grid_size))

        # CONCEPT: Separate bias spline coefficients per output neuron
        # WHY: In the KA theorem, there's a constant term for each outer function.
        #      Instead of a simple scalar bias, this uses a spline-based bias —
        #      though in practice it's evaluated at a fixed input (0.5), making it
        #      act like a fancy constant. The Optimized version simplifies this to a scalar.
        self.b_coeffs = nn.Parameter(torch.randn(out_features, grid_size))

        self.register_buffer('grid', torch.linspace(0, 1, grid_size))

    def forward(self, x):
        batch_size = x.size(0)
        x_norm = torch.sigmoid(x)

        # CONCEPT: unsqueeze(1) for broadcasting across out_features
        # (batch, in_features) → (batch, 1, in_features) after scaling
        scaled = x_norm.unsqueeze(1) * (self.grid_size - 1)
        idx = scaled.long().clamp(0, self.grid_size - 2)
        frac = scaled - idx.float()

        idx_expanded = idx.expand(-1, self.out_features, -1)

        # CONCEPT: torch.gather — the batched version of "look up these indices"
        # WHY: We have coeffs of shape (out, in, grid_size) and need to pick specific
        #      grid points for each (batch, out, in) combination.
        # gather(input, dim=3, index) → picks along the grid_size dimension (dim 3)
        #
        # Step by step:
        # 1. coeffs.unsqueeze(0) → (1, out, in, grid_size)
        # 2. .expand(batch_size, ...) → (batch, out, in, grid_size) — virtual copies
        # 3. idx_expanded.unsqueeze(3) → (batch, out, in, 1) — the index into grid_size dim
        # 4. gather picks one value per (batch, out, in) → (batch, out, in, 1)
        # 5. .squeeze(3) → (batch, out, in)
        coeffs_left = torch.gather(
            self.coeffs.unsqueeze(0).expand(batch_size, -1, -1, -1),
            3,
            idx_expanded.unsqueeze(3)
        ).squeeze(3)

        coeffs_right = torch.gather(
            self.coeffs.unsqueeze(0).expand(batch_size, -1, -1, -1),
            3,
            (idx_expanded + 1).unsqueeze(3)
        ).squeeze(3)

        # Linear interpolation — same as before
        frac_expanded = frac.expand(-1, self.out_features, -1)
        activated_inputs = (1 - frac_expanded) * coeffs_left + frac_expanded * coeffs_right

        # CONCEPT: Sum over input features → Σ_i φ_ji(x_i)
        # This is the pure KA theorem sum — no weight matrix multiplied in
        summed = torch.sum(activated_inputs, dim=2)   # (batch, out_features)

        # Bias computation (evaluated at fixed point 0.5)
        b_values = torch.zeros(batch_size, self.out_features, device=x.device)
        for o in range(self.out_features):
            b_idx = torch.zeros(batch_size, dtype=torch.long, device=x.device)
            b_frac = torch.ones(batch_size, device=x.device) * 0.5
            b_left = self.b_coeffs[o, b_idx]
            b_right = self.b_coeffs[o, b_idx + 1]
            b_values[:, o] = (1 - b_frac) * b_left + b_frac * b_right

        return summed + b_values
```

#### `OptimizedPureKANLayer` — Simplified Bias

The `PureKANLayer`'s spline-based bias is overcomplicated (it's always evaluated at 0.5, so it's effectively a constant). The optimized version just uses a scalar bias:

```python
class OptimizedPureKANLayer(nn.Module):
    def __init__(self, in_features, out_features, grid_size=10):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size

        self.coeffs = nn.Parameter(torch.randn(out_features, in_features, grid_size))

        # CONCEPT: Simple scalar bias instead of spline-based bias
        # WHY: The PureKANLayer's b_coeffs evaluated at a fixed point is just a constant.
        #      A scalar nn.Parameter does the same job with less code and no loop.
        self.bias = nn.Parameter(torch.randn(out_features))

        self.register_buffer('grid', torch.linspace(0, 1, grid_size))

    def forward(self, x):
        batch_size = x.size(0)
        x_norm = torch.sigmoid(x)
        scaled = x_norm * (self.grid_size - 1)
        idx = scaled.long().clamp(0, self.grid_size - 2)
        frac = scaled - idx.float()

        # CONCEPT: Back to simple Python loops (trading vectorization for clarity)
        # NOTE: This is actually SLOWER than the gather-based PureKANLayer.
        #       It's called "Optimized" because it simplified the bias, not the loops.
        output = torch.zeros(batch_size, self.out_features, device=x.device)
        for o in range(self.out_features):
            for i in range(self.in_features):
                left_coeff = self.coeffs[o, i, idx[:, i]]
                right_coeff = self.coeffs[o, i, idx[:, i] + 1]
                output[:, o] += (1 - frac[:, i]) * left_coeff + frac[:, i] * right_coeff

        # CONCEPT: unsqueeze(0) broadcasts scalar bias across batch dimension
        # bias shape: (out_features,) → (1, out_features) → broadcasts to (batch, out_features)
        output += self.bias.unsqueeze(0)
        return output
```

---

### Version 4: Enhanced Pure KAN Layer (Cell 7)

Three improvements over Version 3: better normalization, smarter initialization, and LayerNorm between layers.

```python
class EnhancedPureKANLayer(nn.Module):
    def __init__(self, in_features, out_features, grid_size=50):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size

        # CONCEPT: Smaller initialization variance (* 0.1) for training stability
        # WHY: With large grid_size (50), random N(0,1) coefficients create wild splines
        #      that produce huge activations. Scaling down keeps initial outputs reasonable,
        #      similar to Xavier/He init philosophy for weight matrices.
        self.coeffs = nn.Parameter(torch.randn(out_features, in_features, grid_size) * 0.1)

        # CONCEPT: Zero-initialized bias (not random)
        # WHY: Starting with zero bias means the initial output is purely from the splines.
        #      This is standard practice — let the learned part (coeffs) determine the
        #      initial behavior rather than adding random offsets.
        self.bias = nn.Parameter(torch.zeros(out_features))

        # CONCEPT: Nonlinear grid spacing via tanh
        # WHY: linspace(-1, 1, 50).tanh() bunches grid points TOWARD ZERO.
        #      tanh is nearly linear near zero and saturates at ±1.
        #      More grid points near zero = finer resolution where inputs are most likely
        #      (since tanh-normalized inputs cluster near zero for moderate values).
        # NOTE: This grid is used conceptually (for understanding) but the actual
        #       interpolation still uses the index-based approach, not grid positions.
        self.register_buffer('grid', torch.linspace(-1, 1, grid_size).tanh())

    def forward(self, x):
        batch_size = x.size(0)

        # CONCEPT: tanh normalization instead of sigmoid
        # WHY: tanh maps inputs to (-1, 1) instead of (0, 1).
        #      This preserves the sign of inputs — negative inputs stay negative.
        #      sigmoid always outputs positive values, losing sign information.
        x_norm = torch.tanh(x)

        # CONCEPT: Map from (-1, 1) to (0, grid_size-1) for index lookup
        # (x_norm + 1) shifts from (-1,1) to (0,2), then * (grid_size-1)/2 scales
        scaled = (x_norm + 1) * (self.grid_size - 1) / 2
        idx = scaled.long().clamp(0, self.grid_size - 2)
        frac = scaled - idx.float()

        # Same loop-based computation as OptimizedPureKANLayer
        output = torch.zeros(batch_size, self.out_features, device=x.device)
        for o in range(self.out_features):
            for i in range(self.in_features):
                left_coeff = self.coeffs[o, i, idx[:, i]]
                right_coeff = self.coeffs[o, i, idx[:, i] + 1]
                output[:, o] += (1 - frac[:, i]) * left_coeff + frac[:, i] * right_coeff

        output += self.bias.unsqueeze(0)
        return output
```

#### `PureKANClassifier` with LayerNorm

```python
class PureKANClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64], output_dim=3, grid_size=50):
        super().__init__()
        dims = [input_dim] + hidden_dims + [output_dim]
        self.layers = nn.ModuleList([
            EnhancedPureKANLayer(dims[i], dims[i+1], grid_size)
            for i in range(len(dims)-1)
        ])

        # CONCEPT: nn.LayerNorm normalizes each sample to mean=0, std=1
        # WHY: KAN layers can produce outputs with wildly different scales
        #      across neurons. LayerNorm stabilizes the inputs to the next layer,
        #      preventing exploding/vanishing activations deeper in the network.
        # Math: for each sample x, compute (x - mean(x)) / std(x), then scale+shift
        # NOTE: Only applied to hidden layers, NOT the output layer
        self.norms = nn.ModuleList([
            nn.LayerNorm(dims[i+1]) for i in range(len(dims)-2)
        ])

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.norms[i](x)     # Normalize between hidden layers
        x = self.layers[-1](x)       # No norm on output — raw logits
        return x
```

**Key insight:** LayerNorm between KAN layers is important because spline outputs don't have a naturally bounded range (unlike ReLU which is ≥0, or tanh which is in [-1,1]). The spline coefficients are random — they can produce any scale. LayerNorm is the safety net that keeps values reasonable between layers.

---

## Full Network Assembly

All versions follow the same stacking pattern. Here's the `PureKAN` as an example:

```python
class PureKAN(nn.Module):
    def __init__(self, input_dim, hidden_dims=[32, 16], grid_size=10, optimized=True):
        super().__init__()
        # CONCEPT: dims includes output dimension (unlike KANNetwork which uses a separate output_layer)
        # WHY: In PureKAN, the last KAN layer IS the output layer.
        #      In KANNetwork, the last layer is nn.Linear — a design choice, not a requirement.
        self.dims = [input_dim] + hidden_dims + [1]

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            if optimized:
                self.layers.append(OptimizedPureKANLayer(self.dims[i], self.dims[i+1], grid_size))
            else:
                self.layers.append(PureKANLayer(self.dims[i], self.dims[i+1], grid_size))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
```

### The `nn.Linear` Output Layer Question

Notice that `KANNetwork` uses `nn.Linear` for the output, while `PureKAN` uses a KAN layer all the way through. Both approaches work:

| Approach | Code | Tradeoff |
|---|---|---|
| KAN + Linear output | `KANNetwork` | Fewer parameters in the output layer. The Linear layer is just a weighted sum — fast. |
| KAN all the way | `PureKAN` | Consistent architecture. Even the output transform is a learned spline. More parameters. |

For our CPPN project, the output is `(h, s, v)` — three simple channels. A linear projection from the last hidden layer is probably sufficient. But if we wanted the output to have nonlinear characteristics (e.g., the hue wrapping behavior), a KAN output layer could learn that mapping.

---

## Connecting KAN to CPPN

Here's the side-by-side comparison of what happens in one layer:

```
CPPN layer (Tutorial 02):
    x → nn.Linear(W @ x) → split [15,4,2,1] → [cache, gaussian, identity, sin] → cat
    Learnable: weight matrix W
    Fixed: which activation goes where (architect decides)

KAN layer (this tutorial):
    x → for each (out, in) pair: spline_ji(x_i) → sum across inputs
    Learnable: spline coefficients for every edge
    Fixed: nothing — every edge learns its own function
```

**The key architectural difference:**

| | CPPN (MLP) | KAN |
|---|---|---|
| Where's the nonlinearity? | After the linear transform | Inside each edge connection |
| What's learnable? | Weight matrix W | Spline coefficients per edge |
| Activation functions | Fixed by arch string | Learned per edge |
| Parameters per layer | `in × out` weights | `in × out × grid_size` coefficients |

**Parameter count comparison:**
- CPPN layer (22→22): 22 × 22 = **484** parameters
- KAN layer (22→22, grid_size=10): 22 × 22 × 10 = **4,840** parameters

KAN layers use ~10x more parameters (scales with grid_size). This is why the notebook uses smaller hidden dimensions for KAN (hidden_dims=[32,16]) compared to what you'd use for an MLP.

---

## Swarm-KAN: Evolutionary Coefficient Optimization

The final cell in the notebook introduces `SwarmKAN` — a hybrid that combines gradient-based training with **swarm intelligence** for the spline coefficients. This is directly relevant to our project because CPPNs were originally evolved (Picbreeder/NEAT), and we want to explore evolutionary approaches for KAN layers too.

### The Core Idea

Standard training: gradient descent updates ALL parameters. Swarm-KAN adds a second update mechanism — periodically, each spline activation "looks at" a few randomly chosen neighbor activations and **blends toward their average**. This is inspired by the **cohesion** behavior in swarm algorithms (like Boids or PSO).

```
Every N epochs:
    For each spline activation φ in the network:
        Pick 3 random other activations φ₁, φ₂, φ₃
        Blend: coeffs ← (1 - rate) * coeffs + rate * mean(φ₁, φ₂, φ₃)
```

Why would this help? Two reasons:
1. **Regularization** — blending toward neighbors prevents any one spline from becoming too extreme (wild oscillations)
2. **Exploration** — it injects variation from other parts of the network, potentially escaping local minima

### `SwarmKANActivation` — A Spline That Talks to Its Neighbors

```python
class SwarmKANActivation(nn.Module):
    def __init__(self, grid_size=20, influence_rate=0.05):
        super().__init__()
        self.grid_size = grid_size

        # CONCEPT: influence_rate controls how much each swarm step changes the coefficients
        # WHY: Too high (e.g., 0.5) = coefficients converge to the mean, losing diversity
        #      Too low (e.g., 0.001) = swarm has no effect
        #      0.05 means "nudge 5% toward your neighbors each time"
        self.influence_rate = influence_rate

        self.register_buffer('grid', torch.linspace(0, 1, grid_size))
        self.coeffs = nn.Parameter(torch.randn(grid_size))

        # CONCEPT: register_buffer for memory — stores a snapshot of past coefficients
        # WHY: This could be used for momentum-style updates or reverting to previous states.
        #      In this implementation it's initialized but not actively used during swarm_update.
        #      Think of it as a "personal best" from PSO — a hook for future extensions.
        self.register_buffer('memory', self.coeffs.detach().clone())

    def forward(self, x):
        # Same spline interpolation as Version 1
        x_norm = torch.sigmoid(x)
        scaled = x_norm * (self.grid_size - 1)
        idx = scaled.long().clamp(0, self.grid_size - 2)
        frac = scaled - idx.float()
        left = self.coeffs[idx]
        right = self.coeffs[idx + 1]
        return (1 - frac) * left + frac * right

    def swarm_update(self, neighbor_coeffs):
        # CONCEPT: Cohesion-style blending — move toward the average of your neighbors
        # WHY: In swarm intelligence, cohesion = "steer toward average position of neighbors"
        #      Here, "position" = the spline coefficient vector
        #
        # neighbor_coeffs: shape (num_neighbors, grid_size)
        # mean_neighbor: shape (grid_size,)
        mean_neighbor = neighbor_coeffs.mean(dim=0)

        # CONCEPT: .data modifies the tensor WITHOUT recording it in the autograd graph
        # WHY: The swarm update is NOT a gradient-based operation. It's a direct
        #      manipulation of the parameter values between training steps.
        #      If we didn't use .data, PyTorch would try to backpropagate through this update,
        #      which makes no sense (there's no loss function driving it).
        self.coeffs.data = (1 - self.influence_rate) * self.coeffs.data + self.influence_rate * mean_neighbor
```

**Key insight about `.data`:** This is how you modify parameters outside of gradient descent. `self.coeffs.data = ...` directly writes new values into the parameter tensor. The optimizer doesn't know this happened — it will continue computing gradients against whatever values are there. This is the "evolutionary" part of the hybrid: direct mutation between gradient steps.

### `SwarmKANLayer` — Orchestrating the Swarm

```python
class SwarmKANLayer(nn.Module):
    def __init__(self, in_features, out_features, grid_size=20, influence_rate=0.05):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # CONCEPT: This version re-adds the weight matrix (like VectorizedKANLayer)
        # WHY: Having both learnable splines AND weights gives the most flexibility.
        #      Gradient descent optimizes the weights, while swarm optimizes the splines.
        #      This separates "what shape is the function" (swarm) from
        #      "how much does each connection matter" (gradient descent).
        self.weights = nn.Parameter(torch.randn(out_features, in_features))

        self.activations = nn.ModuleList([
            SwarmKANActivation(grid_size, influence_rate)
            for _ in range(in_features * out_features)
        ])

    def forward(self, x):
        batch_size = x.size(0)
        out = torch.zeros(batch_size, self.out_features, device=x.device)
        idx = 0
        for o in range(self.out_features):
            for i in range(self.in_features):
                # weight * spline(input) — gradient descent tunes weight, swarm tunes spline
                out[:, o] += self.weights[o, i] * self.activations[idx](x[:, i])
                idx += 1
        return out

    def swarm_step(self):
        # CONCEPT: Collect all spline coefficients, then update each one
        # Step 1: Stack all coefficients into one tensor for easy neighbor sampling
        all_coeffs = torch.stack([a.coeffs.detach() for a in self.activations])
        # all_coeffs shape: (num_activations, grid_size)

        # Step 2: For each activation, randomly pick 3 neighbors and blend toward them
        for i, a in enumerate(self.activations):
            # CONCEPT: torch.randint randomly selects neighbor indices
            # WHY: Random neighbor selection (vs nearest neighbors) is simpler and
            #      provides more diverse exploration. In PSO terms, these are "informants."
            neighbor_ids = torch.randint(0, len(self.activations), (3,))
            neighbor_coeffs = all_coeffs[neighbor_ids]   # shape: (3, grid_size)
            a.swarm_update(neighbor_coeffs)
```

### `SwarmKAN` — The Full Network

```python
class SwarmKAN(nn.Module):
    def __init__(self, input_dim, hidden_dims=[32, 16], output_dim=1, grid_size=20, influence_rate=0.05):
        super().__init__()
        dims = [input_dim] + hidden_dims + [output_dim]
        self.layers = nn.ModuleList([
            SwarmKANLayer(dims[i], dims[i+1], grid_size, influence_rate)
            for i in range(len(dims)-1)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def swarm_step_all(self):
        # CONCEPT: Trigger swarm update on ALL layers
        # Called between training steps, not during forward pass
        for layer in self.layers:
            layer.swarm_step()
```

### The Training Loop: Gradient + Swarm Hybrid

```python
def train_swarm_kan(model, X_train, y_train, epochs=50, swarm_interval=5):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        # Standard gradient descent step
        model.train()
        optimizer.zero_grad()
        preds = model(X_train)
        loss = criterion(preds, y_train)
        loss.backward()
        optimizer.step()

        # CONCEPT: Swarm step every N epochs (not every epoch)
        # WHY: The swarm update is a perturbation — doing it too frequently
        #      destabilizes gradient descent. Every 5 epochs is a balance:
        #      gradients settle the splines, then swarm nudges them, then
        #      gradients adapt to the new positions.
        if (epoch + 1) % swarm_interval == 0:
            model.swarm_step_all()
```

**Key insight:** The training loop interleaves two completely different optimization strategies. Gradient descent (Adam) handles the smooth, local optimization of all parameters. The swarm step introduces non-gradient perturbations to the spline coefficients, acting like a mild mutation operator in evolutionary computation. This is the kind of hybrid neuroevolution approach that connects directly to how Picbreeder's NEAT evolved CPPN weights — but now applied to learnable activation functions instead.

### How Swarm-KAN Connects to Our Project

| Picbreeder (original CPPNs) | Swarm-KAN | Our future direction |
|---|---|---|
| NEAT evolves topology + weights | Swarm perturbs spline coefficients | Evolve KAN spline shapes in a CPPN |
| No gradient descent at all | Gradient + swarm hybrid | Could do gradient-free evolution too |
| Human selection as fitness | MSE loss function | Could use aesthetic/diversity fitness |
| Fixed activation set (sin, cos, gaussian) | Learned activations (splines) | Best of both: evolved structure with learned edges |

The Swarm-KAN is a prototype, but it points toward a key insight for our project: **we don't have to choose between evolution and gradient descent**. We can use gradient descent to train the spline coefficients (fine-tuning the shape of each activation function) while using evolutionary methods to search over architectures, initial conditions, or higher-level hyperparameters.

---

## Gotchas & Common Mistakes

### 1. Sigmoid vs tanh normalization misaligns with the grid

If your grid covers `[0, 1]` but you normalize with `tanh` (output range `[-1, 1]`), half your inputs map to negative values that the grid can't represent. The `EnhancedPureKANLayer` handles this correctly by using `(x_norm + 1) * (grid_size-1) / 2` to shift tanh's `[-1,1]` to the `[0, grid_size-1]` index range. If you mix sigmoid normalization with a `[-1,1]` grid, or tanh with a `[0,1]` grid, you'll get indices that cluster at one end and waste half the grid resolution.

### 2. `.long()` truncates, it doesn't round

```python
torch.tensor(2.99).long()   # → 2, NOT 3
torch.tensor(-0.1).long()   # → 0 (truncates toward zero)
```

This is correct for spline indexing (we want the LEFT grid point), but surprising if you expect rounding. Combined with `.clamp(0, grid_size-2)`, negative or out-of-range values all get pushed to the boundary grid cells, effectively "clamping" the spline's output at the edges.

### 3. Nested `nn.ModuleList` works but watch parameter discovery

```python
# This works — PyTorch finds all parameters
self.acts = nn.ModuleList([nn.ModuleList([...]) for _ in range(n)])

# This DOES NOT work — inner lists are invisible
self.acts = nn.ModuleList([[SplineAct() for _ in range(m)] for _ in range(n)])
# The inner list is a plain Python list, not nn.ModuleList!
```

The outer `nn.ModuleList` only registers its direct children. If the inner containers are plain Python lists, their parameters are invisible to `model.parameters()`. This means the optimizer won't train them and `model.to('cuda')` won't move them.

### 4. Python loops over `in_features * out_features` kill GPU performance

The naive and "optimized pure" versions use nested Python `for` loops. On CPU with small layers (8→16), this is fine. On GPU with larger layers (64→32), the Python loop overhead dominates — you're launching thousands of tiny GPU kernels instead of one big one. The `VectorizedKANLayer` avoids this entirely, which is why it's 10-100x faster for larger layers.

### 5. `register_buffer` vs `nn.Parameter` for the grid — easy to mix up

```python
# CORRECT: grid is fixed, not learned
self.register_buffer('grid', torch.linspace(0, 1, grid_size))

# WRONG: grid becomes a learnable parameter — gradient descent will move the knot points!
self.grid = nn.Parameter(torch.linspace(0, 1, grid_size))

# ALSO WRONG: grid won't move to GPU with model.to('cuda')
self.grid = torch.linspace(0, 1, grid_size)
```

The first version of the notebook (cell 2) actually makes this mistake — it uses `nn.Parameter(... requires_grad=False)` instead of `register_buffer`. This works but is semantically wrong — it stores a non-learnable constant as a "parameter," which is confusing and causes it to show up in `model.parameters()` (though it won't be trained because `requires_grad=False`).

### 6. The swarm `.data` bypass — when to use it and when not to

```python
# For swarm updates (non-gradient modifications): USE .data
self.coeffs.data = new_values

# For normal forward pass computation: NEVER use .data
result = self.coeffs[idx]  # ← correct, gradient flows through this
result = self.coeffs.data[idx]  # ← WRONG, breaks gradient computation
```

`.data` bypasses autograd. This is correct for the swarm step (which is not a differentiable operation), but if you accidentally use `.data` during the forward pass, gradients won't flow through that operation and the optimizer can't learn those parameters.

---

## What's Next

- **Tutorial 04**: Particle swarm optimization for KAN-CPPNs — attaching PSO to spline coefficients for hybrid exploration alongside SGD
- **Tutorial 05**: Memetic KAN with Natural Evolution Strategy + SGD — why crossover fails for KANs, antithetic gradient estimation, and selective parameter perturbation

---

## Quick Reference: MLP vs KAN

| "I want to..." | MLP (nn.Linear) | KAN |
|---|---|---|
| Define one layer | `nn.Linear(in, out)` | `KANLayer(in, out, grid_size)` |
| Forward pass | `y = W @ x + b` | `y_j = Σ_i spline_ji(x_i)` |
| Learnable parts | Weight matrix W, bias b | Spline coefficients per edge |
| Parameters per layer | `in × out` | `in × out × grid_size` |
| Activation function | Fixed (ReLU, sin, etc.) | Learned (spline per edge) |
| Input normalization | Not needed | `sigmoid(x)` or `tanh(x)` to map to grid |
| Speed (Python loops) | Fast (single matmul) | Slow (Naive), Fast (Vectorized) |
| Interpretability | Opaque weight matrix | Can visualize each learned spline |
