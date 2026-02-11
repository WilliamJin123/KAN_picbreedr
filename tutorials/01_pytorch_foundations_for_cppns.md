# Tutorial 01: PyTorch Foundations for CPPNs

## What We're Building (Big Picture)

We have a CPPN (Compositional Pattern-Producing Network) that takes pixel coordinates `(x, y, d, bias)` and outputs a color `(h, s, v)`. It's just a neural network where:
- Input = position on a canvas
- Output = what color to paint there
- The internal layers learn spatial patterns (circles, stripes, gradients)

The fer repo does this in JAX. We're rebuilding it in PyTorch so we can swap in KAN layers later. **Nothing about the math changes - only the syntax.**

---

## Concept 1: Tensors (PyTorch's Arrays)

**What you already know:** NumPy arrays / JAX arrays / matrices of numbers.

**PyTorch equivalent:** `torch.Tensor`. It's the same thing, but with two superpowers:
1. It can live on a GPU
2. It can track gradients (for backprop)

### Math vs PyTorch

| Math | NumPy | PyTorch |
|------|-------|---------|
| vector `v = [1, 2, 3]` | `np.array([1, 2, 3])` | `torch.tensor([1, 2, 3])` |
| matrix multiply `A @ x` | `np.dot(A, x)` or `A @ x` | `A @ x` (same!) |
| element-wise `sin(x)` | `np.sin(x)` | `torch.sin(x)` |
| shape of tensor | `x.shape` | `x.shape` (same!) |
| zeros | `np.zeros((3, 4))` | `torch.zeros(3, 4)` |
| linspace | `np.linspace(-1, 1, 256)` | `torch.linspace(-1, 1, 256)` |

### Minimal Example

```python
import torch

# Create a 256x256 grid of x,y coordinates (exactly like the CPPN input)
x = torch.linspace(-1, 1, 256)          # 256 values from -1 to 1
y = torch.linspace(-1, 1, 256)
grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')  # two 256x256 grids
d = torch.sqrt(grid_x**2 + grid_y**2)   # distance from center
b = torch.ones_like(grid_x)             # bias = 1 everywhere

# Stack into input tensor: shape (256, 256, 4)
inputs = torch.stack([grid_x, grid_y, d, b], dim=-1)
print(inputs.shape)  # torch.Size([256, 256, 4])
```

**Key insight:** Tensors work almost identically to NumPy. The main difference shows up when we need gradients.

---

## Concept 2: nn.Module (The Building Block of Everything)

**What you already know:** A neural network layer is a function with learnable parameters (weights).

**PyTorch's approach:** Every "thing with parameters" inherits from `nn.Module`. A layer is a Module. A whole network is a Module (containing other Modules). It's turtles all the way down.

### Why does PyTorch do it this way?

Because `nn.Module` automatically:
- Keeps track of ALL learnable parameters (recursively through sub-modules)
- Handles `.to(device)` to move everything to GPU at once
- Handles `model.train()` / `model.eval()` mode switching
- Handles saving/loading (`torch.save` / `torch.load`)

### The Pattern (Every Module Looks Like This)

```python
import torch.nn as nn

class MyLayer(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()                    # REQUIRED: initializes Module internals
        # Define structure here (what parameters exist)
        self.weights = nn.Parameter(torch.randn(out_size, in_size))  # learnable!

    def forward(self, x):
        # Define computation here (what happens when data flows through)
        return x @ self.weights.T
```

Two methods. That's it:
- `__init__`: declare what parameters/sub-layers exist
- `forward`: define the math

### nn.Parameter vs regular Tensor

```python
self.weights = nn.Parameter(torch.randn(3, 4))  # PyTorch WILL update this during training
self.grid = torch.linspace(0, 1, 10)             # PyTorch will NOT update this (just a constant)
```

**Key insight:** `nn.Parameter` is just a tensor with a flag that says "optimize me." That's literally the only difference. When you call `model.parameters()`, it collects all `nn.Parameter`s.

### register_buffer: Constants That Travel With the Model

```python
self.register_buffer('grid', torch.linspace(0, 1, 10))
```

This is for constants that:
- Should NOT be trained (not a Parameter)
- But SHOULD move to GPU when you call `model.to('cuda')`
- And SHOULD be saved when you call `torch.save(model)`

The spline grid in KAN layers is a perfect example - it's a fixed set of knot points, not learned.

---

## Concept 3: nn.Linear (The Dense Layer)

**What you already know:** A dense/fully-connected layer computes `y = Wx + b`.

**PyTorch:** `nn.Linear(in_features, out_features, bias=True)`

```python
layer = nn.Linear(4, 22, bias=False)  # 4 inputs, 22 outputs, no bias
# Internally this creates: self.weight = nn.Parameter(torch.randn(22, 4))

x = torch.randn(1000, 4)   # 1000 samples, 4 features each
y = layer(x)                # shape: (1000, 22)
# This computes: y = x @ weight.T   (matrix multiply)
```

**For our CPPN:** The fer code does `nn.Dense(sum(d_hidden), use_bias=False)(x)` in Flax. The PyTorch equivalent is `nn.Linear(in_features, sum(d_hidden), bias=False)`.

### Flax vs PyTorch Side-by-Side

| Flax (fer repo) | PyTorch (what we'll write) |
|---|---|
| `x = nn.Dense(22, use_bias=False)(x)` | `x = self.linear(x)` where `self.linear = nn.Linear(in, 22, bias=False)` |
| Layer created inline in `forward` | Layer created in `__init__`, used in `forward` |
| Params passed explicitly | Params stored inside the Module |

**Key insight:** In Flax, you pass params around as a separate dict. In PyTorch, params live INSIDE the model object. This is the biggest conceptual difference between the two frameworks.

---

## Concept 4: How Data Flows Through (forward and __call__)

You never call `model.forward(x)` directly. You call `model(x)`, which internally calls `forward` but also handles hooks, gradient tracking, etc.

```python
class MyCPPN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(4, 22, bias=False)
        self.layer2 = nn.Linear(22, 22, bias=False)
        self.output = nn.Linear(22, 3, bias=False)

    def forward(self, x):
        x = self.layer1(x)       # (batch, 4) -> (batch, 22)
        x = torch.sin(x)         # apply activation
        x = self.layer2(x)       # (batch, 22) -> (batch, 22)
        x = torch.sin(x)         # apply activation
        x = self.output(x)       # (batch, 22) -> (batch, 3)
        return x                  # (batch, 3) = (h, s, v)

model = MyCPPN()
pixel_coords = torch.randn(65536, 4)  # 256*256 pixels, 4 inputs each
colors = model(pixel_coords)           # (65536, 3) = h,s,v per pixel
```

---

## Concept 5: Reshaping (view, reshape, and why CPPNs need it)

The CPPN takes a 2D grid of pixels but processes them as a flat batch. We need to reshape back and forth.

```python
# Start with 256x256 grid, 4 features per pixel
inputs = torch.randn(256, 256, 4)

# Flatten to (65536, 4) so the network sees a batch of 65536 "samples"
flat = inputs.view(-1, 4)          # -1 means "figure out this dimension"
# or equivalently:
flat = inputs.reshape(-1, 4)       # same thing (reshape is slightly more general)

# Run through network
output = model(flat)               # (65536, 3)

# Reshape back to image: (256, 256, 3)
image = output.view(256, 256, 3)
```

**Key insight:** `view(-1, 4)` is the PyTorch equivalent of `jax.vmap(jax.vmap(...))` from the fer code. Instead of "apply function to every pixel" (vmap style), PyTorch says "flatten all pixels into a batch, run them all at once, reshape back."

---

## Concept 6: No Grad Context (for inference/visualization)

When generating images or doing weight sweeps, we don't need gradients. Turning them off saves memory and speeds things up.

```python
with torch.no_grad():
    image = model(pixel_coords)    # no gradient tracking, faster
```

This is analogous to the `@jax.jit` decorator in the fer code - it's about performance, not changing the math.

---

## Concept 7: How This All Maps to the CPPN Pipeline

Here's the fer pipeline and what each part becomes in PyTorch:

```
FER PIPELINE (JAX/Flax)              OUR PIPELINE (PyTorch)
========================              ======================
CPPN(nn.Module) class          -->   CPPN(nn.Module) class
  arch string parsing          -->   same (just string parsing)
  nn.Dense layers              -->   nn.Linear layers
  activation_fn_map split      -->   same dict + torch.split
  features.append(x)           -->   same (list of tensors)

FlattenCPPNParameters          -->   torch.nn.utils.parameters_to_vector
  evosax.ParameterReshaper     -->   (built-in PyTorch utility)

generate_image()               -->   generate_image()
  jnp.linspace + meshgrid     -->   torch.linspace + meshgrid
  jax.vmap(jax.vmap(...))     -->   input.view(-1, 4) + model(flat) + .view(H, W, 3)
  hsv2rgb conversion          -->   same math, torch functions

sweep_weight()                 -->   sweep_weight()
  params flat vector           -->   parameters_to_vector / vector_to_parameters
  jnp.linspace for sweep      -->   torch.linspace for sweep
  jax.vmap over sweep values  -->   simple for-loop or torch.stack
```

---

## What's Next

The next tutorial will cover:
- **Tutorial 02:** Building the CPPN in PyTorch (translating `cppn.py` line by line)
- **Tutorial 03:** KAN layers - how spline-based edges replace Linear+activation
- **Tutorial 04:** Swapping MLP layers for KAN layers in the CPPN
- **Tutorial 05:** Weight sweeps and feature map visualization in PyTorch
- **Tutorial 06:** Swarm-based optimization for KAN spline coefficients

---

## Quick Reference Card

| "I want to..." | PyTorch code |
|---|---|
| Make a learnable weight matrix | `self.W = nn.Parameter(torch.randn(m, n))` |
| Make a fixed constant | `self.register_buffer('c', torch.tensor(...))` |
| Make a dense layer | `self.layer = nn.Linear(in, out, bias=False)` |
| Apply sin activation | `x = torch.sin(x)` |
| Flatten a grid to a batch | `x = x.view(-1, features)` |
| Reshape batch back to image | `img = x.view(H, W, channels)` |
| Turn off gradients | `with torch.no_grad():` |
| Get all parameters as 1D vector | `torch.nn.utils.parameters_to_vector(model.parameters())` |
| Count parameters | `sum(p.numel() for p in model.parameters())` |
