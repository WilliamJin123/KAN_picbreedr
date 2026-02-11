# Tutorial 02: What is a CPPN & Building One in PyTorch

## Part A: What is a CPPN?

### The One-Sentence Version

A CPPN is a neural network that takes *where* a pixel is and outputs *what color* it should be.

### The Longer Version

CPPN stands for **Compositional Pattern-Producing Network**. Let's break that name down:

- **Pattern-Producing**: It generates visual patterns (images)
- **Compositional**: It does this by composing simple mathematical functions (sin, cos, gaussian, etc.)
- **Network**: It's a neural network — layers of weighted connections

Here's the mental model:

```
For every pixel at position (x, y) on a canvas:
    feed (x, y) into the network
    get back a color (hue, saturation, brightness)
    paint that pixel
```

That's it. A CPPN is just a function `f(x, y) → color`. The network's weights determine WHAT pattern appears. Change the weights, change the image.

### Why Not Just Use a Regular Neural Network?

You are! A CPPN IS a regular neural network. The difference is purely conceptual — it's about what we chose for the inputs and outputs:

| | Regular image classifier | CPPN |
|---|---|---|
| **Input** | An entire image (pixels) | A single coordinate (x, y) |
| **Output** | A label ("cat") | A color (h, s, v) |
| **What it learns** | "What is this?" | "What color should this location be?" |
| **# of forward passes per image** | 1 | 1 per pixel (or batched) |

### Why Do CPPNs Make Interesting Patterns?

The secret is in the **activation functions**. In a standard MLP you'd use ReLU everywhere. CPPNs use a *mix* of activations — and each one creates a different kind of spatial pattern:

| Activation | Math | What it looks like spatially |
|---|---|---|
| `sin(x)` | Periodic wave | **Stripes, ripples** |
| `cos(x)` | Periodic wave (phase-shifted) | **Stripes, ripples** (offset) |
| `gaussian(x)` = e^(-x²) | Bell curve | **Blobs, circles, soft spots** |
| `sigmoid(x)` | S-curve | **Smooth gradients, soft edges** |
| `tanh(x)` | S-curve (centered at 0) | **Smooth gradients** |
| `identity(x)` = x | Pass-through | **Linear gradients** |

When you *compose* these (stack layers), you get compositions of compositions. `sin(gaussian(x,y))` creates rippling blobs. `gaussian(sin(x) + cos(y))` creates spotted grids. The network learns which combinations produce the desired image.

### The Four Inputs

The fer CPPN doesn't just take `(x, y)`. It takes 4 values:

```
(y, x, d, b)
```

- **y**: vertical position, from -1 (top) to +1 (bottom)
- **x**: horizontal position, from -1 (left) to +1 (right)
- **d**: distance from center = sqrt(x² + y²) * 1.4
- **b**: bias = 1.0 everywhere (a constant the network can scale)

Why `d`? Because it gives the network "free" radial symmetry information. Without it, the network would have to learn `sqrt(x² + y²)` from scratch. With it, making a circle is trivial — just apply gaussian to `d`.

Why `b`? Same as a bias neuron in any neural network — it lets the network shift outputs without depending on position.

### The Three Outputs

The CPPN outputs 3 values per pixel:

```
(h, s, v)  →  (hue, saturation, value/brightness)
```

These get converted to RGB for display. HSV is more natural for generation because:
- **Hue**: what color (red, green, blue, yellow...) — a circular value 0→1→wraps
- **Saturation**: how vivid (0 = gray, 1 = full color)
- **Value**: how bright (0 = black, 1 = full brightness)

### Where Do CPPNs Come From? (NEAT and Picbreeder)

CPPNs were invented by Kenneth Stanley. The original CPPNs were **evolved, not trained with gradients**. The tool was called **Picbreeder** — a website where humans selected images they liked, and a genetic algorithm (NEAT) evolved the CPPN weights and topology.

NEAT produces messy, arbitrary graphs — neurons connected in any pattern, not clean layers. The fer paper's key contribution is **layerizing** these messy NEAT graphs into dense MLPs (regular layered networks) so you can analyze their internal representations.

That's what `process_pb.py` does: takes a NEAT genome zip file → topological sort → inserts "cache" neurons to handle skip connections → outputs a dense MLP with an arch string.

Look at the images in `references/fer/assets/pngs/`:
- **teaser.png**: Left side shows Picbreeder's "unified factored representation" (clean internal features), right side shows SGD's "fractured entangled representation" (messy internal features). Same output image, completely different internal structure.
- **fmaps_576_pb.png**: Feature maps of the skull CPPN — each small square is one neuron's activation over the entire image. Notice how clean they are: circles, stripes, skull-shaped masks. Each neuron "does one thing."

---

## Part B: The fer CPPN Architecture (What We're Translating)

### The Arch String

The architecture is encoded as a string like:

```
"12;cache:15,gaussian:4,identity:2,sin:1"
```

This means:
- **12** hidden layers
- Each layer has **22 neurons total** (15 + 4 + 2 + 1)
- Those 22 neurons are split into groups by activation:
  - 15 neurons use `cache` (identity — these are pass-through neurons for skip connections)
  - 4 neurons use `gaussian`
  - 2 neurons use `identity`
  - 1 neuron uses `sin`

### What "cache" Means

When NEAT evolves a network, neuron A in layer 2 might connect directly to neuron B in layer 8, skipping layers 3-7. In a dense MLP, you can't do that — every layer connects to the next layer only.

The solution: insert **cache neurons** in layers 3-7 that just copy neuron A's value forward. The weight matrix has a 1.0 on the diagonal for cache neurons (identity passthrough) and 0.0 everywhere else in that row.

```
Layer 2: [neuron A outputs 0.7]
Layer 3: [cache neuron copies 0.7]     ← weight = 1.0 on diagonal
Layer 4: [cache neuron copies 0.7]     ← weight = 1.0 on diagonal
...
Layer 8: [neuron B receives 0.7]       ← actual learned connection
```

This is why most neurons in the Picbreeder feature maps are solid blue (constant -1 or 0) — they're cache neurons just passing values through.

### How One Layer Works (The Core Loop)

Here's what happens in a single layer, step by step:

```
Input x:  shape (22,)  ← 22 values from previous layer

Step 1: Dense multiply
    x = W @ x            shape: (22,) → (22,)

Step 2: Split by activation groups
    x_cache    = x[0:15]      15 neurons
    x_gaussian = x[15:19]      4 neurons
    x_identity = x[19:21]      2 neurons
    x_sin      = x[21:22]      1 neuron

Step 3: Apply each activation to its group
    x_cache    = identity(x_cache)      = x_cache  (no-op)
    x_gaussian = exp(-x²)*2 - 1         scaled gaussian
    x_identity = identity(x_identity)   = x_identity
    x_sin      = sin(x_sin)

Step 4: Concatenate back together
    x = [x_cache, x_gaussian, x_identity, x_sin]    shape: (22,)
```

**Key insight:** This is NOT like a standard MLP where every neuron uses the same activation. Each group of neurons has its own activation function. This is what makes CPPNs produce diverse spatial patterns — different neurons "see" the same input but respond with different mathematical functions.

### Full Forward Pass

```
Input: (y, x, d, b)  →  shape (4,)

Layer 0 (input→hidden):   Dense(4 → 22)  →  split/activate  →  (22,)
Layer 1:                   Dense(22 → 22) →  split/activate  →  (22,)
Layer 2:                   Dense(22 → 22) →  split/activate  →  (22,)
...
Layer 11:                  Dense(22 → 22) →  split/activate  →  (22,)
Output layer:              Dense(22 → 3)   →  (h, s, v)

Total: 12 hidden layers + 1 output layer = 13 Dense layers
```

---

## Part C: New PyTorch Concepts

### Concept 1: nn.ModuleList (A List That PyTorch Can See)

**The problem:** We need 13 `nn.Linear` layers. If you put them in a regular Python list, PyTorch **doesn't know they exist** — it can't find their parameters, can't move them to GPU, can't save them.

```python
# BAD - PyTorch can't see these layers
self.layers = [nn.Linear(22, 22) for _ in range(12)]

# GOOD - PyTorch tracks everything inside a ModuleList
self.layers = nn.ModuleList([nn.Linear(22, 22) for _ in range(12)])
```

**Why:** `nn.Module` only auto-discovers sub-modules that are stored as direct attributes (`self.something = nn.Linear(...)`) or inside `nn.ModuleList` / `nn.ModuleDict`. A plain Python list is invisible.

```python
import torch.nn as nn

class Example(nn.Module):
    def __init__(self):
        super().__init__()
        self.visible = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])
        self.invisible = [nn.Linear(4, 4) for _ in range(3)]  # BAD

model = Example()
print(sum(p.numel() for p in model.parameters()))
# Only counts visible layers! invisible ones are lost.
```

**Key insight:** If it has parameters and lives in a collection, wrap it in `nn.ModuleList`. It works exactly like a regular list (indexing, iteration, len) but PyTorch can see inside it.

### Concept 2: torch.split (Dividing a Tensor Along a Dimension)

The CPPN needs to split its 22 neurons into groups of [15, 4, 2, 1]. This is `torch.split`:

```python
import torch

x = torch.randn(22)
groups = torch.split(x, [15, 4, 2, 1])   # tuple of 4 tensors

print(groups[0].shape)  # torch.Size([15])
print(groups[1].shape)  # torch.Size([4])
print(groups[2].shape)  # torch.Size([2])
print(groups[3].shape)  # torch.Size([1])
```

**Batched version** (what we'll actually use — a batch of pixels):
```python
x = torch.randn(65536, 22)   # 65536 pixels, 22 features each
groups = torch.split(x, [15, 4, 2, 1], dim=-1)  # split along last dimension

print(groups[0].shape)  # torch.Size([65536, 15])
print(groups[1].shape)  # torch.Size([65536, 4])
```

**Flax equivalent:** `jnp.split(x, cumsum_indices)` — note Flax uses cumulative split *points* (15, 19, 21), while PyTorch uses group *sizes* (15, 4, 2, 1). PyTorch's way is more readable.

### Concept 3: torch.cat (Concatenate Tensors)

After applying activations to each group, we glue them back together:

```python
groups = [torch.randn(65536, 15), torch.randn(65536, 4),
          torch.randn(65536, 2), torch.randn(65536, 1)]
x = torch.cat(groups, dim=-1)  # shape: (65536, 22)
```

**Flax equivalent:** `jnp.concatenate(x)` — identical concept.

### Concept 4: parameters_to_vector / vector_to_parameters

The fer code flattens all weights into a single 1D vector for analysis and weight sweeps. PyTorch has built-in utilities for this:

```python
from torch.nn.utils import parameters_to_vector, vector_to_parameters

model = nn.Linear(4, 22, bias=False)  # has 88 parameters (4*22)

# Flatten all parameters to one 1D vector
flat = parameters_to_vector(model.parameters())
print(flat.shape)  # torch.Size([88])

# Write a 1D vector back into the model's parameters
new_weights = torch.randn(88)
vector_to_parameters(new_weights, model.parameters())
# Now model.weight has been overwritten with values from new_weights
```

**Key insight:** `parameters_to_vector` reads from the model. `vector_to_parameters` writes into the model. This replaces fer's `evosax.ParameterReshaper` entirely.

### Concept 5: Custom Activation Functions (No Class Needed)

In PyTorch, activation functions can just be regular Python functions — they don't need to be `nn.Module`s unless they have learnable parameters:

```python
import torch

# These are all valid activation functions
cache = lambda x: x                                   # identity
sigmoid = lambda x: torch.sigmoid(x) * 2.0 - 1.0     # scaled to [-1, 1]
gaussian = lambda x: torch.exp(-x**2) * 2.0 - 1.0    # scaled to [-1, 1]

# Store in a dict for lookup
activation_fn_map = {
    'cache': cache,
    'identity': lambda x: x,
    'cos': torch.cos,
    'sin': torch.sin,
    'tanh': torch.tanh,
    'sigmoid': sigmoid,
    'gaussian': gaussian,
    'relu': torch.relu,
}
```

Notice `sigmoid` and `gaussian` are **scaled to [-1, 1]** instead of the usual [0, 1]. This is because CPPN neuron values need to swing negative — it's how you get both "positive" and "negative" spatial features (the red vs blue in the feature maps).

---

## Part D: Annotated Planned PyTorch Code

Here's the complete CPPN we'll build, annotated line by line. Read through this first — we won't write the actual file until you've understood it.

### The CPPN Model

```python
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters

# CONCEPT: Activation functions are just plain functions, no nn.Module needed
# WHY: They have no learnable parameters, so there's nothing for PyTorch to track
# NOTE: sigmoid and gaussian are scaled to [-1, 1] range, not standard [0, 1]
cache = lambda x: x
activation_fn_map = {
    'cache': lambda x: x,
    'identity': lambda x: x,
    'cos': torch.cos,
    'sin': torch.sin,
    'tanh': torch.tanh,
    'sigmoid': lambda x: torch.sigmoid(x) * 2.0 - 1.0,
    'gaussian': lambda x: torch.exp(-x**2) * 2.0 - 1.0,
    'relu': torch.relu,
}


class CPPN(nn.Module):
    """
    CPPN model. Translates directly from fer's Flax CPPN.

    arch: "12;cache:15,gaussian:4,identity:2,sin:1"
          = 12 layers, each with 15 cache + 4 gaussian + 2 identity + 1 sin neurons
    """

    # CONCEPT: __init__ parses the architecture and creates all layers upfront
    # WHY: Unlike Flax (where layers are created inline during forward), PyTorch
    #       requires all layers to exist before forward() is called. This is because
    #       PyTorch stores parameters INSIDE the model object, so it needs to know
    #       what parameters exist at construction time.
    def __init__(self, arch, init_scale='default'):
        super().__init__()

        # Parse arch string — pure string manipulation, nothing PyTorch-specific
        n_layers_str, activation_neurons_str = arch.split(';')
        self.n_layers = int(n_layers_str)

        # Parse "cache:15,gaussian:4,identity:2,sin:1" into two lists
        self.activations = []     # ['cache', 'gaussian', 'identity', 'sin']
        self.d_hidden = []        # [15, 4, 2, 1]
        for item in activation_neurons_str.split(','):
            name, count = item.split(':')
            self.activations.append(name)
            self.d_hidden.append(int(count))

        self.width = sum(self.d_hidden)   # 22 total neurons per layer

        # CONCEPT: nn.ModuleList is a Python list that PyTorch can see inside
        # WHY: If we used a plain list, model.parameters() wouldn't find the layers
        self.layers = nn.ModuleList()

        d_in = 4  # (y, x, d, b)
        for i in range(self.n_layers):
            # CONCEPT: nn.Linear(in, out, bias=False) creates a weight matrix W
            # and computes x @ W.T (same as Flax's nn.Dense with use_bias=False)
            layer = nn.Linear(d_in, self.width, bias=False)

            # CONCEPT: Custom weight initialization
            # WHY: fer supports a custom init_scale for controlling initial weight variance.
            #       Default Flax uses Lecun init; PyTorch default uses Kaiming uniform.
            #       When init_scale is specified, we match fer's truncated normal init.
            if init_scale != 'default':
                scale = float(init_scale)
                nn.init.trunc_normal_(layer.weight, std=(scale / d_in) ** 0.5)

            self.layers.append(layer)
            d_in = self.width  # after first layer, input size = hidden width

        # Output layer: hidden → 3 (h, s, v)
        self.output_layer = nn.Linear(self.width, 3, bias=False)
        if init_scale != 'default':
            nn.init.trunc_normal_(self.output_layer.weight, std=(float(init_scale) / self.width) ** 0.5)

    # CONCEPT: forward() defines what happens when data flows through the model
    # WHY: You call model(x), which internally calls forward(x) plus gradient hooks
    # NOTE: x comes in as shape (batch, 4) and exits as (batch, 3)
    def forward(self, x, return_features=False):
        features = [x]  # save input as "layer 0 features" (for visualization)

        for layer in self.layers:
            x = layer(x)            # Dense multiply: (batch, d_in) → (batch, 22)

            # CONCEPT: torch.split divides tensor along a dimension by GROUP SIZES
            # This is different from jnp.split which uses cumulative INDICES
            # split [15, 4, 2, 1] on a (batch, 22) tensor → four chunks
            groups = torch.split(x, self.d_hidden, dim=-1)

            # Apply each activation to its corresponding group
            groups = [
                activation_fn_map[act](group)
                for act, group in zip(self.activations, groups)
            ]

            # CONCEPT: torch.cat glues tensors back together (reverse of split)
            x = torch.cat(groups, dim=-1)   # → (batch, 22)

            features.append(x)

        x = self.output_layer(x)   # (batch, 22) → (batch, 3) = (h, s, v)
        features.append(x)

        if return_features:
            return x, features
        return x
```

### The Image Generation Function

```python
def generate_image(model, params_vector=None, img_size=256, return_features=False):
    """
    Generate a (img_size, img_size, 3) RGB image from a CPPN.

    If params_vector is given, loads those weights into the model first.
    This replaces fer's FlattenCPPNParameters.generate_image().
    """
    # CONCEPT: vector_to_parameters writes a 1D tensor back into model weights
    # WHY: For weight sweeps, we need to swap weights in and out quickly
    if params_vector is not None:
        vector_to_parameters(params_vector, model.parameters())

    # Build the coordinate grid — same math as fer's generate_image
    # torch.linspace works exactly like jnp.linspace / np.linspace
    coords = torch.linspace(-1, 1, img_size)

    # CONCEPT: meshgrid creates two 2D grids from two 1D vectors
    # grid_y[i,j] = coords[i] for all j  (varies down rows)
    # grid_x[i,j] = coords[j] for all i  (varies across columns)
    grid_y, grid_x = torch.meshgrid(coords, coords, indexing='ij')

    d = torch.sqrt(grid_x**2 + grid_y**2) * 1.4   # distance from center
    b = torch.ones_like(grid_x)                     # bias = 1 everywhere

    # Stack into (img_size, img_size, 4) then flatten to (img_size*img_size, 4)
    # CONCEPT: This flatten+unflatten replaces fer's jax.vmap(jax.vmap(...))
    # Instead of "apply function to each pixel," we say:
    #   "treat all pixels as one big batch, run them all at once"
    inputs = torch.stack([grid_y, grid_x, d, b], dim=-1)
    flat_inputs = inputs.view(-1, 4)   # (65536, 4) for 256x256

    # CONCEPT: torch.no_grad() turns off gradient tracking
    # WHY: We're just generating an image, not training. Saves memory + speed.
    with torch.no_grad():
        if return_features:
            flat_output, features = model(flat_inputs, return_features=True)
            # Reshape each feature layer back to (H, W, neurons)
            features = [f.view(img_size, img_size, -1) for f in features]
        else:
            flat_output = model(flat_inputs)

    # Reshape back to (img_size, img_size, 3) = (H, W, [h, s, v])
    hsv = flat_output.view(img_size, img_size, 3)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    # Convert HSV → RGB (same math as fer's color.py)
    rgb = hsv_to_rgb(h, s, v)

    if return_features:
        return rgb, features
    return rgb
```

### HSV to RGB Conversion

```python
def hsv_to_rgb(h, s, v):
    """
    Convert CPPN output (h, s, v) to RGB.

    The CPPN outputs raw values (any range), so we need to normalize:
    - h: shift by +1, then mod 1 to wrap into [0, 1]
    - s: clamp to [0, 1]
    - v: take absolute value, clamp to [0, 1]

    This matches fer's color.py exactly.
    """
    # CONCEPT: The CPPN outputs aren't already in [0,1] — we normalize here
    # WHY: tanh/sin/gaussian outputs can be negative. h wraps (it's circular),
    #       s clips (can't have negative saturation), v uses abs (negative = same brightness)
    h = (h + 1) % 1       # wrap hue into [0, 1)
    s = s.clamp(0, 1)     # clamp saturation
    v = v.abs().clamp(0, 1)  # absolute value, then clamp brightness

    # Standard HSV → RGB math (piecewise over 6 hue sectors)
    h360 = h * 360.0
    c = v * s                                    # chroma
    x = c * (1 - ((h360 / 60) % 2 - 1).abs())   # second-largest component
    m = v - c                                     # match value

    # Select RGB based on which 60-degree sector of the hue wheel
    c1 = ((0 <= h360) & (h360 < 60)).float()
    c2 = ((60 <= h360) & (h360 < 120)).float()
    c3 = ((120 <= h360) & (h360 < 180)).float()
    c4 = ((180 <= h360) & (h360 < 240)).float()
    c5 = ((240 <= h360) & (h360 < 300)).float()
    c6 = ((300 <= h360) & (h360 < 360)).float()

    r = c*c1 + x*c2 + 0*c3 + 0*c4 + x*c5 + c*c6
    g = x*c1 + c*c2 + c*c3 + x*c4 + 0*c5 + 0*c6
    b_ch = 0*c1 + 0*c2 + x*c3 + c*c4 + c*c5 + x*c6

    r = (r + m).clamp(0, 1)
    g = (g + m).clamp(0, 1)
    b_ch = (b_ch + m).clamp(0, 1)

    return torch.stack([r, g, b_ch], dim=-1)  # (H, W, 3)
```

### Weight Sweep Utility

```python
def sweep_weight(model, params_vector, weight_idx, r=1.0, n=5):
    """
    Vary a single weight from (original - r) to (original + r) in n steps.
    Returns n images showing how that one weight affects the output.

    This is how you find "interesting" weights — ones that control
    semantic features like mouth opening, eye shape, wing angle.

    Replaces fer's sweep_weight() from fer.ipynb.
    """
    images = []
    original_val = params_vector[weight_idx].item()

    for val in torch.linspace(original_val - r, original_val + r, n):
        # CONCEPT: .clone() makes a copy so we don't modify the original
        swept = params_vector.clone()
        swept[weight_idx] = val
        img = generate_image(model, swept)
        images.append(img)

    return torch.stack(images)  # (n, H, W, 3)
```

---

## Part E: The Full Data Flow (End to End)

Here's what happens when you generate a skull image:

```
1. Load pre-computed params from pickle file
   params_vector: shape (5346,) ← a 1D vector of all weights
   arch: "12;cache:15,gaussian:4,identity:2,sin:1"

2. Create CPPN model from arch string
   model = CPPN(arch)
   → creates 12 hidden nn.Linear(22, 22) + 1 input nn.Linear(4, 22) + 1 output nn.Linear(22, 3)

3. Load weights into model
   vector_to_parameters(params_vector, model.parameters())
   → fills all weight matrices from the 1D vector

4. Build coordinate grid
   y, x = meshgrid(linspace(-1,1,256), linspace(-1,1,256))
   d = sqrt(x² + y²) * 1.4
   b = ones(256, 256)
   inputs: shape (256, 256, 4)

5. Flatten and forward pass
   flat_inputs: (65536, 4)
   → Layer 0:  Linear(4→22) → split → [cache(15), gaussian(4), identity(2), sin(1)] → cat → (65536, 22)
   → Layer 1:  Linear(22→22) → split → activations → cat → (65536, 22)
   → ...
   → Layer 11: Linear(22→22) → split → activations → cat → (65536, 22)
   → Output:   Linear(22→3) → (65536, 3)

6. Reshape + HSV→RGB
   hsv: (256, 256, 3)
   rgb: (256, 256, 3) ← ready to display with matplotlib
```

---

## Part F: Gotchas

### 1. Flax creates layers inline; PyTorch creates them upfront

In Flax, you can write `nn.Dense(22)(x)` inside `forward`, and Flax creates the layer on the fly. In PyTorch, you **must** create all `nn.Linear` layers in `__init__`. If you create them in `forward`, they get recreated every call (new random weights each time).

### 2. torch.split uses sizes, not indices

```python
# Flax/JAX: split at cumulative indices [15, 19, 21]
jnp.split(x, [15, 19, 21])

# PyTorch: split by group sizes [15, 4, 2, 1]
torch.split(x, [15, 4, 2, 1], dim=-1)
```

Both give the same result, but the API is different. PyTorch's is more intuitive — the numbers tell you how big each group is.

### 3. The first hidden layer has different input size

The first layer is `Linear(4, 22)` (4 inputs from coordinates). All subsequent layers are `Linear(22, 22)`. Our code handles this with the `d_in` variable that starts at 4 and becomes `self.width` after the first layer.

### 4. vector_to_parameters modifies weights IN PLACE

```python
vector_to_parameters(new_weights, model.parameters())
# model's weights have been CHANGED — the old weights are gone
```

This is different from Flax where params are an immutable dict you pass around. In PyTorch, calling this function **mutates the model**. If you need to go back, you must save the old vector first with `parameters_to_vector`.

### 5. Weight initialization matters for matching fer's outputs

If you load pre-computed Picbreeder weights (from the pickle files), initialization doesn't matter — you're overwriting all weights anyway. But if you train from scratch (SGD), the initial random weights affect the final result. fer uses Lecun init (Flax default); PyTorch defaults to Kaiming uniform. We handle this with the `init_scale` parameter.

### 6. `.float()` for boolean masks

In the HSV→RGB conversion, conditions like `(0 <= h360) & (h360 < 60)` produce boolean tensors. You can't multiply booleans with floats in PyTorch — you need `.float()` to convert True/False to 1.0/0.0 first.

---

## What's Next

- **Tutorial 03**: KAN layers — how B-spline edge functions replace Linear+activation
- **Tutorial 04**: Swapping the MLP layers for KAN layers in the CPPN
- **Tutorial 05**: Weight sweeps, feature maps, and visualizing what the CPPN learns
- **Tutorial 06**: Swarm-based optimization for KAN spline coefficients

---

## Quick Reference: Flax → PyTorch Translation

| fer (Flax/JAX) | Our code (PyTorch) |
|---|---|
| `nn.Dense(22, use_bias=False)(x)` | `self.layer(x)` where layer = `nn.Linear(d_in, 22, bias=False)` |
| `jnp.split(x, [15, 19, 21])` | `torch.split(x, [15, 4, 2, 1], dim=-1)` |
| `jnp.concatenate(groups)` | `torch.cat(groups, dim=-1)` |
| `jax.vmap(jax.vmap(fn))` | `x.view(-1, 4)` → `model(flat)` → `.view(H, W, 3)` |
| `evosax.ParameterReshaper` | `parameters_to_vector` / `vector_to_parameters` |
| `jnp.exp(-x**2)` | `torch.exp(-x**2)` |
| `jax.nn.sigmoid(x)` | `torch.sigmoid(x)` |
| `x.clip(0, 1)` | `x.clamp(0, 1)` |
| `jnp.abs(v)` | `v.abs()` or `torch.abs(v)` |
