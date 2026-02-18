import numpy as np
import torch
import matplotlib.pyplot as plt


def viz_feature_maps(features, title=""):
    """Visualize feature maps for all layers of a CPPN.

    Args:
        features: List of tensors, each of shape (H, W, D) representing
                  per-layer activations from CPPN.generate_image(return_features=True).
        title: Optional figure title.

    Returns:
        matplotlib Figure object.
    """
    # Convert to numpy
    features_np = [f.detach().cpu().numpy() if isinstance(f, torch.Tensor) else f
                   for f in features]

    max_features_per_layer = max(f.shape[-1] for f in features_np)
    n_layers = len(features_np)

    fig, axs = plt.subplots(
        n_layers, max_features_per_layer,
        figsize=(6.5, n_layers / max_features_per_layer * 6.5),
        dpi=150,
    )
    if n_layers == 1:
        axs = axs[np.newaxis, :]
    if max_features_per_layer == 1:
        axs = axs[:, np.newaxis]

    for i in range(n_layers):
        # (H, W, D) -> (D, H, W)
        layer_features = np.transpose(features_np[i], (2, 0, 1))
        for j in range(max_features_per_layer):
            ax = axs[n_layers - 1 - i, j]
            if j >= len(layer_features):
                ax.set_visible(False)
                continue
            fmap = layer_features[j]
            ax.imshow(fmap, cmap='bwr_r', vmin=-1.0, vmax=1.0)
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0 and i % 2 == 0:
                ax.set_ylabel(f"{i}", fontsize=8)
            if i == 0:
                input_labels = ["$x$", "$y$", "$d$", "$1$"]
                if j < len(input_labels):
                    ax.set_xlabel(input_labels[j], fontsize=8)
            if i == n_layers - 1:
                output_labels = ["$h$", "$s$", "$v$"]
                if j < len(output_labels):
                    ax.set_title(output_labels[j], fontsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(0.1)

    plt.subplots_adjust(left=0.06, right=1.0, bottom=0.05, top=0.95,
                        wspace=0.15, hspace=0.15)
    fig.supylabel("Layer", fontsize=12, x=0.0)
    fig.supxlabel("Neuron", fontsize=12, y=0.0)
    if title:
        fig.suptitle(title, fontsize=14)
    return fig


def get_kan_param_info(kan_cppn, flat_idx):
    """Map a flat parameter index to structured KAN parameter info.

    Given a flat index into the concatenated parameter vector of a KAN_CPPN,
    identifies which layer, parameter type, and local element it refers to.

    Args:
        kan_cppn: A KAN_CPPN instance.
        flat_idx: Integer index into the flattened parameter vector.

    Returns:
        Dict with keys:
            layer_idx: Which KAN layer (0-indexed).
            param_type: One of 'base_weight', 'coeffs', 'weights'.
            local_shape_indices: Tuple of indices into the parameter tensor.
            description: Human-readable string like "Layer 5 base_weight[3,7]".
    """
    offset = 0
    for name, param in kan_cppn.named_parameters():
        n_elem = param.numel()
        if flat_idx < offset + n_elem:
            local_idx = flat_idx - offset
            local_shape_indices = np.unravel_index(local_idx, param.shape)

            # Parse name like "layers.5.base_weight"
            parts = name.split('.')
            layer_idx = int(parts[1])
            param_type = parts[2]

            idx_str = ','.join(str(i) for i in local_shape_indices)
            description = f"Layer {layer_idx} {param_type}[{idx_str}]"

            return {
                'layer_idx': layer_idx,
                'param_type': param_type,
                'local_shape_indices': local_shape_indices,
                'description': description,
            }
        offset += n_elem

    raise IndexError(f"flat_idx {flat_idx} is out of range (total params: {offset})")


def discover_interesting_kan_sweeps(kan_cppn, kan_flat, target_img, img_size=64,
                                     n_candidates_per_group=5, top_k=8):
    """Find the most visually impactful KAN weight indices for sweep visualization.

    For each layer and parameter type (base_weight, coeffs, weights), randomly
    samples candidate flat indices, measures the visual impact of perturbing each
    one, and returns the top_k most impactful.

    Args:
        kan_cppn: A KAN_CPPN instance.
        kan_flat: A FlattenKANParameters instance wrapping kan_cppn.
        target_img: Target image tensor (used only for context, not compared).
        img_size: Resolution for discovery phase images (small = fast).
        n_candidates_per_group: Random indices to sample per (layer, param_type).
        top_k: Number of top candidates to return.

    Returns:
        List of top_k dicts sorted by visual_impact (descending), each with:
            flat_idx: Index into the flat parameter vector.
            visual_impact: Mean absolute pixel difference between endpoints.
            description: e.g. "Layer 5 base_weight[3,7]".
    """
    params = kan_flat.flatten()

    # Build a map of (layer_idx, param_type) -> list of flat indices
    groups = {}
    offset = 0
    for name, param in kan_cppn.named_parameters():
        n_elem = param.numel()
        parts = name.split('.')
        layer_idx = int(parts[1])
        param_type = parts[2]
        key = (layer_idx, param_type)
        groups[key] = (offset, n_elem)
        offset += n_elem

    # Sample candidates from each group
    rng = np.random.RandomState(42)
    candidates = []
    for (layer_idx, param_type), (group_offset, group_size) in groups.items():
        n_sample = min(n_candidates_per_group, group_size)
        local_indices = rng.choice(group_size, size=n_sample, replace=False)
        for local_idx in local_indices:
            flat_idx = group_offset + local_idx
            candidates.append(flat_idx)

    # Evaluate visual impact for each candidate
    scored = []
    with torch.no_grad():
        for flat_idx in candidates:
            original_val = params[flat_idx].item()

            # Image at w - 1
            p_low = params.clone()
            p_low[flat_idx] = original_val - 1.0
            img_low = kan_flat.generate_image(p_low, img_size=img_size)

            # Image at w + 1
            p_high = params.clone()
            p_high[flat_idx] = original_val + 1.0
            img_high = kan_flat.generate_image(p_high, img_size=img_size)

            # Visual impact = mean absolute pixel difference
            impact = torch.mean(torch.abs(img_high - img_low)).item()

            info = get_kan_param_info(kan_cppn, flat_idx)
            scored.append({
                'flat_idx': flat_idx,
                'visual_impact': impact,
                'description': info['description'],
            })

    # Restore original parameters
    kan_flat.unflatten(params)

    # Sort by visual impact descending, return top_k
    scored.sort(key=lambda x: x['visual_impact'], reverse=True)
    return scored[:top_k]


def sweep_weight(params, weight_id, cppn_flat, img_size=256, center_weight=None, r=1, n=5):
    """Sweep a single weight across a range of values.

    Args:
        params: Flat parameter tensor (1D).
        weight_id: Index of the weight to sweep.
        cppn_flat: FlattenCPPNParameters instance.
        img_size: Resolution of generated images.
        center_weight: Center the sweep around this value (None = use original).
        r: Sweep radius, from -r to +r around center.
        n: Number of samples.

    Returns:
        Tensor of shape (n, img_size, img_size, 3).
    """
    weight_sweep = torch.linspace(-r, r, n)
    if center_weight is not None:
        weight_sweep = weight_sweep + center_weight
    else:
        weight_sweep = weight_sweep + params[weight_id].item()

    imgs = []
    for w_val in weight_sweep:
        p = params.clone()
        p[weight_id] = w_val
        img = cppn_flat.generate_image(p, img_size=img_size)
        imgs.append(img)

    return torch.stack(imgs)


def sweep_weight_random_direction(params, seed, cppn_flat, img_size=256, r=1, n=5):
    """Sweep a random direction in weight space.

    Selects a random layer, a random input row, and a random unit direction
    in the output space, then sweeps along that direction.

    Args:
        params: Flat parameter tensor (1D).
        seed: Random seed for reproducibility.
        cppn_flat: FlattenCPPNParameters instance.
        img_size: Resolution of generated images.
        r: Sweep radius.
        n: Number of samples.

    Returns:
        Tensor of shape (n, img_size, img_size, 3).
    """
    cppn = cppn_flat.cppn

    # Use numpy RNG for layer selection (matches JAX reference)
    np.random.seed(seed)
    layer_idx = np.random.randint(0, len(cppn.layers))
    layer = cppn.layers[layer_idx]

    # Use torch RNG for row and direction (matches JAX reference using PRNGKey)
    rng = torch.Generator().manual_seed(seed)
    n_rows = layer.in_features
    n_cols = layer.out_features
    i_row = torch.randint(0, n_rows, (1,), generator=rng).item()
    vec = torch.randn(n_cols, generator=rng)
    vec = vec / vec.norm()

    # Build perturbation matrix: e_i @ vec^T -> shape (n_rows, n_cols)
    e_i = torch.zeros(n_rows)
    e_i[i_row] = 1.0
    dW = e_i[:, None] @ vec[None, :]  # (n_rows, n_cols)

    ts = torch.linspace(-r, r, n)
    imgs = []
    for t in ts:
        # Load base params
        cppn_flat.load_jax_flat_params(params)
        # Perturb the specific layer's weight
        # dW is in Flax layout (in, out), PyTorch weight is (out, in)
        layer.weight.data += (t * dW).T
        img = cppn.generate_image(img_size=img_size)
        imgs.append(img)

    return torch.stack(imgs)


def plot_sweep_strip(imgs, title="", padding=6):
    """Plot a horizontal strip of images from a weight sweep.

    Args:
        imgs: Tensor of shape (N, H, W, 3) or numpy array.
        title: Plot title.
        padding: Black padding between images in pixels.

    Returns:
        matplotlib Figure object.
    """
    if isinstance(imgs, torch.Tensor):
        imgs = imgs.detach().cpu().numpy()

    imgs = np.pad(imgs, ((0, 0), (padding, 0), (padding, 0), (0, 0)),
                  mode='constant', constant_values=0.0)
    # (N, H, W, D) -> (H, N*W, D)
    strip = np.concatenate(imgs, axis=1)
    strip = np.pad(strip, ((0, padding), (0, padding), (0, 0)),
                   mode='constant', constant_values=0.0)

    fig, ax = plt.subplots(figsize=(20, 5), dpi=50)
    ax.imshow(strip)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    if title:
        ax.set_title(title, fontsize=30)
    return fig


def plot_sweep_grid(sweep_data, title="", padding=6):
    """Plot multiple weight sweeps in a grid.

    Args:
        sweep_data: List of dicts with keys:
            'imgs': tensor (N, H, W, 3)
            'weight_id': int
            'description': str (optional)
        title: Overall figure title.
        padding: Black padding between images.

    Returns:
        matplotlib Figure object.
    """
    n_rows = len(sweep_data)
    fig, axs = plt.subplots(n_rows, 1, figsize=(10, 10), dpi=50)
    if n_rows == 1:
        axs = [axs]

    for iplt, entry in enumerate(sweep_data):
        imgs = entry['imgs']
        if isinstance(imgs, torch.Tensor):
            imgs = imgs.detach().cpu().numpy()

        imgs = np.pad(imgs, ((0, 0), (padding, 0), (padding, 0), (0, 0)),
                      mode='constant', constant_values=0.0)
        strip = np.concatenate(imgs, axis=1)
        strip = np.pad(strip, ((0, padding), (0, padding), (0, 0)),
                       mode='constant', constant_values=0.0)

        ax = axs[iplt]
        ax.imshow(strip)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        desc = entry.get('description', '')
        if desc:
            ax.set_title(desc, fontsize=20)
        ax.set_ylabel(str(entry.get('weight_id', '')), fontsize=20)

        if iplt == n_rows - 1:
            img_size_p = imgs.shape[2]  # W + padding
            n_imgs = len(entry['imgs'])
            mid = n_imgs // 2
            ax.set_xticks(
                [img_size_p * 0 + img_size_p / 2,
                 img_size_p * mid + img_size_p / 2,
                 img_size_p * (n_imgs - 1) + img_size_p / 2],
                [r"$\Delta w = -$1", r"$\Delta w = $0", r"$\Delta w = +$1"],
                fontsize=25,
            )

    fig.supylabel("Weight ID", fontsize=30, x=0.03)
    fig.supxlabel("Sweeping Weight Value", fontsize=30)
    if title:
        fig.suptitle(title, fontsize=35)
    fig.tight_layout()
    return fig
