# src/graph_viz.py
"""Directed graph visualization of KAN architectures with spline insets."""

import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure project root is on path for analysis imports
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from analysis.spline_inspector import extract_spline_curve, fit_known_function


def build_kan_graph_data(kan_cppn, impact_img_size=32):
    """Extract graph structure + spline data from a trained KAN-CPPN."""
    layers = kan_cppn.layers
    n_kan_layers = len(layers)

    nodes = []
    input_labels = ['y', 'x', 'd', 'b']
    for i in range(layers[0].in_features):
        label = input_labels[i] if i < len(input_labels) else f'in_{i}'
        nodes.append({'layer_idx': 0, 'neuron_idx': i, 'label': label})
    for l_idx, layer in enumerate(layers):
        node_layer = l_idx + 1
        for n_idx in range(layer.out_features):
            if l_idx == n_kan_layers - 1:
                out_labels = ['h', 's', 'v']
                label = out_labels[n_idx] if n_idx < len(out_labels) else f'out_{n_idx}'
            else:
                label = f'h{node_layer}_{n_idx}'
            nodes.append({'layer_idx': node_layer, 'neuron_idx': n_idx, 'label': label})

    edges = []
    for l_idx, layer in enumerate(layers):
        for out_idx in range(layer.out_features):
            for in_idx in range(layer.in_features):
                raw_inputs, spline_values, _ = extract_spline_curve(layer, in_idx, out_idx, n_points=200)
                best_match = fit_known_function(raw_inputs, spline_values)

                impact = float(np.sqrt(np.mean(spline_values ** 2)))

                edges.append({
                    'src_layer': l_idx,
                    'src_neuron': in_idx,
                    'dst_layer': l_idx + 1,
                    'dst_neuron': out_idx,
                    'kan_layer_idx': l_idx,
                    'raw_inputs': raw_inputs,
                    'spline_values': spline_values,
                    'best_fit_name': best_match['name'],
                    'best_fit_score': best_match['l2_distance'],
                    'fitted_curve': best_match['fitted_curve'],
                    'visual_impact': impact,
                })

    return {
        'nodes': nodes,
        'edges': edges,
        'n_node_layers': n_kan_layers + 1,
    }


def render_pruned_graph(graph_data, output_path, title="KAN Architecture",
                        percentile_threshold=50):
    """Render a pruned architecture graph with only high-impact edges."""
    edges = graph_data['edges']
    nodes = graph_data['nodes']
    n_node_layers = graph_data['n_node_layers']

    impacts = [e['visual_impact'] for e in edges]
    threshold = np.percentile(impacts, percentile_threshold)
    max_impact = max(impacts) if impacts else 1.0

    visible_edges = [e for e in edges if e['visual_impact'] >= threshold]

    layer_sizes = {}
    for node in nodes:
        l = node['layer_idx']
        layer_sizes[l] = layer_sizes.get(l, 0) + 1

    max_layer_size = max(layer_sizes.values())

    def node_pos(layer_idx, neuron_idx):
        size = layer_sizes[layer_idx]
        y = (neuron_idx - (size - 1) / 2) * 1.0
        x = layer_idx * 3.0
        return x, y

    fig_width = n_node_layers * 3 + 2
    fig_height = max_layer_size * 1.0 + 2
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=150)

    for edge in visible_edges:
        x0, y0 = node_pos(edge['src_layer'], edge['src_neuron'])
        x1, y1 = node_pos(edge['dst_layer'], edge['dst_neuron'])

        lw = 0.3 + 2.0 * (edge['visual_impact'] / max_impact)
        alpha = 0.3 + 0.7 * (edge['visual_impact'] / max_impact)

        ax.plot([x0, x1], [y0, y1], '-', color='steelblue', linewidth=lw, alpha=alpha)

        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        inset_size = 0.3
        inset_ax = ax.inset_axes(
            [mx - inset_size/2, my - inset_size/2, inset_size, inset_size],
            transform=ax.transData,
        )
        raw = edge['raw_inputs']
        vals = edge['spline_values']
        inset_ax.plot(raw, vals, 'b-', linewidth=0.5)
        if edge['fitted_curve'] is not None:
            inset_ax.plot(raw, edge['fitted_curve'], 'r--', linewidth=0.3, alpha=0.5)
        inset_ax.set_xticks([])
        inset_ax.set_yticks([])
        inset_ax.patch.set_alpha(0.8)
        inset_ax.patch.set_facecolor('white')
        for spine in inset_ax.spines.values():
            spine.set_linewidth(0.3)

        label = f"{edge['best_fit_name']}"
        ax.text(mx, my - inset_size/2 - 0.05, label,
                ha='center', va='top', fontsize=3, color='dimgray')

    for node in nodes:
        x, y = node_pos(node['layer_idx'], node['neuron_idx'])
        circle = plt.Circle((x, y), 0.15, facecolor='lightcoral', edgecolor='black',
                           linewidth=0.5, zorder=5)
        ax.add_patch(circle)
        ax.text(x, y, node['label'], ha='center', va='center', fontsize=4, zorder=6)

    ax.set_xlim(-1, n_node_layers * 3)
    ax.set_ylim(-max_layer_size / 2 - 1, max_layer_size / 2 + 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=12)

    try:
        fig.savefig(output_path, bbox_inches='tight', dpi=150)
    finally:
        plt.close(fig)


def render_full_graph_by_layer(graph_data, output_dir, title_prefix=""):
    """Render one bipartite graph per layer transition with ALL edges."""
    os.makedirs(output_dir, exist_ok=True)

    edges = graph_data['edges']
    nodes = graph_data['nodes']

    edges_by_layer = {}
    for edge in edges:
        l = edge['kan_layer_idx']
        edges_by_layer.setdefault(l, []).append(edge)

    nodes_by_layer = {}
    for node in nodes:
        l = node['layer_idx']
        nodes_by_layer.setdefault(l, []).append(node)

    for kan_layer_idx, layer_edges in sorted(edges_by_layer.items()):
        src_layer = kan_layer_idx
        dst_layer = kan_layer_idx + 1
        src_nodes = nodes_by_layer.get(src_layer, [])
        dst_nodes = nodes_by_layer.get(dst_layer, [])

        n_src = len(src_nodes)
        n_dst = len(dst_nodes)
        n_edges = len(layer_edges)

        max_impact = max(e['visual_impact'] for e in layer_edges) if layer_edges else 1.0

        spacing = 1.2
        col_gap = 4.0

        fig_height = max(n_src, n_dst) * spacing + 2
        fig_width = col_gap + 4
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=150)

        def src_pos(i):
            return 0, (i - (n_src - 1) / 2) * spacing

        def dst_pos(i):
            return col_gap, (i - (n_dst - 1) / 2) * spacing

        for edge in layer_edges:
            x0, y0 = src_pos(edge['src_neuron'])
            x1, y1 = dst_pos(edge['dst_neuron'])

            lw = 0.2 + 1.5 * (edge['visual_impact'] / max_impact)
            alpha = 0.2 + 0.6 * (edge['visual_impact'] / max_impact)
            ax.plot([x0, x1], [y0, y1], '-', color='steelblue', linewidth=lw, alpha=alpha)

            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            inset_size = 0.4
            inset_ax = ax.inset_axes(
                [mx - inset_size/2, my - inset_size/2, inset_size, inset_size],
                transform=ax.transData,
            )
            inset_ax.plot(edge['raw_inputs'], edge['spline_values'], 'b-', linewidth=0.4)
            if edge['fitted_curve'] is not None:
                inset_ax.plot(edge['raw_inputs'], edge['fitted_curve'], 'r--', linewidth=0.2, alpha=0.5)
            inset_ax.set_xticks([])
            inset_ax.set_yticks([])
            inset_ax.patch.set_alpha(0.85)
            inset_ax.patch.set_facecolor('white')
            for spine in inset_ax.spines.values():
                spine.set_linewidth(0.2)

            ax.text(mx, my - inset_size/2 - 0.03,
                    f"{edge['best_fit_name']}",
                    ha='center', va='top', fontsize=3, color='dimgray')

        for i, node in enumerate(src_nodes):
            x, y = src_pos(i)
            circle = plt.Circle((x, y), 0.2, facecolor='lightcoral', edgecolor='black',
                               linewidth=0.5, zorder=5)
            ax.add_patch(circle)
            ax.text(x, y, node['label'], ha='center', va='center', fontsize=5, zorder=6)

        for i, node in enumerate(dst_nodes):
            x, y = dst_pos(i)
            circle = plt.Circle((x, y), 0.2, facecolor='lightskyblue', edgecolor='black',
                               linewidth=0.5, zorder=5)
            ax.add_patch(circle)
            ax.text(x, y, node['label'], ha='center', va='center', fontsize=5, zorder=6)

        ax.set_xlim(-1, col_gap + 1)
        ax.set_ylim(-max(n_src, n_dst) * spacing / 2 - 1,
                     max(n_src, n_dst) * spacing / 2 + 1)
        ax.set_aspect('equal')
        ax.axis('off')

        prefix = f"{title_prefix} " if title_prefix else ""
        ax.set_title(f"{prefix}Layer {src_layer} -> {dst_layer} ({n_edges} edges)", fontsize=10)

        filename = f'layer_pair_{src_layer:02d}_{dst_layer:02d}.png'
        try:
            fig.savefig(os.path.join(output_dir, filename), bbox_inches='tight', dpi=150)
        finally:
            plt.close(fig)
