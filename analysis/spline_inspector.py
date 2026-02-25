"""Inspect and analyze learned spline activation functions in KAN layers.

Extracts the effective spline curve for each (layer, in, out) edge,
compares against known activation functions, and identifies what
the KAN learned.
"""

import torch
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import BSpline


# Library of known activation functions (evaluated on raw input domain)
# The KAN applies sigmoid first, so we define these on [0, 1] (normalized domain)
# AND on the raw domain [-3, 3] for comparison.
KNOWN_FUNCTIONS = {
    'identity': lambda x: x,
    'sin': lambda x: np.sin(x),
    'cos': lambda x: np.cos(x),
    'tanh': lambda x: np.tanh(x),
    'sigmoid': lambda x: 1 / (1 + np.exp(-x)) * 2 - 1,
    'gaussian': lambda x: np.exp(-x**2) * 2 - 1,
    'relu': lambda x: np.maximum(0, x),
    'quadratic': lambda x: x**2,
    'abs': lambda x: np.abs(x),
    'constant': lambda x: np.zeros_like(x),
}

# Parameterized versions for curve fitting
def _scaled_shifted(func):
    """Wrap a base function with learnable scale and shift: a * f(b*x + c) + d"""
    def parameterized(x, a, b, c, d):
        return a * func(b * x + c) + d
    return parameterized

FITTABLE_FUNCTIONS = {
    name: _scaled_shifted(fn) for name, fn in KNOWN_FUNCTIONS.items()
    if name != 'constant'
}


def extract_spline_curve(layer, in_idx, out_idx, n_points=1000):
    """Extract the effective spline curve for one (in, out) edge.

    Evaluates the spline at n_points uniformly spaced inputs by:
    1. Creating inputs in the raw domain [-3, 3]
    2. Applying sigmoid to map to [0, 1] (the grid domain)
    3. Interpolating on the spline grid (linear for degree 1, scipy BSpline for higher)
    4. Multiplying by the edge weight

    Args:
        layer: A KANCPPNLayer instance.
        in_idx: Input feature index.
        out_idx: Output feature index.
        n_points: Number of evaluation points.

    Returns:
        raw_inputs: numpy array of shape (n_points,) in [-3, 3]
        spline_values: numpy array of shape (n_points,) — the spline output
        normalized_inputs: numpy array in [0, 1] (after sigmoid)
    """
    raw_inputs = np.linspace(-3, 3, n_points)
    normalized = 1 / (1 + np.exp(-raw_inputs))  # sigmoid

    # Get coefficients and weight for this edge
    coeffs = layer.coeffs[out_idx, in_idx].detach().cpu().numpy()
    weight = layer.weights[out_idx, in_idx].detach().cpu().item()

    degree = getattr(layer, 'spline_degree', 1)

    if degree == 1:
        # Original linear interpolation (unchanged)
        grid_size = layer.grid_size
        scaled = normalized * (grid_size - 1)
        idx = np.clip(scaled.astype(int), 0, grid_size - 2)
        frac = scaled - idx

        left = coeffs[idx]
        right = coeffs[idx + 1]
        spline_values = (left + frac * (right - left)) * weight
    else:
        # Higher-degree B-spline via scipy
        knots = layer.knots.detach().cpu().numpy()
        spl = BSpline(knots, coeffs, degree, extrapolate=False)
        # Clamp to valid domain to avoid NaN at boundaries
        t_clamped = np.clip(normalized, knots[0], knots[-1])
        spline_values = spl(t_clamped) * weight

    return raw_inputs, spline_values, normalized


def fit_known_function(raw_inputs, spline_values):
    """Find the best-matching known activation function for a spline curve.

    Tries fitting each known function with scale/shift parameters.

    Args:
        raw_inputs: numpy array of input values.
        spline_values: numpy array of spline output values.

    Returns:
        best_match: dict with keys:
            name: str — name of best-matching function
            l2_distance: float — L2 error of the best fit
            params: tuple — (a, b, c, d) fitted parameters
            fitted_curve: numpy array — the fitted function evaluated at raw_inputs
            all_fits: dict mapping name -> (l2_distance, fitted_curve)
    """
    all_fits = {}

    for name, func in FITTABLE_FUNCTIONS.items():
        try:
            popt, _ = curve_fit(func, raw_inputs, spline_values,
                                p0=[1.0, 1.0, 0.0, 0.0],
                                maxfev=5000)
            fitted = func(raw_inputs, *popt)
            l2 = np.sqrt(np.mean((spline_values - fitted) ** 2))
            all_fits[name] = (l2, fitted, popt)
        except (RuntimeError, ValueError):
            # curve_fit failed — skip this function
            all_fits[name] = (float('inf'), None, None)

    # Also try constant (just the mean)
    mean_val = np.mean(spline_values)
    const_curve = np.full_like(spline_values, mean_val)
    const_l2 = np.sqrt(np.mean((spline_values - const_curve) ** 2))
    all_fits['constant'] = (const_l2, const_curve, (mean_val,))

    # Find best match
    best_name = min(all_fits, key=lambda k: all_fits[k][0])
    best_l2, best_curve, best_params = all_fits[best_name]

    return {
        'name': best_name,
        'l2_distance': best_l2,
        'params': best_params,
        'fitted_curve': best_curve,
        'all_fits': {k: (v[0], v[1]) for k, v in all_fits.items()},
    }


def analyze_all_edges(kan_cppn, top_k=20):
    """Analyze all spline edges in a KAN-CPPN and find what they learned.

    Args:
        kan_cppn: A KAN_CPPN instance.
        top_k: Return top_k edges by signal magnitude.

    Returns:
        List of dicts sorted by signal magnitude (descending), each with:
            layer_idx: int
            in_idx: int
            out_idx: int
            raw_inputs: numpy array
            spline_values: numpy array
            best_match: result from fit_known_function
            signal_magnitude: float — RMS of spline values
    """
    edges = []

    for layer_idx, layer in enumerate(kan_cppn.layers):
        for out_idx in range(layer.out_features):
            for in_idx in range(layer.in_features):
                raw_inputs, spline_values, _ = extract_spline_curve(layer, in_idx, out_idx)
                signal_mag = np.sqrt(np.mean(spline_values ** 2))

                best_match = fit_known_function(raw_inputs, spline_values)

                edges.append({
                    'layer_idx': layer_idx,
                    'in_idx': in_idx,
                    'out_idx': out_idx,
                    'raw_inputs': raw_inputs,
                    'spline_values': spline_values,
                    'best_match': best_match,
                    'signal_magnitude': signal_mag,
                })

    # Sort by signal magnitude (most active edges first)
    edges.sort(key=lambda e: e['signal_magnitude'], reverse=True)
    return edges[:top_k]
