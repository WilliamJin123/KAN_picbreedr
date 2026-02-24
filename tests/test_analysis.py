"""Tests for analysis modules."""

import torch
import numpy as np
import pytest


def test_mse():
    from analysis.comparison import mse
    img1 = torch.zeros(4, 4, 3)
    img2 = torch.ones(4, 4, 3)
    assert abs(mse(img1, img2) - 1.0) < 1e-6

    # Identical images
    assert mse(img1, img1) == 0.0


def test_ssim_identical():
    from analysis.comparison import ssim
    img = torch.rand(32, 32, 3)
    s = ssim(img, img)
    assert s > 0.99  # Should be ~1.0 for identical images


def test_ssim_different():
    from analysis.comparison import ssim
    img1 = torch.zeros(32, 32, 3)
    img2 = torch.ones(32, 32, 3)
    s = ssim(img1, img2)
    assert s < 0.1  # Very different images


def test_extract_spline_curve():
    from src.kan import KANCPPNLayer
    from analysis.spline_inspector import extract_spline_curve

    layer = KANCPPNLayer(4, 8, grid_size=20)
    raw, vals, normed = extract_spline_curve(layer, in_idx=0, out_idx=0)

    assert raw.shape == (1000,)
    assert vals.shape == (1000,)
    assert normed.shape == (1000,)
    assert raw[0] == -3.0
    assert raw[-1] == 3.0
    assert np.all(normed >= 0) and np.all(normed <= 1)


def test_fit_known_function():
    from analysis.spline_inspector import fit_known_function

    # Create a sine-like curve
    x = np.linspace(-3, 3, 1000)
    y = np.sin(x)

    result = fit_known_function(x, y)
    assert result['name'] == 'sin'
    assert result['l2_distance'] < 0.1
    assert result['fitted_curve'] is not None


def test_sequence_kan():
    from analysis.text_prototype import SequenceKAN

    model = SequenceKAN(n_layers=2, hidden_size=8, output_size=1)
    positions, signal = model.generate_signal(seq_len=50)

    assert positions.shape == (50,)
    assert signal.shape == (50, 1)


def test_benchmark_loads_target():
    from analysis.benchmark import load_target_image

    img = load_target_image('skull', img_size=64)
    assert img.shape == (64, 64, 3)
    assert img.min() >= 0 and img.max() <= 1


def test_feature_cosine_similarity():
    from analysis.comparison import feature_cosine_similarity

    f1 = [torch.randn(8, 8, 4), torch.randn(8, 8, 4)]
    f2 = [torch.randn(8, 8, 4), torch.randn(8, 8, 4)]

    sims = feature_cosine_similarity(f1, f2)
    assert len(sims) == 2
    assert all(-1 <= s <= 1 for s in sims)

    # Identical features should have similarity ~1.0
    sims_same = feature_cosine_similarity(f1, f1)
    assert all(s > 0.99 for s in sims_same)
