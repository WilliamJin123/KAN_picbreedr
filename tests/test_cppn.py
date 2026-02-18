"""Test that the PyTorch CPPN port reproduces the JAX reference images."""

import os
import sys
import numpy as np
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.cppn import CPPN, FlattenCPPNParameters
from src.data import load_genome

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'references', 'fer', 'data')
THRESHOLD = 0.01


def test_genome(source, genome):
    """Test a single genome by comparing generated image to reference."""
    arch, params = load_genome(source, genome)
    print(f"\n--- {source}_{genome} ---")
    print(f"  arch: {arch}")
    print(f"  params: shape={params.shape}, dtype={params.dtype}")

    # Create CPPN and load parameters
    cppn = CPPN(arch)
    cppn_flat = FlattenCPPNParameters(cppn)
    print(f"  model n_params: {cppn_flat.n_params}")
    assert cppn_flat.n_params == params.shape[0], \
        f"Param count mismatch: model={cppn_flat.n_params}, data={params.shape[0]}"

    # Generate image
    img = cppn_flat.generate_image(params, img_size=256)
    img_np = img.detach().cpu().numpy()
    print(f"  generated image: shape={img_np.shape}, range=[{img_np.min():.4f}, {img_np.max():.4f}]")

    # Load reference image
    ref_path = os.path.join(DATA_DIR, f"{source}_{genome}", "img.png")
    ref_img = np.array(Image.open(ref_path).convert('RGB')).astype(np.float32) / 255.0
    print(f"  reference image: shape={ref_img.shape}, range=[{ref_img.min():.4f}, {ref_img.max():.4f}]")

    # Resize reference if needed (reference may be different resolution)
    if ref_img.shape[0] != 256 or ref_img.shape[1] != 256:
        from PIL import Image as PILImage
        ref_pil = PILImage.fromarray((ref_img * 255).astype(np.uint8))
        ref_pil = ref_pil.resize((256, 256), PILImage.BILINEAR)
        ref_img = np.array(ref_pil).astype(np.float32) / 255.0
        print(f"  resized reference to: {ref_img.shape}")

    # Compute MSE
    mse = np.mean((img_np - ref_img) ** 2)
    print(f"  MSE: {mse:.6f}")

    status = "PASS" if mse < THRESHOLD else "FAIL"
    print(f"  Result: {status}")
    return mse < THRESHOLD


def test_roundtrip():
    """Test that flatten -> load roundtrip preserves parameters exactly."""
    print("\n--- Roundtrip test ---")
    arch, params = load_genome('picbreeder', 'skull')
    cppn = CPPN(arch)
    cppn_flat = FlattenCPPNParameters(cppn)

    # Load JAX params
    cppn_flat.load_jax_flat_params(params)

    # Flatten back to JAX order
    roundtripped = cppn_flat.flatten()

    diff = (params - roundtripped).abs().max().item()
    print(f"  Max absolute difference: {diff:.10f}")
    status = "PASS" if diff < 1e-6 else "FAIL"
    print(f"  Result: {status}")
    return diff < 1e-6


if __name__ == "__main__":
    all_passed = True

    # Test roundtrip first
    all_passed &= test_roundtrip()

    # Test all picbreeder genomes
    for genome in ['skull', 'butterfly', 'apple']:
        all_passed &= test_genome('picbreeder', genome)

    # Test all SGD genomes
    for genome in ['skull', 'butterfly', 'apple']:
        all_passed &= test_genome('sgd', genome)

    print("\n" + "=" * 50)
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    sys.exit(0 if all_passed else 1)
