import os
import pickle

import numpy as np
import torch

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'references', 'fer', 'data')


class _JaxToNumpyUnpickler(pickle.Unpickler):
    """Custom unpickler that converts JAX DeviceArray objects to numpy arrays."""

    def find_class(self, module, name):
        if module == 'jax._src.array' and name == '_reconstruct_array':
            def reconstruct(np_reconstruct, init_args, state, attrs):
                arr = np_reconstruct(*init_args)
                arr.__setstate__(state)
                return arr
            return reconstruct
        return super().find_class(module, name)


def load_pkl(load_dir, name):
    path = os.path.join(load_dir, f"{name}.pkl")
    with open(path, "rb") as f:
        try:
            return pickle.load(f)
        except ModuleNotFoundError:
            pass
    # Retry with JAX-aware unpickler
    with open(path, "rb") as f:
        return _JaxToNumpyUnpickler(f).load()


def save_pkl(save_dir, name, item):
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, f"{name}.pkl"), "wb") as f:
        pickle.dump(item, f)


def load_genome(source, genome):
    """Load architecture and parameters for a genome.

    Args:
        source: 'picbreeder' or 'sgd'
        genome: 'skull', 'butterfly', or 'apple'

    Returns:
        (arch_string, params_tensor) where params_tensor is a flat 1D torch tensor.
    """
    save_dir = os.path.join(DATA_DIR, f"{source}_{genome}")
    arch = load_pkl(save_dir, "arch")
    params = load_pkl(save_dir, "params")
    params = torch.tensor(np.array(params), dtype=torch.float32)
    return arch, params
