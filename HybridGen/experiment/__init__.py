"""
HybridGen Experiment Registry — Auto-discovery with algorithm injection.

Usage:
    from HybridGen.algorithm import load_algorithm
    from HybridGen.experiment import load_experiment

    algo = load_algorithm("v2.1")
    exp  = load_experiment("v2", algo)   # injects algo, returns module
    exp.main()
"""

import os
import importlib

from ..utils import import_modules


EXPERIMENT_REGISTRY = {}


def load_experiment(name: str, algo=None):
    """
    Load an experiment module and optionally inject an algorithm.

    Args:
        name: Experiment version (e.g., "v1", "v2", "v21", "v2.1").
        algo: Algorithm module (from load_algorithm). Injected as module.algo.

    Returns:
        The loaded Python module (call .main() to run).
    """
    normalized = name.replace(".", "").lower()
    if not normalized.startswith("v"):
        normalized = "v" + normalized

    if normalized in EXPERIMENT_REGISTRY:
        mod = EXPERIMENT_REGISTRY[normalized]
        if algo is not None:
            mod.algo = algo
        return mod

    # Fuzzy match
    for key in EXPERIMENT_REGISTRY:
        if key.lower() in normalized or normalized in key.lower():
            mod = EXPERIMENT_REGISTRY[key]
            if algo is not None:
                mod.algo = algo
            return mod

    raise ValueError(
        f"Unknown experiment: '{name}'. Registered: {list(EXPERIMENT_REGISTRY.keys())}"
    )


def register_experiment(name: str):
    """
    Register the calling module under the given experiment name.
    Automatically detects the caller's module via sys.modules inspection.
    """
    import inspect
    import sys
    caller_frame = inspect.currentframe().f_back
    caller_module_name = caller_frame.f_globals.get('__name__', None)
    if caller_module_name and caller_module_name in sys.modules:
        EXPERIMENT_REGISTRY[name] = sys.modules[caller_module_name]
    else:
        raise RuntimeError(f"Cannot determine caller module for experiment '{name}'")


# Auto-discover experiment modules
_exp_dir = os.path.dirname(__file__)
import_modules(_exp_dir, "HybridGen.experiment")
