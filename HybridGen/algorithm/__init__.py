"""
HybridGen Algorithm Registry.

Usage:
    from HybridGen.algorithm import load_algorithm

    algo = load_algorithm("v2")       # returns hybrid_algorithm_v2 module
    algo = load_algorithm("v2.1")     # returns hybrid_algorithm_v21 module
    algo = load_algorithm("v1")       # returns hybrid_algorithm module

    # Or get the evaluate_miner function directly:
    evaluate = load_algorithm("v2").evaluate_miner
"""

import os
import importlib

from ..utils import import_modules


ALGORITHM_REGISTRY = {}


def load_algorithm(name: str):
    """
    Load an algorithm module by name.

    Args:
        name: Algorithm version (e.g., "v1", "v2", "v2.1").

    Returns:
        The loaded Python module.
    """
    # Normalize: strip dots, allow "2.1" -> "v21"
    normalized = name.replace(".", "").lower()
    if not normalized.startswith("v"):
        normalized = "v" + normalized

    if normalized in ALGORITHM_REGISTRY:
        return ALGORITHM_REGISTRY[normalized]

    # Fuzzy match
    for key in ALGORITHM_REGISTRY:
        if key.lower() in normalized or normalized in key.lower():
            return ALGORITHM_REGISTRY[key]

    raise ValueError(
        f"Unknown algorithm: '{name}'. Registered: {list(ALGORITHM_REGISTRY.keys())}"
    )


def register_algorithm(name: str):
    """
    Register the calling module under the given algorithm name.
    Automatically detects the caller's module via sys.modules inspection.
    """
    import inspect
    import sys
    caller_frame = inspect.currentframe().f_back
    caller_module_name = caller_frame.f_globals.get('__name__', None)
    if caller_module_name and caller_module_name in sys.modules:
        ALGORITHM_REGISTRY[name] = sys.modules[caller_module_name]
    else:
        raise RuntimeError(f"Cannot determine caller module for algorithm '{name}'")


# Auto-discover algorithm modules
_alg_dir = os.path.dirname(__file__)
import_modules(_alg_dir, "HybridGen.algorithm")
