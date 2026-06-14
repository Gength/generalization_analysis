"""
HybridGen Algorithm Registry.

Usage:
    from HybridGen.algorithm import load_algorithm

    algo = load_algorithm("v2.0")     # returns hybrid_algorithm_v2 module
    algo = load_algorithm("v2.1")     # returns hybrid_algorithm_v21 module
    algo = load_algorithm("v1.0")     # returns hybrid_algorithm module

    # Undotted aliases also work for backward compat:
    algo = load_algorithm("v2")       # same as "v2.0"
    algo = load_algorithm("v21")      # same as "v2.1"

    # Or get the evaluate_miner function directly:
    evaluate = load_algorithm("v2.0").evaluate_miner
"""

import os
import importlib

from ..utils import import_modules


ALGORITHM_REGISTRY = {}


def load_algorithm(name: str):
    """
    Load an algorithm module by name.

    Args:
        name: Algorithm version in canonical dot notation, e.g. "v2.1", "v2.4".
              Undotted aliases (e.g. "v21", "v24") also work for backward compat.

    Returns:
        The loaded Python module.
    """
    # Normalize: strip dots to support both "v2.1" and "v21"
    normalized = name.replace(".", "").lower()
    if not normalized.startswith("v"):
        normalized = "v" + normalized

    if normalized in ALGORITHM_REGISTRY:
        return ALGORITHM_REGISTRY[normalized]

    # Fuzzy match (fallback for partial names)
    for key in ALGORITHM_REGISTRY:
        if key.lower() in normalized or normalized in key.lower():
            return ALGORITHM_REGISTRY[key]

    raise ValueError(
        f"Unknown algorithm: '{name}'. Registered: {sorted(ALGORITHM_REGISTRY.keys())}"
    )


def register_algorithm(name: str):
    """
    Register the calling module under the given algorithm name.
    Automatically detects the caller's module via sys.modules inspection.

    The canonical name should use dot notation (e.g. "v2.1").
    An undotted alias (e.g. "v21") is also registered automatically for backward compat.
    """
    import inspect
    import sys
    caller_frame = inspect.currentframe().f_back
    caller_module_name = caller_frame.f_globals.get('__name__', None)
    if caller_module_name and caller_module_name in sys.modules:
        mod = sys.modules[caller_module_name]
        ALGORITHM_REGISTRY[name] = mod
        # Also register the dot-stripped alias for backward compat
        stripped = name.replace(".", "").lower()
        if stripped != name:
            ALGORITHM_REGISTRY[stripped] = mod
    else:
        raise RuntimeError(f"Cannot determine caller module for algorithm '{name}'")


# Auto-discover algorithm modules
_alg_dir = os.path.dirname(__file__)
import_modules(_alg_dir, "HybridGen.algorithm")
