"""
Utility: auto-import all Python modules in a directory for registry discovery.
"""

import importlib
import os
import pkgutil


def import_modules(package_dir: str, package_prefix: str):
    """
    Automatically import all Python modules in a directory.

    This triggers any @register_* decorators defined in those modules,
    populating the corresponding factory dictionaries.

    Args:
        package_dir: Absolute path to the directory containing modules.
        package_prefix: Dotted Python package prefix (e.g., "HybridGen.algorithm").
    """
    for _, module_name, _ in pkgutil.iter_modules([package_dir]):
        full_name = f"{package_prefix}.{module_name}"
        importlib.import_module(full_name)
