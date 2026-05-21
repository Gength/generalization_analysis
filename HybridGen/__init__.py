"""
HybridGen: Hybrid Generative-Structural Generalization Evaluation.

Usage:
    from HybridGen.algorithm import load_algorithm
    from HybridGen.experiment import load_experiment

    algo = load_algorithm("v2.1")
    exp  = load_experiment("v2", algo)
    exp.main()

    # Or from CLI (auto-wires default algo):
    #   python -m HybridGen.experiment.v2 --miner all --weight 0.5
"""

# Import algorithm FIRST so registry is populated before experiment modules
# try to call load_algorithm() at module level.
from .algorithm import load_algorithm  # triggers algorithm module discovery
from .experiment import load_experiment  # now safe

__all__ = ["load_algorithm", "load_experiment"]
