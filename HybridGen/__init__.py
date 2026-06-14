"""
HybridGen: Hybrid Generative-Structural Generalization Evaluation.

    CLI:
        python -m HybridGen -a v2.1 -e v2 --miner all --weight 0.5
        python -m HybridGen --list

    Python API:
        from HybridGen.algorithm import load_algorithm
        from HybridGen.experiment import load_experiment

        algo = load_algorithm("v2.1")
        exp  = load_experiment("v2", algo)
        exp.main()                         # parse args from CLI
        exp.main(args=namespace)           # or pass Namespace directly

    Output directory defaults to ``output/{algorithm_version}``
    (e.g., output/v2.1 for v2.1), overridable via ``--output-dir``.
"""

from .algorithm import load_algorithm
from .experiment import load_experiment

__all__ = ["load_algorithm", "load_experiment"]
