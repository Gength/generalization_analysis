"""
Usage:
    python -m HybridGen -a v2.1 -e v2 --miner all --weight 0.5
    python -m HybridGen --list
    python -m HybridGen -a v1 -h               # show experiment help

Entry point: selects algorithm + experiment via CLI flags,
then delegates remaining arguments to the experiment runner.
"""

import sys
import argparse

from HybridGen.algorithm import load_algorithm
from HybridGen.experiment import load_experiment


WRAPPER_HELP = """
HybridGen — Hybrid Generative-Structural Generalization Evaluation

Wrapper arguments (consumed by HybridGen, not passed to experiment):
  -a, --algorithm ALGO    Algorithm version: v1, v2, v2.1  [default: v2]
  -e, --experiment EXP    Experiment runner version         [default: same as --algorithm]
  --list                  List registered algorithms and experiments, then exit.

All other arguments are forwarded to the experiment runner.
Use -h after -a/-e to see experiment-specific options.

Examples:
  python -m HybridGen -a v2 -e v2 --miner all --weight 0.5
  python -m HybridGen -a v1 -h
  python -m HybridGen --list
"""


def main():
    p = argparse.ArgumentParser(
        usage=argparse.SUPPRESS,
        add_help=False,
    )
    p.add_argument("-a", "--algorithm", default="v2")
    p.add_argument("-e", "--experiment", default=None)
    p.add_argument("--list", action="store_true")
    p.add_argument("-h", "--help", action="store_true")

    known, remaining = p.parse_known_args()

    if known.help and not known.list:
        print(WRAPPER_HELP)
        return

    if known.list:
        from HybridGen.algorithm import ALGORITHM_REGISTRY
        from HybridGen.experiment import EXPERIMENT_REGISTRY
        print("Algorithms:", sorted(ALGORITHM_REGISTRY.keys()))
        print("Experiments:", sorted(EXPERIMENT_REGISTRY.keys()))
        return

    exp_name = known.experiment or known.algorithm
    algo = load_algorithm(known.algorithm)
    exp = load_experiment(exp_name, algo)

    sys.argv = [sys.argv[0]] + remaining
    exp.main()
