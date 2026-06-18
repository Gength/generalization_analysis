#!/bin/bash
# M6: Bootstrap Generalization (Entropia -bgen with fixed JAR)
# Uses fixed JAR (jbpt-pm-entropia-1.7.1.jar) for k=2 support on all datasets.
# For full methodology v2 runs (5 reps, config JSON output), use:
#   uv run python benchmark/bridges/run_m6_bgen.py --dataset D1 --k 2 --m 5
uv run python benchmark/bridges/run_m6_bgen.py "$@"
