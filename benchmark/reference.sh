#!/bin/bash
# R1-R3: Reference / Sanity-Check Metrics (Methodology v2)
#   R1 = K-Fold CV (k=5, variant-based, 3 shuffles)
#   R2 = Leave-One-Variant-Out (default: all variants; use --r2-sample N to cap)
#   R3 = Naive Random Baseline (5 iterations)

# Usage:
#   bash benchmark/reference.sh                    # D1, all R methods
#   bash benchmark/reference.sh --dataset D2       # D2, all R methods
#   bash benchmark/reference.sh --r2-sample 50     # D1, R2 samples 50 variants

uv run python benchmark/run_r_family.py "$@"
