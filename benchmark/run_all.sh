#!/bin/bash
# Full Generalization Benchmark
# Each script is independently runnable.
# Run this to execute all available methods sequentially.
#
# Usage:
#   bash benchmark/run_all.sh                    # D1 (default)
#   bash benchmark/run_all.sh D2                 # D2 (BPI 2013 Incidents)
#
# Excluded (not feasible on real-life logs):
#   M4 (Anti-Alignment)   — single-threaded O(n²~n³), 14h+ per miner
#   M8 (Pattern-based)    — JAR crashes internally
#
# Optional (requires Docker GPU):
#   M5 (AVATAR)           — bash benchmark/m5.sh --quick

set -e
DATASET="${1:-D1}"

echo "=========================================="
echo "Full Generalization Benchmark — $DATASET"
echo "=========================================="
echo "Started: $(date)"
echo ""

echo "[prepare] Discovering models..."
bash benchmark/prepare.sh --dataset "$DATASET"

echo "[m1]     HybridGen methods (M1a-M1g)..."
bash benchmark/m1.sh --dataset "$DATASET"

echo "[m2]     PM4Py Built-in Generalization..."
bash benchmark/m2.sh --dataset "$DATASET"

echo "[m3]     Entropic Relevance..."
bash benchmark/m3.sh --dataset "$DATASET"

echo "[m6]     Bootstrap Generalization..."
bash benchmark/m6.sh --dataset "$DATASET"

echo "[m7]     SpeciAL4PM..."
bash benchmark/m7.sh --dataset "$DATASET"

echo "[m5]     AVATAR (RelGAN) — Docker GPU required (FULL: 5000 adv steps)..."
bash benchmark/m5.sh --dataset "$DATASET"

echo "[r]       Reference / Sanity-Check Metrics (R1-R3)..."
bash benchmark/reference.sh --dataset "$DATASET"

echo ""
echo "=========================================="
echo "Completed: $(date)"
echo "Configs in benchmark/results/configs_v2/"
echo "=========================================="
