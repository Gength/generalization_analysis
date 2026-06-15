#!/bin/bash
# D1 Sepsis — Full Generalization Benchmark
# Each script is independently runnable.
# Run this to execute all available methods sequentially.
#
# Excluded (not feasible on real-life logs):
#   M4 (Anti-Alignment)   — single-threaded O(n²~n³), 14h+ per miner
#   M8 (Pattern-based)    — JAR crashes internally
#
# Optional (requires Docker GPU):
#   M5 (AVATAR)           — bash benchmark/m5.sh --quick

set -e
echo "=========================================="
echo "D1 Sepsis — Full Generalization Benchmark"
echo "=========================================="
echo "Started: $(date)"
echo ""

echo "[prepare] Discovering models..."
bash benchmark/prepare.sh

echo "[m1]     HybridGen methods (M1a-M1g)..."
bash benchmark/m1.sh

echo "[reference]     Reference / Sanity-Check Metrics (R1-R3)..."
bash benchmark/reference.sh

echo "[m3]     Entropic Relevance..."
bash benchmark/m3.sh

echo "[m6]     Bootstrap Generalization..."
bash benchmark/m6.sh

echo "[m7]     SpeciAL4PM..."
bash benchmark/m7.sh

echo "[m5]     AVATAR (RelGAN) — Docker GPU required (FULL: 5000 adv steps)..."
bash benchmark/m5.sh

echo ""
echo "=========================================="
echo "Completed: $(date)"
echo "Configs in benchmark/results/configs/ (v1) and configs_v2/ (v2)"
echo "=========================================="
