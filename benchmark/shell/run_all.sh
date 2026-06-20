#!/bin/bash
#SBATCH --partition=Krater
#SBATCH --job-name=bench_ALL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=benchmark/logs/bench_ALL_%j.log
#
# Full pipeline — runs all methods sequentially (excl. M5 GPU)
# ─────────────────────────────────────────────────────────────────────────────
# Usage:
#   bash benchmark/shell/run_all.sh          # D1 (default)
#   bash benchmark/shell/run_all.sh D2       # D2
#
# Each method is a self-contained /tmp job; no preparation step needed.
# All results land in benchmark/results/configs_v2/.
#
# To include M5 (AVATAR GPU), uncomment the line below.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$SCRIPT_DIR"
export TMPDIR=/tmp
DATASET="${1:-D1}"
OUTPUT="benchmark/results/configs_v2"

echo "=========================================="
echo "Full Generalization Benchmark — $DATASET"
echo "=========================================="
echo "Started: $(date)"

bash benchmark/shell/m1.sh --dataset "$DATASET" --output "$OUTPUT"
bash benchmark/shell/m2.sh --dataset "$DATASET" --output "$OUTPUT"
bash benchmark/shell/m3.sh --dataset "$DATASET" --output "$OUTPUT"
bash benchmark/shell/m6.sh --dataset "$DATASET" --output "$OUTPUT"
bash benchmark/shell/m7.sh --dataset "$DATASET" --output "$OUTPUT"
# bash benchmark/shell/m5.sh --dataset "$DATASET" --output "$OUTPUT"  # requires Docker GPU
bash benchmark/shell/r1.sh --dataset "$DATASET" --output "$OUTPUT"
bash benchmark/shell/r2.sh --dataset "$DATASET" --output "$OUTPUT"
bash benchmark/shell/r3.sh --dataset "$DATASET" --output "$OUTPUT"

echo ""
echo "Completed: $(date)"
echo "Configs in $OUTPUT"
