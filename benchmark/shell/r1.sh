#!/bin/bash
#SBATCH --partition=Krater
#SBATCH --job-name=bench_R1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=benchmark/logs/bench_R1_%j.log
#
# R1  K-Fold Cross-Validation Fitness (5-fold, variant-based, 3 shuffles)
# ─────────────────────────────────────────────────────────────────────────────
# CLI arguments (all optional except --dataset):
#   --dataset D1..D21          Dataset key (required)
#   --output <dir>             Output directory (default: /tmp/<workdir>/results/)
#   --seed N                   Random seed (default: 42)
#   --miners Alpha Flower ...  Subset of miners (default: all 8)
#
# Examples:
#   bash benchmark/shell/r1.sh --dataset D1
#   bash benchmark/shell/r1.sh --dataset D1 --seed 1
#   bash benchmark/shell/r1.sh --dataset D2 --output benchmark/results/configs_v2
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$SCRIPT_DIR"
export TMPDIR=/tmp

source ~/.bashrc
uv run python benchmark/job_r1.py "$@"
