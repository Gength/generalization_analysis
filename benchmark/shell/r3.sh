#!/bin/bash
#SBATCH --partition=Krater
#SBATCH --job-name=bench_R3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=benchmark/logs/bench_R3_%j.log
#
# R3  Naive Random Baseline (uniform random activity traces, 5 iterations)
# ─────────────────────────────────────────────────────────────────────────────
# CLI arguments (all optional except --dataset):
#   --dataset D1..D21          Dataset key (required)
#   --output <dir>             Output directory (default: /tmp/<workdir>/results/)
#   --seed N                   Random seed (default: 42)
#   --miners Alpha Flower ...  Subset of miners (default: all 8)
#   --num-traces N             Shadow traces per iteration (default: 1000)
#
# Examples:
#   bash benchmark/shell/r3.sh --dataset D1
#   bash benchmark/shell/r3.sh --dataset D1 --num-traces 500
#   bash benchmark/shell/r3.sh --dataset D2 --output benchmark/results/configs_v2
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$SCRIPT_DIR"
export TMPDIR=/tmp

source ~/.bashrc
uv run python benchmark/job_r3.py "$@"
