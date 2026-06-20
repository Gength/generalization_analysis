#!/bin/bash
#SBATCH --partition=Krater
#SBATCH --job-name=bench_R2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=benchmark/logs/bench_R2_%j.log
#
# R2  Leave-One-Variant-Out Fitness (parallelized, 8 workers)
# ─────────────────────────────────────────────────────────────────────────────
# CLI arguments (all optional except --dataset):
#   --dataset D1..D21          Dataset key (required)
#   --output <dir>             Output directory (default: /tmp/<workdir>/results/)
#   --seed N                   Random seed (default: 42)
#   --miners Alpha Flower ...  Subset of miners (default: all 8)
#   --r2-sample N              Cap variants to N (0 = all, default: 0)
#   --workers N                Parallel worker processes (default: 8)
#
# Examples:
#   bash benchmark/shell/r2.sh --dataset D1 --r2-sample 50     # fast smoke test
#   bash benchmark/shell/r2.sh --dataset D1                     # all variants
#   bash benchmark/shell/r2.sh --dataset D2 --output benchmark/results/configs_v2
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$SCRIPT_DIR"
export TMPDIR=/tmp

source ~/.bashrc
uv run python benchmark/job_r2.py "$@"
