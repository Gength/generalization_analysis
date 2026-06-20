#!/bin/bash
#SBATCH --partition=Krater
#SBATCH --job-name=bench_M2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=benchmark/logs/bench_M2_%j.log
#
# M2  PM4Py Built-in Generalization (deterministic, ~0.3s/miner)
# ─────────────────────────────────────────────────────────────────────────────
# CLI arguments (all optional except --dataset):
#   --dataset D1..D21          Dataset key (required)
#   --output <dir>             Output directory (default: /tmp/<workdir>/results/)
#   --miners Alpha Flower ...  Subset of miners (default: all 8)
#
# Examples:
#   bash benchmark/shell/m2.sh --dataset D1
#   bash benchmark/shell/m2.sh --dataset D1 --miners Alpha Heuristics
#   bash benchmark/shell/m2.sh --dataset D2 --output benchmark/results/configs_v2
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$SCRIPT_DIR"
export TMPDIR=/tmp

source ~/.bashrc
uv run python benchmark/job_m2.py "$@"
