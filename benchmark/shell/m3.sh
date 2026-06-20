#!/bin/bash
#SBATCH --partition=Krater
#SBATCH --job-name=bench_M3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=benchmark/logs/bench_M3_%j.log
#
# M3  Entropic Relevance (Entropia JAR, DFG-based, one score for all miners)
# ─────────────────────────────────────────────────────────────────────────────
# CLI arguments (all optional except --dataset):
#   --dataset D1..D21          Dataset key (required)
#   --output <dir>             Output directory (default: /tmp/<workdir>/results/)
#   --miners Alpha Flower ...  Subset of miners (default: all)
#
# Examples:
#   bash benchmark/shell/m3.sh --dataset D1
#   bash benchmark/shell/m3.sh --dataset D2
#   bash benchmark/shell/m3.sh --dataset D1 --output benchmark/results/configs_v2
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$SCRIPT_DIR"
export TMPDIR=/tmp

source ~/.bashrc
uv run python benchmark/job_m3.py "$@"
