#!/bin/bash
#SBATCH --partition=Krater
#SBATCH --job-name=bench_R2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --output=benchmark/logs/bench_R2_%j.log
#
# R2  Leave-One-Variant-Out (reference metric)
# ─────────────────────────────────────────────────────────────────────────────
# CLI arguments:
#   --dataset D1..D21          Dataset key (required)
#   --output <dir>             Output directory (default: /tmp/<workdir>/results/)
#
# Examples:
#   bash benchmark/shell/r2.sh --dataset D1
#   bash benchmark/shell/r2.sh --dataset D1 --output benchmark/results/configs_v2
# ─────────────────────────────────────────────────────────────────────────────

set -eo pipefail
export TMPDIR=/tmp
export PATH="$HOME/.local/bin:$PATH"

uv run python benchmark/job_r2.py "$@"
