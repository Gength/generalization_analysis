#!/bin/bash
#SBATCH --partition=Krater
#SBATCH --job-name=bench_M3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=benchmark/logs/bench_M3_%j.log
#
# M3  Entropic Relevance (Go-Back-N simulation)
# ─────────────────────────────────────────────────────────────────────────────
# CLI arguments:
#   --dataset D1..D21          Dataset key (required)
#   --output <dir>             Output directory (default: /tmp/<workdir>/results/)
#
# Examples:
#   bash benchmark/shell/m3.sh --dataset D1
#   bash benchmark/shell/m3.sh --dataset D1 --output benchmark/results/configs_v2
# ─────────────────────────────────────────────────────────────────────────────

set -eo pipefail
export TMPDIR=/tmp
export PATH="$HOME/.local/bin:$PATH"

uv run python benchmark/job_m3.py "$@"
