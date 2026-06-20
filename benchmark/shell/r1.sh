#!/bin/bash
#SBATCH --partition=Krater
#SBATCH --job-name=bench_R1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=benchmark/logs/bench_R1_%j.log
#
# R1  K-Fold Cross-Validation (reference metric)
# ─────────────────────────────────────────────────────────────────────────────
# CLI arguments:
#   --dataset D1..D21          Dataset key (required)
#   --output <dir>             Output directory (default: /tmp/<workdir>/results/)
#
# Examples:
#   bash benchmark/shell/r1.sh --dataset D1
#   bash benchmark/shell/r1.sh --dataset D1 --output benchmark/results/configs_v2
# ─────────────────────────────────────────────────────────────────────────────

set -eo pipefail
export TMPDIR=/tmp
export PATH="$HOME/.local/bin:$PATH"

uv run python benchmark/job_r1.py "$@"
