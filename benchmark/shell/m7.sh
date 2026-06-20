#!/bin/bash
#SBATCH --partition=Krater
#SBATCH --job-name=bench_M7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=benchmark/logs/bench_M7_%j.log
#
# M7  SpeciAL4PM — species-based generalization (C1 coverage ratio)
# ─────────────────────────────────────────────────────────────────────────────
# CLI arguments (all optional except --dataset):
#   --dataset D1..D21          Dataset key (required)
#   --output <dir>             Output directory (default: /tmp/<workdir>/results/)
#   --miners Alpha Flower ...  Subset of miners (default: all)
#
# Examples:
#   bash benchmark/shell/m7.sh --dataset D1
#   bash benchmark/shell/m7.sh --dataset D1 --miners Alpha Inductive_Strict
#   bash benchmark/shell/m7.sh --dataset D2 --output benchmark/results/configs_v2
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$SCRIPT_DIR"
export TMPDIR=/tmp

source ~/.bashrc
uv run python benchmark/job_m7.py "$@"
