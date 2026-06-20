#!/bin/bash
#SBATCH --partition=Krater
#SBATCH --job-name=bench_M6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=benchmark/logs/bench_M6_%j.log
#
# M6  Bootstrap Generalization (Entropia -bgen, eigenvalue precision & recall)
# ─────────────────────────────────────────────────────────────────────────────
# CLI arguments (all optional except --dataset):
#   --dataset D1..D21          Dataset key (required)
#   --output <dir>             Output directory (default: /tmp/<workdir>/results/)
#   --jar fixed|vanilla        JAR version (default: fixed = 1.7.1)
#   --k N                      Subtrace length (default: 2)
#   --m N                      Bootstrap replicates (default: 10)
#   --n N                      Sample size per replicate (default: 200)
#   --g N                      Breeding generations (default: 10)
#   --p F                      Breeding probability (default: 1.0)
#   --miners Alpha Flower ...  Subset of miners (default: all)
#
# Examples:
#   bash benchmark/shell/m6.sh --dataset D1
#   bash benchmark/shell/m6.sh --dataset D1 --m 5 --k 2
#   bash benchmark/shell/m6.sh --dataset D2 --output benchmark/results/configs_v2
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$SCRIPT_DIR"
export TMPDIR=/tmp

source ~/.bashrc
uv run python benchmark/job_m6.py "$@"
